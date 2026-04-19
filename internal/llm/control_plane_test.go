package llm

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func TestControlPlaneBuildsRegistries(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})

	cfg := &RouterConfig{
		Providers: map[string]ProviderMetadata{
			"openai": {Endpoint: "https://api.openai.com/v1", Regions: []string{"global"}},
		},
		Capabilities: map[string]CapabilityMetadata{
			string(CapabilityTextGeneration): {Description: "Text reasoning", Modalities: []string{"text"}},
		},
		Models: map[string]ModelMetadata{
			"gpt-4o": {
				Provider:                "openai",
				Capabilities:            []Capability{CapabilityTextGeneration},
				Modalities:              []string{"text"},
				ContextLimit:            128000,
				StreamingSupported:      true,
				StructuredOutputSupport: true,
				ToolSupport:             true,
			},
		},
		Strategies: map[string]RoutingStrategy{
			"default": {QualityWeight: 1},
		},
		RoutePolicies: map[string]RoutingPolicy{
			"default": {Strategy: "default", AllowDegradedFallback: true},
		},
	}

	InitializeModelCosts(map[string]map[string]float64{
		"gpt-4o": {"input": 1, "output": 1},
	})
	profiler := NewProfiler(rdb, cfg)
	router := NewRouter(profiler, cfg)
	controlPlane := NewControlPlane(cfg, profiler, router, []string{"gpt-4o"}, map[string]string{"gpt-4o": "test-key"})

	provider, ok := controlPlane.Providers().Get("openai")
	if !ok || !provider.AuthConfigured {
		t.Fatalf("expected provider registry to capture configured auth, got %+v", provider)
	}

	model, ok := controlPlane.Models().Get("gpt-4o")
	if !ok || !model.StreamingSupported || !model.ToolSupport {
		t.Fatalf("expected model registry metadata, got %+v", model)
	}

	capability, ok := controlPlane.Capabilities().Get(CapabilityTextGeneration)
	if !ok || len(capability.Models) != 1 || capability.Models[0] != "gpt-4o" {
		t.Fatalf("expected capability registry to reference gpt-4o, got %+v", capability)
	}
}

func TestControlPlaneDecoratesRouteWithPolicyIntentAndFallbackGraph(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})

	cfg := &RouterConfig{
		Thresholds: map[string]interface{}{
			"health_check_staleness":            "5m",
			"max_error_rate":                    0.5,
			"min_request_count":                 1,
			"circuit_breaker_failure_threshold": 3,
		},
		Models: map[string]ModelMetadata{
			"gpt-4o":          {QualityScore: 9.8, CodingScore: 9.9, ToolSupport: true, Capabilities: []Capability{CapabilityTextGeneration}},
			"claude-3":        {QualityScore: 9.5, CodingScore: 9.7, ToolSupport: true, Capabilities: []Capability{CapabilityTextGeneration}},
			"mistral-large-1": {QualityScore: 8.8, CodingScore: 8.6, Capabilities: []Capability{CapabilityTextGeneration}},
		},
		Strategies: map[string]RoutingStrategy{
			"default":         {QualityWeight: 1},
			"best-for-coding": {UseCodingScore: true, QualityWeight: 1},
		},
		RoutePolicies: map[string]RoutingPolicy{
			"default":         {Strategy: "default", AllowDegradedFallback: true},
			"best-for-coding": {Strategy: "best-for-coding", AllowDegradedFallback: true, Description: "Coding-first policy."},
		},
		FallbackGraphs: map[string]map[string][]string{
			string(CapabilityTextGeneration): {
				"best-for-coding": {"gpt-4o", "claude-3", "mistral-large-1"},
			},
		},
	}

	InitializeModelCosts(map[string]map[string]float64{
		"gpt-4o":          {"input": 1, "output": 1},
		"claude-3":        {"input": 1, "output": 1},
		"mistral-large-1": {"input": 1, "output": 1},
	})
	profiler := NewProfiler(rdb, cfg)
	router := NewRouter(profiler, cfg)
	controlPlane := NewControlPlane(cfg, profiler, router, []string{"gpt-4o", "claude-3", "mistral-large-1"}, map[string]string{
		"gpt-4o":          "openai-key",
		"claude-3":        "anthropic-key",
		"mistral-large-1": "mistral-key",
	})

	ctx := context.Background()
	for _, modelID := range []string{"gpt-4o", "claude-3", "mistral-large-1"} {
		seedHealthyModelForControlPlane(t, profiler, ctx, modelID, HealthStatusOnline)
	}

	selection, err := controlPlane.PlanRoute(ctx, RoutePlanningRequest{
		Prompt:          "Implement a Go function and explain the tradeoffs.",
		ConversationID:  "convo-123",
		AvailableModels: []string{"gpt-4o", "claude-3", "mistral-large-1"},
		ModelBudgets:    map[string]float64{},
		PromptTokens:    120,
		Capability:      CapabilityTextGeneration,
	})
	if err != nil {
		t.Fatalf("PlanRoute returned error: %v", err)
	}
	if selection.ModelID != "gpt-4o" {
		t.Fatalf("expected highest coding-score model to win, got %s", selection.ModelID)
	}
	if selection.Explanation.PolicyName != "best-for-coding" {
		t.Fatalf("expected coding policy, got %+v", selection.Explanation)
	}
	if selection.Explanation.Intent.TaskType != "coding" || selection.Explanation.Intent.SessionConstraint != "conversation_continuity" {
		t.Fatalf("expected structured intent metadata, got %+v", selection.Explanation.Intent)
	}
	if len(selection.Explanation.FallbackGraph) != 3 {
		t.Fatalf("expected fallback graph metadata, got %+v", selection.Explanation.FallbackGraph)
	}
	if len(selection.Explanation.FallbackCandidates) != 2 || selection.Explanation.FallbackCandidates[0].ModelID != "claude-3" {
		t.Fatalf("expected fallback candidates ordered by graph, got %+v", selection.Explanation.FallbackCandidates)
	}
}

func seedHealthyModelForControlPlane(t *testing.T, profiler *Profiler, ctx context.Context, modelID string, status HealthStatus) {
	t.Helper()
	if _, err := profiler.GetProfile(ctx, modelID); err != nil {
		t.Fatalf("GetProfile(%s): %v", modelID, err)
	}
	provider := ProviderForModel(modelID)
	probe := HealthProbeResult{
		Status:        status,
		AccessAllowed: true,
		Latency:       20 * time.Millisecond,
	}
	profiler.UpdateModelHealthCheck(ctx, modelID, probe)
	profiler.UpdateCapabilityHealthCheck(ctx, provider, modelID, CapabilityTextGeneration, probe)
	profiler.UpdateProviderHealthCheck(ctx, provider, probe)
}
