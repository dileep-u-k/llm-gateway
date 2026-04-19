package llm

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func TestSelectOptimalModelPrefersOnlineOverDegraded(t *testing.T) {
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
			"gpt-4o":   {QualityScore: 9.8, CodingScore: 9.8},
			"claude-3": {QualityScore: 9.4, CodingScore: 9.4},
		},
		Strategies: map[string]RoutingStrategy{
			"default": {QualityWeight: 1, CostWeight: 0, LatencyWeight: 0, ReliabilityWeight: 0},
		},
		SmartBalancedCostThreshold: 0.001,
	}

	InitializeModelCosts(map[string]map[string]float64{
		"gpt-4o":   {"input": 1, "output": 1},
		"claude-3": {"input": 1, "output": 1},
	})

	profiler := NewProfiler(rdb, cfg)
	router := NewRouter(profiler, cfg)
	ctx := context.Background()

	seedHealthyModel(t, profiler, ctx, "gpt-4o", HealthStatusOnline)
	seedHealthyModel(t, profiler, ctx, "claude-3", HealthStatusDegraded)

	modelID, err := router.SelectOptimalModel(ctx, []string{"gpt-4o", "claude-3"}, "default", 100, map[string]float64{})
	if err != nil {
		t.Fatalf("SelectOptimalModel returned error: %v", err)
	}
	if modelID != "gpt-4o" {
		t.Fatalf("expected online model to win, got %s", modelID)
	}
}

func TestSelectOptimalModelSkipsCircuitOpenModel(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})

	cfg := &RouterConfig{
		Thresholds: map[string]interface{}{
			"health_check_staleness":            "5m",
			"max_error_rate":                    0.5,
			"min_request_count":                 1,
			"circuit_breaker_failure_threshold": 1,
			"circuit_breaker_cooldown":          "10m",
		},
		Models: map[string]ModelMetadata{
			"gpt-4o":   {QualityScore: 9.8, CodingScore: 9.8},
			"claude-3": {QualityScore: 9.4, CodingScore: 9.4},
		},
		Strategies: map[string]RoutingStrategy{
			"default": {QualityWeight: 1, CostWeight: 0, LatencyWeight: 0, ReliabilityWeight: 0},
		},
		SmartBalancedCostThreshold: 0.001,
	}

	InitializeModelCosts(map[string]map[string]float64{
		"gpt-4o":   {"input": 1, "output": 1},
		"claude-3": {"input": 1, "output": 1},
	})

	profiler := NewProfiler(rdb, cfg)
	router := NewRouter(profiler, cfg)
	ctx := context.Background()

	seedHealthyModel(t, profiler, ctx, "gpt-4o", HealthStatusOnline)
	seedHealthyModel(t, profiler, ctx, "claude-3", HealthStatusOnline)
	profiler.UpdateProfileOnFailure(ctx, "gpt-4o", context.DeadlineExceeded)

	modelID, err := router.SelectOptimalModel(ctx, []string{"gpt-4o", "claude-3"}, "default", 100, map[string]float64{})
	if err != nil {
		t.Fatalf("SelectOptimalModel returned error: %v", err)
	}
	if modelID != "claude-3" {
		t.Fatalf("expected circuit-open model to be skipped, got %s", modelID)
	}
}

func TestSelectOptimalRouteIncludesFallbackChainAndFilteredCandidates(t *testing.T) {
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
			"gpt-4o":          {QualityScore: 9.8, CodingScore: 9.8},
			"claude-3":        {QualityScore: 9.4, CodingScore: 9.4},
			"mistral-large-1": {QualityScore: 7.2, CodingScore: 7.0},
		},
		Strategies: map[string]RoutingStrategy{
			"default": {QualityWeight: 1, CostWeight: 0, LatencyWeight: 0, ReliabilityWeight: 0},
		},
		SmartBalancedCostThreshold: 0.001,
	}

	InitializeModelCosts(map[string]map[string]float64{
		"gpt-4o":          {"input": 1, "output": 1},
		"claude-3":        {"input": 1, "output": 1},
		"mistral-large-1": {"input": 1, "output": 1},
	})

	profiler := NewProfiler(rdb, cfg)
	router := NewRouter(profiler, cfg)
	ctx := context.Background()

	seedHealthyModel(t, profiler, ctx, "gpt-4o", HealthStatusOnline)
	seedHealthyModel(t, profiler, ctx, "claude-3", HealthStatusOnline)
	seedHealthyModel(t, profiler, ctx, "mistral-large-1", HealthStatusOffline)

	selection, err := router.SelectOptimalRoute(ctx, []string{"gpt-4o", "claude-3", "mistral-large-1"}, "default", 100, map[string]float64{}, ForceMetadata{})
	if err != nil {
		t.Fatalf("SelectOptimalRoute returned error: %v", err)
	}
	if selection.ModelID != "gpt-4o" {
		t.Fatalf("expected highest-quality model to win, got %s", selection.ModelID)
	}
	if len(selection.Explanation.FallbackCandidates) != 1 || selection.Explanation.FallbackCandidates[0].ModelID != "claude-3" {
		t.Fatalf("expected ordered fallback chain, got %+v", selection.Explanation.FallbackCandidates)
	}
	if len(selection.Explanation.FilteredCandidates) != 1 || selection.Explanation.FilteredCandidates[0].ModelID != "mistral-large-1" {
		t.Fatalf("expected offline candidate to be reported as filtered, got %+v", selection.Explanation.FilteredCandidates)
	}
}

func seedHealthyModel(t *testing.T, profiler *Profiler, ctx context.Context, modelID string, status HealthStatus) {
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
