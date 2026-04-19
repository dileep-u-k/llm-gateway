package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/dileep-u-k/llm-gateway/internal/llm"
	"github.com/dileep-u-k/llm-gateway/internal/ops"
	"github.com/dileep-u-k/llm-gateway/internal/platform"
	"github.com/dileep-u-k/llm-gateway/internal/tools"
	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

func TestDetermineModelIDMidSessionForcedOverride(t *testing.T) {
	handler, cleanup := newTestHandler(t)
	defer cleanup()

	ctx := context.Background()
	session := &sessionState{
		ConversationID:     "convo-1",
		ModelID:            "gpt-4o",
		EffectiveModelID:   "gpt-4o",
		IsForced:           true,
		ForceScope:         forceScopePrimaryReasoner,
		SessionModeVersion: sessionModeVersion,
	}
	if err := handler.saveSession(ctx, session); err != nil {
		t.Fatalf("saveSession: %v", err)
	}

	req := &api.GenerationRequest{
		Prompt:         "switch models",
		ConversationID: "convo-1",
		Config: api.GenerationConfig{
			ForceModel: "claude-3",
		},
	}

	modelID, failoverInfo, updatedSession, routeSelection, err := handler.determineModelID(ctx, req)
	if err != nil {
		t.Fatalf("determineModelID returned error: %v", err)
	}
	if failoverInfo != nil {
		t.Fatalf("expected no failover info, got %+v", failoverInfo)
	}
	if modelID != "claude-3" {
		t.Fatalf("expected forced override to choose claude-3, got %s", modelID)
	}
	if updatedSession == nil || updatedSession.OverrideCount != 1 {
		t.Fatalf("expected override count to increment, got %+v", updatedSession)
	}
	if updatedSession.LastOverrideFrom != "gpt-4o" || updatedSession.LastOverrideTo != "claude-3" {
		t.Fatalf("unexpected override metadata: %+v", updatedSession)
	}
	if updatedSession.EffectiveModelID != "claude-3" || updatedSession.ForceScope != forceScopePrimaryReasoner {
		t.Fatalf("expected effective forced metadata to update, got %+v", updatedSession)
	}
	if routeSelection == nil || !routeSelection.Explanation.ForcedSemantics.IsForced {
		t.Fatalf("expected forced route explanation, got %+v", routeSelection)
	}
}

func TestDetermineModelIDForcedFallbackWhenRequestedModelUnavailable(t *testing.T) {
	handler, cleanup := newTestHandler(t)
	defer cleanup()

	ctx := context.Background()
	handler.profiler.UpdateModelHealthCheck(ctx, "claude-3", llm.HealthProbeResult{
		Status:        llm.HealthStatusOffline,
		AccessAllowed: false,
		Latency:       10 * time.Millisecond,
		Err:           context.DeadlineExceeded,
	})
	handler.profiler.UpdateCapabilityHealthCheck(ctx, llm.ProviderForModel("claude-3"), "claude-3", llm.CapabilityTextGeneration, llm.HealthProbeResult{
		Status:        llm.HealthStatusOffline,
		AccessAllowed: false,
		Latency:       10 * time.Millisecond,
		Err:           context.DeadlineExceeded,
	})

	req := &api.GenerationRequest{
		Prompt:         "switch if possible",
		ConversationID: "convo-2",
		Config: api.GenerationConfig{
			ForceModel:                   "claude-3",
			ForceIfAvailableElseFallback: true,
		},
	}

	modelID, failoverInfo, updatedSession, routeSelection, err := handler.determineModelID(ctx, req)
	if err != nil {
		t.Fatalf("determineModelID returned error: %v", err)
	}
	if modelID != "gpt-4o" {
		t.Fatalf("expected healthy fallback model, got %s", modelID)
	}
	if failoverInfo == nil || failoverInfo.NewModel != "gpt-4o" {
		t.Fatalf("expected fallback failover info, got %+v", failoverInfo)
	}
	if updatedSession == nil || updatedSession.ModelID != "gpt-4o" || !updatedSession.IsForced {
		t.Fatalf("expected forced session to pin fallback model, got %+v", updatedSession)
	}
	if routeSelection == nil || routeSelection.ModelID != "gpt-4o" {
		t.Fatalf("expected route explanation for fallback model, got %+v", routeSelection)
	}
}

func TestDetermineModelIDForcedSessionContinuityUsesEffectiveModel(t *testing.T) {
	handler, cleanup := newTestHandler(t)
	defer cleanup()

	ctx := context.Background()
	session := &sessionState{
		ConversationID:     "convo-3",
		ModelID:            "claude-3",
		EffectiveModelID:   "gpt-4o",
		IsForced:           true,
		ForceScope:         forceScopeCapability,
		SessionModeVersion: sessionModeVersion,
	}
	if err := handler.saveSession(ctx, session); err != nil {
		t.Fatalf("saveSession: %v", err)
	}

	modelID, failoverInfo, updatedSession, routeSelection, err := handler.determineModelID(ctx, &api.GenerationRequest{
		Prompt:         "continue",
		ConversationID: "convo-3",
	})
	if err != nil {
		t.Fatalf("determineModelID returned error: %v", err)
	}
	if failoverInfo != nil {
		t.Fatalf("expected no failover, got %+v", failoverInfo)
	}
	if modelID != "gpt-4o" {
		t.Fatalf("expected effective model to be reused, got %s", modelID)
	}
	if updatedSession == nil || updatedSession.EffectiveModelID != "gpt-4o" || updatedSession.ModelID != "claude-3" {
		t.Fatalf("expected pinned/effective model distinction to persist, got %+v", updatedSession)
	}
	if routeSelection == nil || routeSelection.Explanation.ForcedSemantics.PinnedModel != "claude-3" {
		t.Fatalf("expected route explanation to expose pinned model, got %+v", routeSelection)
	}
}

func TestDetermineModelIDStrictForceRejectsFallback(t *testing.T) {
	handler, cleanup := newTestHandler(t)
	defer cleanup()

	ctx := context.Background()
	handler.profiler.UpdateModelHealthCheck(ctx, "claude-3", llm.HealthProbeResult{
		Status:        llm.HealthStatusOffline,
		AccessAllowed: false,
		Latency:       10 * time.Millisecond,
		Err:           context.DeadlineExceeded,
	})
	handler.profiler.UpdateCapabilityHealthCheck(ctx, llm.ProviderForModel("claude-3"), "claude-3", llm.CapabilityTextGeneration, llm.HealthProbeResult{
		Status:        llm.HealthStatusOffline,
		AccessAllowed: false,
		Latency:       10 * time.Millisecond,
		Err:           context.DeadlineExceeded,
	})

	_, _, _, _, err := handler.determineModelID(ctx, &api.GenerationRequest{
		Prompt:         "force strictly",
		ConversationID: "convo-4",
		Config: api.GenerationConfig{
			ForceModel:                   "claude-3",
			ForceScope:                   forceScopeStrictEndToEnd,
			StrictForce:                  true,
			ForceIfAvailableElseFallback: true,
		},
	})
	if err == nil {
		t.Fatal("expected strict force to reject fallback")
	}
	reqErr, ok := err.(*requestFailure)
	if !ok {
		t.Fatalf("expected requestFailure, got %T", err)
	}
	if reqErr.StatusCode != 424 {
		t.Fatalf("expected failed dependency status, got %d", reqErr.StatusCode)
	}
}

func TestPersistOrchestrationMetadataStoresRecord(t *testing.T) {
	handler, cleanup := newTestHandler(t)
	defer cleanup()

	recordID := handler.persistOrchestrationMetadata(context.Background(), api.GenerationRequest{
		Prompt:         "Write production-grade Go code",
		UserID:         "user-1",
		ConversationID: "convo-meta",
	}, &api.GenerationResponse{
		ModelUsed:   "gpt-4o",
		CacheStatus: "MISS",
		Route: &api.RouteMetadata{
			SelectedProvider: "openai",
			SelectedModel:    "gpt-4o",
		},
		Intent: &api.IntentMetadata{
			TaskType:          "coding",
			SessionConstraint: "conversation_continuity",
		},
	})
	if recordID == "" {
		t.Fatal("expected orchestration metadata record ID")
	}

	payload, err := handler.rdb.Get(context.Background(), "orchestration:"+recordID).Result()
	if err != nil {
		t.Fatalf("expected persisted orchestration record, got error: %v", err)
	}
	if !strings.Contains(payload, "\"conversation_id\":\"convo-meta\"") || !strings.Contains(payload, "\"task_type\":\"coding\"") {
		t.Fatalf("unexpected orchestration payload: %s", payload)
	}
}

func TestPrepareExecutionContextBuildsPlanAndArtifacts(t *testing.T) {
	handler, cleanup := newTestHandler(t)
	defer cleanup()

	prepared, err := handler.prepareExecutionContext(context.Background(), api.GenerationRequest{
		Prompt:     "Compare the screenshot with the document and explain the mismatch.",
		InputType:  "mixed",
		OutputType: "summary",
		Assets: []api.AssetInput{
			{AssetID: "doc-1", Type: "document", Name: "spec.md", InlineText: "Expected value is 5."},
			{AssetID: "img-1", Type: "image", Name: "screen.png", Caption: "Displayed value is 9."},
		},
	})
	if err != nil {
		t.Fatalf("prepareExecutionContext returned error: %v", err)
	}
	if prepared == nil || prepared.Plan == nil {
		t.Fatalf("expected prepared execution plan, got %+v", prepared)
	}
	if len(prepared.Artifacts) != 2 {
		t.Fatalf("expected two artifacts, got %+v", prepared.Artifacts)
	}
	if prepared.Plan.PrimaryStageID != "reason_answer" {
		t.Fatalf("expected reason_answer primary stage, got %+v", prepared.Plan)
	}
}

func TestHandleGenerationAppliesWorkspaceGovernance(t *testing.T) {
	handler, cleanup := newTestHandler(t)
	defer cleanup()

	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.POST("/api/v1/generate", handler.HandleGeneration)

	body := `{"prompt":"Generate a new logo","tenant_id":"default","workspace_id":"ops","task_type":"image_generation","requires_generation":true,"config":{"preference":"balanced"}}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/generate", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	engine.ServeHTTP(resp, req)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected policy rejection, got %d body=%s", resp.Code, resp.Body.String())
	}
	if !strings.Contains(resp.Body.String(), "does not allow generation workflows") {
		t.Fatalf("expected governance rejection in body, got %s", resp.Body.String())
	}
}

func TestNormalizeRequestForPlatformRejectsInvalidAsyncOptions(t *testing.T) {
	handler, cleanup := newTestHandler(t)
	defer cleanup()

	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	req := httptest.NewRequest(http.MethodPost, "/api/v1/generate", bytes.NewBufferString(`{}`))
	c.Request = req
	principal := &platform.Principal{ID: "user-1", Role: "admin", Mode: "open"}

	_, _, _, _, err := handler.normalizeRequestForPlatform(context.Background(), c, principal, api.GenerationRequest{
		Prompt:                "async callback test",
		SyncOrAsyncPreference: "eventual",
	})
	if err == nil || !strings.Contains(err.Error(), "sync_or_async_preference") {
		t.Fatalf("expected sync preference validation error, got %v", err)
	}

	_, _, _, _, err = handler.normalizeRequestForPlatform(context.Background(), c, principal, api.GenerationRequest{
		Prompt:                "async callback test",
		SyncOrAsyncPreference: "sync",
		CallbackURL:           "https://example.com/hook",
	})
	if err == nil || !strings.Contains(err.Error(), "only supported for async") {
		t.Fatalf("expected callback async-only validation error, got %v", err)
	}

	_, _, _, _, err = handler.normalizeRequestForPlatform(context.Background(), c, principal, api.GenerationRequest{
		Prompt:                "async callback test",
		SyncOrAsyncPreference: "async",
		CallbackURL:           "ftp://example.com/hook",
	})
	if err == nil || !strings.Contains(err.Error(), "http or https") {
		t.Fatalf("expected callback scheme validation error, got %v", err)
	}
}

func TestServeStaticAssetSupportsPrefixedPath(t *testing.T) {
	handler, cleanup := newTestHandler(t)
	defer cleanup()

	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.GET("/ui/*filepath", handler.ServeStaticAsset)

	req := httptest.NewRequest(http.MethodGet, "/ui/styles.css", nil)
	resp := httptest.NewRecorder()
	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected static asset to resolve, got status %d body=%s", resp.Code, resp.Body.String())
	}
	if ct := resp.Header().Get("Content-Type"); !strings.Contains(ct, "text/css") {
		t.Fatalf("expected css content type, got %s", ct)
	}
}

func newTestHandler(t *testing.T) (*GatewayHandler, func()) {
	t.Helper()
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})

	routerConfig := &llm.RouterConfig{
		Thresholds: map[string]interface{}{
			"health_check_staleness":            "5m",
			"max_error_rate":                    0.5,
			"min_request_count":                 1,
			"relevance_threshold":               0.45,
			"circuit_breaker_failure_threshold": 3,
		},
		Models: map[string]llm.ModelMetadata{
			"gpt-4o":   {QualityScore: 9.8, CodingScore: 9.8, Modalities: []string{"text", "image"}, Capabilities: []llm.Capability{llm.CapabilityTextGeneration, llm.CapabilityImageUnderstanding}},
			"claude-3": {QualityScore: 9.2, CodingScore: 9.2, Modalities: []string{"text"}, Capabilities: []llm.Capability{llm.CapabilityTextGeneration}},
		},
		Capabilities: map[string]llm.CapabilityMetadata{
			string(llm.CapabilityTextGeneration):     {Description: "Text reasoning", Modalities: []string{"text"}},
			string(llm.CapabilityImageUnderstanding): {Description: "Image understanding", Modalities: []string{"image"}},
			string(llm.CapabilityOCR):                {Description: "OCR", Modalities: []string{"document"}},
			string(llm.CapabilityEmbeddings):         {Description: "Embeddings", Modalities: []string{"text"}},
		},
		Strategies: map[string]llm.RoutingStrategy{
			"default":  {QualityWeight: 1, CostWeight: 0, LatencyWeight: 0, ReliabilityWeight: 0},
			"balanced": {QualityWeight: 1, CostWeight: 0, LatencyWeight: 0, ReliabilityWeight: 0},
		},
		SmartBalancedCostThreshold: 0.001,
	}

	llm.InitializeModelCosts(map[string]map[string]float64{
		"gpt-4o":   {"input": 1, "output": 1},
		"claude-3": {"input": 1, "output": 1},
	})
	profiler := llm.NewProfiler(rdb, routerConfig)
	router := llm.NewRouter(profiler, routerConfig)
	controlPlane := llm.NewControlPlane(routerConfig, profiler, router, []string{"gpt-4o", "claude-3"}, map[string]string{
		"gpt-4o":   "test-openai-key",
		"claude-3": "test-anthropic-key",
	})
	multimodalRuntime := llm.NewMultimodalRuntime(rdb, controlPlane)
	ctx := context.Background()
	for _, modelID := range []string{"gpt-4o", "claude-3"} {
		if _, err := profiler.GetProfile(ctx, modelID); err != nil {
			t.Fatalf("GetProfile(%s): %v", modelID, err)
		}
		probe := llm.HealthProbeResult{Status: llm.HealthStatusOnline, AccessAllowed: true, Latency: 10 * time.Millisecond}
		profiler.UpdateModelHealthCheck(ctx, modelID, probe)
		profiler.UpdateCapabilityHealthCheck(ctx, llm.ProviderForModel(modelID), modelID, llm.CapabilityTextGeneration, probe)
		profiler.UpdateProviderHealthCheck(ctx, llm.ProviderForModel(modelID), probe)
	}

	handler := &GatewayHandler{
		clients: map[string]llm.LLMClient{
			"gpt-4o":   stubLLMClient{},
			"claude-3": stubLLMClient{},
		},
		profiler:       profiler,
		router:         router,
		controlPlane:   controlPlane,
		multimodal:     multimodalRuntime,
		promptAnalyzer: llm.NewPromptAnalyzer(),
		config: &AppConfig{
			EnabledModels: []string{"gpt-4o", "claude-3"},
			ModelBudgets:  map[string]float64{},
			RouterConfig:  routerConfig,
			RAGConfig:     &llm.Config{RetrievalTopK: 2, RAGFailurePolicy: "graceful_no_rag"},
			ModelCosts: map[string]map[string]float64{
				"gpt-4o":   {"input": 0.000005},
				"claude-3": {"input": 0.000003},
			},
			PlatformConfig: platform.DefaultConfig(),
		},
		rdb:       rdb,
		evaluator: ops.Evaluator{},
	}
	handler.authenticator = platform.NewAuthenticatorFromEnv()
	handler.policyEngine = platform.NewEngine(handler.config.PlatformConfig, platform.NewPolicyBundleStore(rdb), handler.config.ModelCosts)
	handler.auditLogger = platform.NewAuditLogger(rdb)
	handler.artifactAccess = platform.NewSignedArtifactAccessLayerFromEnv("15m")
	handler.artifactRoot = t.TempDir()

	return handler, func() {
		_ = rdb.Close()
		mr.Close()
	}
}

type stubLLMClient struct{}

func (stubLLMClient) Generate(context.Context, []llm.Message, *llm.GenerationConfig, []tools.Tool) (*llm.GenerationResult, error) {
	return &llm.GenerationResult{Content: "ok", Usage: api.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}}, nil
}

func (stubLLMClient) GenerateStream(context.Context, []llm.Message, *llm.GenerationConfig, []tools.Tool) (<-chan *llm.StreamingResult, error) {
	return nil, nil
}
