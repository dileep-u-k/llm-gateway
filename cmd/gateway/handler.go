package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/dileep-u-k/llm-gateway/internal/llm"
	"github.com/dileep-u-k/llm-gateway/internal/observability"
	"github.com/dileep-u-k/llm-gateway/internal/ops"
	"github.com/dileep-u-k/llm-gateway/internal/platform"
	"github.com/dileep-u-k/llm-gateway/internal/tools"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

const sessionModeVersion = 3

const (
	forceScopePrimaryReasoner = "primary_reasoner_force"
	forceScopeCapability      = "capability_scoped_force"
	forceScopeStrictEndToEnd  = "strict_end_to_end_force"
)

type GatewayHandler struct {
	clients         map[string]llm.LLMClient
	profiler        *llm.Profiler
	router          *llm.Router
	controlPlane    *llm.ControlPlane
	ragService      *llm.RAGService
	intentAnalyzer  *llm.IntentAnalyzer
	toolManager     *tools.ToolManager
	promptAnalyzer  *llm.PromptAnalyzer
	memoryEngine    *llm.MemoryEngine
	contextComposer *llm.ContextComposer
	groundingEngine *llm.GroundingPolicyEngine
	multimodal      *llm.MultimodalRuntime
	generation      *llm.GenerationRuntime
	config          *AppConfig
	rdb             *redis.Client
	imageClients    map[string]llm.ImageClient
	speechClients   map[string]llm.SpeechClient
	imageRouter     *llm.ImageRouter
	opsRuntime      *ops.Runtime
	evaluator       ops.Evaluator
	canary          ops.CanaryPolicyEngine
	authenticator   *platform.Authenticator
	policyEngine    *platform.Engine
	auditLogger     *platform.AuditLogger
	artifactAccess  *platform.SignedArtifactAccessLayer
	artifactRoot    string
}

type requestFailure struct {
	StatusCode int
	Message    string
	Details    gin.H
}

func (e *requestFailure) Error() string {
	return e.Message
}

type sessionState struct {
	ConversationID     string
	ModelID            string
	EffectiveModelID   string
	IsForced           bool
	ForceScope         string
	StrictForce        bool
	OverrideCount      int
	LastOverrideAt     time.Time
	LastOverrideFrom   string
	LastOverrideTo     string
	SessionModeVersion int
	FailoverCount      int
	LastFailoverAt     time.Time
}

func NewGatewayHandler(clients map[string]llm.LLMClient, profiler *llm.Profiler, router *llm.Router, controlPlane *llm.ControlPlane, ragService *llm.RAGService, intentAnalyzer *llm.IntentAnalyzer, toolManager *tools.ToolManager, promptAnalyzer *llm.PromptAnalyzer, memoryEngine *llm.MemoryEngine, contextComposer *llm.ContextComposer, groundingEngine *llm.GroundingPolicyEngine, multimodal *llm.MultimodalRuntime, generation *llm.GenerationRuntime, config *AppConfig, rdb *redis.Client, imageClients map[string]llm.ImageClient, speechClients map[string]llm.SpeechClient, imageRouter *llm.ImageRouter, opsRuntime *ops.Runtime, authenticator *platform.Authenticator, policyEngine *platform.Engine, auditLogger *platform.AuditLogger, artifactAccess *platform.SignedArtifactAccessLayer, artifactRoot string) *GatewayHandler {
	return &GatewayHandler{
		clients:         clients,
		profiler:        profiler,
		router:          router,
		controlPlane:    controlPlane,
		ragService:      ragService,
		intentAnalyzer:  intentAnalyzer,
		toolManager:     toolManager,
		promptAnalyzer:  promptAnalyzer,
		memoryEngine:    memoryEngine,
		contextComposer: contextComposer,
		groundingEngine: groundingEngine,
		multimodal:      multimodal,
		generation:      generation,
		config:          config,
		rdb:             rdb,
		imageClients:    imageClients,
		speechClients:   speechClients,
		imageRouter:     imageRouter,
		opsRuntime:      opsRuntime,
		authenticator:   authenticator,
		policyEngine:    policyEngine,
		auditLogger:     auditLogger,
		artifactAccess:  artifactAccess,
		artifactRoot:    artifactRoot,
	}
}

func (h *GatewayHandler) HandleGeneration(c *gin.Context) {
	startTime := time.Now()
	var req api.GenerationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request: " + err.Error()})
		return
	}
	principal, err := h.resolvePrincipal(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}
	req, workspaceCtx, governanceDecision, auditEventID, err := h.normalizeRequestForPlatform(c.Request.Context(), c, principal, req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error(), "audit_event_id": auditEventID})
		return
	}

	log.Printf("--- New Request (User: %s, Convo: %s, Prompt: '%.30s...') ---", req.UserID, req.ConversationID, req.Prompt)
	intent := h.intentAnalyzer.AnalyzeIntent(req.Prompt)
	log.Printf("🔍 Intent Detected: %s", intent)

	prepared, err := h.prepareExecutionContext(c.Request.Context(), req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to prepare multimodal execution context: %v", err)})
		return
	}
	if h.policyEngine != nil {
		req, governanceDecision, err = h.policyEngine.ValidatePrepared(req, prepared, workspaceCtx, governanceDecision)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error(), "governance": governanceDecision, "audit_event_id": auditEventID})
			return
		}
	}

	if h.shouldRunAsync(req, prepared) && h.opsRuntime != nil {
		job, jobErr := h.opsRuntime.Submit(c.Request.Context(), req, prepared)
		if jobErr != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to submit async job: %v", jobErr)})
			return
		}
		c.JSON(http.StatusAccepted, gin.H{
			"job":        job.AcceptedResponse(),
			"governance": buildAPIGovernanceMetadata(governanceDecision),
			"security": &api.SecurityMetadata{
				AuthenticationMode: h.authenticationMode(),
				PrincipalID:        principal.ID,
				PrincipalRole:      principal.Role,
				TenantID:           workspaceCtx.Tenant.ID,
				WorkspaceID:        workspaceCtx.Workspace.ID,
				AuditEventID:       auditEventID,
				SignedArtifactMode: "signed_url",
			},
		})
		return
	}

	finalResponse, err := h.executeManagedSync(c.Request.Context(), req, prepared)
	if err != nil {
		if requestErr, ok := err.(*requestFailure); ok {
			payload := gin.H{"error": requestErr.Message}
			for key, value := range requestErr.Details {
				payload[key] = value
			}
			c.JSON(requestErr.StatusCode, payload)
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("An unexpected error occurred: %v", err)})
		return
	}

	latency := time.Since(startTime)
	finalResponse.LatencyMS = latency.Milliseconds()
	h.attachPlatformMetadata(&finalResponse, principal, governanceDecision, workspaceCtx, auditEventID)
	h.refreshOrchestrationMetadata(c.Request.Context(), req, &finalResponse)
	h.observeFinalResponse(req, prepared, finalResponse, latency)
	c.JSON(http.StatusOK, finalResponse)
}

func (h *GatewayHandler) HandleMetrics(c *gin.Context) {
	c.JSON(http.StatusOK, observability.Default().Snapshot())
}

func (h *GatewayHandler) HandleDashboards(c *gin.Context) {
	snapshot := observability.Default().Snapshot()
	c.JSON(http.StatusOK, snapshot["dashboards"])
}

func (h *GatewayHandler) HandleJobStatus(c *gin.Context) {
	if h.opsRuntime == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "async runtime is not configured"})
		return
	}
	job, err := h.opsRuntime.GetJob(c.Request.Context(), c.Param("id"))
	if err != nil || job == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "job not found"})
		return
	}
	c.JSON(http.StatusOK, job.StatusResponse())
}

func (h *GatewayHandler) HandleJobResult(c *gin.Context) {
	if h.opsRuntime == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "async runtime is not configured"})
		return
	}
	job, err := h.opsRuntime.GetJob(c.Request.Context(), c.Param("id"))
	if err != nil || job == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "job not found"})
		return
	}
	if job.Result == nil {
		c.JSON(http.StatusAccepted, job.StatusResponse())
		return
	}
	c.JSON(http.StatusOK, job.Result)
}

func (h *GatewayHandler) HandleCancelJob(c *gin.Context) {
	if h.opsRuntime == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "async runtime is not configured"})
		return
	}
	if err := h.opsRuntime.CancelJob(c.Request.Context(), c.Param("id")); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	job, _ := h.opsRuntime.GetJob(c.Request.Context(), c.Param("id"))
	c.JSON(http.StatusOK, job.StatusResponse())
}

func (h *GatewayHandler) HandleEvaluationRun(c *gin.Context) {
	var req api.EvaluationRunRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid evaluation request: " + err.Error()})
		return
	}

	generationReq, originalResp, err := h.resolveEvaluationRequest(c.Request.Context(), req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	generationReq.SyncOrAsyncPreference = "sync"
	generationReq.Evaluation.Enabled = true
	prepared, err := h.prepareExecutionContext(c.Request.Context(), generationReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	response, err := h.executeManagedSync(c.Request.Context(), generationReq, prepared)
	if err != nil {
		if requestErr, ok := err.(*requestFailure); ok {
			c.JSON(requestErr.StatusCode, gin.H{"error": requestErr.Message})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	comparison := &api.RolloutMetadata{}
	if originalResp != nil {
		comparison = ops.CompareResponses("evaluation_replay", *originalResp, response, req.OrchestrationID, 0, 0)
	}
	c.JSON(http.StatusOK, gin.H{
		"response":   response,
		"evaluation": response.Evaluation,
		"comparison": comparison,
	})
}

func (h *GatewayHandler) HandleReplayExecution(c *gin.Context) {
	req, originalResp, err := h.loadRecordedRequest(c.Request.Context(), c.Param("id"))
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	req.SyncOrAsyncPreference = "sync"
	prepared, err := h.prepareExecutionContext(c.Request.Context(), req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	response, err := h.executeManagedSync(c.Request.Context(), req, prepared)
	if err != nil {
		if requestErr, ok := err.(*requestFailure); ok {
			c.JSON(requestErr.StatusCode, gin.H{"error": requestErr.Message})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	comparison := (*api.RolloutMetadata)(nil)
	if originalResp != nil {
		comparison = ops.CompareResponses("offline_replay", *originalResp, response, c.Param("id"), 0, 0)
		response.Rollout = comparison
	}
	c.JSON(http.StatusOK, gin.H{
		"response":   response,
		"comparison": comparison,
	})
}

func (h *GatewayHandler) executeManagedSync(ctx context.Context, req api.GenerationRequest, prepared *llm.PreparedExecution) (api.GenerationResponse, error) {
	adjustedReq, canaryRollout := h.applyCanary(req, prepared)
	response, err := h.executePreparedSync(ctx, adjustedReq, prepared)
	if err != nil {
		return api.GenerationResponse{}, err
	}
	if canaryRollout != nil {
		response.Rollout = canaryRollout
	}
	if shadowRollout := h.runShadowIfConfigured(ctx, adjustedReq, prepared, response); shadowRollout != nil {
		if response.Rollout == nil {
			response.Rollout = shadowRollout
		} else {
			response.Rollout.Mode = joinModes(response.Rollout.Mode, shadowRollout.Mode)
			response.Rollout.ShadowModel = shadowRollout.ShadowModel
			response.Rollout.PrimaryLatencyMS = shadowRollout.PrimaryLatencyMS
			response.Rollout.ShadowLatencyMS = shadowRollout.ShadowLatencyMS
			response.Rollout.SimilarityScore = shadowRollout.SimilarityScore
			response.Rollout.OutputDelta = shadowRollout.OutputDelta
			response.Rollout.ReplaySourceID = shadowRollout.ReplaySourceID
			response.Rollout.Warnings = append(response.Rollout.Warnings, shadowRollout.Warnings...)
		}
	}
	if evaluation := h.evaluator.Evaluate(adjustedReq, prepared, response); evaluation != nil {
		response.Evaluation = evaluation
	}
	return response, nil
}

func (h *GatewayHandler) executePreparedSync(ctx context.Context, req api.GenerationRequest, prepared *llm.PreparedExecution) (api.GenerationResponse, error) {
	intent := h.intentAnalyzer.AnalyzeIntent(req.Prompt)
	switch intent {
	case llm.IntentImageCreation:
		return h.handleCreativeGeneration(ctx, req, prepared)
	case llm.IntentWeather, llm.IntentCalculator, llm.IntentNews:
		return h.handleToolLoop(ctx, req, prepared)
	default:
		if prepared != nil && prepared.Task.RequiresGeneration {
			return h.handleCreativeGeneration(ctx, req, prepared)
		}
		if prepared != nil && prepared.Task.NeedsTooling && len(prepared.Artifacts) == 0 {
			return h.handleToolLoop(ctx, req, prepared)
		}
		return h.handleTextGeneration(ctx, req, prepared)
	}
}

func (h *GatewayHandler) shouldRunAsync(req api.GenerationRequest, prepared *llm.PreparedExecution) bool {
	if strings.EqualFold(req.SyncOrAsyncPreference, "sync") {
		return false
	}
	if strings.EqualFold(req.SyncOrAsyncPreference, "async") {
		return true
	}
	if prepared == nil || prepared.Plan == nil {
		return false
	}
	if prepared.Plan.RequiresAsync {
		return true
	}
	return prepared.Task.ComplexityClass == "high" && (prepared.Task.RequiresGeneration || prepared.Task.RequiresTranscription || prepared.Task.RequiresOCR)
}

func (h *GatewayHandler) applyCanary(req api.GenerationRequest, prepared *llm.PreparedExecution) (api.GenerationRequest, *api.RolloutMetadata) {
	applied, modelID, bucket := h.canary.Decide(req)
	if !applied {
		return req, nil
	}
	if prepared != nil && !h.modelSupportsCapability(modelID, prepared.Task.PrimaryCapability) {
		return req, &api.RolloutMetadata{
			Mode:         "canary_skipped",
			AppliedModel: req.Config.ForceModel,
			Warnings:     []string{fmt.Sprintf("bucket %d selected canary model %s, but it does not support capability %s", bucket, modelID, prepared.Task.PrimaryCapability)},
		}
	}
	nextReq := req
	nextReq.Config.ForceModel = modelID
	if nextReq.Config.ForceScope == "" {
		nextReq.Config.ForceScope = forceScopeCapability
	}
	return nextReq, &api.RolloutMetadata{
		Mode:         "canary",
		AppliedModel: modelID,
		Warnings:     []string{fmt.Sprintf("deterministic canary bucket=%d", bucket)},
	}
}

func (h *GatewayHandler) runShadowIfConfigured(ctx context.Context, req api.GenerationRequest, prepared *llm.PreparedExecution, primary api.GenerationResponse) *api.RolloutMetadata {
	shadowModel := strings.TrimSpace(req.Rollout.ShadowModel)
	if shadowModel == "" {
		return nil
	}
	if prepared != nil && !h.modelSupportsCapability(shadowModel, prepared.Task.PrimaryCapability) {
		return &api.RolloutMetadata{
			Mode:        "shadow_skipped",
			Warnings:    []string{fmt.Sprintf("shadow model %s does not support capability %s", shadowModel, prepared.Task.PrimaryCapability)},
			ShadowModel: shadowModel,
		}
	}
	shadowReq := req
	shadowReq.SyncOrAsyncPreference = "sync"
	shadowReq.CallbackURL = ""
	shadowReq.Rollout = api.RolloutOptions{}
	shadowReq.Evaluation = api.EvaluationOptions{}
	shadowReq.Config.ForceModel = shadowModel
	if shadowReq.Config.ForceScope == "" {
		shadowReq.Config.ForceScope = forceScopeCapability
	}

	traceID := observability.Default().StartTrace("shadow_execution", shadowModel+"|"+req.ConversationID)
	start := time.Now()
	shadowResp, err := h.executePreparedSync(ctx, shadowReq, prepared)
	shadowLatency := time.Since(start)
	if err != nil {
		observability.Default().FinishTrace(traceID, "failed")
		return &api.RolloutMetadata{
			Mode:        "shadow_failed",
			ShadowModel: shadowModel,
			Warnings:    []string{err.Error()},
		}
	}
	observability.Default().RecordTraceSpan(traceID, "shadow.compare", "completed", map[string]string{
		"primary_model": primary.ModelUsed,
		"shadow_model":  shadowResp.ModelUsed,
	})
	observability.Default().FinishTrace(traceID, "completed")
	rollout := ops.CompareResponses("shadow", primary, shadowResp, "", time.Duration(primary.LatencyMS)*time.Millisecond, shadowLatency)
	rollout.ComparisonTraceID = traceID
	return rollout
}

func (h *GatewayHandler) observeFinalResponse(req api.GenerationRequest, prepared *llm.PreparedExecution, resp api.GenerationResponse, latency time.Duration) {
	observability.Default().ObserveLatency(latency)
	if resp.ModelUsed != "" {
		observability.Default().IncRequest(llm.ProviderForModel(resp.ModelUsed), resp.ModelUsed)
	}
	modalities := []string{"text"}
	taskType := "general"
	if prepared != nil {
		if len(prepared.Task.Modalities) > 0 {
			modalities = prepared.Task.Modalities
		}
		taskType = firstNonEmpty(prepared.Task.TaskType, taskType)
	}
	for _, modality := range modalities {
		observability.Default().RecordModality(modality)
	}
	if resp.Route != nil {
		observability.Default().RecordRoute(resp.Route.SelectedProvider, resp.Route.SelectedModel, resp.Route.Strategy, resp.Route.RouteFamily, resp.Route.ForcedSemantics.Scope)
	}
	cost := llm.EstimateRequestCost(resp.ModelUsed, resp.Usage, len(resp.GeneratedArtifacts))
	observability.Default().ObserveCost(llm.ProviderForModel(resp.ModelUsed), taskType, strings.Join(modalities, "+"), cost)
}

func (h *GatewayHandler) resolveEvaluationRequest(ctx context.Context, req api.EvaluationRunRequest) (api.GenerationRequest, *api.GenerationResponse, error) {
	if req.OrchestrationID != "" {
		return h.loadRecordedRequest(ctx, req.OrchestrationID)
	}
	if strings.TrimSpace(req.Request.Prompt) == "" {
		return api.GenerationRequest{}, nil, fmt.Errorf("evaluation request.prompt is required")
	}
	return req.Request, nil, nil
}

func (h *GatewayHandler) loadRecordedRequest(ctx context.Context, orchestrationID string) (api.GenerationRequest, *api.GenerationResponse, error) {
	if h.rdb == nil {
		return api.GenerationRequest{}, nil, fmt.Errorf("redis is not configured")
	}
	payload, err := h.rdb.Get(ctx, fmt.Sprintf("orchestration:%s", orchestrationID)).Bytes()
	if err == redis.Nil {
		return api.GenerationRequest{}, nil, fmt.Errorf("orchestration record %s not found", orchestrationID)
	}
	if err != nil {
		return api.GenerationRequest{}, nil, err
	}
	var record struct {
		Request  api.GenerationRequest   `json:"request"`
		Response *api.GenerationResponse `json:"response"`
	}
	if err := json.Unmarshal(payload, &record); err != nil {
		return api.GenerationRequest{}, nil, err
	}
	if strings.TrimSpace(record.Request.Prompt) == "" {
		return api.GenerationRequest{}, nil, fmt.Errorf("stored orchestration record does not include a replayable request")
	}
	return record.Request, record.Response, nil
}

func joinModes(current, next string) string {
	switch {
	case current == "":
		return next
	case next == "", current == next:
		return current
	default:
		return current + "+" + next
	}
}

func (h *GatewayHandler) handleTextGeneration(ctx context.Context, req api.GenerationRequest, prepared *llm.PreparedExecution) (api.GenerationResponse, error) {
	modelID, failoverInfo, session, routeSelection, err := h.determinePrimaryRoute(ctx, &req, prepared)
	if err != nil {
		return api.GenerationResponse{}, err
	}
	observability.Default().RecordLog("info", "routing", "selected primary text route", map[string]string{
		"model":    modelID,
		"strategy": firstNonEmpty(req.Config.Preference, "default"),
	})

	memorySnapshot, err := h.loadMemorySnapshot(ctx, req)
	if err != nil {
		return api.GenerationResponse{}, err
	}

	retrievalResult := &llm.RetrievalResult{}
	ragContextUsed := false
	if prepared == nil || prepared.Task.NeedsRetrieval {
		var retrievalErr error
		retrievalResult, _, ragContextUsed, retrievalErr = h.performRAGRetrieval(ctx, req.Prompt)
		if retrievalErr != nil {
			return api.GenerationResponse{}, retrievalErr
		}
	}
	if failoverInfo != nil {
		observability.Default().RecordLog("warn", "routing", "failover applied", map[string]string{
			"from": failoverInfo.OriginalModel,
			"to":   failoverInfo.NewModel,
		})
	}

	groundingDecision := h.decideGrounding(req, routeSelection, retrievalResult, nil)
	retrievalForContext := retrievalResult
	if !ragContextUsed {
		retrievalForContext = nil
	}
	composedContext := h.composeContext(req, memorySnapshot, retrievalForContext, nil, routeSelection, groundingDecision, modelID, prepared)
	if groundingDecision.ShouldAbstain {
		finalResponse := api.GenerationResponse{
			Content:        llm.FormatGroundedResponse("", retrievalResult, nil, groundingDecision),
			ModelUsed:      modelID,
			Usage:          api.Usage{},
			RAGContextUsed: ragContextUsed,
			CacheStatus:    "MISS",
			FailoverInfo:   failoverInfo,
			Route:          buildAPIRouteMetadata(routeSelection),
			Intent:         buildAPIIntentMetadata(routeSelection),
			Session:        buildAPISessionMetadata(session),
			Retrieval:      buildAPIRetrievalMetadata(retrievalResult),
			Grounding:      buildAPIGroundingMetadata(groundingDecision),
			Memory:         buildAPIMemoryMetadata(memorySnapshot),
			Context:        buildAPIContextMetadata(composedContext),
			Artifacts:      buildAPIArtifactMetadata(prepared),
			ExecutionPlan:  buildAPIExecutionPlanMetadata(prepared),
		}
		h.persistMemory(ctx, req, finalResponse, retrievalResult, routeSelection, groundingDecision, nil, session, prepared)
		orchestrationID := h.persistOrchestrationMetadata(ctx, req, &finalResponse)
		if finalResponse.Route != nil {
			finalResponse.Route.OrchestrationID = orchestrationID
		}
		return finalResponse, nil
	}

	cacheKey := h.ragService.BuildResponseCacheKeyWithContext(
		ctx,
		llm.GenerateCacheKey(req.Prompt),
		historySignature(req.History),
		modelID,
		effectiveRoutingMode(req.Config.Preference, routeSelection)+"|"+string(groundingDecision.Mode),
		retrievalResult.VersionSignature,
	)
	if cachedVal, found := h.ragService.CheckCache(ctx, cacheKey); found {
		var cachedResp api.GenerationResponse
		if json.Unmarshal([]byte(cachedVal), &cachedResp) == nil {
			observability.Default().IncCounter("response_cache_hits")
			cachedResp.CacheStatus = "HIT"
			cachedResp.ModelUsed = modelID
			cachedResp.FailoverInfo = failoverInfo
			cachedResp.Route = buildAPIRouteMetadata(routeSelection)
			cachedResp.Intent = buildAPIIntentMetadata(routeSelection)
			cachedResp.Session = buildAPISessionMetadata(session)
			cachedResp.Retrieval = buildAPIRetrievalMetadata(retrievalResult)
			cachedResp.Grounding = buildAPIGroundingMetadata(groundingDecision)
			cachedResp.Memory = buildAPIMemoryMetadata(memorySnapshot)
			cachedResp.Context = buildAPIContextMetadata(composedContext)
			cachedResp.Artifacts = buildAPIArtifactMetadata(prepared)
			cachedResp.ExecutionPlan = buildAPIExecutionPlanMetadata(prepared)
			h.persistMemory(ctx, req, cachedResp, retrievalResult, routeSelection, groundingDecision, nil, session, prepared)
			orchestrationID := h.persistOrchestrationMetadata(ctx, req, &cachedResp)
			if cachedResp.Route != nil {
				cachedResp.Route.OrchestrationID = orchestrationID
			}
			return cachedResp, nil
		}
	}
	observability.Default().IncCounter("response_cache_misses")

	finalContent, usage, err := h.executeGeneration(ctx, req, modelID, composedContext)
	if err != nil {
		return api.GenerationResponse{}, err
	}
	finalContent = llm.FormatGroundedResponse(finalContent, retrievalResult, nil, groundingDecision)

	finalResponse := api.GenerationResponse{
		Content:        finalContent,
		ModelUsed:      modelID,
		Usage:          usage,
		RAGContextUsed: ragContextUsed,
		CacheStatus:    "MISS",
		FailoverInfo:   failoverInfo,
		Route:          buildAPIRouteMetadata(routeSelection),
		Intent:         buildAPIIntentMetadata(routeSelection),
		Session:        buildAPISessionMetadata(session),
		Retrieval:      buildAPIRetrievalMetadata(retrievalResult),
		Grounding:      buildAPIGroundingMetadata(groundingDecision),
		Memory:         buildAPIMemoryMetadata(memorySnapshot),
		Context:        buildAPIContextMetadata(composedContext),
		Artifacts:      buildAPIArtifactMetadata(prepared),
		ExecutionPlan:  buildAPIExecutionPlanMetadata(prepared),
	}
	h.persistMemory(ctx, req, finalResponse, retrievalResult, routeSelection, groundingDecision, nil, session, prepared)
	orchestrationID := h.persistOrchestrationMetadata(ctx, req, &finalResponse)
	if finalResponse.Route != nil {
		finalResponse.Route.OrchestrationID = orchestrationID
	}

	respBytes, err := json.Marshal(finalResponse)
	if err != nil {
		log.Printf("WARNING: Failed to marshal response for caching: %v", err)
	} else {
		h.ragService.SetCache(ctx, cacheKey, string(respBytes))
	}

	return finalResponse, nil
}

func (h *GatewayHandler) handleCreativeGeneration(ctx context.Context, req api.GenerationRequest, prepared *llm.PreparedExecution) (api.GenerationResponse, error) {
	if h.generation == nil {
		return api.GenerationResponse{}, &requestFailure{StatusCode: http.StatusServiceUnavailable, Message: "generation runtime is not configured"}
	}

	result, err := h.generation.Execute(ctx, req, prepared)
	if err != nil {
		return api.GenerationResponse{}, &requestFailure{StatusCode: http.StatusBadGateway, Message: err.Error()}
	}
	observability.Default().RecordLog("info", "generation", "creative pipeline completed", map[string]string{
		"pipeline": result.Pipeline,
		"model":    result.ModelUsed,
	})

	response := api.GenerationResponse{
		Content:            result.Content,
		ImageURL:           result.ImageURL,
		AudioURL:           result.AudioURL,
		ModelUsed:          result.ModelUsed,
		CacheStatus:        "MISS",
		Route:              buildAPIGenerationRouteMetadata(result, req, prepared),
		Session:            buildAPISessionMetadataForGeneration(req, result),
		Artifacts:          buildAPIArtifactMetadata(prepared),
		GeneratedArtifacts: buildAPIArtifactMetadataFromRecords(result.OutputArtifacts),
		ExecutionPlan:      buildAPIExecutionPlanMetadata(prepared),
		Generation:         buildAPIGenerationMetadata(result),
	}
	orchestrationID := h.persistOrchestrationMetadata(ctx, req, &response)
	if response.Route != nil {
		response.Route.OrchestrationID = orchestrationID
	}
	return response, nil
}

func (h *GatewayHandler) determineModelID(ctx context.Context, req *api.GenerationRequest) (string, *api.FailoverInfo, *sessionState, *llm.RouteSelection, error) {
	currentSession, err := h.loadSession(ctx, req.ConversationID)
	if err != nil {
		return "", nil, nil, nil, err
	}

	if req.Config.ForceModel != "" {
		return h.resolveForcedModel(ctx, req, currentSession)
	}

	if currentSession != nil && currentSession.IsForced {
		activeModelID := currentEffectiveModelID(currentSession)
		if err := h.ensureModelAvailable(ctx, activeModelID); err == nil {
			h.refreshSessionTTL(ctx, currentSession.ConversationID)
			observability.Default().IncCounter("session_mode_forced")
			routeSelection, routeErr := h.explainForcedTextRoute(ctx, req, activeModelID, forceMetadataFromSession(currentSession))
			if routeErr != nil {
				return "", nil, currentSession, nil, routeErr
			}
			return activeModelID, nil, currentSession, routeSelection, nil
		}

		if currentSession.StrictForce {
			return "", nil, currentSession, nil, &requestFailure{
				StatusCode: http.StatusFailedDependency,
				Message:    fmt.Sprintf("The forced model '%s' is unavailable and strict force mode forbids fallback.", activeModelID),
				Details: gin.H{
					"available_models": h.availableTextModels(ctx, activeModelID),
					"force_scope":      currentSession.ForceScope,
				},
			}
		}

		previousModel := activeModelID
		routeSelection, routeErr := h.routeTextModel(ctx, req, forceMetadataFromSession(currentSession))
		if routeErr != nil {
			return "", nil, currentSession, nil, routeErr
		}
		fallbackModel := routeSelection.ModelID
		currentSession.FailoverCount++
		currentSession.LastFailoverAt = time.Now().UTC()
		currentSession.EffectiveModelID = fallbackModel
		if err := h.saveSession(ctx, currentSession); err != nil {
			log.Printf("WARNING: failed to persist failover session state: %v", err)
		}
		observability.Default().IncCounter("failover_count")
		routeSelection.Explanation.SelectionReason = fmt.Sprintf("selected as forced-session failover because %s became unavailable", previousModel)
		return fallbackModel, &api.FailoverInfo{
			OriginalModel: previousModel,
			NewModel:      fallbackModel,
			Reason:        fmt.Sprintf("Pinned model became unavailable; failed over to %s.", fallbackModel),
		}, currentSession, routeSelection, nil
	}

	routeSelection, err := h.routeTextModel(ctx, req, llm.ForceMetadata{})
	if err != nil {
		return "", nil, currentSession, nil, err
	}
	modelID := routeSelection.ModelID

	if req.ConversationID != "" {
		nextSession := currentSession
		if nextSession == nil {
			nextSession = &sessionState{ConversationID: req.ConversationID}
		}
		nextSession.ModelID = modelID
		nextSession.EffectiveModelID = modelID
		nextSession.IsForced = false
		nextSession.ForceScope = ""
		nextSession.StrictForce = false
		nextSession.SessionModeVersion = sessionModeVersion
		if err := h.saveSession(ctx, nextSession); err != nil {
			log.Printf("WARNING: failed to persist session state: %v", err)
		}
		currentSession = nextSession
	}
	observability.Default().IncCounter("session_mode_dynamic")
	return modelID, nil, currentSession, routeSelection, nil
}

func (h *GatewayHandler) resolveForcedModel(ctx context.Context, req *api.GenerationRequest, currentSession *sessionState) (string, *api.FailoverInfo, *sessionState, *llm.RouteSelection, error) {
	requestedModel := req.Config.ForceModel
	forceScope := normalizeForceScope(req.Config.ForceScope)
	strictForce := req.Config.StrictForce || forceScope == forceScopeStrictEndToEnd
	forceMeta := llm.ForceMetadata{
		IsForced:       true,
		Scope:          forceScope,
		Strict:         strictForce,
		PinnedModel:    requestedModel,
		EffectiveModel: requestedModel,
	}
	if err := h.ensureModelAvailable(ctx, requestedModel); err != nil {
		alternatives := h.availableTextModels(ctx, requestedModel)
		if strictForce || !req.Config.ForceIfAvailableElseFallback {
			return "", nil, currentSession, nil, &requestFailure{
				StatusCode: http.StatusFailedDependency,
				Message:    fmt.Sprintf("The requested model '%s' is currently unavailable.", requestedModel),
				Details: gin.H{
					"available_models": alternatives,
					"force_scope":      forceScope,
					"strict_force":     strictForce,
				},
			}
		}

		forceMeta.EffectiveModel = ""
		routeSelection, routeErr := h.routeTextModel(ctx, req, forceMeta)
		if routeErr != nil {
			return "", nil, currentSession, nil, routeErr
		}
		fallbackModel := routeSelection.ModelID
		nextSession := h.nextForcedSession(req.ConversationID, currentSession, requestedModel, fallbackModel)
		nextSession.ForceScope = forceScope
		nextSession.StrictForce = strictForce
		nextSession.ModelID = fallbackModel
		nextSession.EffectiveModelID = fallbackModel
		if err := h.saveSession(ctx, nextSession); err != nil {
			log.Printf("WARNING: failed to persist forced fallback session state: %v", err)
		}
		observability.Default().IncCounter("forced_override_count")
		routeSelection.Explanation.SelectionReason = fmt.Sprintf("requested forced model %s was unavailable; selected healthy fallback", requestedModel)
		return fallbackModel, &api.FailoverInfo{
			OriginalModel: requestedModel,
			NewModel:      fallbackModel,
			Reason:        fmt.Sprintf("Requested forced model '%s' was unavailable; routed to healthy fallback '%s'.", requestedModel, fallbackModel),
		}, nextSession, routeSelection, nil
	}

	nextSession := h.nextForcedSession(req.ConversationID, currentSession, currentEffectiveModelID(currentSession), requestedModel)
	nextSession.ForceScope = forceScope
	nextSession.StrictForce = strictForce
	nextSession.ModelID = requestedModel
	nextSession.EffectiveModelID = requestedModel
	if err := h.saveSession(ctx, nextSession); err != nil {
		log.Printf("WARNING: failed to persist forced session state: %v", err)
	}
	observability.Default().IncCounter("forced_override_count")
	routeSelection, routeErr := h.explainForcedTextRoute(ctx, req, requestedModel, forceMetadataFromSession(nextSession))
	if routeErr != nil {
		return "", nil, nextSession, nil, routeErr
	}
	return requestedModel, nil, nextSession, routeSelection, nil
}

func (h *GatewayHandler) nextForcedSession(conversationID string, currentSession *sessionState, overrideFrom, overrideTo string) *sessionState {
	next := &sessionState{ConversationID: conversationID, SessionModeVersion: sessionModeVersion, IsForced: true}
	if currentSession != nil {
		*next = *currentSession
		next.ConversationID = conversationID
	}
	next.ModelID = overrideTo
	next.EffectiveModelID = overrideTo
	next.IsForced = true
	next.OverrideCount++
	next.LastOverrideAt = time.Now().UTC()
	next.LastOverrideFrom = overrideFrom
	next.LastOverrideTo = overrideTo
	next.SessionModeVersion = sessionModeVersion
	return next
}

func (h *GatewayHandler) routeTextModel(ctx context.Context, req *api.GenerationRequest, forceMeta llm.ForceMetadata) (*llm.RouteSelection, error) {
	if req.Config.Preference == "" {
		req.Config.Preference = h.promptAnalyzer.Analyze(req.Prompt)
	}
	estimatedTokens := estimatePromptTokens(req.Prompt, req.History)
	availableModels := h.config.EnabledModels
	if h.policyEngine != nil {
		filtered, _, decision, err := h.policyEngine.FilterModels(ctx, *req, availableModels, llm.CapabilityTextGeneration, estimatedTokens)
		if err != nil {
			return nil, &requestFailure{StatusCode: http.StatusForbidden, Message: err.Error(), Details: gin.H{"governance": decision}}
		}
		availableModels = filtered
	}
	if h.controlPlane == nil {
		selection, err := h.router.SelectOptimalRoute(ctx, availableModels, req.Config.Preference, estimatedTokens, h.config.ModelBudgets, forceMeta)
		if err != nil {
			return nil, &requestFailure{StatusCode: http.StatusServiceUnavailable, Message: err.Error(), Details: gin.H{"available_models": h.availableTextModels(ctx, "")}}
		}
		return selection, nil
	}
	selection, err := h.controlPlane.PlanRoute(ctx, llm.RoutePlanningRequest{
		Prompt:          req.Prompt,
		History:         convertAPIMessagesToLLMMessages(req.History),
		ConversationID:  req.ConversationID,
		Preference:      req.Config.Preference,
		AvailableModels: availableModels,
		ModelBudgets:    h.config.ModelBudgets,
		PromptTokens:    estimatedTokens,
		Capability:      llm.CapabilityTextGeneration,
		ForceMetadata:   forceMeta,
	})
	if err != nil {
		return nil, &requestFailure{StatusCode: http.StatusServiceUnavailable, Message: err.Error(), Details: gin.H{"available_models": h.availableTextModels(ctx, "")}}
	}
	return selection, nil
}

func (h *GatewayHandler) explainForcedTextRoute(ctx context.Context, req *api.GenerationRequest, selectedModel string, forceMeta llm.ForceMetadata) (*llm.RouteSelection, error) {
	if req.Config.Preference == "" {
		req.Config.Preference = h.promptAnalyzer.Analyze(req.Prompt)
	}
	estimatedTokens := estimatePromptTokens(req.Prompt, req.History)
	if h.controlPlane == nil {
		return h.router.ExplainForcedRoute(ctx, selectedModel, h.config.EnabledModels, req.Config.Preference, estimatedTokens, h.config.ModelBudgets, forceMeta)
	}
	selection, err := h.controlPlane.ExplainForcedRoute(ctx, selectedModel, llm.RoutePlanningRequest{
		Prompt:          req.Prompt,
		History:         convertAPIMessagesToLLMMessages(req.History),
		ConversationID:  req.ConversationID,
		Preference:      req.Config.Preference,
		AvailableModels: h.config.EnabledModels,
		ModelBudgets:    h.config.ModelBudgets,
		PromptTokens:    estimatedTokens,
		Capability:      llm.CapabilityTextGeneration,
		ForceMetadata:   forceMeta,
	})
	if err != nil {
		return nil, err
	}
	return selection, nil
}

func (h *GatewayHandler) ensureModelAvailable(ctx context.Context, modelID string) error {
	if modelID == "" {
		return fmt.Errorf("model is empty")
	}
	if _, ok := h.clients[modelID]; !ok {
		return fmt.Errorf("model '%s' is not enabled", modelID)
	}
	effectiveHealth, providerHealth, capabilityHealth, profile, err := h.profiler.CombinedModelHealth(ctx, modelID, llm.CapabilityTextGeneration)
	if err != nil {
		return err
	}
	if !providerHealth.AccessAllowed || !capabilityHealth.AccessAllowed || !capabilityHealth.Supported {
		return fmt.Errorf("model '%s' cannot serve the requested capability", modelID)
	}
	if effectiveHealth == llm.HealthStatusOffline {
		return fmt.Errorf("model '%s' is offline", modelID)
	}
	staleness := h.profilerHealthStaleness()
	if !profile.LastHealthCheck.IsZero() && time.Since(profile.LastHealthCheck) > staleness {
		return fmt.Errorf("model '%s' health check is stale", modelID)
	}
	if !providerHealth.LastHealthCheck.IsZero() && time.Since(providerHealth.LastHealthCheck) > staleness {
		return fmt.Errorf("provider health for '%s' is stale", modelID)
	}
	if !capabilityHealth.LastHealthCheck.IsZero() && time.Since(capabilityHealth.LastHealthCheck) > staleness {
		return fmt.Errorf("capability health for '%s' is stale", modelID)
	}
	if !profile.CircuitOpenUntil.IsZero() && time.Now().UTC().Before(profile.CircuitOpenUntil) {
		return fmt.Errorf("model '%s' circuit is open", modelID)
	}
	return nil
}

func (h *GatewayHandler) availableTextModels(ctx context.Context, exclude string) []string {
	models := make([]string, 0, len(h.config.EnabledModels))
	for _, modelID := range h.config.EnabledModels {
		if modelID == "" || modelID == exclude {
			continue
		}
		if err := h.ensureModelAvailable(ctx, modelID); err == nil {
			models = append(models, modelID)
		}
	}
	return models
}

func (h *GatewayHandler) saveSession(ctx context.Context, session *sessionState) error {
	if session == nil || session.ConversationID == "" {
		return nil
	}
	sessionKey := fmt.Sprintf("session:%s", session.ConversationID)
	fields := map[string]interface{}{
		"model_id":             session.ModelID,
		"effective_model_id":   session.EffectiveModelID,
		"is_forced":            session.IsForced,
		"force_scope":          session.ForceScope,
		"strict_force":         session.StrictForce,
		"override_count":       session.OverrideCount,
		"last_override_at":     formatTime(session.LastOverrideAt),
		"last_override_from":   session.LastOverrideFrom,
		"last_override_to":     session.LastOverrideTo,
		"session_mode_version": session.SessionModeVersion,
		"failover_count":       session.FailoverCount,
		"last_failover_at":     formatTime(session.LastFailoverAt),
	}
	if err := h.rdb.HSet(ctx, sessionKey, fields).Err(); err != nil {
		return err
	}
	h.refreshSessionTTL(ctx, session.ConversationID)
	log.Printf("📌 Session %s pinned=%s effective=%s (forced=%v scope=%s strict=%v overrides=%d failovers=%d)", session.ConversationID, session.ModelID, session.EffectiveModelID, session.IsForced, session.ForceScope, session.StrictForce, session.OverrideCount, session.FailoverCount)
	return nil
}

func (h *GatewayHandler) loadSession(ctx context.Context, conversationID string) (*sessionState, error) {
	if conversationID == "" {
		return nil, nil
	}
	sessionKey := fmt.Sprintf("session:%s", conversationID)
	data, err := h.rdb.HGetAll(ctx, sessionKey).Result()
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}
	return &sessionState{
		ConversationID:     conversationID,
		ModelID:            data["model_id"],
		EffectiveModelID:   firstNonEmpty(data["effective_model_id"], data["model_id"]),
		IsForced:           parseBoolValue(data["is_forced"]),
		ForceScope:         storedForceScope(data["force_scope"]),
		StrictForce:        parseBoolValue(data["strict_force"]),
		OverrideCount:      parseIntValue(data["override_count"]),
		LastOverrideAt:     parseTimestamp(data["last_override_at"]),
		LastOverrideFrom:   data["last_override_from"],
		LastOverrideTo:     data["last_override_to"],
		SessionModeVersion: parseIntValue(data["session_mode_version"]),
		FailoverCount:      parseIntValue(data["failover_count"]),
		LastFailoverAt:     parseTimestamp(data["last_failover_at"]),
	}, nil
}

func (h *GatewayHandler) refreshSessionTTL(ctx context.Context, conversationID string) {
	if conversationID == "" {
		return
	}
	h.rdb.Expire(ctx, fmt.Sprintf("session:%s", conversationID), time.Hour)
}

func (h *GatewayHandler) executeGeneration(ctx context.Context, req api.GenerationRequest, modelID string, composedContext *llm.ComposedContext) (string, api.Usage, error) {
	client := h.clients[modelID]
	if client == nil {
		return "", api.Usage{}, &requestFailure{StatusCode: http.StatusServiceUnavailable, Message: fmt.Sprintf("no client available for model %s", modelID)}
	}

	messages := convertAPIMessagesToLLMMessages(req.History)
	if composedContext != nil && strings.TrimSpace(composedContext.SystemPrompt) != "" {
		messages = append([]llm.Message{{Role: llm.RoleSystem, Content: composedContext.SystemPrompt}}, messages...)
	}
	userPrompt := req.Prompt
	if composedContext != nil && strings.TrimSpace(composedContext.UserPrompt) != "" {
		userPrompt = composedContext.UserPrompt
	}
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: userPrompt})
	llmConfig := &llm.GenerationConfig{Model: modelID, MaxTokens: req.Config.MaxTokens, Temperature: req.Config.Temperature, TopP: req.Config.TopP, Stream: req.Config.Stream}

	start := time.Now()
	result, err := client.Generate(ctx, messages, llmConfig, nil)
	if err != nil {
		h.profiler.UpdateProfileOnFailure(ctx, modelID, err)
		observability.Default().IncError(llm.ProviderForModel(modelID), modelID)
		return "", api.Usage{}, &requestFailure{StatusCode: http.StatusBadGateway, Message: fmt.Sprintf("LLM generation failed for model %s: %v", modelID, err)}
	}
	h.profiler.UpdateProfileOnSuccess(ctx, modelID, time.Since(start), result.Usage)
	return result.Content, result.Usage, nil
}

func (h *GatewayHandler) performRAGRetrieval(ctx context.Context, prompt string) (*llm.RetrievalResult, string, bool, error) {
	retrievalResult, err := h.ragService.RetrieveContext(ctx, prompt, h.config.RAGConfig.RetrievalTopK)
	if err != nil {
		observability.Default().IncCounter("rag_failures")
		if h.config.RAGConfig.RAGFailurePolicy == "graceful_no_rag" {
			log.Printf("WARNING: RAG retrieval failed, continuing without grounding: %v", err)
			return &llm.RetrievalResult{}, prompt, false, nil
		}
		return nil, "", false, &requestFailure{StatusCode: http.StatusServiceUnavailable, Message: fmt.Sprintf("RAG retrieval failed: %v", err)}
	}

	observability.Default().ObserveRetrievalLatency(retrievalResult.RetrievalLatency)
	observability.Default().RecordLog("info", "retrieval", "retrieval completed", map[string]string{
		"retrieval_id": retrievalResult.RetrievalID,
		"sources":      strconv.Itoa(retrievalResult.SelectedCount),
	})
	if len(retrievalResult.StaleSources) > 0 {
		observability.Default().AddCounter("rag_stale_source_skips", int64(len(retrievalResult.StaleSources)))
	}
	threshold, _ := h.config.RouterConfig.Thresholds["relevance_threshold"].(float64)
	if retrievalResult.Context != "" && retrievalResult.Score >= threshold {
		observability.Default().IncCounter("rag_hits")
		return retrievalResult, fmt.Sprintf("Using the following retrieved context, answer the question. If the context is insufficient, say so clearly.\n\nContext:\n%s\n\nQuestion: %s", retrievalResult.Context, prompt), true, nil
	}
	observability.Default().IncCounter("rag_misses")
	return retrievalResult, prompt, false, nil
}

func (h *GatewayHandler) loadMemorySnapshot(ctx context.Context, req api.GenerationRequest) (*llm.SessionMemorySnapshot, error) {
	if h.memoryEngine == nil {
		return nil, nil
	}
	return h.memoryEngine.LoadSnapshot(ctx, req.ConversationID, convertAPIMessagesToLLMMessages(req.History))
}

func (h *GatewayHandler) decideGrounding(req api.GenerationRequest, routeSelection *llm.RouteSelection, retrieval *llm.RetrievalResult, toolResults []llm.ContextToolResult) llm.GroundingDecision {
	if h.groundingEngine == nil {
		return llm.GroundingDecision{Mode: llm.AnswerModeBestEffort, EvidenceStatus: llm.EvidenceStatusInsufficient}
	}
	intent := llm.ExecutionIntent{}
	if routeSelection != nil {
		intent = routeSelection.Explanation.Intent
	}
	return h.groundingEngine.Decide(req.Config.AnswerMode, intent, retrieval, toolResults)
}

func (h *GatewayHandler) composeContext(req api.GenerationRequest, memory *llm.SessionMemorySnapshot, retrieval *llm.RetrievalResult, toolResults []llm.ContextToolResult, routeSelection *llm.RouteSelection, decision llm.GroundingDecision, modelID string, prepared *llm.PreparedExecution) *llm.ComposedContext {
	if h.contextComposer == nil {
		return &llm.ComposedContext{UserPrompt: req.Prompt}
	}
	modelLimit := 0
	if metadata, ok := h.config.RouterConfig.Models[modelID]; ok {
		modelLimit = metadata.ContextLimit
	}
	return h.contextComposer.Compose(llm.ContextComposerInput{
		UserPrompt:        req.Prompt,
		Memory:            memory,
		Retrieval:         retrieval,
		ToolResults:       toolResults,
		RouteSelection:    routeSelection,
		GroundingDecision: decision,
		ForceMetadata:     routeSelectionForceMetadata(routeSelection),
		ModelContextLimit: modelLimit,
		Artifacts:         preparedArtifacts(prepared),
		ExecutionPlan:     preparedPlan(prepared),
	})
}

func routeSelectionForceMetadata(selection *llm.RouteSelection) llm.ForceMetadata {
	if selection == nil {
		return llm.ForceMetadata{}
	}
	return selection.Explanation.ForcedSemantics
}

func (h *GatewayHandler) persistMemory(ctx context.Context, req api.GenerationRequest, resp api.GenerationResponse, retrieval *llm.RetrievalResult, routeSelection *llm.RouteSelection, decision llm.GroundingDecision, toolRecords []tools.ExecutionRecord, session *sessionState, prepared *llm.PreparedExecution) {
	if h.memoryEngine == nil {
		return
	}
	sessionMode := "dynamic"
	forceScope := ""
	effectiveModel := resp.ModelUsed
	if session != nil && session.IsForced {
		sessionMode = "forced"
		forceScope = session.ForceScope
		effectiveModel = currentEffectiveModelID(session)
	}
	var toolCalls []llm.MemoryToolCall
	for _, record := range toolRecords {
		toolCalls = append(toolCalls, llm.MemoryToolCall{
			Name:   record.Name,
			Result: firstNonEmpty(record.Result, record.Error),
			Status: record.Status,
		})
	}
	artifactLinks := preparedArtifactLinks(prepared)
	if _, err := h.memoryEngine.UpdateSession(ctx, llm.MemoryUpdate{
		ConversationID:  req.ConversationID,
		UserPrompt:      req.Prompt,
		ResponseContent: resp.Content,
		ToolCalls:       toolCalls,
		Retrieval:       retrieval,
		Route:           routeSelection,
		SessionMode:     sessionMode,
		ForceScope:      forceScope,
		EffectiveModel:  effectiveModel,
		AnswerMode:      string(decision.Mode),
		ArtifactLinks:   artifactLinks,
	}); err != nil {
		log.Printf("WARNING: failed to persist session memory: %v", err)
	}
}

func (h *GatewayHandler) routeToolModel(ctx context.Context, req *api.GenerationRequest) (*llm.RouteSelection, error) {
	availableModels := make([]string, 0, len(h.config.EnabledModels))
	for _, modelID := range h.config.EnabledModels {
		if metadata, ok := h.config.RouterConfig.Models[modelID]; ok && metadata.ToolSupport {
			availableModels = append(availableModels, modelID)
		}
	}
	if len(availableModels) == 0 {
		return nil, &requestFailure{StatusCode: http.StatusServiceUnavailable, Message: "no tool-capable models are enabled"}
	}
	estimatedTokens := estimatePromptTokens(req.Prompt, req.History)
	if h.controlPlane == nil {
		selection, err := h.router.SelectOptimalRoute(ctx, availableModels, firstNonEmpty(req.Config.Preference, "balanced"), estimatedTokens, h.config.ModelBudgets, llm.ForceMetadata{})
		if err != nil {
			return nil, &requestFailure{StatusCode: http.StatusServiceUnavailable, Message: err.Error(), Details: gin.H{"available_models": availableModels}}
		}
		return selection, nil
	}
	selection, err := h.controlPlane.PlanRoute(ctx, llm.RoutePlanningRequest{
		Prompt:          req.Prompt,
		History:         convertAPIMessagesToLLMMessages(req.History),
		ConversationID:  req.ConversationID,
		Preference:      firstNonEmpty(req.Config.Preference, "balanced"),
		AvailableModels: availableModels,
		ModelBudgets:    h.config.ModelBudgets,
		PromptTokens:    estimatedTokens,
		Capability:      llm.CapabilityTextGeneration,
	})
	if err != nil {
		return nil, &requestFailure{StatusCode: http.StatusServiceUnavailable, Message: err.Error(), Details: gin.H{"available_models": availableModels}}
	}
	return selection, nil
}

func (h *GatewayHandler) handleToolLoop(ctx context.Context, req api.GenerationRequest, prepared *llm.PreparedExecution) (api.GenerationResponse, error) {
	const maxToolCalls = 5
	var cumulativeUsage api.Usage
	toolPlan := h.toolManager.Plan(req.Prompt)
	routeSelection, err := h.routeToolModel(ctx, &req)
	if err != nil {
		return api.GenerationResponse{}, err
	}
	modelID := routeSelection.ModelID
	client, ok := h.clients[modelID]
	if !ok {
		return api.GenerationResponse{}, &requestFailure{StatusCode: http.StatusServiceUnavailable, Message: fmt.Sprintf("tool-use model '%s' is not available or enabled", modelID)}
	}

	memorySnapshot, err := h.loadMemorySnapshot(ctx, req)
	if err != nil {
		return api.GenerationResponse{}, err
	}

	messages := convertAPIMessagesToLLMMessages(req.History)
	initialGrounding := h.decideGrounding(req, routeSelection, nil, nil)
	if initialGrounding.Instruction != "" {
		messages = append([]llm.Message{{Role: llm.RoleSystem, Content: initialGrounding.Instruction}}, messages...)
	}
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: req.Prompt})
	llmConfig := &llm.GenerationConfig{Model: modelID, MaxTokens: req.Config.MaxTokens, Temperature: req.Config.Temperature, TopP: req.Config.TopP, Stream: req.Config.Stream}
	definitions := h.toolManager.DefinitionsForPlan(toolPlan)
	var executionRecords []tools.ExecutionRecord

	for i := 0; i < maxToolCalls; i++ {
		start := time.Now()
		result, err := client.Generate(ctx, messages, llmConfig, definitions)
		if err != nil {
			h.profiler.UpdateProfileOnFailure(ctx, modelID, err)
			observability.Default().IncError(llm.ProviderForModel(modelID), modelID)
			return api.GenerationResponse{}, &requestFailure{StatusCode: http.StatusInternalServerError, Message: fmt.Sprintf("LLM generation failed during tool loop: %v", err)}
		}
		h.profiler.UpdateProfileOnSuccess(ctx, modelID, time.Since(start), result.Usage)
		cumulativeUsage.Add(result.Usage)

		if len(result.ToolCalls) == 0 {
			toolResults := toolResultsForContext(executionRecords)
			groundingDecision := h.decideGrounding(req, routeSelection, nil, toolResults)
			finalContent := llm.FormatGroundedResponse(result.Content, nil, toolResults, groundingDecision)
			response := api.GenerationResponse{
				Content:       finalContent,
				ModelUsed:     modelID,
				Usage:         cumulativeUsage,
				CacheStatus:   "MISS",
				Route:         buildAPIRouteMetadata(routeSelection),
				Intent:        buildAPIIntentMetadata(routeSelection),
				ToolPlan:      buildAPIToolPlanMetadata(toolPlan),
				Grounding:     buildAPIGroundingMetadata(groundingDecision),
				Memory:        buildAPIMemoryMetadata(memorySnapshot),
				ToolCalls:     buildAPIToolCalls(executionRecords),
				Artifacts:     buildAPIArtifactMetadata(prepared),
				ExecutionPlan: buildAPIExecutionPlanMetadata(prepared),
			}
			h.persistMemory(ctx, req, response, nil, routeSelection, groundingDecision, executionRecords, nil, prepared)
			orchestrationID := h.persistOrchestrationMetadata(ctx, req, &response)
			if response.Route != nil {
				response.Route.OrchestrationID = orchestrationID
			}
			return response, nil
		}

		messages = append(messages, llm.Message{Role: llm.RoleAssistant, Content: result.Content, ToolCalls: result.ToolCalls})
		for _, toolCall := range result.ToolCalls {
			record, execErr := h.toolManager.ExecuteWithContext(ctx, toolCall.Function.Name, toolCall.Function.Arguments)
			executionRecords = append(executionRecords, record)
			observability.Default().RecordLog("info", "tools", "executed tool call", map[string]string{
				"tool":   toolCall.Function.Name,
				"status": firstNonEmpty(record.Status, "unknown"),
			})
			toolResult := record.Result
			if execErr != nil {
				toolResult = fmt.Sprintf("Error executing tool %s: %v", toolCall.Function.Name, execErr)
			}
			messages = append(messages, llm.Message{Role: llm.RoleTool, ToolCallID: toolCall.ID, Content: toolResult})
		}
	}

	return api.GenerationResponse{}, &requestFailure{StatusCode: http.StatusRequestTimeout, Message: "exceeded maximum number of tool calls"}
}

func (h *GatewayHandler) prepareExecutionContext(ctx context.Context, req api.GenerationRequest) (*llm.PreparedExecution, error) {
	if h.multimodal == nil {
		return nil, nil
	}
	forceMeta := requestForceMetadata(req.Config)
	if !forceMeta.IsForced && req.ConversationID != "" {
		session, err := h.loadSession(ctx, req.ConversationID)
		if err != nil {
			return nil, err
		}
		forceMeta = forceMetadataFromSession(session)
	}
	return h.multimodal.Prepare(ctx, req, convertAPIMessagesToLLMMessages(req.History), forceMeta)
}

func (h *GatewayHandler) determinePrimaryRoute(ctx context.Context, req *api.GenerationRequest, prepared *llm.PreparedExecution) (string, *api.FailoverInfo, *sessionState, *llm.RouteSelection, error) {
	if prepared == nil || prepared.Task.PrimaryCapability == "" || prepared.Task.PrimaryCapability == llm.CapabilityTextGeneration {
		return h.determineModelID(ctx, req)
	}
	currentSession, err := h.loadSession(ctx, req.ConversationID)
	if err != nil {
		return "", nil, nil, nil, err
	}

	forceMeta := requestForceMetadata(req.Config)
	if !forceMeta.IsForced {
		forceMeta = forceMetadataFromSession(currentSession)
	}

	return h.routeCapabilityModel(ctx, req, prepared.Task.PrimaryCapability, currentSession, forceMeta)
}

func (h *GatewayHandler) routeCapabilityModel(ctx context.Context, req *api.GenerationRequest, capability llm.Capability, session *sessionState, forceMeta llm.ForceMetadata) (string, *api.FailoverInfo, *sessionState, *llm.RouteSelection, error) {
	if req.Config.Preference == "" {
		req.Config.Preference = h.promptAnalyzer.Analyze(req.Prompt)
	}
	estimatedTokens := estimatePromptTokens(req.Prompt, req.History)
	availableModels := h.config.EnabledModels
	if h.policyEngine != nil {
		filtered, _, decision, err := h.policyEngine.FilterModels(ctx, *req, availableModels, capability, estimatedTokens)
		if err != nil {
			return "", nil, session, nil, &requestFailure{StatusCode: http.StatusForbidden, Message: err.Error(), Details: gin.H{"governance": decision}}
		}
		availableModels = filtered
	}

	if forceMeta.IsForced && forceMeta.PinnedModel != "" {
		compatible := h.modelSupportsCapability(forceMeta.PinnedModel, capability)
		available := h.ensureCapabilityModelAvailable(ctx, forceMeta.PinnedModel, capability) == nil
		if compatible && available {
			selection, err := h.controlPlane.ExplainForcedRoute(ctx, forceMeta.PinnedModel, llm.RoutePlanningRequest{
				Prompt:          req.Prompt,
				History:         convertAPIMessagesToLLMMessages(req.History),
				ConversationID:  req.ConversationID,
				Preference:      req.Config.Preference,
				AvailableModels: availableModels,
				ModelBudgets:    h.config.ModelBudgets,
				PromptTokens:    estimatedTokens,
				Capability:      capability,
				ForceMetadata:   forceMeta,
			})
			if err != nil {
				return "", nil, session, nil, &requestFailure{StatusCode: http.StatusServiceUnavailable, Message: err.Error()}
			}
			return forceMeta.PinnedModel, nil, session, selection, nil
		}
		if forceMeta.Strict {
			return "", nil, session, nil, &requestFailure{
				StatusCode: http.StatusFailedDependency,
				Message:    fmt.Sprintf("forced model '%s' cannot satisfy primary capability '%s' in strict mode", forceMeta.PinnedModel, capability),
			}
		}
	}

	selection, err := h.controlPlane.PlanRoute(ctx, llm.RoutePlanningRequest{
		Prompt:          req.Prompt,
		History:         convertAPIMessagesToLLMMessages(req.History),
		ConversationID:  req.ConversationID,
		Preference:      req.Config.Preference,
		AvailableModels: availableModels,
		ModelBudgets:    h.config.ModelBudgets,
		PromptTokens:    estimatedTokens,
		Capability:      capability,
		ForceMetadata:   forceMeta,
	})
	if err != nil {
		return "", nil, session, nil, &requestFailure{StatusCode: http.StatusServiceUnavailable, Message: err.Error()}
	}
	return selection.ModelID, nil, session, selection, nil
}

func (h *GatewayHandler) selectImageModel(ctx context.Context, req api.GenerationRequest, prepared *llm.PreparedExecution) (string, error) {
	if prepared != nil && prepared.Plan != nil {
		if stage := prepared.Plan.PrimaryStage(); stage != nil && stage.ModelBinding != "" {
			if _, ok := h.imageClients[stage.ModelBinding]; ok {
				return stage.ModelBinding, nil
			}
			if stage.Strict {
				return "", fmt.Errorf("strict image stage binding requested unavailable model %s", stage.ModelBinding)
			}
		}
	}
	if req.Config.ForceModel != "" {
		if _, ok := h.imageClients[req.Config.ForceModel]; ok {
			return req.Config.ForceModel, nil
		}
		if req.Config.StrictForce {
			return "", fmt.Errorf("forced image model %s is unavailable", req.Config.ForceModel)
		}
	}
	return h.imageRouter.SelectModel(ctx, req.Prompt, req.Config.ImagePreference)
}

func requestForceMetadata(cfg api.GenerationConfig) llm.ForceMetadata {
	if cfg.ForceModel == "" {
		return llm.ForceMetadata{}
	}
	scope := normalizeForceScope(cfg.ForceScope)
	return llm.ForceMetadata{
		IsForced:       true,
		Scope:          scope,
		Strict:         cfg.StrictForce || scope == forceScopeStrictEndToEnd,
		PinnedModel:    cfg.ForceModel,
		EffectiveModel: cfg.ForceModel,
	}
}

func (h *GatewayHandler) modelSupportsCapability(modelID string, capability llm.Capability) bool {
	if capability == "" || modelID == "" {
		return true
	}
	if h.controlPlane != nil {
		return h.controlPlane.Models().SupportsCapability(modelID, capability)
	}
	return llm.CapabilityForModel(modelID) == capability
}

func (h *GatewayHandler) ensureCapabilityModelAvailable(ctx context.Context, modelID string, capability llm.Capability) error {
	if capability == llm.CapabilityImageGeneration || capability == llm.CapabilityImageEditing {
		if _, ok := h.imageClients[modelID]; !ok {
			return fmt.Errorf("model '%s' is not enabled for image generation", modelID)
		}
		return nil
	}
	return h.ensureModelAvailable(ctx, modelID)
}

func preparedArtifacts(prepared *llm.PreparedExecution) []llm.ArtifactRecord {
	if prepared == nil {
		return nil
	}
	return prepared.Artifacts
}

func preparedPlan(prepared *llm.PreparedExecution) *llm.ExecutionPlan {
	if prepared == nil {
		return nil
	}
	return prepared.Plan
}

func preparedArtifactLinks(prepared *llm.PreparedExecution) []llm.ArtifactMemoryLink {
	if prepared == nil || len(prepared.Artifacts) == 0 {
		return nil
	}
	links := make([]llm.ArtifactMemoryLink, 0, len(prepared.Artifacts))
	for _, artifact := range prepared.Artifacts {
		links = append(links, llm.ArtifactMemoryLink{
			Name:      artifact.Name,
			Kind:      artifact.Type,
			Source:    artifact.SourceURI,
			Reference: artifact.ArtifactID,
		})
	}
	return links
}

func convertAPIMessagesToLLMMessages(apiMessages []api.Message) []llm.Message {
	llmMessages := make([]llm.Message, len(apiMessages))
	for i, msg := range apiMessages {
		llmMessages[i] = llm.Message{Role: llm.Role(msg.Role), Content: msg.Content}
	}
	return llmMessages
}

func historySignature(history []api.Message) string {
	if len(history) == 0 {
		return llm.GenerateCacheKey("empty-history")
	}
	parts := make([]string, 0, len(history))
	for _, message := range history {
		parts = append(parts, message.Role+":"+message.Content)
	}
	return llm.GenerateCacheKey(strings.Join(parts, "|"))
}

func effectiveRoutingMode(preference string, routeSelection *llm.RouteSelection) string {
	if routeSelection != nil && routeSelection.Explanation.Strategy != "" {
		return routeSelection.Explanation.Strategy
	}
	if preference == "" {
		return "default"
	}
	return preference
}

func estimatePromptTokens(prompt string, history []api.Message) int {
	totalPromptLength := len(prompt)
	for _, msg := range history {
		totalPromptLength += len(msg.Content)
	}
	if totalPromptLength == 0 {
		return 1
	}
	return totalPromptLength / 4
}

func formatTime(ts time.Time) string {
	if ts.IsZero() {
		return ""
	}
	return ts.UTC().Format(time.RFC3339Nano)
}

func parseBoolValue(value string) bool {
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return false
	}
	return parsed
}

func parseIntValue(value string) int {
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0
	}
	return parsed
}

func parseTimestamp(value string) time.Time {
	if value == "" {
		return time.Time{}
	}
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return time.Time{}
	}
	return parsed
}

func currentPinnedModelID(session *sessionState) string {
	if session == nil {
		return ""
	}
	if session.ModelID != "" {
		return session.ModelID
	}
	return session.LastOverrideTo
}

func currentEffectiveModelID(session *sessionState) string {
	if session == nil {
		return ""
	}
	if session.EffectiveModelID != "" {
		return session.EffectiveModelID
	}
	return currentPinnedModelID(session)
}

func (h *GatewayHandler) profilerHealthStaleness() time.Duration {
	if raw, ok := h.config.RouterConfig.Thresholds["health_check_staleness"].(string); ok {
		if parsed, err := time.ParseDuration(raw); err == nil {
			return parsed
		}
	}
	return 5 * time.Minute
}

func normalizeForceScope(scope string) string {
	switch scope {
	case forceScopeCapability, forceScopeStrictEndToEnd:
		return scope
	case "", forceScopePrimaryReasoner:
		return forceScopePrimaryReasoner
	default:
		return forceScopePrimaryReasoner
	}
}

func storedForceScope(scope string) string {
	if scope == "" {
		return ""
	}
	return normalizeForceScope(scope)
}

func forceMetadataFromSession(session *sessionState) llm.ForceMetadata {
	if session == nil || !session.IsForced {
		return llm.ForceMetadata{}
	}
	return llm.ForceMetadata{
		IsForced:       session.IsForced,
		Scope:          normalizeForceScope(session.ForceScope),
		Strict:         session.StrictForce,
		PinnedModel:    currentPinnedModelID(session),
		EffectiveModel: currentEffectiveModelID(session),
	}
}

func buildAPIRouteMetadata(selection *llm.RouteSelection) *api.RouteMetadata {
	if selection == nil {
		return nil
	}
	metadata := &api.RouteMetadata{
		SelectedProvider:       selection.Explanation.SelectedProvider,
		SelectedModel:          selection.Explanation.SelectedModel,
		Strategy:               selection.Explanation.Strategy,
		PolicyName:             selection.Explanation.PolicyName,
		PolicySummary:          selection.Explanation.PolicySummary,
		RouteFamily:            selection.Explanation.RouteFamily,
		SelectionReason:        selection.Explanation.SelectionReason,
		UsedDegradedCandidates: selection.Explanation.UsedDegradedCandidates,
		HealthInputs:           buildAPIRouteHealth(selection.Explanation.HealthInputs),
		FallbackGraph:          append([]string(nil), selection.Explanation.FallbackGraph...),
		OrchestrationID:        selection.Explanation.OrchestrationID,
		ForcedSemantics: api.ForceMetadata{
			IsForced:       selection.Explanation.ForcedSemantics.IsForced,
			Scope:          selection.Explanation.ForcedSemantics.Scope,
			Strict:         selection.Explanation.ForcedSemantics.Strict,
			PinnedModel:    selection.Explanation.ForcedSemantics.PinnedModel,
			EffectiveModel: selection.Explanation.ForcedSemantics.EffectiveModel,
		},
	}
	for _, candidate := range selection.Explanation.FallbackCandidates {
		metadata.FallbackCandidates = append(metadata.FallbackCandidates, buildAPIRouteCandidate(candidate))
	}
	for _, candidate := range selection.Explanation.FilteredCandidates {
		metadata.FilteredCandidates = append(metadata.FilteredCandidates, buildAPIRouteCandidate(candidate))
	}
	return metadata
}

func buildAPIIntentMetadata(selection *llm.RouteSelection) *api.IntentMetadata {
	if selection == nil {
		return nil
	}
	intent := selection.Explanation.Intent
	return &api.IntentMetadata{
		TaskType:            intent.TaskType,
		ModalityType:        intent.ModalityType,
		ComplexityClass:     intent.ComplexityClass,
		CostPriority:        intent.CostPriority,
		LatencyPriority:     intent.LatencyPriority,
		QualityPriority:     intent.QualityPriority,
		GroundingRequired:   intent.GroundingRequired,
		ToolLikelihood:      intent.ToolLikelihood,
		AsyncLikelihood:     intent.AsyncLikelihood,
		SessionConstraint:   intent.SessionConstraint,
		ForceScopeIfAny:     intent.ForceScopeIfAny,
		RequestedCapability: string(intent.RequestedCapability),
		PreferenceHint:      intent.PreferenceHint,
	}
}

func buildAPIRouteCandidate(candidate llm.RouteCandidate) api.RouteCandidate {
	return api.RouteCandidate{
		ModelID:       candidate.ModelID,
		Provider:      candidate.Provider,
		Score:         candidate.Score,
		EstimatedCost: candidate.EstimatedCost,
		AvgLatencyMS:  candidate.AvgLatencyMS,
		Health:        buildAPIRouteHealth(candidate.Health),
		Reason:        candidate.Reason,
	}
}

func buildAPIRouteHealth(health llm.RouteHealth) api.RouteHealth {
	return api.RouteHealth{
		ProviderStatus:   string(health.ProviderStatus),
		ModelStatus:      string(health.ModelStatus),
		CapabilityStatus: string(health.CapabilityStatus),
	}
}

func buildAPISessionMetadata(session *sessionState) *api.SessionMetadata {
	if session == nil || session.ConversationID == "" {
		return nil
	}
	mode := "dynamic"
	if session.IsForced {
		mode = "forced"
	}
	return &api.SessionMetadata{
		ConversationID: session.ConversationID,
		Mode:           mode,
		PinnedModel:    currentPinnedModelID(session),
		EffectiveModel: currentEffectiveModelID(session),
		IsForced:       session.IsForced,
		ForceScope:     storedForceScope(session.ForceScope),
		StrictForce:    session.StrictForce,
		OverrideCount:  session.OverrideCount,
		FailoverCount:  session.FailoverCount,
		LastOverrideAt: formatTime(session.LastOverrideAt),
		LastFailoverAt: formatTime(session.LastFailoverAt),
	}
}

func buildAPIRetrievalMetadata(result *llm.RetrievalResult) *api.RetrievalMetadata {
	if result == nil {
		return nil
	}
	metadata := &api.RetrievalMetadata{
		RetrievalID:        result.RetrievalID,
		Score:              result.Score,
		CandidateCount:     result.CandidateCount,
		SelectedCount:      result.SelectedCount,
		SourceDiversity:    result.SourceDiversity,
		BudgetUsedChars:    result.BudgetUsedChars,
		RetrievalLatencyMS: result.RetrievalLatency.Milliseconds(),
		StaleSources:       append([]string(nil), result.StaleSources...),
	}
	for _, provenance := range result.Provenance {
		metadata.Sources = append(metadata.Sources, api.RetrievalSource{
			Source:      provenance.Source,
			DocumentID:  provenance.DocumentID,
			DocTitle:    provenance.DocTitle,
			Section:     provenance.Section,
			SectionPath: provenance.SectionPath,
			Version:     provenance.Version,
			UpdatedAt:   provenance.Timestamp,
			IngestedAt:  provenance.IngestedAt,
			ChunkIndex:  provenance.ChunkIndex,
			Score:       provenance.Score,
			RerankScore: provenance.RerankScore,
			Freshness:   provenance.Freshness,
		})
	}
	return metadata
}

func buildAPIGroundingMetadata(decision llm.GroundingDecision) *api.GroundingMetadata {
	if decision.Mode == "" && decision.EvidenceStatus == "" {
		return nil
	}
	return &api.GroundingMetadata{
		AnswerMode:       string(decision.Mode),
		RequiresEvidence: decision.RequiresEvidence,
		EvidenceStatus:   string(decision.EvidenceStatus),
		Abstained:        decision.ShouldAbstain,
		Caution:          decision.Caution,
	}
}

func buildAPIMemoryMetadata(snapshot *llm.SessionMemorySnapshot) *api.MemoryMetadata {
	if snapshot == nil {
		return nil
	}
	return &api.MemoryMetadata{
		ConversationID:      snapshot.ConversationID,
		ShortTermEventCount: len(snapshot.ShortTerm),
		Summary:             snapshot.Summary.Summary,
		KnowledgeSources:    append([]string(nil), snapshot.Working.KnowledgeSources...),
		ActiveConstraints:   append([]string(nil), snapshot.Working.ActiveConstraints...),
	}
}

func buildAPIContextMetadata(context *llm.ComposedContext) *api.ContextMetadata {
	if context == nil {
		return nil
	}
	return &api.ContextMetadata{
		PromptChars:      context.PromptChars,
		BudgetChars:      context.BudgetChars,
		IncludedSections: append([]string(nil), context.IncludedSections...),
		OmittedSections:  append([]string(nil), context.OmittedSections...),
	}
}

func buildAPIToolPlanMetadata(plan tools.ToolPlan) *api.ToolPlanMetadata {
	if !plan.NeedTools && len(plan.SelectedTools) == 0 && plan.Reason == "" {
		return nil
	}
	return &api.ToolPlanMetadata{
		NeedTools:            plan.NeedTools,
		SelectedTools:        append([]string(nil), plan.SelectedTools...),
		Reason:               plan.Reason,
		RetrievalBeforeTools: plan.RetrievalBeforeTools,
		ToolBeforeReasoning:  plan.ToolBeforeReasoning,
		UseMultiTool:         plan.UseMultiTool,
	}
}

func buildAPIArtifactMetadata(prepared *llm.PreparedExecution) []api.ArtifactMetadata {
	if prepared == nil || len(prepared.Artifacts) == 0 {
		return nil
	}
	return buildAPIArtifactMetadataFromRecords(prepared.Artifacts)
}

func buildAPIArtifactMetadataFromRecords(records []llm.ArtifactRecord) []api.ArtifactMetadata {
	if len(records) == 0 {
		return nil
	}
	artifacts := make([]api.ArtifactMetadata, 0, len(records))
	for _, artifact := range records {
		artifacts = append(artifacts, api.ArtifactMetadata{
			ArtifactID:     artifact.ArtifactID,
			Name:           artifact.Name,
			Type:           artifact.Type,
			MimeType:       artifact.MimeType,
			SourceURI:      artifact.SourceURI,
			Version:        artifact.Version,
			SizeBytes:      artifact.SizeBytes,
			Role:           artifact.Role,
			DerivedFrom:    artifact.DerivedFrom,
			Lineage:        append([]string(nil), artifact.Lineage...),
			GeneratorModel: artifact.GeneratorModel,
			PromptSummary:  artifact.PromptSummary,
			Metadata:       cloneMetadata(artifact.Metadata),
		})
	}
	return artifacts
}

func buildAPIGenerationMetadata(result *llm.GenerationExecutionResult) *api.GenerationMetadata {
	if result == nil {
		return nil
	}
	metadata := &api.GenerationMetadata{
		Pipeline:       result.Pipeline,
		PromptStrategy: result.Prompt.Strategy,
		RefinedPrompt:  result.Prompt.RefinedPrompt,
		PolicyStatus:   result.PolicyStatus,
		QualityStatus:  result.QualityStatus,
		Voice:          result.Prompt.Voice,
		Attempts:       result.Attempts,
		Warnings:       append([]string(nil), result.Warnings...),
		StageModelMap:  cloneMetadata(result.StageModelMap),
		VideoHook:      result.VideoHook,
	}
	for _, artifact := range result.OutputArtifacts {
		if artifact.ArtifactID != "" {
			metadata.OutputArtifactIDs = append(metadata.OutputArtifactIDs, artifact.ArtifactID)
		}
	}
	return metadata
}

func buildAPIGenerationRouteMetadata(result *llm.GenerationExecutionResult, req api.GenerationRequest, prepared *llm.PreparedExecution) *api.RouteMetadata {
	if result == nil {
		return nil
	}
	routeFamily := ""
	if prepared != nil {
		routeFamily = string(prepared.Task.PrimaryCapability)
	}
	return &api.RouteMetadata{
		SelectedProvider: llm.ProviderForModel(result.ModelUsed),
		SelectedModel:    result.ModelUsed,
		Strategy:         firstNonEmpty(req.Config.ImagePreference, result.Prompt.Strategy, req.Config.Preference, "generation"),
		RouteFamily:      routeFamily,
		SelectionReason:  firstNonEmpty(result.RouteReason, "selected by creative generation runtime"),
		ForcedSemantics: api.ForceMetadata{
			IsForced:       req.Config.ForceModel != "",
			Scope:          req.Config.ForceScope,
			Strict:         req.Config.StrictForce,
			PinnedModel:    req.Config.ForceModel,
			EffectiveModel: result.ModelUsed,
		},
	}
}

func buildAPISessionMetadataForGeneration(req api.GenerationRequest, result *llm.GenerationExecutionResult) *api.SessionMetadata {
	if req.ConversationID == "" {
		return nil
	}
	mode := "dynamic"
	if req.Config.ForceModel != "" {
		mode = "forced"
	}
	return &api.SessionMetadata{
		ConversationID: req.ConversationID,
		Mode:           mode,
		PinnedModel:    req.Config.ForceModel,
		EffectiveModel: firstNonEmpty(req.Config.ForceModel, result.ModelUsed),
		IsForced:       req.Config.ForceModel != "",
		ForceScope:     req.Config.ForceScope,
		StrictForce:    req.Config.StrictForce,
	}
}

func buildAPIExecutionPlanMetadata(prepared *llm.PreparedExecution) *api.ExecutionPlanMetadata {
	if prepared == nil || prepared.Plan == nil {
		return nil
	}
	plan := prepared.Plan
	metadata := &api.ExecutionPlanMetadata{
		PlanID:            plan.PlanID,
		PlanType:          plan.PlanType,
		SyncMode:          plan.SyncMode,
		CostTier:          plan.CostTier,
		LatencyTier:       plan.LatencyTier,
		PrimaryStageID:    plan.PrimaryStageID,
		PrimaryCapability: string(plan.PrimaryCapability),
		ForceScope:        plan.ForceScope,
		RequiresAsync:     plan.RequiresAsync,
		Modalities:        append([]string(nil), plan.Modalities...),
		Notes:             append([]string(nil), plan.Notes...),
	}
	for _, capability := range plan.RequiredCapabilities {
		metadata.RequiredCapabilities = append(metadata.RequiredCapabilities, string(capability))
	}
	for _, stage := range plan.Stages {
		metadata.Stages = append(metadata.Stages, api.ExecutionStageMetadata{
			StageID:          stage.StageID,
			StageType:        stage.StageType,
			Title:            stage.Title,
			Capability:       string(stage.Capability),
			ModelBinding:     stage.ModelBinding,
			DependsOn:        append([]string(nil), stage.DependsOn...),
			ForcePolicy:      stage.ForcePolicy,
			BindingViolation: stage.BindingViolation,
			Status:           stage.Status,
			Optional:         stage.Optional,
			ForceApplied:     stage.ForceApplied,
			Strict:           stage.Strict,
		})
	}
	return metadata
}

func buildAPIToolCalls(records []tools.ExecutionRecord) []api.ExecutedToolCall {
	if len(records) == 0 {
		return nil
	}
	calls := make([]api.ExecutedToolCall, 0, len(records))
	for _, record := range records {
		calls = append(calls, api.ExecutedToolCall{
			Name:       record.Name,
			Args:       record.Args,
			Result:     firstNonEmpty(record.Result, record.Error),
			Status:     record.Status,
			DurationMS: record.DurationMS,
			Attempts:   record.Attempts,
		})
	}
	return calls
}

func toolResultsForContext(records []tools.ExecutionRecord) []llm.ContextToolResult {
	if len(records) == 0 {
		return nil
	}
	results := make([]llm.ContextToolResult, 0, len(records))
	for _, record := range records {
		results = append(results, llm.ContextToolResult{
			Name:    record.Name,
			Status:  record.Status,
			Summary: firstNonEmpty(record.Result, record.Error),
		})
	}
	return results
}

func (h *GatewayHandler) persistOrchestrationMetadata(ctx context.Context, req api.GenerationRequest, resp *api.GenerationResponse) string {
	if h.rdb == nil || resp == nil || resp.Route == nil {
		return ""
	}
	recordID := llm.GenerateCacheKey(strings.Join([]string{
		req.ConversationID,
		req.UserID,
		req.Prompt,
		resp.ModelUsed,
		resp.CacheStatus,
		time.Now().UTC().Format(time.RFC3339Nano),
	}, "|"))

	record := map[string]any{
		"id":                  recordID,
		"request":             req,
		"response":            resp,
		"governance":          resp.Governance,
		"security":            resp.Security,
		"user_id":             req.UserID,
		"tenant_id":           req.TenantID,
		"workspace_id":        req.WorkspaceID,
		"conversation_id":     req.ConversationID,
		"prompt_hash":         llm.GenerateCacheKey(req.Prompt),
		"prompt_preview":      truncateForMetadata(req.Prompt, 160),
		"model_used":          resp.ModelUsed,
		"cache_status":        resp.CacheStatus,
		"rag_context_used":    resp.RAGContextUsed,
		"route":               resp.Route,
		"intent":              resp.Intent,
		"session":             resp.Session,
		"failover_info":       resp.FailoverInfo,
		"artifacts":           resp.Artifacts,
		"generated_artifacts": resp.GeneratedArtifacts,
		"execution_plan":      resp.ExecutionPlan,
		"generation":          resp.Generation,
		"persisted_at":        time.Now().UTC().Format(time.RFC3339Nano),
	}

	payload, err := json.Marshal(record)
	if err != nil {
		log.Printf("WARNING: failed to marshal orchestration metadata: %v", err)
		return ""
	}
	ttl := h.orchestrationMetadataTTL()
	if err := h.rdb.Set(ctx, fmt.Sprintf("orchestration:%s", recordID), payload, ttl).Err(); err != nil {
		log.Printf("WARNING: failed to persist orchestration metadata: %v", err)
		return ""
	}
	globalIndexKey := "orchestration_index:all"
	pipe := h.rdb.TxPipeline()
	pipe.LPush(ctx, globalIndexKey, recordID)
	pipe.LTrim(ctx, globalIndexKey, 0, 199)
	pipe.Expire(ctx, globalIndexKey, ttl)
	if req.ConversationID != "" {
		indexKey := fmt.Sprintf("orchestration_index:%s", req.ConversationID)
		pipe.LPush(ctx, indexKey, recordID)
		pipe.LTrim(ctx, indexKey, 0, 99)
		pipe.Expire(ctx, indexKey, ttl)
	}
	if _, err := pipe.Exec(ctx); err != nil {
		log.Printf("WARNING: failed to persist orchestration index: %v", err)
	}
	return recordID
}

func (h *GatewayHandler) orchestrationMetadataTTL() time.Duration {
	if raw := strings.TrimSpace(h.config.RouterConfig.OrchestrationMetadataTTL); raw != "" {
		if parsed, err := time.ParseDuration(raw); err == nil {
			return parsed
		}
	}
	return 24 * time.Hour
}

func (h *GatewayHandler) refreshOrchestrationMetadata(ctx context.Context, req api.GenerationRequest, resp *api.GenerationResponse) {
	if h.rdb == nil || resp == nil || resp.Route == nil || resp.Route.OrchestrationID == "" {
		return
	}
	recordID := resp.Route.OrchestrationID
	record := map[string]any{
		"id":                  recordID,
		"request":             req,
		"response":            resp,
		"governance":          resp.Governance,
		"security":            resp.Security,
		"user_id":             req.UserID,
		"tenant_id":           req.TenantID,
		"workspace_id":        req.WorkspaceID,
		"conversation_id":     req.ConversationID,
		"prompt_hash":         llm.GenerateCacheKey(req.Prompt),
		"prompt_preview":      truncateForMetadata(req.Prompt, 160),
		"model_used":          resp.ModelUsed,
		"cache_status":        resp.CacheStatus,
		"rag_context_used":    resp.RAGContextUsed,
		"route":               resp.Route,
		"intent":              resp.Intent,
		"session":             resp.Session,
		"failover_info":       resp.FailoverInfo,
		"artifacts":           resp.Artifacts,
		"generated_artifacts": resp.GeneratedArtifacts,
		"execution_plan":      resp.ExecutionPlan,
		"generation":          resp.Generation,
		"persisted_at":        time.Now().UTC().Format(time.RFC3339Nano),
	}
	payload, err := json.Marshal(record)
	if err != nil {
		return
	}
	_ = h.rdb.Set(ctx, fmt.Sprintf("orchestration:%s", recordID), payload, h.orchestrationMetadataTTL()).Err()
}

func truncateForMetadata(value string, maxLen int) string {
	if maxLen <= 0 || len(value) <= maxLen {
		return value
	}
	return value[:maxLen]
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func cloneMetadata(value map[string]string) map[string]string {
	if len(value) == 0 {
		return nil
	}
	out := make(map[string]string, len(value))
	for key, item := range value {
		out[key] = item
	}
	return out
}
