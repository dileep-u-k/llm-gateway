// In file: cmd/gateway/handler.go
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"llm-gateway/internal/api"
	"llm-gateway/internal/llm"
	"llm-gateway/internal/tools"
	cacheversion "llm-gateway/internal/version"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

// =================================================================================
// Gateway Handler v3 (Definitive Production Version)
// =================================================================================
// This definitive version implements a complete, production-grade logic flow for
// a sophisticated and user-friendly conversational AI application.
//
// Key Features:
// 1.  **Two Chat Modes:** Seamlessly supports both permanently "locked" chats (forced
//     model for consistency) and "dynamic" chats (flexible model selection).
// 2.  **Automatic Session Failover:** If a pinned model in any chat goes offline,
//     the gateway automatically fails over to the next-best healthy model.
// 3.  **Conditional Re-routing:** In "dynamic" chats, users can override the
//     current model by sending a new preference.
// =================================================================================

type GatewayHandler struct {
	clients        map[string]llm.LLMClient
	profiler       *llm.Profiler
	router         *llm.Router
	ragService     *llm.RAGService
	intentAnalyzer *llm.IntentAnalyzer
	toolManager    *tools.ToolManager
	promptAnalyzer *llm.PromptAnalyzer
	config         *AppConfig
	rdb            *redis.Client
}

func NewGatewayHandler(clients map[string]llm.LLMClient, profiler *llm.Profiler, router *llm.Router, ragService *llm.RAGService, intentAnalyzer *llm.IntentAnalyzer, toolManager *tools.ToolManager, promptAnalyzer *llm.PromptAnalyzer, config *AppConfig, rdb *redis.Client) *GatewayHandler {
	return &GatewayHandler{
		clients:        clients,
		profiler:       profiler,
		router:         router,
		ragService:     ragService,
		intentAnalyzer: intentAnalyzer,
		toolManager:    toolManager,
		promptAnalyzer: promptAnalyzer,
		config:         config,
		rdb:            rdb,
	}
}

func (h *GatewayHandler) HandleGeneration(c *gin.Context) {
	startTime := time.Now()
	var req api.GenerationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request: " + err.Error()})
		return
	}

	log.Printf("--- New Request (User: %s, Convo: %s, Prompt: '%.30s...') ---", req.UserID, req.ConversationID, req.Prompt)

	cacheKey := cacheversion.GenerateVersionedCacheKey("llmcache", req.Prompt)
	if cachedVal, found := h.ragService.CheckCache(c.Request.Context(), cacheKey); found {
		var cachedResp api.GenerationResponse
		if json.Unmarshal([]byte(cachedVal), &cachedResp) == nil {
			log.Println("✅ Cache HIT")
			cachedResp.LatencyMS = time.Since(startTime).Milliseconds()
			cachedResp.CacheStatus = "HIT"
			c.JSON(http.StatusOK, cachedResp)
			return
		}
	}
	log.Println("⚠️ Cache MISS")

	modelID, failoverInfo, err := h.determineModelID(c, &req)
	if err != nil {
		return // An error response has already been sent.
	}

	intent := h.intentAnalyzer.AnalyzeIntent(req.Prompt)
	log.Printf("🔍 Intent Detected: %s", intent)

	var finalContent string
	var usage api.Usage
	var ragContextUsed bool

	// This is the only change in this function: pass the history to the tool loop.
	switch intent {
	case llm.IntentWeather, llm.IntentCalculator, llm.IntentNews:
		// --- THIS IS THE CHANGE ---
		finalContent, usage, _, err = h.handleToolLoop(c, req)
	default:
		finalContent, usage, ragContextUsed, err = h.executeRAGAndGenerate(c, req, modelID)
	}

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	latency := time.Since(startTime)
	h.profiler.UpdateProfileOnSuccess(c.Request.Context(), modelID, latency, usage)

	finalResponse := api.GenerationResponse{
		Content:        finalContent,
		ModelUsed:      modelID,
		Usage:          usage,
		LatencyMS:      latency.Milliseconds(),
		RAGContextUsed: ragContextUsed,
		CacheStatus:    "MISS",
		FailoverInfo:   failoverInfo,
	}

	respBytes, err := json.Marshal(finalResponse)
	if err != nil {
		log.Printf("WARNING: Failed to marshal response for caching: %v", err)
	} else {
		h.ragService.SetCache(c.Request.Context(), cacheKey, string(respBytes))
		log.Println("✅ Response CACHED")
	}

	c.JSON(http.StatusOK, finalResponse)
}

// determineModelID encapsulates the complete, final logic with all bug fixes.
func (h *GatewayHandler) determineModelID(c *gin.Context, req *api.GenerationRequest) (string, *api.FailoverInfo, error) {
	var failoverInfo *api.FailoverInfo

	// A. SESSION HANDLING: Check for an existing conversation first.
	if req.ConversationID != "" {
		sessionKey := fmt.Sprintf("session:%s", req.ConversationID)
		sessionData, err := h.rdb.HGetAll(c.Request.Context(), sessionKey).Result()

		if err == nil && len(sessionData) > 0 {
			pinnedModel := sessionData["model_id"]
			isForcedSession := sessionData["is_forced"] == "true"

			if isForcedSession {
				// --- FORCED SESSION LOGIC ---
				log.Printf("... Detected a Forced Session. Verifying model health...")
				profile, profilerErr := h.profiler.GetProfile(c.Request.Context(), pinnedModel)
				if profilerErr == nil && profile.Status == "online" {
					log.Printf("📌 Forced Session HIT. Reusing locked model: %s", pinnedModel)
					h.refreshSessionTTL(c.Request.Context(), sessionKey)
					return pinnedModel, nil, nil
				} else {
					// FAILOVER for a forced session.
					log.Printf("🚨 Forced-pinned model '%s' is offline. Failing over...", pinnedModel)
					req.Config.Preference = "max_quality"
					failoverInfo = &api.FailoverInfo{ OriginalModel: pinnedModel, Reason: fmt.Sprintf("Model '%s' was offline.", pinnedModel) }
					// Let the request fall through to the router.
				}
			} else {
				// --- DYNAMIC SESSION LOGIC ---
				profile, profilerErr := h.profiler.GetProfile(c.Request.Context(), pinnedModel)
				if profilerErr == nil && profile.Status == "online" {
					if req.Config.Preference == "" {
						log.Printf("📌 Dynamic Session HIT. Reusing model: %s", pinnedModel)
						h.refreshSessionTTL(c.Request.Context(), sessionKey)
						return pinnedModel, nil, nil
					}
					log.Printf("⚠️ User provided a new preference in a dynamic chat. Overriding session...")
				} else {
					// FAILOVER for a dynamic session.
					log.Printf("🚨 Pinned model '%s' is offline. Failing over...", pinnedModel)
					failoverInfo = &api.FailoverInfo{ OriginalModel: pinnedModel, Reason: fmt.Sprintf("Model '%s' was offline.", pinnedModel) }
				}
			}
		}
	}

	// B. NEW CHAT / ROUTING LOGIC
	// This block runs for the first message of a chat, one-off queries, or failovers.

	// Handle the creation of a NEW forced chat as a special, separate case.
	if req.ConversationID != "" && req.Config.ForceModel != "" {
		forcedModelID := req.Config.ForceModel
		log.Printf("... User is attempting to force-start a new chat with model: %s", forcedModelID)
		profile, err := h.profiler.GetProfile(c.Request.Context(), forcedModelID)
		if err != nil || profile.Status != "online" {
			h.suggestHealthyAlternatives(c, forcedModelID)
			return "", nil, errors.New("response sent")
		}
		// Pin the new forced session and return immediately.
		h.pinSession(c.Request.Context(), req.ConversationID, forcedModelID, true)
		return forcedModelID, nil, nil
	}

	// This is the path for new dynamic chats, one-off queries, or any failover.
	if req.Config.Preference == "" {
		req.Config.Preference = h.promptAnalyzer.Analyze(req.Prompt)
	}

	// --- THIS IS THE FINAL ENHANCEMENT ---
	// Calculate the total length of the prompt including all historical messages.
	totalPromptLength := len(req.Prompt)
	for _, msg := range req.History {
		totalPromptLength += len(msg.Content)
	}
	// Use this more accurate total length for the token estimation.
	estimatedTokens := totalPromptLength / 4
	log.Printf("... Total estimated input tokens (including history): %d", estimatedTokens)
	// --- END OF ENHANCEMENT ---
	
	modelID, err := h.router.SelectOptimalModel(c.Request.Context(), h.config.EnabledModels, req.Config.Preference, estimatedTokens, h.config.ModelBudgets)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
		return "", nil, errors.New("response sent")
	}

	if failoverInfo != nil {
		failoverInfo.NewModel = modelID
	}

	if req.ConversationID != "" {
		// Pin the session, ensuring isForced is false because we came through the dynamic path.
		h.pinSession(c.Request.Context(), req.ConversationID, modelID, false)
	}

	return modelID, failoverInfo, nil
}


// --- HELPER FUNCTIONS ---

func (h *GatewayHandler) pinSession(ctx context.Context, conversationID, modelID string, isForced bool) {
	sessionKey := fmt.Sprintf("session:%s", conversationID)
	sessionData := map[string]interface{}{
		"model_id":  modelID,
		"is_forced": fmt.Sprintf("%v", isForced), // Converts true to "true"
	}
	if err := h.rdb.HSet(ctx, sessionKey, sessionData).Err(); err != nil {
		log.Printf("WARNING: Failed to HSet session key in Redis: %v", err)
	} else {
		h.refreshSessionTTL(ctx, sessionKey)
		log.Printf("📌 Pinned model %s to conversation %s (Forced=%v).", modelID, conversationID, isForced)
	}
}

func (h *GatewayHandler) refreshSessionTTL(ctx context.Context, sessionKey string) {
	h.rdb.Expire(ctx, sessionKey, 1*time.Hour)
}

func (h *GatewayHandler) suggestHealthyAlternatives(c *gin.Context, failedModelID string) {
	var healthyModels []string
	for _, model := range h.config.EnabledModels {
		if model == failedModelID {
			continue
		}
		p, err := h.profiler.GetProfile(c.Request.Context(), model)
		if err == nil && p.Status == "online" {
			healthyModels = append(healthyModels, model)
		}
	}
	errorMsg := fmt.Sprintf("The requested model '%s' is currently offline.", failedModelID)
	c.JSON(http.StatusFailedDependency, gin.H{
		"error":            errorMsg,
		"available_models": healthyModels,
	})
}

// --- THIS FUNCTION IS NOW UPDATED ---
// It now accepts the full request to handle conversation history.
func (h *GatewayHandler) executeRAGAndGenerate(c *gin.Context, req api.GenerationRequest, modelID string) (string, api.Usage, bool, error) {
	finalPrompt, ragContextUsed, err := h.performRAGRetrieval(c, req.Prompt, "relevance_threshold")
	if err != nil {
		return "", api.Usage{}, false, fmt.Errorf("RAG retrieval failed: %w", err)
	}
	client := h.clients[modelID]
	if client == nil {
		return "", api.Usage{}, false, fmt.Errorf("no client available for model %s", modelID)
	}

	// --- THIS IS THE NEW LOGIC ---
	// Construct the full conversation history to give the model memory.
	// Convert the API message history to the internal LLM message type.
	messages := convertAPIMessagesToLLMMessages(req.History)
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: finalPrompt})
	// --- END OF NEW LOGIC ---

	llmConfig := &llm.GenerationConfig{
		Model:       modelID,
		MaxTokens:   req.Config.MaxTokens,
		Temperature: req.Config.Temperature,
		TopP:        req.Config.TopP,
		Stream:      req.Config.Stream,
	}

	// Pass the complete message history to the LLM.
	result, err := client.Generate(c.Request.Context(), messages, llmConfig, nil)
	if err != nil {
		h.profiler.UpdateProfileOnFailure(c.Request.Context(), modelID)
		return "", api.Usage{}, ragContextUsed, fmt.Errorf("LLM generation failed for model %s: %w", modelID, err)
	}
	return result.Content, result.Usage, ragContextUsed, nil
}

func (h *GatewayHandler) performRAGRetrieval(c *gin.Context, prompt string, thresholdKey string) (string, bool, error) {
	contextText, score, err := h.ragService.RetrieveContext(c.Request.Context(), prompt, 2)
	if err != nil {
		return prompt, false, err
	}
	threshold := h.config.RouterConfig.Thresholds[thresholdKey].(float64)
	if score >= threshold {
		log.Printf("📝 RAG context found (score %.2f >= %.2f). Augmenting prompt.", score, threshold)
		return fmt.Sprintf("Using the following context, answer the question.\n\nContext:\n%s\n\nQuestion: %s", contextText, prompt), true, nil
	}
	log.Printf("RAG context score (%.2f) is below threshold (%.2f). Proceeding with original prompt.", score, threshold)
	return prompt, false, nil
}

// --- THIS FUNCTION IS NOW UPDATED ---
// It now accepts the full request to handle conversation history.
func (h *GatewayHandler) handleToolLoop(c *gin.Context, req api.GenerationRequest) (string, api.Usage, string, error) {
	log.Println("Entering tool loop...")
	const maxToolCalls = 5
	var cumulativeUsage api.Usage
	modelID := "gpt-4o"
	client, ok := h.clients[modelID]
	if !ok {
		return "", api.Usage{}, "", fmt.Errorf("tool-use model '%s' is not available or enabled", modelID)
	}

	// --- THIS IS THE NEW LOGIC ---
	// Construct the full conversation history for the tool-using agent.
	// Convert the API message history to the internal LLM message type.
	messages := convertAPIMessagesToLLMMessages(req.History)
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: req.Prompt})
	// --- END OF NEW LOGIC ---


	llmConfig := &llm.GenerationConfig{
		Model:       modelID,
		MaxTokens:   req.Config.MaxTokens,
		Temperature: req.Config.Temperature,
		TopP:        req.Config.TopP,
		Stream:      req.Config.Stream,
	}

	for i := 0; i < maxToolCalls; i++ {
		result, err := client.Generate(c.Request.Context(), messages, llmConfig, h.toolManager.GetDefinitions())
		if err != nil {
			h.profiler.UpdateProfileOnFailure(c.Request.Context(), modelID)
			return "", api.Usage{}, "", fmt.Errorf("LLM generation failed during tool loop: %w", err)
		}
		cumulativeUsage.Add(result.Usage)
		if len(result.ToolCalls) == 0 {
			log.Println("LLM provided final answer. Exiting tool loop.")
			return result.Content, cumulativeUsage, modelID, nil
		}
		messages = append(messages, llm.Message{Role: llm.RoleAssistant, Content: result.Content, ToolCalls: result.ToolCalls})
		for _, toolCall := range result.ToolCalls {
			log.Printf("🛠️ Executing tool: %s (ID: %s) with args: %s", toolCall.Function.Name, toolCall.ID, toolCall.Function.Arguments)
			toolResult, err := h.toolManager.Execute(toolCall.Function.Name, toolCall.Function.Arguments)
			if err != nil {
				toolResult = fmt.Sprintf("Error executing tool %s: %v", toolCall.Function.Name, err)
			}
			messages = append(messages, llm.Message{Role: llm.RoleTool, ToolCallID: toolCall.ID, Content: toolResult})
		}
	}
	return "", api.Usage{}, "", errors.New("exceeded maximum number of tool calls")
}

// --- NEW HELPER FUNCTION ---
// convertAPIMessagesToLLMMessages handles the type conversion between the public API and internal logic.
func convertAPIMessagesToLLMMessages(apiMessages []api.Message) []llm.Message {
	llmMessages := make([]llm.Message, len(apiMessages))
	for i, msg := range apiMessages {
		llmMessages[i] = llm.Message{
			Role:    llm.Role(msg.Role), // Cast the role string to the llm.Role type
			Content: msg.Content,
		}
	}
	return llmMessages
}
