// In file: cmd/gateway/main.go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/dileep-u-k/llm-gateway/internal/llm"
	"github.com/dileep-u-k/llm-gateway/internal/ops"
	"github.com/dileep-u-k/llm-gateway/internal/platform"
	"github.com/dileep-u-k/llm-gateway/internal/tools"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

// main is the entry point for the application.
// Its primary role is the "Composition Root": it loads configuration,
// initializes all services, injects dependencies, and starts the server.
func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	buildInfo := GetBuildInfo()
	log.Printf("🚀 Starting LLM Gateway | Version: %s | Commit: %s", buildInfo.Version, buildInfo.GitCommit)

	// 1. LOAD CONFIGURATION
	cfg, err := LoadConfig()
	if err != nil {
		log.Fatalf("❌ FATAL: Configuration Error: %v", err)
	}
	llm.InitializeModelCosts(cfg.ModelCosts)
	log.Println("✅ Configuration loaded.")

	// 2. INITIALIZE SERVICES
	rdb := redis.NewClient(&redis.Options{Addr: cfg.RedisAddr})
	if _, err := rdb.Ping(context.Background()).Result(); err != nil {
		log.Fatalf("❌ FATAL: Could not connect to Redis: %v", err)
	}

	llmClients, err := initializeLLMClients(cfg)
	if err != nil {
		log.Fatalf("❌ FATAL: %v", err)
	}

	profiler := llm.NewProfiler(rdb, cfg.RouterConfig)
	ragService, err := llm.NewRAGService(cfg.RAGConfig)
	if err != nil {
		log.Fatalf("❌ FATAL: Could not create RAG service: %v", err)
	}

	intentAnalyzer := llm.NewIntentAnalyzer()
	router := llm.NewRouter(profiler, cfg.RouterConfig)
	toolManager, err := initializeToolManager(cfg)
	if err != nil {
		log.Fatalf("❌ FATAL: %v", err)
	}
	memoryEngine := llm.NewMemoryEngine(rdb)
	contextComposer := llm.NewContextComposer()
	relevanceThreshold, _ := cfg.RouterConfig.Thresholds["relevance_threshold"].(float64)
	groundingEngine := llm.NewGroundingPolicyEngine(relevanceThreshold)

	// *** NEW: Initialize the PromptAnalyzer service. ***
	// This service will automatically select a routing preference if the user does not provide one.
	promptAnalyzer := llm.NewPromptAnalyzer()

	// Initialize Image Clients with robust, fatal error handling
	imageClients := make(map[string]llm.ImageClient)
	// --- FIX START: Iterate over ENABLED_IMAGE_MODELS for client creation ---
	enabledImageModelsList := cleanModelList(strings.Split(cfg.EnabledImageModels, ","))
	for _, imageModelID := range enabledImageModelsList {
		imageModelID = strings.TrimSpace(imageModelID) // Clean up any whitespace

		var imageClient llm.ImageClient
		var clientErr error

		switch {
		case strings.HasPrefix(imageModelID, "dall-e"):
			apiKey := os.Getenv("OPENAI_API_KEY")
			if apiKey == "" {
				log.Fatalf("❌ FATAL: OPENAI_API_KEY is not set, required for Dall-E models.")
			}
			imageClient, clientErr = llm.NewOpenAIImageClient(apiKey)
			if clientErr != nil {
				log.Fatalf("❌ FATAL: Failed to create OpenAI Image Client for %s: %v", imageModelID, clientErr)
			}
			imageClients[imageModelID] = imageClient

		case strings.HasPrefix(imageModelID, "imagen"):
			apiKey := os.Getenv("GEMINI_API_KEY")
			if apiKey == "" {
				log.Fatalf("❌ FATAL: GEMINI_API_KEY is not set, required for Imagen models.")
			}
			gcsBucket := os.Getenv("GCS_BUCKET_NAME")
			if gcsBucket == "" {
				log.Fatalf("❌ FATAL: GCS_BUCKET_NAME is not set, required for Imagen models.")
			}
			imageClient, clientErr = llm.NewGeminiImageClient(apiKey, gcsBucket)
			if clientErr != nil {
				log.Fatalf("❌ FATAL: Failed to create Gemini Image Client for %s: %v", imageModelID, clientErr)
			}
			imageClients[imageModelID] = imageClient

		default:
			log.Printf("WARNING: Unknown image model provider for %s, skipping client creation.", imageModelID)
		}
	}
	log.Printf("✅ %d image clients initialized.", len(imageClients))
	// --- FIX END ---

	imageRouter := llm.NewImageRouter(cfg.EnabledImageModels, profiler, cfg.RouterConfig)
	speechClients := make(map[string]llm.SpeechClient)
	enabledSpeechModelsList := cleanModelList(strings.Split(cfg.EnabledSpeechModels, ","))
	for _, speechModelID := range enabledSpeechModelsList {
		speechModelID = strings.TrimSpace(speechModelID)
		if speechModelID == "" {
			continue
		}
		switch {
		case strings.HasPrefix(speechModelID, "gpt"), strings.HasPrefix(speechModelID, "tts-"):
			apiKey := os.Getenv("OPENAI_API_KEY")
			if apiKey == "" {
				log.Fatalf("❌ FATAL: OPENAI_API_KEY is not set, required for speech models.")
			}
			speechClient, clientErr := llm.NewOpenAISpeechClient(apiKey)
			if clientErr != nil {
				log.Fatalf("❌ FATAL: Failed to create OpenAI Speech Client for %s: %v", speechModelID, clientErr)
			}
			speechClients[speechModelID] = speechClient
		default:
			log.Printf("WARNING: Unknown speech model provider for %s, skipping client creation.", speechModelID)
		}
	}
	log.Printf("✅ %d speech clients initialized.", len(speechClients))

	allEnabledModels := append(append([]string{}, cfg.EnabledModels...), enabledImageModelsList...)
	allEnabledModels = append(allEnabledModels, enabledSpeechModelsList...)
	controlPlane := llm.NewControlPlane(cfg.RouterConfig, profiler, router, allEnabledModels, cfg.APIKeys)
	multimodalRuntime := llm.NewMultimodalRuntime(rdb, controlPlane)
	generationRuntime := llm.NewGenerationRuntime(multimodalRuntime.ArtifactRegistry(), imageRouter, imageClients, speechClients, cfg.RouterConfig, platform.CleanArtifactStorageRoot(cfg.ArtifactStorageRoot))
	authenticator := platform.NewAuthenticatorFromEnv()
	policyStore := platform.NewPolicyBundleStore(rdb)
	policyEngine := platform.NewEngine(cfg.PlatformConfig, policyStore, cfg.ModelCosts)
	auditLogger := platform.NewAuditLogger(rdb)
	artifactAccess := platform.NewSignedArtifactAccessLayerFromEnv(cfg.PlatformConfig.Defaults.SignedURLTTL)

	// Correctly inject ALL components into the GatewayHandler.
	gatewayHandler := NewGatewayHandler(llmClients, profiler, router, controlPlane, ragService, intentAnalyzer, toolManager, promptAnalyzer, memoryEngine, contextComposer, groundingEngine, multimodalRuntime, generationRuntime, cfg, rdb, imageClients, speechClients, imageRouter, nil, authenticator, policyEngine, auditLogger, artifactAccess, cfg.ArtifactStorageRoot)
	asyncConfig := ops.DefaultConfig()
	asyncConfig.Workers = cfg.AsyncWorkers
	asyncRuntime := ops.NewRuntime(rdb, ops.PrepareFunc(gatewayHandler.prepareExecutionContext), ops.ExecuteFunc(gatewayHandler.executeManagedSync), asyncConfig, nil)
	gatewayHandler.opsRuntime = asyncRuntime
	log.Println("✅ All services initialized.")
	// --- END OF CORRECTION ---

	// 3. START BACKGROUND PROCESSES
	// Combine both text and image models into a single list for the health checker.
	go startHealthChecker(allEnabledModels, llmClients, imageClients, speechClients, ragService, profiler, cfg.RAGConfig.EmbeddingModel)
	if cfg.AsyncWorkersEnabled {
		asyncRuntime.Start(context.Background())
		log.Printf("✅ Async workers started (%d workers).", cfg.AsyncWorkers)
	}
	// --- END OF CORRECTION ---

	// 4. SETUP AND RUN THE WEB SERVER
	if !cfg.HTTPEnabled {
		log.Println("ℹ️ HTTP server disabled; running worker-only mode.")
		waitForShutdownSignal()
		return
	}
	gin.SetMode(os.Getenv("GIN_MODE"))
	engine := gin.Default()
	engine.GET("/", gatewayHandler.ServeProductUI)
	engine.GET("/admin", gatewayHandler.ServeAdminUI)
	engine.GET("/ui/*filepath", gatewayHandler.ServeStaticAsset)
	engine.GET("/healthz", gatewayHandler.HandleHealthz)
	engine.GET("/readyz", gatewayHandler.HandleReadyz)
	v1 := engine.Group("/api/v1")
	{
		v1.POST("/generate", gatewayHandler.HandleGeneration)
		v1.POST("/assets/upload", gatewayHandler.HandleAssetUpload)
		v1.GET("/artifacts", gatewayHandler.HandleArtifacts)
		v1.GET("/artifacts/:id", gatewayHandler.HandleArtifactMetadata)
		v1.GET("/artifacts/:id/content", gatewayHandler.HandleArtifactContent)
		v1.GET("/jobs/:id", gatewayHandler.HandleJobStatus)
		v1.GET("/jobs/:id/result", gatewayHandler.HandleJobResult)
		v1.POST("/jobs/:id/cancel", gatewayHandler.HandleCancelJob)
		v1.GET("/metrics", gatewayHandler.HandleMetrics)
		v1.GET("/dashboards", gatewayHandler.HandleDashboards)
		v1.POST("/evaluations/run", gatewayHandler.HandleEvaluationRun)
		v1.POST("/replay/:id", gatewayHandler.HandleReplayExecution)
		v1.GET("/platform/bootstrap", gatewayHandler.HandlePlatformBootstrap)
		v1.GET("/platform/me", gatewayHandler.HandlePlatformMe)
		v1.GET("/platform/admin/overview", gatewayHandler.HandleAdminOverview)
		v1.GET("/platform/admin/policies", gatewayHandler.HandleAdminPolicies)
		v1.PUT("/platform/admin/policies/:tenant/:workspace", gatewayHandler.HandleAdminPolicyUpsert)
		v1.POST("/platform/admin/policies/simulate", gatewayHandler.HandleAdminPolicySimulation)
		v1.GET("/platform/admin/jobs", gatewayHandler.HandleAdminJobs)
		v1.GET("/platform/admin/orchestrations", gatewayHandler.HandleAdminOrchestrations)
		v1.GET("/platform/admin/audit", gatewayHandler.HandleAdminAudit)
		v1.GET("/platform/admin/schema", gatewayHandler.HandleSchema)
	}

	srv := &http.Server{Addr: fmt.Sprintf(":%s", os.Getenv("PORT")), Handler: engine}
	runServerWithGracefulShutdown(srv)
}

func waitForShutdownSignal() {
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
}

// initializeLLMClients creates instances of the LLM clients based on config.
func initializeLLMClients(cfg *AppConfig) (map[string]llm.LLMClient, error) {
	clients := make(map[string]llm.LLMClient)
	var err error
	// --- THIS LINE IS THE FIX ---
	for _, modelID := range cfg.EnabledModels {
		modelID = strings.TrimSpace(modelID)
		if modelID == "" {
			continue
		}
		apiKey := cfg.APIKeys[modelID]
		var client llm.LLMClient
		switch {
		case strings.HasPrefix(modelID, "gpt"):
			client, err = llm.NewOpenAIClient(apiKey)
		case strings.HasPrefix(modelID, "claude"):
			client, err = llm.NewAnthropicClient(apiKey)
		case strings.HasPrefix(modelID, "gemini"):
			client, err = llm.NewGeminiClient(apiKey, modelID)
		case strings.HasPrefix(modelID, "mistral"):
			client, err = llm.NewMistralClient(apiKey)
		default:
			log.Printf("WARNING: Unknown model provider for %s, skipping.", modelID)
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("failed to create client for %s: %w", modelID, err)
		}
		clients[modelID] = client
	}
	log.Printf("✅ %d LLM clients initialized.", len(clients))
	return clients, nil
}

// initializeToolManager creates and registers all available tools.
func initializeToolManager(cfg *AppConfig) (*tools.ToolManager, error) {
	manager := tools.NewToolManager()

	manager.Register(tools.NewCalculatorTool())
	manager.Register(tools.NewWeatherTool())

	if cfg.NewsAPIKey != "" {
		newsTool, err := tools.NewNewsTool(cfg.NewsAPIKey)
		if err != nil {
			return nil, fmt.Errorf("failed to create news tool: %w", err)
		}
		manager.Register(newsTool)
	}

	log.Printf("✅ Tool Manager initialized with %d tools.", manager.ToolCount())
	return manager, nil
}

// startHealthChecker runs a background goroutine to proactively check model health.
func startHealthChecker(allEnabledModels []string, clients map[string]llm.LLMClient, imageClients map[string]llm.ImageClient, speechClients map[string]llm.SpeechClient, ragService *llm.RAGService, profiler *llm.Profiler, embeddingModel string) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	log.Println("🩺 Health checker started.")

	runChecks := func() {
		log.Println("🩺 Running proactive health checks...")
		providerStatuses := make(map[string][]llm.HealthStatus)
		providerAccess := make(map[string]bool)

		for _, modelID := range allEnabledModels {
			modelID = strings.TrimSpace(modelID) // Crucial: trim whitespace from model IDs
			provider := llm.ProviderForModel(modelID)

			// Check if it's a text model
			if client, ok := clients[modelID]; ok {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				config := &llm.GenerationConfig{Model: modelID, MaxTokens: 5}
				healthCheckPrompt := []llm.Message{{Role: llm.RoleUser, Content: "What is the capital of India?"}}
				start := time.Now()
				_, err := client.Generate(ctx, healthCheckPrompt, config, nil)
				cancel()
				status, accessAllowed := llm.ClassifyProbeError(err)
				if err == nil {
					status = llm.HealthStatusOnline
					accessAllowed = true
				}
				probe := llm.HealthProbeResult{Status: status, AccessAllowed: accessAllowed, Latency: time.Since(start), Err: err}
				profiler.UpdateModelHealthCheck(context.Background(), modelID, probe)
				profiler.UpdateCapabilityHealthCheck(context.Background(), provider, modelID, llm.CapabilityTextGeneration, probe)
				providerStatuses[provider] = append(providerStatuses[provider], probe.Status)
				providerAccess[provider] = providerAccess[provider] || probe.AccessAllowed
				log.Printf("Health check for text model %s: status=%s access=%v", modelID, probe.Status, probe.AccessAllowed)
				continue // Move to the next model
			}

			// Check if it's an image model using the modelID as the key
			if imageClient, ok := imageClients[modelID]; ok { // Now `imageClients` is keyed by modelID
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				start := time.Now()
				_, err := imageClient.GenerateImage(ctx, "a single red square", modelID)
				cancel()
				status, accessAllowed := llm.ClassifyProbeError(err)
				if err == nil {
					status = llm.HealthStatusOnline
					accessAllowed = true
				}
				probe := llm.HealthProbeResult{Status: status, AccessAllowed: accessAllowed, Latency: time.Since(start), Err: err}
				profiler.UpdateModelHealthCheck(context.Background(), modelID, probe)
				profiler.UpdateCapabilityHealthCheck(context.Background(), provider, modelID, llm.CapabilityImageGeneration, probe)
				providerStatuses[provider] = append(providerStatuses[provider], probe.Status)
				providerAccess[provider] = providerAccess[provider] || probe.AccessAllowed
				log.Printf("Health check for image model %s: status=%s access=%v", modelID, probe.Status, probe.AccessAllowed)
				continue // Move to the next model
			}

			if speechClient, ok := speechClients[modelID]; ok {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				start := time.Now()
				_, err := speechClient.SynthesizeSpeech(ctx, llm.SpeechSynthesisRequest{
					Model:  modelID,
					Input:  "Health check.",
					Voice:  "alloy",
					Format: "mp3",
				})
				cancel()
				status, accessAllowed := llm.ClassifyProbeError(err)
				if err == nil {
					status = llm.HealthStatusOnline
					accessAllowed = true
				}
				probe := llm.HealthProbeResult{Status: status, AccessAllowed: accessAllowed, Latency: time.Since(start), Err: err}
				profiler.UpdateModelHealthCheck(context.Background(), modelID, probe)
				profiler.UpdateCapabilityHealthCheck(context.Background(), provider, modelID, llm.CapabilityTTS, probe)
				providerStatuses[provider] = append(providerStatuses[provider], probe.Status)
				providerAccess[provider] = providerAccess[provider] || probe.AccessAllowed
				log.Printf("Health check for speech model %s: status=%s access=%v", modelID, probe.Status, probe.AccessAllowed)
				continue
			}

			log.Printf("WARNING: Model '%s' is in the enabled list but no client was found for it. Skipping health check.", modelID)
		}

		if ragService != nil && embeddingModel != "" {
			provider := llm.ProviderForModel(embeddingModel)
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			probe := ragService.ProbeEmbeddings(ctx)
			cancel()
			profiler.UpdateCapabilityHealthCheck(context.Background(), provider, embeddingModel, llm.CapabilityEmbeddings, probe)
			providerStatuses[provider] = append(providerStatuses[provider], probe.Status)
			providerAccess[provider] = providerAccess[provider] || probe.AccessAllowed
			log.Printf("Health check for embeddings %s: status=%s access=%v", embeddingModel, probe.Status, probe.AccessAllowed)
		}

		for provider, statuses := range providerStatuses {
			aggregateStatus := llm.CombineHealthStatuses(statuses...)
			profiler.UpdateProviderHealthCheck(context.Background(), provider, llm.HealthProbeResult{
				Status:        aggregateStatus,
				AccessAllowed: providerAccess[provider],
			})
		}
	}

	go runChecks()
	for range ticker.C {
		runChecks()
	}
}

// runServerWithGracefulShutdown handles the server lifecycle.
func runServerWithGracefulShutdown(srv *http.Server) {
	go func() {
		log.Printf("👂 Gateway is listening on http://localhost%s", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("❌ Listen error: %s\n", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("🛑 Shutting down server...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("❌ Server shutdown failed:", err)
	}

	log.Println("👋 Server exited gracefully.")
}

func cleanModelList(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}
