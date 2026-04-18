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

	profiler := llm.NewProfiler(rdb)
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

	// *** NEW: Initialize the PromptAnalyzer service. ***
	// This service will automatically select a routing preference if the user does not provide one.
	promptAnalyzer := llm.NewPromptAnalyzer()

	// Initialize Image Clients with robust, fatal error handling
	imageClients := make(map[string]llm.ImageClient)
	// --- FIX START: Iterate over ENABLED_IMAGE_MODELS for client creation ---
	enabledImageModelsList := strings.Split(cfg.EnabledImageModels, ",")
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

	// Correctly inject ALL components into the GatewayHandler.
	gatewayHandler := NewGatewayHandler(llmClients, profiler, router, ragService, intentAnalyzer, toolManager, promptAnalyzer, cfg, rdb, imageClients, imageRouter)
	log.Println("✅ All services initialized.")
	// --- END OF CORRECTION ---

	// 3. START BACKGROUND PROCESSES
	// Combine both text and image models into a single list for the health checker.
	allEnabledModels := append(cfg.EnabledModels, enabledImageModelsList...) // Use the parsed list
	go startHealthChecker(allEnabledModels, llmClients, imageClients, profiler)
	// --- END OF CORRECTION ---

	// 4. SETUP AND RUN THE WEB SERVER
	gin.SetMode(os.Getenv("GIN_MODE"))
	engine := gin.Default()
	v1 := engine.Group("/api/v1")
	{
		v1.POST("/generate", gatewayHandler.HandleGeneration)
	}

	srv := &http.Server{Addr: fmt.Sprintf(":%s", os.Getenv("PORT")), Handler: engine}
	runServerWithGracefulShutdown(srv)
}

// initializeLLMClients creates instances of the LLM clients based on config.
func initializeLLMClients(cfg *AppConfig) (map[string]llm.LLMClient, error) {
	clients := make(map[string]llm.LLMClient)
	var err error
	// --- THIS LINE IS THE FIX ---
	for _, modelID := range cfg.EnabledModels {
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
func startHealthChecker(allEnabledModels []string, clients map[string]llm.LLMClient, imageClients map[string]llm.ImageClient, profiler *llm.Profiler) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	log.Println("🩺 Health checker started.")

	runChecks := func() {
		log.Println("🩺 Running proactive health checks...")

		for _, modelID := range allEnabledModels {
			modelID = strings.TrimSpace(modelID) // Crucial: trim whitespace from model IDs

			// Check if it's a text model
			if client, ok := clients[modelID]; ok {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				config := &llm.GenerationConfig{Model: modelID, MaxTokens: 5}
				healthCheckPrompt := []llm.Message{{Role: llm.RoleUser, Content: "What is the capital of India?"}}

				_, err := client.Generate(ctx, healthCheckPrompt, config, nil)
				cancel()

				isHealthy := err == nil
				profiler.UpdateProfileOnHealthCheck(context.Background(), modelID, isHealthy)
				log.Printf("Health check for text model %s: Healthy = %v", modelID, isHealthy)
				continue // Move to the next model
			}

			// Check if it's an image model using the modelID as the key
			if imageClient, ok := imageClients[modelID]; ok { // Now `imageClients` is keyed by modelID
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				// Use a simple, generic prompt for image health checks.
				// The actual generated image is not used, just the success/failure of the call.
				_, err := imageClient.GenerateImage(ctx, "a single red square", modelID)
				cancel()

				isHealthy := err == nil
				profiler.UpdateProfileOnHealthCheck(context.Background(), modelID, isHealthy)
				log.Printf("Health check for image model %s: Healthy = %v", modelID, isHealthy)
				continue // Move to the next model
			}

			log.Printf("WARNING: Model '%s' is in the enabled list but no client was found for it. Skipping health check.", modelID)
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
