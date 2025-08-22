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

    "llm-gateway/internal/llm"
    "llm-gateway/internal/tools"

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

    // *** MODIFIED: Inject the new promptAnalyzer into the GatewayHandler. ***
    gatewayHandler := NewGatewayHandler(llmClients, profiler, router, ragService, intentAnalyzer, toolManager, promptAnalyzer, cfg, rdb)
    log.Println("✅ All services initialized.")

    // 3. START BACKGROUND PROCESSES
    go startHealthChecker(cfg.EnabledModels, llmClients, profiler)

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
    for modelID := range cfg.APIKeys {
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
func startHealthChecker(models []string, clients map[string]llm.LLMClient, profiler *llm.Profiler) {
    ticker := time.NewTicker(5 * time.Minute)
    defer ticker.Stop()

    log.Println("🩺 Health checker started.")

    runChecks := func() {
        log.Println("🩺 Running proactive health checks...")
        for _, modelID := range models {
            client, ok := clients[modelID]
            if !ok {
                continue
            }
            ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
            config := &llm.GenerationConfig{Model: modelID, MaxTokens: 5}
            healthCheckPrompt := []llm.Message{{Role: llm.RoleUser, Content: "What is the capital of India?"}}

            _, err := client.Generate(ctx, healthCheckPrompt, config, nil)
            cancel()

            isHealthy := err == nil
            profiler.UpdateProfileOnHealthCheck(context.Background(), modelID, isHealthy)
            log.Printf("Health check for %s: Healthy = %v", modelID, isHealthy)
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