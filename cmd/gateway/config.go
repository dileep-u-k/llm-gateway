// In file: cmd/gateway/config.go
package main

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/dileep-u-k/llm-gateway/internal/llm"
	"github.com/dileep-u-k/llm-gateway/internal/platform"

	"github.com/joho/godotenv"
	"gopkg.in/yaml.v3"
)

// AppConfig holds all configuration for the gateway, loaded from the environment and config files.
type AppConfig struct {
	EnabledModels []string
	// --- ADD THIS LINE ---
	EnabledImageModels  string
	EnabledSpeechModels string
	APIKeys             map[string]string
	ModelCosts          map[string]map[string]float64
	ModelBudgets        map[string]float64
	RouterConfig        *llm.RouterConfig
	RAGConfig           *llm.Config // Assuming RAG config is needed
	RedisAddr           string
	NewsAPIKey          string
	PlatformConfig      *platform.Config
	ArtifactStorageRoot string
	HTTPEnabled         bool
	AsyncWorkersEnabled bool
	AsyncWorkers        int
}

// LoadConfig loads all configuration from a .env file, environment variables, and config.yaml.
func LoadConfig() (*AppConfig, error) {
	//if err := godotenv.Load(); err != nil {
	//  log.Println("WARNING: No .env file found, relying on system environment variables.")
	//}

	// --- THIS IS THE FIX ---
	// Only attempt to load a .env file if we are in a local development environment.
	// In Docker (where GIN_MODE="release"), configuration is provided directly as
	// environment variables by Docker Compose.
	if os.Getenv("GIN_MODE") != "release" {
		if err := godotenv.Load(); err != nil {
			log.Println("WARNING: No .env file found for local development.")
		}
	}
	// --- END OF FIX ---

	cfg := &AppConfig{
		APIKeys:      make(map[string]string),
		ModelCosts:   make(map[string]map[string]float64),
		ModelBudgets: make(map[string]float64),
		RedisAddr:    os.Getenv("REDIS_ADDR"),
		NewsAPIKey:   os.Getenv("NEWS_API_KEY"),
		// --- ADD THIS LINE ---
		EnabledImageModels:  os.Getenv("ENABLED_IMAGE_MODELS"),
		EnabledSpeechModels: os.Getenv("ENABLED_SPEECH_MODELS"),
		ArtifactStorageRoot: os.Getenv("ARTIFACT_STORAGE_ROOT"),
		HTTPEnabled:         parseEnvBool("HTTP_ENABLED", true),
		AsyncWorkersEnabled: parseEnvBool("ASYNC_WORKERS_ENABLED", true),
		AsyncWorkers:        parseEnvInt("ASYNC_WORKERS", 3),
	}

	// This loop now correctly handles both text and image model costs and budgets.
	allModelsStr := os.Getenv("ENABLED_MODELS") + "," + os.Getenv("ENABLED_IMAGE_MODELS") + "," + os.Getenv("ENABLED_SPEECH_MODELS")
	allModels := strings.Split(allModelsStr, ",")
	cfg.EnabledModels = splitNonEmpty(os.Getenv("ENABLED_MODELS"))

	for _, modelID := range allModels {
		if modelID == "" {
			continue
		}
		var apiKey string
		switch {
		case strings.HasPrefix(modelID, "gpt"), strings.HasPrefix(modelID, "dall-e"):
			apiKey = os.Getenv("OPENAI_API_KEY")
		case strings.HasPrefix(modelID, "claude"):
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
		case strings.HasPrefix(modelID, "gemini"), strings.HasPrefix(modelID, "imagen"):
			apiKey = os.Getenv("GEMINI_API_KEY")
		case strings.HasPrefix(modelID, "mistral"):
			apiKey = os.Getenv("MISTRAL_API_KEY")
		}

		if apiKey != "" {
			cfg.APIKeys[modelID] = apiKey
		}

		sanitizedModelID := strings.ReplaceAll(strings.ReplaceAll(modelID, "-", "_"), ".", "_")

		envCostInput := fmt.Sprintf("%s_COST_INPUT", strings.ToUpper(sanitizedModelID))
		envCostOutput := fmt.Sprintf("%s_COST_OUTPUT", strings.ToUpper(sanitizedModelID))
		costInput, errI := strconv.ParseFloat(os.Getenv(envCostInput), 64)
		costOutput, errO := strconv.ParseFloat(os.Getenv(envCostOutput), 64)

		// Note: For images, output cost is often 0.
		if errI == nil && errO == nil {
			cfg.ModelCosts[modelID] = map[string]float64{
				"input":  costInput / 1_000_000,
				"output": costOutput / 1_000_000,
			}
			// For image models, cost is per-image, not per-token.
			if strings.HasPrefix(modelID, "dall-e") || strings.HasPrefix(modelID, "imagen") {
				cfg.ModelCosts[modelID]["input"] = costInput
				cfg.ModelCosts[modelID]["output"] = costOutput
			}
		}

		envBudget := fmt.Sprintf("%s_BUDGET_USD", strings.ToUpper(sanitizedModelID))
		if budget, err := strconv.ParseFloat(os.Getenv(envBudget), 64); err == nil {
			cfg.ModelBudgets[modelID] = budget
		}
	}

	routerConfigFile, err := os.ReadFile("config.yaml")
	if err != nil {
		return nil, fmt.Errorf("failed to read router config.yaml: %w", err)
	}
	if err := yaml.Unmarshal(routerConfigFile, &cfg.RouterConfig); err != nil {
		return nil, fmt.Errorf("failed to parse router config.yaml: %w", err)
	}

	ragCfg, err := llm.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load RAG config: %w", err)
	}
	cfg.RAGConfig = ragCfg

	platformCfg, err := platform.LoadConfig("platform.yaml")
	if err != nil {
		return nil, fmt.Errorf("failed to load platform config: %w", err)
	}
	cfg.PlatformConfig = platformCfg
	if strings.TrimSpace(cfg.ArtifactStorageRoot) == "" {
		cfg.ArtifactStorageRoot = platformCfg.Defaults.ArtifactRoot
	}

	return cfg, nil
}

func splitNonEmpty(value string) []string {
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func parseEnvBool(key string, fallback bool) bool {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	switch strings.ToLower(raw) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func parseEnvInt(key string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}
