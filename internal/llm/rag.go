// In file: internal/llm/rag.go
package llm

import (
	"bytes"
	"context"
	"strings"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/redis/go-redis/v9"
)

// =================================================================================
// Configuration
// =================================================================================

const (
	// Default values for configuration if not set in the .env file.
	defaultEmbeddingModel = "text-embedding-3-small"
	defaultOpenAIAPIURL   = "https://api.openai.com/v1/embeddings"

	// Constants for caching and API interaction.
	embeddingCachePrefix = "embeddingcache:"
	responseCachePrefix  = "llmcache:"
	embeddingCacheTTL    = 7 * 24 * time.Hour // Cache embeddings for a week.
	responseCacheTTL     = 24 * time.Hour     // Cache final responses for a day.
	
)

// Config holds all the configuration for the RAG service.
// Loading from the environment makes the service portable and easy to configure.
type Config struct {
	OpenAIKey       string
	PineconeKey     string
	PineconeHost    string
	RedisAddr       string
	EmbeddingModel  string
	OpenAIAPIURL    string
}

// LoadConfig loads configuration from environment variables.
func LoadConfig() (*Config, error) {
	cfg := &Config{
		OpenAIKey:       os.Getenv("OPENAI_API_KEY"),
		PineconeKey:     os.Getenv("PINECONE_API_KEY"),
		PineconeHost:    os.Getenv("PINECONE_INDEX_HOST"),
		RedisAddr:       os.Getenv("REDIS_ADDR"),
		EmbeddingModel:  getEnv("EMBEDDING_MODEL", defaultEmbeddingModel),
		OpenAIAPIURL:    getEnv("OPENAI_API_URL", defaultOpenAIAPIURL),
	}

	if cfg.OpenAIKey == "" || cfg.PineconeKey == "" || cfg.PineconeHost == "" || cfg.RedisAddr == "" {
		return nil, errors.New("OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_HOST, and REDIS_ADDR must be set")
	}
	return cfg, nil
}

// getEnv is a helper to read an env var or return a default.
func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}


// =================================================================================
// RAG Service
// =================================================================================

// RAGService encapsulates all logic and dependencies needed for the RAG pipeline.
// This struct-based approach is a best practice for building testable and maintainable services.
type RAGService struct {
	config      *Config
	httpClient  *http.Client
	redisClient *redis.Client
}

// NewRAGService is the constructor for our RAG service.
// It initializes all necessary clients and returns a new service instance.
func NewRAGService(cfg *Config) (*RAGService, error) {
	// Initialize Redis client
	rdb := redis.NewClient(&redis.Options{
		Addr: cfg.RedisAddr,
	})
	// Ping Redis to ensure the connection is alive.
	if _, err := rdb.Ping(context.Background()).Result(); err != nil {
		return nil, fmt.Errorf("could not connect to Redis: %w", err)
	}

	return &RAGService{
		config: cfg,
		httpClient: &http.Client{
			Timeout: 30 * time.Second, // Set a reasonable timeout for external API calls.
		},
		redisClient: rdb,
	}, nil
}

// GetEmbedding retrieves a vector embedding for a given text string.
// It implements a caching layer to avoid re-calculating embeddings for the same text,
// saving both time and money on API calls.
func (s *RAGService) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	// 1. Check cache first.
	cacheKey := embeddingCachePrefix + GenerateCacheKey(text)
	cachedEmbedding, err := s.redisClient.Get(ctx, cacheKey).Bytes()
	if err == nil {
		// Cache hit!
		var embedding []float32
		if err := json.Unmarshal(cachedEmbedding, &embedding); err == nil {
			log.Println("Embedding cache HIT")
			return embedding, nil
		}
		log.Printf("Error unmarshalling cached embedding: %v", err) // Log error but proceed to fetch fresh.
	} else if err != redis.Nil {
		log.Printf("Redis GET error for embedding: %v", err) // Log error but proceed.
	}
	log.Println("Embedding cache MISS")

	// 2. If cache miss, call the API.
	type APIRequest struct {
		Input string `json:"input"`
		Model string `json:"model"`
	}
	type APIResponse struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}

	payload := APIRequest{Input: text, Model: s.config.EmbeddingModel}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal OpenAI request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.config.OpenAIAPIURL, bytes.NewBuffer(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create OpenAI request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.config.OpenAIKey)

	body, err := s.doRequestWithRetry(req)
	if err != nil {
		return nil, fmt.Errorf("OpenAI embedding API request failed: %w", err)
	}

	var apiResp APIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal OpenAI response: %w", err)
	}
	if len(apiResp.Data) == 0 {
		return nil, errors.New("no embedding data returned from API")
	}
	embedding := apiResp.Data[0].Embedding

	// 3. Store the new embedding in the cache before returning.
	embeddingBytes, err := json.Marshal(embedding)
	if err != nil {
		log.Printf("Error marshalling embedding for cache: %v", err)
	} else {
		err := s.redisClient.Set(ctx, cacheKey, embeddingBytes, embeddingCacheTTL).Err()
		if err != nil {
			log.Printf("Failed to set embedding cache in Redis: %v", err)
		}
	}

	return embedding, nil
}


// QueryPinecone queries the Pinecone index to find the most relevant document chunks.
// It returns the concatenated context text, the topic of the top match, and its confidence score.
func (s *RAGService) QueryPinecone(ctx context.Context, embedding []float32, topK int) (string, string, float64, error) {
	type Match struct {
		Score    float64 `json:"score"`
		Metadata struct {
			Text  string `json:"text"`
			Topic string `json:"topic"`
		} `json:"metadata"`
	}
	type APIResponse struct {
		Matches []Match `json:"matches"`
	}
	type APIRequest struct {
		Vector          []float32 `json:"vector"`
		TopK            int       `json:"topK"`
		IncludeMetadata bool      `json:"includeMetadata"`
	}

	payload := APIRequest{
		Vector:          embedding,
		TopK:            topK,
		IncludeMetadata: true,
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", "", 0.0, fmt.Errorf("failed to marshal Pinecone request: %w", err)
	}

	queryURL := s.config.PineconeHost + "/query"
	req, err := http.NewRequestWithContext(ctx, "POST", queryURL, bytes.NewBuffer(payloadBytes))
	if err != nil {
		return "", "", 0.0, fmt.Errorf("failed to create Pinecone request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Api-Key", s.config.PineconeKey)

	body, err := s.doRequestWithRetry(req)
	if err != nil {
		return "", "", 0.0, fmt.Errorf("Pinecone query API request failed: %w", err)
	}

	var apiResp APIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", "", 0.0, fmt.Errorf("failed to unmarshal Pinecone response: %w", err)
	}

	if len(apiResp.Matches) == 0 {
		return "", "", 0.0, nil // No matches found is not an error, just an empty result.
	}

	// Build the context from all matched documents.
	var contextBuilder strings.Builder
	for _, match := range apiResp.Matches {
		contextBuilder.WriteString(match.Metadata.Text)
		contextBuilder.WriteString("\n\n")
	}

	// The top match determines the primary topic and score.
	topMatch := apiResp.Matches[0]
	return strings.TrimSpace(contextBuilder.String()), topMatch.Metadata.Topic, topMatch.Score, nil
}

// =================================================================================
// Caching for Final Responses
// =================================================================================

// CheckCache looks for a final LLM response in Redis.
// It uses a hash of the prompt as the key for efficiency and consistency.
func (s *RAGService) CheckCache(ctx context.Context, prompt string) (string, bool) {
	cacheKey := responseCachePrefix + GenerateCacheKey(prompt)
	val, err := s.redisClient.Get(ctx, cacheKey).Result()
	if err == redis.Nil {
		return "", false // Cache miss.
	} else if err != nil {
		log.Printf("Redis GET error for response cache: %v", err)
		return "", false // Treat error as a cache miss.
	}
	return val, true // Cache hit.
}

// SetCache adds a final LLM response to the Redis cache.
func (s *RAGService) SetCache(ctx context.Context, prompt, response string) {
	cacheKey := responseCachePrefix + GenerateCacheKey(prompt)
	err := s.redisClient.Set(ctx, cacheKey, response, responseCacheTTL).Err()
	if err != nil {
		log.Printf("Redis SET error for response cache: %v", err)
	}
}

// =================================================================================
// Utility and Helper Functions
// =================================================================================

// doRequestWithRetry is a robust utility to perform an HTTP request with automatic retries.
// It uses exponential backoff to gracefully handle transient network or API errors.
// CORRECTED: This is the robust, production-grade retry function.
func (s *RAGService) doRequestWithRetry(req *http.Request) ([]byte, error) {
	var lastErr error
	delay := initialRetryDelay
	for i := 0; i < maxRetries; i++ {
		// IMPORTANT: For retries, the request body must be readable multiple times.
		// We ensure this by setting GetBody, which the http.Client will use on retries.
		if req.Body != nil {
			bodyBytes, err := io.ReadAll(req.Body)
			if err != nil {
				return nil, fmt.Errorf("failed to read request body: %w", err)
			}
			req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes)) // Restore body for the first attempt
			req.GetBody = func() (io.ReadCloser, error) {
				return io.NopCloser(bytes.NewBuffer(bodyBytes)), nil // Provide fresh body for retries
			}
		}

		resp, err := s.httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("request failed (attempt %d/%d): %w", i+1, maxRetries, err)
			log.Println(lastErr)
			time.Sleep(delay)
			delay *= 2
			continue
		}

		body, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("failed to read response body: %w", readErr)
		}
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return body, nil
		}
		lastErr = fmt.Errorf("API error (attempt %d/%d): status %d, body: %s", i+1, maxRetries, resp.StatusCode, string(body))
		if resp.StatusCode >= 400 && resp.StatusCode < 500 {
			return nil, lastErr // Do not retry on client errors like 4xx.
		}
		time.Sleep(delay)
		delay *= 2
	}
	return nil, lastErr
}

// RetrieveContext is a high-level method that gets an embedding and queries Pinecone.
func (s *RAGService) RetrieveContext(ctx context.Context, text string, topK int) (string, float64, error) {
    embedding, err := s.GetEmbedding(ctx, text)
    if err != nil {
        return "", 0.0, fmt.Errorf("failed to get embedding for RAG context: %w", err)
    }

    // The QueryPinecone function now returns context, topic, and score. We only need context and score here.
    contextText, _, score, err := s.QueryPinecone(ctx, embedding, topK)
    if err != nil {
        return "", 0.0, fmt.Errorf("failed to query pinecone for RAG context: %w", err)
    }

    return contextText, score, nil
}

// GenerateVectorsForChunks is a new batch-processing method for the ingestor.
func (s *RAGService) GenerateVectorsForChunks(ctx context.Context, chunks []string, topic string) ([]Vector, error) {
	// This logic is moved from the ingestor to ensure consistency.
	// It calls the OpenAI batch embedding endpoint.
	type APIRequest struct {
		Input []string `json:"input"`
		Model string   `json:"model"`
	}
	type APIResponse struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}

	payload := APIRequest{Input: chunks, Model: s.config.EmbeddingModel}
	payloadBytes, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(ctx, "POST", s.config.OpenAIAPIURL, bytes.NewBuffer(payloadBytes))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.config.OpenAIKey)
	body, err := s.doRequestWithRetry(req)
	if err != nil {
		return nil, err
	}
	var apiResp APIResponse
	if json.Unmarshal(body, &apiResp) != nil {
		return nil, fmt.Errorf("failed to unmarshal OpenAI embedding response")
	}
	if len(apiResp.Data) != len(chunks) {
		return nil, errors.New("mismatch between chunks and embeddings count")
	}
	vectors := make([]Vector, len(chunks))
	for i, chunk := range chunks {
		vectors[i] = Vector{
			ID:     GenerateCacheKey(topic + "::" + chunk), // Using the central helper
			Values: apiResp.Data[i].Embedding,
			Metadata: map[string]interface{}{
				"text":  chunk,
				"topic": topic,
			},
		}
	}
	return vectors, nil
}