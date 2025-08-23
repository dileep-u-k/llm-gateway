// In file: cmd/ingestor/main.go

// Package main implements the data ingestion service for the LLM Gateway.
// This is an offline command-line tool responsible for processing source documents,
// generating vector embeddings, and populating the necessary databases.
// It's a dual-purpose pipeline:
// 1. It ingests knowledge documents into a vector database (Pinecone) for the RAG system.
// 2. It ingests training examples into Redis to power the Intent Analyzer.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"llm-gateway/internal/llm"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
)

// =================================================================================
// Configuration
// =================================================================================

const (
	defaultEmbeddingModel = "text-embedding-3-small"
	defaultOpenAIAPIURL   = "https://api.openai.com/v1/embeddings"
	defaultSourceDataDir  = "./data"
	pineconeUpsertPath    = "/vectors/upsert"
	upsertBatchSize       = 100
	maxRetries            = 3
	initialRetryDelay     = 2 * time.Second
)

// Config is now simplified, as Redis is no longer needed by the ingestor.
type Config struct {
	OpenAIKey      string
	PineconeKey    string
	PineconeHost   string
	EmbeddingModel string
	OpenAIAPIURL   string
	SourceDataDir  string
}

// loadConfig is simplified.
func loadConfig() (*Config, error) {
	if err := godotenv.Load(".env"); err != nil {
		log.Println("Warning: .env file not found. Relying on environment variables.")
	}
	cfg := &Config{
		OpenAIKey:      os.Getenv("OPENAI_API_KEY"),
		PineconeKey:    os.Getenv("PINECONE_API_KEY"),
		PineconeHost:   os.Getenv("PINECONE_INDEX_HOST"),
		EmbeddingModel: getEnv("EMBEDDING_MODEL", defaultEmbeddingModel),
		OpenAIAPIURL:   getEnv("OPENAI_API_URL", defaultOpenAIAPIURL),
		SourceDataDir:  getEnv("SOURCE_DATA_DIR", defaultSourceDataDir),
	}
	if cfg.OpenAIKey == "" || cfg.PineconeKey == "" || cfg.PineconeHost == "" {
		return nil, errors.New("OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX_HOST must be set")
	}
	return cfg, nil
}

// getEnv is a helper to read an env var or return a default value.
func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

// =================================================================================
// Ingestor Service
// =================================================================================

// Ingestor is now a dedicated Pinecone pipeline.
type Ingestor struct {
	config     *Config
	httpClient *http.Client
	ragService *llm.RAGService
}

// NewIngestor is simpler without Redis dependencies.
func NewIngestor(cfg *Config, ragService *llm.RAGService) (*Ingestor, error) {
	return &Ingestor{
		config:     cfg,
		httpClient: &http.Client{Timeout: 60 * time.Second},
		ragService: ragService,
	}, nil
}

// main is simplified to reflect the ingestor's new focus.
func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("❌ Configuration Error: %v", err)
	}
	// The RAGService is still needed to get embeddings consistently.
	ragConfig, err := llm.LoadConfig()
	if err != nil {
		log.Fatalf("❌ Failed to load RAG Service config: %v", err)
	}
	ragService, err := llm.NewRAGService(ragConfig)
	if err != nil {
		log.Fatalf("❌ Failed to create RAG Service: %v", err)
	}
	ingestor, err := NewIngestor(cfg, ragService)
	if err != nil {
		log.Fatalf("❌ Failed to create ingestor: %v", err)
	}
	if err := ingestor.Run(); err != nil {
		log.Fatalf("❌ Ingestion process failed: %v", err)
	}
}

// Run is now a simpler loop that only processes RAG topics for Pinecone.
func (i *Ingestor) Run() error {
	log.Println("🚀 Starting RAG data ingestion process for Pinecone...")
	topics, err := i.discoverTopics()
	if err != nil {
		return fmt.Errorf("failed to discover document topics: %w", err)
	}
	var wg sync.WaitGroup
	for _, topic := range topics {
		wg.Add(1)
		go func(t string) {
			defer wg.Done()
			if err := i.ingestTopicToPinecone(t); err != nil {
				log.Printf("❌ Error ingesting topic %s to Pinecone: %v", t, err)
			}
		}(topic)
	}
	wg.Wait()
	log.Println("✅ Data ingestion complete.")
	return nil
}

// discoverTopics now explicitly ignores the 'intents' folder.
func (i *Ingestor) discoverTopics() ([]string, error) {
	var topics []string
	entries, err := os.ReadDir(i.config.SourceDataDir)
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		if entry.IsDir() && entry.Name() != "intents" {
			topics = append(topics, entry.Name())
		}
	}
	return topics, nil
}

func (i *Ingestor) ingestTopicToPinecone(topic string) error {
	topicPath := filepath.Join(i.config.SourceDataDir, topic)
	log.Printf("📚 Processing RAG topic for Pinecone: '%s'", topic)
	allChunks, err := i.extractChunksFromPath(topicPath)
	if err != nil {
		return fmt.Errorf("error extracting chunks for topic %s: %w", topic, err)
	}
	if len(allChunks) == 0 {
		log.Printf("No chunks found for topic %s, skipping.", topic)
		return nil
	}
	log.Printf("Found %d total text chunks for topic '%s'. Processing in batches...", len(allChunks), topic)
	const embeddingBatchSize = 500
	for j := 0; j < len(allChunks); j += embeddingBatchSize {
		end := j + embeddingBatchSize
		if end > len(allChunks) {
			end = len(allChunks)
		}
		chunkBatch := allChunks[j:end]
		batchNum := (j / embeddingBatchSize) + 1
		totalBatches := (len(allChunks) + embeddingBatchSize - 1) / embeddingBatchSize
		log.Printf("  -> Processing batch %d of %d for topic '%s'", batchNum, totalBatches, topic)

		// CORRECTED: Use the single, consistent RAGService for embeddings.
		vectors, err := i.ragService.GenerateVectorsForChunks(context.Background(), chunkBatch, topic)
		if err != nil {
			return fmt.Errorf("failed to generate embeddings for batch %d of topic %s: %w", batchNum, topic, err)
		}
		if err := i.upsertToPinecone(vectors); err != nil {
			return fmt.Errorf("failed to upsert vectors for batch %d of topic %s: %w", batchNum, topic, err)
		}
	}
	return nil
}

// =================================================================================
// Helper Functions
// =================================================================================
// (These functions are kept from the previous version as they are still needed)

// extractChunksFromPath walks a directory and extracts all text chunks from valid files.
func (i *Ingestor) extractChunksFromPath(rootPath string) ([]string, error) {
	var chunks []string
	err := filepath.Walk(rootPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			fileChunks, err := extractChunksFromFile(path)
			if err != nil {
				log.Printf("⚠️  Could not extract chunks from file %s: %v", path, err)
				return nil
			}
			chunks = append(chunks, fileChunks...)
		}
		return nil
	})
	return chunks, err
}

// extractChunksFromFile uses a hybrid strategy for the most robust chunking.
func extractChunksFromFile(path string) ([]string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		// Only process supported files, otherwise skip.
		if os.IsNotExist(err) || !strings.HasSuffix(path, ".md") && !strings.HasSuffix(path, ".txt") {
			log.Printf("Unsupported file type or not found: %s. Skipping.", path)
			return nil, nil
		}
		return nil, err
	}

	finalChunks := []string{}

	// --- Pass 1: Semantic Chunking ---
	// First, split the document by major headings to respect semantic boundaries.
	sections := strings.Split(string(content), "\n# ")

	// --- Pass 2: Fixed-Size Chunking (if needed) ---
	const targetTokensPerChunk = 500
	const overlapTokens = 50

	for i, section := range sections {
		// Add the heading back to all sections after the first one.
		if i > 0 {
			section = "# " + section
		}

		// If the semantic section is already a good size, just add it.
		if len(section)/4 <= targetTokensPerChunk {
			if strings.TrimSpace(section) != "" {
				finalChunks = append(finalChunks, strings.TrimSpace(section))
			}
			continue
		}

		// If the section is too long, apply fixed-size chunking to it.
		var currentChunk strings.Builder
		lines := strings.Split(section, "\n")

		for _, line := range lines {
			lineTokenCount := len(line) / 4
			currentChunkTokenCount := len(currentChunk.String()) / 4

			if currentChunkTokenCount+lineTokenCount > targetTokensPerChunk && currentChunk.Len() > 0 {
				finalChunks = append(finalChunks, currentChunk.String())

				lastChunk := currentChunk.String()
				overlapStart := len(lastChunk) - (overlapTokens * 4)
				if overlapStart < 0 {
					overlapStart = 0
				}

				currentChunk.Reset()
				currentChunk.WriteString(lastChunk[overlapStart:])
			}
			currentChunk.WriteString(line + "\n")
		}
		if currentChunk.Len() > 0 {
			finalChunks = append(finalChunks, currentChunk.String())
		}
	}

	return finalChunks, nil
}

// upsertToPinecone sends batches of vectors to the Pinecone API.
func (i *Ingestor) upsertToPinecone(vectors []llm.Vector) error {
	type APIRequest struct {
		Vectors []llm.Vector `json:"vectors"`
	}

	totalBatches := (len(vectors) + upsertBatchSize - 1) / upsertBatchSize
	for j := 0; j < len(vectors); j += upsertBatchSize {
		end := j + upsertBatchSize
		if end > len(vectors) {
			end = len(vectors)
		}
		batch := vectors[j:end]
		batchNumber := (j / upsertBatchSize) + 1

		log.Printf("Upserting batch %d/%d to Pinecone (%d vectors)...", batchNumber, totalBatches, len(batch))

		payload := APIRequest{Vectors: batch}
		payloadBytes, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("failed to marshal Pinecone request payload for batch %d: %w", batchNumber, err)
		}

		upsertURL := i.config.PineconeHost + pineconeUpsertPath
		req, err := http.NewRequest("POST", upsertURL, bytes.NewBuffer(payloadBytes))
		if err != nil {
			return fmt.Errorf("failed to create Pinecone request for batch %d: %w", batchNumber, err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Api-Key", i.config.PineconeKey)

		if _, err := i.doRequestWithRetry(req); err != nil {
			return fmt.Errorf("pinecone API request for batch %d failed after retries: %w", batchNumber, err)
		}
	}
	return nil
}

// doRequestWithRetry performs a robust HTTP request with retries.
func (i *Ingestor) doRequestWithRetry(req *http.Request) ([]byte, error) {
	var body []byte
	var err error
	delay := initialRetryDelay

	for k := 0; k < maxRetries; k++ {
		// Clone the request so we can reuse it in case of a retry.
		reqClone := req.Clone(req.Context())
		if req.Body != nil {
			reqClone.Body, err = req.GetBody()
			if err != nil {
				return nil, fmt.Errorf("failed to get request body for retry: %w", err)
			}
		}

		resp, err := i.httpClient.Do(reqClone)
		if err != nil {
			log.Printf("Request failed (attempt %d/%d): %v. Retrying in %v...", k+1, maxRetries, err, delay)
			time.Sleep(delay)
			delay *= 2 // Exponential backoff.
			continue
		}

		body, err = io.ReadAll(resp.Body)
		resp.Body.Close() // Close the body immediately after reading.
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %w", err)
		}

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return body, nil // Success!
		}

		// Handle non-successful status codes.
		err = fmt.Errorf("API returned non-2xx status: %d %s - %s", resp.StatusCode, resp.Status, string(body))
		log.Printf("Request failed (attempt %d/%d): %v. Retrying in %v...", k+1, maxRetries, err, delay)
		time.Sleep(delay)
		delay *= 2
	}
	return nil, fmt.Errorf("request failed after %d attempts: %w", maxRetries, err)
}
