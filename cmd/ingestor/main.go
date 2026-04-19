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
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/dileep-u-k/llm-gateway/internal/llm"

	"github.com/joho/godotenv"
)

type semanticSection struct {
	Title string
	Path  string
	Body  string
}

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
	mu         sync.Mutex
	sourceRefs map[string]string
}

// NewIngestor is simpler without Redis dependencies.
func NewIngestor(cfg *Config, ragService *llm.RAGService) (*Ingestor, error) {
	return &Ingestor{
		config:     cfg,
		httpClient: &http.Client{Timeout: 60 * time.Second},
		ragService: ragService,
		sourceRefs: make(map[string]string),
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
	if err := i.publishCorpusVersion(); err != nil {
		return fmt.Errorf("failed to publish corpus version: %w", err)
	}
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
	allChunks, err := i.extractChunksFromPath(topicPath, topic)
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
		vectors, err := i.ragService.GenerateVectorsForChunks(context.Background(), chunkBatch)
		if err != nil {
			return fmt.Errorf("failed to generate embeddings for batch %d of topic %s: %w", batchNum, topic, err)
		}
		if err := i.upsertToPinecone(vectors); err != nil {
			return fmt.Errorf("failed to upsert vectors for batch %d of topic %s: %w", batchNum, topic, err)
		}
		for _, chunk := range chunkBatch {
			if err := i.ragService.StoreSourceVersion(context.Background(), chunk.Source, chunk.Version); err != nil {
				log.Printf("⚠️ Failed to store source version for %s: %v", chunk.Source, err)
			}
			i.recordSourceVersion(chunk.Source, chunk.Version)
		}
	}
	return nil
}

// =================================================================================
// Helper Functions
// =================================================================================
// (These functions are kept from the previous version as they are still needed)

// extractChunksFromPath walks a directory and extracts all text chunks from valid files.
func (i *Ingestor) extractChunksFromPath(rootPath, topic string) ([]llm.DocumentChunk, error) {
	var chunks []llm.DocumentChunk
	err := filepath.Walk(rootPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			fileChunks, err := extractChunksFromFile(path, topic)
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
func extractChunksFromFile(path, topic string) ([]llm.DocumentChunk, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		// Only process supported files, otherwise skip.
		if os.IsNotExist(err) || !strings.HasSuffix(path, ".md") && !strings.HasSuffix(path, ".txt") {
			log.Printf("Unsupported file type or not found: %s. Skipping.", path)
			return nil, nil
		}
		return nil, err
	}

	fileInfo, statErr := os.Stat(path)
	if statErr != nil {
		return nil, statErr
	}

	finalChunks := []llm.DocumentChunk{}
	fileVersion := llm.GenerateCacheKey(string(content))
	sourceID := filepath.ToSlash(path)
	documentTitle := baseDocumentTitle(path)
	sections := hierarchicalSections(string(content), documentTitle)
	const targetTokensPerChunk = 500
	ingestedAt := time.Now().UTC().Format(time.RFC3339)

	for _, section := range sections {
		for _, chunkText := range splitSectionIntoChunks(section, targetTokensPerChunk) {
			chunkText = strings.TrimSpace(chunkText)
			if chunkText == "" {
				continue
			}
			finalChunks = append(finalChunks, llm.DocumentChunk{
				Text:        chunkText,
				Topic:       topic,
				Source:      sourceID,
				DocumentID:  sourceID,
				DocTitle:    documentTitle,
				Section:     section.Title,
				SectionPath: section.Path,
				Version:     fileVersion,
				Timestamp:   fileInfo.ModTime().UTC().Format(time.RFC3339),
				IngestedAt:  ingestedAt,
				ChunkIndex:  len(finalChunks) + 1,
				ContentHash: llm.GenerateCacheKey(chunkText),
			})
		}
	}

	return finalChunks, nil
}

func extractSectionTitle(section, fallback string) string {
	lines := strings.Split(strings.TrimSpace(section), "\n")
	if len(lines) == 0 {
		return fallback
	}
	title := strings.TrimSpace(strings.TrimPrefix(lines[0], "#"))
	if title == "" {
		return fallback
	}
	return title
}

func baseDocumentTitle(path string) string {
	base := filepath.Base(path)
	return strings.TrimSuffix(base, filepath.Ext(base))
}

func hierarchicalSections(content, fallbackTitle string) []semanticSection {
	lines := strings.Split(content, "\n")
	headings := []string{fallbackTitle}
	currentTitle := fallbackTitle
	var currentBody strings.Builder
	var sections []semanticSection

	flush := func() {
		body := strings.TrimSpace(currentBody.String())
		if body == "" {
			currentBody.Reset()
			return
		}
		sections = append(sections, semanticSection{
			Title: currentTitle,
			Path:  strings.Join(trimHeadingStack(headings), " > "),
			Body:  body,
		})
		currentBody.Reset()
	}

	for _, line := range lines {
		level, heading, ok := markdownHeading(line)
		if ok {
			flush()
			if level < 1 {
				level = 1
			}
			for len(headings) >= level+1 {
				headings = headings[:len(headings)-1]
			}
			headings = append(headings, heading)
			currentTitle = heading
			continue
		}
		currentBody.WriteString(line)
		currentBody.WriteString("\n")
	}
	flush()
	if len(sections) == 0 {
		return []semanticSection{{Title: fallbackTitle, Path: fallbackTitle, Body: strings.TrimSpace(content)}}
	}
	return sections
}

func markdownHeading(line string) (int, string, bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(trimmed, "#") {
		return 0, "", false
	}
	level := 0
	for level < len(trimmed) && trimmed[level] == '#' {
		level++
	}
	heading := strings.TrimSpace(trimmed[level:])
	if heading == "" {
		return 0, "", false
	}
	return level, heading, true
}

func trimHeadingStack(values []string) []string {
	var trimmed []string
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			trimmed = append(trimmed, value)
		}
	}
	return trimmed
}

func splitSectionIntoChunks(section semanticSection, targetTokensPerChunk int) []string {
	body := strings.TrimSpace(section.Body)
	if body == "" {
		return nil
	}
	if len(body)/4 <= targetTokensPerChunk {
		return []string{body}
	}
	paragraphs := strings.Split(body, "\n\n")
	var chunks []string
	var current strings.Builder
	var lastParagraph string

	flush := func() {
		text := strings.TrimSpace(current.String())
		if text != "" {
			chunks = append(chunks, text)
		}
		current.Reset()
		if lastParagraph != "" {
			current.WriteString(lastParagraph)
			current.WriteString("\n\n")
		}
	}

	for _, paragraph := range paragraphs {
		paragraph = strings.TrimSpace(paragraph)
		if paragraph == "" {
			continue
		}
		candidate := current.String()
		if len(candidate+paragraph)/4 > targetTokensPerChunk && current.Len() > 0 {
			flush()
		}
		if current.Len() > 0 {
			current.WriteString("\n\n")
		}
		current.WriteString(paragraph)
		lastParagraph = paragraph
	}
	if strings.TrimSpace(current.String()) != "" {
		chunks = append(chunks, strings.TrimSpace(current.String()))
	}
	return chunks
}

func (i *Ingestor) recordSourceVersion(source, version string) {
	if source == "" || version == "" {
		return
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	i.sourceRefs[source] = version
}

func (i *Ingestor) publishCorpusVersion() error {
	i.mu.Lock()
	refs := make([]string, 0, len(i.sourceRefs))
	for source, version := range i.sourceRefs {
		refs = append(refs, source+"@"+version)
	}
	i.mu.Unlock()

	if len(refs) == 0 {
		return nil
	}
	sort.Strings(refs)
	return i.ragService.SetCorpusVersion(context.Background(), llm.GenerateCacheKey(strings.Join(refs, "|")))
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

		//hjgvhjgghffgdd
		//resp.Body.Close() // Close the body immediately after reading.
		if err := resp.Body.Close(); err != nil {
			log.Printf("warning: failed to close response body: %v", err)
		}

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
