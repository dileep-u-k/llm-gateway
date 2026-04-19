package llm

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
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	defaultEmbeddingModel     = "text-embedding-3-small"
	defaultOpenAIAPIURL       = "https://api.openai.com/v1/embeddings"
	embeddingCachePrefix      = "embeddingcache:"
	responseCachePrefix       = "llmcache:"
	sourceManifestPrefix      = "rag:source:"
	corpusVersionKey          = "rag:corpus:version"
	retrievalProvenancePrefix = "rag:retrieval:"
	embeddingCacheTTL         = 7 * 24 * time.Hour
	responseCacheTTL          = 24 * time.Hour
	defaultRetrievalTopK      = 4
	defaultRetrievalMaxChunks = 3
	defaultRetrievalMaxChars  = 6000
	defaultRAGFailurePolicy   = "graceful_no_rag"
)

type Config struct {
	OpenAIKey                string
	PineconeKey              string
	PineconeHost             string
	RedisAddr                string
	EmbeddingModel           string
	OpenAIAPIURL             string
	RetrievalTopK            int
	RetrievalMaxChunks       int
	RetrievalMaxContextChars int
	EmbeddingCacheVersion    string
	ResponseCacheVersion     string
	CorpusVersion            string
	RAGFailurePolicy         string
}

type DocumentChunk struct {
	Text        string
	Topic       string
	Source      string
	DocumentID  string
	DocTitle    string
	Section     string
	SectionPath string
	Version     string
	Timestamp   string
	IngestedAt  string
	ChunkIndex  int
	ContentHash string
}

type RetrievedChunk struct {
	Text        string
	Topic       string
	Source      string
	DocumentID  string
	DocTitle    string
	Section     string
	SectionPath string
	Version     string
	Timestamp   string
	IngestedAt  string
	ChunkIndex  int
	ContentHash string
	Score       float64
	RerankScore float64
	Freshness   float64
	BudgetCost  int
}

type RetrievalPlan struct {
	RequestedTopK   int `json:"requested_top_k"`
	CandidatePool   int `json:"candidate_pool"`
	MaxChunks       int `json:"max_chunks"`
	MaxContextChars int `json:"max_context_chars"`
	MaxChunksPerDoc int `json:"max_chunks_per_doc"`
}

type RetrievalProvenance struct {
	Source       string  `json:"source,omitempty"`
	DocumentID   string  `json:"document_id,omitempty"`
	DocTitle     string  `json:"doc_title,omitempty"`
	Section      string  `json:"section,omitempty"`
	SectionPath  string  `json:"section_path,omitempty"`
	Version      string  `json:"version,omitempty"`
	Timestamp    string  `json:"timestamp,omitempty"`
	IngestedAt   string  `json:"ingested_at,omitempty"`
	ChunkIndex   int     `json:"chunk_index,omitempty"`
	Score        float64 `json:"score,omitempty"`
	RerankScore  float64 `json:"rerank_score,omitempty"`
	Freshness    float64 `json:"freshness,omitempty"`
	ContentHash  string  `json:"content_hash,omitempty"`
	ContextChars int     `json:"context_chars,omitempty"`
}

type RetrievalResult struct {
	RetrievalID      string
	Context          string
	Score            float64
	CandidateCount   int
	SelectedCount    int
	SourceDiversity  int
	BudgetUsedChars  int
	Chunks           []RetrievedChunk
	Plan             RetrievalPlan
	Provenance       []RetrievalProvenance
	VersionSignature string
	RetrievalLatency time.Duration
	StaleSources     []string
}

type RAGService struct {
	config      *Config
	httpClient  *http.Client
	redisClient *redis.Client
}

func LoadConfig() (*Config, error) {
	cfg := &Config{
		OpenAIKey:                os.Getenv("OPENAI_API_KEY"),
		PineconeKey:              os.Getenv("PINECONE_API_KEY"),
		PineconeHost:             os.Getenv("PINECONE_INDEX_HOST"),
		RedisAddr:                os.Getenv("REDIS_ADDR"),
		EmbeddingModel:           getEnv("EMBEDDING_MODEL", defaultEmbeddingModel),
		OpenAIAPIURL:             getEnv("OPENAI_API_URL", defaultOpenAIAPIURL),
		RetrievalTopK:            getEnvInt("RAG_TOP_K", defaultRetrievalTopK),
		RetrievalMaxChunks:       getEnvInt("RAG_MAX_CHUNKS", defaultRetrievalMaxChunks),
		RetrievalMaxContextChars: getEnvInt("RAG_MAX_CONTEXT_CHARS", defaultRetrievalMaxChars),
		EmbeddingCacheVersion:    getEnv("EMBEDDING_CACHE_VERSION", "v2"),
		ResponseCacheVersion:     getEnv("RESPONSE_CACHE_VERSION", "v2"),
		CorpusVersion:            getEnv("RAG_CORPUS_VERSION", ""),
		RAGFailurePolicy:         getEnv("RAG_FAILURE_POLICY", defaultRAGFailurePolicy),
	}

	if cfg.OpenAIKey == "" || cfg.PineconeKey == "" || cfg.PineconeHost == "" || cfg.RedisAddr == "" {
		return nil, errors.New("OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_HOST, and REDIS_ADDR must be set")
	}
	return cfg, nil
}

func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func NewRAGService(cfg *Config) (*RAGService, error) {
	rdb := redis.NewClient(&redis.Options{Addr: cfg.RedisAddr})
	if _, err := rdb.Ping(context.Background()).Result(); err != nil {
		return nil, fmt.Errorf("could not connect to Redis: %w", err)
	}

	return &RAGService{
		config:      cfg,
		httpClient:  &http.Client{Timeout: 30 * time.Second},
		redisClient: rdb,
	}, nil
}

func (s *RAGService) BuildEmbeddingCacheKey(text string) string {
	seed := strings.Join([]string{s.config.EmbeddingModel, s.config.EmbeddingCacheVersion, text}, "|")
	return embeddingCachePrefix + GenerateCacheKey(seed)
}

func (s *RAGService) BuildResponseCacheKey(promptHash, historyHash, modelFamily, routingMode, retrievedVersionSignature string) string {
	seed := strings.Join([]string{s.config.ResponseCacheVersion, promptHash, historyHash, modelFamily, routingMode, s.config.CorpusVersion, retrievedVersionSignature}, "|")
	return responseCachePrefix + GenerateCacheKey(seed)
}

func (s *RAGService) BuildResponseCacheKeyWithContext(ctx context.Context, promptHash, historyHash, modelFamily, routingMode, retrievedVersionSignature string) string {
	corpusVersion := s.CurrentCorpusVersion(ctx)
	seed := strings.Join([]string{s.config.ResponseCacheVersion, promptHash, historyHash, modelFamily, routingMode, corpusVersion, retrievedVersionSignature}, "|")
	return responseCachePrefix + GenerateCacheKey(seed)
}

func (s *RAGService) CheckCache(ctx context.Context, cacheKey string) (string, bool) {
	val, err := s.redisClient.Get(ctx, cacheKey).Result()
	if err == redis.Nil {
		return "", false
	}
	if err != nil {
		log.Printf("Redis GET error for response cache: %v", err)
		return "", false
	}
	return val, true
}

func (s *RAGService) SetCache(ctx context.Context, cacheKey, response string) {
	if err := s.redisClient.Set(ctx, cacheKey, response, responseCacheTTL).Err(); err != nil {
		log.Printf("Redis SET error for response cache: %v", err)
	}
}

func (s *RAGService) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	cacheKey := s.BuildEmbeddingCacheKey(text)
	cachedEmbedding, err := s.redisClient.Get(ctx, cacheKey).Bytes()
	if err == nil {
		var embedding []float32
		if unmarshalErr := json.Unmarshal(cachedEmbedding, &embedding); unmarshalErr == nil {
			log.Println("Embedding cache HIT")
			return embedding, nil
		}
		log.Printf("Error unmarshalling cached embedding: %v", err)
	} else if err != redis.Nil {
		log.Printf("Redis GET error for embedding: %v", err)
	}
	log.Println("Embedding cache MISS")

	type apiRequest struct {
		Input string `json:"input"`
		Model string `json:"model"`
	}
	type apiResponse struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}

	payloadBytes, err := json.Marshal(apiRequest{Input: text, Model: s.config.EmbeddingModel})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal OpenAI request: %w", err)
	}
	body, err := s.postJSONWithRetry(ctx, s.config.OpenAIAPIURL, payloadBytes, map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + s.config.OpenAIKey,
	})
	if err != nil {
		return nil, fmt.Errorf("OpenAI embedding API request failed: %w", err)
	}

	var apiResp apiResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal OpenAI response: %w", err)
	}
	if len(apiResp.Data) == 0 {
		return nil, errors.New("no embedding data returned from API")
	}
	embedding := apiResp.Data[0].Embedding

	embeddingBytes, err := json.Marshal(embedding)
	if err != nil {
		log.Printf("Error marshalling embedding for cache: %v", err)
	} else if err := s.redisClient.Set(ctx, cacheKey, embeddingBytes, embeddingCacheTTL).Err(); err != nil {
		log.Printf("Failed to set embedding cache in Redis: %v", err)
	}

	return embedding, nil
}

func (s *RAGService) ProbeEmbeddings(ctx context.Context) HealthProbeResult {
	start := time.Now()
	_, err := s.GetEmbedding(ctx, "health check")
	status, accessAllowed := classifyProbeError(err)
	if err == nil {
		status = HealthStatusOnline
		accessAllowed = true
	}
	return HealthProbeResult{
		Status:        status,
		AccessAllowed: accessAllowed,
		Latency:       time.Since(start),
		Err:           err,
	}
}

func (s *RAGService) QueryPinecone(ctx context.Context, embedding []float32, topK int, query string) ([]RetrievedChunk, error) {
	type match struct {
		Score    float64                `json:"score"`
		Metadata map[string]interface{} `json:"metadata"`
	}
	type apiResponse struct {
		Matches []match `json:"matches"`
	}
	type apiRequest struct {
		Vector          []float32 `json:"vector"`
		TopK            int       `json:"topK"`
		IncludeMetadata bool      `json:"includeMetadata"`
	}

	requestedTopK := topK * 2
	if requestedTopK < topK {
		requestedTopK = topK
	}
	payloadBytes, err := json.Marshal(apiRequest{Vector: embedding, TopK: requestedTopK, IncludeMetadata: true})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Pinecone request: %w", err)
	}
	body, err := s.postJSONWithRetry(ctx, s.config.PineconeHost+"/query", payloadBytes, map[string]string{
		"Content-Type": "application/json",
		"Api-Key":      s.config.PineconeKey,
	})
	if err != nil {
		return nil, fmt.Errorf("pinecone query API request failed: %w", err)
	}

	var apiResp apiResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Pinecone response: %w", err)
	}
	if len(apiResp.Matches) == 0 {
		return nil, nil
	}

	chunks := make([]RetrievedChunk, 0, len(apiResp.Matches))
	for _, match := range apiResp.Matches {
		chunk := RetrievedChunk{
			Text:        toString(match.Metadata["text"]),
			Topic:       toString(match.Metadata["topic"]),
			Source:      toString(match.Metadata["source"]),
			DocumentID:  firstNonEmptyString(toString(match.Metadata["document_id"]), toString(match.Metadata["doc_id"]), toString(match.Metadata["source"])),
			DocTitle:    firstNonEmptyString(toString(match.Metadata["doc_title"]), toString(match.Metadata["title"]), toString(match.Metadata["source"])),
			Section:     toString(match.Metadata["section"]),
			SectionPath: firstNonEmptyString(toString(match.Metadata["section_path"]), toString(match.Metadata["section"])),
			Version:     toString(match.Metadata["version"]),
			Timestamp:   toString(match.Metadata["timestamp"]),
			IngestedAt:  firstNonEmptyString(toString(match.Metadata["ingested_at"]), toString(match.Metadata["timestamp"])),
			ChunkIndex:  toInt(match.Metadata["chunk_index"]),
			ContentHash: toString(match.Metadata["content_hash"]),
			Score:       match.Score,
		}
		chunk.Freshness = recencyBoost(chunk.Timestamp, chunk.IngestedAt)
		chunk.RerankScore = rerankChunk(query, chunk)
		chunk.BudgetCost = estimatedChunkBudgetCost(chunk)
		chunks = append(chunks, chunk)
	}

	sort.SliceStable(chunks, func(i, j int) bool {
		if chunks[i].RerankScore == chunks[j].RerankScore {
			return chunks[i].ChunkIndex < chunks[j].ChunkIndex
		}
		return chunks[i].RerankScore > chunks[j].RerankScore
	})
	return chunks, nil
}

func (s *RAGService) RetrieveContext(ctx context.Context, text string, topK int) (*RetrievalResult, error) {
	start := time.Now()
	if topK <= 0 {
		topK = s.config.RetrievalTopK
	}
	if topK <= 0 {
		topK = defaultRetrievalTopK
	}

	maxChunks := s.config.RetrievalMaxChunks
	if maxChunks <= 0 {
		maxChunks = defaultRetrievalMaxChunks
	}
	maxChars := s.config.RetrievalMaxContextChars
	if maxChars <= 0 {
		maxChars = defaultRetrievalMaxChars
	}
	plan := RetrievalPlan{
		RequestedTopK:   topK,
		CandidatePool:   maxRetrievalInt(topK*3, topK),
		MaxChunks:       maxChunks,
		MaxContextChars: maxChars,
		MaxChunksPerDoc: 2,
	}

	embedding, err := s.GetEmbedding(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding for RAG context: %w", err)
	}
	chunks, err := s.QueryPinecone(ctx, embedding, plan.CandidatePool, text)
	if err != nil {
		return nil, fmt.Errorf("failed to query pinecone for RAG context: %w", err)
	}

	result := &RetrievalResult{
		RetrievalID:      GenerateCacheKey(text + "|" + time.Now().UTC().Format(time.RFC3339Nano)),
		RetrievalLatency: time.Since(start),
		CandidateCount:   len(chunks),
		Plan:             plan,
	}
	if len(chunks) == 0 {
		return result, nil
	}

	seen := make(map[string]struct{})
	perDoc := make(map[string]int)
	versionParts := make([]string, 0, maxChunks)
	staleSources := make(map[string]struct{})
	var selected []RetrievedChunk
	var contextBuilder strings.Builder

	for _, chunk := range chunks {
		if len(selected) >= plan.MaxChunks {
			break
		}
		if chunk.Text == "" {
			continue
		}
		if s.chunkIsStale(ctx, chunk) {
			staleSources[chunk.Source] = struct{}{}
			continue
		}
		dupeKey := chunk.ContentHash
		if dupeKey == "" {
			dupeKey = GenerateCacheKey(chunk.Source + "|" + chunk.Section + "|" + chunk.Text)
		}
		if _, exists := seen[dupeKey]; exists {
			continue
		}
		docKey := firstNonEmptyString(chunk.DocumentID, chunk.Source)
		if docKey != "" && perDoc[docKey] >= plan.MaxChunksPerDoc {
			continue
		}
		candidate := s.formatChunkContext(chunk)
		if contextBuilder.Len()+len(candidate) > plan.MaxContextChars {
			candidate = truncateChunkContext(candidate, plan.MaxContextChars-contextBuilder.Len())
			if candidate == "" {
				continue
			}
		}
		seen[dupeKey] = struct{}{}
		selected = append(selected, chunk)
		if docKey != "" {
			perDoc[docKey]++
		}
		versionParts = append(versionParts, chunk.Source+"@"+chunk.Version)
		contextBuilder.WriteString(candidate)
		contextBuilder.WriteString("\n\n")
	}

	result.Context = strings.TrimSpace(contextBuilder.String())
	result.Chunks = selected
	result.Score = selectedScore(selected)
	result.SelectedCount = len(selected)
	result.BudgetUsedChars = len(result.Context)
	result.SourceDiversity = len(perDoc)
	result.Provenance = buildRetrievalProvenance(selected)
	result.VersionSignature = GenerateCacheKey(strings.Join(versionParts, "|"))
	for source := range staleSources {
		result.StaleSources = append(result.StaleSources, source)
	}
	sort.Strings(result.StaleSources)
	result.RetrievalLatency = time.Since(start)
	s.StoreRetrievalProvenance(ctx, result)
	return result, nil
}

func (s *RAGService) chunkIsStale(ctx context.Context, chunk RetrievedChunk) bool {
	if chunk.Source == "" || chunk.Version == "" {
		return false
	}
	currentVersion, err := s.GetSourceVersion(ctx, chunk.Source)
	if err != nil || currentVersion == "" {
		return false
	}
	return currentVersion != chunk.Version
}

func (s *RAGService) formatChunkContext(chunk RetrievedChunk) string {
	var parts []string
	if chunk.Source != "" {
		parts = append(parts, "Source: "+chunk.Source)
	}
	if chunk.DocTitle != "" && chunk.DocTitle != chunk.Source {
		parts = append(parts, "Title: "+chunk.DocTitle)
	}
	if chunk.Section != "" {
		parts = append(parts, "Section: "+chunk.Section)
	}
	if chunk.Version != "" {
		parts = append(parts, "Version: "+chunk.Version)
	}
	if chunk.Timestamp != "" {
		parts = append(parts, "Updated: "+chunk.Timestamp)
	}
	header := strings.Join(parts, " | ")
	if header == "" {
		return chunk.Text
	}
	return header + "\n" + chunk.Text
}

func (s *RAGService) GenerateVectorsForChunks(ctx context.Context, chunks []DocumentChunk) ([]Vector, error) {
	type apiRequest struct {
		Input []string `json:"input"`
		Model string   `json:"model"`
	}
	type apiResponse struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}

	inputs := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		inputs = append(inputs, chunk.Text)
	}
	payloadBytes, err := json.Marshal(apiRequest{Input: inputs, Model: s.config.EmbeddingModel})
	if err != nil {
		return nil, err
	}
	body, err := s.postJSONWithRetry(ctx, s.config.OpenAIAPIURL, payloadBytes, map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + s.config.OpenAIKey,
	})
	if err != nil {
		return nil, err
	}
	var apiResp apiResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal OpenAI embedding response: %w", err)
	}
	if len(apiResp.Data) != len(chunks) {
		return nil, errors.New("mismatch between chunks and embeddings count")
	}

	vectors := make([]Vector, len(chunks))
	for i, chunk := range chunks {
		vectors[i] = Vector{
			ID:     GenerateCacheKey(chunk.Source + "::" + chunk.Version + "::" + chunk.ContentHash),
			Values: apiResp.Data[i].Embedding,
			Metadata: map[string]interface{}{
				"text":         chunk.Text,
				"topic":        chunk.Topic,
				"source":       chunk.Source,
				"document_id":  chunk.DocumentID,
				"doc_title":    chunk.DocTitle,
				"section":      chunk.Section,
				"section_path": chunk.SectionPath,
				"version":      chunk.Version,
				"timestamp":    chunk.Timestamp,
				"ingested_at":  chunk.IngestedAt,
				"chunk_index":  chunk.ChunkIndex,
				"content_hash": chunk.ContentHash,
			},
		}
	}
	return vectors, nil
}

func (s *RAGService) StoreSourceVersion(ctx context.Context, source, version string) error {
	if source == "" || version == "" {
		return nil
	}
	return s.redisClient.Set(ctx, sourceManifestPrefix+GenerateCacheKey(source), version, 30*24*time.Hour).Err()
}

func (s *RAGService) GetSourceVersion(ctx context.Context, source string) (string, error) {
	if source == "" {
		return "", nil
	}
	value, err := s.redisClient.Get(ctx, sourceManifestPrefix+GenerateCacheKey(source)).Result()
	if err == redis.Nil {
		return "", nil
	}
	return value, err
}

func (s *RAGService) SetCorpusVersion(ctx context.Context, version string) error {
	if version == "" {
		return nil
	}
	s.config.CorpusVersion = version
	return s.redisClient.Set(ctx, corpusVersionKey, version, 30*24*time.Hour).Err()
}

func (s *RAGService) CurrentCorpusVersion(ctx context.Context) string {
	if s == nil || s.config == nil {
		return ""
	}
	if s.redisClient != nil {
		value, err := s.redisClient.Get(ctx, corpusVersionKey).Result()
		if err == nil && value != "" {
			s.config.CorpusVersion = value
			return value
		}
	}
	return s.config.CorpusVersion
}

func (s *RAGService) StoreRetrievalProvenance(ctx context.Context, result *RetrievalResult) {
	if s == nil || s.redisClient == nil || result == nil || result.RetrievalID == "" {
		return
	}
	payload, err := json.Marshal(result)
	if err != nil {
		log.Printf("Failed to marshal retrieval provenance: %v", err)
		return
	}
	if err := s.redisClient.Set(ctx, retrievalProvenancePrefix+result.RetrievalID, payload, responseCacheTTL).Err(); err != nil {
		log.Printf("Failed to store retrieval provenance: %v", err)
	}
}

func (s *RAGService) postJSONWithRetry(ctx context.Context, url string, payload []byte, headers map[string]string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(payload))
	if err != nil {
		return nil, err
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	return s.doRequestWithRetry(req)
}

func (s *RAGService) doRequestWithRetry(req *http.Request) ([]byte, error) {
	var lastErr error
	delay := initialRetryDelay
	for i := 0; i < maxRetries; i++ {
		if req.Body != nil {
			bodyBytes, err := io.ReadAll(req.Body)
			if err != nil {
				return nil, fmt.Errorf("failed to read request body: %w", err)
			}
			req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
			req.GetBody = func() (io.ReadCloser, error) {
				return io.NopCloser(bytes.NewBuffer(bodyBytes)), nil
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
		if closeErr := resp.Body.Close(); closeErr != nil {
			log.Printf("warning: failed to close response body: %v", closeErr)
		}
		if readErr != nil {
			return nil, fmt.Errorf("failed to read response body: %w", readErr)
		}
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return body, nil
		}
		lastErr = fmt.Errorf("API error (attempt %d/%d): status %d, body: %s", i+1, maxRetries, resp.StatusCode, string(body))
		if resp.StatusCode >= 400 && resp.StatusCode < 500 && resp.StatusCode != 429 {
			return nil, lastErr
		}
		time.Sleep(delay)
		delay *= 2
	}
	return nil, lastErr
}

func lexicalOverlapScore(query, candidate string) float64 {
	queryTerms := tokenize(query)
	candidateTerms := tokenize(candidate)
	if len(queryTerms) == 0 || len(candidateTerms) == 0 {
		return 0
	}
	candidateSet := make(map[string]struct{}, len(candidateTerms))
	for _, term := range candidateTerms {
		candidateSet[term] = struct{}{}
	}
	matches := 0
	for _, term := range queryTerms {
		if _, ok := candidateSet[term]; ok {
			matches++
		}
	}
	return float64(matches) / float64(len(queryTerms))
}

func rerankChunk(query string, chunk RetrievedChunk) float64 {
	score := (0.62 * chunk.Score) + (0.23 * lexicalOverlapScore(query, chunk.Text)) + (0.15 * chunk.Freshness)
	if querySectionBoost(query, chunk.Section, chunk.SectionPath, chunk.DocTitle) > 0 {
		score += querySectionBoost(query, chunk.Section, chunk.SectionPath, chunk.DocTitle)
	}
	return score
}

func querySectionBoost(query string, fields ...string) float64 {
	normalizedQuery := strings.ToLower(query)
	for _, field := range fields {
		field = strings.ToLower(strings.TrimSpace(field))
		if field == "" {
			continue
		}
		for _, token := range tokenize(field) {
			if strings.Contains(normalizedQuery, token) {
				return 0.08
			}
		}
	}
	return 0
}

func recencyBoost(values ...string) float64 {
	for _, value := range values {
		ts, ok := parseRAGTime(value)
		if !ok {
			continue
		}
		ageHours := time.Since(ts).Hours()
		switch {
		case ageHours <= 24:
			return 1
		case ageHours <= 24*7:
			return 0.75
		case ageHours <= 24*30:
			return 0.4
		default:
			return 0.15
		}
	}
	return 0.1
}

func parseRAGTime(value string) (time.Time, bool) {
	if value == "" {
		return time.Time{}, false
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02 15:04:05", "2006-01-02"} {
		if parsed, err := time.Parse(layout, value); err == nil {
			return parsed.UTC(), true
		}
	}
	return time.Time{}, false
}

func estimatedChunkBudgetCost(chunk RetrievedChunk) int {
	headerCost := len(chunk.Source) + len(chunk.DocTitle) + len(chunk.Section) + len(chunk.Version) + len(chunk.Timestamp) + 32
	return len(chunk.Text) + headerCost
}

func truncateChunkContext(value string, remaining int) string {
	if remaining <= 80 {
		return ""
	}
	if len(value) <= remaining {
		return value
	}
	trimmed := strings.TrimSpace(value[:remaining-3])
	if trimmed == "" {
		return ""
	}
	return trimmed + "..."
}

func buildRetrievalProvenance(chunks []RetrievedChunk) []RetrievalProvenance {
	provenance := make([]RetrievalProvenance, 0, len(chunks))
	for _, chunk := range chunks {
		provenance = append(provenance, RetrievalProvenance{
			Source:       chunk.Source,
			DocumentID:   chunk.DocumentID,
			DocTitle:     chunk.DocTitle,
			Section:      chunk.Section,
			SectionPath:  chunk.SectionPath,
			Version:      chunk.Version,
			Timestamp:    chunk.Timestamp,
			IngestedAt:   chunk.IngestedAt,
			ChunkIndex:   chunk.ChunkIndex,
			Score:        chunk.Score,
			RerankScore:  chunk.RerankScore,
			Freshness:    chunk.Freshness,
			ContentHash:  chunk.ContentHash,
			ContextChars: estimatedChunkBudgetCost(chunk),
		})
	}
	return provenance
}

func maxRetrievalInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func tokenize(text string) []string {
	text = strings.ToLower(text)
	replacer := strings.NewReplacer(",", " ", ".", " ", ":", " ", ";", " ", "?", " ", "!", " ", "\n", " ", "\t", " ")
	text = replacer.Replace(text)
	parts := strings.Fields(text)
	filtered := parts[:0]
	for _, part := range parts {
		if len(part) < 3 {
			continue
		}
		filtered = append(filtered, part)
	}
	return filtered
}

func selectedScore(chunks []RetrievedChunk) float64 {
	if len(chunks) == 0 {
		return 0
	}
	return chunks[0].RerankScore
}

func toString(value interface{}) string {
	switch v := value.(type) {
	case string:
		return v
	case fmt.Stringer:
		return v.String()
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case int:
		return strconv.Itoa(v)
	default:
		return ""
	}
}

func toInt(value interface{}) int {
	switch v := value.(type) {
	case int:
		return v
	case float64:
		return int(v)
	case string:
		parsed, err := strconv.Atoi(v)
		if err == nil {
			return parsed
		}
	}
	return 0
}
