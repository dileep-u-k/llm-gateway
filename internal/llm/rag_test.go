package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func TestBuildResponseCacheKeyIncludesVersionSignature(t *testing.T) {
	mr := miniredis.RunT(t)
	service := newTestRAGService(t, mr, nil)

	keyA := service.BuildResponseCacheKey("prompt", "history", "gpt-4o", "balanced", "version-a")
	keyB := service.BuildResponseCacheKey("prompt", "history", "gpt-4o", "balanced", "version-b")
	if keyA == keyB {
		t.Fatalf("expected different version signatures to produce different cache keys")
	}
}

func TestBuildResponseCacheKeyWithContextIncludesCorpusVersion(t *testing.T) {
	mr := miniredis.RunT(t)
	service := newTestRAGService(t, mr, nil)
	ctx := context.Background()

	if err := service.SetCorpusVersion(ctx, "corpus-v1"); err != nil {
		t.Fatalf("SetCorpusVersion: %v", err)
	}
	keyA := service.BuildResponseCacheKeyWithContext(ctx, "prompt", "history", "gpt-4o", "balanced", "retrieved-v1")

	if err := service.SetCorpusVersion(ctx, "corpus-v2"); err != nil {
		t.Fatalf("SetCorpusVersion: %v", err)
	}
	keyB := service.BuildResponseCacheKeyWithContext(ctx, "prompt", "history", "gpt-4o", "balanced", "retrieved-v1")

	if keyA == keyB {
		t.Fatal("expected corpus version changes to invalidate response cache keys")
	}
}

func TestRetrieveContextFiltersStaleAndDuplicateChunks(t *testing.T) {
	mr := miniredis.RunT(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/embeddings":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"embedding": []float32{0.1, 0.2, 0.3}}},
			})
		case "/query":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"matches": []map[string]any{
					{
						"score": 0.92,
						"metadata": map[string]any{
							"text":         "Fresh architecture note",
							"topic":        "gateway",
							"source":       "docs/fresh.md",
							"doc_title":    "Fresh Notes",
							"section":      "Architecture",
							"section_path": "Gateway > Architecture",
							"version":      "fresh-v1",
							"timestamp":    "2026-04-19T00:00:00Z",
							"ingested_at":  "2026-04-19T00:00:00Z",
							"chunk_index":  0,
							"content_hash": "dup-hash",
						},
					},
					{
						"score": 0.91,
						"metadata": map[string]any{
							"text":         "Fresh architecture note",
							"topic":        "gateway",
							"source":       "docs/fresh.md",
							"section":      "Architecture",
							"version":      "fresh-v1",
							"timestamp":    "2026-04-19T00:00:00Z",
							"chunk_index":  1,
							"content_hash": "dup-hash",
						},
					},
					{
						"score": 0.95,
						"metadata": map[string]any{
							"text":         "Stale note should be ignored",
							"topic":        "gateway",
							"source":       "docs/stale.md",
							"section":      "Old",
							"version":      "old-v1",
							"timestamp":    "2026-04-10T00:00:00Z",
							"chunk_index":  0,
							"content_hash": "stale-hash",
						},
					},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	service := newTestRAGService(t, mr, server)
	if err := service.StoreSourceVersion(context.Background(), "docs/fresh.md", "fresh-v1"); err != nil {
		t.Fatalf("StoreSourceVersion fresh: %v", err)
	}
	if err := service.StoreSourceVersion(context.Background(), "docs/stale.md", "stale-v2"); err != nil {
		t.Fatalf("StoreSourceVersion stale: %v", err)
	}

	result, err := service.RetrieveContext(context.Background(), "architecture note", 2)
	if err != nil {
		t.Fatalf("RetrieveContext returned error: %v", err)
	}
	if len(result.Chunks) != 1 {
		t.Fatalf("expected one fresh deduplicated chunk, got %d", len(result.Chunks))
	}
	if result.Chunks[0].Source != "docs/fresh.md" {
		t.Fatalf("expected fresh chunk to survive filtering, got %+v", result.Chunks[0])
	}
	if result.Chunks[0].DocTitle != "Fresh Notes" || result.Chunks[0].SectionPath != "Gateway > Architecture" {
		t.Fatalf("expected provenance metadata to survive retrieval, got %+v", result.Chunks[0])
	}
	if len(result.StaleSources) != 1 || result.StaleSources[0] != "docs/stale.md" {
		t.Fatalf("expected stale source to be recorded, got %v", result.StaleSources)
	}
	if result.SelectedCount != 1 || result.SourceDiversity != 1 || len(result.Provenance) != 1 {
		t.Fatalf("expected retrieval metadata to be populated, got %+v", result)
	}
}

func newTestRAGService(t *testing.T, mr *miniredis.Miniredis, server *httptest.Server) *RAGService {
	t.Helper()
	baseURL := "http://example.invalid"
	if server != nil {
		baseURL = server.URL
	}
	cfg := &Config{
		OpenAIKey:                "test-key",
		PineconeKey:              "test-pinecone",
		PineconeHost:             baseURL,
		RedisAddr:                mr.Addr(),
		EmbeddingModel:           "text-embedding-3-small",
		OpenAIAPIURL:             baseURL + "/embeddings",
		RetrievalTopK:            2,
		RetrievalMaxChunks:       2,
		RetrievalMaxContextChars: 4096,
		EmbeddingCacheVersion:    "v2",
		ResponseCacheVersion:     "v2",
		RAGFailurePolicy:         "graceful_no_rag",
	}
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	return &RAGService{
		config:      cfg,
		httpClient:  &http.Client{},
		redisClient: rdb,
	}
}
