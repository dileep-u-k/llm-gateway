package llm

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func TestMemoryEngineUpdateSessionPersistsKnowledgeSources(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	engine := NewMemoryEngine(rdb)

	snapshot, err := engine.UpdateSession(context.Background(), MemoryUpdate{
		ConversationID:  "convo-1",
		UserPrompt:      "Explain the routing design",
		ResponseContent: "Here is the routing design summary",
		Retrieval: &RetrievalResult{
			Chunks: []RetrievedChunk{
				{Source: "docs/architecture.md", DocTitle: "Architecture", Section: "Routing", Text: "Routing details"},
			},
		},
		Route: &RouteSelection{
			ModelID: "gpt-4o",
			Explanation: RouteExplanation{
				PolicyName:      "default",
				Strategy:        "default",
				SelectionReason: "selected highest-scoring healthy candidate",
			},
		},
		SessionMode:    "dynamic",
		EffectiveModel: "gpt-4o",
		AnswerMode:     string(AnswerModeGroundedPreferred),
	})
	if err != nil {
		t.Fatalf("UpdateSession returned error: %v", err)
	}
	if len(snapshot.Working.KnowledgeSources) != 1 || snapshot.Working.KnowledgeSources[0] != "docs/architecture.md" {
		t.Fatalf("expected knowledge source to be persisted, got %+v", snapshot.Working)
	}
	if snapshot.Summary.Summary == "" {
		t.Fatal("expected summary memory to be generated")
	}
}

func TestMemoryEngineCompactsOlderEvents(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	engine := NewMemoryEngine(rdb)

	for i := 0; i < 14; i++ {
		_, err := engine.UpdateSession(context.Background(), MemoryUpdate{
			ConversationID:  "convo-2",
			UserPrompt:      "step",
			ResponseContent: "response",
			EffectiveModel:  "gpt-4o",
		})
		if err != nil {
			t.Fatalf("UpdateSession iteration %d: %v", i, err)
		}
		time.Sleep(time.Millisecond)
	}

	snapshot, err := engine.LoadSnapshot(context.Background(), "convo-2", nil)
	if err != nil {
		t.Fatalf("LoadSnapshot returned error: %v", err)
	}
	if len(snapshot.ShortTerm) > engine.shortTermLimit {
		t.Fatalf("expected compacted short-term memory, got %d events", len(snapshot.ShortTerm))
	}
	if snapshot.Summary.Summary == "" {
		t.Fatal("expected compacted summary memory to be populated")
	}
}
