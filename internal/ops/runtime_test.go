package ops

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/dileep-u-k/llm-gateway/internal/llm"
	"github.com/redis/go-redis/v9"
)

func TestRuntimeProcessesAsyncJob(t *testing.T) {
	runtime, cleanup := newTestRuntime(t, func(context.Context, api.GenerationRequest, *llm.PreparedExecution) (api.GenerationResponse, error) {
		return api.GenerationResponse{
			Content:   "done",
			ModelUsed: "gpt-4o",
			Usage:     api.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			Route:     &api.RouteMetadata{SelectedProvider: "openai", SelectedModel: "gpt-4o", Strategy: "default"},
		}, nil
	})
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	runtime.Start(ctx)

	job, err := runtime.Submit(ctx, api.GenerationRequest{
		Prompt:                "Summarize this document.",
		SyncOrAsyncPreference: "async",
	}, preparedExecution())
	if err != nil {
		t.Fatalf("Submit returned error: %v", err)
	}

	waitForJobState(t, runtime, job.JobID, JobStateCompleted)
	stored, err := runtime.GetJob(context.Background(), job.JobID)
	if err != nil {
		t.Fatalf("GetJob returned error: %v", err)
	}
	if stored.Result == nil || stored.Result.Content != "done" {
		t.Fatalf("expected completed job result, got %+v", stored)
	}
	if len(stored.Checkpoints) == 0 || stored.Checkpoints[len(stored.Checkpoints)-1].Status != "completed" {
		t.Fatalf("expected completed checkpoints, got %+v", stored.Checkpoints)
	}
}

func TestRuntimeRetriesTransientFailure(t *testing.T) {
	attempts := 0
	runtime, cleanup := newTestRuntime(t, func(context.Context, api.GenerationRequest, *llm.PreparedExecution) (api.GenerationResponse, error) {
		attempts++
		if attempts == 1 {
			return api.GenerationResponse{}, errors.New("provider timeout while generating")
		}
		return api.GenerationResponse{
			Content:   "recovered",
			ModelUsed: "gpt-4o",
			Usage:     api.Usage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12},
			Route:     &api.RouteMetadata{SelectedProvider: "openai", SelectedModel: "gpt-4o", Strategy: "default"},
		}, nil
	})
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	runtime.Start(ctx)

	job, err := runtime.Submit(ctx, api.GenerationRequest{
		Prompt:                "Create the answer with retry.",
		SyncOrAsyncPreference: "async",
	}, preparedExecution())
	if err != nil {
		t.Fatalf("Submit returned error: %v", err)
	}

	waitForJobState(t, runtime, job.JobID, JobStateCompleted)
	stored, err := runtime.GetJob(context.Background(), job.JobID)
	if err != nil {
		t.Fatalf("GetJob returned error: %v", err)
	}
	if stored.RetryCount < 1 {
		t.Fatalf("expected at least one retry, got %+v", stored)
	}
	if stored.Result == nil || stored.Result.Content != "recovered" {
		t.Fatalf("expected recovered result, got %+v", stored.Result)
	}
}

func TestEvaluatorAndCanaryPolicy(t *testing.T) {
	evaluator := Evaluator{}
	req := api.GenerationRequest{
		Prompt: "Generate an infographic.",
		Rollout: api.RolloutOptions{
			CanaryModel:   "gpt-4o",
			CanaryPercent: 100,
		},
		Evaluation: api.EvaluationOptions{Enabled: true},
	}
	canary := CanaryPolicyEngine{}
	applied, modelID, _ := canary.Decide(req)
	if !applied || modelID != "gpt-4o" {
		t.Fatalf("expected canary to apply, got applied=%v model=%s", applied, modelID)
	}

	report := evaluator.Evaluate(req, preparedExecution(), api.GenerationResponse{
		Content:   "visual ready",
		ModelUsed: "gpt-4o",
		Route:     &api.RouteMetadata{SelectedProvider: "openai", SelectedModel: "gpt-4o", Strategy: "default"},
		ExecutionPlan: &api.ExecutionPlanMetadata{
			RequiresAsync: true,
		},
		Generation:         &api.GenerationMetadata{Pipeline: "image_generation", QualityStatus: "passed"},
		GeneratedArtifacts: []api.ArtifactMetadata{{ArtifactID: "art-1"}},
	})
	if report == nil || report.OverallScore == 0 {
		t.Fatalf("expected evaluation report, got %+v", report)
	}
}

func newTestRuntime(t *testing.T, executor ExecuteFunc) (*Runtime, func()) {
	t.Helper()
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	cfg := DefaultConfig()
	cfg.Workers = 1
	cfg.QueuePollTimeout = 100 * time.Millisecond
	runtime := NewRuntime(rdb, PrepareFunc(func(context.Context, api.GenerationRequest) (*llm.PreparedExecution, error) {
		return preparedExecution(), nil
	}), executor, cfg, nil)
	runtime.retry.BaseDelay = 50 * time.Millisecond
	runtime.retry.MaxDelay = 100 * time.Millisecond
	return runtime, func() {
		_ = rdb.Close()
		mr.Close()
	}
}

func preparedExecution() *llm.PreparedExecution {
	return &llm.PreparedExecution{
		Task: llm.TaskProfile{
			TaskType:          "direct_text_generation",
			PrimaryCapability: llm.CapabilityTextGeneration,
			Modalities:        []string{"text"},
		},
		Plan: &llm.ExecutionPlan{
			PlanID:         "plan-1",
			PlanType:       "reasoning",
			PrimaryStageID: "reason_answer",
			RequiresAsync:  true,
			Stages: []llm.ExecutionStage{
				{StageID: "retrieve_context", StageType: "retrieval"},
				{StageID: "reason_answer", StageType: "reasoning"},
			},
		},
	}
}

func waitForJobState(t *testing.T, runtime *Runtime, jobID string, expected JobState) {
	t.Helper()
	deadline := time.Now().Add(4 * time.Second)
	for time.Now().Before(deadline) {
		job, err := runtime.GetJob(context.Background(), jobID)
		if err == nil && job != nil && job.State == expected {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	job, _ := runtime.GetJob(context.Background(), jobID)
	t.Fatalf("timed out waiting for job %s to reach %s, got %+v", jobID, expected, job)
}
