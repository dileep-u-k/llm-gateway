package llm

import (
	"context"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/redis/go-redis/v9"
)

func TestMultimodalRuntimeBuildsDocumentImagePlan(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})

	cfg := &RouterConfig{
		Providers: map[string]ProviderMetadata{
			"openai": {Endpoint: "https://api.openai.com/v1"},
			"google": {Endpoint: "https://generativelanguage.googleapis.com"},
		},
		Capabilities: map[string]CapabilityMetadata{
			string(CapabilityTextGeneration):     {Description: "Text reasoning", Modalities: []string{"text"}},
			string(CapabilityImageUnderstanding): {Description: "Image understanding", Modalities: []string{"image"}},
			string(CapabilityOCR):                {Description: "OCR", Modalities: []string{"document"}},
			string(CapabilityEmbeddings):         {Description: "Embeddings", Modalities: []string{"text"}},
		},
		Models: map[string]ModelMetadata{
			"gpt-4o": {
				Provider:     "openai",
				Modalities:   []string{"text", "image"},
				Capabilities: []Capability{CapabilityTextGeneration, CapabilityImageUnderstanding},
			},
			"gemini-1.5-flash-latest": {
				Provider:     "google",
				Modalities:   []string{"text", "image"},
				Capabilities: []Capability{CapabilityTextGeneration, CapabilityImageUnderstanding},
			},
		},
		Strategies: map[string]RoutingStrategy{
			"default": {QualityWeight: 1},
		},
		RoutePolicies: map[string]RoutingPolicy{
			"default": {Strategy: "default", AllowDegradedFallback: true},
		},
	}

	controlPlane := NewControlPlane(cfg, nil, nil, []string{"gpt-4o", "gemini-1.5-flash-latest"}, map[string]string{
		"gpt-4o":                  "openai-key",
		"gemini-1.5-flash-latest": "google-key",
	})
	runtime := NewMultimodalRuntime(rdb, controlPlane)

	prepared, err := runtime.Prepare(context.Background(), api.GenerationRequest{
		Prompt:     "Compare the PDF and screenshot and explain the discrepancy.",
		InputType:  "mixed",
		OutputType: "summary",
		Assets: []api.AssetInput{
			{AssetID: "doc-1", Type: "document", Name: "policy.md", InlineText: "Section A: Threshold is 5."},
			{AssetID: "img-1", Type: "image", Name: "dashboard.png", Caption: "Dashboard shows threshold 7."},
		},
	}, nil, ForceMetadata{})
	if err != nil {
		t.Fatalf("Prepare returned error: %v", err)
	}
	if prepared.Task.TaskType != "document_image_reasoning" {
		t.Fatalf("expected document_image_reasoning task, got %+v", prepared.Task)
	}
	if prepared.Plan == nil || prepared.Plan.PlanType != "document_and_image_reasoning" {
		t.Fatalf("expected multimodal execution plan, got %+v", prepared.Plan)
	}
	if prepared.Plan.PrimaryStageID != "reason_answer" {
		t.Fatalf("expected reason_answer primary stage, got %+v", prepared.Plan)
	}
	if !containsString(prepared.Modalities, "document") || !containsString(prepared.Modalities, "image") {
		t.Fatalf("expected document and image modalities, got %+v", prepared.Modalities)
	}
}

func TestMultimodalRuntimeAppliesStrictForceViolationForUnsupportedStage(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})

	cfg := &RouterConfig{
		Models: map[string]ModelMetadata{
			"claude-3": {
				Provider:     "anthropic",
				Modalities:   []string{"text"},
				Capabilities: []Capability{CapabilityTextGeneration},
			},
		},
		Strategies: map[string]RoutingStrategy{
			"default": {QualityWeight: 1},
		},
		RoutePolicies: map[string]RoutingPolicy{
			"default": {Strategy: "default", AllowDegradedFallback: true},
		},
	}

	controlPlane := NewControlPlane(cfg, nil, nil, []string{"claude-3"}, map[string]string{"claude-3": "anthropic-key"})
	runtime := NewMultimodalRuntime(rdb, controlPlane)

	prepared, err := runtime.Prepare(context.Background(), api.GenerationRequest{
		Prompt:      "Read this scanned contract and answer the question.",
		InputType:   "document",
		OutputType:  "text",
		RequiresOCR: true,
		Assets: []api.AssetInput{
			{AssetID: "scan-1", Type: "document", Name: "scan.pdf"},
		},
	}, nil, ForceMetadata{
		IsForced:       true,
		Scope:          "strict_end_to_end_force",
		Strict:         true,
		PinnedModel:    "claude-3",
		EffectiveModel: "claude-3",
	})
	if err != nil {
		t.Fatalf("Prepare returned error: %v", err)
	}
	var foundViolation bool
	for _, stage := range prepared.Plan.Stages {
		if stage.StageID == "ocr_extract" && stage.BindingViolation != "" {
			foundViolation = true
		}
	}
	if !foundViolation {
		t.Fatalf("expected strict force violation on OCR stage, got %+v", prepared.Plan.Stages)
	}
}
