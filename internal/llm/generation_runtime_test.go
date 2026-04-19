package llm

import (
	"context"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/redis/go-redis/v9"
)

type stubImageClient struct {
	generated []string
	edited    []string
}

func (s *stubImageClient) GenerateImage(_ context.Context, prompt, model string) (string, error) {
	s.generated = append(s.generated, prompt+"|"+model)
	return "https://example.com/" + model + "/generated.png", nil
}

func (s *stubImageClient) EditImage(_ context.Context, req ImageEditRequest) (string, error) {
	s.edited = append(s.edited, req.Prompt+"|"+req.Model)
	return "https://example.com/" + req.Model + "/edited.png", nil
}

type stubSpeechClient struct{}

func (s *stubSpeechClient) SynthesizeSpeech(_ context.Context, req SpeechSynthesisRequest) (*SpeechSynthesisResult, error) {
	audio := make([]byte, 512)
	for i := range audio {
		audio[i] = byte(i % 251)
	}
	return &SpeechSynthesisResult{
		Audio:     audio,
		MimeType:  "audio/mpeg",
		Format:    "mp3",
		Voice:     req.Voice,
		Model:     req.Model,
		InputText: req.Input,
	}, nil
}

func TestMultimodalRuntimeBuildsSpeechSynthesisPlan(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})

	cfg := &RouterConfig{
		Models: map[string]ModelMetadata{
			"gpt-4o-mini-tts": {
				Provider:     "openai",
				Modalities:   []string{"audio"},
				Capabilities: []Capability{CapabilityTTS},
			},
		},
		Strategies: map[string]RoutingStrategy{
			"default": {QualityWeight: 1},
		},
		RoutePolicies: map[string]RoutingPolicy{
			"default": {Strategy: "default", AllowDegradedFallback: true},
		},
	}

	controlPlane := NewControlPlane(cfg, nil, nil, []string{"gpt-4o-mini-tts"}, map[string]string{"gpt-4o-mini-tts": "openai-key"})
	runtime := NewMultimodalRuntime(rdb, controlPlane)

	prepared, err := runtime.Prepare(context.Background(), api.GenerationRequest{
		Prompt:             "Turn this product summary into a warm spoken narration.",
		RequiresGeneration: true,
		OutputType:         "audio",
	}, nil, ForceMetadata{})
	if err != nil {
		t.Fatalf("Prepare returned error: %v", err)
	}
	if prepared.Task.TaskType != "speech_synthesis" {
		t.Fatalf("expected speech_synthesis task, got %+v", prepared.Task)
	}
	if prepared.Plan == nil || prepared.Plan.PrimaryStageID != "synthesize_speech" {
		t.Fatalf("expected synthesize_speech primary stage, got %+v", prepared.Plan)
	}
}

func TestGenerationRuntimeRegistersImageEditLineage(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})

	cfg := &RouterConfig{
		Models: map[string]ModelMetadata{
			"dall-e-3": {
				Provider:     "openai",
				Modalities:   []string{"image"},
				Capabilities: []Capability{CapabilityImageGeneration, CapabilityImageEditing},
			},
		},
		Strategies: map[string]RoutingStrategy{
			"default": {QualityWeight: 1},
		},
		RoutePolicies: map[string]RoutingPolicy{
			"default": {Strategy: "default", AllowDegradedFallback: true},
		},
	}

	controlPlane := NewControlPlane(cfg, nil, nil, []string{"dall-e-3"}, map[string]string{"dall-e-3": "openai-key"})
	multimodal := NewMultimodalRuntime(rdb, controlPlane)
	prepared, err := multimodal.Prepare(context.Background(), api.GenerationRequest{
		Prompt:             "Make this product photo look more premium and cinematic.",
		RequiresGeneration: true,
		OutputType:         "image",
		Assets: []api.AssetInput{
			{AssetID: "img-1", Type: "image", Name: "product.png", URI: "/tmp/product.png"},
		},
		Config: api.GenerationConfig{
			ForceModel: "dall-e-3",
			ForceScope: "capability_scoped_force",
		},
	}, nil, ForceMetadata{
		IsForced:       true,
		Scope:          "capability_scoped_force",
		PinnedModel:    "dall-e-3",
		EffectiveModel: "dall-e-3",
	})
	if err != nil {
		t.Fatalf("Prepare returned error: %v", err)
	}

	imageClient := &stubImageClient{}
	runtime := NewGenerationRuntime(multimodal.ArtifactRegistry(), nil, map[string]ImageClient{"dall-e-3": imageClient}, nil, cfg, t.TempDir())
	result, err := runtime.Execute(context.Background(), api.GenerationRequest{
		Prompt:             "Make this product photo look more premium and cinematic.",
		RequiresGeneration: true,
		OutputType:         "image",
		Config: api.GenerationConfig{
			ForceModel: "dall-e-3",
			ForceScope: "capability_scoped_force",
		},
	}, prepared)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.ModelUsed != "dall-e-3" {
		t.Fatalf("expected forced model to be used, got %s", result.ModelUsed)
	}
	if len(result.OutputArtifacts) != 1 {
		t.Fatalf("expected one edited artifact, got %+v", result.OutputArtifacts)
	}
	if result.OutputArtifacts[0].DerivedFrom != "img-1" {
		t.Fatalf("expected derived artifact lineage, got %+v", result.OutputArtifacts[0])
	}
}

func TestGenerationRuntimeBuildsCreativeBundle(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})

	cfg := &RouterConfig{
		Models: map[string]ModelMetadata{
			"dall-e-3": {
				Provider:     "openai",
				Modalities:   []string{"image"},
				Capabilities: []Capability{CapabilityImageGeneration, CapabilityImageEditing},
			},
			"gpt-4o-mini-tts": {
				Provider:     "openai",
				Modalities:   []string{"audio"},
				Capabilities: []Capability{CapabilityTTS},
				QualityTier:  "high",
			},
		},
		Strategies: map[string]RoutingStrategy{
			"default": {QualityWeight: 1},
		},
		RoutePolicies: map[string]RoutingPolicy{
			"default": {Strategy: "default", AllowDegradedFallback: true},
		},
	}

	controlPlane := NewControlPlane(cfg, nil, nil, []string{"dall-e-3", "gpt-4o-mini-tts"}, map[string]string{
		"dall-e-3":        "openai-key",
		"gpt-4o-mini-tts": "openai-key",
	})
	multimodal := NewMultimodalRuntime(rdb, controlPlane)
	prepared, err := multimodal.Prepare(context.Background(), api.GenerationRequest{
		Prompt:             "Create a poster and narration bundle for this launch memo.",
		RequiresGeneration: true,
		OutputType:         "bundle",
		Assets: []api.AssetInput{
			{AssetID: "doc-1", Type: "document", Name: "memo.md", InlineText: "Launch memo: emphasize speed, reliability, and premium finish."},
		},
	}, nil, ForceMetadata{})
	if err != nil {
		t.Fatalf("Prepare returned error: %v", err)
	}
	imageClient := &stubImageClient{}
	speechClient := &stubSpeechClient{}
	runtime := NewGenerationRuntime(multimodal.ArtifactRegistry(), nil, map[string]ImageClient{"dall-e-3": imageClient}, map[string]SpeechClient{"gpt-4o-mini-tts": speechClient}, cfg, t.TempDir())
	result, err := runtime.Execute(context.Background(), api.GenerationRequest{
		Prompt:             "Create a poster and narration bundle for this launch memo.",
		RequiresGeneration: true,
		OutputType:         "bundle",
	}, prepared)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Pipeline != "creative_bundle" {
		t.Fatalf("expected creative_bundle pipeline, got %+v", result)
	}
	if result.ImageURL == "" || result.AudioURL == "" {
		t.Fatalf("expected both image and audio outputs, got %+v", result)
	}
	if len(result.OutputArtifacts) < 3 {
		t.Fatalf("expected image, audio, and bundle manifest artifacts, got %+v", result.OutputArtifacts)
	}
}
