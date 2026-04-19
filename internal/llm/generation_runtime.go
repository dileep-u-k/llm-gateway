package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/dileep-u-k/llm-gateway/internal/api"
)

type CreativeModelProfile struct {
	ModelID             string
	Provider            string
	Capabilities        []Capability
	ConditioningInputs  []string
	CostTier            string
	LatencyTier         string
	QualityTier         string
	MaxOutput           string
	SupportsEditing     bool
	SupportsIterations  bool
	OutputArtifactRules map[string]string
}

type GenerationCapabilityRegistry struct {
	profiles     map[string]CreativeModelProfile
	byCapability map[Capability][]CreativeModelProfile
}

func NewGenerationCapabilityRegistry(cfg *RouterConfig, imageModels []string, speechModels []string) *GenerationCapabilityRegistry {
	registry := &GenerationCapabilityRegistry{
		profiles:     make(map[string]CreativeModelProfile),
		byCapability: make(map[Capability][]CreativeModelProfile),
	}

	for _, modelID := range imageModels {
		modelID = strings.TrimSpace(modelID)
		if modelID == "" {
			continue
		}
		profile := CreativeModelProfile{
			ModelID:            modelID,
			Provider:           ProviderForModel(modelID),
			Capabilities:       []Capability{CapabilityImageGeneration, CapabilityImageEditing},
			ConditioningInputs: []string{"text_prompt", "reference_image", "style_constraints"},
			CostTier:           "premium",
			LatencyTier:        "medium",
			QualityTier:        "high",
			MaxOutput:          "1024x1024",
			SupportsEditing:    true,
			SupportsIterations: true,
			OutputArtifactRules: map[string]string{
				"type": "image",
			},
		}
		if cfg != nil {
			if meta, ok := cfg.Models[modelID]; ok {
				profile.CostTier = firstNonEmpty(meta.CostTier, profile.CostTier)
				profile.LatencyTier = firstNonEmpty(meta.LatencyTier, profile.LatencyTier)
				profile.QualityTier = firstNonEmpty(meta.QualityTier, profile.QualityTier)
			}
		}
		registry.register(profile)
	}

	for _, modelID := range speechModels {
		modelID = strings.TrimSpace(modelID)
		if modelID == "" {
			continue
		}
		profile := CreativeModelProfile{
			ModelID:            modelID,
			Provider:           ProviderForModel(modelID),
			Capabilities:       []Capability{CapabilityTTS},
			ConditioningInputs: []string{"script", "voice", "style"},
			CostTier:           "balanced",
			LatencyTier:        "fast",
			QualityTier:        "high",
			MaxOutput:          "2000_tokens",
			SupportsIterations: true,
			OutputArtifactRules: map[string]string{
				"type": "audio",
			},
		}
		if cfg != nil {
			if meta, ok := cfg.Models[modelID]; ok {
				profile.CostTier = firstNonEmpty(meta.CostTier, profile.CostTier)
				profile.LatencyTier = firstNonEmpty(meta.LatencyTier, profile.LatencyTier)
				profile.QualityTier = firstNonEmpty(meta.QualityTier, profile.QualityTier)
			}
		}
		registry.register(profile)
	}

	return registry
}

func (r *GenerationCapabilityRegistry) register(profile CreativeModelProfile) {
	r.profiles[profile.ModelID] = profile
	for _, capability := range profile.Capabilities {
		r.byCapability[capability] = append(r.byCapability[capability], profile)
	}
}

func (r *GenerationCapabilityRegistry) Profiles(capability Capability) []CreativeModelProfile {
	if r == nil {
		return nil
	}
	return append([]CreativeModelProfile(nil), r.byCapability[capability]...)
}

type ConditioningAssetSelector struct{}

func (s *ConditioningAssetSelector) Select(task TaskProfile, artifacts []ArtifactRecord) []ArtifactRecord {
	if len(artifacts) == 0 {
		return nil
	}
	selected := make([]ArtifactRecord, 0, len(artifacts))
	for _, artifact := range artifacts {
		switch task.TaskType {
		case "image_editing":
			if artifact.Type == "image" {
				selected = append(selected, artifact)
			}
		case "speech_synthesis":
			if artifact.Type == "document" || artifact.Type == "audio" {
				selected = append(selected, artifact)
			}
		default:
			if artifact.Type == "document" || artifact.Type == "image" || artifact.Type == "audio" {
				selected = append(selected, artifact)
			}
		}
	}
	if len(selected) == 0 {
		return append(selected, artifacts...)
	}
	return selected
}

type CreativePrompt struct {
	Strategy                string
	RefinedPrompt           string
	Voice                   string
	ConditioningArtifactIDs []string
	Warnings                []string
}

type CreativePromptComposer struct {
	selector *ConditioningAssetSelector
}

func NewCreativePromptComposer() *CreativePromptComposer {
	return &CreativePromptComposer{selector: &ConditioningAssetSelector{}}
}

func (c *CreativePromptComposer) Compose(req api.GenerationRequest, prepared *PreparedExecution) CreativePrompt {
	if prepared == nil {
		return CreativePrompt{RefinedPrompt: req.Prompt}
	}
	selected := c.selector.Select(prepared.Task, prepared.Artifacts)
	summaries := make([]string, 0, len(selected))
	artifactIDs := make([]string, 0, len(selected))
	for _, artifact := range selected {
		artifactIDs = append(artifactIDs, artifact.ArtifactID)
		if summary := artifactSummary(artifact); summary != "" {
			summaries = append(summaries, summary)
		}
	}

	strategy := "balanced"
	switch prepared.Task.TaskType {
	case "image_generation":
		strategy = "image_generation"
	case "image_editing":
		strategy = "image_editing"
	case "speech_synthesis":
		strategy = "speech_synthesis"
	case "video_generation_hook":
		strategy = "video_storyboard"
	case "creative_bundle":
		strategy = "composite_bundle"
	}

	voice := inferVoice(req.Prompt, selected)
	refined := strings.TrimSpace(req.Prompt)
	if len(summaries) > 0 {
		refined = strings.TrimSpace(refined + "\n\nReference context:\n- " + strings.Join(summaries, "\n- "))
	}

	switch prepared.Task.TaskType {
	case "image_generation":
		refined = "Create a polished image asset.\nGoal: " + refined + "\nConstraints: preserve factual details from the provided references, use a clear focal point, and deliver a production-ready composition."
	case "image_editing":
		refined = "Edit the provided image while preserving its core identity.\nInstructions: " + refined + "\nConstraints: keep the original subject recognizable, apply only the requested transformation, and maintain high visual coherence."
	case "speech_synthesis":
		refined = "Prepare this narration for speech synthesis.\nScript: " + refined + "\nDelivery: natural pacing, clear articulation, and clean sentence boundaries."
	case "video_generation_hook":
		refined = "Build a storyboard and production brief for a downstream video generation system.\nCreative brief: " + refined + "\nDeliver scenes, narration, and reusable asset prompts."
	case "creative_bundle":
		refined = "Create a coordinated multi-asset campaign bundle.\nCreative brief: " + refined + "\nDeliver a visually aligned image asset plus a matching narration concept."
	}

	return CreativePrompt{
		Strategy:                strategy,
		RefinedPrompt:           strings.TrimSpace(refined),
		Voice:                   voice,
		ConditioningArtifactIDs: artifactIDs,
	}
}

type GenerationPolicyGuard struct{}

func (g *GenerationPolicyGuard) Validate(task TaskProfile, prompt CreativePrompt, artifacts []ArtifactRecord) error {
	if strings.TrimSpace(prompt.RefinedPrompt) == "" {
		return fmt.Errorf("creative prompt cannot be empty")
	}
	lower := strings.ToLower(prompt.RefinedPrompt)
	for _, blocked := range []string{"sexual content involving minors", "child sexual", "explosive attack instructions"} {
		if strings.Contains(lower, blocked) {
			return fmt.Errorf("request violated generation safety policy")
		}
	}
	if task.TaskType == "image_editing" && firstImageArtifact(artifacts) == nil {
		return fmt.Errorf("image editing requires at least one image artifact")
	}
	return nil
}

type OutputQualityChecker struct{}

func (c *OutputQualityChecker) CheckImage(location string) error {
	if strings.TrimSpace(location) == "" {
		return fmt.Errorf("image output was empty")
	}
	return nil
}

func (c *OutputQualityChecker) CheckAudio(result *SpeechSynthesisResult) error {
	if result == nil || len(result.Audio) < 256 {
		return fmt.Errorf("audio output was too small to be valid")
	}
	return nil
}

func (c *OutputQualityChecker) CheckTextArtifact(content string) error {
	if strings.TrimSpace(content) == "" {
		return fmt.Errorf("generated artifact content was empty")
	}
	return nil
}

type CreativeWorkflowValidator struct{}

func (v *CreativeWorkflowValidator) Validate(task TaskProfile) error {
	switch task.TaskType {
	case "video_generation_hook", "creative_bundle", "image_generation", "image_editing", "speech_synthesis":
		return nil
	default:
		return fmt.Errorf("unsupported generation workflow %q", task.TaskType)
	}
}

type GenerationForceScopeResolver struct{}

func (r *GenerationForceScopeResolver) BoundModel(plan *ExecutionPlan, stageID string, fallback string) string {
	if plan == nil {
		return fallback
	}
	for _, stage := range plan.Stages {
		if stage.StageID == stageID && stage.ModelBinding != "" {
			return stage.ModelBinding
		}
	}
	return fallback
}

type GenerationArtifactManager struct {
	registry    *ArtifactRegistry
	storageRoot string
}

func NewGenerationArtifactManager(registry *ArtifactRegistry, storageRoot string) *GenerationArtifactManager {
	if strings.TrimSpace(storageRoot) == "" {
		storageRoot = filepath.Join(os.TempDir(), "llm-gateway-generated")
	}
	return &GenerationArtifactManager{
		registry:    registry,
		storageRoot: storageRoot,
	}
}

func (m *GenerationArtifactManager) RegisterRemoteImage(ctx context.Context, prompt CreativePrompt, modelID string, derivedFrom string, location string, prepared *PreparedExecution) (ArtifactRecord, error) {
	record := ArtifactRecord{
		Name:           "generated-image",
		Type:           "image",
		MimeType:       inferMimeType("image", location),
		SourceURI:      location,
		Role:           "generated_output",
		DerivedFrom:    derivedFrom,
		Lineage:        buildLineage(derivedFrom),
		GeneratorModel: modelID,
		PromptSummary:  truncatePrompt(prompt.RefinedPrompt, 220),
		Metadata: map[string]string{
			"generation_prompt":     truncatePrompt(prompt.RefinedPrompt, 600),
			"generation_strategy":   prompt.Strategy,
			"upstream_context_refs": strings.Join(prompt.ConditioningArtifactIDs, ","),
			"artifact_family":       "phase4_generation",
			"primary_task":          prepared.Task.TaskType,
		},
	}
	return m.registry.Register(ctx, record)
}

func (m *GenerationArtifactManager) RegisterAudio(ctx context.Context, prompt CreativePrompt, result *SpeechSynthesisResult, derivedFrom string, prepared *PreparedExecution) (ArtifactRecord, error) {
	if err := os.MkdirAll(filepath.Join(m.storageRoot, "audio"), 0o755); err != nil {
		return ArtifactRecord{}, err
	}
	path := filepath.Join(m.storageRoot, "audio", GenerateCacheKey(result.Model+"|"+prompt.RefinedPrompt+"|"+time.Now().UTC().Format(time.RFC3339Nano))+"."+firstNonEmpty(result.Format, "mp3"))
	if err := os.WriteFile(path, result.Audio, 0o644); err != nil {
		return ArtifactRecord{}, err
	}
	record := ArtifactRecord{
		Name:           filepath.Base(path),
		Type:           "audio",
		MimeType:       result.MimeType,
		SourceURI:      path,
		SizeBytes:      int64(len(result.Audio)),
		Role:           "generated_output",
		DerivedFrom:    derivedFrom,
		Lineage:        buildLineage(derivedFrom),
		GeneratorModel: result.Model,
		PromptSummary:  truncatePrompt(prompt.RefinedPrompt, 220),
		Transcript:     truncatePrompt(result.InputText, 1200),
		Metadata: map[string]string{
			"generation_prompt":     truncatePrompt(prompt.RefinedPrompt, 600),
			"generation_strategy":   prompt.Strategy,
			"voice":                 result.Voice,
			"upstream_context_refs": strings.Join(prompt.ConditioningArtifactIDs, ","),
			"artifact_family":       "phase4_generation",
			"primary_task":          prepared.Task.TaskType,
		},
	}
	return m.registry.Register(ctx, record)
}

func (m *GenerationArtifactManager) RegisterTextArtifact(ctx context.Context, name, kind, content string, prompt CreativePrompt, modelID string, derivedFrom []string, prepared *PreparedExecution) (ArtifactRecord, error) {
	if err := os.MkdirAll(filepath.Join(m.storageRoot, "manifests"), 0o755); err != nil {
		return ArtifactRecord{}, err
	}
	path := filepath.Join(m.storageRoot, "manifests", GenerateCacheKey(name+"|"+modelID+"|"+time.Now().UTC().Format(time.RFC3339Nano))+".md")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return ArtifactRecord{}, err
	}
	record := ArtifactRecord{
		Name:           name,
		Type:           kind,
		MimeType:       "text/markdown",
		SourceURI:      path,
		SizeBytes:      int64(len(content)),
		Role:           "generated_output",
		DerivedFrom:    strings.Join(derivedFrom, ","),
		Lineage:        append([]string(nil), derivedFrom...),
		GeneratorModel: modelID,
		PromptSummary:  truncatePrompt(prompt.RefinedPrompt, 220),
		Text:           content,
		Metadata: map[string]string{
			"generation_prompt":     truncatePrompt(prompt.RefinedPrompt, 600),
			"generation_strategy":   prompt.Strategy,
			"upstream_context_refs": strings.Join(prompt.ConditioningArtifactIDs, ","),
			"artifact_family":       "phase4_generation",
			"primary_task":          prepared.Task.TaskType,
		},
	}
	return m.registry.Register(ctx, record)
}

type GenerationExecutionResult struct {
	Pipeline        string
	Content         string
	ImageURL        string
	AudioURL        string
	ModelUsed       string
	Prompt          CreativePrompt
	PolicyStatus    string
	QualityStatus   string
	Attempts        int
	Warnings        []string
	StageModelMap   map[string]string
	OutputArtifacts []ArtifactRecord
	VideoHook       *api.VideoHookMetadata
	RouteReason     string
}

type ImageGenerationPipeline struct {
	router    *ImageRouter
	clients   map[string]ImageClient
	quality   *OutputQualityChecker
	artifacts *GenerationArtifactManager
	force     *GenerationForceScopeResolver
}

type ImageEditPipeline struct {
	router    *ImageRouter
	clients   map[string]ImageClient
	quality   *OutputQualityChecker
	artifacts *GenerationArtifactManager
	force     *GenerationForceScopeResolver
}

type SpeechSynthesisPipeline struct {
	clients   map[string]SpeechClient
	registry  *GenerationCapabilityRegistry
	quality   *OutputQualityChecker
	artifacts *GenerationArtifactManager
	force     *GenerationForceScopeResolver
}

type VideoGenerationHookLayer struct {
	artifacts *GenerationArtifactManager
}

type CompositeCreativeWorkflowPlanner struct {
	image     *ImageGenerationPipeline
	speech    *SpeechSynthesisPipeline
	video     *VideoGenerationHookLayer
	artifacts *GenerationArtifactManager
}

type GenerationRuntime struct {
	registry         *GenerationCapabilityRegistry
	composer         *CreativePromptComposer
	policy           *GenerationPolicyGuard
	quality          *OutputQualityChecker
	validator        *CreativeWorkflowValidator
	imageGeneration  *ImageGenerationPipeline
	imageEditing     *ImageEditPipeline
	speechSynthesis  *SpeechSynthesisPipeline
	videoHooks       *VideoGenerationHookLayer
	compositePlanner *CompositeCreativeWorkflowPlanner
}

func NewGenerationRuntime(registry *ArtifactRegistry, router *ImageRouter, imageClients map[string]ImageClient, speechClients map[string]SpeechClient, cfg *RouterConfig, storageRoot string) *GenerationRuntime {
	imageModels := sortedMapKeys(imageClients)
	speechModels := sortedSpeechKeys(speechClients)
	artifactManager := NewGenerationArtifactManager(registry, storageRoot)
	capabilities := NewGenerationCapabilityRegistry(cfg, imageModels, speechModels)
	quality := &OutputQualityChecker{}
	force := &GenerationForceScopeResolver{}

	imageGeneration := &ImageGenerationPipeline{
		router:    router,
		clients:   imageClients,
		quality:   quality,
		artifacts: artifactManager,
		force:     force,
	}
	speechSynthesis := &SpeechSynthesisPipeline{
		clients:   speechClients,
		registry:  capabilities,
		quality:   quality,
		artifacts: artifactManager,
		force:     force,
	}
	videoHooks := &VideoGenerationHookLayer{artifacts: artifactManager}
	composite := &CompositeCreativeWorkflowPlanner{
		image:     imageGeneration,
		speech:    speechSynthesis,
		video:     videoHooks,
		artifacts: artifactManager,
	}

	return &GenerationRuntime{
		registry:        capabilities,
		composer:        NewCreativePromptComposer(),
		policy:          &GenerationPolicyGuard{},
		quality:         quality,
		validator:       &CreativeWorkflowValidator{},
		imageGeneration: imageGeneration,
		imageEditing: &ImageEditPipeline{
			router:    router,
			clients:   imageClients,
			quality:   quality,
			artifacts: artifactManager,
			force:     force,
		},
		speechSynthesis:  speechSynthesis,
		videoHooks:       videoHooks,
		compositePlanner: composite,
	}
}

func (r *GenerationRuntime) Execute(ctx context.Context, req api.GenerationRequest, prepared *PreparedExecution) (*GenerationExecutionResult, error) {
	if prepared == nil {
		return nil, fmt.Errorf("generation runtime requires prepared execution context")
	}
	if err := r.validator.Validate(prepared.Task); err != nil {
		return nil, err
	}
	prompt := r.composer.Compose(req, prepared)
	if err := r.policy.Validate(prepared.Task, prompt, prepared.Artifacts); err != nil {
		return nil, err
	}

	switch prepared.Task.TaskType {
	case "image_generation":
		return r.imageGeneration.Run(ctx, req, prepared, prompt)
	case "image_editing":
		return r.imageEditing.Run(ctx, req, prepared, prompt)
	case "speech_synthesis":
		return r.speechSynthesis.Run(ctx, req, prepared, prompt, "synthesize_speech")
	case "video_generation_hook":
		return r.videoHooks.Run(ctx, req, prepared, prompt)
	case "creative_bundle":
		return r.compositePlanner.Run(ctx, req, prepared, prompt)
	default:
		return nil, fmt.Errorf("generation runtime does not support task %q", prepared.Task.TaskType)
	}
}

func (p *ImageGenerationPipeline) Run(ctx context.Context, req api.GenerationRequest, prepared *PreparedExecution, prompt CreativePrompt) (*GenerationExecutionResult, error) {
	modelID, reason, err := p.selectModel(ctx, req, prepared, prompt, "generate_image")
	if err != nil {
		return nil, err
	}
	client := p.clients[modelID]
	if client == nil {
		return nil, fmt.Errorf("no image client available for model %s", modelID)
	}

	var location string
	var warnings []string
	attempts := 0
	for attempts < 2 {
		attempts++
		currentPrompt := prompt.RefinedPrompt
		if attempts > 1 {
			currentPrompt += "\nRefinement pass: increase composition clarity, consistency, and artifact quality."
		}
		location, err = client.GenerateImage(ctx, currentPrompt, modelID)
		if err != nil {
			return nil, err
		}
		if err := p.quality.CheckImage(location); err == nil {
			break
		}
		warnings = append(warnings, fmt.Sprintf("image quality check failed on attempt %d", attempts))
	}
	if err := p.quality.CheckImage(location); err != nil {
		return nil, err
	}
	artifact, err := p.artifacts.RegisterRemoteImage(ctx, prompt, modelID, "", location, prepared)
	if err != nil {
		return nil, err
	}

	return &GenerationExecutionResult{
		Pipeline:        "image_generation",
		ImageURL:        location,
		ModelUsed:       modelID,
		Prompt:          prompt,
		PolicyStatus:    "passed",
		QualityStatus:   "passed",
		Attempts:        attempts,
		Warnings:        warnings,
		StageModelMap:   map[string]string{"generate_image": modelID},
		OutputArtifacts: []ArtifactRecord{artifact},
		RouteReason:     reason,
	}, nil
}

func (p *ImageGenerationPipeline) selectModel(ctx context.Context, req api.GenerationRequest, prepared *PreparedExecution, prompt CreativePrompt, stageID string) (string, string, error) {
	if prepared != nil && prepared.Plan != nil {
		if bound := p.force.BoundModel(prepared.Plan, stageID, ""); bound != "" {
			if _, ok := p.clients[bound]; ok {
				return bound, "selected via generation force binding", nil
			}
		}
	}
	if req.Config.ForceModel != "" {
		if _, ok := p.clients[req.Config.ForceModel]; ok {
			return req.Config.ForceModel, "selected via explicit forced generation model", nil
		}
	}
	if p.router == nil {
		keys := sortedMapKeys(p.clients)
		if len(keys) == 0 {
			return "", "", fmt.Errorf("no image models are enabled")
		}
		return keys[0], "selected first available image model", nil
	}
	modelID, err := p.router.SelectModel(ctx, prompt.RefinedPrompt, req.Config.ImagePreference)
	if err != nil {
		return "", "", err
	}
	return modelID, "selected by image router strategy", nil
}

func (p *ImageEditPipeline) Run(ctx context.Context, req api.GenerationRequest, prepared *PreparedExecution, prompt CreativePrompt) (*GenerationExecutionResult, error) {
	modelID, reason, err := p.selectModel(ctx, req, prepared, prompt)
	if err != nil {
		return nil, err
	}
	client := p.clients[modelID]
	if client == nil {
		return nil, fmt.Errorf("no image client available for model %s", modelID)
	}
	base := firstImageArtifact(prepared.Artifacts)
	if base == nil {
		return nil, fmt.Errorf("image edit pipeline requires a source image")
	}

	var location string
	if editor, ok := client.(ImageEditingClient); ok && isLocalFile(base.SourceURI) {
		location, err = editor.EditImage(ctx, ImageEditRequest{
			Model:     modelID,
			Prompt:    prompt.RefinedPrompt,
			ImagePath: base.SourceURI,
			MaskPath:  firstNonEmpty(base.Metadata["mask_path"]),
		})
	} else {
		fallbackPrompt := prompt.RefinedPrompt + "\nUse the source image only as a reference and preserve the subject identity."
		location, err = client.GenerateImage(ctx, fallbackPrompt, modelID)
	}
	if err != nil {
		return nil, err
	}
	if err := p.quality.CheckImage(location); err != nil {
		return nil, err
	}
	artifact, err := p.artifacts.RegisterRemoteImage(ctx, prompt, modelID, base.ArtifactID, location, prepared)
	if err != nil {
		return nil, err
	}

	return &GenerationExecutionResult{
		Pipeline:        "image_editing",
		ImageURL:        location,
		ModelUsed:       modelID,
		Prompt:          prompt,
		PolicyStatus:    "passed",
		QualityStatus:   "passed",
		Attempts:        1,
		StageModelMap:   map[string]string{"edit_image": modelID},
		OutputArtifacts: []ArtifactRecord{artifact},
		RouteReason:     reason,
	}, nil
}

func (p *ImageEditPipeline) selectModel(ctx context.Context, req api.GenerationRequest, prepared *PreparedExecution, prompt CreativePrompt) (string, string, error) {
	if prepared != nil && prepared.Plan != nil {
		if bound := p.force.BoundModel(prepared.Plan, "edit_image", ""); bound != "" {
			if _, ok := p.clients[bound]; ok {
				return bound, "selected via generation force binding", nil
			}
		}
	}
	if req.Config.ForceModel != "" {
		if _, ok := p.clients[req.Config.ForceModel]; ok {
			return req.Config.ForceModel, "selected via explicit forced generation model", nil
		}
	}
	if p.router == nil {
		keys := sortedMapKeys(p.clients)
		if len(keys) == 0 {
			return "", "", fmt.Errorf("no image models are enabled")
		}
		return keys[0], "selected first available image model", nil
	}
	modelID, err := p.router.SelectModel(ctx, prompt.RefinedPrompt, req.Config.ImagePreference)
	if err != nil {
		return "", "", err
	}
	return modelID, "selected by image router strategy", nil
}

func (p *SpeechSynthesisPipeline) Run(ctx context.Context, req api.GenerationRequest, prepared *PreparedExecution, prompt CreativePrompt, stageID string) (*GenerationExecutionResult, error) {
	modelID, reason, err := p.selectModel(req, prepared, stageID)
	if err != nil {
		return nil, err
	}
	client := p.clients[modelID]
	if client == nil {
		return nil, fmt.Errorf("no speech client available for model %s", modelID)
	}

	result, err := client.SynthesizeSpeech(ctx, SpeechSynthesisRequest{
		Model:        modelID,
		Input:        speechInput(req, prepared, prompt),
		Voice:        prompt.Voice,
		Format:       "mp3",
		Instructions: "Deliver clean, natural, expressive speech that matches the request.",
	})
	if err != nil {
		return nil, err
	}
	if err := p.quality.CheckAudio(result); err != nil {
		return nil, err
	}

	artifact, err := p.artifacts.RegisterAudio(ctx, prompt, result, firstArtifactID(prepared.Artifacts), prepared)
	if err != nil {
		return nil, err
	}
	return &GenerationExecutionResult{
		Pipeline:        "speech_synthesis",
		AudioURL:        artifact.SourceURI,
		ModelUsed:       modelID,
		Prompt:          prompt,
		PolicyStatus:    "passed",
		QualityStatus:   "passed",
		Attempts:        1,
		StageModelMap:   map[string]string{stageID: modelID},
		OutputArtifacts: []ArtifactRecord{artifact},
		RouteReason:     reason,
	}, nil
}

func (p *SpeechSynthesisPipeline) selectModel(req api.GenerationRequest, prepared *PreparedExecution, stageID string) (string, string, error) {
	if prepared != nil && prepared.Plan != nil {
		if bound := p.force.BoundModel(prepared.Plan, stageID, ""); bound != "" {
			if _, ok := p.clients[bound]; ok {
				return bound, "selected via generation force binding", nil
			}
		}
	}
	if req.Config.ForceModel != "" {
		if _, ok := p.clients[req.Config.ForceModel]; ok {
			return req.Config.ForceModel, "selected via explicit forced generation model", nil
		}
	}
	profiles := p.registry.Profiles(CapabilityTTS)
	if len(profiles) == 0 {
		return "", "", fmt.Errorf("no speech synthesis models are enabled")
	}
	sort.SliceStable(profiles, func(i, j int) bool {
		return qualityRank(profiles[i].QualityTier) > qualityRank(profiles[j].QualityTier)
	})
	for _, profile := range profiles {
		if _, ok := p.clients[profile.ModelID]; ok {
			return profile.ModelID, "selected from generation capability registry", nil
		}
	}
	return "", "", fmt.Errorf("no speech client is available for enabled speech models")
}

func (v *VideoGenerationHookLayer) Run(ctx context.Context, req api.GenerationRequest, prepared *PreparedExecution, prompt CreativePrompt) (*GenerationExecutionResult, error) {
	_ = ctx
	storyboard := buildStoryboard(prompt.RefinedPrompt, prepared.Artifacts)
	if err := (&OutputQualityChecker{}).CheckTextArtifact(storyboard); err != nil {
		return nil, err
	}
	storyboardArtifact, err := v.artifacts.RegisterTextArtifact(ctx, "video-storyboard", "document", storyboard, prompt, "video_generation_hook", artifactIDs(prepared.Artifacts), prepared)
	if err != nil {
		return nil, err
	}
	bundle := fmt.Sprintf("Video generation bundle\n\nStoryboard Artifact: %s\nPrimary brief:\n%s\n", storyboardArtifact.ArtifactID, truncatePrompt(prompt.RefinedPrompt, 600))
	bundleArtifact, err := v.artifacts.RegisterTextArtifact(ctx, "video-bundle", "document", bundle, prompt, "video_generation_hook", append(artifactIDs(prepared.Artifacts), storyboardArtifact.ArtifactID), prepared)
	if err != nil {
		return nil, err
	}
	return &GenerationExecutionResult{
		Pipeline:        "video_generation_hook",
		Content:         "Generated a storyboard-ready video creation bundle with reusable scene prompts and a packaging manifest.",
		ModelUsed:       "video_generation_hook",
		Prompt:          prompt,
		PolicyStatus:    "passed",
		QualityStatus:   "passed",
		Attempts:        1,
		StageModelMap:   map[string]string{"package_bundle": "video_generation_hook"},
		OutputArtifacts: []ArtifactRecord{storyboardArtifact, bundleArtifact},
		VideoHook: &api.VideoHookMetadata{
			StoryboardArtifactID: storyboardArtifact.ArtifactID,
			BundleArtifactID:     bundleArtifact.ArtifactID,
			SceneCount:           countStoryboardScenes(storyboard),
			Status:               "ready_for_downstream_video_provider",
		},
		RouteReason: "selected composite video hook pipeline",
	}, nil
}

func (p *CompositeCreativeWorkflowPlanner) Run(ctx context.Context, req api.GenerationRequest, prepared *PreparedExecution, prompt CreativePrompt) (*GenerationExecutionResult, error) {
	imageResult, err := p.image.Run(ctx, req, prepared, prompt)
	if err != nil {
		return nil, err
	}
	speechResult, err := p.speech.Run(ctx, req, prepared, prompt, "synthesize_speech")
	if err != nil {
		return nil, err
	}

	bundleManifest := map[string]any{
		"type":               "creative_bundle",
		"prompt":             prompt.RefinedPrompt,
		"image_artifact_ids": artifactIDs(imageResult.OutputArtifacts),
		"audio_artifact_ids": artifactIDs(speechResult.OutputArtifacts),
		"created_at":         time.Now().UTC().Format(time.RFC3339Nano),
	}
	payload, err := json.MarshalIndent(bundleManifest, "", "  ")
	if err != nil {
		return nil, err
	}
	bundleArtifact, err := p.artifacts.RegisterTextArtifact(ctx, "creative-bundle", "document", string(payload), prompt, "creative_bundle", append(artifactIDs(imageResult.OutputArtifacts), artifactIDs(speechResult.OutputArtifacts)...), prepared)
	if err != nil {
		return nil, err
	}

	outputArtifacts := append([]ArtifactRecord{}, imageResult.OutputArtifacts...)
	outputArtifacts = append(outputArtifacts, speechResult.OutputArtifacts...)
	outputArtifacts = append(outputArtifacts, bundleArtifact)

	stageMap := map[string]string{}
	for key, value := range imageResult.StageModelMap {
		stageMap[key] = value
	}
	for key, value := range speechResult.StageModelMap {
		stageMap[key] = value
	}
	stageMap["package_bundle"] = "creative_bundle"

	return &GenerationExecutionResult{
		Pipeline:        "creative_bundle",
		Content:         "Generated a coordinated creative bundle with a visual asset, narration audio, and a reusable manifest.",
		ImageURL:        imageResult.ImageURL,
		AudioURL:        speechResult.AudioURL,
		ModelUsed:       imageResult.ModelUsed,
		Prompt:          prompt,
		PolicyStatus:    "passed",
		QualityStatus:   "passed",
		Attempts:        imageResult.Attempts + speechResult.Attempts,
		StageModelMap:   stageMap,
		OutputArtifacts: outputArtifacts,
		RouteReason:     "selected composite creative workflow planner",
	}, nil
}

func inferVoice(prompt string, artifacts []ArtifactRecord) string {
	lower := strings.ToLower(prompt)
	switch {
	case strings.Contains(lower, "warm"), strings.Contains(lower, "friendly"):
		return "alloy"
	case strings.Contains(lower, "energetic"), strings.Contains(lower, "upbeat"):
		return "nova"
	case strings.Contains(lower, "calm"), strings.Contains(lower, "professional"):
		return "echo"
	}
	for _, artifact := range artifacts {
		if voice := artifact.Metadata["voice"]; voice != "" {
			return voice
		}
	}
	return "alloy"
}

func artifactSummary(artifact ArtifactRecord) string {
	switch artifact.Type {
	case "document":
		return firstNonEmpty(truncatePrompt(artifact.Text, 180), truncatePrompt(artifact.OCRText, 180), truncatePrompt(artifact.Caption, 120), artifact.Name)
	case "image":
		return firstNonEmpty(truncatePrompt(artifact.Caption, 160), artifact.Name)
	case "audio":
		return firstNonEmpty(truncatePrompt(artifact.Transcript, 180), artifact.Name)
	default:
		return firstNonEmpty(truncatePrompt(artifact.Text, 120), artifact.Name)
	}
}

func truncatePrompt(value string, maxLen int) string {
	value = strings.TrimSpace(value)
	if maxLen <= 0 || len(value) <= maxLen {
		return value
	}
	return value[:maxLen]
}

func buildLineage(derivedFrom string) []string {
	if strings.TrimSpace(derivedFrom) == "" {
		return nil
	}
	return []string{derivedFrom}
}

func firstImageArtifact(artifacts []ArtifactRecord) *ArtifactRecord {
	for i := range artifacts {
		if artifacts[i].Type == "image" {
			return &artifacts[i]
		}
	}
	return nil
}

func firstArtifactID(artifacts []ArtifactRecord) string {
	for _, artifact := range artifacts {
		if artifact.ArtifactID != "" {
			return artifact.ArtifactID
		}
	}
	return ""
}

func artifactIDs(artifacts []ArtifactRecord) []string {
	ids := make([]string, 0, len(artifacts))
	for _, artifact := range artifacts {
		if artifact.ArtifactID != "" {
			ids = append(ids, artifact.ArtifactID)
		}
	}
	return ids
}

func speechInput(req api.GenerationRequest, prepared *PreparedExecution, prompt CreativePrompt) string {
	if strings.TrimSpace(req.Prompt) != "" && prepared != nil && prepared.Task.TaskType == "speech_synthesis" && len(prepared.Artifacts) == 0 {
		return req.Prompt
	}
	if prepared != nil {
		for _, artifact := range prepared.Artifacts {
			if text := firstNonEmpty(strings.TrimSpace(artifact.Text), strings.TrimSpace(artifact.Transcript), strings.TrimSpace(artifact.OCRText)); text != "" {
				return text
			}
		}
	}
	return prompt.RefinedPrompt
}

func buildStoryboard(prompt string, artifacts []ArtifactRecord) string {
	sourceContext := make([]string, 0, len(artifacts))
	for _, artifact := range artifacts {
		if summary := artifactSummary(artifact); summary != "" {
			sourceContext = append(sourceContext, summary)
		}
	}
	scenes := []string{
		"Scene 1: Establish the problem or setup with a concise visual hook.",
		"Scene 2: Introduce the key subject, product, or insight with a clear focal frame.",
		"Scene 3: Show the transformation, proof point, or comparison.",
		"Scene 4: Close with a memorable call to action and narration cue.",
	}
	return strings.TrimSpace(fmt.Sprintf("Storyboard Brief\n\nPrompt:\n%s\n\nReference Signals:\n- %s\n\nScenes:\n1. %s\n2. %s\n3. %s\n4. %s\n", truncatePrompt(prompt, 1200), firstNonEmpty(strings.Join(sourceContext, "\n- "), "No additional source context provided."), scenes[0], scenes[1], scenes[2], scenes[3]))
}

func countStoryboardScenes(content string) int {
	count := 0
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "1.") || strings.HasPrefix(line, "2.") || strings.HasPrefix(line, "3.") || strings.HasPrefix(line, "4.") {
			count++
		}
	}
	return count
}

func qualityRank(tier string) int {
	switch strings.ToLower(strings.TrimSpace(tier)) {
	case "frontier":
		return 4
	case "high":
		return 3
	case "balanced":
		return 2
	case "economy", "value":
		return 1
	default:
		return 0
	}
}

func sortedMapKeys[T any](values map[string]T) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func sortedSpeechKeys(values map[string]SpeechClient) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func isLocalFile(path string) bool {
	return path != "" && !strings.Contains(path, "://")
}
