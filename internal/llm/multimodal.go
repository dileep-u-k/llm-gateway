package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/redis/go-redis/v9"
)

const (
	artifactRegistryPrefix = "artifact:registry:"
	artifactIndexKey       = "artifact:registry:index"
	executionPlanPrefix    = "execution:plan:"
	multimodalTTL          = 24 * time.Hour
)

type ArtifactRecord struct {
	ArtifactID     string            `json:"artifact_id,omitempty"`
	Name           string            `json:"name,omitempty"`
	Type           string            `json:"type,omitempty"`
	MimeType       string            `json:"mime_type,omitempty"`
	SourceURI      string            `json:"source_uri,omitempty"`
	Version        string            `json:"version,omitempty"`
	SizeBytes      int64             `json:"size_bytes,omitempty"`
	Role           string            `json:"role,omitempty"`
	DerivedFrom    string            `json:"derived_from,omitempty"`
	Lineage        []string          `json:"lineage,omitempty"`
	GeneratorModel string            `json:"generator_model,omitempty"`
	PromptSummary  string            `json:"prompt_summary,omitempty"`
	Text           string            `json:"text,omitempty"`
	OCRText        string            `json:"ocr_text,omitempty"`
	Transcript     string            `json:"transcript,omitempty"`
	Caption        string            `json:"caption,omitempty"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

type UnifiedRequest struct {
	Prompt                string
	History               []Message
	ConversationID        string
	InputType             string
	TaskType              string
	OutputType            string
	Assets                []api.AssetInput
	ArtifactRefs          []api.ArtifactReference
	RequiresOCR           bool
	RequiresTranscription bool
	RequiresGeneration    bool
	SyncOrAsyncPreference string
	StageBindingHints     []api.StageBindingHint
	ForceMetadata         ForceMetadata
	Preference            string
}

type TaskProfile struct {
	TaskType              string
	OutputType            string
	Modalities            []string
	ComplexityClass       string
	SyncMode              string
	PrimaryCapability     Capability
	NeedsRetrieval        bool
	NeedsTooling          bool
	RequiresOCR           bool
	RequiresTranscription bool
	RequiresGeneration    bool
}

type ExecutionStage struct {
	StageID          string     `json:"stage_id,omitempty"`
	StageType        string     `json:"stage_type,omitempty"`
	Title            string     `json:"title,omitempty"`
	Capability       Capability `json:"capability,omitempty"`
	DependsOn        []string   `json:"depends_on,omitempty"`
	ModelBinding     string     `json:"model_binding,omitempty"`
	ForcePolicy      string     `json:"force_policy,omitempty"`
	BindingViolation string     `json:"binding_violation,omitempty"`
	Status           string     `json:"status,omitempty"`
	Optional         bool       `json:"optional,omitempty"`
	ForceApplied     bool       `json:"force_applied,omitempty"`
	Strict           bool       `json:"strict,omitempty"`
}

type ExecutionPlan struct {
	PlanID               string           `json:"plan_id,omitempty"`
	PlanType             string           `json:"plan_type,omitempty"`
	SyncMode             string           `json:"sync_mode,omitempty"`
	CostTier             string           `json:"cost_tier,omitempty"`
	LatencyTier          string           `json:"latency_tier,omitempty"`
	PrimaryStageID       string           `json:"primary_stage_id,omitempty"`
	PrimaryCapability    Capability       `json:"primary_capability,omitempty"`
	ForceScope           string           `json:"force_scope,omitempty"`
	RequiresAsync        bool             `json:"requires_async,omitempty"`
	Modalities           []string         `json:"modalities,omitempty"`
	RequiredCapabilities []Capability     `json:"required_capabilities,omitempty"`
	Notes                []string         `json:"notes,omitempty"`
	Stages               []ExecutionStage `json:"stages,omitempty"`
}

func (p *ExecutionPlan) PrimaryStage() *ExecutionStage {
	if p == nil {
		return nil
	}
	for i := range p.Stages {
		if p.Stages[i].StageID == p.PrimaryStageID {
			return &p.Stages[i]
		}
	}
	return nil
}

type PreparedExecution struct {
	Request    UnifiedRequest
	Artifacts  []ArtifactRecord
	Task       TaskProfile
	Plan       *ExecutionPlan
	Modalities []string
}

type ArtifactRegistry struct {
	redisClient *redis.Client
	mu          sync.RWMutex
	local       map[string]ArtifactRecord
}

func NewArtifactRegistry(redisClient *redis.Client) *ArtifactRegistry {
	return &ArtifactRegistry{
		redisClient: redisClient,
		local:       make(map[string]ArtifactRecord),
	}
}

func (r *ArtifactRegistry) Register(ctx context.Context, artifact ArtifactRecord) (ArtifactRecord, error) {
	if artifact.ArtifactID == "" {
		artifact.ArtifactID = GenerateCacheKey(strings.Join([]string{
			artifact.Name,
			artifact.Type,
			artifact.SourceURI,
			artifact.Version,
			artifact.Text,
			time.Now().UTC().Format(time.RFC3339Nano),
		}, "|"))
	}
	if artifact.Version == "" {
		artifact.Version = time.Now().UTC().Format(time.RFC3339Nano)
	}

	r.mu.Lock()
	r.local[artifact.ArtifactID] = artifact
	r.mu.Unlock()

	if r.redisClient != nil {
		payload, err := json.Marshal(artifact)
		if err != nil {
			return artifact, err
		}
		pipe := r.redisClient.TxPipeline()
		pipe.Set(ctx, artifactRegistryPrefix+artifact.ArtifactID, payload, multimodalTTL)
		pipe.LPush(ctx, artifactIndexKey, artifact.ArtifactID)
		pipe.LTrim(ctx, artifactIndexKey, 0, 199)
		pipe.Expire(ctx, artifactIndexKey, multimodalTTL)
		if _, err := pipe.Exec(ctx); err != nil {
			return artifact, err
		}
	}
	return artifact, nil
}

func (r *ArtifactRegistry) Resolve(ctx context.Context, refs []api.ArtifactReference) ([]ArtifactRecord, error) {
	if len(refs) == 0 {
		return nil, nil
	}
	artifacts := make([]ArtifactRecord, 0, len(refs))
	for _, ref := range refs {
		if strings.TrimSpace(ref.ArtifactID) == "" {
			continue
		}
		artifact, ok, err := r.Get(ctx, ref.ArtifactID)
		if err != nil {
			return nil, err
		}
		if ok {
			artifacts = append(artifacts, artifact)
		}
	}
	return artifacts, nil
}

func (r *ArtifactRegistry) Get(ctx context.Context, artifactID string) (ArtifactRecord, bool, error) {
	r.mu.RLock()
	artifact, ok := r.local[artifactID]
	r.mu.RUnlock()
	if ok {
		return artifact, true, nil
	}
	if r.redisClient == nil {
		return ArtifactRecord{}, false, nil
	}
	value, err := r.redisClient.Get(ctx, artifactRegistryPrefix+artifactID).Bytes()
	if err == redis.Nil {
		return ArtifactRecord{}, false, nil
	}
	if err != nil {
		return ArtifactRecord{}, false, err
	}
	if err := json.Unmarshal(value, &artifact); err != nil {
		return ArtifactRecord{}, false, err
	}
	r.mu.Lock()
	r.local[artifactID] = artifact
	r.mu.Unlock()
	return artifact, true, nil
}

func (r *ArtifactRegistry) List(ctx context.Context, limit int64) ([]ArtifactRecord, error) {
	if limit <= 0 {
		limit = 20
	}
	records := make([]ArtifactRecord, 0)
	seen := make(map[string]struct{})

	r.mu.RLock()
	for _, record := range r.local {
		records = append(records, record)
		seen[record.ArtifactID] = struct{}{}
		if int64(len(records)) >= limit {
			r.mu.RUnlock()
			return records[:limit], nil
		}
	}
	r.mu.RUnlock()

	if r.redisClient == nil {
		return records, nil
	}
	ids, err := r.redisClient.LRange(ctx, artifactIndexKey, 0, limit-1).Result()
	if err != nil {
		return records, err
	}
	for _, artifactID := range ids {
		if _, ok := seen[artifactID]; ok {
			continue
		}
		record, ok, err := r.Get(ctx, artifactID)
		if err != nil {
			return records, err
		}
		if ok {
			records = append(records, record)
			if int64(len(records)) >= limit {
				break
			}
		}
	}
	return records, nil
}

type AssetIngestor struct {
	registry *ArtifactRegistry
}

func NewAssetIngestor(registry *ArtifactRegistry) *AssetIngestor {
	return &AssetIngestor{registry: registry}
}

func (i *AssetIngestor) Ingest(ctx context.Context, req UnifiedRequest) ([]ArtifactRecord, error) {
	var artifacts []ArtifactRecord
	for _, asset := range req.Assets {
		artifact, err := i.ingestAsset(ctx, asset)
		if err != nil {
			return nil, err
		}
		artifacts = append(artifacts, artifact)
	}
	referenced, err := i.registry.Resolve(ctx, req.ArtifactRefs)
	if err != nil {
		return nil, err
	}
	artifacts = append(artifacts, referenced...)
	return dedupeArtifacts(artifacts), nil
}

func (i *AssetIngestor) ingestAsset(ctx context.Context, asset api.AssetInput) (ArtifactRecord, error) {
	assetType := normalizeAssetType(asset.Type, asset.URI, asset.MimeType, asset.Name)
	artifact := ArtifactRecord{
		ArtifactID: firstNonEmpty(asset.AssetID, GenerateCacheKey(asset.Name+"|"+asset.URI+"|"+asset.InlineText)),
		Name:       firstNonEmpty(asset.Name, filepath.Base(asset.URI), asset.AssetID),
		Type:       assetType,
		MimeType:   firstNonEmpty(asset.MimeType, inferMimeType(assetType, asset.URI)),
		SourceURI:  asset.URI,
		SizeBytes:  asset.SizeBytes,
		Text:       strings.TrimSpace(asset.InlineText),
		OCRText:    strings.TrimSpace(asset.OCRText),
		Transcript: strings.TrimSpace(asset.Transcript),
		Caption:    strings.TrimSpace(asset.Caption),
		Metadata:   cloneStringMap(asset.Metadata),
	}

	if artifact.Metadata == nil {
		artifact.Metadata = make(map[string]string)
	}
	if asset.URI != "" {
		if err := hydrateArtifactFromPath(&artifact); err != nil {
			return ArtifactRecord{}, err
		}
	}

	return i.registry.Register(ctx, artifact)
}

type ModalityDetector struct{}

func NewModalityDetector() *ModalityDetector {
	return &ModalityDetector{}
}

func (d *ModalityDetector) Detect(req UnifiedRequest, artifacts []ArtifactRecord) []string {
	modalitySet := make(map[string]struct{})
	add := func(value string) {
		value = strings.TrimSpace(strings.ToLower(value))
		if value != "" && value != "mixed" {
			modalitySet[value] = struct{}{}
		}
	}

	add(req.InputType)
	if len(artifacts) == 0 && strings.TrimSpace(req.Prompt) != "" {
		add("text")
	}
	for _, artifact := range artifacts {
		add(artifact.Type)
	}

	out := make([]string, 0, len(modalitySet))
	for modality := range modalitySet {
		out = append(out, modality)
	}
	sort.Strings(out)
	if len(out) == 0 {
		return []string{"text"}
	}
	return out
}

type TaskClassifier struct{}

func NewTaskClassifier() *TaskClassifier {
	return &TaskClassifier{}
}

func (c *TaskClassifier) Classify(req UnifiedRequest, artifacts []ArtifactRecord, modalities []string) TaskProfile {
	prompt := strings.ToLower(strings.TrimSpace(req.Prompt))
	taskType := strings.TrimSpace(req.TaskType)
	outputType := firstNonEmpty(strings.TrimSpace(req.OutputType), inferOutputType(prompt, req, artifacts))
	hasImage := containsString(modalities, "image")
	hasDocument := containsString(modalities, "document")
	hasAudio := containsString(modalities, "audio")
	hasVideo := containsString(modalities, "video")
	hasMixed := len(modalities) > 1

	if taskType == "" {
		switch {
		case outputType == "bundle":
			taskType = "creative_bundle"
		case outputType == "video":
			taskType = "video_generation_hook"
		case outputType == "audio":
			taskType = "speech_synthesis"
		case outputType == "image" && hasImage:
			taskType = "image_editing"
		case outputType == "image":
			taskType = "image_generation"
		case containsAny(prompt, "voiceover", "spoken", "read aloud", "tts", "narration", "podcast intro"):
			taskType = "speech_synthesis"
		case containsAny(prompt, "storyboard", "scene plan", "shot list", "video teaser", "video trailer"):
			taskType = "video_generation_hook"
		case containsAny(prompt, "bundle", "campaign kit", "poster and narration", "poster + narration", "voiceover and image"):
			taskType = "creative_bundle"
		case hasVideo:
			taskType = "video_summarization"
		case hasAudio:
			taskType = "audio_summarization"
		case hasDocument && hasImage:
			taskType = "document_image_reasoning"
		case hasDocument && (req.RequiresOCR || missingStructuredText(artifacts, "document")):
			taskType = "ocr_qa"
		case hasDocument:
			taskType = "document_qa"
		case hasImage:
			taskType = "image_understanding"
		default:
			taskType = "direct_text_generation"
		}
	}

	primaryCapability := CapabilityTextGeneration
	switch taskType {
	case "image_generation":
		primaryCapability = CapabilityImageGeneration
	case "image_editing":
		primaryCapability = CapabilityImageEditing
	case "speech_synthesis":
		primaryCapability = CapabilityTTS
	case "video_generation_hook", "creative_bundle":
		primaryCapability = CapabilityVideoGeneration
	case "image_understanding":
		primaryCapability = CapabilityImageUnderstanding
	}

	complexity := "simple"
	switch {
	case len(req.Prompt) > 800 || len(artifacts) > 3 || hasVideo || hasMixed:
		complexity = "high"
	case len(req.Prompt) > 240 || len(artifacts) > 1:
		complexity = "medium"
	}

	syncMode := "sync"
	if strings.EqualFold(req.SyncOrAsyncPreference, "async") || hasVideo || (hasAudio && len(artifacts) > 1) || complexity == "high" && (req.RequiresTranscription || req.RequiresOCR) {
		syncMode = "async_recommended"
	}

	return TaskProfile{
		TaskType:              taskType,
		OutputType:            outputType,
		Modalities:            append([]string(nil), modalities...),
		ComplexityClass:       complexity,
		SyncMode:              syncMode,
		PrimaryCapability:     primaryCapability,
		NeedsRetrieval:        hasDocument || containsAny(prompt, "cite", "reference", "according to", "source"),
		NeedsTooling:          containsAny(prompt, "weather", "forecast", "calculate", "calculator", "news", "headline"),
		RequiresOCR:           req.RequiresOCR || hasDocument && missingStructuredText(artifacts, "document"),
		RequiresTranscription: req.RequiresTranscription || hasAudio && missingStructuredText(artifacts, "audio"),
		RequiresGeneration:    req.RequiresGeneration || outputType == "image" || outputType == "audio" || outputType == "video" || outputType == "bundle",
	}
}

type ExecutionPlanner struct {
	modelRegistry *ModelRegistry
}

func NewExecutionPlanner(modelRegistry *ModelRegistry) *ExecutionPlanner {
	return &ExecutionPlanner{modelRegistry: modelRegistry}
}

func (p *ExecutionPlanner) Build(req UnifiedRequest, task TaskProfile, artifacts []ArtifactRecord) *ExecutionPlan {
	plan := &ExecutionPlan{
		PlanID:            GenerateCacheKey(strings.Join([]string{req.ConversationID, req.Prompt, task.TaskType, time.Now().UTC().Format(time.RFC3339Nano)}, "|")),
		PlanType:          planTypeForTask(task.TaskType),
		SyncMode:          task.SyncMode,
		CostTier:          costTierForTask(task),
		LatencyTier:       latencyTierForTask(task),
		PrimaryCapability: task.PrimaryCapability,
		ForceScope:        req.ForceMetadata.Scope,
		RequiresAsync:     strings.Contains(task.SyncMode, "async"),
		Modalities:        append([]string(nil), task.Modalities...),
		Notes:             buildPlanNotes(task, artifacts),
	}

	addStage := func(stage ExecutionStage) {
		if stage.Status == "" {
			stage.Status = "planned"
		}
		plan.Stages = append(plan.Stages, stage)
		if stage.Capability != "" {
			plan.RequiredCapabilities = append(plan.RequiredCapabilities, stage.Capability)
		}
	}

	if len(artifacts) > 0 {
		addStage(ExecutionStage{
			StageID:   "ingest_assets",
			StageType: "asset_ingestion",
			Title:     "Ingest and normalize assets",
		})
	}
	if task.RequiresOCR {
		addStage(ExecutionStage{
			StageID:    "ocr_extract",
			StageType:  "ocr",
			Title:      "Extract text from documents or screenshots",
			Capability: CapabilityOCR,
			DependsOn:  maybeDependsOn(len(artifacts) > 0, "ingest_assets"),
		})
	}
	if task.RequiresTranscription {
		addStage(ExecutionStage{
			StageID:    "transcribe_audio",
			StageType:  "transcription",
			Title:      "Transcribe speech into timestamped text",
			Capability: CapabilityTranscription,
			DependsOn:  maybeDependsOn(len(artifacts) > 0, "ingest_assets"),
		})
	}
	if containsString(task.Modalities, "video") {
		addStage(ExecutionStage{
			StageID:    "segment_video",
			StageType:  "video_understanding",
			Title:      "Segment and summarize video timeline signals",
			Capability: CapabilityVideoUnderstanding,
			DependsOn:  maybeDependsOn(len(artifacts) > 0, "ingest_assets"),
		})
	}
	if task.NeedsRetrieval {
		addStage(ExecutionStage{
			StageID:    "retrieve_context",
			StageType:  "retrieval",
			Title:      "Retrieve supporting evidence and context",
			Capability: CapabilityEmbeddings,
			DependsOn:  planDependencies(plan, "ocr_extract", "transcribe_audio", "segment_video"),
			Optional:   !containsString(task.Modalities, "document"),
		})
	}
	if task.NeedsTooling {
		addStage(ExecutionStage{
			StageID:    "tool_runtime",
			StageType:  "tool_execution",
			Title:      "Execute deterministic tools before synthesis",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "retrieve_context"),
			Optional:   false,
		})
	}

	switch task.TaskType {
	case "image_generation":
		addStage(ExecutionStage{
			StageID:    "creative_conditioning",
			StageType:  "creative_conditioning",
			Title:      "Select reference assets and creative constraints",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "retrieve_context", "ocr_extract"),
			Optional:   true,
		})
		addStage(ExecutionStage{
			StageID:    "prompt_refine",
			StageType:  "prompt_refinement",
			Title:      "Refine the visual generation prompt",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "creative_conditioning"),
			Optional:   true,
		})
		addStage(ExecutionStage{
			StageID:    "policy_validate",
			StageType:  "policy_validation",
			Title:      "Validate the generation request against creative safety rules",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "prompt_refine"),
		})
		addStage(ExecutionStage{
			StageID:    "generate_image",
			StageType:  "image_generation",
			Title:      "Generate the requested image artifact",
			Capability: CapabilityImageGeneration,
			DependsOn:  planDependencies(plan, "policy_validate"),
		})
		addStage(ExecutionStage{
			StageID:    "quality_check",
			StageType:  "quality_validation",
			Title:      "Validate image quality and artifact completeness",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "generate_image"),
		})
		addStage(ExecutionStage{
			StageID:   "store_artifact",
			StageType: "artifact_storage",
			Title:     "Register generated image artifact with lineage metadata",
			DependsOn: planDependencies(plan, "quality_check"),
		})
		plan.PrimaryStageID = "generate_image"
	case "image_editing":
		addStage(ExecutionStage{
			StageID:    "creative_conditioning",
			StageType:  "creative_conditioning",
			Title:      "Analyze source image and desired transformation constraints",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "ingest_assets", "retrieve_context"),
			Optional:   true,
		})
		addStage(ExecutionStage{
			StageID:    "prompt_refine",
			StageType:  "prompt_refinement",
			Title:      "Refine the edit instructions for the source image",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "creative_conditioning"),
			Optional:   true,
		})
		addStage(ExecutionStage{
			StageID:    "policy_validate",
			StageType:  "policy_validation",
			Title:      "Validate the image edit instructions",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "prompt_refine"),
		})
		addStage(ExecutionStage{
			StageID:    "edit_image",
			StageType:  "image_editing",
			Title:      "Transform the provided image asset",
			Capability: CapabilityImageEditing,
			DependsOn:  planDependencies(plan, "ingest_assets", "policy_validate"),
		})
		addStage(ExecutionStage{
			StageID:    "quality_check",
			StageType:  "quality_validation",
			Title:      "Validate edited image quality and lineage",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "edit_image"),
		})
		addStage(ExecutionStage{
			StageID:   "store_artifact",
			StageType: "artifact_storage",
			Title:     "Register edited image artifact with edit lineage",
			DependsOn: planDependencies(plan, "quality_check"),
		})
		plan.PrimaryStageID = "edit_image"
	case "speech_synthesis":
		addStage(ExecutionStage{
			StageID:    "text_cleanup",
			StageType:  "text_cleanup",
			Title:      "Clean and normalize the narration script",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "retrieve_context", "transcribe_audio"),
		})
		addStage(ExecutionStage{
			StageID:    "voice_select",
			StageType:  "voice_selection",
			Title:      "Choose a voice and speaking style",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "text_cleanup"),
		})
		addStage(ExecutionStage{
			StageID:    "policy_validate",
			StageType:  "policy_validation",
			Title:      "Validate the speech synthesis request",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "voice_select"),
		})
		addStage(ExecutionStage{
			StageID:    "synthesize_speech",
			StageType:  "speech_synthesis",
			Title:      "Generate spoken audio from the prepared script",
			Capability: CapabilityTTS,
			DependsOn:  planDependencies(plan, "policy_validate"),
		})
		addStage(ExecutionStage{
			StageID:    "post_process_audio",
			StageType:  "audio_post_processing",
			Title:      "Finalize audio packaging and playback metadata",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "synthesize_speech"),
		})
		addStage(ExecutionStage{
			StageID:    "quality_check",
			StageType:  "quality_validation",
			Title:      "Validate audio quality and artifact integrity",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "post_process_audio"),
		})
		addStage(ExecutionStage{
			StageID:   "store_artifact",
			StageType: "artifact_storage",
			Title:     "Register generated audio artifact for reuse",
			DependsOn: planDependencies(plan, "quality_check"),
		})
		plan.PrimaryStageID = "synthesize_speech"
	case "video_generation_hook":
		addStage(ExecutionStage{
			StageID:    "storyboard_plan",
			StageType:  "storyboard_generation",
			Title:      "Create a storyboard and scene plan for downstream video generation",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "retrieve_context", "transcribe_audio", "segment_video"),
		})
		addStage(ExecutionStage{
			StageID:    "synthesize_voiceover",
			StageType:  "speech_synthesis",
			Title:      "Optionally generate voiceover audio for the storyboard",
			Capability: CapabilityTTS,
			DependsOn:  planDependencies(plan, "storyboard_plan"),
			Optional:   true,
		})
		addStage(ExecutionStage{
			StageID:    "package_bundle",
			StageType:  "asset_bundle",
			Title:      "Bundle storyboard, prompts, and optional narration for video tools",
			Capability: CapabilityVideoGeneration,
			DependsOn:  planDependencies(plan, "storyboard_plan", "synthesize_voiceover"),
		})
		plan.PrimaryStageID = "package_bundle"
	case "creative_bundle":
		addStage(ExecutionStage{
			StageID:    "creative_conditioning",
			StageType:  "creative_conditioning",
			Title:      "Build a composite brief for multi-asset creative generation",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "retrieve_context", "ocr_extract"),
		})
		addStage(ExecutionStage{
			StageID:    "prompt_refine",
			StageType:  "prompt_refinement",
			Title:      "Refine prompts for each asset in the bundle",
			Capability: CapabilityTextGeneration,
			DependsOn:  planDependencies(plan, "creative_conditioning"),
		})
		addStage(ExecutionStage{
			StageID:    "generate_image",
			StageType:  "image_generation",
			Title:      "Generate the visual anchor asset",
			Capability: CapabilityImageGeneration,
			DependsOn:  planDependencies(plan, "prompt_refine"),
		})
		addStage(ExecutionStage{
			StageID:    "synthesize_speech",
			StageType:  "speech_synthesis",
			Title:      "Generate the complementary narration asset",
			Capability: CapabilityTTS,
			DependsOn:  planDependencies(plan, "prompt_refine"),
			Optional:   true,
		})
		addStage(ExecutionStage{
			StageID:    "package_bundle",
			StageType:  "asset_bundle",
			Title:      "Register a reusable creative asset bundle",
			Capability: CapabilityVideoGeneration,
			DependsOn:  planDependencies(plan, "generate_image", "synthesize_speech"),
		})
		plan.PrimaryStageID = "package_bundle"
	default:
		addStage(ExecutionStage{
			StageID:    "reason_answer",
			StageType:  "reasoning",
			Title:      "Synthesize the final answer across modalities",
			Capability: task.PrimaryCapability,
			DependsOn:  planDependencies(plan, "retrieve_context", "tool_runtime", "ocr_extract", "transcribe_audio", "segment_video"),
		})
		plan.PrimaryStageID = "reason_answer"
	}

	plan.RequiredCapabilities = dedupeCapabilities(plan.RequiredCapabilities)
	return plan
}

type StageBindingResolver struct {
	modelRegistry *ModelRegistry
}

func NewStageBindingResolver(modelRegistry *ModelRegistry) *StageBindingResolver {
	return &StageBindingResolver{modelRegistry: modelRegistry}
}

func (r *StageBindingResolver) Apply(plan *ExecutionPlan, req UnifiedRequest) {
	if plan == nil {
		return
	}
	for i := range plan.Stages {
		stage := &plan.Stages[i]
		stage.ForcePolicy = "dynamic"
		stage.Strict = req.ForceMetadata.Strict

		hint := findStageBindingHint(req.StageBindingHints, stage.StageID, stage.Capability)
		if hint != nil && hint.ModelID != "" {
			stage.ModelBinding = hint.ModelID
			stage.Status = "hinted"
		}

		if !req.ForceMetadata.IsForced || req.ForceMetadata.PinnedModel == "" {
			continue
		}

		applies := false
		switch req.ForceMetadata.Scope {
		case "strict_end_to_end_force":
			applies = true
			stage.ForcePolicy = "strict_end_to_end_force"
		case "capability_scoped_force":
			applies = r.supports(req.ForceMetadata.PinnedModel, stage.Capability)
			stage.ForcePolicy = "capability_scoped_force"
		default:
			applies = stage.StageID == plan.PrimaryStageID
			stage.ForcePolicy = "primary_reasoner_force"
		}
		if !applies {
			continue
		}

		stage.ForceApplied = true
		if r.supports(req.ForceMetadata.PinnedModel, stage.Capability) {
			stage.ModelBinding = req.ForceMetadata.PinnedModel
		} else {
			stage.BindingViolation = fmt.Sprintf("forced model %s does not support capability %s", req.ForceMetadata.PinnedModel, stage.Capability)
			stage.Status = "needs_override"
		}
	}
}

func (r *StageBindingResolver) supports(modelID string, capability Capability) bool {
	if capability == "" || modelID == "" {
		return true
	}
	if r.modelRegistry == nil {
		return CapabilityForModel(modelID) == capability
	}
	record, ok := r.modelRegistry.Get(modelID)
	if !ok {
		return CapabilityForModel(modelID) == capability
	}
	if len(record.Capabilities) == 0 {
		return CapabilityForModel(modelID) == capability
	}
	for _, supported := range record.Capabilities {
		if supported == capability {
			return true
		}
	}
	return false
}

type ExecutionPlanStore struct {
	redisClient *redis.Client
	mu          sync.RWMutex
	local       map[string]ExecutionPlan
}

func NewExecutionPlanStore(redisClient *redis.Client) *ExecutionPlanStore {
	return &ExecutionPlanStore{
		redisClient: redisClient,
		local:       make(map[string]ExecutionPlan),
	}
}

func (s *ExecutionPlanStore) Save(ctx context.Context, plan *ExecutionPlan) error {
	if plan == nil || plan.PlanID == "" {
		return nil
	}
	s.mu.Lock()
	s.local[plan.PlanID] = *plan
	s.mu.Unlock()

	if s.redisClient == nil {
		return nil
	}
	payload, err := json.Marshal(plan)
	if err != nil {
		return err
	}
	return s.redisClient.Set(ctx, executionPlanPrefix+plan.PlanID, payload, multimodalTTL).Err()
}

type MultimodalRuntime struct {
	registry        *ArtifactRegistry
	ingestor        *AssetIngestor
	detector        *ModalityDetector
	classifier      *TaskClassifier
	planner         *ExecutionPlanner
	bindingResolver *StageBindingResolver
	planStore       *ExecutionPlanStore
}

func NewMultimodalRuntime(redisClient *redis.Client, controlPlane *ControlPlane) *MultimodalRuntime {
	var modelRegistry *ModelRegistry
	if controlPlane != nil {
		modelRegistry = controlPlane.Models()
	}
	registry := NewArtifactRegistry(redisClient)
	return &MultimodalRuntime{
		registry:        registry,
		ingestor:        NewAssetIngestor(registry),
		detector:        NewModalityDetector(),
		classifier:      NewTaskClassifier(),
		planner:         NewExecutionPlanner(modelRegistry),
		bindingResolver: NewStageBindingResolver(modelRegistry),
		planStore:       NewExecutionPlanStore(redisClient),
	}
}

func (r *MultimodalRuntime) ArtifactRegistry() *ArtifactRegistry {
	if r == nil {
		return nil
	}
	return r.registry
}

func (r *MultimodalRuntime) Prepare(ctx context.Context, req api.GenerationRequest, history []Message, forceMeta ForceMetadata) (*PreparedExecution, error) {
	unified := UnifiedRequest{
		Prompt:                req.Prompt,
		History:               history,
		ConversationID:        req.ConversationID,
		InputType:             req.InputType,
		TaskType:              req.TaskType,
		OutputType:            req.OutputType,
		Assets:                append([]api.AssetInput(nil), req.Assets...),
		ArtifactRefs:          append([]api.ArtifactReference(nil), req.ArtifactRefs...),
		RequiresOCR:           req.RequiresOCR,
		RequiresTranscription: req.RequiresTranscription,
		RequiresGeneration:    req.RequiresGeneration,
		SyncOrAsyncPreference: req.SyncOrAsyncPreference,
		StageBindingHints:     append([]api.StageBindingHint(nil), req.StageBindingHints...),
		ForceMetadata:         forceMeta,
		Preference:            req.Config.Preference,
	}

	artifacts, err := r.ingestor.Ingest(ctx, unified)
	if err != nil {
		return nil, err
	}
	modalities := r.detector.Detect(unified, artifacts)
	task := r.classifier.Classify(unified, artifacts, modalities)
	plan := r.planner.Build(unified, task, artifacts)
	r.bindingResolver.Apply(plan, unified)
	if err := r.planStore.Save(ctx, plan); err != nil {
		return nil, err
	}

	return &PreparedExecution{
		Request:    unified,
		Artifacts:  artifacts,
		Task:       task,
		Plan:       plan,
		Modalities: modalities,
	}, nil
}

func hydrateArtifactFromPath(artifact *ArtifactRecord) error {
	path := strings.TrimSpace(artifact.SourceURI)
	if path == "" || strings.Contains(path, "://") {
		return nil
	}
	info, err := os.Stat(path)
	if err != nil {
		return nil
	}
	if artifact.SizeBytes == 0 {
		artifact.SizeBytes = info.Size()
	}
	if artifact.Name == "" {
		artifact.Name = info.Name()
	}
	if artifact.Type == "document" && artifact.Text == "" && isTextLikePath(path) {
		bytes, err := os.ReadFile(path)
		if err == nil {
			artifact.Text = strings.TrimSpace(string(limitBytes(bytes, 64*1024)))
			artifact.Metadata["ingested_from"] = "file"
		}
	}
	if artifact.Type == "image" {
		file, err := os.Open(path)
		if err == nil {
			defer file.Close()
			if cfg, _, decodeErr := image.DecodeConfig(file); decodeErr == nil {
				artifact.Metadata["width"] = fmt.Sprintf("%d", cfg.Width)
				artifact.Metadata["height"] = fmt.Sprintf("%d", cfg.Height)
			}
		}
	}
	return nil
}

func normalizeAssetType(assetType, uri, mimeType, name string) string {
	value := strings.TrimSpace(strings.ToLower(assetType))
	if value != "" {
		if value == "mixed_multimodal" {
			return "mixed"
		}
		return value
	}
	ext := strings.ToLower(filepath.Ext(firstNonEmpty(uri, name)))
	switch {
	case strings.HasPrefix(mimeType, "image/"), containsString([]string{".png", ".jpg", ".jpeg", ".gif", ".webp"}, ext):
		return "image"
	case strings.HasPrefix(mimeType, "audio/"), containsString([]string{".mp3", ".wav", ".m4a", ".aac", ".flac"}, ext):
		return "audio"
	case strings.HasPrefix(mimeType, "video/"), containsString([]string{".mp4", ".mov", ".mkv", ".avi"}, ext):
		return "video"
	case strings.HasPrefix(mimeType, "application/pdf"), containsString([]string{".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}, ext):
		return "document"
	default:
		return "document"
	}
}

func inferMimeType(assetType, uri string) string {
	ext := strings.ToLower(filepath.Ext(uri))
	switch assetType {
	case "image":
		switch ext {
		case ".png":
			return "image/png"
		case ".jpg", ".jpeg":
			return "image/jpeg"
		default:
			return "image/*"
		}
	case "audio":
		return "audio/*"
	case "video":
		return "video/*"
	default:
		if ext == ".pdf" {
			return "application/pdf"
		}
		return "text/plain"
	}
}

func inferOutputType(prompt string, req UnifiedRequest, artifacts []ArtifactRecord) string {
	if strings.TrimSpace(req.OutputType) != "" {
		return req.OutputType
	}
	switch {
	case containsAny(prompt, "bundle", "campaign kit", "poster and narration", "poster + narration"):
		return "bundle"
	case containsAny(prompt, "storyboard", "scene plan", "video", "teaser", "trailer"):
		return "video"
	case containsAny(prompt, "voiceover", "spoken", "audio", "speech", "narration", "tts", "podcast"):
		return "audio"
	case containsAny(prompt, "generate an image", "create an image", "poster", "illustration", "diagram", "picture", "logo", "thumbnail"):
		return "image"
	case containsAny(prompt, "json", "schema", "fields", "structured output"):
		return "json"
	case req.RequiresTranscription || containsAny(prompt, "transcribe", "transcript"):
		return "transcript"
	case containsAny(prompt, "summary", "summarize"):
		return "summary"
	case len(artifacts) > 0 && containsAny(prompt, "extract", "field"):
		return "extracted_fields"
	default:
		return "text"
	}
}

func planTypeForTask(taskType string) string {
	switch taskType {
	case "image_generation":
		return "generation_pipeline"
	case "image_editing":
		return "image_editing_pipeline"
	case "speech_synthesis":
		return "speech_synthesis_pipeline"
	case "video_generation_hook":
		return "video_generation_hook_pipeline"
	case "creative_bundle":
		return "creative_bundle_pipeline"
	case "ocr_qa":
		return "ocr_then_reason"
	case "audio_summarization":
		return "transcribe_then_summarize"
	case "video_summarization":
		return "segment_video_then_summarize"
	case "document_image_reasoning":
		return "document_and_image_reasoning"
	case "document_qa":
		return "retrieve_then_answer"
	default:
		return "direct_text_generation"
	}
}

func buildPlanNotes(task TaskProfile, artifacts []ArtifactRecord) []string {
	notes := []string{
		fmt.Sprintf("task=%s", task.TaskType),
		fmt.Sprintf("complexity=%s", task.ComplexityClass),
		fmt.Sprintf("sync_mode=%s", task.SyncMode),
	}
	if len(artifacts) > 0 {
		notes = append(notes, fmt.Sprintf("artifact_count=%d", len(artifacts)))
	}
	if task.RequiresOCR {
		notes = append(notes, "ocr_required")
	}
	if task.RequiresTranscription {
		notes = append(notes, "transcription_required")
	}
	if task.RequiresGeneration {
		notes = append(notes, "generation_required")
	}
	return notes
}

func costTierForTask(task TaskProfile) string {
	switch {
	case task.RequiresGeneration || task.ComplexityClass == "high":
		return "premium"
	case task.ComplexityClass == "medium":
		return "balanced"
	default:
		return "economy"
	}
}

func latencyTierForTask(task TaskProfile) string {
	switch task.SyncMode {
	case "async_recommended":
		return "background"
	case "sync":
		return "interactive"
	default:
		return "standard"
	}
}

func dedupeCapabilities(values []Capability) []Capability {
	var out []Capability
	seen := make(map[Capability]struct{}, len(values))
	for _, value := range values {
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func dedupeArtifacts(values []ArtifactRecord) []ArtifactRecord {
	var out []ArtifactRecord
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		key := firstNonEmpty(value.ArtifactID, value.SourceURI+"|"+value.Name)
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, value)
	}
	return out
}

func maybeDependsOn(ok bool, value string) []string {
	if ok && value != "" {
		return []string{value}
	}
	return nil
}

func planDependencies(plan *ExecutionPlan, stageIDs ...string) []string {
	if plan == nil {
		return nil
	}
	present := make(map[string]struct{}, len(plan.Stages))
	for _, stage := range plan.Stages {
		present[stage.StageID] = struct{}{}
	}
	var out []string
	for _, stageID := range stageIDs {
		if _, ok := present[stageID]; ok {
			out = append(out, stageID)
		}
	}
	return out
}

func findStageBindingHint(hints []api.StageBindingHint, stageID string, capability Capability) *api.StageBindingHint {
	for _, hint := range hints {
		if hint.StageID != "" && hint.StageID == stageID {
			return &hint
		}
		if hint.Capability != "" && hint.Capability == string(capability) {
			return &hint
		}
	}
	return nil
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if strings.EqualFold(value, target) {
			return true
		}
	}
	return false
}

func missingStructuredText(artifacts []ArtifactRecord, kind string) bool {
	for _, artifact := range artifacts {
		if artifact.Type != kind {
			continue
		}
		if strings.TrimSpace(firstNonEmpty(artifact.Text, artifact.OCRText, artifact.Transcript, artifact.Caption)) != "" {
			return false
		}
	}
	return true
}

func isTextLikePath(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".tsv", ".html", ".xml", ".go", ".py", ".js", ".ts", ".java":
		return true
	default:
		return false
	}
}

func limitBytes(value []byte, max int) []byte {
	if max <= 0 || len(value) <= max {
		return value
	}
	return value[:max]
}

func cloneStringMap(value map[string]string) map[string]string {
	if len(value) == 0 {
		return nil
	}
	out := make(map[string]string, len(value))
	for key, item := range value {
		out[key] = item
	}
	return out
}
