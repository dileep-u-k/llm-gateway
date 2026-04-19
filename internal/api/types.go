// In file: internal/api/types.go

// Package api defines the public API contract for the LLM Gateway.
// These data structures are used for request binding and response serialization,
// serving as a stable, versioned interface for all client interactions.
package api

// Message defines the structure for a single message in a conversation history.
// This is part of the public API and is used in the GenerationRequest.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// GenerationRequest defines the structure for an incoming request to the /generate endpoint.
// It includes the user's prompt and all necessary configuration for the gateway to process it.
type GenerationRequest struct {
	// Prompt is the user's query or instruction.
	Prompt string `json:"prompt" binding:"required"`
	// UserID is an identifier for the end-user, crucial for logging, auditing, and rate-limiting.
	UserID string `json:"user_id"`
	// TenantID identifies the tenant boundary for policy, storage, and audit enforcement.
	TenantID string `json:"tenant_id,omitempty"`
	// WorkspaceID identifies the workspace boundary within a tenant.
	WorkspaceID string `json:"workspace_id,omitempty"`
	// --- ADD THIS LINE ---
	// ConversationID links multiple requests together into a single chat session,
	// enabling features like model stickiness.
	ConversationID string `json:"conversation_id,omitempty"`
	// --- THIS FIELD IS NEW ---
	// History contains the list of previous messages in the conversation for context.
	History []Message `json:"history,omitempty"`
	// InputType describes the primary modality family for the request.
	InputType string `json:"input_type,omitempty"`
	// TaskType lets callers provide an explicit workflow hint for multimodal planning.
	TaskType string `json:"task_type,omitempty"`
	// OutputType captures the requested response or artifact type.
	OutputType string `json:"output_type,omitempty"`
	// Assets contains inline or referenced multimodal inputs that should participate in one workflow.
	Assets []AssetInput `json:"assets,omitempty"`
	// ArtifactRefs references previously registered artifacts that should be brought into this request.
	ArtifactRefs []ArtifactReference `json:"artifact_refs,omitempty"`
	// RequiresOCR signals that OCR or text extraction is expected before reasoning.
	RequiresOCR bool `json:"requires_ocr,omitempty"`
	// RequiresTranscription signals that speech-to-text should be planned before reasoning.
	RequiresTranscription bool `json:"requires_transcription,omitempty"`
	// RequiresGeneration signals that the workflow is expected to create an artifact as its end result.
	RequiresGeneration bool `json:"requires_generation,omitempty"`
	// SyncOrAsyncPreference captures whether the caller prefers immediate execution or async planning.
	SyncOrAsyncPreference string `json:"sync_or_async_preference,omitempty"`
	// CallbackURL optionally receives a completion webhook for async executions.
	CallbackURL string `json:"callback_url,omitempty"`
	// StageBindingHints lets callers guide the planner toward or away from specific stage bindings.
	StageBindingHints []StageBindingHint `json:"stage_binding_hints,omitempty"`
	// Rollout controls optional shadow, canary, and replay safety workflows.
	Rollout RolloutOptions `json:"rollout,omitempty"`
	// Evaluation controls optional benchmark and scoring passes for this request.
	Evaluation EvaluationOptions `json:"evaluation,omitempty"`
	// Config holds all the parameters that control how the gateway processes and routes the request.
	Config GenerationConfig `json:"config"`
}

type RolloutOptions struct {
	ShadowModel           string `json:"shadow_model,omitempty"`
	CanaryModel           string `json:"canary_model,omitempty"`
	CanaryPercent         int    `json:"canary_percent,omitempty"`
	ReplayOrchestrationID string `json:"replay_orchestration_id,omitempty"`
}

type EvaluationOptions struct {
	Enabled   bool     `json:"enabled,omitempty"`
	Suites    []string `json:"suites,omitempty"`
	Baselines []string `json:"baselines,omitempty"`
}

type AssetInput struct {
	AssetID    string            `json:"asset_id,omitempty"`
	Type       string            `json:"type,omitempty"`
	Name       string            `json:"name,omitempty"`
	URI        string            `json:"uri,omitempty"`
	MimeType   string            `json:"mime_type,omitempty"`
	InlineText string            `json:"inline_text,omitempty"`
	OCRText    string            `json:"ocr_text,omitempty"`
	Transcript string            `json:"transcript,omitempty"`
	Caption    string            `json:"caption,omitempty"`
	SizeBytes  int64             `json:"size_bytes,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

type ArtifactReference struct {
	ArtifactID string `json:"artifact_id,omitempty"`
	Role       string `json:"role,omitempty"`
	Version    string `json:"version,omitempty"`
}

type StageBindingHint struct {
	StageID    string `json:"stage_id,omitempty"`
	Capability string `json:"capability,omitempty"`
	ModelID    string `json:"model_id,omitempty"`
}

// GenerationConfig holds all user-configurable parameters for a single LLM request.
type GenerationConfig struct {
	// Preference is the routing strategy the user prefers. The gateway's router will
	// use this to select the optimal model. Examples: "cost", "latency", "max_quality".
	Preference string `json:"preference,omitempty"`

	// --- ADD THIS LINE ---
	ImagePreference string `json:"image_preference,omitempty"`
	// --- ADD THIS LINE ---
	// ForceModel allows the user to bypass the router and pin a specific model to the
	// start of a new conversation.
	ForceModel string `json:"force_model,omitempty"`
	// ForceScope controls how widely a forced model pin should apply across execution.
	// Phase 0 currently supports text-first semantics, but the field is future-proofed
	// for stage-level routing in later phases.
	ForceScope string `json:"force_scope,omitempty"`
	// StrictForce forbids automatic substitution when the forced model becomes unhealthy.
	StrictForce bool `json:"strict_force,omitempty"`
	// ForceIfAvailableElseFallback allows a forced-model request to gracefully fall back
	// to a healthy alternative instead of failing the request.
	ForceIfAvailableElseFallback bool `json:"force_if_available_else_fallback,omitempty"`
	// AnswerMode allows callers to control how strongly the system should ground itself in
	// retrieved evidence or tool outputs.
	AnswerMode string `json:"answer_mode,omitempty"`
	// MaxTokens sets the maximum number of tokens to generate in the response.
	MaxTokens int `json:"max_tokens,omitempty"`
	// Temperature controls the randomness of the output. Higher values (e.g., 0.8)
	// make the output more creative, while lower values (e.g., 0.2) make it more deterministic.
	// We use a pointer to distinguish between an unset value (nil) and a deliberate value of 0.
	Temperature *float32 `json:"temperature,omitempty"`
	// TopP is an alternative to temperature sampling, known as nucleus sampling.
	TopP *float32 `json:"top_p,omitempty"`
	// Stream determines whether to send back a single response or a stream of events.
	Stream bool `json:"stream,omitempty"`
}

// FailoverInfo provides details about an automatic model failover event.
type FailoverInfo struct {
	OriginalModel string `json:"original_model"`
	NewModel      string `json:"new_model"`
	Reason        string `json:"reason"`
}

type RouteHealth struct {
	ProviderStatus   string `json:"provider_status,omitempty"`
	ModelStatus      string `json:"model_status,omitempty"`
	CapabilityStatus string `json:"capability_status,omitempty"`
}

type RouteCandidate struct {
	ModelID       string      `json:"model_id"`
	Provider      string      `json:"provider"`
	Score         float64     `json:"score,omitempty"`
	EstimatedCost float64     `json:"estimated_cost,omitempty"`
	AvgLatencyMS  int64       `json:"avg_latency_ms,omitempty"`
	Health        RouteHealth `json:"health"`
	Reason        string      `json:"reason,omitempty"`
}

type ForceMetadata struct {
	IsForced       bool   `json:"is_forced"`
	Scope          string `json:"scope,omitempty"`
	Strict         bool   `json:"strict"`
	PinnedModel    string `json:"pinned_model,omitempty"`
	EffectiveModel string `json:"effective_model,omitempty"`
}

type IntentMetadata struct {
	TaskType            string  `json:"task_type,omitempty"`
	ModalityType        string  `json:"modality_type,omitempty"`
	ComplexityClass     string  `json:"complexity_class,omitempty"`
	CostPriority        int     `json:"cost_priority,omitempty"`
	LatencyPriority     int     `json:"latency_priority,omitempty"`
	QualityPriority     int     `json:"quality_priority,omitempty"`
	GroundingRequired   bool    `json:"grounding_required,omitempty"`
	ToolLikelihood      float64 `json:"tool_likelihood,omitempty"`
	AsyncLikelihood     float64 `json:"async_likelihood,omitempty"`
	SessionConstraint   string  `json:"session_constraint,omitempty"`
	ForceScopeIfAny     string  `json:"force_scope_if_any,omitempty"`
	RequestedCapability string  `json:"requested_capability,omitempty"`
	PreferenceHint      string  `json:"preference_hint,omitempty"`
}

type RouteMetadata struct {
	SelectedProvider       string           `json:"selected_provider,omitempty"`
	SelectedModel          string           `json:"selected_model,omitempty"`
	Strategy               string           `json:"strategy,omitempty"`
	PolicyName             string           `json:"policy_name,omitempty"`
	PolicySummary          string           `json:"policy_summary,omitempty"`
	RouteFamily            string           `json:"route_family,omitempty"`
	SelectionReason        string           `json:"selection_reason,omitempty"`
	UsedDegradedCandidates bool             `json:"used_degraded_candidates,omitempty"`
	HealthInputs           RouteHealth      `json:"health_inputs"`
	FallbackCandidates     []RouteCandidate `json:"fallback_candidates,omitempty"`
	FallbackGraph          []string         `json:"fallback_graph,omitempty"`
	FilteredCandidates     []RouteCandidate `json:"filtered_candidates,omitempty"`
	OrchestrationID        string           `json:"orchestration_id,omitempty"`
	ForcedSemantics        ForceMetadata    `json:"forced_semantics"`
}

type SessionMetadata struct {
	ConversationID string `json:"conversation_id,omitempty"`
	Mode           string `json:"mode,omitempty"`
	PinnedModel    string `json:"pinned_model,omitempty"`
	EffectiveModel string `json:"effective_model,omitempty"`
	IsForced       bool   `json:"is_forced"`
	ForceScope     string `json:"force_scope,omitempty"`
	StrictForce    bool   `json:"strict_force,omitempty"`
	OverrideCount  int    `json:"override_count,omitempty"`
	FailoverCount  int    `json:"failover_count,omitempty"`
	LastOverrideAt string `json:"last_override_at,omitempty"`
	LastFailoverAt string `json:"last_failover_at,omitempty"`
}

type RetrievalSource struct {
	Source      string  `json:"source,omitempty"`
	DocumentID  string  `json:"document_id,omitempty"`
	DocTitle    string  `json:"doc_title,omitempty"`
	Section     string  `json:"section,omitempty"`
	SectionPath string  `json:"section_path,omitempty"`
	Version     string  `json:"version,omitempty"`
	UpdatedAt   string  `json:"updated_at,omitempty"`
	IngestedAt  string  `json:"ingested_at,omitempty"`
	ChunkIndex  int     `json:"chunk_index,omitempty"`
	Score       float64 `json:"score,omitempty"`
	RerankScore float64 `json:"rerank_score,omitempty"`
	Freshness   float64 `json:"freshness,omitempty"`
}

type RetrievalMetadata struct {
	RetrievalID        string            `json:"retrieval_id,omitempty"`
	Score              float64           `json:"score,omitempty"`
	CandidateCount     int               `json:"candidate_count,omitempty"`
	SelectedCount      int               `json:"selected_count,omitempty"`
	SourceDiversity    int               `json:"source_diversity,omitempty"`
	BudgetUsedChars    int               `json:"budget_used_chars,omitempty"`
	RetrievalLatencyMS int64             `json:"retrieval_latency_ms,omitempty"`
	StaleSources       []string          `json:"stale_sources,omitempty"`
	Sources            []RetrievalSource `json:"sources,omitempty"`
}

type GroundingMetadata struct {
	AnswerMode       string `json:"answer_mode,omitempty"`
	RequiresEvidence bool   `json:"requires_evidence,omitempty"`
	EvidenceStatus   string `json:"evidence_status,omitempty"`
	Abstained        bool   `json:"abstained,omitempty"`
	Caution          string `json:"caution,omitempty"`
}

type MemoryMetadata struct {
	ConversationID      string   `json:"conversation_id,omitempty"`
	ShortTermEventCount int      `json:"short_term_event_count,omitempty"`
	Summary             string   `json:"summary,omitempty"`
	KnowledgeSources    []string `json:"knowledge_sources,omitempty"`
	ActiveConstraints   []string `json:"active_constraints,omitempty"`
}

type ContextMetadata struct {
	PromptChars      int      `json:"prompt_chars,omitempty"`
	BudgetChars      int      `json:"budget_chars,omitempty"`
	IncludedSections []string `json:"included_sections,omitempty"`
	OmittedSections  []string `json:"omitted_sections,omitempty"`
}

type ToolPlanMetadata struct {
	NeedTools            bool     `json:"need_tools,omitempty"`
	SelectedTools        []string `json:"selected_tools,omitempty"`
	Reason               string   `json:"reason,omitempty"`
	RetrievalBeforeTools bool     `json:"retrieval_before_tools,omitempty"`
	ToolBeforeReasoning  bool     `json:"tool_before_reasoning,omitempty"`
	UseMultiTool         bool     `json:"use_multi_tool,omitempty"`
}

type ArtifactMetadata struct {
	ArtifactID     string            `json:"artifact_id,omitempty"`
	Name           string            `json:"name,omitempty"`
	Type           string            `json:"type,omitempty"`
	MimeType       string            `json:"mime_type,omitempty"`
	SourceURI      string            `json:"source_uri,omitempty"`
	AccessURL      string            `json:"access_url,omitempty"`
	Version        string            `json:"version,omitempty"`
	SizeBytes      int64             `json:"size_bytes,omitempty"`
	Role           string            `json:"role,omitempty"`
	DerivedFrom    string            `json:"derived_from,omitempty"`
	Lineage        []string          `json:"lineage,omitempty"`
	GeneratorModel string            `json:"generator_model,omitempty"`
	PromptSummary  string            `json:"prompt_summary,omitempty"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

type VideoHookMetadata struct {
	StoryboardArtifactID string `json:"storyboard_artifact_id,omitempty"`
	BundleArtifactID     string `json:"bundle_artifact_id,omitempty"`
	SceneCount           int    `json:"scene_count,omitempty"`
	Status               string `json:"status,omitempty"`
}

type GenerationMetadata struct {
	Pipeline          string             `json:"pipeline,omitempty"`
	PromptStrategy    string             `json:"prompt_strategy,omitempty"`
	RefinedPrompt     string             `json:"refined_prompt,omitempty"`
	PolicyStatus      string             `json:"policy_status,omitempty"`
	QualityStatus     string             `json:"quality_status,omitempty"`
	Voice             string             `json:"voice,omitempty"`
	Attempts          int                `json:"attempts,omitempty"`
	Warnings          []string           `json:"warnings,omitempty"`
	StageModelMap     map[string]string  `json:"stage_model_map,omitempty"`
	OutputArtifactIDs []string           `json:"output_artifact_ids,omitempty"`
	VideoHook         *VideoHookMetadata `json:"video_hook,omitempty"`
}

type ExecutionStageMetadata struct {
	StageID          string   `json:"stage_id,omitempty"`
	StageType        string   `json:"stage_type,omitempty"`
	Title            string   `json:"title,omitempty"`
	Capability       string   `json:"capability,omitempty"`
	ModelBinding     string   `json:"model_binding,omitempty"`
	DependsOn        []string `json:"depends_on,omitempty"`
	ForcePolicy      string   `json:"force_policy,omitempty"`
	BindingViolation string   `json:"binding_violation,omitempty"`
	Status           string   `json:"status,omitempty"`
	Optional         bool     `json:"optional,omitempty"`
	ForceApplied     bool     `json:"force_applied,omitempty"`
	Strict           bool     `json:"strict,omitempty"`
}

type ExecutionPlanMetadata struct {
	PlanID               string                   `json:"plan_id,omitempty"`
	PlanType             string                   `json:"plan_type,omitempty"`
	SyncMode             string                   `json:"sync_mode,omitempty"`
	CostTier             string                   `json:"cost_tier,omitempty"`
	LatencyTier          string                   `json:"latency_tier,omitempty"`
	PrimaryStageID       string                   `json:"primary_stage_id,omitempty"`
	PrimaryCapability    string                   `json:"primary_capability,omitempty"`
	ForceScope           string                   `json:"force_scope,omitempty"`
	RequiresAsync        bool                     `json:"requires_async,omitempty"`
	Modalities           []string                 `json:"modalities,omitempty"`
	RequiredCapabilities []string                 `json:"required_capabilities,omitempty"`
	Notes                []string                 `json:"notes,omitempty"`
	Stages               []ExecutionStageMetadata `json:"stages,omitempty"`
}

type AsyncMetadata struct {
	Accepted    bool   `json:"accepted,omitempty"`
	JobID       string `json:"job_id,omitempty"`
	State       string `json:"state,omitempty"`
	StatusURL   string `json:"status_url,omitempty"`
	ResultURL   string `json:"result_url,omitempty"`
	CancelURL   string `json:"cancel_url,omitempty"`
	PollAfterMS int    `json:"poll_after_ms,omitempty"`
}

type JobCheckpointMetadata struct {
	StageID      string            `json:"stage_id,omitempty"`
	StageType    string            `json:"stage_type,omitempty"`
	WorkerClass  string            `json:"worker_class,omitempty"`
	Status       string            `json:"status,omitempty"`
	Attempts     int               `json:"attempts,omitempty"`
	StartedAt    string            `json:"started_at,omitempty"`
	CompletedAt  string            `json:"completed_at,omitempty"`
	FailureClass string            `json:"failure_class,omitempty"`
	Error        string            `json:"error,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

type JobAcceptedResponse struct {
	JobID         string                 `json:"job_id"`
	State         string                 `json:"state"`
	AcceptedAt    string                 `json:"accepted_at,omitempty"`
	ExecutionPlan *ExecutionPlanMetadata `json:"execution_plan,omitempty"`
	Async         *AsyncMetadata         `json:"async,omitempty"`
}

type JobStatusResponse struct {
	JobID            string                  `json:"job_id"`
	State            string                  `json:"state"`
	WorkerClass      string                  `json:"worker_class,omitempty"`
	FailureClass     string                  `json:"failure_class,omitempty"`
	Error            string                  `json:"error,omitempty"`
	RetryCount       int                     `json:"retry_count,omitempty"`
	DeadLettered     bool                    `json:"dead_lettered,omitempty"`
	ResultAvailable  bool                    `json:"result_available,omitempty"`
	CreatedAt        string                  `json:"created_at,omitempty"`
	StartedAt        string                  `json:"started_at,omitempty"`
	CompletedAt      string                  `json:"completed_at,omitempty"`
	UpdatedAt        string                  `json:"updated_at,omitempty"`
	ExecutionPlan    *ExecutionPlanMetadata  `json:"execution_plan,omitempty"`
	Checkpoints      []JobCheckpointMetadata `json:"checkpoints,omitempty"`
	PartialResponse  *GenerationResponse     `json:"partial_response,omitempty"`
	GeneratedTraceID string                  `json:"generated_trace_id,omitempty"`
	Async            *AsyncMetadata          `json:"async,omitempty"`
}

type RolloutMetadata struct {
	Mode              string   `json:"mode,omitempty"`
	AppliedModel      string   `json:"applied_model,omitempty"`
	ShadowModel       string   `json:"shadow_model,omitempty"`
	PrimaryLatencyMS  int64    `json:"primary_latency_ms,omitempty"`
	ShadowLatencyMS   int64    `json:"shadow_latency_ms,omitempty"`
	SimilarityScore   float64  `json:"similarity_score,omitempty"`
	OutputDelta       string   `json:"output_delta,omitempty"`
	ReplaySourceID    string   `json:"replay_source_id,omitempty"`
	ComparisonTraceID string   `json:"comparison_trace_id,omitempty"`
	Warnings          []string `json:"warnings,omitempty"`
}

type BaselineComparison struct {
	Name      string  `json:"name,omitempty"`
	Delta     float64 `json:"delta,omitempty"`
	Direction string  `json:"direction,omitempty"`
	Summary   string  `json:"summary,omitempty"`
}

type EvaluationMetadata struct {
	Suites              []string             `json:"suites,omitempty"`
	OverallScore        float64              `json:"overall_score,omitempty"`
	DomainScores        map[string]float64   `json:"domain_scores,omitempty"`
	BaselineComparisons []BaselineComparison `json:"baseline_comparisons,omitempty"`
	Notes               []string             `json:"notes,omitempty"`
}

type EvaluationRunRequest struct {
	Request         GenerationRequest `json:"request"`
	OrchestrationID string            `json:"orchestration_id,omitempty"`
}

type GovernanceMetadata struct {
	TenantID            string   `json:"tenant_id,omitempty"`
	WorkspaceID         string   `json:"workspace_id,omitempty"`
	PolicyBundle        string   `json:"policy_bundle,omitempty"`
	Status              string   `json:"status,omitempty"`
	AppliedRules        []string `json:"applied_rules,omitempty"`
	Warnings            []string `json:"warnings,omitempty"`
	AllowedModels       []string `json:"allowed_models,omitempty"`
	AllowedCapabilities []string `json:"allowed_capabilities,omitempty"`
}

type SecurityMetadata struct {
	AuthenticationMode string `json:"authentication_mode,omitempty"`
	PrincipalID        string `json:"principal_id,omitempty"`
	PrincipalRole      string `json:"principal_role,omitempty"`
	TenantID           string `json:"tenant_id,omitempty"`
	WorkspaceID        string `json:"workspace_id,omitempty"`
	AuditEventID       string `json:"audit_event_id,omitempty"`
	SignedArtifactMode string `json:"signed_artifact_mode,omitempty"`
}

// GenerationResponse defines the successful response structure sent back to the client.
// It includes the LLM's content plus rich metadata about the generation process.
type GenerationResponse struct {
	// Content is the text generated by the LLM.
	Content string `json:"content,omitempty"`

	ImageURL string `json:"image_url,omitempty"`
	AudioURL string `json:"audio_url,omitempty"`
	// ModelUsed is the ID of the model that was ultimately selected by the router to process the request.
	ModelUsed string `json:"model_used"`
	// Usage provides token metrics for the entire request, including all tool-use rounds.
	Usage Usage `json:"usage"`
	// LatencyMS is the total end-to-end processing time for the request in milliseconds.
	LatencyMS int64 `json:"latency_ms"`
	// RAGContextUsed indicates whether context from the RAG system was used to augment the prompt.
	RAGContextUsed bool `json:"rag_context_used"`
	// ToolCalls provides a log of any tools that were executed by the agent during the request.
	ToolCalls []ExecutedToolCall `json:"tool_calls,omitempty"`
	// CacheStatus indicates whether the response was served from the cache ("HIT") or generated live ("MISS").
	CacheStatus string `json:"cache_status"`
	// --- ADD THIS LINE ---
	// FailoverInfo will be populated if a session failover occurred during the request.
	FailoverInfo *FailoverInfo `json:"failover_info,omitempty"`
	// Route captures the inspectable routing metadata for the request.
	Route *RouteMetadata `json:"route,omitempty"`
	// Intent captures the structured Phase 1 intent profile that informed route planning.
	Intent *IntentMetadata `json:"intent,omitempty"`
	// Session captures the effective session semantics after request processing.
	Session *SessionMetadata `json:"session,omitempty"`
	// Retrieval captures provenance-aware Phase 2 retrieval metadata.
	Retrieval *RetrievalMetadata `json:"retrieval,omitempty"`
	// Grounding captures the answer-mode and evidence sufficiency decision.
	Grounding *GroundingMetadata `json:"grounding,omitempty"`
	// Memory captures the session-memory snapshot that influenced prompt composition.
	Memory *MemoryMetadata `json:"memory,omitempty"`
	// Context describes how the final prompt was assembled and budgeted.
	Context *ContextMetadata `json:"context,omitempty"`
	// ToolPlan captures the planner's decision before any tool execution.
	ToolPlan *ToolPlanMetadata `json:"tool_plan,omitempty"`
	// Artifacts captures the registered multimodal artifacts that shaped the request.
	Artifacts []ArtifactMetadata `json:"artifacts,omitempty"`
	// GeneratedArtifacts captures newly created artifacts that can be reused in future workflows.
	GeneratedArtifacts []ArtifactMetadata `json:"generated_artifacts,omitempty"`
	// ExecutionPlan captures the stage-level multimodal execution plan for the request.
	ExecutionPlan *ExecutionPlanMetadata `json:"execution_plan,omitempty"`
	// Generation captures Phase 4 creative pipeline metadata.
	Generation *GenerationMetadata `json:"generation,omitempty"`
	// Async captures Phase 5 job metadata when a request was accepted for background execution.
	Async *AsyncMetadata `json:"async,omitempty"`
	// Rollout captures shadow/canary/replay safety metadata.
	Rollout *RolloutMetadata `json:"rollout,omitempty"`
	// Evaluation captures Phase 5 benchmark and scoring output.
	Evaluation *EvaluationMetadata `json:"evaluation,omitempty"`
	// Governance captures Phase 6 tenant/workspace policy enforcement.
	Governance *GovernanceMetadata `json:"governance,omitempty"`
	// Security captures Phase 6 auth, audit, and signed-access details.
	Security *SecurityMetadata `json:"security,omitempty"`
}

// ExecutedToolCall provides a transparent record of a tool that was executed by the agent.
type ExecutedToolCall struct {
	Name       string `json:"name"`
	Args       string `json:"args"`
	Result     string `json:"result"`
	Status     string `json:"status,omitempty"`
	DurationMS int64  `json:"duration_ms,omitempty"`
	Attempts   int    `json:"attempts,omitempty"`
}

// Usage mirrors the token usage structure from providers like OpenAI and Anthropic.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Add accumulates the token usage from another Usage struct into this one.
// This is essential for correctly tracking total tokens in multi-step agentic chains.
func (u *Usage) Add(other Usage) {
	u.PromptTokens += other.PromptTokens
	u.CompletionTokens += other.CompletionTokens
	u.TotalTokens += other.TotalTokens
}
