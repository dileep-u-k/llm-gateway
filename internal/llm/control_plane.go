package llm

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

type ExecutionIntent struct {
	TaskType            string     `json:"task_type"`
	ModalityType        string     `json:"modality_type"`
	ComplexityClass     string     `json:"complexity_class"`
	CostPriority        int        `json:"cost_priority"`
	LatencyPriority     int        `json:"latency_priority"`
	QualityPriority     int        `json:"quality_priority"`
	GroundingRequired   bool       `json:"grounding_required"`
	ToolLikelihood      float64    `json:"tool_likelihood"`
	AsyncLikelihood     float64    `json:"async_likelihood"`
	SessionConstraint   string     `json:"session_constraint"`
	ForceScopeIfAny     string     `json:"force_scope_if_any,omitempty"`
	RequestedCapability Capability `json:"requested_capability"`
	PreferenceHint      string     `json:"preference_hint,omitempty"`
}

type RoutePlanningRequest struct {
	Prompt          string
	History         []Message
	ConversationID  string
	Preference      string
	AvailableModels []string
	ModelBudgets    map[string]float64
	PromptTokens    int
	Capability      Capability
	ForceMetadata   ForceMetadata
}

type ProviderRecord struct {
	Name           string       `json:"name"`
	Endpoint       string       `json:"endpoint,omitempty"`
	RegionSupport  []string     `json:"region_support,omitempty"`
	AuthConfigured bool         `json:"auth_configured"`
	CoarseHealth   HealthStatus `json:"coarse_health"`
}

type ModelRecord struct {
	ModelID                 string       `json:"model_id"`
	Provider                string       `json:"provider"`
	Modalities              []string     `json:"modalities,omitempty"`
	Capabilities            []Capability `json:"capabilities,omitempty"`
	CostTier                string       `json:"cost_tier,omitempty"`
	LatencyTier             string       `json:"latency_tier,omitempty"`
	QualityTier             string       `json:"quality_tier,omitempty"`
	ContextLimit            int          `json:"context_limit,omitempty"`
	StreamingSupported      bool         `json:"streaming_supported"`
	StructuredOutputSupport bool         `json:"structured_output_supported"`
	ToolSupport             bool         `json:"tool_supported"`
	Enabled                 bool         `json:"enabled"`
	AccessConfigured        bool         `json:"access_configured"`
}

type CapabilityRecord struct {
	Capability  Capability `json:"capability"`
	Description string     `json:"description,omitempty"`
	Modalities  []string   `json:"modalities,omitempty"`
	AsyncLikely bool       `json:"async_likely"`
	Models      []string   `json:"models,omitempty"`
}

type ProviderRegistry struct {
	providers map[string]ProviderRecord
}

func (r *ProviderRegistry) Get(name string) (ProviderRecord, bool) {
	if r == nil {
		return ProviderRecord{}, false
	}
	record, ok := r.providers[name]
	return record, ok
}

func (r *ProviderRegistry) List() []ProviderRecord {
	if r == nil {
		return nil
	}
	records := make([]ProviderRecord, 0, len(r.providers))
	for _, record := range r.providers {
		records = append(records, record)
	}
	sort.Slice(records, func(i, j int) bool { return records[i].Name < records[j].Name })
	return records
}

type ModelRegistry struct {
	models map[string]ModelRecord
}

func (r *ModelRegistry) Get(modelID string) (ModelRecord, bool) {
	if r == nil {
		return ModelRecord{}, false
	}
	record, ok := r.models[modelID]
	return record, ok
}

func (r *ModelRegistry) SupportsCapability(modelID string, capability Capability) bool {
	record, ok := r.Get(modelID)
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

func (r *ModelRegistry) List() []ModelRecord {
	if r == nil {
		return nil
	}
	records := make([]ModelRecord, 0, len(r.models))
	for _, record := range r.models {
		records = append(records, record)
	}
	sort.Slice(records, func(i, j int) bool { return records[i].ModelID < records[j].ModelID })
	return records
}

type CapabilityRegistry struct {
	capabilities map[Capability]CapabilityRecord
}

func (r *CapabilityRegistry) Get(capability Capability) (CapabilityRecord, bool) {
	if r == nil {
		return CapabilityRecord{}, false
	}
	record, ok := r.capabilities[capability]
	return record, ok
}

func (r *CapabilityRegistry) List() []CapabilityRecord {
	if r == nil {
		return nil
	}
	records := make([]CapabilityRecord, 0, len(r.capabilities))
	for _, record := range r.capabilities {
		records = append(records, record)
	}
	sort.Slice(records, func(i, j int) bool { return string(records[i].Capability) < string(records[j].Capability) })
	return records
}

type RoutePolicyDecision struct {
	Name     string
	Strategy string
	Policy   RoutingPolicy
	Reasons  []string
}

type IntentClassifier struct{}

func NewIntentClassifier() *IntentClassifier {
	return &IntentClassifier{}
}

func (c *IntentClassifier) Classify(input RoutePlanningRequest) ExecutionIntent {
	prompt := strings.ToLower(strings.TrimSpace(input.Prompt))
	taskType := "general_reasoning"
	switch {
	case input.Capability == CapabilityImageGeneration:
		taskType = "image_generation"
	case containsAny(prompt, "code", "function", "golang", "python", "bug", "refactor", "implement", "debug"):
		taskType = "coding"
	case containsAny(prompt, "weather", "forecast", "calculate", "calculator", "news", "headline"):
		taskType = "tool_augmented"
	case containsAny(prompt, "according to", "source", "cite", "document", "knowledge base", "latest", "grounded"):
		taskType = "knowledge_grounded"
	case containsAny(prompt, "summarize", "analyze", "extract", "classify", "compare"):
		taskType = "analysis"
	}

	modalityType := "text"
	switch input.Capability {
	case CapabilityImageGeneration:
		modalityType = "image_generation"
	case CapabilityEmbeddings:
		modalityType = "embedding"
	default:
		modalityType = "text"
	}

	complexityClass := "simple"
	totalChars := len(input.Prompt)
	for _, msg := range input.History {
		totalChars += len(msg.Content)
	}
	if totalChars > 1200 || len(input.History) > 8 || containsAny(prompt, "step by step", "architecture", "planner", "multi-stage", "deep dive") {
		complexityClass = "high"
	} else if totalChars > 300 || len(input.History) > 2 || containsAny(prompt, "compare", "tradeoff", "design", "strategy") {
		complexityClass = "medium"
	}

	costPriority, latencyPriority, qualityPriority := resolveIntentPriorities(input.Preference, taskType, complexityClass)
	groundingRequired := taskType == "knowledge_grounded" || containsAny(prompt, "source", "cite", "reference", "latest", "accurate")
	toolLikelihood := scoreLikelihood(prompt, []string{"weather", "forecast", "calculate", "calculator", "news", "headline", "tool"}, 0.05, 0.85)
	asyncLikelihood := scoreLikelihood(prompt, []string{"batch", "large file", "thousands", "long running", "async", "queue"}, 0.05, 0.8)

	sessionConstraint := "stateless"
	if input.ConversationID != "" {
		sessionConstraint = "conversation_continuity"
	}
	if input.ForceMetadata.IsForced {
		sessionConstraint = "forced_session_continuity"
	}

	return ExecutionIntent{
		TaskType:            taskType,
		ModalityType:        modalityType,
		ComplexityClass:     complexityClass,
		CostPriority:        costPriority,
		LatencyPriority:     latencyPriority,
		QualityPriority:     qualityPriority,
		GroundingRequired:   groundingRequired,
		ToolLikelihood:      toolLikelihood,
		AsyncLikelihood:     asyncLikelihood,
		SessionConstraint:   sessionConstraint,
		ForceScopeIfAny:     input.ForceMetadata.Scope,
		RequestedCapability: input.Capability,
		PreferenceHint:      input.Preference,
	}
}

type RoutingPolicyEngine struct {
	policies map[string]RoutingPolicy
}

func NewRoutingPolicyEngine(config *RouterConfig) *RoutingPolicyEngine {
	policies := make(map[string]RoutingPolicy)
	for name, policy := range config.RoutePolicies {
		policies[name] = policy
	}
	if len(policies) == 0 {
		for name := range config.Strategies {
			policies[name] = RoutingPolicy{
				Strategy:              name,
				AllowDegradedFallback: true,
				Description:           fmt.Sprintf("Auto-generated policy mirroring strategy %q.", name),
			}
		}
	}
	if _, ok := policies["default"]; !ok {
		policies["default"] = RoutingPolicy{
			Strategy:              "default",
			AllowDegradedFallback: true,
			Description:           "Default quality-leaning text routing policy.",
		}
	}
	if _, ok := policies["force_strict"]; !ok {
		policies["force_strict"] = RoutingPolicy{
			Strategy:               "default",
			AvoidDegradedProviders: true,
			AllowDegradedFallback:  false,
			Description:            "Strict forced routing that preserves the requested route family whenever possible.",
		}
	}
	return &RoutingPolicyEngine{policies: policies}
}

func (e *RoutingPolicyEngine) Resolve(intent ExecutionIntent, requestedPreference string, forceMeta ForceMetadata) RoutePolicyDecision {
	var reasons []string
	name := strings.TrimSpace(requestedPreference)

	if forceMeta.IsForced && forceMeta.Strict {
		if _, ok := e.policies["force_strict"]; ok {
			name = "force_strict"
			reasons = append(reasons, "strict force semantics requested")
		}
	}

	if name == "" {
		switch {
		case intent.TaskType == "coding":
			name = "best-for-coding"
			reasons = append(reasons, "coding task detected")
		case intent.CostPriority >= 5:
			name = "cost"
			reasons = append(reasons, "cost-first intent detected")
		case intent.LatencyPriority >= 5:
			name = "latency"
			reasons = append(reasons, "latency-first intent detected")
		case intent.QualityPriority >= 5:
			name = "max_quality"
			reasons = append(reasons, "quality-first intent detected")
		default:
			name = "default"
			reasons = append(reasons, "default policy selected")
		}
	}

	policy, ok := e.policies[name]
	if !ok {
		reasons = append(reasons, fmt.Sprintf("policy %q not configured, falling back to default", name))
		name = "default"
		policy = e.policies[name]
	}
	strategy := policy.Strategy
	if strategy == "" {
		if name == "force_strict" {
			strategy = "default"
		} else {
			strategy = name
		}
	}

	return RoutePolicyDecision{
		Name:     name,
		Strategy: strategy,
		Policy:   policy,
		Reasons:  reasons,
	}
}

type FallbackGraphStore struct {
	graphs map[string]map[string][]string
}

func NewFallbackGraphStore(config *RouterConfig) *FallbackGraphStore {
	return &FallbackGraphStore{graphs: config.FallbackGraphs}
}

func (s *FallbackGraphStore) Chain(capability Capability, policyName string) []string {
	if s == nil || len(s.graphs) == 0 {
		return nil
	}
	byCapability, ok := s.graphs[string(capability)]
	if !ok {
		return nil
	}
	if chain, ok := byCapability[policyName]; ok {
		return append([]string(nil), chain...)
	}
	if chain, ok := byCapability["default"]; ok {
		return append([]string(nil), chain...)
	}
	return nil
}

type ControlPlane struct {
	profiler           *Profiler
	router             *Router
	intentClassifier   *IntentClassifier
	policyEngine       *RoutingPolicyEngine
	fallbackGraphs     *FallbackGraphStore
	providerRegistry   *ProviderRegistry
	modelRegistry      *ModelRegistry
	capabilityRegistry *CapabilityRegistry
}

func NewControlPlane(config *RouterConfig, profiler *Profiler, router *Router, enabledModels []string, apiKeys map[string]string) *ControlPlane {
	cp := &ControlPlane{
		profiler:         profiler,
		router:           router,
		intentClassifier: NewIntentClassifier(),
		policyEngine:     NewRoutingPolicyEngine(config),
		fallbackGraphs:   NewFallbackGraphStore(config),
	}
	cp.providerRegistry = cp.buildProviderRegistry(config, enabledModels, apiKeys)
	cp.modelRegistry = cp.buildModelRegistry(config, enabledModels, apiKeys)
	cp.capabilityRegistry = cp.buildCapabilityRegistry(config, enabledModels)
	return cp
}

func (cp *ControlPlane) PlanRoute(ctx context.Context, input RoutePlanningRequest) (*RouteSelection, error) {
	intent := cp.intentClassifier.Classify(input)
	decision := cp.policyEngine.Resolve(intent, input.Preference, input.ForceMetadata)
	filteredModels := cp.filterModels(ctx, input.AvailableModels, input.Capability, decision.Policy, intent)
	selection, err := cp.router.SelectOptimalRoute(ctx, filteredModels, decision.Strategy, input.PromptTokens, input.ModelBudgets, input.ForceMetadata)
	if err != nil {
		return nil, err
	}
	cp.decorateSelection(selection, intent, input.Capability, decision)
	return selection, nil
}

func (cp *ControlPlane) ExplainForcedRoute(ctx context.Context, selectedModel string, input RoutePlanningRequest) (*RouteSelection, error) {
	intent := cp.intentClassifier.Classify(input)
	decision := cp.policyEngine.Resolve(intent, input.Preference, input.ForceMetadata)
	filteredModels := cp.filterModels(ctx, input.AvailableModels, input.Capability, decision.Policy, intent)
	selection, err := cp.router.ExplainForcedRoute(ctx, selectedModel, filteredModels, decision.Strategy, input.PromptTokens, input.ModelBudgets, input.ForceMetadata)
	if err != nil {
		return nil, err
	}
	cp.decorateSelection(selection, intent, input.Capability, decision)
	return selection, nil
}

func (cp *ControlPlane) Providers() *ProviderRegistry {
	return cp.providerRegistry
}

func (cp *ControlPlane) Models() *ModelRegistry {
	return cp.modelRegistry
}

func (cp *ControlPlane) Capabilities() *CapabilityRegistry {
	return cp.capabilityRegistry
}

func (cp *ControlPlane) buildProviderRegistry(config *RouterConfig, enabledModels []string, apiKeys map[string]string) *ProviderRegistry {
	records := make(map[string]ProviderRecord)
	for name, meta := range config.Providers {
		records[name] = ProviderRecord{
			Name:          name,
			Endpoint:      meta.Endpoint,
			RegionSupport: append([]string(nil), meta.Regions...),
			CoarseHealth:  HealthStatusOnline,
		}
	}
	for _, modelID := range enabledModels {
		provider := ProviderForModel(modelID)
		record := records[provider]
		record.Name = provider
		record.AuthConfigured = record.AuthConfigured || apiKeys[modelID] != ""
		if record.CoarseHealth == "" {
			record.CoarseHealth = HealthStatusOnline
		}
		records[provider] = record
	}
	return &ProviderRegistry{providers: records}
}

func (cp *ControlPlane) buildModelRegistry(config *RouterConfig, enabledModels []string, apiKeys map[string]string) *ModelRegistry {
	enabled := make(map[string]struct{}, len(enabledModels))
	for _, modelID := range enabledModels {
		modelID = strings.TrimSpace(modelID)
		if modelID != "" {
			enabled[modelID] = struct{}{}
		}
	}

	records := make(map[string]ModelRecord)
	for modelID, meta := range config.Models {
		_, isEnabled := enabled[modelID]
		record := ModelRecord{
			ModelID:                 modelID,
			Provider:                firstNonEmpty(meta.Provider, ProviderForModel(modelID)),
			Modalities:              append([]string(nil), meta.Modalities...),
			Capabilities:            append([]Capability(nil), meta.Capabilities...),
			CostTier:                meta.CostTier,
			LatencyTier:             meta.LatencyTier,
			QualityTier:             meta.QualityTier,
			ContextLimit:            meta.ContextLimit,
			StreamingSupported:      meta.StreamingSupported,
			StructuredOutputSupport: meta.StructuredOutputSupport,
			ToolSupport:             meta.ToolSupport,
			Enabled:                 isEnabled,
			AccessConfigured:        apiKeys[modelID] != "",
		}
		if len(record.Modalities) == 0 {
			record.Modalities = defaultModalitiesForCapability(record.Capabilities, CapabilityForModel(modelID))
		}
		if len(record.Capabilities) == 0 {
			record.Capabilities = []Capability{CapabilityForModel(modelID)}
		}
		records[modelID] = record
	}

	for modelID := range enabled {
		if _, ok := records[modelID]; ok {
			continue
		}
		capability := CapabilityForModel(modelID)
		records[modelID] = ModelRecord{
			ModelID:          modelID,
			Provider:         ProviderForModel(modelID),
			Modalities:       defaultModalitiesForCapability(nil, capability),
			Capabilities:     []Capability{capability},
			Enabled:          true,
			AccessConfigured: apiKeys[modelID] != "",
		}
	}

	return &ModelRegistry{models: records}
}

func (cp *ControlPlane) buildCapabilityRegistry(config *RouterConfig, enabledModels []string) *CapabilityRegistry {
	records := map[Capability]CapabilityRecord{
		CapabilityTextGeneration:     {Capability: CapabilityTextGeneration},
		CapabilityEmbeddings:         {Capability: CapabilityEmbeddings},
		CapabilityOCR:                {Capability: CapabilityOCR},
		CapabilityImageUnderstanding: {Capability: CapabilityImageUnderstanding},
		CapabilityImageGeneration:    {Capability: CapabilityImageGeneration},
		CapabilityImageEditing:       {Capability: CapabilityImageEditing},
		CapabilityTranscription:      {Capability: CapabilityTranscription},
		CapabilityTTS:                {Capability: CapabilityTTS},
		CapabilityVideoUnderstanding: {Capability: CapabilityVideoUnderstanding},
		CapabilityVideoGeneration:    {Capability: CapabilityVideoGeneration},
	}

	for rawCapability, meta := range config.Capabilities {
		capability := Capability(rawCapability)
		record := records[capability]
		record.Capability = capability
		record.Description = meta.Description
		record.Modalities = append([]string(nil), meta.Modalities...)
		record.AsyncLikely = meta.AsyncLikely
		records[capability] = record
	}

	for _, modelID := range enabledModels {
		record, ok := cp.modelRegistry.Get(modelID)
		if !ok {
			record = ModelRecord{
				ModelID:      modelID,
				Capabilities: []Capability{CapabilityForModel(modelID)},
			}
		}
		for _, capability := range record.Capabilities {
			capRecord := records[capability]
			capRecord.Capability = capability
			capRecord.Models = append(capRecord.Models, modelID)
			if len(capRecord.Modalities) == 0 {
				capRecord.Modalities = defaultModalitiesForCapability([]Capability{capability}, capability)
			}
			records[capability] = capRecord
		}
	}
	for capability, record := range records {
		sort.Strings(record.Models)
		records[capability] = record
	}

	return &CapabilityRegistry{capabilities: records}
}

func (cp *ControlPlane) filterModels(ctx context.Context, availableModels []string, capability Capability, policy RoutingPolicy, intent ExecutionIntent) []string {
	unique := make([]string, 0, len(availableModels))
	seen := make(map[string]struct{}, len(availableModels))
	for _, modelID := range availableModels {
		modelID = strings.TrimSpace(modelID)
		if modelID == "" {
			continue
		}
		if _, ok := seen[modelID]; ok {
			continue
		}
		seen[modelID] = struct{}{}
		if !cp.modelRegistry.SupportsCapability(modelID, capability) {
			continue
		}
		record, _ := cp.modelRegistry.Get(modelID)
		if policy.RequireToolSupport && !record.ToolSupport {
			continue
		}
		if contains(policy.AvoidProviders, ProviderForModel(modelID)) {
			continue
		}
		unique = append(unique, modelID)
	}
	if len(unique) == 0 {
		unique = dedupeModels(availableModels)
	}

	ordered := unique
	if len(policy.PreferredProviders) > 0 {
		var preferred, others []string
		for _, modelID := range unique {
			if contains(policy.PreferredProviders, ProviderForModel(modelID)) {
				preferred = append(preferred, modelID)
			} else {
				others = append(others, modelID)
			}
		}
		ordered = append(preferred, others...)
	}

	if policy.AvoidDegradedProviders && cp.profiler != nil {
		var healthyProviders, degradedProviders []string
		for _, modelID := range ordered {
			health, err := cp.profiler.GetProviderHealth(ctx, ProviderForModel(modelID))
			if err == nil && health.Status == HealthStatusOnline {
				healthyProviders = append(healthyProviders, modelID)
				continue
			}
			degradedProviders = append(degradedProviders, modelID)
		}
		if len(healthyProviders) > 0 {
			return healthyProviders
		}
		if policy.AllowDegradedFallback {
			return ordered
		}
		return nil
	}

	if policy.RequireGrounding && !intent.GroundingRequired {
		return ordered
	}
	return ordered
}

func (cp *ControlPlane) decorateSelection(selection *RouteSelection, intent ExecutionIntent, capability Capability, decision RoutePolicyDecision) {
	if selection == nil {
		return
	}
	selection.Explanation.PolicyName = decision.Name
	selection.Explanation.PolicySummary = decision.Policy.Description
	selection.Explanation.RouteFamily = string(capability)
	selection.Explanation.Intent = intent
	fallbackGraph := cp.fallbackGraphs.Chain(capability, decision.Name)
	selection.Explanation.FallbackGraph = fallbackGraph
	if len(fallbackGraph) > 0 && len(selection.Explanation.FallbackCandidates) > 0 {
		selection.Explanation.FallbackCandidates = reorderFallbackCandidates(selection.ModelID, fallbackGraph, selection.Explanation.FallbackCandidates)
	}
	if len(decision.Reasons) > 0 {
		selection.Explanation.SelectionReason = selection.Explanation.SelectionReason + " (" + strings.Join(decision.Reasons, "; ") + ")"
	}
}

func resolveIntentPriorities(preference, taskType, complexity string) (costPriority, latencyPriority, qualityPriority int) {
	costPriority, latencyPriority, qualityPriority = 3, 3, 4
	switch preference {
	case "cost":
		costPriority, latencyPriority, qualityPriority = 5, 2, 2
	case "latency":
		costPriority, latencyPriority, qualityPriority = 2, 5, 2
	case "max_quality", "best-for-coding":
		costPriority, latencyPriority, qualityPriority = 1, 2, 5
	case "balanced", "default", "smart-balanced":
		costPriority, latencyPriority, qualityPriority = 3, 3, 4
	}
	if taskType == "coding" {
		qualityPriority = maxInt(qualityPriority, 5)
	}
	if complexity == "high" {
		qualityPriority = maxInt(qualityPriority, 5)
		latencyPriority = minInt(latencyPriority, 2)
	}
	return costPriority, latencyPriority, qualityPriority
}

func reorderFallbackCandidates(selectedModel string, fallbackGraph []string, candidates []RouteCandidate) []RouteCandidate {
	order := make(map[string]int, len(fallbackGraph))
	selectedIndex := -1
	for idx, modelID := range fallbackGraph {
		order[modelID] = idx
		if modelID == selectedModel {
			selectedIndex = idx
		}
	}

	reordered := append([]RouteCandidate(nil), candidates...)
	sort.SliceStable(reordered, func(i, j int) bool {
		leftOrder, leftKnown := order[reordered[i].ModelID]
		rightOrder, rightKnown := order[reordered[j].ModelID]
		if leftKnown && selectedIndex >= 0 && leftOrder <= selectedIndex {
			leftKnown = false
		}
		if rightKnown && selectedIndex >= 0 && rightOrder <= selectedIndex {
			rightKnown = false
		}
		switch {
		case leftKnown && rightKnown:
			return leftOrder < rightOrder
		case leftKnown:
			return true
		case rightKnown:
			return false
		default:
			return reordered[i].Score > reordered[j].Score
		}
	})
	return reordered
}

func defaultModalitiesForCapability(capabilities []Capability, fallback Capability) []string {
	if len(capabilities) == 0 {
		capabilities = []Capability{fallback}
	}
	modalitySet := make(map[string]struct{})
	for _, capability := range capabilities {
		switch capability {
		case CapabilityTextGeneration, CapabilityEmbeddings:
			modalitySet["text"] = struct{}{}
		case CapabilityImageUnderstanding:
			modalitySet["image"] = struct{}{}
		case CapabilityImageGeneration, CapabilityImageEditing:
			modalitySet["image"] = struct{}{}
		case CapabilityTranscription, CapabilityTTS:
			modalitySet["audio"] = struct{}{}
		case CapabilityVideoUnderstanding, CapabilityVideoGeneration:
			modalitySet["video"] = struct{}{}
		case CapabilityOCR:
			modalitySet["document"] = struct{}{}
		default:
			modalitySet["text"] = struct{}{}
		}
	}
	out := make([]string, 0, len(modalitySet))
	for modality := range modalitySet {
		out = append(out, modality)
	}
	sort.Strings(out)
	return out
}

func dedupeModels(models []string) []string {
	var out []string
	seen := make(map[string]struct{}, len(models))
	for _, modelID := range models {
		modelID = strings.TrimSpace(modelID)
		if modelID == "" {
			continue
		}
		if _, ok := seen[modelID]; ok {
			continue
		}
		seen[modelID] = struct{}{}
		out = append(out, modelID)
	}
	return out
}

func containsAny(input string, keywords ...string) bool {
	for _, keyword := range keywords {
		if strings.Contains(input, keyword) {
			return true
		}
	}
	return false
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func scoreLikelihood(prompt string, keywords []string, base, max float64) float64 {
	score := base
	for _, keyword := range keywords {
		if strings.Contains(prompt, keyword) {
			score += 0.2
		}
	}
	if score > max {
		return max
	}
	return score
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
