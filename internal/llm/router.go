package llm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"sort"
	"time"
)

type RoutingStrategy struct {
	UseCodingScore    bool    `yaml:"use_coding_score"`
	QualityWeight     float64 `yaml:"quality_weight"`
	CostWeight        float64 `yaml:"cost_weight"`
	LatencyWeight     float64 `yaml:"latency_weight"`
	ReliabilityWeight float64 `yaml:"reliability_weight"`
}

type ModelMetadata struct {
	QualityScore            float64      `yaml:"quality_score"`
	CodingScore             float64      `yaml:"coding_score"`
	Provider                string       `yaml:"provider"`
	Modalities              []string     `yaml:"modalities"`
	Capabilities            []Capability `yaml:"capabilities"`
	CostTier                string       `yaml:"cost_tier"`
	LatencyTier             string       `yaml:"latency_tier"`
	QualityTier             string       `yaml:"quality_tier"`
	ContextLimit            int          `yaml:"context_limit"`
	StreamingSupported      bool         `yaml:"streaming_supported"`
	StructuredOutputSupport bool         `yaml:"structured_output_supported"`
	ToolSupport             bool         `yaml:"tool_supported"`
}

type ProviderMetadata struct {
	Endpoint string   `yaml:"endpoint"`
	Regions  []string `yaml:"regions"`
}

type CapabilityMetadata struct {
	Description string   `yaml:"description"`
	Modalities  []string `yaml:"modalities"`
	AsyncLikely bool     `yaml:"async_likely"`
}

type RoutingPolicy struct {
	Strategy               string   `yaml:"strategy"`
	PreferredProviders     []string `yaml:"preferred_providers"`
	AvoidProviders         []string `yaml:"avoid_providers"`
	AvoidDegradedProviders bool     `yaml:"avoid_degraded_providers"`
	AllowDegradedFallback  bool     `yaml:"allow_degraded_fallback"`
	RequireToolSupport     bool     `yaml:"require_tool_support"`
	RequireGrounding       bool     `yaml:"require_grounding"`
	Description            string   `yaml:"description"`
}

type RouterConfig struct {
	Thresholds                 map[string]interface{}          `yaml:"pre_check_thresholds"`
	Models                     map[string]ModelMetadata        `yaml:"models"`
	Providers                  map[string]ProviderMetadata     `yaml:"providers"`
	Capabilities               map[string]CapabilityMetadata   `yaml:"capabilities"`
	Strategies                 map[string]RoutingStrategy      `yaml:"strategies"`
	RoutePolicies              map[string]RoutingPolicy        `yaml:"route_policies"`
	FallbackGraphs             map[string]map[string][]string  `yaml:"fallback_graphs"`
	SmartBalancedCostThreshold float64                         `yaml:"smart_balanced_cost_threshold"`
	OrchestrationMetadataTTL   string                          `yaml:"orchestration_metadata_ttl"`
	ImageModels                map[string]ImageModelMetadata   `yaml:"image_models"`
	ImageStrategies            map[string]ImageRoutingStrategy `yaml:"image_strategies"`
}

type ImageModelMetadata struct {
	QualityScore  float64 `yaml:"quality_score"`
	ArtisticScore float64 `yaml:"artistic_score"`
	SpeedScore    float64 `yaml:"speed_score"`
}

type ImageRoutingStrategy struct {
	UseArtisticScore  bool    `yaml:"use_artistic_score"`
	QualityWeight     float64 `yaml:"quality_weight"`
	SpeedWeight       float64 `yaml:"speed_weight"`
	CostWeight        float64 `yaml:"cost_weight"`
	ReliabilityWeight float64 `yaml:"reliability_weight"`
}

type Router struct {
	profiler *Profiler
	config   *RouterConfig
}

type ForceMetadata struct {
	IsForced       bool
	Scope          string
	Strict         bool
	PinnedModel    string
	EffectiveModel string
}

type RouteHealth struct {
	ProviderStatus   HealthStatus
	ModelStatus      HealthStatus
	CapabilityStatus HealthStatus
}

type RouteCandidate struct {
	ModelID       string
	Provider      string
	Score         float64
	EstimatedCost float64
	AvgLatencyMS  int64
	Health        RouteHealth
	Reason        string
}

type RouteExplanation struct {
	SelectedProvider       string
	SelectedModel          string
	Strategy               string
	PolicyName             string
	PolicySummary          string
	RouteFamily            string
	SelectionReason        string
	UsedDegradedCandidates bool
	HealthInputs           RouteHealth
	FallbackCandidates     []RouteCandidate
	FallbackGraph          []string
	FilteredCandidates     []RouteCandidate
	Intent                 ExecutionIntent
	OrchestrationID        string
	ForcedSemantics        ForceMetadata
}

type RouteSelection struct {
	ModelID       string
	Explanation   RouteExplanation
	EstimatedCost float64
}

type routeLandscape struct {
	ranked                 []RouteCandidate
	filtered               []RouteCandidate
	usedDegradedCandidates bool
	activePoolLabel        string
	strategyName           string
}

func NewRouter(profiler *Profiler, config *RouterConfig) *Router {
	if config.SmartBalancedCostThreshold <= 0 {
		config.SmartBalancedCostThreshold = 0.001
	}
	return &Router{profiler: profiler, config: config}
}

type contender struct {
	Profile          *ModelProfile
	Metadata         ModelMetadata
	EstimatedCost    float64
	EffectiveHealth  HealthStatus
	CapabilityHealth *CapabilityHealth
	ProviderHealth   *ProviderHealth
}

func (r *Router) SelectOptimalModel(
	ctx context.Context,
	availableModels []string,
	preference string,
	promptTokens int,
	modelBudgets map[string]float64,
) (string, error) {
	selection, err := r.SelectOptimalRoute(ctx, availableModels, preference, promptTokens, modelBudgets, ForceMetadata{})
	if err != nil {
		return "", err
	}
	return selection.ModelID, nil
}

func (r *Router) SelectOptimalRoute(
	ctx context.Context,
	availableModels []string,
	preference string,
	promptTokens int,
	modelBudgets map[string]float64,
	forceMeta ForceMetadata,
) (*RouteSelection, error) {
	log.Printf("--- Starting Model Selection (Preference: %q) ---", preference)
	landscape, err := r.buildRouteLandscape(ctx, availableModels, preference, promptTokens, modelBudgets)
	if err != nil {
		return nil, err
	}
	if len(landscape.ranked) == 0 {
		return nil, errors.New("no suitable model after filtering")
	}

	selected := landscape.ranked[0]
	explanation := RouteExplanation{
		SelectedProvider:       selected.Provider,
		SelectedModel:          selected.ModelID,
		Strategy:               landscape.strategyName,
		SelectionReason:        fmt.Sprintf("selected highest-scoring %s candidate using strategy %q", landscape.activePoolLabel, landscape.strategyName),
		UsedDegradedCandidates: landscape.usedDegradedCandidates,
		HealthInputs:           selected.Health,
		FilteredCandidates:     append([]RouteCandidate(nil), landscape.filtered...),
		ForcedSemantics:        forceMeta,
	}
	if len(landscape.ranked) > 1 {
		explanation.FallbackCandidates = append([]RouteCandidate(nil), landscape.ranked[1:]...)
	}

	log.Printf("🏆 Selected best model: %s (Score=%.4f)", selected.ModelID, selected.Score)
	return &RouteSelection{
		ModelID:       selected.ModelID,
		Explanation:   explanation,
		EstimatedCost: selected.EstimatedCost,
	}, nil
}

func (r *Router) ExplainForcedRoute(
	ctx context.Context,
	selectedModel string,
	availableModels []string,
	preference string,
	promptTokens int,
	modelBudgets map[string]float64,
	forceMeta ForceMetadata,
) (*RouteSelection, error) {
	landscape, err := r.buildRouteLandscape(ctx, availableModels, preference, promptTokens, modelBudgets)
	if err != nil {
		return nil, err
	}
	for _, candidate := range landscape.ranked {
		if candidate.ModelID != selectedModel {
			continue
		}
		explanation := RouteExplanation{
			SelectedProvider:       candidate.Provider,
			SelectedModel:          candidate.ModelID,
			Strategy:               landscape.strategyName,
			SelectionReason:        fmt.Sprintf("selected via forced-session semantics (%s)", forceMeta.Scope),
			UsedDegradedCandidates: landscape.usedDegradedCandidates,
			HealthInputs:           candidate.Health,
			FilteredCandidates:     append([]RouteCandidate(nil), landscape.filtered...),
			ForcedSemantics:        forceMeta,
		}
		for _, fallback := range landscape.ranked {
			if fallback.ModelID == selectedModel {
				continue
			}
			explanation.FallbackCandidates = append(explanation.FallbackCandidates, fallback)
		}
		return &RouteSelection{
			ModelID:       selectedModel,
			Explanation:   explanation,
			EstimatedCost: candidate.EstimatedCost,
		}, nil
	}
	return nil, fmt.Errorf("forced model %q was not available in the route landscape", selectedModel)
}

func (r *Router) buildRouteLandscape(
	ctx context.Context,
	availableModels []string,
	preference string,
	promptTokens int,
	modelBudgets map[string]float64,
) (*routeLandscape, error) {
	onlineContenders := make(map[string]contender)
	degradedContenders := make(map[string]contender)
	var filtered []RouteCandidate

	for _, rawModelID := range availableModels {
		modelID := rawModelID
		if modelID == "" {
			continue
		}
		profile, providerHealth, capabilityHealth, effectiveHealth, estimatedCost, meta, reason, err := r.buildContender(ctx, modelID, promptTokens, modelBudgets[modelID])
		if err != nil {
			log.Printf("⚠️ Skipping model %s: profiler error: %v", modelID, err)
			filtered = append(filtered, RouteCandidate{
				ModelID:  modelID,
				Provider: ProviderForModel(modelID),
				Reason:   err.Error(),
			})
			continue
		}
		candidate := contender{
			Profile:          profile,
			Metadata:         meta,
			EstimatedCost:    estimatedCost,
			EffectiveHealth:  effectiveHealth,
			CapabilityHealth: capabilityHealth,
			ProviderHealth:   providerHealth,
		}
		if reason != "" {
			log.Printf("⚠️ Filtering model %s: %s", modelID, reason)
			filtered = append(filtered, r.describeContender(candidate, 0, reason))
			continue
		}

		switch effectiveHealth {
		case HealthStatusOnline:
			onlineContenders[modelID] = candidate
		case HealthStatusDegraded:
			degradedContenders[modelID] = candidate
		default:
			filtered = append(filtered, r.describeContender(candidate, 0, "offline"))
		}
	}

	contenders := onlineContenders
	activePoolLabel := "online"
	usedDegraded := false
	if len(contenders) == 0 {
		contenders = degradedContenders
		activePoolLabel = "degraded"
		usedDegraded = len(degradedContenders) > 0
	}
	if len(contenders) == 0 {
		return nil, errors.New("no suitable model after filtering")
	}

	strategyName, strategy, err := r.getStrategy(preference, contenders)
	if err != nil {
		return nil, err
	}

	minCost, maxCost, minLatency, maxLatency := getNormalizationBounds(contenders)
	modelIDs := make([]string, 0, len(contenders))
	for modelID := range contenders {
		modelIDs = append(modelIDs, modelID)
	}
	sort.Strings(modelIDs)

	ranked := make([]RouteCandidate, 0, len(modelIDs))
	for _, modelID := range modelIDs {
		c := contenders[modelID]
		score := r.calculateNormalizedScore(c, strategy, minCost, maxCost, minLatency, maxLatency)
		log.Printf("- Score[%s]: health=%s latency=%dms cost=%.6f quality=%.2f final=%.4f",
			modelID, c.EffectiveHealth, c.Profile.AvgLatencyMS, c.EstimatedCost, c.Metadata.QualityScore, score)
		ranked = append(ranked, r.describeContender(c, score, ""))
	}

	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].Score == ranked[j].Score {
			return ranked[i].ModelID < ranked[j].ModelID
		}
		return ranked[i].Score > ranked[j].Score
	})
	sort.SliceStable(filtered, func(i, j int) bool {
		return filtered[i].ModelID < filtered[j].ModelID
	})

	return &routeLandscape{
		ranked:                 ranked,
		filtered:               filtered,
		usedDegradedCandidates: usedDegraded,
		activePoolLabel:        activePoolLabel,
		strategyName:           strategyName,
	}, nil
}

func (r *Router) buildContender(
	ctx context.Context,
	modelID string,
	promptTokens int,
	monthlyBudget float64,
) (*ModelProfile, *ProviderHealth, *CapabilityHealth, HealthStatus, float64, ModelMetadata, string, error) {
	provider := ProviderForModel(modelID)
	effectiveHealth, providerHealth, capabilityHealth, profile, err := r.profiler.CombinedModelHealth(ctx, modelID, CapabilityTextGeneration)
	if err != nil {
		return nil, nil, nil, HealthStatusOffline, 0, ModelMetadata{}, "", err
	}
	if ok, reason := r.passesPreChecks(profile, providerHealth, capabilityHealth, effectiveHealth, monthlyBudget); !ok {
		return profile, providerHealth, capabilityHealth, effectiveHealth, 0, ModelMetadata{}, reason, nil
	}
	modelMeta, ok := r.config.Models[modelID]
	if !ok {
		return profile, providerHealth, capabilityHealth, effectiveHealth, 0, ModelMetadata{}, "metadata not found in config", nil
	}
	estimatedOutputTokens := promptTokens * 2
	estimatedCost := (float64(promptTokens) * profile.CostPerInputToken) +
		(float64(estimatedOutputTokens) * profile.CostPerOutputToken)
	if provider == "unknown" {
		return profile, providerHealth, capabilityHealth, effectiveHealth, estimatedCost, modelMeta, "unknown provider", nil
	}
	return profile, providerHealth, capabilityHealth, effectiveHealth, estimatedCost, modelMeta, "", nil
}

func (r *Router) getStrategy(preference string, contenders map[string]contender) (string, RoutingStrategy, error) {
	if preference == "smart-balanced" {
		avgCost := 0.0
		for _, c := range contenders {
			avgCost += c.EstimatedCost
		}
		if len(contenders) > 0 {
			avgCost /= float64(len(contenders))
		}

		if avgCost < r.config.SmartBalancedCostThreshold {
			log.Printf("Smart-balanced: avg cost %.6f < threshold %.6f → latency-priority",
				avgCost, r.config.SmartBalancedCostThreshold)
			return "latency-focused-balanced", r.config.Strategies["latency-focused-balanced"], nil
		}
		log.Printf("Smart-balanced: avg cost %.6f ≥ threshold %.6f → quality-priority",
			avgCost, r.config.SmartBalancedCostThreshold)
		return "quality-focused-balanced", r.config.Strategies["quality-focused-balanced"], nil
	}

	strategy, ok := r.config.Strategies[preference]
	if !ok {
		log.Printf("⚠️ Strategy %q not found, falling back to 'default'", preference)
		strategy, ok = r.config.Strategies["default"]
		if !ok {
			return "", RoutingStrategy{}, errors.New("no 'default' strategy configured")
		}
		return "default", strategy, nil
	}
	return preference, strategy, nil
}

func (r *Router) calculateNormalizedScore(
	c contender,
	strategy RoutingStrategy,
	minCost, maxCost, minLatency, maxLatency float64,
) float64 {
	latencyFactor := 0.5
	if maxLatency > minLatency {
		latencyFactor = (maxLatency - float64(c.Profile.AvgLatencyMS)) / (maxLatency - minLatency)
	}

	costFactor := 0.5
	if maxCost > minCost {
		costFactor = (maxCost - c.EstimatedCost) / (maxCost - minCost)
	}

	qualityFactor := c.Metadata.QualityScore / 10.0
	if strategy.UseCodingScore {
		qualityFactor = c.Metadata.CodingScore / 10.0
	}

	reliabilityFactor := 1.0 - c.Profile.ErrorRate
	healthFactor := 1.0
	if c.EffectiveHealth == HealthStatusDegraded {
		healthFactor = 0.6
	}

	latencyFactor = clamp01(latencyFactor)
	costFactor = clamp01(costFactor)
	qualityFactor = clamp01(qualityFactor)
	reliabilityFactor = clamp01(reliabilityFactor)

	return ((strategy.QualityWeight * qualityFactor) +
		(strategy.CostWeight * costFactor) +
		(strategy.LatencyWeight * latencyFactor) +
		(strategy.ReliabilityWeight * reliabilityFactor)) * healthFactor
}

func getNormalizationBounds(contenders map[string]contender) (minCost, maxCost, minLatency, maxLatency float64) {
	minCost, maxCost = math.MaxFloat64, 0.0
	minLatency, maxLatency = math.MaxFloat64, 0.0

	for _, c := range contenders {
		if c.EstimatedCost < minCost {
			minCost = c.EstimatedCost
		}
		if c.EstimatedCost > maxCost {
			maxCost = c.EstimatedCost
		}
		lat := float64(c.Profile.AvgLatencyMS)
		if lat < minLatency {
			minLatency = lat
		}
		if lat > maxLatency {
			maxLatency = lat
		}
	}
	if minCost == math.MaxFloat64 {
		minCost = 0
	}
	if minLatency == math.MaxFloat64 {
		minLatency = 0
	}
	return
}

func (r *Router) passesPreChecks(
	profile *ModelProfile,
	providerHealth *ProviderHealth,
	capabilityHealth *CapabilityHealth,
	effectiveHealth HealthStatus,
	monthlyBudget float64,
) (bool, string) {
	staleness := r.profiler.healthCheckStaleness()
	if !providerHealth.AccessAllowed {
		return false, "provider access denied"
	}
	if !capabilityHealth.AccessAllowed {
		return false, "capability access denied"
	}
	if !capabilityHealth.Supported {
		return false, "capability unsupported"
	}
	if effectiveHealth == HealthStatusOffline {
		return false, "offline"
	}
	if !profile.CircuitOpenUntil.IsZero() && time.Now().UTC().Before(profile.CircuitOpenUntil) {
		return false, fmt.Sprintf("circuit open until %s", profile.CircuitOpenUntil.Format(time.RFC3339))
	}
	if !profile.LastHealthCheck.IsZero() && time.Since(profile.LastHealthCheck) > staleness {
		return false, fmt.Sprintf("health check stale > %s", staleness)
	}
	if !providerHealth.LastHealthCheck.IsZero() && time.Since(providerHealth.LastHealthCheck) > staleness {
		return false, fmt.Sprintf("provider health check stale > %s", staleness)
	}
	if !capabilityHealth.LastHealthCheck.IsZero() && time.Since(capabilityHealth.LastHealthCheck) > staleness {
		return false, fmt.Sprintf("capability health check stale > %s", staleness)
	}
	if monthlyBudget > 0 && profile.CostSpentMonthly >= monthlyBudget {
		return false, fmt.Sprintf("over budget $%.4f / $%.2f", profile.CostSpentMonthly, monthlyBudget)
	}

	maxErrorRate := 0.15
	if val, ok := r.config.Thresholds["max_error_rate"].(float64); ok {
		maxErrorRate = val
	}

	minRequests := int64(20)
	switch val := r.config.Thresholds["min_request_count"].(type) {
	case int:
		minRequests = int64(val)
	case float64:
		minRequests = int64(val)
	}

	totalReq := profile.TotalSuccesses + profile.TotalFailures
	if totalReq >= minRequests && profile.ErrorRate > maxErrorRate {
		return false, fmt.Sprintf("error rate too high %.2f%% > %.2f%%", profile.ErrorRate*100, maxErrorRate*100)
	}

	return true, ""
}

func clamp01(v float64) float64 {
	return math.Max(0, math.Min(1, v))
}

func (r *Router) describeContender(c contender, score float64, reason string) RouteCandidate {
	return RouteCandidate{
		ModelID:       c.Profile.ModelID,
		Provider:      c.Profile.Provider,
		Score:         score,
		EstimatedCost: c.EstimatedCost,
		AvgLatencyMS:  c.Profile.AvgLatencyMS,
		Health: RouteHealth{
			ProviderStatus:   c.ProviderHealth.Status,
			ModelStatus:      normalizeHealthStatus(c.Profile.Status),
			CapabilityStatus: c.CapabilityHealth.Status,
		},
		Reason: reason,
	}
}
