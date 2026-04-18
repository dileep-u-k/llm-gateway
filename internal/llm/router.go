// In file: internal/llm/router.go
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

// =================================================================================
// Configuration Structs
// =================================================================================

type RoutingStrategy struct {
	UseCodingScore    bool    `yaml:"use_coding_score"`
	QualityWeight     float64 `yaml:"quality_weight"`
	CostWeight        float64 `yaml:"cost_weight"`
	LatencyWeight     float64 `yaml:"latency_weight"`
	ReliabilityWeight float64 `yaml:"reliability_weight"`
}

type ModelMetadata struct {
	QualityScore float64 `yaml:"quality_score"` // 0–10
	CodingScore  float64 `yaml:"coding_score"`  // 0–10
}

type RouterConfig struct {
	Thresholds                 map[string]interface{}     `yaml:"pre_check_thresholds"`
	Models                     map[string]ModelMetadata   `yaml:"models"`
	Strategies                 map[string]RoutingStrategy `yaml:"strategies"`
	SmartBalancedCostThreshold float64                    `yaml:"smart_balanced_cost_threshold"`

	// --- ADD THESE LINES ---
	ImageModels     map[string]ImageModelMetadata   `yaml:"image_models"`
	ImageStrategies map[string]ImageRoutingStrategy `yaml:"image_strategies"`
}

// --- ADD THESE NEW STRUCTS FOR IMAGE GENERATION ---

// ImageModelMetadata defines static capabilities of an image model.
type ImageModelMetadata struct {
	QualityScore  float64 `yaml:"quality_score"`
	ArtisticScore float64 `yaml:"artistic_score"` // Specific for image models
	SpeedScore    float64 `yaml:"speed_score"`
}

// ImageRoutingStrategy defines the weighting for different image routing preferences.
type ImageRoutingStrategy struct {
	UseArtisticScore  bool    `yaml:"use_artistic_score"` // Flag to use artistic score instead of quality
	QualityWeight     float64 `yaml:"quality_weight"`
	SpeedWeight       float64 `yaml:"speed_weight"`
	CostWeight        float64 `yaml:"cost_weight"`
	ReliabilityWeight float64 `yaml:"reliability_weight"`
}

// --- END OF NEW STRUCTS ---

// =================================================================================
// Router Service
// =================================================================================

type Router struct {
	profiler *Profiler
	config   *RouterConfig
}

func NewRouter(profiler *Profiler, config *RouterConfig) *Router {
	if config.SmartBalancedCostThreshold <= 0 {
		config.SmartBalancedCostThreshold = 0.001 // safe fallback
	}
	return &Router{profiler: profiler, config: config}
}

type contender struct {
	Profile       *ModelProfile
	Metadata      ModelMetadata
	EstimatedCost float64
}

// =================================================================================
// Core Routing
// =================================================================================

func (r *Router) SelectOptimalModel(
	ctx context.Context,
	availableModels []string,
	preference string,
	promptTokens int,
	modelBudgets map[string]float64,
) (string, error) {
	log.Printf("--- Starting Model Selection (Preference: %q) ---", preference)

	// Gather contenders
	contenders := make(map[string]contender)
	for _, modelID := range availableModels {
		profile, err := r.profiler.GetProfile(ctx, modelID)
		if err != nil {
			log.Printf("⚠️ Skipping model %s: profiler error: %v", modelID, err)
			continue
		}
		if ok, reason := r.passesPreChecks(profile, modelBudgets[modelID]); !ok {
			log.Printf("⚠️ Filtering model %s: %s", modelID, reason)
			continue
		}
		modelMeta, ok := r.config.Models[profile.ModelID]
		if !ok {
			log.Printf("⚠️ Skipping model %s: metadata not found in config", modelID)
			continue
		}
		// Conservative: assume output ≈ 2× input tokens
		estimatedOutputTokens := promptTokens * 2
		estimatedCost := (float64(promptTokens)*profile.CostPerInputToken +
			float64(estimatedOutputTokens)*profile.CostPerOutputToken)

		contenders[modelID] = contender{
			Profile:       profile,
			Metadata:      modelMeta,
			EstimatedCost: estimatedCost,
		}
		log.Printf("✅ Model %s is a contender", modelID)
	}

	if len(contenders) == 0 {
		return "", errors.New("no suitable model after filtering")
	}
	if len(contenders) == 1 {
		for modelID := range contenders {
			log.Printf("🏆 Only one contender: %s", modelID)
			return modelID, nil
		}
	}

	// Get routing strategy
	strategy, err := r.getStrategy(preference, contenders)
	if err != nil {
		return "", err
	}

	// Scoring loop
	bestModel := ""
	bestScore := -1.0
	var tieBreakers []string
	minCost, maxCost, minLatency, maxLatency := getNormalizationBounds(contenders)

	for modelID, c := range contenders {
		score := r.calculateNormalizedScore(c, strategy, minCost, maxCost, minLatency, maxLatency)
		log.Printf("- Score[%s]: Latency=%dms Cost=%.6f Quality=%.2f Final=%.4f",
			modelID, c.Profile.AvgLatencyMS, c.EstimatedCost, c.Metadata.QualityScore, score)

		if score > bestScore {
			bestScore = score
			bestModel = modelID
			tieBreakers = []string{modelID}
		} else if score == bestScore {
			tieBreakers = append(tieBreakers, modelID)
		}
	}

	if bestModel == "" {
		return "", errors.New("failed to select best model")
	}

	// Deterministic tie-breaking
	if len(tieBreakers) > 1 {
		sort.Strings(tieBreakers)
		bestModel = tieBreakers[0]
		log.Printf("⭐ Tie-break: multiple models scored %.4f, chose %s", bestScore, bestModel)
	}

	log.Printf("🏆 Selected best model: %s (Score=%.4f)", bestModel, bestScore)
	return bestModel, nil
}

// =================================================================================
// Strategy & Scoring
// =================================================================================

func (r *Router) getStrategy(preference string, contenders map[string]contender) (RoutingStrategy, error) {
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
			return r.config.Strategies["latency-focused-balanced"], nil
		}
		log.Printf("Smart-balanced: avg cost %.6f ≥ threshold %.6f → quality-priority",
			avgCost, r.config.SmartBalancedCostThreshold)
		return r.config.Strategies["quality-focused-balanced"], nil
	}

	strategy, ok := r.config.Strategies[preference]
	if !ok {
		log.Printf("⚠️ Strategy %q not found, falling back to 'default'", preference)
		strategy, ok = r.config.Strategies["default"]
		if !ok {
			return RoutingStrategy{}, errors.New("no 'default' strategy configured")
		}
	}
	return strategy, nil
}

func (r *Router) calculateNormalizedScore(
	c contender,
	strategy RoutingStrategy,
	minCost, maxCost, minLatency, maxLatency float64,
) float64 {
	// Normalize latency (lower = better)
	latencyFactor := 0.5
	if maxLatency > minLatency {
		latencyFactor = (maxLatency - float64(c.Profile.AvgLatencyMS)) / (maxLatency - minLatency)
	}

	// Normalize cost (lower = better)
	costFactor := 0.5
	if maxCost > minCost {
		costFactor = (maxCost - c.EstimatedCost) / (maxCost - minCost)
	}

	// Normalize quality/coding score (higher = better)
	qualityFactor := c.Metadata.QualityScore / 10.0
	if strategy.UseCodingScore {
		qualityFactor = c.Metadata.CodingScore / 10.0
	}

	// Reliability (1 - error rate)
	reliabilityFactor := 1.0 - c.Profile.ErrorRate

	// Clamp all values between 0–1
	latencyFactor = clamp01(latencyFactor)
	costFactor = clamp01(costFactor)
	qualityFactor = clamp01(qualityFactor)
	reliabilityFactor = clamp01(reliabilityFactor)

	// Weighted score
	return (strategy.QualityWeight * qualityFactor) +
		(strategy.CostWeight * costFactor) +
		(strategy.LatencyWeight * latencyFactor) +
		(strategy.ReliabilityWeight * reliabilityFactor)
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

// =================================================================================
// Pre-checks
// =================================================================================

func (r *Router) passesPreChecks(profile *ModelProfile, monthlyBudget float64) (bool, string) {
	// Staleness threshold
	staleness := 5 * time.Minute
	if val, ok := r.config.Thresholds["health_check_staleness"].(string); ok {
		if parsed, err := time.ParseDuration(val); err == nil {
			staleness = parsed
		}
	}

	// Hard filters
	if profile.Status == "offline" {
		return false, "offline"
	}
	if time.Since(profile.LastHealthCheck) > staleness {
		return false, fmt.Sprintf("health check stale > %s", staleness)
	}
	if monthlyBudget > 0 && profile.CostSpentMonthly >= monthlyBudget {
		return false, fmt.Sprintf("over budget $%.4f / $%.2f", profile.CostSpentMonthly, monthlyBudget)
	}

	// Error rate threshold
	maxErrorRate := 0.15
	if val, ok := r.config.Thresholds["max_error_rate"].(float64); ok {
		maxErrorRate = val
	}

	minRequests := int64(20)
	if val, ok := r.config.Thresholds["min_request_count"].(int); ok {
		minRequests = int64(val)
	}

	totalReq := profile.TotalSuccesses + profile.TotalFailures
	if totalReq >= minRequests && profile.ErrorRate > maxErrorRate {
		return false, fmt.Sprintf("error rate too high %.2f%% > %.2f%%", profile.ErrorRate*100, maxErrorRate*100)
	}

	return true, ""
}

// =================================================================================
// Utils
// =================================================================================

func clamp01(v float64) float64 {
	return math.Max(0, math.Min(1, v))
}
