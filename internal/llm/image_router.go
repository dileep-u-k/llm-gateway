// In file: internal/llm/image_router.go
package llm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
	"time"
)

// ImageRouter intelligently selects the best image model using data-driven logic.
type ImageRouter struct {
	profiler *Profiler
	config   *RouterConfig
	models   []string
	analyzer *ImagePromptAnalyzer
}

// NewImageRouter creates a new, configured image router.
func NewImageRouter(enabledModels string, profiler *Profiler, cfg *RouterConfig) *ImageRouter {
	return &ImageRouter{
		profiler: profiler,
		config:   cfg,
		models:   strings.Split(enabledModels, ","),
		analyzer: NewImagePromptAnalyzer(),
	}
}

// imageContender holds the profile and metadata for a model during scoring.
type imageContender struct {
	Profile  *ModelProfile
	Metadata ImageModelMetadata
}

// SelectModel performs a full, data-driven filter-and-score analysis.
func (r *ImageRouter) SelectModel(ctx context.Context, prompt, preference string) (string, error) {
	log.Printf("--- Starting Image Model Selection ---")

	strategyName := preference
	if strategyName == "" {
		strategyName = r.analyzer.Analyze(prompt)
		log.Printf("🤖 No image preference specified. Auto-selected: '%s'", strategyName)
	}
	strategy, ok := r.config.ImageStrategies[strategyName]
	if !ok {
		return "", fmt.Errorf("image strategy '%s' not found in configuration", strategyName)
	}

	// Pass 1: Filter models that pass pre-checks.
	var contenders []imageContender
	for _, modelID := range r.models {
		if modelID == "" {
			continue
		}
		effectiveHealth, providerHealth, capabilityHealth, profile, err := r.profiler.CombinedModelHealth(ctx, modelID, CapabilityImageGeneration)
		if err != nil {
			log.Printf("Could not get profile for image model %s, skipping: %v", modelID, err)
			continue
		}
		if ok, reason := r.passesPreChecks(profile, providerHealth, capabilityHealth, effectiveHealth); !ok {
			log.Printf("- Filtering Image Model: %s | Reason: %s", modelID, reason)
			continue
		}
		contenders = append(contenders, imageContender{
			Profile:  profile,
			Metadata: r.config.ImageModels[modelID],
		})
	}

	if len(contenders) == 0 {
		return "", errors.New("no suitable, healthy image models found after filtering")
	}
	if len(contenders) == 1 {
		return contenders[0].Profile.ModelID, nil
	}

	// Pass 2: Score the healthy contenders.
	bestModel := ""
	bestScore := -1.0
	var tieBreakers []string
	minCost, maxCost := getImageCostBounds(contenders)

	for _, c := range contenders {
		score := r.calculateScore(c, strategy, minCost, maxCost)
		log.Printf("- Scoring Image Model [%s]: Quality=%.2f, Cost=%.4f -> Final Score=%.4f",
			c.Profile.ModelID, c.Metadata.QualityScore, c.Profile.CostPerInputToken, score)

		if score > bestScore {
			bestScore = score
			bestModel = c.Profile.ModelID
			tieBreakers = []string{bestModel}
		} else if score == bestScore {
			tieBreakers = append(tieBreakers, c.Profile.ModelID)
		}
	}

	if len(tieBreakers) > 1 {
		sort.Strings(tieBreakers)
		bestModel = tieBreakers[0]
	}

	log.Printf("🏆 Selected best image model: %s (Score=%.2f)", bestModel, bestScore)
	return bestModel, nil
}

// calculateScore applies the weighted strategy to an image model.
func (r *ImageRouter) calculateScore(c imageContender, strategy ImageRoutingStrategy, minCost, maxCost float64) float64 {
	qualityScore := c.Metadata.QualityScore / 10.0
	if strategy.UseArtisticScore {
		qualityScore = c.Metadata.ArtisticScore / 10.0
	}
	speedScore := c.Metadata.SpeedScore / 10.0

	costFactor := 0.5
	if maxCost > minCost {
		costFactor = (maxCost - c.Profile.CostPerInputToken) / (maxCost - minCost)
	}

	reliabilityFactor := 1.0 - c.Profile.ErrorRate

	// Clamp values to a 0-1 range.
	// --- IMPORTANT: Ensure 'clamp01' is defined elsewhere in your llm package ---
	qualityScore = clamp01(qualityScore)
	speedScore = clamp01(speedScore)
	costFactor = clamp01(costFactor)
	reliabilityFactor = clamp01(reliabilityFactor)

	return (strategy.QualityWeight * qualityScore) +
		(strategy.SpeedWeight * speedScore) +
		(strategy.CostWeight * costFactor) +
		(strategy.ReliabilityWeight * reliabilityFactor)
}

// passesPreChecks evaluates an image model against the main configured thresholds.
func (r *ImageRouter) passesPreChecks(
	profile *ModelProfile,
	providerHealth *ProviderHealth,
	capabilityHealth *CapabilityHealth,
	effectiveHealth HealthStatus,
) (bool, string) {
	staleness := r.profiler.healthCheckStaleness()
	if !providerHealth.AccessAllowed {
		return false, "Provider access denied."
	}
	if !capabilityHealth.AccessAllowed {
		return false, "Capability access denied."
	}
	if !capabilityHealth.Supported {
		return false, "Capability unsupported."
	}
	if effectiveHealth == HealthStatusOffline {
		return false, "Model is offline."
	}
	if !profile.CircuitOpenUntil.IsZero() && time.Now().UTC().Before(profile.CircuitOpenUntil) {
		return false, fmt.Sprintf("Circuit breaker open until %s.", profile.CircuitOpenUntil.Format(time.RFC3339))
	}
	if !profile.LastHealthCheck.IsZero() && time.Since(profile.LastHealthCheck) > staleness {
		return false, fmt.Sprintf("Health check is stale (last check > %s ago).", staleness.Round(time.Second))
	}
	if !providerHealth.LastHealthCheck.IsZero() && time.Since(providerHealth.LastHealthCheck) > staleness {
		return false, fmt.Sprintf("Provider health check is stale (>%s).", staleness.Round(time.Second))
	}

	maxErrorRate := 0.50
	if rate, ok := r.config.Thresholds["max_error_rate"].(float64); ok {
		maxErrorRate = rate
	}
	if profile.ErrorRate > maxErrorRate {
		return false, fmt.Sprintf("Error rate is too high (%.2f%% > %.2f%%).", profile.ErrorRate*100, maxErrorRate*100)
	}

	return true, ""
}

// getImageCostBounds is a helper function for normalization.
func getImageCostBounds(contenders []imageContender) (minCost, maxCost float64) {
	minCost = math.MaxFloat64
	maxCost = 0.0
	for _, c := range contenders {
		if c.Profile.CostPerInputToken < minCost {
			minCost = c.Profile.CostPerInputToken
		}
		if c.Profile.CostPerInputToken > maxCost {
			maxCost = c.Profile.CostPerInputToken
		}
	}
	// Handle case where all costs are the same to prevent division by zero in costFactor calculation
	if minCost == math.MaxFloat64 { // No contenders
		minCost = 0
		maxCost = 0
	} else if maxCost == minCost { // All contenders have same cost
		maxCost += 1 // Small offset to avoid division by zero in costFactor if range is 0
	}
	return
}
