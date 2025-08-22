// In file: internal/llm/router.go
package llm

import (
    "context"
    "errors"
    "fmt"
    "log"
    "math"
    "time"
)

// =================================================================================
// Configuration Structs
// =================================================================================

// RoutingStrategy defines the weights for scoring models based on a preference.
type RoutingStrategy struct {
    UseCodingScore bool    `yaml:"use_coding_score"`
    QualityWeight  float64 `yaml:"quality_weight"`
    CostWeight     float64 `yaml:"cost_weight"`
    LatencyWeight  float64 `yaml:"latency_weight"`
}

// ModelMetadata holds static, configured information about a model.
type ModelMetadata struct {
    QualityScore float64 `yaml:"quality_score"`
    CodingScore  float64 `yaml:"coding_score"`
}

// RouterConfig holds the complete configuration for the router.
type RouterConfig struct {
    Thresholds map[string]interface{}    `yaml:"pre_check_thresholds"`
    Models     map[string]ModelMetadata   `yaml:"models"`
    Strategies map[string]RoutingStrategy `yaml:"strategies"`
}

// =================================================================================
// Router Service
// =================================================================================

// Router is a service that selects the best LLM for a given request.
type Router struct {
    profiler *Profiler
    config   *RouterConfig
}

// NewRouter creates a new, configured router service.
func NewRouter(profiler *Profiler, config *RouterConfig) *Router {
    return &Router{
        profiler: profiler,
        config:   config,
    }
}

// contender holds the profile and metadata for a model that has passed pre-checks.
type contender struct {
    Profile       *ModelProfile
    Metadata      ModelMetadata
    EstimatedCost float64
}

// SelectOptimalModel is the core routing algorithm.
// It now uses a two-pass approach:
// 1. Filter models that pass pre-checks to create a pool of "contenders".
// 2. Normalize and score the contenders to find the best one.
func (r *Router) SelectOptimalModel(ctx context.Context, availableModels []string, preference string, promptTokens int, modelBudgets map[string]float64) (string, error) {
    log.Printf("--- Starting Model Selection (Preference: '%s') ---", preference)

    // --- Pass 1: Filter models and create a pool of contenders ---
    contenders := make(map[string]contender)
    for _, modelID := range availableModels {
        profile, err := r.profiler.GetProfile(ctx, modelID)
        if err != nil {
            log.Printf("Could not get profile for model %s, skipping: %v", modelID, err)
            continue
        }

        monthlyBudget := modelBudgets[modelID]
        if ok, reason := r.passesPreChecks(profile, monthlyBudget); !ok {
            log.Printf("- Filtering Model: %s | Reason: %s", modelID, reason)
            continue
        }

        modelMeta, ok := r.config.Models[profile.ModelID]
        if !ok {
            log.Printf("- Filtering Model: %s | Reason: Model metadata not found in config.", modelID)
            continue
        }

        // Estimate cost for this specific call for scoring purposes.
        estimatedOutputTokens := promptTokens * 2 // A simple heuristic.
        estimatedCost := (float64(promptTokens) * profile.CostPerInputToken) + (float64(estimatedOutputTokens) * profile.CostPerOutputToken)

        contenders[modelID] = contender{
            Profile:       profile,
            Metadata:      modelMeta,
            EstimatedCost: estimatedCost,
        }
        log.Printf("- Model %s is a contender.", modelID)
    }

    if len(contenders) == 0 {
        return "", errors.New("no suitable, healthy, and in-budget model found after filtering")
    }

    // If there's only one contender, select it immediately.
    if len(contenders) == 1 {
        for modelID := range contenders {
            log.Printf("🏆 Only one contender found. Selecting: %s", modelID)
            return modelID, nil
        }
    }

    // --- Pass 2: Normalize and score the contenders ---
    strategy, err := r.getStrategy(preference, promptTokens, contenders)
    if err != nil {
        return "", err
    }

    bestModel := ""
    bestScore := -1.0

    // Calculate min/max values across contenders for normalization.
    minCost, maxCost, minLatency, maxLatency := getNormalizationBounds(contenders)

    for modelID, c := range contenders {
        score := r.calculateNormalizedScore(c, strategy, minCost, maxCost, minLatency, maxLatency)
        log.Printf("- Scoring Model: %s | Latency: %dms | Est. Cost: %.6f | Quality: %.2f | Final Score: %.4f",
            modelID, c.Profile.AvgLatencyMS, c.EstimatedCost, c.Metadata.QualityScore, score)

        if score > bestScore {
            bestScore = score
            bestModel = modelID
        }
    }

    if bestModel == "" {
        // This should theoretically not be reached if there are contenders, but it's a safe fallback.
        return "", errors.New("failed to select a model after scoring")
    }

    log.Printf("🏆 Best model selected: %s (Score: %.4f)", bestModel, bestScore)
    return bestModel, nil
}

// getStrategy retrieves the appropriate routing strategy based on the preference.
// It also handles the dynamic logic for "smart-balanced".
func (r *Router) getStrategy(preference string, promptTokens int, contenders map[string]contender) (RoutingStrategy, error) {
    // The "smart-balanced" strategy has dynamic weights based on the prompt size.
    if preference == "smart-balanced" {
        // Example dynamic logic: for cheap requests, prioritize speed; for expensive ones, quality.
        // This could be made more sophisticated by moving thresholds to the config.
        avgCost := 0.0
        for _, c := range contenders {
            avgCost += c.EstimatedCost
        }
        avgCost /= float64(len(contenders))

        if avgCost < 0.001 { // For cheap requests, prioritize speed.
            log.Println("Smart-balanced mode: Prioritizing latency for low-cost request.")
            return r.config.Strategies["latency-focused-balanced"], nil
        } else { // For expensive requests, prioritize quality.
            log.Println("Smart-balanced mode: Prioritizing quality for high-cost request.")
            return r.config.Strategies["quality-focused-balanced"], nil
        }
    }

    strategy, ok := r.config.Strategies[preference]
    if !ok {
        // Fallback to the default strategy if the preference is unknown.
        log.Printf("Warning: preference '%s' not found, falling back to 'default' strategy.", preference)
        strategy, ok = r.config.Strategies["default"]
        if !ok {
            return RoutingStrategy{}, errors.New("default strategy not found in configuration")
        }
    }
    return strategy, nil
}

// calculateNormalizedScore computes a model's score using linear normalization.
// This ensures that weights have a predictable, proportional impact.
func (r *Router) calculateNormalizedScore(c contender, strategy RoutingStrategy, minCost, maxCost, minLatency, maxLatency float64) float64 {
    // --- Normalize Component Factors (so that 1.0 is best, 0.0 is worst) ---

    // Latency: Lower is better.
    latencyFactor := 0.5 // Default to average if min/max are the same
    if maxLatency > minLatency {
        latencyFactor = (maxLatency - float64(c.Profile.AvgLatencyMS)) / (maxLatency - minLatency)
    }

    // Cost: Lower is better.
    costFactor := 0.5 // Default to average if min/max are the same
    if maxCost > minCost {
        costFactor = (maxCost - c.EstimatedCost) / (maxCost - minCost)
    }

    // Quality: Higher is better. This score is already a relative value, so no normalization needed.
    qualityFactor := c.Metadata.QualityScore / 10.0 // Normalize to a 0-1 scale
    if strategy.UseCodingScore {
        qualityFactor = c.Metadata.CodingScore / 10.0
    }

    // Reliability: Higher is better.
    reliabilityFactor := 1.0 - c.Profile.ErrorRate

    // --- Final Weighted Score Calculation ---
    // The reliability factor acts as a multiplier on the weighted average of other factors.
    score := (
        (strategy.QualityWeight * qualityFactor) +
        (strategy.CostWeight * costFactor) +
        (strategy.LatencyWeight * latencyFactor)) * reliabilityFactor

    return score
}

// getNormalizationBounds finds the min/max cost and latency from the pool of contenders.
func getNormalizationBounds(contenders map[string]contender) (minCost, maxCost, minLatency, maxLatency float64) {
    minCost = math.MaxFloat64
    maxCost = 0.0
    minLatency = math.MaxFloat64
    maxLatency = 0.0

    for _, c := range contenders {
        if c.EstimatedCost < minCost {
            minCost = c.EstimatedCost
        }
        if c.EstimatedCost > maxCost {
            maxCost = c.EstimatedCost
        }
        latency := float64(c.Profile.AvgLatencyMS)
        if latency < minLatency {
            minLatency = latency
        }
        if latency > maxLatency {
            maxLatency = latency
        }
    }
    return
}

// passesPreChecks evaluates a model against configured health, budget, and reliability thresholds.
func (r *Router) passesPreChecks(profile *ModelProfile, monthlyBudget float64) (bool, string) {
    // Health Check
    staleness, _ := time.ParseDuration(r.config.Thresholds["health_check_staleness"].(string))
    if profile.Status == "offline" {
        return false, "Model is marked as offline."
    }
    if time.Since(profile.LastHealthCheck) > staleness {
        return false, fmt.Sprintf("Health check is stale (last check > %s ago).", staleness)
    }

    // Budget Check
    if monthlyBudget > 0 && profile.CostSpentMonthly >= monthlyBudget {
        return false, fmt.Sprintf("Over monthly budget ($%.4f / $%.2f).", profile.CostSpentMonthly, monthlyBudget)
    }

    // Error Rate Check
    maxErrorRate := r.config.Thresholds["max_error_rate"].(float64)
    minRequests := int64(r.config.Thresholds["min_request_count"].(int))
    totalRequests := profile.TotalSuccesses + profile.TotalFailures

    if totalRequests > minRequests && profile.ErrorRate > maxErrorRate {
        return false, fmt.Sprintf("Error rate is too high (%.2f%% > %.2f%%).", profile.ErrorRate*100, maxErrorRate*100)
    }

    return true, ""
}
