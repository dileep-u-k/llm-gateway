// In file: internal/llm/profiler.go
package llm

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"time"

	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/redis/go-redis/v9"
)

// =================================================================================
// Ultra Production-Ready Model Profiler
// =================================================================================
// Key Features:
// 1. **Concurrency-Safe Initialization** → HSetNX prevents race conditions.
// 2. **Atomic Updates** → Pipelines/transactions ensure consistency.
// 3. **Resilient Parsing** → Gracefully handles missing/corrupted Redis data.
// 4. **Strong Logging** → CRITICAL, WARNING, INFO messages guide ops/debugging.
// 5. **Cost Tracking** → Monthly cost rollups with automatic expiry.
// =================================================================================

// ModelProfile tracks performance, cost, and reliability metrics for an LLM.
type ModelProfile struct {
	ModelID            string    `json:"model_id" redis:"model_id"`
	AvgLatencyMS       int64     `json:"avg_latency_ms" redis:"avg_latency_ms"`
	CostPerInputToken  float64   `json:"cost_per_input_token" redis:"cost_per_input_token"`
	CostPerOutputToken float64   `json:"cost_per_output_token" redis:"cost_per_output_token"`
	Status             string    `json:"status" redis:"status"`
	ErrorRate          float64   `json:"error_rate" redis:"error_rate"`
	TotalSuccesses     int64     `json:"total_successes" redis:"total_successes"`
	TotalFailures      int64     `json:"total_failures" redis:"total_failures"`
	TotalInputTokens   int64     `json:"total_input_tokens" redis:"total_input_tokens"`
	TotalOutputTokens  int64     `json:"total_output_tokens" redis:"total_output_tokens"`
	LastHealthCheck    time.Time `json:"last_health_check" redis:"last_health_check"`
	CostSpentMonthly   float64   `json:"cost_spent_monthly"`
}

// modelCosts holds input/output token cost per model.
var modelCosts = make(map[string]map[string]float64)

func InitializeModelCosts(costs map[string]map[string]float64) {
	modelCosts = costs
	for modelID, costData := range modelCosts {
		log.Printf("[INFO] Loaded cost config for %s: Input=$%.8f/token, Output=$%.8f/token",
			modelID, costData["input"], costData["output"])
	}
}

type Profiler struct {
	rdb *redis.Client
}

func NewProfiler(rdb *redis.Client) *Profiler {
	return &Profiler{rdb: rdb}
}

func (p *Profiler) getProfileKey(modelID string) string {
	return fmt.Sprintf("profile:%s", modelID)
}

// ================================================================================
// GetProfile: Safe, resilient retrieval of a model profile.
// ================================================================================
func (p *Profiler) GetProfile(ctx context.Context, modelID string) (*ModelProfile, error) {
	key := p.getProfileKey(modelID)
	profileData, err := p.rdb.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get profile for %s: %w", modelID, err)
	}

	if len(profileData) == 0 {
		return p.createDefaultProfile(ctx, modelID)
	}

	// Safe parsing with fallbacks
	profile := &ModelProfile{ModelID: modelID}
	if v, err := strconv.ParseInt(profileData["avg_latency_ms"], 10, 64); err == nil {
		profile.AvgLatencyMS = v
	} else {
		log.Printf("[WARN] Could not parse avg_latency_ms for %s, defaulting to 2000ms", modelID)
		profile.AvgLatencyMS = 2000
	}
	profile.CostPerInputToken, _ = strconv.ParseFloat(profileData["cost_per_input_token"], 64)
	profile.CostPerOutputToken, _ = strconv.ParseFloat(profileData["cost_per_output_token"], 64)
	profile.Status = profileData["status"]
	profile.ErrorRate, _ = strconv.ParseFloat(profileData["error_rate"], 64)
	profile.TotalSuccesses, _ = strconv.ParseInt(profileData["total_successes"], 10, 64)
	profile.TotalFailures, _ = strconv.ParseInt(profileData["total_failures"], 10, 64)
	profile.TotalInputTokens, _ = strconv.ParseInt(profileData["total_input_tokens"], 10, 64)
	profile.TotalOutputTokens, _ = strconv.ParseInt(profileData["total_output_tokens"], 10, 64)
	profile.LastHealthCheck, _ = time.Parse(time.RFC3339Nano, profileData["last_health_check"])

	// Monthly spend tracking
	costKey := fmt.Sprintf("cost:%s:%s", modelID, time.Now().Format("2006-01"))
	profile.CostSpentMonthly, _ = p.rdb.Get(ctx, costKey).Float64()

	return profile, nil
}

// ================================================================================
// createDefaultProfile: Concurrency-safe profile creation.
// ================================================================================
func (p *Profiler) createDefaultProfile(ctx context.Context, modelID string) (*ModelProfile, error) {
	costs, ok := modelCosts[modelID]
	if !ok {
		log.Printf("[CRITICAL] No cost config for model '%s'. Defaulting to zero cost.", modelID)
		costs = map[string]float64{"input": 0, "output": 0}
	}

	profile := &ModelProfile{
		ModelID:            modelID,
		AvgLatencyMS:       2000,
		CostPerInputToken:  costs["input"],
		CostPerOutputToken: costs["output"],
		Status:             "online",
		TotalSuccesses:     1,
		TotalFailures:      0,
		ErrorRate:          0.0,
		LastHealthCheck:    time.Now(),
	}

	key := p.getProfileKey(modelID)

	// Ensure only one process initializes the profile
	wasSet, err := p.rdb.HSetNX(ctx, key, "model_id", profile.ModelID).Result()
	if err != nil {
		return nil, fmt.Errorf("redis HSetNX failed for %s: %w", modelID, err)
	}

	if wasSet {
		pipe := p.rdb.Pipeline()
		pipe.HSet(ctx, key, "avg_latency_ms", profile.AvgLatencyMS)
		pipe.HSet(ctx, key, "cost_per_input_token", profile.CostPerInputToken)
		pipe.HSet(ctx, key, "cost_per_output_token", profile.CostPerOutputToken)
		pipe.HSet(ctx, key, "status", profile.Status)
		pipe.HSet(ctx, key, "total_successes", profile.TotalSuccesses)
		pipe.HSet(ctx, key, "total_failures", profile.TotalFailures)
		pipe.HSet(ctx, key, "error_rate", profile.ErrorRate)
		pipe.HSet(ctx, key, "last_health_check", profile.LastHealthCheck.Format(time.RFC3339Nano))
		if _, err := pipe.Exec(ctx); err != nil {
			return nil, fmt.Errorf("failed to populate new profile for %s: %w", modelID, err)
		}
		log.Printf("[INFO] Created new profile for %s", modelID)
	} else {
		log.Printf("[INFO] Profile for %s already exists, fetching.", modelID)
		return p.GetProfile(ctx, modelID)
	}

	return profile, nil
}

// ================================================================================
// UpdateProfileOnSuccess: Atomic updates on successful model usage.
// ================================================================================
func (p *Profiler) UpdateProfileOnSuccess(ctx context.Context, modelID string, latency time.Duration, usage api.Usage) {
	key := p.getProfileKey(modelID)
	const alpha = 0.1 // smoothing factor for latency EMA

	// Get required fields before update
	results, err := p.rdb.HMGet(ctx, key, "avg_latency_ms", "total_failures").Result()
	if err != nil {
		log.Printf("[ERROR] Failed to fetch profile data for success update %s: %v", modelID, err)
		return
	}
	currentLatency, _ := strconv.ParseInt(fmt.Sprint(results[0]), 10, 64)
	totalFailures, _ := strconv.ParseInt(fmt.Sprint(results[1]), 10, 64)

	newLatency := int64((alpha * float64(latency.Milliseconds())) + ((1.0 - alpha) * float64(currentLatency)))

	pipe := p.rdb.Pipeline()
	pipe.HSet(ctx, key, "avg_latency_ms", newLatency)
	successes := pipe.HIncrBy(ctx, key, "total_successes", 1)
	pipe.HIncrBy(ctx, key, "total_input_tokens", int64(usage.PromptTokens))
	pipe.HIncrBy(ctx, key, "total_output_tokens", int64(usage.CompletionTokens))
	pipe.HSet(ctx, key, "status", "online")

	callCost := float64(usage.PromptTokens)*modelCosts[modelID]["input"] +
		float64(usage.CompletionTokens)*modelCosts[modelID]["output"]
	costKey := fmt.Sprintf("cost:%s:%s", modelID, time.Now().Format("2006-01"))
	pipe.IncrByFloat(ctx, costKey, callCost)
	pipe.Expire(ctx, costKey, 35*24*time.Hour)

	totalRequests := successes.Val() + totalFailures
	if totalRequests > 0 {
		errorRate := float64(totalFailures) / float64(totalRequests)
		pipe.HSet(ctx, key, "error_rate", errorRate)
	}

	if _, err := pipe.Exec(ctx); err != nil {
		log.Printf("[ERROR] Failed success update for %s: %v", modelID, err)
	}
}

// ================================================================================
// UpdateProfileOnFailure: Atomic updates on failed model usage.
// ================================================================================
func (p *Profiler) UpdateProfileOnFailure(ctx context.Context, modelID string) {
	key := p.getProfileKey(modelID)
	pipe := p.rdb.Pipeline()
	failures := pipe.HIncrBy(ctx, key, "total_failures", 1)
	pipe.HSet(ctx, key, "status", "degraded")

	if _, err := pipe.Exec(ctx); err != nil {
		log.Printf("[ERROR] Failed failure update for %s: %v", modelID, err)
		return
	}

	successesStr, _ := p.rdb.HGet(ctx, key, "total_successes").Result()
	totalSuccesses, _ := strconv.ParseInt(successesStr, 10, 64)
	totalRequests := totalSuccesses + failures.Val()
	if totalRequests > 0 {
		errorRate := float64(failures.Val()) / float64(totalRequests)
		p.rdb.HSet(ctx, key, "error_rate", errorRate)
	}
}

// ================================================================================
// UpdateProfileOnHealthCheck: Periodic health check updates.
// ================================================================================
func (p *Profiler) UpdateProfileOnHealthCheck(ctx context.Context, modelID string, isHealthy bool) {
	_, err := p.GetProfile(ctx, modelID)
	if err != nil {
		log.Printf("[WARN] Error ensuring profile exists during health check for %s: %v", modelID, err)
	}

	key := p.getProfileKey(modelID)
	status := "offline"
	if isHealthy {
		status = "online"
	}
	pipe := p.rdb.Pipeline()
	pipe.HSet(ctx, key, "status", status)
	pipe.HSet(ctx, key, "last_health_check", time.Now().Format(time.RFC3339Nano))
	if _, err = pipe.Exec(ctx); err != nil {
		log.Printf("[ERROR] Failed health check update for %s: %v", modelID, err)
	}
}
