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

var modelCosts = make(map[string]map[string]float64)

func InitializeModelCosts(costs map[string]map[string]float64) {
	modelCosts = costs
	for modelID, costData := range modelCosts {
		log.Printf("Loaded cost config for %s: Input=$%.8f/token, Output=$%.8f/token", modelID, costData["input"], costData["output"])
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

// GetProfile retrieves a model's profile, creating a default one if it doesn't exist.
func (p *Profiler) GetProfile(ctx context.Context, modelID string) (*ModelProfile, error) {
	key := p.getProfileKey(modelID)
	profileData, err := p.rdb.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, err
	}

	if len(profileData) == 0 {
		return p.createDefaultProfile(ctx, modelID)
	}

	profile := &ModelProfile{}
	profile.ModelID = modelID
	profile.AvgLatencyMS, _ = strconv.ParseInt(profileData["avg_latency_ms"], 10, 64)
	profile.CostPerInputToken, _ = strconv.ParseFloat(profileData["cost_per_input_token"], 64)
	profile.CostPerOutputToken, _ = strconv.ParseFloat(profileData["cost_per_output_token"], 64)
	profile.Status = profileData["status"]
	profile.ErrorRate, _ = strconv.ParseFloat(profileData["error_rate"], 64)
	profile.TotalSuccesses, _ = strconv.ParseInt(profileData["total_successes"], 10, 64)
	profile.TotalFailures, _ = strconv.ParseInt(profileData["total_failures"], 10, 64)
	profile.TotalInputTokens, _ = strconv.ParseInt(profileData["total_input_tokens"], 10, 64)
	profile.TotalOutputTokens, _ = strconv.ParseInt(profileData["total_output_tokens"], 10, 64)
	profile.LastHealthCheck, _ = time.Parse(time.RFC3339Nano, profileData["last_health_check"])

	costKey := fmt.Sprintf("cost:%s:%s", modelID, time.Now().Format("2006-01"))
	profile.CostSpentMonthly, _ = p.rdb.Get(ctx, costKey).Float64()

	return profile, nil
}

func (p *Profiler) createDefaultProfile(ctx context.Context, modelID string) (*ModelProfile, error) {
	costs, ok := modelCosts[modelID]
	if !ok {
		// Return a zero-cost profile but log a critical error.
		// This prevents crashes but makes it clear that config is missing.
		log.Printf("CRITICAL: No cost information for model '%s'. Defaulting to zero cost.", modelID)
		costs = map[string]float64{"input": 0, "output": 0}
	}

	profile := &ModelProfile{
		ModelID:            modelID,
		AvgLatencyMS:       2000, // Start with a reasonable default latency.
		CostPerInputToken:  costs["input"],
		CostPerOutputToken: costs["output"],
		Status:             "online",
		TotalSuccesses:     1,
		TotalFailures:      0,
		TotalInputTokens:   0,
		TotalOutputTokens:  0,
		ErrorRate:          0.0,
		LastHealthCheck:    time.Now(),
		CostSpentMonthly:   0.0,
	}

	key := p.getProfileKey(modelID)
	pipe := p.rdb.Pipeline()
	pipe.HSet(ctx, key, "model_id", profile.ModelID)
	pipe.HSet(ctx, key, "avg_latency_ms", profile.AvgLatencyMS)
	pipe.HSet(ctx, key, "cost_per_input_token", profile.CostPerInputToken)
	pipe.HSet(ctx, key, "cost_per_output_token", profile.CostPerOutputToken)
	pipe.HSet(ctx, key, "status", profile.Status)
	pipe.HSet(ctx, key, "total_successes", profile.TotalSuccesses)
	pipe.HSet(ctx, key, "total_failures", profile.TotalFailures)
	pipe.HSet(ctx, key, "error_rate", profile.ErrorRate)
	pipe.HSet(ctx, key, "last_health_check", profile.LastHealthCheck.Format(time.RFC3339Nano))
	_, err := pipe.Exec(ctx)

	log.Printf("Created new, complete profile for %s", modelID)
	return profile, err
}

func (p *Profiler) UpdateProfileOnSuccess(ctx context.Context, modelID string, latency time.Duration, usage api.Usage) {
	key := p.getProfileKey(modelID)
	const alpha = 0.1

	err := p.rdb.Watch(ctx, func(tx *redis.Tx) error {
		currentLatencyStr, err := tx.HGet(ctx, key, "avg_latency_ms").Result()
		if err != nil && err != redis.Nil {
			return err
		}
		currentLatency, _ := strconv.ParseInt(currentLatencyStr, 10, 64)
		newLatency := int64((alpha * float64(latency.Milliseconds())) + ((1.0 - alpha) * float64(currentLatency)))
		_, err = tx.Pipelined(ctx, func(pipe redis.Pipeliner) error {
			pipe.HSet(ctx, key, "avg_latency_ms", newLatency)
			return nil
		})
		return err
	}, key)
	if err != nil {
		log.Printf("Error updating latency for %s: %v", modelID, err)
	}

	pipe := p.rdb.Pipeline()
	successes := pipe.HIncrBy(ctx, key, "total_successes", 1)
	failures := pipe.HGet(ctx, key, "total_failures")
	pipe.HIncrBy(ctx, key, "total_input_tokens", int64(usage.PromptTokens))
	pipe.HIncrBy(ctx, key, "total_output_tokens", int64(usage.CompletionTokens))
	pipe.HSet(ctx, key, "status", "online")

	callCost := (float64(usage.PromptTokens) * modelCosts[modelID]["input"]) + (float64(usage.CompletionTokens) * modelCosts[modelID]["output"])
	costKey := fmt.Sprintf("cost:%s:%s", modelID, time.Now().Format("2006-01"))
	pipe.IncrByFloat(ctx, costKey, callCost)
	pipe.Expire(ctx, costKey, 35*24*time.Hour)

	_, err = pipe.Exec(ctx)
	if err != nil {
		log.Printf("Error in success update pipeline for %s: %v", modelID, err)
		return
	}

	totalFailures, _ := strconv.ParseInt(failures.Val(), 10, 64)
	totalRequests := successes.Val() + totalFailures
	if totalRequests > 0 {
		errorRate := float64(totalFailures) / float64(totalRequests)
		p.rdb.HSet(ctx, key, "error_rate", errorRate)
	}
}

func (p *Profiler) UpdateProfileOnFailure(ctx context.Context, modelID string) {
	key := p.getProfileKey(modelID)
	pipe := p.rdb.Pipeline()
	failures := pipe.HIncrBy(ctx, key, "total_failures", 1)
	successes := pipe.HGet(ctx, key, "total_successes")
	pipe.HSet(ctx, key, "status", "degraded")

	_, err := pipe.Exec(ctx)
	if err != nil {
		log.Printf("Error in failure update pipeline for %s: %v", modelID, err)
		return
	}

	totalSuccesses, _ := strconv.ParseInt(successes.Val(), 10, 64)
	totalRequests := totalSuccesses + failures.Val()
	if totalRequests > 0 {
		errorRate := float64(failures.Val()) / float64(totalRequests)
		p.rdb.HSet(ctx, key, "error_rate", errorRate)
	}
}

// UpdateProfileOnHealthCheck updates status based on a proactive check.
// *** THIS IS THE FIX ***
// It now ensures a full profile exists before writing health status to prevent creating partial profiles.
func (p *Profiler) UpdateProfileOnHealthCheck(ctx context.Context, modelID string, isHealthy bool) {
	// First, ensure a profile exists. GetProfile will create one if it's missing.
	// This is the critical fix that prevents the health checker from creating partial, empty profiles.
	_, err := p.GetProfile(ctx, modelID)
	if err != nil {
		// Log the error but continue, as setting the health status is still important.
		log.Printf("Error ensuring profile exists during health check for %s: %v", modelID, err)
	}

	key := p.getProfileKey(modelID)
	status := "offline"
	if isHealthy {
		status = "online"
	}

	pipe := p.rdb.Pipeline()
	pipe.HSet(ctx, key, "status", status)
	pipe.HSet(ctx, key, "last_health_check", time.Now().Format(time.RFC3339Nano))
	_, err = pipe.Exec(ctx)

	if err != nil {
		log.Printf("Error updating health check for %s: %v", modelID, err)
	}
}
