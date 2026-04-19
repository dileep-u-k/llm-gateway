package llm

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"time"

	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/dileep-u-k/llm-gateway/internal/observability"
	"github.com/redis/go-redis/v9"
)

type ModelProfile struct {
	ModelID             string    `json:"model_id" redis:"model_id"`
	Provider            string    `json:"provider" redis:"provider"`
	AvgLatencyMS        int64     `json:"avg_latency_ms" redis:"avg_latency_ms"`
	CostPerInputToken   float64   `json:"cost_per_input_token" redis:"cost_per_input_token"`
	CostPerOutputToken  float64   `json:"cost_per_output_token" redis:"cost_per_output_token"`
	Status              string    `json:"status" redis:"status"`
	ErrorRate           float64   `json:"error_rate" redis:"error_rate"`
	TotalSuccesses      int64     `json:"total_successes" redis:"total_successes"`
	TotalFailures       int64     `json:"total_failures" redis:"total_failures"`
	TotalInputTokens    int64     `json:"total_input_tokens" redis:"total_input_tokens"`
	TotalOutputTokens   int64     `json:"total_output_tokens" redis:"total_output_tokens"`
	LastHealthCheck     time.Time `json:"last_health_check" redis:"last_health_check"`
	CostSpentMonthly    float64   `json:"cost_spent_monthly"`
	ConsecutiveFailures int64     `json:"consecutive_failures" redis:"consecutive_failures"`
	CircuitOpenUntil    time.Time `json:"circuit_open_until" redis:"circuit_open_until"`
	LastError           string    `json:"last_error" redis:"last_error"`
}

var modelCosts = make(map[string]map[string]float64)

func InitializeModelCosts(costs map[string]map[string]float64) {
	modelCosts = costs
	for modelID, costData := range modelCosts {
		log.Printf("[INFO] Loaded cost config for %s: Input=$%.8f/token, Output=$%.8f/token",
			modelID, costData["input"], costData["output"])
	}
}

func EstimateRequestCost(modelID string, usage api.Usage, artifactCount int) float64 {
	costs, ok := modelCosts[modelID]
	if !ok {
		return 0
	}
	cost := float64(usage.PromptTokens)*costs["input"] + float64(usage.CompletionTokens)*costs["output"]
	if cost == 0 && artifactCount > 0 {
		cost = costs["input"] * float64(artifactCount)
	}
	if cost < 0 {
		return 0
	}
	return cost
}

type Profiler struct {
	rdb    *redis.Client
	config *RouterConfig
}

func NewProfiler(rdb *redis.Client, config *RouterConfig) *Profiler {
	return &Profiler{rdb: rdb, config: config}
}

func (p *Profiler) getProfileKey(modelID string) string {
	return fmt.Sprintf("profile:%s", modelID)
}

func (p *Profiler) getProviderHealthKey(provider string) string {
	return fmt.Sprintf("provider_health:%s", provider)
}

func (p *Profiler) getCapabilityHealthKey(provider, modelID string, capability Capability) string {
	return fmt.Sprintf("capability_health:%s:%s:%s", provider, modelID, capability)
}

func (p *Profiler) GetProfile(ctx context.Context, modelID string) (*ModelProfile, error) {
	key := p.getProfileKey(modelID)
	profileData, err := p.rdb.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get profile for %s: %w", modelID, err)
	}

	if len(profileData) == 0 {
		return p.createDefaultProfile(ctx, modelID)
	}

	profile := &ModelProfile{ModelID: modelID, Provider: ProviderForModel(modelID)}
	profile.AvgLatencyMS = parseInt64(profileData["avg_latency_ms"], 2000)
	profile.CostPerInputToken = parseFloat64(profileData["cost_per_input_token"], 0)
	profile.CostPerOutputToken = parseFloat64(profileData["cost_per_output_token"], 0)
	profile.Status = string(normalizeHealthStatus(profileData["status"]))
	profile.ErrorRate = parseFloat64(profileData["error_rate"], 0)
	profile.TotalSuccesses = parseInt64(profileData["total_successes"], 1)
	profile.TotalFailures = parseInt64(profileData["total_failures"], 0)
	profile.TotalInputTokens = parseInt64(profileData["total_input_tokens"], 0)
	profile.TotalOutputTokens = parseInt64(profileData["total_output_tokens"], 0)
	profile.LastHealthCheck = parseTime(profileData["last_health_check"])
	profile.ConsecutiveFailures = parseInt64(profileData["consecutive_failures"], 0)
	profile.CircuitOpenUntil = parseTime(profileData["circuit_open_until"])
	profile.LastError = profileData["last_error"]

	costKey := fmt.Sprintf("cost:%s:%s", modelID, time.Now().Format("2006-01"))
	profile.CostSpentMonthly, _ = p.rdb.Get(ctx, costKey).Float64()

	return profile, nil
}

func (p *Profiler) createDefaultProfile(ctx context.Context, modelID string) (*ModelProfile, error) {
	costs, ok := modelCosts[modelID]
	if !ok {
		log.Printf("[CRITICAL] No cost config for model '%s'. Defaulting to zero cost.", modelID)
		costs = map[string]float64{"input": 0, "output": 0}
	}

	profile := &ModelProfile{
		ModelID:            modelID,
		Provider:           ProviderForModel(modelID),
		AvgLatencyMS:       2000,
		CostPerInputToken:  costs["input"],
		CostPerOutputToken: costs["output"],
		Status:             string(HealthStatusOnline),
		TotalSuccesses:     1,
		TotalFailures:      0,
		ErrorRate:          0.0,
		LastHealthCheck:    time.Now().UTC(),
	}

	key := p.getProfileKey(modelID)
	wasSet, err := p.rdb.HSetNX(ctx, key, "model_id", profile.ModelID).Result()
	if err != nil {
		return nil, fmt.Errorf("redis HSetNX failed for %s: %w", modelID, err)
	}

	if wasSet {
		pipe := p.rdb.Pipeline()
		pipe.HSet(ctx, key,
			"provider", profile.Provider,
			"avg_latency_ms", profile.AvgLatencyMS,
			"cost_per_input_token", profile.CostPerInputToken,
			"cost_per_output_token", profile.CostPerOutputToken,
			"status", profile.Status,
			"total_successes", profile.TotalSuccesses,
			"total_failures", profile.TotalFailures,
			"error_rate", profile.ErrorRate,
			"last_health_check", profile.LastHealthCheck.Format(time.RFC3339Nano),
			"consecutive_failures", 0,
			"circuit_open_until", "",
			"last_error", "",
		)
		if _, err := pipe.Exec(ctx); err != nil {
			return nil, fmt.Errorf("failed to populate new profile for %s: %w", modelID, err)
		}
		log.Printf("[INFO] Created new profile for %s", modelID)
	} else {
		return p.GetProfile(ctx, modelID)
	}

	return profile, nil
}

func (p *Profiler) UpdateProfileOnSuccess(ctx context.Context, modelID string, latency time.Duration, usage api.Usage) {
	key := p.getProfileKey(modelID)
	const alpha = 0.1

	results, err := p.rdb.HMGet(ctx, key, "avg_latency_ms", "total_failures").Result()
	if err != nil {
		log.Printf("[ERROR] Failed to fetch profile data for success update %s: %v", modelID, err)
		return
	}
	currentLatency, _ := strconv.ParseInt(fmt.Sprint(results[0]), 10, 64)
	totalFailures, _ := strconv.ParseInt(fmt.Sprint(results[1]), 10, 64)
	if currentLatency == 0 {
		currentLatency = latency.Milliseconds()
	}

	newLatency := int64((alpha * float64(latency.Milliseconds())) + ((1.0 - alpha) * float64(currentLatency)))

	pipe := p.rdb.Pipeline()
	successes := pipe.HIncrBy(ctx, key, "total_successes", 1)
	pipe.HIncrBy(ctx, key, "total_input_tokens", int64(usage.PromptTokens))
	pipe.HIncrBy(ctx, key, "total_output_tokens", int64(usage.CompletionTokens))
	pipe.HSet(ctx, key,
		"avg_latency_ms", newLatency,
		"status", string(HealthStatusOnline),
		"last_error", "",
		"consecutive_failures", 0,
		"circuit_open_until", "",
		"last_health_check", time.Now().UTC().Format(time.RFC3339Nano),
	)

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

func (p *Profiler) UpdateProfileOnFailure(ctx context.Context, modelID string, err error) {
	profile, profileErr := p.GetProfile(ctx, modelID)
	if profileErr != nil {
		log.Printf("[ERROR] Failed to fetch profile for failure update %s: %v", modelID, profileErr)
		return
	}

	key := p.getProfileKey(modelID)
	pipe := p.rdb.Pipeline()
	failures := pipe.HIncrBy(ctx, key, "total_failures", 1)
	consecutiveFailures := pipe.HIncrBy(ctx, key, "consecutive_failures", 1)
	status := string(HealthStatusDegraded)
	circuitOpenUntil := ""
	if consecutiveFailures.Val() >= p.circuitBreakerFailureThreshold() {
		status = string(HealthStatusOffline)
		circuitOpenUntil = time.Now().UTC().Add(p.circuitBreakerCooldown()).Format(time.RFC3339Nano)
	}
	pipe.HSet(ctx, key,
		"status", status,
		"last_error", stringifyError(err),
		"circuit_open_until", circuitOpenUntil,
		"last_health_check", time.Now().UTC().Format(time.RFC3339Nano),
	)

	if _, execErr := pipe.Exec(ctx); execErr != nil {
		log.Printf("[ERROR] Failed failure update for %s: %v", modelID, execErr)
		return
	}

	totalRequests := profile.TotalSuccesses + failures.Val()
	if totalRequests > 0 {
		errorRate := float64(failures.Val()) / float64(totalRequests)
		if setErr := p.rdb.HSet(ctx, key, "error_rate", errorRate).Err(); setErr != nil {
			log.Printf("[ERROR] Failed to update error rate for %s: %v", modelID, setErr)
		}
	}
}

func (p *Profiler) UpdateModelHealthCheck(ctx context.Context, modelID string, probe HealthProbeResult) {
	profile, err := p.GetProfile(ctx, modelID)
	if err != nil {
		log.Printf("[WARN] Error ensuring profile exists during model health check for %s: %v", modelID, err)
		return
	}

	status := probe.Status
	if status == "" {
		status, probe.AccessAllowed = classifyProbeError(probe.Err)
	}
	if !probe.AccessAllowed {
		status = HealthStatusOffline
	}

	key := p.getProfileKey(modelID)
	previousStatus := normalizeHealthStatus(profile.Status)
	pipe := p.rdb.Pipeline()
	fields := []interface{}{
		"status", string(status),
		"last_health_check", time.Now().UTC().Format(time.RFC3339Nano),
		"last_error", stringifyError(probe.Err),
	}
	if probe.Latency > 0 {
		fields = append(fields, "avg_latency_ms", probe.Latency.Milliseconds())
	}
	if status == HealthStatusOnline {
		fields = append(fields, "consecutive_failures", 0, "circuit_open_until", "")
	} else {
		fields = append(fields, "consecutive_failures", profile.ConsecutiveFailures+1)
		if profile.ConsecutiveFailures+1 >= p.circuitBreakerFailureThreshold() {
			fields = append(fields, "circuit_open_until", time.Now().UTC().Add(p.circuitBreakerCooldown()).Format(time.RFC3339Nano))
		}
	}
	pipe.HSet(ctx, key, fields...)
	if _, err := pipe.Exec(ctx); err != nil {
		log.Printf("[ERROR] Failed model health check update for %s: %v", modelID, err)
		return
	}
	if previousStatus != status {
		observability.Default().RecordHealthTransition("model", modelID, string(previousStatus), string(status))
	}
}

func (p *Profiler) UpdateProviderHealthCheck(ctx context.Context, provider string, probe HealthProbeResult) {
	key := p.getProviderHealthKey(provider)
	current, _ := p.GetProviderHealth(ctx, provider)
	status := probe.Status
	if status == "" {
		status, probe.AccessAllowed = classifyProbeError(probe.Err)
	}
	if !probe.AccessAllowed {
		status = HealthStatusOffline
	}
	fields := map[string]interface{}{
		"provider":             provider,
		"status":               string(status),
		"access_allowed":       probe.AccessAllowed,
		"last_health_check":    time.Now().UTC().Format(time.RFC3339Nano),
		"last_error":           stringifyError(probe.Err),
		"consecutive_failures": 0,
	}
	if probe.Latency > 0 {
		fields["avg_latency_ms"] = probe.Latency.Milliseconds()
	}
	if status != HealthStatusOnline {
		fields["consecutive_failures"] = current.ConsecutiveFailures + 1
	}
	if err := p.rdb.HSet(ctx, key, fields).Err(); err != nil {
		log.Printf("[ERROR] Failed provider health update for %s: %v", provider, err)
		return
	}
	if current.Status != status {
		observability.Default().RecordHealthTransition("provider", provider, string(current.Status), string(status))
	}
}

func (p *Profiler) UpdateCapabilityHealthCheck(ctx context.Context, provider, modelID string, capability Capability, probe HealthProbeResult) {
	key := p.getCapabilityHealthKey(provider, modelID, capability)
	current, err := p.GetCapabilityHealth(ctx, provider, modelID, capability)
	if err != nil || current == nil {
		current = &CapabilityHealth{Status: HealthStatusOnline}
	}
	status := probe.Status
	if status == "" {
		status, probe.AccessAllowed = classifyProbeError(probe.Err)
	}
	if !probe.AccessAllowed {
		status = HealthStatusOffline
	}
	fields := map[string]interface{}{
		"provider":             provider,
		"model_id":             modelID,
		"capability":           string(capability),
		"status":               string(status),
		"access_allowed":       probe.AccessAllowed,
		"supported":            true,
		"last_health_check":    time.Now().UTC().Format(time.RFC3339Nano),
		"last_error":           stringifyError(probe.Err),
		"consecutive_failures": 0,
	}
	if probe.Latency > 0 {
		fields["avg_latency_ms"] = probe.Latency.Milliseconds()
	}
	if status != HealthStatusOnline {
		fields["consecutive_failures"] = current.ConsecutiveFailures + 1
	}
	if err := p.rdb.HSet(ctx, key, fields).Err(); err != nil {
		log.Printf("[ERROR] Failed capability health update for %s/%s/%s: %v", provider, modelID, capability, err)
		return
	}
	if current.Status != status {
		observability.Default().RecordHealthTransition("capability", BuildCapabilityHealthID(provider, modelID, capability), string(current.Status), string(status))
	}
}

func (p *Profiler) GetProviderHealth(ctx context.Context, provider string) (*ProviderHealth, error) {
	key := p.getProviderHealthKey(provider)
	data, err := p.rdb.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		defaultHealth := &ProviderHealth{
			Provider:        provider,
			Status:          HealthStatusOnline,
			AccessAllowed:   true,
			LastHealthCheck: time.Now().UTC(),
		}
		if err := p.rdb.HSet(ctx, key,
			"provider", provider,
			"status", string(defaultHealth.Status),
			"access_allowed", defaultHealth.AccessAllowed,
			"last_health_check", defaultHealth.LastHealthCheck.Format(time.RFC3339Nano),
		).Err(); err != nil {
			return nil, err
		}
		return defaultHealth, nil
	}
	return &ProviderHealth{
		Provider:            provider,
		Status:              normalizeHealthStatus(data["status"]),
		AccessAllowed:       parseBool(data["access_allowed"], true),
		AvgLatencyMS:        parseInt64(data["avg_latency_ms"], 0),
		LastHealthCheck:     parseTime(data["last_health_check"]),
		ConsecutiveFailures: parseInt64(data["consecutive_failures"], 0),
		LastError:           data["last_error"],
	}, nil
}

func (p *Profiler) GetCapabilityHealth(ctx context.Context, provider, modelID string, capability Capability) (*CapabilityHealth, error) {
	key := p.getCapabilityHealthKey(provider, modelID, capability)
	data, err := p.rdb.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		defaultHealth := &CapabilityHealth{
			Provider:        provider,
			ModelID:         modelID,
			Capability:      capability,
			Status:          HealthStatusOnline,
			AccessAllowed:   true,
			Supported:       true,
			LastHealthCheck: time.Now().UTC(),
		}
		if err := p.rdb.HSet(ctx, key,
			"provider", provider,
			"model_id", modelID,
			"capability", string(capability),
			"status", string(defaultHealth.Status),
			"access_allowed", defaultHealth.AccessAllowed,
			"supported", defaultHealth.Supported,
			"last_health_check", defaultHealth.LastHealthCheck.Format(time.RFC3339Nano),
		).Err(); err != nil {
			return nil, err
		}
		return defaultHealth, nil
	}
	return &CapabilityHealth{
		Provider:            provider,
		ModelID:             modelID,
		Capability:          Capability(data["capability"]),
		Status:              normalizeHealthStatus(data["status"]),
		AccessAllowed:       parseBool(data["access_allowed"], true),
		Supported:           parseBool(data["supported"], true),
		AvgLatencyMS:        parseInt64(data["avg_latency_ms"], 0),
		ErrorRate:           parseFloat64(data["error_rate"], 0),
		LastHealthCheck:     parseTime(data["last_health_check"]),
		ConsecutiveFailures: parseInt64(data["consecutive_failures"], 0),
		LastError:           data["last_error"],
	}, nil
}

func (p *Profiler) CombinedModelHealth(ctx context.Context, modelID string, capability Capability) (HealthStatus, *ProviderHealth, *CapabilityHealth, *ModelProfile, error) {
	profile, err := p.GetProfile(ctx, modelID)
	if err != nil {
		return HealthStatusOffline, nil, nil, nil, err
	}
	providerHealth, err := p.GetProviderHealth(ctx, ProviderForModel(modelID))
	if err != nil {
		return HealthStatusOffline, nil, nil, nil, err
	}
	capabilityHealth, err := p.GetCapabilityHealth(ctx, ProviderForModel(modelID), modelID, capability)
	if err != nil {
		return HealthStatusOffline, nil, nil, nil, err
	}
	return combineHealthStatuses(normalizeHealthStatus(profile.Status), providerHealth.Status, capabilityHealth.Status), providerHealth, capabilityHealth, profile, nil
}

func (p *Profiler) healthCheckStaleness() time.Duration {
	if p == nil || p.config == nil {
		return 5 * time.Minute
	}
	if val, ok := p.config.Thresholds["health_check_staleness"].(string); ok {
		if parsed, err := time.ParseDuration(val); err == nil {
			return parsed
		}
	}
	return 5 * time.Minute
}

func (p *Profiler) circuitBreakerFailureThreshold() int64 {
	return int64(p.getThresholdInt("circuit_breaker_failure_threshold", 3))
}

func (p *Profiler) circuitBreakerCooldown() time.Duration {
	if p == nil || p.config == nil {
		return 2 * time.Minute
	}
	if raw, ok := p.config.Thresholds["circuit_breaker_cooldown"].(string); ok {
		if parsed, err := time.ParseDuration(raw); err == nil {
			return parsed
		}
	}
	return 2 * time.Minute
}

func (p *Profiler) getThresholdInt(key string, fallback int) int {
	if p == nil || p.config == nil {
		return fallback
	}
	switch v := p.config.Thresholds[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return fallback
	}
}

func parseInt64(value string, fallback int64) int64 {
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

func parseFloat64(value string, fallback float64) float64 {
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

func parseBool(value string, fallback bool) bool {
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func parseTime(value string) time.Time {
	if value == "" {
		return time.Time{}
	}
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return time.Time{}
	}
	return parsed
}

func stringifyError(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
