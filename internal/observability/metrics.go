package observability

import (
	"math"
	"sort"
	"sync"
	"time"
)

const (
	maxLatencySamples = 512
	maxEventEntries   = 200
	maxTraceSpans     = 32
)

type LogEntry struct {
	Timestamp string            `json:"timestamp,omitempty"`
	Level     string            `json:"level,omitempty"`
	Category  string            `json:"category,omitempty"`
	Message   string            `json:"message,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type TraceSpan struct {
	Name       string            `json:"name,omitempty"`
	Status     string            `json:"status,omitempty"`
	StartedAt  string            `json:"started_at,omitempty"`
	FinishedAt string            `json:"finished_at,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

type TraceRecord struct {
	TraceID     string      `json:"trace_id,omitempty"`
	Kind        string      `json:"kind,omitempty"`
	Status      string      `json:"status,omitempty"`
	RequestID   string      `json:"request_id,omitempty"`
	StartedAt   string      `json:"started_at,omitempty"`
	FinishedAt  string      `json:"finished_at,omitempty"`
	Spans       []TraceSpan `json:"spans,omitempty"`
	CurrentSpan string      `json:"current_span,omitempty"`
}

type Registry struct {
	mu               sync.Mutex
	counters         map[string]int64
	requests         map[string]int64
	errors           map[string]int64
	healthChanges    map[string]int64
	latencySamples   []int64
	retrievalTimings []int64
	jobDurations     []int64
	costTotals       map[string]float64
	modalityCounts   map[string]int64
	stageCounts      map[string]int64
	routeSelections  map[string]int64
	forceScopes      map[string]int64
	jobStates        map[string]int64
	logs             []LogEntry
	traces           []TraceRecord
	activeTraces     map[string]TraceRecord
}

func NewRegistry() *Registry {
	return &Registry{
		counters:        make(map[string]int64),
		requests:        make(map[string]int64),
		errors:          make(map[string]int64),
		healthChanges:   make(map[string]int64),
		costTotals:      make(map[string]float64),
		modalityCounts:  make(map[string]int64),
		stageCounts:     make(map[string]int64),
		routeSelections: make(map[string]int64),
		forceScopes:     make(map[string]int64),
		jobStates:       make(map[string]int64),
		activeTraces:    make(map[string]TraceRecord),
	}
}

var defaultRegistry = NewRegistry()

func Default() *Registry {
	return defaultRegistry
}

func (r *Registry) IncRequest(provider, model string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.requests[provider+"|"+model]++
}

func (r *Registry) IncError(provider, model string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.errors[provider+"|"+model]++
}

func (r *Registry) IncCounter(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.counters[name]++
}

func (r *Registry) AddCounter(name string, delta int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.counters[name] += delta
}

func (r *Registry) ObserveLatency(d time.Duration) {
	r.observe(&r.latencySamples, d)
}

func (r *Registry) ObserveRetrievalLatency(d time.Duration) {
	r.observe(&r.retrievalTimings, d)
}

func (r *Registry) ObserveJobDuration(d time.Duration) {
	r.observe(&r.jobDurations, d)
}

func (r *Registry) ObserveCost(provider, taskType, modality string, cost float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := provider + "|" + taskType + "|" + modality
	r.costTotals[key] += maxFloat(cost, 0)
}

func (r *Registry) RecordModality(modality string) {
	if modality == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.modalityCounts[modality]++
}

func (r *Registry) RecordStage(stageType, status string) {
	if stageType == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	key := stageType
	if status != "" {
		key += "|" + status
	}
	r.stageCounts[key]++
}

func (r *Registry) RecordRoute(provider, model, strategy, routeFamily, forceScope string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.routeSelections[provider+"|"+model+"|"+strategy+"|"+routeFamily]++
	if forceScope == "" {
		forceScope = "dynamic"
	}
	r.forceScopes[forceScope]++
}

func (r *Registry) RecordJobState(state string) {
	if state == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.jobStates[state]++
}

func (r *Registry) RecordLog(level, category, message string, metadata map[string]string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	entry := LogEntry{
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		Level:     level,
		Category:  category,
		Message:   message,
		Metadata:  cloneStringMap(metadata),
	}
	r.logs = append(r.logs, entry)
	if len(r.logs) > maxEventEntries {
		r.logs = append([]LogEntry(nil), r.logs[len(r.logs)-maxEventEntries:]...)
	}
}

func (r *Registry) StartTrace(kind, requestID string) string {
	traceID := requestID
	if traceID == "" {
		traceID = time.Now().UTC().Format("20060102150405.000000000")
	}
	record := TraceRecord{
		TraceID:   traceID,
		Kind:      kind,
		Status:    "running",
		RequestID: requestID,
		StartedAt: time.Now().UTC().Format(time.RFC3339Nano),
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.activeTraces[traceID] = record
	return traceID
}

func (r *Registry) RecordTraceSpan(traceID, name, status string, metadata map[string]string) {
	if traceID == "" || name == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	record, ok := r.activeTraces[traceID]
	if !ok {
		record = TraceRecord{
			TraceID:   traceID,
			Status:    "running",
			StartedAt: time.Now().UTC().Format(time.RFC3339Nano),
		}
	}
	record.CurrentSpan = name
	record.Spans = append(record.Spans, TraceSpan{
		Name:       name,
		Status:     status,
		StartedAt:  time.Now().UTC().Format(time.RFC3339Nano),
		FinishedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Metadata:   cloneStringMap(metadata),
	})
	if len(record.Spans) > maxTraceSpans {
		record.Spans = append([]TraceSpan(nil), record.Spans[len(record.Spans)-maxTraceSpans:]...)
	}
	r.activeTraces[traceID] = record
}

func (r *Registry) FinishTrace(traceID, status string) {
	if traceID == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	record, ok := r.activeTraces[traceID]
	if !ok {
		return
	}
	record.Status = firstNonEmpty(status, "completed")
	record.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
	record.CurrentSpan = ""
	delete(r.activeTraces, traceID)
	r.traces = append(r.traces, record)
	if len(r.traces) > maxEventEntries {
		r.traces = append([]TraceRecord(nil), r.traces[len(r.traces)-maxEventEntries:]...)
	}
}

func (r *Registry) observe(samples *[]int64, d time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	ms := d.Milliseconds()
	if ms < 0 {
		ms = 0
	}
	*samples = append(*samples, ms)
	if len(*samples) > maxLatencySamples {
		*samples = append([]int64(nil), (*samples)[len(*samples)-maxLatencySamples:]...)
	}
}

func (r *Registry) RecordHealthTransition(kind, target, from, to string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := kind + "|" + target + "|" + from + "->" + to
	r.healthChanges[key]++
}

func (r *Registry) Snapshot() map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()

	return map[string]any{
		"generated_at":                   time.Now().UTC().Format(time.RFC3339),
		"counters":                       cloneMap(r.counters),
		"requests_per_provider_model":    cloneMap(r.requests),
		"errors_per_provider_model":      cloneMap(r.errors),
		"error_rate_per_provider_model":  computeRates(r.errors, r.requests),
		"health_transitions":             cloneMap(r.healthChanges),
		"cache_hit_rate":                 hitRate(r.counters["response_cache_hits"], r.counters["response_cache_misses"]),
		"rag_hit_rate":                   hitRate(r.counters["rag_hits"], r.counters["rag_misses"]),
		"latency_ms":                     quantiles(r.latencySamples),
		"retrieval_latency_ms":           quantiles(r.retrievalTimings),
		"job_duration_ms":                quantiles(r.jobDurations),
		"cost_by_provider_task_modality": cloneFloatMap(r.costTotals),
		"modality_distribution":          cloneMap(r.modalityCounts),
		"stage_distribution":             cloneMap(r.stageCounts),
		"route_selection_distribution":   cloneMap(r.routeSelections),
		"force_scope_distribution":       cloneMap(r.forceScopes),
		"job_state_distribution":         cloneMap(r.jobStates),
		"recent_logs":                    append([]LogEntry(nil), r.logs...),
		"recent_traces":                  append([]TraceRecord(nil), r.traces...),
		"dashboards":                     r.dashboardSnapshotLocked(),
	}
}

func (r *Registry) dashboardSnapshotLocked() map[string]any {
	return map[string]any{
		"provider_health": map[string]any{
			"requests": cloneMap(r.requests),
			"errors":   cloneMap(r.errors),
		},
		"routing": map[string]any{
			"routes":      cloneMap(r.routeSelections),
			"force_scope": cloneMap(r.forceScopes),
		},
		"latency": map[string]any{
			"request_latency_ms":   quantiles(r.latencySamples),
			"retrieval_latency_ms": quantiles(r.retrievalTimings),
			"job_duration_ms":      quantiles(r.jobDurations),
		},
		"failover": map[string]any{
			"failovers": cloneMap(filterCountersByPrefix(r.counters, "failover")),
			"retries":   cloneMap(filterCountersByPrefix(r.counters, "async_retry")),
		},
		"rag": map[string]any{
			"rag_hit_rate": hitRate(r.counters["rag_hits"], r.counters["rag_misses"]),
			"stale_skips":  r.counters["rag_stale_source_skips"],
		},
		"generation": map[string]any{
			"stage_distribution": cloneMap(filterStagesByName(r.stageCounts, "image_", "speech_", "creative_", "quality_", "asset_bundle")),
		},
		"async_jobs": map[string]any{
			"states": cloneMap(r.jobStates),
		},
		"cost": map[string]any{
			"totals": cloneFloatMap(r.costTotals),
		},
	}
}

func cloneMap(in map[string]int64) map[string]int64 {
	out := make(map[string]int64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneFloatMap(in map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func quantiles(samples []int64) map[string]any {
	if len(samples) == 0 {
		return map[string]any{"count": 0, "p50": int64(0), "p95": int64(0), "p99": int64(0)}
	}
	copySamples := append([]int64(nil), samples...)
	sort.Slice(copySamples, func(i, j int) bool { return copySamples[i] < copySamples[j] })
	return map[string]any{
		"count": len(copySamples),
		"p50":   percentile(copySamples, 0.50),
		"p95":   percentile(copySamples, 0.95),
		"p99":   percentile(copySamples, 0.99),
	}
}

func computeRates(numerators, denominators map[string]int64) map[string]float64 {
	out := make(map[string]float64, len(denominators))
	for key, denominator := range denominators {
		if denominator <= 0 {
			out[key] = 0
			continue
		}
		out[key] = float64(numerators[key]) / float64(denominator)
	}
	return out
}

func hitRate(hits, misses int64) float64 {
	total := hits + misses
	if total <= 0 {
		return 0
	}
	return float64(hits) / float64(total)
}

func percentile(samples []int64, p float64) int64 {
	if len(samples) == 0 {
		return 0
	}
	idx := int(math.Round(float64(len(samples)-1) * p))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(samples) {
		idx = len(samples) - 1
	}
	return samples[idx]
}

func filterCountersByPrefix(in map[string]int64, prefix string) map[string]int64 {
	out := make(map[string]int64)
	for key, value := range in {
		if len(key) >= len(prefix) && key[:len(prefix)] == prefix {
			out[key] = value
		}
	}
	return out
}

func filterStagesByName(in map[string]int64, prefixes ...string) map[string]int64 {
	out := make(map[string]int64)
	for key, value := range in {
		for _, prefix := range prefixes {
			if len(key) >= len(prefix) && key[:len(prefix)] == prefix {
				out[key] = value
				break
			}
		}
	}
	return out
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
