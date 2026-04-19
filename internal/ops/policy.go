package ops

import (
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/dileep-u-k/llm-gateway/internal/llm"
)

type CanaryPolicyEngine struct{}

func (e CanaryPolicyEngine) Decide(req api.GenerationRequest) (bool, string, int) {
	model := strings.TrimSpace(req.Rollout.CanaryModel)
	percent := req.Rollout.CanaryPercent
	if model == "" || percent <= 0 {
		return false, "", -1
	}
	if percent > 100 {
		percent = 100
	}
	seed := llm.GenerateCacheKey(strings.Join([]string{req.UserID, req.ConversationID, req.Prompt}, "|"))
	bucket := 0
	for _, ch := range seed[:minInt(8, len(seed))] {
		bucket += int(ch)
	}
	bucket = bucket % 100
	return bucket < percent, model, bucket
}

type Evaluator struct{}

func (Evaluator) Evaluate(req api.GenerationRequest, prepared *llm.PreparedExecution, resp api.GenerationResponse) *api.EvaluationMetadata {
	if !req.Evaluation.Enabled && len(req.Evaluation.Suites) == 0 {
		return nil
	}
	suites := append([]string(nil), req.Evaluation.Suites...)
	if len(suites) == 0 {
		suites = []string{"default"}
	}

	domainScores := map[string]float64{
		"routing":     routingScore(resp),
		"retrieval":   retrievalScore(prepared, resp),
		"memory":      memoryScore(resp),
		"tools":       toolsScore(prepared, resp),
		"multimodal":  multimodalScore(prepared, resp),
		"generation":  generationScore(prepared, resp),
		"reliability": reliabilityScore(resp),
	}
	overall := 0.0
	for _, score := range domainScores {
		overall += score
	}
	overall /= float64(len(domainScores))

	baselines := req.Evaluation.Baselines
	if len(baselines) == 0 {
		baselines = []string{
			"no-routing baseline",
			"single-provider baseline",
			"no-RAG baseline",
			"no-memory baseline",
			"no-tool baseline",
			"no-generation-refinement baseline",
			"no-failover baseline",
		}
	}

	var comparisons []api.BaselineComparison
	for _, baseline := range baselines {
		comparisons = append(comparisons, baselineComparison(baseline, domainScores, resp))
	}

	notes := []string{
		"Routing score reflects explicit route metadata, strategy visibility, and forced-scope explainability.",
		"Reliability score rewards clean completion, low failover churn, and observable execution metadata.",
	}
	if prepared != nil && prepared.Plan != nil && prepared.Plan.RequiresAsync {
		notes = append(notes, "Async-eligible plan detected and included in evaluation weighting.")
	}

	return &api.EvaluationMetadata{
		Suites:              suites,
		OverallScore:        roundScore(overall),
		DomainScores:        roundScores(domainScores),
		BaselineComparisons: comparisons,
		Notes:               notes,
	}
}

func CompareResponses(mode string, primary api.GenerationResponse, shadow api.GenerationResponse, replaySourceID string, primaryLatency, shadowLatency time.Duration) *api.RolloutMetadata {
	return &api.RolloutMetadata{
		Mode:             mode,
		AppliedModel:     primary.ModelUsed,
		ShadowModel:      shadow.ModelUsed,
		PrimaryLatencyMS: primaryLatency.Milliseconds(),
		ShadowLatencyMS:  shadowLatency.Milliseconds(),
		SimilarityScore:  roundScore(similarity(primary.Content, shadow.Content)),
		OutputDelta:      summarizeDelta(primary, shadow),
		ReplaySourceID:   replaySourceID,
	}
}

func routingScore(resp api.GenerationResponse) float64 {
	score := 0.25
	if resp.Route != nil {
		score += 0.35
		if resp.Route.Strategy != "" {
			score += 0.1
		}
		if resp.Route.SelectionReason != "" {
			score += 0.1
		}
		if resp.Route.ForcedSemantics.Scope != "" || !resp.Route.ForcedSemantics.IsForced {
			score += 0.1
		}
	}
	if resp.ModelUsed != "" {
		score += 0.1
	}
	return clamp(score)
}

func retrievalScore(prepared *llm.PreparedExecution, resp api.GenerationResponse) float64 {
	if prepared == nil || !prepared.Task.NeedsRetrieval {
		return 0.75
	}
	score := 0.2
	if resp.Retrieval != nil {
		if resp.Retrieval.SelectedCount > 0 {
			score += 0.35
		}
		if len(resp.Retrieval.Sources) > 0 {
			score += 0.25
		}
	}
	if resp.Grounding != nil && resp.Grounding.EvidenceStatus != "" {
		score += 0.2
	}
	return clamp(score)
}

func memoryScore(resp api.GenerationResponse) float64 {
	if resp.Memory == nil {
		return 0.45
	}
	score := 0.4
	if resp.Memory.Summary != "" {
		score += 0.3
	}
	if len(resp.Memory.ActiveConstraints) > 0 || len(resp.Memory.KnowledgeSources) > 0 {
		score += 0.2
	}
	if resp.Memory.ConversationID != "" {
		score += 0.1
	}
	return clamp(score)
}

func toolsScore(prepared *llm.PreparedExecution, resp api.GenerationResponse) float64 {
	if prepared == nil || !prepared.Task.NeedsTooling {
		return 0.75
	}
	score := 0.2
	if resp.ToolPlan != nil && len(resp.ToolPlan.SelectedTools) > 0 {
		score += 0.2
	}
	if len(resp.ToolCalls) > 0 {
		score += 0.4
	}
	failed := 0
	for _, call := range resp.ToolCalls {
		if strings.EqualFold(call.Status, "failed") {
			failed++
		}
	}
	if failed == 0 && len(resp.ToolCalls) > 0 {
		score += 0.2
	}
	return clamp(score)
}

func multimodalScore(prepared *llm.PreparedExecution, resp api.GenerationResponse) float64 {
	if prepared == nil || len(prepared.Task.Modalities) <= 1 {
		return 0.8
	}
	score := 0.2
	if len(resp.Artifacts) > 0 {
		score += 0.4
	}
	if resp.ExecutionPlan != nil && len(resp.ExecutionPlan.Stages) > 1 {
		score += 0.2
	}
	if resp.Context != nil || resp.Generation != nil {
		score += 0.2
	}
	return clamp(score)
}

func generationScore(prepared *llm.PreparedExecution, resp api.GenerationResponse) float64 {
	if prepared == nil || !prepared.Task.RequiresGeneration {
		return 0.75
	}
	score := 0.2
	if resp.Generation != nil {
		score += 0.25
		if resp.Generation.QualityStatus != "" {
			score += 0.15
		}
	}
	if len(resp.GeneratedArtifacts) > 0 {
		score += 0.25
	}
	if resp.ImageURL != "" || resp.AudioURL != "" {
		score += 0.15
	}
	return clamp(score)
}

func reliabilityScore(resp api.GenerationResponse) float64 {
	score := 0.35
	if resp.FailoverInfo == nil {
		score += 0.15
	}
	if resp.Route != nil && resp.Route.OrchestrationID != "" {
		score += 0.15
	}
	if resp.ExecutionPlan != nil {
		score += 0.15
	}
	if resp.LatencyMS > 0 {
		score += 0.1
	}
	if resp.CacheStatus != "" {
		score += 0.1
	}
	return clamp(score)
}

func baselineComparison(name string, domainScores map[string]float64, resp api.GenerationResponse) api.BaselineComparison {
	lower := strings.ToLower(name)
	score := 0.0
	summary := "Execution beat the heuristic baseline."
	direction := "higher_is_better"
	switch {
	case strings.Contains(lower, "routing"):
		score = domainScores["routing"] - 0.45
	case strings.Contains(lower, "single-provider"):
		score = domainScores["routing"] - 0.35
	case strings.Contains(lower, "rag"):
		score = domainScores["retrieval"] - 0.4
	case strings.Contains(lower, "memory"):
		score = domainScores["memory"] - 0.35
	case strings.Contains(lower, "tool"):
		score = domainScores["tools"] - 0.35
	case strings.Contains(lower, "generation"):
		score = domainScores["generation"] - 0.35
	case strings.Contains(lower, "failover"):
		score = domainScores["reliability"] - 0.4
	default:
		score = averageScore(domainScores) - 0.4
	}
	if resp.FailoverInfo != nil && strings.Contains(lower, "failover") {
		summary = "Failover occurred, so the reliability edge over the baseline is narrower."
	}
	return api.BaselineComparison{
		Name:      name,
		Delta:     roundScore(score),
		Direction: direction,
		Summary:   summary,
	}
}

func summarizeDelta(primary, shadow api.GenerationResponse) string {
	delta := []string{}
	if primary.ModelUsed != shadow.ModelUsed {
		delta = append(delta, "model_changed="+primary.ModelUsed+"->"+shadow.ModelUsed)
	}
	if len(primary.GeneratedArtifacts) != len(shadow.GeneratedArtifacts) {
		delta = append(delta, "artifact_count="+strconv.Itoa(len(primary.GeneratedArtifacts))+"->"+strconv.Itoa(len(shadow.GeneratedArtifacts)))
	}
	if len(primary.ToolCalls) != len(shadow.ToolCalls) {
		delta = append(delta, "tool_calls="+strconv.Itoa(len(primary.ToolCalls))+"->"+strconv.Itoa(len(shadow.ToolCalls)))
	}
	if len(delta) == 0 {
		return "no material metadata delta"
	}
	return strings.Join(delta, ", ")
}

func similarity(a, b string) float64 {
	left := tokenSet(a)
	right := tokenSet(b)
	if len(left) == 0 && len(right) == 0 {
		return 1
	}
	if len(left) == 0 || len(right) == 0 {
		return 0
	}
	intersection := 0
	for token := range left {
		if right[token] {
			intersection++
		}
	}
	union := len(left) + len(right) - intersection
	if union == 0 {
		return 1
	}
	return float64(intersection) / float64(union)
}

func tokenSet(value string) map[string]bool {
	out := make(map[string]bool)
	for _, token := range strings.Fields(strings.ToLower(value)) {
		token = strings.Trim(token, ".,!?;:\"'`()[]{}")
		if token != "" {
			out[token] = true
		}
	}
	return out
}

func roundScores(in map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(in))
	for key, value := range in {
		out[key] = roundScore(value)
	}
	return out
}

func averageScore(scores map[string]float64) float64 {
	if len(scores) == 0 {
		return 0
	}
	total := 0.0
	for _, score := range scores {
		total += score
	}
	return total / float64(len(scores))
}

func clamp(value float64) float64 {
	return math.Max(0, math.Min(1, value))
}

func roundScore(value float64) float64 {
	return math.Round(clamp(value)*100) / 100
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func replayWindow(start time.Time) string {
	if start.IsZero() {
		return ""
	}
	return start.UTC().Format(time.RFC3339Nano)
}
