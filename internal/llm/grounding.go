package llm

import (
	"fmt"
	"strings"
)

type AnswerMode string

const (
	AnswerModeGroundedRequired          AnswerMode = "grounded-required"
	AnswerModeGroundedPreferred         AnswerMode = "grounded-preferred"
	AnswerModeBestEffort                AnswerMode = "best-effort"
	AnswerModeAbstainIfInsufficient     AnswerMode = "abstain-if-insufficient"
	AnswerModeSummarizeOnlyFromEvidence AnswerMode = "summarize-only-from-evidence"
	AnswerModeToolOutputPriority        AnswerMode = "tool-output-priority"
)

type EvidenceStatus string

const (
	EvidenceStatusSufficient   EvidenceStatus = "sufficient"
	EvidenceStatusWeak         EvidenceStatus = "weak"
	EvidenceStatusInsufficient EvidenceStatus = "insufficient"
)

type GroundingDecision struct {
	Mode             AnswerMode     `json:"mode"`
	RequiresEvidence bool           `json:"requires_evidence"`
	EvidenceStatus   EvidenceStatus `json:"evidence_status"`
	ShouldAbstain    bool           `json:"should_abstain"`
	Caution          string         `json:"caution,omitempty"`
	Instruction      string         `json:"instruction,omitempty"`
}

type GroundingPolicyEngine struct {
	relevanceThreshold float64
}

func NewGroundingPolicyEngine(relevanceThreshold float64) *GroundingPolicyEngine {
	if relevanceThreshold <= 0 {
		relevanceThreshold = 0.45
	}
	return &GroundingPolicyEngine{relevanceThreshold: relevanceThreshold}
}

func (g *GroundingPolicyEngine) Decide(requestedMode string, intent ExecutionIntent, retrieval *RetrievalResult, toolResults []ContextToolResult) GroundingDecision {
	mode := g.resolveMode(requestedMode, intent, toolResults)
	status := g.checkEvidence(retrieval, toolResults)
	decision := GroundingDecision{
		Mode:             mode,
		RequiresEvidence: mode != AnswerModeBestEffort,
		EvidenceStatus:   status,
	}

	switch mode {
	case AnswerModeGroundedRequired:
		decision.Instruction = "Use only grounded evidence when answering. If the evidence is weak or missing, say so clearly."
	case AnswerModeGroundedPreferred:
		decision.Instruction = "Prefer grounded evidence and mention uncertainty if the evidence is incomplete."
	case AnswerModeAbstainIfInsufficient:
		decision.Instruction = "Answer only when the evidence is sufficient. Otherwise abstain."
	case AnswerModeSummarizeOnlyFromEvidence:
		decision.Instruction = "Summarize only what is supported by the provided evidence. Do not add external facts."
	case AnswerModeToolOutputPriority:
		decision.Instruction = "Prioritize tool outputs as the primary source of truth. Use retrieval only as supporting context."
	default:
		decision.Instruction = "Provide the best answer you can while being honest about uncertainty."
	}

	if status == EvidenceStatusWeak {
		decision.Caution = "Evidence is available but weak; answer conservatively."
	}
	if status == EvidenceStatusInsufficient && decision.RequiresEvidence {
		decision.Caution = "Evidence is insufficient for a confident grounded answer."
	}

	switch mode {
	case AnswerModeGroundedRequired, AnswerModeAbstainIfInsufficient, AnswerModeSummarizeOnlyFromEvidence:
		decision.ShouldAbstain = status == EvidenceStatusInsufficient
	case AnswerModeToolOutputPriority:
		decision.ShouldAbstain = len(toolResults) == 0 && status == EvidenceStatusInsufficient
	}
	return decision
}

func (g *GroundingPolicyEngine) resolveMode(requestedMode string, intent ExecutionIntent, toolResults []ContextToolResult) AnswerMode {
	switch AnswerMode(strings.TrimSpace(requestedMode)) {
	case AnswerModeGroundedRequired, AnswerModeGroundedPreferred, AnswerModeBestEffort, AnswerModeAbstainIfInsufficient, AnswerModeSummarizeOnlyFromEvidence, AnswerModeToolOutputPriority:
		return AnswerMode(strings.TrimSpace(requestedMode))
	}
	if intent.GroundingRequired {
		return AnswerModeGroundedRequired
	}
	if len(toolResults) > 0 {
		return AnswerModeToolOutputPriority
	}
	if intent.TaskType == "knowledge_grounded" {
		return AnswerModeGroundedPreferred
	}
	return AnswerModeBestEffort
}

func (g *GroundingPolicyEngine) checkEvidence(retrieval *RetrievalResult, toolResults []ContextToolResult) EvidenceStatus {
	if len(toolResults) > 0 {
		for _, result := range toolResults {
			if strings.EqualFold(result.Status, "success") {
				return EvidenceStatusSufficient
			}
		}
	}
	if retrieval == nil || len(retrieval.Chunks) == 0 {
		return EvidenceStatusInsufficient
	}
	if retrieval.Score >= g.relevanceThreshold && retrieval.SelectedCount > 0 {
		return EvidenceStatusSufficient
	}
	if retrieval.Score > 0 {
		return EvidenceStatusWeak
	}
	return EvidenceStatusInsufficient
}

func FormatGroundedResponse(content string, retrieval *RetrievalResult, toolResults []ContextToolResult, decision GroundingDecision) string {
	content = strings.TrimSpace(content)
	if decision.ShouldAbstain {
		return "I don’t have enough grounded evidence to answer confidently."
	}
	var references []string
	for _, result := range toolResults {
		if strings.EqualFold(result.Status, "success") {
			references = append(references, fmt.Sprintf("Tool: %s", result.Name))
		}
	}
	if retrieval != nil {
		for _, provenance := range retrieval.Provenance {
			label := firstNonEmptyString(provenance.DocTitle, provenance.Source)
			if label == "" {
				continue
			}
			section := strings.TrimSpace(provenance.Section)
			if section != "" {
				label = label + " / " + section
			}
			references = append(references, label)
		}
	}
	references = dedupeStrings(references)
	if len(references) == 0 || decision.Mode == AnswerModeBestEffort {
		return content
	}
	return content + "\n\nSources:\n- " + strings.Join(references, "\n- ")
}

func dedupeStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	var deduped []string
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		deduped = append(deduped, value)
	}
	return deduped
}
