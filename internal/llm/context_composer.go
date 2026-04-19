package llm

import (
	"fmt"
	"strings"
)

type ContextToolResult struct {
	Name    string
	Status  string
	Summary string
}

type ContextComposerInput struct {
	UserPrompt        string
	Memory            *SessionMemorySnapshot
	Retrieval         *RetrievalResult
	ToolResults       []ContextToolResult
	RouteSelection    *RouteSelection
	GroundingDecision GroundingDecision
	ForceMetadata     ForceMetadata
	ModelContextLimit int
	Artifacts         []ArtifactRecord
	ExecutionPlan     *ExecutionPlan
}

type ComposedContext struct {
	SystemPrompt     string
	UserPrompt       string
	PromptChars      int
	BudgetChars      int
	IncludedSections []string
	OmittedSections  []string
}

type ContextComposer struct {
	defaultBudgetChars int
}

func NewContextComposer() *ContextComposer {
	return &ContextComposer{defaultBudgetChars: 12000}
}

func (c *ContextComposer) Compose(input ContextComposerInput) *ComposedContext {
	budget := c.defaultBudgetChars
	if input.ModelContextLimit > 0 {
		derived := input.ModelContextLimit * 3
		if derived > 0 && derived < budget {
			budget = derived
		}
		if derived > budget && derived < 32000 {
			budget = derived
		}
	}

	systemLines := []string{
		"You are operating inside a knowledge-aware orchestration gateway.",
	}
	if instruction := strings.TrimSpace(input.GroundingDecision.Instruction); instruction != "" {
		systemLines = append(systemLines, instruction)
	}
	if input.ForceMetadata.IsForced {
		systemLines = append(systemLines, fmt.Sprintf("Respect the current force semantics: %s.", input.ForceMetadata.Scope))
	}
	if input.RouteSelection != nil && input.RouteSelection.Explanation.PolicySummary != "" {
		systemLines = append(systemLines, "Route policy: "+input.RouteSelection.Explanation.PolicySummary)
	}
	systemPrompt := strings.Join(systemLines, "\n")

	var userSections []string
	var included []string
	var omitted []string

	appendSection := func(title, body string) {
		body = strings.TrimSpace(body)
		if body == "" {
			omitted = append(omitted, title)
			return
		}
		userSections = append(userSections, fmt.Sprintf("%s:\n%s", title, body))
		included = append(included, title)
	}

	appendSection("User Question", input.UserPrompt)
	appendSection("Execution Constraints", c.executionConstraints(input))
	appendSection("Session Memory Summary", c.sessionMemorySummary(input.Memory))
	appendSection("Working Memory", c.workingMemorySummary(input.Memory))
	appendSection("Execution Plan", c.executionPlanSummary(input.ExecutionPlan))
	appendSection("Retrieved Evidence", c.retrievedEvidence(input.Retrieval))
	appendSection("Multimodal Artifacts", c.multimodalArtifacts(input.Artifacts))
	appendSection("Tool Outputs", c.toolOutputs(input.ToolResults))
	appendSection("Artifact Notes", c.artifactNotes(input.Memory))
	appendSection("Answer Mode", c.answerModeSummary(input.GroundingDecision))

	userPrompt := c.fitBudget(strings.Join(userSections, "\n\n"), budget)
	return &ComposedContext{
		SystemPrompt:     systemPrompt,
		UserPrompt:       userPrompt,
		PromptChars:      len(systemPrompt) + len(userPrompt),
		BudgetChars:      budget,
		IncludedSections: included,
		OmittedSections:  omitted,
	}
}

func (c *ContextComposer) executionConstraints(input ContextComposerInput) string {
	parts := []string{}
	if input.RouteSelection != nil && input.RouteSelection.Explanation.SelectedModel != "" {
		parts = append(parts, "Selected model: "+input.RouteSelection.Explanation.SelectedModel)
	}
	if input.RouteSelection != nil && input.RouteSelection.Explanation.PolicyName != "" {
		parts = append(parts, "Routing policy: "+input.RouteSelection.Explanation.PolicyName)
	}
	if input.GroundingDecision.Mode != "" {
		parts = append(parts, "Answer mode: "+string(input.GroundingDecision.Mode))
	}
	if input.ForceMetadata.IsForced {
		parts = append(parts, "Forced semantics active")
	}
	return strings.Join(parts, "\n")
}

func (c *ContextComposer) sessionMemorySummary(memory *SessionMemorySnapshot) string {
	if memory == nil {
		return ""
	}
	lines := []string{}
	if memory.Summary.Summary != "" {
		lines = append(lines, memory.Summary.Summary)
	}
	if memory.Summary.UserGoalProgression != "" {
		lines = append(lines, "User goal: "+memory.Summary.UserGoalProgression)
	}
	if memory.Summary.ActiveReasoning != "" {
		lines = append(lines, "Reasoning thread: "+memory.Summary.ActiveReasoning)
	}
	if memory.Summary.CurrentSubtask != "" {
		lines = append(lines, "Current subtask: "+memory.Summary.CurrentSubtask)
	}
	return strings.Join(lines, "\n")
}

func (c *ContextComposer) workingMemorySummary(memory *SessionMemorySnapshot) string {
	if memory == nil {
		return ""
	}
	lines := []string{}
	if memory.Working.SessionMode != "" {
		lines = append(lines, "Session mode: "+memory.Working.SessionMode)
	}
	if memory.Working.EffectiveModel != "" {
		lines = append(lines, "Effective model: "+memory.Working.EffectiveModel)
	}
	if len(memory.Working.ActiveConstraints) > 0 {
		lines = append(lines, "Constraints: "+strings.Join(memory.Working.ActiveConstraints, ", "))
	}
	if len(memory.Working.KnowledgeSources) > 0 {
		lines = append(lines, "Knowledge sources: "+strings.Join(memory.Working.KnowledgeSources, ", "))
	}
	return strings.Join(lines, "\n")
}

func (c *ContextComposer) retrievedEvidence(result *RetrievalResult) string {
	if result == nil || len(result.Chunks) == 0 {
		return ""
	}
	blocks := make([]string, 0, len(result.Chunks))
	for _, chunk := range result.Chunks {
		header := []string{}
		if chunk.Source != "" {
			header = append(header, "Source="+chunk.Source)
		}
		if chunk.DocTitle != "" {
			header = append(header, "Title="+chunk.DocTitle)
		}
		if chunk.Section != "" {
			header = append(header, "Section="+chunk.Section)
		}
		if chunk.Version != "" {
			header = append(header, "Version="+chunk.Version)
		}
		blocks = append(blocks, fmt.Sprintf("[%s]\n%s", strings.Join(header, " | "), chunk.Text))
	}
	return strings.Join(blocks, "\n\n")
}

func (c *ContextComposer) executionPlanSummary(plan *ExecutionPlan) string {
	if plan == nil {
		return ""
	}
	lines := []string{
		"Plan type: " + plan.PlanType,
		"Sync mode: " + plan.SyncMode,
	}
	if plan.PrimaryCapability != "" {
		lines = append(lines, "Primary capability: "+string(plan.PrimaryCapability))
	}
	if len(plan.Modalities) > 0 {
		lines = append(lines, "Modalities: "+strings.Join(plan.Modalities, ", "))
	}
	for _, stage := range plan.Stages {
		line := fmt.Sprintf("%s [%s]", stage.Title, stage.StageType)
		if stage.ModelBinding != "" {
			line += " -> " + stage.ModelBinding
		}
		if stage.BindingViolation != "" {
			line += " (" + stage.BindingViolation + ")"
		}
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func (c *ContextComposer) multimodalArtifacts(artifacts []ArtifactRecord) string {
	if len(artifacts) == 0 {
		return ""
	}
	blocks := make([]string, 0, len(artifacts))
	for _, artifact := range artifacts {
		lines := []string{fmt.Sprintf("%s [%s]", firstNonEmpty(artifact.Name, artifact.ArtifactID), artifact.Type)}
		if artifact.SourceURI != "" {
			lines = append(lines, "Source: "+artifact.SourceURI)
		}
		if artifact.Caption != "" {
			lines = append(lines, "Caption: "+artifact.Caption)
		}
		if artifact.OCRText != "" {
			lines = append(lines, "OCR: "+artifact.OCRText)
		}
		if artifact.Transcript != "" {
			lines = append(lines, "Transcript: "+artifact.Transcript)
		}
		if artifact.Text != "" {
			lines = append(lines, "Text: "+artifact.Text)
		}
		blocks = append(blocks, strings.Join(lines, "\n"))
	}
	return strings.Join(blocks, "\n\n")
}

func (c *ContextComposer) toolOutputs(results []ContextToolResult) string {
	if len(results) == 0 {
		return ""
	}
	blocks := make([]string, 0, len(results))
	for _, result := range results {
		blocks = append(blocks, fmt.Sprintf("%s (%s): %s", result.Name, result.Status, result.Summary))
	}
	return strings.Join(blocks, "\n")
}

func (c *ContextComposer) artifactNotes(memory *SessionMemorySnapshot) string {
	if memory == nil || len(memory.Artifacts) == 0 {
		return ""
	}
	lines := make([]string, 0, len(memory.Artifacts))
	for _, artifact := range memory.Artifacts {
		lines = append(lines, fmt.Sprintf("%s [%s] %s", artifact.Name, artifact.Kind, artifact.Source))
	}
	return strings.Join(lines, "\n")
}

func (c *ContextComposer) answerModeSummary(decision GroundingDecision) string {
	if decision.Mode == "" {
		return ""
	}
	lines := []string{"Answer mode: " + string(decision.Mode)}
	if decision.EvidenceStatus != "" {
		lines = append(lines, "Evidence status: "+string(decision.EvidenceStatus))
	}
	if decision.Caution != "" {
		lines = append(lines, "Caution: "+decision.Caution)
	}
	return strings.Join(lines, "\n")
}

func (c *ContextComposer) fitBudget(value string, budget int) string {
	value = strings.TrimSpace(value)
	if budget <= 0 || len(value) <= budget {
		return value
	}
	return strings.TrimSpace(value[:budget-3]) + "..."
}
