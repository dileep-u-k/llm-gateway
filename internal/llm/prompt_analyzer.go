// In file: internal/llm/prompt_analyzer.go
package llm

import (
	"regexp"
	"strings"
)

// =================================================================================
// Enterprise-Grade Prompt Analyzer v5 (Definitive Production Version)
// =================================================================================
// This definitive version implements a more robust logic flow to correctly
// analyze prompts that start simply but contain complex instructions.
//
// Key Logic Change:
// - **Score First, Filter Later:** The analyzer now calculates a full complexity
//   score for every non-coding prompt. It only classifies a prompt as "simple"
//   if it both matches a simple pattern AND has a very low complexity score.
//   This prevents false positives on nuanced questions like "What are... explain in detail?".
// =================================================================================

var (
	// --- Pre-compiled Regex for Archetype Matching (Optimized for Performance) ---

	// High-priority check for any coding-related prompt.
	codingArchetypes = regexp.MustCompile(
		`(?i)\b(write|create|generate|implement|fix|debug|refactor|optimize|show me the)\b.*\b(code|script|function|class|method|api|endpoint|query|dockerfile|unit test|algorithm)\b|` +
			`\b(python|java|go|javascript|typescript|rust|c\+\+|swift|kotlin|php|html|css|sql)\b|` +
			`\b(react|vue|angular|django|flask|fastapi|pandas|numpy|tensorflow|terraform|kubernetes)\b`,
	)

	// Pattern for identifying potentially simple, factual recall questions.
	simpleQueryArchetypes = regexp.MustCompile(
		`(?i)^(what|who|which|where|when)\s(is|was|are|were)\s|` +
			`(?i)^(list|define)\s`,
	)

	// Patterns for scoring different levels of complexity.
	mediumComplexityArchetypes = regexp.MustCompile(
		`(?i)\b(explain|summarize|describe|how (do|does|to))\b|` +
			`\bwhat is the (process|method|significance) of|give me an overview of|elaborate on`,
	)
	highComplexityArchetypes = regexp.MustCompile(
		`(?i)\b(compare (and contrast)?|analyze the (impact|effect)|evaluate the|what are the (pros and cons|advantages and disadvantages)|discuss the implications of|critically evaluate)\b`,
	)
	ultraComplexityArchetypes = regexp.MustCompile(
		`(?i)\b(design a|create a (comprehensive|detailed) plan for|develop a (business plan|framework|strategy)|invent a|write a detailed report on|compose a|draft a|propose a solution for)\b|` +
			`\b(poem|short story|song lyrics|screenplay|marketing copy|thesis statement|legal clause)\b|` + // Creative & Professional Writing
			`\b(act as a|you are a|imagine you are)\b|` + // Role-playing
			`\b(solve the equation|calculate the|prove the theorem)\b|` + // Mathematical/Logical
			`\b(analyze this dataset|given this data|create a visualization for)\b`, // Data Analysis
	)

	codeBlockRegex = regexp.MustCompile("(?s)```.*```")
)

// PromptAnalyzer service is responsible for determining the complexity of a user's
// prompt and selecting an appropriate routing preference if none is provided.
type PromptAnalyzer struct{}

// NewPromptAnalyzer creates a new instance of the PromptAnalyzer.
func NewPromptAnalyzer() *PromptAnalyzer {
	return &PromptAnalyzer{}
}

// Analyze is the core classification function. It uses a new, more robust logic flow.
func (pa *PromptAnalyzer) Analyze(prompt string) string {
	// 1. Pre-processing: Normalize the prompt.
	normalizedPrompt := strings.ToLower(strings.TrimSpace(prompt))
	if normalizedPrompt == "" {
		return "cost" // Handle empty prompts gracefully.
	}

	// 2. High-Priority Override: Handle coding tasks first as they are a distinct category.
	if codeBlockRegex.MatchString(normalizedPrompt) || codingArchetypes.MatchString(normalizedPrompt) {
		return "best-for-coding"
	}

	// 3. Comprehensive Scoring: For ALL other prompts, calculate a score first.
	var complexityScore int
	complexityScore += len(normalizedPrompt) / 200      // Score for length
	complexityScore += strings.Count(normalizedPrompt, "\n") * 2 // Score for structure (paragraphs)
	if mediumComplexityArchetypes.MatchString(normalizedPrompt) {
		complexityScore += 5
	}
	if highComplexityArchetypes.MatchString(normalizedPrompt) {
		complexityScore += 15
	}
	if ultraComplexityArchetypes.MatchString(normalizedPrompt) {
		complexityScore += 30
	}

	// 4. Final Classification with "Simplicity Filter"
	// A prompt is only "simple" if it matches a simple pattern AND has a very low complexity score.
	// This correctly handles your "what are... explain in detail" example.
	if simpleQueryArchetypes.MatchString(normalizedPrompt) && complexityScore < 5 {
		return "cost"
	}

	// Otherwise, classify based on the calculated score.
	switch {
	case complexityScore > 25:
		return "max_quality" // Ultra-Complex
	case complexityScore > 10:
		return "default" // Complex
	default:
		return "balanced" // Medium
	}
}