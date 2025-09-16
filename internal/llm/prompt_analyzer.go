// In file: internal/llm/prompt_analyzer.go
package llm

import (
	"regexp"
	"strings"
)

// =================================================================================
// Enterprise-Grade Prompt Analyzer v6 (Final Production Release)
// =================================================================================
// - "Score First, Filter Later" pipeline prevents false positives on nuanced queries.
// - Caps and normalization ensure predictable scoring across prompt lengths.
// - Explicit overrides guarantee correctness for coding vs. simple vs. ultra-complex.
// - Safe defaults prevent costly model selection for trivial prompts.
//
// Usage:
//   analyzer := NewPromptAnalyzer()
//   preference := analyzer.Analyze("Explain quantum entanglement in detail")
//   // preference → "default" | "max_quality" | "balanced" | "cost" | "best-for-coding"
// =================================================================================

var (
	// High-priority check for any coding-related prompt.
	codingArchetypes = regexp.MustCompile(
		`(?i)\b(write|create|generate|implement|fix|debug|refactor|optimize|show me the)\b.*\b(code|script|function|class|method|api|endpoint|query|dockerfile|unit test|algorithm)\b|` +
			`\b(python|java|go|golang|javascript|typescript|rust|c\+\+|c#|swift|kotlin|php|ruby|html|css|sql)\b|` +
			`\b(react|vue|angular|django|flask|fastapi|spring boot|\.net|express\.js|pandas|numpy|tensorflow|pytorch|kubernetes|terraform)\b`,
	)

	// Stricter pattern for simple questions (1-liners).
	simpleQueryArchetypes = regexp.MustCompile(
		`(?i)^(what|who|which|where|when)\s+(is|was|are|were)\s.{1,75}(\?|$)|` +
			`(?i)^(list|define|how many)\s`,
	)

	// Medium-complexity cues.
	mediumComplexityArchetypes = regexp.MustCompile(
		`(?i)\b(explain|summarize|describe|how (do|does|to))\b|` +
			`\bwhat is the (process|method|significance) of|give me an overview of|elaborate on`,
	)

	// High-complexity cues.
	highComplexityArchetypes = regexp.MustCompile(
		`(?i)\b(compare (and contrast)?|analyze the (impact|effect)|evaluate the|what are the (pros and cons|advantages and disadvantages)|discuss the implications of|critically evaluate|what is the relationship between)\b`,
	)

	// Ultra-complex: design, creative, role-play, logical proofs, data analysis.
	ultraComplexityArchetypes = regexp.MustCompile(
		`(?i)\b(design a|create a (comprehensive|detailed) plan for|develop a (business plan|framework|strategy)|invent a|write a detailed report on|compose a|draft a|propose a solution for|write an essay on)\b|` +
			`\b(poem|short story|screenplay|marketing copy|thesis statement|legal clause|press release|official statement)\b|` +
			`\b(act as a|you are a|imagine you are)\b|` +
			`\b(solve the equation|calculate the|prove the theorem)\b|` +
			`\b(analyze this dataset|given this data|create a visualization for)\b`,
	)

	// Regex to detect embedded code blocks.
	codeBlockRegex = regexp.MustCompile("(?s)```.*```")

	// Negative keywords: long prompts with trivial intent → downscore.
	negativeKeywords = regexp.MustCompile(`(?i)\b(list all|what is|who are|define the)\b`)
)

// PromptAnalyzer determines prompt complexity and routes to the right model preference.
type PromptAnalyzer struct{}

// NewPromptAnalyzer creates a new analyzer instance.
func NewPromptAnalyzer() *PromptAnalyzer {
	return &PromptAnalyzer{}
}

// Analyze classifies a prompt into a routing preference.
// Returns one of: cost | balanced | default | max_quality | best-for-coding
func (pa *PromptAnalyzer) Analyze(prompt string) string {
	// 1. Pre-processing
	normalized := strings.ToLower(strings.TrimSpace(prompt))
	if normalized == "" {
		return "cost"
	}
	length := len(normalized)

	// 2. Coding Override
	if codeBlockRegex.MatchString(normalized) || codingArchetypes.MatchString(normalized) {
		return "best-for-coding"
	}

	// 3. Complexity Scoring
	var score int
	score += length / 200                        // Length contribution
	score += strings.Count(normalized, "\n") * 2 // Structure contribution

	if mediumComplexityArchetypes.MatchString(normalized) {
		score += 5
	}
	if highComplexityArchetypes.MatchString(normalized) {
		score += 15
	}
	if ultraComplexityArchetypes.MatchString(normalized) {
		score += 30
	}
	if negativeKeywords.MatchString(normalized) {
		score -= 10
	}

	// Normalize score within a sane range.
	if score < 0 {
		score = 0
	}
	if score > 50 {
		score = 50
	}

	// 4. Simplicity filter
	if simpleQueryArchetypes.MatchString(normalized) && score < 5 {
		return "cost"
	}

	// 5. Map score → preference
	var preference string
	switch {
	case score > 25:
		preference = "max_quality" // Ultra-Complex
	case score > 10:
		preference = "default" // Complex
	default:
		preference = "balanced" // Medium
	}

	// 6. Final Overrides
	if length < 25 {
		return "cost" // Very short prompts always cheap
	}
	if length > 1000 && preference == "balanced" {
		return "default" // Long prompts are at least complex
	}

	return preference
}
