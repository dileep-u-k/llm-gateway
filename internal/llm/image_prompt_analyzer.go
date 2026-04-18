// In file: internal/llm/image_prompt_analyzer.go
package llm

import (
	"regexp"
	"strings"
)

// Expanded archetypes with broader coverage.
var (
	artisticArchetypes = regexp.MustCompile(
		`(?i)\b(artistic|style of|painting|watercolor|oil painting|digital art|illustration|sketchbook|concept art|impressionist|surrealist|pop art|abstract|cubism|fantasy|sci-fi|anime|manga|steampunk|render|vfx)\b`,
	)

	photoArchetypes = regexp.MustCompile(
		`(?i)\b(photorealistic|realistic|photo of|photograph|high detail|ultra realistic|sharp focus|8k|4k|hdr|dslr|85mm|50mm|portrait shot|cinematic)\b`,
	)

	previewArchetypes = regexp.MustCompile(
		`(?i)\b(fast|quick|preview|draft|cheap|low quality|thumbnail|prototype|sketch|concept|low-res|simple drawing|mockup)\b`,
	)

	diagramArchetypes = regexp.MustCompile(
		`(?i)\b(diagram|flowchart|schema|blueprint|graph|map|chart|architecture|workflow|pipeline|uml|system design)\b`,
	)
)

// ImagePromptAnalyzer intelligently selects the best strategy for image generation.
type ImagePromptAnalyzer struct{}

func NewImagePromptAnalyzer() *ImagePromptAnalyzer {
	return &ImagePromptAnalyzer{}
}

// Analyze maps prompt keywords to strategies with sensible defaults.
func (ipa *ImagePromptAnalyzer) Analyze(prompt string) string {
	lowerPrompt := strings.ToLower(prompt)

	// 1. Explicit technical diagrams
	if diagramArchetypes.MatchString(lowerPrompt) {
		return "diagram"
	}

	// 2. Quick/low fidelity previews
	if previewArchetypes.MatchString(lowerPrompt) {
		return "fast_preview"
	}

	// 3. Creative or stylized
	if artisticArchetypes.MatchString(lowerPrompt) {
		return "artistic"
	}

	// 4. Realistic / photo-like
	if photoArchetypes.MatchString(lowerPrompt) {
		return "photorealistic"
	}

	// 5. Default: balanced (safe fallback for generic prompts)
	return "balanced"
}
