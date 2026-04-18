// In file: internal/llm/intent_analyzer.go
package llm

import (
	"log"
	"regexp"
	"strings"
)

// Define constants for the different intents we can detect.
type Intent string

const (
	IntentWeather       Intent = "weather"
	IntentCalculator    Intent = "calculator"
	IntentNews          Intent = "news"
	IntentImageCreation Intent = "image_creation"
	IntentRAG           Intent = "rag_knowledge_query"
)

var (
	calculatorRegex = regexp.MustCompile(`\d+\s*[\+\-\*\/]\s*\d+`)
	// This final, more comprehensive pattern now includes "sketch" and other keywords.
	imageCreationArchetypes = regexp.MustCompile(
		`(?i)\b(create|generate|make|draw|show me|imagine|illustrate|render|sketch)\b.*\b(image|picture|photo|drawing|illustration|logo|artwork|scene|portrait|design|diagram|flowchart|schema|graph)\b`,
	)
)

// IntentAnalyzer is a stateless service for fast intent detection.
type IntentAnalyzer struct{}

// NewIntentAnalyzer creates a new analyzer.
func NewIntentAnalyzer() *IntentAnalyzer {
	return &IntentAnalyzer{}
}

// AnalyzeIntent performs fast checks for specific intents.
// It prioritizes image creation, then tools, and defaults to a knowledge query.
func (ia *IntentAnalyzer) AnalyzeIntent(prompt string) Intent {
	lowerPrompt := strings.ToLower(prompt)

	// Highest priority: Check for any image or diagram creation intent first.
	// The main imageCreationArchetypes regex now includes all visual generation keywords.
	if imageCreationArchetypes.MatchString(lowerPrompt) {
		log.Printf("Intent detected by regex: %s", IntentImageCreation)
		return IntentImageCreation
	}

	// Second priority: Check for tool intents.
	weatherKeywords := []string{"weather", "forecast", "temperature", "how hot is it", "is it raining"}
	for _, keyword := range weatherKeywords {
		if strings.Contains(lowerPrompt, keyword) {
			log.Printf("Intent detected by keyword '%s': %s", keyword, IntentWeather)
			return IntentWeather
		}
	}
	newsKeywords := []string{"news", "headlines", "latest on", "what's happening in"}
	for _, keyword := range newsKeywords {
		if strings.Contains(lowerPrompt, keyword) {
			log.Printf("Intent detected by keyword '%s': %s", keyword, IntentNews)
			return IntentNews
		}
	}
	if calculatorRegex.MatchString(lowerPrompt) {
		log.Printf("Intent detected by regex: %s", IntentCalculator)
		return IntentCalculator
	}

	// Default to a RAG knowledge query.
	log.Println("No specific intent detected. Defaulting to RAG knowledge query.")
	return IntentRAG
}
