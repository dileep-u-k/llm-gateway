// In file: internal/llm/intent_analyzer.go
package llm

import (
	"log"
	"regexp"
	"strings"
)

// Define constants for the different intents we can detect.
const (
	IntentWeather    = "weather"
	IntentCalculator = "calculator"
	IntentNews       = "news"
	IntentRAG        = "rag_knowledge_query"
)

// calculatorRegex is a simple regex to detect mathematical expressions.
var calculatorRegex = regexp.MustCompile(`\d+\s*[\+\-\*\/]\s*\d+`)

// IntentAnalyzer is now a simpler service. It no longer needs Redis.
type IntentAnalyzer struct{}

// NewIntentAnalyzer is now simpler and has no dependencies.
func NewIntentAnalyzer() *IntentAnalyzer {
	return &IntentAnalyzer{}
}

// AnalyzeIntent now only performs fast checks for tool intents.
// If no tool is found, it defaults to assuming the user is asking a knowledge question.
func (ia *IntentAnalyzer) AnalyzeIntent(prompt string) string {
	lowerPrompt := strings.ToLower(prompt)

	// --- Fast Path: Keyword and Regex Checks for Tools ---
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

	// If no specific tool is detected, default to a RAG knowledge query.
	// The RAG system's own confidence score will then decide if the context is used.
	log.Println("No tool intent detected. Defaulting to RAG knowledge query.")
	return IntentRAG
}