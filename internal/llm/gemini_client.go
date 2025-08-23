// In file: internal/llm/gemini_client.go
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"

	"github.com/dileep-u-k/llm-gateway/internal/tools"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

// GeminiClient is the client for interacting with Google's Gemini models.
type GeminiClient struct {
	client *genai.GenerativeModel
}

var _ LLMClient = (*GeminiClient)(nil)

func NewGeminiClient(apiKey, modelID string) (*GeminiClient, error) {
	if apiKey == "" {
		return nil, errors.New("gemini API key cannot be empty")
	}
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}
	model := client.GenerativeModel(modelID)
	return &GeminiClient{client: model}, nil
}

// Generate performs a standard, blocking request to the Gemini API.
func (c *GeminiClient) Generate(
	ctx context.Context,
	messages []Message,
	config *GenerationConfig,
	availableTools []tools.Tool,
) (*GenerationResult, error) {
	c.configureModel(config, availableTools)
	chat := c.client.StartChat()
	chat.History = toGeminiContentHistory(messages)

	lastMessage := messages[len(messages)-1]
	resp, err := chat.SendMessage(ctx, genai.Text(lastMessage.Content))
	if err != nil {
		return nil, fmt.Errorf("gemini API call failed: %w", err)
	}
	return parseGeminiResponse(ctx, c.client, resp)
}

// GenerateStream performs a streaming request to the Gemini API.
func (c *GeminiClient) GenerateStream(
	ctx context.Context,
	messages []Message,
	config *GenerationConfig,
	availableTools []tools.Tool,
) (<-chan *StreamingResult, error) {
	c.configureModel(config, availableTools)
	chat := c.client.StartChat()
	chat.History = toGeminiContentHistory(messages)
	lastMessage := messages[len(messages)-1]

	outChan := make(chan *StreamingResult)
	go func() {
		defer close(outChan)
		iter := chat.SendMessageStream(ctx, genai.Text(lastMessage.Content))
		for {
			resp, err := iter.Next()
			if err == iterator.Done {
				break
			}
			if err != nil {
				outChan <- &StreamingResult{Err: fmt.Errorf("gemini stream error: %w", err)}
				return
			}
			if resp != nil && len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
				var contentBuilder strings.Builder
				for _, part := range resp.Candidates[0].Content.Parts {
					if txt, ok := part.(genai.Text); ok {
						contentBuilder.WriteString(string(txt))
					}
				}
				outChan <- &StreamingResult{ContentDelta: contentBuilder.String()}
			}
		}
	}()
	return outChan, nil
}

// configureModel applies dynamic settings using the SDK's setter methods for safety.
func (c *GeminiClient) configureModel(config *GenerationConfig, availableTools []tools.Tool) {
	// CORRECTED: Use SDK setter methods to safely handle configuration.
	// This avoids pointer mismatch errors.
	if config != nil {
		if config.Temperature != nil {
			c.client.SetTemperature(*config.Temperature)
		}
		if config.TopP != nil {
			c.client.SetTopP(*config.TopP)
		}
		if config.MaxTokens > 0 {
			c.client.SetMaxOutputTokens(int32(config.MaxTokens))
		} else {
			c.client.SetMaxOutputTokens(4096) // Set a default of 4096
		}
	} else {
		// Also set a default if no config was provided at all.
		c.client.SetMaxOutputTokens(4096)
	}

	if len(availableTools) > 0 {
		c.client.Tools = toGeminiTools(availableTools)
	} else {
		c.client.Tools = nil
	}
}

// toGeminiTools converts our internal tool definition to the Gemini SDK's format.
func toGeminiTools(toolsToConvert []tools.Tool) []*genai.Tool {
	var geminiTools []*genai.Tool
	for _, t := range toolsToConvert {
		// CORRECTED: Access fields through t.Function
		funcDecl := &genai.FunctionDeclaration{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  convertSchema(t.Function.Parameters),
		}
		geminiTools = append(geminiTools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{funcDecl},
		})
	}
	return geminiTools
}

// convertSchema is a helper function to convert our JSONSchema to the Gemini SDK's schema type.
func convertSchema(s tools.JSONSchema) *genai.Schema {
	genaiSchema := &genai.Schema{
		Description: s.Description,
		Required:    s.Required,
	}
	switch s.Type {
	case "object":
		genaiSchema.Type = genai.TypeObject
	case "string":
		genaiSchema.Type = genai.TypeString
	case "number":
		genaiSchema.Type = genai.TypeNumber
	case "integer":
		genaiSchema.Type = genai.TypeInteger
	}
	if s.Properties != nil {
		genaiSchema.Properties = make(map[string]*genai.Schema)
		for k, v := range s.Properties {
			genaiSchema.Properties[k] = convertSchema(*v)
		}
	}
	return genaiSchema
}

// toGeminiContentHistory converts our message history to the Gemini SDK's format.
func toGeminiContentHistory(messages []Message) []*genai.Content {
	var history []*genai.Content
	// The last message is the new prompt, so we exclude it from history
	for _, msg := range messages[:len(messages)-1] {
		role := "user"
		if msg.Role == RoleAssistant {
			role = "model"
		}
		history = append(history, &genai.Content{
			Role:  role,
			Parts: []genai.Part{genai.Text(msg.Content)},
		})
	}
	return history
}

// parseGeminiResponse converts a Gemini API response into our internal GenerationResult.
func parseGeminiResponse(
	ctx context.Context, // ADDED: Pass context for the new API call
	client *genai.GenerativeModel, // ADDED: Pass the client for the new API call
	resp *genai.GenerateContentResponse,
) (*GenerationResult, error) {
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return nil, errors.New("no content returned from Gemini")
	}

	candidate := resp.Candidates[0]
	var contentBuilder strings.Builder
	var toolCalls []*tools.ToolCall

	for _, part := range candidate.Content.Parts {
		switch v := part.(type) {
		case genai.Text:
			contentBuilder.WriteString(string(v))
		case genai.FunctionCall:
			argsMap, err := json.Marshal(v.Args)
			if err != nil {
				log.Printf("Warning: could not marshal tool call args: %v", err)
				continue
			}
			toolCalls = append(toolCalls, &tools.ToolCall{
				ID:   fmt.Sprintf("gemini-toolcall-%s", v.Name),
				Type: tools.ToolTypeFunction,
				Function: tools.ToolCallFunction{
					Name:      v.Name,
					Arguments: string(argsMap),
				},
			})
		}
	}

	result := &GenerationResult{
		Content:   strings.TrimSpace(contentBuilder.String()),
		ToolCalls: toolCalls,
	}

	// *** THIS IS THE FIX ***
	// First, try to get usage data directly from the response.
	if resp.UsageMetadata != nil {
		result.Usage.PromptTokens = int(resp.UsageMetadata.PromptTokenCount)
		result.Usage.CompletionTokens = int(resp.UsageMetadata.CandidatesTokenCount)
		result.Usage.TotalTokens = int(resp.UsageMetadata.TotalTokenCount)
	}

	// Fallback: If completion tokens are zero but we have content, count them manually.
	// This handles cases where the API doesn't return completion tokens in the metadata.
	if result.Usage.CompletionTokens == 0 && result.Content != "" {
		// log.Println("Gemini completion tokens were 0, performing manual count...")
		countResp, err := client.CountTokens(ctx, genai.Text(result.Content))
		if err != nil {
			log.Printf("Warning: Failed to manually count completion tokens: %v", err)
		} else {
			result.Usage.CompletionTokens = int(countResp.TotalTokens)
			// Recalculate total
			result.Usage.TotalTokens = result.Usage.PromptTokens + result.Usage.CompletionTokens
		}
	}

	return result, nil
}
