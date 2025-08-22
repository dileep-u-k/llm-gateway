// In file: internal/llm/openai_client.go
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"llm-gateway/internal/api"
	"llm-gateway/internal/tools"
	"net/http"
	"strings"
	"time"
)


// openAIRequest defines the top-level structure for an OpenAI API call.
type openAIRequest struct {
	Model       string          `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	Tools       []openAITool    `json:"tools,omitempty"`
	ToolChoice  string          `json:"tool_choice,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature *float32        `json:"temperature,omitempty"`
	TopP        *float32        `json:"top_p,omitempty"`
}

// openAIMessage represents a single message in a conversation.
type openAIMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	ToolCalls  []tools.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

// openAITool defines the structure for a tool that the API can use.
type openAITool struct {
	Type     string       `json:"type"`
	Function tools.Function `json:"function"`
}

// openAIResponse is the structure of a successful non-streaming response from the API.
type openAIResponse struct {
	Choices []struct {
		Message openAIMessage `json:"message"`
	} `json:"choices"`
	Usage api.Usage `json:"usage"`
}

// openAIStreamChunk is the structure of a single event in a streaming response.
type openAIStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content   string           `json:"content"`
			ToolCalls []tools.ToolCall `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
}

// --- END OF STRUCTS TO PASTE ---

const (
	openAIAPIURL     = "https://api.openai.com/v1/chat/completions"
	
)

// OpenAIClient is the client for interacting with OpenAI models like GPT-4.
// It implements the LLMClient interface, providing robust, production-ready features.
type OpenAIClient struct {
	apiKey     string
	httpClient *http.Client
}

// Statically verify that OpenAIClient implements the LLMClient interface.
var _ LLMClient = (*OpenAIClient)(nil)

// NewOpenAIClient creates a new, configured client for the OpenAI API.
// The modelID is now specified per-request via GenerationConfig, not on the client itself.
func NewOpenAIClient(apiKey string) (*OpenAIClient, error) {
	if apiKey == "" {
		return nil, errors.New("OpenAI API key cannot be empty")
	}
	return &OpenAIClient{
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: defaultTimeout, // Set a default timeout for all HTTP requests.
		},
	}, nil
}

// Generate performs a standard, blocking request to the OpenAI API.
func (c *OpenAIClient) Generate(
	ctx context.Context,
	messages []Message,
	config *GenerationConfig,
	availableTools []tools.Tool,
) (*GenerationResult, error) {
	// Build the request payload from our generic structures.
	payload, err := c.buildRequestPayload(messages, config, availableTools, false)
	if err != nil {
		return nil, fmt.Errorf("failed to build openai request payload: %w", err)
	}

	// Make the API call with our robust retry mechanism.
	respBody, err := c.doRequest(ctx, payload)
	if err != nil {
		return nil, err // The error from doRequest is already descriptive.
	}

	// Parse the complete JSON response body.
	return parseOpenAIResponse(respBody)
}

// GenerateStream performs a streaming request to the OpenAI API.
func (c *OpenAIClient) GenerateStream(
	ctx context.Context,
	messages []Message,
	config *GenerationConfig,
	availableTools []tools.Tool,
) (<-chan *StreamingResult, error) {
	// Build the request payload with the stream flag enabled.
	payload, err := c.buildRequestPayload(messages, config, availableTools, true)
	if err != nil {
		return nil, fmt.Errorf("failed to build openai stream payload: %w", err)
	}

	// The streaming version of the request returns a response body to be processed.
	respBody, err := c.doRequestStream(ctx, payload)
	if err != nil {
		return nil, err
	}

	// Create the channel to stream results back to the caller.
	outChan := make(chan *StreamingResult)

	// Start a goroutine to process the Server-Sent Events (SSE) stream.
	go c.processStream(respBody, outChan)

	return outChan, nil
}

// buildRequestPayload constructs the JSON body for the OpenAI API call.
func (c *OpenAIClient) buildRequestPayload(messages []Message, config *GenerationConfig, availableTools []tools.Tool, stream bool) (*bytes.Buffer, error) {
	// Convert our internal message and tool formats to OpenAI's specific format.
	openAIMsgs := toOpenAIMessages(messages)
	openAITools := toOpenAITools(availableTools)

	req := openAIRequest{
		Model:    config.Model,
		Messages: openAIMsgs,
		Tools:    openAITools,
		Stream:   stream,
	}

	// Apply generation parameters from the config.
	if config.MaxTokens > 0 {
		req.MaxTokens = config.MaxTokens
	}
	if config.Temperature != nil {
		req.Temperature = config.Temperature
	}
	if config.TopP != nil {
		req.TopP = config.TopP
	}

	// OpenAI allows forcing a tool call.
	if len(openAITools) > 0 {
		req.ToolChoice = "auto"
	}

	payloadBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}

	return bytes.NewBuffer(payloadBytes), nil
}

// doRequest performs the HTTP call with retries for non-streaming requests.
func (c *OpenAIClient) doRequest(ctx context.Context, payload *bytes.Buffer) ([]byte, error) {
	var lastErr error
	delay := initialRetryDelay

	for i := 0; i < maxRetries; i++ {
		// Use a bytes.Reader so the request body can be re-read on retry.
		req, err := c.createRequest(ctx, bytes.NewReader(payload.Bytes()))
		if err != nil {
			return nil, err
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("request failed (attempt %d/%d): %w", i+1, maxRetries, err)
			time.Sleep(delay)
			delay *= 2
			continue
		}

		body, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("failed to read response body: %w", readErr)
		}

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return body, nil // Success!
		}
		
		lastErr = fmt.Errorf("openai API error (attempt %d/%d): status %d, body: %s", i+1, maxRetries, resp.StatusCode, string(body))
		
		// Do not retry on client errors (e.g., 400 Bad Request).
		if resp.StatusCode >= 400 && resp.StatusCode < 500 {
			return nil, lastErr
		}
		
		time.Sleep(delay)
		delay *= 2
	}
	return nil, lastErr
}


// doRequestStream prepares and executes the HTTP request for streaming.
func (c *OpenAIClient) doRequestStream(ctx context.Context, payload *bytes.Buffer) (io.ReadCloser, error) {
	req, err := c.createRequest(ctx, payload)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to start stream request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("openai API stream error: status %d, body: %s", resp.StatusCode, string(body))
	}

	return resp.Body, nil
}

// createRequest is a helper to build the common parts of an http.Request.
func (c *OpenAIClient) createRequest(ctx context.Context, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", openAIAPIURL, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create http request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	return req, nil
}

// processStream reads the SSE stream from the response body and sends results to a channel.
func (c *OpenAIClient) processStream(body io.ReadCloser, outChan chan<- *StreamingResult) {
	defer body.Close()
	defer close(outChan)

	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			return // Stream is complete.
		}

		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			outChan <- &StreamingResult{Err: fmt.Errorf("error unmarshalling stream chunk: %w", err)}
			return
		}

		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta
			result := &StreamingResult{}

			if delta.Content != "" {
				result.ContentDelta = delta.Content
			}

			// Handle tool call chunks, which are streamed progressively.
			if len(delta.ToolCalls) > 0 {
				tcChunk := delta.ToolCalls[0]
				// CORRECTED: Initialize the nested 'Function' struct here as well.
				result.ToolCallChunk = &tools.ToolCall{
					ID:   tcChunk.ID,
					Type: tools.ToolTypeFunction,
					Function: tools.ToolCallFunction{
						Name:      tcChunk.Function.Name,
						Arguments: tcChunk.Function.Arguments,
					},
				}
			}
			outChan <- result
		}
	}

	if err := scanner.Err(); err != nil {
		outChan <- &StreamingResult{Err: fmt.Errorf("error reading stream: %w", err)}
	}
}

// toOpenAIMessages converts our internal message slice to the OpenAI API format.

func toOpenAIMessages(messages []Message) []openAIMessage {
	openAIMsgs := make([]openAIMessage, 0, len(messages))
	for _, msg := range messages {
		m := openAIMessage{Role: string(msg.Role)}

		// This switch handles all message types correctly for the OpenAI API
		switch msg.Role {
		case RoleTool:
			m.ToolCallID = msg.ToolCallID
			m.Content = msg.Content
		case RoleAssistant:
			// CORRECTED: This now includes both Content and the crucial ToolCalls field
			m.Content = msg.Content
			if len(msg.ToolCalls) > 0 {
				m.ToolCalls = make([]tools.ToolCall, len(msg.ToolCalls))
				for i, tc := range msg.ToolCalls {
					m.ToolCalls[i] = *tc
				}
			}
		default: // Handles RoleUser and RoleSystem
			m.Content = msg.Content
		}
		openAIMsgs = append(openAIMsgs, m)
	}
	return openAIMsgs
}

// toOpenAITools converts our internal tool slice to the OpenAI API format.
func toOpenAITools(availableTools []tools.Tool) []openAITool {
	if len(availableTools) == 0 {
		return nil
	}
	openAITools := make([]openAITool, 0, len(availableTools))
	for _, tool := range availableTools {
		openAITools = append(openAITools, openAITool{
			Type:     "function",
			Function: tool.Function, // CORRECTED: Access the nested Function struct
		})
	}
	return openAITools
}

// parseOpenAIResponse converts a full OpenAI API response to our internal GenerationResult.
func parseOpenAIResponse(body []byte) (*GenerationResult, error) {
	var openAIResp openAIResponse
	if err := json.Unmarshal(body, &openAIResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal openai response: %w", err)
	}
	if len(openAIResp.Choices) == 0 {
		return nil, errors.New("no choices returned from OpenAI")
	}

	choice := openAIResp.Choices[0]
	result := &GenerationResult{
		Content: choice.Message.Content,
		Usage:   openAIResp.Usage,
	}

	// Check for tool calls and create the struct correctly
	if len(choice.Message.ToolCalls) > 0 {
		result.ToolCalls = make([]*tools.ToolCall, 0, len(choice.Message.ToolCalls))
		for _, tc := range choice.Message.ToolCalls {
			// CORRECTED: Initialize the nested 'Function' struct
			toolCall := &tools.ToolCall{
				ID:   tc.ID,
				Type: tools.ToolTypeFunction,
				Function: tools.ToolCallFunction{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}
			result.ToolCalls = append(result.ToolCalls, toolCall)
		}
	}

	return result, nil
}