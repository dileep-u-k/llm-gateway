// In file: internal/llm/mistral_client.go
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

const (
	mistralAPIURL = "https://api.mistral.ai/v1/chat/completions"
)

// --- API Data Structures ---
type mistralRequest struct {
	Model       string           `json:"model"`
	Messages    []mistralMessage `json:"messages"`
	Tools       []mistralTool    `json:"tools,omitempty"`
	ToolChoice  string           `json:"tool_choice,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Temperature *float32         `json:"temperature,omitempty"`
	TopP        *float32         `json:"top_p,omitempty"`
}
type mistralMessage struct {
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	ToolCalls []mistralToolCall `json:"tool_calls,omitempty"`
}
type mistralToolCall struct {
	ID       string          `json:"id"`
	Function mistralFunction `json:"function"`
}
type mistralFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}
type mistralTool struct {
	Type     string         `json:"type"`
	Function tools.Function `json:"function"`
}
type mistralResponse struct {
	Choices []struct {
		Message mistralMessage `json:"message"`
	} `json:"choices"`
	Usage api.Usage `json:"usage"`
}
type mistralStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content   string            `json:"content"`
			ToolCalls []mistralToolCall `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
}

// --- Main Client ---
type MistralClient struct {
	apiKey     string
	httpClient *http.Client
}

var _ LLMClient = (*MistralClient)(nil)

func NewMistralClient(apiKey string) (*MistralClient, error) {
	if apiKey == "" {
		return nil, errors.New("Mistral API key cannot be empty")
	}
	return &MistralClient{
		apiKey:     apiKey,
		httpClient: &http.Client{Timeout: defaultTimeout},
	}, nil
}

func (c *MistralClient) Generate(ctx context.Context, messages []Message, config *GenerationConfig, availableTools []tools.Tool) (*GenerationResult, error) {
	payload, err := c.buildRequestPayload(messages, config, availableTools, false)
	if err != nil {
		return nil, fmt.Errorf("failed to build mistral request payload: %w", err)
	}
	respBody, err := c.doRequest(ctx, payload)
	if err != nil {
		return nil, err
	}
	return parseMistralResponse(respBody)
}

func (c *MistralClient) GenerateStream(ctx context.Context, messages []Message, config *GenerationConfig, availableTools []tools.Tool) (<-chan *StreamingResult, error) {
	payload, err := c.buildRequestPayload(messages, config, availableTools, true)
	if err != nil {
		return nil, fmt.Errorf("failed to build mistral stream payload: %w", err)
	}
	respBody, err := c.doRequestStream(ctx, payload)
	if err != nil {
		return nil, err
	}
	outChan := make(chan *StreamingResult)
	go c.processStream(respBody, outChan)
	return outChan, nil
}

// --- Helper Functions ---
func (c *MistralClient) buildRequestPayload(messages []Message, config *GenerationConfig, availableTools []tools.Tool, stream bool) (*bytes.Buffer, error) {
	mistralMsgs := toMistralMessages(messages)
	mistralTools := toMistralTools(availableTools)
	req := mistralRequest{
		Model:       config.Model,
		Messages:    mistralMsgs,
		Tools:       mistralTools,
		Stream:      stream,
		MaxTokens:   config.MaxTokens,
		Temperature: config.Temperature,
		TopP:        config.TopP,
	}
	if len(mistralTools) > 0 {
		req.ToolChoice = "auto"
	}
	payloadBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}
	return bytes.NewBuffer(payloadBytes), nil
}

func (c *MistralClient) doRequest(ctx context.Context, payload *bytes.Buffer) ([]byte, error) {
	// ... (Implementation is the same as the Anthropic client's doRequest)
	var lastErr error
	delay := initialRetryDelay
	for i := 0; i < maxRetries; i++ {
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
			return body, nil
		}
		lastErr = fmt.Errorf("anthropic API error (attempt %d/%d): status %d, body: %s", i+1, maxRetries, resp.StatusCode, string(body))
		if resp.StatusCode >= 400 && resp.StatusCode < 500 {
			return nil, lastErr
		}
		time.Sleep(delay)
		delay *= 2
	}
	return nil, lastErr
}
func (c *MistralClient) doRequestStream(ctx context.Context, payload *bytes.Buffer) (io.ReadCloser, error) {
    // ... (Implementation is the same as the Anthropic client's doRequestStream)
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
		return nil, fmt.Errorf("anthropic API stream error: status %d, body: %s", resp.StatusCode, string(body))
	}
	return resp.Body, nil
}
func (c *MistralClient) createRequest(ctx context.Context, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", mistralAPIURL, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create http request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Accept", "application/json")
	return req, nil
}

func (c *MistralClient) processStream(body io.ReadCloser, outChan chan<- *StreamingResult) {
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
			return
		}
		var chunk mistralStreamChunk
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
			if len(delta.ToolCalls) > 0 {
				tcChunk := delta.ToolCalls[0]
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

func toMistralMessages(messages []Message) []mistralMessage {
	mistralMsgs := make([]mistralMessage, 0, len(messages))
	for _, msg := range messages {
		mistralMsgs = append(mistralMsgs, mistralMessage{
			Role:    string(msg.Role),
			Content: msg.Content,
		})
	}
	return mistralMsgs
}

func toMistralTools(availableTools []tools.Tool) []mistralTool {
	if len(availableTools) == 0 {
		return nil
	}
	mistralTools := make([]mistralTool, 0, len(availableTools))
	for _, tool := range availableTools {
		mistralTools = append(mistralTools, mistralTool{
			Type:     "function",
			Function: tool.Function, // CORRECTED: Access the nested Function struct
		})
	}
	return mistralTools
}

func parseMistralResponse(body []byte) (*GenerationResult, error) {
	var mistralResp mistralResponse
	if err := json.Unmarshal(body, &mistralResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal mistral response: %w", err)
	}
	if len(mistralResp.Choices) == 0 {
		return nil, errors.New("no choices returned from Mistral")
	}
	choice := mistralResp.Choices[0]
	result := &GenerationResult{
		Content: choice.Message.Content,
		Usage:   mistralResp.Usage,
	}
	if len(choice.Message.ToolCalls) > 0 {
		result.ToolCalls = make([]*tools.ToolCall, 0, len(choice.Message.ToolCalls))
		for _, tc := range choice.Message.ToolCalls {
			result.ToolCalls = append(result.ToolCalls, &tools.ToolCall{
				ID:   tc.ID,
				Type: tools.ToolTypeFunction,
				Function: tools.ToolCallFunction{ // CORRECTED: Initialize the nested struct
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}
	}
	return result, nil
}