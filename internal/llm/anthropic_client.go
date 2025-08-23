// In file: internal/llm/anthropic_client.go
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/dileep-u-k/llm-gateway/internal/tools"
)

const (
	anthropicAPIURL  = "https://api.anthropic.com/v1/messages"
	anthropicVersion = "2023-06-01"
	defaultMaxTokens = 4096
)

// --- API Data Structures ---

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicRequest struct {
	Model       string             `json:"model"`
	Messages    []anthropicMessage `json:"messages"`
	System      string             `json:"system,omitempty"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
	MaxTokens   int                `json:"max_tokens"`
	Stream      bool               `json:"stream"`
	Temperature *float32           `json:"temperature,omitempty"`
}
type anthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}
type anthropicTool struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema interface{} `json:"input_schema"`
}
type anthropicContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	Content   string          `json:"content,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
}
type anthropicResponse struct {
	Content []anthropicContentBlock `json:"content"`
	Usage   anthropicUsage          `json:"usage"`
}
type anthropicStreamEvent struct {
	Type  string          `json:"type"`
	Delta json.RawMessage `json:"delta"`
	Usage anthropicUsage  `json:"usage"`
}
type anthropicStreamTextDelta struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// --- Main Client ---
type AnthropicClient struct {
	apiKey     string
	httpClient *http.Client
}

var _ LLMClient = (*AnthropicClient)(nil)

func NewAnthropicClient(apiKey string) (*AnthropicClient, error) {
	if apiKey == "" {
		return nil, errors.New("anthropic API key cannot be empty")
	}
	return &AnthropicClient{
		apiKey:     apiKey,
		httpClient: &http.Client{Timeout: defaultTimeout},
	}, nil
}

func (c *AnthropicClient) Generate(ctx context.Context, messages []Message, config *GenerationConfig, availableTools []tools.Tool) (*GenerationResult, error) {
	payload, err := c.buildRequestPayload(messages, config, availableTools, false)
	if err != nil {
		return nil, fmt.Errorf("failed to build anthropic request payload: %w", err)
	}
	respBody, err := c.doRequest(ctx, payload)
	if err != nil {
		return nil, err
	}
	return parseAnthropicResponse(respBody)
}

func (c *AnthropicClient) GenerateStream(ctx context.Context, messages []Message, config *GenerationConfig, availableTools []tools.Tool) (<-chan *StreamingResult, error) {
	payload, err := c.buildRequestPayload(messages, config, availableTools, true)
	if err != nil {
		return nil, fmt.Errorf("failed to build anthropic stream payload: %w", err)
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
func (c *AnthropicClient) buildRequestPayload(messages []Message, config *GenerationConfig, availableTools []tools.Tool, stream bool) (*bytes.Buffer, error) {
	systemPrompt, anthropicMsgs := toAnthropicMessages(messages)
	anthropicTools, err := toAnthropicTools(availableTools)
	if err != nil {
		return nil, fmt.Errorf("failed to convert tools: %w", err)
	}

	req := anthropicRequest{
		Model:       config.Model,
		Messages:    anthropicMsgs,
		System:      systemPrompt,
		Tools:       anthropicTools,
		MaxTokens:   defaultMaxTokens,
		Stream:      stream,
		Temperature: config.Temperature,
	}
	if config.MaxTokens > 0 {
		req.MaxTokens = config.MaxTokens
	}
	payloadBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}
	return bytes.NewBuffer(payloadBytes), nil
}

func toAnthropicMessages(messages []Message) (string, []anthropicMessage) {
	var systemPrompt string
	var anthropicMsgs []anthropicMessage
	for _, msg := range messages {
		if msg.Role == RoleSystem {
			systemPrompt = msg.Content
			continue
		}
		aMsg := anthropicMessage{Role: string(msg.Role)}
		if msg.Role == RoleTool {
			aMsg.Role = "user"
			aMsg.Content = []anthropicContentBlock{{
				Type:      "tool_result",
				ToolUseID: msg.ToolCallID,
				Content:   msg.Content,
			}}
		} else {
			aMsg.Content = msg.Content
		}
		anthropicMsgs = append(anthropicMsgs, aMsg)
	}
	return systemPrompt, anthropicMsgs
}

func toAnthropicTools(toolsToConvert []tools.Tool) ([]anthropicTool, error) {
	if len(toolsToConvert) == 0 {
		return nil, nil
	}
	anthropicTools := make([]anthropicTool, 0, len(toolsToConvert))
	for _, t := range toolsToConvert {
		// FIX 1: Use direct conversion and check for errors.
		paramsBytes, err := json.Marshal(t.Function.Parameters)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal tool parameters: %w", err)
		}
		var paramsMap map[string]interface{}
		if err := json.Unmarshal(paramsBytes, &paramsMap); err != nil {
			return nil, fmt.Errorf("failed to unmarshal tool parameters: %w", err)
		}
		anthropicTools = append(anthropicTools, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: paramsMap,
		})
	}
	return anthropicTools, nil
}

func parseAnthropicResponse(body []byte) (*GenerationResult, error) {
	var anthropicResp anthropicResponse
	if err := json.Unmarshal(body, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal anthropic response: %w", err)
	}
	if len(anthropicResp.Content) == 0 {
		return nil, errors.New("no content returned from Anthropic")
	}
	var contentBuilder strings.Builder
	var toolCalls []*tools.ToolCall
	for _, block := range anthropicResp.Content {
		switch block.Type {
		case "text":
			contentBuilder.WriteString(block.Text)
		case "tool_use":
			toolCalls = append(toolCalls, &tools.ToolCall{
				ID:   block.ID,
				Type: tools.ToolTypeFunction,
				Function: tools.ToolCallFunction{
					Name:      block.Name,
					Arguments: string(block.Input),
				},
			})
		}
	}
	usage := api.Usage{
		PromptTokens:     anthropicResp.Usage.InputTokens,
		CompletionTokens: anthropicResp.Usage.OutputTokens,
		TotalTokens:      anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
	}

	return &GenerationResult{
		Content:   strings.TrimSpace(contentBuilder.String()),
		ToolCalls: toolCalls,
		Usage:     usage,
	}, nil
}

func (c *AnthropicClient) processStream(body io.ReadCloser, outChan chan<- *StreamingResult) {
	defer func() {
		if err := body.Close(); err != nil {
			log.Printf("Error closing anthropic stream body: %v", err)
		}
		close(outChan)
	}()

	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data:") {
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data == "" || !strings.HasPrefix(data, "{") {
				continue
			}
			var event anthropicStreamEvent
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				log.Printf("Error unmarshalling stream event: %v, data: %s", err, data)
				continue
			}
			switch event.Type {
			case "content_block_delta":
				var textDelta anthropicStreamTextDelta
				if json.Unmarshal(event.Delta, &textDelta) == nil && textDelta.Type == "text_delta" {
					outChan <- &StreamingResult{ContentDelta: textDelta.Text}
				}
			case "message_stop":
				usage := &api.Usage{
					PromptTokens:     event.Usage.InputTokens,
					CompletionTokens: event.Usage.OutputTokens,
					TotalTokens:      event.Usage.InputTokens + event.Usage.OutputTokens,
				}
				outChan <- &StreamingResult{Usage: usage}
				return
			}
		}
	}
	if err := scanner.Err(); err != nil {
		outChan <- &StreamingResult{Err: fmt.Errorf("error reading stream: %w", err)}
	}
}

// FIX 2: Check the error returned from body.Close() and handle it.
func (c *AnthropicClient) doRequest(ctx context.Context, payload *bytes.Buffer) ([]byte, error) {
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
		if err := resp.Body.Close(); err != nil {
			log.Printf("Warning: Failed to close response body: %v", err)
		}
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

func (c *AnthropicClient) doRequestStream(ctx context.Context, payload *bytes.Buffer) (io.ReadCloser, error) {
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
		if err := resp.Body.Close(); err != nil {
			log.Printf("Warning: Failed to close stream response body: %v", err)
		}
		return nil, fmt.Errorf("anthropic API stream error: status %d, body: %s", resp.StatusCode, string(body))
	}
	return resp.Body, nil
}

func (c *AnthropicClient) createRequest(ctx context.Context, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", anthropicAPIURL, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create http request: %w", err)
	}
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", anthropicVersion)
	req.Header.Set("content-type", "application/json")
	return req, nil
}
