// In file: internal/llm/client.go
package llm

import (
	"context"

	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/dileep-u-k/llm-gateway/internal/tools"
)

// =================================================================================
// Core Data Structures
// =================================================================================

// Role represents the originator of a message in a conversation.
// Using a defined type and constants prevents typos and improves code clarity.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message represents a single message in a conversation history.
type Message struct {
	Role       Role              `json:"role"`
	Content    string            `json:"content"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
	ToolCalls  []*tools.ToolCall `json:"tool_calls,omitempty"` // <-- This field was missing
}

// GenerationConfig holds all the parameters to control the LLM's generation behavior.
// This allows for fine-tuned control over the model's output for different use cases.
type GenerationConfig struct {
	// The specific model to use for the generation (e.g., "gpt-4o", "claude-3-opus-20240229").
	Model string
	// Controls randomness. A lower value makes the output more deterministic.
	// Using a pointer allows us to distinguish between a value of 0.0 and an unset value.
	Temperature *float32
	// The maximum number of tokens to generate in the response.
	MaxTokens int
	// An alternative to sampling with temperature, called nucleus sampling.
	TopP *float32
	// Indicates whether to use streaming. The client implementation uses this to
	// decide which underlying API method to call.
	Stream bool
}

// GenerationResult holds the complete, non-streamed output from an LLM call.
type GenerationResult struct {
	// The generated text content from the model.
	Content string
	// A slice of tool calls requested by the model. Modern models can request
	// multiple tools to be called in parallel, so this is a slice.
	ToolCalls []*tools.ToolCall
	// Token usage statistics for the generation request.
	Usage api.Usage
}

// StreamingResult holds a chunk of a streamed response from an LLM.
// The channel returned by GenerateStream will send these structs.
type StreamingResult struct {
	// The chunk of text generated in this part of the stream.
	ContentDelta string
	// If the model decides to call a tool, the details will be streamed here.
	// This might be sent in chunks as well, depending on the provider.
	ToolCallChunk *tools.ToolCall
	// The final token usage, which is typically sent as the last item in the stream.
	Usage *api.Usage
	// An error that may have occurred during the stream.
	Err error
}

// =================================================================================
// LLM Client Interface
// =================================================================================

// LLMClient is the universal interface that all model clients (e.g., OpenAI, Anthropic) must implement.
// It supports both standard (unary) and streaming generation for maximum flexibility.
type LLMClient interface {
	// Generate performs a standard, blocking request to the LLM.
	// It takes the full conversation history and returns a single, complete result.
	// This is suitable for backend processing or non-interactive tasks.
	Generate(
		ctx context.Context,
		messages []Message,
		config *GenerationConfig,
		availableTools []tools.Tool,
	) (*GenerationResult, error)

	// GenerateStream performs a streaming request to the LLM.
	// It returns a channel that the caller can read from to receive results
	// token-by-token as they are generated.
	// This is essential for interactive applications like chatbots to provide a responsive user experience.
	GenerateStream(
		ctx context.Context,
		messages []Message,
		config *GenerationConfig,
		availableTools []tools.Tool,
	) (<-chan *StreamingResult, error)
}
