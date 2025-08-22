// In file: internal/tools/types.go

// Package tools defines the data structures for function calling (tool use)
// capabilities of the LLM gateway. These types provide a universal, provider-agnostic
// representation of tools that can be translated into the specific format required
// by different LLM APIs (like OpenAI, Gemini, or Anthropic).
package tools

// ToolTypeFunction is the standard type for function-based tools.
const ToolTypeFunction = "function"

// Tool defines the schema for a function that can be described to an LLM.
// This is the information you send *to* the model to make it aware of a tool's existence.
type Tool struct {
	// Type specifies the type of tool, which is almost always "function".
	Type string `json:"type"`
	// Function holds the detailed definition of the function.
	Function Function `json:"function"`
}

// Function defines the name, description, and parameters of a callable tool.
// This structure is based on the common JSON Schema format used by major LLM providers.
type Function struct {
	// Name is the name of the function to be called (e.g., "get_current_weather").
	Name string `json:"name"`
	// Description is a clear, concise explanation of what the function does.
	// This is critical, as the LLM uses this description to decide when to use the tool.
	Description string `json:"description"`
	// Parameters defines the arguments the function accepts, structured as a JSON Schema.
	Parameters JSONSchema `json:"parameters"`
}

// JSONSchema provides a structured, type-safe representation of the JSON Schema
// used for defining tool parameters. Using this struct instead of `any` or `map[string]interface{}`
// prevents common errors and makes tool definitions much clearer.
type JSONSchema struct {
	// Type defines the data type for a schema node (e.g., "object", "string", "number").
	// For the top-level parameters object, this should always be "object".
	Type string `json:"type"`
	// Description explains what a specific parameter is for.
	Description string `json:"description,omitempty"`
	// Properties describes the parameters of an object. The keys are parameter names,
	// and the values are further JSONSchema definitions for each parameter.
	Properties map[string]*JSONSchema `json:"properties,omitempty"`
	// Required is a list of parameter names that are mandatory for a function call.
	Required []string `json:"required,omitempty"`
}

// ToolCall represents a request *from* the LLM to execute a specific tool with given arguments.
// Your application code will receive this struct, execute the corresponding function,
// and send the result back to the LLM.
type ToolCall struct {
	// ID is a unique identifier for this specific tool call. It's crucial for matching
	// the tool's execution result back to the LLM's request in a multi-turn conversation.
	ID string `json:"id"`
	// Type indicates the type of tool being called, which is almost always "function".
	Type string `json:"type"`
	// Function contains the name and arguments for the function the LLM wants to execute.
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction holds the name and arguments of a function call requested by the LLM.
type ToolCallFunction struct {
	// Name is the name of the function the LLM has decided to call.
	Name string `json:"name"`
	// Arguments is a JSON string containing the arguments for the function.
	// Your application code needs to unmarshal this string into a struct
	// that matches the function's expected parameters.
	Arguments string `json:"arguments"`
}

// NewFunctionTool is a helper function that simplifies the creation of a new Tool.
// It reduces boilerplate and ensures the tool is created with the correct "function" type.
//
// Parameters:
//   - name: The name of the function.
//   - description: A clear description of what the function does.
//   - parameters: A JSONSchema struct defining the function's arguments.
func NewFunctionTool(name, description string, parameters JSONSchema) Tool {
	return Tool{
		Type: ToolTypeFunction,
		Function: Function{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}