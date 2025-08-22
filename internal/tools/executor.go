// In file: internal/tools/executor.go
package tools

// ToolExecutor defines the standard interface for any tool that can be
// executed by the gateway's agentic core.
//
// By having all tools implement this interface, the system can manage and
// execute them in a standardized, plug-and-play fashion without needing to
// know the specific details of each tool's implementation.
type ToolExecutor interface {
	// Definition returns the tool's schema, which is provided to the LLM
	// so it understands the tool's capabilities, name, and arguments.
	Definition() Tool

	// Execute runs the actual logic of the tool. It receives the arguments
	// as a JSON string, which the LLM generates based on the tool's schema.
	// It returns a string result, which will be sent back to the LLM.
	Execute(arguments string) (string, error)
}