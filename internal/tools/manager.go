// In file: internal/tools/manager.go
package tools

import "fmt"

// ToolManager holds a registry of all available tools.
type ToolManager struct {
	tools map[string]ToolExecutor
}

func NewToolManager() *ToolManager {
	return &ToolManager{
		tools: make(map[string]ToolExecutor),
	}
}

// Register adds a new tool to the manager's registry.
func (tm *ToolManager) Register(tool ToolExecutor) {
	name := tool.Definition().Function.Name
	tm.tools[name] = tool
}

// GetDefinitions returns a slice of all registered tool definitions.
func (tm *ToolManager) GetDefinitions() []Tool {
	defs := make([]Tool, 0, len(tm.tools))
	for _, tool := range tm.tools {
		defs = append(defs, tool.Definition())
	}
	return defs
}

// Execute runs a tool by name with the given arguments.
func (tm *ToolManager) Execute(name, arguments string) (string, error) {
	tool, ok := tm.tools[name]
	if !ok {
		return "", fmt.Errorf("tool '%s' not found", name)
	}
	return tool.Execute(arguments)
}

// ToolCount returns the number of registered tools.
func (tm *ToolManager) ToolCount() int {
	return len(tm.tools)
}