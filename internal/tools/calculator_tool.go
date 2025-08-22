// In file: internal/tools/calculator_tool.go
package tools

import (
	"encoding/json"
	"fmt"
)

// --- Calculator Tool Implementation ---

// CalculatorTool is a concrete implementation of a tool that performs basic arithmetic.
type CalculatorTool struct{}

// Statically verify that CalculatorTool implements the ToolExecutor interface.
// This ensures our tool adheres to the standard contract for all tools in the system.
var _ ToolExecutor = (*CalculatorTool)(nil)

// NewCalculatorTool creates a new instance of the CalculatorTool.
// Even though this tool has no dependencies (like an HTTP client), a constructor
// provides a consistent creation pattern across all tools.
func NewCalculatorTool() *CalculatorTool {
	return &CalculatorTool{}
}

// Definition describes the tool to the LLM using our type-safe structures.
//
// The key improvement here is asking for structured parameters (`operand1`, `operator`, `operand2`)
// instead of a single "expression" string. This makes the tool far more robust, as it
// eliminates the need for fragile string parsing in our Go code.
func (ct *CalculatorTool) Definition() Tool {
	return NewFunctionTool(
		"calculate",
		"Performs a basic arithmetic calculation (add, subtract, multiply, divide).",
		JSONSchema{
			Type: "object",
			Properties: map[string]*JSONSchema{
				"operand1": {
					Type:        "number",
					Description: "The first number in the calculation.",
				},
				"operator": {
					Type:        "string",
					Description: "The operator to use. Must be one of '+', '-', '*', '/'.",
				},
				"operand2": {
					Type:        "number",
					Description: "The second number in the calculation.",
				},
			},
			Required: []string{"operand1", "operator", "operand2"},
		},
	)
}

// Execute runs the tool's actual logic. It takes the structured arguments
// provided by the LLM and performs the requested calculation.
func (ct *CalculatorTool) Execute(arguments string) (string, error) {
	// Unmarshal the JSON arguments string from the LLM into our new, structured Go type.
	var args struct {
		Operand1 float64 `json:"operand1"`
		Operand2 float64 `json:"operand2"`
		Operator string  `json:"operator"`
	}
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return "", fmt.Errorf("invalid arguments for calculator: %w", err)

	}

	// The fragile string parsing logic is now gone. We can directly use the structured data.
	var result float64
	switch args.Operator {
	case "+":
		result = args.Operand1 + args.Operand2
	case "-":
		result = args.Operand1 - args.Operand2
	case "*":
		result = args.Operand1 * args.Operand2
	case "/":
		if args.Operand2 == 0 {
			// Return a user-friendly error message that the LLM can understand and relay.
			return "Error: Division by zero is not allowed.", nil
		}
		result = args.Operand1 / args.Operand2
	default:
		// This case is less likely to be hit now, thanks to the improved schema,
		// but it's still good practice to handle it.
		return fmt.Sprintf("Error: Unsupported operator '%s'. Please use +, -, *, or /.", args.Operator), nil
	}

	// Return a clear, natural language result for the LLM.
	// We use a precision-aware format specifier `%g` to avoid trailing zeros (e.g., "10.000000").
	return fmt.Sprintf("The result is %g.", result), nil
}