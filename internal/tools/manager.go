// In file: internal/tools/manager.go
package tools

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"
)

// ToolManager holds a registry of all available tools.
type ToolManager struct {
	tools           map[string]ToolExecutor
	specs           map[string]ToolSpecification
	capabilityIndex map[string][]string
}

func NewToolManager() *ToolManager {
	return &ToolManager{
		tools:           make(map[string]ToolExecutor),
		specs:           make(map[string]ToolSpecification),
		capabilityIndex: make(map[string][]string),
	}
}

// Register adds a new tool to the manager's registry.
func (tm *ToolManager) Register(tool ToolExecutor) {
	name := tool.Definition().Function.Name
	tm.tools[name] = tool
	spec := defaultSpecification(tool.Definition())
	if provider, ok := tool.(ToolSpecProvider); ok {
		spec = provider.Specification()
		if spec.Name == "" {
			spec.Name = name
		}
		if spec.InputSchema.Type == "" {
			spec.InputSchema = tool.Definition().Function.Parameters
		}
	}
	tm.specs[name] = spec
	if spec.Capability != "" {
		tm.capabilityIndex[spec.Capability] = appendUnique(tm.capabilityIndex[spec.Capability], name)
	}
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
	record, err := tm.ExecuteWithContext(context.Background(), name, arguments)
	if err != nil {
		return "", err
	}
	return record.Result, nil
}

func (tm *ToolManager) ExecuteWithContext(ctx context.Context, name, arguments string) (ExecutionRecord, error) {
	tool, ok := tm.tools[name]
	if !ok {
		return ExecutionRecord{}, fmt.Errorf("tool '%s' not found", name)
	}
	spec := tm.specs[name]
	if err := validateArguments(spec.InputSchema, arguments); err != nil {
		normalized := normalizeResult(name, "", 0, 0, err)
		return ExecutionRecord{Name: name, Args: arguments, Status: "error", Error: err.Error(), Normalized: normalized}, err
	}

	attempts := spec.RetryPolicy.MaxAttempts
	if attempts <= 0 {
		attempts = 1
	}
	backoff := spec.RetryPolicy.Backoff
	if backoff <= 0 {
		backoff = 250 * time.Millisecond
	}
	timeout := spec.Timeout
	if timeout <= 0 {
		timeout = 15 * time.Second
	}

	var raw string
	var execErr error
	start := time.Now()
	for attempt := 1; attempt <= attempts; attempt++ {
		execCtx, cancel := context.WithTimeout(ctx, timeout)
		if contextual, ok := tool.(ContextToolExecutor); ok {
			raw, execErr = contextual.ExecuteContext(execCtx, arguments)
		} else {
			raw, execErr = tool.Execute(arguments)
		}
		cancel()
		if execErr == nil {
			duration := time.Since(start)
			normalized := normalizeResult(name, raw, duration, attempt, nil)
			return ExecutionRecord{
				Name:       name,
				Args:       arguments,
				Status:     "success",
				Result:     raw,
				DurationMS: duration.Milliseconds(),
				Attempts:   attempt,
				Normalized: normalized,
			}, nil
		}
		if attempt < attempts {
			time.Sleep(backoff)
		}
	}
	duration := time.Since(start)
	normalized := normalizeResult(name, raw, duration, attempts, execErr)
	return ExecutionRecord{
		Name:       name,
		Args:       arguments,
		Status:     "error",
		Result:     raw,
		Error:      execErr.Error(),
		DurationMS: duration.Milliseconds(),
		Attempts:   attempts,
		Normalized: normalized,
	}, execErr
}

// ToolCount returns the number of registered tools.
func (tm *ToolManager) ToolCount() int {
	return len(tm.tools)
}

func (tm *ToolManager) Plan(prompt string) ToolPlan {
	normalized := strings.ToLower(strings.TrimSpace(prompt))
	plan := ToolPlan{}

	if containsAny(normalized, "weather", "forecast", "temperature") {
		plan.NeedTools = true
		plan.SelectedTools = append(plan.SelectedTools, "getCurrentWeather")
		plan.Reason = "live weather data is required"
	}
	if containsAny(normalized, "calculate", "sum", "multiply", "divide", "+", "-", "*", "/") {
		plan.NeedTools = true
		plan.SelectedTools = append(plan.SelectedTools, "calculate")
		if plan.Reason == "" {
			plan.Reason = "calculation is better handled by a deterministic tool"
		}
	}
	if containsAny(normalized, "news", "headline", "latest") {
		if _, ok := tm.tools["getNewsHeadlines"]; ok {
			plan.NeedTools = true
			plan.SelectedTools = append(plan.SelectedTools, "getNewsHeadlines")
			if plan.Reason == "" {
				plan.Reason = "fresh news requires a live tool"
			}
		}
	}

	plan.SelectedTools = dedupe(plan.SelectedTools)
	plan.UseMultiTool = len(plan.SelectedTools) > 1
	plan.ToolBeforeReasoning = plan.NeedTools
	return plan
}

func (tm *ToolManager) DefinitionsForPlan(plan ToolPlan) []Tool {
	if !plan.NeedTools || len(plan.SelectedTools) == 0 {
		return tm.GetDefinitions()
	}
	defs := make([]Tool, 0, len(plan.SelectedTools))
	for _, name := range plan.SelectedTools {
		if tool, ok := tm.tools[name]; ok {
			defs = append(defs, tool.Definition())
		}
	}
	return defs
}

func (tm *ToolManager) Specifications() []ToolSpecification {
	specs := make([]ToolSpecification, 0, len(tm.specs))
	for _, spec := range tm.specs {
		specs = append(specs, spec)
	}
	sort.Slice(specs, func(i, j int) bool { return specs[i].Name < specs[j].Name })
	return specs
}

func appendUnique(values []string, candidate string) []string {
	for _, value := range values {
		if value == candidate {
			return values
		}
	}
	return append(values, candidate)
}

func containsAny(value string, terms ...string) bool {
	for _, term := range terms {
		if strings.Contains(value, term) {
			return true
		}
	}
	return false
}

func dedupe(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	var deduped []string
	for _, value := range values {
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		deduped = append(deduped, value)
	}
	return deduped
}
