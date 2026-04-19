package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

type TrustLevel string

const (
	TrustLevelHigh   TrustLevel = "high"
	TrustLevelMedium TrustLevel = "medium"
	TrustLevelLow    TrustLevel = "low"
)

type RetryPolicy struct {
	MaxAttempts int           `json:"max_attempts"`
	Backoff     time.Duration `json:"backoff"`
}

type ToolSpecification struct {
	Name            string        `json:"name"`
	Capability      string        `json:"capability"`
	InputSchema     JSONSchema    `json:"input_schema"`
	OutputSchema    JSONSchema    `json:"output_schema"`
	Timeout         time.Duration `json:"timeout"`
	RetryPolicy     RetryPolicy   `json:"retry_policy"`
	PermissionScope string        `json:"permission_scope"`
	TrustLevel      TrustLevel    `json:"trust_level"`
	AsyncSuitable   bool          `json:"async_suitable"`
}

type ToolPlan struct {
	NeedTools            bool     `json:"need_tools"`
	SelectedTools        []string `json:"selected_tools,omitempty"`
	Reason               string   `json:"reason,omitempty"`
	RetrievalBeforeTools bool     `json:"retrieval_before_tools"`
	ToolBeforeReasoning  bool     `json:"tool_before_reasoning"`
	UseMultiTool         bool     `json:"use_multi_tool"`
}

type NormalizedResult struct {
	ToolName     string         `json:"tool_name"`
	Status       string         `json:"status"`
	Summary      string         `json:"summary"`
	Data         map[string]any `json:"data,omitempty"`
	Raw          string         `json:"raw,omitempty"`
	DurationMS   int64          `json:"duration_ms,omitempty"`
	AttemptCount int            `json:"attempt_count,omitempty"`
}

type ExecutionRecord struct {
	Name       string           `json:"name"`
	Args       string           `json:"args"`
	Status     string           `json:"status"`
	Result     string           `json:"result,omitempty"`
	Error      string           `json:"error,omitempty"`
	DurationMS int64            `json:"duration_ms,omitempty"`
	Attempts   int              `json:"attempts,omitempty"`
	Normalized NormalizedResult `json:"normalized"`
}

type ToolSpecProvider interface {
	Specification() ToolSpecification
}

type ContextToolExecutor interface {
	ExecuteContext(ctx context.Context, arguments string) (string, error)
}

func defaultSpecification(def Tool) ToolSpecification {
	return ToolSpecification{
		Name:            def.Function.Name,
		Capability:      "utility",
		InputSchema:     def.Function.Parameters,
		OutputSchema:    JSONSchema{Type: "object"},
		Timeout:         15 * time.Second,
		RetryPolicy:     RetryPolicy{MaxAttempts: 1, Backoff: 250 * time.Millisecond},
		PermissionScope: "gateway:tool",
		TrustLevel:      TrustLevelMedium,
		AsyncSuitable:   false,
	}
}

func validateArguments(schema JSONSchema, arguments string) error {
	if strings.TrimSpace(arguments) == "" {
		if len(schema.Required) > 0 {
			return fmt.Errorf("tool arguments are required")
		}
		return nil
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(arguments), &payload); err != nil {
		return fmt.Errorf("invalid tool arguments JSON: %w", err)
	}
	for _, required := range schema.Required {
		if _, ok := payload[required]; !ok {
			return fmt.Errorf("missing required tool argument %q", required)
		}
	}
	return nil
}

func normalizeResult(name, raw string, duration time.Duration, attempts int, err error) NormalizedResult {
	status := "success"
	summary := strings.TrimSpace(raw)
	if err != nil {
		status = "error"
		summary = err.Error()
	}
	return NormalizedResult{
		ToolName:     name,
		Status:       status,
		Summary:      summary,
		Raw:          raw,
		DurationMS:   duration.Milliseconds(),
		AttemptCount: attempts,
		Data: map[string]any{
			"summary": summary,
		},
	}
}
