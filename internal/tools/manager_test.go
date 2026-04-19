package tools

import (
	"context"
	"errors"
	"testing"
	"time"
)

type flakyTool struct {
	attempts int
}

func (t *flakyTool) Definition() Tool {
	return NewFunctionTool("flaky", "test tool", JSONSchema{
		Type: "object",
		Properties: map[string]*JSONSchema{
			"value": {Type: "string"},
		},
		Required: []string{"value"},
	})
}

func (t *flakyTool) Specification() ToolSpecification {
	return ToolSpecification{
		Name:        "flaky",
		Capability:  "utility_transform",
		InputSchema: t.Definition().Function.Parameters,
		Timeout:     time.Second,
		RetryPolicy: RetryPolicy{MaxAttempts: 2, Backoff: time.Millisecond},
	}
}

func (t *flakyTool) ExecuteContext(context.Context, string) (string, error) {
	t.attempts++
	if t.attempts == 1 {
		return "", errors.New("temporary failure")
	}
	return "ok", nil
}

func (t *flakyTool) Execute(arguments string) (string, error) {
	return t.ExecuteContext(context.Background(), arguments)
}

func TestToolManagerExecuteWithContextRetriesAndNormalizes(t *testing.T) {
	manager := NewToolManager()
	tool := &flakyTool{}
	manager.Register(tool)

	record, err := manager.ExecuteWithContext(context.Background(), "flaky", `{"value":"x"}`)
	if err != nil {
		t.Fatalf("expected retry to succeed, got error: %v", err)
	}
	if record.Attempts != 2 {
		t.Fatalf("expected two attempts, got %d", record.Attempts)
	}
	if record.Normalized.Status != "success" {
		t.Fatalf("expected normalized success, got %+v", record.Normalized)
	}
}

func TestToolManagerExecuteWithContextValidatesArguments(t *testing.T) {
	manager := NewToolManager()
	manager.Register(&flakyTool{})

	if _, err := manager.ExecuteWithContext(context.Background(), "flaky", `{}`); err == nil {
		t.Fatal("expected missing required arguments to fail validation")
	}
}

func TestToolManagerPlanSelectsLiveTools(t *testing.T) {
	manager := NewToolManager()
	manager.Register(NewCalculatorTool())
	manager.Register(NewWeatherTool())

	plan := manager.Plan("What is the weather in Tokyo and calculate 2+2")
	if !plan.NeedTools {
		t.Fatal("expected plan to require tools")
	}
	if len(plan.SelectedTools) != 2 {
		t.Fatalf("expected two planned tools, got %+v", plan)
	}
}
