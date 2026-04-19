package platform

import (
	"context"
	"testing"

	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/dileep-u-k/llm-gateway/internal/llm"
)

func TestEngineNormalizeAndFilterModels(t *testing.T) {
	engine := NewEngine(DefaultConfig(), NewPolicyBundleStore(nil), map[string]map[string]float64{
		"gpt-4o":   {"input": 0.000005},
		"claude-3": {"input": 0.000003},
	})
	req := api.GenerationRequest{
		Prompt:      "Summarize the latest doc",
		TenantID:    "default",
		WorkspaceID: "ops",
	}
	req, workspace, decision, err := engine.NormalizeRequest(context.Background(), req, "user")
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}
	if req.Config.Preference != "cost" {
		t.Fatalf("expected workspace route default to apply, got %q", req.Config.Preference)
	}
	if workspace.Workspace.ID != "ops" || decision.BundleName != "restricted" {
		t.Fatalf("unexpected workspace resolution: %+v %+v", workspace, decision)
	}

	filtered, _, filterDecision, err := engine.FilterModels(context.Background(), req, []string{"gpt-4o", "claude-3"}, llm.CapabilityTextGeneration, 1000)
	if err != nil {
		t.Fatalf("FilterModels returned error: %v", err)
	}
	if len(filtered) != 1 || filtered[0] != "gpt-4o" {
		t.Fatalf("expected only gpt-4o to survive restricted provider policy, got %v", filtered)
	}
	if filterDecision == nil || len(filterDecision.AllowedModels) != 1 {
		t.Fatalf("expected filter decision metadata, got %+v", filterDecision)
	}
}

func TestStrictForceRejectedForRestrictedRole(t *testing.T) {
	engine := NewEngine(DefaultConfig(), NewPolicyBundleStore(nil), nil)
	_, _, _, err := engine.NormalizeRequest(context.Background(), api.GenerationRequest{
		Prompt: "Force this strictly",
		Config: api.GenerationConfig{
			StrictForce: true,
			ForceScope:  "strict_end_to_end_force",
		},
	}, "user")
	if err == nil {
		t.Fatal("expected strict force to be rejected for non-admin role")
	}
}

func TestSignedArtifactURLValidation(t *testing.T) {
	layer := NewSignedArtifactAccessLayerFromEnv("15m")
	url := layer.SignedURL("/api/v1/artifacts/art-1/content", "art-1", "default", "sandbox")
	if url == "" {
		t.Fatal("expected signed URL")
	}
}
