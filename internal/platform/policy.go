package platform

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/dileep-u-k/llm-gateway/internal/llm"
	"github.com/redis/go-redis/v9"
)

const policyOverrideKeyPrefix = "platform:policy:"

type PolicyBundleStore struct {
	rdb   *redis.Client
	mu    sync.RWMutex
	local map[string]PolicyBundle
}

type PolicyDecision struct {
	TenantID            string   `json:"tenant_id,omitempty"`
	WorkspaceID         string   `json:"workspace_id,omitempty"`
	BundleName          string   `json:"bundle_name,omitempty"`
	Status              string   `json:"status,omitempty"`
	AppliedRules        []string `json:"applied_rules,omitempty"`
	Warnings            []string `json:"warnings,omitempty"`
	AllowedModels       []string `json:"allowed_models,omitempty"`
	AllowedCapabilities []string `json:"allowed_capabilities,omitempty"`
}

type Engine struct {
	cfg        *Config
	store      *PolicyBundleStore
	modelCosts map[string]map[string]float64
}

func NewPolicyBundleStore(rdb *redis.Client) *PolicyBundleStore {
	return &PolicyBundleStore{rdb: rdb, local: make(map[string]PolicyBundle)}
}

func NewEngine(cfg *Config, store *PolicyBundleStore, modelCosts map[string]map[string]float64) *Engine {
	if cfg == nil {
		cfg = DefaultConfig()
	}
	if store == nil {
		store = NewPolicyBundleStore(nil)
	}
	return &Engine{cfg: cfg, store: store, modelCosts: modelCosts}
}

func (s *PolicyBundleStore) Get(ctx context.Context, tenantID, workspaceID string) (PolicyBundle, bool, error) {
	key := policyKey(tenantID, workspaceID)
	s.mu.RLock()
	bundle, ok := s.local[key]
	s.mu.RUnlock()
	if ok {
		return bundle, true, nil
	}
	if s.rdb == nil {
		return PolicyBundle{}, false, nil
	}
	payload, err := s.rdb.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return PolicyBundle{}, false, nil
	}
	if err != nil {
		return PolicyBundle{}, false, err
	}
	if err := json.Unmarshal(payload, &bundle); err != nil {
		return PolicyBundle{}, false, err
	}
	s.mu.Lock()
	s.local[key] = bundle
	s.mu.Unlock()
	return bundle, true, nil
}

func (s *PolicyBundleStore) Put(ctx context.Context, tenantID, workspaceID string, bundle PolicyBundle) error {
	bundle.Name = strings.TrimSpace(bundle.Name)
	key := policyKey(tenantID, workspaceID)
	s.mu.Lock()
	s.local[key] = bundle
	s.mu.Unlock()
	if s.rdb == nil {
		return nil
	}
	payload, err := json.Marshal(bundle)
	if err != nil {
		return err
	}
	return s.rdb.Set(ctx, key, payload, 0).Err()
}

func (e *Engine) ResolveWorkspace(ctx context.Context, tenantID, workspaceID string) (WorkspaceContext, error) {
	if e == nil || e.cfg == nil {
		return WorkspaceContext{}, fmt.Errorf("policy engine is not configured")
	}
	tenantID = strings.TrimSpace(firstNonEmpty(tenantID, e.cfg.Defaults.TenantID))
	workspaceID = strings.TrimSpace(firstNonEmpty(workspaceID, e.cfg.Defaults.WorkspaceID))
	for _, tenant := range e.cfg.Tenants {
		if tenant.ID != tenantID {
			continue
		}
		targetWorkspaceID := workspaceID
		if targetWorkspaceID == "" {
			targetWorkspaceID = firstNonEmpty(tenant.DefaultWorkspace, e.cfg.Defaults.WorkspaceID)
		}
		for _, workspace := range tenant.Workspaces {
			if workspace.ID != targetWorkspaceID {
				continue
			}
			bundleName := firstNonEmpty(workspace.PolicyBundle, tenant.PolicyBundle, defaultPolicyBundle)
			if override, ok, err := e.store.Get(ctx, tenant.ID, workspace.ID); err == nil && ok {
				override.Normalize(bundleName)
				return WorkspaceContext{Tenant: tenant, Workspace: workspace, Policy: override}, nil
			}
			bundle, ok := e.cfg.Policies[bundleName]
			if !ok {
				return WorkspaceContext{}, fmt.Errorf("policy bundle %q is not configured", bundleName)
			}
			bundle.Normalize(bundleName)
			return WorkspaceContext{Tenant: tenant, Workspace: workspace, Policy: bundle}, nil
		}
		return WorkspaceContext{}, fmt.Errorf("workspace %q not found in tenant %q", targetWorkspaceID, tenant.ID)
	}
	return WorkspaceContext{}, fmt.Errorf("tenant %q not found", tenantID)
}

func (e *Engine) NormalizeRequest(ctx context.Context, req api.GenerationRequest, role string) (api.GenerationRequest, WorkspaceContext, *PolicyDecision, error) {
	workspaceCtx, err := e.ResolveWorkspace(ctx, req.TenantID, req.WorkspaceID)
	if err != nil {
		return req, WorkspaceContext{}, nil, err
	}
	req.TenantID = workspaceCtx.Tenant.ID
	req.WorkspaceID = workspaceCtx.Workspace.ID

	decision := &PolicyDecision{
		TenantID:            workspaceCtx.Tenant.ID,
		WorkspaceID:         workspaceCtx.Workspace.ID,
		BundleName:          workspaceCtx.Policy.Name,
		Status:              "allowed",
		AllowedCapabilities: append([]string(nil), workspaceCtx.Policy.CapabilityAllowlist...),
	}

	if workspaceCtx.Policy.MaxPromptChars > 0 && len(req.Prompt) > workspaceCtx.Policy.MaxPromptChars {
		return req, workspaceCtx, decision, fmt.Errorf("policy %q rejected prompt larger than %d characters", workspaceCtx.Policy.Name, workspaceCtx.Policy.MaxPromptChars)
	}
	totalArtifacts := len(req.Assets) + len(req.ArtifactRefs)
	if workspaceCtx.Policy.MaxArtifactsPerRequest > 0 && totalArtifacts > workspaceCtx.Policy.MaxArtifactsPerRequest {
		return req, workspaceCtx, decision, fmt.Errorf("policy %q allows at most %d assets per request", workspaceCtx.Policy.Name, workspaceCtx.Policy.MaxArtifactsPerRequest)
	}
	if req.RequiresGeneration && !workspaceCtx.Policy.GenerationAllowed {
		return req, workspaceCtx, decision, fmt.Errorf("policy %q does not allow generation workflows in workspace %s", workspaceCtx.Policy.Name, workspaceCtx.Workspace.ID)
	}
	if req.Config.StrictForce && !roleAllowed(role, workspaceCtx.Policy.StrictForceAllowedRoles) {
		return req, workspaceCtx, decision, fmt.Errorf("strict force is restricted by policy %q for role %s", workspaceCtx.Policy.Name, firstNonEmpty(role, "anonymous"))
	}
	if req.Config.ForceScope != "" && len(workspaceCtx.Policy.ForceScopeAllowlist) > 0 && !containsIgnoreCase(workspaceCtx.Policy.ForceScopeAllowlist, req.Config.ForceScope) {
		return req, workspaceCtx, decision, fmt.Errorf("force scope %q is not allowed by policy %q", req.Config.ForceScope, workspaceCtx.Policy.Name)
	}
	if req.Config.Preference == "" && workspaceCtx.Workspace.RouteDefault != "" {
		req.Config.Preference = workspaceCtx.Workspace.RouteDefault
		decision.AppliedRules = append(decision.AppliedRules, "workspace_default_route")
	}
	if workspaceCtx.Workspace.AsyncPreferred && req.SyncOrAsyncPreference == "" {
		req.SyncOrAsyncPreference = "async"
		decision.AppliedRules = append(decision.AppliedRules, "workspace_async_preference")
	}
	if shouldRequireGrounding(workspaceCtx.Policy, req.TaskType) && req.Config.AnswerMode == "" {
		req.Config.AnswerMode = "grounded"
		decision.AppliedRules = append(decision.AppliedRules, "grounding_required")
	}
	return req, workspaceCtx, decision, nil
}

func (e *Engine) ValidatePrepared(req api.GenerationRequest, prepared *llm.PreparedExecution, workspaceCtx WorkspaceContext, decision *PolicyDecision) (api.GenerationRequest, *PolicyDecision, error) {
	if decision == nil {
		decision = &PolicyDecision{
			TenantID:    workspaceCtx.Tenant.ID,
			WorkspaceID: workspaceCtx.Workspace.ID,
			BundleName:  workspaceCtx.Policy.Name,
			Status:      "allowed",
		}
	}
	if prepared == nil {
		return req, decision, nil
	}
	for _, capability := range prepared.Plan.RequiredCapabilities {
		name := string(capability)
		if len(workspaceCtx.Policy.CapabilityAllowlist) > 0 && !containsIgnoreCase(workspaceCtx.Policy.CapabilityAllowlist, name) {
			return req, decision, fmt.Errorf("policy %q blocks capability %s", workspaceCtx.Policy.Name, name)
		}
	}
	decision.AllowedCapabilities = append([]string(nil), workspaceCtx.Policy.CapabilityAllowlist...)
	if containsIgnoreCase(workspaceCtx.Policy.AsyncOnlyTaskTypes, prepared.Task.TaskType) && !strings.EqualFold(req.SyncOrAsyncPreference, "async") {
		req.SyncOrAsyncPreference = "async"
		decision.AppliedRules = append(decision.AppliedRules, "task_forced_async")
	}
	if shouldRequireGrounding(workspaceCtx.Policy, prepared.Task.TaskType) && req.Config.AnswerMode == "" {
		req.Config.AnswerMode = "grounded"
		decision.AppliedRules = append(decision.AppliedRules, "grounding_required")
	}
	return req, decision, nil
}

func (e *Engine) FilterModels(ctx context.Context, req api.GenerationRequest, enabledModels []string, capability llm.Capability, promptTokens int) ([]string, WorkspaceContext, *PolicyDecision, error) {
	workspaceCtx, err := e.ResolveWorkspace(ctx, req.TenantID, req.WorkspaceID)
	if err != nil {
		return nil, WorkspaceContext{}, nil, err
	}
	filtered := make([]string, 0, len(enabledModels))
	for _, modelID := range enabledModels {
		if modelID == "" {
			continue
		}
		provider := llm.ProviderForModel(modelID)
		if len(workspaceCtx.Policy.ProviderAllowlist) > 0 && !containsIgnoreCase(workspaceCtx.Policy.ProviderAllowlist, provider) {
			continue
		}
		if len(workspaceCtx.Policy.ModelAllowlist) > 0 && !containsIgnoreCase(workspaceCtx.Policy.ModelAllowlist, modelID) {
			continue
		}
		if len(workspaceCtx.Policy.CapabilityAllowlist) > 0 && !containsIgnoreCase(workspaceCtx.Policy.CapabilityAllowlist, string(capability)) {
			continue
		}
		if workspaceCtx.Policy.MaxEstimatedInputCost > 0 && estimatedInputCost(e.modelCosts, modelID, promptTokens) > workspaceCtx.Policy.MaxEstimatedInputCost {
			continue
		}
		filtered = append(filtered, modelID)
	}
	decision := &PolicyDecision{
		TenantID:      workspaceCtx.Tenant.ID,
		WorkspaceID:   workspaceCtx.Workspace.ID,
		BundleName:    workspaceCtx.Policy.Name,
		Status:        "allowed",
		AllowedModels: append([]string(nil), filtered...),
	}
	if len(filtered) == 0 {
		decision.Status = "blocked"
		return nil, workspaceCtx, decision, fmt.Errorf("policy %q left no models available for capability %s", workspaceCtx.Policy.Name, capability)
	}
	return filtered, workspaceCtx, decision, nil
}

func (e *Engine) Policies(ctx context.Context) ([]WorkspaceContext, error) {
	contexts := make([]WorkspaceContext, 0)
	for _, tenant := range e.cfg.Tenants {
		for _, workspace := range tenant.Workspaces {
			workspaceCtx, err := e.ResolveWorkspace(ctx, tenant.ID, workspace.ID)
			if err != nil {
				return nil, err
			}
			contexts = append(contexts, workspaceCtx)
		}
	}
	return contexts, nil
}

func (e *Engine) SavePolicyOverride(ctx context.Context, tenantID, workspaceID string, bundle PolicyBundle) error {
	bundle.Normalize(firstNonEmpty(bundle.Name, "custom"))
	return e.store.Put(ctx, tenantID, workspaceID, bundle)
}

func policyKey(tenantID, workspaceID string) string {
	return policyOverrideKeyPrefix + firstNonEmpty(tenantID, defaultTenantID) + ":" + firstNonEmpty(workspaceID, defaultWorkspaceID)
}

func (p *PolicyBundle) Normalize(name string) {
	if p == nil {
		return
	}
	if p.Name == "" {
		p.Name = name
	}
	if p.ArtifactAccessMode == "" {
		p.ArtifactAccessMode = "signed_url"
	}
	if p.ArtifactRetentionHours <= 0 {
		p.ArtifactRetentionHours = 72
	}
	if p.MaxPromptChars <= 0 {
		p.MaxPromptChars = 24000
	}
	if p.MaxArtifactsPerRequest <= 0 {
		p.MaxArtifactsPerRequest = 8
	}
}

func shouldRequireGrounding(policy PolicyBundle, taskType string) bool {
	if taskType == "" {
		return false
	}
	return containsIgnoreCase(policy.RequireGroundingFor, taskType)
}

func estimatedInputCost(modelCosts map[string]map[string]float64, modelID string, promptTokens int) float64 {
	if promptTokens <= 0 {
		return 0
	}
	costs := modelCosts[modelID]
	if len(costs) == 0 {
		return 0
	}
	return float64(promptTokens) * costs["input"]
}

func roleAllowed(role string, allowlist []string) bool {
	if len(allowlist) == 0 {
		return true
	}
	return containsIgnoreCase(allowlist, role)
}

func containsIgnoreCase(values []string, target string) bool {
	target = strings.ToLower(strings.TrimSpace(target))
	for _, value := range values {
		if strings.ToLower(strings.TrimSpace(value)) == target {
			return true
		}
	}
	return false
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
