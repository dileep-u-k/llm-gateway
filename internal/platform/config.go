package platform

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

const (
	defaultTenantID     = "default"
	defaultWorkspaceID  = "sandbox"
	defaultPolicyBundle = "default"
)

type Config struct {
	Defaults DefaultsConfig          `yaml:"defaults"`
	Policies map[string]PolicyBundle `yaml:"policies"`
	Tenants  []TenantConfig          `yaml:"tenants"`
}

type DefaultsConfig struct {
	TenantID     string `yaml:"tenant_id"`
	WorkspaceID  string `yaml:"workspace_id"`
	SignedURLTTL string `yaml:"signed_url_ttl"`
	ArtifactRoot string `yaml:"artifact_root"`
}

type TenantConfig struct {
	ID               string            `yaml:"id"`
	Name             string            `yaml:"name"`
	Description      string            `yaml:"description"`
	PolicyBundle     string            `yaml:"policy_bundle"`
	DefaultWorkspace string            `yaml:"default_workspace"`
	Labels           map[string]string `yaml:"labels"`
	Workspaces       []WorkspaceConfig `yaml:"workspaces"`
}

type WorkspaceConfig struct {
	ID               string            `yaml:"id"`
	Name             string            `yaml:"name"`
	Description      string            `yaml:"description"`
	PolicyBundle     string            `yaml:"policy_bundle"`
	RouteDefault     string            `yaml:"route_default"`
	AsyncPreferred   bool              `yaml:"async_preferred"`
	ToolPermissions  []string          `yaml:"tool_permissions"`
	MonthlyBudgetUSD float64           `yaml:"monthly_budget_usd"`
	Labels           map[string]string `yaml:"labels"`
}

type PolicyBundle struct {
	Name                    string   `yaml:"name" json:"name,omitempty"`
	Description             string   `yaml:"description" json:"description,omitempty"`
	ProviderAllowlist       []string `yaml:"provider_allowlist" json:"provider_allowlist,omitempty"`
	ModelAllowlist          []string `yaml:"model_allowlist" json:"model_allowlist,omitempty"`
	CapabilityAllowlist     []string `yaml:"capability_allowlist" json:"capability_allowlist,omitempty"`
	ToolAllowlist           []string `yaml:"tool_allowlist" json:"tool_allowlist,omitempty"`
	ForceScopeAllowlist     []string `yaml:"force_scope_allowlist" json:"force_scope_allowlist,omitempty"`
	StrictForceAllowedRoles []string `yaml:"strict_force_allowed_roles" json:"strict_force_allowed_roles,omitempty"`
	RequireGroundingFor     []string `yaml:"require_grounding_for" json:"require_grounding_for,omitempty"`
	AsyncOnlyTaskTypes      []string `yaml:"async_only_task_types" json:"async_only_task_types,omitempty"`
	RegionAllowlist         []string `yaml:"region_allowlist" json:"region_allowlist,omitempty"`
	MaxPromptChars          int      `yaml:"max_prompt_chars" json:"max_prompt_chars,omitempty"`
	MaxArtifactsPerRequest  int      `yaml:"max_artifacts_per_request" json:"max_artifacts_per_request,omitempty"`
	MaxEstimatedInputCost   float64  `yaml:"max_estimated_input_cost_usd" json:"max_estimated_input_cost_usd,omitempty"`
	GenerationAllowed       bool     `yaml:"generation_allowed" json:"generation_allowed"`
	ArtifactAccessMode      string   `yaml:"artifact_access_mode" json:"artifact_access_mode,omitempty"`
	ArtifactRetentionHours  int      `yaml:"artifact_retention_hours" json:"artifact_retention_hours,omitempty"`
}

type WorkspaceContext struct {
	Tenant    TenantConfig    `json:"tenant"`
	Workspace WorkspaceConfig `json:"workspace"`
	Policy    PolicyBundle    `json:"policy"`
}

func LoadConfig(path string) (*Config, error) {
	cfg := DefaultConfig()
	if strings.TrimSpace(path) == "" {
		return cfg, nil
	}
	payload, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, fmt.Errorf("read platform config: %w", err)
	}
	if err := yaml.Unmarshal(payload, cfg); err != nil {
		return nil, fmt.Errorf("parse platform config: %w", err)
	}
	cfg.Normalize()
	return cfg, nil
}

func DefaultConfig() *Config {
	cfg := &Config{
		Defaults: DefaultsConfig{
			TenantID:     defaultTenantID,
			WorkspaceID:  defaultWorkspaceID,
			SignedURLTTL: "15m",
			ArtifactRoot: "data/artifacts",
		},
		Policies: map[string]PolicyBundle{
			defaultPolicyBundle: {
				Name:                    "default",
				Description:             "Balanced enterprise-safe default policy with broad multimodal access and guarded force semantics.",
				ProviderAllowlist:       []string{"openai", "anthropic", "google", "mistral"},
				CapabilityAllowlist:     []string{"text_generation", "image_understanding", "image_generation", "image_editing", "transcription", "tts", "video_understanding"},
				ToolAllowlist:           []string{"calculator", "weather", "news"},
				ForceScopeAllowlist:     []string{"primary_reasoner_force", "capability_scoped_force"},
				StrictForceAllowedRoles: []string{"admin", "operator"},
				RequireGroundingFor:     []string{"knowledge_grounded"},
				AsyncOnlyTaskTypes:      []string{"video_generation_hook"},
				MaxPromptChars:          24000,
				MaxArtifactsPerRequest:  8,
				GenerationAllowed:       true,
				ArtifactAccessMode:      "signed_url",
				ArtifactRetentionHours:  72,
			},
			"restricted": {
				Name:                    "restricted",
				Description:             "Conservative policy for lower-cost workspaces with grounded answers and limited generation.",
				ProviderAllowlist:       []string{"openai", "google"},
				CapabilityAllowlist:     []string{"text_generation", "image_understanding", "tts"},
				ToolAllowlist:           []string{"calculator", "weather"},
				ForceScopeAllowlist:     []string{"primary_reasoner_force"},
				StrictForceAllowedRoles: []string{"admin"},
				RequireGroundingFor:     []string{"analysis", "knowledge_grounded"},
				MaxPromptChars:          12000,
				MaxArtifactsPerRequest:  4,
				MaxEstimatedInputCost:   0.02,
				GenerationAllowed:       false,
				ArtifactAccessMode:      "signed_url",
				ArtifactRetentionHours:  48,
			},
		},
		Tenants: []TenantConfig{
			{
				ID:               defaultTenantID,
				Name:             "Default Tenant",
				Description:      "Single-tenant starter configuration for local development and demos.",
				PolicyBundle:     defaultPolicyBundle,
				DefaultWorkspace: defaultWorkspaceID,
				Workspaces: []WorkspaceConfig{
					{
						ID:               defaultWorkspaceID,
						Name:             "Sandbox",
						Description:      "General-purpose workspace with full Phase 6 surface area enabled.",
						PolicyBundle:     defaultPolicyBundle,
						RouteDefault:     "balanced",
						AsyncPreferred:   false,
						ToolPermissions:  []string{"calculator", "weather", "news"},
						MonthlyBudgetUSD: 200,
					},
					{
						ID:               "ops",
						Name:             "Operations",
						Description:      "Policy-tight workspace intended for cost-aware platform operations.",
						PolicyBundle:     "restricted",
						RouteDefault:     "cost",
						AsyncPreferred:   true,
						ToolPermissions:  []string{"calculator", "weather"},
						MonthlyBudgetUSD: 75,
					},
				},
			},
		},
	}
	cfg.Normalize()
	return cfg
}

func (c *Config) Normalize() {
	if c == nil {
		return
	}
	if c.Defaults.TenantID == "" {
		c.Defaults.TenantID = defaultTenantID
	}
	if c.Defaults.WorkspaceID == "" {
		c.Defaults.WorkspaceID = defaultWorkspaceID
	}
	if c.Defaults.SignedURLTTL == "" {
		c.Defaults.SignedURLTTL = "15m"
	}
	if c.Defaults.ArtifactRoot == "" {
		c.Defaults.ArtifactRoot = "data/artifacts"
	}
	if len(c.Policies) == 0 {
		c.Policies = DefaultConfig().Policies
	}
	for key, bundle := range c.Policies {
		if bundle.Name == "" {
			bundle.Name = key
		}
		if bundle.ArtifactAccessMode == "" {
			bundle.ArtifactAccessMode = "signed_url"
		}
		if bundle.ArtifactRetentionHours <= 0 {
			bundle.ArtifactRetentionHours = 72
		}
		if bundle.MaxPromptChars <= 0 {
			bundle.MaxPromptChars = 24000
		}
		if bundle.MaxArtifactsPerRequest <= 0 {
			bundle.MaxArtifactsPerRequest = 8
		}
		c.Policies[key] = bundle
	}
	if len(c.Tenants) == 0 {
		c.Tenants = DefaultConfig().Tenants
	}
	for i := range c.Tenants {
		if c.Tenants[i].ID == "" {
			c.Tenants[i].ID = defaultTenantID
		}
		if c.Tenants[i].Name == "" {
			c.Tenants[i].Name = c.Tenants[i].ID
		}
		if c.Tenants[i].PolicyBundle == "" {
			c.Tenants[i].PolicyBundle = defaultPolicyBundle
		}
		if c.Tenants[i].DefaultWorkspace == "" && len(c.Tenants[i].Workspaces) > 0 {
			c.Tenants[i].DefaultWorkspace = c.Tenants[i].Workspaces[0].ID
		}
		for j := range c.Tenants[i].Workspaces {
			if c.Tenants[i].Workspaces[j].ID == "" {
				c.Tenants[i].Workspaces[j].ID = defaultWorkspaceID
			}
			if c.Tenants[i].Workspaces[j].Name == "" {
				c.Tenants[i].Workspaces[j].Name = c.Tenants[i].Workspaces[j].ID
			}
			if c.Tenants[i].Workspaces[j].PolicyBundle == "" {
				c.Tenants[i].Workspaces[j].PolicyBundle = c.Tenants[i].PolicyBundle
			}
		}
	}
}
