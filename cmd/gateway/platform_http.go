package main

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/dileep-u-k/llm-gateway/internal/api"
	"github.com/dileep-u-k/llm-gateway/internal/llm"
	"github.com/dileep-u-k/llm-gateway/internal/observability"
	"github.com/dileep-u-k/llm-gateway/internal/ops"
	"github.com/dileep-u-k/llm-gateway/internal/platform"
	"github.com/gin-gonic/gin"
)

//go:embed web/*
var webAssets embed.FS

func (h *GatewayHandler) resolvePrincipal(c *gin.Context) (*platform.Principal, error) {
	if h.authenticator == nil {
		return &platform.Principal{ID: "anonymous", Role: "admin", Mode: "open"}, nil
	}
	return h.authenticator.Resolve(c.Request)
}

func (h *GatewayHandler) normalizeRequestForPlatform(ctx context.Context, c *gin.Context, principal *platform.Principal, req api.GenerationRequest) (api.GenerationRequest, platform.WorkspaceContext, *platform.PolicyDecision, string, error) {
	req.TenantID = firstNonEmpty(req.TenantID, c.GetHeader("X-Tenant-ID"))
	req.WorkspaceID = firstNonEmpty(req.WorkspaceID, c.GetHeader("X-Workspace-ID"))
	req.UserID = firstNonEmpty(req.UserID, c.GetHeader("X-User-ID"), principal.ID)
	req.TenantID = strings.TrimSpace(req.TenantID)
	req.WorkspaceID = strings.TrimSpace(req.WorkspaceID)
	req.UserID = strings.TrimSpace(req.UserID)
	req.ConversationID = strings.TrimSpace(req.ConversationID)
	req.SyncOrAsyncPreference = strings.TrimSpace(strings.ToLower(req.SyncOrAsyncPreference))
	req.CallbackURL = strings.TrimSpace(req.CallbackURL)

	if req.SyncOrAsyncPreference != "" && req.SyncOrAsyncPreference != "sync" && req.SyncOrAsyncPreference != "async" {
		return req, platform.WorkspaceContext{}, nil, "", fmt.Errorf("sync_or_async_preference must be one of: sync, async")
	}
	if req.CallbackURL != "" {
		if req.SyncOrAsyncPreference == "" {
			return req, platform.WorkspaceContext{}, nil, "", fmt.Errorf("callback_url requires sync_or_async_preference=async")
		}
		if req.SyncOrAsyncPreference != "async" {
			return req, platform.WorkspaceContext{}, nil, "", fmt.Errorf("callback_url is only supported for async requests")
		}
		callbackURL, parseErr := url.ParseRequestURI(req.CallbackURL)
		if parseErr != nil || callbackURL == nil || callbackURL.Host == "" {
			return req, platform.WorkspaceContext{}, nil, "", fmt.Errorf("callback_url must be a valid absolute URL")
		}
		if callbackURL.Scheme != "https" && callbackURL.Scheme != "http" {
			return req, platform.WorkspaceContext{}, nil, "", fmt.Errorf("callback_url must use http or https")
		}
	}

	if h.policyEngine == nil {
		return req, platform.WorkspaceContext{}, nil, "", nil
	}

	normalized, workspaceCtx, decision, err := h.policyEngine.NormalizeRequest(ctx, req, principal.Role)
	status := "allowed"
	summary := "generation request accepted"
	if err != nil {
		status = "blocked"
		summary = err.Error()
	}
	auditID := h.recordAudit(ctx, principal, "generation.request", "api/generate", normalized.TenantID, normalized.WorkspaceID, status, summary, map[string]string{
		"conversation_id": normalized.ConversationID,
		"task_type":       normalized.TaskType,
		"input_type":      normalized.InputType,
		"output_type":     normalized.OutputType,
	})
	if err != nil {
		return normalized, workspaceCtx, decision, auditID, err
	}
	return normalized, workspaceCtx, decision, auditID, nil
}

func (h *GatewayHandler) attachPlatformMetadata(resp *api.GenerationResponse, principal *platform.Principal, decision *platform.PolicyDecision, workspaceCtx platform.WorkspaceContext, auditID string) {
	if resp == nil {
		return
	}
	resp.Governance = buildAPIGovernanceMetadata(decision)
	authMode := "open"
	if h.authenticator != nil {
		authMode = h.authenticator.Mode()
	}
	resp.Security = &api.SecurityMetadata{
		AuthenticationMode: authMode,
		PrincipalID:        principal.ID,
		PrincipalRole:      principal.Role,
		TenantID:           firstNonEmpty(workspaceCtx.Tenant.ID, governanceTenantID(resp)),
		WorkspaceID:        firstNonEmpty(workspaceCtx.Workspace.ID, governanceWorkspaceID(resp)),
		AuditEventID:       auditID,
		SignedArtifactMode: "signed_url",
	}
	h.secureResponseArtifacts(resp)
}

func buildAPIGovernanceMetadata(decision *platform.PolicyDecision) *api.GovernanceMetadata {
	if decision == nil {
		return nil
	}
	return &api.GovernanceMetadata{
		TenantID:            decision.TenantID,
		WorkspaceID:         decision.WorkspaceID,
		PolicyBundle:        decision.BundleName,
		Status:              firstNonEmpty(decision.Status, "allowed"),
		AppliedRules:        append([]string(nil), decision.AppliedRules...),
		Warnings:            append([]string(nil), decision.Warnings...),
		AllowedModels:       append([]string(nil), decision.AllowedModels...),
		AllowedCapabilities: append([]string(nil), decision.AllowedCapabilities...),
	}
}

func (h *GatewayHandler) secureResponseArtifacts(resp *api.GenerationResponse) {
	if resp == nil || h.artifactAccess == nil {
		return
	}
	tenantID := ""
	workspaceID := ""
	if resp.Governance != nil {
		tenantID = resp.Governance.TenantID
		workspaceID = resp.Governance.WorkspaceID
	}
	sign := func(artifact *api.ArtifactMetadata) {
		if artifact == nil || artifact.ArtifactID == "" {
			return
		}
		artifact.AccessURL = h.artifactAccess.SignedURL("/api/v1/artifacts/"+artifact.ArtifactID+"/content", artifact.ArtifactID, tenantID, workspaceID)
	}
	for i := range resp.Artifacts {
		sign(&resp.Artifacts[i])
	}
	for i := range resp.GeneratedArtifacts {
		sign(&resp.GeneratedArtifacts[i])
	}
	if resp.ImageURL == "" {
		for _, artifact := range resp.GeneratedArtifacts {
			if artifact.Type == "image" && artifact.AccessURL != "" {
				resp.ImageURL = artifact.AccessURL
				break
			}
		}
	}
	if resp.AudioURL == "" {
		for _, artifact := range resp.GeneratedArtifacts {
			if artifact.Type == "audio" && artifact.AccessURL != "" {
				resp.AudioURL = artifact.AccessURL
				break
			}
		}
	}
}

func (h *GatewayHandler) recordAudit(ctx context.Context, principal *platform.Principal, action, resource, tenantID, workspaceID, status, summary string, metadata map[string]string) string {
	if h.auditLogger == nil {
		return ""
	}
	return h.auditLogger.Record(ctx, platform.AuditEvent{
		ActorID:     principal.ID,
		ActorRole:   principal.Role,
		Action:      action,
		Resource:    resource,
		TenantID:    tenantID,
		WorkspaceID: workspaceID,
		Status:      status,
		Summary:     summary,
		Metadata:    metadata,
	})
}

func (h *GatewayHandler) requireAdmin(c *gin.Context) *platform.Principal {
	principal, err := h.resolvePrincipal(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return nil
	}
	if !principal.CanAdmin() {
		c.JSON(http.StatusForbidden, gin.H{"error": "admin or operator role required"})
		return nil
	}
	return principal
}

func (h *GatewayHandler) ServeProductUI(c *gin.Context) {
	h.serveEmbeddedFile(c, "web/index.html")
}

func (h *GatewayHandler) ServeAdminUI(c *gin.Context) {
	h.serveEmbeddedFile(c, "web/admin.html")
}

func (h *GatewayHandler) ServeStaticAsset(c *gin.Context) {
	path := filepath.Clean(c.Param("filepath"))
	path = strings.TrimPrefix(path, "/")
	if path == "." || strings.Contains(path, "..") {
		c.Status(http.StatusNotFound)
		return
	}
	h.serveEmbeddedFile(c, "web/"+path)
}

func (h *GatewayHandler) serveEmbeddedFile(c *gin.Context, path string) {
	payload, err := webAssets.ReadFile(path)
	if err != nil {
		c.Status(http.StatusNotFound)
		return
	}
	switch filepath.Ext(path) {
	case ".css":
		c.Data(http.StatusOK, "text/css; charset=utf-8", payload)
	case ".js":
		c.Data(http.StatusOK, "application/javascript; charset=utf-8", payload)
	default:
		c.Data(http.StatusOK, "text/html; charset=utf-8", payload)
	}
}

func (h *GatewayHandler) HandleHealthz(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok", "time": time.Now().UTC().Format(time.RFC3339Nano)})
}

func (h *GatewayHandler) HandleReadyz(c *gin.Context) {
	state := gin.H{"status": "ready", "redis": "unknown"}
	if h.rdb != nil {
		if err := h.rdb.Ping(c.Request.Context()).Err(); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"status": "degraded", "redis": err.Error()})
			return
		}
		state["redis"] = "ok"
	}
	c.JSON(http.StatusOK, state)
}

func (h *GatewayHandler) HandlePlatformBootstrap(c *gin.Context) {
	principal, err := h.resolvePrincipal(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}
	policies, _ := h.policyEngine.Policies(c.Request.Context())
	c.JSON(http.StatusOK, gin.H{
		"principal": principal,
		"platform": gin.H{
			"auth_mode":             h.authenticationMode(),
			"default_tenant":        h.config.PlatformConfig.Defaults.TenantID,
			"default_workspace":     h.config.PlatformConfig.Defaults.WorkspaceID,
			"enabled_models":        h.config.EnabledModels,
			"enabled_image_models":  cleanModelList(strings.Split(h.config.EnabledImageModels, ",")),
			"enabled_speech_models": cleanModelList(strings.Split(h.config.EnabledSpeechModels, ",")),
			"tenants":               h.config.PlatformConfig.Tenants,
			"workspace_policies":    policies,
		},
		"sample_request": api.GenerationRequest{
			Prompt:      "Compare the uploaded design spec with the screenshot and summarize the differences.",
			TenantID:    h.config.PlatformConfig.Defaults.TenantID,
			WorkspaceID: h.config.PlatformConfig.Defaults.WorkspaceID,
			InputType:   "mixed",
			OutputType:  "summary",
			Config: api.GenerationConfig{
				Preference: "balanced",
				AnswerMode: "grounded",
			},
		},
	})
}

func (h *GatewayHandler) HandlePlatformMe(c *gin.Context) {
	principal, err := h.resolvePrincipal(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"principal": principal,
		"auth_mode": h.authenticationMode(),
	})
}

func (h *GatewayHandler) HandleAdminOverview(c *gin.Context) {
	principal := h.requireAdmin(c)
	if principal == nil {
		return
	}
	jobs, _ := h.listRecentJobs(c.Request.Context(), 20)
	orchestrations, _ := h.listRecentOrchestrations(c.Request.Context(), 20)
	artifacts, _ := h.listRecentArtifacts(c.Request.Context(), 20)
	auditEvents, _ := h.auditLogger.List(c.Request.Context(), 50)
	policies, _ := h.policyEngine.Policies(c.Request.Context())

	c.JSON(http.StatusOK, gin.H{
		"principal":      principal,
		"metrics":        observability.Default().Snapshot(),
		"providers":      h.controlPlane.Providers().List(),
		"models":         h.controlPlane.Models().List(),
		"capabilities":   h.controlPlane.Capabilities().List(),
		"tenants":        h.config.PlatformConfig.Tenants,
		"policies":       policies,
		"recent_jobs":    jobs,
		"orchestrations": orchestrations,
		"artifacts":      artifacts,
		"audit_events":   auditEvents,
	})
}

func (h *GatewayHandler) HandleAdminPolicies(c *gin.Context) {
	if h.requireAdmin(c) == nil {
		return
	}
	policies, err := h.policyEngine.Policies(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"policies": policies})
}

func (h *GatewayHandler) HandleAdminPolicyUpsert(c *gin.Context) {
	principal := h.requireAdmin(c)
	if principal == nil {
		return
	}
	var bundle platform.PolicyBundle
	if err := c.ShouldBindJSON(&bundle); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	tenantID := c.Param("tenant")
	workspaceID := c.Param("workspace")
	if err := h.policyEngine.SavePolicyOverride(c.Request.Context(), tenantID, workspaceID, bundle); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	auditID := h.recordAudit(c.Request.Context(), principal, "policy.override", "policy/"+tenantID+"/"+workspaceID, tenantID, workspaceID, "updated", "workspace policy override updated", map[string]string{"bundle": bundle.Name})
	c.JSON(http.StatusOK, gin.H{"status": "updated", "audit_event_id": auditID})
}

func (h *GatewayHandler) HandleAdminPolicySimulation(c *gin.Context) {
	principal := h.requireAdmin(c)
	if principal == nil {
		return
	}
	var req api.GenerationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	req, workspaceCtx, decision, err := h.policyEngine.NormalizeRequest(c.Request.Context(), req, principal.Role)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	prepared, prepErr := h.prepareExecutionContext(c.Request.Context(), req)
	if prepErr == nil && prepared != nil {
		req, decision, prepErr = h.policyEngine.ValidatePrepared(req, prepared, workspaceCtx, decision)
	}
	if prepErr != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": prepErr.Error(), "decision": decision})
		return
	}
	estimatedTokens := estimatePromptTokens(req.Prompt, req.History)
	available, _, filterDecision, filterErr := h.policyEngine.FilterModels(c.Request.Context(), req, h.config.EnabledModels, primaryCapability(prepared), estimatedTokens)
	c.JSON(http.StatusOK, gin.H{
		"normalized_request": req,
		"workspace":          workspaceCtx,
		"decision":           decision,
		"prepared":           buildAPIExecutionPlanMetadata(prepared),
		"allowed_models":     available,
		"filter_decision":    filterDecision,
		"filter_error":       errorString(filterErr),
	})
}

func (h *GatewayHandler) HandleAdminJobs(c *gin.Context) {
	if h.requireAdmin(c) == nil {
		return
	}
	jobs, err := h.listRecentJobs(c.Request.Context(), 50)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"jobs": jobs})
}

func (h *GatewayHandler) HandleAdminOrchestrations(c *gin.Context) {
	if h.requireAdmin(c) == nil {
		return
	}
	records, err := h.listRecentOrchestrations(c.Request.Context(), 50)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"orchestrations": records})
}

func (h *GatewayHandler) HandleAdminAudit(c *gin.Context) {
	if h.requireAdmin(c) == nil {
		return
	}
	events, err := h.auditLogger.List(c.Request.Context(), 100)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"events": events})
}

func (h *GatewayHandler) HandleSchema(c *gin.Context) {
	if h.requireAdmin(c) == nil {
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"generation_request_fields": []gin.H{
			{"name": "prompt", "required": true, "type": "string"},
			{"name": "tenant_id", "required": false, "type": "string"},
			{"name": "workspace_id", "required": false, "type": "string"},
			{"name": "conversation_id", "required": false, "type": "string"},
			{"name": "history", "required": false, "type": "[]message"},
			{"name": "assets", "required": false, "type": "[]asset"},
			{"name": "artifact_refs", "required": false, "type": "[]artifact_reference"},
			{"name": "config", "required": false, "type": "generation_config"},
		},
		"key_endpoints": []string{
			"POST /api/v1/generate",
			"POST /api/v1/assets/upload",
			"GET /api/v1/artifacts/:id/content",
			"GET /api/v1/platform/admin/overview",
			"POST /api/v1/platform/admin/policies/simulate",
		},
	})
}

func (h *GatewayHandler) HandleAssetUpload(c *gin.Context) {
	principal, err := h.resolvePrincipal(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}
	file, header, err := c.Request.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "multipart field 'file' is required"})
		return
	}
	defer file.Close()

	req := api.GenerationRequest{
		TenantID:    firstNonEmpty(c.PostForm("tenant_id"), c.GetHeader("X-Tenant-ID")),
		WorkspaceID: firstNonEmpty(c.PostForm("workspace_id"), c.GetHeader("X-Workspace-ID")),
	}
	req, workspaceCtx, _, auditID, err := h.normalizeRequestForPlatform(c.Request.Context(), c, principal, req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	artifact, err := h.persistUploadedArtifact(c.Request.Context(), req, workspaceCtx, header, file, c.PostForm("role"), principal)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	metadata := buildAPIArtifactMetadataFromRecords([]llm.ArtifactRecord{artifact})
	if len(metadata) > 0 {
		metadata[0].AccessURL = h.artifactAccess.SignedURL("/api/v1/artifacts/"+artifact.ArtifactID+"/content", artifact.ArtifactID, req.TenantID, req.WorkspaceID)
	}
	h.recordAudit(c.Request.Context(), principal, "artifact.upload", "artifact/"+artifact.ArtifactID, req.TenantID, req.WorkspaceID, "completed", "artifact uploaded", map[string]string{
		"artifact_id": artifact.ArtifactID,
		"filename":    header.Filename,
	})
	c.JSON(http.StatusCreated, gin.H{
		"artifact":       metadata[0],
		"audit_event_id": auditID,
	})
}

func (h *GatewayHandler) HandleArtifactMetadata(c *gin.Context) {
	principal, req, _, _, ok := h.authorizeArtifactRequest(c)
	if !ok {
		return
	}
	record, found, err := h.multimodal.ArtifactRegistry().Get(c.Request.Context(), c.Param("id"))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if !found {
		c.JSON(http.StatusNotFound, gin.H{"error": "artifact not found"})
		return
	}
	metadata := buildAPIArtifactMetadataFromRecords([]llm.ArtifactRecord{record})
	if len(metadata) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "artifact not found"})
		return
	}
	metadata[0].AccessURL = h.artifactAccess.SignedURL("/api/v1/artifacts/"+record.ArtifactID+"/content", record.ArtifactID, req.TenantID, req.WorkspaceID)
	h.recordAudit(c.Request.Context(), principal, "artifact.metadata", "artifact/"+record.ArtifactID, req.TenantID, req.WorkspaceID, "completed", "artifact metadata viewed", map[string]string{"artifact_id": record.ArtifactID})
	c.JSON(http.StatusOK, metadata[0])
}

func (h *GatewayHandler) HandleArtifactContent(c *gin.Context) {
	principal, req, workspaceCtx, auditID, ok := h.authorizeArtifactRequest(c)
	if !ok {
		return
	}
	artifactID := c.Param("id")
	record, found, err := h.multimodal.ArtifactRegistry().Get(c.Request.Context(), artifactID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if !found {
		c.JSON(http.StatusNotFound, gin.H{"error": "artifact not found"})
		return
	}
	if !h.canAccessArtifact(record, req.TenantID, req.WorkspaceID) {
		c.JSON(http.StatusForbidden, gin.H{"error": "artifact is outside the active tenant/workspace boundary"})
		return
	}
	h.recordAudit(c.Request.Context(), principal, "artifact.read", "artifact/"+artifactID, req.TenantID, req.WorkspaceID, "completed", "artifact content accessed", map[string]string{"artifact_id": artifactID, "audit_event_id": auditID})
	if isRemoteURI(record.SourceURI) {
		c.Redirect(http.StatusTemporaryRedirect, record.SourceURI)
		return
	}
	if strings.TrimSpace(record.Text) != "" {
		c.Data(http.StatusOK, firstNonEmpty(record.MimeType, "text/plain; charset=utf-8"), []byte(record.Text))
		return
	}
	path := record.SourceURI
	if !filepath.IsAbs(path) {
		path = filepath.Join(h.artifactRoot, path)
	}
	c.FileAttachment(path, filepath.Base(path))
	_ = workspaceCtx
}

func (h *GatewayHandler) HandleArtifacts(c *gin.Context) {
	principal, err := h.resolvePrincipal(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}
	req := api.GenerationRequest{
		TenantID:    firstNonEmpty(c.Query("tenant_id"), c.GetHeader("X-Tenant-ID")),
		WorkspaceID: firstNonEmpty(c.Query("workspace_id"), c.GetHeader("X-Workspace-ID")),
	}
	req, workspaceCtx, _, _, err := h.normalizeRequestForPlatform(c.Request.Context(), c, principal, req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	artifacts, err := h.listRecentArtifacts(c.Request.Context(), 50)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	filtered := make([]api.ArtifactMetadata, 0, len(artifacts))
	for _, artifact := range artifacts {
		if !h.canAccessArtifactRecord(artifact, req.TenantID, req.WorkspaceID) {
			continue
		}
		artifact.AccessURL = h.artifactAccess.SignedURL("/api/v1/artifacts/"+artifact.ArtifactID+"/content", artifact.ArtifactID, req.TenantID, req.WorkspaceID)
		filtered = append(filtered, artifact)
	}
	c.JSON(http.StatusOK, gin.H{"artifacts": filtered, "workspace": workspaceCtx})
}

func (h *GatewayHandler) authorizeArtifactRequest(c *gin.Context) (*platform.Principal, api.GenerationRequest, platform.WorkspaceContext, string, bool) {
	principal, err := h.resolvePrincipal(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return nil, api.GenerationRequest{}, platform.WorkspaceContext{}, "", false
	}

	req := api.GenerationRequest{
		TenantID:    firstNonEmpty(c.Query("tenant"), c.Query("tenant_id"), c.GetHeader("X-Tenant-ID")),
		WorkspaceID: firstNonEmpty(c.Query("workspace"), c.Query("workspace_id"), c.GetHeader("X-Workspace-ID")),
	}
	req, workspaceCtx, _, auditID, err := h.normalizeRequestForPlatform(c.Request.Context(), c, principal, req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return nil, api.GenerationRequest{}, platform.WorkspaceContext{}, "", false
	}
	if sig := strings.TrimSpace(c.Query("signature")); sig != "" {
		expires, _ := strconv.ParseInt(c.Query("expires"), 10, 64)
		if !h.artifactAccess.Validate(c.Param("id"), req.TenantID, req.WorkspaceID, expires, sig) {
			c.JSON(http.StatusForbidden, gin.H{"error": "signed artifact URL is invalid or expired"})
			return nil, api.GenerationRequest{}, platform.WorkspaceContext{}, "", false
		}
	}
	return principal, req, workspaceCtx, auditID, true
}

func (h *GatewayHandler) persistUploadedArtifact(ctx context.Context, req api.GenerationRequest, workspaceCtx platform.WorkspaceContext, header *multipart.FileHeader, file multipart.File, role string, principal *platform.Principal) (llm.ArtifactRecord, error) {
	registry := h.multimodal.ArtifactRegistry()
	if registry == nil {
		return llm.ArtifactRecord{}, fmt.Errorf("artifact registry is not configured")
	}
	root := platform.CleanArtifactStorageRoot(h.artifactRoot)
	dir := filepath.Join(root, req.TenantID, req.WorkspaceID, "uploads")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return llm.ArtifactRecord{}, err
	}
	targetPath := filepath.Join(dir, llm.GenerateCacheKey(header.Filename+"|"+time.Now().UTC().Format(time.RFC3339Nano))+"_"+sanitizeFilename(header.Filename))
	bytesWritten, extractedText, err := writeUploadedFile(targetPath, file, header)
	if err != nil {
		return llm.ArtifactRecord{}, err
	}
	artifactType := inferAssetType(header.Header.Get("Content-Type"), header.Filename)
	return registry.Register(ctx, llm.ArtifactRecord{
		Name:      header.Filename,
		Type:      artifactType,
		MimeType:  header.Header.Get("Content-Type"),
		SourceURI: targetPath,
		SizeBytes: bytesWritten,
		Role:      firstNonEmpty(role, "uploaded_input"),
		Text:      extractedText,
		Metadata: map[string]string{
			"tenant_id":     req.TenantID,
			"workspace_id":  req.WorkspaceID,
			"uploaded_by":   principal.ID,
			"uploaded_role": principal.Role,
			"policy_bundle": workspaceCtx.Policy.Name,
		},
	})
}

func writeUploadedFile(targetPath string, file multipart.File, header *multipart.FileHeader) (int64, string, error) {
	out, err := os.Create(targetPath)
	if err != nil {
		return 0, "", err
	}
	defer out.Close()

	var builder strings.Builder
	multi := io.MultiWriter(out, &limitedStringWriter{Builder: &builder, Limit: 64 * 1024})
	written, err := io.Copy(multi, file)
	if err != nil {
		return 0, "", err
	}
	contentType := strings.ToLower(header.Header.Get("Content-Type"))
	if strings.HasPrefix(contentType, "text/") || strings.Contains(header.Filename, ".md") || strings.Contains(header.Filename, ".txt") || strings.Contains(header.Filename, ".json") {
		return written, builder.String(), nil
	}
	return written, "", nil
}

type limitedStringWriter struct {
	Builder *strings.Builder
	Limit   int
}

func (w *limitedStringWriter) Write(p []byte) (int, error) {
	if w.Builder == nil || w.Limit <= 0 {
		return len(p), nil
	}
	remaining := w.Limit - w.Builder.Len()
	if remaining <= 0 {
		return len(p), nil
	}
	if len(p) > remaining {
		p = p[:remaining]
	}
	_, _ = w.Builder.Write(p)
	return len(p), nil
}

func (h *GatewayHandler) canAccessArtifact(record llm.ArtifactRecord, tenantID, workspaceID string) bool {
	return h.canAccessArtifactRecord(buildAPIArtifactMetadataFromRecords([]llm.ArtifactRecord{record})[0], tenantID, workspaceID)
}

func (h *GatewayHandler) canAccessArtifactRecord(record api.ArtifactMetadata, tenantID, workspaceID string) bool {
	if tenantID == "" && workspaceID == "" {
		return true
	}
	recordTenant := record.Metadata["tenant_id"]
	recordWorkspace := record.Metadata["workspace_id"]
	return (recordTenant == "" || recordTenant == tenantID) && (recordWorkspace == "" || recordWorkspace == workspaceID)
}

func (h *GatewayHandler) listRecentJobs(ctx context.Context, limit int64) ([]api.JobStatusResponse, error) {
	if h.rdb == nil {
		return nil, nil
	}
	ids, err := h.rdb.LRange(ctx, "async:job_index", 0, limit-1).Result()
	if err != nil {
		return nil, err
	}
	jobs := make([]api.JobStatusResponse, 0, len(ids))
	for _, id := range ids {
		payload, err := h.rdb.Get(ctx, "async:job:"+id).Bytes()
		if err != nil {
			continue
		}
		var job ops.JobRecord
		if json.Unmarshal(payload, &job) != nil {
			continue
		}
		if status := job.StatusResponse(); status != nil {
			jobs = append(jobs, *status)
		}
	}
	return jobs, nil
}

func (h *GatewayHandler) listRecentOrchestrations(ctx context.Context, limit int64) ([]map[string]any, error) {
	if h.rdb == nil {
		return nil, nil
	}
	ids, err := h.rdb.LRange(ctx, "orchestration_index:all", 0, limit-1).Result()
	if err != nil {
		return nil, err
	}
	records := make([]map[string]any, 0, len(ids))
	for _, id := range ids {
		payload, err := h.rdb.Get(ctx, "orchestration:"+id).Bytes()
		if err != nil {
			continue
		}
		var record map[string]any
		if json.Unmarshal(payload, &record) == nil {
			records = append(records, record)
		}
	}
	return records, nil
}

func (h *GatewayHandler) listRecentArtifacts(ctx context.Context, limit int64) ([]api.ArtifactMetadata, error) {
	if h.multimodal == nil {
		return nil, nil
	}
	records, err := h.multimodal.ArtifactRegistry().List(ctx, limit)
	if err != nil {
		return nil, err
	}
	metadata := buildAPIArtifactMetadataFromRecords(records)
	sort.SliceStable(metadata, func(i, j int) bool { return metadata[i].Version > metadata[j].Version })
	return metadata, nil
}

func primaryCapability(prepared *llm.PreparedExecution) llm.Capability {
	if prepared == nil || prepared.Task.PrimaryCapability == "" {
		return llm.CapabilityTextGeneration
	}
	return prepared.Task.PrimaryCapability
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func governanceTenantID(resp *api.GenerationResponse) string {
	if resp == nil || resp.Governance == nil {
		return ""
	}
	return resp.Governance.TenantID
}

func governanceWorkspaceID(resp *api.GenerationResponse) string {
	if resp == nil || resp.Governance == nil {
		return ""
	}
	return resp.Governance.WorkspaceID
}

func (h *GatewayHandler) authenticationMode() string {
	if h.authenticator == nil {
		return "open"
	}
	return h.authenticator.Mode()
}

func inferAssetType(contentType, name string) string {
	lowerType := strings.ToLower(contentType)
	lowerName := strings.ToLower(name)
	switch {
	case strings.HasPrefix(lowerType, "image/"), strings.HasSuffix(lowerName, ".png"), strings.HasSuffix(lowerName, ".jpg"), strings.HasSuffix(lowerName, ".jpeg"), strings.HasSuffix(lowerName, ".webp"):
		return "image"
	case strings.HasPrefix(lowerType, "audio/"), strings.HasSuffix(lowerName, ".mp3"), strings.HasSuffix(lowerName, ".wav"), strings.HasSuffix(lowerName, ".m4a"):
		return "audio"
	case strings.HasPrefix(lowerType, "video/"), strings.HasSuffix(lowerName, ".mp4"), strings.HasSuffix(lowerName, ".mov"), strings.HasSuffix(lowerName, ".mkv"):
		return "video"
	case strings.HasPrefix(lowerType, "text/"), strings.HasSuffix(lowerName, ".md"), strings.HasSuffix(lowerName, ".txt"), strings.HasSuffix(lowerName, ".json"):
		return "document"
	default:
		return "document"
	}
}

func sanitizeFilename(value string) string {
	value = strings.TrimSpace(value)
	value = strings.ReplaceAll(value, string(filepath.Separator), "_")
	value = strings.ReplaceAll(value, " ", "_")
	if value == "" {
		return "artifact"
	}
	return value
}

func isRemoteURI(value string) bool {
	lower := strings.ToLower(strings.TrimSpace(value))
	return strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "https://")
}
