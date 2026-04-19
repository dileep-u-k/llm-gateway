package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	memoryShortTermPrefix = "memory:short:"
	memorySummaryPrefix   = "memory:summary:"
	memoryWorkingPrefix   = "memory:working:"
	memoryArtifactPrefix  = "memory:artifacts:"
	memoryTTL             = 24 * time.Hour
)

type MemoryEventKind string

const (
	MemoryEventUser      MemoryEventKind = "user_message"
	MemoryEventAssistant MemoryEventKind = "assistant_message"
	MemoryEventTool      MemoryEventKind = "tool_output"
	MemoryEventEvidence  MemoryEventKind = "retrieved_evidence"
	MemoryEventRoute     MemoryEventKind = "route_decision"
)

type MemoryEvent struct {
	Kind      MemoryEventKind   `json:"kind"`
	Content   string            `json:"content"`
	Role      string            `json:"role,omitempty"`
	ToolName  string            `json:"tool_name,omitempty"`
	Source    string            `json:"source,omitempty"`
	Section   string            `json:"section,omitempty"`
	Timestamp string            `json:"timestamp,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type SessionSummaryMemory struct {
	Summary             string   `json:"summary,omitempty"`
	UserGoalProgression string   `json:"user_goal_progression,omitempty"`
	ActiveReasoning     string   `json:"active_reasoning,omitempty"`
	CurrentSubtask      string   `json:"current_subtask,omitempty"`
	ActiveArtifacts     []string `json:"active_artifacts,omitempty"`
	UpdatedAt           string   `json:"updated_at,omitempty"`
}

type StructuredWorkingMemory struct {
	SessionMode       string            `json:"session_mode,omitempty"`
	ForceScope        string            `json:"force_scope,omitempty"`
	EffectiveModel    string            `json:"effective_model,omitempty"`
	ActiveConstraints []string          `json:"active_constraints,omitempty"`
	ActivePolicies    []string          `json:"active_policies,omitempty"`
	KnowledgeSources  []string          `json:"knowledge_sources,omitempty"`
	PendingRefs       []string          `json:"pending_refs,omitempty"`
	WorkflowState     map[string]string `json:"workflow_state,omitempty"`
	UpdatedAt         string            `json:"updated_at,omitempty"`
}

type ArtifactMemoryLink struct {
	Name      string `json:"name,omitempty"`
	Kind      string `json:"kind,omitempty"`
	Source    string `json:"source,omitempty"`
	Reference string `json:"reference,omitempty"`
}

type SessionMemorySnapshot struct {
	ConversationID string                  `json:"conversation_id,omitempty"`
	ShortTerm      []MemoryEvent           `json:"short_term,omitempty"`
	Summary        SessionSummaryMemory    `json:"summary,omitempty"`
	Working        StructuredWorkingMemory `json:"working,omitempty"`
	Artifacts      []ArtifactMemoryLink    `json:"artifacts,omitempty"`
}

type MemoryUpdate struct {
	ConversationID  string
	UserPrompt      string
	ResponseContent string
	ToolCalls       []MemoryToolCall
	Retrieval       *RetrievalResult
	Route           *RouteSelection
	SessionMode     string
	ForceScope      string
	EffectiveModel  string
	AnswerMode      string
	ArtifactLinks   []ArtifactMemoryLink
}

type MemoryToolCall struct {
	Name   string
	Result string
	Status string
}

type MemoryEngine struct {
	redisClient    *redis.Client
	shortTermLimit int
}

func NewMemoryEngine(redisClient *redis.Client) *MemoryEngine {
	return &MemoryEngine{
		redisClient:    redisClient,
		shortTermLimit: 12,
	}
}

func (m *MemoryEngine) LoadSnapshot(ctx context.Context, conversationID string, history []Message) (*SessionMemorySnapshot, error) {
	if conversationID == "" || m == nil || m.redisClient == nil {
		return m.ephemeralSnapshot(conversationID, history), nil
	}

	snapshot := &SessionMemorySnapshot{ConversationID: conversationID}
	if err := m.loadJSON(ctx, memoryShortTermPrefix+conversationID, &snapshot.ShortTerm); err != nil {
		return nil, err
	}
	if err := m.loadJSON(ctx, memorySummaryPrefix+conversationID, &snapshot.Summary); err != nil {
		return nil, err
	}
	if err := m.loadJSON(ctx, memoryWorkingPrefix+conversationID, &snapshot.Working); err != nil {
		return nil, err
	}
	if err := m.loadJSON(ctx, memoryArtifactPrefix+conversationID, &snapshot.Artifacts); err != nil {
		return nil, err
	}

	if len(snapshot.ShortTerm) == 0 && snapshot.Summary.Summary == "" {
		return m.ephemeralSnapshot(conversationID, history), nil
	}
	return snapshot, nil
}

func (m *MemoryEngine) UpdateSession(ctx context.Context, update MemoryUpdate) (*SessionMemorySnapshot, error) {
	snapshot, err := m.LoadSnapshot(ctx, update.ConversationID, nil)
	if err != nil {
		return nil, err
	}
	if snapshot == nil {
		snapshot = &SessionMemorySnapshot{ConversationID: update.ConversationID}
	}

	now := time.Now().UTC().Format(time.RFC3339Nano)
	if strings.TrimSpace(update.UserPrompt) != "" {
		snapshot.ShortTerm = append(snapshot.ShortTerm, MemoryEvent{
			Kind:      MemoryEventUser,
			Role:      string(RoleUser),
			Content:   truncateMemoryText(update.UserPrompt, 480),
			Timestamp: now,
		})
	}
	for _, call := range update.ToolCalls {
		snapshot.ShortTerm = append(snapshot.ShortTerm, MemoryEvent{
			Kind:      MemoryEventTool,
			Role:      string(RoleTool),
			ToolName:  call.Name,
			Content:   truncateMemoryText(call.Result, 320),
			Timestamp: now,
			Metadata: map[string]string{
				"status": call.Status,
			},
		})
	}
	if update.Retrieval != nil {
		for _, chunk := range update.Retrieval.Chunks {
			snapshot.ShortTerm = append(snapshot.ShortTerm, MemoryEvent{
				Kind:      MemoryEventEvidence,
				Content:   truncateMemoryText(chunk.Text, 260),
				Source:    chunk.Source,
				Section:   chunk.Section,
				Timestamp: now,
			})
			snapshot.Artifacts = upsertArtifact(snapshot.Artifacts, ArtifactMemoryLink{
				Name:      firstNonEmptyString(chunk.DocTitle, chunk.Source),
				Kind:      "retrieved_document",
				Source:    chunk.Source,
				Reference: chunk.Version,
			})
		}
	}
	for _, artifact := range update.ArtifactLinks {
		snapshot.Artifacts = upsertArtifact(snapshot.Artifacts, artifact)
	}
	if update.Route != nil {
		snapshot.ShortTerm = append(snapshot.ShortTerm, MemoryEvent{
			Kind:      MemoryEventRoute,
			Content:   truncateMemoryText(update.Route.Explanation.SelectionReason, 240),
			Timestamp: now,
			Metadata: map[string]string{
				"model":    update.Route.ModelID,
				"provider": update.Route.Explanation.SelectedProvider,
				"policy":   update.Route.Explanation.PolicyName,
			},
		})
	}
	if strings.TrimSpace(update.ResponseContent) != "" {
		snapshot.ShortTerm = append(snapshot.ShortTerm, MemoryEvent{
			Kind:      MemoryEventAssistant,
			Role:      string(RoleAssistant),
			Content:   truncateMemoryText(update.ResponseContent, 480),
			Timestamp: now,
		})
	}

	snapshot.ShortTerm, snapshot.Summary = m.compact(snapshot.ShortTerm, snapshot.Summary)
	snapshot.Working = buildWorkingMemory(snapshot.Working, update, now)
	snapshot.Summary = buildSummaryMemory(snapshot.Summary, snapshot.ShortTerm, now)
	snapshot.Artifacts = trimArtifacts(snapshot.Artifacts, 8)

	if update.ConversationID == "" || m.redisClient == nil {
		return snapshot, nil
	}
	if err := m.storeJSON(ctx, memoryShortTermPrefix+update.ConversationID, snapshot.ShortTerm); err != nil {
		return nil, err
	}
	if err := m.storeJSON(ctx, memorySummaryPrefix+update.ConversationID, snapshot.Summary); err != nil {
		return nil, err
	}
	if err := m.storeJSON(ctx, memoryWorkingPrefix+update.ConversationID, snapshot.Working); err != nil {
		return nil, err
	}
	if err := m.storeJSON(ctx, memoryArtifactPrefix+update.ConversationID, snapshot.Artifacts); err != nil {
		return nil, err
	}
	return snapshot, nil
}

func (m *MemoryEngine) compact(events []MemoryEvent, summary SessionSummaryMemory) ([]MemoryEvent, SessionSummaryMemory) {
	if len(events) <= m.shortTermLimit {
		return events, summary
	}
	carry := events[:len(events)-m.shortTermLimit]
	events = append([]MemoryEvent(nil), events[len(events)-m.shortTermLimit:]...)

	summaryNotes := make([]string, 0, len(carry))
	for _, event := range carry {
		switch event.Kind {
		case MemoryEventUser:
			summaryNotes = append(summaryNotes, "User asked: "+event.Content)
		case MemoryEventTool:
			summaryNotes = append(summaryNotes, fmt.Sprintf("Tool %s returned: %s", event.ToolName, event.Content))
		case MemoryEventEvidence:
			summaryNotes = append(summaryNotes, fmt.Sprintf("Evidence from %s/%s", event.Source, event.Section))
		case MemoryEventRoute:
			summaryNotes = append(summaryNotes, "Routing note: "+event.Content)
		}
	}
	summary.Summary = strings.TrimSpace(strings.Join(compactText(summary.Summary, summaryNotes, 5), "\n"))
	return events, summary
}

func (m *MemoryEngine) ephemeralSnapshot(conversationID string, history []Message) *SessionMemorySnapshot {
	snapshot := &SessionMemorySnapshot{ConversationID: conversationID}
	if len(history) == 0 {
		return snapshot
	}
	start := len(history) - 4
	if start < 0 {
		start = 0
	}
	for _, msg := range history[start:] {
		snapshot.ShortTerm = append(snapshot.ShortTerm, MemoryEvent{
			Kind:      mapRoleToEventKind(msg.Role),
			Role:      string(msg.Role),
			Content:   truncateMemoryText(msg.Content, 320),
			Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		})
	}
	snapshot.Summary = buildSummaryMemory(SessionSummaryMemory{}, snapshot.ShortTerm, time.Now().UTC().Format(time.RFC3339Nano))
	return snapshot
}

func (m *MemoryEngine) loadJSON(ctx context.Context, key string, target any) error {
	if m == nil || m.redisClient == nil {
		return nil
	}
	value, err := m.redisClient.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return nil
	}
	if err != nil {
		return err
	}
	return json.Unmarshal(value, target)
}

func (m *MemoryEngine) storeJSON(ctx context.Context, key string, value any) error {
	if m == nil || m.redisClient == nil {
		return nil
	}
	payload, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return m.redisClient.Set(ctx, key, payload, memoryTTL).Err()
}

func mapRoleToEventKind(role Role) MemoryEventKind {
	switch role {
	case RoleAssistant:
		return MemoryEventAssistant
	default:
		return MemoryEventUser
	}
}

func buildSummaryMemory(existing SessionSummaryMemory, events []MemoryEvent, now string) SessionSummaryMemory {
	summary := existing
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Kind == MemoryEventUser && summary.UserGoalProgression == "" {
			summary.UserGoalProgression = truncateMemoryText(events[i].Content, 180)
		}
		if events[i].Kind == MemoryEventRoute && summary.ActiveReasoning == "" {
			summary.ActiveReasoning = truncateMemoryText(events[i].Content, 180)
		}
		if events[i].Kind == MemoryEventAssistant && summary.CurrentSubtask == "" {
			summary.CurrentSubtask = truncateMemoryText(events[i].Content, 180)
		}
	}
	if summary.Summary == "" {
		lines := make([]string, 0, len(events))
		for _, event := range events {
			switch event.Kind {
			case MemoryEventUser:
				lines = append(lines, "User: "+event.Content)
			case MemoryEventTool:
				lines = append(lines, fmt.Sprintf("Tool %s: %s", event.ToolName, event.Content))
			case MemoryEventEvidence:
				lines = append(lines, fmt.Sprintf("Evidence: %s (%s)", event.Source, event.Section))
			case MemoryEventAssistant:
				lines = append(lines, "Assistant: "+event.Content)
			}
		}
		summary.Summary = strings.Join(compactText("", lines, 5), "\n")
	}
	summary.ActiveArtifacts = trimStrings(summary.ActiveArtifacts, 6)
	summary.UpdatedAt = now
	return summary
}

func buildWorkingMemory(existing StructuredWorkingMemory, update MemoryUpdate, now string) StructuredWorkingMemory {
	working := existing
	working.SessionMode = update.SessionMode
	working.ForceScope = update.ForceScope
	working.EffectiveModel = update.EffectiveModel
	working.ActiveConstraints = trimStrings(appendUnique(working.ActiveConstraints, nonEmptyStrings(update.AnswerMode)), 6)
	if update.Route != nil {
		working.ActivePolicies = trimStrings(appendUnique(working.ActivePolicies, nonEmptyStrings(update.Route.Explanation.PolicyName, update.Route.Explanation.Strategy)), 6)
	}
	if update.Retrieval != nil {
		for _, chunk := range update.Retrieval.Chunks {
			working.KnowledgeSources = appendUnique(working.KnowledgeSources, nonEmptyStrings(chunk.Source))
		}
	}
	if working.WorkflowState == nil {
		working.WorkflowState = make(map[string]string)
	}
	if update.Route != nil {
		working.WorkflowState["last_route_model"] = update.Route.ModelID
	}
	if update.AnswerMode != "" {
		working.WorkflowState["answer_mode"] = update.AnswerMode
	}
	working.KnowledgeSources = trimStrings(working.KnowledgeSources, 6)
	working.UpdatedAt = now
	return working
}

func upsertArtifact(artifacts []ArtifactMemoryLink, candidate ArtifactMemoryLink) []ArtifactMemoryLink {
	if candidate.Source == "" && candidate.Name == "" {
		return artifacts
	}
	for i, artifact := range artifacts {
		if artifact.Source == candidate.Source && artifact.Reference == candidate.Reference {
			artifacts[i] = candidate
			return artifacts
		}
	}
	return append(artifacts, candidate)
}

func trimArtifacts(artifacts []ArtifactMemoryLink, limit int) []ArtifactMemoryLink {
	if len(artifacts) <= limit {
		return artifacts
	}
	return append([]ArtifactMemoryLink(nil), artifacts[len(artifacts)-limit:]...)
}

func compactText(existing string, additions []string, limit int) []string {
	lines := make([]string, 0, len(additions)+1)
	if strings.TrimSpace(existing) != "" {
		lines = append(lines, strings.Split(existing, "\n")...)
	}
	for _, addition := range additions {
		if strings.TrimSpace(addition) != "" {
			lines = append(lines, addition)
		}
	}
	if len(lines) > limit {
		lines = append([]string(nil), lines[len(lines)-limit:]...)
	}
	return lines
}

func appendUnique(current []string, next []string) []string {
	seen := make(map[string]struct{}, len(current))
	for _, value := range current {
		seen[value] = struct{}{}
	}
	for _, value := range next {
		if _, ok := seen[value]; ok || strings.TrimSpace(value) == "" {
			continue
		}
		current = append(current, value)
		seen[value] = struct{}{}
	}
	return current
}

func nonEmptyStrings(values ...string) []string {
	var filtered []string
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			filtered = append(filtered, value)
		}
	}
	return filtered
}

func trimStrings(values []string, limit int) []string {
	if len(values) == 0 {
		return nil
	}
	values = append([]string(nil), values...)
	sort.Strings(values)
	if len(values) > limit {
		values = values[:limit]
	}
	return values
}

func truncateMemoryText(value string, maxLen int) string {
	value = strings.TrimSpace(value)
	if maxLen <= 0 || len(value) <= maxLen {
		return value
	}
	return strings.TrimSpace(value[:maxLen-3]) + "..."
}
