package platform

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

const auditEventKey = "platform:audit:events"

type AuditEvent struct {
	ID          string            `json:"id,omitempty"`
	Timestamp   string            `json:"timestamp,omitempty"`
	ActorID     string            `json:"actor_id,omitempty"`
	ActorRole   string            `json:"actor_role,omitempty"`
	Action      string            `json:"action,omitempty"`
	Resource    string            `json:"resource,omitempty"`
	TenantID    string            `json:"tenant_id,omitempty"`
	WorkspaceID string            `json:"workspace_id,omitempty"`
	Status      string            `json:"status,omitempty"`
	Summary     string            `json:"summary,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

type AuditLogger struct {
	rdb   *redis.Client
	mu    sync.Mutex
	local []AuditEvent
}

func NewAuditLogger(rdb *redis.Client) *AuditLogger {
	return &AuditLogger{rdb: rdb}
}

func (l *AuditLogger) Record(ctx context.Context, event AuditEvent) string {
	if l == nil {
		return ""
	}
	if event.ID == "" {
		event.ID = time.Now().UTC().Format("20060102150405.000000000")
	}
	if event.Timestamp == "" {
		event.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	}
	event.Metadata = RedactStringMap(event.Metadata)
	payload, err := json.Marshal(event)
	if err == nil && l.rdb != nil {
		pipe := l.rdb.TxPipeline()
		pipe.LPush(ctx, auditEventKey, payload)
		pipe.LTrim(ctx, auditEventKey, 0, 199)
		pipe.Expire(ctx, auditEventKey, 7*24*time.Hour)
		_, _ = pipe.Exec(ctx)
	}
	l.mu.Lock()
	l.local = append(l.local, event)
	if len(l.local) > 200 {
		l.local = append([]AuditEvent(nil), l.local[len(l.local)-200:]...)
	}
	l.mu.Unlock()
	return event.ID
}

func (l *AuditLogger) List(ctx context.Context, limit int64) ([]AuditEvent, error) {
	if limit <= 0 {
		limit = 50
	}
	if l != nil && l.rdb != nil {
		values, err := l.rdb.LRange(ctx, auditEventKey, 0, limit-1).Result()
		if err == nil {
			events := make([]AuditEvent, 0, len(values))
			for _, value := range values {
				var event AuditEvent
				if json.Unmarshal([]byte(value), &event) == nil {
					events = append(events, event)
				}
			}
			if len(events) > 0 {
				return events, nil
			}
		}
	}
	if l == nil {
		return nil, nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.local) == 0 {
		return nil, nil
	}
	if int(limit) > len(l.local) {
		limit = int64(len(l.local))
	}
	start := len(l.local) - int(limit)
	return append([]AuditEvent(nil), l.local[start:]...), nil
}

func RedactStringMap(metadata map[string]string) map[string]string {
	if len(metadata) == 0 {
		return nil
	}
	out := make(map[string]string, len(metadata))
	for key, value := range metadata {
		out[key] = RedactValue(key, value)
	}
	return out
}

func RedactValue(key, value string) string {
	lowerKey := strings.ToLower(key)
	lowerValue := strings.ToLower(value)
	switch {
	case strings.Contains(lowerKey, "token"),
		strings.Contains(lowerKey, "secret"),
		strings.Contains(lowerKey, "authorization"),
		strings.Contains(lowerKey, "api_key"),
		strings.Contains(lowerValue, "sk-"),
		strings.Contains(lowerValue, "bearer "):
		return "[REDACTED]"
	default:
		return value
	}
}
