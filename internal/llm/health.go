package llm

import (
	"strings"
	"time"
)

type HealthStatus string

type Capability string

const (
	HealthStatusOnline   HealthStatus = "online"
	HealthStatusDegraded HealthStatus = "degraded"
	HealthStatusOffline  HealthStatus = "offline"

	CapabilityTextGeneration     Capability = "text_generation"
	CapabilityEmbeddings         Capability = "embeddings"
	CapabilityOCR                Capability = "ocr"
	CapabilityImageUnderstanding Capability = "image_understanding"
	CapabilityImageGeneration    Capability = "image_generation"
	CapabilityImageEditing       Capability = "image_editing"
	CapabilityTranscription      Capability = "transcription"
	CapabilityTTS                Capability = "tts"
	CapabilityVideoUnderstanding Capability = "video_understanding"
	CapabilityVideoGeneration    Capability = "video_generation_hook"
)

type ProviderHealth struct {
	Provider            string       `json:"provider"`
	Status              HealthStatus `json:"status"`
	AccessAllowed       bool         `json:"access_allowed"`
	AvgLatencyMS        int64        `json:"avg_latency_ms"`
	LastHealthCheck     time.Time    `json:"last_health_check"`
	ConsecutiveFailures int64        `json:"consecutive_failures"`
	LastError           string       `json:"last_error,omitempty"`
}

type CapabilityHealth struct {
	Provider            string       `json:"provider"`
	ModelID             string       `json:"model_id"`
	Capability          Capability   `json:"capability"`
	Status              HealthStatus `json:"status"`
	AccessAllowed       bool         `json:"access_allowed"`
	Supported           bool         `json:"supported"`
	AvgLatencyMS        int64        `json:"avg_latency_ms"`
	ErrorRate           float64      `json:"error_rate"`
	LastHealthCheck     time.Time    `json:"last_health_check"`
	ConsecutiveFailures int64        `json:"consecutive_failures"`
	LastError           string       `json:"last_error,omitempty"`
}

type HealthProbeResult struct {
	Status        HealthStatus
	AccessAllowed bool
	Latency       time.Duration
	Err           error
}

func ProviderForModel(modelID string) string {
	switch {
	case strings.HasPrefix(modelID, "gpt"), strings.HasPrefix(modelID, "dall-e"), strings.HasPrefix(modelID, "text-embedding"), strings.HasPrefix(modelID, "tts-"):
		return "openai"
	case strings.HasPrefix(modelID, "claude"):
		return "anthropic"
	case strings.HasPrefix(modelID, "gemini"), strings.HasPrefix(modelID, "imagen"):
		return "google"
	case strings.HasPrefix(modelID, "mistral"):
		return "mistral"
	default:
		return "unknown"
	}
}

func CapabilityForModel(modelID string) Capability {
	switch {
	case strings.HasPrefix(modelID, "dall-e"), strings.HasPrefix(modelID, "imagen"), strings.HasPrefix(modelID, "gpt-image"):
		return CapabilityImageGeneration
	case strings.Contains(modelID, "tts"), strings.HasPrefix(modelID, "tts-"):
		return CapabilityTTS
	case strings.Contains(modelID, "transcribe"):
		return CapabilityTranscription
	default:
		return CapabilityTextGeneration
	}
}

func BuildCapabilityHealthID(provider, modelID string, capability Capability) string {
	return provider + ":" + modelID + ":" + string(capability)
}

func normalizeHealthStatus(status string) HealthStatus {
	switch HealthStatus(strings.ToLower(status)) {
	case HealthStatusOnline:
		return HealthStatusOnline
	case HealthStatusDegraded:
		return HealthStatusDegraded
	case HealthStatusOffline:
		return HealthStatusOffline
	default:
		return HealthStatusOnline
	}
}

func combineHealthStatuses(statuses ...HealthStatus) HealthStatus {
	seenDegraded := false
	seenOnline := false
	for _, status := range statuses {
		switch status {
		case HealthStatusOffline:
			continue
		case HealthStatusDegraded:
			seenDegraded = true
		case HealthStatusOnline:
			seenOnline = true
		}
	}
	if seenOnline && !seenDegraded {
		return HealthStatusOnline
	}
	if seenOnline || seenDegraded {
		return HealthStatusDegraded
	}
	return HealthStatusOffline
}

func classifyProbeError(err error) (HealthStatus, bool) {
	if err == nil {
		return HealthStatusOnline, true
	}
	message := strings.ToLower(err.Error())
	switch {
	case strings.Contains(message, "401"), strings.Contains(message, "403"), strings.Contains(message, "forbidden"), strings.Contains(message, "unauthorized"), strings.Contains(message, "permission"), strings.Contains(message, "model_not_found"), strings.Contains(message, "not found"):
		return HealthStatusOffline, false
	case strings.Contains(message, "429"), strings.Contains(message, "rate limit"), strings.Contains(message, "timeout"), strings.Contains(message, "deadline exceeded"), strings.Contains(message, "temporarily unavailable"), strings.Contains(message, "overloaded"):
		return HealthStatusDegraded, true
	default:
		return HealthStatusDegraded, true
	}
}

func CombineHealthStatuses(statuses ...HealthStatus) HealthStatus {
	return combineHealthStatuses(statuses...)
}

func ClassifyProbeError(err error) (HealthStatus, bool) {
	return classifyProbeError(err)
}
