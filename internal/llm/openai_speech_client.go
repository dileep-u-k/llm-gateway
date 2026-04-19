package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type OpenAISpeechClient struct {
	apiKey string
	client *http.Client
}

var _ SpeechClient = (*OpenAISpeechClient)(nil)

type openAISpeechRequest struct {
	Model        string  `json:"model"`
	Input        string  `json:"input"`
	Voice        string  `json:"voice"`
	ResponseFmt  string  `json:"response_format,omitempty"`
	Instructions string  `json:"instructions,omitempty"`
	Speed        float64 `json:"speed,omitempty"`
}

func NewOpenAISpeechClient(apiKey string) (*OpenAISpeechClient, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("OpenAI API key cannot be empty")
	}
	return &OpenAISpeechClient{
		apiKey: apiKey,
		client: &http.Client{},
	}, nil
}

func (c *OpenAISpeechClient) SynthesizeSpeech(ctx context.Context, req SpeechSynthesisRequest) (*SpeechSynthesisResult, error) {
	body, err := json.Marshal(openAISpeechRequest{
		Model:        req.Model,
		Input:        req.Input,
		Voice:        firstNonEmpty(req.Voice, "alloy"),
		ResponseFmt:  firstNonEmpty(req.Format, "mp3"),
		Instructions: req.Instructions,
		Speed:        req.Speed,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal speech request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/audio/speech", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create speech request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("OpenAI speech API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		payload, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OpenAI speech API returned non-200 status: %s, body: %s", resp.Status, string(payload))
	}

	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read speech response body: %w", err)
	}

	return &SpeechSynthesisResult{
		Audio:     audio,
		MimeType:  mimeTypeForSpeechFormat(firstNonEmpty(req.Format, "mp3")),
		Format:    firstNonEmpty(req.Format, "mp3"),
		Voice:     firstNonEmpty(req.Voice, "alloy"),
		Model:     req.Model,
		InputText: req.Input,
	}, nil
}

func mimeTypeForSpeechFormat(format string) string {
	switch format {
	case "wav":
		return "audio/wav"
	case "opus":
		return "audio/ogg"
	case "aac":
		return "audio/aac"
	case "flac":
		return "audio/flac"
	default:
		return "audio/mpeg"
	}
}
