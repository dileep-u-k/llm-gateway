// In file: internal/llm/openai_image_client.go
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// OpenAIImageClient is the client for DALL-E models.
type OpenAIImageClient struct {
	apiKey string
	client *http.Client
}

// Statically verify that OpenAIImageClient implements the ImageClient interface.
var _ ImageClient = (*OpenAIImageClient)(nil)

// openAIImageRequest defines the request body for the DALL-E API.
type openAIImageRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	N      int    `json:"n"`
	Size   string `json:"size"`
}

// openAIImageResponse defines the successful response body.
type openAIImageResponse struct {
	Data []struct {
		URL string `json:"url"`
	} `json:"data"`
}

// NewOpenAIImageClient creates a new client for interacting with the DALL-E API.
func NewOpenAIImageClient(apiKey string) (*OpenAIImageClient, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("OpenAI API key cannot be empty")
	}
	return &OpenAIImageClient{
		apiKey: apiKey,
		client: &http.Client{},
	}, nil
}

// --- ADD THIS METHOD ---
// Provider returns the client's identifier.
func (c *OpenAIImageClient) Provider() string {
	return "openai"
}

// --- END OF ADDITION ---

// GenerateImage implements the ImageClient interface for DALL-E.
func (c *OpenAIImageClient) GenerateImage(ctx context.Context, prompt, model string) (string, error) {
	reqBody := openAIImageRequest{
		Model:  model,
		Prompt: prompt,
		N:      1,
		Size:   "1024x1024",
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal OpenAI image request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/images/generations", bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", fmt.Errorf("failed to create OpenAI image request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("OpenAI image API request failed: %w", err)
	}
	defer resp.Body.Close()

	// ... (inside the GenerateImage function)
	if resp.StatusCode != http.StatusOK {
		// --- THIS IS THE FIX ---
		// We now include the response body in the error message for better debugging.
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("OpenAI image API returned non-200 status: %s, body: %s", resp.Status, string(body))
		// --- END OF FIX ---
	}

	var apiResp openAIImageResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return "", fmt.Errorf("failed to decode OpenAI image response: %w", err)
	}

	if len(apiResp.Data) == 0 {
		return "", fmt.Errorf("OpenAI image API returned no image data")
	}

	return apiResp.Data[0].URL, nil
}
