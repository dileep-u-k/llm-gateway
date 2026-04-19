// In file: internal/llm/openai_image_client.go
package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
)

// OpenAIImageClient is the client for DALL-E models.
type OpenAIImageClient struct {
	apiKey string
	client *http.Client
}

// Statically verify that OpenAIImageClient implements the ImageClient interface.
var _ ImageClient = (*OpenAIImageClient)(nil)
var _ ImageEditingClient = (*OpenAIImageClient)(nil)

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
		URL     string `json:"url"`
		B64JSON string `json:"b64_json"`
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

	return c.resolveImageLocation(apiResp.Data[0].URL, apiResp.Data[0].B64JSON)
}

func (c *OpenAIImageClient) EditImage(ctx context.Context, req ImageEditRequest) (string, error) {
	imageBytes, imageName, err := loadImageBytes(req.ImagePath)
	if err != nil {
		return "", err
	}

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	if err := writer.WriteField("model", req.Model); err != nil {
		return "", fmt.Errorf("failed to write edit model field: %w", err)
	}
	if err := writer.WriteField("prompt", req.Prompt); err != nil {
		return "", fmt.Errorf("failed to write edit prompt field: %w", err)
	}
	if err := writer.WriteField("size", "1024x1024"); err != nil {
		return "", fmt.Errorf("failed to write image size field: %w", err)
	}

	imagePart, err := writer.CreateFormFile("image", imageName)
	if err != nil {
		return "", fmt.Errorf("failed to create image form file: %w", err)
	}
	if _, err := imagePart.Write(imageBytes); err != nil {
		return "", fmt.Errorf("failed to write image bytes: %w", err)
	}

	if req.MaskPath != "" {
		maskBytes, maskName, err := loadImageBytes(req.MaskPath)
		if err != nil {
			return "", err
		}
		maskPart, err := writer.CreateFormFile("mask", maskName)
		if err != nil {
			return "", fmt.Errorf("failed to create mask form file: %w", err)
		}
		if _, err := maskPart.Write(maskBytes); err != nil {
			return "", fmt.Errorf("failed to write mask bytes: %w", err)
		}
	}

	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("failed to finalize edit payload: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/images/edits", body)
	if err != nil {
		return "", fmt.Errorf("failed to create OpenAI image edit request: %w", err)
	}
	httpReq.Header.Set("Content-Type", writer.FormDataContentType())
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("OpenAI image edit API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		payload, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("OpenAI image edit API returned non-200 status: %s, body: %s", resp.Status, string(payload))
	}

	var apiResp openAIImageResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return "", fmt.Errorf("failed to decode OpenAI image edit response: %w", err)
	}
	if len(apiResp.Data) == 0 {
		return "", fmt.Errorf("OpenAI image edit API returned no image data")
	}

	return c.resolveImageLocation(apiResp.Data[0].URL, apiResp.Data[0].B64JSON)
}

func (c *OpenAIImageClient) resolveImageLocation(url, b64 string) (string, error) {
	if url != "" {
		return url, nil
	}
	if b64 == "" {
		return "", fmt.Errorf("OpenAI image response contained neither url nor b64_json")
	}
	bytes, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return "", fmt.Errorf("failed to decode image b64 payload: %w", err)
	}
	file, err := os.CreateTemp("", "llm-gateway-image-*.png")
	if err != nil {
		return "", fmt.Errorf("failed to persist generated image: %w", err)
	}
	defer file.Close()
	if _, err := file.Write(bytes); err != nil {
		return "", fmt.Errorf("failed to write generated image: %w", err)
	}
	return file.Name(), nil
}

func loadImageBytes(path string) ([]byte, string, error) {
	if path == "" {
		return nil, "", fmt.Errorf("image path is required for edits")
	}
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, "", fmt.Errorf("failed to read image %q: %w", path, err)
	}
	return bytes, filepath.Base(path), nil
}
