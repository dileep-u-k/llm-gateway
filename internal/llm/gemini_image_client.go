// In file: internal/llm/gemini_image_client.go
package llm

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"

	// --- THIS IMPORT IS REQUIRED ---
	"strings"
	"time"

	"cloud.google.com/go/storage"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// GeminiImageClient is a concrete implementation of the ImageClient interface for Google's image models.
type GeminiImageClient struct {
	client     *genai.Client
	storage    *storage.Client
	bucketName string
}

// Statically verify that GeminiImageClient implements the ImageClient interface.
var _ ImageClient = (*GeminiImageClient)(nil)

// NewGeminiImageClient creates a new, configured client for Google's generative AI services.
func NewGeminiImageClient(apiKey string, gcsBucket string) (*GeminiImageClient, error) {
	if apiKey == "" {
		return nil, errors.New("gemini API key cannot be empty for image client")
	}
	if gcsBucket == "" {
		return nil, errors.New("gcs bucket name cannot be empty for image client")
	}
	ctx := context.Background()

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	storageClient, err := storage.NewClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCS client: %w", err)
	}

	return &GeminiImageClient{
		client:     client,
		storage:    storageClient,
		bucketName: gcsBucket,
	}, nil
}

// Provider returns the client's identifier.
func (c *GeminiImageClient) Provider() string {
	return "google"
}

// GenerateImage sends a prompt to a Gemini/Imagen model and returns a public URL to the image.
func (c *GeminiImageClient) GenerateImage(ctx context.Context, prompt string, model string) (string, error) {
	imgModel := c.client.GenerativeModel(model)
	resp, err := imgModel.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("gemini image generation failed: %w", err)
	}

	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return "", errors.New("no image content returned from Gemini")
	}

	for _, part := range resp.Candidates[0].Content.Parts {
		if img, ok := part.(genai.Blob); ok && strings.HasPrefix(img.MIMEType, "image/") {
			data, err := base64.StdEncoding.DecodeString(string(img.Data))
			if err != nil {
				return "", fmt.Errorf("failed to decode base64 image: %w", err)
			}
			fileName := fmt.Sprintf("gemini-images/%d.png", time.Now().UnixNano())
			return c.uploadToGCS(ctx, fileName, data)
		}
	}

	return "", errors.New("no image blob found in Gemini response")
}

// uploadToGCS saves image bytes to GCS and returns the public URL.
func (c *GeminiImageClient) uploadToGCS(ctx context.Context, objectName string, data []byte) (string, error) {
	w := c.storage.Bucket(c.bucketName).Object(objectName).NewWriter(ctx)
	w.ContentType = "image/png"
	if _, err := w.Write(data); err != nil {
		return "", fmt.Errorf("failed to write image to GCS: %w", err)
	}
	if err := w.Close(); err != nil {
		return "", fmt.Errorf("failed to close GCS writer: %w", err)
	}
	return fmt.Sprintf("https://storage.googleapis.com/%s/%s", c.bucketName, objectName), nil
}
