// In file: internal/llm/image_client.go
package llm

import "context"

// ImageClient defines the standard interface for any image generation provider.
type ImageClient interface {
	// GenerateImage takes a prompt and model, and returns a public URL to the generated image.
	GenerateImage(ctx context.Context, prompt, model string) (string, error)
}

type ImageEditRequest struct {
	Model     string
	Prompt    string
	ImagePath string
	MaskPath  string
}

// ImageEditingClient is implemented by providers that support native image editing.
type ImageEditingClient interface {
	EditImage(ctx context.Context, req ImageEditRequest) (string, error)
}
