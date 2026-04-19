package llm

import "context"

type SpeechSynthesisRequest struct {
	Model        string
	Input        string
	Voice        string
	Format       string
	Instructions string
	Speed        float64
}

type SpeechSynthesisResult struct {
	Audio     []byte
	MimeType  string
	Format    string
	Voice     string
	Model     string
	InputText string
}

type SpeechClient interface {
	SynthesizeSpeech(ctx context.Context, req SpeechSynthesisRequest) (*SpeechSynthesisResult, error)
}
