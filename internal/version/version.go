// In file: internal/version/version.go

// Package version centralizes the versioning for different logical components of the gateway.
//
// This is a powerful caching strategy. By including these version strings in our
// cache keys, we can automatically invalidate old, incorrect cached entries whenever
// a piece of underlying logic or data changes. For example, if we fix a bug in a
// tool and update ToolsVersion from "v1.0" to "v1.1", all cache keys containing
// the old version string will no longer be matched, forcing the gateway to
// re-generate fresh, correct responses.
package version

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// ComponentVersions holds the version strings for different logical parts of the application.
// Manually increment a version number here before you deploy a change to that component.
var ComponentVersions = struct {
	// ToolsVersion should be updated whenever you fix a bug or change the logic
	// in any of your tool files (e.g., weather_tool.go, calculator_tool.go).
	Tools string

	// RAGDataVersion should be updated whenever you add, remove, or edit any of the
	// documents in your /data folder after running the ingestor. This signals
	// that the gateway's knowledge base has changed.
	RAGData string

	// PromptLogicVersion should be updated whenever you change the core prompt
	// templates or the main logic that constructs prompts for the LLM.
	PromptLogic string
}{
	Tools:       "v1.0",
	RAGData:     "v1.0",
	PromptLogic: "v1.0",
}

// GenerateVersionedCacheKey creates a consistent, version-aware key for caching LLM responses.
//
// It combines a prefix, a hash of the user's prompt, and the current versions of all
// logical components. This ensures that if the prompt or any underlying logic changes,
// a new cache key is generated, effectively invalidating the old cache entry.
//
// Example output: "llmcache:a1b2c3d4...:tv1.0_rv1.0_pv1.0"
func GenerateVersionedCacheKey(prefix, prompt string) string {
	// 1. Hash the prompt to create a fixed-length, unique identifier for the user's input.
	hasher := sha256.New()
	hasher.Write([]byte(prompt))
	promptHash := hex.EncodeToString(hasher.Sum(nil))

	// 2. Create a compact string representing the current state of all components.
	versionString := fmt.Sprintf("tv%s_rv%s_pv%s",
		ComponentVersions.Tools,
		ComponentVersions.RAGData,
		ComponentVersions.PromptLogic,
	)

	// 3. Combine all parts into the final cache key.
	return fmt.Sprintf("%s:%s:%s", prefix, promptHash, versionString)
}