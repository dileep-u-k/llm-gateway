// In file: internal/llm/constants.go
package llm

import "time"

// This file centralizes constants shared across multiple clients and services
// in the llm package to avoid redeclaration errors.
const (
    defaultTimeout      = 120 * time.Second
    maxRetries          = 3
    initialRetryDelay   = 2 * time.Second
)