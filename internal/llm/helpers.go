// In file: internal/llm/helpers.go

// Package llm contains the core logic for interacting with Large Language Models,
// including client interfaces, routing, profiling, and intent analysis.
package llm

import (
	"crypto/sha256" // <-- Make sure this is imported
	"encoding/binary"
	"encoding/hex" // <-- Make sure this is imported
	"math"
)

// This file contains shared data structures and stateless utility functions
// used across various parts of the llm package.

// Vector is a generic struct representing a text embedding.
// It is used by the data ingestor when creating vectors and by the RAG (retrieval)
// system when querying for context.
type Vector struct {
	// ID is the unique identifier for the vector, often a hash of its content.
	ID string `json:"id"`
	// Values is the slice of floating-point numbers that represents the text in a high-dimensional space.
	Values []float32 `json:"values"`
	// Metadata holds supplementary, filterable information about the vector's source text.
	Metadata map[string]interface{} `json:"metadata"`
}

// VectorToBytes is a crucial helper function that converts a float32 slice (an embedding)
// into a byte slice. This specific binary format is required by Redis when storing
// and querying vectors in a vector search index.
func VectorToBytes(vector []float32) []byte {
	// Each float32 value requires 4 bytes for its representation.
	// We pre-allocate a byte slice of the exact required size for efficiency.
	byteSlice := make([]byte, 4*len(vector))

	for i, f := range vector {
		// math.Float32bits converts the float into its IEEE 754 binary representation as a uint32.
		bits := math.Float32bits(f)

		// binary.LittleEndian.PutUint32 writes those 4 bytes into the slice at the correct offset.
		// LittleEndian is the standard byte order for Redis vector storage.
		binary.LittleEndian.PutUint32(byteSlice[i*4:], bits)
	}
	return byteSlice
}


// GenerateCacheKey creates a stable, fixed-length SHA256 hash of a string.
// It's used for creating consistent cache keys for prompts and intents.
func GenerateCacheKey(prompt string) string {
	hasher := sha256.New()
	hasher.Write([]byte(prompt))
	return hex.EncodeToString(hasher.Sum(nil))
}
