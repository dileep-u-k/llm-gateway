// In file: internal/tools/weather_tool.go
package tools

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// --- Weather Tool Implementation ---

// WeatherTool is a concrete implementation of a tool that fetches weather data.
// It holds its own configured HTTP client for making robust external API calls.
type WeatherTool struct {
	httpClient *http.Client
}

// Statically verify that WeatherTool implements the ToolExecutor interface.
// This is a best practice that ensures your tool meets the required contract.
var _ ToolExecutor = (*WeatherTool)(nil)

// NewWeatherTool creates a new instance of the WeatherTool.
// It initializes a dedicated HTTP client with a timeout, which is crucial
// for preventing hung requests to external services in a production environment.
func NewWeatherTool() *WeatherTool {
	return &WeatherTool{
		httpClient: &http.Client{
			Timeout: 15 * time.Second, // Set a reasonable timeout.
		},
	}
}

// Definition describes the tool to the LLM using our type-safe structures.
// This replaces the brittle `map[string]interface{}` with the clear and robust
// `JSONSchema` struct, preventing schema errors and improving maintainability.
func (wt *WeatherTool) Definition() Tool {
	// Use the NewFunctionTool helper for clean and consistent tool creation.
	return NewFunctionTool(
		"getCurrentWeather",
		"Get the current weather for a specific location",
		JSONSchema{
			Type: "object",
			Properties: map[string]*JSONSchema{
				"location": {
					Type:        "string",
					Description: "The city and state, e.g., San Francisco, CA or Kharagpur, India",
				},
			},
			Required: []string{"location"},
		},
	)
}

// Execute runs the tool's actual logic. It takes the arguments provided by the LLM,
// calls an external API, and returns the result as a simple string.
func (wt *WeatherTool) Execute(arguments string) (string, error) {
	// Unmarshal the JSON arguments string from the LLM into a Go struct for type-safe access.
	var args struct {
		Location string `json:"location"`
	}
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return "", fmt.Errorf("invalid arguments for weather tool: %w", err)
	}
	if args.Location == "" {
		return "Error: Location cannot be empty.", nil
	}

	// Use wttr.in, a simple text-based weather API perfect for LLM consumption.
	// We use the configured httpClient with a timeout for this call.
	url := fmt.Sprintf("https://wttr.in/%s?format=3", strings.ReplaceAll(args.Location, " ", "+"))
	
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create weather API request: %w", err)
	}
	// Some services might block default Go HTTP clients, so setting a common User-Agent is a good practice.
	req.Header.Set("User-Agent", "LLM-Gateway-Agent/1.0")

	resp, err := wt.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to call weather API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("weather API returned non-200 status: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read weather API response: %w", err)
	}
	
	responseString := string(body)
	if strings.Contains(responseString, "Unknown location") {
		return fmt.Sprintf("I couldn't find the weather for '%s'. Please try another location.", args.Location), nil
	}

	return responseString, nil
}