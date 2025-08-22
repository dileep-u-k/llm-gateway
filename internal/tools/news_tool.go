// In file: internal/tools/news_tool.go
package tools

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// --- News Tool Implementation ---

const newsAPIURL = "https://newsapi.org/v2/top-headlines"

// NewsTool is a concrete implementation of a tool that fetches the latest news headlines.
// To use this tool, you need a free API key from https://newsapi.org
type NewsTool struct {
	apiKey     string
	httpClient *http.Client
}

// Statically verify that NewsTool implements the ToolExecutor interface.
var _ ToolExecutor = (*NewsTool)(nil)

// NewNewsTool creates a new instance of the NewsTool.
// It requires an API key and initializes a dedicated HTTP client with a timeout
// for robust communication with the external NewsAPI service.
func NewNewsTool(apiKey string) (*NewsTool, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("NewsAPI key cannot be empty")
	}
	return &NewsTool{
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: 20 * time.Second, // Set a reasonable timeout.
		},
	}, nil
}

// Definition describes the tool to the LLM using our type-safe structures.
// The descriptions for the parameters are crucial for the LLM to understand
// how to use the tool effectively.
func (nt *NewsTool) Definition() Tool {
	return NewFunctionTool(
		"getNewsHeadlines",
		"Fetches the latest news headlines about a specific topic, category, or from a particular country.",
		JSONSchema{
			Type: "object",
			Properties: map[string]*JSONSchema{
				"query": {
					Type:        "string",
					Description: "The topic or keyword to search for in the news, e.g., 'artificial intelligence' or 'latest space missions'.",
				},
				"category": {
					Type: "string",
					Description: `The category of news. Must be one of: business, entertainment, general, health, science, sports, technology.`,
				},
				"country": {
					Type:        "string",
					Description: "The 2-letter ISO 3166-1 code of the country to get headlines from, e.g., 'us' for USA, 'in' for India, or 'gb' for Great Britain.",
				},
			},
			// The LLM can provide any combination of these optional parameters.
			Required: []string{},
		},
	)
}

// Execute runs the tool's logic. It builds a request to the NewsAPI,
// parses the response, and formats it into a clean summary for the LLM.
func (nt *NewsTool) Execute(arguments string) (string, error) {
	// 1. Unmarshal the LLM's arguments into a structured Go type.
	var args struct {
		Query    string `json:"query"`
		Category string `json:"category"`
		Country  string `json:"country"`
	}
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return "", fmt.Errorf("invalid arguments for news tool: %w", err)
	}

	// 2. Build the request URL with the provided parameters.
	base, _ := url.Parse(newsAPIURL)
	params := url.Values{}
	if args.Query != "" {
		params.Add("q", args.Query)
	}
	if args.Category != "" {
		params.Add("category", args.Category)
	}
	if args.Country != "" {
		params.Add("country", args.Country)
	}
	// Limit the number of results to keep the response concise.
	params.Add("pageSize", "5")
	base.RawQuery = params.Encode()

	// 3. Make the external API call using the configured HTTP client.
	req, err := http.NewRequest("GET", base.String(), nil)
	if err != nil {
		return "", fmt.Errorf("failed to create news API request: %w", err)
	}
	req.Header.Set("X-Api-Key", nt.apiKey)
	req.Header.Set("User-Agent", "LLM-Gateway-Agent/1.0")

	resp, err := nt.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to call news API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Sprintf("Error: News API returned a non-200 status code: %d. Please check the parameters or API key.", resp.StatusCode), nil
	}

	// 4. Parse the JSON response from the NewsAPI.
	var apiResp struct {
		Status       string `json:"status"`
		TotalResults int    `json:"totalResults"`
		Articles     []struct {
			Title       string `json:"title"`
			Description string `json:"description"`
			Source      struct {
				Name string `json:"name"`
			} `json:"source"`
		} `json:"articles"`
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read news API response: %w", err)
	}
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", fmt.Errorf("failed to parse news API JSON response: %w", err)
	}

	// 5. Format the result into a clean, LLM-friendly string.
	if apiResp.TotalResults == 0 {
		return "No news articles found for the given criteria.", nil
	}

	var resultBuilder strings.Builder
	resultBuilder.WriteString(fmt.Sprintf("Here are the top %d headlines:\n", len(apiResp.Articles)))
	for i, article := range apiResp.Articles {
		resultBuilder.WriteString(fmt.Sprintf("%d. %s (Source: %s)\n", i+1, article.Title, article.Source.Name))
	}

	return resultBuilder.String(), nil
}