package platform

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"strings"
)

type Principal struct {
	ID          string   `json:"id,omitempty"`
	DisplayName string   `json:"display_name,omitempty"`
	Role        string   `json:"role,omitempty"`
	Scopes      []string `json:"scopes,omitempty"`
	Mode        string   `json:"mode,omitempty"`
}

type Authenticator struct {
	mode        string
	tokenLookup map[string]Principal
}

func NewAuthenticatorFromEnv() *Authenticator {
	tokenLookup := make(map[string]Principal)

	if raw := strings.TrimSpace(os.Getenv("PLATFORM_TOKENS_JSON")); raw != "" {
		var principals map[string]Principal
		if json.Unmarshal([]byte(raw), &principals) == nil {
			for token, principal := range principals {
				if principal.Role == "" {
					principal.Role = "user"
				}
				principal.Mode = "configured_token"
				tokenLookup[token] = principal
			}
		}
	}

	registerSimpleToken := func(envKey, role string) {
		if token := strings.TrimSpace(os.Getenv(envKey)); token != "" {
			tokenLookup[token] = Principal{
				ID:          role,
				DisplayName: strings.Title(strings.ReplaceAll(role, "_", " ")),
				Role:        role,
				Mode:        "configured_token",
			}
		}
	}
	registerSimpleToken("PLATFORM_ADMIN_TOKEN", "admin")
	registerSimpleToken("PLATFORM_OPERATOR_TOKEN", "operator")
	registerSimpleToken("PLATFORM_USER_TOKEN", "user")

	mode := "open"
	if len(tokenLookup) > 0 {
		mode = "token"
	}
	return &Authenticator{mode: mode, tokenLookup: tokenLookup}
}

func (a *Authenticator) Resolve(r *http.Request) (*Principal, error) {
	if a == nil || a.mode == "open" {
		return &Principal{
			ID:          "anonymous",
			DisplayName: "Open Local Access",
			Role:        "admin",
			Scopes:      []string{"user", "admin"},
			Mode:        "open",
		}, nil
	}
	token := bearerToken(r)
	if token == "" {
		token = strings.TrimSpace(r.Header.Get("X-API-Key"))
	}
	if principal, ok := a.tokenLookup[token]; ok {
		copyPrincipal := principal
		if copyPrincipal.ID == "" {
			copyPrincipal.ID = hashToken(token)
		}
		return &copyPrincipal, nil
	}
	return nil, ErrUnauthorized
}

func (a *Authenticator) Mode() string {
	if a == nil {
		return "open"
	}
	return a.mode
}

func (p *Principal) CanAdmin() bool {
	if p == nil {
		return false
	}
	return p.Role == "admin" || p.Role == "operator"
}

func (p *Principal) CanStrictForce() bool {
	return p != nil && (p.Role == "admin" || p.Role == "operator")
}

func bearerToken(r *http.Request) string {
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	if auth == "" {
		return ""
	}
	parts := strings.SplitN(auth, " ", 2)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "bearer") {
		return ""
	}
	return strings.TrimSpace(parts[1])
}

func hashToken(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:8])
}
