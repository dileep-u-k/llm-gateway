package platform

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type SignedArtifactAccessLayer struct {
	secret []byte
	ttl    time.Duration
}

func NewSignedArtifactAccessLayerFromEnv(defaultTTL string) *SignedArtifactAccessLayer {
	secret := strings.TrimSpace(os.Getenv("PLATFORM_SIGNING_SECRET"))
	if secret == "" {
		secret = "llm-gateway-phase6-local-signing-secret"
	}
	rawTTL := strings.TrimSpace(os.Getenv("PLATFORM_SIGNED_URL_TTL"))
	if rawTTL == "" {
		rawTTL = defaultTTL
	}
	ttl, err := time.ParseDuration(rawTTL)
	if err != nil || ttl <= 0 {
		ttl = 15 * time.Minute
	}
	return &SignedArtifactAccessLayer{
		secret: []byte(secret),
		ttl:    ttl,
	}
}

func (s *SignedArtifactAccessLayer) SignedURL(basePath, artifactID, tenantID, workspaceID string) string {
	if s == nil || artifactID == "" {
		return ""
	}
	expiresAt := time.Now().UTC().Add(s.ttl).Unix()
	signature := s.sign(artifactID, tenantID, workspaceID, expiresAt)
	values := url.Values{}
	values.Set("tenant", tenantID)
	values.Set("workspace", workspaceID)
	values.Set("expires", strconv.FormatInt(expiresAt, 10))
	values.Set("signature", signature)
	return fmt.Sprintf("%s?%s", strings.TrimRight(basePath, "/"), values.Encode())
}

func (s *SignedArtifactAccessLayer) Validate(artifactID, tenantID, workspaceID string, expiresAt int64, signature string) bool {
	if s == nil || artifactID == "" || signature == "" {
		return false
	}
	if expiresAt < time.Now().UTC().Unix() {
		return false
	}
	expected := s.sign(artifactID, tenantID, workspaceID, expiresAt)
	return hmac.Equal([]byte(expected), []byte(signature))
}

func (s *SignedArtifactAccessLayer) sign(artifactID, tenantID, workspaceID string, expiresAt int64) string {
	mac := hmac.New(sha256.New, s.secret)
	_, _ = mac.Write([]byte(strings.Join([]string{
		artifactID,
		tenantID,
		workspaceID,
		strconv.FormatInt(expiresAt, 10),
	}, "|")))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func CleanArtifactStorageRoot(root string) string {
	root = strings.TrimSpace(root)
	if root == "" {
		root = "data/artifacts"
	}
	if filepath.IsAbs(root) {
		return root
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return root
	}
	return abs
}
