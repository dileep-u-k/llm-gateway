package platform

import "errors"

var (
	ErrUnauthorized = errors.New("authentication required")
	ErrForbidden    = errors.New("forbidden")
)
