# Security Model

## Auth

- Open local mode is the default when no platform tokens are configured.
- Token mode is enabled automatically when any `PLATFORM_*_TOKEN` or `PLATFORM_TOKENS_JSON` value is present.
- Admin and operator roles can access the control plane.

## RBAC

- `admin`: full control-plane access and strict-force permission
- `operator`: control-plane access and strict-force permission
- `user`: end-user APIs only

## Artifact protection

- Uploaded and generated artifacts are tagged with tenant and workspace metadata.
- Artifact content is exposed through signed URLs instead of raw filesystem paths.
- Access checks enforce tenant/workspace boundaries before serving content.

## Audit

- Generation requests
- Policy override updates
- Policy simulations
- Artifact uploads and reads

Sensitive metadata is redacted before audit persistence.
