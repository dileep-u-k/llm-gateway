# Phase 6 Platform Completion

Phase 6 turns the existing orchestration engine into a productized platform layer.

## What was added

- End-user surface served directly by the gateway at `/`
- Admin and developer control plane served at `/admin`
- Tenant/workspace-aware governance with policy bundles from [`platform.yaml`](/Users/dileepuk/Desktop/Developer/llm-gateway/platform.yaml)
- RBAC-capable auth with open local mode or token mode via `PLATFORM_*_TOKEN`
- Audit logging for generation, policy, and artifact actions
- Signed artifact access for uploaded and generated files
- Artifact upload endpoints for multimodal workflows
- Worker-only and API-only runtime modes for cloud deployment
- Kubernetes manifests for split API and worker deployments

## Key APIs

- `POST /api/v1/generate`
- `POST /api/v1/assets/upload`
- `GET /api/v1/artifacts`
- `GET /api/v1/artifacts/:id`
- `GET /api/v1/artifacts/:id/content`
- `GET /api/v1/platform/bootstrap`
- `GET /api/v1/platform/admin/overview`
- `POST /api/v1/platform/admin/policies/simulate`
- `PUT /api/v1/platform/admin/policies/:tenant/:workspace`

## Governance flow

1. Resolve principal from bearer token or open-local mode.
2. Resolve tenant and workspace policy bundle.
3. Normalize the request with workspace defaults.
4. Validate prompt, force scope, generation eligibility, and async constraints.
5. Filter route candidates by provider/model/capability/cost policy.
6. Annotate the final response with governance and security metadata.

## Runtime modes

- API mode: `HTTP_ENABLED=true`, `ASYNC_WORKERS_ENABLED=false`
- Worker mode: `HTTP_ENABLED=false`, `ASYNC_WORKERS_ENABLED=true`
- Monolith mode: both enabled for local development
