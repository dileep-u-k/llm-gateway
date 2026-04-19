# Runbooks

## Gateway is up but requests fail

1. Check `/readyz` and `/api/v1/metrics`.
2. Confirm Redis connectivity and worker queue depth.
3. Inspect `/api/v1/platform/admin/overview` for recent logs, traces, and audit events.
4. Verify provider keys and policy restrictions before rotating models.

## Async backlog grows

1. Scale the worker deployment from [`deploy/kubernetes/worker-deployment.yaml`](/Users/dileepuk/Desktop/Developer/llm-gateway/deploy/kubernetes/worker-deployment.yaml).
2. Inspect `job_state_distribution` and retry counters in metrics.
3. Review recent failed jobs in the admin control plane.

## Artifact access fails

1. Check that the signed URL has not expired.
2. Confirm the tenant/workspace in the query matches the artifact metadata.
3. Verify `PLATFORM_SIGNING_SECRET` consistency across API replicas.

## Policy change causes route failures

1. Use `POST /api/v1/platform/admin/policies/simulate` with the target request.
2. Inspect the `allowed_models` and `filter_error` fields.
3. If needed, apply a scoped override with `PUT /api/v1/platform/admin/policies/:tenant/:workspace`.
