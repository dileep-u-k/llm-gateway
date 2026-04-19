# Phase 0.5 Hardening Notes

## Session invariants

- A conversation has exactly one active mode at a time: `dynamic` or `forced`.
- A forced conversation has exactly one active pinned model.
- Forced sessions now persist `force_scope`, `strict_force`, and `effective_model_id` so the runtime can distinguish the user pin from the currently active failover target.
- An explicit `force_model` request always overrides the previous forced pin when the requested model is available.
- If `force_if_available_else_fallback=true`, an unavailable forced model request is converted into a healthy forced fallback and the fallback becomes the new pin.
- If `strict_force=true` or `force_scope=strict_end_to_end_force`, an unavailable forced model fails fast instead of silently substituting.
- Forced-session failover updates the effective model while preserving the original pin and conversation id.
- Every override persists `override_count`, `last_override_at`, `last_override_from`, `last_override_to`, and `session_mode_version` in Redis.

## Layered health model

The health checker now persists three layers of runtime health:

- Provider health: coarse reachability and account access for a provider.
- Model health: rolling status for a concrete model including circuit-breaker state.
- Capability health: status for `text_generation`, `image_generation`, and `embeddings`.

Routing rules:

1. Reject models without provider or capability access.
2. Reject unsupported capabilities.
3. Reject offline models and models behind an open circuit.
4. Prefer online candidates over degraded candidates.
5. Score surviving candidates by quality, cost, latency, and reliability.
6. Persist an inspectable route explanation with health inputs, filtered candidates, and ordered fallback candidates.

## RAG hardening

- Chunk ingestion now stores `source`, `section`, `timestamp`, `version`, `chunk_index`, and `content_hash` metadata.
- Retrieval applies local reranking, duplicate suppression, stale-source filtering, and max-context budgeting.
- Response cache keys include prompt hash, history hash, routed model, routing mode, and retrieved version signature.
- Source versions are published into Redis during ingestion so runtime retrieval can detect stale Pinecone matches.

## Metrics

`GET /api/v1/metrics` exposes a JSON snapshot for:

- requests per provider/model
- errors per provider/model
- failover and forced override counters
- cache and RAG counters
- provider/model/capability health transitions
- latency and retrieval latency p50/p95
- cache hit rate, RAG hit rate, and error rate per provider/model

## Test coverage added in this phase

- router health gating and circuit-breaker selection
- route explanation and fallback chain generation
- mid-session forced override behavior
- forced override fallback behavior
- strict-force rejection behavior
- forced-session continuity with separate pinned/effective models
- version-aware RAG cache keys
- stale/duplicate retrieval filtering
