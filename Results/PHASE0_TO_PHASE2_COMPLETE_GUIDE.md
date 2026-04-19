# Autonomous AI Orchestration
## Complete Guide Through Phase 2

This document explains the current state of the project through Phase 2. It covers:

* what the system is
* what has been implemented so far
* the architecture and workflow
* how to run it
* how to ingest knowledge for retrieval
* how to test it
* how to demo it
* what results are currently achieved
* what success metrics matter at this stage
* what remains for later phases

## 1. Project Vision

The project is evolving from a text-first multi-model gateway into a provider-agnostic AI orchestration platform.

The long-term target is:

* capability-aware routing
* resilient model execution
* retrieval and grounding
* memory-aware orchestration
* structured tool use
* multimodal execution
* async AI ops and enterprise controls

At the current point in the roadmap, the project has completed:

* Phase 0: Core Orchestration Hardening and Correctness Upgrade
* Phase 1: Unified Control Plane, Capability Registry, and Route Intelligence
* Phase 2: Retrieval, Memory, Grounding, and Tool-Augmented Reasoning

That means the platform is currently best described as:

A knowledge-aware execution runtime built on top of a resilient text-first orchestration gateway.

## 2. What Has Been Built So Far

## 2.1 Core Platform Baseline

The repository currently provides:

* a unified HTTP gateway built with Go and Gin
* multi-provider abstraction across OpenAI, Anthropic, Gemini, and Mistral
* Redis-backed state, caching, profiling, session tracking, and metadata persistence
* Pinecone-backed vector retrieval
* Docker and Docker Compose local runtime support
* automated Go test coverage across core orchestration paths

Main runtime entrypoints:

* `cmd/gateway`
* `cmd/ingestor`

## 2.2 Phase 0 Results

Phase 0 focused on making the orchestration baseline correct and explicit.

Implemented outcomes:

* dynamic and forced conversation sessions
* mid-session forced-model override support
* explicit force semantics:
  * `primary_reasoner_force`
  * `capability_scoped_force`
  * `strict_end_to_end_force`
* strict-force failure behavior when fallback is forbidden
* graceful forced fallback when enabled
* layered health semantics:
  * provider health
  * model health
  * capability health
* failover metadata surfaced in API responses
* orchestration metadata persistence in Redis
* stronger test coverage around session continuity and failover behavior

What this means:

* model pinning is not a blind override anymore
* failover behavior is inspectable and deterministic
* the system can explain why a route was selected or changed

## 2.3 Phase 1 Results

Phase 1 turned the gateway into a runtime control plane.

Implemented outcomes:

* provider registry
* model registry
* capability registry
* routing strategy framework
* route policy engine
* intent classification for route planning
* structured route explanation metadata
* fallback graph support
* orchestration metadata TTL and persistence
* metrics endpoint for runtime inspection

What this means:

* the platform does not just pick a model by heuristics
* it reasons over providers, models, capabilities, health, preferences, and route policies
* the route is observable and explainable

## 2.4 Phase 2 Results

Phase 2 upgraded the system from model routing to context-aware execution.

Implemented outcomes:

### Retrieval

* richer chunk metadata:
  * source
  * document ID
  * document title
  * section
  * section path
  * version
  * updated timestamp
  * ingestion timestamp
  * retrieval score
  * rerank score
  * freshness
* retrieval reranking using vector score, lexical overlap, and freshness
* duplicate suppression
* stale-source filtering
* source diversity selection
* context budgeting
* retrieval provenance persistence in Redis
* hierarchical ingestion with section-aware chunking

### Memory

* short-term memory store
* session summary memory
* structured working memory
* memory compaction
* artifact linking for retrieved sources

Stored signals now include:

* recent messages
* recent tool outputs
* recent retrieved evidence
* recent route decisions
* user goal progression
* current subtask
* active knowledge sources
* active constraints
* answer mode
* effective model

### Context Composition

* context composer
* prompt budget shaping
* structured prompt assembly
* clear separation of:
  * system instructions
  * execution constraints
  * session memory
  * evidence
  * tool outputs
  * answer mode

### Tools

* structured tool specifications
* tool planning
* tool validation
* timeout handling
* retry handling
* normalized tool outputs
* richer executed tool metadata

Built-in tools currently include:

* calculator
* weather
* news

### Grounding

* grounding policy engine
* evidence sufficiency checks
* answer modes:
  * `grounded-required`
  * `grounded-preferred`
  * `best-effort`
  * `abstain-if-insufficient`
  * `summarize-only-from-evidence`
  * `tool-output-priority`
* abstention behavior when grounded evidence is insufficient
* provenance-aware response shaping

### Response Metadata

Phase 2 responses can now include:

* `route`
* `intent`
* `session`
* `retrieval`
* `grounding`
* `memory`
* `context`
* `tool_plan`
* `tool_calls`

## 3. Current Architecture

At the current stage, the project is organized around these main subsystems:

* API contract layer: `internal/api`
* model clients and orchestration core: `internal/llm`
* tool subsystem: `internal/tools`
* observability and metrics: `internal/observability`
* gateway server: `cmd/gateway`
* ingestion pipeline: `cmd/ingestor`

## 3.1 Execution Path

The current high-level request flow is:

1. Client sends `POST /api/v1/generate`
2. Request is classified by intent
3. Session state is loaded from Redis if `conversation_id` is present
4. Route planning selects the model or explains a forced route
5. Memory snapshot is loaded
6. Retrieval runs if needed
7. Grounding policy decides how strict the answer must be
8. Context composer assembles the final prompt
9. Model generation runs
10. Tool loop runs if the selected path requires tools
11. Memory is updated
12. Response metadata is persisted
13. Final response is returned

## 3.2 Retrieval Workflow

Current retrieval flow:

1. Source documents are ingested by `cmd/ingestor`
2. Documents are split by semantic sections and chunked
3. Embeddings are generated
4. Vectors are stored in Pinecone with metadata
5. On query:
   * query embedding is created
   * Pinecone candidates are fetched
   * candidates are reranked
   * stale sources are removed
   * duplicates are removed
   * prompt budget is applied
   * provenance is attached

## 3.3 Tool Workflow

Current tool workflow:

1. Tool planner decides whether tools are likely needed
2. Tool-capable model is selected
3. Tool definitions are exposed to the model
4. Tool calls are validated against schemas
5. Tool runtime applies timeout and retry policy
6. Tool output is normalized
7. Tool results are sent back into the conversation
8. Final answer is generated and returned with tool metadata

## 3.4 Memory Workflow

Current memory workflow:

1. Session memory is loaded from Redis
2. Recent short-term events are read
3. Existing summary memory is read
4. Structured working memory is read
5. Retrieved sources and tool outputs are added back after execution
6. Old events are compacted into summary memory
7. Updated memory is written back to Redis

## 4. Repository Structure

Important implementation areas:

* `cmd/gateway/main.go`
  Composition root and server startup

* `cmd/gateway/handler.go`
  Main orchestration flow for generation requests

* `cmd/ingestor/main.go`
  RAG ingestion pipeline and Pinecone upsert flow

* `internal/llm/router.go`
  Routing engine and route selection logic

* `internal/llm/control_plane.go`
  Phase 1 route planning and policy orchestration

* `internal/llm/rag.go`
  Retrieval, reranking, provenance, caching

* `internal/llm/memory.go`
  Phase 2 memory engine

* `internal/llm/context_composer.go`
  Phase 2 context composition

* `internal/llm/grounding.go`
  Grounding policy engine and answer-mode enforcement

* `internal/tools/manager.go`
  Tool registry, planner, runtime entrypoint

* `internal/tools/phase2.go`
  Tool specification, retry, timeout, normalization structures

* `internal/api/types.go`
  Public request and response contract

## 5. How To Run The Project

## 5.1 Prerequisites

You need:

* Go `1.24.5+`
* Docker and Docker Compose
* Redis
* Pinecone index and host
* provider API keys for the models you want enabled

Optional:

* News API key for the news tool
* GCS bucket if you want `imagen-*` image generation

## 5.2 Required Environment Variables

Create `/Users/dileepuk/Desktop/Developer/llm-gateway/.env` with at least:

```env
PORT=8081
GIN_MODE=release

REDIS_ADDR=redis:6379

ENABLED_MODELS=gpt-4o,gemini-1.5-flash-latest,claude-sonnet-4-20250514,mistral-large-latest
ENABLED_IMAGE_MODELS=

OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
MISTRAL_API_KEY=your_mistral_key

PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_HOST=https://your-index-host

EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_URL=https://api.openai.com/v1/embeddings

NEWS_API_KEY=your_newsapi_key_optional
```

If you want image demos:

```env
ENABLED_IMAGE_MODELS=dall-e-3
```

For Imagen:

```env
ENABLED_IMAGE_MODELS=imagen-2
GCS_BUCKET_NAME=your_bucket_name
```

## 5.3 Run With Docker

From the repository root:

```bash
cd /Users/dileepuk/Desktop/Developer/llm-gateway
docker compose up --build
```

The API will be available at:

```text
http://localhost:8081
```

Main endpoints:

* `POST /api/v1/generate`
* `GET /api/v1/metrics`

## 5.4 Run Locally Without Docker

Start Redis separately, then run:

```bash
cd /Users/dileepuk/Desktop/Developer/llm-gateway
set -a; source .env; set +a
export GIN_MODE=debug
export REDIS_ADDR=localhost:6379
go run ./cmd/gateway
```

## 5.5 Ingest Data For RAG

Phase 2 retrieval demos only make sense after ingestion.

Run:

```bash
cd /Users/dileepuk/Desktop/Developer/llm-gateway
docker compose up -d redis
set -a; source .env; set +a
export REDIS_ADDR=localhost:6379
go run ./cmd/ingestor
```

What ingestion does:

* reads files under `data/`
* chunk-splits documents
* embeds chunks
* upserts vectors into Pinecone
* stores source-version manifest in Redis
* publishes corpus version

## 6. How To Test The Project

## 6.1 Automated Tests

Run all tests:

```bash
cd /Users/dileepuk/Desktop/Developer/llm-gateway
go test ./...
```

At the time this document was added, this passed successfully.

Focused test runs:

```bash
go test ./cmd/gateway
go test ./internal/llm
go test ./internal/tools
```

## 6.2 What The Current Tests Cover

Covered areas include:

* forced session continuity
* forced override behavior
* strict force rejection
* forced fallback behavior
* orchestration metadata persistence
* response-cache version invalidation
* stale and duplicate RAG filtering
* retrieval provenance metadata
* grounding abstention logic
* tool-output grounding satisfaction
* memory persistence and compaction
* tool validation and retry behavior

## 6.3 Manual Validation Checklist

Manual checks for the current system:

* gateway starts cleanly
* `/api/v1/metrics` responds
* basic text route works
* coding preference works
* RAG works after ingestion
* grounding modes appear in responses
* memory metadata appears for conversations with `conversation_id`
* tool planning and tool calls appear for weather/calculator/news prompts
* forced session metadata appears and persists
* failover metadata appears when applicable

## 7. End-to-End Workflow

This is the current full workflow from document ingestion to grounded answer delivery.

## 7.1 Offline Knowledge Preparation

1. Place source files under `data/`
2. Run `go run ./cmd/ingestor`
3. Documents are chunked and embedded
4. Pinecone receives vectors
5. Redis receives corpus/source version metadata

## 7.2 Online Query Workflow

1. User sends generation request
2. Intent analyzer identifies whether the request is:
   * text/RAG
   * weather tool
   * calculator tool
   * news tool
   * image creation
3. Session state is restored if the request belongs to an existing conversation
4. Control plane selects the best route
5. Retrieval runs for text knowledge workflows
6. Grounding policy determines evidence requirements
7. Memory is loaded and joined with evidence
8. Context composer constructs the final prompt
9. Model generates a response
10. Tools are executed if needed
11. Memory is updated
12. Final response and orchestration metadata are persisted

## 8. Results Achieved So Far

Through Phase 2, the project now demonstrates:

* correct multi-provider model selection
* explicit and inspectable routing logic
* session-aware forced execution semantics
* layered health-based resilience
* retrieval with provenance and freshness handling
* memory-aware context shaping
* structured tool use with retries and validation
* grounded answer control with abstention behavior
* persistent orchestration metadata
* observable runtime counters and latencies

This is a substantial shift from a plain gateway wrapper.

The project now behaves like:

* a runtime control plane
* a knowledge-aware execution runtime
* a memory-aware reasoning layer

## 9. Current Success Metrics

At this stage, the most meaningful success metrics are operational and behavioral rather than business metrics.

## 9.1 Correctness Metrics

Success indicators:

* forced sessions remain consistent across requests
* strict force rejects fallback correctly
* fallback activates only when allowed
* route metadata matches actual selected model/provider
* stale RAG chunks do not leak into retrieval results
* duplicate chunks are suppressed
* answer abstains when grounding mode requires evidence but evidence is missing

## 9.2 Retrieval Metrics

What to monitor:

* retrieval hit rate
* stale source skips
* retrieval latency
* selected chunk count
* source diversity
* prompt budget usage

Available today:

* retrieval latency through metrics
* retrieval hit/miss counters
* stale-source skip counters
* provenance metadata in response payloads

## 9.3 Tool Metrics

What to monitor:

* whether tool planning selected the right tool
* tool execution success vs failure
* retry behavior
* tool latency
* usefulness of final tool output in the answer

Available today:

* `tool_plan`
* `tool_calls`
* tool status
* attempt count
* duration per tool call

## 9.4 Memory Metrics

What to monitor:

* coherence across multiple turns
* persistence of knowledge sources
* compaction behavior in longer sessions
* summary usefulness

Available today:

* `memory.short_term_event_count`
* `memory.summary`
* `memory.knowledge_sources`
* `memory.active_constraints`

## 9.5 Runtime Metrics

Metrics endpoint:

```text
GET /api/v1/metrics
```

Current examples include:

* requests per provider/model
* errors per provider/model
* error rate
* health transitions
* cache hit rate
* RAG hit rate
* request latency percentiles
* retrieval latency percentiles

## 10. Expected Demo Results

When the system is working correctly, demo requests should show:

* text requests return `model_used`, `route`, `intent`
* grounded requests return `retrieval` and `grounding`
* conversation requests return `session` and `memory`
* tool requests return `tool_plan` and `tool_calls`
* repeated grounded requests may return cache hits if the prompt, history, route, and retrieval signature match

## 11. Current Limitations

These are expected because only Phases 0, 1, and 2 are complete.

Not fully implemented yet:

* multimodal stage planner
* OCR/document execution stages
* audio and video pipelines
* async heavy-job runtime
* tracing dashboard
* evaluation dashboard
* governance and enterprise control plane

Also important:

* retrieval quality still depends heavily on the quality and coverage of ingested source data
* tool selection is rule-guided and model-assisted, not yet a full multi-stage planner
* grounding is controlled, but not yet backed by a large benchmark suite or dashboard

## 12. What Counts As “Done So Far”

The most honest summary of the current system is:

Phase 0 through Phase 2 are complete enough to demonstrate:

* resilient orchestration
* route intelligence
* forced execution correctness
* retrieval intelligence
* memory-aware prompt assembly
* disciplined tool augmentation
* grounding-aware answer shaping

This is already a strong and technically impressive platform stage.

It is no longer just a multi-provider wrapper.

It is now a structured AI execution runtime with routing, retrieval, memory, grounding, and tool orchestration.

## 13. Recommended Next Step

The clean next milestone is Phase 3:

* multimodal request model
* artifact pipeline
* stage-level execution planning
* modality-aware route planning

That is the next major architecture jump after the current work.
