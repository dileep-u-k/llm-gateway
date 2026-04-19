# Demo And Test Examples

This file contains copy-pasteable examples for manual testing and live demos.

Set:

```bash
BASE=http://localhost:8081/api/v1/generate
```

## 1. Health And Metrics

```bash
curl -s http://localhost:8081/api/v1/metrics
```

Expected:

* JSON response
* counters
* latency metrics
* retrieval metrics after RAG requests

## 2. Basic Text Generation

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Explain dynamic model routing in AI gateways.",
  "user_id":"demo-user",
  "config":{"preference":"balanced","max_tokens":250}
}'
```

Inspect:

* `content`
* `model_used`
* `route`
* `intent`

## 3. Coding Route

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Write a Go retry helper with exponential backoff.",
  "user_id":"demo-user",
  "config":{"preference":"best-for-coding","max_tokens":300}
}'
```

Inspect:

* coding-friendly route choice
* route explanation

## 4. Grounded Retrieval

Run ingestion first:

```bash
cd /Users/dileepuk/Desktop/Developer/llm-gateway
set -a; source .env; set +a
export REDIS_ADDR=localhost:6379
go run ./cmd/ingestor
```

Then:

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Based on the project knowledge base, summarize the system architecture.",
  "user_id":"demo-user",
  "config":{"answer_mode":"grounded-required","max_tokens":400}
}'
```

Inspect:

* `retrieval`
* `grounding`
* `rag_context_used`
* `context`

## 5. Summarize Only From Evidence

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Summarize only what the ingested project documents say about retrieval and memory.",
  "user_id":"demo-user",
  "config":{"answer_mode":"summarize-only-from-evidence","max_tokens":400}
}'
```

Inspect:

* evidence-bound answer behavior
* `grounding.answer_mode`

## 6. Conversation Memory

Turn 1:

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"We are building a reliable LLM gateway. Remember reliability matters more than cost.",
  "user_id":"demo-user",
  "conversation_id":"memory-demo-1",
  "config":{"max_tokens":200}
}'
```

Turn 2:

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Now propose the routing policy for that system.",
  "user_id":"demo-user",
  "conversation_id":"memory-demo-1",
  "history":[
    {"role":"user","content":"We are building a reliable LLM gateway. Remember reliability matters more than cost."}
  ],
  "config":{"max_tokens":250}
}'
```

Inspect:

* `session`
* `memory`
* `context`

## 7. Force A Model

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Answer using the forced model.",
  "user_id":"demo-user",
  "conversation_id":"forced-demo-1",
  "config":{
    "force_model":"claude-sonnet-4-20250514",
    "force_scope":"primary_reasoner_force",
    "max_tokens":200
  }
}'
```

Inspect:

* `session.mode`
* `session.pinned_model`
* `route.forced_semantics`

## 8. Mid-Session Forced Override

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Switch this session to Gemini.",
  "user_id":"demo-user",
  "conversation_id":"forced-demo-1",
  "config":{
    "force_model":"gemini-1.5-flash-latest",
    "force_scope":"primary_reasoner_force",
    "max_tokens":200
  }
}'
```

Inspect:

* `session.override_count`
* `session.effective_model`

## 9. Strict Force

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Use only the forced model strictly.",
  "user_id":"demo-user",
  "conversation_id":"strict-force-demo-1",
  "config":{
    "force_model":"gpt-4o",
    "force_scope":"strict_end_to_end_force",
    "strict_force":true,
    "max_tokens":200
  }
}'
```

Inspect:

* forced semantics in route metadata
* failure behavior if the forced model becomes unavailable

## 10. Weather Tool

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"What is the weather in Bengaluru right now?",
  "user_id":"demo-user",
  "config":{"max_tokens":200}
}'
```

Inspect:

* `tool_plan`
* `tool_calls`
* tool output in final answer

## 11. Calculator Tool

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Calculate 245 * 18 / 3",
  "user_id":"demo-user",
  "config":{"max_tokens":120}
}'
```

Inspect:

* calculator tool call
* normalized tool metadata

## 12. News Tool

Only if `NEWS_API_KEY` is set:

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Give me the latest headlines about artificial intelligence.",
  "user_id":"demo-user",
  "config":{"max_tokens":250}
}'
```

Inspect:

* `tool_plan`
* `tool_calls`
* fresh-news behavior

## 13. Image Generation

Only if `ENABLED_IMAGE_MODELS` is configured:

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Create a clean architecture diagram of an LLM gateway with routing, Redis, Pinecone, and multiple model providers.",
  "user_id":"demo-user",
  "config":{"image_preference":"diagram"}
}'
```

Inspect:

* `image_url`
* `model_used`

## 14. Cache Demonstration

Send the same grounded request twice:

```bash
curl -s "$BASE" -H 'Content-Type: application/json' -d '{
  "prompt":"Based on the project knowledge base, summarize the architecture and routing design.",
  "user_id":"demo-user",
  "config":{"answer_mode":"grounded-required","max_tokens":300}
}'
```

Expected:

* first response: likely `cache_status = "MISS"`
* second response: may become `cache_status = "HIT"` if route, retrieval signature, and context match

## 15. Recommended Demo Flow

Best live-demo order:

1. show `/api/v1/metrics`
2. basic text request
3. coding request
4. grounded retrieval request
5. conversation memory request
6. forced-session request
7. tool request
8. image request if enabled
9. show metrics again

This order highlights:

* routing
* control plane
* retrieval
* grounding
* memory
* tools
* observability
