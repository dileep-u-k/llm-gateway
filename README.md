# In file: README.md

# 🧠 LLM Gateway - Phase 1: Core MVP

This is the foundational implementation of a high-performance LLM Gateway built in Go. This MVP establishes the core architecture, including a standardized API, a modular client for OpenAI, and a production-ready HTTP server with graceful shutdown.

## 🚀 How to Run
1.  **Configure**: Copy `.env.example` to `.env` and add your OpenAI API key.
2.  **Install Dependencies**: `go mod tidy`
3.  **Run**: `go run ./cmd/gateway`



# 🧠 LLM Gateway - Phase 2: RAG & Caching

This repository contains the code for a high-performance, intelligent LLM Gateway built in Go. This version builds upon the core MVP by adding a complete Retrieval-Augmented Generation (RAG) pipeline and a high-speed semantic cache.

The gateway can now answer questions using a private knowledge base and cache results to provide near-instantaneous responses for repeated queries, drastically reducing latency and API costs.

# 🏛️ Architecture (Phase 2)
The request flow is now significantly more intelligent. The gateway leverages Docker for local services (Redis) and connects to cloud services (Pinecone, OpenAI) for its knowledge and generation capabilities.

# Request Flow:

Cache Check: The gateway first checks a Redis cache to see if the exact prompt has been answered recently. If a cache hit occurs, the stored response is returned instantly.

RAG Retrieval: On a cache miss, the gateway queries a Pinecone vector database to find relevant documents from our knowledge base.

Prompt Augmentation: The retrieved context is prepended to the user's original prompt.

LLM Generation: The new, context-rich prompt is sent to the OpenAI API for a final, accurate answer.

Cache Update: The new response is stored in Redis before being sent back to the client, ensuring future requests are fast.

# 🚀 Getting Started
Follow these steps to run the complete system on your local machine.

1. Prerequisites

Go (version 1.22 or later)

Docker Desktop (running)

An OpenAI API Key

A Pinecone API Key and Index

2. Configure Environment Variables

The application uses a .env file to manage all secret keys.

Copy the template:

cp .env.example .env

Add your API Keys: Open the new .env file and add your keys from OpenAI and Pinecone.

3. Start Local Infrastructure

We use Docker Compose to start the Redis cache server.

docker-compose up -d

This command will run a Redis container in the background.

4. Ingest Knowledge Base

This is a one-time step to populate your Pinecone vector database with the documents from the /data directory.

go run ./cmd/ingestor

5. Install Dependencies & Run the Gateway

Tidy dependencies: This command downloads all necessary Go packages.

go mod tidy

Run the server:

go run ./cmd/gateway

The server will start and listen on http://localhost:8080.

⚙️ API Testing
You can now test the full RAG and caching pipeline.

# Test 1: Cache Miss (RAG-Powered Response)

This is the first time you ask this question. The gateway will use RAG and contact the LLM. Notice the higher latency.

curl -X POST http://localhost:8080/api/v1/generate \
-H "Content-Type: application/json" \
-d '{
    "prompt": "How does Go handle concurrency?",
    "config": {
        "preferred_model": "gpt-4"
    }
}'

# Expected Response: A detailed answer about goroutines and channels, with a latency_ms of over 1000.

# Test 2: Cache Hit (Instant Response)

Run the exact same command again. This time, the response will be served directly from the Redis cache.

curl -X POST http://localhost:8080/api/v1/generate \
-H "Content-Type: application/json" \
-d '{
    "prompt": "How does Go handle concurrency?",
    "config": {
        "preferred_model": "gpt-4"
    }
}'

# Expected Response: The same content, but with a latency_ms of less than 50, demonstrating the power of the cache.