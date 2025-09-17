# Autonomous AI Orchestration: A Cloud-Native Platform for Resilient, Cost-Optimized LLM Operations

![Go Version](https://img.shields.io/badge/Go-1.24.5-blue.svg)
![Build Status](https://github.com/dileep-u-k/llm-gateway/actions/workflows/ci-cd.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project is an enterprise-grade API gateway designed to intelligently route, manage, and optimize requests to multiple Large Language Models (e.g., GPT-4o, Claude Sonnet, Gemini). It serves as a unified, resilient, and cost-optimized interface to the world of AI, demonstrating advanced software engineering principles for building scalable and fault-tolerant AI applications in Go.

---

## Architecture Diagram

*(This is a placeholder for your final Phase 5 diagram. After deploying to Kubernetes and setting up monitoring, you will replace this with a diagram showing the complete, scalable cloud architecture.)*



The gateway sits between the user and multiple LLM providers. It enriches prompts with a RAG pipeline, makes intelligent routing decisions based on real-time performance data, and manages conversational state, providing a single, unified API endpoint for any client application.

---

## 🚀 Key Features

* **🧠 Intelligent, Data-Driven Routing:** Selects the optimal model in real-time based on performance profiling (latency, cost, error rates) stored in Redis, aligning with user-defined preferences like "cost," "latency," or "max_quality."

* **🤖 Smart Prompt Analysis:** Automatically analyzes user prompts to infer the best routing strategy when none is specified, distinguishing between simple, complex, and coding-related tasks.

* **💬 Stateful, Multi-Turn Conversations:**
    * **Two Chat Modes:** Seamlessly supports "dynamic" chats (for flexibility) and "forced model" chats (for absolute consistency).
    * **Full Conversation History:** Processes chat history sent from the client to provide models with conversational memory, enabling a true ChatGPT-like experience.

* **🛡️ Automatic Failover & Resilience:**
    * **Proactive Health Checks:** A background service constantly monitors the health of all connected LLMs.
    * **Seamless Failover:** If a pinned model goes offline mid-conversation, the gateway automatically re-routes to the next-best healthy model. The API response includes failover details for the UI.

* **📄 Retrieval-Augmented Generation (RAG):** Enriches user prompts with relevant, up-to-date context from a Pinecone vector database.

* **⚡ Multi-Layer Caching:**
    * **Response Cache:** Caches final LLM responses in Redis for repeated queries.
    * **Embedding Cache:** Caches vector embeddings to reduce redundant API calls and lower costs.

* **🛠️ Agentic Tool Use:** Can leverage an LLM's function-calling ability to execute external tools (e.g., weather APIs, calculators).

* **📦 Production-Ready:** Fully containerized with Docker and features a complete CI/CD pipeline using GitHub Actions for automated testing, linting, and publishing.

* **🔌 Designed for Extensibility:** The gateway is built on a modular, interface-driven architecture (`LLMClient`, `ToolExecutor`). This makes it trivial to integrate **new AI models, providers, or custom tools** with minimal code changes, ensuring the platform can adapt to the rapidly evolving AI landscape.

---

### ## 🏛️ Architectural Principles

This project was not just built to be functional, but was architected around a set of core principles essential for modern, cloud-native AI systems:

* **Design for Failure:** The entire system assumes that external APIs can and will fail. Features like proactive health checks, automatic failover, and intelligent retries are built in to ensure the application is resilient and self-healing.
* **Configuration as Code:** All routing logic, model metadata, and thresholds are defined in a version-controlled `config.yaml` file. This allows for an agile, auditable, and GitOps-friendly approach to managing the system's intelligence without requiring a code redeployment.
* **Stateless by Default:** The core gateway service is designed to be stateless, allowing it to be scaled horizontally with ease in a Kubernetes environment. All state (caches, profiles, sessions) is correctly externalized to a dedicated, high-performance Redis layer.
* **Abstraction and Modularity:** By using a clean `LLMClient` interface, the system is decoupled from any single AI provider, preventing vendor lock-in and making it trivial to integrate new models as they become available.

---

### ## 🧠 Key Learnings & Challenges

Building this platform provided deep insights into the practical challenges of operating AI systems at scale.

* **The "Best" Model is Constantly Changing:** The key takeaway is that there is no single "best" LLM. The optimal choice is a dynamic target that depends on the specific task, the user's preference (cost vs. latency), and the real-time performance of the provider's API. This validated the need for a data-driven, dynamic routing engine.
* **The Importance of Atomic Operations:** Early in the development of the `Profiler`, it became clear that simple read/write operations to Redis were not sufficient for a high-concurrency environment. Implementing atomic transactions (`MULTI`/`EXEC`) was a critical step to prevent race conditions and ensure data consistency.
* **Heuristics as a Double-Edged Sword:** Designing the `PromptAnalyzer` highlighted the challenges of heuristic-based systems. While powerful, they require careful tuning and the implementation of safety nets (like negative keywords and length overrides) to handle the vast and unpredictable nature of user input in a live application.

---

##  📊 (Phase 5) Grafana Dashboard

*(This is a placeholder for your Phase 5 deliverable. This will be the most impactful visual in your README.)*

**This Grafana dashboard will display the real-time performance of the gateway, tracking request latency, cost-per-model, and the effectiveness of the routing and caching engines.**



---

## 🛠 Tech Stack

* **Backend:** Go, Gin
* **Database & Caching:** Redis, Pinecone
* **AI Providers:** OpenAI API, Anthropic API, Google Gemini API, Mistral API
* **DevOps & CI/CD:** Docker, Docker Compose, GitHub Actions
* **Cloud & Orchestration (Phase 5):** Kubernetes, Helm
* **Observability (Phase 5):** Prometheus, Grafana

---

## ⚙️ Local Setup & Running the Project

1.  **Prerequisites:**
    * Go 1.24.5+
    * Docker and Docker Compose

2.  **Clone the repository:**
    ```sh
    git clone [https://github.com/YOUR_USERNAME/llm-gateway.git](https://github.com/YOUR_USERNAME/llm-gateway.git)
    cd llm-gateway
    ```

3.  **Set up your environment:**
    * Rename `.env.example` (if you have one) to `.env`.
    * Fill in your secret API keys in the `.env` file.

4.  **Run the application:**
    ```sh
    docker-compose up --build
    ```
    The gateway will be available at `http://localhost:8081`.

---

## 🎯 Project Goals & Future Work

### Project Goals

This project was built to demonstrate a production-grade, scalable, and resilient architecture for leveraging multiple LLMs. The key goals achieved are:

-   **Cost Optimization:** Intelligently routing to cheaper models for simple tasks.
-   **Performance:** Minimizing latency by selecting the fastest model and using multi-layer caching.
-   **Reliability:** Ensuring high availability through automatic health checks and seamless failover.
-   **Flexibility:** Supporting multiple chat modes and conversational contexts.

### Future Work

- **Integrate More Providers & Tools:** Leverage the gateway's modular design to add support for other leading models (e.g., Meta's Llama series, xAI's Grok) and new external tools, further expanding the platform's capabilities.
-   **Implementing a UI:** Building a React/Next.js frontend with a full chat interface.
-   **Building a Cost-Tracking Dashboard:** Creating a UI to visualize the cost savings and performance metrics.
-   **Advanced Deployment Strategies:** Implementing canary deployments or A/B testing for new models using a service mesh like Istio.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.