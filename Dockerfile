# Dockerfile

# --- STAGE 1: Build ---
# This stage compiles the Go application.
FROM golang:1.24.5-alpine AS build

WORKDIR /app

# Cache module downloads for faster subsequent builds
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the application source code
COPY . .

# Build the statically-linked, production-ready binary
RUN CGO_ENABLED=0 GOOS=linux go build -o /app/llm-gateway -ldflags="-w -s" ./cmd/gateway


# --- STAGE 2: Final Production Image ---
# Use a specific, minimal Alpine image. This includes SSL certificates and is more secure than scratch.
# Pinning the version (e.g., :3.20) ensures reproducible builds.
FROM alpine:3.20

# Set the working directory
WORKDIR /app

# Copy only the compiled binary from the 'build' stage.
COPY --from=build /app/llm-gateway /app/llm-gateway

# The config.yaml will be mounted into this directory by Docker Compose or Kubernetes.

# Expose the port that the gateway listens on.
EXPOSE 8081

# Set the command to run when the container starts.
ENTRYPOINT ["/app/llm-gateway"]