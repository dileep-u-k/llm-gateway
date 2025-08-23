# Dockerfile

# --- STAGE 1: Build ---
# Use a Go image to compile the app
FROM golang:1.24.5-alpine AS build

WORKDIR /app

# Cache module downloads
COPY go.mod go.sum ./
RUN go mod download

# Copy project sources
COPY . .

# Build the statically-linked binary (output in current dir, /app)
RUN CGO_ENABLED=0 GOOS=linux go build -o llm-gateway -ldflags="-w -s" ./cmd/gateway

# --- STAGE 2: Production with scratch ---
FROM scratch

WORKDIR /app

# Copy the statically-built binary from the build stage
COPY --from=build /app/llm-gateway /app/llm-gateway
COPY --from=build /app/config.yaml /app/config.yaml
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
# If you need config.yaml, also include:
# COPY --from=build /app/config.yaml /app/config.yaml

EXPOSE 8081

ENTRYPOINT ["/app/llm-gateway"]