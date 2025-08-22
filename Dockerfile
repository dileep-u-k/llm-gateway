# Dockerfile

# --- STAGE 1: Build ---
# This stage compiles the Go application. We use a specific Go version for reproducibility.
FROM golang:1.24.5-alpine AS build

# Set the working directory inside the container
WORKDIR /app

# Copy the Go module files and download dependencies.
# This is done in a separate step to leverage Docker's layer caching.
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the application source code
COPY . .

# Build the application.
# The -o flag specifies the output file name.
# CGO_ENABLED=0 is important for creating a static binary that runs in a minimal container.
# -ldflags "-w -s" strips debug information, making the binary smaller.
RUN CGO_ENABLED=0 GOOS=linux go build -o /llm-gateway -ldflags="-w -s" ./cmd/gateway

# --- STAGE 2: Final ---
# This stage creates the final, lightweight image for production.
# "scratch" is a special, empty Docker image, providing the smallest possible base.
FROM scratch

# Set the working directory to /app
WORKDIR /app

# Copy only the compiled binary from the 'build' stage into the /app directory.
# Nothing else (source code, build tools) is included in the final image.
COPY --from=build /llm-gateway /app/llm-gateway

# Expose the port that the gateway listens on (defined in your .env file).
EXPOSE 8081

# Set the command to run when the container starts.
ENTRYPOINT ["/app/llm-gateway"]