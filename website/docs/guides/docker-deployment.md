---
sidebar_position: 7
title: Docker Deployment
description: Deploy Transcode with Docker and container orchestration
---

# Docker Deployment

Run Transcode in containers for reproducible, scalable deployments.

## Quick Start

```bash
# Pull the image
docker pull transcode/transcode:latest

# Run a transcode job
docker run -v $(pwd):/data transcode/transcode \
  -i /data/input.mp4 \
  -o /data/output.mp4
```

## Available Images

| Image | Size | Description |
|-------|------|-------------|
| `transcode/transcode:latest` | ~50MB | Production CLI |
| `transcode/transcode:dev` | ~2GB | Development with toolchain |
| `transcode/transcode:alpine` | ~30MB | Minimal Alpine-based |
| `transcode/coordinator` | ~60MB | Distributed coordinator |
| `transcode/worker` | ~80MB | Distributed worker |

## Basic Usage

### Transcode a File

```bash
docker run -v /path/to/media:/data transcode/transcode \
  -i /data/input.mp4 \
  -o /data/output.mp4 \
  --video-codec h264 \
  --video-bitrate 5000
```

### With Progress Output

```bash
docker run -it -v $(pwd):/data transcode/transcode \
  -i /data/input.mp4 \
  -o /data/output.mp4 \
  --verbose
```

### JSON Output for Automation

```bash
docker run -v $(pwd):/data transcode/transcode \
  -i /data/input.mp4 \
  -o /data/output.mp4 \
  --json > result.json
```

## Docker Compose

### Basic Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  transcode:
    image: transcode/transcode:latest
    volumes:
      - ./input:/input:ro
      - ./output:/output:rw
    command: ["-i", "/input/video.mp4", "-o", "/output/video.mp4"]
```

Run:
```bash
docker compose run --rm transcode
```

### Development Environment

```yaml
version: '3.8'

services:
  transcode:
    image: transcode/transcode:latest
    volumes:
      - ./data:/data

  dev:
    image: transcode/transcode:dev
    volumes:
      - .:/workspace
    working_dir: /workspace
    command: ["cargo", "watch", "-x", "test"]

  test:
    image: transcode/transcode:dev
    volumes:
      - .:/workspace
    working_dir: /workspace
    command: ["cargo", "test"]

  lint:
    image: transcode/transcode:dev
    volumes:
      - .:/workspace
    working_dir: /workspace
    command: ["cargo", "clippy", "--", "-D", "warnings"]
```

## Distributed Deployment

### Coordinator + Workers

```yaml
version: '3.8'

services:
  coordinator:
    image: transcode/coordinator:latest
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
    volumes:
      - coordinator-data:/data

  worker-1:
    image: transcode/worker:latest
    environment:
      - COORDINATOR_URL=http://coordinator:8080
      - WORKER_ID=worker-1
    volumes:
      - shared-storage:/media
    depends_on:
      - coordinator

  worker-2:
    image: transcode/worker:latest
    environment:
      - COORDINATOR_URL=http://coordinator:8080
      - WORKER_ID=worker-2
    volumes:
      - shared-storage:/media
    depends_on:
      - coordinator

volumes:
  coordinator-data:
  shared-storage:
```

### Scale Workers

```bash
docker compose up -d --scale worker=5
```

## GPU Support

### NVIDIA GPU

```yaml
version: '3.8'

services:
  transcode-gpu:
    image: transcode/transcode:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./data:/data
```

Or with `docker run`:
```bash
docker run --gpus all -v $(pwd):/data transcode/transcode \
  -i /data/input.mp4 \
  -o /data/output.mp4 \
  --hardware-acceleration
```

## Kubernetes

### Single Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: transcode-job
spec:
  template:
    spec:
      containers:
      - name: transcode
        image: transcode/transcode:latest
        args:
          - "-i"
          - "/data/input.mp4"
          - "-o"
          - "/data/output.mp4"
          - "--video-bitrate"
          - "5000"
        volumeMounts:
        - name: media
          mountPath: /data
      volumes:
      - name: media
        persistentVolumeClaim:
          claimName: media-pvc
      restartPolicy: Never
  backoffLimit: 3
```

### Distributed Cluster

```yaml
# coordinator-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transcode-coordinator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: transcode-coordinator
  template:
    metadata:
      labels:
        app: transcode-coordinator
    spec:
      containers:
      - name: coordinator
        image: transcode/coordinator:latest
        ports:
        - containerPort: 8080
        env:
        - name: RUST_LOG
          value: info
---
apiVersion: v1
kind: Service
metadata:
  name: transcode-coordinator
spec:
  selector:
    app: transcode-coordinator
  ports:
  - port: 8080
    targetPort: 8080
---
# worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transcode-worker
spec:
  replicas: 5
  selector:
    matchLabels:
      app: transcode-worker
  template:
    metadata:
      labels:
        app: transcode-worker
    spec:
      containers:
      - name: worker
        image: transcode/worker:latest
        env:
        - name: COORDINATOR_URL
          value: "http://transcode-coordinator:8080"
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        volumeMounts:
        - name: media
          mountPath: /media
      volumes:
      - name: media
        persistentVolumeClaim:
          claimName: media-pvc
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: transcode-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: transcode-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Building Custom Images

### Dockerfile

```dockerfile
FROM transcode/transcode:latest as base

# Add custom configuration
COPY transcode.toml /etc/transcode/config.toml

# Set default environment
ENV TRANSCODE_CONFIG=/etc/transcode/config.toml

ENTRYPOINT ["transcode"]
```

### Multi-Stage Build

```dockerfile
# Build stage
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release -p transcode-cli

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/transcode /usr/local/bin/
ENTRYPOINT ["transcode"]
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSCODE_THREADS` | Number of threads | Auto-detect |
| `TRANSCODE_LOG_LEVEL` | Log level | info |
| `TRANSCODE_CONFIG` | Config file path | None |
| `COORDINATOR_URL` | Coordinator address | None |
| `WORKER_ID` | Worker identifier | hostname |

### Config File Mount

```yaml
services:
  transcode:
    image: transcode/transcode:latest
    volumes:
      - ./config.toml:/etc/transcode/config.toml:ro
      - ./data:/data
    environment:
      - TRANSCODE_CONFIG=/etc/transcode/config.toml
```

## Security

### Non-Root User

The official images run as non-root user `transcode` (UID 1000):

```bash
# Ensure volume permissions
docker run -v $(pwd):/data:rw transcode/transcode ...
```

### Read-Only Filesystem

```yaml
services:
  transcode:
    image: transcode/transcode:latest
    read_only: true
    tmpfs:
      - /tmp
    volumes:
      - ./data:/data
```

### Resource Limits

```yaml
services:
  transcode:
    image: transcode/transcode:latest
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

## Monitoring

### Health Checks

```yaml
services:
  coordinator:
    image: transcode/coordinator:latest
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Observability Stack

```yaml
version: '3.8'

services:
  coordinator:
    image: transcode/coordinator:latest
    ports:
      - "8080:8080"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

`prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'transcode'
    static_configs:
      - targets: ['coordinator:8080']
```

## Next Steps

- [Distributed Processing](/docs/guides/distributed-processing) - Scale across workers
- [CLI Reference](/docs/reference/cli) - All CLI options
- [Configuration](/docs/reference/configuration) - Full configuration guide
