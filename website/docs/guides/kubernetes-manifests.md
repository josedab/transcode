---
sidebar_position: 12
title: Kubernetes Manifest Generation
description: Generate Kubernetes deployment manifests for transcode worker clusters
---

# Kubernetes Manifest Generation

The `transcode-k8s` crate generates production-ready Kubernetes manifests for deploying transcoding worker clusters, including Deployments, Horizontal Pod Autoscalers, Services, and Helm value files.

## Overview

Instead of hand-writing YAML, generate type-safe Kubernetes manifests directly from your Rust configuration:

```rust
use transcode_k8s::manifest::{
    generate_worker_deployment,
    generate_hpa,
    generate_service,
    generate_helm_values,
};

// Generate a worker Deployment
let deployment = generate_worker_deployment(
    "transcode-worker",  // name
    "transcode:1.0",     // image
    3,                    // replicas
    "500m",              // CPU request
    "1Gi",               // memory request
    "2",                 // CPU limit
    "4Gi",               // memory limit
);

println!("{}", deployment);
```

## Quick Start

```toml
[dependencies]
transcode-k8s = "1.0"
```

### Generate a Complete Deployment

```rust
use transcode_k8s::manifest::*;

fn main() {
    // Worker Deployment
    let deploy = generate_worker_deployment(
        "video-workers",
        "ghcr.io/transcode/worker:latest",
        5,       // 5 replicas
        "1",     // 1 CPU request
        "2Gi",   // 2GB RAM request
        "4",     // 4 CPU limit
        "8Gi",   // 8GB RAM limit
    );

    // Horizontal Pod Autoscaler
    let hpa = generate_hpa(
        "video-workers",   // target deployment
        2,                 // min replicas
        20,                // max replicas
        75,                // target CPU utilization %
    );

    // Service for worker discovery
    let service = generate_service("video-workers", 8080);

    // Prometheus ServiceMonitor
    let monitor = generate_service_monitor("video-workers", 9090, "30s");

    // Write manifests
    std::fs::write("deploy.yaml", &deploy).unwrap();
    std::fs::write("hpa.yaml", &hpa).unwrap();
    std::fs::write("service.yaml", &service).unwrap();
    std::fs::write("monitor.yaml", &monitor).unwrap();

    println!("Generated 4 Kubernetes manifests");
}
```

### Helm Values

Generate a Helm `values.yaml` for templated deployments:

```rust
let values = generate_helm_values(
    "transcode-cluster",
    "ghcr.io/transcode/worker:latest",
    3,       // default replicas
    "1",     // CPU request
    "2Gi",   // memory request
    "4",     // CPU limit
    "8Gi",   // memory limit
);

std::fs::write("values.yaml", &values).unwrap();
```

Output:

```yaml
replicaCount: 3
image:
  repository: ghcr.io/transcode/worker
  tag: latest
  pullPolicy: IfNotPresent
resources:
  requests:
    cpu: "1"
    memory: 2Gi
  limits:
    cpu: "4"
    memory: 8Gi
```

## Generated Manifest Examples

### Worker Deployment

The generated Deployment includes production best practices:

- Resource requests and limits
- Liveness and readiness probes
- Graceful shutdown with `terminationGracePeriodSeconds`
- Pod anti-affinity for high availability

### Horizontal Pod Autoscaler

Auto-scales workers based on CPU utilization:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: video-workers
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: video-workers
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 75
```

## Use Cases

- **CI/CD pipelines** — Generate manifests as part of your build process
- **Multi-environment deployments** — Different configs for staging vs. production
- **Helm chart generation** — Create values files for Helm-based deployments
- **GitOps workflows** — Commit generated manifests to a config repo

## API Reference

| Function | Description |
|----------|-------------|
| `generate_worker_deployment()` | Kubernetes Deployment YAML |
| `generate_hpa()` | HorizontalPodAutoscaler YAML |
| `generate_service()` | ClusterIP Service YAML |
| `generate_service_monitor()` | Prometheus ServiceMonitor YAML |
| `generate_helm_values()` | Helm values.yaml |

## Next Steps

- [Distributed Processing](/docs/guides/distributed-processing) — Coordinator/worker architecture
- [Docker Deployment](/docs/guides/docker-deployment) — Container image setup
