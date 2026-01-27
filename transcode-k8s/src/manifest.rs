//! Kubernetes manifest generation for transcoding deployments.

use crate::crd::{GpuType, TranscodeWorkerPool};
use crate::scaling::ScalingPolicy;
use serde::Serialize;
use std::collections::HashMap;

/// Container image configuration.
#[derive(Debug, Clone, Serialize)]
pub struct ImageConfig {
    pub repository: String,
    pub tag: String,
    pub pull_policy: String,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            repository: "ghcr.io/transcode/transcode-worker".into(),
            tag: "latest".into(),
            pull_policy: "IfNotPresent".into(),
        }
    }
}

/// Health probe configuration.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeConfig {
    pub path: String,
    pub port: u16,
    pub initial_delay_secs: u32,
    pub period_secs: u32,
    pub timeout_secs: u32,
    pub failure_threshold: u32,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            path: "/healthz".into(),
            port: 8080,
            initial_delay_secs: 10,
            period_secs: 15,
            timeout_secs: 5,
            failure_threshold: 3,
        }
    }
}

/// Resource requests and limits for a container.
#[derive(Debug, Clone, Serialize)]
pub struct ResourceSpec {
    pub cpu_request: String,
    pub cpu_limit: String,
    pub memory_request: String,
    pub memory_limit: String,
    pub gpu_count: u32,
    pub gpu_type: Option<GpuType>,
}

impl Default for ResourceSpec {
    fn default() -> Self {
        Self {
            cpu_request: "2".into(),
            cpu_limit: "4".into(),
            memory_request: "4Gi".into(),
            memory_limit: "8Gi".into(),
            gpu_count: 0,
            gpu_type: None,
        }
    }
}

/// Generates a Kubernetes Deployment YAML for a transcoding worker pool.
pub fn generate_worker_deployment(
    pool: &TranscodeWorkerPool,
    image: &ImageConfig,
    resources: &ResourceSpec,
    liveness: &ProbeConfig,
    readiness: &ProbeConfig,
) -> String {
    let name = &pool.metadata.name;
    let namespace = &pool.metadata.namespace;
    let replicas = pool.spec.replicas;

    let mut labels = HashMap::new();
    labels.insert("app".to_string(), "transcode-worker".to_string());
    labels.insert("pool".to_string(), name.clone());

    let mut env_vars = vec![
        ("TRANSCODE_WORKER_POOL".to_string(), name.clone()),
        ("TRANSCODE_NAMESPACE".to_string(), namespace.clone()),
    ];

    // GPU environment
    if let Some(gpu_type) = &resources.gpu_type {
        env_vars.push(("NVIDIA_VISIBLE_DEVICES".into(), "all".into()));
        env_vars.push(("NVIDIA_DRIVER_CAPABILITIES".into(), "compute,video".into()));
        labels.insert(
            "gpu-type".to_string(),
            format!("{:?}", gpu_type).to_lowercase(),
        );
    }

    let env_yaml: String = env_vars
        .iter()
        .map(|(k, v)| format!("        - name: {}\n          value: \"{}\"", k, v))
        .collect::<Vec<_>>()
        .join("\n");

    let mut resource_limits = format!(
        "            cpu: \"{}\"\n            memory: \"{}\"",
        resources.cpu_limit, resources.memory_limit
    );
    if resources.gpu_count > 0 {
        let gpu_key = resources
            .gpu_type
            .as_ref()
            .map(|g| g.resource_key())
            .unwrap_or("nvidia.com/gpu");
        resource_limits.push_str(&format!(
            "\n            {}: \"{}\"",
            gpu_key, resources.gpu_count
        ));
    }

    // Node selector
    let node_selector = if pool.spec.node_selector.is_empty() {
        String::new()
    } else {
        let selectors: Vec<String> = pool
            .spec
            .node_selector
            .iter()
            .map(|(k, v)| format!("        {}: \"{}\"", k, v))
            .collect();
        format!("      nodeSelector:\n{}", selectors.join("\n"))
    };

    // Tolerations
    let tolerations = if pool.spec.tolerations.is_empty() {
        String::new()
    } else {
        let tols: Vec<String> = pool
            .spec
            .tolerations
            .iter()
            .map(|t| {
                let mut tol = format!(
                    "        - key: \"{}\"\n          operator: \"{}\"\n          effect: \"{}\"",
                    t.key, t.operator, t.effect
                );
                if let Some(v) = &t.value {
                    tol.push_str(&format!("\n          value: \"{}\"", v));
                }
                tol
            })
            .collect();
        format!("      tolerations:\n{}", tols.join("\n"))
    };

    let labels_yaml: String = labels
        .iter()
        .map(|(k, v)| format!("        {}: \"{}\"", k, v))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: transcode-worker
    pool: "{name}"
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: transcode-worker
      pool: "{name}"
  template:
    metadata:
      labels:
{labels_yaml}
    spec:
      containers:
      - name: transcode-worker
        image: {repo}:{tag}
        imagePullPolicy: {pull_policy}
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP
        env:
{env_yaml}
        resources:
          requests:
            cpu: "{cpu_req}"
            memory: "{mem_req}"
          limits:
{resource_limits}
        livenessProbe:
          httpGet:
            path: {liveness_path}
            port: {liveness_port}
          initialDelaySeconds: {liveness_initial}
          periodSeconds: {liveness_period}
          timeoutSeconds: {liveness_timeout}
          failureThreshold: {liveness_failure}
        readinessProbe:
          httpGet:
            path: {readiness_path}
            port: {readiness_port}
          initialDelaySeconds: {readiness_initial}
          periodSeconds: {readiness_period}
          timeoutSeconds: {readiness_timeout}
          failureThreshold: {readiness_failure}
{node_selector}
{tolerations}"#,
        name = name,
        namespace = namespace,
        replicas = replicas,
        labels_yaml = labels_yaml,
        repo = image.repository,
        tag = image.tag,
        pull_policy = image.pull_policy,
        env_yaml = env_yaml,
        cpu_req = resources.cpu_request,
        mem_req = resources.memory_request,
        resource_limits = resource_limits,
        liveness_path = liveness.path,
        liveness_port = liveness.port,
        liveness_initial = liveness.initial_delay_secs,
        liveness_period = liveness.period_secs,
        liveness_timeout = liveness.timeout_secs,
        liveness_failure = liveness.failure_threshold,
        readiness_path = readiness.path,
        readiness_port = readiness.port,
        readiness_initial = readiness.initial_delay_secs,
        readiness_period = readiness.period_secs,
        readiness_timeout = readiness.timeout_secs,
        readiness_failure = readiness.failure_threshold,
        node_selector = node_selector,
        tolerations = tolerations,
    )
}

/// Generate a Kubernetes HorizontalPodAutoscaler YAML.
pub fn generate_hpa(pool_name: &str, namespace: &str, policy: &ScalingPolicy) -> String {
    format!(
        r#"apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {name}-hpa
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {name}
  minReplicas: {min}
  maxReplicas: {max}
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 4
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: {cooldown}
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
  metrics:
  - type: Pods
    pods:
      metric:
        name: transcode_queue_depth_per_worker
      target:
        type: AverageValue
        averageValue: "{target_depth}""#,
        name = pool_name,
        namespace = namespace,
        min = policy.min_replicas,
        max = policy.max_replicas,
        cooldown = policy.cooldown_secs,
        target_depth = policy.target_queue_depth_per_worker,
    )
}

/// Generate a Kubernetes Service YAML.
pub fn generate_service(pool_name: &str, namespace: &str) -> String {
    format!(
        r#"apiVersion: v1
kind: Service
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: transcode-worker
    pool: "{name}"
spec:
  selector:
    app: transcode-worker
    pool: "{name}"
  ports:
  - name: http
    port: 8080
    targetPort: http
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: metrics
    protocol: TCP
  type: ClusterIP"#,
        name = pool_name,
        namespace = namespace,
    )
}

/// Generate a Prometheus ServiceMonitor YAML.
pub fn generate_service_monitor(pool_name: &str, namespace: &str) -> String {
    format!(
        r#"apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {name}-monitor
  namespace: {namespace}
  labels:
    app: transcode-worker
spec:
  selector:
    matchLabels:
      app: transcode-worker
      pool: "{name}"
  endpoints:
  - port: metrics
    path: /metrics
    interval: 15s"#,
        name = pool_name,
        namespace = namespace,
    )
}

/// Generate Helm values YAML for a worker pool deployment.
pub fn generate_helm_values(
    pool: &TranscodeWorkerPool,
    image: &ImageConfig,
    resources: &ResourceSpec,
) -> String {
    let gpu_section = if let Some(gpu_type) = &resources.gpu_type {
        format!(
            r#"gpu:
  enabled: true
  type: "{:?}"
  count: {}"#,
            gpu_type, resources.gpu_count
        )
    } else {
        "gpu:\n  enabled: false".into()
    };

    let scaling_section = if let Some(policy) = &pool.spec.scaling {
        format!(
            r#"autoscaling:
  enabled: true
  minReplicas: {}
  maxReplicas: {}
  targetQueueDepth: {}
  cooldownSeconds: {}"#,
            policy.min_replicas,
            policy.max_replicas,
            policy.target_queue_depth_per_worker,
            policy.cooldown_secs
        )
    } else {
        "autoscaling:\n  enabled: false".into()
    };

    format!(
        r#"# Transcode Worker Pool Helm Values
# Pool: {name}

replicaCount: {replicas}

image:
  repository: "{repo}"
  tag: "{tag}"
  pullPolicy: "{pull_policy}"

resources:
  requests:
    cpu: "{cpu_req}"
    memory: "{mem_req}"
  limits:
    cpu: "{cpu_lim}"
    memory: "{mem_lim}"

{gpu}

{scaling}

service:
  type: ClusterIP
  httpPort: 8080
  metricsPort: 9090

probes:
  liveness:
    path: /healthz
    initialDelaySeconds: 10
    periodSeconds: 15
  readiness:
    path: /readyz
    initialDelaySeconds: 5
    periodSeconds: 10

metrics:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 15s"#,
        name = pool.metadata.name,
        replicas = pool.spec.replicas,
        repo = image.repository,
        tag = image.tag,
        pull_policy = image.pull_policy,
        cpu_req = resources.cpu_request,
        mem_req = resources.memory_request,
        cpu_lim = resources.cpu_limit,
        mem_lim = resources.memory_limit,
        gpu = gpu_section,
        scaling = scaling_section,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crd::*;

    fn test_pool() -> TranscodeWorkerPool {
        TranscodeWorkerPool::new(
            "gpu-pool",
            "production",
            WorkerPoolSpec {
                replicas: 3,
                gpu_type: Some(GpuType::NvidiaT4),
                scaling: Some(ScalingPolicy {
                    min_replicas: 1,
                    max_replicas: 10,
                    ..Default::default()
                }),
                tolerations: vec![Toleration {
                    key: "nvidia.com/gpu".into(),
                    operator: "Exists".into(),
                    value: None,
                    effect: "NoSchedule".into(),
                }],
                node_selector: HashMap::from([("gpu-node".into(), "true".into())]),
                ..Default::default()
            },
        )
    }

    #[test]
    fn test_generate_deployment() {
        let pool = test_pool();
        let image = ImageConfig::default();
        let resources = ResourceSpec {
            gpu_count: 1,
            gpu_type: Some(GpuType::NvidiaT4),
            ..Default::default()
        };
        let yaml = generate_worker_deployment(
            &pool,
            &image,
            &resources,
            &ProbeConfig::default(),
            &ProbeConfig {
                path: "/readyz".into(),
                ..Default::default()
            },
        );

        assert!(yaml.contains("kind: Deployment"));
        assert!(yaml.contains("namespace: production"));
        assert!(yaml.contains("replicas: 3"));
        assert!(yaml.contains("nvidia.com/gpu"));
        assert!(yaml.contains("gpu-node"));
        assert!(yaml.contains("/healthz"));
    }

    #[test]
    fn test_generate_hpa() {
        let policy = ScalingPolicy {
            min_replicas: 2,
            max_replicas: 20,
            ..Default::default()
        };
        let yaml = generate_hpa("gpu-pool", "production", &policy);
        assert!(yaml.contains("HorizontalPodAutoscaler"));
        assert!(yaml.contains("minReplicas: 2"));
        assert!(yaml.contains("maxReplicas: 20"));
    }

    #[test]
    fn test_generate_service() {
        let yaml = generate_service("gpu-pool", "production");
        assert!(yaml.contains("kind: Service"));
        assert!(yaml.contains("port: 8080"));
        assert!(yaml.contains("port: 9090"));
    }

    #[test]
    fn test_generate_service_monitor() {
        let yaml = generate_service_monitor("gpu-pool", "production");
        assert!(yaml.contains("ServiceMonitor"));
        assert!(yaml.contains("interval: 15s"));
    }

    #[test]
    fn test_generate_helm_values() {
        let pool = test_pool();
        let image = ImageConfig::default();
        let resources = ResourceSpec {
            gpu_count: 1,
            gpu_type: Some(GpuType::NvidiaT4),
            ..Default::default()
        };
        let yaml = generate_helm_values(&pool, &image, &resources);
        assert!(yaml.contains("replicaCount: 3"));
        assert!(yaml.contains("gpu:"));
        assert!(yaml.contains("enabled: true"));
        assert!(yaml.contains("autoscaling:"));
    }

    #[test]
    fn test_cpu_only_deployment() {
        let pool = TranscodeWorkerPool::new("cpu-pool", "default", WorkerPoolSpec::default());
        let yaml = generate_worker_deployment(
            &pool,
            &ImageConfig::default(),
            &ResourceSpec::default(),
            &ProbeConfig::default(),
            &ProbeConfig::default(),
        );
        assert!(yaml.contains("kind: Deployment"));
        assert!(!yaml.contains("nvidia.com/gpu"));
    }
}
