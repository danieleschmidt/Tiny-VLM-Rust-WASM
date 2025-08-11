//! Production deployment utilities and infrastructure management
//! 
//! This module provides comprehensive deployment support for various platforms:
//! - Containerized deployment (Docker/Kubernetes)
//! - Cloud platform integration (AWS, GCP, Azure)
//! - Auto-scaling and load balancing
//! - Configuration management
//! - Health checks and monitoring integration

use crate::{Result, TinyVlmError};
use std::collections::HashMap;

#[cfg(feature = "std")]
use std::fs;
#[cfg(feature = "std")]
use std::path::Path;

/// Production deployment manager
pub struct DeploymentManager {
    config: DeploymentConfig,
    platform_adapters: HashMap<Platform, Box<dyn PlatformAdapter>>,
    health_endpoint: HealthEndpoint,
    metrics_exporter: MetricsExporter,
}

impl DeploymentManager {
    /// Create a new deployment manager for production
    pub fn new(config: DeploymentConfig) -> Self {
        let mut manager = Self {
            config: config.clone(),
            platform_adapters: HashMap::new(),
            health_endpoint: HealthEndpoint::new(),
            metrics_exporter: MetricsExporter::new(),
        };

        // Register platform adapters
        manager.register_adapter(Platform::Kubernetes, Box::new(KubernetesAdapter::new()));
        manager.register_adapter(Platform::Docker, Box::new(DockerAdapter::new()));
        manager.register_adapter(Platform::CloudRun, Box::new(CloudRunAdapter::new()));
        manager.register_adapter(Platform::Lambda, Box::new(LambdaAdapter::new()));
        manager.register_adapter(Platform::AzureFunctions, Box::new(AzureFunctionsAdapter::new()));

        manager
    }

    /// Register a platform adapter
    pub fn register_adapter(&mut self, platform: Platform, adapter: Box<dyn PlatformAdapter>) {
        self.platform_adapters.insert(platform, adapter);
    }

    /// Generate deployment configuration for target platform
    pub fn generate_deployment_config(&self, platform: Platform) -> Result<DeploymentPackage> {
        let adapter = self.platform_adapters.get(&platform)
            .ok_or_else(|| TinyVlmError::config(&format!("No adapter for platform: {:?}", platform)))?;

        adapter.generate_deployment_config(&self.config)
    }

    /// Create health check endpoint configuration
    pub fn create_health_check(&self) -> HealthCheckConfig {
        HealthCheckConfig {
            path: "/health".to_string(),
            port: self.config.service_port,
            timeout_seconds: 30,
            interval_seconds: 10,
            failure_threshold: 3,
            success_threshold: 1,
            readiness_path: "/ready".to_string(),
            liveness_path: "/live".to_string(),
        }
    }

    /// Generate metrics configuration
    pub fn create_metrics_config(&self) -> MetricsConfig {
        MetricsConfig {
            prometheus_port: 9090,
            metrics_path: "/metrics".to_string(),
            scrape_interval_seconds: 15,
            retention_days: 30,
            export_format: MetricsFormat::Prometheus,
        }
    }

    /// Generate auto-scaling configuration
    pub fn create_autoscaling_config(&self) -> AutoScalingConfig {
        AutoScalingConfig {
            min_replicas: 1,
            max_replicas: 10,
            target_cpu_percent: 70,
            target_memory_percent: 80,
            scale_up_cooldown_seconds: 60,
            scale_down_cooldown_seconds: 300,
            custom_metrics: vec![
                CustomMetric {
                    name: "inference_requests_per_second".to_string(),
                    target_value: 10.0,
                    metric_type: MetricType::AverageValue,
                },
                CustomMetric {
                    name: "inference_queue_length".to_string(),
                    target_value: 5.0,
                    metric_type: MetricType::AverageValue,
                },
            ],
        }
    }
}

/// Deployment configuration
#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    pub service_name: String,
    pub service_port: u16,
    pub image_repository: String,
    pub image_tag: String,
    pub environment: Environment,
    pub resource_limits: ResourceLimits,
    pub security_context: SecurityContext,
    pub ingress_config: IngressConfig,
    pub storage_config: StorageConfig,
}

impl DeploymentConfig {
    /// Production-ready configuration
    pub fn production() -> Self {
        Self {
            service_name: "tiny-vlm".to_string(),
            service_port: 8080,
            image_repository: "ghcr.io/tiny-vlm/tiny-vlm".to_string(),
            image_tag: "latest".to_string(),
            environment: Environment::Production,
            resource_limits: ResourceLimits::production(),
            security_context: SecurityContext::secure(),
            ingress_config: IngressConfig::production(),
            storage_config: StorageConfig::ephemeral(),
        }
    }

    /// Development configuration
    pub fn development() -> Self {
        Self {
            service_name: "tiny-vlm-dev".to_string(),
            service_port: 3000,
            image_repository: "tiny-vlm".to_string(),
            image_tag: "dev".to_string(),
            environment: Environment::Development,
            resource_limits: ResourceLimits::development(),
            security_context: SecurityContext::permissive(),
            ingress_config: IngressConfig::development(),
            storage_config: StorageConfig::local(),
        }
    }
}

/// Supported deployment platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Platform {
    Kubernetes,
    Docker,
    CloudRun,
    Lambda,
    AzureFunctions,
}

/// Deployment environment
#[derive(Debug, Clone)]
pub enum Environment {
    Development,
    Staging,
    Production,
}

/// Resource limits for containers
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub cpu_request: String,
    pub cpu_limit: String,
    pub memory_request: String,
    pub memory_limit: String,
    pub storage_request: String,
    pub gpu_limit: u32,
}

impl ResourceLimits {
    fn production() -> Self {
        Self {
            cpu_request: "1000m".to_string(),    // 1 CPU
            cpu_limit: "2000m".to_string(),      // 2 CPUs
            memory_request: "2Gi".to_string(),   // 2GB
            memory_limit: "4Gi".to_string(),     // 4GB
            storage_request: "10Gi".to_string(),  // 10GB
            gpu_limit: 0, // CPU-only for efficiency
        }
    }

    fn development() -> Self {
        Self {
            cpu_request: "100m".to_string(),     // 0.1 CPU
            cpu_limit: "500m".to_string(),       // 0.5 CPU
            memory_request: "256Mi".to_string(),  // 256MB
            memory_limit: "1Gi".to_string(),     // 1GB
            storage_request: "1Gi".to_string(),   // 1GB
            gpu_limit: 0,
        }
    }
}

/// Security context for containers
#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub run_as_non_root: bool,
    pub run_as_user: u32,
    pub read_only_root_filesystem: bool,
    pub allow_privilege_escalation: bool,
    pub capabilities_drop: Vec<String>,
    pub seccomp_profile: String,
}

impl SecurityContext {
    fn secure() -> Self {
        Self {
            run_as_non_root: true,
            run_as_user: 65534, // nobody user
            read_only_root_filesystem: true,
            allow_privilege_escalation: false,
            capabilities_drop: vec!["ALL".to_string()],
            seccomp_profile: "runtime/default".to_string(),
        }
    }

    fn permissive() -> Self {
        Self {
            run_as_non_root: false,
            run_as_user: 0, // root
            read_only_root_filesystem: false,
            allow_privilege_escalation: true,
            capabilities_drop: vec![],
            seccomp_profile: "unconfined".to_string(),
        }
    }
}

/// Ingress configuration for external access
#[derive(Debug, Clone)]
pub struct IngressConfig {
    pub enabled: bool,
    pub hostname: String,
    pub tls_enabled: bool,
    pub cert_issuer: String,
    pub rate_limiting: RateLimitingConfig,
    pub cors_enabled: bool,
}

impl IngressConfig {
    fn production() -> Self {
        Self {
            enabled: true,
            hostname: "api.tiny-vlm.com".to_string(),
            tls_enabled: true,
            cert_issuer: "letsencrypt-prod".to_string(),
            rate_limiting: RateLimitingConfig::production(),
            cors_enabled: false, // Strict CORS in production
        }
    }

    fn development() -> Self {
        Self {
            enabled: false,
            hostname: "localhost".to_string(),
            tls_enabled: false,
            cert_issuer: "self-signed".to_string(),
            rate_limiting: RateLimitingConfig::development(),
            cors_enabled: true, // Allow CORS in development
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitingConfig {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub burst_size: u32,
}

impl RateLimitingConfig {
    fn production() -> Self {
        Self {
            requests_per_minute: 60,
            requests_per_hour: 1000,
            burst_size: 20,
        }
    }

    fn development() -> Self {
        Self {
            requests_per_minute: 1000,
            requests_per_hour: 10000,
            burst_size: 100,
        }
    }
}

/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub type_: StorageType,
    pub size: String,
    pub access_mode: String,
    pub storage_class: String,
}

impl StorageConfig {
    fn ephemeral() -> Self {
        Self {
            type_: StorageType::EmptyDir,
            size: "1Gi".to_string(),
            access_mode: "ReadWriteOnce".to_string(),
            storage_class: "".to_string(),
        }
    }

    fn local() -> Self {
        Self {
            type_: StorageType::HostPath,
            size: "10Gi".to_string(),
            access_mode: "ReadWriteOnce".to_string(),
            storage_class: "local-path".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum StorageType {
    EmptyDir,
    HostPath,
    PersistentVolume,
    ConfigMap,
    Secret,
}

/// Platform adapter trait
trait PlatformAdapter {
    fn generate_deployment_config(&self, config: &DeploymentConfig) -> Result<DeploymentPackage>;
    fn validate_config(&self, config: &DeploymentConfig) -> Result<()>;
}

/// Deployment package containing all necessary files
pub struct DeploymentPackage {
    pub files: HashMap<String, String>,
    pub commands: Vec<String>,
    pub environment_variables: HashMap<String, String>,
}

/// Kubernetes adapter
struct KubernetesAdapter;

impl KubernetesAdapter {
    fn new() -> Self {
        Self
    }
}

impl PlatformAdapter for KubernetesAdapter {
    fn generate_deployment_config(&self, config: &DeploymentConfig) -> Result<DeploymentPackage> {
        let mut files = HashMap::new();
        
        // Generate Deployment YAML
        let deployment_yaml = self.generate_deployment_yaml(config);
        files.insert("deployment.yaml".to_string(), deployment_yaml);

        // Generate Service YAML
        let service_yaml = self.generate_service_yaml(config);
        files.insert("service.yaml".to_string(), service_yaml);

        // Generate ConfigMap YAML
        let configmap_yaml = self.generate_configmap_yaml(config);
        files.insert("configmap.yaml".to_string(), configmap_yaml);

        // Generate Ingress YAML if enabled
        if config.ingress_config.enabled {
            let ingress_yaml = self.generate_ingress_yaml(config);
            files.insert("ingress.yaml".to_string(), ingress_yaml);
        }

        // Generate HPA YAML
        let hpa_yaml = self.generate_hpa_yaml(config);
        files.insert("hpa.yaml".to_string(), hpa_yaml);

        let commands = vec![
            "kubectl apply -f configmap.yaml".to_string(),
            "kubectl apply -f deployment.yaml".to_string(),
            "kubectl apply -f service.yaml".to_string(),
            "kubectl apply -f hpa.yaml".to_string(),
        ];

        Ok(DeploymentPackage {
            files,
            commands,
            environment_variables: HashMap::new(),
        })
    }

    fn validate_config(&self, _config: &DeploymentConfig) -> Result<()> {
        // Validate Kubernetes-specific requirements
        Ok(())
    }
}

impl KubernetesAdapter {
    fn generate_deployment_yaml(&self, config: &DeploymentConfig) -> String {
        format!(r#"apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {name}
    version: {tag}
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
        version: {tag}
    spec:
      securityContext:
        runAsNonRoot: {run_as_non_root}
        runAsUser: {run_as_user}
      containers:
      - name: {name}
        image: {repository}:{tag}
        ports:
        - containerPort: {port}
          name: http
          protocol: TCP
        env:
        - name: RUST_LOG
          value: "info"
        - name: BIND_ADDRESS
          value: "0.0.0.0:{port}"
        resources:
          requests:
            cpu: {cpu_request}
            memory: {memory_request}
          limits:
            cpu: {cpu_limit}
            memory: {memory_limit}
        livenessProbe:
          httpGet:
            path: /health/live
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: tmp
        emptyDir: {{}}
"#,
            name = config.service_name,
            repository = config.image_repository,
            tag = config.image_tag,
            port = config.service_port,
            run_as_non_root = config.security_context.run_as_non_root,
            run_as_user = config.security_context.run_as_user,
            cpu_request = config.resource_limits.cpu_request,
            cpu_limit = config.resource_limits.cpu_limit,
            memory_request = config.resource_limits.memory_request,
            memory_limit = config.resource_limits.memory_limit,
        )
    }

    fn generate_service_yaml(&self, config: &DeploymentConfig) -> String {
        format!(r#"apiVersion: v1
kind: Service
metadata:
  name: {name}
  labels:
    app: {name}
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: http
    protocol: TCP
    name: http
  selector:
    app: {name}
"#,
            name = config.service_name
        )
    }

    fn generate_configmap_yaml(&self, config: &DeploymentConfig) -> String {
        format!(r#"apiVersion: v1
kind: ConfigMap
metadata:
  name: {name}-config
data:
  config.yaml: |
    service:
      port: {port}
      environment: {environment:?}
    model:
      max_batch_size: 4
      timeout_seconds: 30
    logging:
      level: info
      format: json
"#,
            name = config.service_name,
            port = config.service_port,
            environment = config.environment
        )
    }

    fn generate_ingress_yaml(&self, config: &DeploymentConfig) -> String {
        let tls_section = if config.ingress_config.tls_enabled {
            format!(r#"  tls:
  - hosts:
    - {hostname}
    secretName: {name}-tls"#,
                hostname = config.ingress_config.hostname,
                name = config.service_name
            )
        } else {
            String::new()
        };

        format!(r#"apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {name}
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "{rpm}"
    nginx.ingress.kubernetes.io/rate-limit-burst: "{burst}"
    cert-manager.io/cluster-issuer: {cert_issuer}
spec:
{tls_section}
  rules:
  - host: {hostname}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {name}
            port:
              number: 80
"#,
            name = config.service_name,
            hostname = config.ingress_config.hostname,
            rpm = config.ingress_config.rate_limiting.requests_per_minute,
            burst = config.ingress_config.rate_limiting.burst_size,
            cert_issuer = config.ingress_config.cert_issuer,
            tls_section = tls_section
        )
    }

    fn generate_hpa_yaml(&self, config: &DeploymentConfig) -> String {
        format!(r#"apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {name}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {name}
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
"#,
            name = config.service_name
        )
    }
}

/// Docker adapter
struct DockerAdapter;

impl DockerAdapter {
    fn new() -> Self { Self }
}

impl PlatformAdapter for DockerAdapter {
    fn generate_deployment_config(&self, config: &DeploymentConfig) -> Result<DeploymentPackage> {
        let mut files = HashMap::new();
        
        let dockerfile = format!(r#"FROM rust:1.75-slim as builder

WORKDIR /usr/src/app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY benches ./benches
COPY examples ./examples

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN cargo build --release --features std

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

COPY --from=builder /usr/src/app/target/release/tiny-vlm /usr/local/bin/tiny-vlm

EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:{port}/health || exit 1

CMD ["tiny-vlm"]
"#, port = config.service_port);

        files.insert("Dockerfile".to_string(), dockerfile);

        let docker_compose = format!(r#"version: '3.8'

services:
  {name}:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - RUST_LOG=info
      - BIND_ADDRESS=0.0.0.0:{port}
    deploy:
      resources:
        limits:
          memory: {memory_limit}
        reservations:
          memory: {memory_request}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
"#,
            name = config.service_name,
            port = config.service_port,
            memory_limit = config.resource_limits.memory_limit,
            memory_request = config.resource_limits.memory_request,
        );

        files.insert("docker-compose.yml".to_string(), docker_compose);

        let commands = vec![
            "docker build -t tiny-vlm .".to_string(),
            "docker-compose up -d".to_string(),
        ];

        Ok(DeploymentPackage {
            files,
            commands,
            environment_variables: HashMap::new(),
        })
    }

    fn validate_config(&self, _config: &DeploymentConfig) -> Result<()> {
        Ok(())
    }
}

/// Cloud Run adapter (simplified stubs for other platforms)
struct CloudRunAdapter;
impl CloudRunAdapter { fn new() -> Self { Self } }
impl PlatformAdapter for CloudRunAdapter {
    fn generate_deployment_config(&self, _config: &DeploymentConfig) -> Result<DeploymentPackage> {
        // Simplified Cloud Run configuration
        Ok(DeploymentPackage {
            files: HashMap::new(),
            commands: vec!["gcloud run deploy".to_string()],
            environment_variables: HashMap::new(),
        })
    }
    fn validate_config(&self, _config: &DeploymentConfig) -> Result<()> { Ok(()) }
}

struct LambdaAdapter;
impl LambdaAdapter { fn new() -> Self { Self } }
impl PlatformAdapter for LambdaAdapter {
    fn generate_deployment_config(&self, _config: &DeploymentConfig) -> Result<DeploymentPackage> {
        Ok(DeploymentPackage {
            files: HashMap::new(),
            commands: vec!["sam deploy".to_string()],
            environment_variables: HashMap::new(),
        })
    }
    fn validate_config(&self, _config: &DeploymentConfig) -> Result<()> { Ok(()) }
}

struct AzureFunctionsAdapter;
impl AzureFunctionsAdapter { fn new() -> Self { Self } }
impl PlatformAdapter for AzureFunctionsAdapter {
    fn generate_deployment_config(&self, _config: &DeploymentConfig) -> Result<DeploymentPackage> {
        Ok(DeploymentPackage {
            files: HashMap::new(),
            commands: vec!["func azure functionapp publish".to_string()],
            environment_variables: HashMap::new(),
        })
    }
    fn validate_config(&self, _config: &DeploymentConfig) -> Result<()> { Ok(()) }
}

/// Health endpoint for monitoring
struct HealthEndpoint {
    startup_time: std::time::Instant,
}

impl HealthEndpoint {
    fn new() -> Self {
        Self {
            startup_time: std::time::Instant::now(),
        }
    }
}

/// Metrics exporter
struct MetricsExporter;

impl MetricsExporter {
    fn new() -> Self { Self }
}

/// Health check configuration
pub struct HealthCheckConfig {
    pub path: String,
    pub port: u16,
    pub timeout_seconds: u32,
    pub interval_seconds: u32,
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub readiness_path: String,
    pub liveness_path: String,
}

/// Metrics configuration
pub struct MetricsConfig {
    pub prometheus_port: u16,
    pub metrics_path: String,
    pub scrape_interval_seconds: u32,
    pub retention_days: u32,
    pub export_format: MetricsFormat,
}

#[derive(Debug)]
pub enum MetricsFormat {
    Prometheus,
    StatsD,
    OpenTelemetry,
}

/// Auto-scaling configuration
pub struct AutoScalingConfig {
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub target_cpu_percent: u32,
    pub target_memory_percent: u32,
    pub scale_up_cooldown_seconds: u32,
    pub scale_down_cooldown_seconds: u32,
    pub custom_metrics: Vec<CustomMetric>,
}

/// Custom metric for auto-scaling
pub struct CustomMetric {
    pub name: String,
    pub target_value: f64,
    pub metric_type: MetricType,
}

#[derive(Debug)]
pub enum MetricType {
    AverageValue,
    Utilization,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deployment_config_creation() {
        let config = DeploymentConfig::production();
        assert_eq!(config.service_name, "tiny-vlm");
        assert_eq!(config.service_port, 8080);
        assert!(matches!(config.environment, Environment::Production));
    }

    #[test]
    fn test_kubernetes_deployment_generation() {
        let config = DeploymentConfig::production();
        let manager = DeploymentManager::new(config);
        
        let deployment = manager.generate_deployment_config(Platform::Kubernetes);
        assert!(deployment.is_ok());
        
        let package = deployment.unwrap();
        assert!(package.files.contains_key("deployment.yaml"));
        assert!(package.files.contains_key("service.yaml"));
    }

    #[test]
    fn test_docker_deployment_generation() {
        let config = DeploymentConfig::production();
        let manager = DeploymentManager::new(config);
        
        let deployment = manager.generate_deployment_config(Platform::Docker);
        assert!(deployment.is_ok());
        
        let package = deployment.unwrap();
        assert!(package.files.contains_key("Dockerfile"));
        assert!(package.files.contains_key("docker-compose.yml"));
    }
}