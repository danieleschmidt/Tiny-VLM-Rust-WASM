//! Monitoring and metrics collection for production deployments
//! 
//! This module provides comprehensive monitoring capabilities including:
//! - Performance metrics collection
//! - Real-time health monitoring  
//! - Resource usage tracking
//! - Error rate analysis
//! - Custom metric collection

use crate::{Result, TinyVlmError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "std")]
use std::fs::OpenOptions;
#[cfg(feature = "std")]
use std::io::Write;

/// Comprehensive monitoring system for production deployment
pub struct MonitoringSystem {
    /// Performance metrics storage
    metrics: Arc<Mutex<MetricsStorage>>,
    /// System health indicators
    health_checks: Vec<Box<dyn HealthCheck + Send + Sync>>,
    /// Alert thresholds and configurations
    alerts: AlertManager,
    /// Metrics export configuration
    export_config: ExportConfig,
}

impl MonitoringSystem {
    /// Create a new monitoring system with production defaults
    pub fn new() -> Self {
        let mut system = Self {
            metrics: Arc::new(Mutex::new(MetricsStorage::new())),
            health_checks: Vec::new(),
            alerts: AlertManager::with_defaults(),
            export_config: ExportConfig::production(),
        };

        // Register default health checks
        system.register_health_check(Box::new(MemoryHealthCheck::new()));
        system.register_health_check(Box::new(PerformanceHealthCheck::new()));
        system.register_health_check(Box::new(ErrorRateHealthCheck::new()));

        system
    }

    /// Record an inference operation with detailed metrics
    pub fn record_inference(&self, duration_ms: f64, memory_mb: f64, tokens_in: usize, tokens_out: usize) {
        if let Ok(mut metrics) = self.metrics.lock() {
            let timestamp = current_timestamp();
            
            // Record performance metrics
            metrics.record_histogram("inference_duration_ms", duration_ms, timestamp);
            metrics.record_gauge("memory_usage_mb", memory_mb, timestamp);
            metrics.record_counter("inference_total", 1.0, timestamp);
            metrics.record_histogram("tokens_input", tokens_in as f64, timestamp);
            metrics.record_histogram("tokens_output", tokens_out as f64, timestamp);
            
            // Calculate derived metrics
            let tokens_per_second = (tokens_out as f64) / (duration_ms / 1000.0);
            metrics.record_gauge("tokens_per_second", tokens_per_second, timestamp);
            
            // Check performance thresholds
            if duration_ms > 200.0 {
                metrics.record_counter("slow_inference_count", 1.0, timestamp);
                self.alerts.trigger_alert(Alert::new(
                    AlertSeverity::Warning,
                    "Slow inference detected",
                    format!("Inference took {:.2}ms (threshold: 200ms)", duration_ms)
                ));
            }

            if memory_mb > 150.0 {
                self.alerts.trigger_alert(Alert::new(
                    AlertSeverity::Warning,
                    "High memory usage",
                    format!("Memory usage: {:.2}MB (threshold: 150MB)", memory_mb)
                ));
            }
        }
    }

    /// Record an error with classification
    pub fn record_error(&self, error: &TinyVlmError, context: &str) {
        if let Ok(mut metrics) = self.metrics.lock() {
            let timestamp = current_timestamp();
            
            metrics.record_counter("errors_total", 1.0, timestamp);
            
            let error_type = match error {
                TinyVlmError::Config(_) => "config",
                TinyVlmError::ImageProcessing(_) => "image_processing",
                TinyVlmError::TextProcessing(_) => "text_processing",
                TinyVlmError::Inference(_) => "inference",
                TinyVlmError::Memory(_) => "memory",
                TinyVlmError::Simd(_) => "simd",
                TinyVlmError::ModelLoading(_) => "model_loading",
                TinyVlmError::InvalidInput(_) => "invalid_input",
                TinyVlmError::InternalError(_) => "internal",
                TinyVlmError::ValidationError(_) => "validation",
                TinyVlmError::CircuitBreakerOpen(_) => "circuit_breaker",
                TinyVlmError::SecurityError(_) => "security",
                TinyVlmError::NetworkError(_) => "network",
                TinyVlmError::ConfigurationError(_) => "configuration",
                #[cfg(feature = "gpu")]
                TinyVlmError::Gpu(_) => "gpu",
                #[cfg(feature = "wasm")]
                TinyVlmError::Wasm(_) => "wasm",
                #[cfg(feature = "std")]
                TinyVlmError::Io(_) => "io",
                #[cfg(feature = "std")]
                TinyVlmError::SerializationError(_) => "serialization",
            };
            
            metrics.record_counter(&format!("errors_{}", error_type), 1.0, timestamp);
            
            // Trigger alert for critical errors
            let severity = match error {
                TinyVlmError::Memory(_) => AlertSeverity::Critical,
                TinyVlmError::Simd(_) => AlertSeverity::Critical,
                _ => AlertSeverity::Warning,
            };
            
            self.alerts.trigger_alert(Alert::new(
                severity,
                &format!("{} Error", error_type),
                format!("{}: {} (context: {})", error_type, error, context)
            ));
        }
    }

    /// Register a custom health check
    pub fn register_health_check(&mut self, check: Box<dyn HealthCheck + Send + Sync>) {
        self.health_checks.push(check);
    }

    /// Run all health checks and return system status
    pub fn health_status(&self) -> HealthReport {
        let mut report = HealthReport::new();
        
        // Run all registered health checks
        for check in &self.health_checks {
            let check_result = check.check(&self.metrics);
            report.add_check(check_result);
        }
        
        // Determine overall system health
        report.finalize();
        report
    }

    /// Export metrics in Prometheus format
    pub fn export_prometheus(&self) -> String {
        if let Ok(metrics) = self.metrics.lock() {
            metrics.export_prometheus()
        } else {
            String::new()
        }
    }

    /// Get current metrics snapshot
    pub fn get_metrics_snapshot(&self) -> HashMap<String, MetricSnapshot> {
        if let Ok(metrics) = self.metrics.lock() {
            metrics.get_snapshot()
        } else {
            HashMap::new()
        }
    }

    /// Export metrics to file (for production logging)
    #[cfg(feature = "std")]
    pub fn export_to_file(&self, file_path: &str) -> Result<()> {
        let prometheus_data = self.export_prometheus();
        
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)
            .map_err(|e| TinyVlmError::inference(&format!("Failed to open metrics file: {}", e)))?;
            
        writeln!(file, "# Timestamp: {}", current_timestamp())
            .map_err(|e| TinyVlmError::inference(&format!("Failed to write timestamp: {}", e)))?;
        writeln!(file, "{}", prometheus_data)
            .map_err(|e| TinyVlmError::inference(&format!("Failed to write metrics: {}", e)))?;
        
        file.flush()
            .map_err(|e| TinyVlmError::inference(&format!("Failed to flush metrics file: {}", e)))?;
            
        Ok(())
    }
}

/// Internal metrics storage with thread-safe operations
struct MetricsStorage {
    counters: HashMap<String, f64>,
    gauges: HashMap<String, f64>,
    histograms: HashMap<String, Vec<f64>>,
    timestamps: HashMap<String, u64>,
}

impl MetricsStorage {
    fn new() -> Self {
        Self {
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            timestamps: HashMap::new(),
        }
    }

    fn record_counter(&mut self, name: &str, value: f64, timestamp: u64) {
        *self.counters.entry(name.to_string()).or_insert(0.0) += value;
        self.timestamps.insert(name.to_string(), timestamp);
    }

    fn record_gauge(&mut self, name: &str, value: f64, timestamp: u64) {
        self.gauges.insert(name.to_string(), value);
        self.timestamps.insert(name.to_string(), timestamp);
    }

    fn record_histogram(&mut self, name: &str, value: f64, timestamp: u64) {
        self.histograms.entry(name.to_string()).or_insert_with(Vec::new).push(value);
        
        // Keep only recent values (last 1000 samples)
        if let Some(hist) = self.histograms.get_mut(name) {
            if hist.len() > 1000 {
                hist.drain(0..500); // Remove oldest half
            }
        }
        
        self.timestamps.insert(name.to_string(), timestamp);
    }

    fn export_prometheus(&self) -> String {
        let mut output = String::new();
        
        // Export counters
        for (name, value) in &self.counters {
            output.push_str(&format!("# TYPE {} counter\n", name));
            output.push_str(&format!("{} {}\n", name, value));
        }
        
        // Export gauges
        for (name, value) in &self.gauges {
            output.push_str(&format!("# TYPE {} gauge\n", name));
            output.push_str(&format!("{} {}\n", name, value));
        }
        
        // Export histogram summaries
        for (name, values) in &self.histograms {
            if values.is_empty() { continue; }
            
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let sum: f64 = sorted.iter().sum();
            let count = sorted.len();
            let avg = sum / count as f64;
            let p50 = percentile(&sorted, 0.5);
            let p90 = percentile(&sorted, 0.9);
            let p99 = percentile(&sorted, 0.99);
            
            output.push_str(&format!("# TYPE {}_summary summary\n", name));
            output.push_str(&format!("{}_summary_sum {}\n", name, sum));
            output.push_str(&format!("{}_summary_count {}\n", name, count));
            output.push_str(&format!("{}_summary_avg {}\n", name, avg));
            output.push_str(&format!("{}_summary{{quantile=\"0.5\"}} {}\n", name, p50));
            output.push_str(&format!("{}_summary{{quantile=\"0.9\"}} {}\n", name, p90));
            output.push_str(&format!("{}_summary{{quantile=\"0.99\"}} {}\n", name, p99));
        }
        
        output
    }

    fn get_snapshot(&self) -> HashMap<String, MetricSnapshot> {
        let mut snapshot = HashMap::new();
        
        for (name, value) in &self.counters {
            snapshot.insert(name.clone(), MetricSnapshot::Counter(*value));
        }
        
        for (name, value) in &self.gauges {
            snapshot.insert(name.clone(), MetricSnapshot::Gauge(*value));
        }
        
        for (name, values) in &self.histograms {
            if !values.is_empty() {
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                snapshot.insert(name.clone(), MetricSnapshot::Histogram(HistogramSnapshot {
                    count: sorted.len(),
                    sum: sorted.iter().sum(),
                    avg: sorted.iter().sum::<f64>() / sorted.len() as f64,
                    min: sorted[0],
                    max: sorted[sorted.len() - 1],
                    p50: percentile(&sorted, 0.5),
                    p90: percentile(&sorted, 0.9),
                    p99: percentile(&sorted, 0.99),
                }));
            }
        }
        
        snapshot
    }
}

/// Metric snapshot for external consumption
#[derive(Debug, Clone)]
pub enum MetricSnapshot {
    Counter(f64),
    Gauge(f64),
    Histogram(HistogramSnapshot),
}

#[derive(Debug, Clone)]
pub struct HistogramSnapshot {
    pub count: usize,
    pub sum: f64,
    pub avg: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
}

/// Health check trait for modular monitoring
pub trait HealthCheck {
    fn name(&self) -> &str;
    fn check(&self, metrics: &Arc<Mutex<MetricsStorage>>) -> HealthCheckResult;
}

/// Result of a health check
pub struct HealthCheckResult {
    pub name: String,
    pub status: HealthStatus,
    pub message: String,
    pub details: HashMap<String, String>,
}

/// Overall system health status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
}

/// Health report aggregating all checks
pub struct HealthReport {
    pub overall_status: HealthStatus,
    pub checks: Vec<HealthCheckResult>,
    pub timestamp: u64,
}

impl HealthReport {
    fn new() -> Self {
        Self {
            overall_status: HealthStatus::Healthy,
            checks: Vec::new(),
            timestamp: current_timestamp(),
        }
    }
    
    fn add_check(&mut self, check: HealthCheckResult) {
        // Update overall status to worst individual status
        match (&self.overall_status, check.status) {
            (_, HealthStatus::Critical) => self.overall_status = HealthStatus::Critical,
            (HealthStatus::Healthy, HealthStatus::Warning) => self.overall_status = HealthStatus::Warning,
            _ => {}
        }
        
        self.checks.push(check);
    }
    
    fn finalize(&mut self) {
        // Additional logic could be added here for complex health determination
    }
}

/// Memory usage health check
struct MemoryHealthCheck {
    warning_threshold_mb: f64,
    critical_threshold_mb: f64,
}

impl MemoryHealthCheck {
    fn new() -> Self {
        Self {
            warning_threshold_mb: 150.0,
            critical_threshold_mb: 250.0,
        }
    }
}

impl HealthCheck for MemoryHealthCheck {
    fn name(&self) -> &str { "memory_usage" }
    
    fn check(&self, metrics: &Arc<Mutex<MetricsStorage>>) -> HealthCheckResult {
        let mut result = HealthCheckResult {
            name: self.name().to_string(),
            status: HealthStatus::Healthy,
            message: "Memory usage normal".to_string(),
            details: HashMap::new(),
        };
        
        if let Ok(metrics) = metrics.lock() {
            if let Some(&memory_mb) = metrics.gauges.get("memory_usage_mb") {
                result.details.insert("current_memory_mb".to_string(), memory_mb.to_string());
                
                if memory_mb > self.critical_threshold_mb {
                    result.status = HealthStatus::Critical;
                    result.message = format!("Critical memory usage: {:.2}MB", memory_mb);
                } else if memory_mb > self.warning_threshold_mb {
                    result.status = HealthStatus::Warning;
                    result.message = format!("High memory usage: {:.2}MB", memory_mb);
                }
            }
        }
        
        result
    }
}

/// Performance health check
struct PerformanceHealthCheck {
    warning_threshold_ms: f64,
    critical_threshold_ms: f64,
}

impl PerformanceHealthCheck {
    fn new() -> Self {
        Self {
            warning_threshold_ms: 200.0,
            critical_threshold_ms: 500.0,
        }
    }
}

impl HealthCheck for PerformanceHealthCheck {
    fn name(&self) -> &str { "inference_performance" }
    
    fn check(&self, metrics: &Arc<Mutex<MetricsStorage>>) -> HealthCheckResult {
        let mut result = HealthCheckResult {
            name: self.name().to_string(),
            status: HealthStatus::Healthy,
            message: "Performance normal".to_string(),
            details: HashMap::new(),
        };
        
        if let Ok(metrics) = metrics.lock() {
            if let Some(durations) = metrics.histograms.get("inference_duration_ms") {
                if !durations.is_empty() {
                    let mut sorted = durations.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p90 = percentile(&sorted, 0.9);
                    
                    result.details.insert("p90_inference_ms".to_string(), p90.to_string());
                    
                    if p90 > self.critical_threshold_ms {
                        result.status = HealthStatus::Critical;
                        result.message = format!("Critical performance degradation: P90 {:.2}ms", p90);
                    } else if p90 > self.warning_threshold_ms {
                        result.status = HealthStatus::Warning;
                        result.message = format!("Performance degradation: P90 {:.2}ms", p90);
                    }
                }
            }
        }
        
        result
    }
}

/// Error rate health check
struct ErrorRateHealthCheck {
    warning_threshold_percent: f64,
    critical_threshold_percent: f64,
}

impl ErrorRateHealthCheck {
    fn new() -> Self {
        Self {
            warning_threshold_percent: 1.0,  // 1% error rate
            critical_threshold_percent: 5.0, // 5% error rate
        }
    }
}

impl HealthCheck for ErrorRateHealthCheck {
    fn name(&self) -> &str { "error_rate" }
    
    fn check(&self, metrics: &Arc<Mutex<MetricsStorage>>) -> HealthCheckResult {
        let mut result = HealthCheckResult {
            name: self.name().to_string(),
            status: HealthStatus::Healthy,
            message: "Error rate normal".to_string(),
            details: HashMap::new(),
        };
        
        if let Ok(metrics) = metrics.lock() {
            let total_inferences = metrics.counters.get("inference_total").unwrap_or(&0.0);
            let total_errors = metrics.counters.get("errors_total").unwrap_or(&0.0);
            
            if *total_inferences > 0.0 {
                let error_rate = (*total_errors / *total_inferences) * 100.0;
                result.details.insert("error_rate_percent".to_string(), error_rate.to_string());
                
                if error_rate > self.critical_threshold_percent {
                    result.status = HealthStatus::Critical;
                    result.message = format!("Critical error rate: {:.2}%", error_rate);
                } else if error_rate > self.warning_threshold_percent {
                    result.status = HealthStatus::Warning;
                    result.message = format!("High error rate: {:.2}%", error_rate);
                }
            }
        }
        
        result
    }
}

/// Alert management system
pub struct AlertManager {
    active_alerts: Arc<Mutex<Vec<Alert>>>,
    alert_history: Arc<Mutex<Vec<Alert>>>,
    max_active_alerts: usize,
    max_history_size: usize,
}

impl AlertManager {
    fn with_defaults() -> Self {
        Self {
            active_alerts: Arc::new(Mutex::new(Vec::new())),
            alert_history: Arc::new(Mutex::new(Vec::new())),
            max_active_alerts: 100,
            max_history_size: 1000,
        }
    }
    
    fn trigger_alert(&self, alert: Alert) {
        if let Ok(mut active_alerts) = self.active_alerts.lock() {
            // Prevent duplicate alerts
            if !active_alerts.iter().any(|a| a.id == alert.id) {
                active_alerts.push(alert.clone());
                
                // Limit active alerts
                if active_alerts.len() > self.max_active_alerts {
                    active_alerts.drain(0..50); // Remove oldest 50
                }
                
                #[cfg(feature = "std")]
                eprintln!("🚨 ALERT [{}]: {} - {}", 
                    alert.severity, alert.title, alert.message);
            }
        }
        
        // Add to history
        if let Ok(mut history) = self.alert_history.lock() {
            history.push(alert);
            
            if history.len() > self.max_history_size {
                history.drain(0..500); // Remove oldest 500
            }
        }
    }
}

/// Alert representation
#[derive(Clone, Debug)]
pub struct Alert {
    pub id: String,
    pub severity: AlertSeverity,
    pub title: String,
    pub message: String,
    pub timestamp: u64,
}

impl Alert {
    fn new(severity: AlertSeverity, title: &str, message: String) -> Self {
        let id = format!("{}_{}_{}", title.replace(" ", "_").to_lowercase(), 
                         severity, current_timestamp());
        
        Self {
            id,
            severity,
            title: title.to_string(),
            message,
            timestamp: current_timestamp(),
        }
    }
}

/// Alert severity levels
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Export configuration for metrics
struct ExportConfig {
    prometheus_enabled: bool,
    file_export_enabled: bool,
    export_interval_seconds: u64,
}

impl ExportConfig {
    fn production() -> Self {
        Self {
            prometheus_enabled: true,
            file_export_enabled: true,
            export_interval_seconds: 60,
        }
    }
}

/// Calculate percentile from sorted data
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    
    let index = (p * (sorted_data.len() - 1) as f64) as usize;
    sorted_data.get(index).copied().unwrap_or(sorted_data[sorted_data.len() - 1])
}

/// Get current timestamp in seconds since epoch
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_system() {
        let system = MonitoringSystem::new();
        
        // Record some test metrics
        system.record_inference(150.0, 95.0, 20, 15);
        system.record_inference(180.0, 100.0, 25, 18);
        
        // Check health status
        let health = system.health_status();
        assert_eq!(health.overall_status, HealthStatus::Healthy);
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 0.5), 3.0);
        assert_eq!(percentile(&data, 0.9), 4.0); // 90th percentile of [1,2,3,4,5] is 4.0
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(AlertSeverity::Warning, "Test Alert", "Test message".to_string());
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.title, "Test Alert");
    }
}