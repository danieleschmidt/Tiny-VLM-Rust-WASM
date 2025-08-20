//! Advanced Monitoring and Observability Framework
//!
//! Comprehensive real-time monitoring, metrics collection, alerting, and distributed tracing
//! for production-grade observability.

use crate::{Result, TinyVlmError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Advanced monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedMonitoringConfig {
    pub enable_metrics_collection: bool,
    pub enable_distributed_tracing: bool,
    pub enable_alerting: bool,
    pub enable_health_checks: bool,
    pub metrics_retention_hours: u64,
    pub metrics_aggregation_interval_seconds: u64,
    pub alert_evaluation_interval_seconds: u64,
    pub health_check_interval_seconds: u64,
    pub enable_custom_metrics: bool,
    pub enable_performance_profiling: bool,
    pub max_trace_depth: usize,
    pub trace_sampling_rate: f64,
}

impl Default for AdvancedMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics_collection: true,
            enable_distributed_tracing: true,
            enable_alerting: true,
            enable_health_checks: true,
            metrics_retention_hours: 24,
            metrics_aggregation_interval_seconds: 60,
            alert_evaluation_interval_seconds: 30,
            health_check_interval_seconds: 10,
            enable_custom_metrics: true,
            enable_performance_profiling: true,
            max_trace_depth: 10,
            trace_sampling_rate: 0.1,
        }
    }
}

/// Metric types for different kinds of measurements
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Timer,
}

/// Metric value with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub value: f64,
    pub timestamp: u64,
    pub labels: HashMap<String, String>,
}

/// Metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub metric_type: MetricType,
    pub description: String,
    pub unit: String,
    pub values: Vec<MetricValue>,
}

impl Metric {
    pub fn new(name: String, metric_type: MetricType, description: String, unit: String) -> Self {
        Self {
            name,
            metric_type,
            description,
            unit,
            values: Vec::new(),
        }
    }

    pub fn record_value(&mut self, value: f64, labels: HashMap<String, String>) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        self.values.push(MetricValue {
            value,
            timestamp,
            labels,
        });
    }

    pub fn get_latest_value(&self) -> Option<f64> {
        self.values.last().map(|v| v.value)
    }

    pub fn get_aggregated_value(&self, window_seconds: u64) -> Option<f64> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let cutoff = now.saturating_sub(window_seconds);
        let recent_values: Vec<f64> = self.values
            .iter()
            .filter(|v| v.timestamp >= cutoff)
            .map(|v| v.value)
            .collect();

        if recent_values.is_empty() {
            return None;
        }

        match self.metric_type {
            MetricType::Counter => Some(recent_values.iter().sum()),
            MetricType::Gauge => recent_values.last().copied(),
            MetricType::Histogram | MetricType::Summary => {
                Some(recent_values.iter().sum::<f64>() / recent_values.len() as f64)
            }
            MetricType::Timer => {
                Some(recent_values.iter().sum::<f64>() / recent_values.len() as f64)
            }
        }
    }
}

/// Distributed tracing span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub duration_microseconds: Option<u64>,
    pub tags: HashMap<String, String>,
    pub logs: Vec<SpanLog>,
    pub status: SpanStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanStatus {
    Ok,
    Error(String),
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLog {
    pub timestamp: u64,
    pub level: String,
    pub message: String,
    pub fields: HashMap<String, String>,
}

impl Span {
    pub fn new(trace_id: String, span_id: String, operation_name: String) -> Self {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        Self {
            trace_id,
            span_id,
            parent_span_id: None,
            operation_name,
            start_time,
            end_time: None,
            duration_microseconds: None,
            tags: HashMap::new(),
            logs: Vec::new(),
            status: SpanStatus::Ok,
        }
    }

    pub fn with_parent(mut self, parent_span_id: String) -> Self {
        self.parent_span_id = Some(parent_span_id);
        self
    }

    pub fn add_tag(&mut self, key: String, value: String) {
        self.tags.insert(key, value);
    }

    pub fn add_log(&mut self, level: String, message: String, fields: HashMap<String, String>) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        self.logs.push(SpanLog {
            timestamp,
            level,
            message,
            fields,
        });
    }

    pub fn finish(&mut self) {
        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        self.end_time = Some(end_time);
        self.duration_microseconds = Some(end_time - self.start_time);
    }

    pub fn finish_with_error(&mut self, error: String) {
        self.finish();
        self.status = SpanStatus::Error(error);
    }
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub description: String,
    pub metric_name: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub severity: AlertSeverity,
    pub evaluation_window_seconds: u64,
    pub cooldown_seconds: u64,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

impl AlertCondition {
    pub fn evaluate(&self, value: f64, threshold: f64) -> bool {
        match self {
            AlertCondition::GreaterThan => value > threshold,
            AlertCondition::LessThan => value < threshold,
            AlertCondition::Equals => (value - threshold).abs() < f64::EPSILON,
            AlertCondition::NotEquals => (value - threshold).abs() > f64::EPSILON,
            AlertCondition::GreaterThanOrEqual => value >= threshold,
            AlertCondition::LessThanOrEqual => value <= threshold,
        }
    }
}

/// Fired alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub rule_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub current_value: f64,
    pub threshold: f64,
    pub fired_at: u64,
    pub resolved_at: Option<u64>,
    pub context: HashMap<String, String>,
}

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub message: String,
    pub response_time_ms: u64,
    pub last_checked: u64,
    pub metadata: HashMap<String, String>,
}

/// Advanced monitoring system
pub struct AdvancedMonitoringSystem {
    config: AdvancedMonitoringConfig,
    metrics: Arc<Mutex<HashMap<String, Metric>>>,
    traces: Arc<Mutex<HashMap<String, Vec<Span>>>>,
    alert_rules: Arc<Mutex<Vec<AlertRule>>>,
    active_alerts: Arc<Mutex<Vec<Alert>>>,
    health_checks: Arc<Mutex<HashMap<String, HealthCheck>>>,
    custom_metrics: Arc<Mutex<HashMap<String, f64>>>,
    performance_counters: PerformanceCounters,
}

impl AdvancedMonitoringSystem {
    pub fn new(config: AdvancedMonitoringConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(Mutex::new(HashMap::new())),
            traces: Arc::new(Mutex::new(HashMap::new())),
            alert_rules: Arc::new(Mutex::new(Vec::new())),
            active_alerts: Arc::new(Mutex::new(Vec::new())),
            health_checks: Arc::new(Mutex::new(HashMap::new())),
            custom_metrics: Arc::new(Mutex::new(HashMap::new())),
            performance_counters: PerformanceCounters::new(),
        }
    }

    /// Record a metric value
    pub fn record_metric(&self, name: &str, value: f64, labels: HashMap<String, String>) -> Result<()> {
        if !self.config.enable_metrics_collection {
            return Ok(());
        }

        let mut metrics = self.metrics.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire metrics lock".to_string())
        })?;

        if let Some(metric) = metrics.get_mut(name) {
            metric.record_value(value, labels);
        } else {
            let mut new_metric = Metric::new(
                name.to_string(),
                MetricType::Gauge,
                format!("Auto-generated metric for {}", name),
                "units".to_string(),
            );
            new_metric.record_value(value, labels);
            metrics.insert(name.to_string(), new_metric);
        }

        Ok(())
    }

    /// Create and register a metric
    pub fn register_metric(&self, metric: Metric) -> Result<()> {
        let mut metrics = self.metrics.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire metrics lock".to_string())
        })?;

        metrics.insert(metric.name.clone(), metric);
        Ok(())
    }

    /// Start a new trace span
    pub fn start_span(&self, trace_id: String, operation_name: String) -> Result<String> {
        if !self.config.enable_distributed_tracing {
            return Ok("disabled".to_string());
        }

        let span_id = format!("span_{}", self.generate_id());
        let span = Span::new(trace_id.clone(), span_id.clone(), operation_name);

        let mut traces = self.traces.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire traces lock".to_string())
        })?;

        traces.entry(trace_id).or_insert_with(Vec::new).push(span);

        Ok(span_id)
    }

    /// Finish a trace span
    pub fn finish_span(&self, trace_id: &str, span_id: &str) -> Result<()> {
        let mut traces = self.traces.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire traces lock".to_string())
        })?;

        if let Some(trace) = traces.get_mut(trace_id) {
            if let Some(span) = trace.iter_mut().find(|s| s.span_id == span_id) {
                span.finish();
            }
        }

        Ok(())
    }

    /// Add an alert rule
    pub fn add_alert_rule(&self, rule: AlertRule) -> Result<()> {
        if !self.config.enable_alerting {
            return Ok(());
        }

        let mut rules = self.alert_rules.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire alert rules lock".to_string())
        })?;

        rules.push(rule);
        Ok(())
    }

    /// Evaluate alert rules and fire alerts
    pub fn evaluate_alerts(&self) -> Result<Vec<Alert>> {
        if !self.config.enable_alerting {
            return Ok(Vec::new());
        }

        let mut new_alerts = Vec::new();
        let rules = self.alert_rules.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire alert rules lock".to_string())
        })?;

        let metrics = self.metrics.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire metrics lock".to_string())
        })?;

        for rule in rules.iter() {
            if !rule.enabled {
                continue;
            }

            if let Some(metric) = metrics.get(&rule.metric_name) {
                if let Some(current_value) = metric.get_aggregated_value(rule.evaluation_window_seconds) {
                    if rule.condition.evaluate(current_value, rule.threshold) {
                        let alert = Alert {
                            rule_name: rule.name.clone(),
                            severity: rule.severity.clone(),
                            message: format!(
                                "Alert: {} - Current value {} {} threshold {}",
                                rule.description,
                                current_value,
                                match rule.condition {
                                    AlertCondition::GreaterThan => ">",
                                    AlertCondition::LessThan => "<",
                                    AlertCondition::Equals => "==",
                                    AlertCondition::NotEquals => "!=",
                                    AlertCondition::GreaterThanOrEqual => ">=",
                                    AlertCondition::LessThanOrEqual => "<=",
                                },
                                rule.threshold
                            ),
                            current_value,
                            threshold: rule.threshold,
                            fired_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                            resolved_at: None,
                            context: HashMap::new(),
                        };
                        new_alerts.push(alert);
                    }
                }
            }
        }

        // Update active alerts
        let mut active_alerts = self.active_alerts.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire active alerts lock".to_string())
        })?;

        active_alerts.extend(new_alerts.clone());

        Ok(new_alerts)
    }

    /// Register a health check
    pub fn register_health_check(&self, name: String, check: HealthCheck) -> Result<()> {
        if !self.config.enable_health_checks {
            return Ok(());
        }

        let mut health_checks = self.health_checks.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire health checks lock".to_string())
        })?;

        health_checks.insert(name, check);
        Ok(())
    }

    /// Get current system metrics summary
    pub fn get_metrics_summary(&self) -> Result<MetricsSummary> {
        let metrics = self.metrics.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire metrics lock".to_string())
        })?;

        let mut summary = MetricsSummary {
            total_metrics: metrics.len(),
            metrics_overview: Vec::new(),
            system_performance: self.performance_counters.get_summary(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };

        for (name, metric) in metrics.iter() {
            if let Some(latest_value) = metric.get_latest_value() {
                summary.metrics_overview.push(MetricOverview {
                    name: name.clone(),
                    metric_type: metric.metric_type.clone(),
                    current_value: latest_value,
                    unit: metric.unit.clone(),
                });
            }
        }

        Ok(summary)
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Result<Vec<Alert>> {
        let active_alerts = self.active_alerts.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire active alerts lock".to_string())
        })?;

        Ok(active_alerts.clone())
    }

    /// Get health status overview
    pub fn get_health_overview(&self) -> Result<HealthOverview> {
        let health_checks = self.health_checks.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire health checks lock".to_string())
        })?;

        let total_checks = health_checks.len();
        let healthy_count = health_checks.values()
            .filter(|check| check.status == HealthStatus::Healthy)
            .count();
        let degraded_count = health_checks.values()
            .filter(|check| check.status == HealthStatus::Degraded)
            .count();
        let unhealthy_count = health_checks.values()
            .filter(|check| check.status == HealthStatus::Unhealthy)
            .count();

        let overall_status = if unhealthy_count > 0 {
            HealthStatus::Unhealthy
        } else if degraded_count > 0 {
            HealthStatus::Degraded
        } else if healthy_count > 0 {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unknown
        };

        Ok(HealthOverview {
            overall_status,
            total_checks,
            healthy_count,
            degraded_count,
            unhealthy_count,
            checks: health_checks.clone(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        })
    }

    /// Record custom metric
    pub fn record_custom_metric(&self, name: String, value: f64) -> Result<()> {
        if !self.config.enable_custom_metrics {
            return Ok(());
        }

        let mut custom_metrics = self.custom_metrics.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire custom metrics lock".to_string())
        })?;

        custom_metrics.insert(name, value);
        Ok(())
    }

    /// Increment performance counter
    pub fn increment_counter(&self, counter: &str) {
        self.performance_counters.increment(counter);
    }

    /// Record timing measurement
    pub fn record_timing(&self, operation: &str, duration: Duration) {
        self.performance_counters.record_timing(operation, duration);
    }

    fn generate_id(&self) -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
    }
}

/// Performance counters for system monitoring
#[derive(Debug)]
pub struct PerformanceCounters {
    counters: Arc<Mutex<HashMap<String, AtomicU64>>>,
    timings: Arc<Mutex<HashMap<String, Vec<u64>>>>,
}

impl PerformanceCounters {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(Mutex::new(HashMap::new())),
            timings: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn increment(&self, name: &str) {
        if let Ok(mut counters) = self.counters.lock() {
            counters.entry(name.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_timing(&self, operation: &str, duration: Duration) {
        if let Ok(mut timings) = self.timings.lock() {
            timings.entry(operation.to_string())
                .or_insert_with(Vec::new)
                .push(duration.as_millis() as u64);
        }
    }

    pub fn get_summary(&self) -> PerformanceSummary {
        let counters = self.counters.lock().unwrap_or_else(|_| {
            std::panic::panic_any("Failed to acquire counters lock")
        });
        let timings = self.timings.lock().unwrap_or_else(|_| {
            std::panic::panic_any("Failed to acquire timings lock")
        });

        let counter_summary: HashMap<String, u64> = counters.iter()
            .map(|(name, counter)| (name.clone(), counter.load(Ordering::Relaxed)))
            .collect();

        let timing_summary: HashMap<String, TimingSummary> = timings.iter()
            .map(|(name, times)| {
                let count = times.len();
                let sum: u64 = times.iter().sum();
                let average = if count > 0 { sum / count as u64 } else { 0 };
                let min = times.iter().min().copied().unwrap_or(0);
                let max = times.iter().max().copied().unwrap_or(0);

                (name.clone(), TimingSummary {
                    count,
                    average_ms: average,
                    min_ms: min,
                    max_ms: max,
                })
            })
            .collect();

        PerformanceSummary {
            counters: counter_summary,
            timings: timing_summary,
        }
    }
}

impl Default for PerformanceCounters {
    fn default() -> Self {
        Self::new()
    }
}

/// Data structures for reporting
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_metrics: usize,
    pub metrics_overview: Vec<MetricOverview>,
    pub system_performance: PerformanceSummary,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetricOverview {
    pub name: String,
    pub metric_type: MetricType,
    pub current_value: f64,
    pub unit: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub counters: HashMap<String, u64>,
    pub timings: HashMap<String, TimingSummary>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TimingSummary {
    pub count: usize,
    pub average_ms: u64,
    pub min_ms: u64,
    pub max_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthOverview {
    pub overall_status: HealthStatus,
    pub total_checks: usize,
    pub healthy_count: usize,
    pub degraded_count: usize,
    pub unhealthy_count: usize,
    pub checks: HashMap<String, HealthCheck>,
    pub timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_monitoring_config() {
        let config = AdvancedMonitoringConfig::default();
        assert!(config.enable_metrics_collection);
        assert!(config.enable_distributed_tracing);
        assert!(config.enable_alerting);
        assert_eq!(config.metrics_retention_hours, 24);
    }

    #[test]
    fn test_metric_creation_and_recording() {
        let mut metric = Metric::new(
            "test_metric".to_string(),
            MetricType::Counter,
            "Test metric".to_string(),
            "count".to_string(),
        );

        metric.record_value(10.0, HashMap::new());
        metric.record_value(20.0, HashMap::new());

        assert_eq!(metric.get_latest_value(), Some(20.0));
        assert_eq!(metric.values.len(), 2);
    }

    #[test]
    fn test_span_creation_and_timing() {
        let mut span = Span::new(
            "trace_123".to_string(),
            "span_456".to_string(),
            "test_operation".to_string(),
        );

        span.add_tag("service".to_string(), "test_service".to_string());
        span.add_log("info".to_string(), "Test log".to_string(), HashMap::new());
        span.finish();

        assert!(span.duration_microseconds.is_some());
        assert!(span.end_time.is_some());
        assert_eq!(span.tags.get("service"), Some(&"test_service".to_string()));
        assert_eq!(span.logs.len(), 1);
    }

    #[test]
    fn test_alert_condition_evaluation() {
        assert!(AlertCondition::GreaterThan.evaluate(10.0, 5.0));
        assert!(!AlertCondition::GreaterThan.evaluate(5.0, 10.0));
        assert!(AlertCondition::LessThan.evaluate(5.0, 10.0));
        assert!(AlertCondition::Equals.evaluate(10.0, 10.0));
        assert!(AlertCondition::NotEquals.evaluate(10.0, 5.0));
        assert!(AlertCondition::GreaterThanOrEqual.evaluate(10.0, 10.0));
        assert!(AlertCondition::LessThanOrEqual.evaluate(5.0, 10.0));
    }

    #[test]
    fn test_monitoring_system() {
        let config = AdvancedMonitoringConfig::default();
        let system = AdvancedMonitoringSystem::new(config);

        // Test metric recording
        let result = system.record_metric("test_metric", 42.0, HashMap::new());
        assert!(result.is_ok());

        // Test span creation
        let span_id = system.start_span("trace_123".to_string(), "test_op".to_string());
        assert!(span_id.is_ok());

        // Test alert rule
        let alert_rule = AlertRule {
            name: "test_alert".to_string(),
            description: "Test alert".to_string(),
            metric_name: "test_metric".to_string(),
            condition: AlertCondition::GreaterThan,
            threshold: 50.0,
            severity: AlertSeverity::Warning,
            evaluation_window_seconds: 60,
            cooldown_seconds: 300,
            enabled: true,
        };

        let result = system.add_alert_rule(alert_rule);
        assert!(result.is_ok());

        // Test health check
        let health_check = HealthCheck {
            name: "test_check".to_string(),
            status: HealthStatus::Healthy,
            message: "All good".to_string(),
            response_time_ms: 50,
            last_checked: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            metadata: HashMap::new(),
        };

        let result = system.register_health_check("test_check".to_string(), health_check);
        assert!(result.is_ok());
    }

    #[test]
    fn test_performance_counters() {
        let counters = PerformanceCounters::new();
        
        counters.increment("requests");
        counters.increment("requests");
        counters.record_timing("operation", Duration::from_millis(100));
        counters.record_timing("operation", Duration::from_millis(200));

        let summary = counters.get_summary();
        assert_eq!(summary.counters.get("requests"), Some(&2));
        assert!(summary.timings.contains_key("operation"));
        
        let timing = summary.timings.get("operation").unwrap();
        assert_eq!(timing.count, 2);
        assert_eq!(timing.average_ms, 150);
        assert_eq!(timing.min_ms, 100);
        assert_eq!(timing.max_ms, 200);
    }

    #[test]
    fn test_metrics_summary() {
        let config = AdvancedMonitoringConfig::default();
        let system = AdvancedMonitoringSystem::new(config);

        // Record some metrics
        system.record_metric("cpu_usage", 75.5, HashMap::new()).unwrap();
        system.record_metric("memory_usage", 60.2, HashMap::new()).unwrap();

        let summary = system.get_metrics_summary().unwrap();
        assert_eq!(summary.total_metrics, 2);
        assert_eq!(summary.metrics_overview.len(), 2);
    }

    #[test]
    fn test_health_overview() {
        let config = AdvancedMonitoringConfig::default();
        let system = AdvancedMonitoringSystem::new(config);

        let healthy_check = HealthCheck {
            name: "database".to_string(),
            status: HealthStatus::Healthy,
            message: "Connected".to_string(),
            response_time_ms: 25,
            last_checked: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            metadata: HashMap::new(),
        };

        let degraded_check = HealthCheck {
            name: "cache".to_string(),
            status: HealthStatus::Degraded,
            message: "High latency".to_string(),
            response_time_ms: 150,
            last_checked: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            metadata: HashMap::new(),
        };

        system.register_health_check("database".to_string(), healthy_check).unwrap();
        system.register_health_check("cache".to_string(), degraded_check).unwrap();

        let overview = system.get_health_overview().unwrap();
        assert_eq!(overview.overall_status, HealthStatus::Degraded);
        assert_eq!(overview.total_checks, 2);
        assert_eq!(overview.healthy_count, 1);
        assert_eq!(overview.degraded_count, 1);
        assert_eq!(overview.unhealthy_count, 0);
    }
}