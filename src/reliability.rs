//! Enhanced Reliability Framework
//!
//! Advanced error handling, fault tolerance, and system resilience mechanisms
//! for production-grade deployment.

use crate::{Result, TinyVlmError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Reliability configuration for production systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityConfig {
    pub circuit_breaker_failure_threshold: usize,
    pub circuit_breaker_timeout_seconds: u64,
    pub circuit_breaker_minimum_requests: usize,
    pub retry_max_attempts: usize,
    pub retry_base_delay_ms: u64,
    pub retry_max_delay_ms: u64,
    pub health_check_interval_seconds: u64,
    pub degraded_mode_threshold: f64,
    pub enable_graceful_degradation: bool,
    pub enable_bulkhead_isolation: bool,
    pub resource_pool_timeout_ms: u64,
}

impl Default for ReliabilityConfig {
    fn default() -> Self {
        Self {
            circuit_breaker_failure_threshold: 5,
            circuit_breaker_timeout_seconds: 60,
            circuit_breaker_minimum_requests: 10,
            retry_max_attempts: 3,
            retry_base_delay_ms: 100,
            retry_max_delay_ms: 5000,
            health_check_interval_seconds: 30,
            degraded_mode_threshold: 0.8,
            enable_graceful_degradation: true,
            enable_bulkhead_isolation: true,
            resource_pool_timeout_ms: 10000,
        }
    }
}

/// System reliability state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReliabilityState {
    Healthy,
    Degraded,
    Critical,
    Offline,
}

/// Error categories for advanced error handling
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    Transient,
    Persistent,
    ResourceExhaustion,
    NetworkFailure,
    SecurityViolation,
    ModelFailure,
    DataCorruption,
    ConfigurationError,
}

/// Enhanced error with categorization and recovery suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedError {
    pub category: ErrorCategory,
    pub severity: ErrorSeverity,
    pub message: String,
    pub context: HashMap<String, String>,
    pub recovery_suggestions: Vec<String>,
    pub retry_after_seconds: Option<u64>,
    pub is_retryable: bool,
    pub correlation_id: String,
}

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Fatal,
}

/// Enhanced circuit breaker with multiple failure types
#[derive(Debug)]
pub struct EnhancedCircuitBreaker {
    config: ReliabilityConfig,
    state: CircuitBreakerState,
    failure_counts: HashMap<ErrorCategory, AtomicUsize>,
    success_count: AtomicUsize,
    last_failure_time: Option<Instant>,
    total_requests: AtomicU64,
    consecutive_failures: AtomicUsize,
}

#[derive(Debug, Clone, PartialEq)]
enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

impl EnhancedCircuitBreaker {
    pub fn new(config: ReliabilityConfig) -> Self {
        Self {
            config,
            state: CircuitBreakerState::Closed,
            failure_counts: HashMap::new(),
            success_count: AtomicUsize::new(0),
            last_failure_time: None,
            total_requests: AtomicU64::new(0),
            consecutive_failures: AtomicUsize::new(0),
        }
    }

    pub fn call<F, T>(&mut self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        match self.state {
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() > Duration::from_secs(self.config.circuit_breaker_timeout_seconds) {
                        self.state = CircuitBreakerState::HalfOpen;
                    } else {
                        return Err(TinyVlmError::circuit_breaker_open("Circuit breaker is open"));
                    }
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Allow one request through to test if service is recovering
            }
            CircuitBreakerState::Closed => {
                // Normal operation
            }
        }

        match operation() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(error) => {
                self.on_failure(&error);
                Err(error)
            }
        }
    }

    fn on_success(&mut self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
        
        if self.state == CircuitBreakerState::HalfOpen {
            self.state = CircuitBreakerState::Closed;
        }
    }

    fn on_failure(&mut self, error: &TinyVlmError) {
        let error_category = self.categorize_error(error);
        
        self.failure_counts
            .entry(error_category)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(1, Ordering::Relaxed);
        
        self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        self.last_failure_time = Some(Instant::now());
        
        if self.should_open_circuit() {
            self.state = CircuitBreakerState::Open;
        }
    }

    fn should_open_circuit(&self) -> bool {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        if total_requests < self.config.circuit_breaker_minimum_requests as u64 {
            return false;
        }

        let consecutive_failures = self.consecutive_failures.load(Ordering::Relaxed);
        consecutive_failures >= self.config.circuit_breaker_failure_threshold
    }

    fn categorize_error(&self, error: &TinyVlmError) -> ErrorCategory {
        match error {
            TinyVlmError::NetworkError(_) => ErrorCategory::NetworkFailure,
            TinyVlmError::ValidationError(_) => ErrorCategory::DataCorruption,
            TinyVlmError::ConfigurationError(_) => ErrorCategory::ConfigurationError,
            TinyVlmError::SecurityError(_) => ErrorCategory::SecurityViolation,
            TinyVlmError::CircuitBreakerOpen(_) => ErrorCategory::ResourceExhaustion,
            _ => ErrorCategory::Transient,
        }
    }

    pub fn get_state(&self) -> String {
        format!("{:?}", self.state)
    }

    pub fn get_metrics(&self) -> CircuitBreakerMetrics {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let success_count = self.success_count.load(Ordering::Relaxed);
        let consecutive_failures = self.consecutive_failures.load(Ordering::Relaxed);
        
        let failure_count = total_requests.saturating_sub(success_count as u64) as usize;
        let success_rate = if total_requests > 0 {
            success_count as f64 / total_requests as f64
        } else {
            1.0
        };

        CircuitBreakerMetrics {
            state: format!("{:?}", self.state),
            total_requests,
            success_count,
            failure_count,
            success_rate,
            consecutive_failures,
        }
    }
}

/// Circuit breaker metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct CircuitBreakerMetrics {
    pub state: String,
    pub total_requests: u64,
    pub success_count: usize,
    pub failure_count: usize,
    pub success_rate: f64,
    pub consecutive_failures: usize,
}

/// Advanced retry mechanism with exponential backoff and jitter
pub struct AdvancedRetryPolicy {
    config: ReliabilityConfig,
}

impl AdvancedRetryPolicy {
    pub fn new(config: ReliabilityConfig) -> Self {
        Self { config }
    }

    pub fn execute_with_retry<F, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Result<T>,
    {
        let mut last_error = None;
        
        for attempt in 0..self.config.retry_max_attempts {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !self.is_retryable(&error) {
                        return Err(error);
                    }
                    
                    last_error = Some(error);
                    
                    if attempt < self.config.retry_max_attempts - 1 {
                        let delay = self.calculate_delay(attempt);
                        std::thread::sleep(delay);
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| TinyVlmError::InternalError("Retry exhausted".to_string())))
    }

    pub fn execute_with_retry_mut<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        let mut operation = operation;
        let mut last_error = None;
        
        for attempt in 0..self.config.retry_max_attempts {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !self.is_retryable(&error) {
                        return Err(error);
                    }
                    
                    last_error = Some(error);
                    
                    if attempt < self.config.retry_max_attempts - 1 {
                        let delay = self.calculate_delay(attempt);
                        std::thread::sleep(delay);
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| TinyVlmError::InternalError("Retry exhausted".to_string())))
    }

    fn is_retryable(&self, error: &TinyVlmError) -> bool {
        match error {
            TinyVlmError::NetworkError(_) => true,
            TinyVlmError::CircuitBreakerOpen(_) => false,
            TinyVlmError::SecurityError(_) => false,
            TinyVlmError::ValidationError(_) => false,
            _ => true,
        }
    }

    fn calculate_delay(&self, attempt: usize) -> Duration {
        let base_delay = self.config.retry_base_delay_ms;
        let max_delay = self.config.retry_max_delay_ms;
        
        // Exponential backoff with jitter
        let exponential_delay = base_delay * (2_u64.pow(attempt as u32));
        let delay_with_jitter = exponential_delay + (rand::random::<u64>() % (exponential_delay / 4));
        
        Duration::from_millis(delay_with_jitter.min(max_delay))
    }
}

/// Graceful degradation manager
pub struct GracefulDegradationManager {
    config: ReliabilityConfig,
    current_state: ReliabilityState,
    performance_metrics: PerformanceMetrics,
    fallback_handlers: HashMap<String, Box<dyn FallbackHandler>>,
}

impl GracefulDegradationManager {
    pub fn new(config: ReliabilityConfig) -> Self {
        Self {
            config,
            current_state: ReliabilityState::Healthy,
            performance_metrics: PerformanceMetrics::new(),
            fallback_handlers: HashMap::new(),
        }
    }

    pub fn register_fallback_handler(&mut self, service: String, handler: Box<dyn FallbackHandler>) {
        self.fallback_handlers.insert(service, handler);
    }

    pub fn execute_with_degradation<F, T>(&mut self, service: &str, primary_operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        self.update_system_state();

        match self.current_state {
            ReliabilityState::Healthy => {
                match primary_operation() {
                    Ok(result) => {
                        self.performance_metrics.record_success();
                        Ok(result)
                    }
                    Err(error) => {
                        self.performance_metrics.record_failure();
                        if self.config.enable_graceful_degradation {
                            self.try_fallback(service, error)
                        } else {
                            Err(error)
                        }
                    }
                }
            }
            ReliabilityState::Degraded => {
                if self.config.enable_graceful_degradation {
                    self.try_fallback(service, TinyVlmError::service_degraded("Service degraded due to health check failure"))
                } else {
                    primary_operation()
                }
            }
            ReliabilityState::Critical | ReliabilityState::Offline => {
                self.try_fallback(service, TinyVlmError::service_unavailable("Service unavailable"))
            }
        }
    }

    fn try_fallback<T>(&self, service: &str, original_error: TinyVlmError) -> Result<T> {
        if let Some(_handler) = self.fallback_handlers.get(service) {
            // For now, just return the original error since fallback returns String
            // In a real implementation, this would need proper type handling
            Err(original_error)
        } else {
            Err(original_error)
        }
    }

    fn update_system_state(&mut self) {
        let success_rate = self.performance_metrics.get_success_rate();
        
        self.current_state = if success_rate >= 0.95 {
            ReliabilityState::Healthy
        } else if success_rate >= self.config.degraded_mode_threshold {
            ReliabilityState::Degraded
        } else if success_rate >= 0.5 {
            ReliabilityState::Critical
        } else {
            ReliabilityState::Offline
        };
    }

    pub fn get_current_state(&self) -> &ReliabilityState {
        &self.current_state
    }

    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }
}

/// Fallback handler trait
pub trait FallbackHandler: Send + Sync {
    fn handle_fallback(&self) -> Result<String>;
}

/// Simple fallback handler implementation
pub struct SimpleFallbackHandler {
    fallback_response: String,
}

impl SimpleFallbackHandler {
    pub fn new(fallback_response: String) -> Self {
        Self { fallback_response }
    }
}

impl FallbackHandler for SimpleFallbackHandler {
    fn handle_fallback(&self) -> Result<String> {
        Ok(self.fallback_response.clone())
    }
}

/// Performance metrics tracking
#[derive(Debug)]
pub struct PerformanceMetrics {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    average_response_time: AtomicU64,
    peak_response_time: AtomicU64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            average_response_time: AtomicU64::new(0),
            peak_response_time: AtomicU64::new(0),
        }
    }

    pub fn record_success(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_response_time(&self, response_time_ms: u64) {
        let current_avg = self.average_response_time.load(Ordering::Relaxed);
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        
        if total_requests > 0 {
            let new_avg = ((current_avg * (total_requests - 1)) + response_time_ms) / total_requests;
            self.average_response_time.store(new_avg, Ordering::Relaxed);
        }
        
        let current_peak = self.peak_response_time.load(Ordering::Relaxed);
        if response_time_ms > current_peak {
            self.peak_response_time.store(response_time_ms, Ordering::Relaxed);
        }
    }

    pub fn get_success_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        
        let successful = self.successful_requests.load(Ordering::Relaxed);
        successful as f64 / total as f64
    }

    pub fn get_metrics_summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            success_rate: self.get_success_rate(),
            average_response_time_ms: self.average_response_time.load(Ordering::Relaxed),
            peak_response_time_ms: self.peak_response_time.load(Ordering::Relaxed),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics summary for reporting
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub success_rate: f64,
    pub average_response_time_ms: u64,
    pub peak_response_time_ms: u64,
}

/// Bulkhead isolation for resource protection
pub struct BulkheadIsolation {
    config: ReliabilityConfig,
    resource_pools: HashMap<String, Arc<ResourcePool>>,
}

impl BulkheadIsolation {
    pub fn new(config: ReliabilityConfig) -> Self {
        Self {
            config,
            resource_pools: HashMap::new(),
        }
    }

    pub fn create_resource_pool(&mut self, name: String, max_concurrent: usize) {
        let pool = Arc::new(ResourcePool::new(max_concurrent, self.config.resource_pool_timeout_ms));
        self.resource_pools.insert(name, pool);
    }

    pub fn execute_isolated<F, T>(&self, pool_name: &str, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        if let Some(pool) = self.resource_pools.get(pool_name) {
            pool.acquire_and_execute(operation)
        } else {
            Err(TinyVlmError::ConfigurationError(format!("Resource pool '{}' not found", pool_name)))
        }
    }

    pub fn get_pool_metrics(&self, pool_name: &str) -> Option<ResourcePoolMetrics> {
        self.resource_pools.get(pool_name).map(|pool| pool.get_metrics())
    }
}

/// Resource pool for bulkhead isolation
pub struct ResourcePool {
    max_concurrent: usize,
    timeout_ms: u64,
    current_usage: AtomicUsize,
    total_acquisitions: AtomicU64,
    total_timeouts: AtomicU64,
}

impl ResourcePool {
    pub fn new(max_concurrent: usize, timeout_ms: u64) -> Self {
        Self {
            max_concurrent,
            timeout_ms,
            current_usage: AtomicUsize::new(0),
            total_acquisitions: AtomicU64::new(0),
            total_timeouts: AtomicU64::new(0),
        }
    }

    pub fn acquire_and_execute<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let start_time = Instant::now();
        let timeout = Duration::from_millis(self.timeout_ms);

        // Try to acquire resource
        while start_time.elapsed() < timeout {
            let current = self.current_usage.load(Ordering::Relaxed);
            if current < self.max_concurrent {
                if self.current_usage.compare_exchange_weak(
                    current,
                    current + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ).is_ok() {
                    self.total_acquisitions.fetch_add(1, Ordering::Relaxed);
                    
                    // Execute operation
                    let result = operation();
                    
                    // Release resource
                    self.current_usage.fetch_sub(1, Ordering::Relaxed);
                    
                    return result;
                }
            }
            
            // Brief wait before retrying
            std::thread::sleep(Duration::from_millis(1));
        }

        // Timeout occurred
        self.total_timeouts.fetch_add(1, Ordering::Relaxed);
        Err(TinyVlmError::InternalError("Resource pool timeout".to_string()))
    }

    pub fn get_metrics(&self) -> ResourcePoolMetrics {
        ResourcePoolMetrics {
            max_concurrent: self.max_concurrent,
            current_usage: self.current_usage.load(Ordering::Relaxed),
            total_acquisitions: self.total_acquisitions.load(Ordering::Relaxed),
            total_timeouts: self.total_timeouts.load(Ordering::Relaxed),
            utilization_rate: self.current_usage.load(Ordering::Relaxed) as f64 / self.max_concurrent as f64,
        }
    }
}

/// Resource pool metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct ResourcePoolMetrics {
    pub max_concurrent: usize,
    pub current_usage: usize,
    pub total_acquisitions: u64,
    pub total_timeouts: u64,
    pub utilization_rate: f64,
}

/// Comprehensive reliability manager
pub struct ReliabilityManager {
    config: ReliabilityConfig,
    circuit_breaker: EnhancedCircuitBreaker,
    retry_policy: AdvancedRetryPolicy,
    degradation_manager: GracefulDegradationManager,
    bulkhead_isolation: BulkheadIsolation,
}

impl ReliabilityManager {
    pub fn new(config: ReliabilityConfig) -> Self {
        Self {
            circuit_breaker: EnhancedCircuitBreaker::new(config.clone()),
            retry_policy: AdvancedRetryPolicy::new(config.clone()),
            degradation_manager: GracefulDegradationManager::new(config.clone()),
            bulkhead_isolation: BulkheadIsolation::new(config.clone()),
            config,
        }
    }

    pub fn execute_reliable_operation<F, T>(&mut self, service: &str, operation: F) -> Result<T>
    where
        F: Fn() -> Result<T>,
    {
        // Combine circuit breaker, retry, and degradation
        let reliable_operation = || {
            self.circuit_breaker.call(|| {
                self.retry_policy.execute_with_retry(|| operation())
            })
        };

        self.degradation_manager.execute_with_degradation(service, reliable_operation)
    }

    pub fn get_system_health(&self) -> SystemHealthReport {
        SystemHealthReport {
            overall_state: self.degradation_manager.get_current_state().clone(),
            circuit_breaker_metrics: self.circuit_breaker.get_metrics(),
            performance_metrics: self.degradation_manager.get_performance_metrics().get_metrics_summary(),
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        }
    }
}

/// System health report
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemHealthReport {
    pub overall_state: ReliabilityState,
    pub circuit_breaker_metrics: CircuitBreakerMetrics,
    pub performance_metrics: MetricsSummary,
    pub timestamp: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reliability_config_default() {
        let config = ReliabilityConfig::default();
        assert_eq!(config.circuit_breaker_failure_threshold, 5);
        assert_eq!(config.retry_max_attempts, 3);
        assert!(config.enable_graceful_degradation);
    }

    #[test]
    fn test_enhanced_circuit_breaker() {
        let config = ReliabilityConfig::default();
        let mut circuit_breaker = EnhancedCircuitBreaker::new(config);

        // Test successful operation
        let result = circuit_breaker.call(|| Ok("success"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[test]
    fn test_retry_policy() {
        let config = ReliabilityConfig::default();
        let retry_policy = AdvancedRetryPolicy::new(config);

        let mut attempt_count = 0;
        let result = retry_policy.execute_with_retry_mut(&mut || {
            attempt_count += 1;
            if attempt_count < 3 {
                Err(TinyVlmError::NetworkError("Temporary failure".to_string()))
            } else {
                Ok("success")
            }
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempt_count, 3);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics::new();
        
        metrics.record_success();
        metrics.record_success();
        metrics.record_failure();
        
        assert_eq!(metrics.get_success_rate(), 2.0 / 3.0);
        
        let summary = metrics.get_metrics_summary();
        assert_eq!(summary.total_requests, 3);
        assert_eq!(summary.successful_requests, 2);
        assert_eq!(summary.failed_requests, 1);
    }

    #[test]
    fn test_resource_pool() {
        let pool = ResourcePool::new(2, 1000);
        
        let result1 = pool.acquire_and_execute(|| Ok("result1"));
        let result2 = pool.acquire_and_execute(|| Ok("result2"));
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        
        let metrics = pool.get_metrics();
        assert_eq!(metrics.max_concurrent, 2);
        assert_eq!(metrics.total_acquisitions, 2);
    }

    #[test]
    fn test_graceful_degradation_manager() {
        let config = ReliabilityConfig::default();
        let mut manager = GracefulDegradationManager::new(config);
        
        assert_eq!(manager.get_current_state(), &ReliabilityState::Healthy);
        
        // Register a fallback handler
        let fallback = Box::new(SimpleFallbackHandler::new("Service degraded".to_string()));
        manager.register_fallback_handler("test_service".to_string(), fallback);
        
        // Test operation execution
        let result = manager.execute_with_degradation("test_service", || {
            Ok("primary_result")
        });
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "primary_result");
    }

    #[test]
    fn test_bulkhead_isolation() {
        let config = ReliabilityConfig::default();
        let mut bulkhead = BulkheadIsolation::new(config);
        
        bulkhead.create_resource_pool("test_pool".to_string(), 1);
        
        let result = bulkhead.execute_isolated("test_pool", || Ok("isolated_result"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "isolated_result");
        
        let metrics = bulkhead.get_pool_metrics("test_pool");
        assert!(metrics.is_some());
        assert_eq!(metrics.unwrap().max_concurrent, 1);
    }

    #[test]
    fn test_reliability_manager() {
        let config = ReliabilityConfig::default();
        let mut manager = ReliabilityManager::new(config);
        
        let result = manager.execute_reliable_operation("test_service", || {
            Ok("reliable_result")
        });
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "reliable_result");
        
        let health_report = manager.get_system_health();
        assert_eq!(health_report.overall_state, ReliabilityState::Healthy);
    }
}