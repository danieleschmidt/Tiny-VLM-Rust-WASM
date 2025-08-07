//! System health monitoring and diagnostics for Tiny-VLM
//!
//! Provides comprehensive health checks, performance monitoring, and system diagnostics.

use crate::{Result, TinyVlmError};
#[cfg(feature = "std")]
use std::time::Instant;

/// Overall system health status
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// All systems operating normally
    Healthy,
    /// Some issues detected but system functional
    Degraded,
    /// Critical issues affecting functionality
    Unhealthy,
    /// System status unknown or checks failed
    Unknown,
}

/// Detailed health check results
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Overall system status
    pub status: HealthStatus,
    /// Timestamp when report was generated
    pub timestamp: String,
    /// Individual health check results
    pub checks: Vec<HealthCheck>,
    /// System metrics
    pub metrics: SystemMetrics,
    /// Performance statistics
    pub performance: PerformanceStats,
}

/// Individual health check result
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Name of the health check
    pub name: String,
    /// Status of this specific check
    pub status: HealthStatus,
    /// Additional details or error message
    pub message: String,
    /// Time taken to perform this check (in milliseconds)
    pub duration_ms: f64,
}

/// System resource metrics
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Memory fragmentation percentage
    pub memory_fragmentation_percent: f64,
    /// Number of active tensors
    pub active_tensors: usize,
    /// SIMD support status
    pub simd_enabled: bool,
    /// WebAssembly status (if applicable)
    pub wasm_status: Option<WasmStatus>,
}

/// WebAssembly runtime status
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct WasmStatus {
    /// SIMD support in WASM
    pub simd_support: bool,
    /// Available memory in WASM runtime
    pub available_memory_mb: f64,
    /// JavaScript integration status
    pub js_integration_active: bool,
}

/// Performance statistics
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Average inference time in milliseconds
    pub avg_inference_time_ms: f64,
    /// Peak memory usage in MB
    pub peak_memory_usage_mb: f64,
    /// Number of successful inferences
    pub successful_inferences: u64,
    /// Number of failed inferences
    pub failed_inferences: u64,
    /// Uptime in seconds
    pub uptime_seconds: f64,
}

/// Comprehensive health monitor for the VLM system
pub struct HealthMonitor {
    /// Performance statistics
    performance_stats: PerformanceStats,
    /// Start time for uptime calculation
    #[cfg(feature = "std")]
    start_time: Instant,
    /// Last health check time
    #[cfg(feature = "std")]
    last_check: Option<Instant>,
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new() -> Self {
        Self {
            performance_stats: PerformanceStats {
                avg_inference_time_ms: 0.0,
                peak_memory_usage_mb: 0.0,
                successful_inferences: 0,
                failed_inferences: 0,
                uptime_seconds: 0.0,
            },
            #[cfg(feature = "std")]
            start_time: Instant::now(),
            #[cfg(feature = "std")]
            last_check: None,
        }
    }

    /// Perform comprehensive health checks
    pub fn check_health(&mut self, model: Option<&crate::FastVLM>) -> HealthReport {
        let mut checks = Vec::new();
        let mut overall_status = HealthStatus::Healthy;

        #[cfg(feature = "std")]
        let _start_time = Instant::now();

        // Memory health check
        let memory_check = self.check_memory_health(model);
        if memory_check.status != HealthStatus::Healthy {
            overall_status = self.combine_status(overall_status, memory_check.status);
        }
        checks.push(memory_check);

        // SIMD availability check
        let simd_check = self.check_simd_health();
        if simd_check.status != HealthStatus::Healthy {
            overall_status = self.combine_status(overall_status, simd_check.status);
        }
        checks.push(simd_check);

        // Model integrity check
        if let Some(model) = model {
            let model_check = self.check_model_health(model);
            if model_check.status != HealthStatus::Healthy {
                overall_status = self.combine_status(overall_status, model_check.status);
            }
            checks.push(model_check);
        }

        // Performance check
        let perf_check = self.check_performance_health();
        if perf_check.status != HealthStatus::Healthy {
            overall_status = self.combine_status(overall_status, perf_check.status);
        }
        checks.push(perf_check);

        // WebAssembly check (if applicable)
        #[cfg(feature = "wasm")]
        {
            let wasm_check = self.check_wasm_health();
            if wasm_check.status != HealthStatus::Healthy {
                overall_status = self.combine_status(overall_status, wasm_check.status);
            }
            checks.push(wasm_check);
        }

        // Update timing
        #[cfg(feature = "std")]
        {
            self.last_check = Some(Instant::now());
            self.performance_stats.uptime_seconds = self.start_time.elapsed().as_secs_f64();
        }

        HealthReport {
            status: overall_status,
            timestamp: self.get_timestamp(),
            checks,
            metrics: self.collect_system_metrics(model),
            performance: self.performance_stats.clone(),
        }
    }

    /// Record a successful inference
    pub fn record_inference_success(&mut self, duration_ms: f64, memory_mb: f64) {
        self.performance_stats.successful_inferences += 1;
        
        // Update average inference time
        let total_inferences = self.performance_stats.successful_inferences + self.performance_stats.failed_inferences;
        self.performance_stats.avg_inference_time_ms = 
            (self.performance_stats.avg_inference_time_ms * (total_inferences - 1) as f64 + duration_ms) / total_inferences as f64;
        
        // Update peak memory usage
        if memory_mb > self.performance_stats.peak_memory_usage_mb {
            self.performance_stats.peak_memory_usage_mb = memory_mb;
        }
    }

    /// Record a failed inference
    pub fn record_inference_failure(&mut self) {
        self.performance_stats.failed_inferences += 1;
    }

    /// Check memory health
    fn check_memory_health(&self, model: Option<&crate::FastVLM>) -> HealthCheck {
        #[cfg(feature = "std")]
        let start = Instant::now();

        let (status, message) = if let Some(model) = model {
            let stats = model.memory_stats();
            let fragmentation = stats.fragmentation * 100.0;
            let usage_percent = (stats.allocated_memory as f64 / stats.max_memory as f64) * 100.0;

            if usage_percent > 90.0 {
                (HealthStatus::Unhealthy, format!("Memory usage critical: {:.1}%", usage_percent))
            } else if usage_percent > 75.0 || fragmentation > 50.0 {
                (HealthStatus::Degraded, format!("Memory usage high: {:.1}%, fragmentation: {:.1}%", usage_percent, fragmentation))
            } else {
                (HealthStatus::Healthy, format!("Memory healthy: {:.1}% used, {:.1}% fragmented", usage_percent, fragmentation))
            }
        } else {
            (HealthStatus::Unknown, "No model instance available for memory check".to_string())
        };

        HealthCheck {
            name: "Memory Health".to_string(),
            status,
            message,
            #[cfg(feature = "std")]
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            #[cfg(not(feature = "std"))]
            duration_ms: 0.0,
        }
    }

    /// Check SIMD availability and functionality
    fn check_simd_health(&self) -> HealthCheck {
        #[cfg(feature = "std")]
        let start = Instant::now();

        let (status, message) = {
            // Check if SIMD features are available
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            {
                (HealthStatus::Healthy, "SIMD acceleration available".to_string())
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                (HealthStatus::Degraded, "SIMD acceleration not available on this platform".to_string())
            }
        };

        HealthCheck {
            name: "SIMD Health".to_string(),
            status,
            message,
            #[cfg(feature = "std")]
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            #[cfg(not(feature = "std"))]
            duration_ms: 0.0,
        }
    }

    /// Check model integrity
    fn check_model_health(&self, model: &crate::FastVLM) -> HealthCheck {
        #[cfg(feature = "std")]
        let start = Instant::now();

        let config = model.config();
        let memory_stats = model.memory_stats();

        let (status, message) = {
            // Check if model configuration is reasonable
            if config.vision_dim == 0 || config.text_dim == 0 || config.hidden_dim == 0 {
                (HealthStatus::Unhealthy, "Model has invalid dimensions".to_string())
            } else if memory_stats.total_memory == 0 {
                (HealthStatus::Unhealthy, "Model memory pool not initialized".to_string())
            } else if config.temperature <= 0.0 || config.temperature > 5.0 {
                (HealthStatus::Degraded, format!("Model temperature {} is outside normal range", config.temperature))
            } else {
                (HealthStatus::Healthy, "Model configuration and state appear healthy".to_string())
            }
        };

        HealthCheck {
            name: "Model Health".to_string(),
            status,
            message,
            #[cfg(feature = "std")]
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            #[cfg(not(feature = "std"))]
            duration_ms: 0.0,
        }
    }

    /// Check performance metrics
    fn check_performance_health(&self) -> HealthCheck {
        #[cfg(feature = "std")]
        let start = Instant::now();

        let total_inferences = self.performance_stats.successful_inferences + self.performance_stats.failed_inferences;
        let error_rate = if total_inferences > 0 {
            (self.performance_stats.failed_inferences as f64) / (total_inferences as f64) * 100.0
        } else {
            0.0
        };

        let (status, message) = {
            if error_rate > 50.0 {
                (HealthStatus::Unhealthy, format!("High error rate: {:.1}%", error_rate))
            } else if error_rate > 10.0 || self.performance_stats.avg_inference_time_ms > 5000.0 {
                (HealthStatus::Degraded, format!("Performance issues: {:.1}% errors, {:.0}ms avg inference", error_rate, self.performance_stats.avg_inference_time_ms))
            } else {
                (HealthStatus::Healthy, format!("Good performance: {:.1}% errors, {:.0}ms avg inference", error_rate, self.performance_stats.avg_inference_time_ms))
            }
        };

        HealthCheck {
            name: "Performance Health".to_string(),
            status,
            message,
            #[cfg(feature = "std")]
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            #[cfg(not(feature = "std"))]
            duration_ms: 0.0,
        }
    }

    /// Check WebAssembly runtime health
    #[cfg(feature = "wasm")]
    fn check_wasm_health(&self) -> HealthCheck {
        #[cfg(feature = "std")]
        let start = Instant::now();

        // In a real implementation, this would check WASM-specific metrics
        let (status, message) = (
            HealthStatus::Healthy,
            "WebAssembly runtime operational".to_string()
        );

        HealthCheck {
            name: "WebAssembly Health".to_string(),
            status,
            message,
            #[cfg(feature = "std")]
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            #[cfg(not(feature = "std"))]
            duration_ms: 0.0,
        }
    }

    /// Collect system metrics
    fn collect_system_metrics(&self, model: Option<&crate::FastVLM>) -> SystemMetrics {
        let (memory_usage_mb, memory_fragmentation_percent, active_tensors) = if let Some(model) = model {
            let stats = model.memory_stats();
            (
                stats.allocated_memory as f64 / (1024.0 * 1024.0),
                stats.fragmentation * 100.0,
                1, // Simplified - would count actual active tensors
            )
        } else {
            (0.0, 0.0, 0)
        };

        SystemMetrics {
            memory_usage_mb,
            memory_fragmentation_percent: memory_fragmentation_percent as f64,
            active_tensors,
            simd_enabled: cfg!(any(target_arch = "aarch64", target_arch = "x86_64")),
            wasm_status: self.collect_wasm_status(),
        }
    }

    /// Collect WebAssembly-specific status
    fn collect_wasm_status(&self) -> Option<WasmStatus> {
        #[cfg(feature = "wasm")]
        {
            Some(WasmStatus {
                simd_support: cfg!(target_feature = "simd128"),
                available_memory_mb: 0.0, // Would query actual WASM memory
                js_integration_active: true,
            })
        }
        #[cfg(not(feature = "wasm"))]
        {
            None
        }
    }

    /// Combine two health statuses, returning the worse one
    fn combine_status(&self, current: HealthStatus, new: HealthStatus) -> HealthStatus {
        match (current, new) {
            (HealthStatus::Unhealthy, _) | (_, HealthStatus::Unhealthy) => HealthStatus::Unhealthy,
            (HealthStatus::Degraded, _) | (_, HealthStatus::Degraded) => HealthStatus::Degraded,
            (HealthStatus::Unknown, _) | (_, HealthStatus::Unknown) => HealthStatus::Unknown,
            (HealthStatus::Healthy, HealthStatus::Healthy) => HealthStatus::Healthy,
        }
    }

    /// Get current timestamp as string
    fn get_timestamp(&self) -> String {
        #[cfg(feature = "std")]
        {
            chrono::Utc::now().to_rfc3339()
        }
        #[cfg(not(feature = "std"))]
        {
            "unknown".to_string()
        }
    }
}

impl HealthReport {
    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        self.status == HealthStatus::Healthy
    }

    /// Get error rate as percentage
    pub fn error_rate(&self) -> f64 {
        let total = self.performance.successful_inferences + self.performance.failed_inferences;
        if total > 0 {
            (self.performance.failed_inferences as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Export health report as JSON string
    #[cfg(feature = "std")]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self)
            .map_err(|e| TinyVlmError::serialization(format!("Failed to serialize health report: {}", e)))
    }
}

// Make health report serializable
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor_creation() {
        let monitor = HealthMonitor::new();
        assert_eq!(monitor.performance_stats.successful_inferences, 0);
        assert_eq!(monitor.performance_stats.failed_inferences, 0);
    }

    #[test]
    fn test_health_status_combination() {
        let monitor = HealthMonitor::new();
        
        assert_eq!(
            monitor.combine_status(HealthStatus::Healthy, HealthStatus::Healthy),
            HealthStatus::Healthy
        );
        
        assert_eq!(
            monitor.combine_status(HealthStatus::Healthy, HealthStatus::Unhealthy),
            HealthStatus::Unhealthy
        );
        
        assert_eq!(
            monitor.combine_status(HealthStatus::Degraded, HealthStatus::Healthy),
            HealthStatus::Degraded
        );
    }

    #[test]
    fn test_inference_recording() {
        let mut monitor = HealthMonitor::new();
        
        monitor.record_inference_success(100.0, 50.0);
        assert_eq!(monitor.performance_stats.successful_inferences, 1);
        assert_eq!(monitor.performance_stats.avg_inference_time_ms, 100.0);
        assert_eq!(monitor.performance_stats.peak_memory_usage_mb, 50.0);
        
        monitor.record_inference_failure();
        assert_eq!(monitor.performance_stats.failed_inferences, 1);
    }

    #[test]
    fn test_health_report() {
        let report = HealthReport {
            status: HealthStatus::Healthy,
            timestamp: "2023-01-01T00:00:00Z".to_string(),
            checks: vec![],
            metrics: SystemMetrics {
                memory_usage_mb: 100.0,
                memory_fragmentation_percent: 10.0,
                active_tensors: 5,
                simd_enabled: true,
                wasm_status: None,
            },
            performance: PerformanceStats {
                avg_inference_time_ms: 150.0,
                peak_memory_usage_mb: 200.0,
                successful_inferences: 100,
                failed_inferences: 5,
                uptime_seconds: 3600.0,
            },
        };
        
        assert!(report.is_healthy());
        assert_eq!(report.error_rate(), 5.0); // 5 failures out of 105 total
    }
}