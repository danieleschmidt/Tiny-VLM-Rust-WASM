//! Structured logging system for Tiny-VLM
//!
//! Provides comprehensive logging with different levels and structured output.

#[cfg(feature = "std")]
use tracing::{info, warn, error, debug, trace};
#[cfg(feature = "std")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Logging configuration for the VLM system
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Log level filter (trace, debug, info, warn, error)
    pub level: String,
    /// Whether to include timestamps
    pub include_timestamps: bool,
    /// Whether to include thread information
    pub include_thread_info: bool,
    /// Whether to include source location
    pub include_source_location: bool,
    /// Output format (json, pretty, compact)
    pub format: LogFormat,
}

/// Log output format options
#[derive(Debug, Clone)]
pub enum LogFormat {
    /// Human-readable format
    Pretty,
    /// Compact single-line format
    Compact,
    /// JSON structured format
    Json,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            include_timestamps: true,
            include_thread_info: false,
            include_source_location: false,
            format: LogFormat::Pretty,
        }
    }
}

/// Initialize the logging system with given configuration
#[cfg(feature = "std")]
pub fn init_logging(config: LogConfig) -> Result<(), crate::TinyVlmError> {
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&config.level))
        .map_err(|e| crate::TinyVlmError::config(format!("Invalid log level: {}", e)))?;

    match config.format {
        LogFormat::Pretty => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(tracing_subscriber::fmt::layer().pretty())
                .try_init()
                .map_err(|e| crate::TinyVlmError::config(format!("Failed to initialize logging: {}", e)))?;
        }
        LogFormat::Compact => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(tracing_subscriber::fmt::layer().compact())
                .try_init()
                .map_err(|e| crate::TinyVlmError::config(format!("Failed to initialize logging: {}", e)))?;
        }
        LogFormat::Json => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(tracing_subscriber::fmt::layer())
                .try_init()
                .map_err(|e| crate::TinyVlmError::config(format!("Failed to initialize logging: {}", e)))?;
        }
    }

    info!("Tiny-VLM logging system initialized");
    Ok(())
}

/// Log inference performance metrics
#[cfg(feature = "std")]
pub fn log_inference_metrics(
    model_config: &crate::ModelConfig,
    inference_time_ms: f64,
    memory_usage_mb: f64,
    input_tokens: usize,
    output_tokens: usize,
) {
    info!(
        target: "tiny_vlm::inference",
        inference_time_ms = inference_time_ms,
        memory_usage_mb = memory_usage_mb,
        input_tokens = input_tokens,
        output_tokens = output_tokens,
        model_vision_dim = model_config.vision_dim,
        model_text_dim = model_config.text_dim,
        "Inference completed successfully"
    );
}

/// Log model loading events
#[cfg(feature = "std")]
pub fn log_model_loading(model_path: &str, loading_time_ms: f64, model_size_mb: f64) {
    info!(
        target: "tiny_vlm::model",
        model_path = model_path,
        loading_time_ms = loading_time_ms,
        model_size_mb = model_size_mb,
        "Model loaded successfully"
    );
}

/// Log data processing events
#[cfg(feature = "std")]
pub fn log_data_processing(
    operation: &str,
    samples_processed: usize,
    processing_time_ms: f64,
    errors_encountered: usize,
) {
    if errors_encountered > 0 {
        warn!(
            target: "tiny_vlm::data",
            operation = operation,
            samples_processed = samples_processed,
            processing_time_ms = processing_time_ms,
            errors_encountered = errors_encountered,
            "Data processing completed with errors"
        );
    } else {
        info!(
            target: "tiny_vlm::data",
            operation = operation,
            samples_processed = samples_processed,
            processing_time_ms = processing_time_ms,
            "Data processing completed successfully"
        );
    }
}

/// Log SIMD optimization events
#[cfg(feature = "std")]
pub fn log_simd_optimization(operation: &str, simd_enabled: bool, speedup_factor: f64) {
    debug!(
        target: "tiny_vlm::simd",
        operation = operation,
        simd_enabled = simd_enabled,
        speedup_factor = speedup_factor,
        "SIMD optimization applied"
    );
}

/// Log memory management events
#[cfg(feature = "std")]
pub fn log_memory_event(
    event: &str,
    total_memory_mb: f64,
    allocated_memory_mb: f64,
    fragmentation_percent: f64,
) {
    debug!(
        target: "tiny_vlm::memory",
        event = event,
        total_memory_mb = total_memory_mb,
        allocated_memory_mb = allocated_memory_mb,
        fragmentation_percent = fragmentation_percent,
        "Memory management event"
    );
}

/// Log security events
#[cfg(feature = "std")]
pub fn log_security_event(event: &str, severity: SecuritySeverity, details: &str) {
    match severity {
        SecuritySeverity::Critical => {
            error!(
                target: "tiny_vlm::security",
                event = event,
                severity = "critical",
                details = details,
                "Critical security event detected"
            );
        }
        SecuritySeverity::High => {
            warn!(
                target: "tiny_vlm::security",
                event = event,
                severity = "high",
                details = details,
                "High-severity security event detected"
            );
        }
        SecuritySeverity::Medium => {
            warn!(
                target: "tiny_vlm::security",
                event = event,
                severity = "medium",
                details = details,
                "Medium-severity security event detected"
            );
        }
        SecuritySeverity::Low => {
            debug!(
                target: "tiny_vlm::security",
                event = event,
                severity = "low",
                details = details,
                "Low-severity security event detected"
            );
        }
    }
}

/// Security event severity levels
#[derive(Debug, Clone, Copy)]
pub enum SecuritySeverity {
    /// Critical security issue requiring immediate attention
    Critical,
    /// High-priority security issue
    High,
    /// Medium-priority security issue
    Medium,
    /// Low-priority security issue
    Low,
}

/// Log WASM-specific events
#[cfg(all(feature = "std", feature = "wasm"))]
pub fn log_wasm_event(event: &str, performance_ms: Option<f64>, memory_mb: Option<f64>) {
    info!(
        target: "tiny_vlm::wasm",
        event = event,
        performance_ms = performance_ms,
        memory_mb = memory_mb,
        "WASM runtime event"
    );
}

/// Performance measurement utilities
#[cfg(feature = "std")]
pub struct PerformanceTimer {
    start_time: std::time::Instant,
    operation: String,
}

#[cfg(feature = "std")]
impl PerformanceTimer {
    /// Start timing an operation
    pub fn new(operation: impl Into<String>) -> Self {
        let operation = operation.into();
        trace!(target: "tiny_vlm::perf", operation = %operation, "Starting performance measurement");
        
        Self {
            start_time: std::time::Instant::now(),
            operation,
        }
    }

    /// End timing and log the duration
    pub fn end(self) -> f64 {
        let duration_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        
        debug!(
            target: "tiny_vlm::perf",
            operation = %self.operation,
            duration_ms = duration_ms,
            "Performance measurement completed"
        );
        
        duration_ms
    }

    /// Get current elapsed time without ending the timer
    pub fn elapsed_ms(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64() * 1000.0
    }
}

// Fallback implementations for when logging is disabled
#[cfg(not(feature = "std"))]
pub fn init_logging(_config: LogConfig) -> Result<(), crate::TinyVlmError> {
    // No-op when std feature is disabled
    Ok(())
}

#[cfg(not(feature = "std"))]
pub fn log_inference_metrics(
    _model_config: &crate::ModelConfig,
    _inference_time_ms: f64,
    _memory_usage_mb: f64,
    _input_tokens: usize,
    _output_tokens: usize,
) {
    // No-op when std feature is disabled
}

#[cfg(not(feature = "std"))]
pub fn log_model_loading(_model_path: &str, _loading_time_ms: f64, _model_size_mb: f64) {
    // No-op when std feature is disabled
}

#[cfg(not(feature = "std"))]
pub fn log_data_processing(
    _operation: &str,
    _samples_processed: usize,
    _processing_time_ms: f64,
    _errors_encountered: usize,
) {
    // No-op when std feature is disabled
}

#[cfg(not(feature = "std"))]
pub fn log_simd_optimization(_operation: &str, _simd_enabled: bool, _speedup_factor: f64) {
    // No-op when std feature is disabled
}

#[cfg(not(feature = "std"))]
pub fn log_memory_event(
    _event: &str,
    _total_memory_mb: f64,
    _allocated_memory_mb: f64,
    _fragmentation_percent: f64,
) {
    // No-op when std feature is disabled
}

#[cfg(not(feature = "std"))]
pub fn log_security_event(_event: &str, _severity: SecuritySeverity, _details: &str) {
    // No-op when std feature is disabled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_config_default() {
        let config = LogConfig::default();
        assert_eq!(config.level, "info");
        assert!(config.include_timestamps);
        assert!(!config.include_thread_info);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_performance_timer() {
        let timer = PerformanceTimer::new("test_operation");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let duration = timer.end();
        assert!(duration >= 10.0);
    }

    #[test]
    fn test_security_severity() {
        // Test that security severity levels exist and can be used
        let _critical = SecuritySeverity::Critical;
        let _high = SecuritySeverity::High;
        let _medium = SecuritySeverity::Medium;
        let _low = SecuritySeverity::Low;
    }
}