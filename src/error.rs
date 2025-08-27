//! Error types for Tiny-VLM with enhanced context and recovery

use thiserror::Error;

#[cfg(feature = "wasm")]
use wasm_bindgen::JsValue;

/// Result type alias for Tiny-VLM operations
pub type Result<T> = core::result::Result<T, TinyVlmError>;

/// Enhanced error context with recovery suggestions
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Timestamp when error occurred
    #[cfg(feature = "std")]
    pub timestamp: std::time::Instant,
    /// Operation that was being performed
    pub operation: String,
    /// Additional context data
    pub context: std::collections::HashMap<String, String>,
    /// Suggested recovery actions
    pub recovery_suggestions: Vec<String>,
    /// Whether this error is retryable
    pub is_retryable: bool,
    /// Error severity level
    pub severity: ErrorSeverity,
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self {
            #[cfg(feature = "std")]
            timestamp: std::time::Instant::now(),
            operation: "unknown".to_string(),
            context: std::collections::HashMap::new(),
            recovery_suggestions: Vec::new(),
            is_retryable: false,
            severity: ErrorSeverity::Medium,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Low severity - informational errors
    Low,
    /// Medium severity - expected operational errors
    Medium,
    /// High severity - significant operational errors
    High,
    /// Critical severity - system-threatening errors
    Critical,
}

/// Error types for Tiny-VLM operations
#[derive(Error, Debug)]
pub enum TinyVlmError {
    /// Image processing errors
    #[error("Image processing error: {0}")]
    ImageProcessing(String),

    /// Text processing errors
    #[error("Text processing error: {0}")]
    TextProcessing(String),

    /// Model loading errors
    #[error("Model loading error: {0}")]
    ModelLoading(String),

    /// Inference errors
    #[error("Inference error: {0}")]
    Inference(String),

    /// Memory allocation errors
    #[error("Memory allocation error: {0}")]
    Memory(String),

    /// Invalid input errors
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// SIMD operation errors
    #[error("SIMD operation error: {0}")]
    Simd(String),

    /// GPU operation errors
    #[cfg(feature = "gpu")]
    #[error("GPU operation error: {0}")]
    Gpu(String),

    /// WebAssembly specific errors
    #[cfg(feature = "wasm")]
    #[error("WebAssembly error: {0}")]
    Wasm(String),

    /// IO errors (when std is available)
    #[cfg(feature = "std")]
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[cfg(feature = "std")]
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Internal system errors
    #[error("Internal error: {0}")]
    InternalError(String),
    
    /// Validation errors
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    /// Circuit breaker open errors
    #[error("Circuit breaker is open: {0}")]
    CircuitBreakerOpen(String),
    
    /// Security errors
    #[error("Security error: {0}")]
    SecurityError(String),
    
    /// Network errors
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Configuration errors (alternative name)
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    /// Service degraded errors
    #[error("Service degraded: {0}")]
    ServiceDegraded(String),
    
    /// Service unavailable errors
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

impl TinyVlmError {
    /// Create an image processing error
    pub fn image_processing(msg: impl Into<String>) -> Self {
        Self::ImageProcessing(msg.into())
    }

    /// Create a text processing error
    pub fn text_processing(msg: impl Into<String>) -> Self {
        Self::TextProcessing(msg.into())
    }

    /// Create a model loading error
    pub fn model_loading(msg: impl Into<String>) -> Self {
        Self::ModelLoading(msg.into())
    }

    /// Create an inference error
    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }

    /// Create a memory error
    pub fn memory(msg: impl Into<String>) -> Self {
        Self::Memory(msg.into())
    }

    /// Create an invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create an invalid configuration error
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a deployment error
    pub fn deployment(msg: impl Into<String>) -> Self {
        Self::InternalError(format!("Deployment error: {}", msg.into()))
    }

    /// Create a SIMD error
    pub fn simd(msg: impl Into<String>) -> Self {
        Self::Simd(msg.into())
    }

    /// Create a GPU error
    #[cfg(feature = "gpu")]
    pub fn gpu(msg: impl Into<String>) -> Self {
        Self::Gpu(msg.into())
    }

    /// Create a WebAssembly error
    #[cfg(feature = "wasm")]
    pub fn wasm(msg: impl Into<String>) -> Self {
        Self::Wasm(msg.into())
    }

    /// Create a serialization error
    #[cfg(feature = "std")]
    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::SerializationError(msg.into())
    }
    
    /// Create an internal error
    pub fn internal_error(msg: impl Into<String>) -> Self {
        Self::InternalError(msg.into())
    }
    
    /// Create a validation error
    pub fn validation_error(msg: impl Into<String>) -> Self {
        Self::ValidationError(msg.into())
    }
    
    /// Create a circuit breaker error
    pub fn circuit_breaker_open(msg: impl Into<String>) -> Self {
        Self::CircuitBreakerOpen(msg.into())
    }
    
    /// Create a security error
    pub fn security_error(msg: impl Into<String>) -> Self {
        Self::SecurityError(msg.into())
    }
    
    /// Create a network error
    pub fn network_error(msg: impl Into<String>) -> Self {
        Self::NetworkError(msg.into())
    }
    
    /// Create a configuration error (alternative method)
    pub fn configuration_error(msg: impl Into<String>) -> Self {
        Self::ConfigurationError(msg.into())
    }
    
    /// Create a service degraded error
    pub fn service_degraded(msg: impl Into<String>) -> Self {
        Self::ServiceDegraded(msg.into())
    }
    
    /// Create a service unavailable error
    pub fn service_unavailable(msg: impl Into<String>) -> Self {
        Self::ServiceUnavailable(msg.into())
    }

    /// Create an error with enhanced context
    pub fn with_context(self, context: ErrorContext) -> Self {
        #[cfg(feature = "std")]
        {
            // Log the error with context for debugging
            crate::logging::log_security_event(
                "error_with_context",
                match context.severity {
                    ErrorSeverity::Low => crate::logging::SecuritySeverity::Low,
                    ErrorSeverity::Medium => crate::logging::SecuritySeverity::Medium,
                    ErrorSeverity::High => crate::logging::SecuritySeverity::High,
                    ErrorSeverity::Critical => crate::logging::SecuritySeverity::Critical,
                },
                &format!("Error in operation '{}': {}", context.operation, self),
            );
        }
        self
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::ImageProcessing(_) => ErrorSeverity::Medium,
            Self::TextProcessing(_) => ErrorSeverity::Medium,
            Self::ModelLoading(_) => ErrorSeverity::High,
            Self::Inference(_) => ErrorSeverity::Medium,
            Self::Memory(_) => ErrorSeverity::High,
            Self::InvalidInput(_) => ErrorSeverity::Low,
            Self::Config(_) => ErrorSeverity::High,
            Self::Simd(_) => ErrorSeverity::Low,
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => ErrorSeverity::Medium,
            #[cfg(feature = "wasm")]
            Self::Wasm(_) => ErrorSeverity::Medium,
            #[cfg(feature = "std")]
            Self::Io(_) => ErrorSeverity::Medium,
            #[cfg(feature = "std")]
            Self::SerializationError(_) => ErrorSeverity::Medium,
            Self::InternalError(_) => ErrorSeverity::High,
            Self::ValidationError(_) => ErrorSeverity::Low,
            Self::CircuitBreakerOpen(_) => ErrorSeverity::Medium,
            Self::SecurityError(_) => ErrorSeverity::Critical,
            Self::NetworkError(_) => ErrorSeverity::Medium,
            Self::ConfigurationError(_) => ErrorSeverity::High,
            Self::ServiceDegraded(_) => ErrorSeverity::Medium,
            Self::ServiceUnavailable(_) => ErrorSeverity::High,
        }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::ImageProcessing(_) => false, // Usually input format issues
            Self::TextProcessing(_) => false,  // Usually input format issues
            Self::ModelLoading(_) => true,     // Could be transient I/O issue
            Self::Inference(_) => true,        // Could be temporary resource issue
            Self::Memory(_) => true,           // Could free up memory and retry
            Self::InvalidInput(_) => false,    // Invalid input won't change
            Self::Config(_) => false,          // Configuration issues need fixing
            Self::Simd(_) => false,            // Architecture issues
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => true,              // GPU driver/memory issues can be retried
            #[cfg(feature = "wasm")]
            Self::Wasm(_) => true,             // Could be temporary WASM issue
            #[cfg(feature = "std")]
            Self::Io(_) => true,               // I/O operations can be retried
            #[cfg(feature = "std")]
            Self::SerializationError(_) => false, // Data format issues
            Self::InternalError(_) => true,       // Internal errors might be transient
            Self::ValidationError(_) => false,    // Validation errors need input fixing
            Self::CircuitBreakerOpen(_) => true,  // Circuit breaker can reset
            Self::SecurityError(_) => false,      // Security issues need fixing
            Self::NetworkError(_) => true,        // Network issues are often transient
            Self::ConfigurationError(_) => false, // Configuration issues need fixing
            Self::ServiceDegraded(_) => true,     // Service may recover
            Self::ServiceUnavailable(_) => true,  // Service may become available
        }
    }

    /// Get recovery suggestions for this error
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            Self::ImageProcessing(_) => vec![
                "Verify image format is supported (PNG, JPEG, WebP, GIF)".to_string(),
                "Check image size is within limits (10MB max)".to_string(),
                "Ensure image dimensions are valid".to_string(),
            ],
            Self::TextProcessing(_) => vec![
                "Check text length is within limits (1MB max)".to_string(),
                "Verify text contains valid UTF-8 characters".to_string(),
                "Remove null bytes or control characters".to_string(),
            ],
            Self::ModelLoading(_) => vec![
                "Verify model file exists and is accessible".to_string(),
                "Check available disk space and memory".to_string(),
                "Try loading model again after a brief delay".to_string(),
            ],
            Self::Inference(_) => vec![
                "Reduce batch size or input length".to_string(),
                "Free up system memory".to_string(),
                "Try inference again with simplified input".to_string(),
            ],
            Self::Memory(_) => vec![
                "Reduce memory usage by compacting memory pool".to_string(),
                "Lower memory limits in configuration".to_string(),
                "Process smaller batches".to_string(),
            ],
            Self::InvalidInput(_) => vec![
                "Validate input data before processing".to_string(),
                "Check input format matches expected schema".to_string(),
                "Use input validation functions".to_string(),
            ],
            Self::Config(_) => vec![
                "Review configuration parameters".to_string(),
                "Check configuration file syntax".to_string(),
                "Use default configuration as fallback".to_string(),
            ],
            Self::Simd(_) => vec![
                "Check CPU architecture supports required SIMD instructions".to_string(),
                "Fall back to scalar implementations".to_string(),
                "Update processor microcode if available".to_string(),
            ],
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => vec![
                "Check GPU drivers are up to date".to_string(),
                "Verify sufficient GPU memory available".to_string(),
                "Try reducing GPU batch size".to_string(),
                "Fall back to CPU computation".to_string(),
            ],
            #[cfg(feature = "wasm")]
            Self::Wasm(_) => vec![
                "Check WebAssembly runtime supports required features".to_string(),
                "Verify browser/runtime version compatibility".to_string(),
                "Try reloading the WASM module".to_string(),
            ],
            #[cfg(feature = "std")]
            Self::Io(_) => vec![
                "Check file permissions and accessibility".to_string(),
                "Verify sufficient disk space".to_string(),
                "Retry the operation after a brief delay".to_string(),
            ],
            #[cfg(feature = "std")]
            Self::SerializationError(_) => vec![
                "Verify data format matches expected schema".to_string(),
                "Check for data corruption".to_string(),
                "Try alternative serialization format".to_string(),
            ],
            Self::InternalError(_) => vec![
                "Report this issue to support".to_string(),
                "Restart the application".to_string(),
                "Check system logs for more details".to_string(),
            ],
            Self::ValidationError(_) => vec![
                "Check input data format and constraints".to_string(),
                "Verify data integrity".to_string(),
                "Use validation functions before processing".to_string(),
            ],
            Self::CircuitBreakerOpen(_) => vec![
                "Wait for circuit breaker to reset".to_string(),
                "Check underlying service health".to_string(),
                "Reduce request rate temporarily".to_string(),
            ],
            Self::SecurityError(_) => vec![
                "Review security configuration".to_string(),
                "Check access permissions".to_string(),
                "Verify authentication credentials".to_string(),
            ],
            Self::NetworkError(_) => vec![
                "Check network connectivity".to_string(),
                "Verify network configuration".to_string(),
                "Retry with exponential backoff".to_string(),
            ],
            Self::ConfigurationError(_) => vec![
                "Review configuration parameters".to_string(),
                "Check configuration file syntax".to_string(),
                "Use default configuration as fallback".to_string(),
            ],
            Self::ServiceDegraded(_) => vec![
                "Wait for service to recover".to_string(),
                "Use alternative service endpoint".to_string(),
            ],
            Self::ServiceUnavailable(_) => vec![
                "Retry after delay".to_string(),
                "Use fallback service".to_string(),
            ],
        }
    }
}

#[cfg(feature = "wasm")]
impl From<TinyVlmError> for JsValue {
    fn from(error: TinyVlmError) -> Self {
        JsValue::from_str(&format!("{}", error))
    }
}

#[cfg(feature = "wasm")]
impl From<JsValue> for TinyVlmError {
    fn from(js_value: JsValue) -> Self {
        let error_msg = if let Some(s) = js_value.as_string() {
            s
        } else {
            "Unknown JavaScript error".to_string()
        };
        TinyVlmError::Wasm(error_msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = TinyVlmError::image_processing("test error");
        assert!(matches!(err, TinyVlmError::ImageProcessing(_)));
    }

    #[test]
    fn test_error_display() {
        let err = TinyVlmError::inference("test inference error");
        let error_string = format!("{}", err);
        assert!(error_string.contains("Inference error"));
        assert!(error_string.contains("test inference error"));
    }
}