//! Input validation and security checks for Tiny-VLM
//!
//! Provides comprehensive validation for all public API inputs to ensure security and correctness.

use crate::{Result, TinyVlmError};
#[cfg(feature = "std")]
use crate::logging::{log_security_event, SecuritySeverity};

/// Maximum allowed image size (10MB)
pub const MAX_IMAGE_SIZE_BYTES: usize = 10 * 1024 * 1024;

/// Maximum allowed text length (1MB of text)
pub const MAX_TEXT_LENGTH_CHARS: usize = 1024 * 1024;

/// Maximum allowed model dimensions
pub const MAX_MODEL_DIMENSION: usize = 8192;

/// Minimum allowed model dimensions
pub const MIN_MODEL_DIMENSION: usize = 32;

/// Maximum allowed sequence length
pub const MAX_SEQUENCE_LENGTH: usize = 32768;

/// Maximum allowed batch size
pub const MAX_BATCH_SIZE: usize = 1024;

/// Maximum allowed memory limit (2GB) - reduced to prevent overflow on 32-bit systems
pub const MAX_MEMORY_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Validation result with detailed error information
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors if any
    pub errors: Vec<String>,
    /// Security warnings if any
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a successful validation result
    pub fn success() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create a failed validation result with error
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            is_valid: false,
            errors: vec![message.into()],
            warnings: Vec::new(),
        }
    }

    /// Add an error to the validation result
    pub fn add_error(&mut self, message: impl Into<String>) {
        self.errors.push(message.into());
        self.is_valid = false;
    }

    /// Add a warning to the validation result
    pub fn add_warning(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }

    /// Convert to Result type
    pub fn into_result(self) -> Result<()> {
        if self.is_valid {
            Ok(())
        } else {
            Err(TinyVlmError::invalid_input(
                self.errors.join("; ")
            ))
        }
    }
}

/// Validate image data input
pub fn validate_image_data(data: &[u8]) -> ValidationResult {
    let mut result = ValidationResult::success();

    // Check size limits
    if data.is_empty() {
        result.add_error("Image data cannot be empty");
    }

    if data.len() > MAX_IMAGE_SIZE_BYTES {
        result.add_error(format!(
            "Image size {} bytes exceeds maximum of {} bytes",
            data.len(),
            MAX_IMAGE_SIZE_BYTES
        ));
    }

    // Basic image format validation (check for common headers)
    if data.len() >= 4 {
        let header = &data[0..4];
        let is_valid_format = 
            header == b"\xFF\xD8\xFF" || // JPEG
            header.starts_with(b"\x89PNG") || // PNG
            header.starts_with(b"RIFF") || // WebP (RIFF container)
            header.starts_with(b"GIF8"); // GIF
        
        if !is_valid_format {
            result.add_warning("Image format may not be supported");
        }
    }

    // Security check for suspicious patterns
    if contains_suspicious_patterns(data) {
        #[cfg(feature = "std")]
        log_security_event(
            "suspicious_image_data",
            SecuritySeverity::Medium,
            "Image data contains suspicious binary patterns"
        );
        result.add_warning("Image data contains suspicious patterns");
    }

    result
}

/// Validate text input
pub fn validate_text_input(text: &str) -> ValidationResult {
    let mut result = ValidationResult::success();

    // Check length limits
    if text.len() > MAX_TEXT_LENGTH_CHARS {
        result.add_error(format!(
            "Text length {} exceeds maximum of {} characters",
            text.len(),
            MAX_TEXT_LENGTH_CHARS
        ));
    }

    // Check for null bytes (potential security risk)
    if text.contains('\0') {
        #[cfg(feature = "std")]
        log_security_event(
            "null_byte_in_text",
            SecuritySeverity::High,
            "Text input contains null bytes"
        );
        result.add_error("Text input cannot contain null bytes");
    }

    // Check for extremely long lines (potential DoS vector)
    let max_line_length = 10000;
    for (line_num, line) in text.lines().enumerate() {
        if line.len() > max_line_length {
            result.add_warning(format!(
                "Line {} has length {} which may cause processing issues",
                line_num + 1,
                line.len()
            ));
        }
    }

    // Check for excessive whitespace (potential DoS vector)
    let whitespace_ratio = text.chars().filter(|c| c.is_whitespace()).count() as f64 / text.len() as f64;
    if whitespace_ratio > 0.95 {
        result.add_warning("Text is mostly whitespace");
    }

    // Check for potentially malicious Unicode sequences
    if contains_malicious_unicode(text) {
        #[cfg(feature = "std")]
        log_security_event(
            "malicious_unicode",
            SecuritySeverity::Medium,
            "Text contains potentially malicious Unicode sequences"
        );
        result.add_warning("Text contains potentially problematic Unicode sequences");
    }

    result
}

/// Validate model configuration
pub fn validate_model_config(config: &crate::ModelConfig) -> ValidationResult {
    let mut result = ValidationResult::success();

    // Check dimension bounds
    if config.vision_dim < MIN_MODEL_DIMENSION || config.vision_dim > MAX_MODEL_DIMENSION {
        result.add_error(format!(
            "Vision dimension {} must be between {} and {}",
            config.vision_dim, MIN_MODEL_DIMENSION, MAX_MODEL_DIMENSION
        ));
    }

    if config.text_dim < MIN_MODEL_DIMENSION || config.text_dim > MAX_MODEL_DIMENSION {
        result.add_error(format!(
            "Text dimension {} must be between {} and {}",
            config.text_dim, MIN_MODEL_DIMENSION, MAX_MODEL_DIMENSION
        ));
    }

    if config.hidden_dim < MIN_MODEL_DIMENSION || config.hidden_dim > MAX_MODEL_DIMENSION {
        result.add_error(format!(
            "Hidden dimension {} must be between {} and {}",
            config.hidden_dim, MIN_MODEL_DIMENSION, MAX_MODEL_DIMENSION
        ));
    }

    // Check attention heads
    if config.num_heads == 0 || config.num_heads > 64 {
        result.add_error(format!(
            "Number of attention heads {} must be between 1 and 64",
            config.num_heads
        ));
    }

    // Check that hidden_dim is divisible by num_heads
    if config.hidden_dim % config.num_heads != 0 {
        result.add_error("Hidden dimension must be divisible by number of attention heads");
    }

    // Check generation parameters
    if config.max_gen_length > MAX_SEQUENCE_LENGTH {
        result.add_error(format!(
            "Maximum generation length {} exceeds limit of {}",
            config.max_gen_length, MAX_SEQUENCE_LENGTH
        ));
    }

    if config.temperature <= 0.0 || config.temperature > 2.0 {
        result.add_warning(format!(
            "Temperature {} is outside typical range (0.0, 2.0]",
            config.temperature
        ));
    }

    result
}

/// Validate inference configuration
pub fn validate_inference_config(config: &crate::InferenceConfig) -> ValidationResult {
    let mut result = ValidationResult::success();

    // Check sequence length limits
    if config.max_length > MAX_SEQUENCE_LENGTH {
        result.add_error(format!(
            "Maximum sequence length {} exceeds limit of {}",
            config.max_length, MAX_SEQUENCE_LENGTH
        ));
    }

    // Check sampling parameters
    if config.temperature <= 0.0 || config.temperature > 5.0 {
        result.add_error(format!(
            "Temperature {} must be positive and reasonable (typically 0.1-2.0)",
            config.temperature
        ));
    }

    if config.top_p <= 0.0 || config.top_p > 1.0 {
        result.add_error(format!(
            "Top-p {} must be between 0.0 and 1.0",
            config.top_p
        ));
    }

    if config.top_k == 0 {
        result.add_warning("Top-k of 0 disables top-k sampling");
    }

    if config.top_k > 1000 {
        result.add_warning(format!(
            "Top-k {} is very large and may impact performance",
            config.top_k
        ));
    }

    // Check memory limits
    if config.memory_limit_mb > MAX_MEMORY_LIMIT_BYTES / (1024 * 1024) {
        result.add_error(format!(
            "Memory limit {}MB exceeds system maximum of {}MB",
            config.memory_limit_mb,
            MAX_MEMORY_LIMIT_BYTES / (1024 * 1024)
        ));
    }

    result
}

/// Validate tensor dimensions
pub fn validate_tensor_dimensions(dims: &[usize]) -> ValidationResult {
    let mut result = ValidationResult::success();

    if dims.is_empty() {
        result.add_error("Tensor must have at least one dimension");
        return result;
    }

    if dims.len() > 8 {
        result.add_error("Tensor cannot have more than 8 dimensions");
    }

    for &dim in dims {
        if dim == 0 {
            result.add_error("Tensor dimensions cannot be zero");
        }

        if dim > 1024 * 1024 {
            result.add_warning(format!(
                "Large tensor dimension {} may cause memory issues",
                dim
            ));
        }
    }

    // Check for potential overflow in total elements
    let total_elements: usize = dims.iter().product();
    if total_elements > 1_000_000_000 {
        result.add_error("Tensor would have too many elements (> 1 billion)");
    }

    result
}

/// Validate file path for security
pub fn validate_file_path(path: &str) -> ValidationResult {
    let mut result = ValidationResult::success();

    // Check for path traversal attempts
    if path.contains("..") || path.contains("./") || path.contains("\\") {
        #[cfg(feature = "std")]
        log_security_event(
            "path_traversal_attempt",
            SecuritySeverity::High,
            &format!("Potential path traversal in: {}", path)
        );
        result.add_error("File path contains unsafe traversal sequences");
    }

    // Check for absolute paths (potential security risk)
    if path.starts_with('/') || (path.len() > 1 && path.chars().nth(1) == Some(':')) {
        result.add_warning("Absolute file paths may be a security risk");
    }

    // Check for null bytes
    if path.contains('\0') {
        result.add_error("File path cannot contain null bytes");
    }

    // Check path length
    if path.len() > 260 {
        result.add_warning("File path is very long and may not be supported on all systems");
    }

    result
}

/// Check for suspicious binary patterns in image data
fn contains_suspicious_patterns(data: &[u8]) -> bool {
    // Check for embedded scripts or executables
    let suspicious_patterns: &[&[u8]] = &[
        b"<script", b"javascript", b"exec", b"eval",
        b"MZ", // PE executable header
        b"\x7fELF", // ELF executable header
        b"\xCA\xFE\xBA\xBE", // Java class file
    ];

    for pattern in suspicious_patterns {
        if data.windows(pattern.len()).any(|window| window == *pattern) {
            return true;
        }
    }

    false
}

/// Check for potentially malicious Unicode sequences
fn contains_malicious_unicode(text: &str) -> bool {
    for ch in text.chars() {
        // Check for right-to-left override characters (used in spoofing attacks)
        if matches!(ch, '\u{202E}' | '\u{202D}' | '\u{200F}' | '\u{200E}') {
            return true;
        }

        // Check for zero-width characters (potential for hiding content)
        if matches!(ch, '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}') {
            return true;
        }

        // Check for control characters (except common whitespace)
        if ch.is_control() && !matches!(ch, '\n' | '\r' | '\t') {
            return true;
        }
    }

    false
}

/// Sanitize text input by removing potentially dangerous sequences
pub fn sanitize_text_input(text: &str) -> String {
    text.chars()
        .filter(|&ch| {
            // Keep printable ASCII, common whitespace, and safe Unicode
            ch.is_ascii_graphic() ||
            matches!(ch, ' ' | '\n' | '\r' | '\t') ||
            (ch.is_alphabetic() && ch as u32 <= 0x1F6FF) // Basic multilingual plane minus some problematic ranges
        })
        .collect()
}

/// Rate limiting state for API calls
#[cfg(feature = "std")]
pub struct RateLimiter {
    requests: std::collections::VecDeque<std::time::Instant>,
    max_requests: usize,
    window_duration: std::time::Duration,
}

#[cfg(feature = "std")]
impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(max_requests: usize, window_duration: std::time::Duration) -> Self {
        Self {
            requests: std::collections::VecDeque::new(),
            max_requests,
            window_duration,
        }
    }

    /// Check if request is allowed and update state
    pub fn is_allowed(&mut self) -> bool {
        let now = std::time::Instant::now();
        
        // Remove old requests outside the window
        while let Some(&front_time) = self.requests.front() {
            if now.duration_since(front_time) <= self.window_duration {
                break;
            }
            self.requests.pop_front();
        }

        // Check if we can accept the new request
        if self.requests.len() < self.max_requests {
            self.requests.push_back(now);
            true
        } else {
            #[cfg(feature = "std")]
            log_security_event(
                "rate_limit_exceeded",
                SecuritySeverity::Medium,
                "Request rate limit exceeded"
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_image_data() {
        // Test empty data
        let result = validate_image_data(&[]);
        assert!(!result.is_valid);

        // Test valid JPEG header
        let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG header
        let result = validate_image_data(&jpeg_data);
        assert!(result.is_valid);

        // Test oversized data
        let large_data = vec![0u8; MAX_IMAGE_SIZE_BYTES + 1];
        let result = validate_image_data(&large_data);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_text_input() {
        // Test normal text
        let result = validate_text_input("Hello, world!");
        assert!(result.is_valid);

        // Test text with null byte
        let result = validate_text_input("Hello\0world");
        assert!(!result.is_valid);

        // Test oversized text
        let large_text = "x".repeat(MAX_TEXT_LENGTH_CHARS + 1);
        let result = validate_text_input(&large_text);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_tensor_dimensions() {
        // Test valid dimensions
        let result = validate_tensor_dimensions(&[224, 224, 3]);
        assert!(result.is_valid);

        // Test empty dimensions
        let result = validate_tensor_dimensions(&[]);
        assert!(!result.is_valid);

        // Test zero dimension
        let result = validate_tensor_dimensions(&[224, 0, 3]);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_file_path() {
        // Test safe path
        let result = validate_file_path("models/tiny_vlm.bin");
        assert!(result.is_valid);

        // Test path traversal
        let result = validate_file_path("../../../etc/passwd");
        assert!(!result.is_valid);

        // Test null byte
        let result = validate_file_path("model\0.bin");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_sanitize_text_input() {
        let dirty_text = "Hello\u{202E}world\0test";
        let clean_text = sanitize_text_input(dirty_text);
        assert!(!clean_text.contains('\0'));
        assert!(!clean_text.contains('\u{202E}'));
        assert!(clean_text.contains("Hello"));
        assert!(clean_text.contains("world"));
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(3, std::time::Duration::from_secs(1));
        
        // Should allow first 3 requests
        assert!(limiter.is_allowed());
        assert!(limiter.is_allowed());
        assert!(limiter.is_allowed());
        
        // Should block 4th request
        assert!(!limiter.is_allowed());
    }
}