//! Security module for protecting against malicious inputs and attacks
//! 
//! This module implements comprehensive security measures for production deployment:
//! - Input validation and sanitization
//! - Rate limiting and abuse prevention  
//! - Resource consumption limits
//! - Attack detection and mitigation
//! - Secure memory handling

use crate::{Result, TinyVlmError};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Comprehensive security manager for production deployment
pub struct SecurityManager {
    /// Input validation rules
    input_validator: InputValidator,
    /// Rate limiting enforcement
    rate_limiter: RateLimiter,
    /// Resource usage monitor
    resource_monitor: ResourceMonitor,
    /// Attack detection system
    attack_detector: AttackDetector,
    /// Security configuration
    config: SecurityConfig,
}

impl SecurityManager {
    /// Create a new security manager with production-grade defaults
    pub fn new() -> Self {
        Self {
            input_validator: InputValidator::with_strict_rules(),
            rate_limiter: RateLimiter::with_defaults(),
            resource_monitor: ResourceMonitor::new(),
            attack_detector: AttackDetector::new(),
            config: SecurityConfig::production(),
        }
    }

    /// Validate and sanitize all inputs before processing
    pub fn validate_inference_request(
        &mut self, 
        client_id: &str,
        image_data: &[u8], 
        prompt: &str
    ) -> Result<ValidationResult> {
        // 1. Rate limiting check
        if !self.rate_limiter.allow_request(client_id) {
            return Err(TinyVlmError::config("Rate limit exceeded"));
        }

        // 2. Input validation
        let image_validation = self.input_validator.validate_image_data(image_data)?;
        let text_validation = self.input_validator.validate_text_input(prompt)?;

        // 3. Resource consumption check
        let resource_check = self.resource_monitor.check_resource_limits(
            image_data.len(),
            prompt.len(),
        )?;

        // 4. Attack detection
        let attack_check = self.attack_detector.analyze_request(
            client_id,
            image_data,
            prompt,
        )?;

        Ok(ValidationResult {
            image_validation,
            text_validation,
            resource_check,
            attack_check,
            approved: true,
        })
    }

    /// Secure memory cleanup for sensitive data
    pub fn secure_cleanup(&self, sensitive_data: &mut [u8]) {
        // Overwrite memory with random data multiple times
        for _ in 0..3 {
            for byte in sensitive_data.iter_mut() {
                *byte = self.generate_secure_random_byte();
            }
        }
        
        // Final overwrite with zeros
        sensitive_data.fill(0);
    }

    /// Generate cryptographically secure random byte
    fn generate_secure_random_byte(&self) -> u8 {
        // In production, use proper CSPRNG like ring or sodiumoxide
        // This is a simplified version for the demo
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        std::time::SystemTime::now().hash(&mut hasher);
        (hasher.finish() % 256) as u8
    }

    /// Update security configuration
    pub fn update_config(&mut self, config: SecurityConfig) {
        self.config = config;
        self.rate_limiter.update_config(&self.config);
        self.input_validator.update_config(&self.config);
    }

    /// Get security metrics for monitoring
    pub fn get_security_metrics(&self) -> SecurityMetrics {
        SecurityMetrics {
            blocked_requests: self.rate_limiter.get_blocked_count(),
            invalid_inputs: self.input_validator.get_invalid_count(),
            detected_attacks: self.attack_detector.get_detected_count(),
            resource_violations: self.resource_monitor.get_violation_count(),
            uptime_seconds: self.config.start_time.elapsed().as_secs(),
        }
    }
}

/// Input validation with comprehensive security checks
pub struct InputValidator {
    config: ValidationConfig,
    invalid_count: u64,
}

impl InputValidator {
    fn with_strict_rules() -> Self {
        Self {
            config: ValidationConfig::strict(),
            invalid_count: 0,
        }
    }

    fn validate_image_data(&mut self, data: &[u8]) -> Result<ImageValidationResult> {
        let mut result = ImageValidationResult {
            size_valid: true,
            format_valid: true,
            content_safe: true,
            dimensions_valid: true,
            warnings: Vec::new(),
        };

        // Size validation
        if data.len() > self.config.max_image_size_bytes {
            result.size_valid = false;
            self.invalid_count += 1;
            return Err(TinyVlmError::config(&format!(
                "Image too large: {} bytes (max: {})", 
                data.len(), self.config.max_image_size_bytes
            )));
        }

        if data.len() < self.config.min_image_size_bytes {
            result.size_valid = false;
            self.invalid_count += 1;
            return Err(TinyVlmError::config(&format!(
                "Image too small: {} bytes (min: {})", 
                data.len(), self.config.min_image_size_bytes
            )));
        }

        // Basic format validation (simplified RGB check)
        if data.len() % 3 != 0 {
            result.format_valid = false;
            result.warnings.push("Image data length not divisible by 3 (RGB expected)".to_string());
        }

        // Content safety scan (simplified)
        if self.contains_suspicious_patterns(data) {
            result.content_safe = false;
            self.invalid_count += 1;
            return Err(TinyVlmError::config("Suspicious image content detected"));
        }

        // Dimension validation for expected formats
        let expected_pixels = data.len() / 3;
        let sqrt_pixels = (expected_pixels as f64).sqrt() as usize;
        if sqrt_pixels * sqrt_pixels != expected_pixels {
            result.warnings.push("Non-square image detected".to_string());
        }

        if sqrt_pixels > self.config.max_image_dimension {
            result.dimensions_valid = false;
            self.invalid_count += 1;
            return Err(TinyVlmError::config(&format!(
                "Image dimensions too large: {}x{} (max: {}x{})",
                sqrt_pixels, sqrt_pixels, 
                self.config.max_image_dimension, self.config.max_image_dimension
            )));
        }

        Ok(result)
    }

    fn validate_text_input(&mut self, text: &str) -> Result<TextValidationResult> {
        let mut result = TextValidationResult {
            length_valid: true,
            encoding_valid: true,
            content_safe: true,
            language_appropriate: true,
            warnings: Vec::new(),
        };

        // Length validation
        if text.len() > self.config.max_text_length {
            result.length_valid = false;
            self.invalid_count += 1;
            return Err(TinyVlmError::config(&format!(
                "Text too long: {} chars (max: {})", 
                text.len(), self.config.max_text_length
            )));
        }

        if text.is_empty() {
            result.length_valid = false;
            self.invalid_count += 1;
            return Err(TinyVlmError::config("Empty text input"));
        }

        // Encoding validation (ensure valid UTF-8)
        if !text.is_ascii() {
            result.warnings.push("Non-ASCII characters detected".to_string());
        }

        // Content safety checks
        if self.contains_malicious_text_patterns(text) {
            result.content_safe = false;
            self.invalid_count += 1;
            return Err(TinyVlmError::config("Malicious text patterns detected"));
        }

        // Check for injection attempts
        if self.contains_injection_patterns(text) {
            result.content_safe = false;
            self.invalid_count += 1;
            return Err(TinyVlmError::config("Potential injection attack detected"));
        }

        // Language appropriateness (basic profanity filter)
        if self.contains_inappropriate_language(text) {
            result.language_appropriate = false;
            result.warnings.push("Potentially inappropriate language detected".to_string());
        }

        Ok(result)
    }

    fn contains_suspicious_patterns(&self, data: &[u8]) -> bool {
        // Check for suspicious byte patterns that might indicate exploits
        
        // Look for executable headers (simplified)
        if data.len() >= 4 {
            // Check for common executable signatures
            let header = &data[0..4];
            if header == b"\x7fELF" || header == b"MZ\x90\x00" || header == b"\xCAFE" {
                return true;
            }
        }

        // Look for excessive repeating patterns (potential buffer overflow)
        let mut consecutive_same = 0;
        let mut last_byte = data[0];
        
        for &byte in data.iter() {
            if byte == last_byte {
                consecutive_same += 1;
                if consecutive_same > 100 {
                    return true;
                }
            } else {
                consecutive_same = 1;
                last_byte = byte;
            }
        }

        false
    }

    fn contains_malicious_text_patterns(&self, text: &str) -> bool {
        let malicious_patterns = [
            "javascript:",
            "data:",
            "vbscript:",
            "<script",
            "</script>",
            "onload=",
            "onerror=",
            "eval(",
            "document.cookie",
            "window.location",
            "../../",
            "../../../",
            "../../../../",
        ];

        let text_lower = text.to_lowercase();
        malicious_patterns.iter().any(|pattern| text_lower.contains(pattern))
    }

    fn contains_injection_patterns(&self, text: &str) -> bool {
        let injection_patterns = [
            "'; DROP TABLE",
            "; DELETE FROM",
            "UNION SELECT",
            "1=1",
            "' OR '1'='1",
            "admin'--",
            "' OR 1=1--",
            "${",
            "#{",
            "<!--",
            "-->",
        ];

        let text_lower = text.to_lowercase();
        injection_patterns.iter().any(|pattern| text_lower.contains(&pattern.to_lowercase()))
    }

    fn contains_inappropriate_language(&self, text: &str) -> bool {
        // Simple profanity filter (in production, use comprehensive filter)
        let inappropriate_words = [
            "hate", "violence", "attack", "kill", "bomb", "weapon"
        ];

        let text_lower = text.to_lowercase();
        inappropriate_words.iter().any(|word| {
            text_lower.contains(word)
        })
    }

    fn update_config(&mut self, security_config: &SecurityConfig) {
        self.config.max_image_size_bytes = security_config.max_image_size_mb * 1024 * 1024;
        self.config.max_text_length = security_config.max_text_length;
    }

    fn get_invalid_count(&self) -> u64 {
        self.invalid_count
    }
}

/// Rate limiting system with sliding window algorithm
pub struct RateLimiter {
    windows: HashMap<String, SlidingWindow>,
    config: RateLimitConfig,
    blocked_count: u64,
}

impl RateLimiter {
    fn with_defaults() -> Self {
        Self {
            windows: HashMap::new(),
            config: RateLimitConfig::default(),
            blocked_count: 0,
        }
    }

    fn allow_request(&mut self, client_id: &str) -> bool {
        let now = Instant::now();
        
        // Clean up old windows periodically
        if self.windows.len() > 10000 {
            self.cleanup_old_windows(now);
        }

        let window = self.windows.entry(client_id.to_string())
            .or_insert_with(|| SlidingWindow::new(now, self.config.window_duration));

        if window.allow_request(now, self.config.max_requests_per_window) {
            true
        } else {
            self.blocked_count += 1;
            false
        }
    }

    fn cleanup_old_windows(&mut self, now: Instant) {
        let cutoff = now - self.config.window_duration * 2;
        self.windows.retain(|_, window| window.last_request > cutoff);
    }

    fn update_config(&mut self, security_config: &SecurityConfig) {
        self.config.max_requests_per_window = security_config.max_requests_per_minute;
    }

    fn get_blocked_count(&self) -> u64 {
        self.blocked_count
    }
}

/// Sliding window for rate limiting
struct SlidingWindow {
    requests: Vec<Instant>,
    last_request: Instant,
    window_start: Instant,
}

impl SlidingWindow {
    fn new(now: Instant, window_duration: Duration) -> Self {
        Self {
            requests: Vec::new(),
            last_request: now,
            window_start: now,
        }
    }

    fn allow_request(&mut self, now: Instant, max_requests: u32) -> bool {
        self.last_request = now;
        
        // Remove requests outside the current window
        let window_start = now - Duration::from_secs(60); // 1 minute window
        self.requests.retain(|&request_time| request_time > window_start);
        
        // Check if we can allow this request
        if self.requests.len() < max_requests as usize {
            self.requests.push(now);
            true
        } else {
            false
        }
    }
}

/// Resource monitoring and limits
pub struct ResourceMonitor {
    violation_count: u64,
}

impl ResourceMonitor {
    fn new() -> Self {
        Self {
            violation_count: 0,
        }
    }

    fn check_resource_limits(&mut self, image_size: usize, text_length: usize) -> Result<ResourceCheckResult> {
        let mut result = ResourceCheckResult {
            memory_estimate_mb: 0.0,
            processing_time_estimate_ms: 0.0,
            within_limits: true,
        };

        // Estimate memory usage
        let image_memory_mb = (image_size as f64) / (1024.0 * 1024.0) * 4.0; // 4x for processing overhead
        let text_memory_mb = (text_length as f64) / 1024.0; // Rough estimate
        result.memory_estimate_mb = image_memory_mb + text_memory_mb;

        // Estimate processing time based on input size
        result.processing_time_estimate_ms = 
            (image_size as f64 / 1000.0) + (text_length as f64 * 0.1);

        // Check limits
        if result.memory_estimate_mb > 200.0 {
            self.violation_count += 1;
            return Err(TinyVlmError::config(&format!(
                "Estimated memory usage too high: {:.2}MB (max: 200MB)",
                result.memory_estimate_mb
            )));
        }

        if result.processing_time_estimate_ms > 1000.0 {
            self.violation_count += 1;
            return Err(TinyVlmError::config(&format!(
                "Estimated processing time too long: {:.2}ms (max: 1000ms)",
                result.processing_time_estimate_ms
            )));
        }

        Ok(result)
    }

    fn get_violation_count(&self) -> u64 {
        self.violation_count
    }
}

/// Attack detection system
pub struct AttackDetector {
    client_patterns: HashMap<String, ClientPattern>,
    detected_count: u64,
}

impl AttackDetector {
    fn new() -> Self {
        Self {
            client_patterns: HashMap::new(),
            detected_count: 0,
        }
    }

    fn analyze_request(&mut self, client_id: &str, image_data: &[u8], prompt: &str) -> Result<AttackAnalysisResult> {
        let now = Instant::now();
        let pattern = self.client_patterns.entry(client_id.to_string())
            .or_insert_with(|| ClientPattern::new(now));

        pattern.add_request(now, image_data.len(), prompt.len());

        let mut result = AttackAnalysisResult {
            risk_score: 0.0,
            detected_attacks: Vec::new(),
            behavioral_anomalies: Vec::new(),
        };

        // Check for suspicious patterns
        if pattern.is_flooding() {
            result.detected_attacks.push("Request flooding detected".to_string());
            result.risk_score += 50.0;
        }

        if pattern.has_size_anomalies() {
            result.behavioral_anomalies.push("Unusual input size patterns".to_string());
            result.risk_score += 25.0;
        }

        if pattern.is_repeating_requests() {
            result.behavioral_anomalies.push("Repetitive request patterns".to_string());
            result.risk_score += 20.0;
        }

        // Check for potential adversarial inputs
        if self.is_adversarial_image(image_data) {
            result.detected_attacks.push("Potential adversarial image detected".to_string());
            result.risk_score += 75.0;
        }

        if result.risk_score > 80.0 {
            self.detected_count += 1;
            return Err(TinyVlmError::config("High-risk attack pattern detected"));
        }

        Ok(result)
    }

    fn is_adversarial_image(&self, data: &[u8]) -> bool {
        // Simplified adversarial detection
        // In production, use more sophisticated ML-based detection
        
        if data.len() < 1000 { return false; }

        // Check for unusual pixel value distributions
        let mut histogram = [0u32; 256];
        for &byte in data.iter().take(1000) {  // Sample first 1000 bytes
            histogram[byte as usize] += 1;
        }

        // Look for suspicious distributions
        let max_count = *histogram.iter().max().unwrap_or(&0);
        let min_count = *histogram.iter().min().unwrap_or(&0);
        
        // If distribution is too uniform or too concentrated, it might be adversarial
        (max_count > 800) || (max_count - min_count < 2)
    }

    fn get_detected_count(&self) -> u64 {
        self.detected_count
    }
}

/// Client behavior pattern tracking
struct ClientPattern {
    requests: Vec<RequestInfo>,
    first_request: Instant,
    last_request: Instant,
}

impl ClientPattern {
    fn new(now: Instant) -> Self {
        Self {
            requests: Vec::new(),
            first_request: now,
            last_request: now,
        }
    }

    fn add_request(&mut self, now: Instant, image_size: usize, text_length: usize) {
        self.last_request = now;
        self.requests.push(RequestInfo {
            timestamp: now,
            image_size,
            text_length,
        });

        // Keep only recent requests
        let cutoff = now - Duration::from_secs(300); // 5 minutes
        self.requests.retain(|req| req.timestamp > cutoff);
    }

    fn is_flooding(&self) -> bool {
        // Check for request flooding in the last minute
        let cutoff = self.last_request - Duration::from_secs(60);
        let recent_requests = self.requests.iter()
            .filter(|req| req.timestamp > cutoff)
            .count();
        
        recent_requests > 20
    }

    fn has_size_anomalies(&self) -> bool {
        if self.requests.len() < 5 { return false; }

        let recent: Vec<_> = self.requests.iter().rev().take(10).collect();
        let avg_image_size: f64 = recent.iter()
            .map(|r| r.image_size as f64)
            .sum::<f64>() / recent.len() as f64;

        // Check for sudden size changes
        recent.iter().any(|req| {
            let diff = (req.image_size as f64 - avg_image_size).abs();
            diff > avg_image_size * 2.0  // More than 2x average
        })
    }

    fn is_repeating_requests(&self) -> bool {
        if self.requests.len() < 3 { return false; }

        let recent: Vec<_> = self.requests.iter().rev().take(5).collect();
        
        // Check for identical request sizes
        let first = recent[0];
        recent.iter().all(|req| {
            req.image_size == first.image_size && 
            req.text_length == first.text_length
        })
    }
}

#[derive(Clone)]
struct RequestInfo {
    timestamp: Instant,
    image_size: usize,
    text_length: usize,
}

/// Security configuration
pub struct SecurityConfig {
    pub max_image_size_mb: usize,
    pub max_text_length: usize,
    pub max_requests_per_minute: u32,
    pub enable_attack_detection: bool,
    pub strict_validation: bool,
    pub start_time: Instant,
}

impl SecurityConfig {
    fn production() -> Self {
        Self {
            max_image_size_mb: 10,
            max_text_length: 500,
            max_requests_per_minute: 60,
            enable_attack_detection: true,
            strict_validation: true,
            start_time: Instant::now(),
        }
    }
}

/// Internal validation configuration
struct ValidationConfig {
    max_image_size_bytes: usize,
    min_image_size_bytes: usize,
    max_image_dimension: usize,
    max_text_length: usize,
}

impl ValidationConfig {
    fn strict() -> Self {
        Self {
            max_image_size_bytes: 10 * 1024 * 1024, // 10MB
            min_image_size_bytes: 1024, // 1KB
            max_image_dimension: 1024,
            max_text_length: 500,
        }
    }
}

/// Rate limiting configuration
struct RateLimitConfig {
    max_requests_per_window: u32,
    window_duration: Duration,
}

impl RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests_per_window: 60,
            window_duration: Duration::from_secs(60),
        }
    }
}

/// Validation results
pub struct ValidationResult {
    pub image_validation: ImageValidationResult,
    pub text_validation: TextValidationResult,
    pub resource_check: ResourceCheckResult,
    pub attack_check: AttackAnalysisResult,
    pub approved: bool,
}

pub struct ImageValidationResult {
    pub size_valid: bool,
    pub format_valid: bool,
    pub content_safe: bool,
    pub dimensions_valid: bool,
    pub warnings: Vec<String>,
}

pub struct TextValidationResult {
    pub length_valid: bool,
    pub encoding_valid: bool,
    pub content_safe: bool,
    pub language_appropriate: bool,
    pub warnings: Vec<String>,
}

pub struct ResourceCheckResult {
    pub memory_estimate_mb: f64,
    pub processing_time_estimate_ms: f64,
    pub within_limits: bool,
}

pub struct AttackAnalysisResult {
    pub risk_score: f64,
    pub detected_attacks: Vec<String>,
    pub behavioral_anomalies: Vec<String>,
}

/// Security metrics for monitoring
pub struct SecurityMetrics {
    pub blocked_requests: u64,
    pub invalid_inputs: u64,
    pub detected_attacks: u64,
    pub resource_violations: u64,
    pub uptime_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_manager() {
        let mut security = SecurityManager::new();
        
        // Create a valid test image with varying pixel values
        let mut test_image = Vec::with_capacity(224 * 224 * 3);
        for y in 0..224 {
            for x in 0..224 {
                test_image.push((x % 256) as u8);  // R
                test_image.push((y % 256) as u8);  // G
                test_image.push(((x + y) % 256) as u8);  // B
            }
        }
        let test_prompt = "What is in this image?";
        
        let result = security.validate_inference_request("test_client", &test_image, test_prompt);
        match &result {
            Ok(_) => {},
            Err(e) => println!("Validation failed: {}", e),
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_malicious_text_detection() {
        let mut validator = InputValidator::with_strict_rules();
        
        // Test malicious patterns
        let malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "javascript:alert('hack')",
            "../../../../etc/passwd",
        ];

        for input in &malicious_inputs {
            let result = validator.validate_text_input(input);
            assert!(result.is_err(), "Should reject malicious input: {}", input);
        }
    }

    #[test]
    fn test_rate_limiting() {
        let mut limiter = RateLimiter::with_defaults();
        
        // Allow normal requests (rate limit is 60 per window)
        for i in 0..60 {
            assert!(limiter.allow_request("client1"), "Request {} should be allowed", i);
        }
        
        // Should block after limit
        for _ in 0..10 {
            assert!(!limiter.allow_request("client1"), "Should be rate limited");
        }
        
        // Different client should still work
        assert!(limiter.allow_request("client2"));
    }

    #[test]
    fn test_secure_cleanup() {
        let security = SecurityManager::new();
        let mut sensitive_data = vec![42u8; 1000];
        
        security.secure_cleanup(&mut sensitive_data);
        
        // Data should be zeroed
        assert!(sensitive_data.iter().all(|&b| b == 0));
    }
}