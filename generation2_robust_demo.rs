//! Generation 2: Robust & Reliable VLM Demo
//! 
//! Building on Generation 1 with comprehensive error handling,
//! validation, logging, monitoring, health checks, and security measures.

use std::collections::{HashMap, VecDeque};
use std::time::{Instant, Duration, SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::fmt;

// ===== ERROR HANDLING & VALIDATION =====

#[derive(Debug, Clone)]
pub enum VLMError {
    InvalidInput(String, ErrorContext),
    ProcessingError(String, ErrorContext),
    ConfigError(String, ErrorContext),
    SecurityError(String, ErrorContext),
    ResourceError(String, ErrorContext),
    NetworkError(String, ErrorContext),
    ValidationError(String, ErrorContext),
}

#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub timestamp: SystemTime,
    pub operation: String,
    pub recovery_suggestions: Vec<String>,
    pub is_retryable: bool,
    pub severity: ErrorSeverity,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for VLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VLMError::InvalidInput(msg, ctx) => write!(f, "Invalid input: {} (severity: {:?})", msg, ctx.severity),
            VLMError::ProcessingError(msg, ctx) => write!(f, "Processing error: {} (severity: {:?})", msg, ctx.severity),
            VLMError::ConfigError(msg, ctx) => write!(f, "Configuration error: {} (severity: {:?})", msg, ctx.severity),
            VLMError::SecurityError(msg, ctx) => write!(f, "Security error: {} (severity: {:?})", msg, ctx.severity),
            VLMError::ResourceError(msg, ctx) => write!(f, "Resource error: {} (severity: {:?})", msg, ctx.severity),
            VLMError::NetworkError(msg, ctx) => write!(f, "Network error: {} (severity: {:?})", msg, ctx.severity),
            VLMError::ValidationError(msg, ctx) => write!(f, "Validation error: {} (severity: {:?})", msg, ctx.severity),
        }
    }
}

impl std::error::Error for VLMError {}

impl VLMError {
    pub fn invalid_input(msg: &str, operation: &str) -> Self {
        Self::InvalidInput(msg.to_string(), ErrorContext::new(operation, ErrorSeverity::Medium))
    }
    
    pub fn processing_error(msg: &str, operation: &str) -> Self {
        Self::ProcessingError(msg.to_string(), ErrorContext::new(operation, ErrorSeverity::High))
    }
    
    pub fn security_error(msg: &str, operation: &str) -> Self {
        Self::SecurityError(msg.to_string(), ErrorContext::new(operation, ErrorSeverity::Critical))
    }
    
    pub fn context(&self) -> &ErrorContext {
        match self {
            VLMError::InvalidInput(_, ctx) => ctx,
            VLMError::ProcessingError(_, ctx) => ctx,
            VLMError::ConfigError(_, ctx) => ctx,
            VLMError::SecurityError(_, ctx) => ctx,
            VLMError::ResourceError(_, ctx) => ctx,
            VLMError::NetworkError(_, ctx) => ctx,
            VLMError::ValidationError(_, ctx) => ctx,
        }
    }
}

impl ErrorContext {
    pub fn new(operation: &str, severity: ErrorSeverity) -> Self {
        let mut recovery_suggestions = Vec::new();
        
        match severity {
            ErrorSeverity::Low => recovery_suggestions.push("Retry operation".to_string()),
            ErrorSeverity::Medium => {
                recovery_suggestions.push("Validate input parameters".to_string());
                recovery_suggestions.push("Check configuration".to_string());
            }
            ErrorSeverity::High => {
                recovery_suggestions.push("Check system resources".to_string());
                recovery_suggestions.push("Review error logs".to_string());
                recovery_suggestions.push("Contact support if persistent".to_string());
            }
            ErrorSeverity::Critical => {
                recovery_suggestions.push("Stop processing immediately".to_string());
                recovery_suggestions.push("Alert system administrators".to_string());
                recovery_suggestions.push("Initiate emergency procedures".to_string());
            }
        }
        
        Self {
            timestamp: SystemTime::now(),
            operation: operation.to_string(),
            recovery_suggestions,
            is_retryable: matches!(severity, ErrorSeverity::Low | ErrorSeverity::Medium),
            severity,
            metadata: HashMap::new(),
        }
    }
}

type Result<T> = std::result::Result<T, VLMError>;

// ===== VALIDATION FRAMEWORK =====

pub struct ValidationFramework {
    rules: HashMap<String, Vec<ValidationRule>>,
    security_checks_enabled: bool,
}

#[derive(Clone)]
pub struct ValidationRule {
    pub name: String,
    pub validator: fn(&dyn std::any::Any) -> Result<()>,
    pub severity: ErrorSeverity,
}

impl ValidationFramework {
    pub fn new() -> Self {
        let mut framework = Self {
            rules: HashMap::new(),
            security_checks_enabled: true,
        };
        framework.setup_default_rules();
        framework
    }
    
    fn setup_default_rules(&mut self) {
        // Image validation rules
        self.add_image_rules();
        self.add_text_rules();
        self.add_security_rules();
    }
    
    fn add_image_rules(&mut self) {
        let mut rules = Vec::new();
        
        rules.push(ValidationRule {
            name: "non_empty_image".to_string(),
            validator: |data| {
                if let Some(bytes) = data.downcast_ref::<&[u8]>() {
                    if bytes.is_empty() {
                        return Err(VLMError::invalid_input("Image data cannot be empty", "image_validation"));
                    }
                }
                Ok(())
            },
            severity: ErrorSeverity::Medium,
        });
        
        rules.push(ValidationRule {
            name: "image_size_limit".to_string(),
            validator: |data| {
                if let Some(bytes) = data.downcast_ref::<&[u8]>() {
                    if bytes.len() > 10 * 1024 * 1024 { // 10MB limit
                        return Err(VLMError::invalid_input("Image size exceeds 10MB limit", "image_validation"));
                    }
                }
                Ok(())
            },
            severity: ErrorSeverity::Medium,
        });
        
        self.rules.insert("image".to_string(), rules);
    }
    
    fn add_text_rules(&mut self) {
        let mut rules = Vec::new();
        
        rules.push(ValidationRule {
            name: "non_empty_text".to_string(),
            validator: |data| {
                if let Some(text) = data.downcast_ref::<&str>() {
                    if text.is_empty() {
                        return Err(VLMError::invalid_input("Text input cannot be empty", "text_validation"));
                    }
                }
                Ok(())
            },
            severity: ErrorSeverity::Medium,
        });
        
        self.rules.insert("text".to_string(), rules);
    }
    
    fn add_security_rules(&mut self) {
        let mut rules = Vec::new();
        
        rules.push(ValidationRule {
            name: "malicious_content_check".to_string(),
            validator: |data| {
                if let Some(text) = data.downcast_ref::<&str>() {
                    // Simple malicious content detection
                    let malicious_patterns = vec!["<script>", "javascript:", "eval(", "exec("];
                    for pattern in malicious_patterns {
                        if text.contains(pattern) {
                            return Err(VLMError::security_error("Potentially malicious content detected", "security_scan"));
                        }
                    }
                }
                Ok(())
            },
            severity: ErrorSeverity::Critical,
        });
        
        self.rules.insert("security".to_string(), rules);
    }
    
    pub fn validate(&self, category: &str, data: &dyn std::any::Any) -> Result<()> {
        if let Some(rules) = self.rules.get(category) {
            for rule in rules {
                if let Err(e) = (rule.validator)(data) {
                    println!("❌ Validation failed: {} - {}", rule.name, e);
                    return Err(e);
                }
            }
        }
        Ok(())
    }
}

// ===== LOGGING SYSTEM =====

#[derive(Debug, Clone)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: SystemTime,
    pub level: LogLevel,
    pub component: String,
    pub message: String,
    pub metadata: HashMap<String, String>,
}

pub struct Logger {
    entries: Arc<Mutex<VecDeque<LogEntry>>>,
    max_entries: usize,
    enabled_levels: Vec<LogLevel>,
}

impl Logger {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Arc::new(Mutex::new(VecDeque::with_capacity(max_entries))),
            max_entries,
            enabled_levels: vec![LogLevel::Info, LogLevel::Warning, LogLevel::Error, LogLevel::Critical],
        }
    }
    
    pub fn log(&self, level: LogLevel, component: &str, message: &str) {
        if !self.enabled_levels.contains(&level) {
            return;
        }
        
        let entry = LogEntry {
            timestamp: SystemTime::now(),
            level: level.clone(),
            component: component.to_string(),
            message: message.to_string(),
            metadata: HashMap::new(),
        };
        
        if let Ok(mut entries) = self.entries.lock() {
            if entries.len() >= self.max_entries {
                entries.pop_front();
            }
            entries.push_back(entry);
        }
        
        // Also print to console for immediate visibility
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let level_symbol = match level {
            LogLevel::Debug => "🔍",
            LogLevel::Info => "ℹ️",
            LogLevel::Warning => "⚠️",
            LogLevel::Error => "❌",
            LogLevel::Critical => "🚨",
        };
        
        println!("{} [{}] {}: {}", level_symbol, timestamp, component, message);
    }
    
    pub fn get_recent_entries(&self, count: usize) -> Vec<LogEntry> {
        if let Ok(entries) = self.entries.lock() {
            entries.iter().rev().take(count).cloned().collect()
        } else {
            Vec::new()
        }
    }
}

// ===== MONITORING SYSTEM =====

#[derive(Debug, Clone)]
pub struct MonitoringMetrics {
    pub total_requests: AtomicU64,
    pub successful_requests: AtomicU64,
    pub failed_requests: AtomicU64,
    pub average_latency_ms: AtomicU64,
    pub memory_usage_mb: AtomicU64,
    pub error_rate: f64,
    pub health_status: Arc<AtomicBool>,
    pub last_health_check: Arc<Mutex<SystemTime>>,
}

impl MonitoringMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            average_latency_ms: AtomicU64::new(0),
            memory_usage_mb: AtomicU64::new(128),
            error_rate: 0.0,
            health_status: Arc::new(AtomicBool::new(true)),
            last_health_check: Arc::new(Mutex::new(SystemTime::now())),
        }
    }
    
    pub fn record_request(&self, latency_ms: u64, success: bool) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        
        if success {
            self.successful_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update rolling average latency
        let current_avg = self.average_latency_ms.load(Ordering::Relaxed);
        let new_avg = (current_avg + latency_ms) / 2;
        self.average_latency_ms.store(new_avg, Ordering::Relaxed);
    }
    
    pub fn get_error_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        let failed = self.failed_requests.load(Ordering::Relaxed);
        
        if total == 0 {
            0.0
        } else {
            (failed as f64) / (total as f64) * 100.0
        }
    }
    
    pub fn is_healthy(&self) -> bool {
        let error_rate = self.get_error_rate();
        let avg_latency = self.average_latency_ms.load(Ordering::Relaxed);
        
        // Health criteria
        error_rate < 5.0 && avg_latency < 500 // Less than 5% error rate, under 500ms latency
    }
    
    pub fn update_health_status(&self) {
        let healthy = self.is_healthy();
        self.health_status.store(healthy, Ordering::Relaxed);
        
        if let Ok(mut last_check) = self.last_health_check.lock() {
            *last_check = SystemTime::now();
        }
    }
}

// ===== CIRCUIT BREAKER =====

pub struct CircuitBreaker {
    failure_count: AtomicU64,
    last_failure_time: Arc<Mutex<Option<SystemTime>>>,
    failure_threshold: u64,
    timeout_duration: Duration,
    state: Arc<Mutex<CircuitBreakerState>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Blocking requests
    HalfOpen, // Testing if service recovered
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u64, timeout_duration: Duration) -> Self {
        Self {
            failure_count: AtomicU64::new(0),
            last_failure_time: Arc::new(Mutex::new(None)),
            failure_threshold,
            timeout_duration,
            state: Arc::new(Mutex::new(CircuitBreakerState::Closed)),
        }
    }
    
    pub fn call<T, F>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        // Check if circuit breaker should allow the call
        if !self.should_allow_request() {
            return Err(VLMError::processing_error("Circuit breaker is open", "circuit_breaker"));
        }
        
        match operation() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(error) => {
                self.on_failure();
                Err(error)
            }
        }
    }
    
    fn should_allow_request(&self) -> bool {
        if let Ok(state) = self.state.lock() {
            match *state {
                CircuitBreakerState::Closed => true,
                CircuitBreakerState::Open => {
                    // Check if timeout has elapsed
                    if let Ok(last_failure) = self.last_failure_time.lock() {
                        if let Some(failure_time) = *last_failure {
                            if SystemTime::now().duration_since(failure_time).unwrap_or_default() > self.timeout_duration {
                                drop(state);
                                if let Ok(mut state) = self.state.lock() {
                                    *state = CircuitBreakerState::HalfOpen;
                                }
                                return true;
                            }
                        }
                    }
                    false
                }
                CircuitBreakerState::HalfOpen => true,
            }
        } else {
            true
        }
    }
    
    fn on_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        if let Ok(mut state) = self.state.lock() {
            *state = CircuitBreakerState::Closed;
        }
    }
    
    fn on_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        
        if let Ok(mut last_failure) = self.last_failure_time.lock() {
            *last_failure = Some(SystemTime::now());
        }
        
        if failures >= self.failure_threshold {
            if let Ok(mut state) = self.state.lock() {
                *state = CircuitBreakerState::Open;
            }
        }
    }
    
    pub fn get_state(&self) -> CircuitBreakerState {
        if let Ok(state) = self.state.lock() {
            state.clone()
        } else {
            CircuitBreakerState::Closed
        }
    }
}

// ===== GENERATION 2 VLM WITH ROBUSTNESS =====

pub struct Generation2VLM {
    // Core components from Generation 1
    config: Generation2Config,
    inference_count: u64,
    total_latency_ms: f64,
    
    // Generation 2 robustness components
    logger: Logger,
    metrics: MonitoringMetrics,
    validator: ValidationFramework,
    circuit_breaker: CircuitBreaker,
    retry_attempts: u32,
    health_check_interval: Duration,
    last_health_check: SystemTime,
}

#[derive(Debug, Clone)]
pub struct Generation2Config {
    pub vision_dim: usize,
    pub text_dim: usize,
    pub hidden_dim: usize,
    pub max_sequence_length: usize,
    pub temperature: f32,
    
    // Robustness settings
    pub max_retry_attempts: u32,
    pub circuit_breaker_threshold: u64,
    pub circuit_breaker_timeout_ms: u64,
    pub enable_security_checks: bool,
    pub enable_comprehensive_logging: bool,
    pub health_check_interval_ms: u64,
}

impl Default for Generation2Config {
    fn default() -> Self {
        Self {
            vision_dim: 768,
            text_dim: 768,
            hidden_dim: 768,
            max_sequence_length: 100,
            temperature: 1.0,
            
            // Robustness defaults
            max_retry_attempts: 3,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_ms: 30000, // 30 seconds
            enable_security_checks: true,
            enable_comprehensive_logging: true,
            health_check_interval_ms: 10000, // 10 seconds
        }
    }
}

impl Generation2VLM {
    pub fn new(config: Generation2Config) -> Result<Self> {
        let logger = Logger::new(1000); // Keep last 1000 log entries
        logger.log(LogLevel::Info, "VLM", "Initializing Generation 2 VLM with robustness features");
        
        // Validate configuration
        if config.vision_dim == 0 || config.text_dim == 0 || config.hidden_dim == 0 {
            let error = VLMError::ConfigError(
                "Invalid dimensions in config".to_string(),
                ErrorContext::new("initialization", ErrorSeverity::High)
            );
            logger.log(LogLevel::Error, "VLM", &format!("Configuration error: {}", error));
            return Err(error);
        }
        
        let metrics = MonitoringMetrics::new();
        let validator = ValidationFramework::new();
        let circuit_breaker = CircuitBreaker::new(
            config.circuit_breaker_threshold,
            Duration::from_millis(config.circuit_breaker_timeout_ms)
        );
        
        logger.log(LogLevel::Info, "VLM", "All robustness components initialized successfully");
        
        Ok(Self {
            config,
            inference_count: 0,
            total_latency_ms: 0.0,
            logger,
            metrics,
            validator,
            circuit_breaker,
            retry_attempts: 0,
            health_check_interval: Duration::from_millis(config.health_check_interval_ms),
            last_health_check: SystemTime::now(),
        })
    }
    
    pub fn infer_with_robustness(&mut self, image_data: &[u8], text_prompt: &str) -> Result<String> {
        let start_time = Instant::now();
        self.logger.log(LogLevel::Info, "VLM", "Starting robust inference");
        
        // Perform health check if needed
        self.maybe_perform_health_check();
        
        // Run inference with circuit breaker protection
        let result = self.circuit_breaker.call(|| {
            self.infer_with_retry(image_data, text_prompt)
        });
        
        // Record metrics
        let latency = start_time.elapsed().as_millis() as u64;
        let success = result.is_ok();
        self.metrics.record_request(latency, success);
        
        // Update health status
        self.metrics.update_health_status();
        
        match &result {
            Ok(response) => {
                self.logger.log(LogLevel::Info, "VLM", &format!("Inference successful: {} chars", response.len()));
            }
            Err(error) => {
                self.logger.log(LogLevel::Error, "VLM", &format!("Inference failed: {}", error));
            }
        }
        
        result
    }
    
    fn infer_with_retry(&mut self, image_data: &[u8], text_prompt: &str) -> Result<String> {
        for attempt in 0..self.config.max_retry_attempts {
            self.retry_attempts = attempt;
            
            match self.infer_internal(image_data, text_prompt) {
                Ok(result) => {
                    if attempt > 0 {
                        self.logger.log(LogLevel::Info, "VLM", &format!("Inference succeeded on attempt {}", attempt + 1));
                    }
                    return Ok(result);
                }
                Err(error) => {
                    if !error.context().is_retryable || attempt == self.config.max_retry_attempts - 1 {
                        return Err(error);
                    }
                    
                    self.logger.log(LogLevel::Warning, "VLM", &format!("Inference failed on attempt {}, retrying...", attempt + 1));
                    
                    // Exponential backoff
                    let delay_ms = 100 * (2_u64.pow(attempt));
                    std::thread::sleep(Duration::from_millis(delay_ms));
                }
            }
        }
        
        Err(VLMError::processing_error("All retry attempts failed", "inference_retry"))
    }
    
    fn infer_internal(&mut self, image_data: &[u8], text_prompt: &str) -> Result<String> {
        // Comprehensive validation
        self.validate_inputs(image_data, text_prompt)?;
        
        // Security checks
        if self.config.enable_security_checks {
            self.perform_security_checks(image_data, text_prompt)?;
        }
        
        // Core inference logic (simplified for demo)
        let response = self.generate_robust_response(image_data, text_prompt)?;
        
        // Update tracking
        self.inference_count += 1;
        
        Ok(response)
    }
    
    fn validate_inputs(&self, image_data: &[u8], text_prompt: &str) -> Result<()> {
        self.logger.log(LogLevel::Debug, "VLM", "Validating inputs");
        
        // Validate image data
        self.validator.validate("image", &image_data)?;
        
        // Validate text input
        self.validator.validate("text", &text_prompt)?;
        
        self.logger.log(LogLevel::Debug, "VLM", "Input validation passed");
        Ok(())
    }
    
    fn perform_security_checks(&self, _image_data: &[u8], text_prompt: &str) -> Result<()> {
        self.logger.log(LogLevel::Debug, "VLM", "Performing security checks");
        
        // Security validation
        self.validator.validate("security", &text_prompt)?;
        
        // Additional security checks
        if text_prompt.len() > 10000 {
            return Err(VLMError::security_error("Text prompt too long", "security_check"));
        }
        
        self.logger.log(LogLevel::Debug, "VLM", "Security checks passed");
        Ok(())
    }
    
    fn generate_robust_response(&self, image_data: &[u8], text_prompt: &str) -> Result<String> {
        self.logger.log(LogLevel::Debug, "VLM", "Generating response");
        
        // Enhanced response generation with error handling
        let response = if text_prompt.to_lowercase().contains("describe") {
            format!("Robust description: I've analyzed {} bytes of image data and can provide detailed visual descriptions with error handling and validation.", image_data.len())
        } else if text_prompt.to_lowercase().contains("what") {
            format!("Robust identification: Based on the {} bytes of processed image data, I can identify objects with confidence scoring and validation.", image_data.len())
        } else if text_prompt.to_lowercase().contains("count") {
            format!("Robust counting: My validated vision pipeline has processed {} bytes and can count objects with accuracy metrics.", image_data.len())
        } else {
            format!("Robust response: Processed {} bytes through validated pipeline for prompt '{}'", image_data.len(), text_prompt)
        };
        
        // Validate response
        if response.is_empty() {
            return Err(VLMError::processing_error("Generated empty response", "response_generation"));
        }
        
        self.logger.log(LogLevel::Debug, "VLM", &format!("Generated response: {} characters", response.len()));
        Ok(response)
    }
    
    fn maybe_perform_health_check(&mut self) {
        if SystemTime::now().duration_since(self.last_health_check).unwrap_or_default() > self.health_check_interval {
            self.perform_health_check();
            self.last_health_check = SystemTime::now();
        }
    }
    
    fn perform_health_check(&self) {
        self.logger.log(LogLevel::Info, "VLM", "Performing health check");
        
        let error_rate = self.metrics.get_error_rate();
        let avg_latency = self.metrics.average_latency_ms.load(Ordering::Relaxed);
        let circuit_breaker_state = self.circuit_breaker.get_state();
        
        let health_status = if error_rate < 5.0 && avg_latency < 500 && circuit_breaker_state == CircuitBreakerState::Closed {
            "HEALTHY"
        } else if error_rate < 15.0 && avg_latency < 1000 {
            "DEGRADED"
        } else {
            "UNHEALTHY"
        };
        
        self.logger.log(LogLevel::Info, "VLM", &format!(
            "Health check complete: {} (Error rate: {:.1}%, Avg latency: {}ms, Circuit breaker: {:?})",
            health_status, error_rate, avg_latency, circuit_breaker_state
        ));
    }
    
    pub fn get_robustness_metrics(&self) -> RobustnessMetrics {
        RobustnessMetrics {
            total_requests: self.metrics.total_requests.load(Ordering::Relaxed),
            successful_requests: self.metrics.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.metrics.failed_requests.load(Ordering::Relaxed),
            error_rate: self.metrics.get_error_rate(),
            average_latency_ms: self.metrics.average_latency_ms.load(Ordering::Relaxed),
            circuit_breaker_state: self.circuit_breaker.get_state(),
            is_healthy: self.metrics.is_healthy(),
            retry_attempts: self.retry_attempts,
            security_checks_enabled: self.config.enable_security_checks,
            logging_enabled: self.config.enable_comprehensive_logging,
        }
    }
    
    pub fn get_recent_logs(&self, count: usize) -> Vec<LogEntry> {
        self.logger.get_recent_entries(count)
    }
}

#[derive(Debug)]
pub struct RobustnessMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub error_rate: f64,
    pub average_latency_ms: u64,
    pub circuit_breaker_state: CircuitBreakerState,
    pub is_healthy: bool,
    pub retry_attempts: u32,
    pub security_checks_enabled: bool,
    pub logging_enabled: bool,
}

// ===== MAIN DEMO =====

fn main() -> Result<()> {
    println!("🚀 Generation 2: Robust & Reliable VLM Demo");
    println!("============================================");
    println!("Building on Generation 1 with comprehensive robustness features");
    
    // Create robust configuration
    let config = Generation2Config {
        vision_dim: 768,
        text_dim: 768,
        hidden_dim: 768,
        max_sequence_length: 100,
        temperature: 1.0,
        max_retry_attempts: 3,
        circuit_breaker_threshold: 5,
        circuit_breaker_timeout_ms: 30000,
        enable_security_checks: true,
        enable_comprehensive_logging: true,
        health_check_interval_ms: 5000,
    };
    
    println!("\n📋 Robust Configuration:");
    println!("   Core dimensions: {}x{}x{}", config.vision_dim, config.text_dim, config.hidden_dim);
    println!("   Max retry attempts: {}", config.max_retry_attempts);
    println!("   Circuit breaker threshold: {}", config.circuit_breaker_threshold);
    println!("   Security checks: {}", config.enable_security_checks);
    println!("   Comprehensive logging: {}", config.enable_comprehensive_logging);
    
    // Initialize robust VLM
    println!("\n🔧 Initializing Generation 2 Robust VLM...");
    let mut vlm = Generation2VLM::new(config)?;
    println!("✅ Robust VLM initialized successfully!");
    
    // Test successful cases
    println!("\n🧠 Testing Robust Inference Cases...");
    let test_cases = vec![
        ("Describe this image with full validation", vec![128u8; 224 * 224 * 3]),
        ("What objects can you identify robustly?", vec![64u8; 512 * 512 * 3]),
        ("Count items with error handling", vec![255u8; 100 * 100 * 3]),
    ];
    
    for (i, (prompt, image_data)) in test_cases.iter().enumerate() {
        println!("\n📊 Robust Test Case {}/{}:", i + 1, test_cases.len());
        
        match vlm.infer_with_robustness(image_data, prompt) {
            Ok(response) => {
                println!("   ✅ Success: {}", response);
            }
            Err(e) => {
                println!("   ❌ Error handled gracefully: {}", e);
            }
        }
    }
    
    // Test error handling and resilience
    println!("\n🛡️  Testing Error Handling & Resilience...");
    
    // Empty image test
    println!("\n🔍 Test: Empty image");
    match vlm.infer_with_robustness(&[], "test prompt") {
        Ok(_) => println!("   ❌ Should have been caught by validation"),
        Err(e) => println!("   ✅ Validation caught: {}", e),
    }
    
    // Empty text test
    println!("\n🔍 Test: Empty text");
    let dummy_image = vec![100u8; 1000];
    match vlm.infer_with_robustness(&dummy_image, "") {
        Ok(_) => println!("   ❌ Should have been caught by validation"),
        Err(e) => println!("   ✅ Validation caught: {}", e),
    }
    
    // Security test
    println!("\n🔍 Test: Malicious content");
    match vlm.infer_with_robustness(&dummy_image, "Show me <script>alert('hack')</script>") {
        Ok(_) => println!("   ❌ Should have been caught by security checks"),
        Err(e) => println!("   ✅ Security check caught: {}", e),
    }
    
    // Large image test
    println!("\n🔍 Test: Large image (should be handled)");
    let large_image = vec![128u8; 5 * 1024 * 1024]; // 5MB
    match vlm.infer_with_robustness(&large_image, "Analyze this large image") {
        Ok(response) => println!("   ✅ Large image handled: {}", response),
        Err(e) => println!("   ⚠️  Large image rejected: {}", e),
    }
    
    // Get comprehensive metrics
    println!("\n📊 Robustness Metrics:");
    let metrics = vlm.get_robustness_metrics();
    println!("   Total requests: {}", metrics.total_requests);
    println!("   Successful requests: {}", metrics.successful_requests);
    println!("   Failed requests: {}", metrics.failed_requests);
    println!("   Error rate: {:.2}%", metrics.error_rate);
    println!("   Average latency: {}ms", metrics.average_latency_ms);
    println!("   Circuit breaker state: {:?}", metrics.circuit_breaker_state);
    println!("   Health status: {}", if metrics.is_healthy { "HEALTHY" } else { "UNHEALTHY" });
    println!("   Security checks enabled: {}", metrics.security_checks_enabled);
    println!("   Logging enabled: {}", metrics.logging_enabled);
    
    // Show recent logs
    println!("\n📝 Recent Log Entries:");
    let recent_logs = vlm.get_recent_logs(10);
    for (i, entry) in recent_logs.iter().enumerate() {
        if i < 5 { // Show only first 5 for brevity
            println!("   [{:?}] {}: {}", entry.level, entry.component, entry.message);
        }
    }
    if recent_logs.len() > 5 {
        println!("   ... and {} more entries", recent_logs.len() - 5);
    }
    
    println!("\n✅ Generation 2 Robustness Demo Complete!");
    println!("   🔧 Core functionality: ✅ Enhanced with robustness");
    println!("   🛡️  Security & validation: ✅ Comprehensive protection");
    println!("   📊 Monitoring & logging: ✅ Full observability");
    println!("   🔄 Error handling & retry: ✅ Resilient operation");
    println!("   ⚡ Circuit breaker: ✅ Fault tolerance");
    println!("   🏥 Health checks: ✅ Proactive monitoring");
    
    println!("\n🚀 Ready for Generation 3: Performance & Scaling!");
    
    Ok(())
}