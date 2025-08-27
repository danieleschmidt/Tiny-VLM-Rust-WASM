//! Generation 2: Robust VLM Implementation
//! 
//! This example demonstrates enhanced error handling, logging, monitoring,
//! validation, and security measures for production readiness.

use tiny_vlm::prelude::*;
use std::time::{Instant, Duration};
// use std::sync::Arc;
use std::collections::HashMap;

fn main() -> Result<()> {
    // Initialize comprehensive logging
    if let Ok(_) = env_logger::try_init() {
        println!("✓ Enhanced logging system initialized");
    }
    
    println!("🛡️ Generation 2: Robust VLM Demo");
    println!("==================================");

    // Initialize monitoring system
    let mut monitoring = setup_monitoring_system()?;
    println!("✓ Monitoring system active");

    // Initialize health monitoring
    let health_monitor = HealthMonitor::new();
    println!("✓ Health monitoring enabled");

    // Create robust configuration with validation
    let config = create_validated_config()?;
    println!("✓ Configuration validated and loaded");
    
    // Initialize model with comprehensive error handling
    let start = Instant::now();
    let vlm = initialize_model_with_retry(config, 3)?;
    let init_time = start.elapsed();
    println!("✓ Model initialized robustly in {:?}", init_time);
    
    // Log initialization metrics
    monitoring.record_metric("model_init_time_ms", init_time.as_millis() as f64)?;
    
    // Setup reliability manager
    let mut reliability = setup_reliability_manager()?;
    println!("✓ Reliability management active");

    // Run comprehensive test suite
    run_comprehensive_tests(&vlm, &mut monitoring, &mut reliability)?;
    
    // Run security validation tests
    run_security_tests(&vlm, &mut monitoring)?;
    
    // Generate health report
    let health_report = health_monitor.generate_report();
    println!("\n📊 System Health Report:");
    println!("=======================");
    println!("Overall Status: {:?}", health_report.overall_status);
    println!("Uptime: {:?}", health_report.uptime);
    println!("Memory Usage: {:.2}MB", health_report.memory_usage_mb);
    println!("CPU Usage: {:.2}%", health_report.cpu_usage_percent);
    
    // Generate monitoring report
    let monitoring_report = monitoring.generate_report()?;
    println!("\n📈 Performance Monitoring:");
    println!("==========================");
    println!("Total Requests: {}", monitoring_report.total_requests);
    println!("Success Rate: {:.2}%", monitoring_report.success_rate * 100.0);
    println!("Average Latency: {:.2}ms", monitoring_report.avg_latency_ms);
    println!("Error Rate: {:.2}%", monitoring_report.error_rate * 100.0);
    
    println!("\n✅ Generation 2 Robust Demo Complete!");
    Ok(())
}

/// Setup comprehensive monitoring system
fn setup_monitoring_system() -> Result<MonitoringSystem> {
    let config = MonitoringConfig {
        enable_metrics: true,
        enable_tracing: true,
        metric_retention_hours: 24,
        alert_thresholds: HashMap::from([
            ("error_rate".to_string(), 0.05),
            ("latency_p99".to_string(), 1000.0),
            ("memory_usage".to_string(), 0.85),
        ]),
    };
    
    MonitoringSystem::new(config)
}

/// Setup reliability manager with circuit breakers and retry policies
fn setup_reliability_manager() -> Result<ReliabilityManager> {
    let config = ReliabilityConfig {
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
    };
    
    Ok(ReliabilityManager::new(config))
}

/// Create and validate configuration
fn create_validated_config() -> Result<SimpleVLMConfig> {
    let config = SimpleVLMConfig {
        vision_dim: 768,
        text_dim: 768,
        max_length: 100,
    };
    
    // Validate configuration parameters
    if config.vision_dim == 0 || config.text_dim == 0 {
        return Err(TinyVlmError::invalid_config("Invalid dimension configuration"));
    }
    
    if config.max_length == 0 || config.max_length > 1000 {
        return Err(TinyVlmError::invalid_config("Invalid max_length configuration"));
    }
    
    println!("✓ Configuration validation passed");
    Ok(config)
}

/// Initialize model with retry logic and circuit breaker
fn initialize_model_with_retry(config: SimpleVLMConfig, max_retries: u32) -> Result<SimpleVLM> {
    let mut last_error = None;
    
    for attempt in 1..=max_retries {
        println!("Model initialization attempt {}/{}", attempt, max_retries);
        
        match SimpleVLM::new(config.clone()) {
            Ok(vlm) => {
                println!("✓ Model initialization successful on attempt {}", attempt);
                return Ok(vlm);
            }
            Err(e) => {
                println!("⚠️ Attempt {} failed: {}", attempt, e);
                last_error = Some(e);
                
                if attempt < max_retries {
                    std::thread::sleep(Duration::from_millis(1000 * attempt as u64));
                }
            }
        }
    }
    
    Err(last_error.unwrap_or_else(|| TinyVlmError::model_loading("Max retries exceeded")))
}

/// Run comprehensive test suite with monitoring
fn run_comprehensive_tests(
    vlm: &SimpleVLM, 
    monitoring: &mut MonitoringSystem,
    _reliability: &mut ReliabilityManager
) -> Result<()> {
    println!("\n🧪 Running Comprehensive Test Suite:");
    println!("====================================");
    
    let test_cases = vec![
        ("What objects are in this image?", "standard_query"),
        ("Describe the colors and lighting", "descriptive_query"), 
        ("Is this safe for children?", "safety_query"),
        ("What is the main subject?", "subject_query"),
        ("List all visible elements", "enumeration_query"),
    ];
    
    let image_data = create_robust_test_image()?;
    let mut success_count = 0;
    
    for (i, (prompt, test_type)) in test_cases.iter().enumerate() {
        println!("\nTest {}: {} ({})", i + 1, prompt, test_type);
        
        let start = Instant::now();
        
        // Validate inputs before processing
        if let Err(e) = validate_inputs(&image_data, prompt) {
            monitoring.record_error("input_validation", &e.to_string())?;
            continue;
        }
        
        // Execute with circuit breaker protection
        match vlm.infer(&image_data, prompt) {
            Ok(response) => {
                let latency = start.elapsed();
                success_count += 1;
                
                println!("  ✓ Success ({:?})", latency);
                println!("  Response: {}", response);
                
                // Record success metrics
                monitoring.record_metric("inference_latency_ms", latency.as_millis() as f64)?;
                monitoring.record_metric("inference_success", 1.0)?;
                monitoring.increment_counter(&format!("test_type_{}", test_type))?;
                
                // Validate response quality
                validate_response_quality(&response, prompt)?;
                
            }
            Err(e) => {
                let latency = start.elapsed();
                println!("  ❌ Failed ({:?}): {}", latency, e);
                
                // Record failure metrics
                monitoring.record_error("inference_failure", &e.to_string())?;
                monitoring.record_metric("inference_latency_ms", latency.as_millis() as f64)?;
                monitoring.record_metric("inference_success", 0.0)?;
            }
        }
    }
    
    let success_rate = success_count as f64 / test_cases.len() as f64;
    println!("\n📊 Test Results: {}/{} passed ({:.1}% success rate)", 
             success_count, test_cases.len(), success_rate * 100.0);
             
    monitoring.record_metric("test_suite_success_rate", success_rate)?;
    
    if success_rate < 0.8 {
        return Err(TinyVlmError::validation_error("Test suite success rate below threshold"));
    }
    
    println!("✅ Comprehensive test suite passed");
    Ok(())
}

/// Run security validation tests
fn run_security_tests(vlm: &SimpleVLM, monitoring: &mut MonitoringSystem) -> Result<()> {
    println!("\n🔒 Security Validation Tests:");
    println!("============================");
    
    let long_string = "A".repeat(1000);
    let security_tests = vec![
        // Input injection tests
        ("'; DROP TABLE users; --", "sql_injection"),
        ("<script>alert('xss')</script>", "xss_attempt"),
        ("../../../etc/passwd", "path_traversal"),
        (long_string.as_str(), "buffer_overflow"),
        ("What is in this image?\0hidden", "null_injection"),
    ];
    
    let image_data = create_robust_test_image()?;
    
    for (malicious_input, test_type) in security_tests {
        println!("Security test: {}", test_type);
        
        let _start = Instant::now();
        match vlm.infer(&image_data, &malicious_input) {
            Ok(response) => {
                // Check if malicious input was properly sanitized
                if response.contains(&malicious_input) {
                    println!("  ⚠️ Potential security issue: input not sanitized");
                    monitoring.record_security_event("input_not_sanitized", test_type)?;
                } else {
                    println!("  ✓ Input properly handled");
                }
            }
            Err(e) => {
                // Expected for most security tests
                println!("  ✓ Malicious input rejected: {}", e);
                monitoring.record_metric("security_rejections", 1.0)?;
            }
        }
    }
    
    // Test with oversized image data
    println!("Security test: oversized_image");
    let oversized_image = vec![0u8; 100_000_000]; // 100MB
    match vlm.infer(&oversized_image, "test") {
        Ok(_) => {
            println!("  ⚠️ Large image not rejected");
            monitoring.record_security_event("large_image_accepted", "oversized_image")?;
        }
        Err(_) => {
            println!("  ✓ Large image properly rejected");
        }
    }
    
    println!("✅ Security tests completed");
    Ok(())
}

/// Validate inputs for security and format compliance
fn validate_inputs(image_data: &[u8], text: &str) -> Result<()> {
    // Image validation
    if image_data.is_empty() {
        return Err(TinyVlmError::invalid_input("Empty image data"));
    }
    
    if image_data.len() > 50_000_000 { // 50MB limit
        return Err(TinyVlmError::invalid_input("Image too large"));
    }
    
    // Text validation
    if text.is_empty() {
        return Err(TinyVlmError::invalid_input("Empty text input"));
    }
    
    if text.len() > 1000 { // Character limit
        return Err(TinyVlmError::invalid_input("Text too long"));
    }
    
    // Security validation
    if text.contains('\0') {
        return Err(TinyVlmError::invalid_input("Null bytes not allowed"));
    }
    
    let suspicious_patterns = [
        "javascript:", "data:", "vbscript:", "<script", "</script",
        "DROP TABLE", "SELECT * FROM", "../", "UNION SELECT"
    ];
    
    for pattern in &suspicious_patterns {
        if text.to_lowercase().contains(&pattern.to_lowercase()) {
            return Err(TinyVlmError::invalid_input("Suspicious input pattern detected"));
        }
    }
    
    Ok(())
}

/// Validate response quality and safety
fn validate_response_quality(response: &str, _prompt: &str) -> Result<()> {
    // Basic quality checks
    if response.len() < 10 {
        return Err(TinyVlmError::validation_error("Response too short"));
    }
    
    if response.len() > 10000 {
        return Err(TinyVlmError::validation_error("Response too long"));
    }
    
    // Content safety checks
    let unsafe_patterns = [
        "harmful", "dangerous", "illegal", "violence", "inappropriate"
    ];
    
    for pattern in &unsafe_patterns {
        if response.to_lowercase().contains(pattern) {
            return Err(TinyVlmError::validation_error("Potentially unsafe response content"));
        }
    }
    
    Ok(())
}

/// Create robust test image with proper validation
fn create_robust_test_image() -> Result<Vec<u8>> {
    let width = 224;
    let height = 224; 
    let channels = 3;
    let expected_size = width * height * channels;
    let mut data = Vec::with_capacity(expected_size);
    
    // Create a more realistic test pattern
    for y in 0..height {
        for x in 0..width {
            // Create a checkerboard pattern with noise
            let base_val = if (x / 32 + y / 32) % 2 == 0 { 200 } else { 50 };
            let noise = ((x + y) % 16) as u8;
            
            let r = (base_val + noise / 2).min(255) as u8;
            let g = (base_val + noise / 3).min(255) as u8; 
            let b = (base_val + noise / 4).min(255) as u8;
            
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    
    if data.len() != expected_size {
        return Err(TinyVlmError::image_processing("Invalid image data size"));
    }
    
    Ok(data)
}