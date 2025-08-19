//! Simplified high-performance integration layer for production VLM inference

use crate::{
    models::{FastVLM, InferenceConfig, ModelConfig},
    Result, TinyVlmError,
};

#[cfg(feature = "std")]
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

/// Simplified VLM inference service for production deployment
pub struct SimpleVLMService {
    /// Single model instance
    model: Arc<Mutex<FastVLM>>,
    /// Service configuration
    config: SimpleServiceConfig,
}

/// Simplified service configuration
#[derive(Debug, Clone)]
pub struct SimpleServiceConfig {
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// GPU acceleration enabled
    pub gpu_acceleration: bool,
    /// Model quantization level
    pub quantization_bits: u8,
}

impl Default for SimpleServiceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 100,
            gpu_acceleration: true,
            quantization_bits: 8,
        }
    }
}

/// Simplified inference request
#[derive(Debug, Clone)]
pub struct SimpleInferenceRequest {
    /// Unique request ID
    pub id: String,
    /// Image data
    pub image_data: Vec<u8>,
    /// Text prompt
    pub prompt: String,
}

/// Simplified inference response
#[derive(Debug, Clone)]
pub struct SimpleInferenceResponse {
    /// Request ID
    pub request_id: String,
    /// Generated response
    pub response: String,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Success status
    pub success: bool,
}

impl SimpleVLMService {
    /// Create a new simple inference service
    pub fn new(config: SimpleServiceConfig) -> Result<Self> {
        let model_config = ModelConfig::default();
        let model = FastVLM::new(model_config)?;
        
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            config,
        })
    }
    
    /// Process a simple inference request
    pub fn process_request(&self, request: SimpleInferenceRequest) -> Result<SimpleInferenceResponse> {
        let start_time = Instant::now();
        
        // Basic validation
        if request.prompt.len() > 10000 {
            return Err(TinyVlmError::invalid_input("Prompt too long"));
        }
        
        if request.image_data.len() > 50 * 1024 * 1024 {
            return Err(TinyVlmError::invalid_input("Image too large"));
        }
        
        // Process with model
        let mut model = self.model.lock().unwrap();
        let response_text = model.infer(
            &request.image_data,
            &request.prompt,
            InferenceConfig::default(),
        )?;
        
        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(SimpleInferenceResponse {
            request_id: request.id,
            response: response_text,
            processing_time_ms: processing_time,
            success: true,
        })
    }
    
    /// Get simple health status
    pub fn health_status(&self) -> SimpleHealthStatus {
        SimpleHealthStatus {
            status: "healthy".to_string(),
            gpu_enabled: self.config.gpu_acceleration,
            model_loaded: true,
        }
    }
}

/// Simple health status
#[derive(Debug)]
pub struct SimpleHealthStatus {
    pub status: String,
    pub gpu_enabled: bool,
    pub model_loaded: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_service_creation() {
        let config = SimpleServiceConfig::default();
        let service = SimpleVLMService::new(config);
        assert!(service.is_ok());
    }

    #[test]
    fn test_simple_inference() {
        let config = SimpleServiceConfig::default();
        let service = SimpleVLMService::new(config).unwrap();
        
        let request = SimpleInferenceRequest {
            id: "test-123".to_string(),
            image_data: vec![0u8; 1024],
            prompt: "test prompt".to_string(),
        };
        
        let result = service.process_request(request);
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert_eq!(response.request_id, "test-123");
        assert!(response.success);
        assert!(response.processing_time_ms > 0.0);
    }

    #[test]
    fn test_health_status() {
        let config = SimpleServiceConfig::default();
        let service = SimpleVLMService::new(config).unwrap();
        
        let health = service.health_status();
        assert_eq!(health.status, "healthy");
        assert!(health.model_loaded);
    }

    #[test]
    fn test_validation() {
        let config = SimpleServiceConfig::default();
        let service = SimpleVLMService::new(config).unwrap();
        
        // Test prompt too long
        let long_request = SimpleInferenceRequest {
            id: "long-test".to_string(),
            image_data: vec![0u8; 1024],
            prompt: "a".repeat(20000),
        };
        
        let result = service.process_request(long_request);
        assert!(result.is_err());
    }
}