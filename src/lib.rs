//! Tiny-VLM-Rust-WASM: Ultra-efficient Vision-Language Model
//!
//! A high-performance Vision-Language Model implementation optimized for mobile deployment
//! through WebAssembly compilation and SIMD optimization.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

pub mod benchmarks;
pub mod data;
pub mod deployment;
pub mod error;
pub mod health;
pub mod integration_simple;
pub mod logging;
pub mod memory;
pub mod models;
pub mod monitoring;
pub mod monitoring_advanced;
pub mod optimization;
pub mod recovery;
pub mod reliability;
pub mod research;
pub mod scaling;
pub mod scaling_advanced;
pub mod security;
pub mod security_advanced;
pub mod simd;
pub mod text;
pub mod threat_intelligence;
pub mod validation;
pub mod vision;

#[cfg(feature = "wasm")]
pub mod wasm;
#[cfg(feature = "gpu")]
pub mod gpu;

pub use error::{Result, TinyVlmError};
pub use models::{FastVLM, InferenceConfig, ModelConfig, SimpleTextResult, SimpleImageResult, SimpleVLMResult, BasicPerformanceMetrics};
pub use memory::{MemoryPool, TensorShape};

/// Default tensor type using f32 precision
pub type Tensor = memory::Tensor<f32>;

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{FastVLM, InferenceConfig, ModelConfig, Result, TinyVlmError, Tensor};
    pub use crate::benchmarks::{BenchmarkSuite, BenchmarkConfig, BenchmarkOperation, BenchmarkResult, BenchmarkReport};
    pub use crate::data::{DataLoader, DataSample, VisionLanguageDataset, DatasetConfig};
    pub use crate::health::{HealthMonitor, HealthReport};
    pub use crate::integration_simple::{SimpleVLMService, SimpleInferenceRequest, SimpleInferenceResponse, SimpleServiceConfig};
    pub use crate::logging::{LogConfig, LogFormat, PerformanceTimer};
    pub use crate::memory::{MemoryPool, TensorShape};
    pub use crate::monitoring_advanced::{AdvancedMonitoringSystem, AdvancedMonitoringConfig, Metric, MetricType, Span, Alert, HealthCheck, HealthStatus};
    pub use crate::optimization::{AdaptiveCache, CacheConfig, LoadBalancer, LoadBalancingStrategy};
    pub use crate::recovery::{CircuitBreaker, RecoveryManager, RecoveryStats, RetryPolicy};
    pub use crate::reliability::{ReliabilityManager, ReliabilityConfig, ReliabilityState, EnhancedError, ErrorCategory};
    pub use crate::research::{ResearchFramework, ExperimentConfig, ResearchAlgorithm, AlgorithmResult, ExperimentResults, ExportFormat};
    pub use crate::scaling_advanced::{AdvancedScalingManager, AdvancedScalingConfig, AdvancedAutoScaler, SmartCache, ResourceMetrics, ScalingDecision};
    pub use crate::security_advanced::{AdvancedSecurityManager, AdvancedSecurityConfig, SecurityAnalysis, ThreatLevel, SecurityEvent};
    pub use crate::text::{Tokenizer, TokenizerConfig};
    pub use crate::validation::{ValidationResult, validate_image_data, validate_text_input};
    pub use crate::vision::{ImageProcessor, VisionEncoder};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Basic smoke test to ensure core modules compile
        let config = ModelConfig::default();
        assert!(config.vision_dim > 0);
    }
}