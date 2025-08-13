//! Tiny-VLM-Rust-WASM: Ultra-efficient Vision-Language Model
//!
//! A high-performance Vision-Language Model implementation optimized for mobile deployment
//! through WebAssembly compilation and SIMD optimization.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, rust_2018_idioms)]

pub mod data;
pub mod deployment;
pub mod error;
pub mod health;
pub mod logging;
pub mod memory;
pub mod models;
pub mod monitoring;
pub mod optimization;
pub mod recovery;
pub mod scaling;
pub mod security;
pub mod simd;
pub mod text;
pub mod threat_intelligence;
pub mod validation;
pub mod vision;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use error::{Result, TinyVlmError};
pub use models::{FastVLM, InferenceConfig, ModelConfig};
pub use memory::{MemoryPool, Tensor, TensorShape};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{FastVLM, InferenceConfig, ModelConfig, Result, TinyVlmError};
    pub use crate::data::{DataLoader, DataSample, VisionLanguageDataset, DatasetConfig};
    pub use crate::health::{HealthMonitor, HealthReport, HealthStatus};
    pub use crate::logging::{LogConfig, LogFormat, PerformanceTimer};
    pub use crate::memory::{MemoryPool, Tensor, TensorShape};
    pub use crate::optimization::{AdaptiveCache, CacheConfig, LoadBalancer, LoadBalancingStrategy};
    pub use crate::recovery::{CircuitBreaker, RecoveryManager, RecoveryStats, RetryPolicy};
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