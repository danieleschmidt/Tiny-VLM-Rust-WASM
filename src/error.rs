//! Error types for Tiny-VLM

use thiserror::Error;

/// Result type alias for Tiny-VLM operations
pub type Result<T> = core::result::Result<T, TinyVlmError>;

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

    /// Create a SIMD error
    pub fn simd(msg: impl Into<String>) -> Self {
        Self::Simd(msg.into())
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