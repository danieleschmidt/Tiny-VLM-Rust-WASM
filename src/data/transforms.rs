//! Data transformation utilities for preprocessing

use crate::{Result, TinyVlmError};

/// Trait for image transformations
pub trait ImageTransform {
    fn apply(&self, image_data: &[u8]) -> Result<Vec<u8>>;
}

/// Trait for text transformations
pub trait TextTransform {
    fn apply(&self, text: &str) -> Result<String>;
}

/// Composite transformation that combines multiple transforms
pub struct CompositeTransform<T> {
    transforms: Vec<Box<dyn Fn(&T) -> Result<T>>>,
}

impl<T> CompositeTransform<T> {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    pub fn add_transform<F>(mut self, transform: F) -> Self
    where
        F: Fn(&T) -> Result<T> + 'static,
    {
        self.transforms.push(Box::new(transform));
        self
    }

    pub fn apply(&self, input: &T) -> Result<T>
    where
        T: Clone,
    {
        let mut result = input.clone();
        for transform in &self.transforms {
            result = transform(&result)?;
        }
        Ok(result)
    }
}

/// Resize image transformation
pub struct ResizeTransform {
    pub target_width: u32,
    pub target_height: u32,
}

impl ImageTransform for ResizeTransform {
    fn apply(&self, image_data: &[u8]) -> Result<Vec<u8>> {
        // Simplified resize implementation
        // In practice, would use proper image processing library
        if image_data.len() as u32 == self.target_width * self.target_height * 3 {
            return Ok(image_data.to_vec());
        }
        
        // For now, just return padded or truncated data
        let target_size = (self.target_width * self.target_height * 3) as usize;
        let mut result = vec![0u8; target_size];
        
        let copy_size = image_data.len().min(target_size);
        result[..copy_size].copy_from_slice(&image_data[..copy_size]);
        
        Ok(result)
    }
}

/// Normalize text by converting to lowercase and trimming
pub struct NormalizeTextTransform;

impl TextTransform for NormalizeTextTransform {
    fn apply(&self, text: &str) -> Result<String> {
        Ok(text.trim().to_lowercase())
    }
}

/// Truncate text to maximum length
pub struct TruncateTextTransform {
    pub max_length: usize,
}

impl TextTransform for TruncateTextTransform {
    fn apply(&self, text: &str) -> Result<String> {
        if text.len() <= self.max_length {
            Ok(text.to_string())
        } else {
            Ok(text.chars().take(self.max_length).collect())
        }
    }
}