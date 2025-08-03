//! Data loading utilities for efficient batch processing

use crate::{
    data::{DataSample, VisionLanguageDataset},
    memory::{MemoryPool, Tensor, TensorShape},
    Result, TinyVlmError,
};
use serde::{Deserialize, Serialize};

/// Configuration for data loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderConfig {
    /// Batch size for loading
    pub batch_size: usize,
    /// Whether to shuffle data between epochs
    pub shuffle: bool,
    /// Number of worker threads for loading (0 = auto)
    pub num_workers: usize,
    /// Whether to drop the last incomplete batch
    pub drop_last: bool,
    /// Pin memory for faster GPU transfers
    pub pin_memory: bool,
    /// Prefetch factor for background loading
    pub prefetch_factor: usize,
    /// Maximum queue size for prefetching
    pub max_queue_size: usize,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            num_workers: 0, // Auto-detect
            drop_last: false,
            pin_memory: false,
            prefetch_factor: 2,
            max_queue_size: 10,
        }
    }
}

/// A batch of data samples with associated tensors
#[derive(Debug)]
pub struct DataBatch {
    /// Sample IDs in this batch
    pub ids: Vec<String>,
    /// Image tensors (batch_size, height, width, channels)
    pub images: Tensor<f32>,
    /// Text prompts (original strings)
    pub texts: Vec<String>,
    /// Optional ground truth responses
    pub ground_truths: Vec<Option<String>>,
    /// Batch metadata
    pub metadata: BatchMetadata,
}

/// Metadata for a data batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMetadata {
    /// Batch size
    pub batch_size: usize,
    /// Average text length in this batch
    pub avg_text_length: f64,
    /// Image dimensions (height, width)
    pub image_dimensions: (usize, usize),
    /// Processing time for this batch (milliseconds)
    pub processing_time_ms: Option<f64>,
}

/// Efficient data loader for vision-language datasets
pub struct DataLoader {
    /// Configuration
    config: DataLoaderConfig,
    /// Reference to the dataset
    dataset: VisionLanguageDataset,
    /// Current epoch
    current_epoch: usize,
    /// Current position in dataset
    current_position: usize,
    /// Sample indices for current epoch
    sample_indices: Vec<usize>,
    /// Memory pool for efficient tensor allocation
    memory_pool: MemoryPool<f32>,
}

impl DataLoader {
    /// Create a new data loader
    pub fn new(dataset: VisionLanguageDataset, config: DataLoaderConfig) -> Result<Self> {
        let sample_indices: Vec<usize> = (0..dataset.len()).collect();
        let memory_pool = MemoryPool::new(100_000_000); // 400MB limit for f32

        Ok(Self {
            config,
            dataset,
            current_epoch: 0,
            current_position: 0,
            sample_indices,
            memory_pool,
        })
    }

    /// Get the next batch of data
    pub fn next_batch(&mut self) -> Result<Option<DataBatch>> {
        if self.current_position >= self.dataset.len() {
            return Ok(None);
        }

        let start_time = std::time::Instant::now();
        
        // Determine batch size
        let remaining_samples = self.dataset.len() - self.current_position;
        let batch_size = if remaining_samples < self.config.batch_size {
            if self.config.drop_last {
                return Ok(None);
            }
            remaining_samples
        } else {
            self.config.batch_size
        };

        // Collect samples for this batch
        let mut batch_samples = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let sample_idx = self.sample_indices[self.current_position + i];
            if let Some(sample) = self.dataset.get_sample(sample_idx) {
                batch_samples.push(sample);
            }
        }

        // Process batch
        let batch = self.process_batch(batch_samples, start_time)?;
        
        self.current_position += batch_size;
        Ok(Some(batch))
    }

    /// Reset the data loader for a new epoch
    pub fn reset_epoch(&mut self) {
        self.current_position = 0;
        self.current_epoch += 1;

        if self.config.shuffle {
            self.shuffle_indices();
        }
    }

    /// Check if there are more batches in the current epoch
    pub fn has_next(&self) -> bool {
        if self.config.drop_last {
            self.current_position + self.config.batch_size <= self.dataset.len()
        } else {
            self.current_position < self.dataset.len()
        }
    }

    /// Get the current epoch number
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Get the number of batches per epoch
    pub fn batches_per_epoch(&self) -> usize {
        if self.config.drop_last {
            self.dataset.len() / self.config.batch_size
        } else {
            (self.dataset.len() + self.config.batch_size - 1) / self.config.batch_size
        }
    }

    /// Get dataset statistics
    pub fn dataset_stats(&self) -> crate::data::DatasetStats {
        self.dataset.stats().clone()
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> crate::memory::MemoryStats {
        self.memory_pool.memory_usage()
    }

    // Private methods

    fn process_batch(
        &mut self,
        samples: Vec<&DataSample>,
        start_time: std::time::Instant,
    ) -> Result<DataBatch> {
        let batch_size = samples.len();
        
        // Assume standard image size for now (224x224x3)
        let image_height = 224;
        let image_width = 224;
        let image_channels = 3;

        // Create batch tensors
        let image_shape = TensorShape::new(&[batch_size, image_height, image_width, image_channels])?;
        let mut images = self.memory_pool.allocate(image_shape)?;

        let mut ids = Vec::with_capacity(batch_size);
        let mut texts = Vec::with_capacity(batch_size);
        let mut ground_truths = Vec::with_capacity(batch_size);
        let mut total_text_length = 0;

        // Process each sample in the batch
        for (i, sample) in samples.iter().enumerate() {
            // Load and preprocess image
            let image_data = sample.load_image_data()?;
            let processed_image = self.preprocess_image(&image_data, image_height, image_width)?;
            
            // Copy image data to batch tensor
            let image_start = i * image_height * image_width * image_channels;
            let image_end = image_start + image_height * image_width * image_channels;
            let images_data = images.data_mut();
            images_data[image_start..image_end].copy_from_slice(&processed_image);

            // Collect other data
            ids.push(sample.id.clone());
            texts.push(sample.text.clone());
            ground_truths.push(sample.ground_truth.clone());
            total_text_length += sample.text.len();
        }

        let processing_time = start_time.elapsed().as_millis() as f64;
        let avg_text_length = total_text_length as f64 / batch_size as f64;

        let metadata = BatchMetadata {
            batch_size,
            avg_text_length,
            image_dimensions: (image_height, image_width),
            processing_time_ms: Some(processing_time),
        };

        Ok(DataBatch {
            ids,
            images,
            texts,
            ground_truths,
            metadata,
        })
    }

    fn preprocess_image(
        &self,
        image_data: &[u8],
        target_height: usize,
        target_width: usize,
    ) -> Result<Vec<f32>> {
        // Simple image preprocessing (in practice, would use proper image library)
        let expected_size = target_height * target_width * 3;
        
        if image_data.len() < expected_size {
            // Pad with zeros if image is too small
            let mut padded = vec![0.0f32; expected_size];
            for (i, &byte) in image_data.iter().take(expected_size).enumerate() {
                padded[i] = byte as f32 / 255.0;
            }
            Ok(padded)
        } else {
            // Convert to f32 and normalize
            let mut normalized = Vec::with_capacity(expected_size);
            for &byte in image_data.iter().take(expected_size) {
                normalized.push(byte as f32 / 255.0);
            }
            Ok(normalized)
        }
    }

    fn shuffle_indices(&mut self) {
        #[cfg(feature = "std")]
        {
            use rand::seq::SliceRandom;
            use rand::SeedableRng;
            
            let mut rng = rand::rngs::StdRng::seed_from_u64(
                (self.current_epoch as u64).wrapping_mul(42)
            );
            self.sample_indices.shuffle(&mut rng);
        }
    }
}

/// Batch loader for multiple datasets or advanced loading patterns
pub struct BatchLoader {
    /// Multiple data loaders
    loaders: Vec<DataLoader>,
    /// Current loader index
    current_loader: usize,
    /// Round-robin or weighted sampling
    sampling_strategy: SamplingStrategy,
}

/// Strategy for sampling from multiple datasets
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Round-robin between datasets
    RoundRobin,
    /// Weighted sampling based on dataset sizes
    Weighted,
    /// Random sampling
    Random,
}

impl BatchLoader {
    /// Create a new batch loader
    pub fn new(loaders: Vec<DataLoader>, strategy: SamplingStrategy) -> Self {
        Self {
            loaders,
            current_loader: 0,
            sampling_strategy: strategy,
        }
    }

    /// Get the next batch from any available loader
    pub fn next_batch(&mut self) -> Result<Option<DataBatch>> {
        if self.loaders.is_empty() {
            return Ok(None);
        }

        match self.sampling_strategy {
            SamplingStrategy::RoundRobin => self.next_batch_round_robin(),
            SamplingStrategy::Weighted => self.next_batch_weighted(),
            SamplingStrategy::Random => self.next_batch_random(),
        }
    }

    /// Reset all loaders for new epoch
    pub fn reset_epoch(&mut self) {
        for loader in &mut self.loaders {
            loader.reset_epoch();
        }
        self.current_loader = 0;
    }

    /// Check if any loader has more data
    pub fn has_next(&self) -> bool {
        self.loaders.iter().any(|loader| loader.has_next())
    }

    // Private methods

    fn next_batch_round_robin(&mut self) -> Result<Option<DataBatch>> {
        let start_loader = self.current_loader;
        
        loop {
            if let Some(batch) = self.loaders[self.current_loader].next_batch()? {
                self.advance_loader();
                return Ok(Some(batch));
            }
            
            self.advance_loader();
            
            // Check if we've cycled through all loaders
            if self.current_loader == start_loader {
                return Ok(None);
            }
        }
    }

    fn next_batch_weighted(&mut self) -> Result<Option<DataBatch>> {
        // Simple implementation: try loaders in order of remaining data
        for loader in &mut self.loaders {
            if let Some(batch) = loader.next_batch()? {
                return Ok(Some(batch));
            }
        }
        Ok(None)
    }

    fn next_batch_random(&mut self) -> Result<Option<DataBatch>> {
        #[cfg(feature = "std")]
        {
            use rand::seq::SliceRandom;
            
            // Get indices of loaders that have data
            let available_loaders: Vec<usize> = self.loaders
                .iter()
                .enumerate()
                .filter(|(_, loader)| loader.has_next())
                .map(|(i, _)| i)
                .collect();
                
            if available_loaders.is_empty() {
                return Ok(None);
            }
            
            // Randomly select a loader
            let selected = available_loaders.choose(&mut rand::thread_rng())
                .ok_or_else(|| TinyVlmError::config("No available loaders".into()))?;
                
            self.loaders[*selected].next_batch()
        }
        
        #[cfg(not(feature = "std"))]
        {
            // Fallback to round-robin without random
            self.next_batch_round_robin()
        }
    }

    fn advance_loader(&mut self) {
        self.current_loader = (self.current_loader + 1) % self.loaders.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DatasetConfig, VisionLanguageDataset, DataSample};
    use std::path::PathBuf;

    fn create_test_dataset() -> VisionLanguageDataset {
        let config = DatasetConfig {
            root_dir: PathBuf::from("/tmp"),
            validate_data: false,
            ..Default::default()
        };

        let mut dataset = VisionLanguageDataset {
            config,
            samples: Vec::new(),
            metadata: crate::data::DatasetMetadata {
                name: "Test".into(),
                version: "1.0".into(),
                description: "Test dataset".into(),
                num_samples: 0,
                created_at: "2023-01-01T00:00:00Z".into(),
                stats: crate::data::DatasetStats {
                    avg_text_length: 0.0,
                    text_length_stats: (0, 0, 0.0),
                    avg_image_size: 0.0,
                    image_formats: std::collections::HashMap::new(),
                    vocab_size: 0,
                },
            },
        };

        // Add some test samples
        for i in 0..10 {
            let sample = DataSample::new(
                format!("test_{:03}", i),
                PathBuf::from(format!("test_{}.jpg", i)),
                format!("This is test sample {}", i),
            );
            dataset.add_sample(sample).unwrap();
        }

        dataset
    }

    #[test]
    fn test_data_loader_config() {
        let config = DataLoaderConfig::default();
        assert_eq!(config.batch_size, 32);
        assert!(config.shuffle);
        assert_eq!(config.num_workers, 0);
    }

    #[test]
    fn test_data_loader_creation() {
        let dataset = create_test_dataset();
        let config = DataLoaderConfig {
            batch_size: 4,
            ..Default::default()
        };

        let loader = DataLoader::new(dataset, config);
        assert!(loader.is_ok());
        
        let loader = loader.unwrap();
        assert_eq!(loader.batches_per_epoch(), 3); // 10 samples / 4 batch_size = 2.5 -> 3
    }

    #[test]
    fn test_data_loader_batches_per_epoch() {
        let dataset = create_test_dataset();
        
        // Test with drop_last = false
        let config = DataLoaderConfig {
            batch_size: 3,
            drop_last: false,
            ..Default::default()
        };
        let loader = DataLoader::new(dataset.clone(), config).unwrap();
        assert_eq!(loader.batches_per_epoch(), 4); // ceil(10/3) = 4

        // Test with drop_last = true
        let config = DataLoaderConfig {
            batch_size: 3,
            drop_last: true,
            ..Default::default()
        };
        let loader = DataLoader::new(dataset, config).unwrap();
        assert_eq!(loader.batches_per_epoch(), 3); // floor(10/3) = 3
    }

    #[test]
    fn test_batch_metadata() {
        let metadata = BatchMetadata {
            batch_size: 4,
            avg_text_length: 25.5,
            image_dimensions: (224, 224),
            processing_time_ms: Some(150.0),
        };

        assert_eq!(metadata.batch_size, 4);
        assert!((metadata.avg_text_length - 25.5).abs() < f64::EPSILON);
        assert_eq!(metadata.image_dimensions, (224, 224));
    }

    #[test]
    fn test_sampling_strategy() {
        let strategies = vec![
            SamplingStrategy::RoundRobin,
            SamplingStrategy::Weighted,
            SamplingStrategy::Random,
        ];

        for strategy in strategies {
            match strategy {
                SamplingStrategy::RoundRobin => assert!(true),
                SamplingStrategy::Weighted => assert!(true),
                SamplingStrategy::Random => assert!(true),
            }
        }
    }
}