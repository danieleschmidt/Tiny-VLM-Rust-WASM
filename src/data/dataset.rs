//! Dataset management for vision-language data

use crate::{Result, TinyVlmError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Error types specific to dataset operations
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("Dataset not found: {0}")]
    NotFound(String),
    #[error("Invalid dataset format: {0}")]
    InvalidFormat(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Configuration for dataset loading and processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Root directory containing dataset files
    pub root_dir: PathBuf,
    /// Maximum number of samples to load (-1 for all)
    pub max_samples: i32,
    /// Whether to shuffle the dataset
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Supported image formats
    pub image_formats: Vec<String>,
    /// Maximum image size in bytes
    pub max_image_size: usize,
    /// Maximum text length in characters
    pub max_text_length: usize,
    /// Cache directory for processed data
    pub cache_dir: Option<PathBuf>,
    /// Whether to validate data integrity on load
    pub validate_data: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("./data"),
            max_samples: -1,
            shuffle: false,
            seed: 42,
            image_formats: vec!["jpg".into(), "jpeg".into(), "png".into(), "webp".into()],
            max_image_size: 10 * 1024 * 1024, // 10MB
            max_text_length: 2048,
            cache_dir: None,
            validate_data: true,
        }
    }
}

/// A single data sample containing image and text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSample {
    /// Unique identifier for this sample
    pub id: String,
    /// Path to the image file
    pub image_path: PathBuf,
    /// Associated text prompt or caption
    pub text: String,
    /// Optional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Ground truth response (for evaluation)
    pub ground_truth: Option<String>,
}

impl DataSample {
    /// Create a new data sample
    pub fn new(id: String, image_path: PathBuf, text: String) -> Self {
        Self {
            id,
            image_path,
            text,
            metadata: HashMap::new(),
            ground_truth: None,
        }
    }

    /// Add metadata to the sample
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set ground truth response
    pub fn with_ground_truth(mut self, ground_truth: String) -> Self {
        self.ground_truth = Some(ground_truth);
        self
    }

    /// Load image data from disk
    pub fn load_image_data(&self) -> Result<Vec<u8>> {
        std::fs::read(&self.image_path)
            .map_err(|e| TinyVlmError::image_processing(format!("Failed to load image: {}", e)))
    }

    /// Validate the sample data
    pub fn validate(&self, config: &DatasetConfig) -> Result<()> {
        // Check if image file exists
        if !self.image_path.exists() {
            return Err(TinyVlmError::invalid_input(format!(
                "Image file not found: {:?}",
                self.image_path
            )));
        }

        // Check image format
        let extension = self.image_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        if !config.image_formats.contains(&extension) {
            return Err(TinyVlmError::invalid_input(format!(
                "Unsupported image format: {}",
                extension
            )));
        }

        // Check image size
        let metadata = std::fs::metadata(&self.image_path)?;
        if metadata.len() as usize > config.max_image_size {
            return Err(TinyVlmError::invalid_input(format!(
                "Image too large: {} bytes (max: {})",
                metadata.len(),
                config.max_image_size
            )));
        }

        // Check text length
        if self.text.len() > config.max_text_length {
            return Err(TinyVlmError::invalid_input(format!(
                "Text too long: {} characters (max: {})",
                self.text.len(),
                config.max_text_length
            )));
        }

        Ok(())
    }
}

/// Vision-Language dataset for training and evaluation
#[derive(Clone)]
pub struct VisionLanguageDataset {
    /// Dataset configuration
    config: DatasetConfig,
    /// List of data samples
    samples: Vec<DataSample>,
    /// Dataset metadata
    metadata: DatasetMetadata,
}

/// Metadata about the dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name
    pub name: String,
    /// Dataset version
    pub version: String,
    /// Description
    pub description: String,
    /// Number of samples
    pub num_samples: usize,
    /// Creation timestamp
    pub created_at: String,
    /// Dataset statistics
    pub stats: DatasetStats,
}

/// Statistical information about the dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Average text length
    pub avg_text_length: f64,
    /// Text length distribution (min, max, std)
    pub text_length_stats: (usize, usize, f64),
    /// Average image size in bytes
    pub avg_image_size: f64,
    /// Image format distribution
    pub image_formats: HashMap<String, usize>,
    /// Vocabulary size (unique words)
    pub vocab_size: usize,
}

impl VisionLanguageDataset {
    /// Create a new dataset from configuration
    pub fn new(config: DatasetConfig) -> Result<Self> {
        let mut dataset = Self {
            config,
            samples: Vec::new(),
            metadata: DatasetMetadata {
                name: "Custom Dataset".into(),
                version: "1.0".into(),
                description: "A custom vision-language dataset".into(),
                num_samples: 0,
                created_at: chrono::Utc::now().to_rfc3339(),
                stats: DatasetStats {
                    avg_text_length: 0.0,
                    text_length_stats: (0, 0, 0.0),
                    avg_image_size: 0.0,
                    image_formats: HashMap::new(),
                    vocab_size: 0,
                },
            },
        };

        dataset.load_from_directory()?;
        dataset.compute_statistics()?;
        
        Ok(dataset)
    }

    /// Load dataset from a JSON Lines file
    pub fn from_jsonl_file<P: AsRef<Path>>(
        config: DatasetConfig,
        jsonl_path: P,
    ) -> Result<Self> {
        let mut dataset = Self {
            config,
            samples: Vec::new(),
            metadata: DatasetMetadata {
                name: "JSONL Dataset".into(),
                version: "1.0".into(),
                description: "Dataset loaded from JSONL file".into(),
                num_samples: 0,
                created_at: chrono::Utc::now().to_rfc3339(),
                stats: DatasetStats {
                    avg_text_length: 0.0,
                    text_length_stats: (0, 0, 0.0),
                    avg_image_size: 0.0,
                    image_formats: HashMap::new(),
                    vocab_size: 0,
                },
            },
        };

        dataset.load_from_jsonl(jsonl_path)?;
        dataset.compute_statistics()?;
        
        Ok(dataset)
    }

    /// Create dataset from a vector of samples
    pub fn from_samples(config: DatasetConfig, samples: Vec<DataSample>) -> Result<Self> {
        let mut dataset = Self {
            config,
            samples: Vec::new(),
            metadata: DatasetMetadata {
                name: "Custom Dataset".into(),
                version: "1.0".into(),
                description: "A dataset created from provided samples".into(),
                num_samples: 0,
                created_at: chrono::Utc::now().to_rfc3339(),
                stats: DatasetStats {
                    avg_text_length: 0.0,
                    text_length_stats: (0, 0, 0.0),
                    avg_image_size: 0.0,
                    image_formats: HashMap::new(),
                    vocab_size: 0,
                },
            },
        };

        // Add all samples
        for sample in samples {
            dataset.add_sample(sample)?;
        }
        
        dataset.compute_statistics()?;
        Ok(dataset)
    }

    /// Add a sample to the dataset
    pub fn add_sample(&mut self, sample: DataSample) -> Result<()> {
        if self.config.validate_data {
            sample.validate(&self.config)?;
        }
        
        self.samples.push(sample);
        Ok(())
    }

    /// Get a sample by index
    pub fn get_sample(&self, index: usize) -> Option<&DataSample> {
        self.samples.get(index)
    }

    /// Get the number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Shuffle the dataset
    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.seed);
        self.samples.shuffle(&mut rng);
    }

    /// Split dataset into train/validation sets
    pub fn train_val_split(&self, train_ratio: f64) -> Result<(Self, Self)> {
        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err(TinyVlmError::invalid_input("Train ratio must be between 0 and 1"));
        }

        let split_point = (self.samples.len() as f64 * train_ratio) as usize;
        
        let train_samples = self.samples[..split_point].to_vec();
        let val_samples = self.samples[split_point..].to_vec();

        let mut train_dataset = Self {
            config: self.config.clone(),
            samples: train_samples,
            metadata: self.metadata.clone(),
        };
        train_dataset.metadata.name = format!("{} (Train)", self.metadata.name);

        let mut val_dataset = Self {
            config: self.config.clone(),
            samples: val_samples,
            metadata: self.metadata.clone(),
        };
        val_dataset.metadata.name = format!("{} (Validation)", self.metadata.name);

        Ok((train_dataset, val_dataset))
    }

    /// Get dataset statistics
    pub fn stats(&self) -> &DatasetStats {
        &self.metadata.stats
    }

    /// Save dataset to disk in JSONL format
    pub fn save_to_jsonl<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        for sample in &self.samples {
            let json_line = serde_json::to_string(sample)
                .map_err(|e| TinyVlmError::config(format!("Serialization error: {}", e)))?;
            writeln!(writer, "{}", json_line)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Iterate over samples
    pub fn iter(&self) -> impl Iterator<Item = &DataSample> {
        self.samples.iter()
    }

    /// Iterate over batches of samples
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = &[DataSample]> {
        self.samples.chunks(batch_size)
    }

    // Private methods

    fn load_from_directory(&mut self) -> Result<()> {
        if !self.config.root_dir.exists() {
            return Err(TinyVlmError::config(format!(
                "Dataset directory not found: {:?}",
                self.config.root_dir
            )));
        }

        // Look for common dataset files
        let jsonl_path = self.config.root_dir.join("dataset.jsonl");
        if jsonl_path.exists() {
            return self.load_from_jsonl(jsonl_path);
        }

        // If no structured dataset file, try to create from directory structure
        self.load_from_directory_structure()
    }

    fn load_from_jsonl<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut samples_loaded = 0;
        let max_samples = if self.config.max_samples < 0 {
            usize::MAX
        } else {
            self.config.max_samples as usize
        };

        for line in reader.lines() {
            if samples_loaded >= max_samples {
                break;
            }

            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let sample: DataSample = serde_json::from_str(&line)
                .map_err(|e| TinyVlmError::config(format!("Failed to parse sample: {}", e)))?;

            // Make paths relative to dataset root
            let image_path = if sample.image_path.is_relative() {
                self.config.root_dir.join(&sample.image_path)
            } else {
                sample.image_path
            };

            let adjusted_sample = DataSample {
                image_path,
                ..sample
            };

            self.add_sample(adjusted_sample)?;
            samples_loaded += 1;
        }

        if self.config.shuffle {
            self.shuffle();
        }

        Ok(())
    }

    fn load_from_directory_structure(&mut self) -> Result<()> {
        use std::fs;
        
        // Simple directory structure: images and optional captions
        let images_dir = self.config.root_dir.join("images");
        if !images_dir.exists() {
            return Err(TinyVlmError::config(
                "No images directory found in dataset root"
            ));
        }

        let captions_file = self.config.root_dir.join("captions.txt");
        let captions = if captions_file.exists() {
            self.load_captions_file(captions_file)?
        } else {
            HashMap::new()
        };

        let entries = fs::read_dir(images_dir)?;
        let mut samples_loaded = 0;
        let max_samples = if self.config.max_samples < 0 {
            usize::MAX
        } else {
            self.config.max_samples as usize
        };

        for entry in entries {
            if samples_loaded >= max_samples {
                break;
            }

            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                let extension = path.extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("")
                    .to_lowercase();

                if self.config.image_formats.contains(&extension) {
                    let filename = path.file_stem()
                        .and_then(|name| name.to_str())
                        .unwrap_or("unknown");

                    let text = captions.get(filename)
                        .cloned()
                        .unwrap_or_else(|| format!("Image: {}", filename));

                    let sample = DataSample::new(
                        filename.to_string(),
                        path,
                        text,
                    );

                    self.add_sample(sample)?;
                    samples_loaded += 1;
                }
            }
        }

        if self.config.shuffle {
            self.shuffle();
        }

        Ok(())
    }

    fn load_captions_file<P: AsRef<Path>>(&self, path: P) -> Result<HashMap<String, String>> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut captions = HashMap::new();

        for line in reader.lines() {
            let line = line?;
            if let Some((filename, caption)) = line.split_once('\t') {
                captions.insert(filename.to_string(), caption.to_string());
            }
        }

        Ok(captions)
    }

    fn compute_statistics(&mut self) -> Result<()> {
        if self.samples.is_empty() {
            return Ok(());
        }

        let mut text_lengths = Vec::new();
        let mut image_sizes = Vec::new();
        let mut format_counts = HashMap::new();
        let mut words = std::collections::HashSet::new();

        for sample in &self.samples {
            // Text statistics
            text_lengths.push(sample.text.len());
            for word in sample.text.split_whitespace() {
                words.insert(word.to_lowercase());
            }

            // Image statistics
            if let Ok(metadata) = std::fs::metadata(&sample.image_path) {
                image_sizes.push(metadata.len() as f64);
            }

            // Format statistics
            let extension = sample.image_path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("unknown")
                .to_lowercase();
            *format_counts.entry(extension).or_insert(0) += 1;
        }

        // Compute text statistics
        let avg_text_length = text_lengths.iter().sum::<usize>() as f64 / text_lengths.len() as f64;
        let min_text_length = *text_lengths.iter().min().unwrap_or(&0);
        let max_text_length = *text_lengths.iter().max().unwrap_or(&0);
        
        let text_std = {
            let variance = text_lengths.iter()
                .map(|&len| (len as f64 - avg_text_length).powi(2))
                .sum::<f64>() / text_lengths.len() as f64;
            variance.sqrt()
        };

        // Compute image statistics
        let avg_image_size = if !image_sizes.is_empty() {
            image_sizes.iter().sum::<f64>() / image_sizes.len() as f64
        } else {
            0.0
        };

        self.metadata.stats = DatasetStats {
            avg_text_length,
            text_length_stats: (min_text_length, max_text_length, text_std),
            avg_image_size,
            image_formats: format_counts,
            vocab_size: words.len(),
        };

        self.metadata.num_samples = self.samples.len();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_data_sample_creation() {
        let sample = DataSample::new(
            "test_001".into(),
            PathBuf::from("test.jpg"),
            "A test image".into(),
        );

        assert_eq!(sample.id, "test_001");
        assert_eq!(sample.text, "A test image");
        assert!(sample.metadata.is_empty());
        assert!(sample.ground_truth.is_none());
    }

    #[test]
    fn test_data_sample_with_metadata() {
        let sample = DataSample::new(
            "test_001".into(),
            PathBuf::from("test.jpg"),
            "A test image".into(),
        )
        .with_metadata("width".into(), serde_json::Value::from(224))
        .with_ground_truth("Expected response".into());

        assert_eq!(sample.metadata.len(), 1);
        assert!(sample.ground_truth.is_some());
    }

    #[test]
    fn test_dataset_config_default() {
        let config = DatasetConfig::default();
        assert_eq!(config.max_samples, -1);
        assert!(!config.shuffle);
        assert_eq!(config.seed, 42);
        assert!(config.image_formats.contains(&"jpg".to_string()));
    }

    #[test]
    fn test_empty_dataset() {
        let config = DatasetConfig {
            root_dir: PathBuf::from("/nonexistent"),
            ..Default::default()
        };
        
        let result = VisionLanguageDataset::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_dataset_from_samples() {
        let temp_dir = TempDir::new().unwrap();
        let config = DatasetConfig {
            root_dir: temp_dir.path().to_path_buf(),
            validate_data: false,
            ..Default::default()
        };

        let mut dataset = VisionLanguageDataset {
            config,
            samples: Vec::new(),
            metadata: DatasetMetadata {
                name: "Test".into(),
                version: "1.0".into(),
                description: "Test dataset".into(),
                num_samples: 0,
                created_at: chrono::Utc::now().to_rfc3339(),
                stats: DatasetStats {
                    avg_text_length: 0.0,
                    text_length_stats: (0, 0, 0.0),
                    avg_image_size: 0.0,
                    image_formats: HashMap::new(),
                    vocab_size: 0,
                },
            },
        };

        let sample = DataSample::new(
            "test".into(),
            PathBuf::from("test.jpg"),
            "Test sample".into(),
        );

        dataset.add_sample(sample).unwrap();
        assert_eq!(dataset.len(), 1);
        assert!(!dataset.is_empty());
    }
}