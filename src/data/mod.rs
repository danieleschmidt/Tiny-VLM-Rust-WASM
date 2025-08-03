//! Data management and dataset handling

pub mod dataset;
pub mod loader;
pub mod transforms;
pub mod cache;

pub use dataset::{VisionLanguageDataset, DataSample, DatasetConfig, DatasetError};
pub use loader::{DataLoader, DataLoaderConfig, BatchLoader};
pub use transforms::{ImageTransform, TextTransform, CompositeTransform};
pub use cache::{DataCache, CacheConfig, CacheStats};