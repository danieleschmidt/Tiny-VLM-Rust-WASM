//! Caching system for processed data

use crate::{Result, TinyVlmError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Configuration for data caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,
    /// Cache entry time-to-live in seconds
    pub ttl_seconds: u64,
    /// Whether to compress cached data
    pub compress: bool,
    /// Cache cleanup interval in seconds
    pub cleanup_interval_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from("./cache"),
            max_size_bytes: 1024 * 1024 * 1024, // 1GB
            ttl_seconds: 24 * 60 * 60, // 24 hours
            compress: true,
            cleanup_interval_seconds: 60 * 60, // 1 hour
        }
    }
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Entry key
    pub key: String,
    /// File path
    pub file_path: PathBuf,
    /// Size in bytes
    pub size_bytes: usize,
    /// Creation timestamp
    pub created_at: u64,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Access count
    pub access_count: u64,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total entries
    pub total_entries: usize,
    /// Total size in bytes
    pub total_size_bytes: usize,
    /// Hit count
    pub hits: u64,
    /// Miss count
    pub misses: u64,
    /// Hit ratio
    pub hit_ratio: f64,
}

/// Data cache for storing processed samples
pub struct DataCache {
    /// Configuration
    config: CacheConfig,
    /// Cache entries metadata
    entries: HashMap<String, CacheEntry>,
    /// Statistics
    stats: CacheStats,
    /// Last cleanup time
    last_cleanup: u64,
}

impl DataCache {
    /// Create a new data cache
    pub fn new(config: CacheConfig) -> Result<Self> {
        // Create cache directory if it doesn't exist
        std::fs::create_dir_all(&config.cache_dir)?;

        let mut cache = Self {
            config,
            entries: HashMap::new(),
            stats: CacheStats {
                total_entries: 0,
                total_size_bytes: 0,
                hits: 0,
                misses: 0,
                hit_ratio: 0.0,
            },
            last_cleanup: Self::current_timestamp(),
        };

        // Load existing cache metadata
        cache.load_metadata()?;
        
        Ok(cache)
    }

    /// Store data in cache
    pub fn store(&mut self, key: &str, data: &[u8]) -> Result<()> {
        let file_path = self.get_cache_file_path(key);
        
        // Write data to file
        std::fs::write(&file_path, data)?;
        
        let size_bytes = data.len();
        let now = Self::current_timestamp();
        
        // Update or create cache entry
        let entry = CacheEntry {
            key: key.to_string(),
            file_path,
            size_bytes,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        };

        // Remove old entry if it exists
        if let Some(old_entry) = self.entries.remove(key) {
            self.stats.total_size_bytes -= old_entry.size_bytes;
        }

        // Add new entry
        self.entries.insert(key.to_string(), entry);
        self.stats.total_entries = self.entries.len();
        self.stats.total_size_bytes += size_bytes;

        // Check if cleanup is needed
        if self.needs_cleanup() {
            self.cleanup()?;
        }

        // Save metadata
        self.save_metadata()?;
        
        Ok(())
    }

    /// Retrieve data from cache
    pub fn get(&mut self, key: &str) -> Result<Option<Vec<u8>>> {
        if let Some(entry) = self.entries.get_mut(key) {
            // Check if entry is still valid
            let now = Self::current_timestamp();
            if now - entry.created_at > self.config.ttl_seconds {
                // Entry expired, remove it
                let expired_entry = self.entries.remove(key).unwrap();
                self.stats.total_size_bytes -= expired_entry.size_bytes;
                self.stats.total_entries = self.entries.len();
                let _ = std::fs::remove_file(&expired_entry.file_path);
                
                self.stats.misses += 1;
                self.update_hit_ratio();
                return Ok(None);
            }

            // Update access information
            entry.last_accessed = now;
            entry.access_count += 1;

            // Read data from file
            match std::fs::read(&entry.file_path) {
                Ok(data) => {
                    self.stats.hits += 1;
                    self.update_hit_ratio();
                    Ok(Some(data))
                }
                Err(_) => {
                    // File missing, remove entry
                    let missing_entry = self.entries.remove(key).unwrap();
                    self.stats.total_size_bytes -= missing_entry.size_bytes;
                    self.stats.total_entries = self.entries.len();
                    
                    self.stats.misses += 1;
                    self.update_hit_ratio();
                    Ok(None)
                }
            }
        } else {
            self.stats.misses += 1;
            self.update_hit_ratio();
            Ok(None)
        }
    }

    /// Check if key exists in cache
    pub fn contains(&self, key: &str) -> bool {
        if let Some(entry) = self.entries.get(key) {
            let now = Self::current_timestamp();
            now - entry.created_at <= self.config.ttl_seconds
        } else {
            false
        }
    }

    /// Remove entry from cache
    pub fn remove(&mut self, key: &str) -> Result<bool> {
        if let Some(entry) = self.entries.remove(key) {
            self.stats.total_size_bytes -= entry.size_bytes;
            self.stats.total_entries = self.entries.len();
            let _ = std::fs::remove_file(&entry.file_path);
            self.save_metadata()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Clear all cache entries
    pub fn clear(&mut self) -> Result<()> {
        for entry in self.entries.values() {
            let _ = std::fs::remove_file(&entry.file_path);
        }
        
        self.entries.clear();
        self.stats.total_entries = 0;
        self.stats.total_size_bytes = 0;
        
        self.save_metadata()?;
        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Perform cache cleanup
    pub fn cleanup(&mut self) -> Result<()> {
        let now = Self::current_timestamp();
        let mut to_remove = Vec::new();

        // Find expired entries
        for (key, entry) in &self.entries {
            if now - entry.created_at > self.config.ttl_seconds {
                to_remove.push(key.clone());
            }
        }

        // Remove expired entries
        for key in &to_remove {
            if let Some(entry) = self.entries.remove(key) {
                self.stats.total_size_bytes -= entry.size_bytes;
                let _ = std::fs::remove_file(&entry.file_path);
            }
        }

        // If still over size limit, remove least recently used entries
        if self.stats.total_size_bytes > self.config.max_size_bytes {
            self.evict_lru_entries()?;
        }

        self.stats.total_entries = self.entries.len();
        self.last_cleanup = now;
        self.save_metadata()?;
        
        Ok(())
    }

    // Private methods

    fn get_cache_file_path(&self, key: &str) -> PathBuf {
        // Create a safe filename from the key
        let safe_key = key
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect::<String>();
        
        self.config.cache_dir.join(format!("{}.cache", safe_key))
    }

    fn needs_cleanup(&self) -> bool {
        let now = Self::current_timestamp();
        now - self.last_cleanup > self.config.cleanup_interval_seconds ||
        self.stats.total_size_bytes > self.config.max_size_bytes
    }

    fn evict_lru_entries(&mut self) -> Result<()> {
        // Sort entries by last access time
        let mut entries: Vec<_> = self.entries.iter().collect();
        entries.sort_by_key(|(_, entry)| entry.last_accessed);

        // Remove entries until under size limit
        for (key, entry) in entries {
            if self.stats.total_size_bytes <= self.config.max_size_bytes {
                break;
            }

            let key = key.clone();
            if let Some(entry) = self.entries.remove(&key) {
                self.stats.total_size_bytes -= entry.size_bytes;
                let _ = std::fs::remove_file(&entry.file_path);
            }
        }

        Ok(())
    }

    fn load_metadata(&mut self) -> Result<()> {
        let metadata_path = self.config.cache_dir.join("metadata.json");
        
        if metadata_path.exists() {
            let metadata_data = std::fs::read_to_string(metadata_path)?;
            if let Ok(entries) = serde_json::from_str::<HashMap<String, CacheEntry>>(&metadata_data) {
                // Validate entries and update stats
                let mut valid_entries = HashMap::new();
                let mut total_size = 0;

                for (key, entry) in entries {
                    if entry.file_path.exists() {
                        total_size += entry.size_bytes;
                        valid_entries.insert(key, entry);
                    }
                }

                self.entries = valid_entries;
                self.stats.total_entries = self.entries.len();
                self.stats.total_size_bytes = total_size;
            }
        }

        Ok(())
    }

    fn save_metadata(&self) -> Result<()> {
        let metadata_path = self.config.cache_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&self.entries)?;
        std::fs::write(metadata_path, metadata_json)?;
        Ok(())
    }

    fn update_hit_ratio(&mut self) {
        let total_requests = self.stats.hits + self.stats.misses;
        self.stats.hit_ratio = if total_requests > 0 {
            self.stats.hits as f64 / total_requests as f64
        } else {
            0.0
        };
    }

    fn current_timestamp() -> u64 {
        #[cfg(feature = "std")]
        {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        }
        
        #[cfg(not(feature = "std"))]
        {
            // Fallback for no_std environments
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert!(config.max_size_bytes > 0);
        assert!(config.ttl_seconds > 0);
        assert!(config.compress);
    }

    #[test]
    fn test_cache_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let cache = DataCache::new(config);
        assert!(cache.is_ok());
    }

    #[test]
    fn test_cache_store_and_get() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let mut cache = DataCache::new(config).unwrap();
        
        let key = "test_key";
        let data = b"test data";
        
        // Store data
        cache.store(key, data).unwrap();
        
        // Retrieve data
        let retrieved = cache.get(key).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), data);
        
        // Check stats
        assert_eq!(cache.stats().total_entries, 1);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn test_cache_miss() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let mut cache = DataCache::new(config).unwrap();
        
        // Try to get non-existent key
        let result = cache.get("nonexistent").unwrap();
        assert!(result.is_none());
        
        // Check stats
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hit_ratio, 0.0);
    }

    #[test]
    fn test_cache_contains() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let mut cache = DataCache::new(config).unwrap();
        
        let key = "test_key";
        let data = b"test data";
        
        assert!(!cache.contains(key));
        
        cache.store(key, data).unwrap();
        assert!(cache.contains(key));
    }

    #[test]
    fn test_cache_remove() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let mut cache = DataCache::new(config).unwrap();
        
        let key = "test_key";
        let data = b"test data";
        
        cache.store(key, data).unwrap();
        assert!(cache.contains(key));
        
        let removed = cache.remove(key).unwrap();
        assert!(removed);
        assert!(!cache.contains(key));
        
        // Try to remove again
        let removed_again = cache.remove(key).unwrap();
        assert!(!removed_again);
    }

    #[test]
    fn test_cache_clear() {
        let temp_dir = TempDir::new().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let mut cache = DataCache::new(config).unwrap();
        
        // Store multiple entries
        for i in 0..5 {
            let key = format!("key_{}", i);
            let data = format!("data_{}", i).into_bytes();
            cache.store(&key, &data).unwrap();
        }
        
        assert_eq!(cache.stats().total_entries, 5);
        
        cache.clear().unwrap();
        assert_eq!(cache.stats().total_entries, 0);
        assert_eq!(cache.stats().total_size_bytes, 0);
    }
}