//! Advanced Scaling and Performance Optimization
//!
//! High-performance scaling mechanisms including auto-scaling, advanced caching,
//! concurrent processing, and distributed execution.

use crate::{Result, TinyVlmError, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Advanced scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedScalingConfig {
    pub enable_auto_scaling: bool,
    pub enable_adaptive_batching: bool,
    pub enable_distributed_processing: bool,
    pub enable_smart_caching: bool,
    pub enable_workload_prediction: bool,
    pub min_instances: usize,
    pub max_instances: usize,
    pub target_cpu_percent: f64,
    pub target_latency_ms: f64,
    pub target_cpu_utilization: f64,
    pub target_memory_utilization: f64,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub scale_up_cooldown_seconds: u64,
    pub scale_down_cooldown_seconds: u64,
    pub batch_size_auto_tuning: bool,
    pub predictive_scaling: bool,
    pub batch_size_min: usize,
    pub batch_size_max: usize,
    pub adaptive_batch_window_seconds: u64,
    pub cache_size_mb: usize,
    pub cache_ttl_seconds: u64,
    pub preemptive_scaling: bool,
}

impl Default for AdvancedScalingConfig {
    fn default() -> Self {
        Self {
            enable_auto_scaling: true,
            enable_adaptive_batching: true,
            enable_distributed_processing: true,
            enable_smart_caching: true,
            enable_workload_prediction: true,
            min_instances: 1,
            max_instances: 10,
            target_cpu_percent: 70.0,
            target_latency_ms: 100.0,
            target_cpu_utilization: 70.0,
            target_memory_utilization: 80.0,
            scale_up_threshold: 80.0,
            scale_down_threshold: 30.0,
            scale_up_cooldown_seconds: 300,
            scale_down_cooldown_seconds: 600,
            batch_size_auto_tuning: true,
            predictive_scaling: true,
            batch_size_min: 1,
            batch_size_max: 32,
            adaptive_batch_window_seconds: 60,
            cache_size_mb: 1024,
            cache_ttl_seconds: 3600,
            preemptive_scaling: true,
        }
    }
}

/// Scaling decision types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScalingDecision {
    ScaleUp(usize),
    ScaleDown(usize),
    Maintain,
    Emergency(String),
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub avg_latency_ms: f64,
    pub requests_per_second: f64,
    pub error_rate: f64,
    pub timestamp: std::time::SystemTime,
    pub network_utilization: f64,
    pub gpu_utilization: f64,
    pub queue_length: usize,
    pub active_connections: usize,
    pub average_response_time_ms: f64,
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            avg_latency_ms: 0.0,
            requests_per_second: 0.0,
            error_rate: 0.0,
            timestamp: std::time::SystemTime::now(),
            network_utilization: 0.0,
            gpu_utilization: 0.0,
            queue_length: 0,
            active_connections: 0,
            average_response_time_ms: 0.0,
        }
    }
}

/// Advanced auto-scaler with predictive capabilities
pub struct AdvancedAutoScaler {
    config: AdvancedScalingConfig,
    current_instances: AtomicUsize,
    target_instances: AtomicUsize,
    last_scale_action: Arc<Mutex<Option<Instant>>>,
    metrics_history: Arc<RwLock<Vec<ResourceMetrics>>>,
    workload_predictor: WorkloadPredictor,
    scaling_policies: Vec<ScalingPolicy>,
}

impl AdvancedAutoScaler {
    pub fn new(config: AdvancedScalingConfig) -> Self {
        Self {
            current_instances: AtomicUsize::new(config.min_instances),
            target_instances: AtomicUsize::new(config.min_instances),
            last_scale_action: Arc::new(Mutex::new(None)),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            workload_predictor: WorkloadPredictor::new(),
            scaling_policies: vec![
                ScalingPolicy::CpuBased { threshold: config.scale_up_threshold },
                ScalingPolicy::MemoryBased { threshold: config.target_memory_utilization },
                ScalingPolicy::QueueBased { max_queue_length: 100 },
                ScalingPolicy::ResponseTimeBased { max_response_time_ms: 1000.0 },
            ],
            config,
        }
    }

    /// Evaluate scaling decision based on current metrics
    pub fn evaluate_scaling(&mut self, current_metrics: &ResourceMetrics) -> Result<ScalingDecision> {
        // Store metrics for historical analysis
        {
            let mut history = self.metrics_history.write().map_err(|_| {
                TinyVlmError::InternalError("Failed to acquire metrics history lock".to_string())
            })?;
            history.push(current_metrics.clone());
            
            // Keep only last 1000 metrics
            if history.len() > 1000 {
                history.drain(0..100);
            }
        }

        let current_instances = self.current_instances.load(Ordering::Relaxed);
        let mut scaling_votes = Vec::new();

        // Evaluate each scaling policy
        for policy in &self.scaling_policies {
            if let Some(vote) = policy.evaluate(current_metrics, current_instances) {
                scaling_votes.push(vote);
            }
        }

        // Predictive scaling
        if self.config.enable_workload_prediction && self.config.preemptive_scaling {
            if let Some(predicted_load) = self.workload_predictor.predict_load(current_metrics) {
                if predicted_load > 1.5 * current_metrics.requests_per_second {
                    scaling_votes.push(ScalingVote::ScaleUp(2));
                }
            }
        }

        // Apply scaling decision logic
        let decision = self.make_scaling_decision(scaling_votes, current_instances)?;
        
        // Apply cooldown logic
        if let Some(last_action) = *self.last_scale_action.lock().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire last scale action lock".to_string())
        })? {
            let cooldown_duration = match decision {
                ScalingDecision::ScaleUp(_) => Duration::from_secs(self.config.scale_up_cooldown_seconds),
                ScalingDecision::ScaleDown(_) => Duration::from_secs(self.config.scale_down_cooldown_seconds),
                _ => Duration::from_secs(0),
            };
            
            if last_action.elapsed() < cooldown_duration {
                return Ok(ScalingDecision::Maintain);
            }
        }

        // Update state if scaling
        match &decision {
            ScalingDecision::ScaleUp(instances) | ScalingDecision::ScaleDown(instances) => {
                self.target_instances.store(*instances, Ordering::Relaxed);
                *self.last_scale_action.lock().map_err(|_| {
                    TinyVlmError::InternalError("Failed to acquire last scale action lock".to_string())
                })? = Some(Instant::now());
            }
            _ => {}
        }

        Ok(decision)
    }

    fn make_scaling_decision(&self, votes: Vec<ScalingVote>, current_instances: usize) -> Result<ScalingDecision> {
        let scale_up_votes = votes.iter().filter(|v| matches!(v, ScalingVote::ScaleUp(_))).count();
        let scale_down_votes = votes.iter().filter(|v| matches!(v, ScalingVote::ScaleDown(_))).count();
        let emergency_votes = votes.iter().filter(|v| matches!(v, ScalingVote::Emergency(_))).count();

        // Emergency scaling takes precedence
        if emergency_votes > 0 {
            if let Some(ScalingVote::Emergency(reason)) = votes.iter().find(|v| matches!(v, ScalingVote::Emergency(_))) {
                return Ok(ScalingDecision::Emergency(reason.clone()));
            }
        }

        // Determine scaling direction
        if scale_up_votes > scale_down_votes {
            let scale_amount = votes.iter()
                .filter_map(|v| match v {
                    ScalingVote::ScaleUp(amount) => Some(*amount),
                    _ => None,
                })
                .max()
                .unwrap_or(1);
            
            let new_instances = (current_instances + scale_amount).min(self.config.max_instances);
            if new_instances > current_instances {
                Ok(ScalingDecision::ScaleUp(new_instances))
            } else {
                Ok(ScalingDecision::Maintain)
            }
        } else if scale_down_votes > scale_up_votes {
            let scale_amount = votes.iter()
                .filter_map(|v| match v {
                    ScalingVote::ScaleDown(amount) => Some(*amount),
                    _ => None,
                })
                .max()
                .unwrap_or(1);
            
            let new_instances = current_instances.saturating_sub(scale_amount).max(self.config.min_instances);
            if new_instances < current_instances {
                Ok(ScalingDecision::ScaleDown(new_instances))
            } else {
                Ok(ScalingDecision::Maintain)
            }
        } else {
            Ok(ScalingDecision::Maintain)
        }
    }

    /// Update current instance count (called by external scaling executor)
    pub fn update_current_instances(&self, instances: usize) {
        self.current_instances.store(instances, Ordering::Relaxed);
    }

    pub fn get_current_instances(&self) -> usize {
        self.current_instances.load(Ordering::Relaxed)
    }

    pub fn get_target_instances(&self) -> usize {
        self.target_instances.load(Ordering::Relaxed)
    }

    /// Get scaling metrics summary
    pub fn get_scaling_metrics(&self) -> Result<ScalingMetrics> {
        let history = self.metrics_history.read().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire metrics history lock".to_string())
        })?;

        let current_instances = self.current_instances.load(Ordering::Relaxed);
        let target_instances = self.target_instances.load(Ordering::Relaxed);

        let recent_metrics = history.iter().rev().take(10).collect::<Vec<_>>();
        let avg_cpu = if !recent_metrics.is_empty() {
            recent_metrics.iter().map(|m| m.cpu_utilization).sum::<f64>() / recent_metrics.len() as f64
        } else {
            0.0
        };

        let avg_memory = if !recent_metrics.is_empty() {
            recent_metrics.iter().map(|m| m.memory_utilization).sum::<f64>() / recent_metrics.len() as f64
        } else {
            0.0
        };

        Ok(ScalingMetrics {
            current_instances,
            target_instances,
            min_instances: self.config.min_instances,
            max_instances: self.config.max_instances,
            average_cpu_utilization: avg_cpu,
            average_memory_utilization: avg_memory,
            scaling_events_count: 0, // Would be tracked in a real implementation
            last_scaling_action: self.last_scale_action.lock().unwrap().map(|t| t.elapsed().as_secs()).unwrap_or(0),
        })
    }
}

/// Scaling policy definitions
#[derive(Debug, Clone)]
pub enum ScalingPolicy {
    CpuBased { threshold: f64 },
    MemoryBased { threshold: f64 },
    QueueBased { max_queue_length: usize },
    ResponseTimeBased { max_response_time_ms: f64 },
    ErrorRateBased { max_error_rate: f64 },
}

#[derive(Debug, Clone)]
pub enum ScalingVote {
    ScaleUp(usize),
    ScaleDown(usize),
    Maintain,
    Emergency(String),
}

impl ScalingPolicy {
    pub fn evaluate(&self, metrics: &ResourceMetrics, _current_instances: usize) -> Option<ScalingVote> {
        match self {
            ScalingPolicy::CpuBased { threshold } => {
                if metrics.cpu_utilization > *threshold {
                    Some(ScalingVote::ScaleUp(1))
                } else if metrics.cpu_utilization < threshold * 0.5 {
                    Some(ScalingVote::ScaleDown(1))
                } else {
                    None
                }
            }
            ScalingPolicy::MemoryBased { threshold } => {
                if metrics.memory_utilization > *threshold {
                    Some(ScalingVote::ScaleUp(1))
                } else if metrics.memory_utilization < threshold * 0.5 {
                    Some(ScalingVote::ScaleDown(1))
                } else {
                    None
                }
            }
            ScalingPolicy::QueueBased { max_queue_length } => {
                if metrics.queue_length > *max_queue_length {
                    Some(ScalingVote::ScaleUp(2))
                } else if metrics.queue_length < max_queue_length / 4 {
                    Some(ScalingVote::ScaleDown(1))
                } else {
                    None
                }
            }
            ScalingPolicy::ResponseTimeBased { max_response_time_ms } => {
                if metrics.average_response_time_ms > *max_response_time_ms {
                    Some(ScalingVote::ScaleUp(1))
                } else {
                    None
                }
            }
            ScalingPolicy::ErrorRateBased { max_error_rate } => {
                if metrics.error_rate > *max_error_rate {
                    Some(ScalingVote::Emergency("High error rate detected".to_string()))
                } else {
                    None
                }
            }
        }
    }
}

/// Workload prediction engine
#[derive(Debug)]
pub struct WorkloadPredictor {
    historical_patterns: Vec<WorkloadPattern>,
    trend_analyzer: TrendAnalyzer,
}

#[derive(Debug, Clone)]
pub struct WorkloadPattern {
    pub hour_of_day: u8,
    pub day_of_week: u8,
    pub average_load: f64,
    pub peak_load: f64,
    pub confidence: f64,
}

impl WorkloadPredictor {
    pub fn new() -> Self {
        Self {
            historical_patterns: Vec::new(),
            trend_analyzer: TrendAnalyzer::new(),
        }
    }

    pub fn predict_load(&mut self, current_metrics: &ResourceMetrics) -> Option<f64> {
        // Simple prediction based on current trends
        let current_load = current_metrics.requests_per_second;
        let trend = self.trend_analyzer.analyze_trend(current_load);
        
        match trend {
            LoadTrend::Increasing => Some(current_load * 1.2),
            LoadTrend::Decreasing => Some(current_load * 0.8),
            LoadTrend::Stable => Some(current_load),
            LoadTrend::Volatile => Some(current_load * 1.5), // Be conservative with volatile loads
        }
    }

    pub fn learn_pattern(&mut self, metrics: &ResourceMetrics) {
        // In a real implementation, this would use time-series analysis
        // For now, just store basic pattern information
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let hour = ((now.as_secs() / 3600) % 24) as u8;
        let day_of_week = ((now.as_secs() / 86400) % 7) as u8;

        let pattern = WorkloadPattern {
            hour_of_day: hour,
            day_of_week,
            average_load: metrics.requests_per_second,
            peak_load: metrics.requests_per_second,
            confidence: 0.5,
        };

        self.historical_patterns.push(pattern);
        
        // Keep only recent patterns
        if self.historical_patterns.len() > 1000 {
            self.historical_patterns.drain(0..100);
        }
    }
}

impl Default for WorkloadPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Trend analysis for load prediction
#[derive(Debug)]
pub struct TrendAnalyzer {
    recent_values: Vec<f64>,
    max_values: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoadTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

impl TrendAnalyzer {
    pub fn new() -> Self {
        Self {
            recent_values: Vec::new(),
            max_values: 20,
        }
    }

    pub fn analyze_trend(&mut self, new_value: f64) -> LoadTrend {
        self.recent_values.push(new_value);
        
        if self.recent_values.len() > self.max_values {
            self.recent_values.remove(0);
        }

        if self.recent_values.len() < 3 {
            return LoadTrend::Stable;
        }

        let len = self.recent_values.len();
        let recent_avg = self.recent_values[len-3..].iter().sum::<f64>() / 3.0;
        let older_avg = self.recent_values[0..len-3].iter().sum::<f64>() / (len - 3) as f64;

        let change_ratio = (recent_avg - older_avg) / older_avg.max(0.001);

        // Calculate volatility
        let variance = self.recent_values.iter()
            .map(|&x| (x - recent_avg).powi(2))
            .sum::<f64>() / self.recent_values.len() as f64;
        let volatility = variance.sqrt() / recent_avg.max(0.001);

        if volatility > 0.5 {
            LoadTrend::Volatile
        } else if change_ratio > 0.1 {
            LoadTrend::Increasing
        } else if change_ratio < -0.1 {
            LoadTrend::Decreasing
        } else {
            LoadTrend::Stable
        }
    }
}

impl Default for TrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced adaptive batching system
pub struct AdaptiveBatchProcessor {
    config: AdvancedScalingConfig,
    current_batch_size: AtomicUsize,
    batch_performance_history: Arc<RwLock<Vec<BatchPerformance>>>,
    batch_optimizer: BatchOptimizer,
}

#[derive(Debug, Clone)]
pub struct BatchPerformance {
    pub batch_size: usize,
    pub processing_time_ms: u64,
    pub throughput: f64,
    pub memory_usage_mb: f64,
    pub timestamp: u64,
}

impl AdaptiveBatchProcessor {
    pub fn new(config: AdvancedScalingConfig) -> Self {
        Self {
            current_batch_size: AtomicUsize::new(config.batch_size_min),
            batch_performance_history: Arc::new(RwLock::new(Vec::new())),
            batch_optimizer: BatchOptimizer::new(),
            config,
        }
    }

    /// Process a batch of requests with adaptive sizing
    pub fn process_batch(&mut self, requests: Vec<InferenceRequest>) -> Result<Vec<InferenceResponse>> {
        let start_time = Instant::now();
        let batch_size = requests.len();
        
        // Process the batch (simplified implementation)
        let responses = requests.into_iter().map(|req| {
            InferenceResponse {
                request_id: req.request_id,
                result: format!("Processed: {}", req.input),
                processing_time_ms: 10, // Simulated processing time
                metadata: HashMap::new(),
            }
        }).collect();

        let processing_time = start_time.elapsed();
        let throughput = batch_size as f64 / processing_time.as_secs_f64();

        // Record performance
        let performance = BatchPerformance {
            batch_size,
            processing_time_ms: processing_time.as_millis() as u64,
            throughput,
            memory_usage_mb: self.estimate_memory_usage(batch_size),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        {
            let mut history = self.batch_performance_history.write().map_err(|_| {
                TinyVlmError::InternalError("Failed to acquire batch performance history lock".to_string())
            })?;
            history.push(performance.clone());
            
            if history.len() > 100 {
                history.drain(0..10);
            }
        }

        // Optimize batch size for next iteration
        self.optimize_batch_size(&performance)?;

        Ok(responses)
    }

    fn optimize_batch_size(&mut self, latest_performance: &BatchPerformance) -> Result<()> {
        let history = self.batch_performance_history.read().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire batch performance history lock".to_string())
        })?;

        if let Some(optimal_size) = self.batch_optimizer.find_optimal_batch_size(&history) {
            let clamped_size = optimal_size
                .max(self.config.batch_size_min)
                .min(self.config.batch_size_max);
            
            self.current_batch_size.store(clamped_size, Ordering::Relaxed);
        }

        Ok(())
    }

    fn estimate_memory_usage(&self, batch_size: usize) -> f64 {
        // Simplified memory estimation
        (batch_size as f64 * 2.5).min(self.config.cache_size_mb as f64 * 0.8)
    }

    pub fn get_optimal_batch_size(&self) -> usize {
        self.current_batch_size.load(Ordering::Relaxed)
    }
}

/// Batch size optimizer
#[derive(Debug)]
pub struct BatchOptimizer {
    efficiency_threshold: f64,
}

impl BatchOptimizer {
    pub fn new() -> Self {
        Self {
            efficiency_threshold: 0.8,
        }
    }

    pub fn find_optimal_batch_size(&self, history: &[BatchPerformance]) -> Option<usize> {
        if history.len() < 5 {
            return None;
        }

        // Find batch size with highest efficiency (throughput / memory usage)
        let mut best_efficiency = 0.0;
        let mut best_size = 1;

        for performance in history.iter().rev().take(20) {
            let efficiency = performance.throughput / performance.memory_usage_mb.max(1.0);
            if efficiency > best_efficiency {
                best_efficiency = efficiency;
                best_size = performance.batch_size;
            }
        }

        Some(best_size)
    }
}

impl Default for BatchOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Smart caching system with intelligent eviction
pub struct SmartCache {
    config: AdvancedScalingConfig,
    cache_storage: Arc<RwLock<HashMap<String, CacheEntry>>>,
    access_patterns: Arc<RwLock<HashMap<String, AccessPattern>>>,
    cache_stats: CacheStatistics,
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub data: Vec<u8>,
    pub created_at: u64,
    pub last_accessed: u64,
    pub access_count: usize,
    pub size_bytes: usize,
    pub ttl_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub frequency: f64,
    pub recency: f64,
    pub temporal_locality: f64,
    pub prediction_score: f64,
}

#[derive(Debug)]
pub struct CacheStatistics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub total_size_bytes: AtomicU64,
}

impl SmartCache {
    pub fn new(config: AdvancedScalingConfig) -> Self {
        Self {
            config,
            cache_storage: Arc::new(RwLock::new(HashMap::new())),
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: CacheStatistics {
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                evictions: AtomicU64::new(0),
                total_size_bytes: AtomicU64::new(0),
            },
        }
    }

    /// Get item from cache
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let mut cache = self.cache_storage.write().ok()?;
        
        if let Some(entry) = cache.get_mut(key) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            // Check TTL
            if now > entry.created_at + entry.ttl_seconds {
                cache.remove(key);
                self.cache_stats.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Update access statistics
            entry.last_accessed = now;
            entry.access_count += 1;
            
            self.cache_stats.hits.fetch_add(1, Ordering::Relaxed);
            self.update_access_pattern(key, now);
            
            Some(entry.data.clone())
        } else {
            self.cache_stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Put item in cache with intelligent eviction
    pub fn put(&self, key: String, data: Vec<u8>) -> Result<()> {
        let data_size = data.len();
        let max_cache_size = self.config.cache_size_mb * 1024 * 1024;
        
        // Check if we need to make space
        while self.cache_stats.total_size_bytes.load(Ordering::Relaxed) + data_size as u64 > max_cache_size as u64 {
            self.evict_item()?;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let entry = CacheEntry {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size_bytes: data_size,
            ttl_seconds: self.config.cache_ttl_seconds,
        };

        {
            let mut cache = self.cache_storage.write().map_err(|_| {
                TinyVlmError::InternalError("Failed to acquire cache storage lock".to_string())
            })?;
            cache.insert(key.clone(), entry);
        }

        self.cache_stats.total_size_bytes.fetch_add(data_size as u64, Ordering::Relaxed);
        self.update_access_pattern(&key, now);

        Ok(())
    }

    /// Intelligent cache eviction using LRU + access patterns
    fn evict_item(&self) -> Result<()> {
        let mut cache = self.cache_storage.write().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire cache storage lock".to_string())
        })?;

        let access_patterns = self.access_patterns.read().map_err(|_| {
            TinyVlmError::InternalError("Failed to acquire access patterns lock".to_string())
        })?;

        // Find item with lowest eviction score
        let mut lowest_score = f64::MAX;
        let mut evict_key: Option<String> = None;

        for (key, entry) in cache.iter() {
            let pattern = access_patterns.get(key);
            let score = self.calculate_eviction_score(entry, pattern);
            
            if score < lowest_score {
                lowest_score = score;
                evict_key = Some(key.clone());
            }
        }

        if let Some(key) = evict_key {
            if let Some(entry) = cache.remove(&key) {
                self.cache_stats.total_size_bytes.fetch_sub(entry.size_bytes as u64, Ordering::Relaxed);
                self.cache_stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    fn calculate_eviction_score(&self, entry: &CacheEntry, pattern: Option<&AccessPattern>) -> f64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Base score on recency and frequency
        let recency_score = 1.0 / (now - entry.last_accessed + 1) as f64;
        let frequency_score = entry.access_count as f64;
        
        let pattern_score = pattern.map(|p| p.prediction_score).unwrap_or(0.5);
        
        // Higher score means less likely to be evicted
        recency_score * frequency_score * pattern_score
    }

    fn update_access_pattern(&self, key: &str, timestamp: u64) {
        if let Ok(mut patterns) = self.access_patterns.write() {
            let pattern = patterns.entry(key.to_string()).or_insert(AccessPattern {
                frequency: 0.0,
                recency: 0.0,
                temporal_locality: 0.0,
                prediction_score: 0.5,
            });

            // Update access pattern (simplified)
            pattern.frequency += 1.0;
            pattern.recency = 1.0 / (timestamp + 1) as f64;
            pattern.prediction_score = (pattern.frequency * pattern.recency).min(1.0);
        }
    }

    pub fn get_cache_statistics(&self) -> CacheStats {
        let hits = self.cache_stats.hits.load(Ordering::Relaxed);
        let misses = self.cache_stats.misses.load(Ordering::Relaxed);
        let total_requests = hits + misses;
        let hit_rate = if total_requests > 0 {
            hits as f64 / total_requests as f64
        } else {
            0.0
        };

        CacheStats {
            hits,
            misses,
            hit_rate,
            evictions: self.cache_stats.evictions.load(Ordering::Relaxed),
            total_size_bytes: self.cache_stats.total_size_bytes.load(Ordering::Relaxed),
            max_size_bytes: (self.config.cache_size_mb * 1024 * 1024) as u64,
        }
    }
}

/// Request and response types for batch processing
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub request_id: String,
    pub input: String,
    pub priority: u8,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub request_id: String,
    pub result: String,
    pub processing_time_ms: u64,
    pub metadata: HashMap<String, String>,
}

/// Scaling metrics for monitoring
#[derive(Debug, Serialize, Deserialize)]
pub struct ScalingMetrics {
    pub current_instances: usize,
    pub target_instances: usize,
    pub min_instances: usize,
    pub max_instances: usize,
    pub average_cpu_utilization: f64,
    pub average_memory_utilization: f64,
    pub scaling_events_count: u64,
    pub last_scaling_action: u64,
}

/// Cache statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub evictions: u64,
    pub total_size_bytes: u64,
    pub max_size_bytes: u64,
}

/// Advanced scaling manager that orchestrates all scaling components
pub struct AdvancedScalingManager {
    config: AdvancedScalingConfig,
    auto_scaler: AdvancedAutoScaler,
    batch_processor: AdaptiveBatchProcessor,
    smart_cache: SmartCache,
    is_running: AtomicBool,
}

impl AdvancedScalingManager {
    pub fn new(config: AdvancedScalingConfig) -> Result<Self> {
        Ok(Self {
            auto_scaler: AdvancedAutoScaler::new(config.clone()),
            batch_processor: AdaptiveBatchProcessor::new(config.clone()),
            smart_cache: SmartCache::new(config.clone()),
            is_running: AtomicBool::new(false),
            config,
        })
    }

    /// Start the scaling manager
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Relaxed);
    }

    /// Stop the scaling manager
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    /// Process requests with full scaling optimization
    pub fn process_requests(&mut self, requests: Vec<InferenceRequest>) -> Result<Vec<InferenceResponse>> {
        if !self.is_running.load(Ordering::Relaxed) {
            return Err(TinyVlmError::InternalError("Scaling manager is not running".to_string()));
        }

        // Process batch with adaptive sizing
        self.batch_processor.process_batch(requests)
    }

    /// Evaluate and apply scaling decisions
    pub fn evaluate_scaling(&mut self, metrics: &ResourceMetrics) -> Result<ScalingDecision> {
        self.auto_scaler.evaluate_scaling(metrics)
    }

    /// Get comprehensive scaling status
    pub fn get_scaling_status(&self) -> Result<AdvancedScalingStatus> {
        let scaling_metrics = self.auto_scaler.get_scaling_metrics()?;
        let cache_stats = self.smart_cache.get_cache_statistics();
        let optimal_batch_size = self.batch_processor.get_optimal_batch_size();

        Ok(AdvancedScalingStatus {
            scaling_metrics,
            cache_stats,
            optimal_batch_size,
            is_running: self.is_running.load(Ordering::Relaxed),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    /// Cache data
    pub fn cache_put(&self, key: String, data: Vec<u8>) -> Result<()> {
        self.smart_cache.put(key, data)
    }

    /// Retrieve cached data
    pub fn cache_get(&self, key: &str) -> Option<Vec<u8>> {
        self.smart_cache.get(key)
    }

    /// Evaluate scaling decision based on metrics
    pub fn evaluate_scaling_decision(&mut self, metrics: &ResourceMetrics) -> Result<ScalingDecision> {
        self.auto_scaler.evaluate_scaling(metrics)
    }

    /// Get efficiency statistics for reporting
    pub fn get_efficiency_stats(&self) -> Result<ScalingEfficiencyStats> {
        let scaling_metrics = self.auto_scaler.get_scaling_metrics()?;
        let cache_stats = self.smart_cache.get_cache_statistics();

        Ok(ScalingEfficiencyStats {
            current_instances: scaling_metrics.current_instances,
            scale_up_events: scaling_metrics.scaling_events_count / 2, // Simplified
            scale_down_events: scaling_metrics.scaling_events_count / 2, // Simplified
            efficiency_score: if scaling_metrics.average_cpu_utilization > 0.0 {
                1.0 - (scaling_metrics.average_cpu_utilization - self.config.target_cpu_percent / 100.0).abs()
            } else {
                0.8
            }.max(0.0).min(1.0),
            avg_resource_utilization: scaling_metrics.average_cpu_utilization,
        })
    }
}

/// Comprehensive scaling status
#[derive(Debug, Serialize, Deserialize)]
pub struct AdvancedScalingStatus {
    pub scaling_metrics: ScalingMetrics,
    pub cache_stats: CacheStats,
    pub optimal_batch_size: usize,
    pub is_running: bool,
    pub timestamp: u64,
}

/// Scaling efficiency statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct ScalingEfficiencyStats {
    pub current_instances: usize,
    pub scale_up_events: u64,
    pub scale_down_events: u64,
    pub efficiency_score: f64,
    pub avg_resource_utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_scaling_config() {
        let config = AdvancedScalingConfig::default();
        assert!(config.enable_auto_scaling);
        assert!(config.enable_adaptive_batching);
        assert_eq!(config.min_instances, 1);
        assert_eq!(config.max_instances, 10);
    }

    #[test]
    fn test_auto_scaler_creation() {
        let config = AdvancedScalingConfig::default();
        let scaler = AdvancedAutoScaler::new(config);
        assert_eq!(scaler.get_current_instances(), 1);
        assert_eq!(scaler.get_target_instances(), 1);
    }

    #[test]
    fn test_scaling_policy_evaluation() {
        let policy = ScalingPolicy::CpuBased { threshold: 80.0 };
        let metrics = ResourceMetrics {
            cpu_utilization: 90.0,
            ..Default::default()
        };

        let vote = policy.evaluate(&metrics, 2);
        assert!(matches!(vote, Some(ScalingVote::ScaleUp(1))));
    }

    #[test]
    fn test_workload_predictor() {
        let mut predictor = WorkloadPredictor::new();
        let metrics = ResourceMetrics {
            requests_per_second: 100.0,
            ..Default::default()
        };

        let prediction = predictor.predict_load(&metrics);
        assert!(prediction.is_some());
        assert!(prediction.unwrap() > 0.0);
    }

    #[test]
    fn test_trend_analyzer() {
        let mut analyzer = TrendAnalyzer::new();
        
        // Add increasing trend
        for i in 1..=10 {
            analyzer.analyze_trend(i as f64);
        }
        
        let trend = analyzer.analyze_trend(11.0);
        assert_eq!(trend, LoadTrend::Increasing);
    }

    #[test]
    fn test_adaptive_batch_processor() {
        let config = AdvancedScalingConfig::default();
        let mut processor = AdaptiveBatchProcessor::new(config);

        let requests = vec![
            InferenceRequest {
                request_id: "1".to_string(),
                input: "test1".to_string(),
                priority: 1,
                timeout_ms: 1000,
            },
            InferenceRequest {
                request_id: "2".to_string(),
                input: "test2".to_string(),
                priority: 1,
                timeout_ms: 1000,
            },
        ];

        let responses = processor.process_batch(requests);
        assert!(responses.is_ok());
        let responses = responses.unwrap();
        assert_eq!(responses.len(), 2);
    }

    #[test]
    fn test_smart_cache() {
        let config = AdvancedScalingConfig::default();
        let cache = SmartCache::new(config);

        let key = "test_key".to_string();
        let data = b"test_data".to_vec();

        // Test put and get
        cache.put(key.clone(), data.clone()).unwrap();
        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), data);

        // Test cache miss
        let missing = cache.get("missing_key");
        assert!(missing.is_none());

        // Test statistics
        let stats = cache.get_cache_statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!(stats.hit_rate > 0.0);
    }

    #[test]
    fn test_advanced_scaling_manager() {
        let config = AdvancedScalingConfig::default();
        let mut manager = AdvancedScalingManager::new(config);

        manager.start();
        assert!(manager.is_running.load(Ordering::Relaxed));

        // Test request processing
        let requests = vec![
            InferenceRequest {
                request_id: "1".to_string(),
                input: "test".to_string(),
                priority: 1,
                timeout_ms: 1000,
            },
        ];

        let responses = manager.process_requests(requests);
        assert!(responses.is_ok());

        // Test scaling evaluation
        let metrics = ResourceMetrics::default();
        let decision = manager.evaluate_scaling(&metrics);
        assert!(decision.is_ok());

        // Test status
        let status = manager.get_scaling_status();
        assert!(status.is_ok());

        manager.stop();
        assert!(!manager.is_running.load(Ordering::Relaxed));
    }

    #[test]
    fn test_cache_eviction() {
        let mut config = AdvancedScalingConfig::default();
        config.cache_size_mb = 1; // Very small cache to force eviction
        let cache = SmartCache::new(config);

        // Fill cache beyond capacity
        for i in 0..1000 {
            let key = format!("key_{}", i);
            let data = vec![0u8; 2048]; // 2KB per entry
            let _ = cache.put(key, data);
        }

        let stats = cache.get_cache_statistics();
        assert!(stats.evictions > 0);
        assert!(stats.total_size_bytes <= 1024 * 1024); // Should not exceed 1MB
    }
}