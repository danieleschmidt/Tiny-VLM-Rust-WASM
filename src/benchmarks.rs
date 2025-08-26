//! Comprehensive Benchmarking Suite
//!
//! Advanced benchmarking tools for performance analysis and optimization validation.

use crate::{Result, Tensor, TensorShape};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;


/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub name: String,
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub input_sizes: Vec<Vec<usize>>,
    pub batch_sizes: Vec<usize>,
    pub precision_modes: Vec<PrecisionMode>,
    pub hardware_targets: Vec<HardwareTarget>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: "default_benchmark".to_string(),
            warmup_iterations: 10,
            measurement_iterations: 100,
            input_sizes: vec![
                vec![1, 224, 224, 3],   // Single image
                vec![4, 224, 224, 3],   // Small batch
                vec![16, 224, 224, 3],  // Medium batch
                vec![32, 224, 224, 3],  // Large batch
            ],
            batch_sizes: vec![1, 4, 8, 16, 32],
            precision_modes: vec![PrecisionMode::F32, PrecisionMode::F16],
            hardware_targets: vec![HardwareTarget::CPU, HardwareTarget::SIMD],
        }
    }
}

/// Precision modes for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PrecisionMode {
    F32,
    F16,
    INT8,
    INT4,
}

/// Hardware targets for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HardwareTarget {
    CPU,
    SIMD,
    GPU,
    NeuralEngine,
    WASM,
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub operation: String,
    pub input_shape: Vec<usize>,
    pub batch_size: usize,
    pub precision: PrecisionMode,
    pub hardware: HardwareTarget,
    pub latency_stats: LatencyStats,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub energy_consumption_mw: Option<f64>,
    pub metadata: HashMap<String, f64>,
}

/// Latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub mean_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub std_dev_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

/// Comprehensive benchmark report
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub config: BenchmarkConfig,
    pub system_info: SystemInfo,
    pub results: Vec<BenchmarkResult>,
    pub performance_summary: PerformanceSummary,
    pub optimization_recommendations: Vec<String>,
    pub timestamp: String,
}

/// System information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub os_version: String,
    pub rust_version: String,
    pub simd_features: Vec<String>,
    pub gpu_info: Option<String>,
}

/// Performance summary across all benchmarks
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub fastest_config: String,
    pub highest_throughput_ops_per_sec: f64,
    pub lowest_latency_ms: f64,
    pub most_efficient_ops_per_watt: Option<f64>,
    pub memory_efficiency_score: f64,
    pub scaling_efficiency: f64,
}

/// Benchmark suite for comprehensive performance analysis
pub struct BenchmarkSuite {
    operations: HashMap<String, Box<dyn BenchmarkOperation>>,
    system_info: SystemInfo,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            system_info: Self::detect_system_info(),
        }
    }

    /// Register a benchmarkable operation
    pub fn register_operation(&mut self, operation: Box<dyn BenchmarkOperation>) {
        let name = operation.name().to_string();
        self.operations.insert(name, operation);
    }

    /// Run comprehensive benchmark suite
    pub fn run_benchmarks(&mut self, config: BenchmarkConfig) -> Result<BenchmarkReport> {
        let mut results = Vec::new();
        let start_time = Instant::now();

        println!("🚀 Starting comprehensive benchmark suite: {}", config.name);
        println!("📊 System: {} cores, {:.1}GB RAM", 
            self.system_info.cpu_cores, self.system_info.memory_gb);

        // Pre-filter available hardware targets
        let available_hardware: Vec<_> = config.hardware_targets
            .iter()
            .filter(|&hw| self.is_hardware_available(hw))
            .collect();

        for (op_name, operation) in &mut self.operations {
            println!("🔧 Benchmarking operation: {}", op_name);
            
            for input_size in &config.input_sizes {
                for &batch_size in &config.batch_sizes {
                    for precision in &config.precision_modes {
                        for &hardware in &available_hardware {
                            let result = self.benchmark_operation(
                                operation.as_mut(),
                                op_name,
                                input_size,
                                batch_size,
                                precision,
                                hardware,
                                &config,
                            )?;
                            results.push(result);
                        }
                    }
                }
            }
        }

        let total_time = start_time.elapsed();
        println!("✅ Benchmark suite completed in {:.2}s", total_time.as_secs_f64());

        let performance_summary = self.analyze_performance(&results);
        let optimization_recommendations = self.generate_recommendations(&results);

        Ok(BenchmarkReport {
            config,
            system_info: self.system_info.clone(),
            results,
            performance_summary,
            optimization_recommendations,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        })
    }

    fn benchmark_operation(
        &self,
        operation: &mut dyn BenchmarkOperation,
        name: &str,
        input_size: &[usize],
        batch_size: usize,
        precision: &PrecisionMode,
        hardware: &HardwareTarget,
        config: &BenchmarkConfig,
    ) -> Result<BenchmarkResult> {
        // Create input tensor
        let mut shape = input_size.to_vec();
        shape[0] = batch_size; // Override batch dimension
        let input = Tensor::zeros(TensorShape::new(&shape)?)?;

        // Warmup phase
        for _ in 0..config.warmup_iterations {
            let _ = operation.execute(&input, precision, hardware)?;
        }

        // Measurement phase
        let mut latencies = Vec::new();
        let mut memory_usage = 0.0;
        let mut energy_consumption = None;

        for i in 0..config.measurement_iterations {
            let start_memory = self.get_memory_usage();
            let start_energy = self.get_energy_consumption();
            let start_time = Instant::now();
            
            let _result = operation.execute(&input, precision, hardware)?;
            
            let latency = start_time.elapsed();
            latencies.push(latency.as_millis() as f64);
            
            let end_memory = self.get_memory_usage();
            let end_energy = self.get_energy_consumption();
            
            memory_usage += (end_memory - start_memory).max(0.0);
            
            if let (Some(start), Some(end)) = (start_energy, end_energy) {
                energy_consumption = Some(energy_consumption.unwrap_or(0.0) + (end - start));
            }

            // Progress indicator
            if i % (config.measurement_iterations / 10).max(1) == 0 {
                print!(".");
            }
        }
        println!();

        let latency_stats = self.calculate_latency_stats(&latencies);
        let throughput = (batch_size as f64 * config.measurement_iterations as f64) / 
            (latency_stats.mean_ms * config.measurement_iterations as f64 / 1000.0);

        Ok(BenchmarkResult {
            operation: name.to_string(),
            input_shape: shape,
            batch_size,
            precision: precision.clone(),
            hardware: hardware.clone(),
            latency_stats,
            throughput_ops_per_sec: throughput,
            memory_usage_mb: memory_usage / config.measurement_iterations as f64,
            energy_consumption_mw: energy_consumption.map(|e| e / config.measurement_iterations as f64),
            metadata: HashMap::new(),
        })
    }

    fn calculate_latency_stats(&self, latencies: &[f64]) -> LatencyStats {
        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let variance = latencies.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / latencies.len() as f64;
        let std_dev = variance.sqrt();

        let percentile = |p: f64| -> f64 {
            let index = (p * sorted.len() as f64 / 100.0) as usize;
            sorted[index.min(sorted.len() - 1)]
        };

        LatencyStats {
            mean_ms: mean,
            median_ms: percentile(50.0),
            p95_ms: percentile(95.0),
            p99_ms: percentile(99.0),
            std_dev_ms: std_dev,
            min_ms: sorted[0],
            max_ms: sorted[sorted.len() - 1],
        }
    }

    fn analyze_performance(&self, results: &[BenchmarkResult]) -> PerformanceSummary {
        if results.is_empty() {
            return PerformanceSummary {
                fastest_config: "none".to_string(),
                highest_throughput_ops_per_sec: 0.0,
                lowest_latency_ms: f64::MAX,
                most_efficient_ops_per_watt: None,
                memory_efficiency_score: 0.0,
                scaling_efficiency: 0.0,
            };
        }

        let fastest = results.iter()
            .min_by(|a, b| a.latency_stats.mean_ms.partial_cmp(&b.latency_stats.mean_ms).unwrap())
            .unwrap();

        let highest_throughput = results.iter()
            .map(|r| r.throughput_ops_per_sec)
            .fold(0.0f64, f64::max);

        let lowest_latency = results.iter()
            .map(|r| r.latency_stats.mean_ms)
            .fold(f64::MAX, f64::min);

        let most_efficient_ops_per_watt = results.iter()
            .filter_map(|r| {
                r.energy_consumption_mw.map(|energy| {
                    if energy > 0.0 {
                        r.throughput_ops_per_sec / energy
                    } else {
                        0.0
                    }
                })
            })
            .fold(0.0f64, f64::max);

        let memory_efficiency = results.iter()
            .map(|r| {
                if r.memory_usage_mb > 0.0 {
                    r.throughput_ops_per_sec / r.memory_usage_mb
                } else {
                    0.0
                }
            })
            .fold(0.0f64, f64::max);

        let scaling_efficiency = self.calculate_scaling_efficiency(results);

        PerformanceSummary {
            fastest_config: format!("{} - {:?} on {:?}", 
                fastest.operation, fastest.precision, fastest.hardware),
            highest_throughput_ops_per_sec: highest_throughput,
            lowest_latency_ms: lowest_latency,
            most_efficient_ops_per_watt: if most_efficient_ops_per_watt > 0.0 { 
                Some(most_efficient_ops_per_watt) 
            } else { 
                None 
            },
            memory_efficiency_score: memory_efficiency,
            scaling_efficiency,
        }
    }

    fn calculate_scaling_efficiency(&self, results: &[BenchmarkResult]) -> f64 {
        // Calculate how well performance scales with batch size
        let mut efficiency_scores = Vec::new();

        for operation in results.iter().map(|r| &r.operation).collect::<std::collections::HashSet<_>>() {
            let op_results: Vec<_> = results.iter()
                .filter(|r| &r.operation == operation)
                .collect();

            if op_results.len() > 1 {
                // Compare throughput scaling
                let mut batch_throughputs: Vec<(usize, f64)> = op_results.iter()
                    .map(|r| (r.batch_size, r.throughput_ops_per_sec))
                    .collect();
                batch_throughputs.sort_by_key(|&(batch, _)| batch);

                if batch_throughputs.len() > 1 {
                    let ideal_scaling = batch_throughputs.last().unwrap().0 as f64 / 
                        batch_throughputs.first().unwrap().0 as f64;
                    let actual_scaling = batch_throughputs.last().unwrap().1 / 
                        batch_throughputs.first().unwrap().1;
                    
                    let efficiency = (actual_scaling / ideal_scaling).min(1.0);
                    efficiency_scores.push(efficiency);
                }
            }
        }

        if efficiency_scores.is_empty() {
            0.0
        } else {
            efficiency_scores.iter().sum::<f64>() / efficiency_scores.len() as f64
        }
    }

    fn generate_recommendations(&self, results: &[BenchmarkResult]) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze hardware performance
        let cpu_results: Vec<_> = results.iter()
            .filter(|r| matches!(r.hardware, HardwareTarget::CPU))
            .collect();
        let simd_results: Vec<_> = results.iter()
            .filter(|r| matches!(r.hardware, HardwareTarget::SIMD))
            .collect();

        if !cpu_results.is_empty() && !simd_results.is_empty() {
            let cpu_avg_latency = cpu_results.iter()
                .map(|r| r.latency_stats.mean_ms)
                .sum::<f64>() / cpu_results.len() as f64;
            let simd_avg_latency = simd_results.iter()
                .map(|r| r.latency_stats.mean_ms)
                .sum::<f64>() / simd_results.len() as f64;

            if simd_avg_latency < cpu_avg_latency * 0.8 {
                recommendations.push(format!(
                    "SIMD optimization provides {:.1}% performance improvement - prioritize SIMD code paths",
                    ((cpu_avg_latency - simd_avg_latency) / cpu_avg_latency) * 100.0
                ));
            }
        }

        // Analyze precision trade-offs
        let f32_results: Vec<_> = results.iter()
            .filter(|r| matches!(r.precision, PrecisionMode::F32))
            .collect();
        let f16_results: Vec<_> = results.iter()
            .filter(|r| matches!(r.precision, PrecisionMode::F16))
            .collect();

        if !f32_results.is_empty() && !f16_results.is_empty() {
            let f32_avg_latency = f32_results.iter()
                .map(|r| r.latency_stats.mean_ms)
                .sum::<f64>() / f32_results.len() as f64;
            let f16_avg_latency = f16_results.iter()
                .map(|r| r.latency_stats.mean_ms)
                .sum::<f64>() / f16_results.len() as f64;

            if f16_avg_latency < f32_avg_latency * 0.7 {
                recommendations.push(format!(
                    "F16 precision offers {:.1}% speedup - consider mixed precision training",
                    ((f32_avg_latency - f16_avg_latency) / f32_avg_latency) * 100.0
                ));
            }
        }

        // Analyze batch size efficiency
        let mut batch_efficiency = Vec::new();
        let mut grouped_results: HashMap<(&String, &HardwareTarget, &PrecisionMode), Vec<&BenchmarkResult>> = HashMap::new();
        
        for result in results {
            let key = (&result.operation, &result.hardware, &result.precision);
            grouped_results.entry(key).or_insert_with(Vec::new).push(result);
        }
        
        for (_, group_results) in grouped_results {
            if group_results.len() > 1 {
                let throughputs: Vec<_> = group_results.iter()
                    .map(|r| (r.batch_size, r.throughput_ops_per_sec))
                    .collect();
                
                if let Some(max_throughput) = throughputs.iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                    batch_efficiency.push((max_throughput.0, max_throughput.1));
                }
            }
        }

        if let Some(&(optimal_batch, _)) = batch_efficiency.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
            recommendations.push(format!(
                "Optimal batch size appears to be {} for maximum throughput",
                optimal_batch
            ));
        }

        // Memory usage recommendations
        let high_memory_results: Vec<_> = results.iter()
            .filter(|r| r.memory_usage_mb > 500.0)
            .collect();

        if !high_memory_results.is_empty() {
            recommendations.push(
                "High memory usage detected - consider implementing memory pooling and tensor reuse".to_string()
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Performance appears optimal across all configurations".to_string());
        }

        recommendations
    }

    fn detect_system_info() -> SystemInfo {
        SystemInfo {
            cpu_model: "Unknown CPU".to_string(), // Would detect actual CPU in real implementation
            cpu_cores: num_cpus::get(),
            memory_gb: 16.0, // Would detect actual memory
            os_version: std::env::consts::OS.to_string(),
            rust_version: "1.75.0".to_string(), // Would detect actual Rust version
            simd_features: vec!["AVX2".to_string(), "NEON".to_string()], // Would detect actual features
            gpu_info: None,
        }
    }

    fn is_hardware_available(&self, hardware: &HardwareTarget) -> bool {
        match hardware {
            HardwareTarget::CPU => true,
            HardwareTarget::SIMD => true, // Would check actual SIMD availability
            HardwareTarget::GPU => false, // Would check for GPU availability
            HardwareTarget::NeuralEngine => false, // Would check for Neural Engine
            HardwareTarget::WASM => cfg!(target_arch = "wasm32"),
        }
    }

    fn get_memory_usage(&self) -> f64 {
        // Placeholder - would implement actual memory monitoring
        100.0
    }

    fn get_energy_consumption(&self) -> Option<f64> {
        // Placeholder - would implement actual energy monitoring on supported platforms
        None
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for benchmarkable operations
pub trait BenchmarkOperation {
    fn name(&self) -> &str;
    fn execute(&mut self, input: &Tensor, precision: &PrecisionMode, hardware: &HardwareTarget) -> Result<Tensor>;
}

/// Matrix multiplication benchmark operation
pub struct MatMulBenchmark {
    name: String,
}

impl MatMulBenchmark {
    pub fn new() -> Self {
        Self {
            name: "matrix_multiplication".to_string(),
        }
    }
}

impl Default for MatMulBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkOperation for MatMulBenchmark {
    fn name(&self) -> &str {
        &self.name
    }

    fn execute(&mut self, input: &Tensor, _precision: &PrecisionMode, _hardware: &HardwareTarget) -> Result<Tensor> {
        // Simulate matrix multiplication
        let output_shape = input.shape().clone();
        let mut output = crate::memory::Tensor::zeros(output_shape)?;
        
        // Simple computation to simulate work
        for i in 0..output.data().len().min(1000) {
            output.data_mut()[i] = input.data()[i] * 2.0 + 1.0;
        }
        
        Ok(output)
    }
}

/// Convolution benchmark operation
pub struct ConvBenchmark {
    name: String,
}

impl ConvBenchmark {
    pub fn new() -> Self {
        Self {
            name: "convolution".to_string(),
        }
    }
}

impl Default for ConvBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkOperation for ConvBenchmark {
    fn name(&self) -> &str {
        &self.name
    }

    fn execute(&mut self, input: &Tensor, _precision: &PrecisionMode, _hardware: &HardwareTarget) -> Result<Tensor> {
        // Simulate convolution operation
        let mut output_dims = input.shape().dims[..input.shape().ndim].to_vec();
        output_dims[output_dims.len() - 1] = 64; // Output channels
        let output_shape = TensorShape::new(&output_dims)?;
        let mut output = crate::memory::Tensor::zeros(output_shape)?;
        
        // Simulate convolution computation
        for i in 0..output.data().len().min(1000) {
            output.data_mut()[i] = input.data()[i % input.data().len()] * 0.5;
        }
        
        Ok(output)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.measurement_iterations, 100);
        assert!(!config.input_sizes.is_empty());
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::new();
        assert_eq!(suite.operations.len(), 0);
        assert!(suite.system_info.cpu_cores > 0);
    }

    #[test]
    fn test_matmul_benchmark() {
        let mut benchmark = MatMulBenchmark::new();
        let input = crate::memory::Tensor::zeros(&TensorShape::new(vec![1, 4, 4, 1]));
        
        let result = benchmark.execute(&input, &PrecisionMode::F32, &HardwareTarget::CPU);
        assert!(result.is_ok());
    }

    #[test]
    fn test_conv_benchmark() {
        let mut benchmark = ConvBenchmark::new();
        let input = crate::memory::Tensor::zeros(&TensorShape::new(vec![1, 224, 224, 3]));
        
        let result = benchmark.execute(&input, &PrecisionMode::F32, &HardwareTarget::CPU);
        assert!(result.is_ok());
    }

    #[test]
    fn test_latency_stats_calculation() {
        let suite = BenchmarkSuite::new();
        let latencies = vec![10.0, 12.0, 8.0, 15.0, 11.0];
        let stats = suite.calculate_latency_stats(&latencies);
        
        assert_eq!(stats.mean_ms, 11.2);
        assert_eq!(stats.min_ms, 8.0);
        assert_eq!(stats.max_ms, 15.0);
        assert!(stats.std_dev_ms > 0.0);
    }

    #[test]
    fn test_operation_registration() {
        let mut suite = BenchmarkSuite::new();
        let matmul = Box::new(MatMulBenchmark::new());
        
        suite.register_operation(matmul);
        assert_eq!(suite.operations.len(), 1);
        assert!(suite.operations.contains_key("matrix_multiplication"));
    }
}