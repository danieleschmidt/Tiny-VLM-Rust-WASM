//! Performance benchmark suite for Tiny VLM
//! 
//! Comprehensive benchmarks covering:
//! - Full inference pipeline performance
//! - Memory usage patterns
//! - SIMD optimization effectiveness
//! - Batch processing capabilities

use tiny_vlm::{FastVLM, InferenceConfig, ModelConfig, Result};
use std::time::{Duration, Instant};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("🏁 Tiny VLM Performance Benchmark Suite");
    println!("========================================");

    // Initialize model
    let config = ModelConfig::default();
    let mut model = FastVLM::new(config)?;
    println!("✅ Model initialized");

    let inference_config = InferenceConfig {
        max_length: 50,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        deterministic: true, // For consistent benchmarking
        memory_limit_mb: 200,
    };

    let mut results = BenchmarkResults::new();

    // Benchmark 1: Cold start performance
    println!("\n🧊 Cold Start Benchmark");
    println!("------------------------");
    
    let image_data = create_test_image(224, 224, ImagePattern::Gradient);
    let prompt = "What is in this image?";
    
    let start_time = Instant::now();
    let response = model.infer(&image_data, prompt, inference_config.clone())?;
    let cold_start_time = start_time.elapsed();
    
    println!("⏱️  Cold start time: {:.2}ms", cold_start_time.as_secs_f32() * 1000.0);
    println!("💬 Response: \"{}\"", response);
    results.record("cold_start_ms", cold_start_time.as_secs_f32() * 1000.0);

    // Benchmark 2: Warm inference performance
    println!("\n🔥 Warm Inference Benchmark");
    println!("---------------------------");
    
    let num_warmup = 5;
    let num_iterations = 20;
    
    // Warmup
    for _ in 0..num_warmup {
        let _ = model.infer(&image_data, "warmup", inference_config.clone())?;
    }
    
    let mut inference_times = Vec::new();
    for i in 0..num_iterations {
        let prompt = format!("Describe image iteration {}", i + 1);
        let start_time = Instant::now();
        let _response = model.infer(&image_data, &prompt, inference_config.clone())?;
        let inference_time = start_time.elapsed();
        inference_times.push(inference_time.as_secs_f32() * 1000.0);
    }
    
    let avg_time = inference_times.iter().sum::<f32>() / inference_times.len() as f32;
    let min_time = inference_times.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_time = inference_times.iter().fold(0.0, |a, &b| a.max(b));
    
    println!("📊 Warm inference stats ({} iterations):", num_iterations);
    println!("   Average: {:.2}ms", avg_time);
    println!("   Min: {:.2}ms", min_time);
    println!("   Max: {:.2}ms", max_time);
    println!("   🎯 Target: <200ms");
    
    results.record("warm_avg_ms", avg_time);
    results.record("warm_min_ms", min_time);
    results.record("warm_max_ms", max_time);

    // Benchmark 3: Memory usage patterns
    println!("\n💾 Memory Usage Benchmark");
    println!("-------------------------");
    
    let initial_memory = model.memory_stats();
    println!("📈 Initial memory: {:.2} MB", bytes_to_mb(initial_memory.allocated_memory));
    
    // Run inference with different image sizes
    let sizes = vec![224, 256, 384];
    for size in sizes {
        let test_image = create_test_image(size, size, ImagePattern::Random);
        let _response = model.infer(&test_image, "Test memory usage", inference_config.clone())?;
        let memory_after = model.memory_stats();
        println!("🖼️  {}x{} image - Memory: {:.2} MB", size, size, bytes_to_mb(memory_after.allocated_memory));
        results.record(&format!("memory_{}x{}_mb", size, size), bytes_to_mb(memory_after.allocated_memory));
    }
    
    // Test memory cleanup
    model.compact_memory();
    let memory_after_cleanup = model.memory_stats();
    println!("🧹 After cleanup: {:.2} MB", bytes_to_mb(memory_after_cleanup.allocated_memory));
    results.record("memory_after_cleanup_mb", bytes_to_mb(memory_after_cleanup.allocated_memory));

    // Benchmark 4: Different prompt lengths
    println!("\n📝 Prompt Length Benchmark");
    println!("--------------------------");
    
    let prompts = vec![
        ("Short", "What is this?"),
        ("Medium", "Can you describe what you see in this image in detail?"),
        ("Long", "Please provide a comprehensive analysis of this image, including colors, objects, composition, artistic style, and any notable features or patterns you observe."),
    ];
    
    for (length_type, prompt) in prompts {
        let start_time = Instant::now();
        let response = model.infer(&image_data, prompt, inference_config.clone())?;
        let prompt_time = start_time.elapsed();
        
        println!("📏 {} prompt ({} chars): {:.2}ms", length_type, prompt.len(), prompt_time.as_secs_f32() * 1000.0);
        println!("   Response length: {} chars", response.len());
        
        results.record(&format!("prompt_{}_ms", length_type.to_lowercase()), prompt_time.as_secs_f32() * 1000.0);
    }

    // Benchmark 5: Different image patterns
    println!("\n🖼️  Image Pattern Benchmark");
    println!("---------------------------");
    
    let patterns = vec![
        ("Solid", ImagePattern::Solid),
        ("Gradient", ImagePattern::Gradient),
        ("Checkerboard", ImagePattern::Checkerboard),
        ("Random", ImagePattern::Random),
    ];
    
    for (pattern_name, pattern) in patterns {
        let test_image = create_test_image(224, 224, pattern);
        let start_time = Instant::now();
        let _response = model.infer(&test_image, "Describe this pattern", inference_config.clone())?;
        let pattern_time = start_time.elapsed();
        
        println!("🎨 {} pattern: {:.2}ms", pattern_name, pattern_time.as_secs_f32() * 1000.0);
        results.record(&format!("pattern_{}_ms", pattern_name.to_lowercase()), pattern_time.as_secs_f32() * 1000.0);
    }

    // Benchmark 6: Encoding performance
    println!("\n🔬 Individual Component Benchmark");
    println!("---------------------------------");
    
    // Image encoding
    let start_time = Instant::now();
    let _image_features = model.encode_image(&image_data)?;
    let image_encoding_time = start_time.elapsed();
    println!("🖼️  Image encoding: {:.2}ms", image_encoding_time.as_secs_f32() * 1000.0);
    results.record("image_encoding_ms", image_encoding_time.as_secs_f32() * 1000.0);
    
    // Text encoding
    let start_time = Instant::now();
    let _text_features = model.encode_text("This is a test prompt for encoding performance")?;
    let text_encoding_time = start_time.elapsed();
    println!("📝 Text encoding: {:.2}ms", text_encoding_time.as_secs_f32() * 1000.0);
    results.record("text_encoding_ms", text_encoding_time.as_secs_f32() * 1000.0);

    // Generate benchmark report
    println!("\n📊 BENCHMARK REPORT");
    println!("==================");
    results.print_summary();

    // Performance assessment
    println!("\n🎯 Performance Assessment");
    println!("========================");
    
    if avg_time < 200.0 {
        println!("✅ PASS: Average inference time ({:.2}ms) meets <200ms target", avg_time);
    } else {
        println!("❌ FAIL: Average inference time ({:.2}ms) exceeds 200ms target", avg_time);
    }
    
    let peak_memory = results.get_max_memory();
    if peak_memory < 150.0 {
        println!("✅ PASS: Peak memory usage ({:.2} MB) is reasonable", peak_memory);
    } else {
        println!("⚠️  WARNING: High memory usage ({:.2} MB) - consider optimization", peak_memory);
    }
    
    println!("💡 Note: These are simplified benchmarks with placeholder operations");
    println!("🚀 In production, full SIMD optimization would provide significant speedup");

    Ok(())
}

/// Image pattern types for testing
#[derive(Copy, Clone)]
enum ImagePattern {
    Solid,
    Gradient,
    Checkerboard,
    Random,
}

/// Create test images with different patterns
fn create_test_image(width: usize, height: usize, pattern: ImagePattern) -> Vec<u8> {
    let mut image_data = Vec::with_capacity(width * height * 3);
    
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = match pattern {
                ImagePattern::Solid => (128, 128, 128),
                ImagePattern::Gradient => {
                    let r = ((x * 255) / width) as u8;
                    let g = ((y * 255) / height) as u8;
                    let b = 128;
                    (r, g, b)
                },
                ImagePattern::Checkerboard => {
                    if (x / 32 + y / 32) % 2 == 0 { (255, 255, 255) } else { (0, 0, 0) }
                },
                ImagePattern::Random => {
                    let seed = (x * 1234 + y * 5678) % 65536;
                    let r = ((seed * 123) % 256) as u8;
                    let g = ((seed * 456) % 256) as u8;
                    let b = ((seed * 789) % 256) as u8;
                    (r, g, b)
                }
            };
            
            image_data.push(r);
            image_data.push(g);
            image_data.push(b);
        }
    }
    
    image_data
}

/// Benchmark results storage and analysis
struct BenchmarkResults {
    metrics: HashMap<String, f32>,
}

impl BenchmarkResults {
    fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }
    
    fn record(&mut self, metric_name: &str, value: f32) {
        self.metrics.insert(metric_name.to_string(), value);
    }
    
    fn print_summary(&self) {
        println!("Key Performance Metrics:");
        
        if let Some(cold_start) = self.metrics.get("cold_start_ms") {
            println!("  🧊 Cold start: {:.2}ms", cold_start);
        }
        
        if let Some(warm_avg) = self.metrics.get("warm_avg_ms") {
            println!("  🔥 Warm average: {:.2}ms", warm_avg);
        }
        
        if let Some(min_time) = self.metrics.get("warm_min_ms") {
            println!("  ⚡ Best time: {:.2}ms", min_time);
        }
        
        let memory_keys: Vec<_> = self.metrics.keys()
            .filter(|k| k.contains("memory_") && k.ends_with("_mb"))
            .collect();
        
        if !memory_keys.is_empty() {
            println!("  💾 Memory usage:");
            for key in memory_keys {
                if let Some(value) = self.metrics.get(key) {
                    println!("     {}: {:.2} MB", key, value);
                }
            }
        }
    }
    
    fn get_max_memory(&self) -> f32 {
        self.metrics.iter()
            .filter(|(k, _)| k.contains("memory_") && k.ends_with("_mb"))
            .map(|(_, &v)| v)
            .fold(0.0, f32::max)
    }
}

/// Convert bytes to megabytes
fn bytes_to_mb(bytes: usize) -> f32 {
    bytes as f32 / (1024.0 * 1024.0)
}