//! Performance benchmarks for Tiny-VLM-Rust-WASM
//!
//! Comprehensive benchmarking suite for measuring inference speed and efficiency

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tiny_vlm::{prelude::*, FastVLM, ModelConfig, InferenceConfig};

fn create_test_image_data(width: usize, height: usize) -> Vec<u8> {
    // Create a more realistic test image (simplified PNG structure)
    let mut data = vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
    ];
    
    // Add enough data to simulate image processing load
    let pixel_data = vec![128u8; width * height * 3]; // RGB data
    data.extend_from_slice(&pixel_data);
    data
}

fn create_benchmark_model() -> FastVLM {
    let config = ModelConfig::default();
    FastVLM::new(config).expect("Failed to create benchmark model")
}

fn benchmark_full_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_inference");
    
    let mut model = create_benchmark_model();
    let test_image = create_test_image_data(224, 224);
    let test_prompt = "Describe this image in detail";
    let inference_config = InferenceConfig::default();
    
    group.bench_function("standard_inference", |b| {
        b.iter(|| {
            black_box(model.infer(
                black_box(&test_image),
                black_box(test_prompt),
                black_box(inference_config.clone())
            )).unwrap()
        })
    });
    
    // Benchmark different prompt lengths
    for prompt_len in [10, 50, 100, 200].iter() {
        let long_prompt = "Describe this image ".repeat(*prompt_len / 20);
        group.bench_with_input(
            BenchmarkId::new("variable_prompt_length", prompt_len),
            prompt_len,
            |b, _| {
                b.iter(|| {
                    black_box(model.infer(
                        black_box(&test_image),
                        black_box(&long_prompt),
                        black_box(inference_config.clone())
                    )).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_vision_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("vision_encoding");
    
    let mut model = create_benchmark_model();
    
    // Benchmark different image sizes
    for &size in [64, 128, 224, 512].iter() {
        let test_image = create_test_image_data(size, size);
        group.throughput(Throughput::Bytes((size * size * 3) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("image_encoding", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(model.encode_image(black_box(&test_image))).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_text_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_encoding");
    
    let mut model = create_benchmark_model();
    
    // Benchmark different text lengths
    for &length in [10, 50, 100, 500, 1000].iter() {
        let test_text = "word ".repeat(length / 5);
        group.throughput(Throughput::Bytes(test_text.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("text_encoding", length),
            &length,
            |b, _| {
                b.iter(|| {
                    black_box(model.encode_text(black_box(&test_text))).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    
    let mut model = create_benchmark_model();
    
    group.bench_function("memory_stats", |b| {
        b.iter(|| {
            black_box(model.memory_stats())
        })
    });
    
    group.bench_function("memory_compaction", |b| {
        b.iter(|| {
            model.compact_memory()
        })
    });
    
    group.finish();
}

fn benchmark_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation");
    
    let test_image = create_test_image_data(224, 224);
    let test_text = "This is a test prompt for validation benchmarking";
    
    group.bench_function("image_validation", |b| {
        b.iter(|| {
            black_box(tiny_vlm::validation::validate_image_data(black_box(&test_image)))
        })
    });
    
    group.bench_function("text_validation", |b| {
        b.iter(|| {
            black_box(tiny_vlm::validation::validate_text_input(black_box(test_text)))
        })
    });
    
    let model_config = ModelConfig::default();
    group.bench_function("model_config_validation", |b| {
        b.iter(|| {
            black_box(tiny_vlm::validation::validate_model_config(black_box(&model_config)))
        })
    });
    
    let inference_config = InferenceConfig::default();
    group.bench_function("inference_config_validation", |b| {
        b.iter(|| {
            black_box(tiny_vlm::validation::validate_inference_config(black_box(&inference_config)))
        })
    });
    
    group.finish();
}

fn benchmark_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");
    
    // Test tensor creation and basic operations
    group.bench_function("tensor_creation_small", |b| {
        b.iter(|| {
            let shape = tiny_vlm::memory::TensorShape::new(&[32, 32, 3]).unwrap();
            black_box(tiny_vlm::memory::Tensor::<f32>::zeros(shape).unwrap())
        })
    });
    
    group.bench_function("tensor_creation_large", |b| {
        b.iter(|| {
            let shape = tiny_vlm::memory::TensorShape::new(&[224, 224, 3]).unwrap();
            black_box(tiny_vlm::memory::Tensor::<f32>::zeros(shape).unwrap())
        })
    });
    
    // Test memory pool operations
    group.bench_function("memory_pool_allocation", |b| {
        let mut pool = tiny_vlm::memory::MemoryPool::<f32>::new(1_000_000);
        let shape = tiny_vlm::memory::TensorShape::new(&[100, 100]).unwrap();
        
        b.iter(|| {
            black_box(pool.allocate(shape).unwrap())
        })
    });
    
    group.finish();
}

fn benchmark_sampling_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_strategies");
    
    let mut model = create_benchmark_model();
    let test_image = create_test_image_data(224, 224);
    let test_prompt = "Test sampling";
    
    // Benchmark different sampling configurations
    let configs = vec![
        ("greedy", InferenceConfig {
            deterministic: true,
            ..InferenceConfig::default()
        }),
        ("random", InferenceConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            deterministic: false,
            ..InferenceConfig::default()
        }),
        ("top_k", InferenceConfig {
            temperature: 0.8,
            top_k: 50,
            deterministic: false,
            ..InferenceConfig::default()
        }),
        ("top_p", InferenceConfig {
            temperature: 0.8,
            top_p: 0.9,
            deterministic: false,
            ..InferenceConfig::default()
        }),
    ];
    
    for (name, config) in configs {
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(model.infer(
                    black_box(&test_image),
                    black_box(test_prompt),
                    black_box(config.clone())
                )).unwrap()
            })
        });
    }
    
    group.finish();
}

fn benchmark_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    let mut model = create_benchmark_model();
    let test_image = create_test_image_data(224, 224);
    let inference_config = InferenceConfig::default();
    
    // Simulate batch processing by running multiple inferences
    for &batch_size in [1, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("sequential_batch", batch_size),
            &batch_size,
            |b, &size| {
                b.iter(|| {
                    for i in 0..size {
                        let prompt = format!("Batch item {}", i);
                        black_box(model.infer(
                            black_box(&test_image),
                            black_box(&prompt),
                            black_box(inference_config.clone())
                        )).unwrap();
                    }
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_error_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_handling");
    
    let mut model = create_benchmark_model();
    let valid_image = create_test_image_data(224, 224);
    let invalid_image = vec![]; // Empty image
    let valid_prompt = "Valid prompt";
    let invalid_config = InferenceConfig {
        temperature: -1.0, // Invalid temperature
        ..InferenceConfig::default()
    };
    
    group.bench_function("valid_input_processing", |b| {
        b.iter(|| {
            black_box(model.infer(
                black_box(&valid_image),
                black_box(valid_prompt),
                black_box(InferenceConfig::default())
            )).unwrap()
        })
    });
    
    group.bench_function("invalid_image_handling", |b| {
        b.iter(|| {
            let _ = black_box(model.infer(
                black_box(&invalid_image),
                black_box(valid_prompt),
                black_box(InferenceConfig::default())
            ));
        })
    });
    
    group.bench_function("invalid_config_handling", |b| {
        b.iter(|| {
            let _ = black_box(model.infer(
                black_box(&valid_image),
                black_box(valid_prompt),
                black_box(invalid_config.clone())
            ));
        })
    });
    
    group.finish();
}

// Specialized benchmarks for mobile performance targets
fn benchmark_mobile_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("mobile_performance");
    
    // Target: sub-200ms inference on mobile hardware
    group.measurement_time(std::time::Duration::from_secs(30));
    group.sample_size(50);
    
    let mut model = create_benchmark_model();
    let mobile_image = create_test_image_data(224, 224); // Standard mobile camera input
    let mobile_prompt = "What's in this photo?"; // Typical mobile query
    let mobile_config = InferenceConfig {
        max_length: 50, // Shorter responses for mobile
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40, // Top-k sampling for mobile
        deterministic: false,
        memory_limit_mb: 50, // Conservative memory limit
    };
    
    group.bench_function("mobile_inference_target", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let result = black_box(model.infer(
                black_box(&mobile_image),
                black_box(mobile_prompt),
                black_box(mobile_config.clone())
            )).unwrap();
            let elapsed = start.elapsed();
            
            // Log performance for analysis
            if elapsed > std::time::Duration::from_millis(200) {
                eprintln!("Warning: Mobile inference took {:?} (target: <200ms)", elapsed);
            }
            
            black_box(result)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_full_inference,
    benchmark_vision_encoding,
    benchmark_text_encoding,
    benchmark_memory_operations,
    benchmark_validation,
    benchmark_tensor_operations,
    benchmark_sampling_strategies,
    benchmark_batch_processing,
    benchmark_error_handling,
    benchmark_mobile_targets
);

criterion_main!(benches);