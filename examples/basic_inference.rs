//! Basic inference example for Tiny-VLM-Rust-WASM
//!
//! Demonstrates how to use the Vision-Language Model for basic image understanding

use tiny_vlm::{prelude::*, FastVLM, ModelConfig, InferenceConfig};
use std::time::Instant;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Tiny-VLM Basic Inference Example");
    println!("===================================");

    // Initialize logging
    env_logger::init();

    // Create model configuration
    let model_config = ModelConfig {
        vision_dim: 768,
        text_dim: 768,
        hidden_dim: 768,
        num_heads: 12,
        max_gen_length: 100,
        temperature: 0.8,
        ..ModelConfig::default()
    };

    println!("\n📋 Model Configuration:");
    println!("  Vision dim: {}", model_config.vision_dim);
    println!("  Text dim: {}", model_config.text_dim);
    println!("  Hidden dim: {}", model_config.hidden_dim);
    println!("  Attention heads: {}", model_config.num_heads);

    // Create model
    println!("\n🔧 Initializing model...");
    let start_time = Instant::now();
    let mut model = FastVLM::new(model_config)?;
    let init_time = start_time.elapsed();
    println!("✅ Model initialized in {:?}", init_time);

    // Create test image data
    println!("\n🖼️  Creating test image...");
    let test_image = create_test_image();
    println!("✅ Test image created ({} bytes)", test_image.len());

    // Set up inference configuration
    let inference_config = InferenceConfig {
        max_length: 50,
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        deterministic: false,
        memory_limit_mb: 100,
    };

    println!("\n🔍 Inference Configuration:");
    println!("  Max length: {}", inference_config.max_length);
    println!("  Temperature: {}", inference_config.temperature);
    println!("  Top-p: {}", inference_config.top_p);
    println!("  Top-k: {}", inference_config.top_k);

    // Test different prompts
    let test_prompts = vec![
        "What is in this image?",
        "Describe the contents of this image.",
        "What objects can you see?",
        "Tell me about this picture in detail.",
    ];

    println!("\n🤖 Running inference tests...");
    println!("================================");

    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n📝 Test {}: \"{}\"", i + 1, prompt);
        
        let start_time = Instant::now();
        match model.infer(&test_image, prompt, inference_config.clone()) {
            Ok(response) => {
                let inference_time = start_time.elapsed();
                println!("✅ Response ({:?}): {}", inference_time, response);
                
                // Check if inference meets mobile performance target
                if inference_time.as_millis() <= 200 {
                    println!("🎯 Mobile target achieved! (<200ms)");
                } else {
                    println!("⚠️  Above mobile target (>200ms)");
                }
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
    }

    // Test memory management
    println!("\n🧠 Memory Usage Analysis:");
    let memory_stats = model.memory_stats();
    println!("  Total memory: {} bytes", memory_stats.total_memory);
    println!("  Allocated: {} bytes", memory_stats.allocated_memory);
    println!("  Fragmentation: {:.2}%", memory_stats.fragmentation * 100.0);

    // Test memory compaction
    println!("\n🧹 Testing memory compaction...");
    let before_compaction = model.memory_stats().allocated_memory;
    model.compact_memory();
    let after_compaction = model.memory_stats().allocated_memory;
    let memory_saved = before_compaction.saturating_sub(after_compaction);
    println!("✅ Memory compacted: {} bytes saved", memory_saved);

    // Test individual encoding functions
    println!("\n🔬 Testing individual components:");
    
    // Vision encoding
    let start_time = Instant::now();
    match model.encode_image(&test_image) {
        Ok(vision_features) => {
            let vision_time = start_time.elapsed();
            println!("✅ Vision encoding: {:?} (shape: {:?})", 
                    vision_time, vision_features.shape());
        }
        Err(e) => {
            println!("❌ Vision encoding failed: {}", e);
        }
    }

    // Text encoding
    let start_time = Instant::now();
    match model.encode_text("Test text for encoding") {
        Ok(text_features) => {
            let text_time = start_time.elapsed();
            println!("✅ Text encoding: {:?} (shape: {:?})", 
                    text_time, text_features.shape());
        }
        Err(e) => {
            println!("❌ Text encoding failed: {}", e);
        }
    }

    // Performance benchmark
    println!("\n⚡ Performance Benchmark:");
    let num_iterations = 5;
    let mut total_time = std::time::Duration::ZERO;
    let mut successful_inferences = 0;

    for i in 0..num_iterations {
        let start_time = Instant::now();
        match model.infer(&test_image, "Benchmark test", inference_config.clone()) {
            Ok(_) => {
                total_time += start_time.elapsed();
                successful_inferences += 1;
            }
            Err(e) => {
                println!("❌ Benchmark iteration {} failed: {}", i + 1, e);
            }
        }
    }

    if successful_inferences > 0 {
        let avg_time = total_time / successful_inferences as u32;
        let throughput = successful_inferences as f64 / total_time.as_secs_f64();
        
        println!("📊 Benchmark Results:");
        println!("  Successful inferences: {}/{}", successful_inferences, num_iterations);
        println!("  Average inference time: {:?}", avg_time);
        println!("  Throughput: {:.2} inferences/second", throughput);
        
        if avg_time.as_millis() <= 200 {
            println!("🎯 Average performance meets mobile target!");
        }
    }

    // Error handling test
    println!("\n🚨 Error Handling Tests:");
    
    // Test with invalid image data
    let empty_image = vec![];
    match model.infer(&empty_image, "Test", inference_config.clone()) {
        Ok(_) => println!("⚠️  Expected error but got success with empty image"),
        Err(e) => println!("✅ Empty image correctly rejected: {}", e),
    }

    // Test with invalid text (null bytes)
    let invalid_text = "Invalid\0text";
    match model.infer(&test_image, invalid_text, inference_config.clone()) {
        Ok(_) => println!("⚠️  Expected error but got success with invalid text"),
        Err(e) => println!("✅ Invalid text correctly rejected: {}", e),
    }

    println!("\n🏁 Example completed successfully!");
    println!("Visit https://github.com/danieleschmidt/Tiny-VLM-Rust-WASM for more examples");

    Ok(())
}

/// Create a simple test image (RGB format)
fn create_test_image() -> Vec<u8> {
    // Create 224x224x3 RGB test image data (expected by the model)
    let width = 224;
    let height = 224;
    let channels = 3;
    
    let mut rgb_data = Vec::with_capacity(width * height * channels);
    
    // Create a simple pattern: gradient with some noise
    for y in 0..height {
        for x in 0..width {
            // Create a gradient pattern
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8; 
            let b = (((x + y) * 255) / (width + height)) as u8;
            
            rgb_data.push(r);
            rgb_data.push(g);
            rgb_data.push(b);
        }
    }
    
    rgb_data
}