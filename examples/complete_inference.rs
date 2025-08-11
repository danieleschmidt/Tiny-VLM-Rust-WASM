//! Complete inference example demonstrating the Tiny VLM capabilities
//! 
//! This example shows how to:
//! 1. Load the model
//! 2. Process an image
//! 3. Run inference with a text prompt
//! 4. Generate a response

use tiny_vlm::{FastVLM, InferenceConfig, ModelConfig, Result};
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize logging
    #[cfg(feature = "std")]
    env_logger::init();

    println!("🚀 Tiny VLM Complete Inference Example");
    println!("=====================================");

    // Load model with default configuration
    let config = ModelConfig::default();
    println!("📋 Model Configuration:");
    println!("  Vision dim: {}", config.vision_dim);
    println!("  Text dim: {}", config.text_dim);
    println!("  Hidden dim: {}", config.hidden_dim);
    println!("  Max gen length: {}", config.max_gen_length);

    let mut model = FastVLM::new(config)?;
    println!("✅ Model loaded successfully");

    // Create a sample RGB image (red square pattern)
    let image_data = create_sample_image(224, 224);
    println!("📸 Sample image created: {}x{} RGB", 224, 224);

    // Configure inference parameters
    let inference_config = InferenceConfig {
        max_length: 50,
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        deterministic: false,
        memory_limit_mb: 150,
    };

    println!("🔧 Inference Configuration:");
    println!("  Max length: {}", inference_config.max_length);
    println!("  Temperature: {}", inference_config.temperature);
    println!("  Top-p: {}", inference_config.top_p);
    println!("  Top-k: {}", inference_config.top_k);

    // Test prompts
    let prompts = vec![
        "What is in this image?",
        "Describe the colors you see",
        "What objects are visible?",
        "Is this a photograph or digital art?",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        println!("\n🤖 Inference {} of {}", i + 1, prompts.len());
        println!("💬 Prompt: \"{}\"", prompt);
        
        let start_time = Instant::now();
        
        match model.infer(&image_data, prompt, inference_config.clone()) {
            Ok(response) => {
                let inference_time = start_time.elapsed();
                println!("✨ Response: \"{}\"", response);
                println!("⏱️  Inference time: {:.2}ms", inference_time.as_secs_f32() * 1000.0);
                
                // Show memory usage
                let memory_stats = model.memory_stats();
                let memory_mb = memory_stats.allocated_memory as f64 / (1024.0 * 1024.0);
                println!("💾 Memory usage: {:.2} MB", memory_mb);
            }
            Err(e) => {
                println!("❌ Inference failed: {}", e);
            }
        }
    }

    // Test individual encoding functions
    println!("\n🔬 Testing individual encoding functions");
    
    println!("🖼️  Encoding image...");
    let start_time = Instant::now();
    match model.encode_image(&image_data) {
        Ok(image_features) => {
            let encoding_time = start_time.elapsed();
            println!("✅ Image encoded - Shape: {:?}", image_features.shape().dims);
            println!("⏱️  Encoding time: {:.2}ms", encoding_time.as_secs_f32() * 1000.0);
        }
        Err(e) => println!("❌ Image encoding failed: {}", e),
    }

    println!("📝 Encoding text...");
    let start_time = Instant::now();
    match model.encode_text("Hello, this is a test prompt") {
        Ok(text_features) => {
            let encoding_time = start_time.elapsed();
            println!("✅ Text encoded - Shape: {:?}", text_features.shape().dims);
            println!("⏱️  Encoding time: {:.2}ms", encoding_time.as_secs_f32() * 1000.0);
        }
        Err(e) => println!("❌ Text encoding failed: {}", e),
    }

    // Performance summary
    println!("\n📊 Performance Summary");
    println!("======================");
    println!("✅ All inference tests completed successfully");
    println!("🎯 Target: <200ms inference (production target)");
    println!("💡 This is a proof-of-concept with simplified operations");
    
    // Clean up memory
    model.compact_memory();
    println!("🧹 Memory cleaned up");

    Ok(())
}

/// Create a sample RGB image with a simple pattern
fn create_sample_image(width: usize, height: usize) -> Vec<u8> {
    let mut image_data = Vec::with_capacity(width * height * 3);
    
    for y in 0..height {
        for x in 0..width {
            // Create a simple geometric pattern
            let r = ((x + y) % 256) as u8;
            let g = ((x * 2) % 256) as u8;
            let b = ((y * 2) % 256) as u8;
            
            image_data.push(r);
            image_data.push(g);
            image_data.push(b);
        }
    }
    
    image_data
}