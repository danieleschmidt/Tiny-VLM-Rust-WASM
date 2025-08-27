//! Generation 1: Basic Working VLM Implementation
//! 
//! This example demonstrates a simple but functional vision-language model
//! that can process images and text to generate responses.

use tiny_vlm::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize logging
    if let Ok(_) = env_logger::try_init() {
        println!("✓ Logging initialized");
    }
    
    println!("🚀 Generation 1: Basic VLM Demo");
    println!("================================");

    // Create configuration
    let config = SimpleVLMConfig {
        vision_dim: 512,
        text_dim: 512,
        max_length: 50,
    };
    
    // Initialize model
    let start = Instant::now();
    let vlm = SimpleVLM::new(config)?;
    println!("✓ Model initialized in {:?}", start.elapsed());
    
    // Create sample image data (simulating a 224x224 RGB image)
    let image_data = create_sample_image_data();
    println!("✓ Sample image created ({} bytes)", image_data.len());
    
    // Test different prompts
    let test_cases = vec![
        "What objects are in this image?",
        "Describe the scene",
        "What colors do you see?",
        "Is this indoors or outdoors?",
    ];
    
    println!("\n🔍 Running Inference Tests:");
    println!("--------------------------");
    
    for (i, prompt) in test_cases.iter().enumerate() {
        let start = Instant::now();
        match vlm.infer(&image_data, prompt) {
            Ok(response) => {
                let elapsed = start.elapsed();
                println!("Test {}: ✓ ({:?})", i + 1, elapsed);
                println!("  Prompt: {}", prompt);
                println!("  Response: {}", response);
                println!();
            }
            Err(e) => {
                println!("Test {}: ❌ Error: {}", i + 1, e);
            }
        }
    }
    
    // Test error handling
    println!("🧪 Testing Error Handling:");
    println!("--------------------------");
    
    // Empty image test
    match vlm.infer(&[], "test") {
        Err(e) => println!("✓ Empty image validation: {}", e),
        Ok(_) => println!("❌ Empty image should have failed"),
    }
    
    // Empty text test
    match vlm.infer(&image_data, "") {
        Err(e) => println!("✓ Empty text validation: {}", e),
        Ok(_) => println!("❌ Empty text should have failed"),
    }
    
    // Too long text test
    let long_text = "A".repeat(1000);
    match vlm.infer(&image_data, &long_text) {
        Err(e) => println!("✓ Long text validation: {}", e),
        Ok(_) => println!("❌ Long text should have failed"),
    }
    
    // Performance metrics
    println!("\n📊 Performance Metrics:");
    println!("----------------------");
    let metrics = vlm.performance_metrics();
    println!("Inference count: {}", metrics.inference_count);
    println!("Average latency: {:.2}ms", metrics.avg_latency_ms);
    println!("Memory usage: {:.2}MB", metrics.memory_usage_mb);
    
    println!("\n✅ Generation 1 Demo Complete!");
    Ok(())
}

/// Create sample image data (RGB format)
fn create_sample_image_data() -> Vec<u8> {
    let width = 224;
    let height = 224;
    let channels = 3;
    let mut data = Vec::with_capacity(width * height * channels);
    
    // Create a simple gradient pattern
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = ((x + y) as f32 / (width + height) as f32 * 255.0) as u8;
            
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    
    data
}