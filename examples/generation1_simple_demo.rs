//! Generation 1: Simple VLM Demo
//! 
//! Basic functionality demonstration for the Tiny-VLM model.
//! Shows core inference capabilities with minimal setup.

use tiny_vlm::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Generation 1: Tiny-VLM Simple Demo");
    println!("=====================================");
    
    // Initialize model with default configuration
    let config = ModelConfig::default();
    let mut model = FastVLM::new(config)?;
    
    // Initialize minimal functionality
    model.init_minimal()?;
    
    println!("\n📋 Model Configuration:");
    println!("   Vision dimension: {}", model.config().vision_dim);
    println!("   Text dimension: {}", model.config().text_dim);
    println!("   Hidden dimension: {}", model.config().hidden_dim);
    
    // Test simple text processing
    println!("\n🔤 Testing Text Processing...");
    let text_prompt = "What is in this image?";
    let text_result = model.process_text_simple(text_prompt)?;
    println!("   Input: '{}'", text_prompt);
    println!("   Tokens: {} tokens", text_result.output_tokens.len());
    println!("   Latency: {:.1}ms", text_result.latency_ms);
    
    // Test simple image processing
    println!("\n🖼️  Testing Image Processing...");
    let image_width = 224;
    let image_height = 224;
    let image_result = model.process_image_simple(image_width, image_height)?;
    println!("   Input: {}x{} pixels", image_width, image_height);
    println!("   Features: {:?}", image_result.feature_dims);
    println!("   Latency: {:.1}ms", image_result.latency_ms);
    
    // Test simple VLM inference
    println!("\n🧠 Testing VLM Inference...");
    let vlm_result = model.infer_simple(
        "Describe what you see in this image",
        Some((224, 224))
    )?;
    println!("   Response: '{}'", vlm_result.text_output);
    println!("   Total latency: {:.1}ms", vlm_result.total_latency_ms);
    println!("   Text latency: {:.1}ms", vlm_result.text_latency_ms);
    println!("   Image latency: {:.1}ms", vlm_result.image_latency_ms);
    
    // Get performance metrics
    println!("\n📊 Performance Metrics:");
    let metrics = model.get_performance_metrics();
    println!("   Total inferences: {}", metrics.total_inferences);
    println!("   Average latency: {:.1}ms", metrics.avg_latency_ms);
    println!("   Memory usage: {:.1}MB", metrics.memory_usage_mb);
    
    // Memory statistics
    println!("\n💾 Memory Statistics:");
    let memory_stats = model.memory_stats();
    println!("   Total memory: {:.1}MB", memory_stats.total_memory as f64 / 1_000_000.0);
    println!("   Allocated memory: {:.1}MB", memory_stats.allocated_memory as f64 / 1_000_000.0);
    println!("   Available memory: {:.1}MB", memory_stats.available_memory as f64 / 1_000_000.0);
    
    println!("\n✅ Generation 1 Demo Complete!");
    println!("   🔧 Basic functionality: Working");
    println!("   📱 Mobile-ready: Core components initialized");
    println!("   ⚡ Performance: Sub-200ms target architecture in place");
    
    Ok(())
}