//! Basic VLM Demo - Generation 1: Make It Work
//! 
//! Simple demonstration of core VLM functionality with minimal dependencies.
//! This example focuses on getting basic functionality working.

use tiny_vlm::prelude::*;

fn main() -> Result<()> {
    println!("🚀 Tiny-VLM Basic Demo - Generation 1");
    
    // Initialize basic VLM system
    let config = ModelConfig::default();
    let mut vlm = FastVLM::new(config)?;
    
    // Load minimal model for demo
    println!("📦 Initializing minimal model...");
    vlm.init_minimal()?;
    
    // Process simple text input
    println!("💭 Processing text input...");
    let text_input = "What is machine learning?";
    let text_result = vlm.process_text_simple(text_input)?;
    
    println!("✅ Text processing result: {:.2}ms", text_result.latency_ms);
    println!("📊 Generated tokens: {}", text_result.output_tokens.len());
    
    // Process simple image (simulated for now)
    println!("🖼️ Processing simulated image...");
    let image_result = vlm.process_image_simple(224, 224)?;
    
    println!("✅ Image processing result: {:.2}ms", image_result.latency_ms);
    println!("📊 Feature dimensions: {:?}", image_result.feature_dims);
    
    // Combined VLM inference (basic)
    println!("🧠 Running basic VLM inference...");
    let vlm_result = vlm.infer_simple(text_input, Some((224, 224)))?;
    
    println!("✅ VLM inference completed!");
    println!("📈 Total latency: {:.2}ms", vlm_result.total_latency_ms);
    println!("🎯 Output preview: {:?}", &vlm_result.text_output[..vlm_result.text_output.len().min(50)]);
    
    // Basic performance metrics
    let metrics = vlm.get_performance_metrics();
    println!("\n📊 Performance Summary:");
    println!("   • Total inferences: {}", metrics.total_inferences);
    println!("   • Average latency: {:.2}ms", metrics.avg_latency_ms);
    println!("   • Memory usage: {:.1}MB", metrics.memory_usage_mb);
    
    println!("\n🎉 Basic VLM demo completed successfully!");
    
    Ok(())
}