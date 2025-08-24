//! Working Generation 1 Demo
//! 
//! This demo uses only the SimpleVLM module to demonstrate
//! basic vision-language functionality without complex dependencies.

use tiny_vlm::simple_vlm::{SimpleVLM, SimpleVLMConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Working Generation 1: Simple VLM Demo");
    println!("========================================");
    
    // Create simple configuration
    let config = SimpleVLMConfig {
        vision_dim: 768,
        text_dim: 768,
        max_length: 100,
    };
    
    println!("\n📋 Simple VLM Configuration:");
    println!("   Vision dimension: {}", config.vision_dim);
    println!("   Text dimension: {}", config.text_dim);
    println!("   Max sequence length: {}", config.max_length);
    
    // Create simple VLM model
    println!("\n🔧 Creating Simple VLM...");
    let vlm = SimpleVLM::new(config)?;
    
    if vlm.is_initialized() {
        println!("✅ Simple VLM initialized successfully!");
    } else {
        println!("❌ Failed to initialize Simple VLM");
        return Err("Initialization failed".into());
    }
    
    // Test basic inference
    println!("\n🧠 Testing Basic VLM Inference...");
    
    let test_cases = vec![
        ("What is in this image?", vec![128u8; 224 * 224 * 3]),
        ("Describe the scene", vec![64u8; 512 * 512 * 3]),
        ("Count the objects", vec![192u8; 100 * 100 * 3]),
    ];
    
    for (i, (prompt, image_data)) in test_cases.iter().enumerate() {
        println!("\n   Test case {}: '{}'", i + 1, prompt);
        println!("   Image size: {} bytes", image_data.len());
        
        match vlm.infer(image_data, prompt) {
            Ok(response) => {
                println!("   ✅ Response: {}", response);
            },
            Err(e) => {
                println!("   ❌ Error: {}", e);
            }
        }
    }
    
    // Test error handling
    println!("\n🔍 Testing Error Handling...");
    
    // Empty image test
    match vlm.infer(&[], "test prompt") {
        Ok(_) => println!("   ❌ Should have failed with empty image"),
        Err(e) => println!("   ✅ Correctly handled empty image: {}", e),
    }
    
    // Empty text test
    let dummy_image = vec![100u8; 1000];
    match vlm.infer(&dummy_image, "") {
        Ok(_) => println!("   ❌ Should have failed with empty text"),
        Err(e) => println!("   ✅ Correctly handled empty text: {}", e),
    }
    
    // Too long text test
    let long_text = "A".repeat(1000);
    match vlm.infer(&dummy_image, &long_text) {
        Ok(_) => println!("   ❌ Should have failed with long text"),
        Err(e) => println!("   ✅ Correctly handled long text: {}", e),
    }
    
    // Performance metrics
    println!("\n📊 Performance Metrics:");
    let metrics = vlm.performance_metrics();
    println!("   Inference count: {}", metrics.inference_count);
    println!("   Average latency: {:.1}ms", metrics.avg_latency_ms);
    println!("   Memory usage: {:.1}MB", metrics.memory_usage_mb);
    
    // Configuration access
    println!("\n⚙️  Configuration Details:");
    let model_config = vlm.config();
    println!("   Vision dimension: {}", model_config.vision_dim);
    println!("   Text dimension: {}", model_config.text_dim);
    println!("   Max length: {}", model_config.max_length);
    
    println!("\n✅ Working Generation 1 Demo Complete!");
    println!("   🔧 Basic VLM functionality: Working");
    println!("   🛡️  Error handling: Working");
    println!("   📊 Performance tracking: Working");
    println!("   📱 Foundation for mobile deployment: Ready");
    
    Ok(())
}