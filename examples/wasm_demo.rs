//! WebAssembly demonstration example
//! 
//! This example shows how to use Tiny VLM in a WASM environment
//! with optimized memory usage and browser-compatible APIs.

#[cfg(feature = "wasm")]
use tiny_vlm::{FastVLM, InferenceConfig, ModelConfig};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use web_sys::{console, ImageData};

// Define WASM exports
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    console::log_1(&"🚀 Tiny VLM WASM module loaded successfully".into());
}

/// WASM-compatible VLM model wrapper
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct TinyVlmWasm {
    model: FastVLM,
    config: InferenceConfig,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl TinyVlmWasm {
    /// Create a new WASM VLM instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<TinyVlmWasm, JsValue> {
        console::log_1(&"🏗️  Initializing Tiny VLM WASM".into());
        
        let model_config = ModelConfig::default();
        let model = FastVLM::new(model_config)
            .map_err(|e| JsValue::from_str(&format!("Model initialization failed: {}", e)))?;

        let config = InferenceConfig {
            max_length: 50,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            deterministic: false,
            memory_limit_mb: 50, // Reduced for WASM
        };

        console::log_1(&"✅ Tiny VLM WASM initialized".into());
        
        Ok(TinyVlmWasm { model, config })
    }

    /// Process image data from canvas and generate response
    #[wasm_bindgen]
    pub fn infer_from_image_data(&mut self, image_data: ImageData, prompt: &str) -> Result<String, JsValue> {
        let start_time = js_sys::Date::now();
        
        console::log_1(&format!("🖼️  Processing image: {}x{}", image_data.width(), image_data.height()).into());
        console::log_1(&format!("💬 Prompt: '{}'", prompt).into());

        // Convert ImageData to RGB bytes
        let rgba_data = image_data.data();
        let rgb_data = rgba_to_rgb(&rgba_data);

        // Resize to model input size if necessary
        let target_size = 224;
        let resized_data = if image_data.width() as usize != target_size || image_data.height() as usize != target_size {
            resize_image(&rgb_data, image_data.width() as usize, image_data.height() as usize, target_size, target_size)
        } else {
            rgb_data
        };

        // Run inference
        let result = self.model.infer(&resized_data, prompt, self.config.clone())
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))?;

        let inference_time = js_sys::Date::now() - start_time;
        console::log_1(&format!("⏱️  Inference completed in {:.2}ms", inference_time).into());

        // Log memory usage
        let memory_stats = self.model.memory_stats();
        let memory_mb = memory_stats.allocated_memory as f64 / (1024.0 * 1024.0);
        console::log_1(&format!("💾 Memory usage: {:.2} MB", memory_mb).into());

        Ok(result)
    }

    /// Process base64 encoded image
    #[wasm_bindgen]
    pub fn infer_from_base64(&mut self, base64_data: &str, prompt: &str) -> Result<String, JsValue> {
        console::log_1(&"📸 Processing base64 image".into());
        
        // Decode base64 (simplified - in real implementation would use proper base64 decoder)
        let rgb_data = decode_base64_to_rgb(base64_data)
            .map_err(|e| JsValue::from_str(&format!("Base64 decoding failed: {}", e)))?;

        self.model.infer(&rgb_data, prompt, self.config.clone())
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))
    }

    /// Get model memory statistics
    #[wasm_bindgen]
    pub fn memory_usage(&self) -> f64 {
        let memory_stats = self.model.memory_stats();
        memory_stats.allocated_memory as f64 / (1024.0 * 1024.0)
    }

    /// Clean up model memory
    #[wasm_bindgen]
    pub fn compact_memory(&mut self) {
        self.model.compact_memory();
        console::log_1(&"🧹 Memory cleaned up".into());
    }

    /// Update inference configuration
    #[wasm_bindgen]
    pub fn set_temperature(&mut self, temperature: f32) {
        self.config.temperature = temperature.max(0.1).min(2.0);
        console::log_1(&format!("🌡️  Temperature set to {}", self.config.temperature).into());
    }

    #[wasm_bindgen]
    pub fn set_max_length(&mut self, max_length: usize) {
        self.config.max_length = max_length.min(200);
        console::log_1(&format!("📏 Max length set to {}", self.config.max_length).into());
    }
}

#[cfg(feature = "wasm")]
/// Convert RGBA data to RGB
fn rgba_to_rgb(rgba_data: &[u8]) -> Vec<u8> {
    let mut rgb_data = Vec::with_capacity(rgba_data.len() * 3 / 4);
    
    for chunk in rgba_data.chunks_exact(4) {
        rgb_data.push(chunk[0]); // R
        rgb_data.push(chunk[1]); // G
        rgb_data.push(chunk[2]); // B
        // Skip alpha channel
    }
    
    rgb_data
}

#[cfg(feature = "wasm")]
/// Simple image resize using nearest neighbor
fn resize_image(
    input: &[u8],
    input_width: usize,
    input_height: usize,
    target_width: usize,
    target_height: usize,
) -> Vec<u8> {
    let mut output = vec![0u8; target_width * target_height * 3];

    for y in 0..target_height {
        for x in 0..target_width {
            let src_x = (x * input_width) / target_width;
            let src_y = (y * input_height) / target_height;

            for c in 0..3 {
                let src_idx = (src_y * input_width + src_x) * 3 + c;
                let dst_idx = (y * target_width + x) * 3 + c;
                
                if src_idx < input.len() {
                    output[dst_idx] = input[src_idx];
                }
            }
        }
    }

    output
}

#[cfg(feature = "wasm")]
/// Simplified base64 to RGB decoder (placeholder implementation)
fn decode_base64_to_rgb(base64_data: &str) -> Result<Vec<u8>, String> {
    // In a real implementation, this would properly decode base64
    // For demo purposes, create a sample image
    if base64_data.is_empty() {
        return Err("Empty base64 data".to_string());
    }

    // Return a sample 224x224 RGB pattern
    let mut rgb_data = Vec::with_capacity(224 * 224 * 3);
    for y in 0..224 {
        for x in 0..224 {
            let r = ((x + y) % 256) as u8;
            let g = (x % 256) as u8;
            let b = (y % 256) as u8;
            rgb_data.push(r);
            rgb_data.push(g);
            rgb_data.push(b);
        }
    }
    
    Ok(rgb_data)
}

// Provide a stub for non-WASM builds
#[cfg(not(feature = "wasm"))]
fn main() {
    println!("🚨 This example requires the 'wasm' feature to be enabled");
    println!("💡 Build with: cargo build --features wasm --target wasm32-unknown-unknown");
    println!("📦 Or use: wasm-pack build --features wasm");
}