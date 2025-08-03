//! WebAssembly bindings for browser integration

use crate::{FastVLM, InferenceConfig, ModelConfig, Result, TinyVlmError};
use wasm_bindgen::prelude::*;
use js_sys::{Array, Object, Reflect, Uint8Array};
use web_sys::{console, ImageData};

// Enable console.log! and other web APIs
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// WASM wrapper for the FastVLM model
#[wasm_bindgen]
pub struct WasmFastVLM {
    model: FastVLM,
}

#[wasm_bindgen]
impl WasmFastVLM {
    /// Create a new FastVLM instance with default configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmFastVLM, JsValue> {
        console_log!("Initializing FastVLM with default configuration");
        
        let config = ModelConfig::default();
        let model = FastVLM::new(config)
            .map_err(|e| JsValue::from_str(&format!("Failed to create model: {}", e)))?;
        
        Ok(WasmFastVLM { model })
    }

    /// Create a new FastVLM instance with custom configuration
    #[wasm_bindgen(js_name = newWithConfig)]
    pub fn new_with_config(config_js: &JsValue) -> Result<WasmFastVLM, JsValue> {
        let config = parse_model_config(config_js)?;
        let model = FastVLM::new(config)
            .map_err(|e| JsValue::from_str(&format!("Failed to create model: {}", e)))?;
        
        Ok(WasmFastVLM { model })
    }

    /// Load a model from a URL or file path
    #[wasm_bindgen(js_name = loadFromUrl)]
    pub async fn load_from_url(url: &str) -> Result<WasmFastVLM, JsValue> {
        console_log!("Loading model from URL: {}", url);
        
        // In a real implementation, this would fetch and parse model weights
        // For now, create a default model
        let model = FastVLM::load_from_file(url)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;
        
        Ok(WasmFastVLM { model })
    }

    /// Perform inference on an image with a text prompt
    #[wasm_bindgen]
    pub fn infer(&mut self, image_data: &Uint8Array, prompt: &str) -> Result<String, JsValue> {
        let image_bytes = image_data.to_vec();
        let config = InferenceConfig::default();
        
        self.model
            .infer(&image_bytes, prompt, config)
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))
    }

    /// Perform inference with custom configuration
    #[wasm_bindgen(js_name = inferWithConfig)]
    pub fn infer_with_config(
        &mut self,
        image_data: &Uint8Array,
        prompt: &str,
        config_js: &JsValue,
    ) -> Result<String, JsValue> {
        let image_bytes = image_data.to_vec();
        let config = parse_inference_config(config_js)?;
        
        self.model
            .infer(&image_bytes, prompt, config)
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))
    }

    /// Process image from HTML Canvas ImageData
    #[wasm_bindgen(js_name = inferFromImageData)]
    pub fn infer_from_image_data(
        &mut self,
        image_data: &ImageData,
        prompt: &str,
    ) -> Result<String, JsValue> {
        // Extract RGB data from ImageData (RGBA -> RGB conversion)
        let rgba_data = image_data.data();
        let width = image_data.width() as usize;
        let height = image_data.height() as usize;
        
        let mut rgb_data = Vec::with_capacity(width * height * 3);
        for i in (0..rgba_data.length()).step_by(4) {
            rgb_data.push(rgba_data.get_index(i) as u8);     // R
            rgb_data.push(rgba_data.get_index(i + 1) as u8); // G
            rgb_data.push(rgba_data.get_index(i + 2) as u8); // B
            // Skip alpha channel
        }
        
        let config = InferenceConfig::default();
        self.model
            .infer(&rgb_data, prompt, config)
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))
    }

    /// Encode image to feature vector
    #[wasm_bindgen(js_name = encodeImage)]
    pub fn encode_image(&mut self, image_data: &Uint8Array) -> Result<Array, JsValue> {
        let image_bytes = image_data.to_vec();
        
        let features = self.model
            .encode_image(&image_bytes)
            .map_err(|e| JsValue::from_str(&format!("Image encoding failed: {}", e)))?;
        
        // Convert tensor to JavaScript array
        tensor_to_js_array(&features)
    }

    /// Encode text to feature vector
    #[wasm_bindgen(js_name = encodeText)]
    pub fn encode_text(&mut self, text: &str) -> Result<Array, JsValue> {
        let features = self.model
            .encode_text(text)
            .map_err(|e| JsValue::from_str(&format!("Text encoding failed: {}", e)))?;
        
        // Convert tensor to JavaScript array
        tensor_to_js_array(&features)
    }

    /// Get memory usage statistics
    #[wasm_bindgen(js_name = getMemoryStats)]
    pub fn get_memory_stats(&self) -> JsValue {
        let stats = self.model.memory_stats();
        
        let obj = Object::new();
        Reflect::set(&obj, &"totalMemory".into(), &JsValue::from(stats.total_memory)).unwrap();
        Reflect::set(&obj, &"allocatedMemory".into(), &JsValue::from(stats.allocated_memory)).unwrap();
        Reflect::set(&obj, &"availableMemory".into(), &JsValue::from(stats.available_memory)).unwrap();
        Reflect::set(&obj, &"maxMemory".into(), &JsValue::from(stats.max_memory)).unwrap();
        Reflect::set(&obj, &"fragmentation".into(), &JsValue::from(stats.fragmentation)).unwrap();
        
        obj.into()
    }

    /// Compact memory to reduce fragmentation
    #[wasm_bindgen(js_name = compactMemory)]
    pub fn compact_memory(&mut self) {
        self.model.compact_memory();
        console_log!("Memory compacted");
    }

    /// Check if SIMD is available
    #[wasm_bindgen(js_name = isSimdAvailable)]
    pub fn is_simd_available() -> bool {
        // Check for WebAssembly SIMD support
        // This is a simplified check - real implementation would be more robust
        true // Assume SIMD is available for now
    }

    /// Get model configuration
    #[wasm_bindgen(js_name = getConfig)]
    pub fn get_config(&self) -> JsValue {
        model_config_to_js(self.model.config())
    }

    /// Process image from File or Blob
    #[wasm_bindgen(js_name = inferFromFile)]
    pub async fn infer_from_file(
        &mut self,
        file: &web_sys::File,
        prompt: &str,
    ) -> Result<String, JsValue> {
        // Convert File to ArrayBuffer and then to Uint8Array
        let array_buffer = wasm_bindgen_futures::JsFuture::from(file.array_buffer())
            .await?;
        let uint8_array = Uint8Array::new(&array_buffer);
        
        self.infer(&uint8_array, prompt)
    }
}

/// Utility functions for JavaScript interop

/// Parse ModelConfig from JavaScript object
fn parse_model_config(config_js: &JsValue) -> Result<ModelConfig, JsValue> {
    let mut config = ModelConfig::default();
    
    if let Ok(obj) = config_js.dyn_ref::<Object>() {
        // Parse vision configuration
        if let Ok(vision_dim) = Reflect::get(obj, &"visionDim".into()) {
            if let Some(dim) = vision_dim.as_f64() {
                config.vision_dim = dim as usize;
            }
        }
        
        if let Ok(text_dim) = Reflect::get(obj, &"textDim".into()) {
            if let Some(dim) = text_dim.as_f64() {
                config.text_dim = dim as usize;
            }
        }
        
        if let Ok(hidden_dim) = Reflect::get(obj, &"hiddenDim".into()) {
            if let Some(dim) = hidden_dim.as_f64() {
                config.hidden_dim = dim as usize;
            }
        }
        
        if let Ok(max_gen_length) = Reflect::get(obj, &"maxGenLength".into()) {
            if let Some(length) = max_gen_length.as_f64() {
                config.max_gen_length = length as usize;
            }
        }
        
        if let Ok(temperature) = Reflect::get(obj, &"temperature".into()) {
            if let Some(temp) = temperature.as_f64() {
                config.temperature = temp as f32;
            }
        }
    }
    
    Ok(config)
}

/// Parse InferenceConfig from JavaScript object
fn parse_inference_config(config_js: &JsValue) -> Result<InferenceConfig, JsValue> {
    let mut config = InferenceConfig::default();
    
    if let Ok(obj) = config_js.dyn_ref::<Object>() {
        if let Ok(max_length) = Reflect::get(obj, &"maxLength".into()) {
            if let Some(length) = max_length.as_f64() {
                config.max_length = length as usize;
            }
        }
        
        if let Ok(temperature) = Reflect::get(obj, &"temperature".into()) {
            if let Some(temp) = temperature.as_f64() {
                config.temperature = temp as f32;
            }
        }
        
        if let Ok(top_p) = Reflect::get(obj, &"topP".into()) {
            if let Some(p) = top_p.as_f64() {
                config.top_p = p as f32;
            }
        }
        
        if let Ok(top_k) = Reflect::get(obj, &"topK".into()) {
            if let Some(k) = top_k.as_f64() {
                config.top_k = k as usize;
            }
        }
        
        if let Ok(deterministic) = Reflect::get(obj, &"deterministic".into()) {
            config.deterministic = deterministic.as_bool().unwrap_or(false);
        }
    }
    
    Ok(config)
}

/// Convert ModelConfig to JavaScript object
fn model_config_to_js(config: &ModelConfig) -> JsValue {
    let obj = Object::new();
    
    Reflect::set(&obj, &"visionDim".into(), &JsValue::from(config.vision_dim)).unwrap();
    Reflect::set(&obj, &"textDim".into(), &JsValue::from(config.text_dim)).unwrap();
    Reflect::set(&obj, &"hiddenDim".into(), &JsValue::from(config.hidden_dim)).unwrap();
    Reflect::set(&obj, &"numHeads".into(), &JsValue::from(config.num_heads)).unwrap();
    Reflect::set(&obj, &"maxGenLength".into(), &JsValue::from(config.max_gen_length)).unwrap();
    Reflect::set(&obj, &"temperature".into(), &JsValue::from(config.temperature)).unwrap();
    
    obj.into()
}

/// Convert tensor to JavaScript array
fn tensor_to_js_array(tensor: &crate::memory::Tensor<f32>) -> Result<Array, JsValue> {
    let data = tensor.data();
    let array = Array::new();
    
    for &value in data {
        array.push(&JsValue::from(value));
    }
    
    Ok(array)
}

/// JavaScript-callable utility functions
#[wasm_bindgen]
pub struct WasmUtils;

#[wasm_bindgen]
impl WasmUtils {
    /// Check WebAssembly SIMD support
    #[wasm_bindgen(js_name = checkSimdSupport)]
    pub fn check_simd_support() -> bool {
        // This would check for actual SIMD support in the browser
        // For now, return true as a placeholder
        true
    }

    /// Get library version
    #[wasm_bindgen(js_name = getVersion)]
    pub fn get_version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Initialize panic hook for better error reporting
    #[wasm_bindgen(js_name = initPanicHook)]
    pub fn init_panic_hook() {
        console_error_panic_hook::set_once();
    }

    /// Validate image dimensions
    #[wasm_bindgen(js_name = validateImageDimensions)]
    pub fn validate_image_dimensions(width: u32, height: u32) -> bool {
        // Check if dimensions are reasonable for VLM processing
        width > 0 && height > 0 && width <= 2048 && height <= 2048
    }

    /// Resize image data (simplified nearest neighbor)
    #[wasm_bindgen(js_name = resizeImageData)]
    pub fn resize_image_data(
        input_data: &Uint8Array,
        input_width: u32,
        input_height: u32,
        target_width: u32,
        target_height: u32,
    ) -> Result<Uint8Array, JsValue> {
        let input = input_data.to_vec();
        let input_len = input_width as usize * input_height as usize * 3;
        
        if input.len() != input_len {
            return Err(JsValue::from_str("Input data length mismatch"));
        }
        
        let mut output = vec![0u8; target_width as usize * target_height as usize * 3];
        
        for y in 0..target_height {
            for x in 0..target_width {
                let src_x = (x * input_width) / target_width;
                let src_y = (y * input_height) / target_height;
                
                for c in 0..3 {
                    let src_idx = ((src_y * input_width + src_x) * 3 + c) as usize;
                    let dst_idx = ((y * target_width + x) * 3 + c) as usize;
                    
                    if src_idx < input.len() && dst_idx < output.len() {
                        output[dst_idx] = input[src_idx];
                    }
                }
            }
        }
        
        Ok(Uint8Array::from(&output[..]))
    }
}

/// Performance monitoring utilities
#[wasm_bindgen]
pub struct WasmPerformance {
    start_time: f64,
}

#[wasm_bindgen]
impl WasmPerformance {
    /// Create a new performance monitor
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmPerformance {
        WasmPerformance {
            start_time: js_sys::Date::now(),
        }
    }

    /// Start timing
    #[wasm_bindgen]
    pub fn start(&mut self) {
        self.start_time = js_sys::Date::now();
    }

    /// Get elapsed time in milliseconds
    #[wasm_bindgen]
    pub fn elapsed(&self) -> f64 {
        js_sys::Date::now() - self.start_time
    }

    /// Log performance measurement
    #[wasm_bindgen]
    pub fn log(&self, operation: &str) {
        let elapsed = self.elapsed();
        console_log!("{}: {:.2}ms", operation, elapsed);
    }
}

// Export main types for TypeScript
#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = r#"
export interface ModelConfig {
    visionDim?: number;
    textDim?: number;
    hiddenDim?: number;
    numHeads?: number;
    maxGenLength?: number;
    temperature?: number;
}

export interface InferenceConfig {
    maxLength?: number;
    temperature?: number;
    topP?: number;
    topK?: number;
    deterministic?: boolean;
}

export interface MemoryStats {
    totalMemory: number;
    allocatedMemory: number;
    availableMemory: number;
    maxMemory: number;
    fragmentation: number;
}
"#;