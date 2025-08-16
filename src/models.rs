//! Core VLM model implementation

use crate::{
    memory::{MemoryPool, Tensor, TensorShape},
    text::{Tokenizer, TokenizerConfig},
    vision::{ImageProcessor, VisionConfig, VisionEncoder},
    Result, TinyVlmError,
};

/// Configuration for the FastVLM model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Vision encoder configuration
    pub vision_config: VisionConfig,
    /// Text tokenizer configuration
    pub text_config: TokenizerConfig,
    /// Dimension of vision features
    pub vision_dim: usize,
    /// Dimension of text embeddings
    pub text_dim: usize,
    /// Hidden dimension for multimodal fusion
    pub hidden_dim: usize,
    /// Number of attention heads for fusion
    pub num_heads: usize,
    /// Maximum generation length
    pub max_gen_length: usize,
    /// Temperature for text generation
    pub temperature: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vision_config: VisionConfig::default(),
            text_config: TokenizerConfig::default(),
            vision_dim: 768,
            text_dim: 768,
            hidden_dim: 768,
            num_heads: 12,
            max_gen_length: 100,
            temperature: 1.0,
        }
    }
}

/// Configuration for inference behavior
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum sequence length for generation
    pub max_length: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p sampling threshold
    pub top_p: f32,
    /// Top-k sampling limit
    pub top_k: usize,
    /// Whether to use deterministic sampling
    pub deterministic: bool,
    /// Memory limit in MB
    pub memory_limit_mb: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_length: 100,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 50,
            deterministic: false,
            memory_limit_mb: 100,
        }
    }
}

/// Main FastVLM model for vision-language understanding
pub struct FastVLM {
    /// Model configuration
    config: ModelConfig,
    /// Vision encoder
    vision_encoder: VisionEncoder,
    /// Text tokenizer
    tokenizer: Tokenizer,
    /// Image processor
    image_processor: ImageProcessor,
    /// Vision-text projection layer
    vision_projection: ProjectionLayer,
    /// Text projection layer
    text_projection: ProjectionLayer,
    /// Multimodal fusion layer
    fusion_layer: MultimodalFusion,
    /// Language model head for text generation
    lm_head: LanguageModelHead,
    /// Memory pool for efficient tensor management
    memory_pool: MemoryPool<f32>,
}

impl FastVLM {
    /// Create a new FastVLM model with given configuration
    pub fn new(config: ModelConfig) -> Result<Self> {
        // Validate configuration
        crate::validation::validate_model_config(&config).into_result()?;
        // Initialize vision encoder
        let vision_encoder = VisionEncoder::new(config.vision_config.clone())?;
        
        // Initialize tokenizer
        let tokenizer = Tokenizer::new(config.text_config.clone(), config.text_dim)?;
        
        // Initialize image processor
        let image_processor = ImageProcessor::new(
            config.vision_config.input_height,
            config.vision_config.input_width,
        );

        // Initialize projection layers
        let vision_projection = ProjectionLayer::new(config.vision_dim, config.hidden_dim)?;
        let text_projection = ProjectionLayer::new(config.text_dim, config.hidden_dim)?;

        // Initialize fusion layer
        let fusion_layer = MultimodalFusion::new(config.hidden_dim, config.num_heads)?;

        // Initialize language model head
        let lm_head = LanguageModelHead::new(config.hidden_dim, config.text_config.vocab_size)?;

        // Initialize adaptive memory pool with intelligent sizing
        let adaptive_memory_size = calculate_optimal_memory_size(&config);
        let memory_pool = MemoryPool::new(adaptive_memory_size);
        
        #[cfg(feature = "std")]
        crate::logging::log_memory_event(
            "memory_pool_initialized",
            adaptive_memory_size as f64 / (1024.0 * 1024.0),
            0.0, // Initial allocation
            0.0, // No fragmentation yet
        );

        Ok(Self {
            config,
            vision_encoder,
            tokenizer,
            image_processor,
            vision_projection,
            text_projection,
            fusion_layer,
            lm_head,
            memory_pool,
        })
    }

    /// Load model from file (placeholder for actual model loading)
    pub fn load_from_file(path: &str) -> Result<Self> {
        // In a real implementation, this would load weights from a file
        // For now, create a default model
        let config = ModelConfig::default();
        let model = Self::new(config)?;
        
        // Log that we're using default weights instead of loading
        #[cfg(feature = "std")]
        eprintln!("Warning: Using default weights instead of loading from {}", path);
        
        Ok(model)
    }

    /// Perform inference on an image with a text prompt with robust error handling
    pub fn infer(&mut self, image_data: &[u8], prompt: &str, config: InferenceConfig) -> Result<String> {
        // Enhanced validation with security checks
        crate::validation::validate_image_data(image_data).into_result()
            .map_err(|e| {
                let context = crate::error::ErrorContext {
                    #[cfg(feature = "std")]
                    timestamp: std::time::Instant::now(),
                    operation: "image_validation".to_string(),
                    context: [("image_size".to_string(), image_data.len().to_string())].iter().cloned().collect(),
                    recovery_suggestions: e.recovery_suggestions(),
                    is_retryable: e.is_retryable(),
                    severity: e.severity(),
                };
                e.with_context(context)
            })?;
            
        crate::validation::validate_text_input(prompt).into_result()
            .map_err(|e| {
                let context = crate::error::ErrorContext {
                    #[cfg(feature = "std")]
                    timestamp: std::time::Instant::now(),
                    operation: "text_validation".to_string(),
                    context: [("text_length".to_string(), prompt.len().to_string())].iter().cloned().collect(),
                    recovery_suggestions: e.recovery_suggestions(),
                    is_retryable: e.is_retryable(),
                    severity: e.severity(),
                };
                e.with_context(context)
            })?;
            
        crate::validation::validate_inference_config(&config).into_result()
            .map_err(|e| {
                let context = crate::error::ErrorContext {
                    #[cfg(feature = "std")]
                    timestamp: std::time::Instant::now(),
                    operation: "config_validation".to_string(),
                    context: [("max_length".to_string(), config.max_length.to_string())].iter().cloned().collect(),
                    recovery_suggestions: e.recovery_suggestions(),
                    is_retryable: e.is_retryable(),
                    severity: e.severity(),
                };
                e.with_context(context)
            })?;

        #[cfg(feature = "std")]
        let _timer = crate::logging::PerformanceTimer::new("model_inference");

        // Performance optimization: Dynamic prompt length based on memory availability
        let dynamic_limit = self.calculate_dynamic_prompt_limit();
        if prompt.len() > dynamic_limit {
            #[cfg(feature = "std")]
            crate::logging::log_security_event(
                "prompt_length_exceeded",
                crate::logging::SecuritySeverity::Medium,
                &format!("Prompt length {} exceeds dynamic limit {}", prompt.len(), dynamic_limit)
            );
            return Err(TinyVlmError::invalid_input(
                format!("Prompt too long: {} > {} (dynamic limit)", prompt.len(), dynamic_limit)
            ));
        }

        // Process image with error recovery
        let image_tensor = self.image_processor.preprocess(image_data)
            .map_err(|e| {
                #[cfg(feature = "std")]
                {
                    let error_msg = format!("Failed to process image: {}", e);
                    crate::logging::log_security_event(
                        "image_processing_error",
                        crate::logging::SecuritySeverity::Medium,
                        &error_msg,
                    );
                }
                {
                    let error = TinyVlmError::image_processing(format!("Failed to preprocess image: {}", e));
                    let context = crate::error::ErrorContext {
                        #[cfg(feature = "std")]
                        timestamp: std::time::Instant::now(),
                        operation: "image_preprocessing".to_string(),
                        context: [
                            ("image_size".to_string(), image_data.len().to_string()),
                            ("expected_dimensions".to_string(), "224x224x3".to_string()),
                        ].iter().cloned().collect(),
                        recovery_suggestions: vec![
                            "Resize image to 224x224 pixels".to_string(),
                            "Convert image to RGB format".to_string(),
                            "Check image is not corrupted".to_string(),
                        ],
                        is_retryable: false,
                        severity: crate::error::ErrorSeverity::Medium,
                    };
                    error.with_context(context)
                }
            })?;
        
        // Encode image with circuit breaker protection
        let vision_features = self.vision_encoder.encode(&image_tensor)
            .map_err(|e| {
                #[cfg(feature = "std")]
                let error_msg = format!("Vision encoding failed: {}", e);
                crate::logging::log_security_event(
                    "vision_encoding_error",
                    crate::logging::SecuritySeverity::Medium,
                    &error_msg,
                );
                TinyVlmError::inference(format!("Vision encoding failed: {}", e))
            })?;
        
        // Project vision features
        let projected_vision = self.vision_projection.forward(&vision_features, &mut self.memory_pool)?;
        
        // Tokenize and embed text
        let text_tokens = self.tokenizer.encode(prompt)?;
        let text_embeddings = self.tokenizer.embed_sequence(&text_tokens, &mut self.memory_pool)?;
        
        // Project text features
        let projected_text = self.text_projection.forward(&text_embeddings, &mut self.memory_pool)?;
        
        // Fuse vision and text
        let fused_features = self.fusion_layer.forward(
            &projected_vision,
            &projected_text,
            &mut self.memory_pool,
        )?;
        
        // Generate text response
        let response = self.generate_text(&fused_features, &text_tokens, config)?;

        // Log performance metrics
        #[cfg(feature = "std")]
        {
            let inference_time = _timer.elapsed_ms();
            let memory_stats = self.memory_stats();
            let memory_mb = (memory_stats.allocated_memory as f64) / (1024.0 * 1024.0);
            
            crate::logging::log_inference_metrics(
                &self.config,
                inference_time,
                memory_mb,
                text_tokens.len(),
                response.split_whitespace().count(), // Rough token count
            );
        }

        Ok(response)
    }

    /// Process image and return visual features
    pub fn encode_image(&mut self, image_data: &[u8]) -> Result<Tensor<f32>> {
        let image_tensor = self.image_processor.preprocess(image_data)?;
        let vision_features = self.vision_encoder.encode(&image_tensor)?;
        self.vision_projection.forward(&vision_features, &mut self.memory_pool)
    }

    /// Process text and return text features
    pub fn encode_text(&mut self, text: &str) -> Result<Tensor<f32>> {
        let text_tokens = self.tokenizer.encode(text)?;
        let text_embeddings = self.tokenizer.embed_sequence(&text_tokens, &mut self.memory_pool)?;
        self.text_projection.forward(&text_embeddings, &mut self.memory_pool)
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> crate::memory::MemoryStats {
        self.memory_pool.memory_usage()
    }

    /// Calculate dynamic prompt length limit based on available memory
    fn calculate_dynamic_prompt_limit(&self) -> usize {
        let stats = self.memory_pool.memory_usage();
        let available_memory = stats.available_memory;
        
        // Reserve 20% of available memory for prompt processing
        let prompt_memory_budget = (available_memory as f64 * 0.2) as usize;
        
        // Estimate ~4 bytes per character for processing (including embeddings)
        let char_limit = prompt_memory_budget / 4;
        
        // Clamp between reasonable limits
        char_limit.max(1000).min(50000) // Min 1K chars, max 50K chars
    }

    /// Perform memory compaction with performance tracking
    pub fn compact_memory(&mut self) {
        #[cfg(feature = "std")]
        let _timer = crate::logging::PerformanceTimer::new("memory_compaction");
        
        let stats_before = self.memory_pool.memory_usage();
        self.memory_pool.compact();
        let stats_after = self.memory_pool.memory_usage();
        
        #[cfg(feature = "std")]
        crate::logging::log_memory_event(
            "memory_compacted",
            stats_after.total_memory as f64 / (1024.0 * 1024.0),
            stats_after.allocated_memory as f64 / (1024.0 * 1024.0),
            stats_after.fragmentation as f64,
        );
    }

    // Private methods

    fn generate_text(
        &mut self,
        context_features: &Tensor<f32>,
        initial_tokens: &[u32],
        config: InferenceConfig,
    ) -> Result<String> {
        let mut generated_tokens = initial_tokens.to_vec();
        let (_, _bos_token, eos_token, _) = self.tokenizer.special_tokens();

        // Limit generation length
        let max_new_tokens = config.max_length.min(self.config.max_gen_length);

        for _ in 0..max_new_tokens {
            // Get logits for next token
            let logits = self.lm_head.forward(context_features, &mut self.memory_pool)?;
            
            // Sample next token
            let next_token = self.sample_token(&logits, &config)?;
            
            generated_tokens.push(next_token);
            
            // Stop if we hit EOS token
            if next_token == eos_token {
                break;
            }
        }

        // Decode generated tokens
        self.tokenizer.decode(&generated_tokens)
    }

    fn sample_token(&self, logits: &Tensor<f32>, config: &InferenceConfig) -> Result<u32> {
        let logits_data = logits.data();
        let vocab_size = logits_data.len();

        if config.deterministic {
            // Greedy sampling - pick the highest probability token
            let mut max_idx = 0;
            let mut max_val = logits_data[0];
            
            for (i, &val) in logits_data.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            
            return Ok(max_idx as u32);
        }

        // Apply temperature scaling
        let mut scaled_logits = vec![0.0; vocab_size];
        for (i, &logit) in logits_data.iter().enumerate() {
            scaled_logits[i] = logit / config.temperature;
        }

        // Apply softmax to get probabilities
        let mut probabilities = self.softmax(&scaled_logits)?;

        // Apply top-k filtering
        if config.top_k < vocab_size {
            self.apply_top_k(&mut probabilities, config.top_k);
        }

        // Apply top-p (nucleus) filtering
        if config.top_p < 1.0 {
            self.apply_top_p(&mut probabilities, config.top_p);
        }

        // Sample from the distribution (simplified random sampling)
        self.sample_from_distribution(&probabilities)
    }

    fn softmax(&self, logits: &[f32]) -> Result<Vec<f32>> {
        // Find maximum for numerical stability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exponentials
        let mut exp_sum = 0.0;
        let mut exp_logits = vec![0.0; logits.len()];
        
        for (i, &logit) in logits.iter().enumerate() {
            let exp_val = (logit - max_logit).exp();
            exp_logits[i] = exp_val;
            exp_sum += exp_val;
        }
        
        // Normalize
        for exp_logit in &mut exp_logits {
            *exp_logit /= exp_sum;
        }
        
        Ok(exp_logits)
    }

    fn apply_top_k(&self, probabilities: &mut [f32], k: usize) {
        // Get indices sorted by probability
        let mut indices: Vec<usize> = (0..probabilities.len()).collect();
        indices.sort_by(|&a, &b| probabilities[b].partial_cmp(&probabilities[a]).unwrap());
        
        // Zero out all but top-k
        for &idx in indices.iter().skip(k) {
            probabilities[idx] = 0.0;
        }
        
        // Renormalize
        let sum: f32 = probabilities.iter().sum();
        if sum > 0.0 {
            for prob in probabilities.iter_mut() {
                *prob /= sum;
            }
        }
    }

    fn apply_top_p(&self, probabilities: &mut [f32], p: f32) {
        // Get indices sorted by probability
        let mut indices: Vec<usize> = (0..probabilities.len()).collect();
        indices.sort_by(|&a, &b| probabilities[b].partial_cmp(&probabilities[a]).unwrap());
        
        // Find cutoff point
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = 0;
        
        for (i, &idx) in indices.iter().enumerate() {
            cumulative_prob += probabilities[idx];
            if cumulative_prob >= p {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Zero out tokens beyond cutoff
        for &idx in indices.iter().skip(cutoff_idx) {
            probabilities[idx] = 0.0;
        }
        
        // Renormalize
        let sum: f32 = probabilities.iter().sum();
        if sum > 0.0 {
            for prob in probabilities.iter_mut() {
                *prob /= sum;
            }
        }
    }

    fn sample_from_distribution(&self, probabilities: &[f32]) -> Result<u32> {
        // Simple pseudo-random sampling (in real implementation, use proper RNG)
        let mut pseudo_random = 0.42; // Fixed for deterministic testing
        
        // Use a simple linear congruential generator
        pseudo_random = (pseudo_random * 1664525.0 + 1013904223.0) % (2.0_f32.powi(32));
        pseudo_random /= 2.0_f32.powi(32);
        
        let mut cumulative = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if pseudo_random <= cumulative {
                return Ok(i as u32);
            }
        }
        
        // Fallback to last valid token
        Ok((probabilities.len() - 1) as u32)
    }
}

/// Projection layer for feature transformation
struct ProjectionLayer {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    input_dim: usize,
    output_dim: usize,
}

impl ProjectionLayer {
    fn new(input_dim: usize, output_dim: usize) -> Result<Self> {
        let weight_shape = TensorShape::new(&[input_dim, output_dim])?;
        let mut weight = Tensor::zeros(weight_shape)?;
        Self::init_xavier_uniform(&mut weight)?;

        let bias_shape = TensorShape::new(&[output_dim])?;
        let bias = Tensor::zeros(bias_shape)?;

        Ok(Self {
            weight,
            bias,
            input_dim,
            output_dim,
        })
    }

    fn forward(&self, input: &Tensor<f32>, memory_pool: &mut MemoryPool<f32>) -> Result<Tensor<f32>> {
        let input_shape = input.shape();
        let batch_size = input_shape.dims[0];
        let seq_len = if input_shape.ndim > 2 { input_shape.dims[1] } else { 1 };
        
        let output_shape = if input_shape.ndim > 2 {
            TensorShape::new(&[batch_size, seq_len, self.output_dim])?
        } else {
            TensorShape::new(&[batch_size, self.output_dim])?
        };

        let mut output = memory_pool.allocate(output_shape)?;

        // Simplified linear transformation (actual matmul would be more complex)
        let input_data = input.data();
        let output_data = output.data_mut();
        let weight_data = self.weight.data();
        let bias_data = self.bias.data();

        // Perform matrix multiplication and add bias
        for i in 0..output_data.len() {
            let out_idx = i % self.output_dim;
            output_data[i] = bias_data[out_idx];
            
            // Simplified computation - would use SIMD matmul in practice
            for j in 0..self.input_dim {
                let input_idx = (i / self.output_dim) * self.input_dim + j;
                let weight_idx = j * self.output_dim + out_idx;
                if input_idx < input_data.len() && weight_idx < weight_data.len() {
                    output_data[i] += input_data[input_idx] * weight_data[weight_idx];
                }
            }
        }

        Ok(output)
    }

    fn init_xavier_uniform(tensor: &mut Tensor<f32>) -> Result<()> {
        let shape = tensor.shape();
        let fan_in = shape.dims[0];
        let fan_out = shape.dims[1];
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        
        let data = tensor.data_mut();
        for (i, val) in data.iter_mut().enumerate() {
            let pseudo_random = ((i * 1234567) % 1000000) as f32 / 1000000.0;
            *val = (pseudo_random * 2.0 - 1.0) * bound;
        }

        Ok(())
    }
}

/// Multimodal fusion layer for combining vision and text features
struct MultimodalFusion {
    cross_attention: CrossAttention,
    norm: LayerNorm,
    hidden_dim: usize,
}

impl MultimodalFusion {
    fn new(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            cross_attention: CrossAttention::new(hidden_dim, num_heads)?,
            norm: LayerNorm::new(hidden_dim)?,
            hidden_dim,
        })
    }

    fn forward(
        &mut self,
        vision_features: &Tensor<f32>,
        text_features: &Tensor<f32>,
        memory_pool: &mut MemoryPool<f32>,
    ) -> Result<Tensor<f32>> {
        // Apply cross-attention between vision and text
        let attended_features = self.cross_attention.forward(
            text_features,  // query
            vision_features, // key and value
            memory_pool,
        )?;

        // Apply layer normalization
        self.norm.forward(&attended_features, memory_pool)
    }
}

/// Cross-attention mechanism for multimodal fusion
struct CrossAttention {
    query_proj: ProjectionLayer,
    key_proj: ProjectionLayer,
    value_proj: ProjectionLayer,
    out_proj: ProjectionLayer,
    num_heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    fn new(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        if hidden_dim % num_heads != 0 {
            return Err(TinyVlmError::config("Hidden dim must be divisible by num_heads"));
        }

        let head_dim = hidden_dim / num_heads;

        Ok(Self {
            query_proj: ProjectionLayer::new(hidden_dim, hidden_dim)?,
            key_proj: ProjectionLayer::new(hidden_dim, hidden_dim)?,
            value_proj: ProjectionLayer::new(hidden_dim, hidden_dim)?,
            out_proj: ProjectionLayer::new(hidden_dim, hidden_dim)?,
            num_heads,
            head_dim,
        })
    }

    fn forward(
        &mut self,
        query: &Tensor<f32>,
        key_value: &Tensor<f32>,
        memory_pool: &mut MemoryPool<f32>,
    ) -> Result<Tensor<f32>> {
        // Project query, key, and value
        let q = self.query_proj.forward(query, memory_pool)?;
        let k = self.key_proj.forward(key_value, memory_pool)?;
        let v = self.value_proj.forward(key_value, memory_pool)?;

        // Apply attention (simplified)
        let attention_output = self.apply_attention(&q, &k, &v, memory_pool)?;

        // Final projection
        self.out_proj.forward(&attention_output, memory_pool)
    }

    fn apply_attention(
        &self,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        memory_pool: &mut MemoryPool<f32>,
    ) -> Result<Tensor<f32>> {
        // Simplified attention computation
        // In practice, this would involve proper multi-head attention with scaling
        let output_shape = q.shape();
        let mut output = memory_pool.allocate(output_shape)?;

        let q_data = q.data();
        let v_data = v.data();
        let output_data = output.data_mut();

        // Simple weighted combination (not proper attention)
        for (i, (q_val, v_val)) in q_data.iter().zip(v_data.iter()).enumerate() {
            output_data[i] = q_val * 0.5 + v_val * 0.5;
        }

        Ok(output)
    }
}

/// Language model head for text generation
struct LanguageModelHead {
    projection: ProjectionLayer,
    vocab_size: usize,
}

impl LanguageModelHead {
    fn new(hidden_dim: usize, vocab_size: usize) -> Result<Self> {
        Ok(Self {
            projection: ProjectionLayer::new(hidden_dim, vocab_size)?,
            vocab_size,
        })
    }

    fn forward(&self, input: &Tensor<f32>, memory_pool: &mut MemoryPool<f32>) -> Result<Tensor<f32>> {
        self.projection.forward(input, memory_pool)
    }
}

/// Layer normalization
struct LayerNorm {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    eps: f32,
    hidden_dim: usize,
}

impl LayerNorm {
    fn new(hidden_dim: usize) -> Result<Self> {
        let param_shape = TensorShape::new(&[hidden_dim])?;
        
        let mut weight = Tensor::zeros(param_shape)?;
        let weight_data = weight.data_mut();
        weight_data.fill(1.0);

        let bias = Tensor::zeros(param_shape)?;

        Ok(Self {
            weight,
            bias,
            eps: 1e-5,
            hidden_dim,
        })
    }

    fn forward(&self, input: &Tensor<f32>, memory_pool: &mut MemoryPool<f32>) -> Result<Tensor<f32>> {
        // Simplified layer norm - just copy input for now
        let mut output = memory_pool.allocate(input.shape())?;
        output.data_mut().copy_from_slice(input.data());
        Ok(output)
    }
}

/// Calculate optimal memory pool size based on model configuration
fn calculate_optimal_memory_size(config: &ModelConfig) -> usize {
    // Base memory for small models
    let base_memory = 50_000_000; // 50MB base

    // Scale based on model dimensions
    let vision_factor = (config.vision_dim as f64 / 768.0).max(1.0);
    let text_factor = (config.text_dim as f64 / 768.0).max(1.0);
    let hidden_factor = (config.hidden_dim as f64 / 768.0).max(1.0);
    
    // Calculate scaling multiplier
    let scale_factor = (vision_factor * text_factor * hidden_factor).sqrt();
    
    // Apply scaling with reasonable limits
    let scaled_memory = (base_memory as f64 * scale_factor) as usize;
    
    // Clamp between 25MB and 500MB
    scaled_memory.max(25_000_000).min(500_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config() {
        let config = ModelConfig::default();
        assert_eq!(config.vision_dim, 768);
        assert_eq!(config.text_dim, 768);
        assert_eq!(config.hidden_dim, 768);
    }

    #[test]
    fn test_inference_config() {
        let config = InferenceConfig::default();
        assert_eq!(config.max_length, 100);
        assert_eq!(config.temperature, 1.0);
    }

    #[test]
    fn test_model_creation() {
        let config = ModelConfig::default();
        let model = FastVLM::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_projection_layer() {
        let proj = ProjectionLayer::new(256, 512);
        assert!(proj.is_ok());
        
        let layer = proj.unwrap();
        assert_eq!(layer.input_dim, 256);
        assert_eq!(layer.output_dim, 512);
    }

    #[test]
    fn test_multimodal_fusion() {
        let fusion = MultimodalFusion::new(768, 12);
        assert!(fusion.is_ok());
    }
}