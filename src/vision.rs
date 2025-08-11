//! Vision processing components for image encoding

use crate::{
    memory::{MemoryPool, Tensor, TensorShape},
    simd::{ConvParams, SimdDispatcher},
    Result, TinyVlmError,
};

/// Configuration for the vision encoder
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Input image height
    pub input_height: usize,
    /// Input image width
    pub input_width: usize,
    /// Number of input channels (typically 3 for RGB)
    pub input_channels: usize,
    /// Hidden dimension size
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of encoder layers
    pub num_layers: usize,
    /// Patch size for vision transformer
    pub patch_size: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            input_height: 224,
            input_width: 224,
            input_channels: 3,
            hidden_dim: 768,
            num_heads: 12,
            num_layers: 12,
            patch_size: 16,
        }
    }
}

/// Image processor for preprocessing input images
pub struct ImageProcessor {
    /// Target image dimensions
    target_height: usize,
    target_width: usize,
    /// Normalization parameters (mean, std)
    normalize_mean: [f32; 3],
    normalize_std: [f32; 3],
}

impl ImageProcessor {
    /// Create a new image processor
    pub fn new(target_height: usize, target_width: usize) -> Self {
        Self {
            target_height,
            target_width,
            // ImageNet normalization values
            normalize_mean: [0.485, 0.456, 0.406],
            normalize_std: [0.229, 0.224, 0.225],
        }
    }

    /// Preprocess raw RGB image data into model input format
    pub fn preprocess(&self, rgb_data: &[u8]) -> Result<Tensor<f32>> {
        if rgb_data.len() != self.target_height * self.target_width * 3 {
            return Err(TinyVlmError::image_processing(
                "Input image size does not match target dimensions",
            ));
        }

        let shape = TensorShape::new(&[1, self.target_height, self.target_width, 3])?;
        let mut tensor = Tensor::zeros(shape)?;
        let tensor_data = tensor.data_mut();

        // Convert u8 to f32 and normalize
        for i in 0..rgb_data.len() {
            let channel = i % 3;
            let pixel_value = rgb_data[i] as f32 / 255.0;
            let normalized = (pixel_value - self.normalize_mean[channel]) / self.normalize_std[channel];
            tensor_data[i] = normalized;
        }

        Ok(tensor)
    }

    /// Resize image to target dimensions (simple nearest neighbor)
    pub fn resize(&self, input: &[u8], input_height: usize, input_width: usize) -> Result<Vec<u8>> {
        if input.len() != input_height * input_width * 3 {
            return Err(TinyVlmError::image_processing("Invalid input dimensions"));
        }

        let mut output = vec![0u8; self.target_height * self.target_width * 3];

        for y in 0..self.target_height {
            for x in 0..self.target_width {
                // Nearest neighbor interpolation
                let src_y = (y * input_height) / self.target_height;
                let src_x = (x * input_width) / self.target_width;

                for c in 0..3 {
                    let src_idx = (src_y * input_width + src_x) * 3 + c;
                    let dst_idx = (y * self.target_width + x) * 3 + c;
                    output[dst_idx] = input[src_idx];
                }
            }
        }

        Ok(output)
    }
}

/// Vision encoder implementing a simplified vision transformer
pub struct VisionEncoder {
    /// Configuration
    config: VisionConfig,
    /// Patch embedding layer weights
    patch_embedding: ConvLayer,
    /// Position embeddings
    position_embeddings: Tensor<f32>,
    /// Transformer encoder layers
    layers: Vec<TransformerLayer>,
    /// Memory pool for efficient allocation
    memory_pool: MemoryPool<f32>,
}

impl VisionEncoder {
    /// Create a new vision encoder with given configuration
    pub fn new(config: VisionConfig) -> Result<Self> {
        let num_patches = (config.input_height / config.patch_size) * (config.input_width / config.patch_size);
        
        // Initialize patch embedding (conv layer that converts patches to embeddings)
        let patch_embedding = ConvLayer::new(
            config.input_channels,
            config.hidden_dim,
            config.patch_size,
            config.patch_size,
            0,
        )?;

        // Initialize position embeddings
        let pos_shape = TensorShape::new(&[1, num_patches + 1, config.hidden_dim])?; // +1 for CLS token
        let mut position_embeddings = Tensor::zeros(pos_shape)?;
        
        // Initialize with small random values (simplified)
        Self::init_position_embeddings(&mut position_embeddings)?;

        // Initialize transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TransformerLayer::new(config.hidden_dim, config.num_heads)?);
        }

        // Initialize memory pool (100MB limit)
        let memory_pool = MemoryPool::new(25_000_000); // 100MB for f32

        Ok(Self {
            config,
            patch_embedding,
            position_embeddings,
            layers,
            memory_pool,
        })
    }

    /// Encode input image to feature representation
    pub fn encode(&mut self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Validate input shape
        let expected_shape = [1, self.config.input_height, self.config.input_width, self.config.input_channels];
        if input.shape().dims[..4] != expected_shape {
            return Err(TinyVlmError::image_processing("Invalid input shape"));
        }

        // Convert to patches using patch embedding
        let mut patch_features = self.patch_embedding.forward(input, &mut self.memory_pool)?;

        // Add position embeddings
        self.add_position_embeddings(&mut patch_features)?;

        // Apply transformer layers
        for layer in &mut self.layers {
            patch_features = layer.forward(&patch_features, &mut self.memory_pool)?;
        }

        // Return CLS token representation (first token)
        self.extract_cls_token(&patch_features)
    }

    /// Initialize position embeddings with sinusoidal patterns
    fn init_position_embeddings(embeddings: &mut Tensor<f32>) -> Result<()> {
        let shape = embeddings.shape();
        let seq_len = shape.dims[1];
        let hidden_dim = shape.dims[2];
        let data = embeddings.data_mut();

        for pos in 0..seq_len {
            for i in 0..hidden_dim {
                let angle = pos as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / hidden_dim as f32);
                let value = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
                data[pos * hidden_dim + i] = value * 0.02; // Small magnitude
            }
        }

        Ok(())
    }

    /// Add position embeddings to patch embeddings
    fn add_position_embeddings(&self, patch_features: &mut Tensor<f32>) -> Result<()> {
        let patch_shape = patch_features.shape();
        let pos_shape = self.position_embeddings.shape();
        
        // Ensure the patch features are reshaped to match position embeddings
        // The conv layer outputs [batch, out_height, out_width, channels]
        // We need to flatten to [batch, seq_len, hidden_dim] where seq_len = out_height * out_width
        
        let batch_size = patch_shape.dims[0];
        let out_height = patch_shape.dims[1];
        let out_width = patch_shape.dims[2];
        let hidden_dim = patch_shape.dims[3];
        
        let seq_len = out_height * out_width;
        let expected_seq_len = pos_shape.dims[1] - 1; // -1 for CLS token
        
        // Validate dimensions match
        if seq_len != expected_seq_len {
            return Err(TinyVlmError::inference(&format!(
                "Sequence length mismatch: patch features {} vs position embeddings {}",
                seq_len, expected_seq_len
            )));
        }
        
        if hidden_dim != pos_shape.dims[2] {
            return Err(TinyVlmError::inference(&format!(
                "Hidden dimension mismatch: patch features {} vs position embeddings {}",
                hidden_dim, pos_shape.dims[2]
            )));
        }
        
        let patch_data = patch_features.data_mut();
        let pos_data = self.position_embeddings.data();
        
        // Add position embeddings to patch embeddings
        // Skip the CLS token embedding (index 0) and use patch embeddings (indices 1..seq_len+1)
        for seq_pos in 0..seq_len {
            for hidden_idx in 0..hidden_dim {
                let patch_idx = seq_pos * hidden_dim + hidden_idx;
                let pos_idx = (seq_pos + 1) * hidden_dim + hidden_idx; // +1 to skip CLS token
                
                if patch_idx < patch_data.len() && pos_idx < pos_data.len() {
                    patch_data[patch_idx] += pos_data[pos_idx];
                }
            }
        }
        
        Ok(())
    }

    /// Extract CLS token from sequence
    fn extract_cls_token(&mut self, features: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = features.shape();
        let hidden_dim = shape.dims[2];
        
        let cls_shape = TensorShape::new(&[1, hidden_dim])?;
        let mut cls_token = self.memory_pool.allocate(cls_shape)?;
        
        let features_data = features.data();
        let cls_data = cls_token.data_mut();
        
        // Copy first token (CLS token)
        cls_data.copy_from_slice(&features_data[..hidden_dim]);
        
        Ok(cls_token)
    }
}

/// Convolutional layer for patch embedding
struct ConvLayer {
    weights: Tensor<f32>,
    bias: Tensor<f32>,
    params: ConvParams,
}

impl ConvLayer {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Result<Self> {
        let weight_shape = TensorShape::new(&[out_channels, in_channels, kernel_size, kernel_size])?;
        let mut weights = Tensor::zeros(weight_shape)?;
        
        // Initialize with Xavier uniform
        Self::xavier_uniform_init(&mut weights)?;

        let bias_shape = TensorShape::new(&[out_channels])?;
        let bias = Tensor::zeros(bias_shape)?;

        // Create conv params (will be set dynamically)
        let params = ConvParams::new(224, 224, in_channels, out_channels, kernel_size, stride, padding)?;

        Ok(Self { weights, bias, params })
    }

    fn forward(&self, input: &Tensor<f32>, memory_pool: &mut MemoryPool<f32>) -> Result<Tensor<f32>> {
        let input_shape = input.shape();
        let batch_size = input_shape.dims[0];
        let in_height = input_shape.dims[1];
        let in_width = input_shape.dims[2];

        // Update conv params for this input
        let params = ConvParams::new(
            in_height,
            in_width,
            self.params.in_channels,
            self.params.out_channels,
            self.params.kernel_size,
            self.params.stride,
            self.params.padding,
        )?;

        let output_shape = TensorShape::new(&[
            batch_size,
            params.out_height,
            params.out_width,
            params.out_channels,
        ])?;

        let mut output = memory_pool.allocate(output_shape)?;

        // Perform convolution
        SimdDispatcher::conv2d(
            input.data(),
            self.weights.data(),
            output.data_mut(),
            params,
        )?;

        // Add bias
        self.add_bias(&mut output)?;

        Ok(output)
    }

    fn xavier_uniform_init(tensor: &mut Tensor<f32>) -> Result<()> {
        let shape = tensor.shape();
        let fan_in = shape.dims[1] * shape.dims[2] * shape.dims[3]; // in_channels * kernel_h * kernel_w
        let fan_out = shape.dims[0] * shape.dims[2] * shape.dims[3]; // out_channels * kernel_h * kernel_w
        
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        
        // Simple pseudo-random initialization (in real implementation, use proper RNG)
        let data = tensor.data_mut();
        for (i, val) in data.iter_mut().enumerate() {
            let pseudo_random = ((i * 1234567) % 1000000) as f32 / 1000000.0;
            *val = (pseudo_random * 2.0 - 1.0) * bound;
        }

        Ok(())
    }

    fn add_bias(&self, output: &mut Tensor<f32>) -> Result<()> {
        let output_shape = output.shape();
        let out_channels = output_shape.dims[3];
        let output_data = output.data_mut();
        let bias_data = self.bias.data();

        for i in 0..output_data.len() {
            let channel = i % out_channels;
            output_data[i] += bias_data[channel];
        }

        Ok(())
    }
}

/// Simplified transformer layer
struct TransformerLayer {
    attention: MultiHeadAttention,
    mlp: MLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerLayer {
    fn new(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new(hidden_dim, num_heads)?,
            mlp: MLP::new(hidden_dim, hidden_dim * 4)?,
            norm1: LayerNorm::new(hidden_dim)?,
            norm2: LayerNorm::new(hidden_dim)?,
        })
    }

    fn forward(&mut self, input: &Tensor<f32>, memory_pool: &mut MemoryPool<f32>) -> Result<Tensor<f32>> {
        // Self-attention with residual connection
        let normed1 = self.norm1.forward(input, memory_pool)?;
        let attn_out = self.attention.forward(&normed1, memory_pool)?;
        let residual1 = self.add_tensors(input, &attn_out, memory_pool)?;

        // MLP with residual connection
        let normed2 = self.norm2.forward(&residual1, memory_pool)?;
        let mlp_out = self.mlp.forward(&normed2, memory_pool)?;
        let output = self.add_tensors(&residual1, &mlp_out, memory_pool)?;

        Ok(output)
    }

    fn add_tensors(&self, a: &Tensor<f32>, b: &Tensor<f32>, memory_pool: &mut MemoryPool<f32>) -> Result<Tensor<f32>> {
        let mut result = memory_pool.allocate(a.shape())?;
        let result_data = result.data_mut();
        
        for (i, (a_val, b_val)) in a.data().iter().zip(b.data().iter()).enumerate() {
            result_data[i] = a_val + b_val;
        }
        
        Ok(result)
    }
}

/// Simplified multi-head attention
struct MultiHeadAttention {
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    qkv_proj: Tensor<f32>,
    out_proj: Tensor<f32>,
}

impl MultiHeadAttention {
    fn new(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        if hidden_dim % num_heads != 0 {
            return Err(TinyVlmError::config("Hidden dim must be divisible by num_heads"));
        }

        let head_dim = hidden_dim / num_heads;
        
        let qkv_shape = TensorShape::new(&[hidden_dim, hidden_dim * 3])?;
        let mut qkv_proj = Tensor::zeros(qkv_shape)?;
        ConvLayer::xavier_uniform_init(&mut qkv_proj)?;

        let out_shape = TensorShape::new(&[hidden_dim, hidden_dim])?;
        let mut out_proj = Tensor::zeros(out_shape)?;
        ConvLayer::xavier_uniform_init(&mut out_proj)?;

        Ok(Self {
            hidden_dim,
            num_heads,
            head_dim,
            qkv_proj,
            out_proj,
        })
    }

    fn forward(&self, input: &Tensor<f32>, memory_pool: &mut MemoryPool<f32>) -> Result<Tensor<f32>> {
        let input_shape = input.shape();
        let batch_size = input_shape.dims[0];
        let seq_len = input_shape.dims[1];

        // Project to Q, K, V (simplified - just return scaled input for now)
        let scale = (self.head_dim as f32).sqrt().recip();
        let mut output = memory_pool.allocate(input_shape)?;
        
        let input_data = input.data();
        let output_data = output.data_mut();
        
        // Simplified attention: just scale the input
        for (out, inp) in output_data.iter_mut().zip(input_data.iter()) {
            *out = inp * scale;
        }

        Ok(output)
    }
}

/// Simplified MLP layer
struct MLP {
    linear1: Tensor<f32>,
    linear2: Tensor<f32>,
    hidden_dim: usize,
    intermediate_dim: usize,
}

impl MLP {
    fn new(hidden_dim: usize, intermediate_dim: usize) -> Result<Self> {
        let linear1_shape = TensorShape::new(&[hidden_dim, intermediate_dim])?;
        let mut linear1 = Tensor::zeros(linear1_shape)?;
        ConvLayer::xavier_uniform_init(&mut linear1)?;

        let linear2_shape = TensorShape::new(&[intermediate_dim, hidden_dim])?;
        let mut linear2 = Tensor::zeros(linear2_shape)?;
        ConvLayer::xavier_uniform_init(&mut linear2)?;

        Ok(Self {
            linear1,
            linear2,
            hidden_dim,
            intermediate_dim,
        })
    }

    fn forward(&self, input: &Tensor<f32>, memory_pool: &mut MemoryPool<f32>) -> Result<Tensor<f32>> {
        // Simplified MLP: just apply ReLU to input
        let mut output = memory_pool.allocate(input.shape())?;
        let input_data = input.data();
        let output_data = output.data_mut();
        
        for (out, inp) in output_data.iter_mut().zip(input_data.iter()) {
            *out = inp.max(0.0); // ReLU activation
        }

        Ok(output)
    }
}

/// Simplified layer normalization
struct LayerNorm {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    eps: f32,
}

impl LayerNorm {
    fn new(hidden_dim: usize) -> Result<Self> {
        let param_shape = TensorShape::new(&[hidden_dim])?;
        
        let mut weight = Tensor::zeros(param_shape)?;
        let weight_data = weight.data_mut();
        weight_data.fill(1.0); // Initialize to 1

        let bias = Tensor::zeros(param_shape)?;

        Ok(Self {
            weight,
            bias,
            eps: 1e-5,
        })
    }

    fn forward(&self, input: &Tensor<f32>, memory_pool: &mut MemoryPool<f32>) -> Result<Tensor<f32>> {
        let mut output = memory_pool.allocate(input.shape())?;
        let input_data = input.data();
        let output_data = output.data_mut();
        
        // Simplified layer norm: just copy input (normalization calculation omitted for brevity)
        output_data.copy_from_slice(input_data);
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_processor() {
        let processor = ImageProcessor::new(224, 224);
        let rgb_data = vec![128u8; 224 * 224 * 3];
        
        let result = processor.preprocess(&rgb_data);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape().dims[1], 224);
        assert_eq!(tensor.shape().dims[2], 224);
        assert_eq!(tensor.shape().dims[3], 3);
    }

    #[test]
    fn test_vision_config() {
        let config = VisionConfig::default();
        assert_eq!(config.input_height, 224);
        assert_eq!(config.input_width, 224);
        assert_eq!(config.input_channels, 3);
    }

    #[test]
    fn test_conv_layer_creation() {
        let conv = ConvLayer::new(3, 64, 3, 1, 1);
        assert!(conv.is_ok());
    }

    #[test]
    fn test_vision_encoder_creation() {
        let config = VisionConfig::default();
        let encoder = VisionEncoder::new(config);
        assert!(encoder.is_ok());
    }
}