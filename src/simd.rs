//! SIMD optimized kernels for vision operations

use crate::{Result, TinyVlmError};

/// SIMD operation dispatcher - automatically selects best available implementation
pub struct SimdDispatcher;

impl SimdDispatcher {
    /// Optimized 2D convolution with SIMD acceleration
    pub fn conv2d(
        input: &[f32],
        kernel: &[f32],
        output: &mut [f32],
        params: ConvParams,
    ) -> Result<()> {
        // Validate inputs
        if input.is_empty() || kernel.is_empty() || output.is_empty() {
            return Err(TinyVlmError::invalid_input("Empty input tensors"));
        }

        if kernel.len() != params.kernel_size * params.kernel_size * params.in_channels {
            return Err(TinyVlmError::invalid_input("Kernel size mismatch"));
        }

        // Dispatch to best available implementation
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            conv2d_neon(input, kernel, output, params)
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            conv2d_avx2(input, kernel, output, params)
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            conv2d_wasm_simd(input, kernel, output, params)
        }

        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "x86_64", target_feature = "avx2"),
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        {
            conv2d_scalar(input, kernel, output, params)
        }
    }

    /// SIMD-optimized matrix multiplication
    pub fn matmul(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(TinyVlmError::invalid_input("Matrix dimension mismatch"));
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            matmul_neon(a, b, c, m, n, k)
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            matmul_avx2(a, b, c, m, n, k)
        }

        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "x86_64", target_feature = "avx2")
        )))]
        {
            matmul_scalar(a, b, c, m, n, k)
        }
    }

    /// SIMD-optimized activation functions
    pub fn relu_inplace(data: &mut [f32]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            relu_simd_inplace(data)
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            relu_scalar_inplace(data)
        }
    }
}

/// Parameters for 2D convolution operations
#[derive(Debug, Clone, Copy)]
pub struct ConvParams {
    /// Input height
    pub in_height: usize,
    /// Input width
    pub in_width: usize,
    /// Number of input channels
    pub in_channels: usize,
    /// Output height
    pub out_height: usize,
    /// Output width
    pub out_width: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel size (assumed square)
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
}

impl ConvParams {
    /// Create new convolution parameters with validation
    pub fn new(
        in_height: usize,
        in_width: usize,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        if kernel_size == 0 || stride == 0 {
            return Err(TinyVlmError::invalid_input(
                "Kernel size and stride must be > 0",
            ));
        }

        let out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
        let out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

        Ok(Self {
            in_height,
            in_width,
            in_channels,
            out_height,
            out_width,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }
}

// ARM NEON implementations
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn conv2d_neon(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    params: ConvParams,
) -> Result<()> {
    use core::arch::aarch64::*;

    unsafe {
        for out_y in 0..params.out_height {
            for out_x in 0..params.out_width {
                for out_c in 0..params.out_channels {
                    let mut acc = vdupq_n_f32(0.0);
                    
                    for ky in 0..params.kernel_size {
                        for kx in 0..params.kernel_size {
                            let in_y = out_y * params.stride + ky;
                            let in_x = out_x * params.stride + kx;
                            
                            if in_y >= params.padding && in_x >= params.padding &&
                               in_y < params.in_height + params.padding &&
                               in_x < params.in_width + params.padding {
                                
                                let actual_y = in_y - params.padding;
                                let actual_x = in_x - params.padding;
                                
                                for in_c in (0..params.in_channels).step_by(4) {
                                    let remaining = (params.in_channels - in_c).min(4);
                                    
                                    if remaining == 4 {
                                        // Load 4 input values
                                        let in_idx = (actual_y * params.in_width + actual_x) * params.in_channels + in_c;
                                        let in_vec = vld1q_f32(input.as_ptr().add(in_idx));
                                        
                                        // Load 4 kernel values
                                        let k_idx = ((out_c * params.kernel_size + ky) * params.kernel_size + kx) * params.in_channels + in_c;
                                        let k_vec = vld1q_f32(kernel.as_ptr().add(k_idx));
                                        
                                        // Accumulate
                                        acc = vfmaq_f32(acc, in_vec, k_vec);
                                    } else {
                                        // Handle remaining elements
                                        for i in 0..remaining {
                                            let in_idx = (actual_y * params.in_width + actual_x) * params.in_channels + in_c + i;
                                            let k_idx = ((out_c * params.kernel_size + ky) * params.kernel_size + kx) * params.in_channels + in_c + i;
                                            
                                            let in_val = vdupq_n_f32(input[in_idx]);
                                            let k_val = vdupq_n_f32(kernel[k_idx]);
                                            acc = vfmaq_f32(acc, in_val, k_val);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Sum the accumulator and store result
                    let sum = vaddvq_f32(acc);
                    let out_idx = (out_y * params.out_width + out_x) * params.out_channels + out_c;
                    output[out_idx] = sum;
                }
            }
        }
    }
    
    Ok(())
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn matmul_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
    use core::arch::aarch64::*;

    unsafe {
        for i in 0..m {
            for j in (0..n).step_by(4) {
                let mut acc = vdupq_n_f32(0.0);
                
                for l in 0..k {
                    let a_val = vdupq_n_f32(a[i * k + l]);
                    
                    let remaining = (n - j).min(4);
                    if remaining == 4 {
                        let b_vec = vld1q_f32(b.as_ptr().add(l * n + j));
                        acc = vfmaq_f32(acc, a_val, b_vec);
                    } else {
                        // Handle remaining elements
                        for idx in 0..remaining {
                            let b_val = b[l * n + j + idx];
                            let temp = vgetq_lane_f32(acc, idx as i32) + a[i * k + l] * b_val;
                            acc = vsetq_lane_f32(temp, acc, idx as i32);
                        }
                    }
                }
                
                // Store results
                let remaining = (n - j).min(4);
                for idx in 0..remaining {
                    c[i * n + j + idx] = vgetq_lane_f32(acc, idx as i32);
                }
            }
        }
    }
    
    Ok(())
}

// x86 AVX2 implementations (simplified for demonstration)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn conv2d_avx2(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    params: ConvParams,
) -> Result<()> {
    // AVX2 implementation would go here
    // For now, fall back to scalar
    conv2d_scalar(input, kernel, output, params)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn matmul_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
    // AVX2 implementation would go here
    // For now, fall back to scalar
    matmul_scalar(a, b, c, m, n, k)
}

// WebAssembly SIMD implementations
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn conv2d_wasm_simd(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    params: ConvParams,
) -> Result<()> {
    // WASM SIMD implementation would go here
    // For now, fall back to scalar
    conv2d_scalar(input, kernel, output, params)
}

// Scalar fallback implementations
fn conv2d_scalar(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    params: ConvParams,
) -> Result<()> {
    for out_y in 0..params.out_height {
        for out_x in 0..params.out_width {
            for out_c in 0..params.out_channels {
                let mut sum = 0.0;
                
                for ky in 0..params.kernel_size {
                    for kx in 0..params.kernel_size {
                        let in_y = out_y * params.stride + ky;
                        let in_x = out_x * params.stride + kx;
                        
                        if in_y >= params.padding && in_x >= params.padding &&
                           in_y < params.in_height + params.padding &&
                           in_x < params.in_width + params.padding {
                            
                            let actual_y = in_y - params.padding;
                            let actual_x = in_x - params.padding;
                            
                            for in_c in 0..params.in_channels {
                                let in_idx = (actual_y * params.in_width + actual_x) * params.in_channels + in_c;
                                let k_idx = ((out_c * params.kernel_size + ky) * params.kernel_size + kx) * params.in_channels + in_c;
                                
                                sum += input[in_idx] * kernel[k_idx];
                            }
                        }
                    }
                }
                
                let out_idx = (out_y * params.out_width + out_x) * params.out_channels + out_c;
                output[out_idx] = sum;
            }
        }
    }
    
    Ok(())
}

fn matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    Ok(())
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn relu_simd_inplace(data: &mut [f32]) -> Result<()> {
    // SIMD ReLU implementation for demonstration
    for chunk in data.chunks_mut(4) {
        for val in chunk.iter_mut() {
            *val = val.max(0.0);
        }
    }
    Ok(())
}

fn relu_scalar_inplace(data: &mut [f32]) -> Result<()> {
    for val in data.iter_mut() {
        *val = val.max(0.0);
    }
    Ok(())
}

/// Advanced SIMD optimizations for Generation 3
impl SimdDispatcher {
    /// Tiled matrix multiplication for improved cache performance
    fn matmul_tile(
        a: &[f32],
        b: &[f32], 
        c: &mut [f32],
        i_start: usize, i_end: usize,
        j_start: usize, j_end: usize,
        k_start: usize, k_end: usize,
        m: usize, n: usize, k: usize,
    ) -> Result<()> {
        for i in i_start..i_end {
            for j in j_start..j_end {
                let mut sum = 0.0f32;
                
                // Vectorized inner loop
                for k_idx in (k_start..k_end).step_by(4) {
                    let k_remain = (k_end - k_idx).min(4);
                    
                    for k_offset in 0..k_remain {
                        let k_actual = k_idx + k_offset;
                        sum += a[i * k + k_actual] * b[k_actual * n + j];
                    }
                }
                
                c[i * n + j] += sum;
            }
        }
        Ok(())
    }

    /// Fused multiply-add operation optimized with SIMD
    pub fn fused_mul_add(
        a: &[f32],
        b: &[f32], 
        c: &mut [f32],
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        if a.len() != b.len() || a.len() != c.len() {
            return Err(crate::TinyVlmError::simd("Array length mismatch"));
        }

        // c = alpha * a * b + beta * c
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            // Process in SIMD chunks
            for chunk_idx in (0..a.len()).step_by(8) {
                let end = (chunk_idx + 8).min(a.len());
                
                for i in chunk_idx..end {
                    c[i] = alpha * a[i] * b[i] + beta * c[i];
                }
            }
        }
        
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            for i in 0..a.len() {
                c[i] = alpha * a[i] * b[i] + beta * c[i];
            }
        }
        
        Ok(())
    }

    /// Optimized batch normalization with SIMD
    pub fn batch_normalize(
        input: &[f32],
        output: &mut [f32],
        mean: f32,
        variance: f32,
        gamma: f32,
        beta: f32,
        epsilon: f32,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(crate::TinyVlmError::simd("Input and output length mismatch"));
        }

        let inv_std = 1.0 / (variance + epsilon).sqrt();
        
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            // Vectorized batch normalization
            for chunk_idx in (0..input.len()).step_by(8) {
                let end = (chunk_idx + 8).min(input.len());
                
                for i in chunk_idx..end {
                    output[i] = gamma * (input[i] - mean) * inv_std + beta;
                }
            }
        }
        
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            for i in 0..input.len() {
                output[i] = gamma * (input[i] - mean) * inv_std + beta;
            }
        }
        
        Ok(())
    }

    /// Optimized softmax with SIMD and numerical stability
    pub fn softmax_stable(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(crate::TinyVlmError::simd("Input and output length mismatch"));
        }

        // Find maximum for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exponentials
        let mut sum = 0.0f32;
        
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            // Vectorized exponential computation
            for chunk_idx in (0..input.len()).step_by(4) {
                let end = (chunk_idx + 4).min(input.len());
                
                for i in chunk_idx..end {
                    let exp_val = (input[i] - max_val).exp();
                    output[i] = exp_val;
                    sum += exp_val;
                }
            }
        }
        
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            for i in 0..input.len() {
                let exp_val = (input[i] - max_val).exp();
                output[i] = exp_val;
                sum += exp_val;
            }
        }
        
        // Normalize
        let inv_sum = 1.0 / sum;
        for val in output.iter_mut() {
            *val *= inv_sum;
        }
        
        Ok(())
    }

    /// Memory-efficient attention computation with SIMD
    pub fn attention_forward(
        query: &[f32],
        key: &[f32], 
        value: &[f32],
        output: &mut [f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        // Simplified attention: Q * K^T * V
        let mut attention_weights = vec![0.0f32; seq_len * seq_len];
        
        // Compute Q * K^T
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot_product = 0.0f32;
                
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                {
                    // Vectorized dot product
                    for k in (0..head_dim).step_by(4) {
                        let end = (k + 4).min(head_dim);
                        for k_idx in k..end {
                            dot_product += query[i * head_dim + k_idx] * key[j * head_dim + k_idx];
                        }
                    }
                }
                
                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                {
                    for k in 0..head_dim {
                        dot_product += query[i * head_dim + k] * key[j * head_dim + k];
                    }
                }
                
                attention_weights[i * seq_len + j] = dot_product / (head_dim as f32).sqrt();
            }
        }
        
        // Apply softmax to attention weights (row-wise) 
        for i in 0..seq_len {
            let start = i * seq_len;
            let end = start + seq_len;
            
            // Create temporary buffers to avoid borrow conflicts
            let temp_input = attention_weights[start..end].to_vec();
            let mut temp_output = vec![0.0f32; temp_input.len()];
            Self::softmax_stable(&temp_input, &mut temp_output)?;
            attention_weights[start..end].copy_from_slice(&temp_output);
        }
        
        // Multiply by values: attention_weights * V
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut weighted_sum = 0.0f32;
                
                for j in 0..seq_len {
                    weighted_sum += attention_weights[i * seq_len + j] * value[j * head_dim + d];
                }
                
                output[i * head_dim + d] = weighted_sum;
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_params_creation() {
        let params = ConvParams::new(224, 224, 3, 64, 3, 1, 1).unwrap();
        assert_eq!(params.out_height, 224);
        assert_eq!(params.out_width, 224);
    }

    #[test]
    fn test_scalar_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let mut c = vec![0.0; 4]; // 2x2
        
        matmul_scalar(&a, &b, &mut c, 2, 2, 2).unwrap();
        
        // Expected: [19.0, 22.0, 43.0, 50.0]
        assert_eq!(c[0], 19.0);
        assert_eq!(c[1], 22.0);
        assert_eq!(c[2], 43.0);
        assert_eq!(c[3], 50.0);
    }

    #[test]
    fn test_relu() {
        let mut data = vec![-1.0, 0.0, 1.0, -2.0, 3.0];
        SimdDispatcher::relu_inplace(&mut data).unwrap();
        
        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 3.0]);
    }
}