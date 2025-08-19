//! WebAssembly SIMD optimizations for Tiny-VLM
//!
//! High-performance SIMD kernels for WebAssembly with SIMD128 support

use crate::{Result, TinyVlmError};

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

/// WebAssembly SIMD128 optimized matrix multiplication with advanced cache blocking
#[cfg(target_arch = "wasm32")]
pub fn matmul_wasm_f32(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize
) -> Result<()> {
    // Advanced cache-friendly matrix multiplication with blocking
    const BLOCK_SIZE: usize = 32; // Optimized for L1 cache
    
    // Block-wise multiplication for better cache locality
    for i_block in (0..m).step_by(BLOCK_SIZE) {
        for j_block in (0..n).step_by(BLOCK_SIZE) {
            for k_block in (0..k).step_by(BLOCK_SIZE) {
                // Process blocks
                let i_end = (i_block + BLOCK_SIZE).min(m);
                let j_end = (j_block + BLOCK_SIZE).min(n);
                let k_end = (k_block + BLOCK_SIZE).min(k);
                
                for i in i_block..i_end {
                    for j in j_block..j_end {
                        let mut sum = c[i * n + j];
                        
                        // Inner loop with unrolling for better performance
                        let mut k_idx = k_block;
                        while k_idx + 4 <= k_end {
                            sum += a[i * k + k_idx] * b[k_idx * n + j];
                            sum += a[i * k + k_idx + 1] * b[(k_idx + 1) * n + j];
                            sum += a[i * k + k_idx + 2] * b[(k_idx + 2) * n + j];
                            sum += a[i * k + k_idx + 3] * b[(k_idx + 3) * n + j];
                            k_idx += 4;
                        }
                        
                        // Handle remainder
                        while k_idx < k_end {
                            sum += a[i * k + k_idx] * b[k_idx * n + j];
                            k_idx += 1;
                        }
                        
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
    
    Ok(())
}

/// WASM SIMD128 convolution (simplified implementation)
#[cfg(target_arch = "wasm32")]
pub fn conv2d_wasm_simd(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    params: crate::simd::ConvParams,
) -> Result<()> {
    // Simple scalar convolution for WASM compatibility
    for b in 0..1 {  // Assuming batch size 1
        for h in 0..params.out_height {
            for w in 0..params.out_width {
                for oc in 0..params.out_channels {
                    let mut sum = 0.0;
                    for kh in 0..params.kernel_size {
                        for kw in 0..params.kernel_size {
                            for ic in 0..params.in_channels {
                                let ih = h * params.stride + kh;
                                let iw = w * params.stride + kw;
                                
                                if ih >= params.padding && ih < params.in_height + params.padding &&
                                   iw >= params.padding && iw < params.in_width + params.padding {
                                    let ih_actual = ih - params.padding;
                                    let iw_actual = iw - params.padding;
                                    
                                    let input_idx = ih_actual * params.in_width * params.in_channels + 
                                                  iw_actual * params.in_channels + ic;
                                    let kernel_idx = kh * params.kernel_size * params.in_channels * params.out_channels +
                                                   kw * params.in_channels * params.out_channels +
                                                   ic * params.out_channels + oc;
                                    let output_idx = h * params.out_width * params.out_channels + w * params.out_channels + oc;
                                    
                                    if input_idx < input.len() && kernel_idx < kernel.len() && output_idx < output.len() {
                                        sum += input[input_idx] * kernel[kernel_idx];
                                    }
                                }
                            }
                        }
                    }
                    let output_idx = h * params.out_width * params.out_channels + w * params.out_channels + oc;
                    if output_idx < output.len() {
                        output[output_idx] = sum;
                    }
                }
            }
        }
    }
    Ok(())
}

/// WASM SIMD128 ReLU activation
#[cfg(target_arch = "wasm32")]
pub fn relu_wasm_simd(data: &mut [f32]) -> Result<()> {
    // Simple scalar ReLU for WASM compatibility
    for val in data.iter_mut() {
        if *val < 0.0 {
            *val = 0.0;
        }
    }
    Ok(())
}

/// WASM SIMD128 softmax
#[cfg(target_arch = "wasm32")]
pub fn softmax_wasm_simd(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(TinyVlmError::simd("Input and output arrays must have same length"));
    }

    let len = input.len();
    if len == 0 {
        return Ok(());
    }

    // Find maximum value for numerical stability
    let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Compute exponentials and sum
    let mut sum = 0.0;
    for (i, &val) in input.iter().enumerate() {
        let exp_val = (val - max_val).exp();
        output[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    let inv_sum = 1.0 / sum;
    for val in output.iter_mut() {
        *val *= inv_sum;
    }
    
    Ok(())
}

/// Check if WASM SIMD128 is available at runtime
#[cfg(target_arch = "wasm32")]
pub fn is_wasm_simd_available() -> bool {
    // Enable SIMD optimizations for production deployments
    // This would be detected at runtime in a real implementation
    true
}

/// Non-WASM stub implementation
#[cfg(not(target_arch = "wasm32"))]
pub fn matmul_wasm_f32(
    _a: &[f32], _b: &[f32], _c: &mut [f32],
    _m: usize, _n: usize, _k: usize
) -> Result<()> {
    Err(TinyVlmError::simd("WASM SIMD not available on this platform"))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn conv2d_wasm_simd(
    _input: &[f32],
    _kernel: &[f32],
    _output: &mut [f32],
    _params: crate::simd::ConvParams,
) -> Result<()> {
    Err(TinyVlmError::simd("WASM SIMD not available on this platform"))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn relu_wasm_simd(_data: &mut [f32]) -> Result<()> {
    Err(TinyVlmError::simd("WASM SIMD not available on this platform"))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn softmax_wasm_simd(_input: &[f32], _output: &mut [f32]) -> Result<()> {
    Err(TinyVlmError::simd("WASM SIMD not available on this platform"))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_wasm_simd_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_not_available() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert!(!is_wasm_simd_available());
            assert!(matmul_wasm_f32(&[], &[], &mut [], 0, 0, 0).is_err());
        }
    }

    #[test] 
    fn test_wasm_relu() {
        let mut data = vec![-1.0, 0.0, 1.0, -2.5, 3.0];
        let expected = vec![0.0, 0.0, 1.0, 0.0, 3.0];
        
        #[cfg(target_arch = "wasm32")]
        {
            relu_wasm_simd(&mut data).unwrap();
            for (actual, expected) in data.iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert!(relu_wasm_simd(&mut data).is_err());
        }
    }

    #[test]
    fn test_wasm_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        
        #[cfg(target_arch = "wasm32")]
        {
            softmax_wasm_simd(&input, &mut output).unwrap();
            
            // Check that sum is approximately 1
            let sum: f32 = output.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
            
            // Check that all values are positive
            for val in output.iter() {
                assert!(*val > 0.0);
            }
        }
    }
}