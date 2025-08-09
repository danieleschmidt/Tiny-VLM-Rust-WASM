//! ARM NEON SIMD optimizations for Tiny-VLM
//!
//! High-performance SIMD kernels for ARM64 processors with NEON instructions

use crate::{Result, TinyVlmError};

/// ARM NEON optimized matrix multiplication
#[cfg(target_arch = "aarch64")]
pub fn matmul_neon_f32(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize
) -> Result<()> {
    use core::arch::aarch64::*;
    
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return Err(TinyVlmError::invalid_input("Matrix dimensions mismatch"));
    }

    unsafe {
        // Process 4x4 blocks for optimal NEON utilization
        let m_blocks = m / 4;
        let n_blocks = n / 4;
        let k_blocks = k / 4;
        
        for i in 0..m_blocks {
            for j in 0..n_blocks {
                // Initialize accumulator registers
                let mut acc00 = vdupq_n_f32(0.0);
                let mut acc01 = vdupq_n_f32(0.0);
                let mut acc02 = vdupq_n_f32(0.0);
                let mut acc03 = vdupq_n_f32(0.0);
                
                for l in 0..k_blocks {
                    let k_base = l * 4;
                    
                    // Load 4x4 block from matrix A
                    let a0 = vld1q_f32(a.as_ptr().add((i * 4 + 0) * k + k_base));
                    let a1 = vld1q_f32(a.as_ptr().add((i * 4 + 1) * k + k_base));
                    let a2 = vld1q_f32(a.as_ptr().add((i * 4 + 2) * k + k_base));
                    let a3 = vld1q_f32(a.as_ptr().add((i * 4 + 3) * k + k_base));
                    
                    // Load 4x4 block from matrix B  
                    let b0 = vld1q_f32(b.as_ptr().add((k_base + 0) * n + j * 4));
                    let b1 = vld1q_f32(b.as_ptr().add((k_base + 1) * n + j * 4));
                    let b2 = vld1q_f32(b.as_ptr().add((k_base + 2) * n + j * 4));
                    let b3 = vld1q_f32(b.as_ptr().add((k_base + 3) * n + j * 4));
                    
                    // Perform fused multiply-add operations
                    acc00 = vfmaq_laneq_f32::<0>(acc00, b0, a0);
                    acc00 = vfmaq_laneq_f32::<1>(acc00, b1, a0);
                    acc00 = vfmaq_laneq_f32::<2>(acc00, b2, a0);
                    acc00 = vfmaq_laneq_f32::<3>(acc00, b3, a0);
                    
                    acc01 = vfmaq_laneq_f32::<0>(acc01, b0, a1);
                    acc01 = vfmaq_laneq_f32::<1>(acc01, b1, a1);
                    acc01 = vfmaq_laneq_f32::<2>(acc01, b2, a1);
                    acc01 = vfmaq_laneq_f32::<3>(acc01, b3, a1);
                    
                    acc02 = vfmaq_laneq_f32::<0>(acc02, b0, a2);
                    acc02 = vfmaq_laneq_f32::<1>(acc02, b1, a2);
                    acc02 = vfmaq_laneq_f32::<2>(acc02, b2, a2);
                    acc02 = vfmaq_laneq_f32::<3>(acc02, b3, a2);
                    
                    acc03 = vfmaq_laneq_f32::<0>(acc03, b0, a3);
                    acc03 = vfmaq_laneq_f32::<1>(acc03, b1, a3);
                    acc03 = vfmaq_laneq_f32::<2>(acc03, b2, a3);
                    acc03 = vfmaq_laneq_f32::<3>(acc03, b3, a3);
                }
                
                // Store results
                vst1q_f32(c.as_mut_ptr().add((i * 4 + 0) * n + j * 4), acc00);
                vst1q_f32(c.as_mut_ptr().add((i * 4 + 1) * n + j * 4), acc01);
                vst1q_f32(c.as_mut_ptr().add((i * 4 + 2) * n + j * 4), acc02);
                vst1q_f32(c.as_mut_ptr().add((i * 4 + 3) * n + j * 4), acc03);
            }
        }
        
        // Handle remaining elements with scalar code
        handle_matmul_remainder(a, b, c, m, n, k, m_blocks * 4, n_blocks * 4);
    }
    
    Ok(())
}

/// ARM NEON optimized 2D convolution
#[cfg(target_arch = "aarch64")]
pub fn conv2d_neon_f32(
    input: &[f32], kernel: &[f32], output: &mut [f32],
    batch_size: usize, input_height: usize, input_width: usize, input_channels: usize,
    kernel_height: usize, kernel_width: usize, kernel_out_channels: usize,
    stride: usize, padding: usize
) -> Result<()> {
    use core::arch::aarch64::*;
    
    let output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    let output_width = (input_width + 2 * padding - kernel_width) / stride + 1;
    
    unsafe {
        // Process output channels in groups of 4 for NEON optimization
        let oc_blocks = kernel_out_channels / 4;
        
        for b in 0..batch_size {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    for oc_block in 0..oc_blocks {
                        let mut acc = vdupq_n_f32(0.0);
                        
                        // Convolution computation with NEON
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                
                                if ih >= padding && ih < input_height + padding &&
                                   iw >= padding && iw < input_width + padding {
                                    let ih_actual = ih - padding;
                                    let iw_actual = iw - padding;
                                    
                                    // Load input values (broadcasting across channels)
                                    for ic in (0..input_channels).step_by(4) {
                                        let ic_end = (ic + 4).min(input_channels);
                                        
                                        if ic_end - ic == 4 {
                                            let input_idx = b * input_height * input_width * input_channels +
                                                           ih_actual * input_width * input_channels +
                                                           iw_actual * input_channels + ic;
                                                           
                                            let input_vec = vld1q_f32(input.as_ptr().add(input_idx));
                                            
                                            // Load corresponding kernel weights
                                            let kernel_idx = kh * kernel_width * input_channels * kernel_out_channels +
                                                           kw * input_channels * kernel_out_channels +
                                                           ic * kernel_out_channels + oc_block * 4;
                                                           
                                            let kernel_vec = vld1q_f32(kernel.as_ptr().add(kernel_idx));
                                            
                                            // Fused multiply-add
                                            acc = vfmaq_f32(acc, input_vec, kernel_vec);
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Store result
                        let output_idx = b * output_height * output_width * kernel_out_channels +
                                       oh * output_width * kernel_out_channels +
                                       ow * kernel_out_channels + oc_block * 4;
                                       
                        vst1q_f32(output.as_mut_ptr().add(output_idx), acc);
                    }
                    
                    // Handle remaining output channels
                    for oc in (oc_blocks * 4)..kernel_out_channels {
                        let mut sum = 0.0f32;
                        
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                
                                if ih >= padding && ih < input_height + padding &&
                                   iw >= padding && iw < input_width + padding {
                                    let ih_actual = ih - padding;
                                    let iw_actual = iw - padding;
                                    
                                    for ic in 0..input_channels {
                                        let input_idx = b * input_height * input_width * input_channels +
                                                      ih_actual * input_width * input_channels +
                                                      iw_actual * input_channels + ic;
                                                      
                                        let kernel_idx = kh * kernel_width * input_channels * kernel_out_channels +
                                                       kw * input_channels * kernel_out_channels +
                                                       ic * kernel_out_channels + oc;
                                                       
                                        sum += input[input_idx] * kernel[kernel_idx];
                                    }
                                }
                            }
                        }
                        
                        let output_idx = b * output_height * output_width * kernel_out_channels +
                                       oh * output_width * kernel_out_channels +
                                       ow * kernel_out_channels + oc;
                                       
                        output[output_idx] = sum;
                    }
                }
            }
        }
    }
    
    Ok(())
}

/// ARM NEON optimized ReLU activation
#[cfg(target_arch = "aarch64")]
pub fn relu_neon_f32(data: &mut [f32]) -> Result<()> {
    use core::arch::aarch64::*;
    
    let len = data.len();
    let vector_len = len / 4;
    
    unsafe {
        let zero = vdupq_n_f32(0.0);
        
        // Process 4 elements at a time
        for i in 0..vector_len {
            let idx = i * 4;
            let vec = vld1q_f32(data.as_ptr().add(idx));
            let result = vmaxq_f32(vec, zero);
            vst1q_f32(data.as_mut_ptr().add(idx), result);
        }
        
        // Handle remaining elements
        for i in (vector_len * 4)..len {
            if data[i] < 0.0 {
                data[i] = 0.0;
            }
        }
    }
    
    Ok(())
}

/// ARM NEON optimized softmax
#[cfg(target_arch = "aarch64")]
pub fn softmax_neon_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    use core::arch::aarch64::*;
    
    if input.len() != output.len() {
        return Err(TinyVlmError::invalid_input("Input and output lengths must match"));
    }
    
    let len = input.len();
    let vector_len = len / 4;
    
    unsafe {
        // Find maximum value for numerical stability
        let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
        
        for i in 0..vector_len {
            let idx = i * 4;
            let vec = vld1q_f32(input.as_ptr().add(idx));
            max_vec = vmaxq_f32(max_vec, vec);
        }
        
        // Extract maximum from vector
        let max_array: [f32; 4] = core::mem::transmute(max_vec);
        let mut max_val = max_array[0].max(max_array[1]).max(max_array[2]).max(max_array[3]);
        
        // Check remaining elements
        for i in (vector_len * 4)..len {
            max_val = max_val.max(input[i]);
        }
        
        let max_broadcast = vdupq_n_f32(max_val);
        
        // Compute exponentials and sum
        let mut sum_vec = vdupq_n_f32(0.0);
        
        for i in 0..vector_len {
            let idx = i * 4;
            let vec = vld1q_f32(input.as_ptr().add(idx));
            let shifted = vsubq_f32(vec, max_broadcast);
            
            // Approximate exp using polynomial (for better performance)
            let exp_vec = fast_exp_neon(shifted);
            vst1q_f32(output.as_mut_ptr().add(idx), exp_vec);
            sum_vec = vaddq_f32(sum_vec, exp_vec);
        }
        
        // Extract sum from vector and add remaining elements
        let sum_array: [f32; 4] = core::mem::transmute(sum_vec);
        let mut total_sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        for i in (vector_len * 4)..len {
            let exp_val = (input[i] - max_val).exp();
            output[i] = exp_val;
            total_sum += exp_val;
        }
        
        // Normalize
        let inv_sum = vdupq_n_f32(1.0 / total_sum);
        
        for i in 0..vector_len {
            let idx = i * 4;
            let vec = vld1q_f32(output.as_ptr().add(idx));
            let normalized = vmulq_f32(vec, inv_sum);
            vst1q_f32(output.as_mut_ptr().add(idx), normalized);
        }
        
        for i in (vector_len * 4)..len {
            output[i] /= total_sum;
        }
    }
    
    Ok(())
}

/// Fast exponential approximation using NEON
#[cfg(target_arch = "aarch64")]
unsafe fn fast_exp_neon(x: core::arch::aarch64::float32x4_t) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;
    
    // Clamp input to reasonable range
    let min_val = vdupq_n_f32(-10.0);
    let max_val = vdupq_n_f32(10.0);
    let clamped = vmaxq_f32(vminq_f32(x, max_val), min_val);
    
    // Use polynomial approximation: e^x ≈ 1 + x + x²/2 + x³/6
    let one = vdupq_n_f32(1.0);
    let half = vdupq_n_f32(0.5);
    let sixth = vdupq_n_f32(1.0 / 6.0);
    
    let x2 = vmulq_f32(clamped, clamped);
    let x3 = vmulq_f32(x2, clamped);
    
    let term1 = clamped;
    let term2 = vmulq_f32(x2, half);
    let term3 = vmulq_f32(x3, sixth);
    
    vaddq_f32(vaddq_f32(vaddq_f32(one, term1), term2), term3)
}

/// Handle matrix multiplication remainder (non-vectorized portions)
fn handle_matmul_remainder(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
    m_processed: usize, n_processed: usize
) {
    // Handle remaining rows
    for i in m_processed..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    
    // Handle remaining columns
    for i in 0..m_processed {
        for j in n_processed..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// Stub implementations for non-ARM targets
#[cfg(not(target_arch = "aarch64"))]
pub fn matmul_neon_f32(
    _a: &[f32], _b: &[f32], _c: &mut [f32],
    _m: usize, _n: usize, _k: usize
) -> Result<()> {
    Err(TinyVlmError::simd("NEON not available on this platform"))
}

#[cfg(not(target_arch = "aarch64"))]
pub fn conv2d_neon_f32(
    _input: &[f32], _kernel: &[f32], _output: &mut [f32],
    _batch_size: usize, _input_height: usize, _input_width: usize, _input_channels: usize,
    _kernel_height: usize, _kernel_width: usize, _kernel_out_channels: usize,
    _stride: usize, _padding: usize
) -> Result<()> {
    Err(TinyVlmError::simd("NEON not available on this platform"))
}

#[cfg(not(target_arch = "aarch64"))]
pub fn relu_neon_f32(_data: &mut [f32]) -> Result<()> {
    Err(TinyVlmError::simd("NEON not available on this platform"))
}

#[cfg(not(target_arch = "aarch64"))]
pub fn softmax_neon_f32(_input: &[f32], _output: &mut [f32]) -> Result<()> {
    Err(TinyVlmError::simd("NEON not available on this platform"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_relu() -> Result<()> {
        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 1.5, -3.0];
        relu_neon_f32(&mut data)?;
        
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.5, 0.0];
        
        for (actual, expected) in data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        
        Ok(())
    }

    #[test] 
    #[cfg(target_arch = "aarch64")]
    fn test_neon_matmul_small() -> Result<()> {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![2.0, 1.0, 3.0, 2.0]; // 2x2  
        let mut c = vec![0.0; 4]; // 2x2
        
        matmul_neon_f32(&a, &b, &mut c, 2, 2, 2)?;
        
        // Expected: [[8, 5], [18, 11]]
        assert!((c[0] - 8.0).abs() < 1e-6);
        assert!((c[1] - 5.0).abs() < 1e-6);  
        assert!((c[2] - 18.0).abs() < 1e-6);
        assert!((c[3] - 11.0).abs() < 1e-6);
        
        Ok(())
    }

    #[test]
    #[cfg(not(target_arch = "aarch64"))]
    fn test_neon_not_available() {
        let mut data = vec![1.0, 2.0, 3.0];
        let result = relu_neon_f32(&mut data);
        assert!(result.is_err());
    }
}