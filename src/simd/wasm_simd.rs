//! WebAssembly SIMD optimizations for Tiny-VLM
//!
//! High-performance SIMD kernels for WebAssembly with SIMD128 support

use crate::{Result, TinyVlmError};

/// WebAssembly SIMD128 optimized matrix multiplication
#[cfg(target_arch = "wasm32")]
pub fn matmul_wasm_f32(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize
) -> Result<()> {
    use core::arch::wasm32::*;
    
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return Err(TinyVlmError::invalid_input("Matrix dimensions mismatch"));
    }

    if !cfg!(target_feature = "simd128") {
        return Err(TinyVlmError::simd("WASM SIMD128 not supported"));
    }

    unsafe {
        // Process 4x4 blocks for optimal WASM SIMD utilization (128-bit vectors)
        let m_blocks = m / 4;
        let n_blocks = n / 4;
        let k_blocks = k / 4;
        
        for i in 0..m_blocks {
            for j in 0..n_blocks {
                // Initialize accumulator registers (4x4 block)
                let mut acc = [[f32x4_splat(0.0); 4]; 4];
                
                for l in 0..k_blocks {
                    let k_base = l * 4;
                    
                    // Load 4x4 block from matrix A
                    let mut a_rows = [f32x4_splat(0.0); 4];
                    for row in 0..4 {
                        a_rows[row] = v128_load(a.as_ptr().add((i * 4 + row) * k + k_base) as *const v128) as f32x4;
                    }
                    
                    // Load 4x4 block from matrix B (with gather pattern)
                    for row in 0..4 {
                        for col in 0..4 {
                            // Dot product accumulation
                            let b_col = f32x4(
                                b[(k_base + 0) * n + j * 4 + col],
                                b[(k_base + 1) * n + j * 4 + col],
                                b[(k_base + 2) * n + j * 4 + col],
                                b[(k_base + 3) * n + j * 4 + col]
                            );
                            
                            acc[row][col] = f32x4_add(acc[row][col], f32x4_mul(a_rows[row], b_col));
                        }
                    }
                }
                
                // Horizontal sum and store results
                for row in 0..4 {
                    for col in 0..4 {
                        let sum = horizontal_sum_wasm(acc[row][col]);
                        c[(i * 4 + row) * n + j * 4 + col] = sum;
                    }
                }
            }
        }
        
        // Handle remaining elements with scalar code
        handle_matmul_remainder_wasm(a, b, c, m, n, k, m_blocks * 4, n_blocks * 4);
    }
    
    Ok(())
}

/// WebAssembly SIMD128 optimized 2D convolution
#[cfg(target_arch = "wasm32")]
pub fn conv2d_wasm_f32(
    input: &[f32], kernel: &[f32], output: &mut [f32],
    batch_size: usize, input_height: usize, input_width: usize, input_channels: usize,
    kernel_height: usize, kernel_width: usize, kernel_out_channels: usize,
    stride: usize, padding: usize
) -> Result<()> {
    use core::arch::wasm32::*;
    
    if !cfg!(target_feature = "simd128") {
        return Err(TinyVlmError::simd("WASM SIMD128 not supported"));
    }

    let output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    let output_width = (input_width + 2 * padding - kernel_width) / stride + 1;
    
    unsafe {
        // Process output channels in groups of 4 for WASM SIMD optimization
        let oc_blocks = kernel_out_channels / 4;
        
        for b in 0..batch_size {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    for oc_block in 0..oc_blocks {
                        let mut acc = f32x4_splat(0.0);
                        
                        // Convolution computation with WASM SIMD
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                
                                if ih >= padding && ih < input_height + padding &&
                                   iw >= padding && iw < input_width + padding {
                                    let ih_actual = ih - padding;
                                    let iw_actual = iw - padding;
                                    
                                    // Vectorized across channels
                                    for ic in (0..input_channels).step_by(4) {
                                        let ic_end = (ic + 4).min(input_channels);
                                        
                                        if ic_end - ic == 4 {
                                            let input_idx = b * input_height * input_width * input_channels +
                                                           ih_actual * input_width * input_channels +
                                                           iw_actual * input_channels + ic;
                                                           
                                            let input_vec = v128_load(input.as_ptr().add(input_idx) as *const v128) as f32x4;
                                            
                                            // Load corresponding kernel weights
                                            let kernel_idx = kh * kernel_width * input_channels * kernel_out_channels +
                                                           kw * input_channels * kernel_out_channels +
                                                           ic * kernel_out_channels + oc_block * 4;
                                                           
                                            let kernel_vec = v128_load(kernel.as_ptr().add(kernel_idx) as *const v128) as f32x4;
                                            
                                            // Fused multiply-add
                                            acc = f32x4_add(acc, f32x4_mul(input_vec, kernel_vec));
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Store result
                        let output_idx = b * output_height * output_width * kernel_out_channels +
                                       oh * output_width * kernel_out_channels +
                                       ow * kernel_out_channels + oc_block * 4;
                                       
                        v128_store(output.as_mut_ptr().add(output_idx) as *mut v128, acc as v128);
                    }
                    
                    // Handle remaining output channels with scalar code
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

/// WebAssembly SIMD128 optimized ReLU activation
#[cfg(target_arch = "wasm32")]
pub fn relu_wasm_f32(data: &mut [f32]) -> Result<()> {
    use core::arch::wasm32::*;
    
    if !cfg!(target_feature = "simd128") {
        return Err(TinyVlmError::simd("WASM SIMD128 not supported"));
    }

    let len = data.len();
    let vector_len = len / 4;
    
    unsafe {
        let zero = f32x4_splat(0.0);
        
        // Process 4 elements at a time
        for i in 0..vector_len {
            let idx = i * 4;
            let vec = v128_load(data.as_ptr().add(idx) as *const v128) as f32x4;
            let result = f32x4_max(vec, zero);
            v128_store(data.as_mut_ptr().add(idx) as *mut v128, result as v128);
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

/// WebAssembly SIMD128 optimized softmax
#[cfg(target_arch = "wasm32")]
pub fn softmax_wasm_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    use core::arch::wasm32::*;
    
    if !cfg!(target_feature = "simd128") {
        return Err(TinyVlmError::simd("WASM SIMD128 not supported"));
    }

    if input.len() != output.len() {
        return Err(TinyVlmError::invalid_input("Input and output lengths must match"));
    }
    
    let len = input.len();
    let vector_len = len / 4;
    
    unsafe {
        // Find maximum value for numerical stability
        let mut max_vec = f32x4_splat(f32::NEG_INFINITY);
        
        for i in 0..vector_len {
            let idx = i * 4;
            let vec = v128_load(input.as_ptr().add(idx) as *const v128) as f32x4;
            max_vec = f32x4_max(max_vec, vec);
        }
        
        // Extract maximum from vector
        let max_val = horizontal_max_wasm(max_vec);
        
        // Check remaining elements
        let mut final_max = max_val;
        for i in (vector_len * 4)..len {
            final_max = final_max.max(input[i]);
        }
        
        let max_broadcast = f32x4_splat(final_max);
        
        // Compute exponentials and sum
        let mut sum_vec = f32x4_splat(0.0);
        
        for i in 0..vector_len {
            let idx = i * 4;
            let vec = v128_load(input.as_ptr().add(idx) as *const v128) as f32x4;
            let shifted = f32x4_sub(vec, max_broadcast);
            
            // Fast exponential approximation
            let exp_vec = fast_exp_wasm(shifted);
            v128_store(output.as_mut_ptr().add(idx) as *mut v128, exp_vec as v128);
            sum_vec = f32x4_add(sum_vec, exp_vec);
        }
        
        // Extract sum from vector and add remaining elements
        let mut total_sum = horizontal_sum_wasm(sum_vec);
        
        for i in (vector_len * 4)..len {
            let exp_val = (input[i] - final_max).exp();
            output[i] = exp_val;
            total_sum += exp_val;
        }
        
        // Normalize
        let inv_sum = f32x4_splat(1.0 / total_sum);
        
        for i in 0..vector_len {
            let idx = i * 4;
            let vec = v128_load(output.as_ptr().add(idx) as *const v128) as f32x4;
            let normalized = f32x4_mul(vec, inv_sum);
            v128_store(output.as_mut_ptr().add(idx) as *mut v128, normalized as v128);
        }
        
        for i in (vector_len * 4)..len {
            output[i] /= total_sum;
        }
    }
    
    Ok(())
}

/// Horizontal sum of 4 float elements in WASM SIMD vector
#[cfg(target_arch = "wasm32")]
unsafe fn horizontal_sum_wasm(vec: core::arch::wasm32::f32x4) -> f32 {
    use core::arch::wasm32::*;
    
    // Extract elements and sum them
    let a = f32x4_extract_lane::<0>(vec);
    let b = f32x4_extract_lane::<1>(vec);
    let c = f32x4_extract_lane::<2>(vec);
    let d = f32x4_extract_lane::<3>(vec);
    
    a + b + c + d
}

/// Horizontal max of 4 float elements in WASM SIMD vector
#[cfg(target_arch = "wasm32")]
unsafe fn horizontal_max_wasm(vec: core::arch::wasm32::f32x4) -> f32 {
    use core::arch::wasm32::*;
    
    let a = f32x4_extract_lane::<0>(vec);
    let b = f32x4_extract_lane::<1>(vec);
    let c = f32x4_extract_lane::<2>(vec);
    let d = f32x4_extract_lane::<3>(vec);
    
    a.max(b).max(c).max(d)
}

/// Fast exponential approximation using WASM SIMD
#[cfg(target_arch = "wasm32")]
unsafe fn fast_exp_wasm(x: core::arch::wasm32::f32x4) -> core::arch::wasm32::f32x4 {
    use core::arch::wasm32::*;
    
    // Clamp input to reasonable range
    let min_val = f32x4_splat(-10.0);
    let max_val = f32x4_splat(10.0);
    let clamped = f32x4_max(f32x4_min(x, max_val), min_val);
    
    // Use polynomial approximation: e^x ≈ 1 + x + x²/2 + x³/6
    let one = f32x4_splat(1.0);
    let half = f32x4_splat(0.5);
    let sixth = f32x4_splat(1.0 / 6.0);
    
    let x2 = f32x4_mul(clamped, clamped);
    let x3 = f32x4_mul(x2, clamped);
    
    let term1 = clamped;
    let term2 = f32x4_mul(x2, half);
    let term3 = f32x4_mul(x3, sixth);
    
    f32x4_add(f32x4_add(f32x4_add(one, term1), term2), term3)
}

/// Handle matrix multiplication remainder (non-vectorized portions)
#[cfg(target_arch = "wasm32")]
fn handle_matmul_remainder_wasm(
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

// Stub implementations for non-WASM targets
#[cfg(not(target_arch = "wasm32"))]
pub fn matmul_wasm_f32(
    _a: &[f32], _b: &[f32], _c: &mut [f32],
    _m: usize, _n: usize, _k: usize
) -> Result<()> {
    Err(TinyVlmError::simd("WASM SIMD not available on this platform"))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn conv2d_wasm_f32(
    _input: &[f32], _kernel: &[f32], _output: &mut [f32],
    _batch_size: usize, _input_height: usize, _input_width: usize, _input_channels: usize,
    _kernel_height: usize, _kernel_width: usize, _kernel_out_channels: usize,
    _stride: usize, _padding: usize
) -> Result<()> {
    Err(TinyVlmError::simd("WASM SIMD not available on this platform"))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn relu_wasm_f32(_data: &mut [f32]) -> Result<()> {
    Err(TinyVlmError::simd("WASM SIMD not available on this platform"))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn softmax_wasm_f32(_input: &[f32], _output: &mut [f32]) -> Result<()> {
    Err(TinyVlmError::simd("WASM SIMD not available on this platform"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_wasm_relu() -> Result<()> {
        if !cfg!(target_feature = "simd128") {
            return Ok(()); // Skip if SIMD128 not available
        }

        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 1.5, -3.0];
        relu_wasm_f32(&mut data)?;
        
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.5, 0.0];
        
        for (actual, expected) in data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        
        Ok(())
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_wasm_horizontal_sum() {
        if !cfg!(target_feature = "simd128") {
            return; // Skip if SIMD128 not available
        }

        unsafe {
            use core::arch::wasm32::*;
            
            let vec = f32x4(1.0, 2.0, 3.0, 4.0);
            let sum = horizontal_sum_wasm(vec);
            assert!((sum - 10.0).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_wasm_horizontal_max() {
        if !cfg!(target_feature = "simd128") {
            return; // Skip if SIMD128 not available
        }

        unsafe {
            use core::arch::wasm32::*;
            
            let vec = f32x4(1.0, 8.0, 3.0, 4.0);
            let max = horizontal_max_wasm(vec);
            assert!((max - 8.0).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wasm_not_available() {
        let mut data = vec![1.0, 2.0, 3.0];
        let result = relu_wasm_f32(&mut data);
        assert!(result.is_err());
    }
}