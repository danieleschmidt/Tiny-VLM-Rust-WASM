//! x86 AVX2 SIMD optimizations for Tiny-VLM
//!
//! High-performance SIMD kernels for x86-64 processors with AVX2 and FMA instructions

use crate::{Result, TinyVlmError};

/// x86 AVX2 optimized matrix multiplication
#[cfg(target_arch = "x86_64")]
pub fn matmul_avx2_f32(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize
) -> Result<()> {
    use core::arch::x86_64::*;
    
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return Err(TinyVlmError::invalid_input("Matrix dimensions mismatch"));
    }

    if !is_x86_feature_detected!("avx2") {
        return Err(TinyVlmError::simd("AVX2 not supported on this CPU"));
    }

    unsafe {
        // Process 8x8 blocks for optimal AVX2 utilization (256-bit vectors)
        let m_blocks = m / 8;
        let n_blocks = n / 8;
        let k_blocks = k / 8;
        
        for i in 0..m_blocks {
            for j in 0..n_blocks {
                // Initialize accumulator registers (8x8 block)
                let mut acc = [[_mm256_setzero_ps(); 8]; 8];
                
                for l in 0..k_blocks {
                    let k_base = l * 8;
                    
                    // Load 8x8 block from matrix A
                    let mut a_rows = [_mm256_setzero_ps(); 8];
                    for row in 0..8 {
                        a_rows[row] = _mm256_loadu_ps(a.as_ptr().add((i * 8 + row) * k + k_base));
                    }
                    
                    // Load 8x8 block from matrix B (transposed for better access pattern)
                    let mut b_cols = [_mm256_setzero_ps(); 8];
                    for col in 0..8 {
                        // Gather elements for column (stride access)
                        let mut b_col = [0.0f32; 8];
                        for row in 0..8 {
                            b_col[row] = b[(k_base + row) * n + j * 8 + col];
                        }
                        b_cols[col] = _mm256_loadu_ps(b_col.as_ptr());
                    }
                    
                    // Perform 8x8 outer product with FMA
                    for row in 0..8 {
                        for col in 0..8 {
                            // Broadcast each element of A row across vector
                            for k_elem in 0..8 {
                                let a_elem = _mm256_set1_ps(a[(i * 8 + row) * k + k_base + k_elem]);
                                let b_vec = _mm256_set1_ps(b[(k_base + k_elem) * n + j * 8 + col]);
                                acc[row][col] = _mm256_fmadd_ps(a_elem, b_vec, acc[row][col]);
                            }
                        }
                    }
                }
                
                // Horizontal sum and store results
                for row in 0..8 {
                    for col in 0..8 {
                        // Horizontal sum of 8 elements
                        let sum = horizontal_sum_avx2(acc[row][col]);
                        c[(i * 8 + row) * n + j * 8 + col] = sum;
                    }
                }
            }
        }
        
        // Handle remaining elements with scalar code
        handle_matmul_remainder_avx2(a, b, c, m, n, k, m_blocks * 8, n_blocks * 8);
    }
    
    Ok(())
}

/// x86 AVX2 optimized 2D convolution
#[cfg(target_arch = "x86_64")]
pub fn conv2d_avx2_f32(
    input: &[f32], kernel: &[f32], output: &mut [f32],
    batch_size: usize, input_height: usize, input_width: usize, input_channels: usize,
    kernel_height: usize, kernel_width: usize, kernel_out_channels: usize,
    stride: usize, padding: usize
) -> Result<()> {
    use core::arch::x86_64::*;
    
    if !is_x86_feature_detected!("avx2") {
        return Err(TinyVlmError::simd("AVX2 not supported on this CPU"));
    }

    let output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    let output_width = (input_width + 2 * padding - kernel_width) / stride + 1;
    
    unsafe {
        // Process output channels in groups of 8 for AVX2 optimization
        let oc_blocks = kernel_out_channels / 8;
        
        for b in 0..batch_size {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    for oc_block in 0..oc_blocks {
                        let mut acc = _mm256_setzero_ps();
                        
                        // Convolution computation with AVX2
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                
                                if ih >= padding && ih < input_height + padding &&
                                   iw >= padding && iw < input_width + padding {
                                    let ih_actual = ih - padding;
                                    let iw_actual = iw - padding;
                                    
                                    // Vectorized across input channels
                                    for ic in (0..input_channels).step_by(8) {
                                        let ic_end = (ic + 8).min(input_channels);
                                        
                                        if ic_end - ic == 8 {
                                            let input_idx = b * input_height * input_width * input_channels +
                                                           ih_actual * input_width * input_channels +
                                                           iw_actual * input_channels + ic;
                                                           
                                            let input_vec = _mm256_loadu_ps(input.as_ptr().add(input_idx));
                                            
                                            // Load corresponding kernel weights
                                            let kernel_idx = kh * kernel_width * input_channels * kernel_out_channels +
                                                           kw * input_channels * kernel_out_channels +
                                                           ic * kernel_out_channels + oc_block * 8;
                                                           
                                            let kernel_vec = _mm256_loadu_ps(kernel.as_ptr().add(kernel_idx));
                                            
                                            // Fused multiply-add
                                            acc = _mm256_fmadd_ps(input_vec, kernel_vec, acc);
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Store result
                        let output_idx = b * output_height * output_width * kernel_out_channels +
                                       oh * output_width * kernel_out_channels +
                                       ow * kernel_out_channels + oc_block * 8;
                                       
                        _mm256_storeu_ps(output.as_mut_ptr().add(output_idx), acc);
                    }
                    
                    // Handle remaining output channels
                    for oc in (oc_blocks * 8)..kernel_out_channels {
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

/// x86 AVX2 optimized ReLU activation
#[cfg(target_arch = "x86_64")]
pub fn relu_avx2_f32(data: &mut [f32]) -> Result<()> {
    use core::arch::x86_64::*;
    
    if !is_x86_feature_detected!("avx2") {
        return Err(TinyVlmError::simd("AVX2 not supported on this CPU"));
    }

    let len = data.len();
    let vector_len = len / 8;
    
    unsafe {
        let zero = _mm256_setzero_ps();
        
        // Process 8 elements at a time
        for i in 0..vector_len {
            let idx = i * 8;
            let vec = _mm256_loadu_ps(data.as_ptr().add(idx));
            let result = _mm256_max_ps(vec, zero);
            _mm256_storeu_ps(data.as_mut_ptr().add(idx), result);
        }
        
        // Handle remaining elements
        for i in (vector_len * 8)..len {
            if data[i] < 0.0 {
                data[i] = 0.0;
            }
        }
    }
    
    Ok(())
}

/// x86 AVX2 optimized softmax
#[cfg(target_arch = "x86_64")]
pub fn softmax_avx2_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    use core::arch::x86_64::*;
    
    if !is_x86_feature_detected!("avx2") {
        return Err(TinyVlmError::simd("AVX2 not supported on this CPU"));
    }

    if input.len() != output.len() {
        return Err(TinyVlmError::invalid_input("Input and output lengths must match"));
    }
    
    let len = input.len();
    let vector_len = len / 8;
    
    unsafe {
        // Find maximum value for numerical stability
        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        
        for i in 0..vector_len {
            let idx = i * 8;
            let vec = _mm256_loadu_ps(input.as_ptr().add(idx));
            max_vec = _mm256_max_ps(max_vec, vec);
        }
        
        // Extract maximum from vector
        let max_val = horizontal_max_avx2(max_vec);
        
        // Check remaining elements
        let mut final_max = max_val;
        for i in (vector_len * 8)..len {
            final_max = final_max.max(input[i]);
        }
        
        let max_broadcast = _mm256_set1_ps(final_max);
        
        // Compute exponentials and sum
        let mut sum_vec = _mm256_setzero_ps();
        
        for i in 0..vector_len {
            let idx = i * 8;
            let vec = _mm256_loadu_ps(input.as_ptr().add(idx));
            let shifted = _mm256_sub_ps(vec, max_broadcast);
            
            // Fast exponential approximation
            let exp_vec = fast_exp_avx2(shifted);
            _mm256_storeu_ps(output.as_mut_ptr().add(idx), exp_vec);
            sum_vec = _mm256_add_ps(sum_vec, exp_vec);
        }
        
        // Extract sum from vector and add remaining elements
        let mut total_sum = horizontal_sum_avx2(sum_vec);
        
        for i in (vector_len * 8)..len {
            let exp_val = (input[i] - final_max).exp();
            output[i] = exp_val;
            total_sum += exp_val;
        }
        
        // Normalize
        let inv_sum = _mm256_set1_ps(1.0 / total_sum);
        
        for i in 0..vector_len {
            let idx = i * 8;
            let vec = _mm256_loadu_ps(output.as_ptr().add(idx));
            let normalized = _mm256_mul_ps(vec, inv_sum);
            _mm256_storeu_ps(output.as_mut_ptr().add(idx), normalized);
        }
        
        for i in (vector_len * 8)..len {
            output[i] /= total_sum;
        }
    }
    
    Ok(())
}

/// Horizontal sum of 8 float elements in AVX2 vector
#[cfg(target_arch = "x86_64")]
unsafe fn horizontal_sum_avx2(vec: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    
    unsafe {
        // Sum pairs: [a0+a1, a2+a3, a4+a5, a6+a7, ...]
        let sum1 = _mm256_hadd_ps(vec, vec);
        // Sum pairs of pairs: [a0+a1+a2+a3, a4+a5+a6+a7, ...]
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        
        // Extract high and low 128-bit lanes
        let low = _mm256_castps256_ps128(sum2);
        let high = _mm256_extractf128_ps(sum2, 1);
        
        // Add high and low lanes
        let final_sum = _mm_add_ps(low, high);
        
        // Extract the first element
        _mm_cvtss_f32(final_sum)
    }
}

/// Horizontal max of 8 float elements in AVX2 vector
#[cfg(target_arch = "x86_64")]
unsafe fn horizontal_max_avx2(vec: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    
    unsafe {
        // Extract high and low 128-bit lanes
        let low = _mm256_castps256_ps128(vec);
        let high = _mm256_extractf128_ps(vec, 1);
        
        // Max of high and low lanes
        let max_lanes = _mm_max_ps(low, high);
        
        // Horizontal max within 128-bit vector
        let shuf1 = _mm_shuffle_ps(max_lanes, max_lanes, 0b_11_01_11_01);
        let max1 = _mm_max_ps(max_lanes, shuf1);
        let shuf2 = _mm_shuffle_ps(max1, max1, 0b_10_10_10_10);
        let max2 = _mm_max_ps(max1, shuf2);
        
        _mm_cvtss_f32(max2)
    }
}

/// Fast exponential approximation using AVX2
#[cfg(target_arch = "x86_64")]
unsafe fn fast_exp_avx2(x: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    
    unsafe {
        // Clamp input to reasonable range
        let min_val = _mm256_set1_ps(-10.0);
        let max_val = _mm256_set1_ps(10.0);
        let clamped = _mm256_max_ps(_mm256_min_ps(x, max_val), min_val);
        
        // Use polynomial approximation: e^x ≈ 1 + x + x²/2 + x³/6 + x⁴/24
        let one = _mm256_set1_ps(1.0);
        let half = _mm256_set1_ps(0.5);
        let sixth = _mm256_set1_ps(1.0 / 6.0);
        let twenty_fourth = _mm256_set1_ps(1.0 / 24.0);
        
        let x2 = _mm256_mul_ps(clamped, clamped);
        let x3 = _mm256_mul_ps(x2, clamped);
        let x4 = _mm256_mul_ps(x3, clamped);
        
        let term1 = clamped;
        let term2 = _mm256_mul_ps(x2, half);
        let term3 = _mm256_mul_ps(x3, sixth);
        let term4 = _mm256_mul_ps(x4, twenty_fourth);
        
        _mm256_add_ps(
            _mm256_add_ps(
                _mm256_add_ps(one, term1),
                _mm256_add_ps(term2, term3)
            ),
            term4
        )
    }
}

/// Handle matrix multiplication remainder (non-vectorized portions)
#[cfg(target_arch = "x86_64")]
fn handle_matmul_remainder_avx2(
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

// Stub implementations for non-x86_64 targets
#[cfg(not(target_arch = "x86_64"))]
pub fn matmul_avx2_f32(
    _a: &[f32], _b: &[f32], _c: &mut [f32],
    _m: usize, _n: usize, _k: usize
) -> Result<()> {
    Err(TinyVlmError::simd("AVX2 not available on this platform"))
}

#[cfg(not(target_arch = "x86_64"))]
pub fn conv2d_avx2_f32(
    _input: &[f32], _kernel: &[f32], _output: &mut [f32],
    _batch_size: usize, _input_height: usize, _input_width: usize, _input_channels: usize,
    _kernel_height: usize, _kernel_width: usize, _kernel_out_channels: usize,
    _stride: usize, _padding: usize
) -> Result<()> {
    Err(TinyVlmError::simd("AVX2 not available on this platform"))
}

#[cfg(not(target_arch = "x86_64"))]
pub fn relu_avx2_f32(_data: &mut [f32]) -> Result<()> {
    Err(TinyVlmError::simd("AVX2 not available on this platform"))
}

#[cfg(not(target_arch = "x86_64"))]
pub fn softmax_avx2_f32(_input: &[f32], _output: &mut [f32]) -> Result<()> {
    Err(TinyVlmError::simd("AVX2 not available on this platform"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_relu() -> Result<()> {
        if !is_x86_feature_detected!("avx2") {
            return Ok(()); // Skip if AVX2 not available
        }

        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 1.5, -3.0];
        relu_avx2_f32(&mut data)?;
        
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.5, 0.0];
        
        for (actual, expected) in data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        
        Ok(())
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_horizontal_sum() {
        if !is_x86_feature_detected!("avx2") {
            return; // Skip if AVX2 not available
        }

        unsafe {
            use core::arch::x86_64::*;
            
            let vec = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            let sum = horizontal_sum_avx2(vec);
            assert!((sum - 36.0).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_horizontal_max() {
        if !is_x86_feature_detected!("avx2") {
            return; // Skip if AVX2 not available
        }

        unsafe {
            use core::arch::x86_64::*;
            
            let vec = _mm256_setr_ps(1.0, 8.0, 3.0, 4.0, 5.0, 2.0, 7.0, 6.0);
            let max = horizontal_max_avx2(vec);
            assert!((max - 8.0).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(not(target_arch = "x86_64"))]
    fn test_avx2_not_available() {
        let mut data = vec![1.0, 2.0, 3.0];
        let result = relu_avx2_f32(&mut data);
        assert!(result.is_err());
    }
}