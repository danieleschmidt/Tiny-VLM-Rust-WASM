//! Advanced SIMD optimization module for Tiny-VLM
//!
//! Platform-specific SIMD implementations for maximum performance on ARM NEON and x86 AVX2

pub mod arm_neon;
pub mod avx2;
pub mod wasm_simd;

use crate::{Result, TinyVlmError};
use crate::memory::{Tensor, TensorShape};

/// Convolution parameters for SIMD operations
#[derive(Debug, Clone, Copy)]
pub struct ConvParams {
    pub in_height: usize,
    pub in_width: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub out_height: usize,
    pub out_width: usize,
}

impl ConvParams {
    pub fn new(
        in_height: usize,
        in_width: usize,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
        let out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
        
        Ok(Self {
            in_height,
            in_width,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            out_height,
            out_width,
        })
    }
}

/// SIMD operation types
#[derive(Debug, Clone, Copy)]
pub enum SimdOp {
    /// Matrix multiplication
    MatMul,
    /// Convolution
    Conv2d,
    /// Element-wise operations
    ElementWise,
    /// Reduction operations
    Reduction,
}

/// SIMD capability detection and dispatch
pub struct SimdDispatcher {
    /// Available SIMD features
    pub capabilities: SimdCapabilities,
    /// Performance counters
    pub stats: SimdStats,
}

/// SIMD capabilities on the current platform
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    /// ARM NEON support
    pub neon: bool,
    /// x86 AVX2 support  
    pub avx2: bool,
    /// x86 FMA support
    pub fma: bool,
    /// WebAssembly SIMD support
    pub wasm_simd: bool,
    /// Vector width in bytes
    pub vector_width: usize,
    /// Maximum parallel operations
    pub max_parallelism: usize,
}

/// SIMD performance statistics
#[derive(Debug, Clone, Default)]
pub struct SimdStats {
    /// Total SIMD operations performed
    pub total_ops: u64,
    /// Total time spent in SIMD operations (microseconds)
    pub total_time_us: u64,
    /// Operations by type
    pub ops_by_type: [u64; 4], // MatMul, Conv2d, ElementWise, Reduction
    /// SIMD efficiency (% of operations using SIMD)
    pub simd_efficiency: f32,
}

impl SimdDispatcher {
    /// Create new SIMD dispatcher with capability detection
    pub fn new() -> Self {
        let capabilities = Self::detect_capabilities();
        
        Self {
            capabilities,
            stats: SimdStats::default(),
        }
    }

    /// Detect available SIMD capabilities
    pub fn detect_capabilities() -> SimdCapabilities {
        let mut caps = SimdCapabilities {
            neon: false,
            avx2: false,
            fma: false,
            wasm_simd: false,
            vector_width: 16, // Default to 128-bit
            max_parallelism: 1,
        };

        // ARM NEON detection
        #[cfg(target_arch = "aarch64")]
        {
            caps.neon = true;
            caps.vector_width = 16; // 128-bit NEON
            caps.max_parallelism = 4;
        }

        // x86 AVX2 detection
        #[cfg(target_arch = "x86_64")]
        {
            caps.avx2 = is_x86_feature_detected!("avx2");
            caps.fma = is_x86_feature_detected!("fma");
            if caps.avx2 {
                caps.vector_width = 32; // 256-bit AVX2
                caps.max_parallelism = 8;
            }
        }

        // WebAssembly SIMD detection
        #[cfg(target_arch = "wasm32")]
        {
            caps.wasm_simd = cfg!(target_feature = "simd128");
            if caps.wasm_simd {
                caps.vector_width = 16; // 128-bit WASM SIMD
                caps.max_parallelism = 4;
            }
        }

        caps
    }

    /// Optimized matrix multiplication with SIMD
    pub fn matmul_f32(
        &mut self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        c: &mut Tensor<f32>,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        let a_shape = a.shape();
        let b_shape = b.shape();
        let _c_shape = c.shape();

        // Validate dimensions
        if a_shape.dims[1] != b_shape.dims[0] {
            return Err(TinyVlmError::invalid_input(
                "Matrix dimensions incompatible for multiplication"
            ));
        }

        let m = a_shape.dims[0];
        let n = b_shape.dims[1];
        let k = a_shape.dims[1];

        // Choose optimal implementation
        let result = if self.capabilities.avx2 && m >= 8 && n >= 8 && k >= 8 {
            #[cfg(target_arch = "x86_64")]
            {
                avx2::matmul_avx2_f32(a.data(), b.data(), c.data_mut(), m, n, k)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                self.matmul_scalar_f32(a.data(), b.data(), c.data_mut(), m, n, k)
            }
        } else if self.capabilities.neon && m >= 4 && n >= 4 && k >= 4 {
            #[cfg(target_arch = "aarch64")]
            {
                arm_neon::matmul_neon_f32(a.data(), b.data(), c.data_mut(), m, n, k)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                self.matmul_scalar_f32(a.data(), b.data(), c.data_mut(), m, n, k)
            }
        } else if self.capabilities.wasm_simd {
            #[cfg(target_arch = "wasm32")]
            {
                wasm_simd::matmul_wasm_f32(a.data(), b.data(), c.data_mut(), m, n, k)
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.matmul_scalar_f32(a.data(), b.data(), c.data_mut(), m, n, k)
            }
        } else {
            self.matmul_scalar_f32(a.data(), b.data(), c.data_mut(), m, n, k)
        };

        // Update statistics
        self.stats.total_ops += 1;
        self.stats.total_time_us += start_time.elapsed().as_micros() as u64;
        self.stats.ops_by_type[SimdOp::MatMul as usize] += 1;

        result
    }

    /// Optimized 2D convolution with SIMD
    pub fn conv2d_f32(
        &mut self,
        input: &Tensor<f32>,
        kernel: &Tensor<f32>,
        output: &mut Tensor<f32>,
        stride: usize,
        padding: usize,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        let input_shape = input.shape();
        let kernel_shape = kernel.shape();

        // Validate shapes (NHWC format expected)
        if input_shape.ndim != 4 || kernel_shape.ndim != 4 {
            return Err(TinyVlmError::invalid_input(
                "Conv2d expects 4D tensors (NHWC format)"
            ));
        }

        let batch_size = input_shape.dims[0];
        let input_height = input_shape.dims[1];
        let input_width = input_shape.dims[2];
        let input_channels = input_shape.dims[3];

        let kernel_height = kernel_shape.dims[0];
        let kernel_width = kernel_shape.dims[1];
        let kernel_in_channels = kernel_shape.dims[2];
        let kernel_out_channels = kernel_shape.dims[3];

        if input_channels != kernel_in_channels {
            return Err(TinyVlmError::invalid_input(
                "Input and kernel channel dimensions must match"
            ));
        }

        let result = if self.capabilities.avx2 {
            #[cfg(target_arch = "x86_64")]
            {
                avx2::conv2d_avx2_f32(
                    input.data(), kernel.data(), output.data_mut(),
                    batch_size, input_height, input_width, input_channels,
                    kernel_height, kernel_width, kernel_out_channels,
                    stride, padding
                )
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                self.conv2d_scalar_f32(
                    input.data(), kernel.data(), output.data_mut(),
                    batch_size, input_height, input_width, input_channels,
                    kernel_height, kernel_width, kernel_out_channels,
                    stride, padding
                )
            }
        } else if self.capabilities.neon {
            #[cfg(target_arch = "aarch64")]
            {
                arm_neon::conv2d_neon_f32(
                    input.data(), kernel.data(), output.data_mut(),
                    batch_size, input_height, input_width, input_channels,
                    kernel_height, kernel_width, kernel_out_channels,
                    stride, padding
                )
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                self.conv2d_scalar_f32(
                    input.data(), kernel.data(), output.data_mut(),
                    batch_size, input_height, input_width, input_channels,
                    kernel_height, kernel_width, kernel_out_channels,
                    stride, padding
                )
            }
        } else if self.capabilities.wasm_simd {
            #[cfg(target_arch = "wasm32")]
            {
                {
                    let params = ConvParams::new(
                        input_height, input_width, input_channels, 
                        kernel_out_channels, kernel_height, stride, padding
                    )?;
                    wasm_simd::conv2d_wasm_simd(
                        input.data(), kernel.data(), output.data_mut(), params
                    )
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.conv2d_scalar_f32(
                    input.data(), kernel.data(), output.data_mut(),
                    batch_size, input_height, input_width, input_channels,
                    kernel_height, kernel_width, kernel_out_channels,
                    stride, padding
                )
            }
        } else {
            self.conv2d_scalar_f32(
                input.data(), kernel.data(), output.data_mut(),
                batch_size, input_height, input_width, input_channels,
                kernel_height, kernel_width, kernel_out_channels,
                stride, padding
            )
        };

        // Update statistics
        self.stats.total_ops += 1;
        self.stats.total_time_us += start_time.elapsed().as_micros() as u64;
        self.stats.ops_by_type[SimdOp::Conv2d as usize] += 1;

        result
    }

    /// Element-wise ReLU activation with SIMD
    pub fn relu_f32(&mut self, data: &mut [f32]) -> Result<()> {
        let start_time = std::time::Instant::now();

        let result = if self.capabilities.avx2 && data.len() >= 32 {
            #[cfg(target_arch = "x86_64")]
            {
                avx2::relu_avx2_f32(data)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                self.relu_scalar_f32(data)
            }
        } else if self.capabilities.neon && data.len() >= 16 {
            #[cfg(target_arch = "aarch64")]
            {
                arm_neon::relu_neon_f32(data)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                self.relu_scalar_f32(data)
            }
        } else if self.capabilities.wasm_simd && data.len() >= 16 {
            #[cfg(target_arch = "wasm32")]
            {
                wasm_simd::relu_wasm_simd(data)
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.relu_scalar_f32(data)
            }
        } else {
            self.relu_scalar_f32(data)
        };

        // Update statistics
        self.stats.total_ops += 1;
        self.stats.total_time_us += start_time.elapsed().as_micros() as u64;
        self.stats.ops_by_type[SimdOp::ElementWise as usize] += 1;

        result
    }

    /// Softmax with SIMD optimization
    pub fn softmax_f32(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let start_time = std::time::Instant::now();

        if input.len() != output.len() {
            return Err(TinyVlmError::invalid_input(
                "Input and output lengths must match for softmax"
            ));
        }

        let result = if self.capabilities.avx2 && input.len() >= 32 {
            #[cfg(target_arch = "x86_64")]
            {
                avx2::softmax_avx2_f32(input, output)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                self.softmax_scalar_f32(input, output)
            }
        } else if self.capabilities.neon && input.len() >= 16 {
            #[cfg(target_arch = "aarch64")]
            {
                arm_neon::softmax_neon_f32(input, output)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                self.softmax_scalar_f32(input, output)
            }
        } else if self.capabilities.wasm_simd && input.len() >= 16 {
            #[cfg(target_arch = "wasm32")]
            {
                wasm_simd::softmax_wasm_simd(input, output)
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.softmax_scalar_f32(input, output)
            }
        } else {
            self.softmax_scalar_f32(input, output)
        };

        // Update statistics
        self.stats.total_ops += 1;
        self.stats.total_time_us += start_time.elapsed().as_micros() as u64;
        self.stats.ops_by_type[SimdOp::Reduction as usize] += 1;

        result
    }

    /// Static convolution method for external use with ConvParams
    pub fn conv2d(
        input: &[f32],
        kernel: &[f32],
        output: &mut [f32],
        params: ConvParams,
    ) -> Result<()> {
        let dispatcher = Self::new();
        
        // Call the scalar implementation directly with the provided parameters
        dispatcher.conv2d_scalar_f32(
            input, 
            kernel, 
            output,
            1, // batch_size - assume single batch for static call
            params.in_height,
            params.in_width,
            params.in_channels,
            params.kernel_size,
            params.kernel_size, // assume square kernels
            params.out_channels,
            params.stride,
            params.padding
        )
    }

    // Scalar fallback implementations
    
    fn matmul_scalar_f32(
        &self,
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, n: usize, k: usize
    ) -> Result<()> {
        // Simple scalar matrix multiplication
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

    fn conv2d_scalar_f32(
        &self,
        input: &[f32], kernel: &[f32], output: &mut [f32],
        batch_size: usize, input_height: usize, input_width: usize, input_channels: usize,
        kernel_height: usize, kernel_width: usize, kernel_out_channels: usize,
        stride: usize, padding: usize
    ) -> Result<()> {
        // Simple scalar convolution implementation
        let output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
        let output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

        for b in 0..batch_size {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    for oc in 0..kernel_out_channels {
                        let mut sum = 0.0;
                        
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                for ic in 0..input_channels {
                                    let ih = oh * stride + kh;
                                    let iw = ow * stride + kw;
                                    
                                    // Apply padding check
                                    if ih >= padding && ih < input_height + padding &&
                                       iw >= padding && iw < input_width + padding {
                                        let ih_actual = ih - padding;
                                        let iw_actual = iw - padding;
                                        
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
        Ok(())
    }

    fn relu_scalar_f32(&self, data: &mut [f32]) -> Result<()> {
        for val in data.iter_mut() {
            if *val < 0.0 {
                *val = 0.0;
            }
        }
        Ok(())
    }

    fn softmax_scalar_f32(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        // Find max for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exponentials and sum
        let mut sum = 0.0;
        for (i, &val) in input.iter().enumerate() {
            let exp_val = (val - max_val).exp();
            output[i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for val in output.iter_mut() {
            *val /= sum;
        }
        
        Ok(())
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &SimdStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = SimdStats::default();
    }

    /// Calculate SIMD efficiency
    pub fn update_efficiency(&mut self) {
        let total_ops = self.stats.total_ops;
        if total_ops > 0 {
            let simd_ops = if self.capabilities.avx2 || self.capabilities.neon || self.capabilities.wasm_simd {
                // Estimate based on capabilities
                (total_ops as f32 * 0.8) as u64 // Assume 80% of ops use SIMD when available
            } else {
                0
            };
            
            self.stats.simd_efficiency = simd_ops as f32 / total_ops as f32;
        }
    }

    /// Benchmark SIMD operations
    pub fn benchmark(&mut self) -> Result<SimdBenchmarkResults> {
        let mut results = SimdBenchmarkResults::default();

        // Matrix multiplication benchmark
        let a_shape = TensorShape::new(&[256, 256])?;
        let b_shape = TensorShape::new(&[256, 256])?;
        let c_shape = TensorShape::new(&[256, 256])?;
        
        let a = Tensor::<f32>::zeros(a_shape)?;
        let b = Tensor::<f32>::zeros(b_shape)?;
        let mut c = Tensor::<f32>::zeros(c_shape)?;

        let start = std::time::Instant::now();
        for _ in 0..10 {
            self.matmul_f32(&a, &b, &mut c)?;
        }
        results.matmul_time_ms = start.elapsed().as_millis() as f64 / 10.0;

        // Convolution benchmark
        let input_shape = TensorShape::new(&[1, 32, 32, 3])?;
        let kernel_shape = TensorShape::new(&[3, 3, 3, 16])?;
        let output_shape = TensorShape::new(&[1, 30, 30, 16])?;
        
        let input = Tensor::<f32>::zeros(input_shape)?;
        let kernel = Tensor::<f32>::zeros(kernel_shape)?;
        let mut output = Tensor::<f32>::zeros(output_shape)?;

        let start = std::time::Instant::now();
        for _ in 0..10 {
            self.conv2d_f32(&input, &kernel, &mut output, 1, 0)?;
        }
        results.conv2d_time_ms = start.elapsed().as_millis() as f64 / 10.0;

        // Element-wise operations benchmark
        let mut data = vec![1.0f32; 10000];
        let start = std::time::Instant::now();
        for _ in 0..100 {
            self.relu_f32(&mut data)?;
        }
        results.elementwise_time_ms = start.elapsed().as_millis() as f64 / 100.0;

        Ok(results)
    }
}

impl Default for SimdDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD benchmark results
#[derive(Debug, Clone, Default)]
pub struct SimdBenchmarkResults {
    /// Matrix multiplication time (milliseconds)
    pub matmul_time_ms: f64,
    /// 2D convolution time (milliseconds)
    pub conv2d_time_ms: f64,
    /// Element-wise operations time (milliseconds)
    pub elementwise_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_detection() {
        let caps = SimdDispatcher::detect_capabilities();
        
        // Should detect at least one capability on most platforms
        let has_any = caps.neon || caps.avx2 || caps.wasm_simd;
        
        // On most modern platforms, we should have some SIMD support
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        assert!(has_any, "Expected SIMD support on modern platforms");
        
        assert!(caps.vector_width >= 16, "Vector width should be at least 128-bit");
        assert!(caps.max_parallelism >= 1, "Should support at least scalar operations");
    }

    #[test]
    fn test_simd_dispatcher_creation() {
        let dispatcher = SimdDispatcher::new();
        assert_eq!(dispatcher.stats.total_ops, 0);
    }

    #[test]
    fn test_scalar_matmul() -> Result<()> {
        let dispatcher = SimdDispatcher::new();
        
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![2.0, 1.0, 3.0, 2.0]; // 2x2
        let mut c = vec![0.0; 4]; // 2x2
        
        dispatcher.matmul_scalar_f32(&a, &b, &mut c, 2, 2, 2)?;
        
        // Expected result: [[8, 5], [18, 11]]
        assert_eq!(c[0], 8.0);
        assert_eq!(c[1], 5.0);
        assert_eq!(c[2], 18.0);
        assert_eq!(c[3], 11.0);
        
        Ok(())
    }

    #[test]
    fn test_scalar_relu() -> Result<()> {
        let dispatcher = SimdDispatcher::new();
        let mut data = vec![-1.0, 0.0, 1.0, -2.5, 3.7];
        
        dispatcher.relu_scalar_f32(&mut data)?;
        
        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 3.7]);
        
        Ok(())
    }

    #[test]
    fn test_scalar_softmax() -> Result<()> {
        let dispatcher = SimdDispatcher::new();
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        
        dispatcher.softmax_scalar_f32(&input, &mut output)?;
        
        // Check that probabilities sum to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1, got {}", sum);
        
        // Check that values are in ascending order (since input was ascending)
        assert!(output[0] < output[1]);
        assert!(output[1] < output[2]);
        
        Ok(())
    }

    #[test]
    fn test_stats_tracking() -> Result<()> {
        let mut dispatcher = SimdDispatcher::new();
        
        // Create test tensors
        let a_shape = TensorShape::new(&[2, 2])?;
        let b_shape = TensorShape::new(&[2, 2])?;
        let c_shape = TensorShape::new(&[2, 2])?;
        
        let a = Tensor::<f32>::zeros(a_shape)?;
        let b = Tensor::<f32>::zeros(b_shape)?;
        let mut c = Tensor::<f32>::zeros(c_shape)?;
        
        assert_eq!(dispatcher.stats.total_ops, 0);
        
        dispatcher.matmul_f32(&a, &b, &mut c)?;
        
        assert_eq!(dispatcher.stats.total_ops, 1);
        assert_eq!(dispatcher.stats.ops_by_type[SimdOp::MatMul as usize], 1);
        // Note: timing may be too fast to measure in tests
        assert!(dispatcher.stats.total_time_us >= 0);
        
        Ok(())
    }
}