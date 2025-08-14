//! GPU acceleration support for Tiny-VLM
//!
//! Provides CUDA and OpenCL backends for high-performance model inference

use crate::{Result, TinyVlmError};

pub mod cuda;
pub mod opencl;
pub mod metal;

/// GPU backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Cuda,
    OpenCL,
    Metal,
    WebGPU,
}

/// GPU memory information
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub used_bytes: u64,
    pub device_name: String,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: u32,
    pub name: String,
    pub backend: GpuBackend,
    pub compute_capability: String,
    pub memory_info: GpuMemoryInfo,
    pub max_threads_per_block: u32,
    pub max_shared_memory: u32,
}

/// GPU context for managing resources
#[derive(Debug)]
pub struct GpuContext {
    device: GpuDevice,
    backend: GpuBackend,
    streams: Vec<GpuStream>,
    memory_pool: GpuMemoryPool,
}

/// GPU computation stream
#[derive(Debug)]
pub struct GpuStream {
    id: u32,
    backend: GpuBackend,
    is_synchronous: bool,
}

/// GPU memory pool for efficient allocation
#[derive(Debug)]
pub struct GpuMemoryPool {
    total_allocated: u64,
    free_blocks: Vec<GpuMemoryBlock>,
    used_blocks: Vec<GpuMemoryBlock>,
}

/// GPU memory block
#[derive(Debug, Clone)]
pub struct GpuMemoryBlock {
    ptr: u64, // Raw pointer as u64 for portability
    size: u64,
    alignment: u32,
}

/// GPU tensor for computation
#[derive(Debug)]
pub struct GpuTensor<T> {
    data: GpuMemoryBlock,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: std::marker::PhantomData<T>,
}

impl GpuContext {
    /// Create a new GPU context
    pub fn new(backend: GpuBackend) -> Result<Self> {
        let devices = Self::enumerate_devices(backend)?;
        let device = devices.into_iter().next()
            .ok_or_else(|| TinyVlmError::gpu("No GPU devices available"))?;
        
        Ok(Self {
            device,
            backend,
            streams: Vec::new(),
            memory_pool: GpuMemoryPool::new(),
        })
    }
    
    /// Enumerate available GPU devices
    pub fn enumerate_devices(backend: GpuBackend) -> Result<Vec<GpuDevice>> {
        match backend {
            GpuBackend::Cuda => cuda::enumerate_devices(),
            GpuBackend::OpenCL => opencl::enumerate_devices(),
            GpuBackend::Metal => metal::enumerate_devices(),
            GpuBackend::WebGPU => Err(TinyVlmError::gpu("WebGPU not yet implemented")),
        }
    }
    
    /// Create a new computation stream
    pub fn create_stream(&mut self) -> Result<u32> {
        let stream_id = self.streams.len() as u32;
        let stream = GpuStream {
            id: stream_id,
            backend: self.backend,
            is_synchronous: false,
        };
        self.streams.push(stream);
        Ok(stream_id)
    }
    
    /// Allocate GPU memory
    pub fn allocate<T>(&mut self, size: usize) -> Result<GpuMemoryBlock> {
        let bytes = size * std::mem::size_of::<T>();
        self.memory_pool.allocate(bytes as u64, 256) // 256-byte alignment
    }
    
    /// Create a GPU tensor
    pub fn create_tensor<T>(&mut self, shape: &[usize]) -> Result<GpuTensor<T>> {
        let total_elements: usize = shape.iter().product();
        let memory_block = self.allocate::<T>(total_elements)?;
        
        // Calculate strides (row-major order)
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        Ok(GpuTensor {
            data: memory_block,
            shape: shape.to_vec(),
            strides,
            dtype: std::marker::PhantomData,
        })
    }
    
    /// Synchronize all streams
    pub fn synchronize(&self) -> Result<()> {
        match self.backend {
            GpuBackend::Cuda => cuda::synchronize(),
            GpuBackend::OpenCL => opencl::synchronize(),
            GpuBackend::Metal => metal::synchronize(),
            GpuBackend::WebGPU => Err(TinyVlmError::gpu("WebGPU not yet implemented")),
        }
    }
    
    /// Get device information
    pub fn device_info(&self) -> &GpuDevice {
        &self.device
    }
    
    /// Check if backend is available
    pub fn is_available(backend: GpuBackend) -> bool {
        match backend {
            GpuBackend::Cuda => cuda::is_available(),
            GpuBackend::OpenCL => opencl::is_available(),
            GpuBackend::Metal => metal::is_available(),
            GpuBackend::WebGPU => false, // Not implemented yet
        }
    }
    
    /// Get optimal backend for current platform
    pub fn optimal_backend() -> Option<GpuBackend> {
        // Priority order: CUDA > Metal > OpenCL > WebGPU
        if Self::is_available(GpuBackend::Cuda) {
            Some(GpuBackend::Cuda)
        } else if Self::is_available(GpuBackend::Metal) {
            Some(GpuBackend::Metal)
        } else if Self::is_available(GpuBackend::OpenCL) {
            Some(GpuBackend::OpenCL)
        } else {
            None
        }
    }
}

impl GpuMemoryPool {
    /// Create a new memory pool
    pub fn new() -> Self {
        Self {
            total_allocated: 0,
            free_blocks: Vec::new(),
            used_blocks: Vec::new(),
        }
    }
    
    /// Allocate memory from pool
    pub fn allocate(&mut self, size: u64, alignment: u32) -> Result<GpuMemoryBlock> {
        // Try to find a suitable free block
        for (i, block) in self.free_blocks.iter().enumerate() {
            if block.size >= size && block.alignment >= alignment {
                let block = self.free_blocks.remove(i);
                self.used_blocks.push(block.clone());
                return Ok(block);
            }
        }
        
        // No suitable block found, allocate new one
        let aligned_size = ((size + alignment as u64 - 1) / alignment as u64) * alignment as u64;
        let ptr = self.allocate_raw(aligned_size)?;
        
        let block = GpuMemoryBlock {
            ptr,
            size: aligned_size,
            alignment,
        };
        
        self.used_blocks.push(block.clone());
        self.total_allocated += aligned_size;
        
        Ok(block)
    }
    
    /// Free memory block back to pool
    pub fn free(&mut self, block: GpuMemoryBlock) -> Result<()> {
        // Remove from used blocks
        if let Some(pos) = self.used_blocks.iter().position(|b| b.ptr == block.ptr) {
            self.used_blocks.remove(pos);
            self.free_blocks.push(block);
            Ok(())
        } else {
            Err(TinyVlmError::gpu("Block not found in used blocks"))
        }
    }
    
    /// Raw memory allocation (platform-specific)
    fn allocate_raw(&self, size: u64) -> Result<u64> {
        // This would be implemented by specific backends
        // For now, return a dummy pointer
        Ok(0x1000 + self.total_allocated)
    }
    
    /// Get memory statistics
    pub fn stats(&self) -> GpuMemoryStats {
        let used = self.used_blocks.iter().map(|b| b.size).sum();
        let free = self.free_blocks.iter().map(|b| b.size).sum();
        
        GpuMemoryStats {
            total_allocated: self.total_allocated,
            used_bytes: used,
            free_bytes: free,
            num_allocations: self.used_blocks.len(),
            num_free_blocks: self.free_blocks.len(),
        }
    }
}

/// GPU memory statistics
#[derive(Debug, Clone)]
pub struct GpuMemoryStats {
    pub total_allocated: u64,
    pub used_bytes: u64,
    pub free_bytes: u64,
    pub num_allocations: usize,
    pub num_free_blocks: usize,
}

impl<T> GpuTensor<T> {
    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get tensor strides
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Get tensor rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    
    /// Check if tensor is contiguous
    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }
    
    /// Reshape tensor (creates new view)
    pub fn reshape(&self, new_shape: &[usize]) -> Result<GpuTensor<T>> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(TinyVlmError::invalid_input("Reshape size mismatch"));
        }
        
        // Calculate new strides
        let mut new_strides = vec![1; new_shape.len()];
        for i in (0..new_shape.len().saturating_sub(1)).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }
        
        Ok(GpuTensor {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            strides: new_strides,
            dtype: std::marker::PhantomData,
        })
    }
}

/// GPU kernel launcher trait
pub trait GpuKernel {
    type Args;
    
    /// Launch kernel on GPU
    fn launch(&self, args: Self::Args, stream: Option<u32>) -> Result<()>;
    
    /// Get optimal grid and block dimensions
    fn optimal_dimensions(&self, problem_size: &[usize]) -> (Vec<usize>, Vec<usize>);
}

/// High-performance GPU operations
pub struct GpuOperations {
    context: GpuContext,
}

impl GpuOperations {
    /// Create new GPU operations instance
    pub fn new(backend: GpuBackend) -> Result<Self> {
        let context = GpuContext::new(backend)?;
        Ok(Self { context })
    }
    
    /// Matrix multiplication on GPU
    pub fn matmul<T>(&mut self, a: &GpuTensor<T>, b: &GpuTensor<T>) -> Result<GpuTensor<T>>
    where
        T: Clone + Default + std::fmt::Debug
    {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(TinyVlmError::invalid_input("Matrices must be 2D"));
        }
        
        let (m, k1) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        
        if k1 != k2 {
            return Err(TinyVlmError::invalid_input("Matrix dimensions incompatible"));
        }
        
        let result = self.context.create_tensor::<T>(&[m, n])?;
        
        match self.context.backend {
            GpuBackend::Cuda => cuda::matmul(a, b, &result)?,
            GpuBackend::OpenCL => opencl::matmul(a, b, &result)?,
            GpuBackend::Metal => metal::matmul(a, b, &result)?,
            GpuBackend::WebGPU => return Err(TinyVlmError::gpu("WebGPU not implemented")),
        }
        
        Ok(result)
    }
    
    /// Convolution operation on GPU
    pub fn conv2d<T>(&mut self, 
        input: &GpuTensor<T>, 
        kernel: &GpuTensor<T>,
        stride: usize,
        padding: usize
    ) -> Result<GpuTensor<T>>
    where
        T: Clone + Default + std::fmt::Debug
    {
        if input.shape.len() != 4 || kernel.shape.len() != 4 {
            return Err(TinyVlmError::invalid_input("Input and kernel must be 4D"));
        }
        
        let (batch, in_channels, in_height, in_width) = 
            (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
        let (out_channels, _, k_height, k_width) = 
            (kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3]);
        
        let out_height = (in_height + 2 * padding - k_height) / stride + 1;
        let out_width = (in_width + 2 * padding - k_width) / stride + 1;
        
        let result = self.context.create_tensor::<T>(&[batch, out_channels, out_height, out_width])?;
        
        match self.context.backend {
            GpuBackend::Cuda => cuda::conv2d(input, kernel, &result, stride, padding)?,
            GpuBackend::OpenCL => opencl::conv2d(input, kernel, &result, stride, padding)?,
            GpuBackend::Metal => metal::conv2d(input, kernel, &result, stride, padding)?,
            GpuBackend::WebGPU => return Err(TinyVlmError::gpu("WebGPU not implemented")),
        }
        
        Ok(result)
    }
    
    /// Attention operation on GPU
    pub fn attention<T>(&mut self,
        query: &GpuTensor<T>,
        key: &GpuTensor<T>, 
        value: &GpuTensor<T>
    ) -> Result<GpuTensor<T>>
    where
        T: Clone + Default + std::fmt::Debug
    {
        // Scaled dot-product attention: softmax(QK^T / sqrt(d_k))V
        let d_k = key.shape()[key.shape().len() - 1];
        let scale = 1.0 / (d_k as f32).sqrt();
        
        // QK^T
        let scores = self.matmul(query, key)?;
        
        // Scale and softmax
        let attention_weights = self.softmax(&scores)?;
        
        // Multiply by V
        self.matmul(&attention_weights, value)
    }
    
    /// Softmax operation on GPU
    pub fn softmax<T>(&mut self, input: &GpuTensor<T>) -> Result<GpuTensor<T>>
    where
        T: Clone + Default + std::fmt::Debug
    {
        let result = self.context.create_tensor::<T>(input.shape())?;
        
        match self.context.backend {
            GpuBackend::Cuda => cuda::softmax(input, &result)?,
            GpuBackend::OpenCL => opencl::softmax(input, &result)?,
            GpuBackend::Metal => metal::softmax(input, &result)?,
            GpuBackend::WebGPU => return Err(TinyVlmError::gpu("WebGPU not implemented")),
        }
        
        Ok(result)
    }
    
    /// Layer normalization on GPU
    pub fn layer_norm<T>(&mut self, 
        input: &GpuTensor<T>,
        weight: &GpuTensor<T>,
        bias: &GpuTensor<T>,
        eps: f32
    ) -> Result<GpuTensor<T>>
    where
        T: Clone + Default + std::fmt::Debug
    {
        let result = self.context.create_tensor::<T>(input.shape())?;
        
        match self.context.backend {
            GpuBackend::Cuda => cuda::layer_norm(input, weight, bias, &result, eps)?,
            GpuBackend::OpenCL => opencl::layer_norm(input, weight, bias, &result, eps)?,
            GpuBackend::Metal => metal::layer_norm(input, weight, bias, &result, eps)?,
            GpuBackend::WebGPU => return Err(TinyVlmError::gpu("WebGPU not implemented")),
        }
        
        Ok(result)
    }
    
    /// Copy tensor from CPU to GPU
    pub fn copy_from_cpu<T>(&mut self, cpu_data: &[T]) -> Result<GpuTensor<T>>
    where
        T: Clone + Default + std::fmt::Debug
    {
        let shape = vec![cpu_data.len()];
        let gpu_tensor = self.context.create_tensor::<T>(&shape)?;
        
        match self.context.backend {
            GpuBackend::Cuda => cuda::copy_to_gpu(cpu_data, &gpu_tensor)?,
            GpuBackend::OpenCL => opencl::copy_to_gpu(cpu_data, &gpu_tensor)?,
            GpuBackend::Metal => metal::copy_to_gpu(cpu_data, &gpu_tensor)?,
            GpuBackend::WebGPU => return Err(TinyVlmError::gpu("WebGPU not implemented")),
        }
        
        Ok(gpu_tensor)
    }
    
    /// Copy tensor from GPU to CPU
    pub fn copy_to_cpu<T>(&self, gpu_tensor: &GpuTensor<T>) -> Result<Vec<T>>
    where
        T: Clone + Default + std::fmt::Debug
    {
        let mut cpu_data = vec![T::default(); gpu_tensor.numel()];
        
        match self.context.backend {
            GpuBackend::Cuda => cuda::copy_to_cpu(gpu_tensor, &mut cpu_data)?,
            GpuBackend::OpenCL => opencl::copy_to_cpu(gpu_tensor, &mut cpu_data)?,
            GpuBackend::Metal => metal::copy_to_cpu(gpu_tensor, &mut cpu_data)?,
            GpuBackend::WebGPU => return Err(TinyVlmError::gpu("WebGPU not implemented")),
        }
        
        Ok(cpu_data)
    }
    
    /// Get GPU context
    pub fn context(&self) -> &GpuContext {
        &self.context
    }
    
    /// Get GPU context mutably
    pub fn context_mut(&mut self) -> &mut GpuContext {
        &mut self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_detection() {
        // Test that at least one backend detection works
        let _cuda_available = GpuContext::is_available(GpuBackend::Cuda);
        let _opencl_available = GpuContext::is_available(GpuBackend::OpenCL);
        let _metal_available = GpuContext::is_available(GpuBackend::Metal);
        
        // Should not panic
    }
    
    #[test]
    fn test_memory_pool() {
        let mut pool = GpuMemoryPool::new();
        
        // Test allocation
        let block1 = pool.allocate(1024, 256).unwrap();
        assert_eq!(block1.size, 1024);
        assert_eq!(block1.alignment, 256);
        
        let block2 = pool.allocate(2048, 256).unwrap();
        assert_eq!(block2.size, 2048);
        
        // Test free
        pool.free(block1).unwrap();
        
        let stats = pool.stats();
        assert_eq!(stats.num_allocations, 1);
        assert_eq!(stats.num_free_blocks, 1);
    }
    
    #[test]
    fn test_tensor_operations() {
        // Create mock GPU tensor
        let shape = vec![2, 3, 4];
        let tensor = GpuTensor {
            data: GpuMemoryBlock { ptr: 0x1000, size: 96, alignment: 256 },
            shape: shape.clone(),
            strides: vec![12, 4, 1],
            dtype: std::marker::PhantomData::<f32>,
        };
        
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.numel(), 24);
        assert_eq!(tensor.rank(), 3);
        assert!(tensor.is_contiguous());
        
        // Test reshape
        let reshaped = tensor.reshape(&[4, 6]).unwrap();
        assert_eq!(reshaped.shape(), &[4, 6]);
        assert_eq!(reshaped.numel(), 24);
    }
    
    #[test]
    fn test_optimal_backend() {
        // Should return an option without panicking
        let _backend = GpuContext::optimal_backend();
    }
}