//! Memory management for efficient tensor operations

use crate::{Result, TinyVlmError};
use bytemuck::{Pod, Zeroable};

/// Shape descriptor for multi-dimensional tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    /// Dimensions of the tensor
    pub dims: [usize; 4],
    /// Number of valid dimensions
    pub ndim: usize,
}

impl TensorShape {
    /// Create a new tensor shape
    pub fn new(dims: &[usize]) -> Result<Self> {
        if dims.is_empty() || dims.len() > 4 {
            return Err(TinyVlmError::invalid_input(
                "Tensor shape must have 1-4 dimensions",
            ));
        }

        let mut shape_dims = [1usize; 4];
        for (i, &dim) in dims.iter().enumerate() {
            if dim == 0 {
                return Err(TinyVlmError::invalid_input("Tensor dimensions must be > 0"));
            }
            shape_dims[i] = dim;
        }

        Ok(Self {
            dims: shape_dims,
            ndim: dims.len(),
        })
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.dims[..self.ndim].iter().product()
    }

    /// Get strides for the tensor (row-major order)
    pub fn strides(&self) -> [usize; 4] {
        let mut strides = [1usize; 4];
        for i in (0..self.ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Check if shapes are compatible for broadcasting
    pub fn can_broadcast(&self, other: &TensorShape) -> bool {
        for i in 0..self.ndim.max(other.ndim) {
            let dim1 = if i < self.ndim { self.dims[i] } else { 1 };
            let dim2 = if i < other.ndim { other.dims[i] } else { 1 };
            
            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }
        true
    }
}

/// Generic tensor for efficient data storage and manipulation
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    /// Raw data storage
    data: Vec<T>,
    /// Tensor shape
    shape: TensorShape,
}

impl<T: Pod + Zeroable + Copy> Tensor<T> {
    /// Create a new tensor with given shape, filled with zeros
    pub fn zeros(shape: TensorShape) -> Result<Self> {
        let numel = shape.numel();
        let data = vec![T::zeroed(); numel];
        
        Ok(Self { data, shape })
    }

    /// Create a new tensor from raw data and shape
    pub fn from_data(data: Vec<T>, shape: TensorShape) -> Result<Self> {
        if data.len() != shape.numel() {
            return Err(TinyVlmError::invalid_input(
                "Data length does not match tensor shape",
            ));
        }
        
        Ok(Self { data, shape })
    }

    /// Get tensor shape
    pub fn shape(&self) -> TensorShape {
        self.shape
    }

    /// Get mutable reference to raw data
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get reference to raw data
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Reshape tensor (must preserve total number of elements)
    pub fn reshape(&mut self, new_shape: TensorShape) -> Result<()> {
        if new_shape.numel() != self.shape.numel() {
            return Err(TinyVlmError::invalid_input(
                "New shape must have same number of elements",
            ));
        }
        
        self.shape = new_shape;
        Ok(())
    }

    /// Get element at given indices
    pub fn get(&self, indices: &[usize]) -> Result<T> {
        if indices.len() != self.shape.ndim {
            return Err(TinyVlmError::invalid_input("Invalid number of indices"));
        }

        let mut offset = 0;
        let strides = self.shape.strides();
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape.dims[i] {
                return Err(TinyVlmError::invalid_input("Index out of bounds"));
            }
            offset += idx * strides[i];
        }

        Ok(self.data[offset])
    }

    /// Set element at given indices
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        if indices.len() != self.shape.ndim {
            return Err(TinyVlmError::invalid_input("Invalid number of indices"));
        }

        let mut offset = 0;
        let strides = self.shape.strides();
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape.dims[i] {
                return Err(TinyVlmError::invalid_input("Index out of bounds"));
            }
            offset += idx * strides[i];
        }

        self.data[offset] = value;
        Ok(())
    }
}

/// Memory pool for efficient tensor allocation and reuse
pub struct MemoryPool<T> {
    /// Available tensor buffers
    available: Vec<Vec<T>>,
    /// Currently allocated tensors
    allocated: Vec<Vec<T>>,
    /// Total memory allocated (in elements)
    total_memory: usize,
    /// Maximum memory limit (in elements)
    max_memory: usize,
}

impl<T: Pod + Zeroable + Copy> MemoryPool<T> {
    /// Create a new memory pool with given maximum memory limit
    pub fn new(max_memory: usize) -> Self {
        Self {
            available: Vec::new(),
            allocated: Vec::new(),
            total_memory: 0,
            max_memory,
        }
    }

    /// Allocate a tensor from the pool
    pub fn allocate(&mut self, shape: TensorShape) -> Result<Tensor<T>> {
        let required_size = shape.numel();
        
        // Try to find a suitable buffer from available ones
        for i in 0..self.available.len() {
            if self.available[i].len() >= required_size {
                let mut buffer = self.available.swap_remove(i);
                buffer.resize(required_size, T::zeroed());
                let tensor = Tensor::from_data(buffer, shape)?;
                // Track the buffer pointer for deallocation
                self.allocated.push(tensor.data.clone());
                return Ok(tensor);
            }
        }

        // Check memory limit before allocating new buffer
        if self.total_memory + required_size > self.max_memory {
            return Err(TinyVlmError::memory(
                "Memory pool limit exceeded",
            ));
        }

        // Allocate new buffer
        let buffer = vec![T::zeroed(); required_size];
        self.total_memory += required_size;
        let tensor = Tensor::from_data(buffer, shape)?;
        // Track the buffer pointer for deallocation
        self.allocated.push(tensor.data.clone());
        
        Ok(tensor)
    }

    /// Return a tensor to the pool for reuse
    pub fn deallocate(&mut self, tensor: Tensor<T>) {
        let buffer = tensor.data;
        let buffer_len = buffer.len();
        
        // Remove from allocated list - find by size since pointer may not match due to cloning
        if let Some(pos) = self.allocated.iter().position(|x| x.len() == buffer_len) {
            self.allocated.swap_remove(pos);
        }
        
        // Add to available list
        self.available.push(buffer);
    }

    /// Get current memory usage statistics
    pub fn memory_usage(&self) -> MemoryStats {
        let allocated_memory: usize = self.allocated.iter().map(|buf| buf.len()).sum();
        let available_memory: usize = self.available.iter().map(|buf| buf.len()).sum();
        
        MemoryStats {
            total_memory: self.total_memory,
            allocated_memory,
            available_memory,
            max_memory: self.max_memory,
            fragmentation: if self.total_memory > 0 {
                (available_memory as f32) / (self.total_memory as f32)
            } else {
                0.0
            },
        }
    }

    /// Clear all available buffers to reduce memory usage
    pub fn compact(&mut self) {
        let freed_memory: usize = self.available.iter().map(|buf| buf.len()).sum();
        self.available.clear();
        self.total_memory -= freed_memory;
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory allocated by the pool
    pub total_memory: usize,
    /// Currently allocated memory
    pub allocated_memory: usize,
    /// Available memory in the pool
    pub available_memory: usize,
    /// Maximum memory limit
    pub max_memory: usize,
    /// Fragmentation ratio (available / total)
    pub fragmentation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_creation() {
        let shape = TensorShape::new(&[2, 3, 4]).unwrap();
        assert_eq!(shape.ndim, 3);
        assert_eq!(shape.dims[0], 2);
        assert_eq!(shape.dims[1], 3);
        assert_eq!(shape.dims[2], 4);
        assert_eq!(shape.numel(), 24);
    }

    #[test]
    fn test_tensor_creation() {
        let shape = TensorShape::new(&[2, 3]).unwrap();
        let tensor: Tensor<f32> = Tensor::zeros(shape).unwrap();
        assert_eq!(tensor.data().len(), 6);
        assert_eq!(tensor.shape().numel(), 6);
    }

    #[test]
    fn test_tensor_indexing() {
        let shape = TensorShape::new(&[2, 3]).unwrap();
        let mut tensor: Tensor<f32> = Tensor::zeros(shape).unwrap();
        
        tensor.set(&[1, 2], 5.0).unwrap();
        assert_eq!(tensor.get(&[1, 2]).unwrap(), 5.0);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool: MemoryPool<f32> = MemoryPool::new(1000);
        let shape = TensorShape::new(&[10, 10]).unwrap();
        
        let tensor = pool.allocate(shape).unwrap();
        assert_eq!(tensor.data().len(), 100);
        
        let stats = pool.memory_usage();
        assert_eq!(stats.allocated_memory, 100);
        
        pool.deallocate(tensor);
        let stats = pool.memory_usage();
        assert_eq!(stats.allocated_memory, 0);
        assert_eq!(stats.available_memory, 100);
    }

    #[test]
    fn test_broadcasting_compatibility() {
        let shape1 = TensorShape::new(&[3, 1, 4]).unwrap();
        let shape2 = TensorShape::new(&[1, 2, 4]).unwrap();
        assert!(shape1.can_broadcast(&shape2));
        
        let shape3 = TensorShape::new(&[3, 2, 4]).unwrap();
        let shape4 = TensorShape::new(&[3, 3, 4]).unwrap();
        assert!(!shape3.can_broadcast(&shape4));
    }
}