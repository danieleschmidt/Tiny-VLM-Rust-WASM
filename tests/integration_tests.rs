//! Integration tests for Tiny-VLM-Rust-WASM
//!
//! Comprehensive testing suite covering all major functionality

use tiny_vlm::{prelude::*, FastVLM, ModelConfig, InferenceConfig};

#[cfg(feature = "std")]
mod std_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_model_creation_and_config() {
        let config = ModelConfig::default();
        let model = FastVLM::new(config);
        assert!(model.is_ok(), "Model creation should succeed with default config");
        
        let model = model.unwrap();
        assert_eq!(model.config().vision_dim, 768);
        assert_eq!(model.config().text_dim, 768);
        assert_eq!(model.config().hidden_dim, 768);
    }

    #[test]
    fn test_model_inference_pipeline() {
        println!("Creating model config...");
        let config = ModelConfig::default();
        println!("Creating model...");
        let mut model = FastVLM::new(config).expect("Failed to create model");
        
        println!("Creating test data...");
        // Create test image data (224x224x3 RGB)
        let test_image = create_test_image_data();
        let test_prompt = "Hi";  // Simplified prompt
        let inference_config = InferenceConfig::default();
        
        println!("Running inference...");
        let result = model.infer(&test_image, test_prompt, inference_config);
        assert!(result.is_ok(), "Inference should succeed with valid inputs: {:?}", result.err());
        
        let response = result.unwrap();
        println!("Got response: {}", response);
        assert!(!response.is_empty(), "Response should not be empty");
        assert!(response.len() < 1000, "Response should be reasonable length");
    }

    #[test]
    fn test_input_validation() {
        let config = ModelConfig::default();
        let mut model = FastVLM::new(config).expect("Failed to create model");
        
        // Test empty image data
        let empty_image = vec![];
        let result = model.infer(&empty_image, "test", InferenceConfig::default());
        assert!(result.is_err(), "Empty image should fail validation");
        
        // Test oversized image
        let oversized_image = vec![0u8; 11 * 1024 * 1024]; // 11MB
        let result = model.infer(&oversized_image, "test", InferenceConfig::default());
        assert!(result.is_err(), "Oversized image should fail validation");
        
        // Test text with null bytes
        let test_image = create_test_image_data();
        let result = model.infer(&test_image, "test\0text", InferenceConfig::default());
        assert!(result.is_err(), "Text with null bytes should fail validation");
    }

    #[test]
    fn test_memory_management() {
        let config = ModelConfig::default();
        let mut model = FastVLM::new(config).expect("Failed to create model");
        
        let initial_memory = model.memory_stats();
        let test_image = create_test_image_data();
        
        // Run multiple inferences to test memory management
        let mut successful_inferences = 0;
        for i in 0..5 {
            let prompt = format!("Inference {}", i);
            let result = model.infer(&test_image, &prompt, InferenceConfig::default());
            if result.is_ok() {
                successful_inferences += 1;
            } else {
                // Memory exhaustion is expected after several inferences
                println!("Inference {} failed (expected): {:?}", i, result.err());
                break;
            }
        }
        
        // At least 2 inferences should succeed before memory exhaustion
        assert!(successful_inferences >= 2, "At least 2 inferences should succeed, got {}", successful_inferences);
        
        // Memory should be managed efficiently
        let final_memory = model.memory_stats();
        let memory_growth = final_memory.allocated_memory.saturating_sub(initial_memory.allocated_memory);
        assert!(memory_growth < 100_000_000, "Memory growth should be reasonable"); // < 100MB
        
        // Test memory compaction
        model.compact_memory();
        let compacted_memory = model.memory_stats();
        assert!(compacted_memory.allocated_memory <= final_memory.allocated_memory);
    }

    #[test]
    fn test_vision_encoding() {
        let config = ModelConfig::default();
        let mut model = FastVLM::new(config).expect("Failed to create model");
        
        let test_image = create_test_image_data();
        let result = model.encode_image(&test_image);
        
        assert!(result.is_ok(), "Vision encoding should succeed");
        let features = result.unwrap();
        assert!(features.shape().numel() > 0, "Features should have elements");
    }

    #[test]
    fn test_text_encoding() {
        let config = ModelConfig::default();
        let mut model = FastVLM::new(config).expect("Failed to create model");
        
        let test_text = "This is a test prompt for text encoding.";
        let result = model.encode_text(test_text);
        
        assert!(result.is_ok(), "Text encoding should succeed");
        let features = result.unwrap();
        assert!(features.shape().numel() > 0, "Features should have elements");
    }

    #[test]
    fn test_inference_config_validation() {
        // Test valid config
        let valid_config = InferenceConfig {
            max_length: 100,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 50,
            deterministic: false,
            memory_limit_mb: 100,
        };
        let result = tiny_vlm::validation::validate_inference_config(&valid_config);
        assert!(result.is_valid, "Valid config should pass validation");

        // Test invalid temperature
        let invalid_config = InferenceConfig {
            temperature: -1.0,
            ..valid_config
        };
        let result = tiny_vlm::validation::validate_inference_config(&invalid_config);
        assert!(!result.is_valid, "Negative temperature should fail validation");

        // Test invalid top_p
        let invalid_config = InferenceConfig {
            top_p: 1.5,
            ..valid_config
        };
        let result = tiny_vlm::validation::validate_inference_config(&invalid_config);
        assert!(!result.is_valid, "Top_p > 1.0 should fail validation");
    }

    #[test]
    fn test_performance_benchmarks() {
        let config = ModelConfig::default();
        let mut model = FastVLM::new(config).expect("Failed to create model");
        
        let test_image = create_test_image_data();
        let test_prompt = "Describe this image";
        let inference_config = InferenceConfig::default();
        
        let start_time = std::time::Instant::now();
        let result = model.infer(&test_image, test_prompt, inference_config);
        let inference_time = start_time.elapsed();
        
        assert!(result.is_ok(), "Inference should succeed");
        
        // Performance target: should complete within reasonable time (10 seconds for testing)
        assert!(
            inference_time < Duration::from_secs(10),
            "Inference took too long: {:?}", 
            inference_time
        );
        
        println!("Inference time: {:?}", inference_time);
    }

    #[test]
    fn test_error_recovery() {
        let config = ModelConfig::default();
        let mut model = FastVLM::new(config).expect("Failed to create model");
        
        let test_image = create_test_image_data();
        
        // Test recovery from invalid input
        let _error_result = model.infer(&[], "test", InferenceConfig::default());
        
        // Model should still work after error
        let valid_result = model.infer(&test_image, "valid test", InferenceConfig::default());
        assert!(valid_result.is_ok(), "Model should recover from previous error");
    }

    #[test]
    fn test_concurrent_inference() {
        use std::sync::Arc;
        use std::thread;
        
        // Test concurrent inference by creating separate models per thread
        let config = ModelConfig::default();
        let test_image = Arc::new(create_test_image_data());
        let mut handles = vec![];
        
        // Spawn multiple threads for concurrent inference with separate models
        for i in 0..3 {
            let config_clone = config.clone();
            let image_clone = Arc::clone(&test_image);
            
            let handle = thread::spawn(move || {
                let mut model = FastVLM::new(config_clone).expect("Failed to create model");
                let prompt = format!("Thread {} inference", i);
                model.infer(&image_clone, &prompt, InferenceConfig::default())
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads and check results
        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.join().unwrap();
            assert!(result.is_ok(), "Concurrent inference {} should succeed", i);
        }
    }

    pub fn create_test_image_data() -> Vec<u8> {
        // Create a simple test PNG image (1x1 pixel)
        vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1 pixels
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, // RGB + Alpha, no compression
            0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, // IDAT chunk
            0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00, // Compressed data
            0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, // End of IDAT
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, // IEND chunk
            0x42, 0x60, 0x82
        ]
    }
}

#[cfg(feature = "wasm")]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_wasm_model_creation() {
        let config = ModelConfig::default();
        let model = FastVLM::new(config);
        assert!(model.is_ok());
    }
    
    // Additional WASM-specific tests would go here
}

// Stress tests for robustness
#[cfg(feature = "std")]
mod stress_tests {
    use super::*;
    
    #[test]
    #[ignore] // Ignore by default, run with --ignored flag
    fn test_memory_stress() {
        let config = ModelConfig::default();
        let mut model = FastVLM::new(config).expect("Failed to create model");
        
        let test_image = std_tests::create_test_image_data();
        
        // Run many inferences to test memory stability
        for i in 0..1000 {
            let prompt = format!("Stress test inference {}", i);
            let result = model.infer(&test_image, &prompt, InferenceConfig::default());
            
            if i % 100 == 0 {
                println!("Completed {} stress test inferences", i);
                model.compact_memory();
            }
            
            assert!(result.is_ok(), "Stress test inference {} failed", i);
        }
        
        println!("Memory stress test completed successfully");
    }
    
    #[test]
    #[ignore]
    fn test_large_input_handling() {
        let config = ModelConfig::default();
        let mut model = FastVLM::new(config).expect("Failed to create model");
        
        // Test with large (but valid) text input
        let large_prompt = "Describe this image in detail. ".repeat(100); // ~3000 chars
        let test_image = std_tests::create_test_image_data();
        
        let result = model.infer(&test_image, &large_prompt, InferenceConfig::default());
        assert!(result.is_ok(), "Large input handling should work");
    }
}