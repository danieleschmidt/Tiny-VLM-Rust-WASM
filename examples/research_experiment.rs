//! Research Experiment Example
//!
//! Demonstrates how to use the research framework for comparative algorithm studies.

use tiny_vlm::prelude::*;
use tiny_vlm::research::{BaselineVLMAlgorithm, FastVLMAlgorithm};

fn main() -> Result<()> {
    println!("🔬 Starting VLM Research Experiment");

    // Create research framework
    let mut framework = ResearchFramework::new();

    // Register algorithms for comparison
    let baseline_alg = Box::new(BaselineVLMAlgorithm::new());
    let fast_alg = Box::new(FastVLMAlgorithm::new());
    
    framework.register_algorithm(baseline_alg);
    framework.register_algorithm(fast_alg);

    // Create synthetic dataset for testing
    let dataset = create_synthetic_dataset(10)?;
    framework.register_dataset("synthetic_vqa".to_string(), dataset);

    // Configure experiment
    let experiment_config = ExperimentConfig {
        name: "VLM Performance Comparison".to_string(),
        description: "Comparative study of baseline vs optimized VLM algorithms".to_string(),
        baseline_algorithms: vec!["baseline_vlm".to_string()],
        novel_algorithms: vec!["fast_vlm".to_string()],
        datasets: vec!["synthetic_vqa".to_string()],
        metrics: vec!["accuracy".to_string(), "latency".to_string(), "memory".to_string()],
        significance_threshold: 0.05,
        min_runs: 5,
        max_runs: 10,
    };

    // Run experiment
    println!("📊 Running comparative experiment...");
    let results = framework.run_experiment(experiment_config)?;

    // Print results
    print_experiment_results(&results);

    // Export results in different formats
    #[cfg(feature = "std")]
    {
        let markdown_export = framework.export_results(&results, ExportFormat::Markdown)?;
        println!("\n📄 Markdown Export:\n{}", markdown_export);
        
        let json_export = framework.export_results(&results, ExportFormat::Json)?;
        std::fs::write("experiment_results.json", json_export)?;
        println!("💾 Results saved to experiment_results.json");
    }

    println!("✅ Research experiment completed successfully!");
    Ok(())
}

fn create_synthetic_dataset(size: usize) -> Result<Vec<Tensor>> {
    let mut dataset = Vec::new();
    
    for i in 0..size {
        // Create synthetic image data (224x224x3)
        let shape = TensorShape::new(vec![1, 224, 224, 3]);
        let mut tensor = Tensor::zeros(&shape);
        
        // Fill with synthetic data
        let data = tensor.data_mut();
        for j in 0..data.len() {
            data[j] = ((i * 1000 + j) as f32 * 0.001) % 1.0;
        }
        
        dataset.push(tensor);
    }
    
    Ok(dataset)
}

fn print_experiment_results(results: &ExperimentResults) {
    println!("\n🎯 Experiment Results: {}", results.config.name);
    println!("📝 Description: {}", results.config.description);
    println!("🔒 Reproducibility Hash: {}", results.reproducibility_hash);

    println!("\n📈 Statistical Analysis:");
    for (metric, analysis) in &results.statistical_analysis {
        println!("  {} - Mean: {:.3}, Std Dev: {:.3}, CI: [{:.3}, {:.3}]", 
            metric, analysis.mean, analysis.std_dev, 
            analysis.confidence_interval.0, analysis.confidence_interval.1);
    }

    println!("\n🔄 Comparative Analysis:");
    for (comparison, improvement) in &results.comparative_analysis {
        if *improvement > 0.0 {
            println!("  ✅ {}: {:.2}% improvement", comparison, improvement);
        } else {
            println!("  ❌ {}: {:.2}% regression", comparison, improvement.abs());
        }
    }

    println!("\n💡 Key Findings:");
    for (i, conclusion) in results.conclusions.iter().enumerate() {
        println!("  {}. {}", i + 1, conclusion);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_dataset_creation() {
        let dataset = create_synthetic_dataset(5).unwrap();
        assert_eq!(dataset.len(), 5);
        
        for tensor in &dataset {
            assert_eq!(tensor.shape().dims(), &[1, 224, 224, 3]);
        }
    }

    #[test]
    fn test_research_framework_setup() {
        let mut framework = ResearchFramework::new();
        
        let baseline_alg = Box::new(BaselineVLMAlgorithm::new());
        let fast_alg = Box::new(FastVLMAlgorithm::new());
        
        framework.register_algorithm(baseline_alg);
        framework.register_algorithm(fast_alg);
        
        let dataset = create_synthetic_dataset(2).unwrap();
        framework.register_dataset("test".to_string(), dataset);
        
        // Should be able to create experiment config
        let config = ExperimentConfig::default();
        assert_eq!(config.min_runs, 3);
    }
}