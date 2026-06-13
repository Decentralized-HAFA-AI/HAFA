// ============================================================================
// Accelerated Operations: Backend-Optimized Computations
// ============================================================================
//
// Provides accelerated versions of critical operations using Backend trait.
// Used for benchmarking and performance optimization.
//
// ============================================================================

use std::time::Instant;
use ndarray::{Array2, Array1, Axis};

use super::backend::Backend;
/// Results of a benchmark comparison
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkResult {
    pub operation: String,
    pub direct_time_ms: f64,
    pub backend_time_ms: f64,
    pub speedup: f64,
    pub matrix_size: (usize, usize),
}

pub struct AcceleratedOps<'a> {
    backend: &'a dyn Backend,
}

impl<'a> AcceleratedOps<'a> {
    pub fn new(backend: &'a dyn Backend) -> Self {
        Self { backend }
    }
    
    /// Benchmark matmul: direct ndarray vs backend
    pub fn benchmark_matmul(&self, size: usize, iterations: usize) -> BenchmarkResult {
        let a = Array2::from_elem((size, size), 0.5);
        let b = Array2::from_elem((size, size), 0.5);
        
        // Warmup
        for _ in 0..3 {
            let _ = a.dot(&b);
            let _ = self.backend.matmul(&a, &b);
        }
        
        // Benchmark direct
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = a.dot(&b);
        }
        let direct_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        
        // Benchmark backend
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = self.backend.matmul(&a, &b);
        }
        let backend_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        
        BenchmarkResult {
            operation: "matmul".to_string(),
            direct_time_ms: direct_time,
            backend_time_ms: backend_time,
            speedup: direct_time / backend_time,
            matrix_size: (size, size),
        }
    }
    
    /// Benchmark softmax
    pub fn benchmark_softmax(&self, rows: usize, cols: usize, iterations: usize) -> BenchmarkResult {
        let input = Array2::from_elem((rows, cols), 1.0);
        
        // Warmup
        for _ in 0..3 {
            let _ = self.softmax_direct(&input);
            let _ = self.backend.softmax(&input, -1);
        }
        
        // Benchmark direct
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = self.softmax_direct(&input);
        }
        let direct_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        
        // Benchmark backend
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = self.backend.softmax(&input, -1);
        }
        let backend_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        
        BenchmarkResult {
            operation: "softmax".to_string(),
            direct_time_ms: direct_time,
            backend_time_ms: backend_time,
            speedup: direct_time / backend_time,
            matrix_size: (rows, cols),
        }
    }
    
    /// Benchmark layer_norm
    pub fn benchmark_layer_norm(&self, rows: usize, cols: usize, iterations: usize) -> BenchmarkResult {
        let input = Array2::from_elem((rows, cols), 1.0);
        let gamma = Array1::from_elem(cols, 1.0);
        let beta = Array1::zeros(cols);
        
        // Warmup
        for _ in 0..3 {
            let _ = self.layer_norm_direct(&input, &gamma, &beta);
            let _ = self.backend.layer_norm(&input, &gamma, &beta, 1e-5);
        }
        
        // Benchmark direct
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = self.layer_norm_direct(&input, &gamma, &beta);
        }
        let direct_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        
        // Benchmark backend
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = self.backend.layer_norm(&input, &gamma, &beta, 1e-5);
        }
        let backend_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        
        BenchmarkResult {
            operation: "layer_norm".to_string(),
            direct_time_ms: direct_time,
            backend_time_ms: backend_time,
            speedup: direct_time / backend_time,
            matrix_size: (rows, cols),
        }
    }
    
    /// Run full benchmark suite
    pub fn run_full_benchmark(&self) -> Vec<BenchmarkResult> {
        println!("\n🔬 Running Backend Benchmarks...");
        println!("   Backend: {}", self.backend.name());
        println!("   Device: {}", self.backend.info().device_name);
        println!("   Compute Units: {}\n", self.backend.info().compute_units);
        
        let mut results = Vec::new();
        
        // Small matrices (typical for attention)
        results.push(self.benchmark_matmul(64, 100));
        results.push(self.benchmark_matmul(128, 50));
        
        // Softmax (attention scores)
        results.push(self.benchmark_softmax(32, 64, 100));
        results.push(self.benchmark_softmax(64, 128, 50));
        
        // LayerNorm
        results.push(self.benchmark_layer_norm(32, 128, 100));
        results.push(self.benchmark_layer_norm(64, 256, 50));
        
        // Print results
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│  Operation        │ Direct (ms) │ Backend (ms) │ Speedup    │");
        println!("├─────────────────────────────────────────────────────────────┤");
        
        for result in &results {
            println!(
                "│  {:15} │ {:11.3} │ {:12.3} │ {:8.2}x   │",
                format!("{} {}x{}", result.operation, result.matrix_size.0, result.matrix_size.1),
                result.direct_time_ms,
                result.backend_time_ms,
                result.speedup
            );
        }
        
        println!("└─────────────────────────────────────────────────────────────┘\n");
        
        results
    }
    
    /// Direct softmax implementation (for comparison)
    fn softmax_direct(&self, input: &Array2<f32>) -> Array2<f32> {
        let ax = Axis(1);
        let mut result = input.clone();
        
        let max_vals = result.map_axis(ax, |lane| {
            lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        });
        
        for (mut lane, &max_val) in result.lanes_mut(ax).into_iter().zip(max_vals.iter()) {
            for val in lane.iter_mut() {
                *val = (*val - max_val).exp();
            }
        }
        
        let sum_vals = result.map_axis(ax, |lane| lane.iter().sum::<f32>());
        
        for (mut lane, &sum_val) in result.lanes_mut(ax).into_iter().zip(sum_vals.iter()) {
            for val in lane.iter_mut() {
                *val /= sum_val;
            }
        }
        
        result
    }
    
    /// Direct LayerNorm implementation (for comparison)
    fn layer_norm_direct(
        &self,
        input: &Array2<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
    ) -> Array2<f32> {
        let (_rows, cols) = input.dim();
        let eps = 1e-5;
        let mut result = Array2::zeros(input.dim());
        
        for (i, row) in input.rows().into_iter().enumerate() {
            let mean: f32 = row.iter().sum::<f32>() / cols as f32;
            let variance: f32 = row.iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<f32>() / cols as f32;
            let std_dev = (variance + eps).sqrt();
            
            for (j, (&x, (&g, &b))) in row.iter()
                .zip(gamma.iter().zip(beta.iter()))
                .enumerate()
            {
                result[[i, j]] = ((x - mean) / std_dev) * g + b;
            }
        }
        
        result
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_v3::backend::CpuBackend;
    
    #[test]
    fn test_benchmark_matmul() {
        let backend = CpuBackend::new();
        let ops = AcceleratedOps::new(&backend);
        let result = ops.benchmark_matmul(32, 10);
        assert!(result.direct_time_ms > 0.0);
        assert!(result.backend_time_ms > 0.0);
        assert!(result.speedup > 0.0);
    }
    
    #[test]
    fn test_benchmark_softmax() {
        let backend = CpuBackend::new();
        let ops = AcceleratedOps::new(&backend);
        let result = ops.benchmark_softmax(16, 32, 10);
        assert!(result.direct_time_ms > 0.0);
        assert!(result.backend_time_ms > 0.0);
    }
    
    #[test]
    fn test_benchmark_layer_norm() {
        let backend = CpuBackend::new();
        let ops = AcceleratedOps::new(&backend);
        let result = ops.benchmark_layer_norm(16, 32, 10);
        assert!(result.direct_time_ms > 0.0);
        assert!(result.backend_time_ms > 0.0);
    }
}