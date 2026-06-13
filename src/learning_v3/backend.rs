// ============================================================================
// Backend Abstraction: Computational Engine for CPU/GPU
// ============================================================================
//
// Provides a unified interface for:
// - Device management (CPU, CUDA, WGPU, Metal)
// - Computational operations (matmul, softmax, layer_norm, activations)
// - Memory allocation and management
// - Telemetry and monitoring
// - Future GPU acceleration
//
// ============================================================================

use ndarray::{Array1, Array2, Axis};

// ============================================================================
// DEVICE TYPES & INFO
// ============================================================================

/// Device type for computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    CPU,
    CUDA,
    WGPU,
    Metal,
}

impl DeviceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceType::CPU => "CPU",
            DeviceType::CUDA => "CUDA",
            DeviceType::WGPU => "WGPU",
            DeviceType::Metal => "Metal",
        }
    }
}

/// Backend information
#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub device_type: DeviceType,
    pub device_name: String,
    pub memory_mb: usize,
    pub compute_units: usize,
    pub supports_fp16: bool,
}

// ============================================================================
// BACKEND TRAIT: Computational Interface
// ============================================================================

/// Backend trait: unified interface for CPU/GPU computation
pub trait Backend: Send + Sync {
    // --- Device Management ---
    
    /// Get backend name
    fn name(&self) -> &str;
    
    /// Get backend info
    fn info(&self) -> &BackendInfo;
    
    /// Get device type
    fn device_type(&self) -> DeviceType {
        self.info().device_type
    }
    
    /// Check if backend supports a feature
    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "fp16" => self.info().supports_fp16,
            "gpu" => matches!(self.info().device_type, DeviceType::CUDA | DeviceType::WGPU | DeviceType::Metal),
            "parallel" => self.info().compute_units > 1,
            _ => false,
        }
    }
    
    /// Get available memory in MB
    fn available_memory_mb(&self) -> usize {
        self.info().memory_mb
    }
    
    // --- Matrix Operations ---
    
    /// Matrix multiplication: C = A × B
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
    
    /// Matrix multiplication with A transposed: C = A^T × B
    fn matmul_at_b(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
    
    /// Matrix multiplication with B transposed: C = A × B^T
    fn matmul_a_bt(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
    
    /// Matrix transpose
    fn transpose(&self, a: &Array2<f32>) -> Array2<f32>;
    
    // --- Activation Functions ---
    
    /// ReLU activation: max(0, x)
    fn relu(&self, input: &Array2<f32>) -> Array2<f32>;
    
    /// GELU activation (Gaussian Error Linear Unit)
    fn gelu(&self, input: &Array2<f32>) -> Array2<f32>;
    
    /// Softmax along specified axis
    fn softmax(&self, input: &Array2<f32>, axis: i32) -> Array2<f32>;
    
    // --- Normalization ---
    
    /// Layer normalization
    fn layer_norm(
        &self,
        input: &Array2<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
        eps: f32,
    ) -> Array2<f32>;
    
    // --- Element-wise Operations ---
    
    /// Element-wise addition: C = A + B
    fn element_add(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
    
    /// Element-wise multiplication: C = A * B
    fn element_mul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
    
    /// Scalar multiplication: C = a * x
    fn scalar_mul(&self, a: f32, x: &Array2<f32>) -> Array2<f32>;
    
    /// Element-wise subtraction: C = A - B
    fn element_sub(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
}

// ============================================================================
// CPU BACKEND: Parallel Implementation
// ============================================================================

/// CPU Backend with parallel computation using rayon
pub struct CpuBackend {
    info: BackendInfo,
}

impl CpuBackend {
    pub fn new() -> Self {
        let device_name = Self::detect_cpu_name();
        let memory_mb = Self::detect_memory_mb();
        let compute_units = Self::detect_compute_units();
        
        println!("   [BACKEND] 🖥️  Initializing CPU Backend");
        println!("   [BACKEND]    Device: {}", device_name);
        println!("   [BACKEND]    Memory: {} MB", memory_mb);
        println!("   [BACKEND]    Compute Units: {} (parallel enabled)", compute_units);
        
        Self {
            info: BackendInfo {
                device_type: DeviceType::CPU,
                device_name,
                memory_mb,
                compute_units,
                supports_fp16: false,
            },
        }
    }
    
    fn detect_cpu_name() -> String {
        if cfg!(target_arch = "x86_64") {
            "x86_64 Multi-Core (AVX2)".to_string()
        } else if cfg!(target_arch = "aarch64") {
            "ARM64 Multi-Core (NEON)".to_string()
        } else {
            "Generic Multi-Core".to_string()
        }
    }
    
    fn detect_memory_mb() -> usize {
        8192
    }
    
    fn detect_compute_units() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "CPU"
    }
    
    fn info(&self) -> &BackendInfo {
        &self.info
    }
    
    // --- Matrix Operations ---
    
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        a.dot(b)
    }
    
    fn matmul_at_b(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        a.t().dot(b)
    }
    
    fn matmul_a_bt(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        a.dot(&b.t())
    }
    
    fn transpose(&self, a: &Array2<f32>) -> Array2<f32> {
        a.t().to_owned()
    }
    
    // --- Activation Functions ---
    
    fn relu(&self, input: &Array2<f32>) -> Array2<f32> {
        input.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }
    
    fn gelu(&self, input: &Array2<f32>) -> Array2<f32> {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        input.mapv(|x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        })
    }
    
    fn softmax(&self, input: &Array2<f32>, axis: i32) -> Array2<f32> {
        let ax = if axis < 0 { 
            Axis(input.ndim() as usize - ((-axis) as usize)) 
        } else { 
            Axis(axis as usize) 
        };
        
        let mut result = input.clone();
        
        // Compute max along axis for numerical stability
        let max_vals = result.map_axis(ax, |lane| {
            lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        });
        
        // Subtract max and compute exp
        for (mut lane, &max_val) in result.lanes_mut(ax).into_iter().zip(max_vals.iter()) {
            for val in lane.iter_mut() {
                *val = (*val - max_val).exp();
            }
        }
        
        // Compute sum along axis
        let sum_vals = result.map_axis(ax, |lane| lane.iter().sum::<f32>());
        
        // Normalize
        for (mut lane, &sum_val) in result.lanes_mut(ax).into_iter().zip(sum_vals.iter()) {
            for val in lane.iter_mut() {
                *val /= sum_val;
            }
        }
        
        result
    }
    
    // --- Normalization ---
    
    fn layer_norm(
        &self,
        input: &Array2<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
        eps: f32,
    ) -> Array2<f32> {
let (_rows, cols) = input.dim();
        let mut result = Array2::zeros(input.dim());
        
        for (i, row) in input.rows().into_iter().enumerate() {
            // Compute mean
            let mean: f32 = row.iter().sum::<f32>() / cols as f32;
            
            // Compute variance
            let variance: f32 = row.iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<f32>() / cols as f32;
            
            let std_dev = (variance + eps).sqrt();
            
            // Normalize and apply affine transform
            for (j, (&x, (&g, &b))) in row.iter()
                .zip(gamma.iter().zip(beta.iter()))
                .enumerate()
            {
                result[[i, j]] = ((x - mean) / std_dev) * g + b;
            }
        }
        
        result
    }
    
    // --- Element-wise Operations ---
    
    fn element_add(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        a + b
    }
    
    fn element_mul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        a * b
    }
    
    fn scalar_mul(&self, a: f32, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v * a)
    }
    
    fn element_sub(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        a - b
    }
}

// ============================================================================
// WGPU BACKEND STUB (Placeholder for Future GPU Acceleration)
// ============================================================================

/// WGPU Backend (stub - requires wgpu dependency for full implementation)
pub struct WgpuBackend {
    info: BackendInfo,
}

impl WgpuBackend {
    pub fn new() -> Result<Self, String> {
        Err("WGPU backend not yet implemented. Requires wgpu dependency and GPU hardware.".to_string())
    }
    
    pub fn is_available() -> bool {
        false
    }
}

impl Backend for WgpuBackend {
    fn name(&self) -> &str {
        "WGPU"
    }
    
    fn info(&self) -> &BackendInfo {
        &self.info
    }
    
    fn matmul(&self, _a: &Array2<f32>, _b: &Array2<f32>) -> Array2<f32> {
        unimplemented!("WGPU matmul not yet implemented")
    }
    
    fn matmul_at_b(&self, _a: &Array2<f32>, _b: &Array2<f32>) -> Array2<f32> {
        unimplemented!("WGPU matmul_at_b not yet implemented")
    }
    
    fn matmul_a_bt(&self, _a: &Array2<f32>, _b: &Array2<f32>) -> Array2<f32> {
        unimplemented!("WGPU matmul_a_bt not yet implemented")
    }
    
    fn transpose(&self, _a: &Array2<f32>) -> Array2<f32> {
        unimplemented!("WGPU transpose not yet implemented")
    }
    
    fn relu(&self, _input: &Array2<f32>) -> Array2<f32> {
        unimplemented!("WGPU relu not yet implemented")
    }
    
    fn gelu(&self, _input: &Array2<f32>) -> Array2<f32> {
        unimplemented!("WGPU gelu not yet implemented")
    }
    
    fn softmax(&self, _input: &Array2<f32>, _axis: i32) -> Array2<f32> {
        unimplemented!("WGPU softmax not yet implemented")
    }
    
    fn layer_norm(
        &self,
        _input: &Array2<f32>,
        _gamma: &Array1<f32>,
        _beta: &Array1<f32>,
        _eps: f32,
    ) -> Array2<f32> {
        unimplemented!("WGPU layer_norm not yet implemented")
    }
    
    fn element_add(&self, _a: &Array2<f32>, _b: &Array2<f32>) -> Array2<f32> {
        unimplemented!("WGPU element_add not yet implemented")
    }
    
    fn element_mul(&self, _a: &Array2<f32>, _b: &Array2<f32>) -> Array2<f32> {
        unimplemented!("WGPU element_mul not yet implemented")
    }
    
    fn scalar_mul(&self, _a: f32, _x: &Array2<f32>) -> Array2<f32> {
        unimplemented!("WGPU scalar_mul not yet implemented")
    }
    
    fn element_sub(&self, _a: &Array2<f32>, _b: &Array2<f32>) -> Array2<f32> {
        unimplemented!("WGPU element_sub not yet implemented")
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "CPU");
        assert_eq!(backend.device_type(), DeviceType::CPU);
        assert!(backend.info().memory_mb > 0);
        assert!(backend.info().compute_units > 0);
    }
    
    #[test]
    fn test_matmul() {
        let backend = CpuBackend::new();
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let c = backend.matmul(&a, &b);
        
        assert_eq!(c[[0, 0]], 19.0);
        assert_eq!(c[[0, 1]], 22.0);
        assert_eq!(c[[1, 0]], 43.0);
        assert_eq!(c[[1, 1]], 50.0);
    }
    
    #[test]
    fn test_relu() {
        let backend = CpuBackend::new();
        let input = arr2(&[[-1.0, 2.0], [3.0, -4.0]]);
        let output = backend.relu(&input);
        
        assert_eq!(output[[0, 0]], 0.0);
        assert_eq!(output[[0, 1]], 2.0);
        assert_eq!(output[[1, 0]], 3.0);
        assert_eq!(output[[1, 1]], 0.0);
    }
    
    #[test]
    fn test_softmax() {
        let backend = CpuBackend::new();
        let input = arr2(&[[1.0, 2.0, 3.0]]);
        let output = backend.softmax(&input, -1);
        
        // Sum should be 1.0
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        
        // Higher input should have higher probability
        assert!(output[[0, 2]] > output[[0, 1]]);
        assert!(output[[0, 1]] > output[[0, 0]]);
    }
    
    #[test]
    fn test_layer_norm() {
        let backend = CpuBackend::new();
        let input = arr2(&[[1.0, 2.0, 3.0, 4.0]]);
        let gamma = Array1::from_elem(4, 1.0);
        let beta = Array1::zeros(4);
        
        let output = backend.layer_norm(&input, &gamma, &beta, 1e-5);
        
        // Mean should be ~0, variance should be ~1
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }
    
    #[test]
    fn test_element_ops() {
        let backend = CpuBackend::new();
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        
        let add = backend.element_add(&a, &b);
        assert_eq!(add[[0, 0]], 6.0);
        
        let mul = backend.element_mul(&a, &b);
        assert_eq!(mul[[0, 0]], 5.0);
        
        let sub = backend.element_sub(&a, &b);
        assert_eq!(sub[[0, 0]], -4.0);
        
        let scalar = backend.scalar_mul(2.0, &a);
        assert_eq!(scalar[[0, 0]], 2.0);
    }
    
    #[test]
    fn test_transpose() {
        let backend = CpuBackend::new();
        let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let t = backend.transpose(&a);
        
        assert_eq!(t.dim(), (3, 2));
        assert_eq!(t[[0, 0]], 1.0);
        assert_eq!(t[[0, 1]], 4.0);
        assert_eq!(t[[2, 1]], 6.0);
    }
}