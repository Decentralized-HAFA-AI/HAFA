// ============================================================================
// Tensor Abstraction Layer
// ============================================================================
//
// Simple tensor operations built on top of ndarray.
// Provides a clean API for transformer operations.
//
// ============================================================================

use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_rand::{RandomExt, rand::distributions::Uniform};
use serde::{Deserialize, Serialize};

// ============================================================================
// TENSOR TYPE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Array3<f32>,  // [batch, seq_len, dim]
}

impl Tensor {
    /// Create a new tensor with given shape
    pub fn new(batch: usize, seq_len: usize, dim: usize) -> Self {
        Self {
            data: Array3::zeros((batch, seq_len, dim)),
        }
    }

    /// Create a random tensor (Xavier initialization)
    pub fn random(batch: usize, seq_len: usize, dim: usize) -> Self {
        let std_dev = (2.0 / (seq_len + dim) as f64).sqrt() as f32;
        let uniform = Uniform::new(-std_dev, std_dev);
        Self {
            data: Array3::random((batch, seq_len, dim), uniform),
        }
    }

    /// Get shape
    pub fn shape(&self) -> (usize, usize, usize) {
        self.data.dim()
    }

    /// Matrix multiplication: [batch, seq, dim] × [dim, out_dim] → [batch, seq, out_dim]
    pub fn matmul(&self, weight: &Array2<f32>) -> Tensor {
        let (batch, seq_len, _dim) = self.shape();
        let out_dim = weight.ncols();
        
        let mut result = Array3::zeros((batch, seq_len, out_dim));
        
        for b in 0..batch {
            for s in 0..seq_len {
                let input_vec = self.data.slice(ndarray::s![b, s, ..]);
                let output_vec = weight.t().dot(&input_vec);
                result.slice_mut(ndarray::s![b, s, ..]).assign(&output_vec);
            }
        }
        
        Tensor { data: result }
    }

    /// Add bias: [batch, seq, dim] + [dim] → [batch, seq, dim]
    pub fn add_bias(&self, bias: &Array1<f32>) -> Tensor {
        let mut result = self.data.clone();
        for mut row in result.outer_iter_mut() {
            for mut vec in row.outer_iter_mut() {
                vec += bias;
            }
        }
        Tensor { data: result }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Tensor {
        Tensor {
            data: &self.data + &other.data,
        }
    }

    /// Apply activation function
    pub fn gelu(&self) -> Tensor {
        Tensor {
            data: self.data.mapv(|x| {
                0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }),
        }
    }

    pub fn relu(&self) -> Tensor {
        Tensor {
            data: self.data.mapv(|x| x.max(0.0)),
        }
    }

    pub fn softmax(&self) -> Tensor {
        let mut result = self.data.clone();
        
        for mut matrix in result.outer_iter_mut() {
            for mut row in matrix.outer_iter_mut() {
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum_exp: f32 = exp_vals.iter().sum();
                
                for (i, val) in row.iter_mut().enumerate() {
                    *val = exp_vals[i] / sum_exp;
                }
            }
        }
        
        Tensor { data: result }
    }

    /// Layer normalization (without learnable parameters)
    pub fn layer_norm(&self, eps: f32) -> Tensor {
        let mut result = self.data.clone();
        
        for mut matrix in result.outer_iter_mut() {
            for mut row in matrix.outer_iter_mut() {
                let mean: f32 = row.iter().sum::<f32>() / row.len() as f32;
                let variance: f32 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32;
                let std_dev = (variance + eps).sqrt();
                
                for val in row.iter_mut() {
                    *val = (*val - mean) / std_dev;
                }
            }
        }
        
        Tensor { data: result }
    }

    /// Layer normalization with learnable affine parameters (gamma, beta)
    /// y = gamma * ((x - mean) / sqrt(var + eps)) + beta
    pub fn layer_norm_affine(
        &self, 
        gamma: &Array1<f32>, 
        beta: &Array1<f32>, 
        eps: f32
    ) -> Tensor {
        let (batch, seq_len, dim) = self.shape();
        let mut result = Array3::zeros((batch, seq_len, dim));
        
        for b in 0..batch {
            for s in 0..seq_len {
                // Compute mean
                let mut mean = 0.0f32;
                for d in 0..dim {
                    mean += self.data[[b, s, d]];
                }
                mean /= dim as f32;
                
                // Compute variance
                let mut variance = 0.0f32;
                for d in 0..dim {
                    let diff = self.data[[b, s, d]] - mean;
                    variance += diff * diff;
                }
                variance /= dim as f32;
                let std_dev = (variance + eps).sqrt();
                
                // Normalize and apply affine transformation
                for d in 0..dim {
                    let x_hat = (self.data[[b, s, d]] - mean) / std_dev;
                    result[[b, s, d]] = gamma[d] * x_hat + beta[d];
                }
            }
        }
        
        Tensor { data: result }
    }

    /// Transpose last two dimensions: [batch, seq, dim] → [batch, dim, seq]
    pub fn transpose(&self) -> Tensor {
        let (batch, seq_len, dim) = self.shape();
        let mut result = Array3::zeros((batch, dim, seq_len));
        
        for b in 0..batch {
            for s in 0..seq_len {
                for d in 0..dim {
                    result[[b, d, s]] = self.data[[b, s, d]];
                }
            }
        }
        
        Tensor { data: result }
    }

    /// Reshape to 2D: [batch, seq, dim] → [batch*seq, dim]
    pub fn flatten(&self) -> Array2<f32> {
        let (batch, seq_len, dim) = self.shape();
        let mut result = Array2::zeros((batch * seq_len, dim));
        
        for b in 0..batch {
            for s in 0..seq_len {
                let idx = b * seq_len + s;
                result.row_mut(idx).assign(&self.data.slice(ndarray::s![b, s, ..]));
            }
        }
        
        result
    }

    /// Get slice along sequence dimension
    pub fn slice_seq(&self, start: usize, end: usize) -> Tensor {
        Tensor {
            data: self.data.slice_axis(Axis(1), ndarray::Slice::from(start..end)).to_owned(),
        }
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(2, 10, 64);
        assert_eq!(tensor.shape(), (2, 10, 64));
    }

    #[test]
    fn test_tensor_matmul() {
        let tensor = Tensor::new(1, 4, 8);
        let weight = Array2::zeros((8, 16));
        let result = tensor.matmul(&weight);
        assert_eq!(result.shape(), (1, 4, 16));
    }

    #[test]
    fn test_tensor_softmax() {
        let mut data = Array3::zeros((1, 1, 3));
        data[[0, 0, 0]] = 1.0;
        data[[0, 0, 1]] = 2.0;
        data[[0, 0, 2]] = 3.0;
        
        let tensor = Tensor { data };
        let result = tensor.softmax();
        
        // Softmax should sum to 1
        let sum: f32 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_affine() {
        let mut data = Array3::zeros((1, 1, 4));
        data[[0, 0, 0]] = 1.0;
        data[[0, 0, 1]] = 2.0;
        data[[0, 0, 2]] = 3.0;
        data[[0, 0, 3]] = 4.0;
        
        let tensor = Tensor { data };
        let gamma = Array1::ones(4);  // gamma = 1
        let beta = Array1::zeros(4);  // beta = 0
        
        let result = tensor.layer_norm_affine(&gamma, &beta, 1e-5f32);

// With gamma=1, beta=0, should be same as regular layer_norm
let mean = 2.5f32;
let variance = 1.25f32;
let std_dev = (variance + 1e-5f32).sqrt();
        
        assert!((result.data[[0, 0, 0]] - (1.0 - mean) / std_dev).abs() < 1e-4);
        assert!((result.data[[0, 0, 1]] - (2.0 - mean) / std_dev).abs() < 1e-4);
        assert!((result.data[[0, 0, 2]] - (3.0 - mean) / std_dev).abs() < 1e-4);
        assert!((result.data[[0, 0, 3]] - (4.0 - mean) / std_dev).abs() < 1e-4);
    }
}