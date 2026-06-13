// ============================================================================
// AdamW Optimizer with L2 Norm Gradient Clipping
// ============================================================================
//
// Production-grade AdamW implementation with:
// - Decoupled weight decay
// - Bias correction
// - L2 norm gradient clipping
// - Configurable hyperparameters
//
// ============================================================================

use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub t: usize,
}

impl AdamW {
    /// Creates AdamW with default hyperparameters
    /// Default: lr=0.001, weight_decay=0.001, max_grad_norm=5.0
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.001,        // Optimized: 10x smaller than before
            max_grad_norm: 5.0,         // Optimized: less aggressive clipping
            t: 0,
        }
    }

    /// Creates AdamW with full parameter control
    /// Use this for production training with custom hyperparameters
    pub fn new_with_params(
        learning_rate: f32,
        weight_decay: f32,
        max_grad_norm: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            max_grad_norm,
            t: 0,
        }
    }

    /// Applies L2 Norm Gradient Clipping
    /// If gradient norm exceeds threshold, scales it down proportionally
    pub fn clip_gradients(&self, gradients: &Array2<f32>) -> Array2<f32> {
        // Compute L2 norm: sqrt(sum(g^2))
        let grad_norm: f32 = gradients.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        // If norm exceeds threshold, scale down
        if grad_norm > self.max_grad_norm {
            let scale = self.max_grad_norm / grad_norm;
            gradients * scale
        } else {
            gradients.clone()
        }
    }

    /// Performs one optimization step with AdamW
    /// 
    /// Parameters:
    /// - weights: model weights to update (modified in-place)
    /// - gradients: computed gradients
    /// - m: first moment buffer (momentum)
    /// - v: second moment buffer (velocity)
    pub fn step(
        &mut self,
        weights: &mut Array2<f32>,
        gradients: &Array2<f32>,
        m: &mut Array2<f32>,
        v: &mut Array2<f32>,
    ) {
        self.t += 1;

        // Apply L2 Norm Gradient Clipping
        let clipped_grads = self.clip_gradients(gradients);

       // Update biased first moment estimate: m_t = β₁·m_{t-1} + (1-β₁)·g_t
*m = &*m * self.beta1 + &clipped_grads * (1.0 - self.beta1);

// Update biased second raw moment estimate: v_t = β₂·v_{t-1} + (1-β₂)·g_t²
*v = &*v * self.beta2 + &(&clipped_grads * &clipped_grads) * (1.0 - self.beta2);
        // Compute bias-corrected estimates
        let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));

        // Update weights with adaptive learning rate
        let update = &m_hat / &(&v_hat.mapv(|v| v.sqrt()) + self.epsilon);
        *weights -= &(&update * self.learning_rate);

        // Apply decoupled weight decay
        *weights *= 1.0 - (self.learning_rate * self.weight_decay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_adamw_step() {
        let mut optimizer = AdamW::new(0.001);
        let mut weights = Array2::zeros((3, 3));
        let gradients = Array2::ones((3, 3)) * 0.1;
        let mut m = Array2::zeros((3, 3));
        let mut v = Array2::zeros((3, 3));

        optimizer.step(&mut weights, &gradients, &mut m, &mut v);

        // Weights should have changed
        assert!(weights.iter().any(|&w| w != 0.0));
    }

    #[test]
    fn test_gradient_clipping() {
        let optimizer = AdamW::new(0.001);
        
        // Create gradient with large norm
        let mut gradients = Array2::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                gradients[[i, j]] = 10.0;
            }
        }
        
        let clipped = optimizer.clip_gradients(&gradients);
        
        // Compute norm of clipped gradients
        let clipped_norm: f32 = clipped.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        // Should be <= max_grad_norm (5.0)
        assert!(clipped_norm <= optimizer.max_grad_norm + 1e-6);
    }

    #[test]
    fn test_custom_params() {
        let optimizer = AdamW::new_with_params(0.0001, 0.001, 5.0);
        assert_eq!(optimizer.learning_rate, 0.0001);
        assert_eq!(optimizer.weight_decay, 0.001);
        assert_eq!(optimizer.max_grad_norm, 5.0);
    }

    #[test]
    fn test_weight_decay_effect() {
        let mut optimizer = AdamW::new_with_params(0.01, 0.1, 5.0);
        let mut weights = Array2::ones((2, 2)) * 1.0;
        let gradients = Array2::zeros((2, 2)); // Zero gradients
        let mut m = Array2::zeros((2, 2));
        let mut v = Array2::zeros((2, 2));

        optimizer.step(&mut weights, &gradients, &mut m, &mut v);

        // Weights should decrease due to weight decay
        assert!(weights.iter().all(|&w| w < 1.0));
    }
}