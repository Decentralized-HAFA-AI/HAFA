// ============================================================================
// Loss Functions for Transformer Training
// ============================================================================

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossType {
    CrossEntropy,
    MSE,
}

#[derive(Debug, Clone)]
pub struct LossResult {
    pub loss: f32,
    pub gradient: Vec<f32>,
}

pub struct LossFunction {
    pub loss_type: LossType,
}

impl LossFunction {
    pub fn new(loss_type: LossType) -> Self {
        Self { loss_type }
    }

    /// Compute cross-entropy loss and gradient
    /// logits: [vocab_size] - raw output from model
    /// target: usize - index of correct token
    pub fn compute(&self, logits: &[f32], target: usize) -> LossResult {
        match self.loss_type {
            LossType::CrossEntropy => self.cross_entropy(logits, target),
            LossType::MSE => self.mse(logits, target),
        }
    }

    fn cross_entropy(&self, logits: &[f32], target: usize) -> LossResult {
        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        // Loss = -log(prob[target])
        let prob_target = probs[target].max(1e-10); // Prevent log(0)
        let loss = -prob_target.ln();

        // Gradient: prob - one_hot(target)
        let mut gradient = probs.clone();
        gradient[target] -= 1.0;

        LossResult { loss, gradient }
    }

    fn mse(&self, logits: &[f32], target: usize) -> LossResult {
        let target_vec: Vec<f32> = (0..logits.len())
            .map(|i| if i == target { 1.0 } else { 0.0 })
            .collect();

        let mut loss = 0.0f32;
        let mut gradient = vec![0.0f32; logits.len()];

        for (i, (&pred, &tgt)) in logits.iter().zip(target_vec.iter()).enumerate() {
            let diff = pred - tgt;
            loss += diff * diff;
            gradient[i] = 2.0 * diff;
        }

        loss /= logits.len() as f32;
        gradient.iter_mut().for_each(|g| *g /= logits.len() as f32);

        LossResult { loss, gradient }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_loss() {
        let loss_fn = LossFunction::new(LossType::CrossEntropy);
        let logits = vec![1.0, 2.0, 3.0];
        let target = 2; // Correct answer is index 2

        let result = loss_fn.compute(&logits, target);
        
        // Loss should be small since target has highest logit
        assert!(result.loss < 1.0);
        assert_eq!(result.gradient.len(), 3);
    }
}