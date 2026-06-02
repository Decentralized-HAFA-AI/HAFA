// ============================================================================
// HAFA - src/learning.rs — COGNITIVE LEARNING ENGINE (FROM SCRATCH)
// ============================================================================

use crate::config::Config;
use crate::data_source::ValidatedData;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use thiserror::Error;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum LearningError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Empty experience buffer")]
    EmptyBuffer,
    #[error("Mathematical computation failed: {0}")]
    MathError(String),
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// A single unit of experience (Input -> Output mapping with confidence)
#[derive(Debug, Clone)]
pub struct Experience {
    pub input: Array1<f64>,
    pub target: Array1<f64>,
    pub weight: f64, // Based on epistemic confidence
}

/// Short-term memory buffer for learning (Sliding Window)
/// Note: Not serialized - ephemeral runtime memory only
#[derive(Debug, Clone)]
pub struct ContextBuffer {
    pub experiences: Vec<Experience>,
    pub max_size: usize,
}

/// The core cognitive model (Simple Neural Network / Perceptron weights)
#[derive(Debug, Clone)]
pub struct CognitiveModel {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub input_size: usize,
    pub output_size: usize,
}

/// The main Learning Engine
pub struct Learner {
    config: Config,
    model: CognitiveModel,
    buffer: ContextBuffer,
}

// ============================================================================
// IMPLEMENTATION
// ============================================================================

impl ContextBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            experiences: Vec::new(),
            max_size,
        }
    }

    pub fn push(&mut self, exp: Experience) {
        if self.experiences.len() >= self.max_size {
            self.experiences.remove(0); // Drop oldest
        }
        self.experiences.push(exp);
    }

    pub fn len(&self) -> usize {
        self.experiences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experiences.is_empty()
    }
}

impl CognitiveModel {
    /// Initialize with random weights
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let uniform = Uniform::new(-0.1, 0.1);
        Self {
            weights: Array2::random((output_size, input_size), uniform),
            biases: Array1::random(output_size, uniform),
            input_size,
            output_size,
        }
    }

    /// Forward pass: Input -> Output
    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = self.weights.dot(input);
        output += &self.biases;
        
        // ReLU Activation
        output.mapv(|x| x.max(0.0))
    }

    /// Update weights using Gradient Descent logic (Explicit loops for stability)
    pub fn update_weights(&mut self, input: &Array1<f64>, target: &Array1<f64>, learning_rate: f64) {
        let prediction = self.predict(input);
        let error = target - &prediction;
        
        // Delta Rule: weight += learning_rate * error * input^T
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.weights[[i, j]] += learning_rate * error[i] * input[j];
            }
            self.biases[i] += learning_rate * error[i];
        }
    }
}

impl Learner {
    pub fn new(config: &Config) -> Self {
        // Default architecture for Genesis: 128 inputs -> 64 outputs
        let model = CognitiveModel::new(128, 64);
        let buffer = ContextBuffer::new(1000); // Keep last 1000 experiences
        
        Self {
            config: config.clone(),
            model,
            buffer,
        }
    }

    /// Ingest validated data and add to memory
    pub fn ingest(&mut self, data: &ValidatedData) {
        // Convert raw bytes to normalized float vector (Embedding)
        let input_vec = self.embed_data(&data.content);
        let target_vec = self.embed_data(&data.content); // Self-supervised for now
        
        let exp = Experience {
            input: input_vec,
            target: target_vec,
            weight: data.epistemic_state.confidence,
        };
        
        self.buffer.push(exp);
    }

    /// Train the model on current buffer
    pub fn train_step(&mut self) -> Result<(), LearningError> {
        if self.buffer.is_empty() {
            return Err(LearningError::EmptyBuffer);
        }

        let lr = 0.01; // Learning rate
        
        for exp in &self.buffer.experiences {
            // Weighted update: Higher confidence = stronger update
            let weighted_lr = lr * exp.weight;
            self.model.update_weights(&exp.input, &exp.target, weighted_lr);
        }
        
        Ok(())
    }

    /// Generate output for new input
    pub fn query(&self, input_bytes: &[u8]) -> Vec<f64> {
        let input_vec = self.embed_data(input_bytes);
        self.model.predict(&input_vec).to_vec()
    }

    /// Helper: Raw bytes to normalized float vector (128 dim)
    fn embed_data(&self, data: &[u8]) -> Array1<f64> {
        let mut vec = vec![0.0f64; self.model.input_size];
        for (i, byte) in data.iter().enumerate() {
            if i < self.model.input_size {
                vec[i] = *byte as f64 / 255.0;
            }
        }
        Array1::from(vec)
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_buffer_fifo() {
        let mut buf = ContextBuffer::new(2);
        buf.push(Experience { input: Array1::zeros(1), target: Array1::zeros(1), weight: 1.0 });
        buf.push(Experience { input: Array1::ones(1), target: Array1::ones(1), weight: 1.0 });
        buf.push(Experience { input: Array1::zeros(1), target: Array1::zeros(1), weight: 1.0 });
        
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_model_update_improves_loss() {
        let mut model = CognitiveModel::new(2, 1);
        let input = Array1::from_vec(vec![1.0, 0.0]);
        let target = Array1::from_vec(vec![1.0]);
        
        let pred_before = model.predict(&input);
        model.update_weights(&input, &target, 0.1);
        let pred_after = model.predict(&input);
        
        // Error should decrease after update
        assert!((target[0] - pred_after[0]).abs() < (target[0] - pred_before[0]).abs());
    }
}