// ============================================================================
// HAFA - src/learning.rs — COGNITIVE LEARNING ENGINE (ROBUST & POWERFUL)
// ============================================================================
// 
// Full-power architecture with stability safeguards:
// - Large MLP: [128, 256, 128, 64]
// - Gradient clipping to prevent explosion
// - NaN detection and auto-recovery
// - Adaptive learning rate
//
// ============================================================================

use crate::config::Config;
use crate::data_source::ValidatedData;
use ndarray::{Array1, Array2};
use ndarray_rand::{RandomExt, rand::distributions::Uniform};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
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
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Training failed: {0}")]
    TrainingFailed(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    ReLU,
    GELU,
    Swish,
    Sigmoid,
    Tanh,
    Linear,
}

impl Activation {
    pub fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::GELU => x.mapv(gelu),
            Activation::Swish => x.mapv(swish),
            Activation::Sigmoid => x.mapv(sigmoid),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Linear => x.clone(),
        }
    }

    pub fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::GELU => x.mapv(gelu_derivative),
            Activation::Swish => x.mapv(swish_derivative),
            Activation::Sigmoid => {
                let s = x.mapv(sigmoid);
                &s * &(1.0 - &s)
            }
            Activation::Tanh => {
                let t = x.mapv(|v| v.tanh());
                1.0 - &(&t * &t)
            }
            Activation::Linear => Array1::ones(x.len()),
        }
    }
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

fn gelu_derivative(x: f64) -> f64 {
    let cdf = 0.5 * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());
    let pdf = (2.0 / std::f64::consts::PI).sqrt() * (-0.5 * x * x).exp();
    cdf + 0.5 * x * pdf
}

fn swish(x: f64) -> f64 {
    x * sigmoid(x)
}

fn swish_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s + x * s * (1.0 - s)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LossFunction {
    MSE,
    CrossEntropy,
}

impl LossFunction {
    pub fn compute(&self, prediction: &Array1<f64>, target: &Array1<f64>) -> f64 {
        match self {
            LossFunction::MSE => {
                let diff = prediction - target;
                let loss = diff.mapv(|v| v * v).mean().unwrap_or(0.0);
                if loss.is_nan() || loss.is_infinite() { 1.0 } else { loss }
            }
            LossFunction::CrossEntropy => {
                let eps = 1e-7;
                let pred_clipped = prediction.mapv(|v| v.clamp(eps, 1.0 - eps));
                let loss: f64 = target
                    .iter()
                    .zip(pred_clipped.iter())
                    .map(|(t, p)| -(t * p.ln() + (1.0 - t) * (1.0 - p).ln()))
                    .sum();
                let result = loss / target.len() as f64;
                if result.is_nan() || result.is_infinite() { 1.0 } else { result }
            }
        }
    }

    pub fn gradient(&self, prediction: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        let grad = match self {
            LossFunction::MSE => {
                let n = prediction.len() as f64;
                (prediction - target) * (2.0 / n)
            }
            LossFunction::CrossEntropy => {
                let eps = 1e-7;
                let pred_clipped = prediction.mapv(|v| v.clamp(eps, 1.0 - eps));
                let n = prediction.len() as f64;
                (&pred_clipped - target) / (&pred_clipped * &(1.0 - &pred_clipped) + eps) / n
            }
        };
        // Gradient clipping to prevent explosion
        grad.mapv(|v| v.clamp(-1.0, 1.0))
    }
}

// ============================================================================
// LAYER IMPLEMENTATION
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub activation: Activation,
    #[serde(skip)]
    pub input_cache: Option<Array1<f64>>,
    #[serde(skip)]
    pub pre_activation_cache: Option<Array1<f64>>,
    #[serde(skip)]
    pub weights_grad: Option<Array2<f64>>,
    #[serde(skip)]
    pub biases_grad: Option<Array1<f64>>,
    #[serde(skip)]
    pub m_weights: Option<Array2<f64>>,
    #[serde(skip)]
    pub v_weights: Option<Array2<f64>>,
    #[serde(skip)]
    pub m_biases: Option<Array1<f64>>,
    #[serde(skip)]
    pub v_biases: Option<Array1<f64>>,
}

impl Layer {
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        let std_dev = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let uniform = Uniform::new(-std_dev, std_dev);
        
        Self {
            weights: Array2::random((output_dim, input_dim), uniform),
            biases: Array1::zeros(output_dim),
            activation,
            input_cache: None,
            pre_activation_cache: None,
            weights_grad: None,
            biases_grad: None,
            m_weights: None,
            v_weights: None,
            m_biases: None,
            v_biases: None,
        }
    }

    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input_cache = Some(input.clone());
        
        let mut pre_activation = self.weights.dot(input);
        pre_activation += &self.biases;
        
        self.pre_activation_cache = Some(pre_activation.clone());
        
        self.activation.apply(&pre_activation)
    }

    pub fn backward(&mut self, output_gradient: &Array1<f64>) -> Array1<f64> {
        let pre_activation = self.pre_activation_cache.as_ref()
            .expect("Forward pass must be called before backward");
        let input = self.input_cache.as_ref()
            .expect("Forward pass must be called before backward");

        let activation_derivative = self.activation.derivative(pre_activation);
        let delta = output_gradient * &activation_derivative;
        let weights_gradient = outer_product(&delta, input);
        let biases_gradient = delta.clone();

        let input_gradient = self.weights.t().dot(&delta);

        self.weights_grad = Some(weights_gradient);
        self.biases_grad = Some(biases_gradient);

        input_gradient
    }

    pub fn get_gradients(&self) -> (Array2<f64>, Array1<f64>) {
        let weights_grad = self.weights_grad.as_ref()
            .expect("Backward pass must be called")
            .clone();
        let biases_grad = self.biases_grad.as_ref()
            .expect("Backward pass must be called")
            .clone();
        
        // Clip gradients
        let w_clipped = weights_grad.mapv(|v| v.clamp(-1.0, 1.0));
        let b_clipped = biases_grad.mapv(|v| v.clamp(-1.0, 1.0));
        
        (w_clipped, b_clipped)
    }

    pub fn has_nan(&self) -> bool {
        self.weights.iter().any(|v| v.is_nan() || v.is_infinite())
            || self.biases.iter().any(|v| v.is_nan() || v.is_infinite())
    }

    pub fn reset_if_nan(&mut self) {
        if self.has_nan() {
            let input_dim = self.weights.ncols();
            let output_dim = self.weights.nrows();
            let std_dev = (2.0 / (input_dim + output_dim) as f64).sqrt();
            let uniform = Uniform::new(-std_dev, std_dev);
            self.weights = Array2::random((output_dim, input_dim), uniform);
            self.biases = Array1::zeros(output_dim);
            self.m_weights = None;
            self.v_weights = None;
            self.m_biases = None;
            self.v_biases = None;
        }
    }
}

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let mut result = Array2::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

// ============================================================================
// OPTIMIZERS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Optimizer {
    SGD { learning_rate: f64 },
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        #[serde(skip)]
        t: usize,
    },
}

impl Optimizer {
    pub fn new_sgd(learning_rate: f64) -> Self {
        Optimizer::SGD { learning_rate }
    }

    pub fn new_adam(learning_rate: f64) -> Self {
        Optimizer::Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
        }
    }

    pub fn update_layer(&mut self, layer: &mut Layer) {
        let (weights_grad, biases_grad) = layer.get_gradients();

        match self {
            Optimizer::SGD { learning_rate } => {
                layer.weights -= &(&weights_grad * *learning_rate);
                layer.biases -= &(&biases_grad * *learning_rate);
            }
            Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                t,
            } => {
                *t += 1;

                if layer.m_weights.is_none() {
                    layer.m_weights = Some(Array2::zeros(layer.weights.dim()));
                    layer.v_weights = Some(Array2::zeros(layer.weights.dim()));
                    layer.m_biases = Some(Array1::zeros(layer.biases.len()));
                    layer.v_biases = Some(Array1::zeros(layer.biases.len()));
                }

                let m_w = layer.m_weights.as_mut().unwrap();
                let v_w = layer.v_weights.as_mut().unwrap();
                let m_b = layer.m_biases.as_mut().unwrap();
                let v_b = layer.v_biases.as_mut().unwrap();

                *m_w = &*m_w * *beta1 + &weights_grad * (1.0 - *beta1);
                *v_w = &*v_w * *beta2 + &(&weights_grad * &weights_grad) * (1.0 - *beta2);

                let m_w_hat = &*m_w / (1.0 - beta1.powi(*t as i32));
                let v_w_hat = &*v_w / (1.0 - beta2.powi(*t as i32));

                let update_w = &m_w_hat / &(&v_w_hat.mapv(|v| v.sqrt()) + *epsilon);
                layer.weights -= &(&update_w * *learning_rate);

                *m_b = &*m_b * *beta1 + &biases_grad * (1.0 - *beta1);
                *v_b = &*v_b * *beta2 + &(&biases_grad * &biases_grad) * (1.0 - *beta2);

                let m_b_hat = &*m_b / (1.0 - beta1.powi(*t as i32));
                let v_b_hat = &*v_b / (1.0 - beta2.powi(*t as i32));

                let update_b = &m_b_hat / &(&v_b_hat.mapv(|v| v.sqrt()) + *epsilon);
                layer.biases -= &(&update_b * *learning_rate);
            }
        }

        layer.reset_if_nan();
    }
}

// ============================================================================
// EXPERIENCE BUFFER
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub input: Array1<f64>,
    pub target: Array1<f64>,
    pub weight: f64,
    pub priority: f64,
}

#[derive(Debug, Clone)]
pub struct ExperienceBuffer {
    pub experiences: Vec<Experience>,
    pub max_size: usize,
}

impl ExperienceBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            experiences: Vec::new(),
            max_size,
        }
    }

    pub fn push(&mut self, exp: Experience) {
        if self.experiences.len() >= self.max_size {
            if let Some(min_idx) = self.experiences
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.priority.partial_cmp(&b.priority).unwrap())
                .map(|(idx, _)| idx)
            {
                self.experiences.remove(min_idx);
            }
        }
        self.experiences.push(exp);
    }

    pub fn sample_batch(&self, batch_size: usize) -> Vec<&Experience> {
        let mut rng = rand::thread_rng();
        let actual_size = batch_size.min(self.experiences.len());
        self.experiences
            .choose_multiple(&mut rng, actual_size)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.experiences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experiences.is_empty()
    }
}

// ============================================================================
// COGNITIVE MODEL
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveModel {
    pub layers: Vec<Layer>,
    pub input_size: usize,
    pub output_size: usize,
}

impl CognitiveModel {
    pub fn new(layer_sizes: &[usize], activations: &[Activation]) -> Result<Self, LearningError> {
        if layer_sizes.len() < 2 {
            return Err(LearningError::InvalidConfig(
                "Model must have at least 2 layers".to_string(),
            ));
        }

        if activations.len() != layer_sizes.len() - 1 {
            return Err(LearningError::InvalidConfig(
                "Number of activations must match number of layers - 1".to_string(),
            ));
        }

        let mut layers = Vec::new();
        for i in 0..(layer_sizes.len() - 1) {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1], activations[i]));
        }

        Ok(Self {
            input_size: layer_sizes[0],
            output_size: *layer_sizes.last().unwrap(),
            layers,
        })
    }

    pub fn predict(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current.mapv(|v| {
            if v.is_nan() || v.is_infinite() { 0.0 } else { v }
        })
    }

    pub fn backward(&mut self, output_gradient: &Array1<f64>) {
        let mut gradient = output_gradient.clone();
        for layer in self.layers.iter_mut().rev() {
            gradient = layer.backward(&gradient);
        }
    }

    pub fn serialize_weights(&self) -> Result<Vec<u8>, LearningError> {
        bincode::serialize(self)
            .map_err(|e| LearningError::SerializationError(e.to_string()))
    }

    pub fn deserialize_weights(data: &[u8]) -> Result<Self, LearningError> {
        bincode::deserialize(data)
            .map_err(|e| LearningError::SerializationError(e.to_string()))
    }
}

// ============================================================================
// LEARNER
// ============================================================================

pub struct Learner {
    pub config: Config,
    pub model: CognitiveModel,
    pub buffer: ExperienceBuffer,
    pub optimizer: Optimizer,
    pub loss_fn: LossFunction,
    pub batch_size: usize,
    pub context_size: usize,
    pub predict_size: usize,
}

impl Learner {
    pub fn new(config: &Config) -> Self {
        // FULL POWER architecture
        let layer_sizes = vec![128, 256, 128, 64];
        let activations = vec![Activation::GELU, Activation::GELU, Activation::Linear];
        
        let model = CognitiveModel::new(&layer_sizes, &activations)
            .expect("Failed to create model");
        
        Self {
            config: config.clone(),
            model,
            buffer: ExperienceBuffer::new(10000),
            optimizer: Optimizer::new_adam(0.0005),
            loss_fn: LossFunction::MSE,
            batch_size: 32,
            context_size: 64,
            predict_size: 64,
        }
    }

    pub fn ingest(&mut self, data: &ValidatedData) {
        let content = &data.content;
        let content_len = content.len();
        let step_size = 32;

        if content_len < self.context_size + self.predict_size {
            let mut padded = content.clone();
            padded.resize(self.context_size + self.predict_size, 0);
            self.add_experience(&padded, 0, data.epistemic_state.confidence);
            return;
        }

        let mut i = 0;
        while i + self.context_size + self.predict_size <= content_len {
            self.add_experience(content, i, data.epistemic_state.confidence);
            i += step_size;
        }
    }

    fn add_experience(&mut self, content: &[u8], start_idx: usize, confidence: f64) {
        let context_end = start_idx + self.context_size;
        let target_end = context_end + self.predict_size;

        let input_vec = self.embed_input(&content[start_idx..context_end]);
        let target_vec = self.embed_target(&content[context_end..target_end]);

        let exp = Experience {
            input: input_vec,
            target: target_vec,
            weight: confidence,
            priority: confidence,
        };
        
        self.buffer.push(exp);
    }

    pub fn train_step(&mut self) -> Result<f64, LearningError> {
        if self.buffer.is_empty() {
            return Err(LearningError::EmptyBuffer);
        }

        let batch = self.buffer.sample_batch(self.batch_size);
        let batch_len = batch.len();
        let mut total_loss = 0.0;

        for exp in batch {
            let prediction = self.model.predict(&exp.input);
            let loss = self.loss_fn.compute(&prediction, &exp.target);
            total_loss += loss * exp.weight;

            let output_gradient = self.loss_fn.gradient(&prediction, &exp.target);
            self.model.backward(&output_gradient);

            for layer in &mut self.model.layers {
                self.optimizer.update_layer(layer);
            }
        }

        let avg_loss = total_loss / batch_len as f64;
        
        if avg_loss.is_nan() || avg_loss.is_infinite() {
            Ok(1.0)
        } else {
            Ok(avg_loss)
        }
    }

    pub fn query(&mut self, prompt_bytes: &[u8], generate_steps: usize) -> Vec<u8> {
        let mut current_context = self.embed_input(prompt_bytes);
        let mut generated_bytes = Vec::new();

        for _ in 0..generate_steps {
            let prediction = self.model.predict(&current_context);
            
            let mut next_chunk = vec![0u8; self.predict_size];
            for (i, &val) in prediction.iter().enumerate() {
                if i < self.predict_size {
                    let normalized = sigmoid(val);
                    let byte_val = (normalized * 255.0).round() as u8;
                    next_chunk[i] = byte_val;
                    generated_bytes.push(byte_val);
                }
            }

            current_context = self.embed_input(&next_chunk);
        }

        generated_bytes
    }

    pub fn query_simple(&mut self, input_bytes: &[u8]) -> Vec<f64> {
        let input_vec = self.embed_input(input_bytes);
        self.model.predict(&input_vec).to_vec()
    }

    // ✅ تابع جدید برای input (اندازه context_size)
    fn embed_input(&self, data: &[u8]) -> Array1<f64> {
        let mut vec = vec![0.0f64; self.model.input_size];
        for (i, byte) in data.iter().enumerate() {
            if i < self.model.input_size {
                vec[i] = *byte as f64 / 255.0;
            }
        }
        Array1::from(vec)
    }

    // ✅ تابع جدید برای target (اندازه predict_size)
    fn embed_target(&self, data: &[u8]) -> Array1<f64> {
        let mut vec = vec![0.0f64; self.predict_size];
        for (i, byte) in data.iter().enumerate() {
            if i < self.predict_size {
                vec[i] = *byte as f64 / 255.0;
            }
        }
        Array1::from(vec)
    }

    pub fn get_stats(&self) -> ModelStats {
        ModelStats {
            input_size: self.model.input_size,
            output_size: self.model.output_size,
            num_layers: self.model.layers.len(),
            buffer_size: self.buffer.len(),
            total_parameters: self.model.layers.iter()
                .map(|l| l.weights.len() + l.biases.len())
                .sum(),
            context_size: self.context_size,
            predict_size: self.predict_size,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub input_size: usize,
    pub output_size: usize,
    pub num_layers: usize,
    pub buffer_size: usize,
    pub total_parameters: usize,
    pub context_size: usize,
    pub predict_size: usize,
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_functions() {
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        
        let relu = Activation::ReLU.apply(&x);
        assert_eq!(relu[0], 0.0);
        assert_eq!(relu[1], 0.0);
        assert_eq!(relu[2], 1.0);
    }

    #[test]
    fn test_model_forward_backward() {
        let mut model = CognitiveModel::new(
            &[2, 4, 1],
            &[Activation::ReLU, Activation::Linear],
        ).unwrap();

        let input = Array1::from_vec(vec![1.0, 0.5]);
        let target = Array1::from_vec(vec![1.0]);

        let pred_before = model.predict(&input);
        
        let loss_fn = LossFunction::MSE;
        let gradient = loss_fn.gradient(&pred_before, &target);
        model.backward(&gradient);

        let (w_grad, b_grad) = model.layers[0].get_gradients();
        assert!(w_grad.iter().any(|&v| v != 0.0));
        assert!(b_grad.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_experience_buffer() {
        let mut buffer = ExperienceBuffer::new(3);
        
        buffer.push(Experience {
            input: Array1::zeros(1),
            target: Array1::zeros(1),
            weight: 1.0,
            priority: 0.5,
        });
        
        buffer.push(Experience {
            input: Array1::ones(1),
            target: Array1::ones(1),
            weight: 1.0,
            priority: 0.8,
        });
        
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_model_serialization() {
        let model = CognitiveModel::new(
            &[10, 20, 5],
            &[Activation::ReLU, Activation::Linear],
        ).unwrap();

        let serialized = model.serialize_weights().unwrap();
        let deserialized = CognitiveModel::deserialize_weights(&serialized).unwrap();

        assert_eq!(model.input_size, deserialized.input_size);
        assert_eq!(model.output_size, deserialized.output_size);
    }

    #[test]
    fn test_sliding_window_ingest() {
        use crate::epistemic::EpistemicState;
        
        let config = Config::default();
        let mut learner = Learner::new(&config);
        
        let test_data = vec![1u8; 300];
        let validated_data = ValidatedData {
            content: test_data,
            source: crate::data_source::DataSource::Local { path: "test".to_string() },
            epistemic_state: EpistemicState::new(0.9, true, 0, 0.1, 1, 0.0, 1.0),
            timestamp: 0,
            knowledge_claim: crate::epistemic::KnowledgeClaim::new(
                b"test",
                "local".to_string(),
                "test_id".to_string(),
                true,
                "test".to_string(),
            ),
            metadata: None,
        };
        
        learner.ingest(&validated_data);
        assert!(learner.buffer.len() > 1);
    }

    #[test]
    fn test_training_stability() {
        use crate::epistemic::EpistemicState;
        
        let config = Config::default();
        let mut learner = Learner::new(&config);
        
        let test_data = vec![65u8; 300];
        let validated_data = ValidatedData {
            content: test_data,
            source: crate::data_source::DataSource::Local { path: "test".to_string() },
            epistemic_state: EpistemicState::new(0.9, true, 0, 0.1, 1, 0.0, 1.0),
            timestamp: 0,
            knowledge_claim: crate::epistemic::KnowledgeClaim::new(
                b"test",
                "local".to_string(),
                "test_id".to_string(),
                true,
                "test".to_string(),
            ),
            metadata: None,
        };
        
        learner.ingest(&validated_data);
        
        for _ in 0..10 {
            let result = learner.train_step();
            assert!(result.is_ok());
            let loss = result.unwrap();
            assert!(!loss.is_nan());
            assert!(!loss.is_infinite());
        }
    }
}