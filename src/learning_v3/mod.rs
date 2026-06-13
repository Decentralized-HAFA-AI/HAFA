// ============================================================================
// HAFA Learning v3.0 — Transformer-Based Cognitive Engine
// ============================================================================
//
// Next-generation learning engine with:
// - Transformer architecture (Multi-Head Attention)
// - Byte-level tokenization
// - Positional encoding
// - Autoregressive generation with Temperature & Top-k Sampling
// - Hidden state extraction for training
// - Full backpropagation (Attention + Feed-Forward)
// - Production-Grade Trainer v4 (AdamW, Gradient Accumulation, Binary Checkpoint)
// - NEW: Reasoning Engine (Query & Inference over Knowledge Graph)
//
// This module runs alongside the legacy MLP engine (learning.rs)
// and will eventually replace it.
//
// ============================================================================

// ============================================================================
// TRAINER V3 MODULES (Legacy)
// ============================================================================
pub mod tensor;
pub mod embedding;
pub mod attention;
pub mod transformer;
pub mod loss;
pub mod optimizers;
pub mod trainer;
pub mod auto_learning;
pub mod backend;
pub mod knowledge_graph;
pub mod knowledge_graph_storage;
pub mod reasoning_engine;
pub mod accelerated;
pub mod wgpu_backend;

// ============================================================================
// TRAINER V4 MODULES (Production-Grade)
// ============================================================================
pub mod dataset_stream;
pub mod training_metrics;
pub mod cognitive_proof;
pub mod trainer_v4;
pub mod scheduler;
pub mod telemetry;
pub mod checkpoint;
pub mod gradient_bucket;
pub mod proof_verifier;
pub use wgpu_backend::WgpuBackend;

// ============================================================================
// TRAINER V3 EXPORTS
// ============================================================================
pub use tensor::Tensor;
pub use embedding::{ByteEmbedding, PositionalEncoding};
pub use attention::MultiHeadAttention;
pub use transformer::{TransformerBlock, TransformerStack};
pub use loss::{LossFunction, LossType, LossResult};
pub use optimizers::adamw::AdamW;
pub use trainer::{Trainer, TrainingProof};
pub use scheduler::LRScheduler;
pub use telemetry::SystemTelemetry;
pub use checkpoint::ModelCheckpoint;
pub use gradient_bucket::{GradientBucket, TensorGradient};
pub use proof_verifier::ProofVerifier;
pub use accelerated::{AcceleratedOps, BenchmarkResult};
// ============================================================================
// TRAINER V4 EXPORTS
// ============================================================================
pub use trainer_v4::TrainerV4;
pub use cognitive_proof::CognitiveProofV4;
pub use knowledge_graph::{KnowledgeGraph, Entity, Relation, EntityType, RelationType, KnowledgeGraphStats};
pub use knowledge_graph_storage::KnowledgeGraphStorage;
pub use reasoning_engine::{ReasoningEngine, QueryResult};  // NEW: Reasoning Engine exports

use serde::{Deserialize, Serialize};
use rand::Rng;

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            embed_dim: 128,
            num_heads: 4,
            num_layers: 2,
            ff_dim: 512,
            max_seq_len: 128,
            dropout: 0.1,
        }
    }
}

// ============================================================================
// TRANSFORMER ENGINE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerEngine {
    pub config: TransformerConfig,
    pub stack: TransformerStack,
}

impl TransformerEngine {
    pub fn new(config: &TransformerConfig) -> Self {
        let stack = TransformerStack::new(config);
        Self {
            config: config.clone(),
            stack,
        }
    }

    /// Forward pass: bytes → prediction logits
    /// Mutable because attention saves cache for backward pass
    pub fn forward(&mut self, input_bytes: &[u8]) -> Vec<f32> {
        let embedded = self.stack.embed(input_bytes);
        let hidden = self.stack.forward(&embedded);
        self.stack.predict(&hidden)
    }

    /// Forward pass that returns both logits AND hidden state
    /// Essential for proper gradient computation during training
    pub fn forward_with_hidden(&mut self, input_bytes: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let embedded = self.stack.embed(input_bytes);
        let hidden = self.stack.forward(&embedded);
        let logits = self.stack.predict(&hidden);
        
        let (_, seq_len, embed_dim) = hidden.shape();
        let last_idx = if seq_len > 0 { seq_len - 1 } else { 0 };
        let last_hidden: Vec<f32> = (0..embed_dim)
            .map(|d| hidden.data[[0, last_idx, d]])
            .collect();
        
        (logits, last_hidden)
    }

    /// Autoregressive text generation with Temperature and Top-k Sampling
    /// 
    /// Parameters:
    /// - prompt_bytes: Starting text
    /// - steps: Number of tokens to generate
    /// - temperature: Controls randomness (0.1-2.0, default 0.8)
    /// - top_k: Only sample from top k tokens (default 40, 0 = disabled)
    pub fn generate(
        &mut self, 
        prompt_bytes: &[u8], 
        steps: usize,
        temperature: f32,
        top_k: usize,
    ) -> Vec<u8> {
        let mut current_bytes = prompt_bytes.to_vec();
        let mut generated = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..steps {
            let logits = self.forward(&current_bytes);
            
            // Apply temperature scaling
            let scaled_logits: Vec<f32> = if temperature > 0.0 {
                logits.iter().map(|&l| l / temperature).collect()
            } else {
                logits.clone()
            };

            // Apply Top-k filtering
            let filtered_logits = if top_k > 0 && top_k < scaled_logits.len() {
                let mut indexed: Vec<(usize, f32)> = scaled_logits.iter()
                    .enumerate()
                    .map(|(i, &l)| (i, l))
                    .collect();
                
                // Sort by logit value (descending)
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                // Keep only top_k
                let top_k_items = &indexed[..top_k];
                
                // Create filtered vector (set others to -infinity)
                let mut filtered = vec![f32::NEG_INFINITY; scaled_logits.len()];
                for &(idx, logit) in top_k_items {
                    filtered[idx] = logit;
                }
                filtered
            } else {
                scaled_logits
            };

            // Convert to probabilities using softmax
            let probs = softmax(&filtered_logits);

            // Sample from the distribution
            let next_token = sample_from_distribution(&probs, &mut rng);
            
            generated.push(next_token);
            current_bytes.push(next_token);
            
            // Keep sequence length manageable
            if current_bytes.len() > self.config.max_seq_len {
                current_bytes.remove(0);
            }
        }
        generated
    }

    /// Get model statistics
    pub fn get_stats(&self) -> TransformerStats {
        TransformerStats {
            vocab_size: self.config.vocab_size,
            embed_dim: self.config.embed_dim,
            num_heads: self.config.num_heads,
            num_layers: self.config.num_layers,
            total_parameters: self.stack.count_parameters(),
        }
    }
}

/// Softmax function for converting logits to probabilities
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum_exp).collect()
}

/// Sample from a probability distribution
fn sample_from_distribution(probs: &[f32], rng: &mut impl Rng) -> u8 {
    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return i as u8;
        }
    }
    
    // Fallback to last token (should rarely happen)
    (probs.len() - 1) as u8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerStats {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub total_parameters: usize,
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_creation() {
        let config = TransformerConfig::default();
        let engine = TransformerEngine::new(&config);
        assert_eq!(engine.config.vocab_size, 256);
        assert_eq!(engine.config.num_heads, 4);
    }

    #[test]
    fn test_transformer_forward() {
        let config = TransformerConfig::default();
        let mut engine = TransformerEngine::new(&config);
        
        let input = b"HAFA";
        let output = engine.forward(input);
        
        assert_eq!(output.len(), config.vocab_size);
    }

    #[test]
    fn test_transformer_forward_with_hidden() {
        let config = TransformerConfig::default();
        let mut engine = TransformerEngine::new(&config);
        
        let input = b"HAFA";
        let (logits, hidden) = engine.forward_with_hidden(input);
        
        assert_eq!(logits.len(), config.vocab_size);
        assert_eq!(hidden.len(), config.embed_dim);
    }

    #[test]
    fn test_transformer_generate() {
        let config = TransformerConfig::default();
        let mut engine = TransformerEngine::new(&config);
        
        let input = b"HAFA";
        let output = engine.generate(input, 5, 0.8, 40);
        
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        
        // Sum of probabilities should be 1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        
        // Higher logit should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }
}