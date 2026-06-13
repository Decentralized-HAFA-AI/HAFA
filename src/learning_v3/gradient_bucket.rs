// ============================================================================
// Gradient Bucket: Full-Model Gradient Storage & Management
// ============================================================================
//
// Stores gradients for ALL model parameters (not just pred_head).
// This enables proper full-model training with gradient accumulation.
//
// Structure:
// - embedding: [vocab_size, embed_dim]
// - For each layer:
//   - Attention: w_q, w_k, w_v, w_o (each [embed_dim, embed_dim])
//   - FFN: ffn1_weight [embed_dim, ff_dim], ffn1_bias [1, ff_dim]
//          ffn2_weight [ff_dim, embed_dim], ffn2_bias [1, embed_dim]
//   - LayerNorm 1: ln1_gamma [1, embed_dim], ln1_beta [1, embed_dim]
//   - LayerNorm 2: ln2_gamma [1, embed_dim], ln2_beta [1, embed_dim]
// - final_ln_gamma [1, embed_dim], final_ln_beta [1, embed_dim]
// - pred_head: [embed_dim, vocab_size]
//
// Design Principles:
// - Deterministic hashing (order-independent) for verifiable PoUCW
// - Optimized operations (in-place where possible)
// - Comprehensive proof support (task_id, seed, model state)
//
// ============================================================================

use ndarray::Array2;
use sha3::{Sha3_256, Digest};
use super::TransformerEngine;

#[derive(Debug, Clone)]
pub struct TensorGradient {
    pub name: String,
    pub shape: (usize, usize),
    pub data: Array2<f32>,
}

pub struct GradientBucket {
    pub gradients: Vec<TensorGradient>,
}

impl GradientBucket {
    /// Creates an empty gradient bucket matching the model structure
    /// NOTE: Order of insertion is preserved but hash computation is order-independent
    pub fn new_from_model(model: &TransformerEngine) -> Self {
        let mut gradients = Vec::new();
        let embed_dim = model.config.embed_dim;
        let vocab_size = model.config.vocab_size;
        let ff_dim = model.config.ff_dim;
        
        // 1. Embedding weights
        gradients.push(TensorGradient {
            name: "embedding".into(),
            shape: (vocab_size, embed_dim),
            data: Array2::zeros((vocab_size, embed_dim)),
        });
        
        // 2. Transformer layers
        for layer_idx in 0..model.config.num_layers {
            let prefix = format!("layer_{}", layer_idx);
            
            // Attention: Q, K, V, O projections
            for proj in ["w_q", "w_k", "w_v", "w_o"] {
                gradients.push(TensorGradient {
                    name: format!("{}.{}", prefix, proj),
                    shape: (embed_dim, embed_dim),
                    data: Array2::zeros((embed_dim, embed_dim)),
                });
            }
            
            // Feed-Forward Network (weights + biases)
            gradients.push(TensorGradient {
                name: format!("{}.ffn1_weight", prefix),
                shape: (embed_dim, ff_dim),
                data: Array2::zeros((embed_dim, ff_dim)),
            });
            gradients.push(TensorGradient {
                name: format!("{}.ffn1_bias", prefix),
                shape: (1, ff_dim),
                data: Array2::zeros((1, ff_dim)),
            });
            gradients.push(TensorGradient {
                name: format!("{}.ffn2_weight", prefix),
                shape: (ff_dim, embed_dim),
                data: Array2::zeros((ff_dim, embed_dim)),
            });
            gradients.push(TensorGradient {
                name: format!("{}.ffn2_bias", prefix),
                shape: (1, embed_dim),
                data: Array2::zeros((1, embed_dim)),
            });
            
            // LayerNorm 1 (gamma + beta) - Learnable parameters
            gradients.push(TensorGradient {
                name: format!("{}.ln1_gamma", prefix),
                shape: (1, embed_dim),
                data: Array2::zeros((1, embed_dim)),
            });
            gradients.push(TensorGradient {
                name: format!("{}.ln1_beta", prefix),
                shape: (1, embed_dim),
                data: Array2::zeros((1, embed_dim)),
            });
            
            // LayerNorm 2 (gamma + beta) - Learnable parameters
            gradients.push(TensorGradient {
                name: format!("{}.ln2_gamma", prefix),
                shape: (1, embed_dim),
                data: Array2::zeros((1, embed_dim)),
            });
            gradients.push(TensorGradient {
                name: format!("{}.ln2_beta", prefix),
                shape: (1, embed_dim),
                data: Array2::zeros((1, embed_dim)),
            });
        }
        
        // 3. Final LayerNorm (gamma + beta) - Learnable parameters
        gradients.push(TensorGradient {
            name: "final_ln_gamma".into(),
            shape: (1, embed_dim),
            data: Array2::zeros((1, embed_dim)),
        });
        gradients.push(TensorGradient {
            name: "final_ln_beta".into(),
            shape: (1, embed_dim),
            data: Array2::zeros((1, embed_dim)),
        });
        
        // 4. Prediction Head
        gradients.push(TensorGradient {
            name: "pred_head".into(),
            shape: (embed_dim, vocab_size),
            data: Array2::zeros((embed_dim, vocab_size)),
        });
        
        Self { gradients }
    }
    
    /// Computes global L2 norm of all gradients
    /// Optimized: uses iterator chain to avoid intermediate allocations
    pub fn global_norm(&self) -> f32 {
        self.gradients.iter()
            .flat_map(|g| g.data.iter())
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
    }
    
    /// Applies global gradient clipping
    /// If norm > max_norm, scales all gradients by (max_norm / norm)
    pub fn clip_by_global_norm(&mut self, max_norm: f32) {
        let norm = self.global_norm();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for g in &mut self.gradients {
                g.data.mapv_inplace(|x| x * scale);
            }
        }
    }
    
    /// Zeros all gradients (in-place)
    pub fn zero(&mut self) {
        for g in &mut self.gradients {
            g.data.fill(0.0);
        }
    }
    
    /// Computes SHA3-256 hash of all gradients (for verifiable proof)
    /// ORDER-INDEPENDENT: Sorts by name before hashing to ensure determinism
    pub fn compute_hash(&self) -> String {
        let mut hasher = Sha3_256::new();
        
        // Sort by name for canonical order (deterministic regardless of insertion order)
        let mut sorted: Vec<_> = self.gradients.iter().collect();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        
        for g in sorted {
            // Include name and shape for uniqueness
            hasher.update(g.name.as_bytes());
            hasher.update(&g.shape.0.to_le_bytes());
            hasher.update(&g.shape.1.to_le_bytes());
            
            // Hash gradient data deterministically
            for &val in g.data.iter() {
                hasher.update(&val.to_le_bytes());
            }
        }
        
        format!("{:x}", hasher.finalize())
    }
    
    /// Computes comprehensive hash including task context and model state
    /// This prevents replay attacks and enables full verification
    /// 
    /// # Arguments
    /// * `model_before` - Serialized model weights before training
    /// * `model_after` - Serialized model weights after training
    /// * `task_id` - Unique identifier for this training task
    /// * `seed` - Random seed used for data shuffling
    pub fn compute_comprehensive_hash(
        &self,
        model_before: &[u8],
        model_after: &[u8],
        task_id: &str,
        seed: u64,
    ) -> String {
        let mut hasher = Sha3_256::new();
        
        // 1. Task context (prevents replay attacks)
        hasher.update(task_id.as_bytes());
        hasher.update(&seed.to_le_bytes());
        
        // 2. Model state (before and after)
        hasher.update(&(model_before.len() as u64).to_le_bytes());
        hasher.update(model_before);
        hasher.update(&(model_after.len() as u64).to_le_bytes());
        hasher.update(model_after);
        
        // 3. Gradients (sorted for determinism)
        let mut sorted: Vec<_> = self.gradients.iter().collect();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        
        for g in sorted {
            hasher.update(g.name.as_bytes());
            hasher.update(&g.shape.0.to_le_bytes());
            hasher.update(&g.shape.1.to_le_bytes());
            for &val in g.data.iter() {
                hasher.update(&val.to_le_bytes());
            }
        }
        
        format!("{:x}", hasher.finalize())
    }
    
    /// Adds another bucket's gradients to this one (for accumulation)
    /// Optimized: uses in-place operations
    pub fn add(&mut self, other: &GradientBucket) {
        for (self_g, other_g) in self.gradients.iter_mut().zip(other.gradients.iter()) {
            debug_assert_eq!(self_g.shape, other_g.shape, "Shape mismatch in gradient addition");
            // In-place addition for better performance
            ndarray::Zip::from(&mut self_g.data)
                .and(&other_g.data)
                .for_each(|s, &o| *s += o);
        }
    }
    
    /// Divides all gradients by a scalar (for averaging)
    /// Optimized: uses in-place mapv
    pub fn divide(&mut self, divisor: f32) {
        debug_assert!(divisor > 0.0, "Divisor must be positive");
        let inv = 1.0 / divisor;
        for g in &mut self.gradients {
            g.data.mapv_inplace(|x| x * inv);
        }
    }
    
    /// Gets a mutable reference to a specific gradient by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut TensorGradient> {
        self.gradients.iter_mut().find(|g| g.name == name)
    }
    
    /// Gets a reference to a specific gradient by name
    pub fn get(&self, name: &str) -> Option<&TensorGradient> {
        self.gradients.iter().find(|g| g.name == name)
    }
    
    /// Returns the total number of gradient parameters
    pub fn total_parameters(&self) -> usize {
        self.gradients.iter()
            .map(|g| g.data.len())
            .sum()
    }
    
    /// Returns a summary of gradient statistics for debugging
    pub fn summary(&self) -> String {
        let total_params = self.total_parameters();
        let non_zero = self.gradients.iter()
            .filter(|g| g.data.iter().any(|&x| x.abs() > 1e-10))
            .count();
        
        format!(
            "GradientBucket: {} tensors, {} parameters, {} non-zero",
            self.gradients.len(),
            total_params,
            non_zero
        )
    }
    
    /// Verifies that this bucket's structure matches the given model
    /// Returns Ok(()) if shapes match, Err with details otherwise
    pub fn verify_against_model(&self, model: &TransformerEngine) -> Result<(), String> {
        let expected = Self::new_from_model(model);
        
        if self.gradients.len() != expected.gradients.len() {
            return Err(format!(
                "Gradient count mismatch: got {}, expected {}",
                self.gradients.len(),
                expected.gradients.len()
            ));
        }
        
        for (actual, exp) in self.gradients.iter().zip(expected.gradients.iter()) {
            if actual.name != exp.name {
                return Err(format!(
                    "Gradient name mismatch: got '{}', expected '{}'",
                    actual.name, exp.name
                ));
            }
            if actual.shape != exp.shape {
                return Err(format!(
                    "Shape mismatch for '{}': got {:?}, expected {:?}",
                    actual.name, actual.shape, exp.shape
                ));
            }
        }
        
        Ok(())
    }
    
    /// Computes expected gradient count for a given config (for testing)
    pub fn expected_gradient_count(config: &super::TransformerConfig) -> usize {
        // 1 embedding + N layers * 12 (4 attn + 4 ffn + 4 ln) + 2 final_ln + 1 pred_head
        1 + config.num_layers * 12 + 2 + 1
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_v3::TransformerConfig;
    
    fn test_config(num_layers: usize) -> TransformerConfig {
        TransformerConfig {
            vocab_size: 256,
            embed_dim: 64,
            num_layers,
            num_heads: 4,
            ff_dim: 256,
            max_seq_len: 128,
            dropout: 0.1,
        }
    }
    
    #[test]
    fn test_gradient_bucket_creation_dynamic() {
        for num_layers in [1, 2, 4, 6] {
            let config = test_config(num_layers);
            let model = TransformerEngine::new(&config);
            let bucket = GradientBucket::new_from_model(&model);
            
            // Dynamic calculation (not hardcoded!)
            let expected = GradientBucket::expected_gradient_count(&config);
            assert_eq!(
                bucket.gradients.len(), 
                expected,
                "Failed for num_layers={}", 
                num_layers
            );
            
            // Verify key parameters exist
            assert!(bucket.get("pred_head").is_some());
            assert!(bucket.get("embedding").is_some());
            assert!(bucket.get("final_ln_gamma").is_some());
            assert!(bucket.get("final_ln_beta").is_some());
            
            for layer_idx in 0..num_layers {
                assert!(bucket.get(&format!("layer_{}.ln1_gamma", layer_idx)).is_some());
                assert!(bucket.get(&format!("layer_{}.ln1_beta", layer_idx)).is_some());
                assert!(bucket.get(&format!("layer_{}.ln2_gamma", layer_idx)).is_some());
                assert!(bucket.get(&format!("layer_{}.ln2_beta", layer_idx)).is_some());
                assert!(bucket.get(&format!("layer_{}.w_q", layer_idx)).is_some());
                assert!(bucket.get(&format!("layer_{}.w_k", layer_idx)).is_some());
                assert!(bucket.get(&format!("layer_{}.w_v", layer_idx)).is_some());
                assert!(bucket.get(&format!("layer_{}.w_o", layer_idx)).is_some());
            }
        }
    }
    
    #[test]
    fn test_gradient_operations() {
        let config = test_config(2);
        let model = TransformerEngine::new(&config);
        let mut bucket = GradientBucket::new_from_model(&model);
        
        // Test zero
        bucket.zero();
        assert_eq!(bucket.global_norm(), 0.0);
        
        // Test add
        let mut bucket2 = GradientBucket::new_from_model(&model);
        if let Some(g) = bucket2.get_mut("pred_head") {
            g.data.fill(1.0);
        }
        bucket.add(&bucket2);
        assert!(bucket.global_norm() > 0.0);
        
        // Test divide
        bucket.divide(2.0);
        
        // Test hash determinism
        let hash1 = bucket.compute_hash();
        let hash2 = bucket.compute_hash();
        assert_eq!(hash1, hash2, "Hash should be deterministic");
        assert_eq!(hash1.len(), 64, "SHA3-256 produces 64 hex characters");
    }
    
    #[test]
    fn test_hash_order_independence() {
        let config = test_config(2);
        let model = TransformerEngine::new(&config);
        
        let mut bucket1 = GradientBucket::new_from_model(&model);
        let mut bucket2 = GradientBucket::new_from_model(&model);
        
        // Fill with same values
        if let Some(g) = bucket1.get_mut("pred_head") {
            g.data.fill(1.0);
        }
        if let Some(g) = bucket2.get_mut("pred_head") {
            g.data.fill(1.0);
        }
        
        // Reverse the order of gradients in bucket2
        bucket2.gradients.reverse();
        
        // Hashes should still be equal (order-independent)
        assert_eq!(
            bucket1.compute_hash(),
            bucket2.compute_hash(),
            "Hash should be order-independent"
        );
    }
    
    #[test]
    fn test_clip_by_global_norm() {
        let config = test_config(2);
        let model = TransformerEngine::new(&config);
        let mut bucket = GradientBucket::new_from_model(&model);
        
        // Fill with large values to ensure norm > 50.0
        for g in &mut bucket.gradients {
            g.data.fill(10.0);
        }
        
        let _norm_before = bucket.global_norm();
        
        // Clip to max_norm = 50.0
        let max_norm = 50.0;
        bucket.clip_by_global_norm(max_norm);
        
        let norm_after = bucket.global_norm();
        assert!(
            (norm_after - max_norm).abs() < 0.1,
            "After clipping, norm should be ~max_norm (got {})",
            norm_after
        );
    }
    
    #[test]
    fn test_clip_no_op_when_below_threshold() {
        let config = test_config(2);
        let model = TransformerEngine::new(&config);
        let mut bucket = GradientBucket::new_from_model(&model);
        
        // Fill with very small values so norm is definitely < 50.0
        for g in &mut bucket.gradients {
            g.data.fill(0.001);
        }
        
        let norm_before = bucket.global_norm();
        
        // Attempt to clip to a high value (should do absolutely nothing)
        bucket.clip_by_global_norm(50.0);
        
        let norm_after = bucket.global_norm();
        
        // The norm should remain exactly the same (within FP precision)
        assert!(
            (norm_before - norm_after).abs() < 1e-5,
            "Clipping should be a no-op when norm < max_norm. Before: {}, After: {}",
            norm_before,
            norm_after
        );
    }
    
    #[test]
    fn test_verify_against_model() {
        let config = test_config(2);
        let model = TransformerEngine::new(&config);
        let bucket = GradientBucket::new_from_model(&model);
        
        // Should verify successfully
        assert!(bucket.verify_against_model(&model).is_ok());
    }
    
    #[test]
    fn test_comprehensive_hash() {
        let config = test_config(2);
        let model = TransformerEngine::new(&config);
        let bucket = GradientBucket::new_from_model(&model);
        
        let model_before = b"model_state_before";
        let model_after = b"model_state_after";
        let task_id = "task_123";
        let seed = 42u64;
        
        let hash = bucket.compute_comprehensive_hash(
            model_before,
            model_after,
            task_id,
            seed,
        );
        
        assert_eq!(hash.len(), 64, "SHA3-256 produces 64 hex characters");
        
        // Same inputs should produce same hash
        let hash2 = bucket.compute_comprehensive_hash(
            model_before,
            model_after,
            task_id,
            seed,
        );
        assert_eq!(hash, hash2, "Comprehensive hash should be deterministic");
        
        // Different task_id should produce different hash
        let hash3 = bucket.compute_comprehensive_hash(
            model_before,
            model_after,
            "task_456",
            seed,
        );
        assert_ne!(hash, hash3, "Different task_id should produce different hash");
    }
}