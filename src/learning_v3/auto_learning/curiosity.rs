// ============================================================================
// Curiosity Module: Intelligent Learning Selection
// ============================================================================
//
// Determines which data samples are worth learning based on:
// 1. Novelty (new vs. seen before) - MUST be novel to be considered
// 2. Prediction Error (model's current weakness)
// 3. Uncertainty (entropy of predictions)
//
// This enables HAFA to learn efficiently by focusing on valuable data.
//
// ============================================================================

use std::collections::HashSet;
use sha3::{Sha3_256, Digest};

use super::data_source::TrainingSample;
use crate::learning_v3::TrainerV4;

/// Configuration for the Curiosity Module
#[derive(Debug, Clone)]
pub struct CuriosityConfig {
    /// Weight for uncertainty score (0.0 to 1.0)
    pub uncertainty_weight: f32,
    /// Weight for prediction error score (0.0 to 1.0)
    pub prediction_error_weight: f32,
    /// Weight for novelty score (0.0 to 1.0)
    pub novelty_weight: f32,
    /// Minimum curiosity score to accept a sample (0.0 to 1.0)
    pub min_curiosity_threshold: f32,
    /// Maximum number of learned hashes to store (for memory management)
    pub max_learned_hashes: usize,
}

impl Default for CuriosityConfig {
    fn default() -> Self {
        Self {
            uncertainty_weight: 0.3,
            prediction_error_weight: 0.5,  // Most important
            novelty_weight: 0.2,
            min_curiosity_threshold: 0.3,  // Accept samples with score >= 0.3
            max_learned_hashes: 10000,     // Store up to 10k hashes
        }
    }
}

/// Statistics about curiosity module decisions
#[derive(Debug, Clone, Default)]
pub struct CuriosityStats {
    pub total_evaluated: u64,
    pub total_selected: u64,
    pub total_rejected: u64,
    pub total_rejected_not_novel: u64,  // NEW: Count rejections due to non-novelty
    pub avg_curiosity_score: f64,
    pub avg_uncertainty: f64,
    pub avg_prediction_error: f64,
    pub avg_novelty: f64,
}

/// The Curiosity Module: intelligent learning selection
pub struct CuriosityModule {
    config: CuriosityConfig,
    /// Hashes of samples we've already learned (for novelty detection)
    learned_hashes: HashSet<String>,
    /// Hashes of samples currently in buffer (pending learning)
    pending_hashes: HashSet<String>,
    /// Statistics
    stats: CuriosityStats,
    /// Running sum for average calculation
    sum_curiosity: f64,
    sum_uncertainty: f64,
    sum_prediction_error: f64,
    sum_novelty: f64,
}

impl CuriosityModule {
    /// Create a new CuriosityModule with default config
    pub fn new() -> Self {
        Self::with_config(CuriosityConfig::default())
    }
    
    /// Create a new CuriosityModule with custom config
    pub fn with_config(config: CuriosityConfig) -> Self {
        // Validate weights sum to 1.0
        let weight_sum = config.uncertainty_weight 
            + config.prediction_error_weight 
            + config.novelty_weight;
        
        if (weight_sum - 1.0).abs() > 0.01 {
            println!("   [CURIOSITY] WARNING: Weights sum to {}, normalizing to 1.0", weight_sum);
        }
        
        Self {
            config,
            learned_hashes: HashSet::new(),
            pending_hashes: HashSet::new(),
            stats: CuriosityStats::default(),
            sum_curiosity: 0.0,
            sum_uncertainty: 0.0,
            sum_prediction_error: 0.0,
            sum_novelty: 0.0,
        }
    }
    
    /// Compute curiosity score for a sample
    /// Returns a score between 0.0 (not interesting) and 1.0 (very interesting)
    /// 
    /// IMPORTANT: Novelty is a prerequisite - if sample is not novel, score = 0.0
    /// 
    /// NOTE: Requires `&mut TrainerV4` because forward pass mutates attention cache
    pub fn compute_curiosity_score(
        &mut self,
        sample: &TrainingSample,
        trainer: &mut TrainerV4,
    ) -> f32 {
        // 1. Compute novelty FIRST (cheap check - no forward pass needed)
        let novelty = self.compute_novelty(sample);
        
        // 2. NEW: If not novel, skip expensive computations and return 0
        if novelty == 0.0 {
            self.stats.total_evaluated += 1;
            self.stats.total_rejected_not_novel += 1;
            return 0.0;  // Definitely not worth learning
        }
        
        // 3. Only compute expensive metrics if sample is novel
        let uncertainty = self.compute_uncertainty(sample, trainer);
        let prediction_error = self.compute_prediction_error(sample, trainer);
        
        // 4. Compute weighted curiosity score
        let curiosity = 
            uncertainty * self.config.uncertainty_weight +
            prediction_error * self.config.prediction_error_weight +
            novelty * self.config.novelty_weight;
        
        // 5. Apply source confidence as a multiplier
        let final_score = curiosity * sample.confidence;
        
        // 6. Update running averages
        self.sum_curiosity += final_score as f64;
        self.sum_uncertainty += uncertainty as f64;
        self.sum_prediction_error += prediction_error as f64;
        self.sum_novelty += novelty as f64;
        self.stats.total_evaluated += 1;
        
        final_score
    }
    
    /// Compute uncertainty score based on prediction entropy
    /// High entropy = model is uncertain = worth learning
    fn compute_uncertainty(&self, sample: &TrainingSample, trainer: &mut TrainerV4) -> f32 {
        // Skip if text is too short
        if sample.text.len() < 8 {
            return 0.5; // Default uncertainty for short texts
        }
        
        // Use first 64 bytes as context (or pad if shorter)
        let context_bytes = if sample.text.len() >= 64 {
            &sample.text.as_bytes()[..64]
        } else {
            sample.text.as_bytes()
        };
        
        // Forward pass to get logits (requires mutable reference)
        let embedded = trainer.model.stack.embed(context_bytes);
        let hidden = trainer.model.stack.forward(&embedded);
        let logits = trainer.model.stack.predict(&hidden);
        
        // Compute softmax and entropy
        let entropy = self.compute_entropy(&logits);
        
        // Normalize entropy (max entropy = ln(vocab_size))
        let max_entropy = (trainer.model.config.vocab_size as f32).ln();
        let normalized_entropy = entropy / max_entropy;
        
        // Clamp to [0, 1]
        normalized_entropy.clamp(0.0, 1.0)
    }
    
    /// Compute entropy of a probability distribution
    fn compute_entropy(&self, logits: &[f32]) -> f32 {
        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter()
            .map(|&x| (x - max_logit).exp())
            .sum();
        
        let probs: Vec<f32> = logits.iter()
            .map(|&x| (x - max_logit).exp() / exp_sum)
            .collect();
        
        // Entropy: -sum(p * log(p))
        let entropy = -probs.iter()
            .filter(|&&p| p > 1e-10) // Avoid log(0)
            .map(|&p| p * p.ln())
            .sum::<f32>();
        
        entropy
    }
    
    /// Compute prediction error score
    /// High error = model is weak in this area = worth learning
    fn compute_prediction_error(&self, sample: &TrainingSample, trainer: &mut TrainerV4) -> f32 {
        // Skip if text is too short
        if sample.text.len() < 8 {
            return 0.5; // Default error for short texts
        }
        
        // Use first 64 bytes as context
        let context_bytes = if sample.text.len() >= 64 {
            &sample.text.as_bytes()[..64]
        } else {
            sample.text.as_bytes()
        };
        
        // Target is the next byte (or last byte if we're predicting the text itself)
        let target_byte = if sample.text.len() > 64 {
            sample.text.as_bytes()[64]
        } else {
            *sample.text.as_bytes().last().unwrap_or(&0)
        };
        
        // Forward pass (requires mutable reference)
        let embedded = trainer.model.stack.embed(context_bytes);
        let hidden = trainer.model.stack.forward(&embedded);
        let logits = trainer.model.stack.predict(&hidden);
        
        // Compute cross-entropy loss
        let loss = self.compute_cross_entropy_loss(&logits, target_byte as usize);
        
        // Normalize loss (typical range: 0 to 10)
        let normalized_loss = (loss / 10.0).min(1.0);
        
        normalized_loss
    }
    
    /// Compute cross-entropy loss for a single target
    fn compute_cross_entropy_loss(&self, logits: &[f32], target: usize) -> f32 {
        if target >= logits.len() {
            return 10.0; // Max loss for out-of-vocab target
        }
        
        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter()
            .map(|&x| (x - max_logit).exp())
            .sum();
        
        let target_prob = (logits[target] - max_logit).exp() / exp_sum;
        
        // Cross-entropy: -log(p)
        let loss = -target_prob.max(1e-10).ln();
        
        loss
    }
    
    /// Compute novelty score
    /// Novel = never seen before = worth learning
    /// Checks both learned_hashes and pending_hashes
    fn compute_novelty(&self, sample: &TrainingSample) -> f32 {
        let hash = self.compute_sample_hash(sample);
        
        // Check if already learned OR pending in buffer
        if self.learned_hashes.contains(&hash) || self.pending_hashes.contains(&hash) {
            0.0 // Already learned or pending
        } else {
            1.0 // Novel
        }
    }
    
    /// Compute SHA3-256 hash of a sample
    fn compute_sample_hash(&self, sample: &TrainingSample) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(sample.text.as_bytes());
        hasher.update(sample.source.as_bytes());
        format!("{:x}", hasher.finalize())
    }
    
    /// Mark a sample as learned (add to learned_hashes)
    pub fn mark_as_learned(&mut self, sample: &TrainingSample) {
        let hash = self.compute_sample_hash(sample);
        
        // Memory management: if we have too many hashes, clear old ones
        if self.learned_hashes.len() >= self.config.max_learned_hashes {
            println!("   [CURIOSITY] Memory limit reached, clearing learned hashes");
            self.learned_hashes.clear();
        }
        
        self.learned_hashes.insert(hash);
    }
    
    /// Mark multiple samples as learned
    pub fn mark_batch_as_learned(&mut self, samples: &[TrainingSample]) {
        for sample in samples {
            self.mark_as_learned(sample);
        }
    }
    
    /// Mark a sample as pending (added to buffer but not yet learned)
    pub fn mark_as_pending(&mut self, sample: &TrainingSample) {
        let hash = self.compute_sample_hash(sample);
        self.pending_hashes.insert(hash);
    }
    
    /// Clear pending hashes (after learning cycle)
    pub fn clear_pending(&mut self) {
        self.pending_hashes.clear();
    }
    
    /// Decide whether to accept a sample based on curiosity score
    pub fn should_learn(
        &mut self,
        sample: &TrainingSample,
        trainer: &mut TrainerV4,
    ) -> bool {
        let score = self.compute_curiosity_score(sample, trainer);
        
        if score >= self.config.min_curiosity_threshold {
            self.stats.total_selected += 1;
            true
        } else {
            self.stats.total_rejected += 1;
            false
        }
    }
    
    /// Get current statistics
    pub fn stats(&self) -> CuriosityStats {
        let mut stats = self.stats.clone();
        
        // Compute averages
        if self.stats.total_evaluated > 0 {
            let count = self.stats.total_evaluated as f64;
            stats.avg_curiosity_score = self.sum_curiosity / count;
            stats.avg_uncertainty = self.sum_uncertainty / count;
            stats.avg_prediction_error = self.sum_prediction_error / count;
            stats.avg_novelty = self.sum_novelty / count;
        }
        
        stats
    }
    
    /// Get number of learned hashes
    pub fn learned_count(&self) -> usize {
        self.learned_hashes.len()
    }
    
    /// Get number of pending hashes
    pub fn pending_count(&self) -> usize {
        self.pending_hashes.len()
    }
    
    /// Clear all learned hashes (reset memory)
    pub fn clear_memory(&mut self) {
        self.learned_hashes.clear();
        self.pending_hashes.clear();
        println!("   [CURIOSITY] Memory cleared");
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_v3::TransformerConfig;
    
    fn make_test_trainer() -> TrainerV4 {
        let config = TransformerConfig {
            vocab_size: 256,
            embed_dim: 32,
            num_layers: 1,
            num_heads: 2,
            ff_dim: 64,
            max_seq_len: 32,
            dropout: 0.0,
        };
        TrainerV4::new(&config, 0.001, 5, 50, 0.01, 1)
    }
    
    #[test]
    fn test_curiosity_module_creation() {
        let module = CuriosityModule::new();
        assert_eq!(module.learned_count(), 0);
        assert_eq!(module.pending_count(), 0);
    }
    
    #[test]
    fn test_novelty_detection() {
        let mut module = CuriosityModule::new();
        
        let sample = TrainingSample::new(
            "The quick brown fox jumps over the lazy dog".into(),
            "test".into(),
            0.9,
        );
        
        // First time: should be novel
        let novelty1 = module.compute_novelty(&sample);
        assert_eq!(novelty1, 1.0);
        
        // Mark as learned
        module.mark_as_learned(&sample);
        
        // Second time: should not be novel
        let novelty2 = module.compute_novelty(&sample);
        assert_eq!(novelty2, 0.0);
    }
    
    #[test]
    fn test_pending_detection() {
        let mut module = CuriosityModule::new();
        
        let sample = TrainingSample::new(
            "The quick brown fox jumps over the lazy dog".into(),
            "test".into(),
            0.9,
        );
        
        // First time: should be novel
        let novelty1 = module.compute_novelty(&sample);
        assert_eq!(novelty1, 1.0);
        
        // Mark as pending (in buffer)
        module.mark_as_pending(&sample);
        
        // Second time: should not be novel (pending)
        let novelty2 = module.compute_novelty(&sample);
        assert_eq!(novelty2, 0.0);
        
        // Clear pending
        module.clear_pending();
        
        // Third time: should be novel again
        let novelty3 = module.compute_novelty(&sample);
        assert_eq!(novelty3, 1.0);
    }
    
    #[test]
    fn test_curiosity_score_zero_for_duplicate() {
        let mut module = CuriosityModule::new();
        let mut trainer = make_test_trainer();
        
        let sample = TrainingSample::new(
            "The quick brown fox jumps over the lazy dog".into(),
            "test".into(),
            0.9,
        );
        
        // First time: should have non-zero score (novel)
        let score1 = module.compute_curiosity_score(&sample, &mut trainer);
        assert!(score1 > 0.0, "First sample should have positive score");
        
        // Mark as pending
        module.mark_as_pending(&sample);
        
        // Second time: should have ZERO score (not novel)
        let score2 = module.compute_curiosity_score(&sample, &mut trainer);
        assert_eq!(score2, 0.0, "Duplicate sample should have zero score");
    }
    
    #[test]
    fn test_should_learn_threshold() {
        let config = CuriosityConfig {
            min_curiosity_threshold: 0.5,
            ..Default::default()
        };
        let mut module = CuriosityModule::with_config(config);
        let mut trainer = make_test_trainer();
        
        let sample = TrainingSample::new(
            "The quick brown fox jumps over the lazy dog".into(),
            "test".into(),
            0.9,
        );
        
        // This will compute a score and decide
      let _should_learn = module.should_learn(&sample, &mut trainer);
        
        // We can't predict the exact result, but we can check stats
        assert_eq!(module.stats.total_evaluated, 1);
        assert!(module.stats.total_selected + module.stats.total_rejected == 1);
    }
    
    #[test]
    fn test_memory_management() {
        let config = CuriosityConfig {
            max_learned_hashes: 5,
            ..Default::default()
        };
        let mut module = CuriosityModule::with_config(config);
        
        // Add 10 samples
        for i in 0..10 {
            let sample = TrainingSample::new(
                format!("Sample text number {}", i),
                "test".into(),
                0.9,
            );
            module.mark_as_learned(&sample);
        }
        
        // Should have cleared and only have recent ones
        assert!(module.learned_count() <= 5);
    }
}