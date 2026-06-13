// ============================================================================
// Verifiable Cognitive Proof (PoUCW) - Enhanced Version
// ============================================================================
//
// Cryptographically verifiable proof of useful cognitive work.
// Includes gradient commitment for full reproducibility.
//
// ============================================================================

use serde::Serialize;
use sha3::{Sha3_256, Digest};

#[derive(Serialize, Clone, Debug)]
pub struct CognitiveProofV4 {
    // Model state
    pub model_hash_before: String,
    pub model_hash_after: String,
    
    // Dataset commitment
    pub dataset_commitment: String,
    
    // Gradient commitment (hash of accumulated gradients)
    pub gradient_commitment: String,
    
    // Training metrics
    pub loss_before: f32,
    pub loss_after: f32,
    pub ema_loss_after: f32,
    pub samples_processed: u64,
    
    // Resource usage
    pub wall_time_ms: u64,
    pub cpu_usage_percent: f64,
    pub ram_usage_mb: u64,
}

impl CognitiveProofV4 {
    /// Computes a deterministic SHA3-256 hash of a byte slice
    pub fn compute_hash(data: &[u8]) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }
    
    /// Computes a combined proof hash (for blockchain submission)
    pub fn compute_proof_hash(&self) -> String {
        let mut hasher = Sha3_256::new();
        
        // Hash all fields deterministically
        hasher.update(self.model_hash_before.as_bytes());
        hasher.update(self.model_hash_after.as_bytes());
        hasher.update(self.dataset_commitment.as_bytes());
        hasher.update(self.gradient_commitment.as_bytes());
        hasher.update(&self.loss_before.to_le_bytes());
        hasher.update(&self.loss_after.to_le_bytes());
        hasher.update(&self.ema_loss_after.to_le_bytes());
        hasher.update(&self.samples_processed.to_le_bytes());
        hasher.update(&self.wall_time_ms.to_le_bytes());
        
        format!("{:x}", hasher.finalize())
    }
    
    /// Validates the proof structure (basic sanity checks)
    pub fn is_valid(&self) -> bool {
        // Check that hashes are not empty
        if self.model_hash_before.is_empty() || self.model_hash_after.is_empty() {
            return false;
        }
        if self.dataset_commitment.is_empty() || self.gradient_commitment.is_empty() {
            return false;
        }
        
        // Check that model actually changed
        if self.model_hash_before == self.model_hash_after {
            return false;
        }
        
        // Check that loss didn't explode catastrophically
        if self.loss_after > 100.0 || self.loss_after.is_nan() || self.loss_after.is_infinite() {
            return false;
        }
        
        // Check that samples were actually processed
        if self.samples_processed == 0 {
            return false;
        }
        
        true
    }
    
    /// Calculates a quality score (0.0 to 1.0) based on proof metrics
    /// FIXED: More balanced weights, relaxed efficiency requirements
    pub fn quality_score(&self) -> f64 {
        let mut score: f64 = 0.0;
        
        // 1. Loss improvement (50% weight) - MOST IMPORTANT
        if self.loss_before > 0.0 {
            let improvement = (self.loss_before as f64 - self.loss_after as f64) / self.loss_before as f64;
            score += (improvement.max(0.0) * 0.5).min(0.5);
        }
        
        // 2. Samples processed (25% weight)
        let samples_score = (self.samples_processed as f64 / 1000.0).min(1.0);
        score += samples_score * 0.25;
        
        // 3. Time efficiency (15% weight) - RELAXED for full-model training
        if self.wall_time_ms > 0 {
            // Acceptable: 100ms per sample (full-model training is slower)
            let acceptable_time = self.samples_processed as f64 * 100.0;
            let time_ratio = self.wall_time_ms as f64 / acceptable_time;
            let efficiency_score = (1.0 / time_ratio.max(1.0)).min(1.0);
            score += efficiency_score * 0.15;
        }
        
        // 4. RAM efficiency (10% weight) - RELAXED
        if self.ram_usage_mb > 0 {
            let ram_efficiency = self.samples_processed as f64 / self.ram_usage_mb as f64;
            let ram_score = (ram_efficiency / 0.01).min(1.0);
            score += ram_score * 0.1;
        }
        
        score.min(1.0)
    }
}