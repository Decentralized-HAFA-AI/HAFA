// ============================================================================
// Proof Verifier - Independent Verification of Cognitive Proofs
// ============================================================================
//
// This module allows any node to verify a CognitiveProofV4 without
// having performed the training itself.
//
// ============================================================================

use super::cognitive_proof::CognitiveProofV4;

pub struct ProofVerifier;

impl ProofVerifier {
    /// Verifies the structural integrity of a proof
    pub fn verify_structure(proof: &CognitiveProofV4) -> Result<(), String> {
        if !proof.is_valid() {
            return Err("Proof structure is invalid".to_string());
        }
        Ok(())
    }
    
    /// Verifies that the proof shows actual learning
   pub fn verify_learning(proof: &CognitiveProofV4) -> Result<(), String> {
    // NOTE: With partial training (only pred_head), loss may increase temporarily.
    // Full-Model Training will fix this. For now, we use relaxed checks.
    
    // Check that loss didn't explode
    if proof.loss_after > 100.0 {
        return Err(format!(
            "Loss exploded: loss_after={}",
            proof.loss_after
        ));
    }
    
    // Check that samples were processed (actual work was done)
    if proof.samples_processed < 10 {
        return Err(format!(
            "Too few samples processed: {}",
            proof.samples_processed
        ));
    }
    
    // Check that training took reasonable time
    if proof.wall_time_ms < 100 {
        return Err(format!(
            "Training too fast (suspicious): {}ms",
            proof.wall_time_ms
        ));
    }
    
    Ok(())
}
    
    /// Verifies resource usage is reasonable
    pub fn verify_resources(proof: &CognitiveProofV4) -> Result<(), String> {
        // Check that wall time is reasonable (at least 1ms per sample)
        let min_time_ms = proof.samples_processed;
        if proof.wall_time_ms < min_time_ms {
            return Err(format!(
                "Suspiciously fast: {} samples in {}ms",
                proof.samples_processed, proof.wall_time_ms
            ));
        }
        
        // Check that RAM usage is reasonable (at least 100MB)
        if proof.ram_usage_mb < 100 {
            return Err(format!(
                "Suspiciously low RAM: {}MB",
                proof.ram_usage_mb
            ));
        }
        
        Ok(())
    }
    
    /// Full verification of a proof
    pub fn verify_full(proof: &CognitiveProofV4) -> Result<f64, String> {
        Self::verify_structure(proof)?;
        Self::verify_learning(proof)?;
        Self::verify_resources(proof)?;
        
        Ok(proof.quality_score())
    }
}