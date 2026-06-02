// ============================================================================
// HAFA - src/epistemic.rs — EPISTEMIC EVALUATION ENGINE
// ============================================================================

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum EpistemicError {
    #[error("Invalid constraint values")]
    InvalidConstraints,
    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpistemicState {
    /// Confidence in truthfulness (0.0 to 1.0)
    pub confidence: f64,
    /// Whether claim is grounded in verified data/observation
    pub grounded: bool,
    /// Depth of inference chain (0 = direct, higher = more speculative)
    pub speculation_depth: u8,
    /// Willingness to acknowledge uncertainty (0.0 to 1.0)
    pub humility_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicConstraints {
    pub min_confidence: f64,
    pub max_speculation_depth: u8,
    pub require_grounding: bool,
}

#[derive(Debug, Clone)]
pub struct KnowledgeClaim {
    /// SHA3-256 hash of the raw content
    pub content_hash: String,
    /// Source type: "local", "ipfs", "web", "rss"
    pub source_type: String,
    /// Number of independent nodes/sources corroborating this claim
    pub corroborating_sources: u32,
    /// True if derived from direct measurement/observation
    pub is_direct_observation: bool,
}

// ============================================================================
// IMPLEMENTATION
// ============================================================================

impl Default for EpistemicConstraints {
    fn default() -> Self {
        Self {
            min_confidence: 0.85,
            max_speculation_depth: 2,
            require_grounding: true,
        }
    }
}

impl EpistemicConstraints {
    pub fn validate(&self) -> Result<(), EpistemicError> {
        if !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(EpistemicError::InvalidConstraints);
        }
        Ok(())
    }
}

impl EpistemicState {
    pub fn new(confidence: f64, grounded: bool, speculation_depth: u8, humility_score: f64) -> Self {
        Self {
            confidence: confidence.clamp(0.0, 1.0),
            grounded,
            speculation_depth,
            humility_score: humility_score.clamp(0.0, 1.0),
        }
    }

    /// Check if state meets minimum acceptance criteria
    pub fn is_acceptable(&self, constraints: &EpistemicConstraints) -> bool {
        self.confidence >= constraints.min_confidence
            && self.speculation_depth <= constraints.max_speculation_depth
            && (!constraints.require_grounding || self.grounded)
    }
}

// ============================================================================
// EVALUATION ENGINE
// ============================================================================

pub struct EpistemicEngine;

impl EpistemicEngine {
    /// Evaluate a knowledge claim against constraints
    pub fn evaluate(claim: &KnowledgeClaim, constraints: &EpistemicConstraints) -> EpistemicState {
        constraints.validate().unwrap_or_default();

        let base_confidence = if claim.is_direct_observation { 0.95 } else { 0.70 };
        let corroboration_bonus = (claim.corroborating_sources as f64 * 0.05).min(0.30);
        let confidence = (base_confidence + corroboration_bonus).min(1.0);

        let grounded = claim.is_direct_observation || claim.corroborating_sources > 0;
        let speculation_depth = if claim.is_direct_observation { 0 } else { 1 };
        
        // Humility inversely related to confidence, adjusted for speculation
        let humility_score = (1.0 - confidence) + (speculation_depth as f64 * 0.1);

        EpistemicState::new(confidence, grounded, speculation_depth, humility_score)
    }

    /// Update confidence based on new corroborating evidence
    pub fn update_with_evidence(
        mut state: EpistemicState,
        new_corroborations: u32,
    ) -> EpistemicState {
        let bonus = (new_corroborations as f64 * 0.03).min(0.20);
        state.confidence = (state.confidence + bonus).min(1.0);
        state.humility_score = (1.0 - state.confidence).max(0.0);
        state
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_observation_evaluation() {
        let claim = KnowledgeClaim {
            content_hash: "abc123".into(),
            source_type: "local".into(),
            corroborating_sources: 0,
            is_direct_observation: true,
        };
        let constraints = EpistemicConstraints::default();
        let state = EpistemicEngine::evaluate(&claim, &constraints);

        assert!(state.grounded);
        assert_eq!(state.speculation_depth, 0);
        assert!(state.confidence >= 0.90);
        assert!(state.is_acceptable(&constraints));
    }

    #[test]
    fn test_ungrounded_claim_rejection() {
        let claim = KnowledgeClaim {
            content_hash: "def456".into(),
            source_type: "web".into(),
            corroborating_sources: 0,
            is_direct_observation: false,
        };
        let constraints = EpistemicConstraints::default();
        let state = EpistemicEngine::evaluate(&claim, &constraints);

        assert!(!state.grounded);
        assert!(!state.is_acceptable(&constraints));
    }

    #[test]
    fn test_evidence_boost() {
        let mut state = EpistemicState::new(0.75, false, 1, 0.25);
        let updated = EpistemicEngine::update_with_evidence(state.clone(), 5);
        assert!(updated.confidence > state.confidence);
    }
}