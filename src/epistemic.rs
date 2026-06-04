// ============================================================================
// HAFA - src/epistemic.rs — EPISTEMIC EVALUATION ENGINE (ADVANCED)
// ============================================================================
//
// Advanced epistemic filtering system for validating knowledge claims.
// Features:
// - Source reputation tracking
// - Temporal decay for outdated information
// - Evidence chain tracking
// - Contradiction detection
// - Batch evaluation
// - Advanced confidence calculation
//
// ============================================================================

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use thiserror::Error;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum EpistemicError {
    #[error("Invalid constraint values: {0}")]
    InvalidConstraints(String),
    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),
    #[error("Source not found: {0}")]
    SourceNotFound(String),
    #[error("Contradiction detected: {0}")]
    ContradictionDetected(String),
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Represents the epistemic state of a knowledge claim
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
    /// Number of evidence pieces supporting this claim
    pub evidence_count: u32,
    /// Level of contradiction with other claims (0.0 = no contradiction, 1.0 = full contradiction)
    pub contradiction_level: f64,
    /// Temporal weight (decays over time)
    pub temporal_weight: f64,
    /// Overall weight for learning (combination of all factors)
    pub learning_weight: f64,
}

/// Constraints for accepting knowledge claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicConstraints {
    pub min_confidence: f64,
    pub max_speculation_depth: u8,
    pub require_grounding: bool,
    pub max_contradiction_level: f64,
    pub min_temporal_weight: f64,
    pub temporal_decay_factor: f64,
    pub min_source_reputation: f64,
}

/// Represents a source of information with reputation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReputation {
    pub source_id: String,
    pub source_type: String,
    pub credibility_score: f64,
    pub total_claims: u32,
    pub verified_claims: u32,
    pub last_updated: DateTime<Utc>,
}

/// A single piece of evidence supporting a claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub evidence_id: String,
    pub source_id: String,
    pub timestamp: DateTime<Utc>,
    pub strength: f64,
    pub content_hash: String,
}

/// A knowledge claim with full metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeClaim {
    /// SHA3-256 hash of the raw content
    pub content_hash: String,
    /// Source type: "local", "ipfs", "web", "rss", "sensor"
    pub source_type: String,
    /// Source identifier
    pub source_id: String,
    /// Number of independent nodes/sources corroborating this claim
    pub corroborating_sources: u32,
    /// True if derived from direct measurement/observation
    pub is_direct_observation: bool,
    /// Timestamp when claim was created
    pub timestamp: DateTime<Utc>,
    /// Evidence chain supporting this claim
    pub evidence_chain: Vec<Evidence>,
    /// Semantic category (for contradiction detection)
    pub category: String,
}

// ============================================================================
// IMPLEMENTATION
// ============================================================================

impl Default for EpistemicConstraints {
    fn default() -> Self {
        Self {
            min_confidence: 0.75,
            max_speculation_depth: 3,
            require_grounding: true,
            max_contradiction_level: 0.6,
            min_temporal_weight: 0.3,
            temporal_decay_factor: 0.01, // Decay per hour
            min_source_reputation: 0.5,
        }
    }
}

impl EpistemicConstraints {
    pub fn validate(&self) -> Result<(), EpistemicError> {
        if !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(EpistemicError::InvalidConstraints(
                "min_confidence must be between 0.0 and 1.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.max_contradiction_level) {
            return Err(EpistemicError::InvalidConstraints(
                "max_contradiction_level must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.temporal_decay_factor < 0.0 {
            return Err(EpistemicError::InvalidConstraints(
                "temporal_decay_factor must be non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

impl EpistemicState {
    pub fn new(
        confidence: f64,
        grounded: bool,
        speculation_depth: u8,
        humility_score: f64,
        evidence_count: u32,
        contradiction_level: f64,
        temporal_weight: f64,
    ) -> Self {
        let confidence = confidence.clamp(0.0, 1.0);
        let humility_score = humility_score.clamp(0.0, 1.0);
        let contradiction_level = contradiction_level.clamp(0.0, 1.0);
        let temporal_weight = temporal_weight.clamp(0.0, 1.0);

        // Calculate learning weight as combination of all factors
        let learning_weight = confidence * temporal_weight * (1.0 - contradiction_level * 0.5);

        Self {
            confidence,
            grounded,
            speculation_depth,
            humility_score,
            evidence_count,
            contradiction_level,
            temporal_weight,
            learning_weight,
        }
    }

    /// Check if state meets minimum acceptance criteria
    pub fn is_acceptable(&self, constraints: &EpistemicConstraints) -> bool {
        self.confidence >= constraints.min_confidence
            && self.speculation_depth <= constraints.max_speculation_depth
            && (!constraints.require_grounding || self.grounded)
            && self.contradiction_level <= constraints.max_contradiction_level
            && self.temporal_weight >= constraints.min_temporal_weight
    }
}

impl SourceReputation {
    pub fn new(source_id: String, source_type: String) -> Self {
        Self {
            source_id,
            source_type,
            credibility_score: 0.5, // Start with neutral reputation
            total_claims: 0,
            verified_claims: 0,
            last_updated: Utc::now(),
        }
    }

    /// Update reputation based on claim verification
    pub fn update(&mut self, was_verified: bool) {
        self.total_claims += 1;
        if was_verified {
            self.verified_claims += 1;
        }

        // Exponential moving average for credibility
        let verification_rate = self.verified_claims as f64 / self.total_claims as f64;
        let alpha = 0.1; // Smoothing factor
        self.credibility_score = alpha * verification_rate + (1.0 - alpha) * self.credibility_score;
        self.last_updated = Utc::now();
    }

    /// Get reputation bonus for confidence calculation
    pub fn get_reputation_bonus(&self) -> f64 {
        // Bonus ranges from -0.2 to +0.3 based on credibility
        (self.credibility_score - 0.5) * 0.5
    }
}

impl KnowledgeClaim {
    pub fn new(
        content: &[u8],
        source_type: String,
        source_id: String,
        is_direct_observation: bool,
        category: String,
    ) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(content);
        let content_hash = hex::encode(hasher.finalize());

        Self {
            content_hash,
            source_type,
            source_id,
            corroborating_sources: 0,
            is_direct_observation,
            timestamp: Utc::now(),
            evidence_chain: Vec::new(),
            category,
        }
    }

    /// Add evidence to the claim
    pub fn add_evidence(&mut self, evidence: Evidence) {
        self.evidence_chain.push(evidence);
        self.corroborating_sources += 1;
    }

    /// Calculate age in hours
    pub fn age_hours(&self) -> f64 {
        let now = Utc::now();
        let duration = now.signed_duration_since(self.timestamp);
        duration.num_seconds() as f64 / 3600.0
    }
}

// ============================================================================
// EVALUATION ENGINE
// ============================================================================

pub struct EpistemicEngine;

impl EpistemicEngine {
    /// Evaluate a knowledge claim against constraints
    pub fn evaluate(
        claim: &KnowledgeClaim,
        constraints: &EpistemicConstraints,
        source_reputation: Option<&SourceReputation>,
    ) -> EpistemicState {
        constraints.validate().unwrap_or_default();

        // 1. Base confidence based on observation type
        let base_confidence = if claim.is_direct_observation {
            0.95
        } else {
            match claim.source_type.as_str() {
                "local" => 0.85,
                "sensor" => 0.90,
                "ipfs" => 0.75,
                "web" => 0.65,
                "rss" => 0.60,
                _ => 0.50,
            }
        };

        // 2. Source reputation bonus
        let reputation_bonus = source_reputation
            .map(|r| r.get_reputation_bonus())
            .unwrap_or(0.0);

        // 3. Corroboration bonus (diminishing returns)
        let corroboration_bonus = (claim.corroborating_sources as f64 * 0.05)
            .min(0.25)
            * (1.0 / (1.0 + claim.corroborating_sources as f64 * 0.1));

        // 4. Evidence strength bonus
        let evidence_bonus = if claim.evidence_chain.is_empty() {
            0.0
        } else {
            let avg_strength: f64 = claim
                .evidence_chain
                .iter()
                .map(|e| e.strength)
                .sum::<f64>()
                / claim.evidence_chain.len() as f64;
            avg_strength * 0.15
        };

        // 5. Temporal decay
        let age_hours = claim.age_hours();
        let temporal_weight = (-constraints.temporal_decay_factor * age_hours).exp();

        // 6. Calculate final confidence
        let confidence = (base_confidence + reputation_bonus + corroboration_bonus + evidence_bonus)
            .clamp(0.0, 1.0)
            * temporal_weight;

        // 7. Determine grounded status
        let grounded = claim.is_direct_observation
            || claim.corroborating_sources > 0
            || !claim.evidence_chain.is_empty();

        // 8. Speculation depth
        let speculation_depth = if claim.is_direct_observation {
            0
        } else if claim.evidence_chain.is_empty() {
            2
        } else {
            1
        };

        // 9. Humility score (more sophisticated)
        let uncertainty_factor = 1.0 - confidence;
        let speculation_penalty = speculation_depth as f64 * 0.05;
        let humility_score = (uncertainty_factor + speculation_penalty).clamp(0.0, 1.0);

        // 10. Contradiction level (placeholder - would need semantic analysis)
        let contradiction_level = 0.0; // TODO: Implement contradiction detection

        EpistemicState::new(
            confidence,
            grounded,
            speculation_depth,
            humility_score,
            claim.evidence_chain.len() as u32,
            contradiction_level,
            temporal_weight,
        )
    }

    /// Update confidence based on new corroborating evidence
    pub fn update_with_evidence(
        mut state: EpistemicState,
        new_evidence: &[Evidence],
    ) -> EpistemicState {
        if new_evidence.is_empty() {
            return state;
        }

        let avg_strength: f64 = new_evidence.iter().map(|e| e.strength).sum::<f64>()
            / new_evidence.len() as f64;

        let bonus = (new_evidence.len() as f64 * avg_strength * 0.03).min(0.20);
        state.confidence = (state.confidence + bonus).min(1.0);
        state.evidence_count += new_evidence.len() as u32;
        state.humility_score = ((1.0 - state.confidence) + (state.speculation_depth as f64 * 0.05))
            .clamp(0.0, 1.0);

        // Recalculate learning weight
        state.learning_weight =
            state.confidence * state.temporal_weight * (1.0 - state.contradiction_level * 0.5);

        state
    }

    /// Evaluate multiple claims in batch
    pub fn evaluate_batch(
        claims: &[KnowledgeClaim],
        constraints: &EpistemicConstraints,
        source_reputations: &std::collections::HashMap<String, SourceReputation>,
    ) -> Vec<EpistemicState> {
        claims
            .iter()
            .map(|claim| {
                let reputation = source_reputations.get(&claim.source_id);
                Self::evaluate(claim, constraints, reputation)
            })
            .collect()
    }

    /// Detect potential contradictions between claims
    pub fn detect_contradictions(
        claims: &[KnowledgeClaim],
        states: &[EpistemicState],
    ) -> Vec<(usize, usize, f64)> {
        let mut contradictions = Vec::new();

        for i in 0..claims.len() {
            for j in (i + 1)..claims.len() {
                // Simple heuristic: same category but different content hash
                if claims[i].category == claims[j].category
                    && claims[i].content_hash != claims[j].content_hash
                {
                    // Contradiction strength based on confidence of both claims
                    let contradiction_strength =
                        (states[i].confidence + states[j].confidence) / 2.0;
                    contradictions.push((i, j, contradiction_strength));
                }
            }
        }

        contradictions
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
        let claim = KnowledgeClaim::new(
            b"Temperature is 25C",
            "sensor".into(),
            "sensor_001".into(),
            true,
            "temperature".into(),
        );
        let constraints = EpistemicConstraints::default();
        let state = EpistemicEngine::evaluate(&claim, &constraints, None);

        assert!(state.grounded);
        assert_eq!(state.speculation_depth, 0);
        assert!(state.confidence >= 0.90);
        assert!(state.is_acceptable(&constraints));
    }

    #[test]
    fn test_ungrounded_claim_rejection() {
        let claim = KnowledgeClaim::new(
            b"Unverified claim",
            "web".into(),
            "unknown_source".into(),
            false,
            "general".into(),
        );
        let constraints = EpistemicConstraints::default();
        let state = EpistemicEngine::evaluate(&claim, &constraints, None);

        assert!(!state.grounded);
        assert!(!state.is_acceptable(&constraints));
    }

    #[test]
    fn test_source_reputation_impact() {
        let claim = KnowledgeClaim::new(
            b"Test claim",
            "web".into(),
            "trusted_source".into(),
            false,
            "test".into(),
        );

        let mut reputation = SourceReputation::new("trusted_source".into(), "web".into());
        reputation.credibility_score = 0.9;

        let constraints = EpistemicConstraints::default();

        let state_without_rep = EpistemicEngine::evaluate(&claim, &constraints, None);
        let state_with_rep = EpistemicEngine::evaluate(&claim, &constraints, Some(&reputation));

        assert!(state_with_rep.confidence > state_without_rep.confidence);
    }

    #[test]
    fn test_temporal_decay() {
        let mut claim = KnowledgeClaim::new(
            b"Old claim",
            "local".into(),
            "local_001".into(),
            true,
            "test".into(),
        );

        // Simulate old timestamp (100 hours ago)
        claim.timestamp = Utc::now() - chrono::Duration::hours(100);

        let constraints = EpistemicConstraints::default();
        let state = EpistemicEngine::evaluate(&claim, &constraints, None);

        assert!(state.temporal_weight < 0.5); // Should decay significantly
    }

    #[test]
    fn test_evidence_boost() {
        let claim = KnowledgeClaim::new(
            b"Claim with evidence",
            "web".into(),
            "web_001".into(),
            false,
            "test".into(),
        );
        let constraints = EpistemicConstraints::default();
        let state_before = EpistemicEngine::evaluate(&claim, &constraints, None);

        let new_evidence = vec![
            Evidence {
                evidence_id: "ev1".into(),
                source_id: "source1".into(),
                timestamp: Utc::now(),
                strength: 0.8,
                content_hash: "hash1".into(),
            },
            Evidence {
                evidence_id: "ev2".into(),
                source_id: "source2".into(),
                timestamp: Utc::now(),
                strength: 0.9,
                content_hash: "hash2".into(),
            },
        ];

        let state_after = EpistemicEngine::update_with_evidence(state_before.clone(), &new_evidence);
        assert!(state_after.confidence > state_before.confidence);
        assert_eq!(state_after.evidence_count, 2);
    }

    #[test]
    fn test_source_reputation_update() {
        let mut reputation = SourceReputation::new("test_source".into(), "web".into());
        assert_eq!(reputation.credibility_score, 0.5);

        reputation.update(true);
        assert!(reputation.credibility_score > 0.5);

        reputation.update(false);
        // Should decrease slightly but not below initial
        assert!(reputation.credibility_score > 0.4);
    }

    #[test]
    fn test_batch_evaluation() {
        let claims = vec![
            KnowledgeClaim::new(b"Claim 1", "local".into(), "src1".into(), true, "cat1".into()),
            KnowledgeClaim::new(b"Claim 2", "web".into(), "src2".into(), false, "cat2".into()),
        ];

        let constraints = EpistemicConstraints::default();
        let reputations = std::collections::HashMap::new();

        let states = EpistemicEngine::evaluate_batch(&claims, &constraints, &reputations);
        assert_eq!(states.len(), 2);
    }
}