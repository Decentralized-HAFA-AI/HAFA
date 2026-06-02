// ============================================================================
// HAFA - src/evolution.rs — SELF-EVOLUTION & PROPOSAL ENGINE
// ============================================================================

use crate::config::Config;
use crate::epistemic::{EpistemicEngine, EpistemicConstraints, EpistemicState};
use thiserror::Error;
use serde::{Deserialize, Serialize};

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum EvolutionError {
    #[error("Proposal validation failed: {0}")]
    InvalidProposal(String),
    #[error("Sandbox execution failed: {0}")]
    SandboxError(String),
    #[error("Human approval required for high-risk changes")]
    ApprovalRequired,
    #[error("Epistemic confidence too low: {0}")]
    LowConfidence(f64),
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,      // Client-only changes, no core impact
    Medium,   // Learning/memory module changes, requires testing
    High,     // Core consensus/crypto changes, requires human approval
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionProposal {
    /// Unique identifier (hash of proposal content)
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Target module: "learning", "api", "blockchain", etc.
    pub target_module: String,
    /// Proposed code diff (unified diff format)
    pub code_diff: String,
    /// Epistemic confidence in proposal correctness
    pub epistemic_confidence: f64,
    /// Assessed risk level
    pub risk_level: RiskLevel,
    /// Whether this change could be misused
    pub has_potential_misuse: bool,
    /// Mitigations if misuse is possible
    pub misuse_mitigations: Vec<String>,
    /// Whether human approval is mandatory
    pub requires_human_approval: bool,
}

#[derive(Debug, Clone)]
pub struct SandboxResult {
    pub success: bool,
    pub output: String,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    pub cpu_cycles: u64,
    pub memory_bytes: u64,
    pub execution_time_ms: u64,
}

// ============================================================================
// IMPLEMENTATION
// ============================================================================

impl RiskLevel {
    pub fn requires_approval(&self, is_core_module: bool) -> bool {
        match self {
            RiskLevel::High => true,
            RiskLevel::Medium => is_core_module,
            RiskLevel::Low => false,
        }
    }
}

impl EvolutionProposal {
    pub fn new(
        description: String,
        target_module: String,
        code_diff: String,
        confidence: f64,
        risk: RiskLevel,
    ) -> Self {
        let id = crate::crypto::hash_sha3_256(
            format!("{}|{}|{}", description, target_module, code_diff).as_bytes(),
        );

        let requires_human_approval = risk.requires_approval(
            matches!(target_module.as_str(), "crypto" | "blockchain" | "network" | "config"),
        );

        Self {
            id,
            description,
            target_module,
            code_diff,
            epistemic_confidence: confidence.clamp(0.0, 1.0),
            risk_level: risk,
            has_potential_misuse: false,
            misuse_mitigations: Vec::new(),
            requires_human_approval,
        }
    }

    pub fn validate(&self, constraints: &EpistemicConstraints) -> Result<(), EvolutionError> {
        if self.epistemic_confidence < constraints.min_confidence {
            return Err(EvolutionError::LowConfidence(self.epistemic_confidence));
        }
        if self.code_diff.is_empty() {
            return Err(EvolutionError::InvalidProposal("Empty diff".into()));
        }
        if self.target_module.is_empty() {
            return Err(EvolutionError::InvalidProposal("Empty target module".into()));
        }
        Ok(())
    }
}

pub struct EvolutionEngine {
    config: Config,
    epistemic: EpistemicEngine,
    constraints: EpistemicConstraints,
}

impl EvolutionEngine {
    pub fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
            epistemic: EpistemicEngine,
            constraints: EpistemicConstraints::default(),
        }
    }

    /// Generate a proposal for code evolution
    pub fn propose_change(
        &self,
        description: String,
        target_module: String,
        code_diff: String,
        risk: RiskLevel,
    ) -> Result<EvolutionProposal, EvolutionError> {
        let confidence = self.assess_confidence(&target_module, &code_diff);
        
        let mut proposal = EvolutionProposal::new(
            description,
            target_module,
            code_diff,
            confidence,
            risk,
        );

        proposal.validate(&self.constraints)?;
        Ok(proposal)
    }

    /// Assess epistemic confidence in a proposed change
    fn assess_confidence(&self, _module: &str, _diff: &str) -> f64 {
        // Placeholder: In production, analyze diff complexity, test coverage, etc.
        0.85
    }

    /// Run proposal in sandbox (simulated)
    pub async fn run_sandbox(&self, proposal: &EvolutionProposal) -> Result<SandboxResult, EvolutionError> {
        // Genesis Edition: Simulated sandbox
        // Production: WASM/VM isolation with resource limits
        
        if proposal.has_potential_misuse && proposal.misuse_mitigations.is_empty() {
            return Err(EvolutionError::SandboxError("No mitigations for potential misuse".into()));
        }

        // Simulate execution
        Ok(SandboxResult {
            success: true,
            output: "Sandbox test passed".into(),
            resource_usage: ResourceUsage {
                cpu_cycles: 1000,
                memory_bytes: 1024 * 1024,
                execution_time_ms: 50,
            },
        })
    }

    /// Check if proposal can be auto-applied or needs human approval
    pub fn can_auto_apply(&self, proposal: &EvolutionProposal) -> bool {
        if proposal.requires_human_approval {
            return false;
        }
        if proposal.epistemic_confidence < self.constraints.min_confidence {
            return false;
        }
        true
    }

    /// Submit proposal for human review (returns proposal ID)
    pub fn submit_for_approval(&self, proposal: EvolutionProposal) -> String {
        // In production: broadcast to governance channel / DAO
        proposal.id.clone()
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        let mut cfg = Config::default();
        cfg.founder.genesis_pubkey_hex = "a".repeat(64);
        cfg
    }

    #[test]
    fn test_risk_level_approval_logic() {
        assert!(!RiskLevel::Low.requires_approval(false));
        assert!(!RiskLevel::Low.requires_approval(true));
        assert!(!RiskLevel::Medium.requires_approval(false));
        assert!(RiskLevel::Medium.requires_approval(true));
        assert!(RiskLevel::High.requires_approval(false));
        assert!(RiskLevel::High.requires_approval(true));
    }

    #[test]
    fn test_proposal_validation() {
        let proposal = EvolutionProposal::new(
            "Test change".into(),
            "learning".into(),
            "+fn test() {}".into(),
            0.90,
            RiskLevel::Low,
        );
        let constraints = EpistemicConstraints::default();
        assert!(proposal.validate(&constraints).is_ok());
    }

    #[test]
    fn test_low_confidence_rejection() {
        let proposal = EvolutionProposal::new(
            "Risky change".into(),
            "crypto".into(),
            "+fn hack() {}".into(),
            0.50,
            RiskLevel::High,
        );
        let constraints = EpistemicConstraints::default();
        assert!(matches!(proposal.validate(&constraints), Err(EvolutionError::LowConfidence(_))));
    }

    #[tokio::test]
    async fn test_sandbox_simulation() {
        let config = test_config();
        let engine = EvolutionEngine::new(&config);
        let proposal = EvolutionProposal::new(
            "Safe change".into(),
            "api".into(),
            "+fn safe() {}".into(),
            0.95,
            RiskLevel::Low,
        );
        let result = engine.run_sandbox(&proposal).await;
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }
}