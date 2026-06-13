// ============================================================================
// HAFA - src/evolution.rs — SELF-EVOLUTION & PROPOSAL ENGINE (ADVANCED)
// ============================================================================
//
// Advanced self-evolution system with:
// - Proposal lifecycle management
// - DAO voting mechanism
// - Enhanced sandbox with resource limits
// - Audit trail and history
// - Auto-merge logic
// - Integration with blockchain
//
// ============================================================================

use crate::config::Config;
use crate::crypto::hash_sha3_256;
use crate::epistemic::EpistemicConstraints;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

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
    #[error("Epistemic confidence too low: {0:.2}")]
    LowConfidence(f64),
    #[error("Proposal not found: {0}")]
    ProposalNotFound(String),
    #[error("Proposal already in terminal state: {0}")]
    AlreadyFinalized(String),
    #[error("Insufficient votes: {0} < {1}")]
    InsufficientVotes(u32, u32),
    #[error("Sandbox resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,    // Client-only changes, no core impact
    Medium, // Learning/memory module changes, requires testing
    High,   // Core consensus/crypto changes, requires human approval
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    Pending,     // Awaiting review/voting
    UnderReview, // Being reviewed by humans
    Approved,    // Approved by DAO/humans
    Rejected,    // Rejected by DAO/humans
    Applied,     // Successfully applied to codebase
    Failed,      // Failed during application
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
    /// Current status
    pub status: ProposalStatus,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Votes for approval
    pub votes_for: u32,
    /// Votes against
    pub votes_against: u32,
    /// Addresses that have voted
    pub voters: Vec<String>,
    /// Sandbox test results
    pub sandbox_result: Option<SandboxResult>,
    /// Audit trail
    pub audit_log: Vec<AuditEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub resource_usage: ResourceUsage,
    pub test_results: Vec<TestResult>,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_name: String,
    pub passed: bool,
    pub duration_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_cycles: u64,
    pub memory_bytes: u64,
    pub execution_time_ms: u64,
    pub peak_memory_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub action: String,
    pub actor: String,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    pub voter: String,
    pub approve: bool,
    pub timestamp: DateTime<Utc>,
    pub reasoning: Option<String>,
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

    pub fn min_confidence_required(&self) -> f64 {
        match self {
            RiskLevel::Low => 0.70,
            RiskLevel::Medium => 0.85,
            RiskLevel::High => 0.95,
        }
    }

    pub fn sandbox_timeout_ms(&self) -> u64 {
        match self {
            RiskLevel::Low => 5000,
            RiskLevel::Medium => 10000,
            RiskLevel::High => 30000,
        }
    }

    pub fn max_memory_bytes(&self) -> u64 {
        match self {
            RiskLevel::Low => 64 * 1024 * 1024,      // 64 MB
            RiskLevel::Medium => 256 * 1024 * 1024,  // 256 MB
            RiskLevel::High => 1024 * 1024 * 1024,   // 1 GB
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
        let id = hash_sha3_256(
            format!("{}|{}|{}", description, target_module, code_diff).as_bytes(),
        );

        let requires_human_approval = risk.requires_approval(
            matches!(
                target_module.as_str(),
                "crypto" | "blockchain" | "network" | "config"
            ),
        );

        let now = Utc::now();

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
            status: ProposalStatus::Pending,
            created_at: now,
            updated_at: now,
            votes_for: 0,
            votes_against: 0,
            voters: Vec::new(),
            sandbox_result: None,
            audit_log: vec![AuditEntry {
                timestamp: now,
                action: "created".to_string(),
                actor: "system".to_string(),
                details: format!("Proposal created with confidence {:.2}", confidence),
            }],
        }
    }

    pub fn validate(&self, constraints: &EpistemicConstraints) -> Result<(), EvolutionError> {
        let min_confidence = self.risk_level.min_confidence_required();
        if self.epistemic_confidence < min_confidence {
            return Err(EvolutionError::LowConfidence(self.epistemic_confidence));
        }
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

    pub fn add_audit_entry(&mut self, action: String, actor: String, details: String) {
        self.audit_log.push(AuditEntry {
            timestamp: Utc::now(),
            action,
            actor,
            details,
        });
        self.updated_at = Utc::now();
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            ProposalStatus::Applied | ProposalStatus::Rejected | ProposalStatus::Failed
        )
    }
}

// ============================================================================
// EVOLUTION ENGINE
// ============================================================================

pub struct EvolutionEngine {
    #[allow(dead_code)]
    config: Config,
    constraints: EpistemicConstraints,
    proposals: Arc<DashMap<String, EvolutionProposal>>,
}

impl EvolutionEngine {
    pub fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
            constraints: EpistemicConstraints::default(),
            proposals: Arc::new(DashMap::new()),
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

        let proposal = EvolutionProposal::new(description, target_module, code_diff, confidence, risk);

        proposal.validate(&self.constraints)?;

        // Store proposal
        self.proposals.insert(proposal.id.clone(), proposal.clone());

        Ok(proposal)
    }

    /// Assess epistemic confidence in a proposed change
    fn assess_confidence(&self, module: &str, diff: &str) -> f64 {
        // Advanced confidence assessment based on multiple factors
let mut confidence: f64 = 0.85;

        // Factor 1: Module risk
        let is_core = matches!(module, "crypto" | "blockchain" | "network" | "config");
        if is_core {
            confidence -= 0.10;
        }

        // Factor 2: Diff complexity (simple heuristic: line count)
        let line_count = diff.lines().count();
        if line_count > 100 {
            confidence -= 0.05;
        } else if line_count < 10 {
            confidence += 0.05;
        }

        // Factor 3: Presence of tests in diff
        if diff.contains("#[test]") || diff.contains("test_") {
            confidence += 0.05;
        }

        // Factor 4: Dangerous patterns
        let dangerous_patterns = ["unsafe", "transmute", "unwrap()", "panic!"];
        for pattern in dangerous_patterns {
            if diff.contains(pattern) {
                confidence -= 0.03;
            }
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Run proposal in enhanced sandbox
    pub async fn run_sandbox(
        &self,
        proposal: &EvolutionProposal,
    ) -> Result<SandboxResult, EvolutionError> {
        if proposal.has_potential_misuse && proposal.misuse_mitigations.is_empty() {
            return Err(EvolutionError::SandboxError(
                "No mitigations for potential misuse".into(),
            ));
        }

        let timeout_ms = proposal.risk_level.sandbox_timeout_ms();
        let max_memory = proposal.risk_level.max_memory_bytes();

        // Simulate sandbox execution with resource limits
        let start_time = std::time::Instant::now();

        // Simulate test execution
        let test_results = self.simulate_tests(proposal).await?;

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        if execution_time_ms > timeout_ms {
            return Err(EvolutionError::ResourceLimitExceeded(format!(
                "Execution time {}ms exceeded timeout {}ms",
                execution_time_ms, timeout_ms
            )));
        }

        // Simulate resource usage
        let resource_usage = ResourceUsage {
            cpu_cycles: 1000000 + (proposal.code_diff.len() as u64 * 100),
            memory_bytes: 1024 * 1024 * (1 + proposal.code_diff.lines().count() as u64 / 100),
            execution_time_ms,
            peak_memory_bytes: 2 * 1024 * 1024,
        };

        if resource_usage.memory_bytes > max_memory {
            return Err(EvolutionError::ResourceLimitExceeded(format!(
                "Memory usage {} bytes exceeded limit {} bytes",
                resource_usage.memory_bytes, max_memory
            )));
        }

        let all_tests_passed = test_results.iter().all(|t| t.passed);

        Ok(SandboxResult {
            success: all_tests_passed,
            output: format!(
                "Sandbox test completed. {} tests passed, {} failed.",
                test_results.iter().filter(|t| t.passed).count(),
                test_results.iter().filter(|t| !t.passed).count()
            ),
            error: if all_tests_passed {
                None
            } else {
                Some("Some tests failed".to_string())
            },
            resource_usage,
            test_results,
            execution_time_ms,
        })
    }

    /// Simulate test execution
    async fn simulate_tests(
        &self,
        proposal: &EvolutionProposal,
    ) -> Result<Vec<TestResult>, EvolutionError> {
        // Simulate running tests based on target module
        let tests = match proposal.target_module.as_str() {
            "learning" => vec![
                "test_forward_pass",
                "test_backward_pass",
                "test_activation_functions",
                "test_optimizer",
            ],
            "blockchain" => vec![
                "test_block_validation",
                "test_transaction_verification",
                "test_difficulty_adjustment",
                "test_merkle_tree",
            ],
            "epistemic" => vec![
                "test_confidence_calculation",
                "test_source_reputation",
                "test_temporal_decay",
            ],
            _ => vec!["test_basic_functionality"],
        };

        let mut results = Vec::new();
        for test_name in tests {
            // Simulate test execution (90% pass rate for simulation)
            let passed = rand::random::<f64>() > 0.1;
            results.push(TestResult {
                test_name: test_name.to_string(),
                passed,
                duration_ms: 10 + rand::random::<u64>() % 100,
                error: if passed {
                    None
                } else {
                    Some("Test failed".to_string())
                },
            });
        }

        Ok(results)
    }

    /// Submit a vote for a proposal
    pub fn vote(
        &self,
        proposal_id: &str,
        voter: String,
        approve: bool,
        reasoning: Option<String>,
    ) -> Result<(), EvolutionError> {
        let mut proposal = self
            .proposals
            .get_mut(proposal_id)
            .ok_or_else(|| EvolutionError::ProposalNotFound(proposal_id.to_string()))?;

        if proposal.is_terminal() {
            return Err(EvolutionError::AlreadyFinalized(proposal_id.to_string()));
        }

        if proposal.voters.contains(&voter) {
            return Err(EvolutionError::InvalidProposal(
                "Voter has already voted".to_string(),
            ));
        }

        if approve {
            proposal.votes_for += 1;
        } else {
            proposal.votes_against += 1;
        }

        proposal.voters.push(voter.clone());
        proposal.add_audit_entry(
            "voted".to_string(),
            voter,
            format!(
                "Vote: {}. Reasoning: {}",
                if approve { "Approve" } else { "Reject" },
                reasoning.unwrap_or_else(|| "None".to_string())
            ),
        );

        Ok(())
    }

    /// Check if proposal has enough votes to be approved
    pub fn check_voting_result(&self, proposal_id: &str) -> Result<bool, EvolutionError> {
        let proposal = self
            .proposals
            .get(proposal_id)
            .ok_or_else(|| EvolutionError::ProposalNotFound(proposal_id.to_string()))?;

        if proposal.is_terminal() {
            return Err(EvolutionError::AlreadyFinalized(proposal_id.to_string()));
        }

        // Simple majority voting (can be enhanced with weighted voting)
        let total_votes = proposal.votes_for + proposal.votes_against;
        let min_votes = match proposal.risk_level {
            RiskLevel::Low => 3,
            RiskLevel::Medium => 5,
            RiskLevel::High => 10,
        };

        if total_votes < min_votes {
            return Err(EvolutionError::InsufficientVotes(total_votes, min_votes));
        }

        Ok(proposal.votes_for > proposal.votes_against)
    }

    /// Check if proposal can be auto-applied
    pub fn can_auto_apply(&self, proposal: &EvolutionProposal) -> bool {
        if proposal.requires_human_approval {
            return false;
        }
        if proposal.epistemic_confidence < self.constraints.min_confidence {
            return false;
        }
        if proposal.risk_level == RiskLevel::High {
            return false;
        }
        if let Some(result) = &proposal.sandbox_result {
            if !result.success {
                return false;
            }
        }
        true
    }

    /// Apply proposal to codebase (simulated)
    pub async fn apply_proposal(
        &self,
        proposal_id: &str,
    ) -> Result<(), EvolutionError> {
        let mut proposal = self
            .proposals
            .get_mut(proposal_id)
            .ok_or_else(|| EvolutionError::ProposalNotFound(proposal_id.to_string()))?;

        if proposal.is_terminal() {
            return Err(EvolutionError::AlreadyFinalized(proposal_id.to_string()));
        }

        // Simulate applying the diff
        // In production: actually apply the diff to the codebase
        let success = rand::random::<f64>() > 0.05; // 95% success rate

        if success {
            proposal.status = ProposalStatus::Applied;
            proposal.add_audit_entry(
                "applied".to_string(),
                "system".to_string(),
                "Proposal successfully applied to codebase".to_string(),
            );
            Ok(())
        } else {
            proposal.status = ProposalStatus::Failed;
            proposal.add_audit_entry(
                "failed".to_string(),
                "system".to_string(),
                "Failed to apply proposal".to_string(),
            );
            Err(EvolutionError::SandboxError(
                "Failed to apply proposal".to_string(),
            ))
        }
    }

    /// Get proposal by ID
    pub fn get_proposal(&self, proposal_id: &str) -> Option<EvolutionProposal> {
        self.proposals.get(proposal_id).map(|p| p.clone())
    }

    /// Get all proposals
    pub fn get_all_proposals(&self) -> Vec<EvolutionProposal> {
        self.proposals.iter().map(|p| p.value().clone()).collect()
    }

    /// Get proposals by status
    pub fn get_proposals_by_status(&self, status: ProposalStatus) -> Vec<EvolutionProposal> {
        self.proposals
            .iter()
            .filter(|p| p.value().status == status)
            .map(|p| p.value().clone())
            .collect()
    }

    /// Submit proposal for human review
    pub fn submit_for_approval(&self, proposal: &EvolutionProposal) -> String {
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
    fn test_risk_level_confidence_requirements() {
        assert_eq!(RiskLevel::Low.min_confidence_required(), 0.70);
        assert_eq!(RiskLevel::Medium.min_confidence_required(), 0.85);
        assert_eq!(RiskLevel::High.min_confidence_required(), 0.95);
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
        assert!(matches!(
            proposal.validate(&constraints),
            Err(EvolutionError::LowConfidence(_))
        ));
    }

    #[test]
    fn test_proposal_lifecycle() {
        let proposal = EvolutionProposal::new(
            "Test".into(),
            "api".into(),
            "+fn test() {}".into(),
            0.90,
            RiskLevel::Low,
        );

        assert_eq!(proposal.status, ProposalStatus::Pending);
        assert!(!proposal.is_terminal());

        let mut proposal = proposal;
        proposal.status = ProposalStatus::Applied;
        assert!(proposal.is_terminal());
    }

    #[test]
    fn test_audit_trail() {
        let mut proposal = EvolutionProposal::new(
            "Test".into(),
            "api".into(),
            "+fn test() {}".into(),
            0.90,
            RiskLevel::Low,
        );

        assert_eq!(proposal.audit_log.len(), 1); // Initial creation entry

        proposal.add_audit_entry(
            "reviewed".to_string(),
            "reviewer".to_string(),
            "Looks good".to_string(),
        );

        assert_eq!(proposal.audit_log.len(), 2);
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
        let sandbox_result = result.unwrap();
        assert!(!sandbox_result.test_results.is_empty());
    }

    #[test]
    fn test_proposal_storage() {
        let config = test_config();
        let engine = EvolutionEngine::new(&config);

        let proposal = engine
            .propose_change(
                "Test".into(),
                "api".into(),
                "+fn test() {}".into(),
                RiskLevel::Low,
            )
            .unwrap();

        let retrieved = engine.get_proposal(&proposal.id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, proposal.id);
    }

    #[test]
    fn test_voting_mechanism() {
        let config = test_config();
        let engine = EvolutionEngine::new(&config);

        let proposal = engine
            .propose_change(
                "Test".into(),
                "api".into(),
                "+fn test() {}".into(),
                RiskLevel::Low,
            )
            .unwrap();

        // Cast votes
        engine
            .vote(&proposal.id, "voter1".into(), true, None)
            .unwrap();
        engine
            .vote(&proposal.id, "voter2".into(), true, None)
            .unwrap();
        engine
            .vote(
                &proposal.id,
                "voter3".into(),
                false,
                Some("Not convinced".into()),
            )
            .unwrap();

        let updated = engine.get_proposal(&proposal.id).unwrap();
        assert_eq!(updated.votes_for, 2);
        assert_eq!(updated.votes_against, 1);
        assert_eq!(updated.voters.len(), 3);
    }

    #[test]
    fn test_duplicate_vote_prevention() {
        let config = test_config();
        let engine = EvolutionEngine::new(&config);

        let proposal = engine
            .propose_change(
                "Test".into(),
                "api".into(),
                "+fn test() {}".into(),
                RiskLevel::Low,
            )
            .unwrap();

        engine
            .vote(&proposal.id, "voter1".into(), true, None)
            .unwrap();

        // Try to vote again with same voter
        let result = engine.vote(&proposal.id, "voter1".into(), false, None);
        assert!(result.is_err());
    }
}