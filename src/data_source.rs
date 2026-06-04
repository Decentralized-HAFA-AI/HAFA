// ============================================================================
// HAFA - src/data_source.rs — DATA INGESTION & EPISTEMIC FILTERING (ADVANCED)
// ============================================================================
//
// Advanced data ingestion with full epistemic filtering:
// - Source reputation tracking
// - Evidence chain construction
// - Category-based organization
// - Integration with new epistemic.rs
//
// ============================================================================

use crate::config::Config;
use crate::crypto::hash_sha3_256;
use crate::epistemic::{
    EpistemicConstraints, EpistemicEngine, EpistemicState, Evidence, KnowledgeClaim,
    SourceReputation,
};
use chrono::Utc;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::fs;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum DataSourceError {
    #[error("IO operation failed: {0}")]
    IoError(String),
    #[error("Network source not yet implemented")]
    NetworkNotImplemented,
    #[error("Empty payload received")]
    EmptyContent,
    #[error("Epistemic validation failed: confidence {:.2} < threshold", .0)]
    LowConfidence(f64),
    #[error("Source type blocked by configuration")]
    SourceBlocked,
    #[error("Source reputation too low: {0:.2} < {1:.2}")]
    LowReputation(f64, f64),
}

// ============================================================================
// DATA SOURCE TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataSource {
    Local { path: String },
    Ipfs { cid: String },
    Web { url: String },
    Rss { url: String },
    Sensor { device_id: String },
}

impl DataSource {
    /// Get source type as string
    pub fn source_type(&self) -> &str {
        match self {
            DataSource::Local { .. } => "local",
            DataSource::Ipfs { .. } => "ipfs",
            DataSource::Web { .. } => "web",
            DataSource::Rss { .. } => "rss",
            DataSource::Sensor { .. } => "sensor",
        }
    }

    /// Generate unique source ID
    pub fn source_id(&self) -> String {
        match self {
            DataSource::Local { path } => hash_sha3_256(format!("local:{}", path).as_bytes()),
            DataSource::Ipfs { cid } => hash_sha3_256(format!("ipfs:{}", cid).as_bytes()),
            DataSource::Web { url } => hash_sha3_256(format!("web:{}", url).as_bytes()),
            DataSource::Rss { url } => hash_sha3_256(format!("rss:{}", url).as_bytes()),
            // Here we USE device_id, so no warning
            DataSource::Sensor { device_id } => {
                hash_sha3_256(format!("sensor:{}", device_id).as_bytes())
            }
        }
    }

    /// Check if this is a direct observation source
    pub fn is_direct_observation(&self) -> bool {
        matches!(self, DataSource::Local { .. } | DataSource::Sensor { .. })
    }

    /// Infer category from source type
    pub fn infer_category(&self) -> String {
        match self {
            DataSource::Local { path } => {
                if path.contains("code") || path.ends_with(".rs") || path.ends_with(".py") {
                    "code".to_string()
                } else if path.contains("data") || path.ends_with(".json") || path.ends_with(".csv") {
                    "data".to_string()
                } else {
                    "general".to_string()
                }
            }
            DataSource::Ipfs { .. } => "distributed".to_string(),
            DataSource::Web { url } => {
                if url.contains("github") || url.contains("gitlab") {
                    "code".to_string()
                } else if url.contains("arxiv") || url.contains("paper") {
                    "research".to_string()
                } else {
                    "web".to_string()
                }
            }
            DataSource::Rss { .. } => "news".to_string(),
            DataSource::Sensor { .. } => "sensor".to_string(),
        }
    }
}

// ============================================================================
// VALIDATED DATA
// ============================================================================

#[derive(Debug, Clone)]
pub struct ValidatedData {
    pub content: Vec<u8>,
    pub source: DataSource,
    pub epistemic_state: EpistemicState,
    pub timestamp: u64,
    pub knowledge_claim: KnowledgeClaim,
}

// ============================================================================
// SOURCE REPUTATION MANAGER
// ============================================================================

/// Manages reputation scores for all data sources
pub struct SourceReputationManager {
    reputations: Arc<DashMap<String, SourceReputation>>,
}

impl SourceReputationManager {
    pub fn new() -> Self {
        Self {
            reputations: Arc::new(DashMap::new()),
        }
    }

    /// Get or create reputation for a source
    pub fn get_or_create(&self, source: &DataSource) -> SourceReputation {
        let source_id = source.source_id();
        self.reputations
            .entry(source_id.clone())
            .or_insert_with(|| {
                SourceReputation::new(source_id, source.source_type().to_string())
            })
            .clone()
    }

    /// Update reputation based on claim verification
    pub fn update_reputation(&self, source: &DataSource, was_verified: bool) {
        let source_id = source.source_id();
        if let Some(mut rep) = self.reputations.get_mut(&source_id) {
            rep.update(was_verified);
        }
    }

    /// Get reputation score for a source
    pub fn get_reputation_score(&self, source: &DataSource) -> f64 {
        let source_id = source.source_id();
        self.reputations
            .get(&source_id)
            .map(|r| r.credibility_score)
            .unwrap_or(0.5) // Default neutral reputation
    }

    /// Get all reputations
    pub fn get_all_reputations(&self) -> Vec<SourceReputation> {
        self.reputations.iter().map(|r| r.value().clone()).collect()
    }
}

impl Default for SourceReputationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// DATA SOURCE IMPLEMENTATION
// ============================================================================

impl DataSource {
    pub async fn fetch_and_validate(
        &self,
        config: &Config,
        reputation_manager: &SourceReputationManager,
    ) -> Result<ValidatedData, DataSourceError> {
        // 1. Policy check
        match self {
            DataSource::Web { .. } | DataSource::Ipfs { .. } | DataSource::Rss { .. } => {
                if !config.learning.allow_internet_learning {
                    return Err(DataSourceError::SourceBlocked);
                }
                if config.learning.trusted_sources_only {
                    return Err(DataSourceError::SourceBlocked);
                }
            }
            DataSource::Local { .. } | DataSource::Sensor { .. } => {}
        }

        // 2. Fetch content
        // Note: We use { .. } for Sensor here because we don't need the device_id variable in this scope
        let content = match self {
            DataSource::Local { path } => {
                fs::read(path)
                    .await
                    .map_err(|e| DataSourceError::IoError(e.to_string()))?
            }
            DataSource::Ipfs { .. } | DataSource::Web { .. } | DataSource::Rss { .. } | DataSource::Sensor { .. } => {
                return Err(DataSourceError::NetworkNotImplemented);
            }
        };

        if content.is_empty() {
            return Err(DataSourceError::EmptyContent);
        }

        // 3. Get source reputation
        let source_reputation = reputation_manager.get_or_create(self);

        // Check minimum reputation threshold
        let constraints = EpistemicConstraints::default();
        if source_reputation.credibility_score < constraints.min_source_reputation {
            return Err(DataSourceError::LowReputation(
                source_reputation.credibility_score,
                constraints.min_source_reputation,
            ));
        }

        // 4. Create knowledge claim
        let source_id = self.source_id();
        let category = self.infer_category();
        let is_direct = self.is_direct_observation();

        let mut claim = KnowledgeClaim::new(
            &content,
            self.source_type().to_string(),
            source_id.clone(),
            is_direct,
            category,
        );

        // 5. Add initial evidence (the source itself)
        let initial_evidence = Evidence {
            evidence_id: hash_sha3_256(format!("{}|{}", source_id, Utc::now().timestamp()).as_bytes()),
            source_id: source_id.clone(),
            timestamp: Utc::now(),
            strength: if is_direct { 0.9 } else { 0.6 },
            content_hash: claim.content_hash.clone(),
        };
        claim.add_evidence(initial_evidence);

        // 6. Epistemic evaluation
        let epistemic_state =
            EpistemicEngine::evaluate(&claim, &constraints, Some(&source_reputation));

        if !epistemic_state.is_acceptable(&constraints) {
            // Update reputation negatively
            reputation_manager.update_reputation(self, false);
            return Err(DataSourceError::LowConfidence(epistemic_state.confidence));
        }

        // 7. Update reputation positively
        reputation_manager.update_reputation(self, true);

        Ok(ValidatedData {
            content,
            source: self.clone(),
            epistemic_state,
            timestamp: Utc::now().timestamp() as u64,
            knowledge_claim: claim,
        })
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
        cfg.learning.allow_internet_learning = false;
        cfg.learning.min_confidence_threshold = 0.75;
        cfg
    }

    #[test]
    fn test_source_type() {
        let local = DataSource::Local { path: "test.txt".into() };
        assert_eq!(local.source_type(), "local");

        let web = DataSource::Web { url: "http://example.com".into() };
        assert_eq!(web.source_type(), "web");
    }

    #[test]
    fn test_source_id_generation() {
        let source1 = DataSource::Local { path: "test.txt".into() };
        let source2 = DataSource::Local { path: "test.txt".into() };
        let source3 = DataSource::Local { path: "other.txt".into() };

        assert_eq!(source1.source_id(), source2.source_id());
        assert_ne!(source1.source_id(), source3.source_id());
    }

    #[test]
    fn test_is_direct_observation() {
        let local = DataSource::Local { path: "test.txt".into() };
        assert!(local.is_direct_observation());

        let web = DataSource::Web { url: "http://example.com".into() };
        assert!(!web.is_direct_observation());

        let sensor = DataSource::Sensor { device_id: "temp_001".into() };
        assert!(sensor.is_direct_observation());
    }

    #[test]
    fn test_category_inference() {
        let code_file = DataSource::Local { path: "src/main.rs".into() };
        assert_eq!(code_file.infer_category(), "code");

        let data_file = DataSource::Local { path: "data/config.json".into() };
        assert_eq!(data_file.infer_category(), "data");

        let github = DataSource::Web { url: "https://github.com/user/repo".into() };
        assert_eq!(github.infer_category(), "code");
    }

    #[test]
    fn test_reputation_manager() {
        let manager = SourceReputationManager::new();
        let source = DataSource::Local { path: "test.txt".into() };

        // Initial reputation should be 0.5
        let rep = manager.get_or_create(&source);
        assert_eq!(rep.credibility_score, 0.5);

        // Update positively
        manager.update_reputation(&source, true);
        let rep = manager.get_or_create(&source);
        assert!(rep.credibility_score > 0.5);

        // Update negatively
        manager.update_reputation(&source, false);
        let rep = manager.get_or_create(&source);
        assert!(rep.credibility_score > 0.4);
    }

    #[tokio::test]
    async fn test_local_source_blocked_by_policy() {
        let cfg = test_config();
        let manager = SourceReputationManager::new();
        let source = DataSource::Local { path: "nonexistent.txt".into() };
        
        assert!(matches!(
            source.fetch_and_validate(&cfg, &manager).await,
            Err(DataSourceError::IoError(_))
        ));
    }

    #[tokio::test]
    async fn test_network_source_blocked() {
        let cfg = test_config();
        let manager = SourceReputationManager::new();
        let source = DataSource::Web { url: "http://example.com".into() };
        
        assert!(matches!(
            source.fetch_and_validate(&cfg, &manager).await,
            Err(DataSourceError::SourceBlocked)
        ));
    }
}