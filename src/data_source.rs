// ============================================================================
// HAFA - src/data_source.rs — DATA INGESTION & EPISTEMIC FILTERING
// ============================================================================

use crate::config::Config;
use crate::crypto::hash_sha3_256;
use crate::epistemic::{EpistemicEngine, EpistemicConstraints, EpistemicState, KnowledgeClaim};
use thiserror::Error;
use tokio::fs;
use serde::{Deserialize, Serialize};

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    Local { path: String },
    Ipfs { cid: String },
    Web { url: String },
    Rss { url: String },
}

#[derive(Debug, Clone)]
pub struct ValidatedData {
    pub content: Vec<u8>,
    pub source: DataSource,
    pub epistemic_state: EpistemicState,
    pub timestamp: u64,
}

impl DataSource {
    pub async fn fetch_and_validate(&self, config: &Config) -> Result<ValidatedData, DataSourceError> {
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
            DataSource::Local { .. } => {}
        }

        // 2. Fetch content
        let content = match self {
            DataSource::Local { path } => {
                fs::read(path).await.map_err(|e| DataSourceError::IoError(e.to_string()))?
            }
            DataSource::Ipfs { .. } | DataSource::Web { .. } | DataSource::Rss { .. } => {
                // Network fetchers will be implemented in phase 2
                return Err(DataSourceError::NetworkNotImplemented);
            }
        };

        if content.is_empty() {
            return Err(DataSourceError::EmptyContent);
        }

        // 3. Epistemic evaluation
        let content_hash = hash_sha3_256(&content);
        let claim = KnowledgeClaim {
            content_hash,
            source_type: match self {
                DataSource::Local { .. } => "local",
                DataSource::Ipfs { .. } => "ipfs",
                DataSource::Web { .. } => "web",
                DataSource::Rss { .. } => "rss",
            }
            .to_string(),
            corroborating_sources: 0,
            is_direct_observation: matches!(self, DataSource::Local { .. }),
        };

        let constraints = EpistemicConstraints {
            min_confidence: config.learning.min_confidence_threshold,
            max_speculation_depth: 1,
            require_grounding: config.learning.require_epistemic_validation,
        };

        let epistemic_state = EpistemicEngine::evaluate(&claim, &constraints);

        if !epistemic_state.is_acceptable(&constraints) {
            return Err(DataSourceError::LowConfidence(epistemic_state.confidence));
        }

        Ok(ValidatedData {
            content,
            source: self.clone(),
            epistemic_state,
            timestamp: chrono::Utc::now().timestamp() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        let mut cfg = Config::default();
        cfg.founder.genesis_pubkey_hex = "a".repeat(64);
        cfg.learning.allow_internet_learning = false;
        cfg.learning.min_confidence_threshold = 0.80;
        cfg
    }

    #[tokio::test]
    async fn test_local_source_blocked_by_policy() {
        let cfg = test_config();
        let source = DataSource::Local { path: "nonexistent.txt".into() };
        // Policy allows local, so it proceeds to IO error (expected)
        assert!(matches!(source.fetch_and_validate(&cfg).await, Err(DataSourceError::IoError(_))));
    }

    #[tokio::test]
    async fn test_network_source_blocked() {
        let cfg = test_config();
        let source = DataSource::Web { url: "http://example.com".into() };
        assert!(matches!(source.fetch_and_validate(&cfg).await, Err(DataSourceError::SourceBlocked)));
    }
}