// ============================================================================
// HAFA - src/api.rs — CLIENT CONTROL INTERFACE
// ============================================================================

use crate::config::Config;
use crate::crypto::{Address, is_valid_pubkey};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Invalid client authentication")]
    Unauthorized,
    #[error("Invalid command payload: {0}")]
    InvalidPayload(String),
    #[error("Core processing failed: {0}")]
    CoreError(String),
}

#[derive(Debug, Clone)]
pub enum ClientCommand {
    FeedData { source: String, content: Vec<u8> },
    StartMining { threads: u32 },
    StopMining,
    GetStatus,
    SubmitProposal { title: String, description: String, diff: String },
}

#[derive(Debug, Clone)]
pub enum NodeResponse {
    Success(String),
    Status { uptime_secs: u64, chain_height: u64, pending_txs: u64 },
    Error(String),
}

pub struct ApiSession {
    config: Config,
    client_pubkey: Address,
    is_founder: bool,
}

impl ApiSession {
    /// Initialize a secure session with client authentication
    pub fn new(config: &Config, client_pubkey_hex: &str) -> Result<Self, ApiError> {
        if !is_valid_pubkey(client_pubkey_hex) {
            return Err(ApiError::Unauthorized);
        }
        
        let client_pubkey = Address::from_hex(client_pubkey_hex)
            .map_err(|_| ApiError::Unauthorized)?;

        let is_founder = config.is_founder_key(client_pubkey_hex);

        Ok(Self {
            config: config.clone(),
            client_pubkey,
            is_founder,
        })
    }

    /// Validate and dispatch command to core
    pub async fn execute(&self, cmd: ClientCommand) -> Result<NodeResponse, ApiError> {
        match &cmd {
            ClientCommand::StartMining { threads } => {
                if *threads == 0 {
                    return Err(ApiError::InvalidPayload("Thread count must be > 0".into()));
                }
            }
            ClientCommand::FeedData { source, content } => {
                if !self.config.learning.allow_internet_learning && source != "local" {
                    return Err(ApiError::Unauthorized);
                }
                if content.is_empty() {
                    return Err(ApiError::InvalidPayload("Empty payload".into()));
                }
            }
            ClientCommand::SubmitProposal { .. } => {
                if !self.is_founder {
                    return Err(ApiError::Unauthorized);
                }
            }
            _ => {}
        }

        // Dispatch simulation (actual routing to core event loop happens in main.rs)
        Ok(match cmd {
            ClientCommand::GetStatus => NodeResponse::Status {
                uptime_secs: 0,
                chain_height: 0,
                pending_txs: 0,
            },
            _ => NodeResponse::Success("Command queued".into()),
        })
    }

    pub fn is_founder(&self) -> bool {
        self.is_founder
    }

    pub fn client_address(&self) -> &Address {
        &self.client_pubkey
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        let mut cfg = Config::default();
        cfg.founder.genesis_pubkey_hex = "a".repeat(64);
        cfg
    }

    #[tokio::test]
    async fn test_founder_session_creation() {
        let cfg = test_config();
        let session = ApiSession::new(&cfg, &cfg.founder.genesis_pubkey_hex);
        assert!(session.is_ok());
        assert!(session.unwrap().is_founder());
    }

    #[tokio::test]
    async fn test_normal_client_restricted_proposal() {
        let cfg = test_config();
        let session = ApiSession::new(&cfg, "b".repeat(64).as_str()).unwrap();
        let cmd = ClientCommand::SubmitProposal {
            title: "Test".into(),
            description: "Test".into(),
            diff: "Test".into(),
        };
        assert!(matches!(session.execute(cmd).await, Err(ApiError::Unauthorized)));
    }
}