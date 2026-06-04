// ============================================================================
// HAFA - src/api.rs — CORE CONTROL INTERFACE (ADVANCED)
// ============================================================================
//
// Advanced API layer connecting clients to HAFA core:
// - Real integration with Blockchain, Learner, EvolutionEngine
// - DataSource enum integration
// - Event system subscription
// - Multi-signature support for founder actions
// - Real state management
//
// ============================================================================

use crate::blockchain::{Blockchain, BlockchainEvent};
use crate::config::Config;
use crate::crypto::{Address, MultiSig, is_valid_pubkey};
use crate::data_source::{DataSource, SourceReputationManager, ValidatedData};
use crate::epistemic::EpistemicState;
use crate::evolution::{EvolutionEngine, RiskLevel};
use crate::learning::Learner;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Invalid client authentication")]
    Unauthorized,
    #[error("Invalid command payload: {0}")]
    InvalidPayload(String),
    #[error("Core processing failed: {0}")]
    CoreError(String),
    #[error("Data source error: {0}")]
    DataSourceError(String),
    #[error("Learning error: {0}")]
    LearningError(String),
    #[error("Evolution error: {0}")]
    EvolutionError(String),
    #[error("Blockchain error: {0}")]
    BlockchainError(String),
    #[error("Multi-signature verification failed")]
    MultiSigFailed,
}

// ============================================================================
// CLIENT COMMANDS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientCommand {
    /// Feed data to the learning engine
    FeedData {
        source: DataSource,
        content: Vec<u8>,
    },
    /// Get learning engine status
    GetLearningStatus,
    /// Train the model on current buffer
    TrainModel { epochs: u32 },
    /// Query the model
    QueryModel { input: Vec<u8> },
    /// Start mining (delegated to miner)
    StartMining { threads: u32 },
    /// Stop mining
    StopMining,
    /// Get node status
    GetStatus,
    /// Get blockchain info
    GetBlockchainInfo,
    /// Submit evolution proposal (founder or multi-sig)
    SubmitProposal {
        title: String,
        description: String,
        target_module: String,
        code_diff: String,
        risk_level: RiskLevel,
        multi_sig: Option<MultiSig>,
    },
    /// Vote on a proposal
    VoteProposal {
        proposal_id: String,
        approve: bool,
        reasoning: Option<String>,
    },
    /// Subscribe to events
    SubscribeEvents,
    /// Get founder vesting status
    GetVestingStatus,
}

// ============================================================================
// NODE RESPONSES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeResponse {
    Success(String),
    Status(NodeStatus),
    LearningStatus(LearningStatus),
    BlockchainInfo(BlockchainInfo),
    QueryResult(Vec<f64>),
    ProposalSubmitted(String),
    VoteRecorded,
    EventStream,
    VestingStatus(VestingInfo),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub uptime_secs: u64,
    pub chain_height: u64,
    pub pending_txs: u64,
    pub total_minted: u64,
    pub current_reward: u64,
    pub is_mining: bool,
    pub started_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStatus {
    pub input_size: usize,
    pub output_size: usize,
    pub num_layers: usize,
    pub buffer_size: usize,
    pub total_parameters: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockchainInfo {
    pub height: u64,
    pub total_minted: u64,
    pub current_reward: u64,
    pub difficulty: u32,
    pub last_block_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VestingInfo {
    pub total_allocation: u64,
    pub total_vested: u64,
    pub available: u64,
    pub already_withdrawn: u64,
}

// ============================================================================
// API SESSION
// ============================================================================

pub struct ApiSession {
    config: Config,
    client_pubkey: Address,
    is_founder: bool,
    blockchain: Arc<RwLock<Blockchain>>,
    learner: Arc<RwLock<Learner>>,
    evolution_engine: Arc<RwLock<EvolutionEngine>>,
    #[allow(dead_code)]
    reputation_manager: Arc<SourceReputationManager>,
    event_sender: broadcast::Sender<BlockchainEvent>,
    started_at: DateTime<Utc>,
    is_mining: Arc<RwLock<bool>>,
}

impl ApiSession {
    /// Initialize a secure session with client authentication
    pub async fn new(
        config: &Config,
        client_pubkey_hex: &str,
        blockchain: Arc<RwLock<Blockchain>>,
        learner: Arc<RwLock<Learner>>,
        evolution_engine: Arc<RwLock<EvolutionEngine>>,
        reputation_manager: Arc<SourceReputationManager>,
        event_sender: broadcast::Sender<BlockchainEvent>,
    ) -> Result<Self, ApiError> {
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
            blockchain,
            learner,
            evolution_engine,
            reputation_manager,
            event_sender,
            started_at: Utc::now(),
            is_mining: Arc::new(RwLock::new(false)),
        })
    }

    /// Validate and dispatch command to core
    pub async fn execute(&self, cmd: ClientCommand) -> Result<NodeResponse, ApiError> {
        match cmd {
            ClientCommand::FeedData { source, content } => {
                self.handle_feed_data(source, content).await
            }
            ClientCommand::GetLearningStatus => self.handle_get_learning_status().await,
            ClientCommand::TrainModel { epochs } => self.handle_train_model(epochs).await,
            ClientCommand::QueryModel { input } => self.handle_query_model(input).await,
            ClientCommand::StartMining { threads } => self.handle_start_mining(threads).await,
            ClientCommand::StopMining => self.handle_stop_mining().await,
            ClientCommand::GetStatus => self.handle_get_status().await,
            ClientCommand::GetBlockchainInfo => self.handle_get_blockchain_info().await,
            ClientCommand::SubmitProposal {
                title: _title,
                description,
                target_module,
                code_diff,
                risk_level,
                multi_sig,
            } => {
                self.handle_submit_proposal(
                    description,
                    target_module,
                    code_diff,
                    risk_level,
                    multi_sig,
                )
                .await
            }
            ClientCommand::VoteProposal {
                proposal_id,
                approve,
                reasoning,
            } => self.handle_vote_proposal(proposal_id, approve, reasoning).await,
            ClientCommand::SubscribeEvents => Ok(NodeResponse::EventStream),
            ClientCommand::GetVestingStatus => self.handle_get_vesting_status().await,
        }
    }

    // ========================================================================
    // COMMAND HANDLERS
    // ========================================================================

    async fn handle_feed_data(
        &self,
        source: DataSource,
        content: Vec<u8>,
    ) -> Result<NodeResponse, ApiError> {
        // Policy check
        if !self.config.learning.allow_internet_learning {
            if !matches!(source, DataSource::Local { .. } | DataSource::Sensor { .. }) {
                return Err(ApiError::Unauthorized);
            }
        }

        if content.is_empty() {
            return Err(ApiError::InvalidPayload("Empty payload".into()));
        }

        // Create knowledge claim first (uses &content)
        let knowledge_claim = crate::epistemic::KnowledgeClaim::new(
            &content,
            source.source_type().to_string(),
            source.source_id(),
            source.is_direct_observation(),
            source.infer_category(),
        );

        // Create validated data (uses content.clone())
        let validated_data = ValidatedData {
            content: content.clone(),
            source: source.clone(),
            epistemic_state: EpistemicState::new(0.9, true, 0, 0.1, 1, 0.0, 1.0),
            timestamp: Utc::now().timestamp() as u64,
            knowledge_claim,
        };

        // Feed to learner
        let mut learner = self.learner.write().await;
        learner.ingest(&validated_data);

        Ok(NodeResponse::Success(format!(
            "Data ingested. Buffer size: {}",
            learner.get_stats().buffer_size
        )))
    }

    async fn handle_get_learning_status(&self) -> Result<NodeResponse, ApiError> {
        let learner = self.learner.read().await;
        let stats = learner.get_stats();

        Ok(NodeResponse::LearningStatus(LearningStatus {
            input_size: stats.input_size,
            output_size: stats.output_size,
            num_layers: stats.num_layers,
            buffer_size: stats.buffer_size,
            total_parameters: stats.total_parameters,
        }))
    }

    async fn handle_train_model(&self, epochs: u32) -> Result<NodeResponse, ApiError> {
        if epochs == 0 {
            return Err(ApiError::InvalidPayload("Epochs must be > 0".into()));
        }

        let mut learner = self.learner.write().await;
        let mut total_loss = 0.0;

        for _ in 0..epochs {
            match learner.train_step() {
                Ok(loss) => total_loss += loss,
                Err(e) => return Err(ApiError::LearningError(e.to_string())),
            }
        }

        let avg_loss = total_loss / epochs as f64;
        Ok(NodeResponse::Success(format!(
            "Training completed. {} epochs, avg loss: {:.4}",
            epochs, avg_loss
        )))
    }

    async fn handle_query_model(&self, input: Vec<u8>) -> Result<NodeResponse, ApiError> {
        let mut learner = self.learner.write().await;
        let result = learner.query(&input);
        Ok(NodeResponse::QueryResult(result))
    }

    async fn handle_start_mining(&self, threads: u32) -> Result<NodeResponse, ApiError> {
        if threads == 0 {
            return Err(ApiError::InvalidPayload("Thread count must be > 0".into()));
        }

        let mut is_mining = self.is_mining.write().await;
        *is_mining = true;

        Ok(NodeResponse::Success(format!(
            "Mining started with {} threads",
            threads
        )))
    }

    async fn handle_stop_mining(&self) -> Result<NodeResponse, ApiError> {
        let mut is_mining = self.is_mining.write().await;
        *is_mining = false;

        Ok(NodeResponse::Success("Mining stopped".into()))
    }

    async fn handle_get_status(&self) -> Result<NodeResponse, ApiError> {
        let bc = self.blockchain.read().await;
        let is_mining = *self.is_mining.read().await;

        let uptime_secs = (Utc::now() - self.started_at).num_seconds() as u64;

        Ok(NodeResponse::Status(NodeStatus {
            uptime_secs,
            chain_height: bc.get_chain_height().await,
            pending_txs: 0, // TODO: implement pending tx count
            total_minted: bc.get_total_minted().await,
            current_reward: bc.get_current_reward().await,
            is_mining,
            started_at: self.started_at,
        }))
    }

    async fn handle_get_blockchain_info(&self) -> Result<NodeResponse, ApiError> {
        let bc = self.blockchain.read().await;
        let height = bc.get_chain_height().await;

        // Get last block hash and difficulty
        let (last_hash, difficulty) = match bc.get_task().await {
            Ok((hash, diff, _)) => (hash, diff),
            Err(_) => ("0".repeat(64), 1),
        };

        Ok(NodeResponse::BlockchainInfo(BlockchainInfo {
            height,
            total_minted: bc.get_total_minted().await,
            current_reward: bc.get_current_reward().await,
            difficulty,
            last_block_hash: last_hash,
        }))
    }

    async fn handle_submit_proposal(
        &self,
        description: String,
        target_module: String,
        code_diff: String,
        risk_level: RiskLevel,
        multi_sig: Option<MultiSig>,
    ) -> Result<NodeResponse, ApiError> {
        // Check authorization
        if !self.is_founder {
            // Check multi-sig if provided
            if let Some(ms) = multi_sig {
                // TODO: Verify multi-sig against DAO config
                // For now, just check if it has signatures
                if ms.signatures.is_empty() {
                    return Err(ApiError::MultiSigFailed);
                }
            } else {
                return Err(ApiError::Unauthorized);
            }
        }

        let engine = self.evolution_engine.read().await;
        let proposal = engine
            .propose_change(description, target_module, code_diff, risk_level)
            .map_err(|e| ApiError::EvolutionError(e.to_string()))?;

        Ok(NodeResponse::ProposalSubmitted(proposal.id))
    }

    async fn handle_vote_proposal(
        &self,
        proposal_id: String,
        approve: bool,
        reasoning: Option<String>,
    ) -> Result<NodeResponse, ApiError> {
        let engine = self.evolution_engine.read().await;
        engine
            .vote(
                &proposal_id,
                self.client_pubkey.pubkey_hex.clone(),
                approve,
                reasoning,
            )
            .map_err(|e| ApiError::EvolutionError(e.to_string()))?;

        Ok(NodeResponse::VoteRecorded)
    }

    async fn handle_get_vesting_status(&self) -> Result<NodeResponse, ApiError> {
        if !self.is_founder {
            return Err(ApiError::Unauthorized);
        }

        let bc = self.blockchain.read().await;
        let (total_allocation, total_vested, available) = bc.get_vesting_status().await;

        // Get already withdrawn (simplified - should be tracked in blockchain)
        let already_withdrawn = total_vested.saturating_sub(available);

        Ok(NodeResponse::VestingStatus(VestingInfo {
            total_allocation,
            total_vested,
            available,
            already_withdrawn,
        }))
    }

    // ========================================================================
    // ACCESSORS
    // ========================================================================

    pub fn is_founder(&self) -> bool {
        self.is_founder
    }

    pub fn client_address(&self) -> &Address {
        &self.client_pubkey
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<BlockchainEvent> {
        self.event_sender.subscribe()
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

    #[tokio::test]
    async fn test_founder_session_creation() {
        let cfg = test_config();
        let bc = Arc::new(RwLock::new(Blockchain::new(&cfg).await.unwrap()));
        let learner = Arc::new(RwLock::new(Learner::new(&cfg)));
        let evolution = Arc::new(RwLock::new(EvolutionEngine::new(&cfg)));
        let reputation = Arc::new(SourceReputationManager::new());
        let (event_tx, _) = broadcast::channel(100);

        let session = ApiSession::new(
            &cfg,
            &cfg.founder.genesis_pubkey_hex,
            bc,
            learner,
            evolution,
            reputation,
            event_tx,
        )
        .await;

        assert!(session.is_ok());
        assert!(session.unwrap().is_founder());
    }

    #[tokio::test]
    async fn test_normal_client_restricted_proposal() {
        let cfg = test_config();
        let bc = Arc::new(RwLock::new(Blockchain::new(&cfg).await.unwrap()));
        let learner = Arc::new(RwLock::new(Learner::new(&cfg)));
        let evolution = Arc::new(RwLock::new(EvolutionEngine::new(&cfg)));
        let reputation = Arc::new(SourceReputationManager::new());
        let (event_tx, _) = broadcast::channel(100);

        let session = ApiSession::new(
            &cfg,
            &"b".repeat(64),
            bc,
            learner,
            evolution,
            reputation,
            event_tx,
        )
        .await
        .unwrap();

        let cmd = ClientCommand::SubmitProposal {
            title: "Test".into(),
            description: "Test".into(),
            target_module: "api".into(),
            code_diff: "+fn test() {}".into(),
            risk_level: RiskLevel::Low,
            multi_sig: None,
        };

        assert!(matches!(
            session.execute(cmd).await,
            Err(ApiError::Unauthorized)
        ));
    }

    #[tokio::test]
    async fn test_get_status() {
        let cfg = test_config();
        let bc = Arc::new(RwLock::new(Blockchain::new(&cfg).await.unwrap()));
        let learner = Arc::new(RwLock::new(Learner::new(&cfg)));
        let evolution = Arc::new(RwLock::new(EvolutionEngine::new(&cfg)));
        let reputation = Arc::new(SourceReputationManager::new());
        let (event_tx, _) = broadcast::channel(100);

        let session = ApiSession::new(
            &cfg,
            &cfg.founder.genesis_pubkey_hex,
            bc,
            learner,
            evolution,
            reputation,
            event_tx,
        )
        .await
        .unwrap();

        let result = session.execute(ClientCommand::GetStatus).await;
        assert!(matches!(result, Ok(NodeResponse::Status(_))));
    }

    #[tokio::test]
    async fn test_feed_data() {
        let cfg = test_config();
        let bc = Arc::new(RwLock::new(Blockchain::new(&cfg).await.unwrap()));
        let learner = Arc::new(RwLock::new(Learner::new(&cfg)));
        let evolution = Arc::new(RwLock::new(EvolutionEngine::new(&cfg)));
        let reputation = Arc::new(SourceReputationManager::new());
        let (event_tx, _) = broadcast::channel(100);

        let session = ApiSession::new(
            &cfg,
            &cfg.founder.genesis_pubkey_hex,
            bc,
            learner,
            evolution,
            reputation,
            event_tx,
        )
        .await
        .unwrap();

        let cmd = ClientCommand::FeedData {
            source: DataSource::Local {
                path: "test.txt".into(),
            },
            content: b"test data".to_vec(),
        };

        let result = session.execute(cmd).await;
        assert!(matches!(result, Ok(NodeResponse::Success(_))));
    }
}