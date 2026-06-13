// ============================================================================
// HAFA - src/api.rs — CORE CONTROL INTERFACE (GENERATIVE & REAL DATA)
// ============================================================================
//
// Advanced API layer connecting clients to HAFA core:
// - Real integration with Blockchain, Learner, EvolutionEngine
// - DataSource enum integration with directory scanning
// - Generative AI endpoint (Autoregressive text generation)
// - Event system subscription
// - Multi-signature support for founder actions
// - Real state management
// - NEW: Auto-Learning Engine integration (self-evolving AI)
//
// ============================================================================

use crate::blockchain::{Blockchain, BlockchainEvent};
use crate::config::Config;
use crate::crypto::{Address, MultiSig, is_valid_pubkey};
use crate::data_source::{DataSource, SourceReputationManager, ValidatedData};
use crate::epistemic::EpistemicState;
use crate::evolution::{EvolutionEngine, RiskLevel};
use crate::learning::Learner;
use crate::learning_v3::auto_learning::{
    AutoLearningEngine, TrainingSample,
};
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
    #[error("Buffer empty - no data ingested yet")]
    BufferEmpty,
    #[error("Auto-learning engine not initialized")]
    AutoLearningNotInitialized,
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
    /// Ingest all files from a directory (real data ingestion)
    IngestDirectory {
        path: String,
        recursive: bool,
    },
    /// Get learning engine status
    GetLearningStatus,
    /// Train the model on current buffer
    TrainModel { epochs: u32 },
    /// Query the model (simple prediction)
    QueryModel { input: Vec<u8> },
    /// Generate data autoregressively (NEW - Generative AI)
    Generate {
        prompt: Vec<u8>,
        steps: usize,
    },
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
    
    // ========================================================================
    // NEW: AUTO-LEARNING COMMANDS (Self-Evolving AI)
    // ========================================================================
    
    /// Feed a training sample to the auto-learning engine
    AutoLearnFeed {
        text: String,
        source: String,
        confidence: f32,
    },
    /// Manually trigger an auto-learning cycle
    AutoLearnTrigger,
    /// Get auto-learning engine status
    AutoLearnStatus,
    /// Get auto-learning engine statistics
    AutoLearnStats,
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
    QueryResult(Vec<u8>),
    Generated(GeneratedResult),
    DirectoryIngested(DirectoryIngestResult),
    ProposalSubmitted(String),
    VoteRecorded,
    EventStream,
    VestingStatus(VestingInfo),
    
    // NEW: Auto-Learning Responses
    AutoLearnFeed(AutoLearnFeedResult),
    AutoLearnTrigger(AutoLearnTriggerResult),
    AutoLearnStatus(AutoLearnStatusResult),
    AutoLearnStats(AutoLearnStatsResult),
    
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
    pub context_size: usize,
    pub predict_size: usize,
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

/// Result of generative AI query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedResult {
    pub generated_bytes: Vec<u8>,
    pub generated_text: String,
    pub steps: usize,
    pub bytes_per_step: usize,
}

/// Result of directory ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryIngestResult {
    pub files_processed: usize,
    pub total_bytes: usize,
    pub total_experiences: usize,
    pub buffer_size: usize,
    pub failed_files: Vec<String>,
}

// ============================================================================
// NEW: AUTO-LEARNING RESPONSE TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoLearnFeedResult {
    pub success: bool,
    pub message: String,
    pub buffer_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoLearnTriggerResult {
    pub success: bool,
    pub message: String,
    pub proof: Option<AutoLearnProofSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoLearnProofSummary {
    pub loss_before: f32,
    pub loss_after: f32,
    pub quality_score: f64,
    pub samples_processed: u64,
    pub gradient_commitment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoLearnStatusResult {
    pub is_learning: bool,
    pub buffer_size: usize,
    pub max_buffer_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoLearnStatsResult {
    pub total_cycles: u64,
    pub total_samples_received: u64,
    pub total_samples_rejected: u64,
    pub total_samples_learned: u64,
    pub total_proofs_generated: u64,
    pub last_cycle_time_secs: Option<u64>,
    pub last_cycle_loss: Option<f32>,
    pub buffer_size: usize,
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
    
    // NEW: Auto-Learning Engine (optional)
    auto_learning_engine: Option<Arc<RwLock<AutoLearningEngine>>>,
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
        auto_learning_engine: Option<Arc<RwLock<AutoLearningEngine>>>,
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
            auto_learning_engine,
        })
    }

    /// Validate and dispatch command to core
    pub async fn execute(&self, cmd: ClientCommand) -> Result<NodeResponse, ApiError> {
        match cmd {
            ClientCommand::FeedData { source, content } => {
                self.handle_feed_data(source, content).await
            }
            ClientCommand::IngestDirectory { path, recursive } => {
                self.handle_ingest_directory(path, recursive).await
            }
            ClientCommand::GetLearningStatus => self.handle_get_learning_status().await,
            ClientCommand::TrainModel { epochs } => self.handle_train_model(epochs).await,
            ClientCommand::QueryModel { input } => self.handle_query_model(input).await,
            ClientCommand::Generate { prompt, steps } => {
                self.handle_generate(prompt, steps).await
            }
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
            
            // NEW: Auto-Learning Commands
            ClientCommand::AutoLearnFeed { text, source, confidence } => {
                self.handle_auto_learn_feed(text, source, confidence).await
            }
            ClientCommand::AutoLearnTrigger => self.handle_auto_learn_trigger().await,
            ClientCommand::AutoLearnStatus => self.handle_auto_learn_status().await,
            ClientCommand::AutoLearnStats => self.handle_auto_learn_stats().await,
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
            metadata: None,
        };

        // Feed to learner
        let mut learner = self.learner.write().await;
        learner.ingest(&validated_data);

        Ok(NodeResponse::Success(format!(
            "Data ingested. Buffer size: {}",
            learner.get_stats().buffer_size
        )))
    }

    /// NEW: Ingest all files from a directory (real data ingestion)
    async fn handle_ingest_directory(
        &self,
        path: String,
        recursive: bool,
    ) -> Result<NodeResponse, ApiError> {
        // Policy check
        if !self.config.learning.allow_internet_learning {
            // Local directories are allowed
        }

        // Scan directory and fetch all files
        let validated_data_list = DataSource::fetch_directory_batch(
            &path,
            recursive,
            &self.config,
            &self.reputation_manager,
        )
        .await
        .map_err(|e| ApiError::DataSourceError(e.to_string()))?;

        let files_processed = validated_data_list.len();
        let mut total_bytes = 0;
        let failed_files = Vec::new();
        
        // Feed each file to learner
        let mut learner = self.learner.write().await;
        for data in validated_data_list {
            total_bytes += data.content.len();
            learner.ingest(&data);
        }

        let buffer_size = learner.get_stats().buffer_size;

        Ok(NodeResponse::DirectoryIngested(DirectoryIngestResult {
            files_processed,
            total_bytes,
            total_experiences: buffer_size,
            buffer_size,
            failed_files,
        }))
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
            context_size: stats.context_size,
            predict_size: stats.predict_size,
        }))
    }

    async fn handle_train_model(&self, epochs: u32) -> Result<NodeResponse, ApiError> {
        if epochs == 0 {
            return Err(ApiError::InvalidPayload("Epochs must be > 0".into()));
        }

        let mut learner = self.learner.write().await;
        
        if learner.buffer.is_empty() {
            return Err(ApiError::BufferEmpty);
        }

        let mut total_loss = 0.0;
        let mut successful_epochs = 0;

        for _ in 0..epochs {
            match learner.train_step() {
                Ok(loss) => {
                    total_loss += loss;
                    successful_epochs += 1;
                }
                Err(e) => return Err(ApiError::LearningError(e.to_string())),
            }
        }

        let avg_loss = if successful_epochs > 0 {
            total_loss / successful_epochs as f64
        } else {
            0.0
        };

        Ok(NodeResponse::Success(format!(
            "Training completed. {}/{} epochs successful, avg loss: {:.6}",
            successful_epochs, epochs, avg_loss
        )))
    }

    async fn handle_query_model(&self, input: Vec<u8>) -> Result<NodeResponse, ApiError> {
        let mut learner = self.learner.write().await;
        // Generate 1 step (64 bytes) by default
        let result = learner.query(&input, 1);
        Ok(NodeResponse::QueryResult(result))
    }

    /// NEW: Generative AI endpoint - autoregressive generation
    async fn handle_generate(
        &self,
        prompt: Vec<u8>,
        steps: usize,
    ) -> Result<NodeResponse, ApiError> {
        if prompt.is_empty() {
            return Err(ApiError::InvalidPayload("Empty prompt".into()));
        }

        if steps == 0 || steps > 100 {
            return Err(ApiError::InvalidPayload(
                "Steps must be between 1 and 100".into(),
            ));
        }

        let mut learner = self.learner.write().await;
        
        if learner.buffer.is_empty() {
            return Err(ApiError::BufferEmpty);
        }

        let bytes_per_step = learner.predict_size;
        let generated_bytes = learner.query(&prompt, steps);

        // Try to convert to UTF-8 text for display
        let generated_text = String::from_utf8_lossy(&generated_bytes).to_string();

        Ok(NodeResponse::Generated(GeneratedResult {
            generated_bytes,
            generated_text,
            steps,
            bytes_per_step,
        }))
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

        let already_withdrawn = total_vested.saturating_sub(available);

        Ok(NodeResponse::VestingStatus(VestingInfo {
            total_allocation,
            total_vested,
            available,
            already_withdrawn,
        }))
    }

    // ========================================================================
    // NEW: AUTO-LEARNING HANDLERS
    // ========================================================================

    /// Feed a training sample to the auto-learning engine
    async fn handle_auto_learn_feed(
        &self,
        text: String,
        source: String,
        confidence: f32,
    ) -> Result<NodeResponse, ApiError> {
        let mut engine = match &self.auto_learning_engine {
            Some(e) => e.write().await,
            None => return Err(ApiError::AutoLearningNotInitialized),
        };
        
        if text.is_empty() {
            return Err(ApiError::InvalidPayload("Empty text".into()));
        }
        
        let sample = TrainingSample::new(text, source, confidence);
        let success = engine.push_sample(sample);
        let buffer_size = engine.buffer_size();
        
        Ok(NodeResponse::AutoLearnFeed(AutoLearnFeedResult {
            success,
            message: if success {
                "Sample added to auto-learning buffer".to_string()
            } else {
                "Sample rejected (low confidence or buffer full)".to_string()
            },
            buffer_size,
        }))
    }

    /// Manually trigger an auto-learning cycle
    async fn handle_auto_learn_trigger(&self) -> Result<NodeResponse, ApiError> {
        let mut engine = match &self.auto_learning_engine {
            Some(e) => e.write().await,
            None => return Err(ApiError::AutoLearningNotInitialized),
        };
        
        match engine.trigger_learning() {
            Some(proof) => Ok(NodeResponse::AutoLearnTrigger(AutoLearnTriggerResult {
                success: true,
                message: "Learning cycle completed successfully".to_string(),
                proof: Some(AutoLearnProofSummary {
                    loss_before: proof.loss_before,
                    loss_after: proof.loss_after,
                    quality_score: proof.quality_score(),
                    samples_processed: proof.samples_processed,
                    gradient_commitment: proof.gradient_commitment.clone(),
                }),
            })),
            None => Ok(NodeResponse::AutoLearnTrigger(AutoLearnTriggerResult {
                success: false,
                message: "Learning not triggered (not enough samples or too soon since last cycle)".to_string(),
                proof: None,
            })),
        }
    }

    /// Get auto-learning engine status
    async fn handle_auto_learn_status(&self) -> Result<NodeResponse, ApiError> {
        let engine = match &self.auto_learning_engine {
            Some(e) => e.read().await,
            None => return Err(ApiError::AutoLearningNotInitialized),
        };
        
        Ok(NodeResponse::AutoLearnStatus(AutoLearnStatusResult {
            is_learning: engine.is_learning(),
            buffer_size: engine.buffer_size(),
            max_buffer_size: 1000, // Default max buffer size
        }))
    }

    /// Get auto-learning engine statistics
    async fn handle_auto_learn_stats(&self) -> Result<NodeResponse, ApiError> {
        let engine = match &self.auto_learning_engine {
            Some(e) => e.read().await,
            None => return Err(ApiError::AutoLearningNotInitialized),
        };
        
        let stats = engine.stats();
        
        Ok(NodeResponse::AutoLearnStats(AutoLearnStatsResult {
            total_cycles: stats.total_cycles,
            total_samples_received: stats.total_samples_received,
            total_samples_rejected: stats.total_samples_rejected,
            total_samples_learned: stats.total_samples_learned,
            total_proofs_generated: stats.total_proofs_generated,
            last_cycle_time_secs: stats.last_cycle_time,
            last_cycle_loss: stats.last_cycle_loss,
            buffer_size: stats.buffer_size,
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
            None, // No auto-learning engine for this test
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
            None,
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
            None,
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
            None,
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

    #[tokio::test]
    async fn test_generate_empty_buffer() {
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
            None,
        )
        .await
        .unwrap();

        let cmd = ClientCommand::Generate {
            prompt: b"test".to_vec(),
            steps: 1,
        };

        // Should fail because buffer is empty
        let result = session.execute(cmd).await;
        assert!(matches!(result, Err(ApiError::BufferEmpty)));
    }

    #[tokio::test]
    async fn test_learning_status_includes_new_fields() {
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
            None,
        )
        .await
        .unwrap();

        let result = session.execute(ClientCommand::GetLearningStatus).await;
        if let Ok(NodeResponse::LearningStatus(status)) = result {
            assert_eq!(status.context_size, 64);
            assert_eq!(status.predict_size, 64);
        } else {
            panic!("Expected LearningStatus response");
        }
    }

    #[tokio::test]
    async fn test_auto_learn_not_initialized() {
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
            None, // No auto-learning engine
        )
        .await
        .unwrap();

        let cmd = ClientCommand::AutoLearnStatus;
        let result = session.execute(cmd).await;
        assert!(matches!(result, Err(ApiError::AutoLearningNotInitialized)));
    }
}