// ============================================================================
// HAFA - src/blockchain.rs — PURE BITCOIN-STYLE CONSENSUS + COGNITIVE LAYER
// ============================================================================
//
// Bitcoin-style consensus with cognitive extensions:
// - Structured CognitiveProof (not just a string)
// - Merkle Tree for SPV proofs
// - Founder vesting schedule (10%, 30%, 30%, 30%)
// - Founder royalty (2% on commercial transactions)
// - Learning reports for quality-based rewards
// - Model checkpoints for state tracking
// - Event system for client notifications
//
// ============================================================================

use crate::config::{
    Config, INITIAL_BLOCK_REWARD, HALVING_INTERVAL,
    TARGET_BLOCK_TIME_SECS, DIFFICULTY_ADJUSTMENT_INTERVAL,
};
use crate::crypto::{hash_sha3_256, verify_hex_signature, KeyPair};
use crate::epistemic::EpistemicState;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{RwLock, broadcast};
use std::sync::Arc;
use std::collections::HashMap;
use std::path::PathBuf;
use std::fs;
use chrono::Utc;
use hex::ToHex;
use sha3::{Digest, Sha3_256};

const MAX_DIFFICULTY: u32 = 64;
const MIN_DIFFICULTY: u32 = 1;
const MAX_BLOCK_TIME_DRIFT_SECS: i64 = 7200;
const TRANSACTION_POOL_LIMIT: usize = 10_000;

// Founder vesting schedule: 10%, 30%, 30%, 30% at years 0, 1, 2, 3
const VESTING_PERCENTAGES: [u64; 4] = [10, 30, 30, 30];
const VESTING_CLIFF_SECS: u64 = 365 * 24 * 60 * 60; // 1 year in seconds
const FOUNDER_ROYALTY_BPS: u64 = 200; // 2% = 200 basis points
const ROYALTY_THRESHOLD: u64 = 1000; // Minimum tx amount to apply royalty

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum BlockchainError {
    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),
    #[error("Invalid block: {0}")]
    InvalidBlock(String),
    #[error("Chain validation failed at block {0}")]
    ChainValidationFailed(u64),
    #[error("Insufficient balance: have {have}, need {need}")]
    InsufficientBalance { have: u64, need: u64 },
    #[error("Storage error: {0}")]
    StorageError(String),
    #[error("Consensus error: {0}")]
    ConsensusError(String),
    #[error("Vesting error: {0}")]
    VestingError(String),
    #[error("Merkle tree error: {0}")]
    MerkleError(String),
}

// ============================================================================
// TRANSACTION TYPES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionType {
    Transfer,
    Reward,
    RevenueShare,
    CognitiveWork,
    ContractCall,
    Governance,
    LearningReport,
    EvolutionProposal,
    GenesisVesting,
    LearningSample,
}

// ============================================================================
// COGNITIVE PROOF (Structured)
// ============================================================================

/// Structured proof of useful cognitive work performed by miner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveProof {
    /// Hash of model weights before training
    pub model_hash_before: String,
    /// Hash of model weights after training
    pub model_hash_after: String,
    /// Loss before training step
    pub loss_before: f64,
    /// Loss after training step
    pub loss_after: f64,
    /// Number of experiences processed
    pub experiences_processed: u32,
    /// Average epistemic confidence of processed data
    pub avg_confidence: f64,
    /// Hardware resources used (CPU/GPU/RAM metrics)
    pub resources_used: ResourceUsage,
    /// Training duration in milliseconds
    pub training_duration_ms: u64,
    /// SHA3-256 hash of this entire proof (for block header)
    pub proof_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_percent: f64,
    pub ram_mb: u64,
    pub gpu_percent: f64,
    pub gpu_memory_mb: u64,
}

impl CognitiveProof {
    pub fn new(
        model_hash_before: String,
        model_hash_after: String,
        loss_before: f64,
        loss_after: f64,
        experiences_processed: u32,
        avg_confidence: f64,
        resources_used: ResourceUsage,
        training_duration_ms: u64,
    ) -> Self {
        let mut proof = Self {
            model_hash_before,
            model_hash_after,
            loss_before,
            loss_after,
            experiences_processed,
            avg_confidence,
            resources_used,
            training_duration_ms,
            proof_hash: String::new(),
        };
        proof.proof_hash = proof.calculate_hash();
        proof
    }

    pub fn calculate_hash(&self) -> String {
        let data = format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}",
            self.model_hash_before,
            self.model_hash_after,
            self.loss_before,
            self.loss_after,
            self.experiences_processed,
            self.avg_confidence,
            self.resources_used.cpu_percent,
            self.resources_used.ram_mb,
            self.training_duration_ms
        );
        hash_sha3_256(data.as_bytes())
    }

    /// Calculate quality score (0.0 to 1.0) based on learning improvement
    pub fn quality_score(&self) -> f64 {
        if self.loss_before <= 0.0 {
            return 0.5;
        }

        let loss_reduction = if self.loss_before > self.loss_after {
            ((self.loss_before - self.loss_after) / self.loss_before).min(1.0)
        } else {
            0.0
        };

        let experience_factor = (self.experiences_processed as f64 / 100.0).min(1.0);
        let confidence_factor = self.avg_confidence;

        // Weighted combination
        loss_reduction * 0.5 + experience_factor * 0.3 + confidence_factor * 0.2
    }

    /// Create a dummy proof for genesis block
    pub fn genesis() -> Self {
        Self::new(
            "0".repeat(64),
            "0".repeat(64),
            0.0,
            0.0,
            0,
            1.0,
            ResourceUsage {
                cpu_percent: 0.0,
                ram_mb: 0,
                gpu_percent: 0.0,
                gpu_memory_mb: 0,
            },
            0,
        )
    }
}

// ============================================================================
// MERKLE TREE
// ============================================================================

pub struct MerkleTree;

impl MerkleTree {
    /// Calculate Merkle root from a list of transaction IDs
    pub fn root(tx_ids: &[String]) -> String {
        if tx_ids.is_empty() {
            return "0".repeat(64);
        }

        let mut hashes: Vec<String> = tx_ids.to_vec();

        while hashes.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in hashes.chunks(2) {
                let combined = if chunk.len() == 2 {
                    format!("{}{}", chunk[0], chunk[1])
                } else {
                    // Odd number: duplicate last hash
                    format!("{}{}", chunk[0], chunk[0])
                };
                next_level.push(hash_sha3_256(combined.as_bytes()));
            }

            hashes = next_level;
        }

        hashes[0].clone()
    }

    /// Generate Merkle proof for a specific transaction
pub fn proof(tx_ids: &[String], target_index: usize) -> Result<Vec<(String, bool)>, BlockchainError> {
    if target_index >= tx_ids.len() {
        return Err(BlockchainError::MerkleError("Index out of bounds".into()));
    }

    let mut hashes: Vec<String> = tx_ids.to_vec();
    let mut proof = Vec::new();
    let mut idx = target_index;

    while hashes.len() > 1 {
        let mut next_level = Vec::new();

        for (i, chunk) in hashes.chunks(2).enumerate() {
            let (left, right) = if chunk.len() == 2 {
                (chunk[0].clone(), chunk[1].clone())
            } else {
                (chunk[0].clone(), chunk[0].clone())
            };

            // If our target is in this pair, add sibling to proof
            if i * 2 == idx || i * 2 + 1 == idx {
                if idx % 2 == 0 {
                    proof.push((right.clone(), true)); // ← clone اضافه شد
                } else {
                    proof.push((left.clone(), false)); // ← clone اضافه شد
                }
            }

            next_level.push(hash_sha3_256(format!("{}{}", left, right).as_bytes()));
        }

        hashes = next_level;
        idx /= 2;
    }

    Ok(proof)
}
    /// Verify a Merkle proof
    pub fn verify_proof(leaf: &str, proof: &[(String, bool)], root: &str) -> bool {
        let mut current = leaf.to_string();

        for (sibling, is_right) in proof {
            current = if *is_right {
                hash_sha3_256(format!("{}{}", current, sibling).as_bytes())
            } else {
                hash_sha3_256(format!("{}{}", sibling, current).as_bytes())
            };
        }

        current == root
    }
}

// ============================================================================
// LEARNING REPORT
// ============================================================================

/// Report submitted by miner detailing learning quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningReport {
    pub miner_address: String,
    pub block_height: u64,
    pub cognitive_proof: CognitiveProof,
    pub quality_score: f64,
    pub epistemic_state: EpistemicState,
    pub timestamp: u64,
}

impl LearningReport {
    pub fn new(
        miner_address: String,
        block_height: u64,
        cognitive_proof: CognitiveProof,
        epistemic_state: EpistemicState,
    ) -> Self {
        let quality_score = cognitive_proof.quality_score();
        Self {
            miner_address,
            block_height,
            cognitive_proof,
            quality_score,
            epistemic_state,
            timestamp: Utc::now().timestamp() as u64,
        }
    }
}

// ============================================================================
// MODEL CHECKPOINT
// ============================================================================

/// Checkpoint of model state stored in blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub block_height: u64,
    pub model_hash: String,
    pub total_parameters: u64,
    pub architecture: String,
    pub timestamp: u64,
}

// ============================================================================
// FOUNDER VESTING SCHEDULE
// ============================================================================

pub struct VestingSchedule;

impl VestingSchedule {
    /// Calculate vested amount at a given block height
    pub fn calculate_vested(
        total_allocation: u64,
        genesis_timestamp: u64,
        current_timestamp: u64,
    ) -> u64 {
        let elapsed_secs = current_timestamp.saturating_sub(genesis_timestamp);
        let years_elapsed = elapsed_secs / VESTING_CLIFF_SECS;

        if years_elapsed == 0 {
            return 0;
        }

        let mut vested_percent: u64 = 0;
        for i in 0..years_elapsed.min(VESTING_PERCENTAGES.len() as u64) {
            vested_percent += VESTING_PERCENTAGES[i as usize];
        }

        total_allocation * vested_percent.min(100) / 100
    }

    /// Check if a vesting withdrawal is valid
    pub fn validate_withdrawal(
        total_allocation: u64,
        already_withdrawn: u64,
        genesis_timestamp: u64,
        current_timestamp: u64,
        requested_amount: u64,
    ) -> Result<bool, BlockchainError> {
        let total_vested = Self::calculate_vested(total_allocation, genesis_timestamp, current_timestamp);
        let available = total_vested.saturating_sub(already_withdrawn);

        if requested_amount > available {
            return Err(BlockchainError::VestingError(format!(
                "Requested {} but only {} available (vested: {}, withdrawn: {})",
                requested_amount, available, total_vested, already_withdrawn
            )));
        }

        Ok(true)
    }
}

// ============================================================================
// EVENTS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockchainEvent {
    NewBlock { height: u64, hash: String, reward: u64 },
    NewTransaction { tx_id: String, tx_type: TransactionType },
    DifficultyAdjusted { old_difficulty: u32, new_difficulty: u32 },
    ModelCheckpoint { height: u64, model_hash: String },
    VestingReleased { amount: u64, remaining: u64 },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningTransaction {
    pub tx_id: String,
    pub sample_text: String,
    pub source: String,
    pub confidence: f32,
    pub sender_peer_id: String,
    pub timestamp: u64,
}

// ============================================================================
// TRANSACTION
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: String,
    pub from: String,
    pub to: String,
    pub amount: u64,
    pub tx_type: TransactionType,
    pub timestamp: u64,
    pub signature: String,
    pub data: Option<String>,
    pub founder_share: u64,
    pub epistemic_context: Option<EpistemicState>,
    pub merkle_index: Option<u32>,
}

impl Transaction {
    pub fn new(
        from: String,
        to: String,
        amount: u64,
        tx_type: TransactionType,
        keypair: &KeyPair,
        data: Option<String>,
    ) -> Result<Self, BlockchainError> {
        let timestamp = Utc::now().timestamp() as u64;

        // Calculate founder royalty (2% for transfers above threshold)
        let founder_share = if tx_type == TransactionType::Transfer && amount >= ROYALTY_THRESHOLD {
            amount * FOUNDER_ROYALTY_BPS / 10_000
        } else {
            0
        };

        let message = format!(
            "{}|{}|{}|{}|{}|{}",
            from, to, amount, timestamp, tx_type as i32, founder_share
        );
        let signature = keypair.sign(message.as_bytes());

        Ok(Self {
            id: hash_sha3_256(format!("{}|{}|{}", from, timestamp, rand::random::<u64>()).as_bytes()),
            from,
            to,
            amount,
            tx_type,
            timestamp,
            signature: signature.to_bytes().encode_hex::<String>(),
            data,
            founder_share,
            epistemic_context: None,
            merkle_index: None,
        })
    }

    pub fn verify_signature(&self) -> bool {
        // Genesis/system transactions don't need signature verification
        if self.from == "SYSTEM" || self.signature == "GENESIS" || self.signature == "SYSTEM" {
            return true;
        }

        let message = format!(
            "{}|{}|{}|{}|{}|{}",
            self.from, self.to, self.amount, self.timestamp, self.tx_type as i32, self.founder_share
        );
        verify_hex_signature(&self.from, message.as_bytes(), &self.signature).is_ok()
    }
}

// ============================================================================
// BLOCK
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub index: u64,
    pub timestamp: u64,
    pub transactions: Vec<Transaction>,
    pub previous_hash: String,
    pub hash: String,
    pub nonce: u64,
    pub cognitive_proof: CognitiveProof,
    pub difficulty: u32,
    pub cumulative_work: u64,
    pub merkle_root: String,
    pub model_checkpoint: Option<ModelCheckpoint>,
}

impl Block {
    pub fn calculate_hash(
        index: u64,
        timestamp: u64,
        prev_hash: &str,
        merkle_root: &str,
        nonce: u64,
        difficulty: u32,
        cognitive_proof_hash: &str,
    ) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(index.to_le_bytes());
        hasher.update(timestamp.to_le_bytes());
        hasher.update(prev_hash.as_bytes());
        hasher.update(merkle_root.as_bytes());
        hasher.update(nonce.to_le_bytes());
        hasher.update(difficulty.to_le_bytes());
        hasher.update(cognitive_proof_hash.as_bytes());
        hasher.finalize().encode_hex::<String>()
    }

    pub fn meets_difficulty(&self, difficulty: u32) -> bool {
        let target = Self::calculate_target(difficulty);
        let hash_bytes = hex::decode(&self.hash).unwrap_or_default();
        &hash_bytes[..] <= &target[..]
    }

    pub fn calculate_target(difficulty: u32) -> [u8; 32] {
        let mut target = [0xFFu8; 32];
        let zero_bytes = (difficulty / 8) as usize;
        let zero_bits = (difficulty % 8) as usize;
        for i in 0..zero_bytes.min(32) {
            target[i] = 0x00;
        }
        if zero_bytes < 32 {
            target[zero_bytes] = 0xFF << zero_bits;
        }
        target
    }
}

// ============================================================================
// PERSISTENCE
// ============================================================================

#[derive(Serialize, Deserialize)]
struct SerializableState {
    chain: Vec<Block>,
    total_minted: u64,
    balance_cache: HashMap<String, u64>,
    founder_vesting_withdrawn: u64,
    genesis_timestamp: u64,
}

// ============================================================================
// BLOCKCHAIN
// ============================================================================

pub struct Blockchain {
    config: Config,
    chain: Arc<RwLock<Vec<Block>>>,
    pending_pool: Arc<RwLock<Vec<Transaction>>>,
    storage_path: PathBuf,
    total_minted: Arc<RwLock<u64>>,
    balance_cache: Arc<RwLock<HashMap<String, u64>>>,
    founder_vesting_withdrawn: Arc<RwLock<u64>>,
    genesis_timestamp: Arc<RwLock<u64>>,
    event_sender: broadcast::Sender<BlockchainEvent>,
}

impl Blockchain {
    pub async fn new(config: &Config) -> Result<Self, BlockchainError> {
        let storage_path = config.storage.data_dir.join("blockchain.dat");
        let chain = Arc::new(RwLock::new(Vec::new()));
        let pending_pool = Arc::new(RwLock::new(Vec::new()));
        let total_minted = Arc::new(RwLock::new(0));
        let balance_cache = Arc::new(RwLock::new(HashMap::new()));
        let founder_vesting_withdrawn = Arc::new(RwLock::new(0));
        let genesis_timestamp = Arc::new(RwLock::new(0));
        let (event_sender, _) = broadcast::channel(1000);

        let _ = fs::create_dir_all(&config.storage.data_dir);

        let mut bc = Self {
            config: config.clone(),
            chain: chain.clone(),
            pending_pool,
            storage_path: storage_path.clone(),
            total_minted,
            balance_cache,
            founder_vesting_withdrawn,
            genesis_timestamp,
            event_sender,
        };

        if storage_path.exists() {
            bc.load_from_disk().await?;
            println!("   Loaded blockchain from disk. Height: {}", bc.get_chain_height().await);
        } else {
            bc.create_genesis_block().await?;
            println!("   Created new genesis block.");
        }
        Ok(bc)
    }
    /// Get a block by its height (index)
    pub async fn get_block(&self, height: u64) -> Option<Block> {
        let chain = self.chain.read().await;
        chain.get(height as usize).cloned()
    }
    /// Subscribe to blockchain events
    pub fn subscribe(&self) -> broadcast::Receiver<BlockchainEvent> {
        self.event_sender.subscribe()
    }

    async fn emit_event(&self, event: BlockchainEvent) {
        let _ = self.event_sender.send(event);
    }

    async fn create_genesis_block(&mut self) -> Result<(), BlockchainError> {
        let now = Utc::now().timestamp() as u64;
        let founder_addr = &self.config.founder.genesis_pubkey_hex;
        let founder_allocation = self.config.founder_genesis_amount();

        let genesis_tx = Transaction {
            id: hash_sha3_256(b"genesis"),
            from: "SYSTEM".into(),
            to: founder_addr.clone(),
            amount: founder_allocation,
            tx_type: TransactionType::GenesisVesting,
            timestamp: now,
            signature: "GENESIS".into(),
            data: Some(format!(
                "HAFA Genesis - Founder allocation: {} HAFA (vested 3 years)",
                founder_allocation
            )),
            founder_share: 0,
            epistemic_context: Some(EpistemicState::new(1.0, true, 0, 0.0, 0, 0.0, 1.0)),
            merkle_index: Some(0),
        };

        let merkle_root = MerkleTree::root(&[genesis_tx.id.clone()]);
        let cognitive_proof = CognitiveProof::genesis();

        let mut genesis = Block {
            index: 0,
            timestamp: now,
            transactions: vec![genesis_tx],
            previous_hash: "0".repeat(64),
            hash: String::new(),
            nonce: 0,
            cognitive_proof,
            difficulty: MIN_DIFFICULTY,
            cumulative_work: 1,
            merkle_root,
            model_checkpoint: None,
        };

        genesis.hash = Block::calculate_hash(
            genesis.index,
            genesis.timestamp,
            &genesis.previous_hash,
            &genesis.merkle_root,
            genesis.nonce,
            genesis.difficulty,
            &genesis.cognitive_proof.proof_hash,
        );

        // Note: Founder allocation is recorded but NOT immediately spendable
        // It vests over 3 years according to schedule
        *self.total_minted.write().await = founder_allocation;
        *self.genesis_timestamp.write().await = now;

        // Don't add to balance cache yet - it's locked in vesting
        // Balance will be updated as vesting releases

        self.chain.write().await.push(genesis);
        self.save_to_disk().await
    }

    pub async fn add_transaction(&self, tx: Transaction) -> Result<String, BlockchainError> {
        if !tx.verify_signature() {
            return Err(BlockchainError::InvalidTransaction("Signature invalid".into()));
        }
        if tx.amount == 0 && tx.tx_type != TransactionType::LearningReport {
            return Err(BlockchainError::InvalidTransaction("Zero amount".into()));
        }

        let balance = self.get_balance(&tx.from).await?;
        let total_required = tx.amount.saturating_add(tx.founder_share);
        if balance < total_required {
            return Err(BlockchainError::InsufficientBalance {
                have: balance,
                need: total_required,
            });
        }

        let mut pool = self.pending_pool.write().await;
        if pool.len() >= TRANSACTION_POOL_LIMIT {
            return Err(BlockchainError::ConsensusError("Pool full".into()));
        }

        let tx_id = tx.id.clone();
        self.emit_event(BlockchainEvent::NewTransaction {
            tx_id: tx_id.clone(),
            tx_type: tx.tx_type,
        }).await;

        pool.push(tx);
        Ok(tx_id)
    }

    pub async fn get_task(&self) -> Result<(String, u32, u64), BlockchainError> {
        let chain = self.chain.read().await;
        let last = chain.last().ok_or(BlockchainError::ConsensusError("Empty chain".into()))?;
        let next_diff = self.calculate_next_difficulty(last.index, last.difficulty).await?;
        Ok((last.hash.clone(), next_diff, last.index + 1))
    }

    pub async fn submit_solution(
        &self,
        miner_addr: &str,
        nonce: u64,
        cognitive_proof: CognitiveProof,
        model_checkpoint: Option<ModelCheckpoint>,
    ) -> Result<Block, BlockchainError> {
        let mut txs: Vec<Transaction> = {
            let mut pool = self.pending_pool.write().await;
            pool.drain(..).collect()
        };

        let height = {
            let chain = self.chain.read().await;
            chain.len() as u64
        };
        let halvings = height / HALVING_INTERVAL;
        let base_reward = if halvings >= 64 {
            0
        } else {
            INITIAL_BLOCK_REWARD >> halvings
        };

        // Quality-adjusted reward based on cognitive proof
        let quality_multiplier = 0.5 + cognitive_proof.quality_score() * 0.5; // 0.5x to 1.0x
        let reward = (base_reward as f64 * quality_multiplier) as u64;

        // Founder royalty on reward (2%)
        let founder_royalty = reward * FOUNDER_ROYALTY_BPS / 10_000;
        let miner_reward = reward.saturating_sub(founder_royalty);

        // Create reward transaction for miner
        let reward_tx = Transaction {
            id: hash_sha3_256(format!("reward|{}|{}", miner_addr, Utc::now().timestamp()).as_bytes()),
            from: "SYSTEM".into(),
            to: miner_addr.into(),
            amount: miner_reward,
            tx_type: TransactionType::Reward,
            timestamp: Utc::now().timestamp() as u64,
            signature: "SYSTEM".into(),
            data: Some(format!(
                "PoUCW Reward #{} (quality: {:.2})",
                height + 1,
                cognitive_proof.quality_score()
            )),
            founder_share: 0,
            epistemic_context: None,
            merkle_index: None,
        };
        txs.push(reward_tx);

        // Create royalty transaction for founder
        if founder_royalty > 0 {
            let royalty_tx = Transaction {
                id: hash_sha3_256(
                    format!("royalty|{}|{}", height, Utc::now().timestamp()).as_bytes(),
                ),
                from: "SYSTEM".into(),
                to: self.config.founder.genesis_pubkey_hex.clone(),
                amount: founder_royalty,
                tx_type: TransactionType::RevenueShare,
                timestamp: Utc::now().timestamp() as u64,
                signature: "SYSTEM".into(),
                data: Some(format!("Founder Royalty 2% from Block #{}", height + 1)),
                founder_share: 0,
                epistemic_context: None,
                merkle_index: None,
            };
            txs.push(royalty_tx);
        }

        // Assign merkle indices
        for (i, tx) in txs.iter_mut().enumerate() {
            tx.merkle_index = Some(i as u32);
        }

        // Calculate Merkle root
        let tx_ids: Vec<String> = txs.iter().map(|tx| tx.id.clone()).collect();
        let merkle_root = MerkleTree::root(&tx_ids);

        let (prev_hash, last_index, last_difficulty, last_cumulative_work) = {
            let chain = self.chain.read().await;
            let last = chain.last().ok_or(BlockchainError::ConsensusError("Empty chain".into()))?;
            (
                last.hash.clone(),
                last.index,
                last.difficulty,
                last.cumulative_work,
            )
        };

        let mut block = Block {
            index: last_index + 1,
            timestamp: Utc::now().timestamp() as u64,
            transactions: txs,
            previous_hash: prev_hash,
            hash: String::new(),
            nonce,
            cognitive_proof,
            difficulty: last_difficulty,
            cumulative_work: last_cumulative_work.saturating_add(1u64 << last_difficulty.min(63)),
            merkle_root,
            model_checkpoint,
        };

        block.hash = Block::calculate_hash(
            block.index,
            block.timestamp,
            &block.previous_hash,
            &block.merkle_root,
            block.nonce,
            block.difficulty,
            &block.cognitive_proof.proof_hash,
        );

        if !block.meets_difficulty(block.difficulty) {
            return Err(BlockchainError::InvalidBlock(
                "Hash does not meet difficulty".into(),
            ));
        }

        {
            let mut chain = self.chain.write().await;
            chain.push(block.clone());
        }
        {
            let mut minted = self.total_minted.write().await;
            *minted = minted.saturating_add(reward);
        }
        self.update_balance_cache(miner_addr, miner_reward).await;
        if founder_royalty > 0 {
            self.update_balance_cache(&self.config.founder.genesis_pubkey_hex.clone(), founder_royalty).await;
        }

        self.emit_event(BlockchainEvent::NewBlock {
            height: block.index,
            hash: block.hash.clone(),
            reward,
        }).await;

        if let Some(checkpoint) = &block.model_checkpoint {
            self.emit_event(BlockchainEvent::ModelCheckpoint {
                height: block.index,
                model_hash: checkpoint.model_hash.clone(),
            }).await;
        }

        self.save_to_disk().await?;
        Ok(block)
    }

    /// Process founder vesting withdrawal
    pub async fn process_vesting_withdrawal(
        &self,
        amount: u64,
    ) -> Result<Transaction, BlockchainError> {
        let founder_addr = self.config.founder.genesis_pubkey_hex.clone();
        let total_allocation = self.config.founder_genesis_amount();
        let genesis_ts = *self.genesis_timestamp.read().await;
        let current_ts = Utc::now().timestamp() as u64;
        let already_withdrawn = *self.founder_vesting_withdrawn.read().await;

        VestingSchedule::validate_withdrawal(
            total_allocation,
            already_withdrawn,
            genesis_ts,
            current_ts,
            amount,
        )?;

        let tx = Transaction {
            id: hash_sha3_256(
                format!("vesting|{}|{}", founder_addr, current_ts).as_bytes(),
            ),
            from: "SYSTEM".into(),
            to: founder_addr.clone(),
            amount,
            tx_type: TransactionType::GenesisVesting,
            timestamp: current_ts,
            signature: "SYSTEM".into(),
            data: Some(format!(
                "Vesting withdrawal: {} HAFA (total withdrawn: {})",
                amount,
                already_withdrawn + amount
            )),
            founder_share: 0,
            epistemic_context: None,
            merkle_index: None,
        };

        *self.founder_vesting_withdrawn.write().await = already_withdrawn + amount;
        self.update_balance_cache(&founder_addr, amount).await;

        let remaining = total_allocation.saturating_sub(already_withdrawn + amount);
        self.emit_event(BlockchainEvent::VestingReleased { amount, remaining }).await;

        Ok(tx)
    }

    /// Get founder vesting status
    pub async fn get_vesting_status(&self) -> (u64, u64, u64) {
        let total_allocation = self.config.founder_genesis_amount();
        let genesis_ts = *self.genesis_timestamp.read().await;
        let current_ts = Utc::now().timestamp() as u64;
        let already_withdrawn = *self.founder_vesting_withdrawn.read().await;

        let total_vested = VestingSchedule::calculate_vested(total_allocation, genesis_ts, current_ts);
        let available = total_vested.saturating_sub(already_withdrawn);

        (total_allocation, total_vested, available)
    }

    // EXACT BITCOIN DIFFICULTY ADJUSTMENT FORMULA
    async fn calculate_next_difficulty(
        &self,
        last_index: u64,
        last_difficulty: u32,
    ) -> Result<u32, BlockchainError> {
        if last_index % DIFFICULTY_ADJUSTMENT_INTERVAL != 0 || last_index == 0 {
            return Ok(last_difficulty);
        }

        let (ref_timestamp, last_timestamp) = {
            let chain = self.chain.read().await;
            let ref_block = chain
                .get((last_index - DIFFICULTY_ADJUSTMENT_INTERVAL) as usize)
                .ok_or(BlockchainError::ConsensusError(
                    "Reference block not found".into(),
                ))?;
            let last_block = chain
                .last()
                .ok_or(BlockchainError::ConsensusError("Empty chain".into()))?;
            (ref_block.timestamp, last_block.timestamp)
        };

        let actual_time = last_timestamp.saturating_sub(ref_timestamp);
        let expected_time = DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_BLOCK_TIME_SECS;

        let factor = (expected_time as f64 / actual_time as f64).clamp(0.25, 4.0);
        let new_diff = (last_difficulty as f64 * factor).round() as u32;

        let old_diff = last_difficulty;
        let result = new_diff.clamp(MIN_DIFFICULTY, MAX_DIFFICULTY);

        if result != old_diff {
            self.emit_event(BlockchainEvent::DifficultyAdjusted {
                old_difficulty: old_diff,
                new_difficulty: result,
            }).await;
        }

        Ok(result)
    }

    pub async fn get_balance(&self, addr: &str) -> Result<u64, BlockchainError> {
        {
            let cache = self.balance_cache.read().await;
            if let Some(&bal) = cache.get(addr) {
                return Ok(bal);
            }
        }

        let mut balance: i64 = 0;
        let chain = self.chain.read().await;
        for block in chain.iter() {
            for tx in &block.transactions {
                if tx.to == addr {
                    balance = balance.saturating_add(tx.amount as i64);
                }
                if tx.from == addr && tx.from != "SYSTEM" {
                    balance = balance.saturating_sub((tx.amount + tx.founder_share) as i64);
                }
            }
        }

        let final_bal = if balance < 0 { 0 } else { balance as u64 };
        {
            let mut cache = self.balance_cache.write().await;
            cache.insert(addr.to_string(), final_bal);
        }
        Ok(final_bal)
    }

    async fn update_balance_cache(&self, addr: &str, amount: u64) {
        let mut cache = self.balance_cache.write().await;
        let current = cache.get(addr).copied().unwrap_or(0);
        cache.insert(addr.to_string(), current.saturating_add(amount));
    }

    async fn save_to_disk(&self) -> Result<(), BlockchainError> {
        let chain = self.chain.read().await;
        let total_minted = *self.total_minted.read().await;
        let balance_cache = self.balance_cache.read().await.clone();
        let founder_vesting_withdrawn = *self.founder_vesting_withdrawn.read().await;
        let genesis_timestamp = *self.genesis_timestamp.read().await;

        let state = SerializableState {
            chain: chain.clone(),
            total_minted,
            balance_cache,
            founder_vesting_withdrawn,
            genesis_timestamp,
        };

        let data =
            bincode::serialize(&state).map_err(|e| BlockchainError::StorageError(e.to_string()))?;
        let tmp = self.storage_path.with_extension("tmp");
        fs::write(&tmp, &data).map_err(|e| BlockchainError::StorageError(e.to_string()))?;
        fs::rename(&tmp, &self.storage_path)
            .map_err(|e| BlockchainError::StorageError(e.to_string()))?;
        Ok(())
    }

    async fn load_from_disk(&mut self) -> Result<(), BlockchainError> {
        let data =
            fs::read(&self.storage_path).map_err(|e| BlockchainError::StorageError(e.to_string()))?;

        // Try new format first
        match bincode::deserialize::<SerializableState>(&data) {
            Ok(state) => {
                *self.chain.write().await = state.chain;
                *self.total_minted.write().await = state.total_minted;
                *self.balance_cache.write().await = state.balance_cache;
                *self.founder_vesting_withdrawn.write().await = state.founder_vesting_withdrawn;
                *self.genesis_timestamp.write().await = state.genesis_timestamp;
                Ok(())
            }
            Err(_) => {
                // Migration from old format
                println!("   Old format detected. Migrating...");
                match bincode::deserialize::<Vec<Block>>(&data) {
                    Ok(chain) => {
                        let mut total_minted = 0u64;
                        let mut balance_cache = HashMap::new();

                        for block in &chain {
                            for tx in &block.transactions {
                                if tx.tx_type == TransactionType::Reward && tx.from == "SYSTEM" {
                                    total_minted = total_minted.saturating_add(tx.amount);
                                }
                                if tx.from != "SYSTEM" {
                                    *balance_cache
                                        .entry(tx.from.clone())
                                        .or_insert(0u64) = balance_cache
                                        .get(&tx.from)
                                        .copied()
                                        .unwrap_or(0)
                                        .saturating_sub(tx.amount + tx.founder_share);
                                }
                                *balance_cache
                                    .entry(tx.to.clone())
                                    .or_insert(0u64) = balance_cache
                                    .get(&tx.to)
                                    .copied()
                                    .unwrap_or(0)
                                    .saturating_add(tx.amount);
                            }
                        }

                        let genesis_timestamp = chain.first().map(|b| b.timestamp).unwrap_or(0);

                        *self.chain.write().await = chain;
                        *self.total_minted.write().await = total_minted;
                        *self.balance_cache.write().await = balance_cache;
                        *self.genesis_timestamp.write().await = genesis_timestamp;
                        *self.founder_vesting_withdrawn.write().await = 0;

                        self.save_to_disk().await?;
                        Ok(())
                    }
                    Err(_) => {
                        let _ = fs::remove_file(&self.storage_path);
                        self.create_genesis_block().await
                    }
                }
            }
        }
    }

    pub async fn validate_chain(&self) -> Result<bool, BlockchainError> {
        let chain = self.chain.read().await;
        if chain.is_empty() {
            return Ok(true);
        }

        let genesis = &chain[0];
        if genesis.index != 0 || !genesis.previous_hash.chars().all(|c| c == '0') {
            return Ok(false);
        }

        for i in 1..chain.len() {
            let curr = &chain[i];
            let prev = &chain[i - 1];

            if curr.index != prev.index + 1 {
                return Ok(false);
            }
            if curr.previous_hash != prev.hash {
                return Ok(false);
            }
            if !curr.meets_difficulty(curr.difficulty) {
                return Ok(false);
            }

            // Verify Merkle root
            let tx_ids: Vec<String> = curr.transactions.iter().map(|tx| tx.id.clone()).collect();
            let expected_merkle = MerkleTree::root(&tx_ids);
            if expected_merkle != curr.merkle_root {
                return Ok(false);
            }

            let expected_hash = Block::calculate_hash(
                curr.index,
                curr.timestamp,
                &curr.previous_hash,
                &curr.merkle_root,
                curr.nonce,
                curr.difficulty,
                &curr.cognitive_proof.proof_hash,
            );
            if expected_hash != curr.hash {
                return Ok(false);
            }

            if (curr.timestamp as i64) < (prev.timestamp as i64) {
                return Ok(false);
            }
            if (curr.timestamp as i64) > (Utc::now().timestamp() + MAX_BLOCK_TIME_DRIFT_SECS) {
                return Ok(false);
            }

            for tx in &curr.transactions {
                if !tx.verify_signature() {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Get Merkle proof for a transaction in a block
    pub async fn get_transaction_proof(
        &self,
        block_height: u64,
        tx_id: &str,
    ) -> Result<Vec<(String, bool)>, BlockchainError> {
        let chain = self.chain.read().await;
        let block = chain
            .get(block_height as usize)
            .ok_or(BlockchainError::ConsensusError("Block not found".into()))?;

        let tx_ids: Vec<String> = block.transactions.iter().map(|tx| tx.id.clone()).collect();
        let target_index = tx_ids
            .iter()
            .position(|id| id == tx_id)
            .ok_or(BlockchainError::InvalidTransaction("Transaction not found".into()))?;

        MerkleTree::proof(&tx_ids, target_index)
    }

    pub async fn get_chain_height(&self) -> u64 {
        self.chain.read().await.len() as u64
    }

    pub async fn get_total_minted(&self) -> u64 {
        *self.total_minted.read().await
    }

    pub async fn get_current_reward(&self) -> u64 {
        let height = self.chain.read().await.len() as u64;
        let halvings = height / HALVING_INTERVAL;
        if halvings >= 64 {
            0
        } else {
            INITIAL_BLOCK_REWARD >> halvings
        }
    }

    pub async fn get_latest_model_checkpoint(&self) -> Option<ModelCheckpoint> {
        let chain = self.chain.read().await;
        chain
            .iter()
            .rev()
            .find_map(|block| block.model_checkpoint.clone())
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_root_empty() {
        let root = MerkleTree::root(&[]);
        assert_eq!(root, "0".repeat(64));
    }

    #[test]
    fn test_merkle_root_single() {
        let root = MerkleTree::root(&["tx1".to_string()]);
        assert_eq!(root, "tx1");
    }

    #[test]
    fn test_merkle_root_multiple() {
        let txs = vec!["tx1".to_string(), "tx2".to_string(), "tx3".to_string()];
        let root = MerkleTree::root(&txs);
        assert!(!root.is_empty());
        assert_ne!(root, "0".repeat(64));
    }

    #[test]
    fn test_merkle_proof_verification() {
        let txs = vec![
            "tx1".to_string(),
            "tx2".to_string(),
            "tx3".to_string(),
            "tx4".to_string(),
        ];
        let root = MerkleTree::root(&txs);
        let proof = MerkleTree::proof(&txs, 1).unwrap();
        assert!(MerkleTree::verify_proof(&txs[1], &proof, &root));
    }

    #[test]
    fn test_cognitive_proof_quality_score() {
        let proof = CognitiveProof::new(
            "hash1".into(),
            "hash2".into(),
            1.0,
            0.5,
            100,
            0.9,
            ResourceUsage {
                cpu_percent: 50.0,
                ram_mb: 1024,
                gpu_percent: 0.0,
                gpu_memory_mb: 0,
            },
            1000,
        );

        let score = proof.quality_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_vesting_schedule() {
        let total = 10_500_000;
        let genesis_ts = 1000;

        // At genesis, nothing vested
        assert_eq!(VestingSchedule::calculate_vested(total, genesis_ts, 1000), 0);

        // After 1 year, 10% vested
        let one_year = genesis_ts + VESTING_CLIFF_SECS;
        assert_eq!(
            VestingSchedule::calculate_vested(total, genesis_ts, one_year),
            total * 10 / 100
        );

        // After 2 years, 40% vested (10% + 30%)
        let two_years = genesis_ts + VESTING_CLIFF_SECS * 2;
        assert_eq!(
            VestingSchedule::calculate_vested(total, genesis_ts, two_years),
            total * 40 / 100
        );

        // After 4 years, 100% vested
        let four_years = genesis_ts + VESTING_CLIFF_SECS * 4;
        assert_eq!(
            VestingSchedule::calculate_vested(total, genesis_ts, four_years),
            total
        );
    }

    #[test]
    fn test_founder_royalty_calculation() {
        let amount = 10_000u64;
        let royalty = amount * FOUNDER_ROYALTY_BPS / 10_000;
        assert_eq!(royalty, 200); // 2% of 10,000 = 200
    }

    #[test]
    fn test_difficulty_target() {
        let target = Block::calculate_target(8);
        assert_eq!(target[0], 0x00);
        assert_eq!(target[1], 0xFF);

        let target = Block::calculate_target(16);
        assert_eq!(target[0], 0x00);
        assert_eq!(target[1], 0x00);
        assert_eq!(target[2], 0xFF);
    }
}