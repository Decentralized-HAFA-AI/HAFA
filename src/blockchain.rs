// ============================================================================
// HAFA - src/blockchain.rs — PURE BITCOIN-STYLE CONSENSUS
// ============================================================================

use crate::config::{
    Config, INITIAL_BLOCK_REWARD, HALVING_INTERVAL, 
    TARGET_BLOCK_TIME_SECS, DIFFICULTY_ADJUSTMENT_INTERVAL,
};
use crate::crypto::{hash_sha3_256, verify_hex_signature, KeyPair};
use crate::epistemic::EpistemicState;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;
use std::sync::Arc;
use std::collections::HashMap;
use std::path::PathBuf;
use std::fs;
use chrono::Utc;
use hex::ToHex;
use sha3::Digest;

const MAX_DIFFICULTY: u32 = 64;
const MIN_DIFFICULTY: u32 = 1; // Pure Bitcoin genesis difficulty
const MAX_BLOCK_TIME_DRIFT_SECS: i64 = 7200;
const TRANSACTION_POOL_LIMIT: usize = 10_000;

#[derive(Error, Debug)]
pub enum BlockchainError {
    #[error("Invalid transaction: {0}")] InvalidTransaction(String),
    #[error("Invalid block: {0}")] InvalidBlock(String),
    #[error("Chain validation failed at block {0}")] ChainValidationFailed(u64),
    #[error("Insufficient balance: have {have}, need {need}")] InsufficientBalance { have: u64, need: u64 },
    #[error("Storage error: {0}")] StorageError(String),
    #[error("Consensus error: {0}")] ConsensusError(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionType { Transfer, Reward, RevenueShare, CognitiveWork, ContractCall, Governance }

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub index: u64, 
    pub timestamp: u64, 
    pub transactions: Vec<Transaction>,
    pub previous_hash: String, 
    pub hash: String, 
    pub nonce: u64,
    pub cognitive_proof: String, 
    pub difficulty: u32, 
    pub cumulative_work: u64,
}

#[derive(Serialize, Deserialize)]
struct SerializableState {
    chain: Vec<Block>, 
    total_minted: u64, 
    balance_cache: HashMap<String, u64>,
}

pub struct Blockchain {
    config: Config,
    chain: Arc<RwLock<Vec<Block>>>,
    pending_pool: Arc<RwLock<Vec<Transaction>>>,
    storage_path: PathBuf,
    total_minted: Arc<RwLock<u64>>,
    balance_cache: Arc<RwLock<HashMap<String, u64>>>,
}

impl Transaction {
    pub fn new(from: String, to: String, amount: u64, tx_type: TransactionType, keypair: &KeyPair, data: Option<String>) -> Result<Self, BlockchainError> {
        let timestamp = Utc::now().timestamp() as u64;
        let message = format!("{}|{}|{}|{}|{}", from, to, amount, timestamp, tx_type as i32);
        let signature = keypair.sign(message.as_bytes());
        Ok(Self {
            id: hash_sha3_256(format!("{}|{}|{}", from, timestamp, rand::random::<u64>()).as_bytes()),
            from, to, amount, tx_type, timestamp,
            signature: signature.to_bytes().encode_hex::<String>(),
            data, founder_share: 0, epistemic_context: None,
        })
    }

    pub fn verify_signature(&self) -> bool {
        let message = format!("{}|{}|{}|{}|{}", self.from, self.to, self.amount, self.timestamp, self.tx_type as i32);
        verify_hex_signature(&self.from, message.as_bytes(), &self.signature).is_ok()
    }
}

impl Block {
    pub fn calculate_hash(index: u64, timestamp: u64, prev_hash: &str, txs: &[Transaction], nonce: u64, difficulty: u32, cognitive_proof: &str) -> String {
        let mut hasher = sha3::Sha3_256::new();
        hasher.update(index.to_le_bytes());
        hasher.update(timestamp.to_le_bytes());
        hasher.update(prev_hash.as_bytes());
        hasher.update(nonce.to_le_bytes());
        hasher.update(difficulty.to_le_bytes());
        hasher.update(cognitive_proof.as_bytes());
        for tx in txs { hasher.update(tx.id.as_bytes()); }
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
        for i in 0..zero_bytes.min(32) { target[i] = 0x00; }
        if zero_bytes < 32 { target[zero_bytes] = 0xFF << zero_bits; }
        target
    }
}

impl Blockchain {
    pub async fn new(config: &Config) -> Result<Self, BlockchainError> {
        let storage_path = config.storage.data_dir.join("blockchain.dat");
        let chain = Arc::new(RwLock::new(Vec::new()));
        let pending_pool = Arc::new(RwLock::new(Vec::new()));
        let total_minted = Arc::new(RwLock::new(0));
        let balance_cache = Arc::new(RwLock::new(HashMap::new()));

        let _ = fs::create_dir_all(&config.storage.data_dir);

        let mut bc = Self {
            config: config.clone(), chain: chain.clone(), pending_pool,
            storage_path: storage_path.clone(), total_minted, balance_cache,
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

    async fn create_genesis_block(&mut self) -> Result<(), BlockchainError> {
        let now = Utc::now().timestamp() as u64;
        let founder_addr = &self.config.founder.genesis_pubkey_hex;
        let genesis_tx = Transaction {
            id: hash_sha3_256(b"genesis"), from: "SYSTEM".into(), to: founder_addr.clone(),
            amount: 0, tx_type: TransactionType::Reward, timestamp: now,
            signature: "GENESIS".into(), data: Some("HAFA Genesis".into()),
            founder_share: 0, epistemic_context: Some(EpistemicState::new(1.0, true, 0, 1.0)),
        };
        let mut genesis = Block {
            index: 0, timestamp: now, transactions: vec![genesis_tx],
            previous_hash: "0".repeat(64), hash: String::new(), nonce: 0,
            cognitive_proof: "GENESIS_PROOF".into(), difficulty: MIN_DIFFICULTY, cumulative_work: 1,
        };
        genesis.hash = Block::calculate_hash(genesis.index, genesis.timestamp, &genesis.previous_hash, &genesis.transactions, genesis.nonce, genesis.difficulty, &genesis.cognitive_proof);
        
        let founder_share = self.config.founder_genesis_amount();
        *self.total_minted.write().await = founder_share;
        self.update_balance_cache(founder_addr, founder_share).await;
        self.chain.write().await.push(genesis);
        self.save_to_disk().await
    }

    pub async fn add_transaction(&self, tx: Transaction) -> Result<String, BlockchainError> {
        if !tx.verify_signature() { return Err(BlockchainError::InvalidTransaction("Signature invalid".into())); }
        if tx.amount == 0 { return Err(BlockchainError::InvalidTransaction("Zero amount".into())); }
        let balance = self.get_balance(&tx.from).await?;
        let total_required = tx.amount.saturating_add(tx.founder_share);
        if balance < total_required { return Err(BlockchainError::InsufficientBalance { have: balance, need: total_required }); }
        let mut pool = self.pending_pool.write().await;
        if pool.len() >= TRANSACTION_POOL_LIMIT { return Err(BlockchainError::ConsensusError("Pool full".into())); }
        let tx_id = tx.id.clone();
        pool.push(tx);
        Ok(tx_id)
    }

    pub async fn get_task(&self) -> Result<(String, u32, u64), BlockchainError> {
        let chain = self.chain.read().await;
        let last = chain.last().ok_or(BlockchainError::ConsensusError("Empty chain".into()))?;
        let next_diff = self.calculate_next_difficulty(last.index, last.difficulty).await?;
        Ok((last.hash.clone(), next_diff, last.index + 1))
    }

    pub async fn submit_solution(&self, miner_addr: &str, nonce: u64, cognitive_proof: &str) -> Result<Block, BlockchainError> {
        let mut txs: Vec<Transaction> = {
            let mut pool = self.pending_pool.write().await;
            pool.drain(..).collect()
        };

        let height = { let chain = self.chain.read().await; chain.len() as u64 };
        let halvings = height / HALVING_INTERVAL;
        let reward = if halvings >= 64 { 0 } else { INITIAL_BLOCK_REWARD >> halvings };

        let reward_tx = Transaction {
            id: hash_sha3_256(format!("reward|{}|{}", miner_addr, Utc::now().timestamp()).as_bytes()),
            from: "SYSTEM".into(), to: miner_addr.into(), amount: reward,
            tx_type: TransactionType::Reward, timestamp: Utc::now().timestamp() as u64,
            signature: "SYSTEM".into(), data: Some(format!("PoUCW Reward #{}", height + 1)),
            founder_share: 0, epistemic_context: None,
        };
        txs.push(reward_tx);

        let (prev_hash, last_index, last_difficulty, last_cumulative_work) = {
            let chain = self.chain.read().await;
            let last = chain.last().ok_or(BlockchainError::ConsensusError("Empty chain".into()))?;
            (last.hash.clone(), last.index, last.difficulty, last.cumulative_work)
        };

        let mut block = Block {
            index: last_index + 1, timestamp: Utc::now().timestamp() as u64,
            transactions: txs, previous_hash: prev_hash, hash: String::new(),
            nonce, cognitive_proof: cognitive_proof.into(),
            difficulty: last_difficulty,
            cumulative_work: last_cumulative_work.saturating_add(1u64 << last_difficulty.min(63)),
        };

        block.hash = Block::calculate_hash(block.index, block.timestamp, &block.previous_hash, &block.transactions, block.nonce, block.difficulty, &block.cognitive_proof);

        if !block.meets_difficulty(block.difficulty) {
            return Err(BlockchainError::InvalidBlock("Hash does not meet difficulty".into()));
        }

        { let mut chain = self.chain.write().await; chain.push(block.clone()); }
        { let mut minted = self.total_minted.write().await; *minted = minted.saturating_add(reward); }
        self.update_balance_cache(miner_addr, reward).await;
        self.save_to_disk().await?;
        Ok(block)
    }

    // EXACT BITCOIN DIFFICULTY ADJUSTMENT FORMULA
    async fn calculate_next_difficulty(&self, last_index: u64, last_difficulty: u32) -> Result<u32, BlockchainError> {
        if last_index % DIFFICULTY_ADJUSTMENT_INTERVAL != 0 || last_index == 0 {
            return Ok(last_difficulty);
        }
        
        let (ref_timestamp, last_timestamp) = {
            let chain = self.chain.read().await;
            let ref_block = chain.get((last_index - DIFFICULTY_ADJUSTMENT_INTERVAL) as usize)
                .ok_or(BlockchainError::ConsensusError("Reference block not found".into()))?;
            let last_block = chain.last()
                .ok_or(BlockchainError::ConsensusError("Empty chain".into()))?;
            (ref_block.timestamp, last_block.timestamp)
        };

        let actual_time = last_timestamp.saturating_sub(ref_timestamp);
        let expected_time = DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_BLOCK_TIME_SECS; // 2016 * 600
        
        let factor = (expected_time as f64 / actual_time as f64).clamp(0.25, 4.0);
        let new_diff = (last_difficulty as f64 * factor).round() as u32;
        Ok(new_diff.clamp(MIN_DIFFICULTY, MAX_DIFFICULTY))
    }

    pub async fn get_balance(&self, addr: &str) -> Result<u64, BlockchainError> {
        { let cache = self.balance_cache.read().await; if let Some(&bal) = cache.get(addr) { return Ok(bal); } }
        let mut balance: i64 = 0;
        let chain = self.chain.read().await;
        for block in chain.iter() {
            for tx in &block.transactions {
                if tx.to == addr { balance = balance.saturating_add(tx.amount as i64); }
                if tx.from == addr { balance = balance.saturating_sub((tx.amount + tx.founder_share) as i64); }
            }
        }
        let final_bal = if balance < 0 { 0 } else { balance as u64 };
        { let mut cache = self.balance_cache.write().await; cache.insert(addr.to_string(), final_bal); }
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
        let state = SerializableState { chain: chain.clone(), total_minted, balance_cache };
        let data = bincode::serialize(&state).map_err(|e| BlockchainError::StorageError(e.to_string()))?;
        let tmp = self.storage_path.with_extension("tmp");
        fs::write(&tmp, &data).map_err(|e| BlockchainError::StorageError(e.to_string()))?;
        fs::rename(&tmp, &self.storage_path).map_err(|e| BlockchainError::StorageError(e.to_string()))?;
        Ok(())
    }

    async fn load_from_disk(&mut self) -> Result<(), BlockchainError> {
        let data = fs::read(&self.storage_path).map_err(|e| BlockchainError::StorageError(e.to_string()))?;
        match bincode::deserialize::<SerializableState>(&data) {
            Ok(state) => {
                *self.chain.write().await = state.chain;
                *self.total_minted.write().await = state.total_minted;
                *self.balance_cache.write().await = state.balance_cache;
                Ok(())
            }
            Err(_) => {
                println!("   Old format detected. Migrating...");
                match bincode::deserialize::<Vec<Block>>(&data) {
                    Ok(chain) => {
                        let mut total_minted = 0u64;
                        let mut balance_cache = HashMap::new();
                        for block in &chain {
                            for tx in &block.transactions {
                                if tx.tx_type == TransactionType::Reward && tx.from == "SYSTEM" { total_minted = total_minted.saturating_add(tx.amount); }
                                if tx.from != "SYSTEM" { *balance_cache.entry(tx.from.clone()).or_insert(0u64) = balance_cache.get(&tx.from).copied().unwrap_or(0).saturating_sub(tx.amount + tx.founder_share); }
                                *balance_cache.entry(tx.to.clone()).or_insert(0u64) = balance_cache.get(&tx.to).copied().unwrap_or(0).saturating_add(tx.amount);
                            }
                        }
                        *self.chain.write().await = chain;
                        *self.total_minted.write().await = total_minted;
                        *self.balance_cache.write().await = balance_cache;
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
        if chain.is_empty() { return Ok(true); }
        let genesis = &chain[0];
        if genesis.index != 0 || !genesis.previous_hash.chars().all(|c| c == '0') { return Ok(false); }
        for i in 1..chain.len() {
            let curr = &chain[i]; let prev = &chain[i-1];
            if curr.index != prev.index + 1 { return Ok(false); }
            if curr.previous_hash != prev.hash { return Ok(false); }
            if !curr.meets_difficulty(curr.difficulty) { return Ok(false); }
            let expected_hash = Block::calculate_hash(curr.index, curr.timestamp, &curr.previous_hash, &curr.transactions, curr.nonce, curr.difficulty, &curr.cognitive_proof);
            if expected_hash != curr.hash { return Ok(false); }
            if (curr.timestamp as i64) < (prev.timestamp as i64) { return Ok(false); }
            if (curr.timestamp as i64) > (Utc::now().timestamp() + MAX_BLOCK_TIME_DRIFT_SECS) { return Ok(false); }
            for tx in &curr.transactions { if !tx.verify_signature() { return Ok(false); } }
        }
        Ok(true)
    }

    pub async fn get_chain_height(&self) -> u64 { self.chain.read().await.len() as u64 }
    pub async fn get_total_minted(&self) -> u64 { *self.total_minted.read().await }
    pub async fn get_current_reward(&self) -> u64 {
        let height = self.chain.read().await.len() as u64;
        let halvings = height / HALVING_INTERVAL;
        if halvings >= 64 { 0 } else { INITIAL_BLOCK_REWARD >> halvings }
    }
}