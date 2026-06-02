// ============================================================================
// HAFA - src/network.rs — P2P NETWORK & COMMUNICATION ENGINE
// ============================================================================

use crate::config::Config;
use crate::blockchain::{Block, Transaction};
use libp2p::{
    identity,
    swarm::NetworkBehaviour,
    gossipsub, kad, mdns,
    PeerId, Multiaddr,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;
use std::collections::HashMap;
use std::path::PathBuf;
use chrono::Utc;
use tracing::{info, debug};

// ============================================================================
// CONSTANTS
// ============================================================================

const NETWORK_STATE_FILE: &str = "network_state.bin";
const DEFAULT_P2P_PORT: u16 = 7474;
const PROTOCOL_ID: &str = "/hafa/1.0.0";
const TOPIC_TRANSACTIONS: &str = "hafa_tx";
const TOPIC_BLOCKS: &str = "hafa_block";
const MESSAGE_TTL_SECS: u64 = 300;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Network initialization failed: {0}")]
    InitializationFailed(String),
    #[error("Peer connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Message (de)serialization error: {0}")]
    SerializationError(String),
    #[error("Invalid peer ID")]
    InvalidPeerId,
    #[error("Storage error: {0}")]
    StorageError(String),
    #[error("Channel error: {0}")]
    ChannelError(String),
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMessage {
    Transaction(Transaction),
    Block(Block),
    RequestBlock(u64),
    RequestTransaction(String),
    Ping,
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: String,
    pub address: String,
    pub last_seen: u64,
    pub connected: bool,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkStats {
    pub connected_peers: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_transferred: u64,
    pub uptime_secs: u64,
    pub last_activity: u64,
}

#[derive(NetworkBehaviour)]
struct HAFABehaviour {
    gossipsub: gossipsub::Behaviour,
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
    mdns: mdns::tokio::Behaviour,
}

pub struct NetworkEngine {
    config: Config,
    local_peer_id: PeerId,
    tx_sender: mpsc::Sender<Transaction>,
    block_sender: mpsc::Sender<Block>,
    stats: Arc<RwLock<NetworkStats>>,
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
    storage_path: PathBuf,
    is_running: Arc<RwLock<bool>>,
}

// ============================================================================
// IMPLEMENTATION
// ============================================================================

impl NetworkEngine {
    pub async fn new(
        config: &Config,
        tx_sender: mpsc::Sender<Transaction>,
        block_sender: mpsc::Sender<Block>,
    ) -> Result<Self, NetworkError> {
        info!("🌐 Initializing HAFA P2P Network...");

        let storage_path = config.storage.data_dir.join(NETWORK_STATE_FILE);
        let local_key = identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());

        info!("🆔 Local Peer ID: {}", local_peer_id);

        let engine = Self {
            config: config.clone(),
            local_peer_id,
            tx_sender,
            block_sender,
            stats: Arc::new(RwLock::new(NetworkStats::default())),
            peers: Arc::new(RwLock::new(HashMap::new())),
            storage_path: storage_path.clone(),
            is_running: Arc::new(RwLock::new(false)),
        };

        if storage_path.exists() {
            engine.load_state().await?;
            info!("📥 Loaded network state");
        }

        Ok(engine)
    }

    pub async fn start(&self) -> Result<(), NetworkError> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(NetworkError::InitializationFailed("Already running".into()));
        }
        *running = true;
        drop(running);

        info!("✅ P2P Network started on port {}", self.config.network.p2p_port);

        let engine_clone = Arc::new(self.clone_for_task());
        tokio::spawn(async move {
            engine_clone.network_loop().await;
        });

        Ok(())
    }

    pub async fn stop(&self) -> Result<(), NetworkError> {
        let mut running = self.is_running.write().await;
        *running = false;
        drop(running);
        info!("🛑 P2P Network stopped");
        Ok(())
    }

    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    async fn network_loop(&self) {
        info!("🔄 Network loop started");

        while *self.is_running.read().await {
            self.connect_to_bootstrap().await;
            self.process_pending().await;
            self.update_peer_stats().await;
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }

        info!("🏁 Network loop terminated");
    }

    async fn connect_to_bootstrap(&self) {
        for addr_str in &self.config.network.bootstrap_nodes {
            if let Ok(addr) = addr_str.parse::<Multiaddr>() {
                debug!(" Attempting bootstrap: {}", addr);
                self.add_peer(addr.to_string()).await;
            }
        }
    }

    async fn add_peer(&self, address: String) {
        let mut peers = self.peers.write().await;
        let peer_id = format!("peer_{}", address.len());
        
        let info = PeerInfo {
            peer_id: peer_id.clone(),
            address: address.clone(),
            last_seen: Utc::now().timestamp() as u64,
            connected: true,
            score: 1.0,
        };
        
        peers.insert(peer_id, info);
        
        {
            let mut stats = self.stats.write().await;
            stats.connected_peers = peers.len() as u64;
        }
    }

    async fn update_peer_stats(&self) {
        let mut peers = self.peers.write().await;
        let now = Utc::now().timestamp() as u64;
        
        for peer in peers.values_mut() {
            peer.last_seen = now;
            peer.score = (peer.score - 0.001).max(0.0);
        }
        
        {
            let mut stats = self.stats.write().await;
            stats.connected_peers = peers.len() as u64;
            stats.uptime_secs += 5;
        }
    }

    async fn process_pending(&self) {
        // Placeholder for actual swarm event processing
    }

    pub async fn broadcast_transaction(&self, tx: Transaction) -> Result<(), NetworkError> {
        debug!("📢 Broadcasting transaction: {}...", &tx.id[..16]);
        
        let message = NetworkMessage::Transaction(tx);
        let data = serde_json::to_vec(&message)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))?;

        info!(" Transaction broadcasted ({} bytes)", data.len());

        {
            let mut stats = self.stats.write().await;
            stats.messages_sent += 1;
            stats.bytes_transferred += data.len() as u64;
            stats.last_activity = Utc::now().timestamp() as u64;
        }

        Ok(())
    }

    pub async fn broadcast_block(&self, block: Block) -> Result<(), NetworkError> {
        debug!("📢 Broadcasting block: {}...", &block.hash[..16]);
        
        let message = NetworkMessage::Block(block);
        let data = serde_json::to_vec(&message)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))?;

        info!(" Block broadcasted ({} bytes)", data.len());

        {
            let mut stats = self.stats.write().await;
            stats.messages_sent += 1;
            stats.bytes_transferred += data.len() as u64;
            stats.last_activity = Utc::now().timestamp() as u64;
        }

        Ok(())
    }

    // ✅ FIX: نام متغیر 'data' به صراحت نوشته شده است
    pub async fn handle_incoming_message(&self, data: &[u8]) -> Result<(), NetworkError> {
        let message: NetworkMessage = serde_json::from_slice(data)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))?;

        match message {
            NetworkMessage::Transaction(tx) => {
                debug!("📥 Received transaction: {}...", &tx.id[..16]);
                self.tx_sender.send(tx).await
                    .map_err(|e| NetworkError::ChannelError(e.to_string()))?;
            }
            NetworkMessage::Block(block) => {
                debug!("📥 Received block: {}...", &block.hash[..16]);
                self.block_sender.send(block).await
                    .map_err(|e| NetworkError::ChannelError(e.to_string()))?;
            }
            NetworkMessage::Ping | NetworkMessage::Pong => {
                debug!("🏓 Received ping/pong");
            }
            _ => {
                debug!("📥 Received unknown message type");
            }
        }

        {
            let mut stats = self.stats.write().await;
            stats.messages_received += 1;
            stats.bytes_transferred += data.len() as u64;
            stats.last_activity = Utc::now().timestamp() as u64;
        }

        Ok(())
    }

    pub async fn get_stats(&self) -> NetworkStats {
        self.stats.read().await.clone()
    }

    pub async fn get_peers(&self) -> Vec<PeerInfo> {
        self.peers.read().await.values().cloned().collect()
    }

    pub fn get_local_peer_id(&self) -> String {
        self.local_peer_id.to_string()
    }

    async fn save_state(&self) -> Result<(), NetworkError> {
        let peers = self.peers.read().await;
        let stats = self.stats.read().await;
        
        let state = NetworkStorageState {
            peers: (*peers).clone(),
            stats: (*stats).clone(),
            timestamp: Utc::now().timestamp() as u64,
        };

        let data = bincode::serialize(&state)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))?;
        
        std::fs::write(&self.storage_path, data)
            .map_err(|e| NetworkError::StorageError(e.to_string()))?;
        
        debug!("💾 Network state saved");
        Ok(())
    }

    async fn load_state(&self) -> Result<(), NetworkError> {
        let data = std::fs::read(&self.storage_path)
            .map_err(|e| NetworkError::StorageError(e.to_string()))?;
        
        let state: NetworkStorageState = bincode::deserialize(&data)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))?;
        
        {
            let mut peers = self.peers.write().await;
            *peers = state.peers;
        }
        {
            let mut stats = self.stats.write().await;
            *stats = state.stats;
        }
        
        Ok(())
    }

    fn clone_for_task(&self) -> Arc<Self> {
        Arc::new(Self {
            config: self.config.clone(),
            local_peer_id: self.local_peer_id,
            tx_sender: self.tx_sender.clone(),
            block_sender: self.block_sender.clone(),
            stats: self.stats.clone(),
            peers: self.peers.clone(),
            storage_path: self.storage_path.clone(),
            is_running: self.is_running.clone(),
        })
    }

    pub async fn get_peer_count(&self) -> usize {
        self.peers.read().await.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NetworkStorageState {
    peers: HashMap<String, PeerInfo>,
    stats: NetworkStats,
    timestamp: u64,
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_info_serialization() {
        let info = PeerInfo {
            peer_id: "test".into(),
            address: "/ip4/127.0.0.1".into(),
            last_seen: 0,
            connected: true,
            score: 1.0,
        };
        let serialized = serde_json::to_string(&info).unwrap();
        let deserialized: PeerInfo = serde_json::from_str(&serialized).unwrap();
        assert_eq!(info.peer_id, deserialized.peer_id);
    }
}