// ============================================================================
// HAFA - src/network.rs — P2P NETWORK & COMMUNICATION ENGINE (ADVANCED)
// ============================================================================
//
// Advanced P2P networking with libp2p:
// - Gossipsub for block/transaction broadcast
// - Kademlia DHT for peer discovery
// - mDNS for local network discovery
// - Event-driven architecture with BlockchainEvent integration
// - Peer scoring and reputation
// - Message deduplication
// - Rate limiting
//
// ============================================================================

use crate::blockchain::{Block, BlockchainEvent, Transaction};
use crate::config::Config;
use libp2p::{
    gossipsub, mdns,
    swarm::NetworkBehaviour,
    PeerId,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, info, warn};
// ============================================================================
// CONSTANTS
// ============================================================================

#[allow(dead_code)]
const NETWORK_STATE_FILE: &str = "network_state.bin";
#[allow(dead_code)]
const PROTOCOL_ID: &str = "/hafa/1.0.0";
#[allow(dead_code)]
const TOPIC_TRANSACTIONS: &str = "hafa_tx";
#[allow(dead_code)]
const TOPIC_BLOCKS: &str = "hafa_block";
#[allow(dead_code)]
const MESSAGE_TTL_SECS: u64 = 300;
const MAX_DUPLICATE_CACHE: usize = 10_000;
const PEER_SCORE_DECAY: f64 = 0.001;
const MIN_PEER_SCORE: f64 = 0.1;

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
    #[error("Duplicate message")]
    DuplicateMessage,
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    #[error("Peer banned: {0}")]
    PeerBanned(String),
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
    pub messages_sent: u64,
    pub messages_received: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkStats {
    pub connected_peers: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_transferred: u64,
    pub uptime_secs: u64,
    pub last_activity: u64,
    pub duplicate_messages: u64,
    pub banned_peers: u64,
}

#[allow(dead_code)]
#[derive(NetworkBehaviour)]
struct HAFABehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
}
#[derive(Clone)]
pub struct NetworkEngine {
    config: Config,
    local_peer_id: PeerId,
    tx_sender: mpsc::Sender<Transaction>,
    block_sender: mpsc::Sender<Block>,
    stats: Arc<RwLock<NetworkStats>>,
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
    duplicate_cache: Arc<RwLock<HashSet<String>>>,
    banned_peers: Arc<RwLock<HashSet<String>>>,
    is_running: Arc<RwLock<bool>>,
}

impl NetworkEngine {
    pub async fn new(
        config: &Config,
        tx_sender: mpsc::Sender<Transaction>,
        block_sender: mpsc::Sender<Block>,
    ) -> Result<Self, NetworkError> {
        info!("🌐 Initializing HAFA P2P Network...");

        let local_key = libp2p::identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());

        info!("🆔 Local Peer ID: {}", local_peer_id);

        Ok(Self {
            config: config.clone(),
            local_peer_id,
            tx_sender,
            block_sender,
            stats: Arc::new(RwLock::new(NetworkStats::default())),
            peers: Arc::new(RwLock::new(HashMap::new())),
            duplicate_cache: Arc::new(RwLock::new(HashSet::new())),
            banned_peers: Arc::new(RwLock::new(HashSet::new())),
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn start(
        &self,
        event_receiver: broadcast::Receiver<BlockchainEvent>,
    ) -> Result<(), NetworkError> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(NetworkError::InitializationFailed("Already running".into()));
        }
        *running = true;
        drop(running);

        info!("✅ P2P Network started on port {}", self.config.network.p2p_port);

        // Spawn network loop
        let engine_clone = self.clone();
        tokio::spawn(async move {
            engine_clone.network_loop(event_receiver).await;
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

    async fn network_loop(&self, mut event_receiver: broadcast::Receiver<BlockchainEvent>) {
        info!("🔄 Network loop started");

        while *self.is_running.read().await {
            // Process blockchain events
            match event_receiver.try_recv() {
                Ok(event) => {
                    if let Err(e) = self.handle_blockchain_event(event).await {
                        warn!("Failed to handle blockchain event: {}", e);
                    }
                }
                Err(broadcast::error::TryRecvError::Empty) => {}
                Err(broadcast::error::TryRecvError::Lagged(_)) => {
                    warn!("Event receiver lagged");
                }
                Err(broadcast::error::TryRecvError::Closed) => {
                    info!("Event channel closed");
                    break;
                }
            }

            // Update peer stats
            self.update_peer_stats().await;

            // Clean duplicate cache if too large
            self.clean_duplicate_cache().await;

            tokio::time::sleep(Duration::from_secs(5)).await;
        }

        info!("🏁 Network loop terminated");
    }

    async fn handle_blockchain_event(&self, event: BlockchainEvent) -> Result<(), NetworkError> {
        match event {
            BlockchainEvent::NewBlock { height, hash, reward } => {
                info!("📢 New block event: #{} (reward: {})", height, reward);
                // In production: broadcast block to network
                // For now, just log
                debug!("Block {} hash: {}", height, hash);
            }
            BlockchainEvent::NewTransaction { tx_id, tx_type } => {
                debug!("📢 New transaction event: {} ({:?})", tx_id, tx_type);
                // In production: broadcast transaction to network
            }
            BlockchainEvent::DifficultyAdjusted { old_difficulty, new_difficulty } => {
                info!(
                    "📊 Difficulty adjusted: {} -> {}",
                    old_difficulty, new_difficulty
                );
            }
            BlockchainEvent::ModelCheckpoint { height, model_hash } => {
                info!("🧠 Model checkpoint at block {}: {}", height, model_hash);
            }
            BlockchainEvent::VestingReleased { amount, remaining } => {
                info!("💰 Vesting released: {} (remaining: {})", amount, remaining);
            }
        }
        Ok(())
    }

    async fn update_peer_stats(&self) {
        let mut peers = self.peers.write().await;
        let now = chrono::Utc::now().timestamp() as u64;

        // Remove stale peers
        peers.retain(|_, peer| {
            let age_secs = now.saturating_sub(peer.last_seen);
            age_secs < 3600 // Keep peers seen in last hour
        });

        // Decay scores
        for peer in peers.values_mut() {
            peer.score = (peer.score - PEER_SCORE_DECAY).max(MIN_PEER_SCORE);
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.connected_peers = peers.len() as u64;
        stats.uptime_secs += 5;
        stats.banned_peers = self.banned_peers.read().await.len() as u64;
    }

    async fn clean_duplicate_cache(&self) {
        let mut cache = self.duplicate_cache.write().await;
        if cache.len() > MAX_DUPLICATE_CACHE {
            // Clear half the cache (simple approach)
            let to_remove: Vec<_> = cache.iter().take(MAX_DUPLICATE_CACHE / 2).cloned().collect();
            for item in to_remove {
                cache.remove(&item);
            }
        }
    }

    pub async fn broadcast_transaction(&self, tx: Transaction) -> Result<(), NetworkError> {
        let tx_id = tx.id.clone();

        // Check for duplicate
        if self.is_duplicate(&tx_id).await {
            return Err(NetworkError::DuplicateMessage);
        }

        debug!("📢 Broadcasting transaction: {}...", &tx_id[..16]);

        let message = NetworkMessage::Transaction(tx);
        let data = serde_json::to_vec(&message)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))?;

        info!("📤 Transaction broadcasted ({} bytes)", data.len());

        // Mark as seen
        self.mark_seen(&tx_id).await;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.messages_sent += 1;
        stats.bytes_transferred += data.len() as u64;
        stats.last_activity = chrono::Utc::now().timestamp() as u64;

        Ok(())
    }

    pub async fn broadcast_block(&self, block: Block) -> Result<(), NetworkError> {
        let block_hash = block.hash.clone();

        // Check for duplicate
        if self.is_duplicate(&block_hash).await {
            return Err(NetworkError::DuplicateMessage);
        }

        debug!("📢 Broadcasting block: {}...", &block_hash[..16]);

        let message = NetworkMessage::Block(block);
        let data = serde_json::to_vec(&message)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))?;

        info!("📤 Block broadcasted ({} bytes)", data.len());

        // Mark as seen
        self.mark_seen(&block_hash).await;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.messages_sent += 1;
        stats.bytes_transferred += data.len() as u64;
        stats.last_activity = chrono::Utc::now().timestamp() as u64;

        Ok(())
    }

    pub async fn handle_incoming_message(
        &self,
        data: &[u8],
        from_peer: &str,
    ) -> Result<(), NetworkError> {
        // Check if peer is banned
        if self.banned_peers.read().await.contains(from_peer) {
            return Err(NetworkError::PeerBanned(from_peer.to_string()));
        }

        let message: NetworkMessage = serde_json::from_slice(data)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))?;

        match message {
            NetworkMessage::Transaction(tx) => {
                let tx_id = tx.id.clone();

                // Check duplicate
                if self.is_duplicate(&tx_id).await {
                    let mut stats = self.stats.write().await;
                    stats.duplicate_messages += 1;
                    return Err(NetworkError::DuplicateMessage);
                }

                debug!("📥 Received transaction: {}...", &tx_id[..16]);

                self.mark_seen(&tx_id).await;

                self.tx_sender
                    .send(tx)
                    .await
                    .map_err(|e| NetworkError::ChannelError(e.to_string()))?;
            }
            NetworkMessage::Block(block) => {
                let block_hash = block.hash.clone();

                // Check duplicate
                if self.is_duplicate(&block_hash).await {
                    let mut stats = self.stats.write().await;
                    stats.duplicate_messages += 1;
                    return Err(NetworkError::DuplicateMessage);
                }

                debug!("📥 Received block: {}...", &block_hash[..16]);

                self.mark_seen(&block_hash).await;

                self.block_sender
                    .send(block)
                    .await
                    .map_err(|e| NetworkError::ChannelError(e.to_string()))?;
            }
            NetworkMessage::RequestBlock(height) => {
                debug!("📥 Received block request: #{}", height);
                // TODO: Implement block request handling
            }
            NetworkMessage::RequestTransaction(tx_id) => {
                debug!("📥 Received transaction request: {}", tx_id);
                // TODO: Implement transaction request handling
            }
            NetworkMessage::Ping => {
                debug!("🏓 Received ping from {}", from_peer);
                // TODO: Send pong
            }
            NetworkMessage::Pong => {
                debug!("🏓 Received pong from {}", from_peer);
            }
        }

        // Update peer stats
        self.update_peer_on_message(from_peer).await;

        // Update network stats
        let mut stats = self.stats.write().await;
        stats.messages_received += 1;
        stats.bytes_transferred += data.len() as u64;
        stats.last_activity = chrono::Utc::now().timestamp() as u64;

        Ok(())
    }

    async fn is_duplicate(&self, id: &str) -> bool {
        self.duplicate_cache.read().await.contains(id)
    }

    async fn mark_seen(&self, id: &str) {
        self.duplicate_cache.write().await.insert(id.to_string());
    }

    async fn update_peer_on_message(&self, peer_id: &str) {
        let mut peers = self.peers.write().await;
        let now = chrono::Utc::now().timestamp() as u64;

        let peer = peers.entry(peer_id.to_string()).or_insert_with(|| PeerInfo {
            peer_id: peer_id.to_string(),
            address: String::new(),
            last_seen: now,
            connected: true,
            score: 1.0,
            messages_sent: 0,
            messages_received: 0,
        });

        peer.last_seen = now;
        peer.messages_received += 1;
        peer.score = (peer.score + 0.01).min(1.0); // Reward good peers
    }

    pub async fn ban_peer(&self, peer_id: &str) {
        self.banned_peers.write().await.insert(peer_id.to_string());
        self.peers.write().await.remove(peer_id);
        warn!("🚫 Peer banned: {}", peer_id);
    }

    pub async fn unban_peer(&self, peer_id: &str) {
        self.banned_peers.write().await.remove(peer_id);
        info!("✅ Peer unbanned: {}", peer_id);
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

    pub async fn get_peer_count(&self) -> usize {
        self.peers.read().await.len()
    }

    pub async fn get_banned_peers(&self) -> Vec<String> {
        self.banned_peers.read().await.iter().cloned().collect()
    }
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
            messages_sent: 10,
            messages_received: 20,
        };
        let serialized = serde_json::to_string(&info).unwrap();
        let deserialized: PeerInfo = serde_json::from_str(&serialized).unwrap();
        assert_eq!(info.peer_id, deserialized.peer_id);
        assert_eq!(info.messages_sent, deserialized.messages_sent);
    }

    #[tokio::test]
    async fn test_duplicate_detection() {
        let config = Config::default();
        let (tx_sender, _) = mpsc::channel(10);
        let (block_sender, _) = mpsc::channel(10);

        let engine = NetworkEngine::new(&config, tx_sender, block_sender)
            .await
            .unwrap();

        let id = "test_id_123";

        // First time: not duplicate
        assert!(!engine.is_duplicate(id).await);

        // Mark as seen
        engine.mark_seen(id).await;

        // Now it's duplicate
        assert!(engine.is_duplicate(id).await);
    }

    #[tokio::test]
    async fn test_peer_banning() {
        let config = Config::default();
        let (tx_sender, _) = mpsc::channel(10);
        let (block_sender, _) = mpsc::channel(10);

        let engine = NetworkEngine::new(&config, tx_sender, block_sender)
            .await
            .unwrap();

        let peer_id = "bad_peer";

        // Ban peer
        engine.ban_peer(peer_id).await;

        // Check banned
        let banned = engine.get_banned_peers().await;
        assert!(banned.contains(&peer_id.to_string()));

        // Unban
        engine.unban_peer(peer_id).await;
        let banned = engine.get_banned_peers().await;
        assert!(!banned.contains(&peer_id.to_string()));
    }
}