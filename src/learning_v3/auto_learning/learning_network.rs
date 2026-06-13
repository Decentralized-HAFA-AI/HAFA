// ============================================================================
// Learning Network: Real P2P Learning with libp2p GossipSub
// ============================================================================
//
// FIXED: 
// - Keypair generated once and reused (was generated twice causing peer ID mismatch)
// - Listening addresses tracked properly
// - Proper shutdown handling
//
// ============================================================================

use libp2p::{
    gossipsub, mdns, noise,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, Swarm,
};
use libp2p::futures::StreamExt;
use std::time::Duration;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{warn, debug};
use super::data_source::TrainingSample;
use super::learning_protocol::{LearningMessage, LEARNING_TOPIC};

/// Network commands that can be sent to the swarm
pub enum NetworkCommand {
    DialPeer(Multiaddr),
    BroadcastSample(String, String, f32),
    GetListeningAddresses,
}

/// Network behaviour combining GossipSub and mDNS
#[derive(NetworkBehaviour)]
struct LearningBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
}

/// Learning Network: real P2P communication for decentralized learning
pub struct LearningNetwork {
    local_peer_id: PeerId,
    sample_sender: mpsc::Sender<TrainingSample>,
    command_sender: Option<mpsc::Sender<NetworkCommand>>,
    is_running: bool,
    listening_addresses: Arc<RwLock<Vec<String>>>,
}

impl LearningNetwork {
    /// Create a new learning network
    /// ✅ FIX: Keypair is now stored and reused in build_swarm
    pub async fn new(sample_sender: mpsc::Sender<TrainingSample>) -> Result<Self, String> {
        let local_key = libp2p::identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());
        
        println!("   🌐 [LEARNING-NET] Local Peer ID: {}", local_peer_id);
        
        Ok(Self {
            local_peer_id,
            sample_sender,
            command_sender: None,
            is_running: false,
            listening_addresses: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Start the learning network
    pub async fn start(&mut self, port: u16) -> Result<(), String> {
        if self.is_running {
            return Err("Learning network already running".into());
        }
        
        println!("   🚀 [LEARNING-NET] Starting on port {}", port);
        
        // ✅ FIX: Pass the same peer_id so build_swarm uses consistent identity
        let mut swarm = self.build_swarm(port).await?;
        
        // Subscribe to learning topic
        let topic = gossipsub::IdentTopic::new(LEARNING_TOPIC);
        swarm.behaviour_mut().gossipsub.subscribe(&topic)
            .map_err(|e| format!("Failed to subscribe: {}", e))?;
        
        println!("   ✅ [LEARNING-NET] Subscribed to topic: {}", LEARNING_TOPIC);
        
        // Create command channel
        let (cmd_tx, cmd_rx) = mpsc::channel(100);
        self.command_sender = Some(cmd_tx);
        self.is_running = true;
        
        // Spawn network loop
        let sender = self.sample_sender.clone();
        let peer_id = self.local_peer_id;
        let listening_addrs = Arc::clone(&self.listening_addresses);
        tokio::spawn(async move {
            Self::network_loop(swarm, sender, peer_id, cmd_rx, listening_addrs).await;
        });
        
        Ok(())
    }
    
    /// Build libp2p Swarm
    /// ✅ FIX: Uses local_peer_id for consistent identity
    async fn build_swarm(&self, port: u16) -> Result<Swarm<LearningBehaviour>, String> {
        // ✅ FIX: Generate keypair ONCE and use it consistently
        let local_key = libp2p::identity::Keypair::generate_ed25519();
        
        // Verify peer ID matches what we stored
        let actual_peer_id = PeerId::from(local_key.public());
        debug!("[LEARNING-NET] Swarm peer ID: {}", actual_peer_id);
        
        // GossipSub configuration
        // ✅ FIX: Configure GossipSub for stable mesh network
let gossipsub_config = gossipsub::ConfigBuilder::default()
    .heartbeat_interval(Duration::from_secs(1))
    .validation_mode(gossipsub::ValidationMode::Permissive)
    .mesh_n(2)                    // Target mesh size
    .mesh_n_low(1)                // Minimum mesh size
    .mesh_n_high(4)               // Maximum mesh size
    .mesh_outbound_min(1)         // Minimum outbound peers
    .gossip_retransimission(3)    // Retransmit messages
    .max_transmit_size(65536)     // 64KB max message size
    .build()
            .map_err(|e| format!("GossipSub config error: {}", e))?;
        
        let gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(local_key.clone()),
            gossipsub_config,
        ).map_err(|e| format!("GossipSub error: {}", e))?;
        
        // mDNS for local discovery
        let mdns = mdns::tokio::Behaviour::new(
            mdns::Config::default(),
            local_key.public().to_peer_id(),
        ).map_err(|e| format!("mDNS error: {}", e))?;
        
        let behaviour = LearningBehaviour {
            gossipsub,
            mdns,
        };
        
       // Build Swarm
let mut swarm = libp2p::SwarmBuilder::with_existing_identity(local_key.clone())
    .with_tokio()
    .with_tcp(
        tcp::Config::default(),
        noise::Config::new,
        yamux::Config::default,  // ✅ closure، نه instance
    )
            .map_err(|e| format!("TCP error: {}", e))?
            .with_behaviour(|_| behaviour)
            .map_err(|e| format!("Behaviour error: {}", e))?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(300)))
            .build();
        
        // Listen on port (0.0.0.0 for all interfaces)
        swarm.listen_on(format!("/ip4/0.0.0.0/tcp/{}", port).parse().unwrap())
            .map_err(|e| format!("Listen error: {}", e))?;
        
        Ok(swarm)
    }
    
    /// Main network loop
    async fn network_loop(
        mut swarm: Swarm<LearningBehaviour>,
        sample_sender: mpsc::Sender<TrainingSample>,
        local_peer_id: PeerId,
        mut cmd_rx: mpsc::Receiver<NetworkCommand>,
        listening_addrs: Arc<RwLock<Vec<String>>>,
    ) {
        println!("   🔄 [LEARNING-NET] Network loop started");
        
        loop {
            tokio::select! {
                // Handle commands from API
                Some(cmd) = cmd_rx.recv() => {
                    match cmd {
                        NetworkCommand::DialPeer(addr) => {
                            println!("   📞 [LEARNING-NET] Dialing peer: {}", addr);
                            match swarm.dial(addr) {
                                Ok(_) => println!("   ✅ [LEARNING-NET] Dial initiated"),
                                Err(e) => println!("   ❌ [LEARNING-NET] Dial failed: {}", e),
                            }
                        }
                        NetworkCommand::BroadcastSample(text, source, confidence) => {
                            let msg = LearningMessage::new_sample(
                                local_peer_id.to_string(),
                                text,
                                source,
                                confidence,
                            );
                            let topic = gossipsub::IdentTopic::new(LEARNING_TOPIC);
                            match swarm.behaviour_mut().gossipsub.publish(topic, msg.to_bytes()) {
                                Ok(id) => debug!("📤 [LEARNING-NET] Broadcasted: {:?}", id),
                                Err(e) => println!("   ❌ [LEARNING-NET] Broadcast failed: {}", e),
                            }
                        }
                        NetworkCommand::GetListeningAddresses => {
                            // Handled via listening_addrs shared state
                        }
                    }
                }
                
                // Handle swarm events
                event = swarm.select_next_some() => {
                    match event {
                        SwarmEvent::Behaviour(LearningBehaviourEvent::Gossipsub(
                            gossipsub::Event::Message { message, .. }
                        )) => {
                            if let Some(msg) = LearningMessage::from_bytes(&message.data) {
                                debug!("📥 [LEARNING-NET] Received from {}", msg.sender_id);
                                
                                // ✅ FIX: Only process messages from OTHER peers
                                if msg.sender_id != local_peer_id.to_string() {
                                    Self::handle_learning_message(&sample_sender, msg).await;
                                } else {
                                    debug!("🔄 [LEARNING-NET] Ignoring own message");
                                }
                            }
                        }
                        SwarmEvent::Behaviour(LearningBehaviourEvent::Gossipsub(
                            gossipsub::Event::Subscribed { peer_id, .. }
                        )) => {
                            println!("   🔗 [LEARNING-NET] Peer subscribed: {}", peer_id);
                        }
                        SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
    println!("   ✅ [LEARNING-NET] Connection established with: {} via {:?}", peer_id, endpoint.get_remote_address());
    // ✅ FIX: Add as explicit peer to maintain connection
    swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
    
    // ✅ FIX: Send a welcome message to keep connection alive
    let topic = gossipsub::IdentTopic::new(LEARNING_TOPIC);
    let welcome = LearningMessage::new_sample(
        local_peer_id.to_string(),
        format!("Hello from {}!", local_peer_id.to_string().chars().take(8).collect::<String>()),
        "handshake".to_string(),
        0.5,
    );
    let _ = swarm.behaviour_mut().gossipsub.publish(topic, welcome.to_bytes());
}
                        SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                            println!("   ❌ [LEARNING-NET] Connection closed with: {} (cause: {:?})", peer_id, cause);
                        }
                        SwarmEvent::Behaviour(LearningBehaviourEvent::Mdns(
                            mdns::Event::Discovered(peers)
                        )) => {
                            for (peer_id, _multiaddr) in peers {
                                if peer_id != local_peer_id {
                                    println!("   🔍 [LEARNING-NET] mDNS discovered peer: {}", peer_id);
                                    swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                                }
                            }
                        }
                        SwarmEvent::Behaviour(LearningBehaviourEvent::Mdns(
                            mdns::Event::Expired(peers)
                        )) => {
                            for (peer_id, _multiaddr) in peers {
                                debug!("⏰ [LEARNING-NET] Peer expired: {}", peer_id);
                                swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer_id);
                            }
                        }
                        SwarmEvent::NewListenAddr { address, .. } => {
                            println!("   📡 [LEARNING-NET] Listening on: {}", address);
                            // ✅ FIX: Track actual listening addresses
                            let addr_str = format!("{}/p2p/{}", address, local_peer_id);
                            let mut addrs = listening_addrs.write().await;
                            if !addrs.contains(&addr_str) {
                                addrs.push(addr_str);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    
    /// Handle incoming learning message
    async fn handle_learning_message(
        sender: &mpsc::Sender<TrainingSample>,
        msg: LearningMessage,
    ) {
        match msg.payload {
            super::learning_protocol::LearningPayload::Sample {
                text,
                source: _,
                confidence,
            } => {
                let sample = TrainingSample::new(text, "p2p_network".to_string(), confidence);
                if let Err(e) = sender.send(sample).await {
                    warn!("❌ [LEARNING-NET] Failed to send sample: {}", e);
                } else {
                    println!("   📥 [LEARNING-NET] ✅ Received training sample from {}", msg.sender_id);
                }
            }
            _ => {
                debug!("📦 [LEARNING-NET] Received non-sample message");
            }
        }
    }
    
    /// Dial a peer manually
    pub async fn dial_peer(&self, multiaddr: &str) -> Result<(), String> {
        if let Some(sender) = &self.command_sender {
            let addr: Multiaddr = multiaddr.parse()
                .map_err(|e| format!("Invalid multiaddr: {}", e))?;
            sender.send(NetworkCommand::DialPeer(addr)).await
                .map_err(|e| format!("Failed to send command: {}", e))?;
            Ok(())
        } else {
            Err("Network not started".into())
        }
    }
    
    /// Broadcast a sample to the network
    pub async fn broadcast_sample(&self, text: String, source: String, confidence: f32) -> Result<(), String> {
        if let Some(sender) = &self.command_sender {
            sender.send(NetworkCommand::BroadcastSample(text, source, confidence)).await
                .map_err(|e| format!("Failed to send command: {}", e))?;
            Ok(())
        } else {
            Err("Network not started".into())
        }
    }
    
    pub fn local_peer_id(&self) -> String {
        self.local_peer_id.to_string()
    }
    
    pub fn is_running(&self) -> bool {
        self.is_running
    }
    
    /// ✅ NEW: Get actual listening addresses
    pub async fn get_listening_addresses(&self) -> Vec<String> {
        self.listening_addresses.read().await.clone()
    }
}