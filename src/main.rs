// ============================================================================
// HAFA - src/main.rs â€” GENESIS NODE WITH FULL AI API + AUTO-LEARNING
// ============================================================================
//
// Genesis node providing a comprehensive HTTP API for:
// - Blockchain operations (balance, height, info)
// - Mining pool (task, submit with CognitiveProof)
// - AI Learning (feed, train, query, generate)
// - Directory ingestion (real data processing)
// - Transformer v3 (legacy) and v4 (production-grade) training
// - Verifiable Cognitive Proof generation (PoUCW)
// - Auto-Learning Engine (self-evolving AI)
// - Blockchain Data Source (Meta-Learning from consensus)
// - Background Auto-Learning Task (fully autonomous)
// - Episodic Memory (learning from experience)
// - Backend Abstraction (CPU backend, GPU-ready)
// - Knowledge Graph (structured long-term memory)
// - Knowledge Graph + Auto-Learning Integration
// - Reasoning Engine (query & inference over KG)
// - P2P Learning Network (libp2p GossipSub)
// - Federated Learning Pool (HTTP-based sample sharing)
// - GPU Backend (WGPU acceleration)
// - Inline Web UI Dashboard
// - NEW: Wallet Management System
//
// ============================================================================
use hafa::learning_v3::KnowledgeGraphStorage;
use hafa::wallet::{WalletManager, TransactionRequest};
use std::time::Duration;
use std::collections::HashMap;
use std::collections::VecDeque;
use hafa::learning_v3::auto_learning::{GossipSubDataSource, LearningNetwork};
use hafa::learning_v3::{TransformerConfig, Trainer, TrainerV4, CognitiveProofV4};
use hafa::learning_v3::auto_learning::{
    AutoLearningEngine, AutoLearningConfig, TrainingSample,
};
use hafa::learning_v3::auto_learning::blockchain_source::BlockchainDataSource;
use hafa::learning_v3::{KnowledgeGraph, EntityType, RelationType, ReasoningEngine};
use hafa::blockchain::{Blockchain, CognitiveProof, ModelCheckpoint, ResourceUsage, TransactionType};
use hafa::config::Config;
use hafa::data_source::{DataSource, SourceReputationManager, ValidatedData};
use hafa::epistemic::EpistemicState;
use hafa::evolution::EvolutionEngine;
use hafa::learning::Learner;
use hafa::network::NetworkEngine;
use axum::{
    extract::{Path, State},
    response::Html,
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use hafa::learning_v3::{AcceleratedOps, BenchmarkResult, WgpuBackend};
use hafa::learning_v3::backend::{Backend, CpuBackend};

// ============================================================================
// APPLICATION STATE
// ============================================================================

type SharedBlockchain = Arc<RwLock<Blockchain>>;
type SharedLearner = Arc<RwLock<Learner>>;
type SharedEvolution = Arc<RwLock<EvolutionEngine>>;
type SharedReputation = Arc<SourceReputationManager>;
type SharedTrainer = Arc<RwLock<Trainer>>;
type SharedTrainerV4 = Arc<Mutex<TrainerV4>>;
type SharedAutoLearning = Arc<RwLock<AutoLearningEngine>>;
type SharedKnowledgeGraph = Arc<RwLock<KnowledgeGraph>>;
type SharedReasoning = Arc<RwLock<ReasoningEngine>>;
type SharedLearningNetwork = Arc<LearningNetwork>;
type SharedLearningPool = Arc<RwLock<VecDeque<LearningPoolItem>>>;
type SharedWalletManager = Arc<Mutex<WalletManager>>;

#[derive(Clone)]
struct AppState {
    config: Config,
    blockchain: SharedBlockchain,
    learner: SharedLearner,
    #[allow(dead_code)]
    evolution: SharedEvolution,
    reputation: SharedReputation,
    trainer: SharedTrainer,
    trainer_v4: SharedTrainerV4,
    auto_learning: SharedAutoLearning,
    knowledge_graph: SharedKnowledgeGraph,
    reasoning: SharedReasoning,
    learning_network: Option<SharedLearningNetwork>,
    learning_pool: SharedLearningPool,
    wallet_manager: SharedWalletManager,
    started_at: i64,
}

// ============================================================================
// FEDERATED LEARNING POOL STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LearningPoolItem {
    id: String,
    text: String,
    source: String,
    confidence: f32,
    timestamp: u64,
    peer_id: String,
}

// ============================================================================
// API RESPONSE STRUCTURES
// ============================================================================

#[derive(Serialize)]
struct BalanceResponse {
    address: String,
    balance: u64,
    balance_hafa: f64,
}

#[derive(Serialize)]
struct HeightResponse {
    height: u64,
}

#[derive(Serialize)]
struct InfoResponse {
    version: String,
    height: u64,
    total_minted: u64,
    total_minted_hafa: f64,
    network: String,
    current_reward: u64,
    current_reward_hafa: f64,
    uptime_secs: i64,
}

#[derive(Serialize)]
struct TaskResponse {
    last_hash: String,
    difficulty: u32,
    target_height: u64,
}

#[derive(Serialize)]
struct LearningStatusResponse {
    input_size: usize,
    output_size: usize,
    num_layers: usize,
    buffer_size: usize,
    total_parameters: usize,
    context_size: usize,
    predict_size: usize,
}

// ============================================================================
// MINING REQUEST STRUCTURES
// ============================================================================

#[derive(Deserialize)]
struct SubmitRequest {
    miner_addr: String,
    nonce: u64,
    cognitive_proof: CognitiveProofRequest,
    model_checkpoint: Option<ModelCheckpointRequest>,
}

#[derive(Deserialize)]
struct CognitiveProofRequest {
    model_hash_before: String,
    model_hash_after: String,
    dataset_commitment: String,
    gradient_commitment: String,
    loss_before: f64,
    loss_after: f64,
    ema_loss_after: f64,
    samples_processed: u64,
    resources_used: ResourceUsageRequest,
    training_duration_ms: u64,
}

#[derive(Deserialize)]
struct ResourceUsageRequest {
    cpu_percent: f64,
    ram_mb: u64,
    gpu_percent: f64,
    gpu_memory_mb: u64,
}

#[derive(Deserialize)]
struct ModelCheckpointRequest {
    model_hash: String,
    total_parameters: u64,
    architecture: String,
}

#[derive(Serialize)]
struct SubmitResponse {
    success: bool,
    block_index: Option<u64>,
    reward: u64,
    reward_hafa: f64,
    quality_score: f64,
    message: String,
}

// ============================================================================
// LEGACY MLP API STRUCTURES
// ============================================================================

#[derive(Deserialize)]
struct FeedRequest {
    source_type: String,
    source_id: String,
    content: Vec<u8>,
}

#[derive(Serialize)]
struct FeedResponse {
    success: bool,
    buffer_size: usize,
    message: String,
}

#[derive(Deserialize)]
struct TrainRequest {
    epochs: u32,
}

#[derive(Serialize)]
struct TrainResponse {
    success: bool,
    epochs_completed: u32,
    avg_loss: f64,
    message: String,
}

#[derive(Deserialize)]
struct QueryRequest {
    input: Vec<u8>,
    steps: Option<usize>,
}

#[derive(Serialize)]
struct QueryResponse {
    success: bool,
    generated_bytes: Vec<u8>,
    generated_text: String,
    steps: usize,
    message: String,
}

#[derive(Deserialize)]
struct IngestDirectoryRequest {
    path: String,
    recursive: bool,
}

#[derive(Serialize)]
struct IngestDirectoryResponse {
    success: bool,
    files_processed: usize,
    total_bytes: usize,
    buffer_size: usize,
    message: String,
}

// ============================================================================
// TRANSFORMER V3 API STRUCTURES (Legacy)
// ============================================================================

#[derive(Deserialize)]
struct GenerateV3Request {
    prompt: String,
    steps: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
}

#[derive(Serialize)]
struct GenerateV3Response {
    success: bool,
    generated_text: String,
    steps: usize,
    message: String,
}

#[derive(Deserialize)]
struct TrainV3Request {
    input: String,
    target: String,
    epochs: u32,
}

#[derive(Serialize)]
struct TrainV3Response {
    success: bool,
    final_loss: f32,
    message: String,
}

#[derive(Deserialize)]
struct TrainTextV3Request {
    text: String,
    context_size: Option<usize>,
    epochs: u32,
}

#[derive(Serialize)]
struct TrainTextV3Response {
    success: bool,
    final_loss: f32,
    samples_trained: usize,
    message: String,
    cognitive_proof: Option<hafa::learning_v3::TrainingProof>,
}

#[derive(Deserialize)]
struct SaveModelRequest {
    path: Option<String>,
}

#[derive(Serialize)]
struct SaveModelResponse {
    success: bool,
    message: String,
}

#[derive(Deserialize)]
struct LoadModelRequest {
    path: Option<String>,
}

#[derive(Serialize)]
struct LoadModelResponse {
    success: bool,
    message: String,
}

// ============================================================================
// TRANSFORMER V4 API STRUCTURES (Production-Grade)
// ============================================================================

#[derive(Deserialize)]
struct TrainTextV4Request {
    text: String,
    context_size: Option<usize>,
    epochs: u32,
}

#[derive(Serialize)]
struct TrainTextV4Response {
    success: bool,
    final_loss: f32,
    ema_loss: f32,
    samples_processed: u64,
    wall_time_ms: u64,
    message: String,
    cognitive_proof: Option<CognitiveProofV4>,
}

// ============================================================================
// AUTO-LEARNING API STRUCTURES
// ============================================================================

#[derive(Deserialize)]
struct AutoLearnFeedRequest {
    text: String,
    #[serde(default = "default_source")]
    source: String,
    #[serde(default = "default_confidence")]
    confidence: f32,
}

fn default_source() -> String {
    "api".to_string()
}

fn default_confidence() -> f32 {
    0.8
}

#[derive(Serialize)]
struct AutoLearnFeedResponse {
    success: bool,
    message: String,
    buffer_size: usize,
}

#[derive(Serialize)]
struct AutoLearnTriggerResponse {
    success: bool,
    message: String,
    proof: Option<AutoLearnProofSummary>,
}

#[derive(Serialize)]
struct AutoLearnProofSummary {
    loss_before: f32,
    loss_after: f32,
    quality_score: f64,
    samples_processed: u64,
    gradient_commitment: String,
}

#[derive(Serialize)]
struct AutoLearnStatusResponse {
    is_learning: bool,
    buffer_size: usize,
    max_buffer_size: usize,
}

#[derive(Serialize)]
struct AutoLearnStatsResponse {
    total_cycles: u64,
    total_samples_received: u64,
    total_samples_rejected: u64,
    total_samples_learned: u64,
    total_proofs_generated: u64,
    last_cycle_time_secs: Option<u64>,
    last_cycle_loss: Option<f32>,
    buffer_size: usize,
    curiosity_accepted: u64,
    curiosity_rejected: u64,
    meta_learning_checks: u64,
    meta_learning_skips: u64,
    meta_learning_boosts: u64,
    kg_entities_retrieved: u64,
    kg_entities_added: u64,
    kg_relations_added: u64,
}

// ============================================================================
// BLOCKCHAIN POLL RESPONSE
// ============================================================================

#[derive(Serialize)]
struct BlockchainPollResponse {
    success: bool,
    new_samples: usize,
    last_processed_height: u64,
    current_height: u64,
    message: String,
}

// ============================================================================
// EPISODIC MEMORY API STRUCTURES
// ============================================================================

#[derive(Serialize)]
struct EpisodeResponse {
    id: String,
    timestamp: u64,
    sample_count: usize,
    loss_before: f32,
    loss_after: f32,
    loss_improvement: f32,
    quality_score: f64,
    duration_ms: u64,
    success: bool,
    tags: Vec<String>,
}

#[derive(Serialize)]
struct EpisodicStatsResponse {
    total_episodes: usize,
    successful_episodes: usize,
    failed_episodes: usize,
    avg_loss_improvement: f32,
    avg_quality_score: f64,
    success_rate: f64,
}

// ============================================================================
// NETWORK SIMULATION API STRUCTURES
// ============================================================================

#[derive(Deserialize)]
struct SimulateNetworkRequest {
    text: String,
    confidence: f32,
}

#[derive(Serialize)]
struct SimulateNetworkResponse {
    success: bool,
    message: String,
    buffer_size: usize,
}

// ============================================================================
// KNOWLEDGE GRAPH API STRUCTURES
// ============================================================================

#[derive(Serialize)]
struct EntityResponse {
    id: String,
    name: String,
    entity_type: String,
    confidence: f32,
    mentions: u64,
    created_at: u64,
    properties: HashMap<String, String>,
}

#[derive(Serialize)]
struct RelationResponse {
    id: String,
    source_id: String,
    target_id: String,
    relation_type: String,
    confidence: f32,
    weight: f32,
    created_at: u64,
}

#[derive(Serialize)]
struct KnowledgeGraphStatsResponse {
    total_entities: usize,
    total_relations: usize,
    entities_by_type: HashMap<String, usize>,
    relations_by_type: HashMap<String, usize>,
    avg_entity_confidence: f32,
    avg_relation_confidence: f32,
}

#[derive(Deserialize)]
struct AddEntityRequest {
    name: String,
    entity_type: String,
    confidence: Option<f32>,
}

#[derive(Serialize)]
struct AddEntityResponse {
    success: bool,
    entity_id: String,
    message: String,
}

#[derive(Deserialize)]
struct AddRelationRequest {
    source: String,
    target: String,
    relation_type: String,
    confidence: Option<f32>,
}

#[derive(Serialize)]
struct AddRelationResponse {
    success: bool,
    relation_id: Option<String>,
    message: String,
}

#[derive(Deserialize)]
struct ExtractKnowledgeRequest {
    text: String,
}

#[derive(Serialize)]
struct ExtractKnowledgeResponse {
    success: bool,
    entities_extracted: usize,
    relations_extracted: usize,
    message: String,
}

// ============================================================================
// REASONING ENGINE API STRUCTURES
// ============================================================================

#[derive(Deserialize)]
struct KnowledgeQueryRequest {
    query: String,
}

#[derive(Serialize)]
struct KnowledgeQueryResponse {
    query: String,
    answer: String,
    confidence: f32,
    entities_found: Vec<String>,
    relations_found: Vec<String>,
    inference_path: Vec<String>,
}

// ============================================================================
// P2P NETWORK API STRUCTURES
// ============================================================================

#[derive(Deserialize)]
struct P2PConnectRequest {
    multiaddr: String,
}

#[derive(Serialize)]
struct P2PConnectResponse {
    success: bool,
    message: String,
}

#[derive(Serialize)]
struct P2PInfoResponse {
    peer_id: String,
    is_running: bool,
    listening_addresses: Vec<String>,
}

// ============================================================================
// FEDERATED LEARNING API STRUCTURES
// ============================================================================

#[derive(Deserialize)]
struct FederatedShareRequest {
    text: String,
    #[serde(default = "default_source")]
    source: String,
    #[serde(default = "default_confidence")]
    confidence: f32,
    #[serde(default = "default_peer_id")]
    peer_id: String,
}

fn default_peer_id() -> String {
    "unknown".to_string()
}

#[derive(Serialize)]
struct FederatedShareResponse {
    success: bool,
    item_id: String,
    pool_size: usize,
    message: String,
}

#[derive(Serialize)]
struct FederatedPollResponse {
    success: bool,
    samples: Vec<LearningPoolItem>,
    count: usize,
    message: String,
}

#[derive(Serialize)]
struct FederatedStatsResponse {
    pool_size: usize,
    total_shared: u64,
    total_received: u64,
    oldest_item_age_secs: Option<u64>,
    newest_item_age_secs: Option<u64>,
}

// ============================================================================
// WALLET API STRUCTURES
// ============================================================================

#[derive(Deserialize)]
struct WalletCreateRequest {
    passphrase: String,
    label: Option<String>,
}

#[derive(Serialize)]
struct WalletCreateResponse {
    success: bool,
    address: String,
    label: Option<String>,
    message: String,
}

#[derive(Deserialize)]
struct WalletImportRequest {
    passphrase: String,
    label: Option<String>,
}

#[derive(Serialize)]
struct WalletImportResponse {
    success: bool,
    address: String,
    label: Option<String>,
    message: String,
}

#[derive(Serialize)]
struct WalletListResponse {
    success: bool,
    wallets: Vec<hafa::wallet::WalletInfo>,
    count: usize,
}

#[derive(Serialize)]
struct WalletInfoResponse {
    success: bool,
    wallet: Option<hafa::wallet::WalletInfo>,
    balance: Option<u64>,
    balance_hafa: Option<f64>,
    message: String,
}

#[derive(Deserialize)]
struct WalletSignRequest {
    passphrase: String,
    to_address: String,
    amount: u64,
    fee: u64,
}

#[derive(Serialize)]
struct WalletSignResponse {
    success: bool,
    signed_transaction: Option<hafa::wallet::SignedTransaction>,
    message: String,
}

#[derive(Deserialize)]
struct WalletDeleteRequest {
    #[allow(dead_code)]
    passphrase: String,
}

#[derive(Serialize)]
struct WalletDeleteResponse {
    success: bool,
    message: String,
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

#[tokio::main]
async fn main() {
    println!("ًںڑ€ HAFA Genesis Node Starting...");
    println!("   Version: 5.1.0 - GPU Backend + Federated Learning + Web UI + Wallet\n");
    
    let config = Config::load_or_default();
    
    let bc = Blockchain::new(&config)
        .await
        .expect("Failed to initialize blockchain");
    
    let learner = Learner::new(&config);
    let evolution = EvolutionEngine::new(&config);
    let reputation = SourceReputationManager::new();

    // Initialize HAFA v3 Transformer Trainer (Legacy)
    let transformer_config = TransformerConfig::default();
    let trainer = Trainer::new(&transformer_config, 0.001, 20, 1000, 0.01, 8);
    println!("   ًں§  Transformer v3 initialized: {} parameters", 
             trainer.model.get_stats().total_parameters);

    // Initialize HAFA v4 Trainer (Production-Grade) with Backend
    let trainer_v4 = TrainerV4::new(&transformer_config, 0.00001, 20, 0, 0.0001, 4);
    println!("   ًں§  Transformer v4 initialized: {} parameters (AdamW + Real Accumulation)", 
             trainer_v4.model.get_stats().total_parameters);
    println!("   âڑ™ï¸ڈ  V4 features: AdamW, L2 Gradient Clipping, Binary Checkpoint, Verifiable PoUCW");
    println!("   ًں–¥ï¸ڈ  Backend: {} (GPU-ready architecture) âœ¨", trainer_v4.backend.name());

    // Initialize Knowledge Graph with Persistent Storage
    let kg_storage_path = config.storage.data_dir.join("knowledge_graph.json");
    let kg_storage = KnowledgeGraphStorage::new(kg_storage_path);
    
    let initial_kg = match kg_storage.load() {
        Ok(kg) => {
            println!("   ًں§  Knowledge Graph loaded from disk (structured long-term memory) âœ¨");
            kg
        }
        Err(e) => {
            println!("   âڑ ï¸ڈ  Failed to load KG from disk: {}, starting fresh", e);
            KnowledgeGraph::new()
        }
    };
    
    let knowledge_graph: SharedKnowledgeGraph = Arc::new(RwLock::new(initial_kg));

    // Initialize Reasoning Engine
    let reasoning: SharedReasoning = Arc::new(RwLock::new(ReasoningEngine::new()));
    println!("   ًں§  Reasoning Engine initialized (query & inference) âœ¨");

    // Initialize Auto-Learning Engine with Knowledge Graph Integration
    let trainer_v4_shared: SharedTrainerV4 = Arc::new(Mutex::new(trainer_v4));
    let auto_learning_config = AutoLearningConfig::default();
    let mut auto_learning_engine = AutoLearningEngine::new(
        Arc::clone(&trainer_v4_shared),
        auto_learning_config,
    );
    
    // Attach Knowledge Graph to Auto-Learning Engine
    auto_learning_engine.set_knowledge_graph(Arc::clone(&knowledge_graph));
    println!("   ًں”— Knowledge Graph integrated with Auto-Learning Engine âœ¨");
    
    let auto_learning: SharedAutoLearning = Arc::new(RwLock::new(auto_learning_engine));
    println!("   ًں¤– Auto-Learning Engine initialized (self-evolving AI) âœ¨");

    let (tx_tx, _) = mpsc::channel(100);
    let (block_tx, _) = mpsc::channel(100);
    match NetworkEngine::new(&config, tx_tx, block_tx).await {
        Ok(_engine) => println!("   ًںŒگ Network Engine initialized (mock mode) âœ¨"),
        Err(e) => println!("   âڑ ï¸ڈ  Network Engine failed: {}", e),
    }

    // Initialize Learning Network (Real P2P)
    let (learning_tx, learning_rx) = mpsc::channel(1000);
    let mut learning_network = match LearningNetwork::new(learning_tx).await {
        Ok(net) => {
            println!("   ًںŒگ Learning Network created (Peer ID: {})", net.local_peer_id());
            net
        }
        Err(e) => {
            println!("   âڑ ï¸ڈ  Failed to create learning network: {}", e);
            return;
        }
    };

    // Start learning network on port 7477
    if let Err(e) = learning_network.start(config.network.p2p_port).await {
        println!("   âڑ ï¸ڈ  Failed to start learning network: {}", e);
    } else {
        println!("   ًںŒگ Learning Network started on port 7477 âœ¨");
    }

    let learning_network_shared: Option<SharedLearningNetwork> = Some(Arc::new(learning_network));

    // Initialize Federated Learning Pool
    let learning_pool: SharedLearningPool = Arc::new(RwLock::new(VecDeque::new()));
    println!("   ًںŒگ Federated Learning Pool initialized (HTTP-based sharing) âœ¨");

    // Initialize Wallet Manager
    let wallet_path = config.storage.data_dir.join("wallets.json");
    let wallet_manager: SharedWalletManager = Arc::new(Mutex::new(WalletManager::new(wallet_path)));
    println!("   ًں’¼ Wallet Manager initialized (Ed25519 + ChaCha20 encryption) âœ¨");

    println!("   ًںژ¨ Web UI Dashboard initialized (inline HTML/CSS/JS) âœ¨");

    let state = AppState {
        config: config.clone(),
        blockchain: Arc::new(RwLock::new(bc)),
        learner: Arc::new(RwLock::new(learner)),
        evolution: Arc::new(RwLock::new(evolution)),
        reputation: Arc::new(reputation),
        trainer: Arc::new(RwLock::new(trainer)),
        trainer_v4: trainer_v4_shared,
        auto_learning,
        knowledge_graph: Arc::clone(&knowledge_graph),
        reasoning,
        learning_network: learning_network_shared,
        learning_pool,
        wallet_manager,
        started_at: Utc::now().timestamp(),
    };

    // Register Data Sources for Meta-Learning
    {
        let mut engine = state.auto_learning.write().await;
        
        // 1. Register Blockchain Data Source
        let current_height = state.blockchain.read().await.get_chain_height().await;
        let bc_source = BlockchainDataSource::new(
            Arc::clone(&state.blockchain),
            current_height,
        );
        engine.register_source(Box::new(bc_source));
        println!("   ًں”— Blockchain Data Source registered (Meta-Learning from consensus) âœ¨");
        
        // 2. Register GossipSub Data Source (Real P2P Learning)
        let gossip_source = GossipSubDataSource::new(learning_rx);
        engine.register_source(Box::new(gossip_source));
        println!("   ًںŒگ GossipSub Data Source registered (Real P2P learning via libp2p) âœ¨");
    }

    // ========================================================================
    // BACKGROUND AUTO-LEARNING TASK (Fully Autonomous AI)
    // ========================================================================
    
    let bg_auto_learning = Arc::clone(&state.auto_learning);
    let bg_pool = Arc::clone(&state.learning_pool);
    tokio::spawn(async move {
        println!("   ًں”„ Background Auto-Learning started (polling every 60s) âœ¨");
        
        // Wait 30 seconds before first poll (let the node stabilize)
        tokio::time::sleep(Duration::from_secs(30)).await;
        
        loop {
            // Poll federated learning pool
            {
                let mut pool = bg_pool.write().await;
                let mut engine = bg_auto_learning.write().await;
                
                // Take up to 10 samples from pool
                let mut count = 0;
                while count < 10 {
                    if let Some(item) = pool.pop_front() {
                        let sample = TrainingSample::new(
                            item.text,
                            format!("federated:{}", item.source),
                            item.confidence,
                        );
                        engine.push_sample(sample);
                        count += 1;
                    } else {
                        break;
                    }
                }
                
                if count > 0 {
                    println!("   [FEDERATED] ًںŒگ Polled {} sample(s) from pool", count);
                }
            }
            
            // Poll all registered data sources (blockchain + gossipsub)
            let new_samples = {
                let mut engine = bg_auto_learning.write().await;
                engine.poll_sources().await
            };
            
            if new_samples > 0 {
                let (buffer_size, should_learn) = {
                    let engine = bg_auto_learning.read().await;
                    (engine.buffer_size(), engine.should_learn())
                };
                
                println!("   [BACKGROUND] ًں§  Polled {} new sample(s) | Buffer: {}", 
                         new_samples, buffer_size);
                
                // Auto-trigger learning if conditions are met
                if should_learn {
                    println!("   [BACKGROUND] ًںڑ€ Conditions met! Auto-triggering learning cycle...");
                    
                    let mut engine = bg_auto_learning.write().await;
                    match engine.trigger_learning() {
                        Some(proof) => {
                            println!("   [BACKGROUND] âœ… Learning cycle complete!");
                            println!("   [BACKGROUND]    ًں“‰ Loss: {:.4} â†’ {:.4}", 
                                     proof.loss_before, proof.loss_after);
                            println!("   [BACKGROUND]    â­گ Quality: {:.4}", proof.quality_score());
                            println!("   [BACKGROUND]    ًں“ٹ Samples processed: {}", 
                                     proof.samples_processed);
                            println!("   [BACKGROUND]    âڈ±ï¸ڈ  Duration: {}ms", proof.wall_time_ms);
                        }
                        None => {
                            println!("   [BACKGROUND] âڑ ï¸ڈ  Learning trigger returned None (unexpected)");
                        }
                    }
                } else {
                    println!("   [BACKGROUND] âڈ³ Waiting for more samples before learning...");
                }
            }
            
            // Sleep for 60 seconds before next poll
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    });

    let app = Router::new()
        // Blockchain endpoints
        .route("/balance/:address", get(get_balance))
        .route("/height", get(get_height))
        .route("/info", get(get_info))
        // Mining endpoints
        .route("/task", get(get_task))
        .route("/submit", post(submit_solution))
        // AI Learning endpoints (Legacy MLP)
        .route("/learning-status", get(get_learning_status))
        .route("/feed", post(feed_data))
        .route("/train", post(train_model))
        .route("/query", post(query_model))
        .route("/generate", post(generate_text))
        // Transformer v3 endpoints (Legacy)
        .route("/generate-v3", post(generate_v3))
        .route("/train-v3", post(train_v3))
        .route("/train-text-v3", post(train_text_v3))
        // Transformer v4 endpoints (Production-Grade)
        .route("/train-text-v4", post(train_text_v4))
        // Model checkpointing
        .route("/save-model", post(save_model))
        .route("/load-model", post(load_model))
        // Data ingestion
        .route("/ingest-directory", post(ingest_directory))
        // Auto-Learning endpoints (Self-Evolving AI)
        .route("/auto-learn/feed", post(auto_learn_feed))
        .route("/auto-learn/trigger", post(auto_learn_trigger))
        .route("/auto-learn/status", get(auto_learn_status))
        .route("/auto-learn/stats", get(auto_learn_stats))
        // Blockchain Meta-Learning endpoints
        .route("/auto-learn/poll-blockchain", post(poll_blockchain))
        // Episodic Memory endpoints
        .route("/auto-learn/episodes", get(auto_learn_episodes))
        .route("/auto-learn/episodes/stats", get(auto_learn_episodic_stats))
        // Network Simulation endpoint (for testing P2P learning)
        .route("/debug/simulate-network", post(simulate_network_data))
        // Knowledge Graph endpoints
        .route("/knowledge/entities", get(knowledge_entities))
        .route("/knowledge/relations", get(knowledge_relations))
        .route("/knowledge/stats", get(knowledge_stats))
        .route("/knowledge/entity", post(knowledge_add_entity))
        .route("/knowledge/relation", post(knowledge_add_relation))
        .route("/knowledge/extract", post(knowledge_extract))
        // Reasoning Engine endpoint
        .route("/knowledge/query", post(knowledge_query))
        // Backend Benchmark endpoint
        .route("/debug/benchmark-backend", post(benchmark_backend))
        // P2P Network endpoints
        .route("/p2p/info", get(p2p_info))
        .route("/p2p/connect", post(p2p_connect))
        // Federated Learning endpoints
        .route("/federated/share", post(federated_share))
        .route("/federated/poll", get(federated_poll))
        .route("/federated/stats", get(federated_stats))
        // GPU Backend endpoint
        .route("/gpu/info", get(gpu_info))
        // Inline Web UI Dashboard
        .route("/web", get(web_dashboard))
        // NEW: Wallet endpoints
        .route("/wallet/create", post(wallet_create))
        .route("/wallet/import", post(wallet_import))
        .route("/wallet/list", get(wallet_list))
        .route("/wallet/:address/info", get(wallet_info))
        .route("/wallet/:address/sign", post(wallet_sign_transaction))
        .route("/wallet/:address/delete", post(wallet_delete))
        .with_state(state);

    let api_handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", config.network.http_port))
            .await
            .unwrap();
        println!("   ًںŒگ HTTP API started on http://127.0.0.1:7476");
        println!("   ًںژ¨ Web UI Dashboard: http://127.0.0.1:7476/web âœ¨");
        println!("   ًں“، Available endpoints:");
        println!("      GET  /info              - Node information");
        println!("      GET  /height            - Blockchain height");
        println!("      GET  /balance/:address  - Account balance");
        println!("      GET  /task              - Mining task");
        println!("      POST /submit            - Submit mined block");
        println!("      GET  /learning-status   - AI model status (Legacy MLP)");
        println!("      POST /feed              - Feed data to AI");
        println!("      POST /train             - Train AI model (Legacy MLP)");
        println!("      POST /query             - Query AI model");
        println!("      POST /generate          - Generate text (Legacy MLP)");
        println!("      POST /generate-v3       - Generate text (Transformer v3)");
        println!("      POST /train-v3          - Train Transformer v3");
        println!("      POST /train-text-v3     - Train on raw text (v3 - legacy)");
        println!("      POST /train-text-v4     - Train on raw text (v4 - AdamW + Verifiable Proof) âœ¨");
        println!("      POST /save-model        - Save model weights to disk");
        println!("      POST /load-model        - Load model weights from disk");
        println!("      POST /ingest-directory  - Ingest directory");
        println!("      POST /auto-learn/feed   - Feed sample to auto-learning engine ًں¤–");
        println!("      POST /auto-learn/trigger- Trigger auto-learning cycle ًں¤–");
        println!("      GET  /auto-learn/status - Auto-learning engine status ًں¤–");
        println!("      GET  /auto-learn/stats  - Auto-learning statistics ًں¤–");
        println!("      POST /auto-learn/poll-blockchain - Poll blockchain for meta-learning ًں”—");
        println!("      GET  /auto-learn/episodes      - List all learning episodes ًں“‌");
        println!("      GET  /auto-learn/episodes/stats - Episodic memory statistics ًں“ٹ");
        println!("      POST /debug/simulate-network - Simulate P2P network data ًںŒگ");
        println!("      GET  /knowledge/entities    - List all entities ًں§ ");
        println!("      GET  /knowledge/relations   - List all relations ًں”—");
        println!("      GET  /knowledge/stats       - Knowledge graph statistics ًں“ٹ");
        println!("      POST /knowledge/entity      - Add an entity â‍•");
        println!("      POST /knowledge/relation    - Add a relation ًں”—");
        println!("      POST /knowledge/extract     - Extract knowledge from text ًں“‌");
        println!("      POST /knowledge/query       - Query knowledge graph ًں§ ");
        println!("      POST /debug/benchmark-backend - Run backend benchmarks ًں”¬");
        println!("      GET  /p2p/info              - P2P network info ًںŒگ");
        println!("      POST /p2p/connect           - Connect to peer manually ًںŒگ");
        println!("      POST /federated/share       - Share sample with network ًںŒگ");
        println!("      GET  /federated/poll        - Poll samples from network ًںŒگ");
        println!("      GET  /federated/stats       - Federated learning stats ًںŒگ");
        println!("      GET  /gpu/info              - GPU backend info ًںژ®");
        println!("      GET  /web                   - Web UI Dashboard ًںژ¨");
        println!("      POST /wallet/create         - Create new wallet ًں’¼");
        println!("      POST /wallet/import         - Import wallet from passphrase ًں’¼");
        println!("      GET  /wallet/list           - List all wallets ًں’¼");
        println!("      GET  /wallet/:addr/info     - Wallet info + balance ًں’¼");
        println!("      POST /wallet/:addr/sign     - Sign transaction ًں’¼");
        println!("      POST /wallet/:addr/delete   - Delete wallet ًں’¼");
        println!("      ًں”„ Background Auto-Learning: polls every 60s âœ¨");
        println!();
        axum::serve(listener, app).await.unwrap();
    });

    println!("   âœ… Node is alive. Press Ctrl+C to stop.\n");
    println!("   ًںŒں HAFA is now FULLY AUTONOMOUS + DECENTRALIZED + KNOWLEDGEABLE + REASONING!\n");
    println!("   ًں§  It learns from blockchain, P2P network, builds structured knowledge, AND answers questions!\n");
    println!("   ًںŒگ NEW: Federated Learning via HTTP - Share and receive samples from other nodes!\n");
    println!("   ًںژ® NEW: GPU Backend - Hardware acceleration for AI computations!\n");
    println!("   ًںژ¨ NEW: Web UI Dashboard at http://127.0.0.1:7476/web âœ¨\n");
    println!("   ًں’¼ NEW: Wallet System - Create, import, and manage wallets!\n");
    
    // Save Knowledge Graph on shutdown
    let shutdown_kg = Arc::clone(&knowledge_graph);
    let shutdown_storage = kg_storage;
    
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Failed to listen for ctrl+c");
        println!("\n   ًں›‘ Shutting down...");
        
        // Save KG before exit
        let kg = shutdown_kg.read().await;
        if let Err(e) = shutdown_storage.save(&kg) {
            eprintln!("   â‌Œ Failed to save KG: {}", e);
        } else {
            println!("   âœ… Knowledge Graph saved successfully");
        }
        
        std::process::exit(0);
    });
    api_handle.await.unwrap();
}

// ============================================================================
// BLOCKCHAIN HANDLERS
// ============================================================================

async fn get_balance(State(state): State<AppState>, Path(address): Path<String>) -> Json<BalanceResponse> {
    let bc = state.blockchain.read().await;
    let balance = bc.get_balance(&address).await.unwrap_or(0);
    Json(BalanceResponse {
        address,
        balance,
        balance_hafa: balance as f64 / 100_000_000.0,
    })
}

async fn get_height(State(state): State<AppState>) -> Json<HeightResponse> {
    let bc = state.blockchain.read().await;
    Json(HeightResponse { height: bc.get_chain_height().await })
}

async fn get_info(State(state): State<AppState>) -> Json<InfoResponse> {
    let bc = state.blockchain.read().await;
    let height = bc.get_chain_height().await;
    let total_minted = bc.get_total_minted().await;
    let current_reward = bc.get_current_reward().await;
    let uptime = Utc::now().timestamp() - state.started_at;
    
    Json(InfoResponse {
        version: "5.1.0".into(),
        height,
        total_minted,
        total_minted_hafa: total_minted as f64 / 100_000_000.0,
        network: "mainnet".into(),
        current_reward,
        current_reward_hafa: current_reward as f64 / 100_000_000.0,
        uptime_secs: uptime,
    })
}

async fn get_task(State(state): State<AppState>) -> Json<TaskResponse> {
    let bc = state.blockchain.read().await;
    match bc.get_task().await {
        Ok((hash, diff, height)) => Json(TaskResponse { last_hash: hash, difficulty: diff, target_height: height }),
        Err(_) => Json(TaskResponse { last_hash: "0".repeat(64), difficulty: 1, target_height: 1 }),
    }
}

async fn submit_solution(State(state): State<AppState>, Json(payload): Json<SubmitRequest>) -> Json<SubmitResponse> {
    let bc = state.blockchain.read().await;

    let v4_proof = hafa::learning_v3::CognitiveProofV4 {
        model_hash_before: payload.cognitive_proof.model_hash_before.clone(),
        model_hash_after: payload.cognitive_proof.model_hash_after.clone(),
        dataset_commitment: payload.cognitive_proof.dataset_commitment.clone(),
        gradient_commitment: payload.cognitive_proof.gradient_commitment.clone(),
        loss_before: payload.cognitive_proof.loss_before as f32,
        loss_after: payload.cognitive_proof.loss_after as f32,
        ema_loss_after: payload.cognitive_proof.ema_loss_after as f32,
        samples_processed: payload.cognitive_proof.samples_processed,
        wall_time_ms: payload.cognitive_proof.training_duration_ms,
        cpu_usage_percent: payload.cognitive_proof.resources_used.cpu_percent,
        ram_usage_mb: payload.cognitive_proof.resources_used.ram_mb,
    };

    let quality_score = match hafa::learning_v3::ProofVerifier::verify_full(&v4_proof) {
        Ok(score) => score,
        Err(e) => {
            return Json(SubmitResponse {
                success: false,
                block_index: None,
                reward: 0,
                reward_hafa: 0.0,
                quality_score: 0.0,
                message: format!("Proof verification failed: {}", e),
            });
        }
    };

    let cognitive_proof = CognitiveProof::new(
        payload.cognitive_proof.model_hash_before,
        payload.cognitive_proof.model_hash_after,
        payload.cognitive_proof.loss_before,
        payload.cognitive_proof.loss_after,
        payload.cognitive_proof.samples_processed as u32,
        0.9,
        ResourceUsage {
            cpu_percent: payload.cognitive_proof.resources_used.cpu_percent,
            ram_mb: payload.cognitive_proof.resources_used.ram_mb,
            gpu_percent: payload.cognitive_proof.resources_used.gpu_percent,
            gpu_memory_mb: payload.cognitive_proof.resources_used.gpu_memory_mb,
        },
        payload.cognitive_proof.training_duration_ms,
    );

    let model_checkpoint = payload.model_checkpoint.map(|mc| ModelCheckpoint {
        block_height: 0,
        model_hash: mc.model_hash,
        total_parameters: mc.total_parameters,
        architecture: mc.architecture,
        timestamp: Utc::now().timestamp() as u64,
    });

    match bc.submit_solution(&payload.miner_addr, payload.nonce, cognitive_proof, model_checkpoint).await {
        Ok(block) => {
            let reward = block.transactions.iter()
                .find(|tx| tx.tx_type == TransactionType::Reward && tx.from == "SYSTEM")
                .map(|tx| tx.amount).unwrap_or(0);

            Json(SubmitResponse {
                success: true,
                block_index: Some(block.index),
                reward,
                reward_hafa: reward as f64 / 100_000_000.0,
                quality_score,
                message: format!("Block #{} mined! Quality: {:.2}", block.index, quality_score),
            })
        }
        Err(e) => Json(SubmitResponse {
            success: false, block_index: None, reward: 0, reward_hafa: 0.0, quality_score: 0.0,
            message: format!("Failed: {}", e),
        }),
    }
}

// ============================================================================
// AI LEARNING HANDLERS (Legacy MLP)
// ============================================================================

async fn get_learning_status(State(state): State<AppState>) -> Json<LearningStatusResponse> {
    let learner = state.learner.read().await;
    let stats = learner.get_stats();
    Json(LearningStatusResponse {
        input_size: stats.input_size,
        output_size: stats.output_size,
        num_layers: stats.num_layers,
        buffer_size: stats.buffer_size,
        total_parameters: stats.total_parameters,
        context_size: stats.context_size,
        predict_size: stats.predict_size,
    })
}

async fn feed_data(State(state): State<AppState>, Json(payload): Json<FeedRequest>) -> Json<FeedResponse> {
    if payload.content.is_empty() {
        return Json(FeedResponse { success: false, buffer_size: 0, message: "Empty content".to_string() });
    }

    let source = match payload.source_type.as_str() {
        "local" => DataSource::Local { path: payload.source_id },
        "sensor" => DataSource::Sensor { device_id: payload.source_id },
        _ => DataSource::Local { path: payload.source_id },
    };

    let knowledge_claim = hafa::epistemic::KnowledgeClaim::new(
        &payload.content, source.source_type().to_string(), source.source_id(),
        source.is_direct_observation(), source.infer_category(),
    );

    let validated_data = ValidatedData {
        content: payload.content.clone(), source: source.clone(),
        epistemic_state: EpistemicState::new(0.9, true, 0, 0.1, 1, 0.0, 1.0),
        timestamp: Utc::now().timestamp() as u64, knowledge_claim, metadata: None,
    };

    let mut learner = state.learner.write().await;
    learner.ingest(&validated_data);
    let buffer_size = learner.get_stats().buffer_size;

    Json(FeedResponse { success: true, buffer_size, message: format!("Data ingested. Buffer size: {}", buffer_size) })
}

async fn train_model(State(state): State<AppState>, Json(payload): Json<TrainRequest>) -> Json<TrainResponse> {
    if payload.epochs == 0 {
        return Json(TrainResponse { success: false, epochs_completed: 0, avg_loss: 0.0, message: "Epochs must be > 0".to_string() });
    }

    let mut learner = state.learner.write().await;
    if learner.buffer.is_empty() {
        return Json(TrainResponse { success: false, epochs_completed: 0, avg_loss: 0.0, message: "Buffer empty - feed data first".to_string() });
    }

    let mut total_loss = 0.0;
    let mut successful_epochs = 0;

    for _ in 0..payload.epochs {
        match learner.train_step() {
            Ok(loss) => { total_loss += loss; successful_epochs += 1; }
            Err(_) => break,
        }
    }

    let avg_loss = if successful_epochs > 0 { total_loss / successful_epochs as f64 } else { 0.0 };

    Json(TrainResponse {
        success: true, epochs_completed: successful_epochs, avg_loss,
        message: format!("Training completed. {}/{} epochs, avg loss: {:.6}", successful_epochs, payload.epochs, avg_loss),
    })
}

async fn query_model(State(state): State<AppState>, Json(payload): Json<QueryRequest>) -> Json<QueryResponse> {
    if payload.input.is_empty() {
        return Json(QueryResponse { success: false, generated_bytes: vec![], generated_text: String::new(), steps: 0, message: "Empty input".to_string() });
    }

    let steps = payload.steps.unwrap_or(1);
    let mut learner = state.learner.write().await;
    
    if learner.buffer.is_empty() {
        return Json(QueryResponse { success: false, generated_bytes: vec![], generated_text: String::new(), steps: 0, message: "Buffer empty - feed and train first".to_string() });
    }

    let generated_bytes = learner.query(&payload.input, steps);
    let generated_text = String::from_utf8_lossy(&generated_bytes).to_string();

    Json(QueryResponse {
        success: true, generated_bytes: generated_bytes.clone(), generated_text, steps,
        message: format!("Generated {} bytes in {} steps", generated_bytes.len(), steps),
    })
}

async fn generate_text(State(state): State<AppState>, Json(payload): Json<QueryRequest>) -> Json<QueryResponse> {
    query_model(State(state), Json(payload)).await
}

async fn ingest_directory(State(state): State<AppState>, Json(payload): Json<IngestDirectoryRequest>) -> Json<IngestDirectoryResponse> {
    let path = std::path::Path::new(&payload.path);

    if path.is_file() {
        match tokio::fs::read(&payload.path).await {
            Ok(content) => {
                let total_bytes = content.len();
                let source = DataSource::Local { path: payload.path.clone() };
                let knowledge_claim = hafa::epistemic::KnowledgeClaim::new(
                    &content, source.source_type().to_string(), source.source_id(),
                    source.is_direct_observation(), source.infer_category(),
                );
                let validated_data = ValidatedData {
                    content, source, epistemic_state: EpistemicState::new(0.9, true, 0, 0.1, 1, 0.0, 1.0),
                    timestamp: Utc::now().timestamp() as u64, knowledge_claim, metadata: None,
                };
                let mut learner = state.learner.write().await;
                learner.ingest(&validated_data);
                let buffer_size = learner.get_stats().buffer_size;

                Json(IngestDirectoryResponse {
                    success: true, files_processed: 1, total_bytes, buffer_size,
                    message: format!("Ingested 1 file ({} bytes). Buffer size: {}", total_bytes, buffer_size),
                })
            }
            Err(e) => Json(IngestDirectoryResponse { success: false, files_processed: 0, total_bytes: 0, buffer_size: 0, message: format!("Failed to read file: {}", e) }),
        }
    } else if path.is_dir() {
        match DataSource::fetch_directory_batch(&payload.path, payload.recursive, &state.config, &state.reputation).await {
            Ok(validated_data_list) => {
                let files_processed = validated_data_list.len();
                let mut total_bytes = 0;
                let mut learner = state.learner.write().await;
                for data in validated_data_list {
                    total_bytes += data.content.len();
                    learner.ingest(&data);
                }
                let buffer_size = learner.get_stats().buffer_size;
                Json(IngestDirectoryResponse {
                    success: true, files_processed, total_bytes, buffer_size,
                    message: format!("Ingested {} files ({} bytes). Buffer size: {}", files_processed, total_bytes, buffer_size),
                })
            }
            Err(e) => Json(IngestDirectoryResponse { success: false, files_processed: 0, total_bytes: 0, buffer_size: 0, message: format!("Failed: {}", e) }),
        }
    } else {
        Json(IngestDirectoryResponse { success: false, files_processed: 0, total_bytes: 0, buffer_size: 0, message: format!("Path does not exist: {}", payload.path) })
    }
}

// ============================================================================
// TRANSFORMER V3 HANDLERS (Legacy)
// ============================================================================

async fn generate_v3(State(state): State<AppState>, Json(payload): Json<GenerateV3Request>) -> Json<GenerateV3Response> {
    if payload.prompt.is_empty() {
        return Json(GenerateV3Response { success: false, generated_text: String::new(), steps: 0, message: "Empty prompt".to_string() });
    }
    if payload.steps == 0 || payload.steps > 100 {
        return Json(GenerateV3Response { success: false, generated_text: String::new(), steps: 0, message: "Steps must be between 1 and 100".to_string() });
    }

    let temperature = payload.temperature.unwrap_or(0.8);
    let top_k = payload.top_k.unwrap_or(40);
    
    let mut trainer = state.trainer.write().await;
    let prompt_bytes = payload.prompt.as_bytes();
    
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trainer.model.generate(prompt_bytes, payload.steps, temperature, top_k)
    }));

    match result {
        Ok(generated_bytes) => {
            let generated_text = String::from_utf8_lossy(&generated_bytes).to_string();
            Json(GenerateV3Response {
                success: true, generated_text, steps: payload.steps,
                message: format!("Generated {} bytes using HAFA Transformer v3 (temp={:.2}, top_k={})", generated_bytes.len(), temperature, top_k),
            })
        }
        Err(_) => Json(GenerateV3Response { success: false, generated_text: String::new(), steps: 0, message: "Generation failed due to internal error".to_string() }),
    }
}

async fn train_v3(State(state): State<AppState>, Json(payload): Json<TrainV3Request>) -> Json<TrainV3Response> {
    if payload.input.is_empty() || payload.target.is_empty() {
        return Json(TrainV3Response { success: false, final_loss: 0.0, message: "Input and target cannot be empty".to_string() });
    }

    let target_byte = payload.target.as_bytes()[0];
    let input_bytes = payload.input.as_bytes().to_vec();
    let dataset: Vec<(Vec<u8>, u8)> = (0..10).map(|_| (input_bytes.clone(), target_byte)).collect();

    let mut trainer = state.trainer.write().await;
    let final_loss = trainer.train_epochs(&dataset, payload.epochs);

    Json(TrainV3Response {
        success: true, final_loss,
        message: format!("Training completed for {} epochs. Final Loss: {:.4}", payload.epochs, final_loss),
    })
}

async fn train_text_v3(State(state): State<AppState>, Json(payload): Json<TrainTextV3Request>) -> Json<TrainTextV3Response> {
    if payload.text.is_empty() {
        return Json(TrainTextV3Response { 
            success: false, final_loss: 0.0, samples_trained: 0, 
            message: "Text cannot be empty".to_string(), cognitive_proof: None 
        });
    }
    if payload.epochs == 0 || payload.epochs > 100 {
        return Json(TrainTextV3Response { 
            success: false, final_loss: 0.0, samples_trained: 0, 
            message: "Epochs must be between 1 and 100".to_string(), cognitive_proof: None 
        });
    }

    let context_size = payload.context_size.unwrap_or(8);
    let text_bytes = payload.text.len();
    let mut trainer = state.trainer.write().await;
    
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trainer.train_on_text(&payload.text, context_size, payload.epochs)
    }));

    match result {
        Ok(final_loss) => {
            let samples = if text_bytes > context_size { text_bytes - context_size } else { 0 };
            let proof = trainer.generate_proof();

            Json(TrainTextV3Response {
                success: true, 
                final_loss, 
                samples_trained: samples * payload.epochs as usize,
                message: format!("Text training completed: {} chars, {} samples/epoch, {} epochs", text_bytes, samples, payload.epochs),
                cognitive_proof: Some(proof),
            })
        }
        Err(_) => Json(TrainTextV3Response { 
            success: false, final_loss: 0.0, samples_trained: 0, 
            message: "Training failed due to internal error".to_string(), cognitive_proof: None 
        }),
    }
}

async fn save_model(State(state): State<AppState>, Json(payload): Json<SaveModelRequest>) -> Json<SaveModelResponse> {
    let path = payload.path.unwrap_or_else(|| "hafa_model_v3.json".to_string());
    let trainer = state.trainer.read().await;
    
    match trainer.save(&path) {
        Ok(_) => Json(SaveModelResponse { success: true, message: format!("Model successfully saved to {}", path) }),
        Err(e) => Json(SaveModelResponse { success: false, message: format!("Failed to save model: {}", e) }),
    }
}

async fn load_model(State(state): State<AppState>, Json(payload): Json<LoadModelRequest>) -> Json<LoadModelResponse> {
    let path = payload.path.unwrap_or_else(|| "hafa_model_v3.json".to_string());
    let mut trainer = state.trainer.write().await;
    
    match trainer.load(&path) {
        Ok(_) => Json(LoadModelResponse { success: true, message: format!("Model successfully loaded from {}", path) }),
        Err(e) => Json(LoadModelResponse { success: false, message: format!("Failed to load model: {}", e) }),
    }
}

// ============================================================================
// TRANSFORMER V4 HANDLERS (Production-Grade)
// ============================================================================

async fn train_text_v4(State(state): State<AppState>, Json(payload): Json<TrainTextV4Request>) -> Json<TrainTextV4Response> {
    if payload.text.is_empty() {
        return Json(TrainTextV4Response {
            success: false, final_loss: 0.0, ema_loss: 0.0,
            samples_processed: 0, wall_time_ms: 0,
            message: "Text cannot be empty".to_string(),
            cognitive_proof: None,
        });
    }
    if payload.epochs == 0 || payload.epochs > 100 {
        return Json(TrainTextV4Response {
            success: false, final_loss: 0.0, ema_loss: 0.0,
            samples_processed: 0, wall_time_ms: 0,
            message: "Epochs must be between 1 and 100".to_string(),
            cognitive_proof: None,
        });
    }

    let context_size = payload.context_size.unwrap_or(8);
    let mut trainer = state.trainer_v4.lock().await;
    
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        trainer.train_on_text(&payload.text, context_size, payload.epochs)
    }));

    match result {
        Ok(proof) => {
            let hash_before_short = if proof.model_hash_before.len() >= 16 {
                &proof.model_hash_before[..16]
            } else {
                &proof.model_hash_before
            };
            let hash_after_short = if proof.model_hash_after.len() >= 16 {
                &proof.model_hash_after[..16]
            } else {
                &proof.model_hash_after
            };

            Json(TrainTextV4Response {
                success: true,
                final_loss: proof.loss_after,
                ema_loss: proof.ema_loss_after,
                samples_processed: proof.samples_processed,
                wall_time_ms: proof.wall_time_ms,
                message: format!(
                    "V4 Training complete: {} samples in {}ms | Model hash: {} â†’ {}",
                    proof.samples_processed,
                    proof.wall_time_ms,
                    hash_before_short,
                    hash_after_short
                ),
                cognitive_proof: Some(proof),
            })
        }
        Err(_) => Json(TrainTextV4Response {
            success: false, final_loss: 0.0, ema_loss: 0.0,
            samples_processed: 0, wall_time_ms: 0,
            message: "Training failed due to internal error".to_string(),
            cognitive_proof: None,
        }),
    }
}

// ============================================================================
// AUTO-LEARNING HANDLERS (Self-Evolving AI)
// ============================================================================

async fn auto_learn_feed(
    State(state): State<AppState>,
    Json(payload): Json<AutoLearnFeedRequest>,
) -> Json<AutoLearnFeedResponse> {
    if payload.text.is_empty() {
        return Json(AutoLearnFeedResponse {
            success: false,
            message: "Text cannot be empty".to_string(),
            buffer_size: 0,
        });
    }

    let mut engine = state.auto_learning.write().await;
    let sample = TrainingSample::new(payload.text.clone(), payload.source.clone(), payload.confidence);
    let success = engine.push_sample(sample);
    let buffer_size = engine.buffer_size();

    if success {
        if let Some(network) = &state.learning_network {
            if let Err(e) = network.broadcast_sample(payload.text, payload.source, payload.confidence).await {
                eprintln!("   âڑ ï¸ڈ  Failed to broadcast sample: {}", e);
            }
        }
    }

    Json(AutoLearnFeedResponse {
        success,
        message: if success {
            format!("Sample added to buffer and broadcasted to P2P network. Buffer size: {}", buffer_size)
        } else {
            "Sample rejected (low confidence, duplicate, or buffer full)".to_string()
        },
        buffer_size,
    })
}

async fn auto_learn_trigger(State(state): State<AppState>) -> Json<AutoLearnTriggerResponse> {
    let mut engine = state.auto_learning.write().await;
    
    match engine.trigger_learning() {
        Some(proof) => Json(AutoLearnTriggerResponse {
            success: true,
            message: format!(
                "Learning cycle completed in {}ms. Loss: {:.4} â†’ {:.4}",
                proof.wall_time_ms,
                proof.loss_before,
                proof.loss_after
            ),
            proof: Some(AutoLearnProofSummary {
                loss_before: proof.loss_before,
                loss_after: proof.loss_after,
                quality_score: proof.quality_score(),
                samples_processed: proof.samples_processed,
                gradient_commitment: proof.gradient_commitment,
            }),
        }),
        None => Json(AutoLearnTriggerResponse {
            success: false,
            message: "Learning not triggered (not enough samples or too soon since last cycle)".to_string(),
            proof: None,
        }),
    }
}

async fn auto_learn_status(State(state): State<AppState>) -> Json<AutoLearnStatusResponse> {
    let engine = state.auto_learning.read().await;
    
    Json(AutoLearnStatusResponse {
        is_learning: engine.is_learning(),
        buffer_size: engine.buffer_size(),
        max_buffer_size: 1000,
    })
}

async fn auto_learn_stats(State(state): State<AppState>) -> Json<AutoLearnStatsResponse> {
    let engine = state.auto_learning.read().await;
    let stats = engine.stats();
    
    Json(AutoLearnStatsResponse {
        total_cycles: stats.total_cycles,
        total_samples_received: stats.total_samples_received,
        total_samples_rejected: stats.total_samples_rejected,
        total_samples_learned: stats.total_samples_learned,
        total_proofs_generated: stats.total_proofs_generated,
        last_cycle_time_secs: stats.last_cycle_time,
        last_cycle_loss: stats.last_cycle_loss,
        buffer_size: stats.buffer_size,
        curiosity_accepted: stats.curiosity_accepted,
        curiosity_rejected: stats.curiosity_rejected,
        meta_learning_checks: stats.meta_learning_checks,
        meta_learning_skips: stats.meta_learning_skips,
        meta_learning_boosts: stats.meta_learning_boosts,
        kg_entities_retrieved: stats.kg_entities_retrieved,
        kg_entities_added: stats.kg_entities_added,
        kg_relations_added: stats.kg_relations_added,
    })
}

// ============================================================================
// BLOCKCHAIN META-LEARNING HANDLER
// ============================================================================

async fn poll_blockchain(State(state): State<AppState>) -> Json<BlockchainPollResponse> {
    let mut engine = state.auto_learning.write().await;
    let bc = state.blockchain.read().await;
    
    let current_height = bc.get_chain_height().await;
    let new_samples = engine.poll_sources().await;
    
    Json(BlockchainPollResponse {
        success: true,
        new_samples,
        last_processed_height: current_height,
        current_height,
        message: if new_samples > 0 {
            format!("Polled blockchain: {} new sample(s) added to buffer", new_samples)
        } else {
            "No new blocks found since last poll".to_string()
        },
    })
}

// ============================================================================
// EPISODIC MEMORY HANDLERS
// ============================================================================

async fn auto_learn_episodes(State(state): State<AppState>) -> Json<Vec<EpisodeResponse>> {
    let engine = state.auto_learning.read().await;
    let episodes = engine.episodes();
    
    let response: Vec<EpisodeResponse> = episodes.iter().map(|ep| {
        EpisodeResponse {
            id: ep.id.clone(),
            timestamp: ep.timestamp,
            sample_count: ep.sample_count,
            loss_before: ep.outcome.loss_before,
            loss_after: ep.outcome.loss_after,
            loss_improvement: ep.outcome.loss_improvement,
            quality_score: ep.outcome.quality_score,
            duration_ms: ep.outcome.duration_ms,
            success: ep.outcome.success,
            tags: ep.tags.clone(),
        }
    }).collect();
    
    Json(response)
}

async fn auto_learn_episodic_stats(State(state): State<AppState>) -> Json<EpisodicStatsResponse> {
    let engine = state.auto_learning.read().await;
    let stats = engine.episodic_stats();
    
    let success_rate = if stats.total_episodes > 0 {
        stats.successful_episodes as f64 / stats.total_episodes as f64
    } else {
        0.0
    };
    
    Json(EpisodicStatsResponse {
        total_episodes: stats.total_episodes,
        successful_episodes: stats.successful_episodes,
        failed_episodes: stats.failed_episodes,
        avg_loss_improvement: stats.avg_loss_improvement,
        avg_quality_score: stats.avg_quality_score,
        success_rate,
    })
}

// ============================================================================
// NETWORK SIMULATION HANDLER
// ============================================================================

async fn simulate_network_data(
    State(state): State<AppState>,
    Json(payload): Json<SimulateNetworkRequest>,
) -> Json<SimulateNetworkResponse> {
    let mut engine = state.auto_learning.write().await;
    
    let sample = TrainingSample::new(
        payload.text,
        "p2p_gossip".to_string(),
        payload.confidence,
    );
    
    let success = engine.push_sample(sample);
    let buffer_size = engine.buffer_size();
    
    Json(SimulateNetworkResponse {
        success,
        message: if success {
            "Simulated P2P sample added to buffer".to_string()
        } else {
            "Sample rejected".to_string()
        },
        buffer_size,
    })
}

// ============================================================================
// KNOWLEDGE GRAPH HANDLERS
// ============================================================================

async fn knowledge_entities(State(state): State<AppState>) -> Json<Vec<EntityResponse>> {
    let kg = state.knowledge_graph.read().await;
    let entities = kg.entities();
    
    let response: Vec<EntityResponse> = entities.iter().map(|e| {
        EntityResponse {
            id: e.id.clone(),
            name: e.name.clone(),
            entity_type: e.entity_type.as_str().to_string(),
            confidence: e.confidence,
            mentions: e.mentions,
            created_at: e.created_at,
            properties: e.properties.clone(),
        }
    }).collect();
    
    Json(response)
}

async fn knowledge_relations(State(state): State<AppState>) -> Json<Vec<RelationResponse>> {
    let kg = state.knowledge_graph.read().await;
    let relations = kg.relations();
    
    let response: Vec<RelationResponse> = relations.iter().map(|r| {
        RelationResponse {
            id: r.id.clone(),
            source_id: r.source_id.clone(),
            target_id: r.target_id.clone(),
            relation_type: r.relation_type.as_str().to_string(),
            confidence: r.confidence,
            weight: r.weight,
            created_at: r.created_at,
        }
    }).collect();
    
    Json(response)
}

async fn knowledge_stats(State(state): State<AppState>) -> Json<KnowledgeGraphStatsResponse> {
    let kg = state.knowledge_graph.read().await;
    let stats = kg.stats();
    
    Json(KnowledgeGraphStatsResponse {
        total_entities: stats.total_entities,
        total_relations: stats.total_relations,
        entities_by_type: stats.entities_by_type,
        relations_by_type: stats.relations_by_type,
        avg_entity_confidence: stats.avg_entity_confidence,
        avg_relation_confidence: stats.avg_relation_confidence,
    })
}

async fn knowledge_add_entity(
    State(state): State<AppState>,
    Json(payload): Json<AddEntityRequest>,
) -> Json<AddEntityResponse> {
    let mut kg = state.knowledge_graph.write().await;
    let entity_type = EntityType::from_string(&payload.entity_type);
    let confidence = payload.confidence.unwrap_or(0.8);
    
    let entity_id = kg.add_entity(payload.name.clone(), entity_type, confidence);
    
    Json(AddEntityResponse {
        success: true,
        entity_id,
        message: format!("Entity '{}' added/updated", payload.name),
    })
}

async fn knowledge_add_relation(
    State(state): State<AppState>,
    Json(payload): Json<AddRelationRequest>,
) -> Json<AddRelationResponse> {
    let mut kg = state.knowledge_graph.write().await;
    let relation_type = RelationType::from_string(&payload.relation_type);
    let confidence = payload.confidence.unwrap_or(0.8);
    
    let relation_id = kg.add_relation(
        &payload.source,
        &payload.target,
        relation_type,
        confidence,
    );
    
    Json(AddRelationResponse {
        success: relation_id.is_some(),
        relation_id: relation_id.clone(),
        message: if relation_id.is_some() {
            format!("Relation '{}' â†’ '{}' added/updated", payload.source, payload.target)
        } else {
            "Failed to add relation (entities not found)".to_string()
        },
    })
}

async fn knowledge_extract(
    State(state): State<AppState>,
    Json(payload): Json<ExtractKnowledgeRequest>,
) -> Json<ExtractKnowledgeResponse> {
    let mut kg = state.knowledge_graph.write().await;
    let (entities, relations) = kg.extract_from_text(&payload.text);
    
    Json(ExtractKnowledgeResponse {
        success: true,
        entities_extracted: entities,
        relations_extracted: relations,
        message: format!("Extracted {} entities and {} relations from text", entities, relations),
    })
}

// ============================================================================
// REASONING ENGINE HANDLER
// ============================================================================

async fn knowledge_query(
    State(state): State<AppState>,
    Json(payload): Json<KnowledgeQueryRequest>,
) -> Json<KnowledgeQueryResponse> {
    let kg = state.knowledge_graph.read().await;
    let engine = state.reasoning.read().await;
    
    let result = engine.query(&kg, &payload.query);
    
    Json(KnowledgeQueryResponse {
        query: result.query,
        answer: result.answer,
        confidence: result.confidence,
        entities_found: result.entities_found,
        relations_found: result.relations_found,
        inference_path: result.inference_path,
    })
}

// ============================================================================
// BACKEND BENCHMARK HANDLER
// ============================================================================

async fn benchmark_backend(State(state): State<AppState>) -> Json<Vec<BenchmarkResult>> {
    let trainer = state.trainer_v4.lock().await;
    let ops = AcceleratedOps::new(trainer.backend.as_ref()); 
    let results = ops.run_full_benchmark();
    Json(results)
}

// ============================================================================
// P2P NETWORK HANDLERS
// ============================================================================

async fn p2p_info(State(state): State<AppState>) -> Json<P2PInfoResponse> {
    if let Some(network) = &state.learning_network {
        Json(P2PInfoResponse {
            peer_id: network.local_peer_id(),
            is_running: network.is_running(),
            listening_addresses: network.get_listening_addresses().await,
        })
    } else {
        Json(P2PInfoResponse {
            peer_id: "N/A".to_string(),
            is_running: false,
            listening_addresses: vec![],
        })
    }
}

async fn p2p_connect(
    State(state): State<AppState>,
    Json(payload): Json<P2PConnectRequest>,
) -> Json<P2PConnectResponse> {
    if let Some(network) = &state.learning_network {
        match network.dial_peer(&payload.multiaddr).await {
            Ok(_) => Json(P2PConnectResponse {
                success: true,
                message: format!("Dialing peer: {}", payload.multiaddr),
            }),
            Err(e) => Json(P2PConnectResponse {
                success: false,
                message: format!("Failed to dial: {}", e),
            }),
        }
    } else {
        Json(P2PConnectResponse {
            success: false,
            message: "Learning network not initialized".to_string(),
        })
    }
}

// ============================================================================
// FEDERATED LEARNING HANDLERS
// ============================================================================

async fn federated_share(
    State(state): State<AppState>,
    Json(payload): Json<FederatedShareRequest>,
) -> Json<FederatedShareResponse> {
    if payload.text.is_empty() {
        return Json(FederatedShareResponse {
            success: false,
            item_id: String::new(),
            pool_size: 0,
            message: "Text cannot be empty".to_string(),
        });
    }

    let item_id = format!("{}_{}", Utc::now().timestamp(), rand::random::<u32>());
    let item = LearningPoolItem {
        id: item_id.clone(),
        text: payload.text,
        source: payload.source,
        confidence: payload.confidence,
        timestamp: Utc::now().timestamp() as u64,
        peer_id: payload.peer_id,
    };

    let mut pool = state.learning_pool.write().await;
    
    while pool.len() >= 1000 {
        pool.pop_front();
    }
    
    pool.push_back(item);
    let pool_size = pool.len();

    Json(FederatedShareResponse {
        success: true,
        item_id,
        pool_size,
        message: format!("Sample shared with federated pool. Pool size: {}", pool_size),
    })
}

async fn federated_poll(State(state): State<AppState>) -> Json<FederatedPollResponse> {
    let pool = state.learning_pool.read().await;
    
    let samples: Vec<LearningPoolItem> = pool.iter()
        .rev()
        .take(50)
        .cloned()
        .collect();
    
    let count = samples.len();
    
    Json(FederatedPollResponse {
        success: true,
        samples,
        count,
        message: format!("Retrieved {} samples from federated pool", count),
    })
}

async fn federated_stats(State(state): State<AppState>) -> Json<FederatedStatsResponse> {
    let pool = state.learning_pool.read().await;
    let now = Utc::now().timestamp() as u64;
    
    let oldest_age = pool.front().map(|item| now - item.timestamp);
    let newest_age = pool.back().map(|item| now - item.timestamp);
    
    Json(FederatedStatsResponse {
        pool_size: pool.len(),
        total_shared: pool.len() as u64,
        total_received: 0,
        oldest_item_age_secs: oldest_age,
        newest_item_age_secs: newest_age,
    })
}

// ============================================================================
// GPU BACKEND HANDLER
// ============================================================================

async fn gpu_info() -> Json<serde_json::Value> {
    let gpu_available = WgpuBackend::is_available().await;
    
    if gpu_available {
        match WgpuBackend::new().await {
            Ok(gpu) => {
                let info = gpu.info();
                Json(serde_json::json!({
                    "success": true,
                    "backend": "WGPU",
                    "device_name": info.device_name,
                    "device_type": format!("{:?}", info.device_type),
                    "memory_mb": info.memory_mb,
                    "compute_units": info.compute_units,
                    "supports_fp16": info.supports_fp16,
                    "message": "GPU is available and ready!"
                }))
            }
            Err(e) => {
                Json(serde_json::json!({
                    "success": false,
                    "backend": "WGPU",
                    "message": format!("GPU initialization failed: {}", e)
                }))
            }
        }
    } else {
        let cpu = CpuBackend::new();
        let info = cpu.info();
        Json(serde_json::json!({
            "success": true,
            "backend": "CPU",
            "device_name": info.device_name,
            "device_type": format!("{:?}", info.device_type),
            "memory_mb": info.memory_mb,
            "compute_units": info.compute_units,
            "supports_fp16": info.supports_fp16,
            "message": "GPU not available, using CPU fallback"
        }))
    }
}

// ============================================================================
// WEB UI DASHBOARD HANDLER (Inline HTML/CSS/JS)
// ============================================================================

async fn web_dashboard() -> Html<String> {
    let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HAFA v5.1 Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
            color: #e2e8f0; 
            min-height: 100vh; 
            padding: 2rem; 
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 3rem; }
        header h1 { 
            font-size: 3rem; 
            background: linear-gradient(to right, #38bdf8, #818cf8); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            background-clip: text;
        }
        .subtitle { color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem; }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
            gap: 1.5rem; 
            margin-bottom: 2rem; 
        }
        .card { 
            background: rgba(30, 41, 59, 0.7); 
            backdrop-filter: blur(10px); 
            border: 1px solid rgba(148, 163, 184, 0.1); 
            border-radius: 1rem; 
            padding: 1.5rem; 
            transition: transform 0.2s, border-color 0.2s; 
        }
        .card:hover { transform: translateY(-4px); border-color: #38bdf8; }
        .card h2 { 
            color: #38bdf8; 
            margin-bottom: 1rem; 
            font-size: 1.2rem; 
            border-bottom: 1px solid rgba(148, 163, 184, 0.2); 
            padding-bottom: 0.5rem; 
        }
        .stat { 
            display: flex; 
            justify-content: space-between; 
            padding: 0.5rem 0; 
            border-bottom: 1px solid rgba(148, 163, 184, 0.1); 
        }
        .stat:last-child { border-bottom: none; }
        .stat strong { color: #f8fafc; font-family: 'Courier New', monospace; word-break: break-all; }
        .action-card { text-align: center; }
        .actions { display: flex; gap: 1rem; justify-content: center; margin-top: 1rem; flex-wrap: wrap; }
        button { 
            background: linear-gradient(135deg, #38bdf8, #818cf8); 
            color: white; 
            border: none; 
            padding: 0.75rem 1.5rem; 
            border-radius: 0.5rem; 
            font-weight: 600; 
            cursor: pointer; 
            transition: opacity 0.2s, transform 0.2s; 
        }
        button:hover { opacity: 0.9; transform: scale(1.05); }
        .result { 
            margin-top: 1rem; 
            padding: 1rem; 
            background: rgba(15, 23, 42, 0.5); 
            border-radius: 0.5rem; 
            font-family: monospace; 
            color: #38bdf8; 
            display: none; 
            text-align: left;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
        footer { text-align: center; margin-top: 3rem; color: #64748b; }
        .loading { color: #fbbf24; font-style: italic; }
        .success { color: #10b981; }
        .error { color: #ef4444; }
        .status-dot { 
            display: inline-block; 
            width: 8px; 
            height: 8px; 
            border-radius: 50%; 
            background: #10b981; 
            margin-right: 6px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .wallet-section {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(30, 41, 59, 0.7);
            border-radius: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }
        .wallet-section h2 {
            color: #fbbf24;
            margin-bottom: 1rem;
        }
        .wallet-input {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(148, 163, 184, 0.2);
            color: #e2e8f0;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ًںڑ€ HAFA Dashboard</h1>
            <p class="subtitle"><span class="status-dot"></span>Decentralized AI Blockchain â€¢ v5.1.0 â€¢ Auto-refresh every 5s</p>
        </header>

        <div class="grid">
            <div class="card">
                <h2>â›“ï¸ڈ Blockchain</h2>
                <div class="stat"><span>Height:</span> <strong id="height" class="loading">Loading...</strong></div>
                <div class="stat"><span>Total Minted:</span> <strong id="minted" class="loading">Loading...</strong></div>
                <div class="stat"><span>Reward:</span> <strong id="reward" class="loading">Loading...</strong></div>
                <div class="stat"><span>Uptime:</span> <strong id="uptime" class="loading">Loading...</strong></div>
            </div>

            <div class="card">
                <h2>ًں§  AI Model</h2>
                <div class="stat"><span>Parameters:</span> <strong id="params" class="loading">Loading...</strong></div>
                <div class="stat"><span>Buffer Size:</span> <strong id="buffer" class="loading">Loading...</strong></div>
                <div class="stat"><span>Is Learning:</span> <strong id="learning" class="loading">Loading...</strong></div>
            </div>

            <div class="card">
                <h2>ًںژ® Compute Backend</h2>
                <div class="stat"><span>Backend:</span> <strong id="backend" class="loading">Loading...</strong></div>
                <div class="stat"><span>Device:</span> <strong id="device" class="loading">Loading...</strong></div>
                <div class="stat"><span>FP16 Support:</span> <strong id="fp16" class="loading">Loading...</strong></div>
            </div>

            <div class="card">
                <h2>ًں¤‌ Federated Pool</h2>
                <div class="stat"><span>Pool Size:</span> <strong id="pool" class="loading">Loading...</strong></div>
                <div class="stat"><span>Total Shared:</span> <strong id="shared" class="loading">Loading...</strong></div>
                <div class="stat"><span>Oldest Item:</span> <strong id="oldest" class="loading">Loading...</strong></div>
            </div>

            <div class="card">
                <h2>ًں§  Knowledge Graph</h2>
                <div class="stat"><span>Entities:</span> <strong id="entities" class="loading">Loading...</strong></div>
                <div class="stat"><span>Relations:</span> <strong id="relations" class="loading">Loading...</strong></div>
            </div>

            <div class="card">
                <h2>ًںŒگ P2P Network</h2>
                <div class="stat"><span>Peer ID:</span> <strong id="peer" class="loading">Loading...</strong></div>
                <div class="stat"><span>Running:</span> <strong id="running" class="loading">Loading...</strong></div>
            </div>
        </div>

        <div class="wallet-section">
            <h2>ًں’¼ Wallet Management</h2>
            <div class="actions">
                <button onclick="createWallet()">ًں”‘ Create New Wallet</button>
                <button onclick="listWallets()">ًں“‹ List Wallets</button>
                <input type="text" id="wallet-address" class="wallet-input" placeholder="Wallet address for balance check" style="width: 300px;">
                <button onclick="checkBalance()">ًں’° Check Balance</button>
            </div>
            <div id="wallet-result" class="result"></div>
        </div>

        <div class="card action-card" style="margin-top: 2rem;">
            <h2>âڑ، Quick Actions</h2>
            <div class="actions">
                <button onclick="refreshData()">ًں”„ Refresh Data</button>
                <button onclick="testFederated()">ًں“¤ Test Federated Share</button>
                <button onclick="trainModel()">ًں§  Train Model (5 epochs)</button>
                <button onclick="queryKnowledge()">ًں”چ Query Knowledge</button>
            </div>
            <div id="action-result" class="result"></div>
        </div>

        <footer>
            <p>HAFA - Horizon After Freedom Achieved</p>
            <p style="margin-top: 0.5rem; font-size: 0.9rem;">Decentralized AI Blockchain with Native Transformer</p>
        </footer>
    </div>

    <script>
        const API = window.location.origin;

        async function fetchJSON(endpoint, options) {
            try {
                const res = await fetch(API + endpoint, options);
                if (!res.ok) throw new Error('HTTP ' + res.status);
                return await res.json();
            } catch (e) { 
                console.error('API Error:', endpoint, e);
                return { error: e.message }; 
            }
        }

        function showResult(data, isError, targetId) {
            const div = document.getElementById(targetId || 'action-result');
            div.style.display = 'block';
            div.className = isError ? 'result error' : 'result';
            div.textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
        }

        function formatUptime(secs) {
            if (secs < 60) return secs + 's';
            if (secs < 3600) return Math.floor(secs/60) + 'm ' + (secs%60) + 's';
            const h = Math.floor(secs/3600);
            const m = Math.floor((secs%3600)/60);
            return h + 'h ' + m + 'm';
        }

        async function refreshData() {
            const info = await fetchJSON('/info');
            if (info && !info.error) {
                const el = document.getElementById('height');
                el.textContent = info.height;
                el.className = 'success';
                document.getElementById('minted').textContent = info.total_minted_hafa.toFixed(2) + ' HAFA';
                document.getElementById('minted').className = 'success';
                document.getElementById('reward').textContent = info.current_reward_hafa.toFixed(4) + ' HAFA';
                document.getElementById('reward').className = 'success';
                document.getElementById('uptime').textContent = formatUptime(info.uptime_secs);
                document.getElementById('uptime').className = 'success';
            }

            const learn = await fetchJSON('/learning-status');
            if (learn && !learn.error) {
                document.getElementById('params').textContent = learn.total_parameters.toLocaleString();
                document.getElementById('params').className = 'success';
            }

            const auto = await fetchJSON('/auto-learn/status');
            if (auto && !auto.error) {
                document.getElementById('buffer').textContent = auto.buffer_size;
                document.getElementById('buffer').className = 'success';
                const l = document.getElementById('learning');
                l.textContent = auto.is_learning ? 'âœ… Yes' : 'â‌Œ No';
                l.className = 'success';
            }

            const gpu = await fetchJSON('/gpu/info');
            if (gpu && !gpu.error) {
                document.getElementById('backend').textContent = gpu.backend;
                document.getElementById('backend').className = 'success';
                document.getElementById('device').textContent = gpu.device_name;
                document.getElementById('device').className = 'success';
                const f = document.getElementById('fp16');
                f.textContent = gpu.supports_fp16 ? 'âœ… Yes' : 'â‌Œ No';
                f.className = 'success';
            }

            const fed = await fetchJSON('/federated/stats');
            if (fed && !fed.error) {
                document.getElementById('pool').textContent = fed.pool_size;
                document.getElementById('pool').className = 'success';
                document.getElementById('shared').textContent = fed.total_shared;
                document.getElementById('shared').className = 'success';
                const o = document.getElementById('oldest');
                o.textContent = fed.oldest_item_age_secs ? fed.oldest_item_age_secs + 's ago' : 'N/A';
                o.className = 'success';
            }

            const kg = await fetchJSON('/knowledge/stats');
            if (kg && !kg.error) {
                document.getElementById('entities').textContent = kg.total_entities;
                document.getElementById('entities').className = 'success';
                document.getElementById('relations').textContent = kg.total_relations;
                document.getElementById('relations').className = 'success';
            }

            const p2p = await fetchJSON('/p2p/info');
            if (p2p && !p2p.error) {
                document.getElementById('peer').textContent = p2p.peer_id.substring(0, 20) + '...';
                document.getElementById('peer').className = 'success';
                const r = document.getElementById('running');
                r.textContent = p2p.is_running ? 'âœ… Yes' : 'â‌Œ No';
                r.className = 'success';
            }
        }

        async function createWallet() {
            const passphrase = prompt('Enter a strong passphrase for your new wallet:');
            if (!passphrase) return;
            const label = prompt('Optional label for this wallet:');
            
            showResult('Creating wallet...', false, 'wallet-result');
            const data = await fetchJSON('/wallet/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ passphrase, label: label || null })
            });
            showResult(data, data.error, 'wallet-result');
        }

        async function listWallets() {
            showResult('Loading wallets...', false, 'wallet-result');
            const data = await fetchJSON('/wallet/list');
            showResult(data, data.error, 'wallet-result');
        }

        async function checkBalance() {
            const address = document.getElementById('wallet-address').value.trim();
            if (!address) {
                alert('Please enter a wallet address');
                return;
            }
            showResult('Checking balance...', false, 'wallet-result');
            const data = await fetchJSON('/wallet/' + encodeURIComponent(address) + '/info');
            showResult(data, data.error, 'wallet-result');
        }

        async function testFederated() {
            showResult('Sending sample to federated pool...', false);
            const data = await fetchJSON('/federated/share', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: 'HAFA Web UI test at ' + new Date().toISOString(), 
                    source: 'web_ui', 
                    confidence: 0.95, 
                    peer_id: 'browser' 
                })
            });
            showResult(data, data.error);
            setTimeout(refreshData, 500);
        }

        async function trainModel() {
            showResult('Training model (this may take a few seconds)...', false);
            const data = await fetchJSON('/train-text-v4', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: 'HAFA is a revolutionary decentralized AI blockchain with GPU acceleration and federated learning capabilities', 
                    context_size: 8,
                    epochs: 5
                })
            });
            showResult(data, data.error);
        }

        async function queryKnowledge() {
            const query = prompt('Enter your query:', 'What is HAFA?');
            if (!query) return;
            showResult('Querying knowledge graph...', false);
            const data = await fetchJSON('/knowledge/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });
            showResult(data, data.error);
        }

        // Auto-refresh every 5 seconds
        setInterval(refreshData, 5000);
        refreshData();
    </script>
</body>
</html>"#;
    
    Html(html.to_string())
}

// ============================================================================
// WALLET HANDLERS
// ============================================================================

async fn wallet_create(
    State(state): State<AppState>,
    Json(payload): Json<WalletCreateRequest>,
) -> Json<WalletCreateResponse> {
    let mut manager = state.wallet_manager.lock().await;
    
    match manager.create_wallet(&payload.passphrase, payload.label) {
        Ok(wallet) => Json(WalletCreateResponse {
            success: true,
            address: wallet.address,
            label: wallet.label,
            message: "Wallet created successfully".to_string(),
        }),
        Err(e) => Json(WalletCreateResponse {
            success: false,
            address: String::new(),
            label: None,
            message: format!("Failed to create wallet: {}", e),
        }),
    }
}

async fn wallet_import(
    State(state): State<AppState>,
    Json(payload): Json<WalletImportRequest>,
) -> Json<WalletImportResponse> {
    let mut manager = state.wallet_manager.lock().await;
    
    match manager.import_from_passphrase(&payload.passphrase, payload.label) {
        Ok(wallet) => Json(WalletImportResponse {
            success: true,
            address: wallet.address,
            label: wallet.label,
            message: "Wallet imported successfully".to_string(),
        }),
        Err(e) => Json(WalletImportResponse {
            success: false,
            address: String::new(),
            label: None,
            message: format!("Failed to import wallet: {}", e),
        }),
    }
}

async fn wallet_list(State(state): State<AppState>) -> Json<WalletListResponse> {
    let manager = state.wallet_manager.lock().await;
    let wallets = manager.list_wallets();
    let count = wallets.len();
    
    Json(WalletListResponse {
        success: true,
        wallets,
        count,
    })
}

async fn wallet_info(
    State(state): State<AppState>,
    Path(address): Path<String>,
) -> Json<WalletInfoResponse> {
    let manager = state.wallet_manager.lock().await;
    
    match manager.get_wallet_info(&address) {
        Ok(wallet) => {
            let bc = state.blockchain.read().await;
            let balance = bc.get_balance(&address).await.unwrap_or(0);
            
            Json(WalletInfoResponse {
                success: true,
                wallet: Some(wallet),
                balance: Some(balance),
                balance_hafa: Some(balance as f64 / 100_000_000.0),
                message: "Wallet info retrieved".to_string(),
            })
        }
        Err(e) => Json(WalletInfoResponse {
            success: false,
            wallet: None,
            balance: None,
            balance_hafa: None,
            message: format!("Wallet not found: {}", e),
        }),
    }
}

async fn wallet_sign_transaction(
    State(state): State<AppState>,
    Path(address): Path<String>,
    Json(payload): Json<WalletSignRequest>,
) -> Json<WalletSignResponse> {
    let manager = state.wallet_manager.lock().await;
    
    let tx = TransactionRequest::new(
        address.clone(),
        payload.to_address,
        payload.amount,
        payload.fee,
    );
    
    match manager.sign_transaction(&address, &payload.passphrase, &tx) {
        Ok(signed_tx) => Json(WalletSignResponse {
            success: true,
            signed_transaction: Some(signed_tx),
            message: "Transaction signed successfully".to_string(),
        }),
        Err(e) => Json(WalletSignResponse {
            success: false,
            signed_transaction: None,
            message: format!("Failed to sign transaction: {}", e),
        }),
    }
}

async fn wallet_delete(
    State(state): State<AppState>,
    Path(address): Path<String>,
    Json(_payload): Json<WalletDeleteRequest>,
) -> Json<WalletDeleteResponse> {
    let mut manager = state.wallet_manager.lock().await;
    
    match manager.delete_wallet(&address) {
        Ok(_) => Json(WalletDeleteResponse {
            success: true,
            message: "Wallet deleted successfully".to_string(),
        }),
        Err(e) => Json(WalletDeleteResponse {
            success: false,
            message: format!("Failed to delete wallet: {}", e),
        }),
    }
}