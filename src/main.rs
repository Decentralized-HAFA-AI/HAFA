// ============================================================================
// HAFA - src/main.rs — GENESIS NODE WITH MINING API
// ============================================================================

use hafa::config::Config;
use hafa::blockchain::{Blockchain, TransactionType};
use hafa::network::NetworkEngine;
use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::sync::RwLock;
use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

type SharedBlockchain = Arc<RwLock<Blockchain>>;

#[derive(Serialize)]
struct BalanceResponse { address: String, balance: u64, balance_hafa: f64 }
#[derive(Serialize)]
struct HeightResponse { height: u64 }
#[derive(Serialize)]
struct InfoResponse { 
    version: String, 
    height: u64, 
    total_minted: u64, 
    total_minted_hafa: f64, 
    network: String,
    current_reward: u64,
    current_reward_hafa: f64,
}

#[derive(Serialize)]
struct TaskResponse { last_hash: String, difficulty: u32, target_height: u64 }
#[derive(Deserialize)]
struct SubmitRequest { miner_addr: String, nonce: u64, cognitive_proof: String }
#[derive(Serialize)]
struct SubmitResponse { success: bool, block_index: Option<u64>, reward: u64, reward_hafa: f64, message: String }

#[tokio::main]
async fn main() {
    println!("🚀 HAFA Genesis Node Starting...");
    let config = Config::default();
    let bc = Blockchain::new(&config).await.expect("Failed to initialize blockchain");
    
    let (tx_tx, _) = mpsc::channel(100);
    let (block_tx, _) = mpsc::channel(100);
    let _ = NetworkEngine::new(&config, tx_tx, block_tx).await;

    let shared_bc: SharedBlockchain = Arc::new(RwLock::new(bc));

    let app = Router::new()
        .route("/balance/:address", get(get_balance))
        .route("/height", get(get_height))
        .route("/info", get(get_info))
        .route("/task", get(get_task))
        .route("/submit", post(submit_solution))
        .with_state(shared_bc.clone());

    let api_handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:7476").await.unwrap();
        println!("   🌐 HTTP API & Mining Pool started on http://127.0.0.1:7476");
        axum::serve(listener, app).await.unwrap();
    });

    println!("   ✅ Node is alive. Press Ctrl+C to stop.");
    api_handle.await.unwrap();
}

async fn get_balance(State(bc): State<SharedBlockchain>, Path(address): Path<String>) -> Json<BalanceResponse> {
    let bc = bc.read().await;
    let balance = bc.get_balance(&address).await.unwrap_or(0);
    Json(BalanceResponse { address, balance, balance_hafa: balance as f64 / 100_000_000.0 })
}

async fn get_height(State(bc): State<SharedBlockchain>) -> Json<HeightResponse> {
    let bc = bc.read().await;
    Json(HeightResponse { height: bc.get_chain_height().await })
}

async fn get_info(State(bc): State<SharedBlockchain>) -> Json<InfoResponse> {
    let bc = bc.read().await;
    let height = bc.get_chain_height().await;
    let total_minted = bc.get_total_minted().await;
    let current_reward = bc.get_current_reward().await;
    Json(InfoResponse { 
        version: "1.0.0".into(), 
        height, 
        total_minted, 
        total_minted_hafa: total_minted as f64 / 100_000_000.0, 
        network: "mainnet".into(),
        current_reward,
        current_reward_hafa: current_reward as f64 / 100_000_000.0,
    })
}

async fn get_task(State(bc): State<SharedBlockchain>) -> Json<TaskResponse> {
    let bc = bc.read().await;
    match bc.get_task().await {
        Ok((hash, diff, height)) => Json(TaskResponse { last_hash: hash, difficulty: diff, target_height: height }),
        Err(_) => Json(TaskResponse { last_hash: "0".repeat(64), difficulty: 1, target_height: 1 }),
    }
}

async fn submit_solution(State(bc): State<SharedBlockchain>, Json(payload): Json<SubmitRequest>) -> Json<SubmitResponse> {
    let bc = bc.read().await;
    match bc.submit_solution(&payload.miner_addr, payload.nonce, &payload.cognitive_proof).await {
        Ok(block) => {
            let reward = block.transactions.iter()
                .find(|tx| tx.tx_type == TransactionType::Reward && tx.from == "SYSTEM")
                .map(|tx| tx.amount).unwrap_or(0);
            Json(SubmitResponse { 
                success: true, 
                block_index: Some(block.index), 
                reward, 
                reward_hafa: reward as f64 / 100_000_000.0,
                message: format!("Block #{} mined!", block.index) 
            })
        }
        Err(e) => Json(SubmitResponse { success: false, block_index: None, reward: 0, reward_hafa: 0.0, message: format!("Failed: {}", e) }),
    }
}