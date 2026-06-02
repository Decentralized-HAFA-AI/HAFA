// ============================================================================
// HAFA - src/main.rs — GENESIS NODE WITH HTTP API
// ============================================================================

use hafa::config::Config;
use hafa::blockchain::Blockchain;
use hafa::network::NetworkEngine;
use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::sync::RwLock;
use axum::{
    extract::{Path, State},
    routing::get,
    Json, Router,
};
use serde::Serialize;

// Shared state for API
type SharedBlockchain = Arc<RwLock<Blockchain>>;

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
}

#[tokio::main]
async fn main() {
    println!("🚀 HAFA Genesis Node Starting...");

    // 1. Load Config
    let config = Config::default();
    println!("   ⚙️  Configuration loaded.");

    // 2. Initialize Blockchain
    let bc = Blockchain::new(&config).await.expect("Failed to initialize blockchain");
    println!("   ⛓️  Blockchain initialized.");

    // 3. Auto-Mine if first run
    let height = bc.get_chain_height().await;
    if height <= 1 {
        println!("   ⛏️  Mining Genesis Reward Block...");
        match bc.mine_block("Genesis_Alpha_001", "Genesis_Cognitive_Proof_001").await {
            Ok(block) => {
                println!("   🎉 SUCCESS! Block #{} mined.", block.index);
                println!("   💰 500 HAFA credited to anonymous wallet.");
            }
            Err(e) => println!("   ⚠️  Could not auto-mine: {}", e),
        }
    } else {
        println!("   ✅ Blockchain already has {} blocks. Skipping genesis mining.", height);
    }

    // 4. Initialize Network
    let (tx_tx, _) = mpsc::channel(100);
    let (block_tx, _) = mpsc::channel(100);
    
    match NetworkEngine::new(&config, tx_tx, block_tx).await {
        Ok(_) => println!("   🌐 P2P Network started."),
        Err(e) => println!("   ⚠️  Network warning: {}", e),
    }

    // 5. Wrap blockchain in Arc<RwLock> for API access
    let shared_bc: SharedBlockchain = Arc::new(RwLock::new(bc));

    // 6. Build API routes
    let app = Router::new()
        .route("/balance/:address", get(get_balance))
        .route("/height", get(get_height))
        .route("/info", get(get_info))
        .with_state(shared_bc.clone());

    // 7. Start API server on port 7476
    let api_handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:7476").await.unwrap();
        println!("   🌐 HTTP API started on http://127.0.0.1:7476");
        axum::serve(listener, app).await.unwrap();
    });

    println!("   ✅ Node is alive and running. Press Ctrl+C to stop.");
    println!("   📊 API Endpoints:");
    println!("      - http://127.0.0.1:7476/balance/Genesis_Alpha_001");
    println!("      - http://127.0.0.1:7476/height");
    println!("      - http://127.0.0.1:7476/info");
    
    // 8. Wait for API server
    api_handle.await.unwrap();
}

// API Handler: Get balance
async fn get_balance(
    State(bc): State<SharedBlockchain>,
    Path(address): Path<String>,
) -> Json<BalanceResponse> {
    let bc = bc.read().await;
    let balance = bc.get_balance(&address).await.unwrap_or(0);
    Json(BalanceResponse {
        address,
        balance,
        balance_hafa: balance as f64 / 100_000_000.0,
    })
}

// API Handler: Get height
async fn get_height(State(bc): State<SharedBlockchain>) -> Json<HeightResponse> {
    let bc = bc.read().await;
    let height = bc.get_chain_height().await;
    Json(HeightResponse { height })
}

// API Handler: Get info
async fn get_info(State(bc): State<SharedBlockchain>) -> Json<InfoResponse> {
    let bc = bc.read().await;
    let height = bc.get_chain_height().await;
    let total_minted = bc.get_total_minted().await;
    Json(InfoResponse {
        version: "1.0.0".to_string(),
        height,
        total_minted,
        total_minted_hafa: total_minted as f64 / 100_000_000.0,
        network: "mainnet".to_string(),
    })
}