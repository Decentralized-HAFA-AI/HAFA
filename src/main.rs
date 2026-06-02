// ============================================================================
// HAFA - src/main.rs — PERSISTENT GENESIS NODE
// ============================================================================

use hafa::config::Config;
use hafa::blockchain::Blockchain;
use hafa::network::NetworkEngine;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    println!("🚀 HAFA Genesis Node Starting...");

    // 1. Load Config
    let config = Config::default();
    println!("   ⚙️  Configuration loaded.");

    // 2. Initialize Blockchain (auto-loads from disk if exists)
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

    println!("   ✅ Node is alive and running. Press Ctrl+C to stop.");
    
    // 5. Main Loop
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
    }
}