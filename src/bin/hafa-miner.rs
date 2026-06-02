// ============================================================================
// HAFA - src/bin/hafa-miner.rs — CONNECTED MINING CLIENT
// ============================================================================

use hafa::blockchain::Block;
use hafa::crypto::hash_sha3_256;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Deserialize)]
struct TaskResp {
    last_hash: String,
    difficulty: u32,
    target_height: u64,
}

#[derive(Serialize)]
struct SubmitReq {
    miner_addr: String,
    nonce: u64,
    cognitive_proof: String,
}

#[derive(Deserialize)]
struct SubmitResp {
    success: bool,
    block_index: Option<u64>,
    reward: u64,
    message: String,
}

#[tokio::main]
async fn main() {
    let client = reqwest::Client::new();
    let node_url = "http://127.0.0.1:7476";
    let miner_addr = "Miner_001";

    println!("🧠 HAFA Connected Miner Started");
    println!("   Node: {}", node_url);
    println!("   Address: {}", miner_addr);

    loop {
        let task: TaskResp = match client.get(format!("{}/task", node_url)).send().await {
            Ok(resp) => match resp.json::<TaskResp>().await {
                Ok(t) => t,
                Err(_) => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    continue;
                }
            },
            Err(_) => {
                println!("   ⚠️ Node not reachable. Retrying...");
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        println!(
            "📥 Task: Height #{}, Difficulty: {}",
            task.target_height,
            task.difficulty
        );

        let start = Instant::now();
        let mut nonce = 0u64;
        let target = Block::calculate_target(task.difficulty);

        loop {
            let data = format!("{}{}", task.last_hash, nonce);
            let hash = hash_sha3_256(data.as_bytes());

            let hash_bytes = match hex::decode(&hash) {
                Ok(v) => v,
                Err(_) => {
                    nonce += 1;
                    continue;
                }
            };

            if hash_bytes.len() >= target.len()
                && &hash_bytes[..target.len()] <= &target[..]
            {
                let req = SubmitReq {
                    miner_addr: miner_addr.to_string(),
                    nonce,
                    cognitive_proof: "PoUCW_Real_Proof".to_string(),
                };

                match client
                    .post(format!("{}/submit", node_url))
                    .json(&req)
                    .send()
                    .await
                {
                    Ok(resp) => match resp.json::<SubmitResp>().await {
                        Ok(result) => {
                            if result.success {
                                println!(
                                    "✅ Block #{} mined! Reward: {} HAFA\n",
                                    result.block_index.unwrap_or(0),
                                    result.reward as f64 / 100_000_000.0
                                );
                            } else {
                                println!("❌ Rejected: {}\n", result.message);
                            }
                        }
                        Err(_) => println!("⚠️ Parse error\n"),
                    },
                    Err(_) => println!("⚠️ Submit failed\n"),
                }

                break;
            }

            nonce += 1;

            if nonce % 50_000 == 0 {
                let elapsed = start.elapsed().as_secs_f64();

                if elapsed > 0.0 {
                    let hashrate = nonce as f64 / elapsed;

                    println!(
                        "   ⚡ Hashing... nonce={}, hashrate={:.0} H/s",
                        nonce,
                        hashrate
                    );
                }
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
}