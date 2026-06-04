// ============================================================================
// HAFA - src/bin/hafa-miner.rs — CONNECTED MINING CLIENT (UPGRADED)
// ============================================================================
//
// Connected miner that performs simulated cognitive work (PoUCW)
// and submits structured CognitiveProof to the node.
//
// Features:
// - Simulated cognitive work (learning training)
// - Real CognitiveProof generation
// - Model checkpoint reporting
// - Quality-based reward tracking
// - Hashrate monitoring
//
// ============================================================================

use hafa::blockchain::Block;
use hafa::crypto::hash_sha3_256;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ============================================================================
// API STRUCTURES (Matching main.rs)
// ============================================================================

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
    cognitive_proof: CognitiveProofReq,
    model_checkpoint: Option<ModelCheckpointReq>,
}

#[derive(Serialize)]
struct CognitiveProofReq {
    model_hash_before: String,
    model_hash_after: String,
    loss_before: f64,
    loss_after: f64,
    experiences_processed: u32,
    avg_confidence: f64,
    resources_used: ResourceUsageReq,
    training_duration_ms: u64,
}

#[derive(Serialize)]
struct ResourceUsageReq {
    cpu_percent: f64,
    ram_mb: u64,
    gpu_percent: f64,
    gpu_memory_mb: u64,
}

#[derive(Serialize)]
struct ModelCheckpointReq {
    model_hash: String,
    total_parameters: u64,
    architecture: String,
}

#[derive(Deserialize)]
struct SubmitResp {
    success: bool,
    block_index: Option<u64>,
    reward: u64,
    reward_hafa: f64,
    quality_score: f64,
    message: String,
}

// ============================================================================
// COGNITIVE WORK SIMULATION
// ============================================================================

/// Simulated cognitive work result
struct CognitiveWorkResult {
    model_hash_before: String,
    model_hash_after: String,
    loss_before: f64,
    loss_after: f64,
    experiences_processed: u32,
    avg_confidence: f64,
    training_duration_ms: u64,
}

/// Simulate cognitive work (learning training)
/// In production: this would actually train the neural network
fn simulate_cognitive_work(difficulty: u32) -> CognitiveWorkResult {
    let start = Instant::now();

    // Simulate model hash before training
    let model_hash_before = hash_sha3_256(format!("model_before_{}", difficulty).as_bytes());

    // Simulate training: more difficult blocks = more training
    let experiences_processed = 50 + (difficulty as u32 * 10);
    let training_iterations = 10 + (difficulty as u32 * 2);

    // Simulate loss reduction (better training = lower loss)
    let loss_before = 1.0 + (difficulty as f64 * 0.1);
    let mut loss = loss_before;

    // Simulate training iterations
    for i in 0..training_iterations {
        // Simulate gradient descent: loss decreases over time
        let learning_rate = 0.05;
        let gradient = loss * 0.1; // Simple gradient
        loss -= learning_rate * gradient;
        loss = loss.max(0.01); // Prevent negative loss

        // Simulate some computation
        let _ = (0..1000).map(|x| x * x).sum::<i64>();
    }

    let loss_after = loss;

    // Simulate model hash after training
    let model_hash_after = hash_sha3_256(format!("model_after_{}", loss_after).as_bytes());

    // Simulate average confidence (higher for easier blocks)
    let avg_confidence = 0.7 + (1.0 / (1.0 + difficulty as f64)) * 0.25;

    let training_duration_ms = start.elapsed().as_millis() as u64;

    CognitiveWorkResult {
        model_hash_before,
        model_hash_after,
        loss_before,
        loss_after,
        experiences_processed,
        avg_confidence,
        training_duration_ms,
    }
}

/// Get simulated resource usage
fn get_resource_usage() -> ResourceUsageReq {
    // In production: read actual system metrics
    ResourceUsageReq {
        cpu_percent: 45.0 + (rand::random::<f64>() * 20.0),
        ram_mb: 1024 + (rand::random::<u64>() % 512),
        gpu_percent: if rand::random::<f64>() > 0.5 {
            60.0 + (rand::random::<f64>() * 30.0)
        } else {
            0.0
        },
        gpu_memory_mb: if rand::random::<f64>() > 0.5 {
            2048 + (rand::random::<u64>() % 2048)
        } else {
            0
        },
    }
}

// ============================================================================
// MAIN
// ============================================================================

#[tokio::main]
async fn main() {
    let client = reqwest::Client::new();
    let node_url = "http://127.0.0.1:7476";
    let miner_addr = "Miner_001";

    println!("🧠 HAFA Connected Miner Started (PoUCW)");
    println!("   Node: {}", node_url);
    println!("   Address: {}", miner_addr);
    println!("   Mode: Proof of Useful Cognitive Work\n");

    let mut blocks_mined = 0u64;
    let mut total_reward = 0u64;

    loop {
        // 1. Get mining task from node
        let task: TaskResp = match client.get(format!("{}/task", node_url)).send().await {
            Ok(resp) => match resp.json::<TaskResp>().await {
                Ok(t) => t,
                Err(_) => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    continue;
                }
            },
            Err(_) => {
                println!("   ⚠️  Node not reachable. Retrying...");
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        println!(
            "📥 Task: Height #{}, Difficulty: {}",
            task.target_height, task.difficulty
        );

        // 2. Perform cognitive work (simulated learning)
        println!("   🧠 Performing cognitive work...");
        let cognitive_work = simulate_cognitive_work(task.difficulty);
        println!(
            "   ✅ Cognitive work done: loss {:.3} → {:.3}, {} experiences",
            cognitive_work.loss_before,
            cognitive_work.loss_after,
            cognitive_work.experiences_processed
        );

        // 3. Mine the block (find nonce)
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

            if hash_bytes.len() >= target.len() && &hash_bytes[..target.len()] <= &target[..] {
                // 4. Build CognitiveProof
                let cognitive_proof = CognitiveProofReq {
                    model_hash_before: cognitive_work.model_hash_before.clone(),
                    model_hash_after: cognitive_work.model_hash_after.clone(),
                    loss_before: cognitive_work.loss_before,
                    loss_after: cognitive_work.loss_after,
                    experiences_processed: cognitive_work.experiences_processed,
                    avg_confidence: cognitive_work.avg_confidence,
                    resources_used: get_resource_usage(),
                    training_duration_ms: cognitive_work.training_duration_ms,
                };

                // 5. Build ModelCheckpoint (optional)
                let model_checkpoint = Some(ModelCheckpointReq {
                    model_hash: cognitive_work.model_hash_after.clone(),
                    total_parameters: 50000, // MLP 128->256->128->64
                    architecture: "MLP-128-256-128-64".to_string(),
                });

                // 6. Submit solution
                let req = SubmitReq {
                    miner_addr: miner_addr.to_string(),
                    nonce,
                    cognitive_proof,
                    model_checkpoint,
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
                                blocks_mined += 1;
                                total_reward += result.reward;

                                println!("✅ Block #{} mined!", result.block_index.unwrap_or(0));
                                println!(
                                    "   💰 Reward: {:.8} HAFA (Quality: {:.2})",
                                    result.reward_hafa, result.quality_score
                                );
                                println!(
                                    "   📊 Stats: {} blocks mined, {:.8} HAFA total\n",
                                    blocks_mined,
                                    total_reward as f64 / 100_000_000.0
                                );
                            } else {
                                println!("❌ Rejected: {}\n", result.message);
                            }
                        }
                        Err(_) => println!("⚠️  Parse error\n"),
                    },
                    Err(e) => println!("⚠️  Submit failed: {}\n", e),
                }

                break;
            }

            nonce += 1;

            // Progress report
            if nonce % 50_000 == 0 {
                let elapsed = start.elapsed().as_secs_f64();

                if elapsed > 0.0 {
                    let hashrate = nonce as f64 / elapsed;

                    println!(
                        "   ⚡  Hashing... nonce={}, hashrate={:.0} H/s",
                        nonce, hashrate
                    );
                }
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
}