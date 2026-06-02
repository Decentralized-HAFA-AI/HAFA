// ============================================================================
// HAFA - src/bin/hafa-miner.rs — CONNECTED MINING CLIENT
// ============================================================================

use hafa::crypto::{KeyPair, hash_sha3_256};
use std::io::{self, Write, BufRead, BufReader};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

fn clear_screen() { print!("\x1B[2J\x1B[1;1H"); io::stdout().flush().unwrap(); }

#[derive(Clone)]
struct Stats {
    hashrate: u64,
    blocks: u64,
    earnings: f64,
    status: String,
}

fn main() {
    let keypair = KeyPair::generate();
    let peer_id = keypair.address().pubkey_hex.clone();
    let stats = Arc::new(std::sync::Mutex::new(Stats { hashrate: 0, blocks: 0, earnings: 0.0, status: "IDLE".into() }));
    let stop = Arc::new(AtomicBool::new(false));

    clear_screen();
    println!("🧠 HAFA MINER (Connected Mode)");
    println!("   Peer ID: {}...", &peer_id[..16]);
    println!("   Status: Connecting to Node...");

    loop {
        // Try to connect
        if let Ok(mut stream) = TcpStream::connect("127.0.0.1:7475") {
            stream.set_read_timeout(Some(Duration::from_secs(300))).unwrap();
            println!("   ✅ Connected to Node! Waiting for task...");
            stats.lock().unwrap().status = "CONNECTED".into();
            
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            
            if reader.read_line(&mut line).is_ok() {
                if line.starts_with("TASK ") {
                    let parts: Vec<&str> = line.trim().split_whitespace().collect();
                    if parts.len() == 3 {
                        let last_hash = parts[1];
                        let difficulty: u32 = parts[2].parse().unwrap_or(1);
                        println!("   🎯 Task received. Difficulty: {}", difficulty);
                        stats.lock().unwrap().status = "MINING".into();
                        
                        // Mining Loop
                        let start = Instant::now();
                        let mut hashes = 0u64;
                        let mut found = false;
                        
                        // Simple PoUCW: find nonce where hash(last_hash + nonce) has leading zeros
                        let target = calculate_target(difficulty);
                        for nonce in 0..u64::MAX {
                            if stop.load(Ordering::Relaxed) { break; }
                            
                            let data = format!("{}{}", last_hash, nonce);
                            let hash = hash_sha3_256(data.as_bytes());
                            hashes += 1;
                            
                            // Check target (simplified check)
                            let hash_bytes = hex::decode(&hash).unwrap_or_default();
                            if hash_bytes.len() >= target.len() && &hash_bytes[..target.len()] <= &target[..] {
                                // Found!
                                let mut writer = stream.try_clone().unwrap();
                                let msg = format!("SOLVE {}\n", nonce);
                                writer.write_all(msg.as_bytes()).unwrap();
                                
                                let mut resp = String::new();
                                if reader.read_line(&mut resp).is_ok() && resp.starts_with("OK ") {
                                    let reward: f64 = resp[3..].trim().parse().unwrap_or(0.0);
                                    let mut s = stats.lock().unwrap();
                                    s.blocks += 1;
                                    s.earnings += reward;
                                    s.hashrate = hashes / start.elapsed().as_secs().max(1);
                                    println!("    BLOCK FOUND! +{} HAFA", reward);
                                } else {
                                    println!("   ❌ Proof rejected by node.");
                                }
                                found = true;
                                break;
                            }
                            
                            // Update hashrate occasionally
                            if hashes % 10000 == 0 {
                                stats.lock().unwrap().hashrate = hashes / start.elapsed().as_secs().max(1);
                            }
                        }
                        if !found { stats.lock().unwrap().status = "IDLE".into(); }
                    }
                }
            }
        } else {
            println!("   ⚠️ Node not found. Retrying in 5s...");
            thread::sleep(Duration::from_secs(5));
        }
        
        // UI Refresh
        clear_screen();
        let s = stats.lock().unwrap();
        println!("🧠 HAFA MINER (Connected Mode)");
        println!("   Peer ID: {}...", &peer_id[..16]);
        println!("   ⚡ Hashrate: {} H/s", s.hashrate);
        println!("   🧱 Blocks: {}", s.blocks);
        println!("   💰 Earnings: {:.2} HAFA", s.earnings);
        println!("   🔄 Status: {}", s.status);
        println!("\n   [q] Quit");
        
        // Check for quit
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        if input.trim().to_lowercase() == "q" { break; }
    }
}

fn calculate_target(difficulty: u32) -> Vec<u8> {
    let mut target = vec![0xFFu8; 32];
    let zero_bytes = (difficulty / 8) as usize;
    let zero_bits = (difficulty % 8) as usize;
    for i in 0..zero_bytes.min(32) { target[i] = 0x00; }
    if zero_bytes < 32 { target[zero_bytes] = 0xFF << zero_bits; }
    target
}