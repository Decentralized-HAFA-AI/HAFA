// ============================================================================
// HAFA - src/bin/hafa-miner.rs — REFERENCE MINING CLIENT
// ============================================================================

use hafa::config::Config;
use hafa::crypto::KeyPair;
// ✅ FIX: Added Digest trait import for sha3
use sha3::Digest;
use chrono::Utc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use std::io::{self, Write};

// ============================================================================
// MINING STATS
// ============================================================================

#[derive(Debug, Clone)]
struct MiningStats {
    start_time: Option<Instant>,
    hashes_computed: u64,
    blocks_found: u64,
    estimated_earnings: f64,
    current_difficulty: u32,
    is_mining: bool,
}

impl Default for MiningStats {
    fn default() -> Self {
        Self {
            start_time: None,
            hashes_computed: 0,
            blocks_found: 0,
            estimated_earnings: 0.0,
            current_difficulty: 1,
            is_mining: false,
        }
    }
}

// ============================================================================
// TUI HELPERS (Simple ANSI-based)
// ============================================================================

fn clear_screen() {
    print!("\x1B[2J\x1B[1;1H");
    io::stdout().flush().unwrap();
}

fn print_header() {
    println!("🧠  HAFA MINER — Proof of Useful Cognitive Work");
    println!("   Horizon After Freedom Achieved");
    println!("   {}", "-".repeat(60));
}

fn print_stats(stats: &MiningStats, peer_id: &str) {
    let uptime = if let Some(start) = stats.start_time {
        let secs = start.elapsed().as_secs();
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        "0m 0s".into()
    };

    let hashrate = if let Some(start) = stats.start_time {
        let secs = start.elapsed().as_secs_f64().max(1.0);
        (stats.hashes_computed as f64 / secs) as u64
    } else {
        0
    };

    println!("\n📊 MINING STATS");
    println!("   {}", "-".repeat(40));
    println!("   ⏱️  Uptime:        {}", uptime);
    println!("   ⚡ Hashrate:       {} H/s", hashrate);
    println!("   🔗 Peer ID:       {}...", &peer_id[..20]);
    println!("   🎯 Difficulty:    {}", stats.current_difficulty);
    println!("   🧱 Blocks Found:  {}", stats.blocks_found);
    println!("   💰 Est. Earnings: {:.2} HAFA", stats.estimated_earnings);
    println!("   🔄 Status:        {}", if stats.is_mining { "🟢 MINING" } else { "🔴 IDLE" });
}

fn print_controls() {
    println!("\n🎮 CONTROLS");
    println!("   {}", "-".repeat(40));
    println!("   [Enter] Start/Stop Mining");
    println!("   [q]     Quit");
    println!("   {}", "-".repeat(60));
}

// ============================================================================
// PoUCW SIMULATION (Cognitive Work)
// ============================================================================

/// Simulate a "useful cognitive task": find a nonce such that
/// hash(data || nonce) has at least `difficulty` leading zero bits.
fn perform_cognitive_work(data: &[u8], difficulty: u32) -> Option<(u64, String)> {
    let target = calculate_target(difficulty);
    
    for nonce in 0..u64::MAX {
        // ✅ FIX: Now Digest trait is in scope, so .new() works
        let mut hasher = sha3::Sha3_256::new();
        hasher.update(data);
        hasher.update(nonce.to_le_bytes());
        let hash = hasher.finalize();
        
        // Check if hash meets difficulty
        if &hash[..] <= &target[..] {
            return Some((nonce, hex::encode(hash)));
        }
    }
    None
}

fn calculate_target(difficulty: u32) -> [u8; 32] {
    let mut target = [0xFFu8; 32];
    let zero_bytes = (difficulty / 8) as usize;
    let zero_bits = (difficulty % 8) as usize;
    for i in 0..zero_bytes.min(32) { target[i] = 0x00; }
    if zero_bytes < 32 { target[zero_bytes] = 0xFF << zero_bits; }
    target
}

// ============================================================================
// MINER LOOP
// ============================================================================

fn mining_loop(
    stats: Arc<std::sync::Mutex<MiningStats>>,
    stop_signal: Arc<AtomicBool>,
    _keypair: KeyPair,  // ✅ FIX: Prefix unused param with underscore
    difficulty: u32,
) {
    let mut local_hashes = 0u64;
    let batch_size = 1000u64;
    
    println!("\n   🚀 Mining started... (PoUCW mode)");
    
    while !stop_signal.load(Ordering::Relaxed) {
        // Perform batch of cognitive work
        let task_data = format!("hafa_mining_{}", Utc::now().timestamp()).into_bytes();
        
        for _ in 0..batch_size {
            if stop_signal.load(Ordering::Relaxed) { break; }
            
            // Simulate work (in production: real cognitive task)
            let _ = perform_cognitive_work(&task_data, difficulty);
            local_hashes += 1;
        }
        
        // Update shared stats
        {
            let mut s = stats.lock().unwrap();
            s.hashes_computed += local_hashes;
            
            // Simulate occasional block discovery (for demo)
            if local_hashes % 50_000 == 0 && rand::random::<u8>() < 5 {
                s.blocks_found += 1;
                s.estimated_earnings += 500.0; // 500 HAFA per block
                println!("   🎉 Block found! +500 HAFA");
            }
        }
        local_hashes = 0;
        
        // Small sleep to avoid CPU burn
        thread::sleep(Duration::from_millis(100));
    }
    
    println!("   🛑 Mining stopped.");
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    // ✅ FIX: Prefix unused variable with underscore
    let _config = Config::default();
    
    // Initialize keypair (in production: load from secure storage)
    // For demo: generate ephemeral key (do NOT use in production!)
    let keypair = KeyPair::generate();
    let peer_id = keypair.address().pubkey_hex.clone();
    
    // Mining state
    let stats = Arc::new(std::sync::Mutex::new(MiningStats::default()));
    let stop_signal = Arc::new(AtomicBool::new(false));
    let mut mining_thread: Option<thread::JoinHandle<()>> = None;
    
    // UI loop
    clear_screen();
    print_header();
    print_stats(&stats.lock().unwrap(), &peer_id);
    print_controls();
    
    // Input handling
    let mut input = String::new();
    loop {
        print!("\n   > ");
        io::stdout().flush().unwrap();
        
        input.clear();
        io::stdin().read_line(&mut input).unwrap();
        let cmd = input.trim().to_lowercase();
        
        match cmd.as_str() {
            "" | "start" | "mine" => {
                let mut s = stats.lock().unwrap();
                if !s.is_mining {
                    s.is_mining = true;
                    s.start_time = Some(Instant::now());
                    drop(s);
                    
                    // Spawn mining thread
                    let stats_clone = stats.clone();
                    let stop_clone = stop_signal.clone();
                    let kp_clone = keypair.clone();
                    let diff = 4; // Demo difficulty
                    
                    mining_thread = Some(thread::spawn(move || {
                        mining_loop(stats_clone, stop_clone, kp_clone, diff);
                    }));
                    
                    println!("   ✅ Mining started.");
                } else {
                    println!("   ⚠️  Already mining. Type 'stop' to pause.");
                }
            }
            "stop" | "pause" => {
                let mut s = stats.lock().unwrap();
                if s.is_mining {
                    s.is_mining = false;
                    stop_signal.store(true, Ordering::Relaxed);
                    drop(s);
                    
                    if let Some(handle) = mining_thread.take() {
                        handle.join().unwrap();
                    }
                    stop_signal.store(false, Ordering::Relaxed);
                    println!("   ✅ Mining stopped.");
                } else {
                    println!("   ⚠️  Not mining.");
                }
            }
            "stats" | "status" => {
                let s = stats.lock().unwrap();
                print_stats(&s, &peer_id);
            }
            "q" | "quit" | "exit" => {
                // Graceful shutdown
                stop_signal.store(true, Ordering::Relaxed);
                if let Some(handle) = mining_thread.take() {
                    let _ = handle.join();
                }
                println!("\n   👋 Goodbye! HAFA 🧠✨");
                break;
            }
            "help" => {
                print_controls();
            }
            _ => {
                println!("   ❓ Unknown command. Type 'help' for options.");
            }
        }
        
        // Refresh display
        clear_screen();
        print_header();
        print_stats(&stats.lock().unwrap(), &peer_id);
        print_controls();
    }
}