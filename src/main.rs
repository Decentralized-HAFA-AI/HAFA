use hafa::config::Config;
use hafa::blockchain::Blockchain;
use hafa::network::NetworkEngine;
use tokio::sync::mpsc;
use tracing::{info, warn, error};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    // 1. راه‌اندازی لاگر با سطح پیش‌فرض INFO
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_new("info").unwrap_or_default())
        .init();

    info!("🚀 HAFA Genesis Node Starting...");

    // 2. بارگذاری پیکربندی و ساخت پوشه‌ی داده
    let mut config = Config::default();
    let data_dir = config.storage.data_dir.clone();
    if !data_dir.exists() {
        std::fs::create_dir_all(&data_dir).expect("❌ Failed to create data directory");
    }
    info!("📄 Configuration loaded. Founder Key: {}...", &config.founder.genesis_pubkey_hex[..16]);

    // 3. ایجاد کانال‌های ارتباطی
    let (tx_sender, _) = mpsc::channel::<hafa::blockchain::Transaction>(1000);
    let (block_sender, _) = mpsc::channel::<hafa::blockchain::Block>(100);

    // 4. راه‌اندازی بلاکچین
    info!("⛓️ Initializing Blockchain Ledger...");
    let blockchain = match Blockchain::new(&config).await {
        Ok(bc) => {
            info!("✅ Blockchain initialized. Height: {}, Minted: {}", 
                  bc.get_chain_height().await, bc.get_total_minted().await);
            bc
        }
        Err(e) => {
            error!("❌ Blockchain init failed: {}. Node cannot run.", e);
            std::process::exit(1);
        }
    };

    // 5. راه‌اندازی شبکه (با استفاده از Option برای مدیریت خطا)
    info!("🌐 Initializing P2P Network...");
    let network: Option<NetworkEngine> = match NetworkEngine::new(&config, tx_sender, block_sender).await {
        Ok(net) => {
            if let Err(e) = net.start().await {
                warn!("⚠️ Network start failed: {}. Running in offline mode.", e);
            } else {
                info!("✅ Network started. PeerID: {}", net.get_local_peer_id());
            }
            Some(net)
        }
        Err(e) => {
            warn!("⚠️ Network init failed: {}. Node will run offline.", e);
            None
        }
    };

    // 6. حلقه‌ی اصلی نود
    info!("🔄 Node is alive. Press Ctrl+C to stop.");
    let mut uptime = 0u64;
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
        uptime += 10;
        
        if uptime % 60 == 0 {
            let peer_count = if let Some(ref net) = network {
                net.get_peer_count().await
            } else {
                0
            };
            
            info!("💓 Heartbeat | Uptime: {}s | Peers: {} | Chain: {}", 
                  uptime, 
                  peer_count, 
                  blockchain.get_chain_height().await);
        }
    }
}