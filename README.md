# 🧠 HAFA Protocol
**H**orizon **A**fter **F**reedom **A**chieved

> A Decentralized, Self-Evolving AI Protocol. Learning from Scratch.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)
![Status](https://img.shields.io/badge/Status-Genesis-00ff00.svg)
![Version](https://img.shields.io/badge/Version-2.0.0-blue.svg)

---

## 📖 What is Hafa?

**HAFA** stands for **Horizon After Freedom Achieved**. It is a next-generation decentralized network that merges blockchain consensus with artificial intelligence. Unlike traditional AI that relies on centralized servers and pre-trained models, Hafa is built to **learn from scratch** on a distributed peer-to-peer network.

Hafa introduces a new consensus mechanism called **Proof of Useful Cognitive Work (PoUCW)**, where nodes are rewarded for solving meaningful cognitive tasks rather than wasting energy on random hashing. The protocol is designed to be neutral, secure, and community-driven, with a locked core architecture and evolving client applications.

## ✨ Core Features

*   🧠 **AI from Scratch:** No external frameworks (PyTorch, TensorFlow, ONNX). Hafa uses custom `ndarray`-based learning algorithms with multi-layer perceptrons (MLP), Adam optimizer, and custom backpropagation.
*   🔒 **Locked Core Architecture:** Consensus, cryptography, and networking layers are immutable. Evolution happens safely through the `EvolutionProposal` system with DAO voting.
*   ⛏️ **PoUCW Consensus:** Mining requires solving verifiable cognitive tasks with structured `CognitiveProof`, aligning network security with useful computation.
*   🌐 **Decentralized & Neutral:** Runs on any device. No central authority, no API dependencies, no corporate control.
*   🛡️ **Epistemic Filtering:** All data ingested is validated for trust, confidence, grounding, source reputation, and temporal relevance before being learned.
*   🔐 **Advanced Cryptography:** Ed25519 signatures, SHA3-256 hashing, ChaCha20-Poly1305 encryption, zeroize for secure memory handling, and multi-signature support.
*   🌳 **Merkle Tree:** SPV (Simplified Payment Verification) support for light clients.
*   📡 **Event-Driven Architecture:** Real-time event system for blockchain events, model checkpoints, and vesting releases.
*   💰 **Fair Tokenomics:** Fixed supply, transparent vesting, quality-based rewards, and a built-in royalty mechanism.

## 🏗️ Architecture

Hafa strictly separates the **Immutable Core** from **Evolving Clients** to ensure security and flexibility:

```text
┌─────────────────────────────────────────────────┐
│  CLIENTS (Evolving & Modular)                   │
│  • AI Interfaces, Wallets, Mining UI            │
│  • Custom Plugins & Third-Party Integrations    │
│  • Proposed & Validated via EvolutionEngine     │
└───────────────────┬─────────────────────────────┘
                    │ Proposals & Data
┌───────────────────▼─────────────────────────────┐
│  CORE PROTOCOL (Locked & Secure)                │
│  • config.rs        → Genesis, Vesting, Rules   │
│  • crypto.rs        → Ed25519, Multi-sig, AES   │
│  • blockchain.rs    → Ledger, PoUCW, Merkle     │
│  • network.rs       → libp2p P2P, Deduplication │
│  • epistemic.rs     → Trust, Reputation, Decay  │
│  • data_source.rs   → Ingestion, Validation     │
│  • learning.rs      → MLP, Adam, Backprop       │
│  • evolution.rs     → Proposals, Voting, DAO    │
│  • api.rs           → Client Control Interface  │
└─────────────────────────────────────────────────┘
⛏️ How Mining Works (PoUCW)
Unlike traditional Proof of Work that wastes energy on random hashing, Hafa's Proof of Useful Cognitive Work requires miners to perform meaningful cognitive tasks:
Mining Process:
Get Task: Miner requests mining task from node
Cognitive Work: Miner simulates neural network training:
Processes experiences from validated data
Reduces loss through gradient descent
Tracks model state changes
Generate Proof: Miner creates structured CognitiveProof:
{
  "model_hash_before": "...",
  "model_hash_after": "...",
  "loss_before": 1.5,
  "loss_after": 0.8,
  "experiences_processed": 100,
  "avg_confidence": 0.85,
  "resources_used": {
    "cpu_percent": 45.0,
    "ram_mb": 2048,
    "gpu_percent": 60.0
  },
  "training_duration_ms": 5000
}
Mine Block: Find nonce that satisfies difficulty target
Submit Solution: Send proof + nonce to node
Quality-Based Reward: Reward adjusted by proof quality score (0.5x to 1.0x)
Quality Score Calculation:
quality_score = (loss_reduction * 0.5) + (experience_factor * 0.3) + (confidence_factor * 0.2)
🔐 Cryptographic Features
Multi-Signature Support
Enable DAO governance with multi-sig wallets:
let config = MultiSigConfig::new(vec![pubkey1, pubkey2, pubkey3], 2)?; // 2-of-3
let mut multisig = MultiSig::new(message);
multisig.add_signature(&kp1, message);
multisig.add_signature(&kp2, message);
config.verify(message, &multisig)?;
Encrypted Key Storage
Secure private keys with ChaCha20-Poly1305:
let encrypted = keypair.encrypt_secret("my-passphrase")?;
let decrypted = KeyPair::decrypt_secret(&encrypted, "my-passphrase")?;
Checksum Addresses
Prevent typos with Ethereum-style checksums:
let addr = keypair.address(); // Includes checksum
let addr_str = addr.to_string_with_checksum(); // "pubkey:checksum"
Zeroize
Automatic secure memory cleanup:
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct KeyPair { ... }
🌳 Merkle Tree & SPV
Light clients can verify transactions without downloading the entire blockchain:
let root = MerkleTree::root(&tx_ids);
let proof = MerkleTree::proof(&tx_ids, target_index)?;
let valid = MerkleTree::verify_proof(leaf, &proof, &root);
📡 Event System
Subscribe to real-time blockchain events:
let mut receiver = blockchain.subscribe();
tokio::spawn(async move {
    while let Ok(event) = receiver.recv().await {
        match event {
            BlockchainEvent::NewBlock { height, hash, reward } => { ... }
            BlockchainEvent::ModelCheckpoint { height, model_hash } => { ... }
            BlockchainEvent::VestingReleased { amount, remaining } => { ... }
        }
    }
});
💰 Tokenomics & Economics
Total Supply: 210,000,000 HAFA (Fixed, Non-Inflationary)
Precision: 8 Decimals (1 HAFA = 100,000,000 Satoshis)
Initial Block Reward: 500 HAFA
Halving Interval: Every 210,000 Blocks (~4 Years)
Difficulty Adjustment: Every 2,016 Blocks (Bitcoin-style)
Target Block Time: 10 minutes
📊 Distribution Model
Allocation
Percentage
Amount
Purpose
Mining Rewards
95%
199.5M
Distributed to nodes via PoUCW
Founder Genesis
5%
10.5M
Protocol development & infrastructure
🔐 Founder Vesting Schedule
To guarantee long-term commitment and prevent market dumping, the founder's 5% share is strictly time-locked:
Year 0 (Launch): 10% Unlocked (1.05M HAFA)
Year 1: +30% Unlocked (3.15M HAFA)
Year 2: +30% Unlocked (3.15M HAFA)
Year 3 (Halving): +30% Unlocked (3.15M HAFA)
Enforced directly in config.rs and blockchain.rs.
💸 Client Royalty Mechanism
2% Royalty: Automatically deducted from revenue generated by paid/official clients
Purpose: Funds ongoing protocol maintenance, security audits, and core development
Transparency: All royalty transactions are recorded on-chain and publicly verifiable
🎯 Quality-Based Rewards
Miners are rewarded based on the quality of their cognitive work:
Base Reward: 500 HAFA (halves every 210,000 blocks)
Quality Multiplier: 0.5x to 1.0x based on proof quality
Founder Royalty: 2% of reward goes to founder address
🛠️ Technical Stack
Language: Rust (1.70+)
Concurrency: Tokio (Async Runtime)
Networking: libp2p (TCP, GossipSub, Kademlia, mDNS)
Cryptography: ed25519-dalek, SHA3-256, ChaCha20-Poly1305, zeroize
Math/AI: ndarray, ndarray-rand, custom backpropagation, Adam optimizer
Serialization: Serde, bincode, TOML/JSON
HTTP API: Axum, Tower-HTTP
Concurrency: DashMap (concurrent HashMap)
🌐 HTTP API
The genesis node exposes a REST API on port 7476:
Endpoints:
# Get node info
GET /info

# Get blockchain height
GET /height

# Get balance for address
GET /balance/:address

# Get mining task
GET /task

# Submit mined block
POST /submit
{
  "miner_addr": "...",
  "nonce": 12345,
  "cognitive_proof": { ... },
  "model_checkpoint": { ... }
}
🚀 Getting Started
Prerequisites
Ensure you have Rust and Cargo installed:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
Build & Run
# Clone the repository
git clone https://github.com/Decentralized-HAFA-AI/hafa.git
cd hafa

# Build in release mode
cargo build --release

# Run the genesis node
cargo run --release

# In another terminal, run the miner
cargo run --release --bin hafa-miner
Configuration
Default settings are managed via config.rs. You can customize:
genesis_pubkey_hex: Your founder public key
network.p2p_port: P2P listening port (Default: 7474)
learning.allow_internet_learning: Toggle external data ingestion
mining.cognitive_worker_threads: CPU threads for PoUCW
🗺️ Roadmap
✅ Phase 1: Genesis Core (COMPLETED)
Cryptography (Ed25519, SHA3-256)
Blockchain (Bitcoin-style consensus)
Network (libp2p P2P)
Configuration (Genesis, Vesting)
HTTP API
✅ Phase 2: Cognition (COMPLETED)
Epistemic Engine (Trust & Validation)
Learning from Scratch (MLP, Adam)
Data Sources (Local, IPFS, Web, RSS, Sensor)
Source Reputation System
Temporal Decay
✅ Phase 3: Evolution (COMPLETED)
Proposal System
Sandbox Validation
DAO Voting Mechanism
Multi-Signature Support
Merkle Tree & SPV
Event System
Encrypted Key Storage
Checksum Addresses
🔄 Phase 4: Ecosystem (IN PROGRESS)
Public Client Marketplace
Mainnet Launch
Governance DAO
Mobile Wallets
Hardware Wallet Integration
🔮 Phase 5: Advanced Features (PLANNED)
Zero-Knowledge Proofs
Sharding
Cross-Chain Bridges
AI Model Marketplace
Decentralized Storage (IPFS integration)
🤝 Contributing
Hafa is open-source and thrives on community collaboration. We welcome contributions in:
Client development (UI/UX, Mobile, Web)
Optimization of learning algorithms & PoUCW tasks
Security audits & formal verification
Documentation & translations
Testing & bug reports
Please read our CONTRIBUTING.md (coming soon) and open a Pull Request.
📜 License & Disclaimer
This project is licensed under the MIT License. See the LICENSE file for details.
⚠️ Disclaimer: Hafa is experimental, decentralized infrastructure. It does not endorse, enable, or control specific use cases. Client developers and end-users are solely responsible for their actions and compliance with local laws. The protocol is neutral technology, similar to the internet or TCP/IP. Use at your own risk.
Powered by Rust 🦀 | Built for Decentralization 🌍 | Learning from Scratch 🧠