# 🧠 HAFA Protocol v3.0

**H**orizon **A**fter **F**reedom **A**chieved

> A Decentralized, Self-Evolving AI Protocol. Learning from Scratch.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)
![Status](https://img.shields.io/badge/status-production--grade-green.svg)
![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)
![Tests](https://img.shields.io/badge/tests-166%20passed-brightgreen.svg)

---

## 📖 What is HAFA?

**HAFA** stands for **Horizon After Freedom Achieved**. It is a next-generation decentralized network that merges blockchain consensus with native artificial intelligence. Unlike traditional AI that relies on centralized servers and pre-trained models, HAFA is built to **learn from scratch** on a distributed peer-to-peer network.

HAFA introduces a novel consensus mechanism called **Proof of Useful Cognitive Work (PoUCW)**, where nodes are rewarded for solving meaningful cognitive tasks rather than wasting energy on random hashing. The protocol is designed to be neutral, secure, and community-driven, with a locked core architecture and evolving client applications.

---

## ✨ Core Features

### 🧠 Native AI Engine (No External Frameworks)
- **AI from Scratch:** No PyTorch, TensorFlow, or ONNX. Custom `ndarray`-based learning with multi-layer perceptrons (MLP), Transformer v3/v4, AdamW optimizer, and custom backpropagation
- **Auto-Learning Engine:** Self-evolving AI that learns autonomously from blockchain, P2P network, and user data
- **Knowledge Graph:** Structured long-term memory with NLP extraction and reasoning engine
- **Episodic Memory:** Learning from experience with episode tracking and quality scoring
- **Curiosity Module:** Novelty detection and exploration-driven learning

### ⛏️ Proof of Useful Cognitive Work (PoUCW)
- **Quality-Based Rewards:** Miners rewarded based on cognitive work quality (loss reduction, experience processing, confidence)
- **Verifiable Proofs:** Structured `CognitiveProofV4` with model hashes, gradient commitments, and resource usage
- **Model Checkpoints:** On-chain storage of model state for transparency and reproducibility

### 🔐 Advanced Cryptography
- **Ed25519 Signatures:** Post-quantum resistant digital signatures
- **SHA3-256 Hashing:** Secure hash function for blocks, Merkle trees, addresses
- **ChaCha20-Poly1305:** Authenticated encryption for wallet storage
- **Multi-Signature Support:** DAO governance and enhanced security
- **Checksum Addresses:** Ethereum-style checksum addresses prevent typos
- **Zeroize:** Automatic secure memory cleanup

### 🌐 Decentralized Learning Network
- **P2P Learning:** libp2p GossipSub for real-time knowledge sharing
- **Federated Learning:** HTTP-based sample sharing between nodes
- **Blockchain Meta-Learning:** AI learns from consensus data
- **Epistemic Filtering:** All data validated for trust, confidence, grounding, and reputation

### 🎮 GPU Acceleration
- **WGPU Backend:** Hardware acceleration for AI computations
- **CPU Fallback:** Automatic fallback to optimized CPU backend
- **Benchmark Suite:** Comprehensive performance testing

### 💼 Wallet System
- **Secure Storage:** Ed25519 keys encrypted with ChaCha20-Poly1305
- **Passphrase-Based:** Deterministic wallet generation from passphrase
- **Transaction Signing:** Sign and verify transactions
- **Balance Tracking:** Real-time balance queries

### 🎨 Web UI Dashboard
- **Real-Time Monitoring:** Live blockchain, AI, and network stats
- **Wallet Management:** Create, import, and manage wallets
- **Mining Controls:** Start/stop mining with hash rate display
- **Knowledge Graph Visualization:** View entities and relations

### 📊 Production-Grade Quality
- **166 Unit Tests:** 100% pass rate
- **51+ API Endpoints:** Comprehensive REST API
- **Zero Warnings:** Clean compilation
- **Comprehensive Documentation:** Inline docs and examples

---

## 🏗️ Architecture

HAFA strictly separates the **Immutable Core** from **Evolving Clients**:

```text
┌─────────────────────────────────────────────────────────────┐
│ CLIENTS (Evolving & Modular)                                │
│ • Web UI Dashboard, Mobile Wallets, Mining UI               │
│ • Custom Plugins & Third-Party Integrations                 │
│ • Proposed & Validated via EvolutionEngine                  │
└───────────────────┬─────────────────────────────────────────┘
                    │ Proposals & Data
┌───────────────────▼─────────────────────────────────────────┐
│ CORE PROTOCOL (Locked & Secure)                             │
│ • config.rs → Genesis, Vesting, Rules                       │
│ • crypto.rs → Ed25519, Multi-sig, ChaCha20                  │
│ • blockchain.rs → Ledger, PoUCW, Merkle                     │
│ • network.rs → libp2p P2P, GossipSub                        │
│ • wallet.rs → Wallet Management, Signing                    │
│ • epistemic.rs → Trust, Reputation, Decay                   │
│ • data_source.rs → Ingestion, Validation                    │
│ • learning.rs → MLP, Adam, Backprop                         │
│ • learning_v3/ → Transformer v3/v4, Auto-Learning, KG       │
│ • evolution.rs → Proposals, Voting, DAO                     │
│ • api.rs → 51+ REST Endpoints                               │
└─────────────────────────────────────────────────────────────┘
🚀 Getting Started
Prerequisites
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version  # Should be 1.70+
cargo --version
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
Access Web UI
Open your browser to:
http://127.0.0.1:7476/web
You'll see the real-time dashboard with:
Blockchain stats (height, minted, reward)
AI model status (parameters, buffer, learning)
Wallet management
Mining controls
Knowledge graph stats
P2P network info
🌐 HTTP API (51+ Endpoints)
Blockchain
GET  /info                    # Node information
GET  /height                  # Blockchain height
GET  /balance/:address        # Account balance
GET  /task                    # Mining task
POST /submit                  # Submit mined block
Wallet Management
POST /wallet/create           # Create new wallet
POST /wallet/import           # Import from passphrase
GET  /wallet/list             # List all wallets
GET  /wallet/:addr/info       # Wallet info + balance
POST /wallet/:addr/sign       # Sign transaction
POST /wallet/:addr/delete     # Delete wallet
AI Learning
POST /train-text-v4           # Train Transformer v4
POST /auto-learn/feed         # Feed sample to auto-learning
POST /auto-learn/trigger      # Trigger learning cycle
GET  /auto-learn/status       # Auto-learning status
GET  /auto-learn/stats        # Learning statistics
Knowledge Graph

GET  /knowledge/entities      # List all entities
GET  /knowledge/relations     # List all relations
GET  /knowledge/stats         # KG statistics
POST /knowledge/entity        # Add entity
POST /knowledge/relation      # Add relation
POST /knowledge/extract       # Extract from text
POST /knowledge/query         # Query knowledge graph
P2P Network
GET  /p2p/info                # P2P network info
POST /p2p/connect             # Connect to peer
POST /federated/share         # Share sample
GET  /federated/poll          # Poll samples
GET  /federated/stats         # Federated stats
GPU Backend
GET  /gpu/info                # GPU backend info
POST /debug/benchmark-backend # Run benchmarks
💼 Wallet System
Create Wallet
curl -X POST http://127.0.0.1:7476/wallet/create \
  -H "Content-Type: application/json" \
  -d '{
    "passphrase": "my-secret-passphrase",
    "label": "My First Wallet"
  }'
  Response:
  {
  "success": true,
  "address": "560d2b1d8a70010b4a65a4e05bcfe4efe0a73b713de1b98c8e20d5c02f6ec43b:c373f92d",
  "label": "My First Wallet",
  "message": "Wallet created successfully"
}
Check Balance
curl http://127.0.0.1:7476/wallet/560d2b1d...:c373f92d/info
🧠 Auto-Learning Engine
HAFA's AI learns autonomously from multiple sources:
1. Blockchain Data
POST /auto-learn/poll-blockchain
AI learns from consensus data, block patterns, and transaction metadata.
2. P2P Network
POST /federated/share
{
  "text": "HAFA is revolutionary",
  "source": "web_ui",
  "confidence": 0.95
}
Knowledge shared across the network via GossipSub.
3. Direct Feeding
POST /auto-learn/feed
{
  "text": "Decentralized AI is the future",
  "source": "user",
  "confidence": 0.9
}
Auto-Learning Cycle
The engine automatically triggers learning when:
Buffer size ≥ threshold (default: 100 samples)
Time since last cycle ≥ cooldown (default: 60s)
Quality conditions met
🎮 GPU Acceleration
HAFA supports GPU acceleration via WGPU:
curl http://127.0.0.1:7476/gpu/info
Response:
{
  "success": true,
  "backend": "WGPU",
  "device_name": "NVIDIA RTX 3080",
  "device_type": "DiscreteGpu",
  "memory_mb": 10240,
  "compute_units": 80,
  "supports_fp16": true
}
If GPU not available, automatically falls back to optimized CPU backend.
🔐 Cryptographic Features
Multi-Signature Support
let config = MultiSigConfig::new(vec![pubkey1, pubkey2, pubkey3], 2)?; // 2-of-3
let mut multisig = MultiSig::new(message);
multisig.add_signature(&kp1, message);
multisig.add_signature(&kp2, message);
config.verify(message, &multisig)?;
Encrypted Wallet Storage
let encrypted = keypair.encrypt_secret("my-passphrase")?;
let decrypted = KeyPair::decrypt_secret(&encrypted, "my-passphrase")?;
Checksum Addresses
let addr = keypair.address(); // Includes checksum
let addr_str = addr.to_string_with_checksum(); // "pubkey:checksum"
💰 Tokenomics
Property
Value
Total Supply
210,000,000 HAFA (Fixed)
Precision
8 Decimals
Initial Block Reward
500 HAFA
Halving Interval
210,000 Blocks (~4 Years)
Difficulty Adjustment
Every 2,016 Blocks
Target Block Time
10 Minutes
Distribution
Allocation
Percentage
Amount
Purpose
Mining Rewards
95%
199.5M
PoUCW distribution
Founder Genesis
5%
10.5M
Protocol development
Founder Vesting
Year 0: 10% (1.05M HAFA)
Year 1: +30% (3.15M HAFA)
Year 2: +30% (3.15M HAFA)
Year 3: +30% (3.15M HAFA)
Quality-Based Rewards
quality_score = (loss_reduction × 0.5) + (experience_factor × 0.3) + (confidence_factor × 0.2)
final_reward = base_reward × (0.5 + quality_score × 0.5)
🗺️ Roadmap
✅ Phase 1: Genesis Core (COMPLETED)
Cryptography (Ed25519, SHA3-256, ChaCha20)
Blockchain (PoUCW, Merkle Tree)
Network (libp2p P2P)
Configuration (Genesis, Vesting)
HTTP API
✅ Phase 2: Cognition (COMPLETED)
Epistemic Engine (Trust & Validation)
Learning from Scratch (MLP, Adam)
Data Sources (Local, IPFS, Web, RSS)
Source Reputation System
✅ Phase 3: Evolution (COMPLETED)
Proposal System & DAO Voting
Multi-Signature Support
Event System
Encrypted Key Storage
✅ Phase 4: Advanced AI (COMPLETED - v3.0)
Transformer v3 & v4
Auto-Learning Engine
Knowledge Graph + Reasoning
P2P Learning Network
Federated Learning
GPU Backend (WGPU)
Wallet System
Web UI Dashboard
Episodic Memory
Curiosity Module
166 Unit Tests
🔄 Phase 5: Ecosystem (IN PROGRESS)
Public Client Marketplace
Mainnet Launch
Governance DAO
Mobile Wallets
Hardware Wallet Integration
🔮 Phase 6: Advanced Features (PLANNED)
Zero-Knowledge Proofs
Sharding
Cross-Chain Bridges
AI Model Marketplace
Decentralized Storage (IPFS)
🧪 Testing
# Run all tests
cargo test

# Run specific module tests
cargo test learning_v3::training_metrics

# Run with output
cargo test -- --nocapture
Test Coverage:
166 unit tests
100% pass rate
Zero warnings
Production-grade quality
🛠️ Technical Stack
Component
Technology
Language
Rust 1.70+
Concurrency
Tokio (async runtime)
Networking
libp2p (TCP, GossipSub, Kademlia, mDNS)
Cryptography
ed25519-dalek, SHA3-256, ChaCha20-Poly1305, zeroize
Math/AI
ndarray, ndarray-rand, custom backpropagation, AdamW
GPU
WGPU (WebGPU for Rust)
Serialization
Serde, bincode, TOML/JSON
HTTP API
Axum, Tower-HTTP
Concurrency
DashMap, Arc, RwLock, Mutex
Error Handling
thiserror, anyhow
🤝 Contributing
HAFA is open-source and thrives on community collaboration. We welcome contributions in:
Client Development: UI/UX, Mobile, Web
AI Optimization: Learning algorithms, Transformer improvements
Security: Audits, formal verification
Documentation: Translations, tutorials
Testing: Bug reports, test coverage
Please read our CONTRIBUTING.md (coming soon) and open a Pull Request.
📜 License & Disclaimer
This project is licensed under the MIT License. See the LEGAL.md file for details.
⚠️ Disclaimer: HAFA is experimental, decentralized infrastructure. It does not endorse, enable, or control specific use cases. Client developers and end-users are solely responsible for their actions and compliance with local laws. The protocol is neutral technology, similar to the internet or TCP/IP. Use at your own risk.
📬 Contact & Resources
Repository: https://github.com/Decentralized-HAFA-AI/hafa
Issues: https://github.com/Decentralized-HAFA-AI/hafa/issues
Discussions: https://github.com/Decentralized-HAFA-AI/hafa/discussions
Whitepaper: WHITEPAPER.md
Legal: LEGAL.md
Powered by Rust 🦀 | Built for Decentralization 🌍 | Learning from Scratch 🧠
"Horizon After Freedom Achieved — the future is learned, not told."
