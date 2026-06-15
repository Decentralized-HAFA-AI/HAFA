# 🧠 HAFA Protocol v5.1.0

![Rust](https://img.shields.io/badge/Rust-1.70+-orange?logo=rust)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-184_Passing-brightgreen)
![Status](https://img.shields.io/badge/Status-Active_Success-brightgreen)

**H**orizon **A**fter **F**reedom **A**chieved

> A Decentralized, Self-Evolving AI Protocol. Learning from Scratch.

---

## 📖 What is HAFA?

**HAFA** stands for **Horizon After Freedom Achieved**. It is a next-generation decentralized network that merges blockchain consensus with native artificial intelligence. Unlike traditional AI that relies on centralized servers and pre-trained models, HAFA is built to **learn from scratch** on a distributed peer-to-peer network.

HAFA introduces a novel consensus mechanism called **Proof of Useful Cognitive Work (PoUCW)**, where nodes are rewarded for solving meaningful cognitive tasks rather than wasting energy on random hashing. The protocol is designed to be neutral, secure, and community-driven, with a locked core architecture and evolving client applications.

---

## ✨ Features

### 🔧 Core Architecture
- **Native AI Engine**: Built from scratch in Rust using `ndarray` (no PyTorch/TensorFlow)
- **Transformer v3 & v4**: Production-grade with AdamW optimizer and verifiable proofs
- **Knowledge Graph**: Structured long-term memory with NLP extraction
- **Reasoning Engine**: Query and inference over knowledge graph
- **Epistemic Filtering**: Multi-axis validation for data ingestion

### 🌐 Network & Consensus
- **Proof of Useful Cognitive Work (PoUCW)**: Quality-adjusted rewards based on learning improvement
- **P2P Network**: libp2p with mDNS discovery and GossipSub
- **Federated Learning**: HTTP-based sample sharing between nodes
- **Bitcoin-style Tokenomics**: 210M HAFA, 4-year halving, 500 HAFA initial reward

### 💼 Wallet & Security
- **Ed25519 Signatures**: Post-quantum resistant cryptography
- **ChaCha20 Encryption**: Encrypted key storage with passphrase
- **Multi-signature Support**: DAO governance and enhanced security
- **Checksum Addresses**: Ethereum-style error detection

### 🎮 Performance & UI
- **GPU Acceleration**: WGPU backend with CPU fallback
- **Web UI Dashboard**: Real-time monitoring and control
- **51+ API Endpoints**: Comprehensive REST API
- **Zero Warnings**: Production-grade code quality

---

## 🚀 Quick Start

### Prerequisites

- **Rust** (1.70 or later): [Install Rust](https://rustup.rs/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **8GB RAM** minimum (16GB recommended)
- **4 CPU cores** minimum (8 cores recommended)
- **GPU** (optional): For hardware acceleration

### Installation

```bash
# Clone the repository
git clone https://github.com/Decentralized-HAFA-AI/hafa.git
cd hafa

# Build in release mode (optimized)
cargo build --release

# Run tests (optional)
cargo test --release
```

### Running the Node

**Terminal 1 - Start the Genesis Node:**
```bash
cargo run --release
```

You should see:
```
🚀 HAFA Genesis Node Starting...
   Version: 5.1.0 - GPU Backend + Federated Learning + Web UI + Wallet
   
   🧠 Transformer v3 initialized: 461312 parameters
   🧠 Transformer v4 initialized: 461312 parameters (AdamW + Real Accumulation)
   🌐 Learning Network started on port 7474 ✨
   💼 Wallet Manager initialized (Ed25519 + ChaCha20 encryption) ✨
   🎨 Web UI Dashboard: http://127.0.0.1:7476/web ✨
   
   ✅ Node is alive. Press Ctrl+C to stop.
```

### Running the Miner

**Terminal 2 - Start Mining:**
```bash
cargo run --bin hafa-miner --release
```

You should see:
```
🧠 HAFA Connected Miner Started (REAL PoUCW)
   Node: http://127.0.0.1:7476
   Mode: Proof of Useful Cognitive Work (Real Neural Network Training)

📥 Task: Height #1, Difficulty: 1
   🧠 Training neural network on block data...
   ✅ Training complete: loss 1.0000 → 0.0923, 1 experiences
✅ Block #1 mined!
   💰 Reward: 401.03 HAFA (Quality: 0.61)
```

### Accessing the Web UI

Open your browser and navigate to:
```
http://127.0.0.1:7476/web
```

You'll see a real-time dashboard showing:
- Blockchain height and statistics
- AI model status
- Knowledge graph metrics
- P2P network information
- Wallet management

---

## 📡 API Reference

### Blockchain Endpoints

```bash
# Get node information
curl http://127.0.0.1:7476/info

# Get blockchain height
curl http://127.0.0.1:7476/height

# Get balance for address
curl http://127.0.0.1:7476/balance/{address}

# Get mining task
curl http://127.0.0.1:7476/task
```

### Mining Endpoints

```bash
# Submit mined block
curl -X POST http://127.0.0.1:7476/submit \
  -H "Content-Type: application/json" \
  -d '{
    "miner_addr": "your_address",
    "nonce": 12345,
    "cognitive_proof": { ... },
    "model_checkpoint": { ... }
  }'
```

### AI Learning Endpoints

```bash
# Get learning status
curl http://127.0.0.1:7476/learning-status

# Feed data to AI
curl -X POST http://127.0.0.1:7476/feed \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "local",
    "source_id": "test",
    "content": [72, 65, 70, 65]
  }'

# Train model
curl -X POST http://127.0.0.1:7476/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 5}'

# Train text (v4 - production)
curl -X POST http://127.0.0.1:7476/train-text-v4 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "HAFA is a decentralized AI blockchain",
    "context_size": 8,
    "epochs": 5
  }'
```

### Auto-Learning Endpoints

```bash
# Feed sample to auto-learning engine
curl -X POST http://127.0.0.1:7476/auto-learn/feed \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bitcoin was created by Satoshi Nakamoto",
    "source": "test",
    "confidence": 0.95
  }'

# Get auto-learning status
curl http://127.0.0.1:7476/auto-learn/status

# Get auto-learning statistics
curl http://127.0.0.1:7476/auto-learn/stats

# Trigger learning cycle
curl -X POST http://127.0.0.1:7476/auto-learn/trigger
```

### Knowledge Graph Endpoints

```bash
# Get knowledge graph statistics
curl http://127.0.0.1:7476/knowledge/stats

# Extract knowledge from text
curl -X POST http://127.0.0.1:7476/knowledge/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Satoshi Nakamoto created Bitcoin"}'

# Query knowledge graph
curl -X POST http://127.0.0.1:7476/knowledge/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who created Bitcoin?"}'
```

### Wallet Endpoints

```bash
# Create new wallet
curl -X POST http://127.0.0.1:7476/wallet/create \
  -H "Content-Type: application/json" \
  -d '{
    "passphrase": "your-secure-password",
    "label": "My Wallet"
  }'

# List all wallets
curl http://127.0.0.1:7476/wallet/list

# Get wallet info (using query parameter for addresses with colons)
curl "http://127.0.0.1:7476/wallet/info?address=your_address_here"

# Sign transaction
curl -X POST "http://127.0.0.1:7476/wallet/sign?address=your_address_here" \
  -H "Content-Type: application/json" \
  -d '{
    "passphrase": "your-password",
    "to_address": "recipient_address",
    "amount": 1000000,
    "fee": 10000
  }'
```

### P2P Network Endpoints

```bash
# Get P2P network info
curl http://127.0.0.1:7476/p2p/info

# Connect to peer
curl -X POST http://127.0.0.1:7476/p2p/connect \
  -H "Content-Type: application/json" \
  -d '{"multiaddr": "/ip4/192.168.1.1/tcp/7474/p2p/peer_id"}'
```

### Federated Learning Endpoints

```bash
# Share sample with network
curl -X POST http://127.0.0.1:7476/federated/share \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Sample data",
    "source": "test",
    "confidence": 0.9,
    "peer_id": "your_peer_id"
  }'

# Poll samples from network
curl http://127.0.0.1:7476/federated/poll

# Get federated learning stats
curl http://127.0.0.1:7476/federated/stats
```

---

## ⚙️ Configuration

HAFA uses a configuration file `hafa.toml` (optional). If not present, default values are used.

### Sample Configuration (`hafa.toml`)

```toml
[founder]
genesis_pubkey_hex = "your_64_char_hex_public_key"
vesting_enabled = true

[storage]
data_dir = "./data/hafa"

[network]
p2p_port = 7474
http_port = 7476
bootstrap_nodes = []
enable_mdns = true
enable_kademlia = true
connection_timeout_secs = 30

[learning]
allow_internet_learning = false
require_epistemic_validation = true
min_confidence_threshold = 0.85
trusted_sources_only = true

[mining]
enabled = false
cognitive_worker_threads = 4
target_block_time_secs = 600
```

### Configuration Options

| Section | Option | Description | Default |
|---------|--------|-------------|---------|
| `founder` | `genesis_pubkey_hex` | 64-character hex public key | Built-in |
| `founder` | `vesting_enabled` | Enable 3-year vesting schedule | `true` |
| `storage` | `data_dir` | Directory for blockchain and wallet data | System default |
| `network` | `p2p_port` | Port for P2P network | `7474` |
| `network` | `http_port` | Port for HTTP API | `7476` |
| `network` | `bootstrap_nodes` | List of bootstrap node addresses | `[]` |
| `network` | `enable_mdns` | Enable mDNS for local discovery | `true` |
| `network` | `enable_kademlia` | Enable Kademlia DHT | `true` |
| `learning` | `allow_internet_learning` | Allow fetching from external sources | `false` |
| `learning` | `require_epistemic_validation` | Require validation before accepting data | `true` |
| `learning` | `min_confidence_threshold` | Minimum confidence for external data | `0.85` |
| `mining` | `enabled` | Enable mining on this node | `false` |
| `mining` | `cognitive_worker_threads` | Number of worker threads | `4` |

---

## 📊 Tokenomics

### Supply & Distribution

- **Total Supply**: 210,000,000 HAFA (fixed, non-inflationary)
- **Initial Block Reward**: 500 HAFA
- **Halving Interval**: Every 210,000 blocks (~4 years)
- **Block Time**: 10 minutes (600 seconds)
- **Difficulty Adjustment**: Every 2,016 blocks (~2 weeks)

### Distribution Schedule

| Period | Blocks | Reward | Total Minted | Percentage |
|--------|--------|--------|--------------|------------|
| Era 1 | 0 - 209,999 | 500 HAFA | 105,000,000 | 50% |
| Era 2 | 210,000 - 419,999 | 250 HAFA | 52,500,000 | 25% |
| Era 3 | 420,000 - 629,999 | 125 HAFA | 26,250,000 | 12.5% |
| Era 4 | 630,000 - 839,999 | 62.5 HAFA | 13,125,000 | 6.25% |
| ... | ... | ... | ... | ... |

### Founder Allocation

- **Genesis Allocation**: 5% (10,500,000 HAFA)
- **Vesting Schedule**: 3 years
  - Year 0: 10% unlocked
  - Year 1: +30% unlocked
  - Year 2: +30% unlocked
  - Year 3: +30% unlocked

### Quality-Based Rewards

Mining rewards are adjusted based on the quality of cognitive work:

```
quality_score = (loss_reduction × 0.5) + (experience_factor × 0.3) + (confidence_factor × 0.2)
final_reward = base_reward × (0.5 + quality_score × 0.5)
```

Where:
- `loss_reduction`: Improvement in model loss (0.0 to 1.0)
- `experience_factor`: Number of experiences processed (capped at 1.0)
- `confidence_factor`: Average epistemic confidence (0.0 to 1.0)

---

## 🧪 Testing

### Running All Tests

```bash
# Run all tests
cargo test

# Run tests in release mode
cargo test --release

# Run specific test module
cargo test blockchain
cargo test crypto
cargo test wallet
```

### Test Coverage

- **169 Unit Tests**: All passing ✅
- **Module Coverage**: blockchain, crypto, config, api, data_source, epistemic, evolution, learning, network
- **Zero Warnings**: Production-grade code quality

---

## 🐛 Troubleshooting

### Build Errors

**Problem**: `cargo build` fails with dependency errors

**Solution**:
```bash
# Update Rust
rustup update

# Clean build cache
cargo clean

# Rebuild
cargo build --release
```

### Port Already in Use

**Problem**: `Address already in use` error

**Solution**:
```bash
# Check what's using port 7476
# Windows:
netstat -ano | findstr :7476

# Linux/macOS:
lsof -i :7476

# Kill the process or change port in hafa.toml
```

### Miner Not Connecting

**Problem**: Miner shows "Node not reachable"

**Solution**:
1. Ensure the node is running (Terminal 1)
2. Check that port 7476 is accessible
3. Verify firewall settings
4. Try accessing `http://127.0.0.1:7476/info` in browser

### Slow Mining

**Problem**: Mining is very slow

**Solution**:
1. Increase `cognitive_worker_threads` in `hafa.toml`
2. Ensure you have sufficient CPU cores
3. Check that difficulty is appropriate (starts at 1)
4. Monitor CPU usage

### Memory Issues

**Problem**: Out of memory errors

**Solution**:
1. Increase system RAM
2. Reduce `cognitive_worker_threads`
3. Close other applications
4. Consider using swap space

### Wallet Address Issues

**Problem**: Wallet API returns 404 for addresses with colons

**Solution**:
Use query parameters instead of path parameters:
```bash
# Correct:
curl "http://127.0.0.1:7476/wallet/info?address=your:address:here"

# Incorrect:
curl "http://127.0.0.1:7476/wallet/your:address:here/info"
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Decentralized-HAFA-AI/hafa.git
cd hafa

# Install dependencies
cargo build

# Run tests
cargo test

# Start development
cargo run
```

### Code Style

- Follow Rust standard conventions
- All comments must be in English
- Write unit tests for new features
- Ensure zero warnings before submitting

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- **AI/ML**: Improve learning algorithms, add new model architectures
- **Networking**: Enhance P2P protocol, add new discovery mechanisms
- **Security**: Audit cryptography, improve encryption
- **Performance**: Optimize algorithms, add GPU acceleration
- **Documentation**: Improve guides, add tutorials
- **Testing**: Add more test cases, improve coverage

---

## 📚 Documentation

- **[Whitepaper](WHITEPAPER.md)**: Technical specification of the HAFA protocol
- **[Legal](LEGAL.md)**: Legal terms and conditions
- **[API Reference](#-api-reference)**: Complete API documentation (see above)

---

## 🗺️ Roadmap

### Phase 1: Genesis (✅ Complete)
- [x] Core blockchain implementation
- [x] Native AI engine (MLP)
- [x] Basic P2P network
- [x] Wallet system

### Phase 2: Evolution (✅ Complete)
- [x] Transformer v3 & v4
- [x] Knowledge graph
- [x] Reasoning engine
- [x] Auto-learning engine
- [x] GPU acceleration

### Phase 3: Production (✅ Complete)
- [x] Web UI dashboard
- [x] Federated learning
- [x] Epistemic filtering
- [x] Verifiable proofs
- [x] 169 unit tests

### Phase 4: Ecosystem (🚧 In Progress)
- [ ] Mobile wallets
- [ ] Block explorer
- [ ] Developer SDK
- [ ] Mainnet launch

### Phase 5: Advanced (📋 Planned)
- [ ] Zero-knowledge proofs
- [ ] Sharding
- [ ] Cross-chain bridges
- [ ] DAO governance

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Bitcoin**: For pioneering blockchain technology and tokenomics
- **Ethereum**: For smart contracts and decentralized applications
- **Rust**: For providing a safe and performant programming language
- **libp2p**: For the P2P networking stack
- **ndarray**: For numerical computing in Rust

---

## 📞 Contact

- **GitHub**: [https://github.com/Decentralized-HAFA-AI/hafa](https://github.com/Decentralized-HAFA-AI/hafa)
- **Issues**: [https://github.com/Decentralized-HAFA-AI/hafa/issues](https://github.com/Decentralized-HAFA-AI/hafa/issues)

---

## 🌟 Star History

If you find HAFA interesting, please consider giving us a star on GitHub! It helps others discover the project.

---

<div align="center">

**"Horizon After Freedom Achieved — the future is learned, not told."**

Made with 🦀 Rust | Powered by 🧠 Native AI | Secured by 🔐 Cryptography

</div>
```

---
