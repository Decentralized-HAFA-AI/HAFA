// ============================================================================
// HAFA - src/config.rs — SYSTEM CONFIGURATION & ECONOMIC PARAMETERS
// ============================================================================
//
// Configuration system with external file support.
// Loads from `hafa.toml` if available, otherwise uses defaults.
//
// Key Features:
// - Immutable protocol constants (supply, halving, vesting)
// - Configurable network ports (P2P + HTTP API)
// - Epistemic learning controls
// - Mining parameters
// - Founder vesting schedule
//
// ============================================================================

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;
use thiserror::Error;

// ============================================================================
// PROTOCOL CONSTANTS (IMMUTABLE)
// ============================================================================

/// Target block time: 600 seconds (10 minutes, like Bitcoin)
pub const TARGET_BLOCK_TIME_SECS: u64 = 600;

/// Difficulty adjustment interval: Every 2016 blocks
pub const DIFFICULTY_ADJUSTMENT_INTERVAL: u64 = 2016;

/// Total Supply: 210 Million HAFA (8 decimal precision)
pub const MAX_SUPPLY: u64 = 210_000_000 * 100_000_000;

/// Initial Block Reward: 500 HAFA per block
pub const INITIAL_BLOCK_REWARD: u64 = 500 * 100_000_000;

/// Halving occurs every 210,000 blocks (~4 years)
pub const HALVING_INTERVAL: u64 = 210_000;

/// Founder Genesis Share: 5% of total supply (10.5 Million HAFA)
pub const FOUNDER_GENESIS_PERCENT: f64 = 5.0;

/// Founder Royalty: 2% of commercial transactions
pub const FOUNDER_ROYALTY_PERCENT: f64 = 2.0;

/// Vesting Schedule (seconds from genesis, cumulative unlocked satoshis)
/// Year 0: 10% | Year 1: +30% | Year 2: +30% | Year 3: +30%
pub const VESTING_SCHEDULE: [(u64, u64); 4] = [
    (0,                                      1_050_000 * 100_000_000),
    (365 * 24 * 60 * 60,                     4_200_000 * 100_000_000),
    (2 * 365 * 24 * 60 * 60,                 7_350_000 * 100_000_000),
    (3 * 365 * 24 * 60 * 60,                10_500_000 * 100_000_000),
];

/// Default genesis public key (can be overridden by config file)
pub const DEFAULT_GENESIS_PUBKEY: &str = "6b4719862983b9cd96e280b034124fc0dd52dfac330a6e74b1e5a14b6d282d06";

/// Default P2P port for libp2p learning network
pub const DEFAULT_P2P_PORT: u16 = 7474;

/// Default HTTP API port
pub const DEFAULT_HTTP_PORT: u16 = 7476;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Genesis public key is missing or invalid")]
    InvalidGenesisKey,
    #[error("Configuration file is malformed: {0}")]
    ParseError(String),
    #[error("Storage error: {0}")]
    StorageError(String),
    #[error("Validation failed: {0}")]
    ValidationError(String),
}

// ============================================================================
// CONFIGURATION STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub founder: FounderConfig,
    pub storage: StorageConfig,
    pub network: NetworkConfig,
    pub learning: LearningConfig,
    pub mining: MiningConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FounderConfig {
    /// Hex-encoded Ed25519 public key (64 chars)
    pub genesis_pubkey_hex: String,
    pub vesting_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub data_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// P2P port for libp2p learning network (GossipSub + mDNS)
    pub p2p_port: u16,
    /// HTTP API port for REST endpoints and Web UI
    pub http_port: u16,
    /// Bootstrap nodes for initial peer discovery
    pub bootstrap_nodes: Vec<String>,
    /// Enable mDNS for local network peer discovery
    pub enable_mdns: bool,
    /// Enable Kademlia DHT for global peer discovery
    pub enable_kademlia: bool,
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Allow fetching data from external sources (IPFS, Web, RSS)
    pub allow_internet_learning: bool,
    /// Require epistemic validation before accepting external data
    pub require_epistemic_validation: bool,
    /// Minimum confidence score for internet-sourced data
    pub min_confidence_threshold: f64,
    /// If true, only data from explicitly allowlisted sources is accepted
    pub trusted_sources_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningConfig {
    pub enabled: bool,
    pub cognitive_worker_threads: u32,
    pub target_block_time_secs: u64,
}

// ============================================================================
// IMPLEMENTATION
// ============================================================================

impl Config {
    /// Load configuration from a TOML/JSON file
    pub fn load(path: &PathBuf) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path)
            .map_err(|e| ConfigError::StorageError(e.to_string()))?;

        let config: Config = toml::from_str(&content)
            .or_else(|_| serde_json::from_str(&content))
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;

        config.validate()?;
        Ok(config)
    }

    /// Load configuration with fallback: try `hafa.toml`, then use defaults
    pub fn load_or_default() -> Self {
        let config_paths = vec![
            PathBuf::from("hafa.toml"),
            PathBuf::from("config.toml"),
            dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("hafa")
                .join("config.toml"),
        ];

        for path in config_paths {
            if path.exists() {
                match Self::load(&path) {
                    Ok(config) => {
                        println!("   📄 Loaded config from: {}", path.display());
                        return config;
                    }
                    Err(e) => {
                        eprintln!("   ⚠️  Failed to load {}: {}", path.display(), e);
                    }
                }
            }
        }

        println!("   📄 Using default configuration");
        Self::default()
    }

    /// Save current configuration to disk
    pub fn save(&self, path: &PathBuf) -> Result<(), ConfigError> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;

        fs::write(path, content)
            .map_err(|e| ConfigError::StorageError(e.to_string()))
    }

    /// Validate critical parameters before startup
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.founder.genesis_pubkey_hex.is_empty()
            || self.founder.genesis_pubkey_hex.len() != 64
            || !self.founder.genesis_pubkey_hex.chars().all(|c| c.is_ascii_hexdigit())
        {
            return Err(ConfigError::InvalidGenesisKey);
        }

        if self.learning.min_confidence_threshold < 0.0
            || self.learning.min_confidence_threshold > 1.0
        {
            return Err(ConfigError::ValidationError(
                "min_confidence_threshold must be between 0.0 and 1.0".into(),
            ));
        }

        if self.network.connection_timeout_secs == 0 {
            return Err(ConfigError::ValidationError(
                "connection_timeout_secs must be greater than 0".into(),
            ));
        }

        // Validate ports are different to prevent conflicts
        if self.network.p2p_port == self.network.http_port {
            return Err(ConfigError::ValidationError(
                "p2p_port and http_port must be different".into(),
            ));
        }

        Ok(())
    }

    /// Check if a given pubkey matches the genesis founder
    pub fn is_founder_key(&self, pubkey_hex: &str) -> bool {
        self.founder.genesis_pubkey_hex.eq_ignore_ascii_case(pubkey_hex)
    }

    /// Calculate unlocked founder amount based on current timestamp
    pub fn founder_unlocked_amount(&self, current_timestamp_secs: u64) -> u64 {
        if !self.founder.vesting_enabled {
            return self.founder_genesis_amount();
        }

        let mut unlocked = 0u64;
        for (unlock_time, cumulative_amount) in VESTING_SCHEDULE.iter() {
            if current_timestamp_secs >= *unlock_time {
                unlocked = *cumulative_amount;
            } else {
                break;
            }
        }
        unlocked
    }

    /// Total genesis allocation for founder (5%)
    pub fn founder_genesis_amount(&self) -> u64 {
        (MAX_SUPPLY as f64 * (FOUNDER_GENESIS_PERCENT / 100.0)) as u64
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            founder: FounderConfig {
                genesis_pubkey_hex: DEFAULT_GENESIS_PUBKEY.into(),
                vesting_enabled: true,
            },
            storage: StorageConfig {
                data_dir: dirs::data_dir()
                    .unwrap_or_else(|| PathBuf::from("./data"))
                    .join("hafa"),
            },
            network: NetworkConfig {
                p2p_port: DEFAULT_P2P_PORT,
                http_port: DEFAULT_HTTP_PORT,
                bootstrap_nodes: vec![],
                enable_mdns: true,
                enable_kademlia: true,
                connection_timeout_secs: 30,
            },
            learning: LearningConfig {
                allow_internet_learning: false,
                require_epistemic_validation: true,
                min_confidence_threshold: 0.85,
                trusted_sources_only: true,
            },
            mining: MiningConfig {
                enabled: false,
                cognitive_worker_threads: 4,
                target_block_time_secs: 600,
            },
        }
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let cfg = Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_founder_key_matching() {
        let cfg = Config::default();
        assert!(cfg.is_founder_key(DEFAULT_GENESIS_PUBKEY));
        assert!(!cfg.is_founder_key(&"a".repeat(64)));
    }

    #[test]
    fn test_vesting_schedule() {
        let cfg = Config::default();
        assert_eq!(cfg.founder_unlocked_amount(0), 1_050_000 * 100_000_000);
        assert_eq!(
            cfg.founder_unlocked_amount(365 * 24 * 60 * 60),
            4_200_000 * 100_000_000
        );
        assert_eq!(
            cfg.founder_unlocked_amount(2 * 365 * 24 * 60 * 60),
            7_350_000 * 100_000_000
        );
        assert_eq!(
            cfg.founder_unlocked_amount(3 * 365 * 24 * 60 * 60),
            10_500_000 * 100_000_000
        );
    }

    #[test]
    fn test_founder_genesis_amount() {
        let cfg = Config::default();
        let expected = (MAX_SUPPLY as f64 * 0.05) as u64;
        assert_eq!(cfg.founder_genesis_amount(), expected);
    }

    #[test]
    fn test_invalid_genesis_key() {
        let mut cfg = Config::default();
        
        // Test 1: Key shorter than 64 characters
        cfg.founder.genesis_pubkey_hex = "560d2b1d".to_string();
        assert!(cfg.validate().is_err(), "Should reject short key");
        
        // Test 2: Key with non-hex characters
        cfg.founder.genesis_pubkey_hex = "xyz0d2b1d8a70010b4a65a4e05bcfe4efe0a73b713de1b98c8e20d5c02f6ec43b".to_string();
        assert!(cfg.validate().is_err(), "Should reject non-hex characters");
        
        // Test 3: Empty key
        cfg.founder.genesis_pubkey_hex = "".to_string();
        assert!(cfg.validate().is_err(), "Should reject empty key");
    }

    #[test]
    fn test_genesis_key_length_validation() {
        let mut cfg = Config::default();
        cfg.founder.genesis_pubkey_hex = "560d2b1d8a70010b4a65a4e05bcfe4efe0a73b713de1b98c8e20d5c02f6ec43b".repeat(63);
        assert!(cfg.validate().is_err());

        cfg.founder.genesis_pubkey_hex = "560d2b1d8a70010b4a65a4e05bcfe4efe0a73b713de1b98c8e20d5c02f6ec43b".repeat(65);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_genesis_key_hex_validation() {
        let mut cfg = Config::default();
        cfg.founder.genesis_pubkey_hex = "560d2b1d8a70010b4a65a4e05bcfe4efe0a73b713de1b98c8e20d5c02f6ec43b".repeat(64);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_default_ports() {
        let cfg = Config::default();
        assert_eq!(cfg.network.p2p_port, DEFAULT_P2P_PORT);
        assert_eq!(cfg.network.http_port, DEFAULT_HTTP_PORT);
        assert_ne!(cfg.network.p2p_port, cfg.network.http_port);
    }

    #[test]
    fn test_port_conflict_validation() {
        let mut cfg = Config::default();
        cfg.network.p2p_port = 7476;
        cfg.network.http_port = 7476;
        assert!(cfg.validate().is_err(), "Should reject same p2p and http ports");
    }

    #[test]
    fn test_custom_ports() {
        let mut cfg = Config::default();
        cfg.network.p2p_port = 8477;
        cfg.network.http_port = 8476;
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.network.p2p_port, 8477);
        assert_eq!(cfg.network.http_port, 8476);
    }
}