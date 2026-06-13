// ============================================================================
// HAFA - src/wallet.rs — WALLET MANAGEMENT SYSTEM
// ============================================================================

use crate::crypto::{EncryptedKey, KeyPair};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use thiserror::Error;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum WalletError {
    #[error("Wallet not found: {0}")]
    NotFound(String),
    #[error("Invalid passphrase")]
    InvalidPassphrase,
    #[error("Wallet already exists: {0}")]
    AlreadyExists(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Crypto error: {0}")]
    CryptoError(#[from] crate::crypto::CryptoError),
    #[error("Insufficient balance")]
    InsufficientBalance,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Wallet information (public data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletInfo {
    pub address: String,
    pub label: Option<String>,
    pub created_at: u64,
}

/// Stored wallet (encrypted)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredWallet {
    pub address: String,
    pub encrypted_key: EncryptedKey,
    pub label: Option<String>,
    pub created_at: u64,
}

/// Transaction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionRequest {
    pub from_address: String,
    pub to_address: String,
    pub amount: u64,
    pub fee: u64,
    pub timestamp: u64,
}

/// Signed transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedTransaction {
    pub from_address: String,
    pub to_address: String,
    pub amount: u64,
    pub fee: u64,
    pub signature: String,
    pub timestamp: u64,
}

/// Wallet manager
pub struct WalletManager {
    wallets: HashMap<String, StoredWallet>,
    storage_path: PathBuf,
}

// ============================================================================
// WALLET MANAGER IMPLEMENTATION
// ============================================================================

impl WalletManager {
    /// Create a new wallet manager
    pub fn new(storage_path: PathBuf) -> Self {
        let mut manager = Self {
            wallets: HashMap::new(),
            storage_path,
        };
        
        // Load existing wallets
        if let Err(e) = manager.load_wallets() {
            eprintln!("Warning: Could not load wallets: {}", e);
        }
        
        manager
    }

    /// Create a new wallet with passphrase
    pub fn create_wallet(
        &mut self,
        passphrase: &str,
        label: Option<String>,
    ) -> Result<WalletInfo, WalletError> {
        let keypair = KeyPair::generate();
        let address = keypair.address();
        let address_str = address.to_string_with_checksum();

        // Check if already exists
        if self.wallets.contains_key(&address_str) {
            return Err(WalletError::AlreadyExists(address_str));
        }

        // Encrypt private key
        let encrypted_key = keypair.encrypt_secret(passphrase)?;

        let created_at = chrono::Utc::now().timestamp() as u64;
        let stored = StoredWallet {
            address: address_str.clone(),
            encrypted_key,
            label: label.clone(),
            created_at,
        };

        self.wallets.insert(address_str.clone(), stored);
        self.save_wallets()?;

        Ok(WalletInfo {
            address: address_str,
            label,
            created_at,
        })
    }

    /// Import wallet from passphrase (deterministic)
    pub fn import_from_passphrase(
        &mut self,
        passphrase: &str,
        label: Option<String>,
    ) -> Result<WalletInfo, WalletError> {
        let salt = b"hafa-wallet-salt-v1";
        let keypair = KeyPair::from_passphrase(passphrase, salt);
        let address = keypair.address();
        let address_str = address.to_string_with_checksum();

        // Check if already exists
        if self.wallets.contains_key(&address_str) {
            return Err(WalletError::AlreadyExists(address_str));
        }

        // Encrypt private key
        let encrypted_key = keypair.encrypt_secret(passphrase)?;

        let created_at = chrono::Utc::now().timestamp() as u64;
        let stored = StoredWallet {
            address: address_str.clone(),
            encrypted_key,
            label: label.clone(),
            created_at,
        };

        self.wallets.insert(address_str.clone(), stored);
        self.save_wallets()?;

        Ok(WalletInfo {
            address: address_str,
            label,
            created_at,
        })
    }

    /// Get wallet info (public data)
    pub fn get_wallet_info(&self, address: &str) -> Result<WalletInfo, WalletError> {
        let stored = self
            .wallets
            .get(address)
            .ok_or_else(|| WalletError::NotFound(address.to_string()))?;

        Ok(WalletInfo {
            address: stored.address.clone(),
            label: stored.label.clone(),
            created_at: stored.created_at,
        })
    }

    /// List all wallets
    pub fn list_wallets(&self) -> Vec<WalletInfo> {
        self.wallets
            .values()
            .map(|w| WalletInfo {
                address: w.address.clone(),
                label: w.label.clone(),
                created_at: w.created_at,
            })
            .collect()
    }

    /// Sign a transaction
    pub fn sign_transaction(
        &self,
        address: &str,
        passphrase: &str,
        tx: &TransactionRequest,
    ) -> Result<SignedTransaction, WalletError> {
        let stored = self
            .wallets
            .get(address)
            .ok_or_else(|| WalletError::NotFound(address.to_string()))?;

        // Decrypt private key
        let keypair = KeyPair::decrypt_secret(&stored.encrypted_key, passphrase)
            .map_err(|_| WalletError::InvalidPassphrase)?;

        // Create message to sign
        let message = format!(
            "{}:{}:{}:{}:{}",
            tx.from_address, tx.to_address, tx.amount, tx.fee, tx.timestamp
        );

        // Sign
        let signature = keypair.sign(message.as_bytes());
        let signature_hex = hex::encode(signature.to_bytes());

        Ok(SignedTransaction {
            from_address: tx.from_address.clone(),
            to_address: tx.to_address.clone(),
            amount: tx.amount,
            fee: tx.fee,
            signature: signature_hex,
            timestamp: chrono::Utc::now().timestamp() as u64,
        })
    }

    /// Delete a wallet
    pub fn delete_wallet(&mut self, address: &str) -> Result<(), WalletError> {
        if self.wallets.remove(address).is_none() {
            return Err(WalletError::NotFound(address.to_string()));
        }
        self.save_wallets()?;
        Ok(())
    }

    /// Save wallets to disk
    fn save_wallets(&self) -> Result<(), WalletError> {
        let json = serde_json::to_string_pretty(&self.wallets)
            .map_err(|e| WalletError::SerializationError(e.to_string()))?;
        fs::write(&self.storage_path, json)?;
        Ok(())
    }

    /// Load wallets from disk
    fn load_wallets(&mut self) -> Result<(), WalletError> {
        if !self.storage_path.exists() {
            return Ok(());
        }

        let json = fs::read_to_string(&self.storage_path)?;
        self.wallets = serde_json::from_str(&json)
            .map_err(|e| WalletError::SerializationError(e.to_string()))?;
        Ok(())
    }
}

// ============================================================================
// TRANSACTION REQUEST HELPERS
// ============================================================================

impl TransactionRequest {
    pub fn new(from: String, to: String, amount: u64, fee: u64) -> Self {
        Self {
            from_address: from,
            to_address: to,
            amount,
            fee,
            timestamp: chrono::Utc::now().timestamp() as u64,
        }
    }
}