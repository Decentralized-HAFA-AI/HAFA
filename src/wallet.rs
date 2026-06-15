// ============================================================================
// HAFA - src/wallet.rs — WALLET MANAGEMENT SYSTEM
// ============================================================================
//
// Secure wallet management with:
// - Ed25519 key generation and storage
// - ChaCha20-Poly1305 encryption for private keys
// - Deterministic wallet import from passphrase
// - Transaction signing
// - Persistent storage (JSON)
//
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
        // Ensure parent directory exists
        if let Some(parent) = self.storage_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }
        
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

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Helper: Create a temporary wallet manager for testing
    fn temp_wallet_manager() -> WalletManager {
        let temp_path = env::temp_dir()
            .join("hafa_wallet_test")
            .join(format!("wallets_{}.json", rand::random::<u64>()));
        WalletManager::new(temp_path)
    }

    #[test]
    fn test_create_wallet() {
        let mut manager = temp_wallet_manager();
        let result = manager.create_wallet("test-passphrase-123", Some("Test Wallet".to_string()));
        
        assert!(result.is_ok(), "Wallet creation should succeed");
        
        let wallet = result.unwrap();
        assert!(!wallet.address.is_empty(), "Address should not be empty");
        assert_eq!(wallet.label, Some("Test Wallet".to_string()));
        assert!(wallet.created_at > 0, "Created timestamp should be positive");
        
        // Address should contain checksum (format: pubkey:checksum)
        assert!(wallet.address.contains(':'), "Address should have checksum format");
    }

    #[test]
    fn test_create_wallet_without_label() {
        let mut manager = temp_wallet_manager();
        let result = manager.create_wallet("passphrase", None);
        
        assert!(result.is_ok());
        let wallet = result.unwrap();
        assert!(wallet.label.is_none(), "Label should be None");
    }

    #[test]
    fn test_import_from_passphrase_deterministic() {
        let mut manager1 = temp_wallet_manager();
        let mut manager2 = temp_wallet_manager();
        
        let passphrase = "deterministic-passphrase-test";
        
        let wallet1 = manager1.import_from_passphrase(passphrase, Some("W1".to_string())).unwrap();
        let wallet2 = manager2.import_from_passphrase(passphrase, Some("W2".to_string())).unwrap();
        
        // Same passphrase should generate same address (deterministic)
        assert_eq!(
            wallet1.address, wallet2.address,
            "Same passphrase should produce same address"
        );
    }

    #[test]
    fn test_different_passphrases_different_addresses() {
        let mut manager = temp_wallet_manager();
        
        let wallet1 = manager.import_from_passphrase("passphrase-1", None).unwrap();
        let wallet2 = manager.import_from_passphrase("passphrase-2", None).unwrap();
        
        assert_ne!(
            wallet1.address, wallet2.address,
            "Different passphrases should produce different addresses"
        );
    }

    #[test]
    fn test_wallet_persistence() {
        let temp_path = env::temp_dir()
            .join("hafa_wallet_test")
            .join(format!("persist_{}.json", rand::random::<u64>()));
        
        // Create wallet and save
        let address;
        {
            let mut manager = WalletManager::new(temp_path.clone());
            let wallet = manager.create_wallet("persist-test", Some("Persistent".to_string())).unwrap();
            address = wallet.address.clone();
        } // Manager dropped, file saved
        
        // Load again and verify
        {
            let manager = WalletManager::new(temp_path.clone());
            let info = manager.get_wallet_info(&address);
            assert!(info.is_ok(), "Wallet should persist across manager instances");
            assert_eq!(info.unwrap().label, Some("Persistent".to_string()));
        }
        
        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_sign_transaction_success() {
        let mut manager = temp_wallet_manager();
        let passphrase = "sign-test-pass";
        let wallet = manager.create_wallet(passphrase, None).unwrap();
        
        let tx = TransactionRequest::new(
            wallet.address.clone(),
            "recipient_address".to_string(),
            1000,
            10,
        );
        
        let result = manager.sign_transaction(&wallet.address, passphrase, &tx);
        assert!(result.is_ok(), "Signing should succeed with correct passphrase");
        
        let signed = result.unwrap();
        assert_eq!(signed.from_address, wallet.address);
        assert_eq!(signed.to_address, "recipient_address");
        assert_eq!(signed.amount, 1000);
        assert_eq!(signed.fee, 10);
        assert!(!signed.signature.is_empty(), "Signature should not be empty");
    }

    #[test]
    fn test_sign_transaction_wrong_passphrase() {
        let mut manager = temp_wallet_manager();
        let wallet = manager.create_wallet("correct-pass", None).unwrap();
        
        let tx = TransactionRequest::new(
            wallet.address.clone(),
            "recipient".to_string(),
            100,
            1,
        );
        
        let result = manager.sign_transaction(&wallet.address, "wrong-pass", &tx);
        assert!(result.is_err(), "Signing should fail with wrong passphrase");
        
        match result {
            Err(WalletError::InvalidPassphrase) => (), // Expected
            _ => panic!("Expected InvalidPassphrase error"),
        }
    }

    #[test]
    fn test_sign_nonexistent_wallet() {
        let manager = temp_wallet_manager();
        let tx = TransactionRequest::new(
            "nonexistent_address".to_string(),
            "recipient".to_string(),
            100,
            1,
        );
        
        let result = manager.sign_transaction("nonexistent_address", "pass", &tx);
        assert!(result.is_err(), "Signing should fail for nonexistent wallet");
    }

    #[test]
    fn test_list_wallets() {
        let mut manager = temp_wallet_manager();
        
        // Initially empty
        assert_eq!(manager.list_wallets().len(), 0);
        
        // Create 3 wallets
        manager.create_wallet("pass1", Some("W1".to_string())).unwrap();
        manager.create_wallet("pass2", Some("W2".to_string())).unwrap();
        manager.create_wallet("pass3", Some("W3".to_string())).unwrap();
        
        let wallets = manager.list_wallets();
        assert_eq!(wallets.len(), 3, "Should have 3 wallets");
    }

    #[test]
    fn test_get_wallet_info() {
        let mut manager = temp_wallet_manager();
        let wallet = manager.create_wallet("info-test", Some("Info Wallet".to_string())).unwrap();
        
        let info = manager.get_wallet_info(&wallet.address);
        assert!(info.is_ok());
        
        let info = info.unwrap();
        assert_eq!(info.address, wallet.address);
        assert_eq!(info.label, Some("Info Wallet".to_string()));
    }

    #[test]
    fn test_get_nonexistent_wallet_info() {
        let manager = temp_wallet_manager();
        let result = manager.get_wallet_info("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_wallet() {
        let mut manager = temp_wallet_manager();
        let wallet = manager.create_wallet("delete-test", None).unwrap();
        
        // Verify exists
        assert_eq!(manager.list_wallets().len(), 1);
        
        // Delete
        let result = manager.delete_wallet(&wallet.address);
        assert!(result.is_ok(), "Delete should succeed");
        
        // Verify gone
        assert_eq!(manager.list_wallets().len(), 0);
        assert!(manager.get_wallet_info(&wallet.address).is_err());
    }

    #[test]
    fn test_delete_nonexistent_wallet() {
        let mut manager = temp_wallet_manager();
        let result = manager.delete_wallet("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_transaction_request_new() {
        let tx = TransactionRequest::new(
            "from_addr".to_string(),
            "to_addr".to_string(),
            5000,
            50,
        );
        
        assert_eq!(tx.from_address, "from_addr");
        assert_eq!(tx.to_address, "to_addr");
        assert_eq!(tx.amount, 5000);
        assert_eq!(tx.fee, 50);
        assert!(tx.timestamp > 0, "Timestamp should be set");
    }

    #[test]
    fn test_wallet_error_display() {
        let err = WalletError::NotFound("addr123".to_string());
        assert!(err.to_string().contains("addr123"));
        
        let err = WalletError::InvalidPassphrase;
        assert!(err.to_string().contains("passphrase"));
        
        let err = WalletError::InsufficientBalance;
        assert!(err.to_string().contains("balance"));
    }
}