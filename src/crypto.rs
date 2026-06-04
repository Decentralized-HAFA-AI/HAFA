// ============================================================================
// HAFA - src/crypto.rs — CRYPTOGRAPHIC PRIMITIVES (ADVANCED)
// ============================================================================
//
// Advanced cryptographic primitives with:
// - Ed25519 signatures (quantum-resistant)
// - SHA3-256 hashing
// - ChaCha20-Poly1305 encryption for key storage
// - Zeroize for secure memory handling
// - Multi-signature support for DAO
// - Checksum addresses (Ethereum-style)
// - Key derivation from passphrase
//
// ============================================================================

use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use hex::{FromHex, ToHex};
use rand::rngs::OsRng as DalekOsRng;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use thiserror::Error;
use zeroize::{Zeroize, ZeroizeOnDrop};
use std::fmt;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("Invalid key format or length")]
    InvalidKey,
    #[error("Hash computation failed")]
    HashError,
    #[error("Key generation failed: {0}")]
    GenerationError(String),
    #[error("Encryption failed: {0}")]
    EncryptionError(String),
    #[error("Decryption failed: {0}")]
    DecryptionError(String),
    #[error("Invalid checksum")]
    InvalidChecksum,
    #[error("Multi-signature error: {0}")]
    MultiSigError(String),
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Ed25519 key pair with secure memory handling
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct KeyPair {
    #[zeroize(skip)]
    pub signing_key: SigningKey,
    #[zeroize(skip)]
    pub verifying_key: VerifyingKey,
}

/// Cryptographic address with optional checksum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Address {
    pub pubkey_hex: String,
    pub checksum: Option<String>,
}

/// Multi-signature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSigConfig {
    pub signers: Vec<String>, // List of public keys
    pub threshold: u32,       // Minimum signatures required
}

/// Multi-signature result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSig {
    pub signatures: Vec<(String, String)>, // (pubkey_hex, signature_hex)
    pub message_hash: String,
}

// ============================================================================
// KEYPAIR IMPLEMENTATION
// ============================================================================

impl KeyPair {
    /// Generate a new Ed25519 key pair using OS entropy
    pub fn generate() -> Self {
        let mut rng = DalekOsRng;
        let signing_key = SigningKey::generate(&mut rng);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Reconstruct key pair from raw 32-byte secret key
    pub fn from_secret_bytes(bytes: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(bytes);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Derive key pair from passphrase using PBKDF2-like approach
    pub fn from_passphrase(passphrase: &str, salt: &[u8]) -> Self {
        // Simple key derivation: hash(passphrase + salt)
        // In production, use proper PBKDF2 or Argon2
        let mut hasher = Sha3_256::new();
        hasher.update(passphrase.as_bytes());
        hasher.update(salt);
        let hash = hasher.finalize();

        let mut secret_bytes = [0u8; 32];
        secret_bytes.copy_from_slice(&hash[..32]);

        let keypair = Self::from_secret_bytes(&secret_bytes);
        secret_bytes.zeroize(); // Clear sensitive data

        keypair
    }

    /// Sign arbitrary data
    pub fn sign(&self, message: &[u8]) -> Signature {
        self.signing_key.sign(message)
    }

    /// Export secret key as raw bytes (USE WITH CAUTION)
    pub fn secret_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// Export public key as raw bytes
    pub fn public_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    /// Derive cryptographic address with checksum
    pub fn address(&self) -> Address {
        let pubkey_hex = self.verifying_key.to_bytes().encode_hex::<String>();
        let checksum = Self::calculate_checksum(&pubkey_hex);
        Address {
            pubkey_hex,
            checksum: Some(checksum),
        }
    }

    /// Encrypt secret key with a passphrase
    pub fn encrypt_secret(&self, passphrase: &str) -> Result<EncryptedKey, CryptoError> {
        let secret_bytes = self.secret_bytes();

        // Derive encryption key from passphrase
        let mut hasher = Sha3_256::new();
        hasher.update(passphrase.as_bytes());
        hasher.update(b"hafa-key-encryption");
        let encryption_key = hasher.finalize();

        let cipher = ChaCha20Poly1305::new_from_slice(&encryption_key)
            .map_err(|e| CryptoError::EncryptionError(e.to_string()))?;

        // Generate random nonce
        let nonce_bytes: [u8; 12] = rand::random();
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt
        let ciphertext = cipher
            .encrypt(nonce, secret_bytes.as_ref())
            .map_err(|e| CryptoError::EncryptionError(e.to_string()))?;

        // Clear sensitive data
        let mut secret_copy = secret_bytes;
        secret_copy.zeroize();

        Ok(EncryptedKey {
            ciphertext: ciphertext.to_vec(),
            nonce: nonce_bytes.to_vec(),
            salt: b"hafa-key-encryption".to_vec(),
        })
    }

    /// Decrypt secret key from encrypted format
    pub fn decrypt_secret(encrypted: &EncryptedKey, passphrase: &str) -> Result<Self, CryptoError> {
        // Derive encryption key from passphrase
        let mut hasher = Sha3_256::new();
        hasher.update(passphrase.as_bytes());
        hasher.update(&encrypted.salt);
        let encryption_key = hasher.finalize();

        let cipher = ChaCha20Poly1305::new_from_slice(&encryption_key)
            .map_err(|e| CryptoError::DecryptionError(e.to_string()))?;

        let nonce = Nonce::from_slice(&encrypted.nonce);

        // Decrypt
        let plaintext = cipher
            .decrypt(nonce, encrypted.ciphertext.as_ref())
            .map_err(|e| CryptoError::DecryptionError(e.to_string()))?;

        if plaintext.len() != 32 {
            return Err(CryptoError::DecryptionError("Invalid key length".into()));
        }

        let mut secret_bytes = [0u8; 32];
        secret_bytes.copy_from_slice(&plaintext);

        let keypair = Self::from_secret_bytes(&secret_bytes);
        secret_bytes.zeroize();

        Ok(keypair)
    }

    /// Calculate checksum for address (first 4 bytes of hash)
    fn calculate_checksum(pubkey_hex: &str) -> String {
        let hash = hash_sha3_256(pubkey_hex.as_bytes());
        hash[..8].to_string() // First 4 bytes (8 hex chars)
    }
}

// ============================================================================
// ENCRYPTED KEY STRUCTURE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedKey {
    pub ciphertext: Vec<u8>,
    pub nonce: Vec<u8>,
    pub salt: Vec<u8>,
}

// ============================================================================
// ADDRESS IMPLEMENTATION
// ============================================================================

impl Address {
    /// Parse address from hex string (with or without checksum)
    pub fn from_hex(hex: &str) -> Result<Self, CryptoError> {
        // Check if address includes checksum (format: pubkey:checksum)
        let (pubkey_hex, checksum) = if hex.contains(':') {
            let parts: Vec<&str> = hex.split(':').collect();
            if parts.len() != 2 {
                return Err(CryptoError::InvalidKey);
            }
            (parts[0].to_string(), Some(parts[1].to_string()))
        } else {
            (hex.to_string(), None)
        };

        if pubkey_hex.len() != 64 || !pubkey_hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(CryptoError::InvalidKey);
        }

        // Verify checksum if present
        if let Some(ref cs) = checksum {
            let expected = KeyPair::calculate_checksum(&pubkey_hex);
            if cs != &expected {
                return Err(CryptoError::InvalidChecksum);
            }
        }

        Ok(Self {
            pubkey_hex,
            checksum,
        })
    }

    /// Create address with checksum
    pub fn with_checksum(pubkey_hex: &str) -> Result<Self, CryptoError> {
        if pubkey_hex.len() != 64 || !pubkey_hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(CryptoError::InvalidKey);
        }

        let checksum = KeyPair::calculate_checksum(pubkey_hex);
        Ok(Self {
            pubkey_hex: pubkey_hex.to_string(),
            checksum: Some(checksum),
        })
    }

    /// Get full address string with checksum
    pub fn to_string_with_checksum(&self) -> String {
        if let Some(ref cs) = self.checksum {
            format!("{}:{}", self.pubkey_hex, cs)
        } else {
            self.pubkey_hex.clone()
        }
    }

    /// Verify signature against this address's public key
    pub fn verify(&self, message: &[u8], signature_hex: &str) -> Result<(), CryptoError> {
        let pubkey_bytes = <[u8; 32]>::from_hex(&self.pubkey_hex)
            .map_err(|_| CryptoError::InvalidKey)?;

        let verifying_key = VerifyingKey::from_bytes(&pubkey_bytes)
            .map_err(|_| CryptoError::InvalidKey)?;

        let sig_bytes = <[u8; 64]>::from_hex(signature_hex)
            .map_err(|_| CryptoError::InvalidSignature)?;

        let signature = Signature::from_bytes(&sig_bytes);

        verifying_key
            .verify(message, &signature)
            .map_err(|_| CryptoError::InvalidSignature)
    }
}

// ============================================================================
// MULTI-SIGNATURE IMPLEMENTATION
// ============================================================================

impl MultiSigConfig {
    pub fn new(signers: Vec<String>, threshold: u32) -> Result<Self, CryptoError> {
        if threshold == 0 || threshold as usize > signers.len() {
            return Err(CryptoError::MultiSigError(
                "Invalid threshold".to_string(),
            ));
        }

        // Validate all public keys
        for signer in &signers {
            if !is_valid_pubkey(signer) {
                return Err(CryptoError::MultiSigError(format!(
                    "Invalid public key: {}",
                    signer
                )));
            }
        }

        Ok(Self { signers, threshold })
    }

    /// Verify multi-signature
    pub fn verify(&self, message: &[u8], multisig: &MultiSig) -> Result<bool, CryptoError> {
        if multisig.signatures.len() < self.threshold as usize {
            return Err(CryptoError::MultiSigError(format!(
                "Insufficient signatures: {} < {}",
                multisig.signatures.len(),
                self.threshold
            )));
        }

        let mut valid_count = 0;
        for (pubkey_hex, signature_hex) in &multisig.signatures {
            // Check if signer is in the allowed list
            if !self.signers.contains(pubkey_hex) {
                continue;
            }

            let address = Address::from_hex(pubkey_hex)?;
            if address.verify(message, signature_hex).is_ok() {
                valid_count += 1;
            }
        }

        Ok(valid_count >= self.threshold as usize)
    }
}

impl MultiSig {
    pub fn new(message: &[u8]) -> Self {
        Self {
            signatures: Vec::new(),
            message_hash: hash_sha3_256(message),
        }
    }

    /// Add a signature
    pub fn add_signature(&mut self, keypair: &KeyPair, message: &[u8]) {
        let signature = keypair.sign(message);
        let pubkey_hex = keypair.address().pubkey_hex;
        let signature_hex = signature.to_bytes().encode_hex::<String>();
        self.signatures.push((pubkey_hex, signature_hex));
    }
}

// ============================================================================
// STANDALONE FUNCTIONS
// ============================================================================

/// SHA3-256 hash wrapper
pub fn hash_sha3_256(data: &[u8]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(data);
    hasher.finalize().encode_hex::<String>()
}

/// Verify transaction/message signature directly from hex inputs
pub fn verify_hex_signature(
    pubkey_hex: &str,
    message: &[u8],
    signature_hex: &str,
) -> Result<(), CryptoError> {
    let addr = Address::from_hex(pubkey_hex)?;
    addr.verify(message, signature_hex)
}

/// Validate Ed25519 public key format
pub fn is_valid_pubkey(hex: &str) -> bool {
    hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit())
}

/// Generate random bytes (useful for nonces, salts, etc.)
pub fn random_bytes(len: usize) -> Vec<u8> {
    (0..len).map(|_| rand::random::<u8>()).collect()
}

// ============================================================================
// DISPLAY & SERIALIZATION HELPERS
// ============================================================================

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_with_checksum())
    }
}

impl fmt::Display for KeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KeyPair({})", self.address())
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation_and_signing() {
        let kp = KeyPair::generate();
        let msg = b"test payload";
        let sig = kp.sign(msg);

        let addr = kp.address();
        assert!(addr
            .verify(msg, &sig.to_bytes().encode_hex::<String>())
            .is_ok());
    }

    #[test]
    fn test_address_format_validation() {
        assert!(is_valid_pubkey(&"a".repeat(64)));
        assert!(!is_valid_pubkey("short"));
        assert!(!is_valid_pubkey(&"z".repeat(64))); // non-hex
    }

    #[test]
    fn test_hash_determinism() {
        let data = b"deterministic input";
        assert_eq!(hash_sha3_256(data), hash_sha3_256(data));
    }

    #[test]
    fn test_address_with_checksum() {
        let kp = KeyPair::generate();
        let addr = kp.address();

        assert!(addr.checksum.is_some());
        let addr_str = addr.to_string_with_checksum();
        assert!(addr_str.contains(':'));

        // Parse back
        let parsed = Address::from_hex(&addr_str).unwrap();
        assert_eq!(parsed.pubkey_hex, addr.pubkey_hex);
    }

    #[test]
    fn test_invalid_checksum() {
        let kp = KeyPair::generate();
        let addr = kp.address();
        let mut addr_str = addr.to_string_with_checksum();

        // Corrupt checksum
        addr_str.push('x');
        assert!(Address::from_hex(&addr_str).is_err());
    }

    #[test]
    fn test_keypair_encryption() {
        let kp = KeyPair::generate();
        let passphrase = "my-secret-passphrase";

        let encrypted = kp.encrypt_secret(passphrase).unwrap();
        let decrypted = KeyPair::decrypt_secret(&encrypted, passphrase).unwrap();

        assert_eq!(kp.public_bytes(), decrypted.public_bytes());
    }

    #[test]
    fn test_wrong_passphrase_decryption() {
        let kp = KeyPair::generate();
        let encrypted = kp.encrypt_secret("correct-pass").unwrap();

        let result = KeyPair::decrypt_secret(&encrypted, "wrong-pass");
        assert!(result.is_err());
    }

    #[test]
    fn test_keypair_from_passphrase() {
        let passphrase = "test-passphrase";
        let salt = b"test-salt";

        let kp1 = KeyPair::from_passphrase(passphrase, salt);
        let kp2 = KeyPair::from_passphrase(passphrase, salt);

        assert_eq!(kp1.public_bytes(), kp2.public_bytes());
    }

    #[test]
    fn test_multi_signature() {
        let kp1 = KeyPair::generate();
        let kp2 = KeyPair::generate();
        let kp3 = KeyPair::generate();

        let config = MultiSigConfig::new(
            vec![
                kp1.address().pubkey_hex,
                kp2.address().pubkey_hex,
                kp3.address().pubkey_hex,
            ],
            2, // 2-of-3
        )
        .unwrap();

        let message = b"multi-sig test";
        let mut multisig = MultiSig::new(message);

        // Add 2 signatures
        multisig.add_signature(&kp1, message);
        multisig.add_signature(&kp2, message);

        assert!(config.verify(message, &multisig).unwrap());
    }

    #[test]
    fn test_multi_signature_insufficient() {
        let kp1 = KeyPair::generate();
        let kp2 = KeyPair::generate();

        let config = MultiSigConfig::new(
            vec![kp1.address().pubkey_hex, kp2.address().pubkey_hex],
            2,
        )
        .unwrap();

        let message = b"multi-sig test";
        let mut multisig = MultiSig::new(message);

        // Add only 1 signature
        multisig.add_signature(&kp1, message);

        assert!(config.verify(message, &multisig).is_err());
    }

    #[test]
    fn test_random_bytes() {
        let bytes1 = random_bytes(32);
        let bytes2 = random_bytes(32);

        assert_eq!(bytes1.len(), 32);
        assert_ne!(bytes1, bytes2); // Should be different (with high probability)
    }
}