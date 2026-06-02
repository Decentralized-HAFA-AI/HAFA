// ============================================================================
// HAFA - src/crypto.rs — CRYPTOGRAPHIC PRIMITIVES
// ============================================================================

use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};
use sha3::{Sha3_256, Digest};
use hex::{ToHex, FromHex};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use thiserror::Error;
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
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone)]
pub struct KeyPair {
    pub signing_key: SigningKey,
    pub verifying_key: VerifyingKey,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Address {
    pub pubkey_hex: String,
}

// ============================================================================
// IMPLEMENTATION
// ============================================================================

impl KeyPair {
    /// Generate a new Ed25519 key pair using OS entropy
    pub fn generate() -> Self {
        let mut rng = OsRng;
        let signing_key = SigningKey::generate(&mut rng);
        let verifying_key = signing_key.verifying_key();
        Self { signing_key, verifying_key }
    }

    /// Reconstruct key pair from raw 32-byte secret key
    pub fn from_secret_bytes(bytes: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(bytes);
        let verifying_key = signing_key.verifying_key();
        Self { signing_key, verifying_key }
    }

    /// Sign arbitrary data
    pub fn sign(&self, message: &[u8]) -> Signature {
        self.signing_key.sign(message)
    }

    /// Export secret key as raw bytes
    pub fn secret_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// Export public key as raw bytes
    pub fn public_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    /// Derive cryptographic address
    pub fn address(&self) -> Address {
        Address {
            pubkey_hex: self.verifying_key.to_bytes().encode_hex::<String>(),
        }
    }
}

impl Address {
    /// Parse address from hex string
    pub fn from_hex(hex: &str) -> Result<Self, CryptoError> {
        if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(CryptoError::InvalidKey);
        }
        Ok(Self { pubkey_hex: hex.to_string() })
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

        verifying_key.verify(message, &signature)
            .map_err(|_| CryptoError::InvalidSignature)
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

// ============================================================================
// DISPLAY & SERIALIZATION HELPERS
// ============================================================================

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pubkey_hex)
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
        assert!(addr.verify(msg, &sig.to_bytes().encode_hex::<String>()).is_ok());
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
}