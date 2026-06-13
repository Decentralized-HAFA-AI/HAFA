// ============================================================================
// Embedding Layers
// ============================================================================
//
// Byte-level embedding and positional encoding for transformer architecture.
//
// Components:
// - ByteEmbedding: Converts byte sequences to dense vector representations
// - PositionalEncoding: Adds position information using sinusoidal encoding
//
// Design Principles:
// - Efficient: Optimized for batch processing
// - Robust: Handles edge cases (long sequences, out-of-vocab bytes)
// - Serializable: Can be saved/loaded for model checkpointing
//
// ============================================================================

use ndarray::Array2;
use ndarray_rand::{RandomExt, rand::distributions::Uniform};
use serde::{Deserialize, Serialize};
use super::tensor::Tensor;

// ============================================================================
// BYTE EMBEDDING
// ============================================================================

/// Converts byte sequences to dense vector representations
/// 
/// # Architecture
/// - Vocabulary size: Typically 256 (all possible byte values)
/// - Embedding dimension: Configurable (e.g., 64, 128, 256)
/// - Initialization: Xavier/Glorot uniform initialization
/// 
/// # Example
/// ```ust,ignore
/// let embedding = ByteEmbedding::new(256, 64);
/// let bytes = b"Hello HAFA";
/// let embedded = embedding.embed(bytes);  // Shape: [1, 10, 64]
/// `````
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteEmbedding {
    /// Size of the vocabulary (number of unique tokens)
    pub vocab_size: usize,
    
    /// Dimension of the embedding vectors
    pub embed_dim: usize,
    
    /// Embedding weight matrix [vocab_size, embed_dim]
    pub weights: Array2<f32>,
}

impl ByteEmbedding {
    /// Creates a new ByteEmbedding with random initialization
    /// 
    /// # Arguments
    /// * `vocab_size` - Number of unique tokens (typically 256 for bytes)
    /// * `embed_dim` - Dimension of embedding vectors
    /// 
    /// # Returns
    /// A new ByteEmbedding instance with Xavier-initialized weights
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        // Xavier/Glorot uniform initialization for stable training
        let std_dev = (1.0 / embed_dim as f64).sqrt() as f32;
        let uniform = Uniform::new(-std_dev, std_dev);
        
        Self {
            vocab_size,
            embed_dim,
            weights: Array2::random((vocab_size, embed_dim), uniform),
        }
    }

    /// Embeds a sequence of bytes into dense vectors
    /// 
    /// # Arguments
    /// * `bytes` - Input byte sequence
    /// 
    /// # Returns
    /// Tensor with shape [1, seq_len, embed_dim]
    /// 
    /// # Behavior
    /// - Bytes outside vocabulary range are mapped to zero vectors
    /// - Output is ready for positional encoding or direct transformer input
    pub fn embed(&self, bytes: &[u8]) -> Tensor {
        let seq_len = bytes.len();
        let mut result = Tensor::new(1, seq_len, self.embed_dim);
        
        for (i, &byte) in bytes.iter().enumerate() {
            let byte_idx = byte as usize;
            
            // Only embed if byte is within vocabulary
            if byte_idx < self.vocab_size {
                let embedding = self.weights.row(byte_idx);
                result.data.slice_mut(ndarray::s![0, i, ..]).assign(&embedding);
            }
            // Out-of-vocab bytes remain as zero vectors (graceful degradation)
        }
        
        result
    }

    /// Embeds multiple sequences in batch
    /// 
    /// # Arguments
    /// * `batch_bytes` - Vector of byte sequences
    /// 
    /// # Returns
    /// Tensor with shape [batch_size, max_seq_len, embed_dim]
    pub fn embed_batch(&self, batch_bytes: &[Vec<u8>]) -> Tensor {
        let batch_size = batch_bytes.len();
        let max_seq_len = batch_bytes.iter().map(|b| b.len()).max().unwrap_or(0);
        
        let mut result = Tensor::new(batch_size, max_seq_len, self.embed_dim);
        
        for (b, bytes) in batch_bytes.iter().enumerate() {
            for (i, &byte) in bytes.iter().enumerate() {
                let byte_idx = byte as usize;
                if byte_idx < self.vocab_size {
                    let embedding = self.weights.row(byte_idx);
                    result.data.slice_mut(ndarray::s![b, i, ..]).assign(&embedding);
                }
            }
        }
        
        result
    }

    /// Returns the total number of trainable parameters
    pub fn count_parameters(&self) -> usize {
        self.weights.len()
    }
}

// ============================================================================
// POSITIONAL ENCODING
// ============================================================================

/// Adds position information to token embeddings using sinusoidal encoding
/// 
/// # Architecture
/// - Sinusoidal encoding (Vaswani et al., 2017 "Attention Is All You Need")
/// - Fixed (non-trainable) encoding
/// - Supports sequences up to max_len
/// 
/// # Mathematical Formula
/// For position `pos` and dimension `i`:
/// - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// 
/// # Robustness
/// - Handles sequences longer than max_len gracefully (truncates encoding)
/// - No panics on edge cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionalEncoding {
    /// Maximum sequence length supported
    pub max_len: usize,
    
    /// Dimension of positional encoding (must match embedding dimension)
    pub embed_dim: usize,
    
    /// Precomputed positional encoding matrix [max_len, embed_dim]
    pub encoding: Array2<f32>,
}

impl PositionalEncoding {
    /// Creates a new PositionalEncoding with sinusoidal patterns
    /// 
    /// # Arguments
    /// * `max_len` - Maximum sequence length to support
    /// * `embed_dim` - Dimension of encoding (must match embedding dimension)
    /// 
    /// # Returns
    /// A new PositionalEncoding with precomputed sinusoidal patterns
    pub fn new(max_len: usize, embed_dim: usize) -> Self {
        let mut encoding = Array2::zeros((max_len, embed_dim));
        
        // Precompute sinusoidal positional encoding
        for pos in 0..max_len {
            for i in 0..embed_dim {
                // Calculate angle with exponential decay for higher dimensions
                let angle = pos as f32 / 10000.0f32.powf(2.0 * (i / 2) as f32 / embed_dim as f32);
                
                // Alternate between sin and cos for even/odd dimensions
                if i % 2 == 0 {
                    encoding[[pos, i]] = angle.sin();
                } else {
                    encoding[[pos, i]] = angle.cos();
                }
            }
        }
        
        Self {
            max_len,
            embed_dim,
            encoding,
        }
    }

    /// Adds positional encoding to input tensor
    /// 
    /// # Arguments
    /// * `input` - Input tensor with shape [batch, seq_len, embed_dim]
    /// 
    /// # Returns
    /// Tensor with same shape, with positional encoding added
    /// 
    /// # Robustness
    /// - If seq_len > max_len, only first max_len positions get encoding
    /// - Remaining positions retain original values (no positional info)
    /// - This allows graceful handling of unexpectedly long sequences
    /// 
    /// # Panics
    /// Panics if input dimension doesn't match embed_dim
    pub fn encode(&self, input: &Tensor) -> Tensor {
        let (batch, seq_len, dim) = input.shape();
        
        // Validate dimension compatibility
        assert_eq!(
            dim, self.embed_dim,
            "Input dimension {} doesn't match positional encoding dimension {}",
            dim, self.embed_dim
        );
        
        // Handle sequences longer than max_len gracefully
        let effective_len = seq_len.min(self.max_len);
        
        let mut result = input.data.clone();
        
        // Add positional encoding to each position
        for b in 0..batch {
            for s in 0..effective_len {
                let pos_encoding = self.encoding.row(s);
                let mut row = result.slice_mut(ndarray::s![b, s, ..]);
                row += &pos_encoding;
            }
            // Positions beyond max_len retain original values (no encoding)
        }
        
        Tensor { data: result }
    }

    /// Returns the total number of trainable parameters
    /// 
    /// # Note
    /// Positional encoding is fixed (non-trainable), so this returns 0
    pub fn count_parameters(&self) -> usize {
        0
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_embedding() {
        let embedding = ByteEmbedding::new(256, 64);
        let bytes = b"HAFA";
        let result = embedding.embed(bytes);
        
        assert_eq!(result.shape(), (1, 4, 64), "Output shape should be [1, seq_len, embed_dim]");
    }

    #[test]
    fn test_byte_embedding_out_of_vocab() {
        // Test with vocab_size < 256 to trigger out-of-vocab handling
        let embedding = ByteEmbedding::new(100, 64);
        let bytes = &[200u8, 50u8, 255u8]; // 200 and 255 are out of vocab
        
        let result = embedding.embed(bytes);
        assert_eq!(result.shape(), (1, 3, 64));
        
        // Out-of-vocab bytes should be zero vectors
        let first_token = result.data.slice(ndarray::s![0, 0, ..]);
        assert!(first_token.iter().all(|&v| v == 0.0), "Out-of-vocab should be zero");
    }

    #[test]
    fn test_positional_encoding() {
        let pos_enc = PositionalEncoding::new(128, 64);
        let input = Tensor::new(1, 10, 64);
        let result = pos_enc.encode(&input);
        
        assert_eq!(result.shape(), (1, 10, 64), "Shape should be preserved");
    }

    #[test]
    fn test_positional_encoding_long_sequence() {
        // Critical test: seq_len > max_len should NOT panic
        let pos_enc = PositionalEncoding::new(32, 64);
        let input = Tensor::new(1, 50, 64); // 50 > 32
        
        // Should handle gracefully without panic
        let result = pos_enc.encode(&input);
        assert_eq!(result.shape(), (1, 50, 64), "Shape should be preserved even for long sequences");
    }

    #[test]
    fn test_positional_encoding_dimension_mismatch() {
        let pos_enc = PositionalEncoding::new(128, 64);
        let input = Tensor::new(1, 10, 128); // Wrong dimension
        
        // Should panic with clear error message
        let result = std::panic::catch_unwind(|| pos_enc.encode(&input));
        assert!(result.is_err(), "Should panic on dimension mismatch");
    }

    #[test]
    fn test_combined_embedding() {
        let byte_emb = ByteEmbedding::new(256, 64);
        let pos_enc = PositionalEncoding::new(128, 64);
        
        let bytes = b"Hello HAFA";
        let embedded = byte_emb.embed(bytes);
        let with_pos = pos_enc.encode(&embedded);
        
        assert_eq!(with_pos.shape(), (1, bytes.len(), 64));
    }

    #[test]
    fn test_combined_embedding_long_sequence() {
        // Test full pipeline with long sequence
        let byte_emb = ByteEmbedding::new(256, 64);
        let pos_enc = PositionalEncoding::new(32, 64); // Small max_len
        
        let bytes = vec![65u8; 50]; // 50 bytes, longer than max_len
        let embedded = byte_emb.embed(&bytes);
        
        // Should handle gracefully
        let with_pos = pos_enc.encode(&embedded);
        assert_eq!(with_pos.shape(), (1, 50, 64));
    }

    #[test]
    fn test_parameter_counting() {
        let byte_emb = ByteEmbedding::new(256, 64);
        assert_eq!(byte_emb.count_parameters(), 256 * 64);
        
        let pos_enc = PositionalEncoding::new(128, 64);
        assert_eq!(pos_enc.count_parameters(), 0); // Fixed encoding
    }

    #[test]
    fn test_batch_embedding() {
        let embedding = ByteEmbedding::new(256, 64);
        let batch = vec![
            b"Hello".to_vec(),
            b"HAFA".to_vec(),
            b"World".to_vec(),
        ];
        
        let result = embedding.embed_batch(&batch);
        assert_eq!(result.shape(), (3, 5, 64)); // [batch_size, max_len, embed_dim]
    }
}