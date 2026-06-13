// ============================================================================
// Blockchain Data Source: Meta-Learning from Consensus
// ============================================================================
//
// Autonomously polls the HAFA blockchain for new blocks.
// Extracts CognitiveProof metadata to enable meta-learning:
// The AI learns from the mining dynamics, quality scores, and loss improvements
// of the network, making it truly self-evolving.
//
// ============================================================================

use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;

use super::data_source::{DataSource, TrainingSample};
use crate::blockchain::Blockchain;

pub struct BlockchainDataSource {
    blockchain: Arc<RwLock<Blockchain>>,
    last_processed_height: u64,
    enabled: bool,
}

impl BlockchainDataSource {
    pub fn new(blockchain: Arc<RwLock<Blockchain>>, start_height: u64) -> Self {
        // start_height is the COUNT of blocks, not the last index
        // We want last_processed_height to be the index of the last block
        // So if we have 52 blocks (indices 0..51), start_height = 52
        // We want last_processed_height = 51 (the last existing block)
        let last_processed = if start_height > 0 { start_height - 1 } else { 0 };
        
        println!("   [BLOCKCHAIN-SOURCE] Initializing: {} blocks exist, will track from index {}", 
                 start_height, last_processed);
        
        Self {
            blockchain,
            last_processed_height: last_processed,
            enabled: true,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn last_processed_height(&self) -> u64 {
        self.last_processed_height
    }
}

#[async_trait]
impl DataSource for BlockchainDataSource {
    fn name(&self) -> &str {
        "blockchain"
    }

    fn priority(&self) -> u8 {
        10 // Highest priority: consensus-verified, trusted data
    }

    async fn poll(&mut self) -> Vec<TrainingSample> {
        if !self.enabled {
            return Vec::new();
        }

        let mut samples = Vec::new();
        
        // Acquire read lock asynchronously
        let bc = self.blockchain.read().await;
        let current_height = bc.get_chain_height().await;
        
        // FIXED: current_height is the COUNT of blocks, not the last index
        // So if we have blocks 0..51, current_height = 52
        // We need to process up to index (current_height - 1)
        let last_block_index = if current_height > 0 { current_height - 1 } else { 0 };
        
        // If no new blocks, return empty
        if last_block_index <= self.last_processed_height {
            return samples;
        }

        let new_blocks_count = last_block_index - self.last_processed_height;
        println!("   [BLOCKCHAIN-SOURCE] Found {} new block(s) (height {} → {})", 
                 new_blocks_count, self.last_processed_height + 1, last_block_index);

        // FIXED: Iterate from (last_processed_height + 1) to last_block_index (inclusive)
        for height in (self.last_processed_height + 1)..=last_block_index {
            if let Some(block) = bc.get_block(height).await {
                // Extract rich meta-learning data from the block's cognitive proof
                let proof = &block.cognitive_proof;
                let quality_score = proof.quality_score();
                
                // Calculate loss improvement
                let loss_improvement = if proof.loss_before > 0.0 {
                    (proof.loss_before - proof.loss_after) / proof.loss_before
                } else {
                    0.0
                };
                
                // Create a rich training sample about mining dynamics
                let text = format!(
                    "Block #{} mined. Quality: {:.4}, Loss: {:.4} → {:.4} (improvement: {:.2}%), \
                     Experiences: {}, Duration: {}ms, CPU: {:.1}%, RAM: {}MB, \
                     Hash: {} → {}",
                    height,
                    quality_score,
                    proof.loss_before,
                    proof.loss_after,
                    loss_improvement * 100.0,
                    proof.experiences_processed,
                    proof.training_duration_ms,
                    proof.resources_used.cpu_percent,
                    proof.resources_used.ram_mb,
                    &proof.model_hash_before[..16.min(proof.model_hash_before.len())],
                    &proof.model_hash_after[..16.min(proof.model_hash_after.len())],
                );
                
                // Confidence based on quality score
                let confidence = (0.7 + quality_score * 0.3).clamp(0.0, 1.0);
                
                let mut sample = TrainingSample::new(
                    text,
                    "blockchain".to_string(),
                    confidence as f32,
                );
                
                // Add metadata
                let metadata = format!(
                    r#"{{"height":{},"quality":{:.4},"loss_before":{:.4},"loss_after":{:.4},"experiences":{}}}"#,
                    height, quality_score, proof.loss_before, proof.loss_after, proof.experiences_processed
                );
                sample = sample.with_metadata(metadata);
                
                samples.push(sample);
                
                println!("   [BLOCKCHAIN-SOURCE] Extracted sample from block #{} (quality: {:.4})", 
                         height, quality_score);
            } else {
                println!("   [BLOCKCHAIN-SOURCE] WARNING: Block #{} not found!", height);
            }
        }
        
        // Update the tracker
        self.last_processed_height = last_block_index;
        
        if !samples.is_empty() {
            println!("   [BLOCKCHAIN-SOURCE] Total {} training sample(s) extracted", samples.len());
        }
        
        samples
    }
}