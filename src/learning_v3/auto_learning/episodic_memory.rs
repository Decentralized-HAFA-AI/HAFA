// ============================================================================
// Episodic Memory: Learning from Experience
// ============================================================================
//
// Stores past learning episodes and retrieves similar experiences
// to guide future learning decisions.
//
// ============================================================================

use sha3::{Sha3_256, Digest};
use chrono::Utc;

use super::data_source::TrainingSample;
use crate::learning_v3::CognitiveProofV4;

/// A single learning episode
#[derive(Debug, Clone)]
pub struct Episode {
    /// Unique identifier
    pub id: String,
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Input samples (hash for similarity comparison)
    pub samples_hash: String,
    
    /// Number of samples
    pub sample_count: usize,
    
    /// Learning outcome
    pub outcome: LearningOutcome,
    
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Outcome of a learning episode
#[derive(Debug, Clone)]
pub struct LearningOutcome {
    pub loss_before: f32,
    pub loss_after: f32,
    pub loss_improvement: f32,
    pub quality_score: f64,
    pub duration_ms: u64,
    pub success: bool,
}

/// Statistics about episodic memory
#[derive(Debug, Clone, Default)]
pub struct EpisodicMemoryStats {
    pub total_episodes: usize,
    pub successful_episodes: usize,
    pub failed_episodes: usize,
    pub avg_loss_improvement: f32,
    pub avg_quality_score: f64,
    pub total_retrievals: usize,
}

/// Episodic Memory: stores and retrieves learning experiences
pub struct EpisodicMemory {
    /// All episodes
    episodes: Vec<Episode>,
    
    /// Maximum episodes to store
    max_episodes: usize,
    
    /// Statistics
    stats: EpisodicMemoryStats,
    
    /// Running sums for averages
    sum_loss_improvement: f64,
    sum_quality_score: f64,
}

impl EpisodicMemory {
    /// Create a new EpisodicMemory
    pub fn new(max_episodes: usize) -> Self {
        Self {
            episodes: Vec::new(),
            max_episodes,
            stats: EpisodicMemoryStats::default(),
            sum_loss_improvement: 0.0,
            sum_quality_score: 0.0,
        }
    }
    
    /// Store a new episode from a learning cycle
    pub fn store_episode(
        &mut self,
        samples: &[TrainingSample],
        proof: &CognitiveProofV4,
    ) -> String {
        let id = self.generate_episode_id(samples, proof);
        let timestamp = Utc::now().timestamp() as u64;
        
        let samples_hash = self.compute_samples_hash(samples);
        
        let loss_improvement = if proof.loss_before > 0.0 {
            (proof.loss_before - proof.loss_after) / proof.loss_before
        } else {
            0.0
        };
        
        let success = loss_improvement > 0.0;
        let quality_score = proof.quality_score();
        let duration_ms = proof.wall_time_ms;
        
        let outcome = LearningOutcome {
            loss_before: proof.loss_before,
            loss_after: proof.loss_after,
            loss_improvement,
            quality_score,
            duration_ms,
            success,
        };
        
        // Generate tags based on outcome
        let tags = self.generate_tags(&outcome);
        
        let episode = Episode {
            id: id.clone(),
            timestamp,
            samples_hash,
            sample_count: samples.len(),
            outcome,
            tags,
        };
        
        // Update stats
        self.stats.total_episodes += 1;
        if success {
            self.stats.successful_episodes += 1;
        } else {
            self.stats.failed_episodes += 1;
        }
        self.sum_loss_improvement += loss_improvement as f64;
        self.sum_quality_score += quality_score;
        
        // Update averages
        let count = self.stats.total_episodes as f64;
        self.stats.avg_loss_improvement = (self.sum_loss_improvement / count) as f32;
        self.stats.avg_quality_score = self.sum_quality_score / count;
        
        // Add to episodes (with memory management)
        self.episodes.push(episode);
        if self.episodes.len() > self.max_episodes {
            self.episodes.remove(0); // Remove oldest
        }
        
        println!("   [EPISODIC] 📝 Stored episode {} (success: {}, improvement: {:.2}%)",
                 &id[..8.min(id.len())], success, loss_improvement * 100.0);
        
        id
    }
    
    /// Retrieve similar episodes based on current samples
    pub fn retrieve_similar(
        &self,
        current_samples: &[TrainingSample],
        top_k: usize,
    ) -> Vec<&Episode> {
        if self.episodes.is_empty() {
            return Vec::new();
        }
        
        let current_hash = self.compute_samples_hash(current_samples);
        
        // Simple similarity: exact match on sample count and similar hash prefix
        let mut scored_episodes: Vec<(&Episode, f32)> = self.episodes.iter()
            .map(|ep| {
                let similarity = self.compute_similarity(
                    &current_hash, 
                    &ep.samples_hash, 
                    current_samples.len(), 
                    ep.sample_count
                );
                (ep, similarity)
            })
            .collect();
        
        // Sort by similarity (descending)
        scored_episodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top_k
        scored_episodes.into_iter()
            .take(top_k)
            .map(|(ep, _)| ep)
            .collect()
    }
    
    /// Get statistics
    pub fn stats(&self) -> EpisodicMemoryStats {
        self.stats.clone()
    }
    
    /// Get all episodes
    pub fn episodes(&self) -> &[Episode] {
        &self.episodes
    }
    
    /// Generate unique episode ID
    fn generate_episode_id(&self, samples: &[TrainingSample], proof: &CognitiveProofV4) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(Utc::now().timestamp().to_le_bytes());
        hasher.update(proof.gradient_commitment.as_bytes());
        for sample in samples {
            hasher.update(sample.text.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }
    
    /// Compute hash of samples for similarity comparison
    fn compute_samples_hash(&self, samples: &[TrainingSample]) -> String {
        let mut hasher = Sha3_256::new();
        for sample in samples {
            hasher.update(sample.text.as_bytes());
            hasher.update(sample.source.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }
    
    /// Compute similarity between two episodes
    fn compute_similarity(
        &self,
        hash1: &str,
        hash2: &str,
        count1: usize,
        count2: usize,
    ) -> f32 {
        // Simple similarity metric:
        // 1. Sample count similarity (50% weight)
        let count_sim = 1.0 - (count1 as f32 - count2 as f32).abs() / (count1 + count2) as f32;
        
        // 2. Hash prefix similarity (50% weight)
        let prefix_len = 8.min(hash1.len()).min(hash2.len());
        let hash_sim = if hash1[..prefix_len] == hash2[..prefix_len] {
            1.0
        } else {
            0.0
        };
        
        count_sim * 0.5 + hash_sim * 0.5
    }
    
    /// Generate tags based on outcome
    fn generate_tags(&self, outcome: &LearningOutcome) -> Vec<String> {
        let mut tags = Vec::new();
        
        if outcome.success {
            tags.push("success".to_string());
        } else {
            tags.push("failure".to_string());
        }
        
        if outcome.loss_improvement > 0.2 {
            tags.push("high_improvement".to_string());
        } else if outcome.loss_improvement > 0.1 {
            tags.push("medium_improvement".to_string());
        } else {
            tags.push("low_improvement".to_string());
        }
        
        if outcome.quality_score > 0.5 {
            tags.push("high_quality".to_string());
        } else if outcome.quality_score > 0.3 {
            tags.push("medium_quality".to_string());
        } else {
            tags.push("low_quality".to_string());
        }
        
        if outcome.duration_ms > 300000 {
            tags.push("slow".to_string());
        } else if outcome.duration_ms < 100000 {
            tags.push("fast".to_string());
        }
        
        tags
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_episodic_memory_creation() {
        let memory = EpisodicMemory::new(100);
        assert_eq!(memory.stats.total_episodes, 0);
    }
    
    #[test]
    fn test_store_episode() {
        let mut memory = EpisodicMemory::new(100);
        
        let samples = vec![
            TrainingSample::new("test1".into(), "test".into(), 0.9),
            TrainingSample::new("test2".into(), "test".into(), 0.8),
        ];
        
        let proof = CognitiveProofV4 {
            model_hash_before: "a".repeat(64),
            model_hash_after: "b".repeat(64),
            dataset_commitment: "c".repeat(64),
            gradient_commitment: "d".repeat(64),
            loss_before: 5.0,
            loss_after: 4.0,
            ema_loss_after: 4.5,
            samples_processed: 100,
            wall_time_ms: 50000,
            cpu_usage_percent: 50.0,
            ram_usage_mb: 1024,
        };
        
        let id = memory.store_episode(&samples, &proof);
        
        assert_eq!(memory.stats.total_episodes, 1);
        assert_eq!(memory.stats.successful_episodes, 1);
        assert!(!id.is_empty());
    }
}