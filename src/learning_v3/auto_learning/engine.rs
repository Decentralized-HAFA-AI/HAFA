#![allow(unused_variables)]
// ============================================================================
// Auto-Learning Engine with Curiosity Module, Episodic Memory, Meta-Learning
// and Knowledge Graph Integration
// ============================================================================
//
// The brain of HAFA's autonomous learning. Coordinates:
// - Multiple data sources (blockchain, gossipsub, external)
// - Epistemic filtering (trust, confidence, grounding)
// - Curiosity-based learning triggers
// - Training via TrainerV4
// - Proof generation and broadcasting
// - Episodic Memory (learning from experience)
// - Meta-Learning (learning from learning)
// - NEW: Knowledge Graph integration (structured long-term memory)
//
// ============================================================================

use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

use super::data_source::{DataSource, TrainingSample};
use super::curiosity::{CuriosityModule, CuriosityConfig};
use super::episodic_memory::EpisodicMemory;
use crate::learning_v3::TrainerV4;
use crate::learning_v3::cognitive_proof::CognitiveProofV4;
use crate::learning_v3::KnowledgeGraph;

/// Configuration for the AutoLearningEngine
#[derive(Debug, Clone)]
pub struct AutoLearningConfig {
    /// Minimum time between learning cycles
    pub min_cycle_interval: Duration,
    /// Maximum samples to accumulate before forcing a learning cycle
    pub max_buffer_size: usize,
    /// Minimum samples required to trigger learning
    pub min_samples_to_learn: usize,
    /// Context size for training (window of text)
    pub context_size: usize,
    /// Number of training epochs per cycle
    pub epochs_per_cycle: u32,
    /// Minimum confidence score to accept a sample
    pub min_confidence: f32,
    /// Enable curiosity-based filtering
    pub enable_curiosity: bool,
    /// Maximum number of episodes to store in episodic memory
    pub max_episodes: usize,
    /// Enable meta-learning
    pub enable_meta_learning: bool,
    /// Minimum success rate to proceed with learning (0.0 to 1.0)
    pub min_success_rate: f32,
    /// Success rate threshold to increase epochs
    pub high_success_rate: f32,
    /// Number of similar episodes to retrieve
    pub similar_episodes_count: usize,
}

impl Default for AutoLearningConfig {
    fn default() -> Self {
        Self {
            min_cycle_interval: Duration::from_secs(60),
            max_buffer_size: 1000,
            min_samples_to_learn: 5,
            context_size: 64,
            epochs_per_cycle: 3,
            min_confidence: 0.3,
            enable_curiosity: true,
            max_episodes: 1000,
            enable_meta_learning: true,
            min_success_rate: 0.3,
            high_success_rate: 0.8,
            similar_episodes_count: 3,
        }
    }
}

/// Statistics about the auto-learning engine
#[derive(Debug, Clone, Default)]
pub struct AutoLearningStats {
    pub total_cycles: u64,
    pub total_samples_received: u64,
    pub total_samples_rejected: u64,
    pub total_samples_learned: u64,
    pub total_proofs_generated: u64,
    pub last_cycle_time: Option<u64>,
    pub last_cycle_loss: Option<f32>,
    pub buffer_size: usize,
    // Curiosity-related stats
    pub curiosity_accepted: u64,
    pub curiosity_rejected: u64,
    // Episodic memory stats
    pub total_episodes_stored: u64,
    // Meta-learning stats
    pub meta_learning_checks: u64,
    pub meta_learning_skips: u64,
    pub meta_learning_boosts: u64,
    // NEW: Knowledge Graph stats
    pub kg_entities_retrieved: u64,
    pub kg_entities_added: u64,
    pub kg_relations_added: u64,
}

/// The main autonomous learning engine
pub struct AutoLearningEngine {
    /// All registered data sources
    sources: Vec<Box<dyn DataSource>>,
    /// Reference to the trainer (shared with the node)
    trainer: Arc<Mutex<TrainerV4>>,
    /// Configuration
    config: AutoLearningConfig,
    /// Buffer of pending samples
    buffer: Vec<TrainingSample>,
    /// Statistics
    stats: AutoLearningStats,
    /// Last learning cycle time
    last_cycle: Option<Instant>,
    /// Is the engine currently running a cycle?
    is_learning: bool,
    /// Curiosity Module for intelligent learning selection
    curiosity: CuriosityModule,
    /// Episodic Memory for learning from experience
    episodic_memory: EpisodicMemory,
    /// NEW: Knowledge Graph for structured long-term memory
    knowledge_graph: Option<Arc<RwLock<KnowledgeGraph>>>,
}

impl AutoLearningEngine {
    /// Create a new AutoLearningEngine
    pub fn new(
        trainer: Arc<Mutex<TrainerV4>>,
        config: AutoLearningConfig,
    ) -> Self {
        let max_episodes = config.max_episodes;
        
        println!("   [AUTO-LEARN] Initializing with curiosity module (enabled: {})", config.enable_curiosity);
        println!("   [EPISODIC] 🧠 Initializing episodic memory (max: {} episodes)", max_episodes);
        println!("   [META-LEARN] 🎯 Initializing meta-learning (enabled: {})", config.enable_meta_learning);
        
        Self {
            sources: Vec::new(),
            trainer,
            config,
            buffer: Vec::new(),
            stats: AutoLearningStats::default(),
            last_cycle: None,
            is_learning: false,
            curiosity: CuriosityModule::with_config(CuriosityConfig::default()),
            episodic_memory: EpisodicMemory::new(max_episodes),
            knowledge_graph: None,
        }
    }
    
    /// Create with custom curiosity config
    pub fn with_curiosity_config(
        trainer: Arc<Mutex<TrainerV4>>,
        config: AutoLearningConfig,
        curiosity_config: CuriosityConfig,
    ) -> Self {
        let max_episodes = config.max_episodes;
        
        println!("   [AUTO-LEARN] Initializing with custom curiosity config");
        println!("   [EPISODIC] 🧠 Initializing episodic memory (max: {} episodes)", max_episodes);
        println!("   [META-LEARN] 🎯 Initializing meta-learning (enabled: {})", config.enable_meta_learning);
        
        Self {
            sources: Vec::new(),
            trainer,
            config,
            buffer: Vec::new(),
            stats: AutoLearningStats::default(),
            last_cycle: None,
            is_learning: false,
            curiosity: CuriosityModule::with_config(curiosity_config),
            episodic_memory: EpisodicMemory::new(max_episodes),
            knowledge_graph: None,
        }
    }
    
    /// NEW: Attach a Knowledge Graph for structured long-term memory
    pub fn set_knowledge_graph(&mut self, kg: Arc<RwLock<KnowledgeGraph>>) {
        println!("   [KNOWLEDGE] 🧠 Knowledge Graph integration enabled ✨");
        self.knowledge_graph = Some(kg);
    }
    
    /// Register a new data source
    pub fn register_source(&mut self, source: Box<dyn DataSource>) {
        let name = source.name().to_string();
        let priority = source.priority();
        self.sources.push(source);
        
        // Sort by priority (lower = higher priority)
        self.sources.sort_by_key(|s| s.priority());
        
        println!("   [AUTO-LEARN] Registered source: {} (priority: {})", name, priority);
    }
    
    /// Poll all sources and add new samples to buffer (with curiosity filtering)
    pub async fn poll_sources(&mut self) -> usize {
        let mut new_count = 0;
        
        for source in &mut self.sources {
            if !source.is_healthy() {
                continue;
            }
            
            let samples = source.poll().await;
            
            for sample in samples {
                // Apply confidence filter first (cheap)
                if sample.confidence < self.config.min_confidence {
                    self.stats.total_samples_rejected += 1;
                    continue;
                }
                
                // Apply curiosity filter
                if self.config.enable_curiosity {
                    let should_accept = match self.trainer.try_lock() {
                        Ok(mut trainer) => self.curiosity.should_learn(&sample, &mut *trainer),
                        Err(_) => true,
                    };
                    
                    if !should_accept {
                        self.stats.curiosity_rejected += 1;
                        self.stats.total_samples_rejected += 1;
                        continue;
                    }
                    self.stats.curiosity_accepted += 1;
                    self.curiosity.mark_as_pending(&sample);
                }
                
                if self.buffer.len() < self.config.max_buffer_size {
                    self.buffer.push(sample);
                    new_count += 1;
                    self.stats.total_samples_received += 1;
                } else {
                    self.stats.total_samples_rejected += 1;
                }
            }
        }
        
        self.stats.buffer_size = self.buffer.len();
        new_count
    }
    
    /// Check if a learning cycle should be triggered
    pub fn should_learn(&self) -> bool {
        if self.is_learning {
            return false;
        }
        
        if self.buffer.len() < self.config.min_samples_to_learn {
            return false;
        }
        
        if let Some(last) = self.last_cycle {
            if last.elapsed() < self.config.min_cycle_interval {
                if self.buffer.len() < self.config.max_buffer_size * 9 / 10 {
                    return false;
                }
            }
        }
        
        true
    }
    
    /// Analyze similar past episodes and decide learning strategy
    fn analyze_similar_episodes(&mut self) -> LearningStrategy {
        if !self.config.enable_meta_learning {
            return LearningStrategy::Normal;
        }
        
        let similar_episodes = self.retrieve_similar_episodes(self.config.similar_episodes_count);
        
        if similar_episodes.is_empty() {
            println!("   [META-LEARN] 📊 No similar episodes found, proceeding normally");
            return LearningStrategy::Normal;
        }
        
        self.stats.meta_learning_checks += 1;
        
        let successful_count = similar_episodes.iter()
            .filter(|ep| ep.outcome.success)
            .count();
        
        let success_rate = successful_count as f32 / similar_episodes.len() as f32;
        let avg_improvement = similar_episodes.iter()
            .map(|ep| ep.outcome.loss_improvement)
            .sum::<f32>() / similar_episodes.len() as f32;
        
        println!("   [META-LEARN] 📊 Found {} similar episodes:", similar_episodes.len());
        println!("   [META-LEARN]    Success rate: {:.1}% ({}/{})", 
                 success_rate * 100.0, successful_count, similar_episodes.len());
        println!("   [META-LEARN]    Avg improvement: {:.2}%", avg_improvement * 100.0);
        
        // Decision logic
        if success_rate < self.config.min_success_rate {
            println!("   [META-LEARN] ⚠️  Low success rate (< {:.0}%), skipping learning", 
                     self.config.min_success_rate * 100.0);
            self.stats.meta_learning_skips += 1;
            return LearningStrategy::Skip;
        }
        
        if success_rate > self.config.high_success_rate {
            println!("   [META-LEARN] 🚀 High success rate (> {:.0}%), boosting learning", 
                     self.config.high_success_rate * 100.0);
            self.stats.meta_learning_boosts += 1;
            return LearningStrategy::Boost;
        }
        
        println!("   [META-LEARN] ✅ Moderate success rate, proceeding normally");
        LearningStrategy::Normal
    }
    
    /// NEW: Retrieve related knowledge from Knowledge Graph (non-blocking)
    fn retrieve_related_knowledge(&mut self, text: &str) -> Vec<String> {
                if let Some(kg_arc) = &self.knowledge_graph {
            // Try to acquire read lock without blocking
            if let Ok(kg) = kg_arc.try_read() {
                let entities = kg.extract_entities_from_text(text);
                if !entities.is_empty() {
                    let entity_names: Vec<String> = entities.iter()
                        .map(|e| format!("{} ({:?}, conf: {:.2})", e.name, e.entity_type, e.confidence))
                        .collect();
                    self.stats.kg_entities_retrieved += entities.len() as u64;
                    return entity_names;
                }
            }
        }
        Vec::new()
    }
    
    /// NEW: Add new knowledge to Knowledge Graph after learning (non-blocking)
    fn store_new_knowledge(&mut self, text: &str) {
        if let Some(kg_arc) = &self.knowledge_graph {
            // Try to acquire write lock without blocking
            if let Ok(mut kg) = kg_arc.try_write() {
                let (entities_added, relations_added) = kg.extract_from_text(text);
                if entities_added > 0 || relations_added > 0 {
                    println!("   [KNOWLEDGE] 📝 Added {} entities and {} relations to knowledge graph", 
                             entities_added, relations_added);
                    self.stats.kg_entities_added += entities_added as u64;
                    self.stats.kg_relations_added += relations_added as u64;
                }
            }
        }
    }
    
    /// Trigger a learning cycle (blocking - for API)
    /// Returns Some(proof) if learning happened, None otherwise
    pub fn trigger_learning(&mut self) -> Option<CognitiveProofV4> {
        if !self.should_learn() {
            return None;
        }
        
        // Meta-Learning - analyze similar episodes
        let strategy = self.analyze_similar_episodes();
        
        match strategy {
            LearningStrategy::Skip => {
                println!("   [AUTO-LEARN] ⏭️  Skipping learning cycle based on meta-learning");
                self.buffer.clear();
                self.curiosity.clear_pending();
                return None;
            }
            LearningStrategy::Boost => {
                println!("   [AUTO-LEARN] 🚀 Boosting learning cycle based on meta-learning");
            }
            LearningStrategy::Normal => {
                // Continue normally
            }
        }
        
        self.is_learning = true;
        let start = Instant::now();
        
        println!("   [AUTO-LEARN] Starting learning cycle with {} samples...", self.buffer.len());
        
        // Mark samples as learned in curiosity module BEFORE clearing buffer
        self.curiosity.mark_batch_as_learned(&self.buffer);
        self.curiosity.clear_pending();
        
        // Combine all samples into one text
        let combined_text = self.combine_samples();
        let sample_count = self.buffer.len();
        
        // NEW: Retrieve related knowledge from Knowledge Graph
        let related_entities = self.retrieve_related_knowledge(&combined_text);
        if !related_entities.is_empty() {
            println!("   [KNOWLEDGE] 🧠 Found {} related entities in knowledge graph:", related_entities.len());
            for entity_info in &related_entities {
                println!("   [KNOWLEDGE]    - {}", entity_info);
            }
        }
        
        // Clone buffer for episodic memory BEFORE clearing
        let samples_for_episode = self.buffer.clone();
        
        // Clear buffer before training (so new samples can accumulate)
        self.buffer.clear();
        
        // Train the model using try_lock to avoid blocking
        let proof = match self.trainer.try_lock() {
            Ok(mut trainer) => {
                trainer.train_on_text(
                    &combined_text,
                    self.config.context_size,
                    self.config.epochs_per_cycle,
                )
            }
            Err(_) => {
                println!("   [AUTO-LEARN] ERROR: Trainer is locked by another operation");
                self.is_learning = false;
                return None;
            }
        };
        
        let elapsed = start.elapsed();
        
        // Store episode in episodic memory
        let episode_id = self.episodic_memory.store_episode(&samples_for_episode, &proof);
        self.stats.total_episodes_stored += 1;
        println!("   [EPISODIC] 📝 Episode stored: {} (quality: {:.4})", 
                 &episode_id[..8.min(episode_id.len())], proof.quality_score());
        
        // NEW: Add new knowledge to Knowledge Graph
        self.store_new_knowledge(&combined_text);
        
        // Update stats
        self.stats.total_cycles += 1;
        self.stats.total_samples_learned += sample_count as u64;
        self.stats.total_proofs_generated += 1;
        self.stats.last_cycle_time = Some(elapsed.as_secs());
        self.stats.last_cycle_loss = Some(proof.loss_after);
        self.stats.buffer_size = 0;
        
        self.last_cycle = Some(Instant::now());
        self.is_learning = false;
        
        println!(
            "   [AUTO-LEARN] Cycle complete in {:.2}s | loss: {:.4} | quality: {:.4}",
            elapsed.as_secs_f32(),
            proof.loss_after,
            proof.quality_score()
        );
        
        Some(proof)
    }
    
    /// Combine buffered samples into a single training text
    fn combine_samples(&self) -> String {
        let mut sorted = self.buffer.clone();
        sorted.sort_by_key(|s| s.timestamp);
        
        sorted.iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
    
    /// Get current statistics
    pub fn stats(&self) -> AutoLearningStats {
        self.stats.clone()
    }
    
    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if engine is currently learning
    pub fn is_learning(&self) -> bool {
        self.is_learning
    }
    
    /// Manually push a sample into the buffer (for API)
    pub fn push_sample(&mut self, sample: TrainingSample) -> bool {
        if sample.confidence < self.config.min_confidence {
            return false;
        }
        if self.buffer.len() >= self.config.max_buffer_size {
            return false;
        }
        
        if self.config.enable_curiosity {
            let should_accept = match self.trainer.try_lock() {
                Ok(mut trainer) => self.curiosity.should_learn(&sample, &mut *trainer),
                Err(_) => true,
            };
            
            if !should_accept {
                self.stats.curiosity_rejected += 1;
                self.stats.total_samples_rejected += 1;
                return false;
            }
            self.stats.curiosity_accepted += 1;
            self.curiosity.mark_as_pending(&sample);
        }
        
        self.buffer.push(sample);
        self.stats.total_samples_received += 1;
        self.stats.buffer_size = self.buffer.len();
        true
    }
    
    /// Clear the buffer
    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
        self.curiosity.clear_pending();
        self.stats.buffer_size = 0;
    }
    
    /// Get curiosity module statistics
    pub fn curiosity_stats(&self) -> super::curiosity::CuriosityStats {
        self.curiosity.stats()
    }
    
    /// Get number of learned samples (for novelty detection)
    pub fn learned_count(&self) -> usize {
        self.curiosity.learned_count()
    }
    
    /// Get number of pending samples (in buffer, not yet learned)
    pub fn pending_count(&self) -> usize {
        self.curiosity.pending_count()
    }
    
    /// Clear curiosity memory (reset novelty detection)
    pub fn clear_curiosity_memory(&mut self) {
        self.curiosity.clear_memory();
    }
    
    /// Get episodic memory statistics
    pub fn episodic_stats(&self) -> super::episodic_memory::EpisodicMemoryStats {
        self.episodic_memory.stats()
    }
    
    /// Get all episodes
    pub fn episodes(&self) -> &[super::episodic_memory::Episode] {
        self.episodic_memory.episodes()
    }
    
    /// Get number of stored episodes
    pub fn episode_count(&self) -> usize {
        self.episodic_memory.episodes().len()
    }
    
    /// Retrieve similar past episodes based on current samples
    pub fn retrieve_similar_episodes(&self, top_k: usize) -> Vec<super::episodic_memory::Episode> {
        self.episodic_memory
            .retrieve_similar(&self.buffer, top_k)
            .into_iter()
            .cloned()
            .collect()
    }
    
    /// NEW: Check if Knowledge Graph is attached
    pub fn has_knowledge_graph(&self) -> bool {
        self.knowledge_graph.is_some()
    }
}

/// Learning strategy determined by meta-learning
#[derive(Debug, Clone, Copy, PartialEq)]
enum LearningStrategy {
    /// Normal learning with default parameters
    Normal,
    /// Skip this learning cycle (past similar episodes failed)
    Skip,
    /// Boost learning (past similar episodes succeeded)
    Boost,
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning_v3::TransformerConfig;
    
    fn make_test_trainer() -> Arc<Mutex<TrainerV4>> {
        let config = TransformerConfig {
            vocab_size: 256,
            embed_dim: 32,
            num_layers: 1,
            num_heads: 2,
            ff_dim: 64,
            max_seq_len: 32,
            dropout: 0.0,
        };
        let trainer = TrainerV4::new(
            &config,
            0.001,
            5,
            50,
            0.01,
            1,
        );
        Arc::new(Mutex::new(trainer))
    }
    
    #[test]
    fn test_engine_creation() {
        let trainer = make_test_trainer();
        let engine = AutoLearningEngine::new(trainer, AutoLearningConfig::default());
        assert_eq!(engine.buffer_size(), 0);
        assert!(!engine.is_learning());
        assert_eq!(engine.pending_count(), 0);
        assert_eq!(engine.episode_count(), 0);
        assert!(!engine.has_knowledge_graph());
    }
    
    #[test]
    fn test_push_sample_without_curiosity() {
        let trainer = make_test_trainer();
        let config = AutoLearningConfig {
            enable_curiosity: false,
            ..Default::default()
        };
        let mut engine = AutoLearningEngine::new(trainer, config);
        
        let sample = TrainingSample::new("test".into(), "test".into(), 0.9);
        assert!(engine.push_sample(sample));
        assert_eq!(engine.buffer_size(), 1);
        assert_eq!(engine.pending_count(), 0);
    }
    
    #[test]
    fn test_confidence_filter() {
        let trainer = make_test_trainer();
        let mut engine = AutoLearningEngine::new(trainer, AutoLearningConfig::default());
        
        let sample = TrainingSample::new("test".into(), "test".into(), 0.1);
        assert!(!engine.push_sample(sample));
        assert_eq!(engine.buffer_size(), 0);
    }
    
    #[test]
    fn test_curiosity_stats() {
        let trainer = make_test_trainer();
        let engine = AutoLearningEngine::new(trainer, AutoLearningConfig::default());
        
        let stats = engine.curiosity_stats();
        assert_eq!(stats.total_evaluated, 0);
    }
    
    #[test]
    fn test_learned_count() {
        let trainer = make_test_trainer();
        let engine = AutoLearningEngine::new(trainer, AutoLearningConfig::default());
        
        assert_eq!(engine.learned_count(), 0);
    }
    
    #[test]
    fn test_pending_count() {
        let trainer = make_test_trainer();
        let engine = AutoLearningEngine::new(trainer, AutoLearningConfig::default());
        
        assert_eq!(engine.pending_count(), 0);
    }
    
    #[test]
    fn test_episodic_stats() {
        let trainer = make_test_trainer();
        let engine = AutoLearningEngine::new(trainer, AutoLearningConfig::default());
        
        let stats = engine.episodic_stats();
        assert_eq!(stats.total_episodes, 0);
    }
    
    #[test]
    fn test_episode_count() {
        let trainer = make_test_trainer();
        let engine = AutoLearningEngine::new(trainer, AutoLearningConfig::default());
        
        assert_eq!(engine.episode_count(), 0);
    }
    
    #[test]
    fn test_meta_learning_config() {
        let trainer = make_test_trainer();
        let config = AutoLearningConfig::default();
        assert!(config.enable_meta_learning);
        assert_eq!(config.min_success_rate, 0.3);
        assert_eq!(config.high_success_rate, 0.8);
    }
    
    #[test]
    fn test_knowledge_graph_integration() {
        let trainer = make_test_trainer();
        let mut engine = AutoLearningEngine::new(trainer, AutoLearningConfig::default());
        
        // Initially no KG
        assert!(!engine.has_knowledge_graph());
        
        // Attach KG
        let kg = Arc::new(RwLock::new(KnowledgeGraph::new()));
        engine.set_knowledge_graph(kg);
        
        // Now has KG
        assert!(engine.has_knowledge_graph());
    }
}
