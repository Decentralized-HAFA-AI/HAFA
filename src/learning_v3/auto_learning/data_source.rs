// ============================================================================
// Data Sources for Autonomous Learning
// ============================================================================
//
// Defines the DataSource trait and implementations for different sources:
// - InMemoryDataSource: For testing and manual feeding
// - BlockchainDataSource: Reads from HAFA blockchain (Meta-Learning)
// - GossipSubDataSource: (future) Receives from P2P network
// - ExternalDataSource: (future) RSS, APIs, web
//
// ============================================================================

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use async_trait::async_trait;

/// A single training sample from any source
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// The text content to learn from
    pub text: String,
    /// Source identifier (e.g., "blockchain", "gossipsub", "external:rss")
    pub source: String,
    /// Unix timestamp when sample was created/received
    pub timestamp: u64,
    /// Confidence score from source (0.0 to 1.0)
    pub confidence: f32,
    /// Optional metadata (JSON string)
    pub metadata: Option<String>,
}

impl TrainingSample {
    pub fn new(text: String, source: String, confidence: f32) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Self {
            text,
            source,
            timestamp,
            confidence: confidence.clamp(0.0, 1.0),
            metadata: None,
        }
    }
    
    pub fn with_metadata(mut self, metadata: String) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Trait for all data sources
/// 
/// Implementations must be thread-safe (Send + Sync) because the
/// AutoLearningEngine may poll from multiple sources concurrently.
/// 
/// NOTE: poll is async to support blockchain reading and network I/O
#[async_trait]
pub trait DataSource: Send + Sync {
    /// Human-readable name of this source
    fn name(&self) -> &str;
    
    /// Priority (0 = highest, 255 = lowest)
    /// Lower priority sources are polled first
    fn priority(&self) -> u8;
    
    /// Poll for new samples (async to support I/O operations)
    /// Returns empty Vec if no new data available
    async fn poll(&mut self) -> Vec<TrainingSample>;
    
    /// Health check - is this source currently operational?
    fn is_healthy(&self) -> bool {
        true
    }
    
    /// Number of samples pending (if known)
    fn pending_count(&self) -> Option<usize> {
        None
    }
}

// ============================================================================
// IN-MEMORY DATA SOURCE (for testing and manual feeding)
// ============================================================================

/// A simple in-memory queue-based data source
/// Useful for testing and for external components to feed data
pub struct InMemoryDataSource {
    name: String,
    queue: VecDeque<TrainingSample>,
    max_size: usize,
    total_received: u64,
    total_consumed: u64,
}

impl InMemoryDataSource {
    pub fn new(name: &str, max_size: usize) -> Self {
        Self {
            name: name.to_string(),
            queue: VecDeque::new(),
            max_size,
            total_received: 0,
            total_consumed: 0,
        }
    }
    
    /// Push a new sample into the queue
    /// Returns false if queue is full (sample rejected)
    pub fn push(&mut self, sample: TrainingSample) -> bool {
        if self.queue.len() >= self.max_size {
            return false;
        }
        self.total_received += 1;
        self.queue.push_back(sample);
        true
    }
    
    /// Push text with auto-wrapping as TrainingSample
    pub fn push_text(&mut self, text: &str, confidence: f32) -> bool {
        let sample = TrainingSample::new(
            text.to_string(),
            format!("inmemory:{}", self.name),
            confidence,
        );
        self.push(sample)
    }
    
    /// Clear all pending samples
    pub fn clear(&mut self) {
        self.queue.clear();
    }
    
    /// Get statistics
    pub fn stats(&self) -> InMemoryStats {
        InMemoryStats {
            pending: self.queue.len(),
            max_size: self.max_size,
            total_received: self.total_received,
            total_consumed: self.total_consumed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InMemoryStats {
    pub pending: usize,
    pub max_size: usize,
    pub total_received: u64,
    pub total_consumed: u64,
}

#[async_trait]
impl DataSource for InMemoryDataSource {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn priority(&self) -> u8 {
        100  // Medium priority
    }
    
    async fn poll(&mut self) -> Vec<TrainingSample> {
        let samples: Vec<TrainingSample> = self.queue.drain(..).collect();
        self.total_consumed += samples.len() as u64;
        samples
    }
    
    fn pending_count(&self) -> Option<usize> {
        Some(self.queue.len())
    }
}

// ============================================================================
// BLOCKCHAIN DATA SOURCE (stub for future implementation)
// ============================================================================

/// Reads training data from HAFA blockchain blocks
/// (Stub - will be implemented in Phase 1.2)
pub struct BlockchainDataSource {
    enabled: bool,
    #[allow(dead_code)]
    last_block_height: u64,
}

impl BlockchainDataSource {
    pub fn new() -> Self {
        Self {
            enabled: false,  // Disabled by default until fully implemented
            last_block_height: 0,
        }
    }
    
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

#[async_trait]
impl DataSource for BlockchainDataSource {
    fn name(&self) -> &str {
        "blockchain"
    }
    
    fn priority(&self) -> u8 {
        10  // High priority - trusted source
    }
    
    async fn poll(&mut self) -> Vec<TrainingSample> {
        if !self.enabled {
            return Vec::new();
        }
        // TODO: Read new blocks from blockchain
        // For each block with cognitive proof, extract training data
        Vec::new()
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_training_sample_creation() {
        let sample = TrainingSample::new(
            "Hello world".to_string(),
            "test".to_string(),
            0.9,
        );
        assert_eq!(sample.text, "Hello world");
        assert_eq!(sample.source, "test");
        assert_eq!(sample.confidence, 0.9);
        assert!(sample.timestamp > 0);
    }
    
    #[tokio::test]
    async fn test_inmemory_source_basic() {
        let mut source = InMemoryDataSource::new("test", 100);
        
        // Initially empty
        assert_eq!(source.poll().await.len(), 0);
        assert_eq!(source.pending_count(), Some(0));
        
        // Push some samples
        assert!(source.push_text("Sample 1", 0.8));
        assert!(source.push_text("Sample 2", 0.9));
        assert_eq!(source.pending_count(), Some(2));
        
        // Poll should return all and clear
        let samples = source.poll().await;
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].text, "Sample 1");
        assert_eq!(samples[1].text, "Sample 2");
        assert_eq!(source.pending_count(), Some(0));
        
        // Stats check
        let stats = source.stats();
        assert_eq!(stats.total_received, 2);
        assert_eq!(stats.total_consumed, 2);
    }
    
    #[tokio::test]
    async fn test_inmemory_source_max_size() {
        let mut source = InMemoryDataSource::new("test", 2);
        
        assert!(source.push_text("Sample 1", 0.8));
        assert!(source.push_text("Sample 2", 0.8));
        
        // Third sample should be rejected
        assert!(!source.push_text("Sample 3", 0.8));
        assert_eq!(source.pending_count(), Some(2));
    }
    
    #[test]
    fn test_confidence_clamping() {
        let sample = TrainingSample::new("test".into(), "test".into(), 1.5);
        assert_eq!(sample.confidence, 1.0);
        
        let sample = TrainingSample::new("test".into(), "test".into(), -0.5);
        assert_eq!(sample.confidence, 0.0);
    }
}