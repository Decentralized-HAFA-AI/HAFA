// ============================================================================
// GossipSub Data Source: Decentralized Learning from P2P Network
// ============================================================================
//
// Receives training samples from the HAFA P2P network via GossipSub.
// This enables truly decentralized learning where nodes share knowledge
// without relying on a central server.
//
// ============================================================================

use tokio::sync::mpsc;
use async_trait::async_trait;

use super::data_source::{DataSource, TrainingSample};

/// Data source that receives training samples from the P2P network
pub struct GossipSubDataSource {
    /// Channel to receive training samples from the Network Engine
    rx: mpsc::Receiver<TrainingSample>,
    /// Name of this source
    name: String,
    /// Is this source enabled?
    enabled: bool,
    /// Total samples received
    total_received: u64,
}

impl GossipSubDataSource {
    pub fn new(rx: mpsc::Receiver<TrainingSample>) -> Self {
        println!("   [GOSSIPSUB] 🌐 Initializing P2P data source");
        Self {
            rx,
            name: "gossipsub".to_string(),
            enabled: true,
            total_received: 0,
        }
    }
    
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    
    pub fn total_received(&self) -> u64 {
        self.total_received
    }
}

#[async_trait]
impl DataSource for GossipSubDataSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u8 {
        50 // Medium priority (lower than blockchain's 10, but still important)
    }

    async fn poll(&mut self) -> Vec<TrainingSample> {
        if !self.enabled {
            return Vec::new();
        }
        
        let mut samples = Vec::new();
        
        // Drain all available samples from the channel without blocking
        while let Ok(sample) = self.rx.try_recv() {
            samples.push(sample);
        }
        
        if !samples.is_empty() {
            self.total_received += samples.len() as u64;
            println!("   [GOSSIPSUB] 🌐 Received {} new sample(s) from P2P network (total: {})", 
                     samples.len(), self.total_received);
        }
        
        samples
    }

    fn is_healthy(&self) -> bool {
        self.enabled && !self.rx.is_closed()
    }

    fn pending_count(&self) -> Option<usize> {
        Some(self.rx.len())
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gossipsub_source_creation() {
        let (_tx, rx) = mpsc::channel(100);
        let source = GossipSubDataSource::new(rx);
        assert_eq!(source.name(), "gossipsub");
        assert_eq!(source.priority(), 50);
        assert!(source.is_healthy());
    }
    
    #[tokio::test]
    async fn test_gossipsub_poll_empty() {
        let (_tx, rx) = mpsc::channel(100);
        let mut source = GossipSubDataSource::new(rx);
        
        let samples = source.poll().await;
        assert_eq!(samples.len(), 0);
    }
    
    #[tokio::test]
    async fn test_gossipsub_poll_with_samples() {
        let (tx, rx) = mpsc::channel(100);
        let mut source = GossipSubDataSource::new(rx);
        
        // Send some samples
        let sample1 = TrainingSample::new("test1".into(), "p2p".into(), 0.9);
        let sample2 = TrainingSample::new("test2".into(), "p2p".into(), 0.8);
        
        tx.send(sample1).await.unwrap();
        tx.send(sample2).await.unwrap();
        
        // Poll should receive both
        let samples = source.poll().await;
        assert_eq!(samples.len(), 2);
        assert_eq!(source.total_received(), 2);
    }
    
    #[tokio::test]
    async fn test_gossipsub_disabled() {
        let (tx, rx) = mpsc::channel(100);
        let mut source = GossipSubDataSource::new(rx);
        
        source.disable();
        
        let sample = TrainingSample::new("test".into(), "p2p".into(), 0.9);
        tx.send(sample).await.unwrap();
        
        let samples = source.poll().await;
        assert_eq!(samples.len(), 0); // Should be empty because disabled
    }
}