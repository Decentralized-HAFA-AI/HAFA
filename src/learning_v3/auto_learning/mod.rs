// ============================================================================
// Auto-Learning Module
// ============================================================================
//
// Enables HAFA to learn autonomously from multiple data sources without
// human intervention. This is the core of HAFA's "self-evolving" capability.
//
// ============================================================================

pub mod data_source;
pub mod engine;
pub mod curiosity;
pub mod blockchain_source;
pub mod episodic_memory;
pub mod gossipsub_source;
pub mod learning_network;
pub mod learning_protocol;


pub use data_source::{DataSource, TrainingSample, InMemoryDataSource};
pub use engine::{AutoLearningEngine, AutoLearningConfig, AutoLearningStats};
pub use curiosity::{CuriosityModule, CuriosityConfig, CuriosityStats};
pub use episodic_memory::{EpisodicMemory, Episode, LearningOutcome, EpisodicMemoryStats};
pub use gossipsub_source::GossipSubDataSource;
pub use learning_network::LearningNetwork;
pub use learning_protocol::{LearningMessage, LearningMessageType, LearningPayload, LEARNING_TOPIC};
