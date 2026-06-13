// ============================================================================
// Learning Protocol: Decentralized P2P Learning Messages
// ============================================================================

use serde::{Serialize, Deserialize};
use chrono::Utc;

/// Type of learning message
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum LearningMessageType {
    TrainingSample,
    ModelUpdate,
    KnowledgeShare,
}

/// Payload of learning message
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum LearningPayload {
    Sample {
        text: String,
        source: String,
        confidence: f32,
    },
    ModelUpdate {
        model_hash: String,
        loss_improvement: f32,
    },
    Knowledge {
        entities: Vec<String>,
        relations: Vec<String>,
    },
}

/// Learning message for P2P network
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LearningMessage {
    pub msg_type: LearningMessageType,
    pub sender_id: String,
    pub timestamp: u64,
    pub payload: LearningPayload,
}

impl LearningMessage {
    pub fn new_sample(sender_id: String, text: String, source: String, confidence: f32) -> Self {
        Self {
            msg_type: LearningMessageType::TrainingSample,
            sender_id,
            timestamp: Utc::now().timestamp() as u64,
            payload: LearningPayload::Sample {
                text,
                source,
                confidence,
            },
        }
    }
    
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
    
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        serde_json::from_slice(bytes).ok()
    }
}

/// GossipSub topic for learning
pub const LEARNING_TOPIC: &str = "hafa/learning/v1";