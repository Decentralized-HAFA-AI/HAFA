// ============================================================================
// Knowledge Graph Storage: Persistent Storage for Long-Term Memory
// ============================================================================

use std::path::PathBuf;
use std::fs;
use serde::{Serialize, Deserialize};
use crate::learning_v3::knowledge_graph::{KnowledgeGraph, EntityType, RelationType};

#[derive(Serialize, Deserialize)]
struct KnowledgeGraphData {
    version: u32,
    entities: Vec<EntityData>,
    relations: Vec<RelationData>,
}

#[derive(Serialize, Deserialize)]
struct EntityData {
    id: String,
    name: String,
    entity_type: String,
    confidence: f32,
    created_at: u64,
    mentions: u64,
    properties: std::collections::HashMap<String, String>,
}

#[derive(Serialize, Deserialize)]
struct RelationData {
    source_id: String,
    target_id: String,
    relation_type: String,
    confidence: f32,
    created_at: u64,
    weight: f32,
}

pub struct KnowledgeGraphStorage {
    storage_path: PathBuf,
}

impl KnowledgeGraphStorage {
    pub fn new(storage_path: PathBuf) -> Self {
        Self { storage_path }
    }

    /// Save Knowledge Graph to disk
    pub fn save(&self, kg: &KnowledgeGraph) -> Result<(), String> {
        let entities: Vec<EntityData> = kg.entities().iter().map(|e| {
            EntityData {
                id: e.id.clone(),
                name: e.name.clone(),
                entity_type: e.entity_type.as_str().to_string(),
                confidence: e.confidence,
                created_at: e.created_at,
                mentions: e.mentions,
                properties: e.properties.clone(),
            }
        }).collect();

        let relations: Vec<RelationData> = kg.relations().iter().map(|r| {
            RelationData {
                source_id: r.source_id.clone(),
                target_id: r.target_id.clone(),
                relation_type: r.relation_type.as_str().to_string(),
                confidence: r.confidence,
                created_at: r.created_at,
                weight: r.weight,
            }
        }).collect();

        let data = KnowledgeGraphData {
            version: 1,
            entities,
            relations,
        };

        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| format!("Failed to serialize KG: {}", e))?;

        // Ensure directory exists
        if let Some(parent) = self.storage_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        fs::write(&self.storage_path, json)
            .map_err(|e| format!("Failed to write KG file: {}", e))?;

        println!("   [KG-STORAGE] 💾 Saved {} entities and {} relations to disk", 
                 data.entities.len(), data.relations.len());

        Ok(())
    }

    /// Load Knowledge Graph from disk
    pub fn load(&self) -> Result<KnowledgeGraph, String> {
        if !self.storage_path.exists() {
            println!("   [KG-STORAGE] 📂 No saved KG found, starting fresh");
            return Ok(KnowledgeGraph::new());
        }

        let json = fs::read_to_string(&self.storage_path)
            .map_err(|e| format!("Failed to read KG file: {}", e))?;

        let data: KnowledgeGraphData = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize KG: {}", e))?;

        let mut kg = KnowledgeGraph::new();

        // Restore entities
        for entity_data in data.entities {
            let entity_type = EntityType::from_string(&entity_data.entity_type);
            kg.restore_entity(
                entity_data.id,
                entity_data.name,
                entity_type,
                entity_data.confidence,
                entity_data.created_at,
                entity_data.mentions,
                entity_data.properties,
            );
        }

        // Restore relations
        for relation_data in data.relations {
            let relation_type = RelationType::from_string(&relation_data.relation_type);
            kg.restore_relation(
                relation_data.source_id,
                relation_data.target_id,
                relation_type,
                relation_data.confidence,
                relation_data.created_at,
                relation_data.weight,
            );
        }

        let stats = kg.stats();
        println!("   [KG-STORAGE] 📂 Loaded {} entities and {} relations from disk", 
                 stats.total_entities, stats.total_relations);

        Ok(kg)
    }
}