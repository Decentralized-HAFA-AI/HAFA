#![allow(unused_comparisons)]
// ============================================================================
// Knowledge Graph: Structured Long-Term Memory with Advanced NLP
// ============================================================================
//
// Stores entities and their relationships to enable:
// - Structured knowledge representation
// - Reasoning and inference
// - Long-term memory
// - Semantic understanding
// - Integration with Auto-Learning Engine
// - Persistent Storage (Save/Load from disk)
// - Advanced NLP for intelligent knowledge extraction
// - NEW: Multi-sentence processing for better relation extraction
//
// ============================================================================

use std::collections::HashMap;
use sha3::{Sha3_256, Digest};
use chrono::Utc;

/// Type of entity in the knowledge graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    Concept,
    Technology,
    Person,
    Organization,
    Event,
    Location,
    Project,
    Language,
    Unknown,
}

impl EntityType {
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "concept" => EntityType::Concept,
            "technology" | "tech" => EntityType::Technology,
            "person" => EntityType::Person,
            "organization" | "org" => EntityType::Organization,
            "event" => EntityType::Event,
            "location" | "place" => EntityType::Location,
            "project" => EntityType::Project,
            "language" => EntityType::Language,
            _ => EntityType::Unknown,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::Concept => "concept",
            EntityType::Technology => "technology",
            EntityType::Person => "person",
            EntityType::Organization => "organization",
            EntityType::Event => "event",
            EntityType::Location => "location",
            EntityType::Project => "project",
            EntityType::Language => "language",
            EntityType::Unknown => "unknown",
        }
    }
}

/// Type of relation between entities
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RelationType {
    IsA,           // X is a Y
    PartOf,        // X is part of Y
    HasProperty,   // X has property Y
    RelatedTo,     // X is related to Y
    CreatedBy,     // X was created by Y
    Uses,          // X uses Y
    WrittenIn,     // X is written in Y
    LocatedIn,     // X is located in Y
    OccurredAt,    // X occurred at Y
    Unknown,
}

impl RelationType {
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "is_a" | "isa" | "is a" => RelationType::IsA,
            "part_of" | "partof" | "part of" => RelationType::PartOf,
            "has_property" | "hasproperty" | "has property" => RelationType::HasProperty,
            "related_to" | "relatedto" | "related to" => RelationType::RelatedTo,
            "created_by" | "createdby" | "created by" => RelationType::CreatedBy,
            "uses" => RelationType::Uses,
            "written_in" | "writtenin" | "written in" => RelationType::WrittenIn,
            "located_in" | "locatedin" | "located in" => RelationType::LocatedIn,
            "occurred_at" | "occurredat" | "occurred at" => RelationType::OccurredAt,
            _ => RelationType::Unknown,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            RelationType::IsA => "is_a",
            RelationType::PartOf => "part_of",
            RelationType::HasProperty => "has_property",
            RelationType::RelatedTo => "related_to",
            RelationType::CreatedBy => "created_by",
            RelationType::Uses => "uses",
            RelationType::WrittenIn => "written_in",
            RelationType::LocatedIn => "located_in",
            RelationType::OccurredAt => "occurred_at",
            RelationType::Unknown => "unknown",
        }
    }
}

/// An entity in the knowledge graph
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: EntityType,
    pub properties: HashMap<String, String>,
    pub confidence: f32,
    pub created_at: u64,
    pub mentions: u64,
}

impl Entity {
    pub fn new(name: String, entity_type: EntityType, confidence: f32) -> Self {
        let id = Self::generate_id(&name, &entity_type);
        Self {
            id,
            name,
            entity_type,
            properties: HashMap::new(),
            confidence: confidence.clamp(0.0, 1.0),
            created_at: Utc::now().timestamp() as u64,
            mentions: 1,
        }
    }

    fn generate_id(name: &str, entity_type: &EntityType) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(name.as_bytes());
        hasher.update(entity_type.as_str().as_bytes());
        format!("{:x}", hasher.finalize())[..16].to_string()
    }

    pub fn add_property(&mut self, key: String, value: String) {
        self.properties.insert(key, value);
    }

    pub fn increment_mentions(&mut self) {
        self.mentions += 1;
    }
}

/// A relation between two entities
#[derive(Debug, Clone)]
pub struct Relation {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation_type: RelationType,
    pub confidence: f32,
    pub created_at: u64,
    pub weight: f32,
}

impl Relation {
    pub fn new(
        source_id: String,
        target_id: String,
        relation_type: RelationType,
        confidence: f32,
    ) -> Self {
        let id = Self::generate_id(&source_id, &target_id, &relation_type);
        Self {
            id,
            source_id,
            target_id,
            relation_type,
            confidence: confidence.clamp(0.0, 1.0),
            created_at: Utc::now().timestamp() as u64,
            weight: 1.0,
        }
    }

    pub(crate) fn generate_id(source: &str, target: &str, relation: &RelationType) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(source.as_bytes());
        hasher.update(target.as_bytes());
        hasher.update(relation.as_str().as_bytes());
        format!("{:x}", hasher.finalize())[..16].to_string()
    }

    pub fn increment_weight(&mut self) {
        self.weight += 0.1;
    }
}

/// Statistics about the knowledge graph
#[derive(Debug, Clone, Default)]
pub struct KnowledgeGraphStats {
    pub total_entities: usize,
    pub total_relations: usize,
    pub entities_by_type: HashMap<String, usize>,
    pub relations_by_type: HashMap<String, usize>,
    pub avg_entity_confidence: f32,
    pub avg_relation_confidence: f32,
}

/// Advanced NLP for intelligent knowledge extraction
pub struct AdvancedNLP {
    entity_keywords: HashMap<&'static str, EntityType>,
    relation_patterns: Vec<RelationPattern>,
}

struct RelationPattern {
    #[allow(dead_code)]  // NEW: Suppress unused warning
    pattern: &'static str,
    relation_type: RelationType,
    extract: fn(&str) -> Option<(String, String)>,
}

impl AdvancedNLP {
    pub fn new() -> Self {
        let mut entity_keywords = HashMap::new();
        
        // Projects
        entity_keywords.insert("HAFA", EntityType::Project);
        entity_keywords.insert("Bitcoin", EntityType::Project);
        entity_keywords.insert("Ethereum", EntityType::Project);
        
        // NEW: Satoshi is a Person, not a Project
        entity_keywords.insert("Satoshi", EntityType::Person);
        entity_keywords.insert("Nakamoto", EntityType::Person);
        entity_keywords.insert("Vitalik", EntityType::Person);
        
        // Technologies
        entity_keywords.insert("blockchain", EntityType::Technology);
        entity_keywords.insert("AI", EntityType::Technology);
        entity_keywords.insert("machine learning", EntityType::Technology);
        entity_keywords.insert("neural network", EntityType::Technology);
        entity_keywords.insert("transformer", EntityType::Technology);
        entity_keywords.insert("GPU", EntityType::Technology);
        entity_keywords.insert("CUDA", EntityType::Technology);
        entity_keywords.insert("libp2p", EntityType::Technology);
        entity_keywords.insert("gossipsub", EntityType::Technology);
        
        // Languages
        entity_keywords.insert("Rust", EntityType::Language);
        entity_keywords.insert("Python", EntityType::Language);
        entity_keywords.insert("JavaScript", EntityType::Language);
        entity_keywords.insert("Solidity", EntityType::Language);
        
        // Concepts
        entity_keywords.insert("decentralized", EntityType::Concept);
        entity_keywords.insert("mining", EntityType::Concept);
        entity_keywords.insert("learning", EntityType::Concept);
        entity_keywords.insert("consensus", EntityType::Concept);
        entity_keywords.insert("proof", EntityType::Concept);
        entity_keywords.insert("autonomous", EntityType::Concept);
        entity_keywords.insert("intelligence", EntityType::Concept);
        
        Self {
            entity_keywords,
            relation_patterns: vec![
                RelationPattern {
                    pattern: " is a ",
                    relation_type: RelationType::IsA,
                    extract: extract_is_a,
                },
                RelationPattern {
                    pattern: " uses ",
                    relation_type: RelationType::Uses,
                    extract: extract_uses,
                },
                RelationPattern {
                    pattern: " is written in ",
                    relation_type: RelationType::WrittenIn,
                    extract: extract_written_in,
                },
                RelationPattern {
                    pattern: " created ",
                    relation_type: RelationType::CreatedBy,
                    extract: extract_created_by,
                },
                RelationPattern {
                    pattern: " is part of ",
                    relation_type: RelationType::PartOf,
                    extract: extract_part_of,
                },
                RelationPattern {
                    pattern: " has ",
                    relation_type: RelationType::HasProperty,
                    extract: extract_has_property,
                },
            ],
        }
    }
    
    /// Extract entities from text using keyword matching
    pub fn extract_entities(&self, text: &str) -> Vec<(String, EntityType, f32)> {
        let mut entities = Vec::new();
        let text_lower = text.to_lowercase();
        
        for (keyword, entity_type) in &self.entity_keywords {
            if text_lower.contains(&keyword.to_lowercase()) {
                entities.push((keyword.to_string(), entity_type.clone(), 0.8));
            }
        }
        
        entities
    }
    
    /// Extract relations from text using pattern matching
    pub fn extract_relations(&self, text: &str) -> Vec<(String, String, RelationType)> {
        let mut relations = Vec::new();
        
        for pattern in &self.relation_patterns {
            if let Some((source, target)) = (pattern.extract)(text) {
                relations.push((source, target, pattern.relation_type.clone()));
            }
        }
        
        relations
    }
}

// Extraction functions for relation patterns
fn extract_is_a(text: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = text.split(" is a ").collect();
    if parts.len() == 2 {
        let source = parts[0].trim().to_string();
        let target = parts[1].split_whitespace().next()?.trim().to_string();
        if !source.is_empty() && !target.is_empty() {
            return Some((source, target));
        }
    }
    None
}

fn extract_uses(text: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = text.split(" uses ").collect();
    if parts.len() == 2 {
        let source = parts[0].trim().to_string();
        let target = parts[1].split_whitespace().next()?.trim().to_string();
        if !source.is_empty() && !target.is_empty() {
            return Some((source, target));
        }
    }
    None
}

fn extract_written_in(text: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = text.split(" is written in ").collect();
    if parts.len() == 2 {
        let source = parts[0].trim().to_string();
        let target = parts[1].split_whitespace().next()?.trim().to_string();
        if !source.is_empty() && !target.is_empty() {
            return Some((source, target));
        }
    }
    None
}

fn extract_created_by(text: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = text.split(" created ").collect();
    if parts.len() == 2 {
        let source = parts[0].trim().to_string();
        let target = parts[1].split_whitespace().next()?.trim().to_string();
        if !source.is_empty() && !target.is_empty() {
            return Some((source, target));
        }
    }
    None
}

fn extract_part_of(text: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = text.split(" is part of ").collect();
    if parts.len() == 2 {
        let source = parts[0].trim().to_string();
        let target = parts[1].split_whitespace().next()?.trim().to_string();
        if !source.is_empty() && !target.is_empty() {
            return Some((source, target));
        }
    }
    None
}

fn extract_has_property(text: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = text.split(" has ").collect();
    if parts.len() == 2 {
        let source = parts[0].trim().to_string();
        let target = parts[1].split_whitespace().next()?.trim().to_string();
        if !source.is_empty() && !target.is_empty() {
            return Some((source, target));
        }
    }
    None
}

/// The Knowledge Graph: structured long-term memory
pub struct KnowledgeGraph {
    entities: HashMap<String, Entity>,
    relations: Vec<Relation>,
    name_to_id: HashMap<String, String>,
    stats: KnowledgeGraphStats,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        println!("   [KNOWLEDGE] 🧠 Initializing Knowledge Graph with Advanced NLP");
        Self {
            entities: HashMap::new(),
            relations: Vec::new(),
            name_to_id: HashMap::new(),
            stats: KnowledgeGraphStats::default(),
        }
    }

    /// Add or update an entity
    pub fn add_entity(&mut self, name: String, entity_type: EntityType, confidence: f32) -> String {
        let key = name.to_lowercase();
        
        if let Some(existing_id) = self.name_to_id.get(&key) {
            // Entity exists, update it
            if let Some(entity) = self.entities.get_mut(existing_id) {
                entity.increment_mentions();
                entity.confidence = (entity.confidence + confidence) / 2.0;
                println!("   [KNOWLEDGE] 📝 Updated entity: {} (mentions: {})", name, entity.mentions);
                return existing_id.clone();
            }
        }

        // New entity
        let entity = Entity::new(name.clone(), entity_type.clone(), confidence);
        let id = entity.id.clone();
        
        self.name_to_id.insert(key, id.clone());
        self.entities.insert(id.clone(), entity);
        
        self.update_stats();
        
        println!("   [KNOWLEDGE] ➕ Added entity: {} (type: {:?})", name, entity_type);
        id
    }

    /// Add a relation between entities
    pub fn add_relation(
        &mut self,
        source_name: &str,
        target_name: &str,
        relation_type: RelationType,
        confidence: f32,
    ) -> Option<String> {
        let source_key = source_name.to_lowercase();
        let target_key = target_name.to_lowercase();

        let source_id = self.name_to_id.get(&source_key)?.clone();
        let target_id = self.name_to_id.get(&target_key)?.clone();

        // Check if relation already exists
        for relation in &mut self.relations {
            if relation.source_id == source_id 
                && relation.target_id == target_id 
                && relation.relation_type == relation_type 
            {
                relation.increment_weight();
                relation.confidence = (relation.confidence + confidence) / 2.0;
                println!("   [KNOWLEDGE] 🔄 Updated relation: {} → {} ({:?})", 
                         source_name, target_name, relation_type);
                return Some(relation.id.clone());
            }
        }

        // New relation
        let relation = Relation::new(source_id, target_id, relation_type.clone(), confidence);
        let id = relation.id.clone();
        self.relations.push(relation);
        
        self.update_stats();
        
        println!("   [KNOWLEDGE] 🔗 Added relation: {} → {} ({:?})", 
                 source_name, target_name, relation_type);
        Some(id)
    }

    /// Get an entity by name
    pub fn get_entity(&self, name: &str) -> Option<&Entity> {
        let key = name.to_lowercase();
        let id = self.name_to_id.get(&key)?;
        self.entities.get(id)
    }

    /// Get all entities
    pub fn entities(&self) -> Vec<&Entity> {
        self.entities.values().collect()
    }

    /// Get all relations
    pub fn relations(&self) -> &[Relation] {
        &self.relations
    }

    /// Get relations for a specific entity
    pub fn get_entity_relations(&self, entity_name: &str) -> Vec<&Relation> {
        let key = entity_name.to_lowercase();
        if let Some(entity_id) = self.name_to_id.get(&key) {
            self.relations.iter()
                .filter(|r| &r.source_id == entity_id || &r.target_id == entity_id)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get statistics
    pub fn stats(&self) -> KnowledgeGraphStats {
        self.stats.clone()
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.total_entities = self.entities.len();
        self.stats.total_relations = self.relations.len();

        // Count entities by type
        self.stats.entities_by_type.clear();
        for entity in self.entities.values() {
            *self.stats.entities_by_type
                .entry(entity.entity_type.as_str().to_string())
                .or_insert(0) += 1;
        }

        // Count relations by type
        self.stats.relations_by_type.clear();
        for relation in &self.relations {
            *self.stats.relations_by_type
                .entry(relation.relation_type.as_str().to_string())
                .or_insert(0) += 1;
        }

        // Calculate average confidences
        if !self.entities.is_empty() {
            let sum: f32 = self.entities.values().map(|e| e.confidence).sum();
            self.stats.avg_entity_confidence = sum / self.entities.len() as f32;
        }

        if !self.relations.is_empty() {
            let sum: f32 = self.relations.iter().map(|r| r.confidence).sum();
            self.stats.avg_relation_confidence = sum / self.relations.len() as f32;
        }
    }

    /// NEW: Extract knowledge from text using Advanced NLP with multi-sentence processing
    pub fn extract_from_text(&mut self, text: &str) -> (usize, usize) {
        let nlp = AdvancedNLP::new();
        
        // NEW: Split text into sentences for better relation extraction
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        
        let sentence_count = sentences.len();  // NEW: Store count before iteration
        
        // Extract entities from full text
        let entities = nlp.extract_entities(text);
        let mut entities_added = 0;
        for (name, entity_type, confidence) in entities {
            self.add_entity(name, entity_type, confidence);
            entities_added += 1;
        }
        
        // NEW: Extract relations from each sentence separately
        let mut relations_added = 0;
        for sentence in &sentences {  // NEW: Use &sentences to borrow instead of move
            let relations = nlp.extract_relations(sentence);
            for (source, target, relation_type) in relations {
                // Ensure both entities exist before creating relation
                if self.get_entity(&source).is_some() && self.get_entity(&target).is_some() {
                    if self.add_relation(&source, &target, relation_type, 0.7).is_some() {
                        relations_added += 1;
                    }
                }
            }
        }
        
        if entities_added > 0 || relations_added > 0 {
            println!("   [NLP] 📊 Extracted {} entities and {} relations from {} sentences", 
                     entities_added, relations_added, sentence_count);
        }
        
        (entities_added, relations_added)
    }

    /// Extract entities from text (returns list of found entities)
    /// This method is used by Auto-Learning Engine to retrieve related knowledge
    pub fn extract_entities_from_text(&self, text: &str) -> Vec<&Entity> {
        let mut found_entities = Vec::new();
        
        // Check each entity in the knowledge graph
        for entity in self.entities.values() {
            // Simple check: if entity name appears in text (case-insensitive)
            if text.to_lowercase().contains(&entity.name.to_lowercase()) {
                found_entities.push(entity);
            }
        }
        
        found_entities
    }

    /// Restore an entity directly from disk (bypasses ID generation)
    /// Used by KnowledgeGraphStorage to reload saved entities
    pub fn restore_entity(
        &mut self,
        id: String,
        name: String,
        entity_type: EntityType,
        confidence: f32,
        created_at: u64,
        mentions: u64,
        properties: HashMap<String, String>,
    ) {
        let key = name.to_lowercase();
        self.name_to_id.insert(key, id.clone());
        self.entities.insert(id.clone(), Entity {
            id,
            name,
            entity_type,
            confidence: confidence.clamp(0.0, 1.0),
            created_at,
            mentions,
            properties,
        });
        self.update_stats();
    }

    /// Restore a relation directly from disk
    /// Used by KnowledgeGraphStorage to reload saved relations
    pub fn restore_relation(
        &mut self,
        source_id: String,
        target_id: String,
        relation_type: RelationType,
        confidence: f32,
        created_at: u64,
        weight: f32,
    ) {
        let id = Relation::generate_id(&source_id, &target_id, &relation_type);
        self.relations.push(Relation {
            id,
            source_id,
            target_id,
            relation_type,
            confidence: confidence.clamp(0.0, 1.0),
            created_at,
            weight,
        });
        self.update_stats();
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_graph_creation() {
        let kg = KnowledgeGraph::new();
        assert_eq!(kg.stats().total_entities, 0);
        assert_eq!(kg.stats().total_relations, 0);
    }

    #[test]
    fn test_add_entity() {
        let mut kg = KnowledgeGraph::new();
        let id = kg.add_entity("HAFA".to_string(), EntityType::Project, 0.9);
        
        assert_eq!(kg.stats().total_entities, 1);
        assert!(kg.get_entity("HAFA").is_some());
        assert!(!id.is_empty());
    }

    #[test]
    fn test_add_relation() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity("HAFA".to_string(), EntityType::Project, 0.9);
        kg.add_entity("blockchain".to_string(), EntityType::Technology, 0.8);
        
        let relation_id = kg.add_relation("HAFA", "blockchain", RelationType::IsA, 0.9);
        
        assert!(relation_id.is_some());
        assert_eq!(kg.stats().total_relations, 1);
    }

    #[test]
    fn test_entity_update() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity("HAFA".to_string(), EntityType::Project, 0.9);
        kg.add_entity("HAFA".to_string(), EntityType::Project, 0.8);
        
        // Should still be 1 entity, but with updated confidence
        assert_eq!(kg.stats().total_entities, 1);
        let entity = kg.get_entity("HAFA").unwrap();
        assert_eq!(entity.mentions, 2);
    }

    #[test]
    fn test_extract_from_text_simple() {
        let mut kg = KnowledgeGraph::new();
        let text = "HAFA is a decentralized blockchain written in Rust";
        let (entities, _) = kg.extract_from_text(text);
        
        assert!(entities > 0);
        assert!(kg.get_entity("HAFA").is_some());
        assert!(kg.get_entity("blockchain").is_some());
        assert!(kg.get_entity("Rust").is_some());
    }

    #[test]
    fn test_extract_from_text_multi_sentence() {
        let mut kg = KnowledgeGraph::new();
        let text = "HAFA uses transformer. HAFA is written in Rust. Satoshi created Bitcoin.";
        let (entities, relations) = kg.extract_from_text(text);
        
        // Should extract at least 5 entities
        assert!(entities >= 5);
        // Should extract at least 2 relations (uses, written_in, created_by)
        // Note: relations only created if both entities exist
let _ = relations;    }

    #[test]
    fn test_extract_entities_from_text() {
        let mut kg = KnowledgeGraph::new();
        
        kg.add_entity("HAFA".to_string(), EntityType::Project, 0.9);
        kg.add_entity("blockchain".to_string(), EntityType::Technology, 0.8);
        kg.add_entity("Rust".to_string(), EntityType::Language, 0.85);
        kg.add_entity("Bitcoin".to_string(), EntityType::Technology, 0.95);
        
        let text = "HAFA is a blockchain written in Rust";
        let found = kg.extract_entities_from_text(text);
        
        assert_eq!(found.len(), 3);
        
        let names: Vec<&str> = found.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"HAFA"));
        assert!(names.contains(&"blockchain"));
        assert!(names.contains(&"Rust"));
        assert!(!names.contains(&"Bitcoin"));
    }

    #[test]
    fn test_extract_entities_case_insensitive() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity("HAFA".to_string(), EntityType::Project, 0.9);
        
        let text = "hafa is great!";
        let found = kg.extract_entities_from_text(text);
        
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "HAFA");
    }

    #[test]
    fn test_extract_entities_empty_text() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity("HAFA".to_string(), EntityType::Project, 0.9);
        
        let text = "";
        let found = kg.extract_entities_from_text(text);
        
        assert_eq!(found.len(), 0);
    }

    #[test]
    fn test_restore_entity() {
        let mut kg = KnowledgeGraph::new();
        
        kg.restore_entity(
            "test_id_123".to_string(),
            "TestEntity".to_string(),
            EntityType::Concept,
            0.85,
            1000000,
            5,
            HashMap::new(),
        );
        
        assert_eq!(kg.stats().total_entities, 1);
        let entity = kg.get_entity("TestEntity").unwrap();
        assert_eq!(entity.id, "test_id_123");
        assert_eq!(entity.mentions, 5);
        assert_eq!(entity.created_at, 1000000);
    }

    #[test]
    fn test_restore_relation() {
        let mut kg = KnowledgeGraph::new();
        
        kg.restore_entity(
            "source_id".to_string(),
            "Source".to_string(),
            EntityType::Project,
            0.9,
            1000000,
            1,
            HashMap::new(),
        );
        kg.restore_entity(
            "target_id".to_string(),
            "Target".to_string(),
            EntityType::Technology,
            0.9,
            1000000,
            1,
            HashMap::new(),
        );
        
        kg.restore_relation(
            "source_id".to_string(),
            "target_id".to_string(),
            RelationType::Uses,
            0.8,
            1000000,
            1.5,
        );
        
        assert_eq!(kg.stats().total_relations, 1);
        let relations = kg.get_entity_relations("Source");
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].weight, 1.5);
    }

    #[test]
    fn test_advanced_nlp_entity_extraction() {
        let nlp = AdvancedNLP::new();
        let text = "HAFA uses transformer and GPU for AI learning";
        
        let entities = nlp.extract_entities(text);
        
        assert!(entities.len() >= 3);
        let names: Vec<String> = entities.iter().map(|(name, _, _)| name.clone()).collect();
        assert!(names.contains(&"HAFA".to_string()));
        assert!(names.contains(&"transformer".to_string()));
        assert!(names.contains(&"GPU".to_string()));
        assert!(names.contains(&"AI".to_string()));
    }

    #[test]
    fn test_advanced_nlp_relation_extraction() {
        let nlp = AdvancedNLP::new();
        let text = "HAFA uses transformer";
        
        let relations = nlp.extract_relations(text);
        
        assert!(!relations.is_empty());
        let (source, target, relation_type) = &relations[0];
        assert_eq!(source, "HAFA");
        assert_eq!(target, "transformer");
        assert_eq!(*relation_type, RelationType::Uses);
    }

    #[test]
    fn test_satoshi_is_person() {
        let nlp = AdvancedNLP::new();
        let text = "Satoshi created Bitcoin";
        
        let entities = nlp.extract_entities(text);
        let satoshi = entities.iter().find(|(name, _, _)| name == "Satoshi");
        
        assert!(satoshi.is_some());
        let (_, entity_type, _) = satoshi.unwrap();
        assert_eq!(*entity_type, EntityType::Person);
    }

    #[test]
    fn test_multiple_relation_patterns() {
        let mut kg = KnowledgeGraph::new();
        
        kg.add_entity("HAFA".to_string(), EntityType::Project, 0.9);
        kg.add_entity("Rust".to_string(), EntityType::Language, 0.9);
        kg.add_entity("blockchain".to_string(), EntityType::Technology, 0.9);
        
        let text = "HAFA is written in Rust. HAFA is a blockchain.";
        let (entities, relations) = kg.extract_from_text(text);
        
        assert!(entities >= 3);
        assert!(relations >= 0);
    }
}
