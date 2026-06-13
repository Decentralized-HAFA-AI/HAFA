// ============================================================================
// Reasoning Engine: Query and Inference over Knowledge Graph
// ============================================================================

use std::collections::{HashSet, VecDeque};
use super::knowledge_graph::{KnowledgeGraph, RelationType};

/// Query result from reasoning engine
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub query: String,
    pub answer: String,
    pub confidence: f32,
    pub entities_found: Vec<String>,
    pub relations_found: Vec<String>,
    pub inference_path: Vec<String>,
}

/// Reasoning Engine for Knowledge Graph
pub struct ReasoningEngine;

impl ReasoningEngine {
    pub fn new() -> Self {
        println!("   [REASONING] 🧠 Initializing Reasoning Engine");
        Self
    }

    /// Answer a natural language query about the knowledge graph
    pub fn query(&self, kg: &KnowledgeGraph, query: &str) -> QueryResult {
        let query_lower = query.to_lowercase();
        
        // IMPORTANT: Check specific patterns BEFORE general "what is" pattern
        
      // Pattern 1: "What is X written in?" - Language lookup (MUST be first!)
if query_lower.contains("written in") {
    return self.answer_written_in(kg, query);
}
        
        // Pattern 2: "What does X use?" or "What X uses?" - Relation lookup
        if query_lower.contains(" use") || query_lower.contains(" uses ") {
            return self.answer_what_uses(kg, query);
        }
        
        // Pattern 3: "Who created X?" - Creator lookup
        if query_lower.contains(" created ") || query_lower.contains("who created") {
            return self.answer_who_created(kg, query);
        }
        
        // Pattern 4: "What is X?" - Entity lookup (AFTER specific patterns)
        if query_lower.starts_with("what is ") || query_lower.starts_with("what's ") {
            return self.answer_what_is(kg, query);
        }
        
        // Pattern 5: "Tell me about X" - General info
        if query_lower.starts_with("tell me about ") || query_lower.starts_with("about ") {
            return self.answer_tell_me_about(kg, query);
        }
        
        // Pattern 6: "What are the relations of X?" - Graph traversal
        if query_lower.contains("relations") || query_lower.contains("connected") {
            return self.answer_relations(kg, query);
        }
        
        // Pattern 7: "How is X related to Y?" - Path finding
        if query_lower.contains(" related to ") || query_lower.contains("relation between") {
            return self.answer_how_related(kg, query);
        }
        
        // Default: Search for entities matching query
        self.answer_general_search(kg, query)
    }
    
    /// Answer "What is X?" queries
    fn answer_what_is(&self, kg: &KnowledgeGraph, query: &str) -> QueryResult {
        let query_lower = query.to_lowercase();
        let entity_name = query_lower
            .trim_start_matches("what is ")
            .trim_start_matches("what's ")
            .trim_end_matches('?')
            .trim()
            .to_string();
        
        if let Some(entity) = kg.get_entity(&entity_name) {
            let relations = kg.get_entity_relations(&entity_name);
            let mut description = format!(
                "{} is a {:?} with confidence {:.2} (mentioned {} times).",
                entity.name, entity.entity_type, entity.confidence, entity.mentions
            );
            
            if !relations.is_empty() {
                description.push_str("\n\nRelations:");
                for rel in relations.iter().take(5) {
                    let other_name = if rel.source_id == entity.id {
                        self.find_entity_name_by_id(kg, &rel.target_id)
                    } else {
                        self.find_entity_name_by_id(kg, &rel.source_id)
                    };
                    
                    if let Some(other) = other_name {
                        let direction = if rel.source_id == entity.id { "→" } else { "←" };
                        description.push_str(&format!(
                            "\n  {} {} {} ({:?})",
                            direction, rel.relation_type.as_str(), other, rel.relation_type
                        ));
                    }
                }
            }
            
            QueryResult {
                query: query.to_string(),
                answer: description,
                confidence: entity.confidence,
                entities_found: vec![entity.name.clone()],
                relations_found: relations.iter().map(|r| r.id.clone()).collect(),
                inference_path: vec![format!("Direct lookup: {}", entity.name)],
            }
        } else {
            QueryResult {
                query: query.to_string(),
                answer: format!("I don't have information about '{}' in my knowledge graph.", entity_name),
                confidence: 0.0,
                entities_found: vec![],
                relations_found: vec![],
                inference_path: vec![],
            }
        }
    }
    
    /// Answer "What does X use?" queries
    fn answer_what_uses(&self, kg: &KnowledgeGraph, query: &str) -> QueryResult {
        let query_lower = query.to_lowercase();
        let entity_name = self.extract_entity_from_uses_query(&query_lower);
        
        if let Some(entity) = kg.get_entity(&entity_name) {
            let relations = kg.get_entity_relations(&entity_name);
            let uses_relations: Vec<_> = relations.iter()
                .filter(|r| r.relation_type == RelationType::Uses && r.source_id == entity.id)
                .collect();
            
            if uses_relations.is_empty() {
                return QueryResult {
                    query: query.to_string(),
                    answer: format!("I don't have information about what {} uses.", entity.name),
                    confidence: 0.0,
                    entities_found: vec![entity.name.clone()],
                    relations_found: vec![],
                    inference_path: vec![],
                };
            }
            
            let mut targets = Vec::new();
            for rel in &uses_relations {
                if let Some(name) = self.find_entity_name_by_id(kg, &rel.target_id) {
                    targets.push(name);
                }
            }
            
            QueryResult {
                query: query.to_string(),
                answer: format!("{} uses: {}", entity.name, targets.join(", ")),
                confidence: 0.8,
                entities_found: targets.clone(),
                relations_found: uses_relations.iter().map(|r| r.id.clone()).collect(),
                inference_path: vec![format!("{} --Uses--> {:?}", entity.name, targets)],
            }
        } else {
            QueryResult {
                query: query.to_string(),
                answer: format!("Entity '{}' not found in knowledge graph.", entity_name),
                confidence: 0.0,
                entities_found: vec![],
                relations_found: vec![],
                inference_path: vec![],
            }
        }
    }
    
    /// Answer "Who created X?" queries
    fn answer_who_created(&self, kg: &KnowledgeGraph, query: &str) -> QueryResult {
        let query_lower = query.to_lowercase();
        let entity_name = self.extract_entity_from_created_query(&query_lower);
        
        if let Some(entity) = kg.get_entity(&entity_name) {
            let relations = kg.get_entity_relations(&entity_name);
            let created_relations: Vec<_> = relations.iter()
                .filter(|r| r.relation_type == RelationType::CreatedBy && r.target_id == entity.id)
                .collect();
            
            if created_relations.is_empty() {
                return QueryResult {
                    query: query.to_string(),
                    answer: format!("I don't have information about who created {}.", entity.name),
                    confidence: 0.0,
                    entities_found: vec![entity.name.clone()],
                    relations_found: vec![],
                    inference_path: vec![],
                };
            }
            
            let mut creators = Vec::new();
            for rel in &created_relations {
                if let Some(name) = self.find_entity_name_by_id(kg, &rel.source_id) {
                    creators.push(name);
                }
            }
            
            QueryResult {
                query: query.to_string(),
                answer: format!("{} was created by: {}", entity.name, creators.join(", ")),
                confidence: 0.85,
                entities_found: creators.clone(),
                relations_found: created_relations.iter().map(|r| r.id.clone()).collect(),
                inference_path: vec![format!("{:?} --CreatedBy--> {}", creators, entity.name)],
            }
        } else {
            QueryResult {
                query: query.to_string(),
                answer: format!("Entity '{}' not found.", entity_name),
                confidence: 0.0,
                entities_found: vec![],
                relations_found: vec![],
                inference_path: vec![],
            }
        }
    }
    
    /// Answer "What is X written in?" queries
    fn answer_written_in(&self, kg: &KnowledgeGraph, query: &str) -> QueryResult {
        let query_lower = query.to_lowercase();
        let entity_name = self.extract_entity_from_written_in_query(&query_lower);
        
        if let Some(entity) = kg.get_entity(&entity_name) {
            let relations = kg.get_entity_relations(&entity_name);
            let written_relations: Vec<_> = relations.iter()
                .filter(|r| r.relation_type == RelationType::WrittenIn && r.source_id == entity.id)
                .collect();
            
            if written_relations.is_empty() {
                return QueryResult {
                    query: query.to_string(),
                    answer: format!("I don't have information about what {} is written in.", entity.name),
                    confidence: 0.0,
                    entities_found: vec![entity.name.clone()],
                    relations_found: vec![],
                    inference_path: vec![],
                };
            }
            
            let mut languages = Vec::new();
            for rel in &written_relations {
                if let Some(name) = self.find_entity_name_by_id(kg, &rel.target_id) {
                    languages.push(name);
                }
            }
            
            QueryResult {
                query: query.to_string(),
                answer: format!("{} is written in: {}", entity.name, languages.join(", ")),
                confidence: 0.9,
                entities_found: languages.clone(),
                relations_found: written_relations.iter().map(|r| r.id.clone()).collect(),
                inference_path: vec![format!("{} --WrittenIn--> {:?}", entity.name, languages)],
            }
        } else {
            QueryResult {
                query: query.to_string(),
                answer: format!("Entity '{}' not found.", entity_name),
                confidence: 0.0,
                entities_found: vec![],
                relations_found: vec![],
                inference_path: vec![],
            }
        }
    }
    
    /// Answer "Tell me about X" queries
    fn answer_tell_me_about(&self, kg: &KnowledgeGraph, query: &str) -> QueryResult {
        let query_lower = query.to_lowercase();
        let entity_name = query_lower
            .trim_start_matches("tell me about ")
            .trim_start_matches("about ")
            .trim_end_matches('.')
            .trim()
            .to_string();
        
        if let Some(entity) = kg.get_entity(&entity_name) {
            let relations = kg.get_entity_relations(&entity_name);
            
            let mut answer = format!("📊 **{}**\n", entity.name);
            answer.push_str(&format!("• Type: {:?}\n", entity.entity_type));
            answer.push_str(&format!("• Confidence: {:.2}\n", entity.confidence));
            answer.push_str(&format!("• Mentions: {}\n", entity.mentions));
            
            if !relations.is_empty() {
                answer.push_str(&format!("\n🔗 Relations ({}):\n", relations.len()));
                for rel in relations.iter().take(10) {
                    let other = if rel.source_id == entity.id {
                        self.find_entity_name_by_id(kg, &rel.target_id)
                    } else {
                        self.find_entity_name_by_id(kg, &rel.source_id)
                    };
                    
                    if let Some(other_name) = other {
                        let direction = if rel.source_id == entity.id { "→" } else { "←" };
                        answer.push_str(&format!(
                            "  {} {} {} (weight: {:.2})\n",
                            direction, rel.relation_type.as_str(), other_name, rel.weight
                        ));
                    }
                }
            }
            
            QueryResult {
                query: query.to_string(),
                answer,
                confidence: entity.confidence,
                entities_found: vec![entity.name.clone()],
                relations_found: relations.iter().map(|r| r.id.clone()).collect(),
                inference_path: vec![format!("Full entity profile: {}", entity.name)],
            }
        } else {
            QueryResult {
                query: query.to_string(),
                answer: format!("I don't have information about '{}' in my knowledge graph.", entity_name),
                confidence: 0.0,
                entities_found: vec![],
                relations_found: vec![],
                inference_path: vec![],
            }
        }
    }
    
    /// Answer "What are the relations of X?" queries
    fn answer_relations(&self, kg: &KnowledgeGraph, query: &str) -> QueryResult {
        let query_lower = query.to_lowercase();
        let entity_name = self.extract_entity_from_query(&query_lower);
        
        if let Some(entity) = kg.get_entity(&entity_name) {
            let relations = kg.get_entity_relations(&entity_name);
            
            if relations.is_empty() {
                return QueryResult {
                    query: query.to_string(),
                    answer: format!("{} has no relations in the knowledge graph.", entity.name),
                    confidence: 0.0,
                    entities_found: vec![entity.name.clone()],
                    relations_found: vec![],
                    inference_path: vec![],
                };
            }
            
            let mut answer = format!("{} has {} relations:\n", entity.name, relations.len());
            for rel in relations.iter().take(15) {
                let other = if rel.source_id == entity.id {
                    self.find_entity_name_by_id(kg, &rel.target_id)
                } else {
                    self.find_entity_name_by_id(kg, &rel.source_id)
                };
                
                if let Some(other_name) = other {
                    let direction = if rel.source_id == entity.id { "→" } else { "←" };
                    answer.push_str(&format!(
                        "  {} {} {}\n",
                        direction, rel.relation_type.as_str(), other_name
                    ));
                }
            }
            
            QueryResult {
                query: query.to_string(),
                answer,
                confidence: 0.8,
                entities_found: vec![entity.name.clone()],
                relations_found: relations.iter().map(|r| r.id.clone()).collect(),
                inference_path: vec![format!("Graph traversal from: {}", entity.name)],
            }
        } else {
            QueryResult {
                query: query.to_string(),
                answer: format!("Entity '{}' not found.", entity_name),
                confidence: 0.0,
                entities_found: vec![],
                relations_found: vec![],
                inference_path: vec![],
            }
        }
    }
    
    /// Answer "How is X related to Y?" queries using BFS
    fn answer_how_related(&self, kg: &KnowledgeGraph, query: &str) -> QueryResult {
        let query_lower = query.to_lowercase();
        
        let parts: Vec<&str> = query_lower
            .split(|c| c == '?' || c == '.')
            .next()
            .unwrap_or("")
            .split(" related to ")
            .collect();
        
        if parts.len() < 2 {
            let parts2: Vec<&str> = query_lower
                .split("relation between ")
                .collect();
            if parts2.len() >= 2 {
                let entities: Vec<&str> = parts2[1].split(" and ").collect();
                if entities.len() >= 2 {
                    return self.find_path(kg, entities[0].trim(), entities[1].trim(), query);
                }
            }
            return QueryResult {
                query: query.to_string(),
                answer: "Please specify two entities to find their relationship.".to_string(),
                confidence: 0.0,
                entities_found: vec![],
                relations_found: vec![],
                inference_path: vec![],
            };
        }
        
        let entity_a = parts[0].split_whitespace().last().unwrap_or("").trim();
        let entity_b = parts[1].trim();
        
        self.find_path(kg, entity_a, entity_b, query)
    }
    
    /// BFS path finding between two entities
    fn find_path(&self, kg: &KnowledgeGraph, start_name: &str, end_name: &str, query: &str) -> QueryResult {
        let start = kg.get_entity(start_name);
        let end = kg.get_entity(end_name);
        
        if start.is_none() || end.is_none() {
            return QueryResult {
                query: query.to_string(),
                answer: format!("One or both entities not found: '{}' or '{}'", start_name, end_name),
                confidence: 0.0,
                entities_found: vec![],
                relations_found: vec![],
                inference_path: vec![],
            };
        }
        
        let start = start.unwrap();
        let end = end.unwrap();
        
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();
        
        queue.push_back((start.id.clone(), vec![start.name.clone()]));
        visited.insert(start.id.clone());
        
        while let Some((current_id, path)) = queue.pop_front() {
            if current_id == end.id {
                let path_str = path.join(" → ");
                return QueryResult {
                    query: query.to_string(),
                    answer: format!("Path from {} to {}:\n{}", start.name, end.name, path_str),
                    confidence: 0.9,
                    entities_found: path.clone(),
                    relations_found: vec![],
                    inference_path: vec![format!("BFS path: {}", path_str)],
                };
            }
            
            if let Some(current_name) = self.find_entity_name_by_id(kg, &current_id) {
                let relations = kg.get_entity_relations(&current_name);
                
                for rel in relations {
                    let next_id = if rel.source_id == current_id {
                        &rel.target_id
                    } else {
                        &rel.source_id
                    };
                    
                    if !visited.contains(next_id) {
                        visited.insert(next_id.clone());
                        if let Some(next_name) = self.find_entity_name_by_id(kg, next_id) {
                            let mut new_path = path.clone();
                            new_path.push(format!("{} ({:?})", next_name, rel.relation_type));
                            queue.push_back((next_id.clone(), new_path));
                        }
                    }
                }
            }
        }
        
        QueryResult {
            query: query.to_string(),
            answer: format!("No path found between '{}' and '{}'.", start.name, end.name),
            confidence: 0.0,
            entities_found: vec![start.name.clone(), end.name.clone()],
            relations_found: vec![],
            inference_path: vec![],
        }
    }
    
    /// General search when no pattern matches
    fn answer_general_search(&self, kg: &KnowledgeGraph, query: &str) -> QueryResult {
        let query_lower = query.to_lowercase();
        let mut found_entities = Vec::new();
        
        for entity in kg.entities() {
            if entity.name.to_lowercase().contains(&query_lower) {
                found_entities.push(entity.name.clone());
            }
        }
        
        if found_entities.is_empty() {
            QueryResult {
                query: query.to_string(),
                answer: format!("No matches found for '{}' in knowledge graph.", query),
                confidence: 0.0,
                entities_found: vec![],
                relations_found: vec![],
                inference_path: vec![],
            }
        } else {
            let answer = format!(
                "Found {} entities matching '{}': {}",
                found_entities.len(),
                query,
                found_entities.join(", ")
            );
            
            QueryResult {
                query: query.to_string(),
                answer,
                confidence: 0.7,
                entities_found: found_entities.clone(),
                relations_found: vec![],
                inference_path: vec![format!("Search: {}", query)],
            }
        }
    }
    
    /// Helper: Extract entity name from "uses" query
    fn extract_entity_from_uses_query(&self, query: &str) -> String {
        let cleaned = query
            .replace("what does ", "")
            .replace("what do ", "")
            .replace(" use?", "")
            .replace(" uses?", "")
            .replace(" use", "")
            .replace(" uses", "")
            .replace("?", "")
            .trim()
            .to_string();
        
        cleaned.split_whitespace().next().unwrap_or("").to_string()
    }
    
    /// Helper: Extract entity name from "created" query
    fn extract_entity_from_created_query(&self, query: &str) -> String {
        let cleaned = query
            .replace("who created ", "")
            .replace(" created by ", " ")
            .replace(" created ", " ")
            .replace("?", "")
            .trim()
            .to_string();
        
        cleaned.split_whitespace().next().unwrap_or("").to_string()
    }
    
    /// Helper: Extract entity name from "written in" query
    fn extract_entity_from_written_in_query(&self, query: &str) -> String {
        let cleaned = query
            .replace("what is ", "")
            .replace(" written in?", "")
            .replace(" written in", "")
            .replace("?", "")
            .trim()
            .to_string();
        
        cleaned.split_whitespace().next().unwrap_or("").to_string()
    }
    
    /// Helper: Extract entity name from general query
fn extract_entity_from_query(&self, query: &str) -> String {
    let cleaned = query
        .replace("what are the relations of ", "")
        .replace("what are the ", "")
        .replace("what does ", "")
        .replace("what do ", "")
        .replace("what is ", "")
        .replace("who ", "")
        .replace("where ", "")
        .replace("when ", "")
        .replace("why ", "")
        .replace("how ", "")
        .replace(" is ", " ")
        .replace(" use ", " ")
        .replace(" uses ", " ")
        .replace(" created ", " ")
        .replace(" written in ", " ")
        .replace(" related to ", " ")
        .replace("?", "")
        .replace(".", "")
        .trim()
        .to_string();
    
    cleaned.split_whitespace().next().unwrap_or("").to_string()
}
    
    /// Helper: Find entity name by ID
    fn find_entity_name_by_id(&self, kg: &KnowledgeGraph, id: &str) -> Option<String> {
        for entity in kg.entities() {
            if entity.id == id {
                return Some(entity.name.clone());
            }
        }
        None
    }
}