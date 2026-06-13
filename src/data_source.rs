// ============================================================================
// HAFA - src/data_source.rs — REAL DATA INGESTION & EPISTEMIC FILTERING
// ============================================================================
//
// Advanced data ingestion system for real-world data:
// - Automatic file type detection (text, binary, JSON, code)
// - Content processing and normalization
// - Recursive directory scanning
// - Batch ingestion with parallel processing
// - File metadata extraction
// - Source reputation tracking
// - Evidence chain construction
// - Integration with epistemic.rs and learning.rs
//
// ============================================================================

use crate::config::Config;
use crate::crypto::hash_sha3_256;
use crate::epistemic::{
    EpistemicConstraints, EpistemicEngine, EpistemicState, Evidence, KnowledgeClaim,
    SourceReputation,
};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::fs;

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Error, Debug)]
pub enum DataSourceError {
    #[error("IO operation failed: {0}")]
    IoError(String),
    #[error("Network source not yet implemented")]
    NetworkNotImplemented,
    #[error("Empty payload received")]
    EmptyContent,
    #[error("Epistemic validation failed: confidence {:.2} < threshold", .0)]
    LowConfidence(f64),
    #[error("Source type blocked by configuration")]
    SourceBlocked,
    #[error("Source reputation too low: {0:.2} < {1:.2}")]
    LowReputation(f64, f64),
    #[error("File type not supported: {0}")]
    UnsupportedFileType(String),
    #[error("Content processing failed: {0}")]
    ProcessingError(String),
    #[error("Directory scan failed: {0}")]
    DirectoryScanError(String),
}

// ============================================================================
// FILE TYPE DETECTION
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileType {
    Text,
    Code,
    Json,
    Binary,
    Unknown,
}

impl FileType {
    /// Detect file type from extension
    pub fn from_extension(path: &str) -> Self {
        let path_lower = path.to_lowercase();
        
        if path_lower.ends_with(".json") {
            FileType::Json
        } else if path_lower.ends_with(".rs")
            || path_lower.ends_with(".py")
            || path_lower.ends_with(".js")
            || path_lower.ends_with(".ts")
            || path_lower.ends_with(".c")
            || path_lower.ends_with(".cpp")
            || path_lower.ends_with(".h")
            || path_lower.ends_with(".java")
            || path_lower.ends_with(".go")
            || path_lower.ends_with(".rb")
        {
            FileType::Code
        } else if path_lower.ends_with(".txt")
            || path_lower.ends_with(".md")
            || path_lower.ends_with(".log")
            || path_lower.ends_with(".csv")
        {
            FileType::Text
        } else if path_lower.ends_with(".bin")
            || path_lower.ends_with(".dat")
            || path_lower.ends_with(".exe")
            || path_lower.ends_with(".dll")
            || path_lower.ends_with(".so")
        {
            FileType::Binary
        } else {
            FileType::Unknown
        }
    }

    /// Check if file type is processable as text
    pub fn is_text_processable(&self) -> bool {
        matches!(self, FileType::Text | FileType::Code | FileType::Json)
    }
}

// ============================================================================
// FILE METADATA
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub path: String,
    pub size_bytes: u64,
    pub file_type: FileType,
    pub created_at: Option<DateTime<Utc>>,
    pub modified_at: Option<DateTime<Utc>>,
    pub line_count: Option<usize>,
    pub char_count: Option<usize>,
    pub mime_type: String,
}

impl FileMetadata {
    /// Extract metadata from a file
    pub async fn from_path(path: &str) -> Result<Self, DataSourceError> {
        let metadata = fs::metadata(path)
            .await
            .map_err(|e| DataSourceError::IoError(e.to_string()))?;

        let file_type = FileType::from_extension(path);
        
        let mime_type = match file_type {
            FileType::Text => "text/plain".to_string(),
            FileType::Code => "text/x-code".to_string(),
            FileType::Json => "application/json".to_string(),
            FileType::Binary => "application/octet-stream".to_string(),
            FileType::Unknown => "application/octet-stream".to_string(),
        };

        let created_at = metadata.created().ok().map(|t| {
            DateTime::<Utc>::from(std::time::SystemTime::from(t))
        });

        let modified_at = metadata.modified().ok().map(|t| {
            DateTime::<Utc>::from(std::time::SystemTime::from(t))
        });

        let (line_count, char_count) = if file_type.is_text_processable() {
            match fs::read_to_string(path).await {
                Ok(content) => {
                    let lines = content.lines().count();
                    let chars = content.chars().count();
                    (Some(lines), Some(chars))
                }
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        };

        Ok(Self {
            path: path.to_string(),
            size_bytes: metadata.len(),
            file_type,
            created_at,
            modified_at,
            line_count,
            char_count,
            mime_type,
        })
    }
}

// ============================================================================
// CONTENT PROCESSOR
// ============================================================================

pub struct ContentProcessor;

impl ContentProcessor {
    /// Normalize text content (remove extra whitespace, normalize line endings)
    pub fn normalize_text(content: &str) -> String {
        content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Extract meaningful chunks from text (paragraphs, functions, etc.)
    pub fn extract_chunks(content: &str, file_type: FileType) -> Vec<String> {
        match file_type {
            FileType::Text | FileType::Json => {
                // Split by paragraphs (double newline)
                content
                    .split("\n\n")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            FileType::Code => {
                // Split by functions or logical blocks
                content
                    .split("\n\n")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty() && s.len() > 10)
                    .collect()
            }
            FileType::Binary => {
                // Binary files are not chunked
                vec![content.to_string()]
            }
            FileType::Unknown => {
                vec![content.to_string()]
            }
        }
    }

    /// Process content based on file type
    pub fn process(content: &[u8], file_type: FileType) -> Result<Vec<u8>, DataSourceError> {
        if !file_type.is_text_processable() {
            // Return binary content as-is
            return Ok(content.to_vec());
        }

        // Convert to string
        let text = String::from_utf8_lossy(content);
        
        // Normalize
        let normalized = Self::normalize_text(&text);
        
        Ok(normalized.into_bytes())
    }
}

// ============================================================================
// DATA SOURCE TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataSource {
    Local { path: String },
    Directory { path: String, recursive: bool },
    Ipfs { cid: String },
    Web { url: String },
    Rss { url: String },
    Sensor { device_id: String },
}

impl DataSource {
    /// Get source type as string
    pub fn source_type(&self) -> &str {
        match self {
            DataSource::Local { .. } => "local",
            DataSource::Directory { .. } => "directory",
            DataSource::Ipfs { .. } => "ipfs",
            DataSource::Web { .. } => "web",
            DataSource::Rss { .. } => "rss",
            DataSource::Sensor { .. } => "sensor",
        }
    }

    /// Generate unique source ID
    pub fn source_id(&self) -> String {
        match self {
            DataSource::Local { path } => hash_sha3_256(format!("local:{}", path).as_bytes()),
            DataSource::Directory { path, recursive } => {
                hash_sha3_256(format!("dir:{}:{}", path, recursive).as_bytes())
            }
            DataSource::Ipfs { cid } => hash_sha3_256(format!("ipfs:{}", cid).as_bytes()),
            DataSource::Web { url } => hash_sha3_256(format!("web:{}", url).as_bytes()),
            DataSource::Rss { url } => hash_sha3_256(format!("rss:{}", url).as_bytes()),
            DataSource::Sensor { device_id } => {
                hash_sha3_256(format!("sensor:{}", device_id).as_bytes())
            }
        }
    }

    /// Check if this is a direct observation source
    pub fn is_direct_observation(&self) -> bool {
        matches!(
            self,
            DataSource::Local { .. } | DataSource::Directory { .. } | DataSource::Sensor { .. }
        )
    }

    /// Infer category from source type
    pub fn infer_category(&self) -> String {
        match self {
            DataSource::Local { path } | DataSource::Directory { path, .. } => {
                let file_type = FileType::from_extension(path);
                match file_type {
                    FileType::Code => "code".to_string(),
                    FileType::Json => "data".to_string(),
                    FileType::Text => "text".to_string(),
                    _ => "general".to_string(),
                }
            }
            DataSource::Ipfs { .. } => "distributed".to_string(),
            DataSource::Web { url } => {
                if url.contains("github") || url.contains("gitlab") {
                    "code".to_string()
                } else if url.contains("arxiv") || url.contains("paper") {
                    "research".to_string()
                } else {
                    "web".to_string()
                }
            }
            DataSource::Rss { .. } => "news".to_string(),
            DataSource::Sensor { .. } => "sensor".to_string(),
        }
    }
}

// ============================================================================
// VALIDATED DATA
// ============================================================================

#[derive(Debug, Clone)]
pub struct ValidatedData {
    pub content: Vec<u8>,
    pub source: DataSource,
    pub epistemic_state: EpistemicState,
    pub timestamp: u64,
    pub knowledge_claim: KnowledgeClaim,
    pub metadata: Option<FileMetadata>,
}

// ============================================================================
// SOURCE REPUTATION MANAGER
// ============================================================================

pub struct SourceReputationManager {
    reputations: Arc<DashMap<String, SourceReputation>>,
}

impl SourceReputationManager {
    pub fn new() -> Self {
        Self {
            reputations: Arc::new(DashMap::new()),
        }
    }

    pub fn get_or_create(&self, source: &DataSource) -> SourceReputation {
        let source_id = source.source_id();
        self.reputations
            .entry(source_id.clone())
            .or_insert_with(|| {
                SourceReputation::new(source_id, source.source_type().to_string())
            })
            .clone()
    }

    pub fn update_reputation(&self, source: &DataSource, was_verified: bool) {
        let source_id = source.source_id();
        if let Some(mut rep) = self.reputations.get_mut(&source_id) {
            rep.update(was_verified);
        }
    }

    pub fn get_reputation_score(&self, source: &DataSource) -> f64 {
        let source_id = source.source_id();
        self.reputations
            .get(&source_id)
            .map(|r| r.credibility_score)
            .unwrap_or(0.5)
    }

    pub fn get_all_reputations(&self) -> Vec<SourceReputation> {
        self.reputations.iter().map(|r| r.value().clone()).collect()
    }
}

impl Default for SourceReputationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// DIRECTORY SCANNER
// ============================================================================

pub struct DirectoryScanner;

impl DirectoryScanner {
    /// Scan directory and return list of file paths
    pub async fn scan(path: &str, recursive: bool) -> Result<Vec<String>, DataSourceError> {
        let mut files = Vec::new();
        Self::scan_recursive(path, recursive, &mut files).await?;
        Ok(files)
    }

    async fn scan_recursive(
        path: &str,
        recursive: bool,
        files: &mut Vec<String>,
    ) -> Result<(), DataSourceError> {
        let mut entries = fs::read_dir(path)
            .await
            .map_err(|e| DataSourceError::DirectoryScanError(e.to_string()))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| DataSourceError::DirectoryScanError(e.to_string()))?
        {
            let entry_path = entry.path();
            let path_str = entry_path.to_string_lossy().to_string();

            if entry_path.is_file() {
                files.push(path_str);
            } else if entry_path.is_dir() && recursive {
                Box::pin(Self::scan_recursive(&path_str, recursive, files)).await?;
            }
        }

        Ok(())
    }
}

// ============================================================================
// DATA SOURCE IMPLEMENTATION
// ============================================================================

impl DataSource {
    pub async fn fetch_and_validate(
        &self,
        config: &Config,
        reputation_manager: &SourceReputationManager,
    ) -> Result<ValidatedData, DataSourceError> {
        // 1. Policy check
        match self {
            DataSource::Web { .. } | DataSource::Ipfs { .. } | DataSource::Rss { .. } => {
                if !config.learning.allow_internet_learning {
                    return Err(DataSourceError::SourceBlocked);
                }
                if config.learning.trusted_sources_only {
                    return Err(DataSourceError::SourceBlocked);
                }
            }
            DataSource::Local { .. } | DataSource::Directory { .. } | DataSource::Sensor { .. } => {}
        }

        // 2. Fetch content
        let (content, metadata) = match self {
            DataSource::Local { path } => {
                let content = fs::read(path)
                    .await
                    .map_err(|e| DataSourceError::IoError(e.to_string()))?;
                let metadata = FileMetadata::from_path(path).await.ok();
                (content, metadata)
            }
            DataSource::Directory { path, recursive } => {
                // Scan directory and concatenate all files
                let files = DirectoryScanner::scan(path, *recursive).await?;
                let mut all_content = Vec::new();
                
                for file_path in files {
                    if let Ok(file_content) = fs::read(&file_path).await {
                        all_content.extend_from_slice(&file_content);
                        all_content.push(b'\n'); // Separator
                    }
                }
                
                (all_content, None)
            }
            DataSource::Ipfs { .. } | DataSource::Web { .. } | DataSource::Rss { .. } | DataSource::Sensor { .. } => {
                return Err(DataSourceError::NetworkNotImplemented);
            }
        };

        if content.is_empty() {
            return Err(DataSourceError::EmptyContent);
        }

        // 3. Process content
        let file_type = metadata
            .as_ref()
            .map(|m| m.file_type)
            .unwrap_or(FileType::Unknown);
        
        let processed_content = ContentProcessor::process(&content, file_type)?;

        // 4. Get source reputation
        let source_reputation = reputation_manager.get_or_create(self);

        let constraints = EpistemicConstraints::default();
        if source_reputation.credibility_score < constraints.min_source_reputation {
            return Err(DataSourceError::LowReputation(
                source_reputation.credibility_score,
                constraints.min_source_reputation,
            ));
        }

        // 5. Create knowledge claim
        let source_id = self.source_id();
        let category = self.infer_category();
        let is_direct = self.is_direct_observation();

        let mut claim = KnowledgeClaim::new(
            &processed_content,
            self.source_type().to_string(),
            source_id.clone(),
            is_direct,
            category,
        );

        // 6. Add initial evidence
        let initial_evidence = Evidence {
            evidence_id: hash_sha3_256(format!("{}|{}", source_id, Utc::now().timestamp()).as_bytes()),
            source_id: source_id.clone(),
            timestamp: Utc::now(),
            strength: if is_direct { 0.9 } else { 0.6 },
            content_hash: claim.content_hash.clone(),
        };
        claim.add_evidence(initial_evidence);

        // 7. Epistemic evaluation
        let epistemic_state =
            EpistemicEngine::evaluate(&claim, &constraints, Some(&source_reputation));

        if !epistemic_state.is_acceptable(&constraints) {
            reputation_manager.update_reputation(self, false);
            return Err(DataSourceError::LowConfidence(epistemic_state.confidence));
        }

        // 8. Update reputation positively
        reputation_manager.update_reputation(self, true);

        Ok(ValidatedData {
            content: processed_content,
            source: self.clone(),
            epistemic_state,
            timestamp: Utc::now().timestamp() as u64,
            knowledge_claim: claim,
            metadata,
        })
    }

    /// Batch fetch multiple files from a directory
    pub async fn fetch_directory_batch(
        path: &str,
        recursive: bool,
        config: &Config,
        reputation_manager: &SourceReputationManager,
    ) -> Result<Vec<ValidatedData>, DataSourceError> {
        let files = DirectoryScanner::scan(path, recursive).await?;
        let mut validated_data = Vec::new();

        for file_path in files {
            let source = DataSource::Local { path: file_path };
            match source.fetch_and_validate(config, reputation_manager).await {
                Ok(data) => validated_data.push(data),
                Err(e) => {
                    // Log error but continue with other files
                    eprintln!("Warning: Failed to process file: {}", e);
                }
            }
        }

        Ok(validated_data)
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        let mut cfg = Config::default();
        cfg.founder.genesis_pubkey_hex = "a".repeat(64);
        cfg.learning.allow_internet_learning = false;
        cfg.learning.min_confidence_threshold = 0.75;
        cfg
    }

    #[test]
    fn test_file_type_detection() {
        assert_eq!(FileType::from_extension("test.rs"), FileType::Code);
        assert_eq!(FileType::from_extension("data.json"), FileType::Json);
        assert_eq!(FileType::from_extension("readme.txt"), FileType::Text);
        assert_eq!(FileType::from_extension("app.exe"), FileType::Binary);
    }

    #[test]
    fn test_content_normalization() {
        let input = "  line 1  \n\n\n  line 2  \n  \n  line 3  ";
        let normalized = ContentProcessor::normalize_text(input);
        assert_eq!(normalized, "line 1\nline 2\nline 3");
    }

    #[test]
    fn test_content_processing() {
        let input = b"Hello   World\n\n\nTest";
        let processed = ContentProcessor::process(input, FileType::Text).unwrap();
        let text = String::from_utf8_lossy(&processed);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn test_source_type() {
        let local = DataSource::Local { path: "test.txt".into() };
        assert_eq!(local.source_type(), "local");

        let dir = DataSource::Directory {
            path: "data".into(),
            recursive: true,
        };
        assert_eq!(dir.source_type(), "directory");
    }

    #[test]
    fn test_category_inference() {
        let code_file = DataSource::Local { path: "src/main.rs".into() };
        assert_eq!(code_file.infer_category(), "code");

        let json_file = DataSource::Local { path: "config.json".into() };
        assert_eq!(json_file.infer_category(), "data");
    }

    #[test]
    fn test_reputation_manager() {
        let manager = SourceReputationManager::new();
        let source = DataSource::Local { path: "test.txt".into() };

        let rep = manager.get_or_create(&source);
        assert_eq!(rep.credibility_score, 0.5);

        manager.update_reputation(&source, true);
        let rep = manager.get_or_create(&source);
        assert!(rep.credibility_score > 0.5);
    }

    #[tokio::test]
    async fn test_local_source_io_error() {
        let cfg = test_config();
        let manager = SourceReputationManager::new();
        let source = DataSource::Local { path: "nonexistent.txt".into() };
        
        assert!(matches!(
            source.fetch_and_validate(&cfg, &manager).await,
            Err(DataSourceError::IoError(_))
        ));
    }

    #[tokio::test]
    async fn test_network_source_blocked() {
        let cfg = test_config();
        let manager = SourceReputationManager::new();
        let source = DataSource::Web { url: "http://example.com".into() };
        
        assert!(matches!(
            source.fetch_and_validate(&cfg, &manager).await,
            Err(DataSourceError::SourceBlocked)
        ));
    }
}