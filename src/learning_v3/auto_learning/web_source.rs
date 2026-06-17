// ============================================================================
// HAFA - src/learning_v3/auto_learning/web_source.rs
// ============================================================================
//
// Web Data Source for Auto-Learning Engine
// Fetches data from internet sources (RSS, Wikipedia, Arxiv, etc.)
//
// ============================================================================

use super::TrainingSample;
use async_trait::async_trait;
use reqwest::Client;
use std::time::Duration;
use tracing::{info, warn, error};

// ============================================================================
// WEB DATA SOURCE TRAIT
// ============================================================================

#[async_trait]
pub trait WebSource: Send + Sync {
    /// Source name
    fn name(&self) -> &str;
    
    /// Fetch new samples from this source
    async fn fetch_samples(&self) -> Result<Vec<TrainingSample>, String>;
    
    /// Get priority (lower = higher priority)
    fn priority(&self) -> u32;
}

// ============================================================================
// RSS FEED SOURCE
// ============================================================================

pub struct RSSFeedSource {
    name: String,
    feed_url: String,
    client: Client,
    priority: u32,
}

impl RSSFeedSource {
    pub fn new(name: &str, feed_url: &str, priority: u32) -> Self {
        Self {
            name: name.to_string(),
            feed_url: feed_url.to_string(),
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap(),
            priority,
        }
    }
}

#[async_trait]
impl WebSource for RSSFeedSource {
    fn name(&self) -> &str {
        &self.name
    }

    async fn fetch_samples(&self) -> Result<Vec<TrainingSample>, String> {
        info!("Fetching RSS feed: {}", self.feed_url);
        
        let response = self.client
            .get(&self.feed_url)
            .header("User-Agent", "HAFA-Learning-Engine/1.0")
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        let bytes = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read response: {}", e))?;

        let feed = feed_rs::parser::parse(&bytes[..])
            .map_err(|e| format!("Failed to parse RSS: {}", e))?;

        let mut samples = Vec::new();
        
        for entry in feed.entries.iter().take(10) {
            let title = entry.title.as_ref().map(|t| t.content.as_str()).unwrap_or("");
            let summary = entry.summary.as_ref().map(|s| s.content.as_str()).unwrap_or("");
            let content = entry.content.as_ref()
                .and_then(|c| c.body.as_ref())
                .map(|b| b.as_str())
                .unwrap_or("");

            let text = format!("{}: {} {}", title, summary, content);
            
            if text.len() > 50 {
                let sample = TrainingSample::new(
                    text,
                    format!("rss:{}", self.name),
                    0.85,
                );
                samples.push(sample);
            }
        }

        info!("Fetched {} samples from RSS: {}", samples.len(), self.name);
        Ok(samples)
    }

    fn priority(&self) -> u32 {
        self.priority
    }
}

// ============================================================================
// WIKIPEDIA SOURCE
// ============================================================================

pub struct WikipediaSource {
    client: Client,
    priority: u32,
}

impl WikipediaSource {
    pub fn new(priority: u32) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap(),
            priority,
        }
    }

    async fn fetch_random_article(&self) -> Result<String, String> {
        let url = "https://en.wikipedia.org/api/rest_v1/page/random/summary";
        
        let response = self.client
            .get(url)
            .header("User-Agent", "HAFA-Learning-Engine/1.0")
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let title = json["title"].as_str().unwrap_or("");
        let extract = json["extract"].as_str().unwrap_or("");

        Ok(format!("Wikipedia - {}: {}", title, extract))
    }
}

#[async_trait]
impl WebSource for WikipediaSource {
    fn name(&self) -> &str {
        "wikipedia"
    }

    async fn fetch_samples(&self) -> Result<Vec<TrainingSample>, String> {
        info!("Fetching random Wikipedia articles");
        
        let mut samples = Vec::new();
        
        for _ in 0..5 {
            match self.fetch_random_article().await {
                Ok(text) => {
                    if text.len() > 100 {
                        let sample = TrainingSample::new(
                            text,
                            "wikipedia:random".to_string(),
                            0.90,
                        );
                        samples.push(sample);
                    }
                }
                Err(e) => {
                    warn!("Failed to fetch Wikipedia article: {}", e);
                }
            }
        }

        info!("Fetched {} samples from Wikipedia", samples.len());
        Ok(samples)
    }

    fn priority(&self) -> u32 {
        self.priority
    }
}

// ============================================================================
// ARXIV SOURCE
// ============================================================================

pub struct ArxivSource {
    query: String,
    client: Client,
    priority: u32,
}

impl ArxivSource {
    pub fn new(query: &str, priority: u32) -> Self {
        Self {
            query: query.to_string(),
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap(),
            priority,
        }
    }
}

#[async_trait]
impl WebSource for ArxivSource {
    fn name(&self) -> &str {
        "arxiv"
    }

    async fn fetch_samples(&self) -> Result<Vec<TrainingSample>, String> {
        info!("Fetching Arxiv papers for query: {}", self.query);
        
        let url = format!(
            "http://export.arxiv.org/api/query?search_query={}&max_results=5&sortBy=submittedDate",
            self.query
        );

        let response = self.client
            .get(&url)
            .header("User-Agent", "HAFA-Learning-Engine/1.0")
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        let text = response
            .text()
            .await
            .map_err(|e| format!("Failed to read response: {}", e))?;

        // Parse Atom feed (simplified)
        let mut samples = Vec::new();
        
        // Extract titles and summaries (basic parsing)
        let lines: Vec<&str> = text.lines().collect();
        let mut current_title = String::new();
let mut current_summary;
        
        for line in lines {
            if line.contains("<title>") && !line.contains("ArXiv") {
                current_title = line
                    .replace("<title>", "")
                    .replace("</title>", "")
                    .trim()
                    .to_string();
            } else if line.contains("<summary>") {
                current_summary = line
                    .replace("<summary>", "")
                    .replace("</summary>", "")
                    .trim()
                    .to_string();
                
                if !current_title.is_empty() && !current_summary.is_empty() {
                    let text = format!("Arxiv Paper - {}: {}", current_title, current_summary);
                    
                    if text.len() > 100 {
                        let sample = TrainingSample::new(
                            text,
                            format!("arxiv:{}", self.query),
                            0.95,
                        );
                        samples.push(sample);
                    }
                    
                    current_title.clear();
                    current_summary.clear();
                }
            }
        }

        info!("Fetched {} samples from Arxiv", samples.len());
        Ok(samples)
    }

    fn priority(&self) -> u32 {
        self.priority
    }
}

// ============================================================================
// WEB DATA SOURCE MANAGER
// ============================================================================

pub struct WebDataSourceManager {
    sources: Vec<Box<dyn WebSource>>,
    last_fetch: std::time::Instant,
    fetch_interval: Duration,
}

impl WebDataSourceManager {
    pub fn new(fetch_interval_secs: u64) -> Self {
        Self {
            sources: Vec::new(),
            last_fetch: std::time::Instant::now() - Duration::from_secs(fetch_interval_secs),
            fetch_interval: Duration::from_secs(fetch_interval_secs),
        }
    }

    pub fn add_source(&mut self, source: Box<dyn WebSource>) {
        self.sources.push(source);
        self.sources.sort_by_key(|s| s.priority());
    }

    pub async fn fetch_all(&mut self) -> Vec<TrainingSample> {
        if self.last_fetch.elapsed() < self.fetch_interval {
            return Vec::new();
        }

        info!("Fetching from all web sources");
        let mut all_samples = Vec::new();

        for source in &self.sources {
            match source.fetch_samples().await {
                Ok(samples) => {
                    info!("Fetched {} samples from {}", samples.len(), source.name());
                    all_samples.extend(samples);
                }
                Err(e) => {
                    error!("Failed to fetch from {}: {}", source.name(), e);
                }
            }
        }

        self.last_fetch = std::time::Instant::now();
        info!("Total fetched: {} samples from web", all_samples.len());
        all_samples
    }
}