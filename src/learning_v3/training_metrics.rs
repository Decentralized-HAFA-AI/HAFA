// ============================================================================
// Training Metrics & Telemetry
// ============================================================================
//
// Comprehensive tracking system for training progress including:
// - Current and EMA (Exponential Moving Average) loss tracking
// - Step counter (optimizer updates, not samples)
// - Wall time tracking for performance monitoring
//
// Design Principles:
// - Single Responsibility: Only tracks metrics, no training logic
// - Thread-safe: Can be used in concurrent training scenarios
// - Serializable: Can be saved/loaded for checkpointing
// - Robust: Handles edge cases gracefully
//
// ============================================================================

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

// ============================================================================
// TRAINING METRICS STRUCTURE
// ============================================================================

/// Tracks comprehensive training metrics with EMA smoothing
/// 
/// # Features
/// - Current loss tracking
/// - Exponential Moving Average (EMA) loss for stable monitoring
/// - Step counter for optimizer updates
/// - Wall time tracking for performance analysis
/// 
/// # Example
/// ```rust,ignoreust,ignore
/// let mut metrics = TrainingMetrics::new();
/// metrics.update_loss(5.0, 0.9);  // First update: EMA = 5.0
/// metrics.update_loss(3.0, 0.9);  // Second update: EMA = 0.9*3.0 + 0.1*5.0 = 3.2
/// metrics.increment_step();       // Step counter = 1
/// ```rust,ignore``
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Current loss value (most recent)
    pub current_loss: f32,
    
    /// Exponential Moving Average of loss (smoothed)
    pub ema_loss: f32,
    
    /// Number of optimizer steps completed
    pub steps_processed: u64,
    
    /// Timestamp when training started (milliseconds since UNIX epoch)
    /// Using u64 instead of Instant for serialization support
    pub wall_time_start_ms: u64,
    
    /// Average loss from the last completed epoch
    pub last_epoch_avg_loss: f32,
    
    /// Internal flag to track if update_loss has been called at least once
    /// This is crucial for correct EMA initialization
    #[serde(skip)]
    has_been_updated: bool,
}

impl TrainingMetrics {
    /// Creates a new TrainingMetrics instance with zeroed values
    /// 
    /// # Returns
    /// A fresh TrainingMetrics ready for tracking
    pub fn new() -> Self {
        Self {
            current_loss: 0.0,
            ema_loss: 0.0,
            steps_processed: 0,
            wall_time_start_ms: Self::current_time_ms(),
            last_epoch_avg_loss: 0.0,
            has_been_updated: false,
        }
    }

    /// Returns current time in milliseconds since UNIX epoch
    fn current_time_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_millis(0))
            .as_millis() as u64
    }

    /// Updates loss values with Exponential Moving Average (EMA) smoothing
    /// 
    /// # Arguments
    /// * `new_loss` - The latest loss value from training
    /// * `ema_alpha` - Smoothing factor in range [0.0, 1.0]
    ///   - Higher values (e.g., 0.9) give more weight to recent loss
    ///   - Lower values (e.g., 0.1) give more weight to historical average
    /// 
    /// # Behavior
    /// - First call: Sets EMA directly to new_loss (no blending)
    /// - Subsequent calls: EMA = alpha * new_loss + (1 - alpha) * old_ema
    /// 
    /// # Note
    /// This method does NOT increment steps_processed.
    /// Only optimizer steps should increment the counter via increment_step().
    /// 
    /// # Example
    /// ```rust,ignore``
    /// let mut metrics = TrainingMetrics::new();
    /// metrics.update_loss(5.0, 0.9);  // EMA = 5.0 (first update)
    /// metrics.update_loss(3.0, 0.9);  // EMA = 0.9*3.0 + 0.1*5.0 = 3.2
    /// ```rust,ignore``
    pub fn update_loss(&mut self, new_loss: f32, ema_alpha: f32) {
        // Clamp alpha to valid range to prevent numerical instability
        let alpha = ema_alpha.clamp(0.0, 1.0);
        self.current_loss = new_loss;

        if !self.has_been_updated {
            // First update: Initialize EMA directly (no blending with zero)
            self.ema_loss = new_loss;
            self.has_been_updated = true;
        } else {
            // Subsequent updates: Apply EMA formula
            // EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}
            self.ema_loss = alpha * new_loss + (1.0 - alpha) * self.ema_loss;
        }
    }

    /// Increments the step counter
    /// 
    /// Should be called after each optimizer update (not after each sample).
    /// This tracks the number of parameter updates, which is crucial for
    /// learning rate schedulers and training progress monitoring.
    pub fn increment_step(&mut self) {
        self.steps_processed += 1;
    }

    /// Increments step counter by a specified amount
    /// 
    /// Useful for batch training where multiple samples are processed at once
    pub fn increment_steps(&mut self, count: u64) {
        self.steps_processed += count;
    }

    /// Resets all metrics to initial state
    /// 
    /// Useful when starting a new training session or after loading a checkpoint
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns elapsed wall time in milliseconds since training started
    /// 
    /// # Returns
    /// Wall time in milliseconds (u64)
    /// 
    /// # Note
    /// This is useful for tracking training speed and estimating completion time
    pub fn get_wall_time_ms(&self) -> u64 {
        let current_ms = Self::current_time_ms();
        current_ms.saturating_sub(self.wall_time_start_ms)
    }

    /// Returns elapsed wall time in seconds (as f64 for precision)
    pub fn get_wall_time_secs(&self) -> f64 {
        self.get_wall_time_ms() as f64 / 1000.0
    }

    /// Calculates training speed in steps per second
    /// 
    /// # Returns
    /// Steps per second, or 0.0 if no time has elapsed
    pub fn get_steps_per_second(&self) -> f64 {
        let secs = self.get_wall_time_secs();
        if secs > 0.0 {
            self.steps_processed as f64 / secs
        } else {
            0.0
        }
    }

    /// Updates the last epoch average loss
    /// 
    /// Should be called at the end of each epoch with the average loss
    pub fn set_epoch_avg_loss(&mut self, avg_loss: f32) {
        self.last_epoch_avg_loss = avg_loss;
    }

    /// Returns a summary of current metrics for logging
    pub fn summary(&self) -> String {
        format!(
            "Loss: {:.4} | EMA: {:.4} | Steps: {} | Time: {:.2}s",
            self.current_loss,
            self.ema_loss,
            self.steps_processed,
            self.get_wall_time_secs()
        )
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_calculation() {
        let mut metrics = TrainingMetrics::new();
        
        // First update should set ema_loss directly (no blending)
        metrics.update_loss(5.0, 0.9);
        assert_eq!(metrics.ema_loss, 5.0, "First update should set EMA directly");
        
        // Second update should blend with previous EMA
        metrics.update_loss(3.0, 0.9);
        let expected = 0.9 * 3.0 + 0.1 * 5.0; // 3.2
        assert!(
            (metrics.ema_loss - expected).abs() < 1e-5,
            "EMA calculation failed: expected {}, got {}",
            expected,
            metrics.ema_loss
        );
        
        // Third update to verify continued EMA behavior
        metrics.update_loss(2.0, 0.9);
        let expected2 = 0.9 * 2.0 + 0.1 * 3.2; // 2.12
        assert!(
            (metrics.ema_loss - expected2).abs() < 1e-5,
            "Third EMA update failed: expected {}, got {}",
            expected2,
            metrics.ema_loss
        );
    }

      #[test]
    fn test_ema_alpha_clamping() {
        let mut metrics = TrainingMetrics::new();
        
        // 1. Alpha > 1.0 should be clamped to 1.0
        metrics.update_loss(5.0, 1.5); // First update: ema = 5.0
        metrics.update_loss(3.0, 1.5); // Alpha clamped to 1.0: ema = 1.0*3.0 + 0.0*5.0 = 3.0
        assert_eq!(
            metrics.ema_loss, 3.0, 
            "Alpha > 1.0 should be clamped to 1.0, making ema equal to new_loss"
        );
        
        // 2. Alpha < 0.0 should be clamped to 0.0
        // If alpha is 0, the formula is: ema = 0 * new_loss + 1.0 * old_ema
        // So the ema should NOT change from its previous value (which is 3.0)
        metrics.update_loss(10.0, -0.5); // Alpha clamped to 0.0: ema = 0*10.0 + 1.0*3.0 = 3.0
        assert_eq!(
            metrics.ema_loss, 3.0, 
            "Alpha < 0.0 should be clamped to 0.0, keeping ema unchanged"
        );
        
        metrics.update_loss(20.0, -0.5); // Alpha clamped to 0.0: ema = 0*20.0 + 1.0*3.0 = 3.0
        assert_eq!(
            metrics.ema_loss, 3.0, 
            "EMA should remain unchanged when alpha is 0"
        );
    }

    #[test]
    fn test_step_counter() {
        let mut metrics = TrainingMetrics::new();
        assert_eq!(metrics.steps_processed, 0, "Initial steps should be 0");
        
        metrics.increment_step();
        assert_eq!(metrics.steps_processed, 1, "After one increment");
        
        metrics.increment_step();
        assert_eq!(metrics.steps_processed, 2, "After two increments");
        
        metrics.increment_steps(5);
        assert_eq!(metrics.steps_processed, 7, "After batch increment");
    }

    #[test]
    fn test_reset() {
        let mut metrics = TrainingMetrics::new();
        metrics.update_loss(5.0, 0.9);
        metrics.increment_step();
        metrics.set_epoch_avg_loss(4.5);
        
        metrics.reset();
        
        assert_eq!(metrics.steps_processed, 0, "Steps should reset to 0");
        assert_eq!(metrics.current_loss, 0.0, "Current loss should reset");
        assert_eq!(metrics.ema_loss, 0.0, "EMA loss should reset");
        assert_eq!(metrics.last_epoch_avg_loss, 0.0, "Epoch loss should reset");
        assert!(!metrics.has_been_updated, "Update flag should reset");
    }

    #[test]
    fn test_wall_time() {
        let metrics = TrainingMetrics::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        let elapsed = metrics.get_wall_time_ms();
        assert!(elapsed >= 10, "Wall time should be at least 10ms, got {}", elapsed);
        
        let elapsed_secs = metrics.get_wall_time_secs();
        assert!(elapsed_secs >= 0.01, "Wall time in seconds should be at least 0.01");
    }

    #[test]
    fn test_steps_per_second() {
        let mut metrics = TrainingMetrics::new();
        
        // Initially should be 0
        assert_eq!(metrics.get_steps_per_second(), 0.0);
        
        // After some steps and time
        metrics.increment_steps(100);
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        let speed = metrics.get_steps_per_second();
        assert!(speed > 0.0, "Speed should be positive, got {}", speed);
    }

    #[test]
    fn test_summary() {
        let mut metrics = TrainingMetrics::new();
        metrics.update_loss(2.5, 0.9);
        metrics.increment_step();
        
        let summary = metrics.summary();
        assert!(summary.contains("Loss: 2.5"), "Summary should contain loss");
        assert!(summary.contains("EMA: 2.5"), "Summary should contain EMA");
        assert!(summary.contains("Steps: 1"), "Summary should contain steps");
    }

    #[test]
    fn test_serialization() {
        let mut metrics = TrainingMetrics::new();
        metrics.update_loss(2.5, 0.9);
        metrics.increment_steps(10);
        
        // Serialize to JSON
        let json = serde_json::to_string(&metrics).expect("Serialization failed");
        
        // Deserialize back
        let deserialized: TrainingMetrics = serde_json::from_str(&json)
            .expect("Deserialization failed");
        
        assert_eq!(deserialized.current_loss, metrics.current_loss);
        assert_eq!(deserialized.ema_loss, metrics.ema_loss);
        assert_eq!(deserialized.steps_processed, metrics.steps_processed);
        assert_eq!(deserialized.last_epoch_avg_loss, metrics.last_epoch_avg_loss);
    }
}