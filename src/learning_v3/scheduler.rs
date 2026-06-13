// ============================================================================
// Learning Rate Scheduler (Warmup + Cosine Decay)
// ============================================================================
//
// Implements a standard learning rate schedule used in modern Transformers:
// 1. Linear warmup from 0 to base_lr
// 2. Cosine annealing decay from base_lr to near-zero
//
// This scheduler is stateless and deterministic, making it easy to test
// and reason about.
//
// ============================================================================

use std::f32::consts::PI;

pub struct LRScheduler {
    pub base_lr: f32,
    pub warmup_steps: u32,
    pub total_steps: u32,
}

impl LRScheduler {
    pub fn new(base_lr: f32, warmup_steps: u32, total_steps: u32) -> Self {
        Self {
            base_lr: base_lr.max(1e-7),
            warmup_steps,
            total_steps: total_steps.max(1),
        }
    }

    /// Calculates the learning rate for a given step
    /// 
    /// # Arguments
    /// * `current_step` - The current optimizer step (0-indexed)
    /// 
    /// # Returns
    /// The learning rate, guaranteed to be >= 1e-7
    pub fn get_lr(&self, current_step: u32) -> f32 {
        let base = self.base_lr.max(1e-7);
        let step = current_step as f32;
        let warmup = self.warmup_steps as f32;
        let total = self.total_steps.max(self.warmup_steps + 1) as f32;

        // Phase 1: No warmup - go straight to cosine decay
        if self.warmup_steps == 0 {
            let progress = (step / total).clamp(0.0, 1.0);
            return (base * 0.5 * (1.0 + (PI * progress).cos())).max(1e-7);
        }

        // Phase 2: Linear warmup
        if step < warmup {
            let progress = (step / warmup).clamp(0.0, 1.0);
            // Ensure minimum LR even at step 0
            return (base * progress.max(1e-4)).max(1e-7);
        }

        // Phase 3: Cosine annealing decay
        let decay_steps = (total - warmup).max(1.0);
        let progress = ((step - warmup) / decay_steps).clamp(0.0, 1.0);
        (base * 0.5 * (1.0 + (PI * progress).cos())).max(1e-7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_phase() {
        let scheduler = LRScheduler::new(0.001, 100, 1000);
        
        // Step 0 should have very small LR
        let lr_0 = scheduler.get_lr(0);
        assert!(lr_0 >= 1e-7);
        assert!(lr_0 < 0.0001);
        
        // Step 50 (halfway through warmup) should be ~half of base_lr
        let lr_50 = scheduler.get_lr(50);
        assert!((lr_50 - 0.0005).abs() < 1e-5);
        
        // Step 100 (end of warmup) should be close to base_lr
        let lr_100 = scheduler.get_lr(100);
        assert!(lr_100 > 0.0009);
    }

    #[test]
    fn test_cosine_decay() {
        let scheduler = LRScheduler::new(0.001, 100, 1000);
        
        // After warmup, LR should decrease
        let lr_200 = scheduler.get_lr(200);
        let lr_500 = scheduler.get_lr(500);
        let lr_900 = scheduler.get_lr(900);
        
        assert!(lr_200 > lr_500);
        assert!(lr_500 > lr_900);
        
        // At the end, LR should be very small but not zero
        let lr_end = scheduler.get_lr(1000);
        assert!(lr_end >= 1e-7);
    }

    #[test]
    fn test_no_warmup() {
        let scheduler = LRScheduler::new(0.001, 0, 1000);
        
        // Should start with cosine decay immediately
        let lr_0 = scheduler.get_lr(0);
        let lr_500 = scheduler.get_lr(500);
        
        assert!(lr_0 > lr_500);
    }
}