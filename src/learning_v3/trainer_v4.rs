// ============================================================================
// Trainer V4: Production-Grade Learning Engine
// ============================================================================
//
// Full-Model Training with:
// - Gradient accumulation
// - AdamW optimizer
// - Recursive backward pass (Attention + FFN + LayerNorm + pred_head)
// - Verifiable cognitive proof
// - NEW: Backend abstraction for device management (CPU/GPU ready)
//
// ============================================================================

use rand::{seq::SliceRandom, thread_rng};

use super::{TransformerEngine, TransformerConfig};
use super::loss::{LossFunction, LossType};
use super::optimizers::adamw::AdamW;
use super::dataset_stream::StreamingDataset;
use super::training_metrics::TrainingMetrics;
use super::cognitive_proof::CognitiveProofV4;
use super::scheduler::LRScheduler;
use super::telemetry::SystemTelemetry;
use super::checkpoint::ModelCheckpoint;
use super::gradient_bucket::GradientBucket;
use super::tensor::Tensor;
use super::backend::{Backend, CpuBackend};  // NEW: Backend integration

pub struct TrainerV4 {
    pub last_gradient_hash: String,
    pub model: TransformerEngine,
    pub loss_fn: LossFunction,
    pub optimizer: AdamW,
    
    // NEW: Backend for device management (CPU/GPU abstraction)
    pub backend: Box<dyn Backend>,

    pub accumulation_steps: u32,
    pub current_step_in_batch: u32,

    pub accumulated_grad: GradientBucket,
    pub m_buffers: Vec<ndarray::Array2<f32>>,
    pub v_buffers: Vec<ndarray::Array2<f32>>,

    pub scheduler: LRScheduler,
    pub metrics: TrainingMetrics,
    pub telemetry: SystemTelemetry,
}

impl TrainerV4 {
    pub fn new(
        config: &TransformerConfig,
        lr: f32,
        warmup_steps: u32,
        total_steps: u32,
        weight_decay: f32,
        accumulation_steps: u32,
    ) -> Self {
        let model = TransformerEngine::new(config);
        let accumulated_grad = GradientBucket::new_from_model(&model);
        
        let m_buffers: Vec<ndarray::Array2<f32>> = accumulated_grad.gradients.iter()
            .map(|g| ndarray::Array2::zeros(g.shape))
            .collect();
        let v_buffers: Vec<ndarray::Array2<f32>> = accumulated_grad.gradients.iter()
            .map(|g| ndarray::Array2::zeros(g.shape))
            .collect();

        // NEW: Initialize Backend (CPU for now, GPU-ready architecture)
        let backend: Box<dyn Backend> = Box::new(CpuBackend::new());

        Self {
            model,
            loss_fn: LossFunction::new(LossType::CrossEntropy),
            optimizer: AdamW::new_with_params(lr, weight_decay, 5.0),
            backend,  // NEW
            accumulation_steps: accumulation_steps.max(1),
            current_step_in_batch: 0,
            accumulated_grad,
            m_buffers,
            v_buffers,
            scheduler: LRScheduler::new(lr, warmup_steps, total_steps),
            metrics: TrainingMetrics::new(),
            telemetry: SystemTelemetry::new(),
            last_gradient_hash: String::new(),
        }
    }
    
    /// NEW: Create TrainerV4 with a custom backend (for GPU support in the future)
    pub fn with_backend(
        config: &TransformerConfig,
        lr: f32,
        warmup_steps: u32,
        total_steps: u32,
        weight_decay: f32,
        accumulation_steps: u32,
        backend: Box<dyn Backend>,
    ) -> Self {
        let model = TransformerEngine::new(config);
        let accumulated_grad = GradientBucket::new_from_model(&model);
        
        let m_buffers: Vec<ndarray::Array2<f32>> = accumulated_grad.gradients.iter()
            .map(|g| ndarray::Array2::zeros(g.shape))
            .collect();
        let v_buffers: Vec<ndarray::Array2<f32>> = accumulated_grad.gradients.iter()
            .map(|g| ndarray::Array2::zeros(g.shape))
            .collect();

        println!("   [TRAINER] 🖥️  Using backend: {} ({})", 
                 backend.name(), backend.info().device_name);

        Self {
            model,
            loss_fn: LossFunction::new(LossType::CrossEntropy),
            optimizer: AdamW::new_with_params(lr, weight_decay, 5.0),
            backend,
            accumulation_steps: accumulation_steps.max(1),
            current_step_in_batch: 0,
            accumulated_grad,
            m_buffers,
            v_buffers,
            scheduler: LRScheduler::new(lr, warmup_steps, total_steps),
            metrics: TrainingMetrics::new(),
            telemetry: SystemTelemetry::new(),
            last_gradient_hash: String::new(),
        }
    }

    /// Applies AdamW update to a specific parameter using index-based access
    fn apply_adamw_by_index(&mut self, param_idx: usize) {
        let grad_data = self.accumulated_grad.gradients[param_idx].data.clone();
        let param_name = self.accumulated_grad.gradients[param_idx].name.clone();
        
        let mut m_new = self.m_buffers[param_idx].clone();
        let mut v_new = self.v_buffers[param_idx].clone();
        
        // Apply AdamW based on parameter name
        if param_name == "pred_head" {
            self.optimizer.step(
                &mut self.model.stack.pred_head,
                &grad_data,
                &mut m_new,
                &mut v_new,
            );
        } else if param_name == "final_ln_gamma" {
            // Handle final LayerNorm gamma (Array1 -> Array2 -> Array1)
            let mut gamma_2d = self.model.stack.final_ln_gamma.clone().insert_axis(ndarray::Axis(0));
            self.optimizer.step(&mut gamma_2d, &grad_data, &mut m_new, &mut v_new);
            self.model.stack.final_ln_gamma = gamma_2d.remove_axis(ndarray::Axis(0));
        } else if param_name == "final_ln_beta" {
            let mut beta_2d = self.model.stack.final_ln_beta.clone().insert_axis(ndarray::Axis(0));
            self.optimizer.step(&mut beta_2d, &grad_data, &mut m_new, &mut v_new);
            self.model.stack.final_ln_beta = beta_2d.remove_axis(ndarray::Axis(0));
        } else if param_name.starts_with("layer_") {
            // Parse layer index and parameter type
            if let Some(dot_pos) = param_name.find('.') {
                let layer_part = &param_name[..dot_pos];
                let param_type = &param_name[dot_pos + 1..];
                
                if let Ok(layer_idx) = layer_part.replace("layer_", "").parse::<usize>() {
                    if layer_idx < self.model.stack.blocks.len() {
                        let block = &mut self.model.stack.blocks[layer_idx];
                        
                        match param_type {
                            // Attention weights (Q, K, V, O projections)
                            "w_q" => {
                                self.optimizer.step(&mut block.attention.w_q, &grad_data, &mut m_new, &mut v_new);
                            }
                            "w_k" => {
                                self.optimizer.step(&mut block.attention.w_k, &grad_data, &mut m_new, &mut v_new);
                            }
                            "w_v" => {
                                self.optimizer.step(&mut block.attention.w_v, &grad_data, &mut m_new, &mut v_new);
                            }
                            "w_o" => {
                                self.optimizer.step(&mut block.attention.w_o, &grad_data, &mut m_new, &mut v_new);
                            }
                            
                            // Feed-Forward Network weights and biases
                            "ffn1_weight" => {
                                self.optimizer.step(&mut block.ff_w1, &grad_data, &mut m_new, &mut v_new);
                            }
                            "ffn1_bias" => {
                                let mut bias_2d = block.ff_b1.clone().insert_axis(ndarray::Axis(0));
                                self.optimizer.step(&mut bias_2d, &grad_data, &mut m_new, &mut v_new);
                                block.ff_b1 = bias_2d.remove_axis(ndarray::Axis(0));
                            }
                            "ffn2_weight" => {
                                self.optimizer.step(&mut block.ff_w2, &grad_data, &mut m_new, &mut v_new);
                            }
                            "ffn2_bias" => {
                                let mut bias_2d = block.ff_b2.clone().insert_axis(ndarray::Axis(0));
                                self.optimizer.step(&mut bias_2d, &grad_data, &mut m_new, &mut v_new);
                                block.ff_b2 = bias_2d.remove_axis(ndarray::Axis(0));
                            }
                            
                            // LayerNorm learnable parameters
                            "ln1_gamma" => {
                                let mut gamma_2d = block.ln1_gamma.clone().insert_axis(ndarray::Axis(0));
                                self.optimizer.step(&mut gamma_2d, &grad_data, &mut m_new, &mut v_new);
                                block.ln1_gamma = gamma_2d.remove_axis(ndarray::Axis(0));
                            }
                            "ln1_beta" => {
                                let mut beta_2d = block.ln1_beta.clone().insert_axis(ndarray::Axis(0));
                                self.optimizer.step(&mut beta_2d, &grad_data, &mut m_new, &mut v_new);
                                block.ln1_beta = beta_2d.remove_axis(ndarray::Axis(0));
                            }
                            "ln2_gamma" => {
                                let mut gamma_2d = block.ln2_gamma.clone().insert_axis(ndarray::Axis(0));
                                self.optimizer.step(&mut gamma_2d, &grad_data, &mut m_new, &mut v_new);
                                block.ln2_gamma = gamma_2d.remove_axis(ndarray::Axis(0));
                            }
                            "ln2_beta" => {
                                let mut beta_2d = block.ln2_beta.clone().insert_axis(ndarray::Axis(0));
                                self.optimizer.step(&mut beta_2d, &grad_data, &mut m_new, &mut v_new);
                                block.ln2_beta = beta_2d.remove_axis(ndarray::Axis(0));
                            }
                            
                            _ => {
                                // Unsupported parameter type
                                // Will be handled in future iterations if needed
                            }
                        }
                    }
                }
            }
        }
        
        // Update momentum buffers
        self.m_buffers[param_idx] = m_new;
        self.v_buffers[param_idx] = v_new;
    }

    fn flush_accumulated_grad(&mut self) {
        if self.current_step_in_batch == 0 {
            return;
        }

        // 1. Get learning rate from scheduler
        let lr = self.scheduler.get_lr(self.metrics.steps_processed as u32);
        self.optimizer.learning_rate = lr;

        // 2. Average accumulated gradients
        self.accumulated_grad.divide(self.current_step_in_batch as f32);

        // 3. Apply global gradient clipping
        self.accumulated_grad.clip_by_global_norm(50.0);

        // 4. DEBUG: Log gradient norm
        let grad_norm = self.accumulated_grad.global_norm();
        println!("   [V4 DEBUG] Gradient norm: {:.6}", grad_norm);

        // 5. CRITICAL: Compute hash BEFORE zero()
        self.last_gradient_hash = self.accumulated_grad.compute_hash();

        // 6. Apply AdamW to ALL parameters using index-based access
        let num_params = self.accumulated_grad.gradients.len();
        for param_idx in 0..num_params {
            self.apply_adamw_by_index(param_idx);
        }

        // 7. Reset for next batch
        self.accumulated_grad.zero();
        self.current_step_in_batch = 0;
        self.metrics.increment_step();
    }

    pub fn train_step(&mut self, input_bytes: &[u8], target_byte: u8) -> f32 {
        // 1. Forward pass
        let embedded = self.model.stack.embed(input_bytes);
        let hidden = self.model.stack.forward(&embedded);
        let logits_vec = self.model.stack.predict(&hidden);
        
        // 2. Compute loss
        let target_idx = target_byte as usize;
        let loss_result = self.loss_fn.compute(&logits_vec, target_idx);
        let loss = loss_result.loss;

        if !loss.is_finite() {
            return 1.0;
        }

        self.metrics.update_loss(loss, 0.95);

        // 3. Use recursive backward for full-model training
        let vocab_size = self.model.config.vocab_size;
        let mut grad_tensor = Tensor::new(1, 1, vocab_size);
        for (j, &g) in loss_result.gradient.iter().enumerate() {
            grad_tensor.data[[0, 0, j]] = g;
        }

        self.model.stack.compute_gradients_recursive(
            &embedded,
            &grad_tensor,
            &mut self.accumulated_grad,
        );

        // 4. Accumulate and flush if needed
        self.current_step_in_batch += 1;

        if self.current_step_in_batch >= self.accumulation_steps {
            self.flush_accumulated_grad();
        }

        loss
    }

    pub fn train_on_text(
        &mut self,
        text: &str,
        context_size: usize,
        epochs: u32,
    ) -> CognitiveProofV4 {
        self.reset_state();

        let hash_before = CognitiveProofV4::compute_hash(
            &bincode::serialize(&self.model).unwrap_or_default(),
        );
        let dataset_commitment = CognitiveProofV4::compute_hash(text.as_bytes());

        let mut dataset = StreamingDataset::new(text, context_size);
        let num_samples = dataset.len();

        if num_samples == 0 {
            return ModelCheckpoint::generate_proof(
                &self.model, hash_before, dataset_commitment, 0.0,
                &self.metrics, &self.telemetry,
                String::new(),
            );
        }

        let initial_loss = {
            dataset.reset();
            if let Some((first_input, first_target)) = dataset.next() {
                let embedded = self.model.stack.embed(first_input);
                let hidden = self.model.stack.forward(&embedded);
                let logits_vec = self.model.stack.predict(&hidden);
                self.loss_fn.compute(&logits_vec, first_target as usize).loss
            } else {
                0.0
            }
        };

        let updates_per_epoch =
            (num_samples as u32 + self.accumulation_steps - 1) / self.accumulation_steps;
        let total_updates = updates_per_epoch * epochs;
        
        self.scheduler.total_steps = total_updates.max(self.scheduler.warmup_steps + 1);

        println!(
            "   [V4 INFO] samples/epoch: {} | accumulation: {} | updates/epoch: {} | backend: {}",
            num_samples, self.accumulation_steps, updates_per_epoch, self.backend.name()
        );
        println!(
            "   [V4 DEBUG] base_lr: {:.8} | warmup: {} | total_updates: {}",
            self.scheduler.base_lr, self.scheduler.warmup_steps, self.scheduler.total_steps
        );

        for epoch in 0..epochs {
            let mut indices: Vec<usize> = (0..num_samples).collect();
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
            
            dataset.reset();
            
            let mut epoch_loss = 0.0f32;
            let mut seen = 0u32;

            for idx in indices {
                dataset.set_position(idx);
                if let Some((input, target)) = dataset.next() {
                    epoch_loss += self.train_step(input, target);
                    seen += 1;
                }
            }

            if self.current_step_in_batch > 0 {
                self.flush_accumulated_grad();
            }

            self.metrics.last_epoch_avg_loss = epoch_loss / seen.max(1) as f32;
            self.telemetry.refresh();

            let gradient_hash = &self.last_gradient_hash;

            println!(
                "   [V4 TRAIN] epoch {:3}/{} | loss {:.4} | ema {:.4} | lr {:.8} | updates {} | grad_hash {:.8} | cpu {:.1}% | ram {} MB",
                epoch + 1,
                epochs,
                self.metrics.last_epoch_avg_loss,
                self.metrics.ema_loss,
                self.optimizer.learning_rate,
                self.metrics.steps_processed,
                gradient_hash.chars().take(8).collect::<String>(),
                self.telemetry.cpu_usage(),
                self.telemetry.ram_usage_mb(),
            );
        }

        ModelCheckpoint::generate_proof(
            &self.model, hash_before, dataset_commitment, initial_loss,
            &self.metrics, &self.telemetry,
            self.last_gradient_hash.clone(),
        )
    }

    pub fn save_binary(&self, path: &str) -> Result<(), String> {
        ModelCheckpoint::save(&self.model, path)
    }

    pub fn load_binary(&mut self, path: &str) -> Result<(), String> {
        ModelCheckpoint::load(&mut self.model, path)
    }

    pub fn reset_state(&mut self) {
        self.current_step_in_batch = 0;
        self.accumulated_grad.zero();
        for m in &mut self.m_buffers {
            m.fill(0.0);
        }
        for v in &mut self.v_buffers {
            v.fill(0.0);
        }
        self.optimizer.t = 0;
        self.metrics.reset();
        self.last_gradient_hash.clear();
    }
    
    /// NEW: Get backend name for display
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
    
    /// NEW: Check if backend supports GPU
    pub fn supports_gpu(&self) -> bool {
        self.backend.supports_feature("gpu")
    }
}