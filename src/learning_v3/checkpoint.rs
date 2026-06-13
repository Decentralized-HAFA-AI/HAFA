// ============================================================================
// Model Checkpoint & Proof Generation
// ============================================================================

use std::fs::File;
use std::io::{BufReader, BufWriter};

use super::cognitive_proof::CognitiveProofV4;
use super::training_metrics::TrainingMetrics;
use super::telemetry::SystemTelemetry;
use super::TransformerEngine;

pub struct ModelCheckpoint;

impl ModelCheckpoint {
    pub fn save(model: &TransformerEngine, path: &str) -> Result<(), String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, model).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load(model: &mut TransformerEngine, path: &str) -> Result<(), String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        *model = bincode::deserialize_from(reader).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn generate_proof(
        model: &TransformerEngine,
        hash_before: String,
        dataset_commitment: String,
        loss_before: f32,
        metrics: &TrainingMetrics,
        telemetry: &SystemTelemetry,
        gradient_commitment: String, // NEW parameter
    ) -> CognitiveProofV4 {
        let hash_after = CognitiveProofV4::compute_hash(
            &bincode::serialize(model).unwrap_or_default(),
        );

        CognitiveProofV4 {
            model_hash_before: hash_before,
            model_hash_after: hash_after,
            dataset_commitment,
            gradient_commitment, // NEW field
            loss_before,
            loss_after: metrics.last_epoch_avg_loss,
            ema_loss_after: metrics.ema_loss,
            samples_processed: metrics.steps_processed,
            wall_time_ms: metrics.get_wall_time_ms(),
            cpu_usage_percent: telemetry.cpu_usage(),
            ram_usage_mb: telemetry.ram_usage_mb(),
        }
    }
}