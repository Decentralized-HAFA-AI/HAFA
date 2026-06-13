// ============================================================================
// System Telemetry (Hardware Resource Monitoring)
// ============================================================================
//
// Provides a clean interface to system resource metrics:
// - CPU usage percentage
// - RAM usage in MB
//
// This isolates the sysinfo dependency and makes it easy to:
// - Mock for testing
// - Replace with a different monitoring library
// - Add new metrics (GPU, disk I/O, etc.)
//
// ============================================================================

use sysinfo::System;

pub struct SystemTelemetry {
    sys: System,
}

impl SystemTelemetry {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        
        // First CPU reading is always 0, so we need to wait and refresh
        sys.refresh_cpu_usage();
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        sys.refresh_cpu_usage();
        sys.refresh_memory();

        Self { sys }
    }

    /// Refreshes all system metrics
    /// Should be called before reading cpu_usage() or ram_usage_mb()
    pub fn refresh(&mut self) {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
    }

    /// Returns current CPU usage as a percentage (0.0 to 100.0)
    pub fn cpu_usage(&self) -> f64 {
        // Updated API for sysinfo 0.30+
        self.sys.global_cpu_info().cpu_usage() as f64
    }

    /// Returns current RAM usage in megabytes
    pub fn ram_usage_mb(&self) -> u64 {
        self.sys.used_memory() / (1024 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_creation() {
        let telemetry = SystemTelemetry::new();
        
        // Should be able to read metrics without panic
        let cpu = telemetry.cpu_usage();
        let ram = telemetry.ram_usage_mb();
        
        // CPU should be a valid percentage
        assert!(cpu >= 0.0);
        assert!(cpu <= 100.0);
        
        // RAM should be positive
        assert!(ram > 0);
    }

    #[test]
    fn test_refresh() {
        let mut telemetry = SystemTelemetry::new();
        
        // Should not panic
        telemetry.refresh();
        
        let cpu = telemetry.cpu_usage();
        assert!(cpu >= 0.0);
    }
}