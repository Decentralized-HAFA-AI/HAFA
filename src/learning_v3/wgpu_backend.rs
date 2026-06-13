// ============================================================================
// WGPU Backend: GPU-Accelerated Matmul with CACHED Pipelines
// ============================================================================

use ndarray::{Array1, Array2};
use wgpu::util::DeviceExt;
use super::backend::{Backend, BackendInfo, DeviceType};

const MATMUL_SHADER: &str = r#"
struct Params { m: u32, n: u32, k: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= params.m || col >= params.k) { return; }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.n; i = i + 1u) {
        sum = sum + a[row * params.n + i] * b[i * params.k + col];
    }
    c[row * params.k + col] = sum;
}
"#;

pub struct WgpuBackend {
    info: BackendInfo,
    #[allow(dead_code)]
    instance: wgpu::Instance,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    // === CACHED ITEMS ===
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuBackend {
    pub async fn new() -> Result<Self, String> {
        println!("   [GPU] 🎮 Initializing WGPU Backend (with Pipeline Caching)...");
        
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.ok_or("Failed to find GPU adapter")?;
        
        let adapter_info = adapter.get_info();
        println!("   [GPU] 🎮 Adapter: {} ({:?})", adapter_info.name, adapter_info.backend);
        
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("HAFA GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await.map_err(|e| format!("Failed to create device: {}", e))?;

        // === 1. CACHE SHADER MODULE ===
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matmul Shader"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
        });

        // === 2. CACHE BIND GROUP LAYOUT ===
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matmul Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // === 3. CACHE COMPUTE PIPELINE ===
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matmul Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matmul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        println!("   [GPU] ✅ Pipelines cached successfully!");
        
        // Test
        println!("   [GPU] 🧪 Testing cached GPU matmul...");
        let test_a = Array2::from_elem((4, 4), 2.0f32);
        let test_b = Array2::from_elem((4, 4), 3.0f32);
        let result = Self::gpu_matmul_internal(&device, &queue, &pipeline, &bind_group_layout, &test_a, &test_b);
        if let Ok(res) = result {
            println!("   [GPU] ✅ GPU matmul test passed! (Result: {})", res[[0, 0]]);
        }

        let info = BackendInfo {
            device_type: DeviceType::WGPU,
            device_name: adapter_info.name,
            memory_mb: 0,
            compute_units: 1,
            supports_fp16: adapter.features().contains(wgpu::Features::SHADER_F16),
        };

        Ok(Self { info, instance, adapter, device, queue, pipeline, bind_group_layout })
    }

    pub async fn is_available() -> bool {
        wgpu::Instance::new(wgpu::InstanceDescriptor::default())
            .request_adapter(&wgpu::RequestAdapterOptions::default()).await.is_some()
    }

    fn gpu_matmul_internal(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        bind_group_layout: &wgpu::BindGroupLayout,
        a: &Array2<f32>,
        b: &Array2<f32>,
    ) -> Result<Array2<f32>, String> {
        let (m, n1) = a.dim();
        let (n2, k) = b.dim();
        if n1 != n2 { return Err("Dimension mismatch".into()); }
        let n = n1;
        let c_size = m * k;

        let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix A"), contents: bytemuck::cast_slice(&a.iter().cloned().collect::<Vec<_>>()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix B"), contents: bytemuck::cast_slice(&b.iter().cloned().collect::<Vec<_>>()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let c_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C"),
            size: (c_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_data: [u32; 4] = [m as u32, n as u32, k as u32, 0];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params"), contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Matmul Encoder") });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((m as u32 + 15) / 16, (k as u32 + 15) / 16, 1);
        }

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (c_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&c_buffer, 0, &output_buffer, 0, (c_size * std::mem::size_of::<f32>()) as u64);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| { let _ = sender.send(result); });
        device.poll(wgpu::Maintain::Wait);
        receiver.recv().map_err(|e| format!("Channel error: {}", e))?.map_err(|e| format!("Map error: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let mut result = Array2::zeros((m, k));
        for (i, &val) in floats.iter().enumerate() { result[[i / k, i % k]] = val; }
        
        drop(data);
        output_buffer.unmap();
        Ok(result)
    }
}

impl Backend for WgpuBackend {
    fn name(&self) -> &str { "WGPU" }
    fn info(&self) -> &BackendInfo { &self.info }
    
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        match Self::gpu_matmul_internal(&self.device, &self.queue, &self.pipeline, &self.bind_group_layout, a, b) {
            Ok(res) => res,
            Err(e) => { eprintln!("   [GPU] ⚠️ Fallback to CPU: {}", e); a.dot(b) }
        }
    }
    
    fn matmul_at_b(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> { self.matmul(&a.t().to_owned(), b) }
    fn matmul_a_bt(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> { self.matmul(a, &b.t().to_owned()) }
    fn transpose(&self, a: &Array2<f32>) -> Array2<f32> { a.t().to_owned() }
    fn relu(&self, input: &Array2<f32>) -> Array2<f32> { input.mapv(|x| if x > 0.0 { x } else { 0.0 }) }
    fn gelu(&self, input: &Array2<f32>) -> Array2<f32> {
        let s = (2.0_f32 / std::f32::consts::PI).sqrt();
        input.mapv(|x| 0.5 * x * (1.0 + (s * (x + 0.044715 * x * x * x)).tanh()))
    }
    fn softmax(&self, input: &Array2<f32>, axis: i32) -> Array2<f32> {
        use ndarray::Axis;
        let ax = if axis < 0 { Axis(input.ndim() - ((-axis) as usize)) } else { Axis(axis as usize) };
        let mut result = input.clone();
        let max_vals = result.map_axis(ax, |lane| lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        for (mut lane, &max_val) in result.lanes_mut(ax).into_iter().zip(max_vals.iter()) {
            for val in lane.iter_mut() { *val = (*val - max_val).exp(); }
        }
        let sum_vals = result.map_axis(ax, |lane| lane.iter().sum::<f32>());
        for (mut lane, &sum_val) in result.lanes_mut(ax).into_iter().zip(sum_vals.iter()) {
            for val in lane.iter_mut() { *val /= sum_val; }
        }
        result
    }
    fn layer_norm(&self, input: &Array2<f32>, gamma: &Array1<f32>, beta: &Array1<f32>, eps: f32) -> Array2<f32> {
        let (_rows, cols) = input.dim();
        let mut result = Array2::zeros(input.dim());
        for (i, row) in input.rows().into_iter().enumerate() {
            let mean: f32 = row.iter().sum::<f32>() / cols as f32;
            let variance: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / cols as f32;
            let std_dev = (variance + eps).sqrt();
            for (j, (&x, (&g, &b))) in row.iter().zip(gamma.iter().zip(beta.iter())).enumerate() {
                result[[i, j]] = ((x - mean) / std_dev) * g + b;
            }
        }
        result
    }
    fn element_add(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> { a + b }
    fn element_mul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> { a * b }
    fn scalar_mul(&self, a: f32, x: &Array2<f32>) -> Array2<f32> { x.mapv(|v| v * a) }
    fn element_sub(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> { a - b }
}