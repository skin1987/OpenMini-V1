//! 统一内存计算集成测试
//!
//! 测试智能调度器、负载监测、SIMD 加速等功能

use openmini_server::hardware::{
    get_system_load, LoadMonitor, LoadThresholds, SystemLoad, UnifiedScheduler,
};

#[test]
fn test_unified_scheduler_creation() {
    let scheduler = UnifiedScheduler::new();

    println!("Preferred Device: {:?}", scheduler.preferred_device());
    println!("Unified Memory: {}", scheduler.has_unified_memory());
    println!("Recommended Threads: {}", scheduler.recommended_threads());

    assert!(scheduler.recommended_threads() >= 1);
}

#[test]
fn test_device_selection() {
    let scheduler = UnifiedScheduler::new();

    let small_device = scheduler.select_device(100);
    println!("Small matrix (100): {:?}", small_device);

    let medium_device = scheduler.select_device(1000);
    println!("Medium matrix (1000): {:?}", medium_device);

    let large_device = scheduler.select_device(5000);
    println!("Large matrix (5000): {:?}", large_device);
}

#[test]
fn test_parallelism() {
    let scheduler = UnifiedScheduler::new();
    let parallelism = scheduler.recommended_parallelism();

    println!("Parallelism: {}", parallelism);
    assert!(parallelism >= 0.0 && parallelism <= 1.0);
}

#[test]
fn test_load_monitor_creation() {
    let monitor = LoadMonitor::default();

    assert_eq!(monitor.parallelism(), 100);
    assert!(!monitor.should_pause());
}

#[test]
fn test_load_monitor_thresholds() {
    let thresholds = LoadThresholds {
        cpu_max: 0.7,
        temp_max: 80.0,
        memory_max: 0.85,
        adjust_interval_ms: 50,
    };

    let monitor = LoadMonitor::new(thresholds);
    assert_eq!(monitor.thresholds().cpu_max, 0.7);
}

#[test]
fn test_load_reduction() {
    let mut monitor = LoadMonitor::new(LoadThresholds {
        cpu_max: 0.5,
        temp_max: 70.0,
        memory_max: 0.8,
        adjust_interval_ms: 0,
    });

    monitor.update(SystemLoad {
        cpu_usage: 0.9,
        cpu_temp: 60.0,
        memory_usage: 0.5,
        gpu_usage: 0.0,
        gpu_temp: 0.0,
        timestamp: std::time::Instant::now(),
    });

    let parallelism = monitor.parallelism();
    println!("Parallelism after high load: {}", parallelism);
    assert!(parallelism < 100);
}

#[test]
fn test_system_load_sampling() {
    let load = get_system_load();

    println!("CPU Usage: {:.2}%", load.cpu_usage * 100.0);
    println!("CPU Temp: {:.1}°C", load.cpu_temp);
    println!("Memory Usage: {:.2}%", load.memory_usage * 100.0);

    assert!(load.cpu_usage >= 0.0 && load.cpu_usage <= 1.0);
    assert!(load.cpu_temp >= 0.0);
    assert!(load.memory_usage >= 0.0 && load.memory_usage <= 1.0);
}

#[test]
fn test_model_config_multimodal() {
    use openmini_server::model::inference::model::ModelConfig;

    let config = ModelConfig::default();

    assert_eq!(config.hidden_size, 3584);
    assert_eq!(config.num_hidden_layers, 28);
    assert_eq!(config.image_patch_size, 14);
}

#[test]
#[ignore = "耗时测试，需要完整模型推理"]
fn test_vision_encoder() {
    use ndarray::Array3;
    use openmini_server::model::inference::model::{ModelConfig, MultimodalTransformer};

    let model = MultimodalTransformer::new(ModelConfig::default());

    let image = Array3::from_shape_vec((224, 224, 3), vec![128u8; 224 * 224 * 3]).unwrap();
    let result = model.encode_image(&image);

    assert!(result.is_ok());
    let embedding = result.unwrap();
    println!("Image embedding shape: {:?}", embedding.dim());
}

#[test]
fn test_stream_generator_stats() {
    use openmini_server::model::inference::generator::StreamStats;

    let stats = StreamStats {
        first_token_latency_ms: 300,
        total_latency_ms: 2000,
        tokens_generated: 50,
        total_text: "Hello, world!".to_string(),
    };

    let tps = stats.tokens_per_second();
    println!("Tokens per second: {:.2}", tps);

    assert!((tps - 25.0).abs() < 1.0);
}

#[test]
fn test_smart_quant_strategy() {
    use openmini_server::model::inference::quant_simd::SmartQuantStrategy;

    let strategy = SmartQuantStrategy::new(28);

    println!("Layer 0 quant: {:?}", strategy.get_layer_quant(0));
    println!("Layer 1 quant: {:?}", strategy.get_layer_quant(1));
    println!("Layer 2 quant: {:?}", strategy.get_layer_quant(2));

    assert!(strategy.layer_configs.len() == 28);
}

#[test]
fn test_simd_softmax() {
    use openmini_server::model::inference::quant_simd::softmax_simd;

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = softmax_simd(&input);

    let sum: f32 = result.iter().sum();
    println!("Softmax sum: {}", sum);

    assert!((sum - 1.0).abs() < 0.001);
}

#[test]
fn test_simd_rms_norm() {
    use openmini_server::model::inference::quant_simd::rms_norm_simd;

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let result = rms_norm_simd(&input, &weight, 1e-6);

    println!("RMS Norm result: {:?}", result);
    assert_eq!(result.len(), 5);
}
