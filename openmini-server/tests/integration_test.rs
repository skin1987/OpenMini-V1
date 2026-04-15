use openmini_server::service::thread::pool::create_default_pool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn test_thread_pool_stress() {
    let pool = create_default_pool();
    let counter = Arc::new(AtomicUsize::new(0));
    let iterations = 1000;

    for _ in 0..iterations {
        let counter_clone = Arc::clone(&counter);
        pool.execute(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
    }

    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(counter.load(Ordering::SeqCst), iterations);
}

#[test]
fn test_memory_monitor_under_load() {
    let monitor = openmini_server::hardware::memory::MemoryMonitor::new(1024 * 1024 * 100);

    for _ in 0..10 {
        let result = monitor.allocate(1024 * 1024);
        assert!(result.is_ok());
    }

    assert_eq!(monitor.usage(), 10 * 1024 * 1024);

    for _ in 0..5 {
        monitor.deallocate(1024 * 1024);
    }

    assert_eq!(monitor.usage(), 5 * 1024 * 1024);
}

#[test]
fn test_response_latency() {
    let start = Instant::now();
    let response = openmini_server::service::server::gateway::Response::new(
        "test_request".to_string(),
        vec![12, 3, 4].into(),
        true,
    );
    let elapsed = start.elapsed();

    assert!(elapsed < Duration::from_millis(100));
    assert_eq!(response.session_id, "test_request");
}

#[tokio::test]
async fn test_connection_pool_concurrent_access() {
    use openmini_server::service::server::connection::ConnectionPool;
    use std::thread;

    let pool = Arc::new(ConnectionPool::new(10));
    let mut handles = vec![];

    for _ in 0..10 {
        let pool_clone = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                if let Some(conn) = pool_clone.acquire().await {
                    std::thread::sleep(Duration::from_millis(10));
                    pool_clone.release(conn);
                }
            });
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(pool.stats().active_connections.load(Ordering::Relaxed), 0);
}

#[test]
fn test_multimodal_prompt_building() {
    use openmini_server::model::inference::engine::build_multimodal_prompt;

    // 使用预编码的 token ID（模拟 tokenizer 输出）
    let text_tokens: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let num_image_tokens = 64;
    let tokens = build_multimodal_prompt(&text_tokens, num_image_tokens);

    assert!(!tokens.is_empty());
    // 验证结构：IM_START + image_tokens + IM_END + text_tokens
    assert!(tokens.len() > num_image_tokens);
}

#[test]
fn test_model_config() {
    use openmini_server::model::inference::model::ModelConfig;

    let config = ModelConfig::default();
    assert_eq!(config.hidden_size, 3584);
    assert_eq!(config.num_hidden_layers, 28);
    assert_eq!(config.num_attention_heads, 32);
}

#[test]
fn test_vision_encoder() {
    use ndarray::Array3;
    use openmini_server::model::inference::model::{ModelConfig, MultimodalTransformer};

    let model = MultimodalTransformer::new(ModelConfig::default());
    let image = Array3::from_shape_vec((224, 224, 3), vec![128u8; 224 * 224 * 3]).unwrap();
    let result = model.encode_image(&image);
    assert!(result.is_ok());
}

#[test]
fn test_sampler_parameters() {
    use openmini_server::model::inference::sampler::GenerateParams;

    let params = GenerateParams::default()
        .with_temperature(0.7)
        .with_top_p(0.8)
        .with_top_k(100)
        .with_max_new_tokens(512);

    assert!((params.temperature - 0.7).abs() < 1e-5);
    assert!((params.top_p - 0.8).abs() < 1e-5);
    assert_eq!(params.top_k, 100);
    assert_eq!(params.max_new_tokens, 512);
}
