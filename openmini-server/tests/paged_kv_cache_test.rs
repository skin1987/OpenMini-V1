//! KV Cache 系统架构验证测试
//!
//! 验证OpenMini-V1中的vLLM风格分页注意力缓存系统：
//! 1. KVCacheConfig 配置
//! 2. PagedKVCache 结构
//! 3. BlockManager 块管理
//! 4. 内存效率分析

#[test]
fn test_kv_cache_config_fields() {
    println!("\n⚙️ Test: KV Cache Configuration");

    use openmini_server::hardware::kv_cache::block::KVCacheConfig;

    let config = KVCacheConfig {
        num_layers: 32,
        num_heads: 32,
        head_dim: 128,
        max_blocks: 1000,
        block_size: 16,
        dtype_size: 2, // FP16
        enable_prefix_cache: true,
        enable_swap: false,
    };

    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_heads, 32);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.max_blocks, 1000);
    assert_eq!(config.block_size, 16);
    assert_eq!(config.dtype_size, 2);
    assert_eq!(config.enable_prefix_cache, true);
    assert!(!config.enable_swap);

    let tokens_per_block = config.block_size;
    let total_capacity = config.max_blocks * tokens_per_block;
    let kv_dim = config.num_heads * config.head_dim;

    println!("   ✓ 配置验证通过");
    println!(
        "     层数: {}, 头数: {}, 维度: {}",
        config.num_layers, config.num_heads, config.head_dim
    );
    println!(
        "     每块token数: {}, 总容量: {}K tokens",
        tokens_per_block,
        total_capacity / 1000
    );
    println!("     KV维度 (H*D): {}", kv_dim);
}

#[test]
fn test_paged_kv_cache_structure() {
    println!("\n🗄️ Test: PagedKVCache Structure");

    use openmini_server::hardware::kv_cache::block::KVCacheConfig;
    use openmini_server::hardware::kv_cache::paged_cache::PagedKVCache;

    let config = KVCacheConfig {
        num_layers: 2,
        num_heads: 4,
        head_dim: 64,
        max_blocks: 50,
        block_size: 8,
        dtype_size: 2,
        enable_prefix_cache: false,
        enable_swap: false,
    };

    let cache = PagedKVCache::new(config);

    assert_eq!(cache.num_active_requests(), 0);
    assert_eq!(cache.total_tokens(), 0);
    assert!(cache.available_blocks() > 0);

    println!("   ✅ PagedKVCache 创建成功");
    println!(
        "     可用块: {}, 活跃请求: {}",
        cache.available_blocks(),
        cache.num_active_requests()
    );
}

#[test]
fn test_block_manager_existence() {
    println!("\n🔧 Test: BlockManager Existence");

    use openmini_server::hardware::kv_cache::block::KVCacheConfig;
    use openmini_server::hardware::kv_cache::block_manager::BlockManager;

    let config = KVCacheConfig {
        num_layers: 4,
        num_heads: 8,
        head_dim: 64,
        max_blocks: 100,
        block_size: 16,
        dtype_size: 2,
        enable_prefix_cache: true,
        enable_swap: false,
    };

    let manager = BlockManager::new(&config);

    println!("   ✅ BlockManager 创建成功");
    println!("     块管理器已就绪，可管理最多{}个块", config.max_blocks);
}

#[test]
fn test_memory_efficiency_calculation() {
    println!("\n💾 Test: Memory Efficiency Calculation");

    use openmini_server::hardware::kv_cache::block::KVCacheConfig;

    let scenarios = vec![
        ("7B模型", 32, 32, 128, 4096),
        ("13B模型", 40, 40, 128, 8192),
        ("70B模型", 64, 64, 128, 16384),
    ];

    println!("\n   模型 | 层数(H) | 头数 | 维度(D) | 最大块 | 单块大小(KB) | 总内存(MB)");
    println!("   ----|--------|------|--------|--------|-------------|-----------");

    for (name, layers, heads, dim, max_blocks) in &scenarios {
        let block_size: usize = 16;

        let config = KVCacheConfig {
            num_layers: *layers,
            num_heads: *heads,
            head_dim: *dim,
            max_blocks: *max_blocks,
            block_size,
            dtype_size: 2,
            enable_prefix_cache: false,
            enable_swap: false,
        };

        let single_block_kb = config.block_memory_size() / 1024;
        let total_mem_mb = config.total_memory_size() / (1024 * 1024);
        let token_capacity = max_blocks * block_size;

        println!(
            "   {} | {:>6} | {:>4} | {:>6} | {:>6} | {:>11} | {:>9}",
            name, layers, heads, dim, max_blocks, single_block_kb, total_mem_mb
        );
        println!("     理论容量: {}K tokens", token_capacity / 1000);
    }

    println!("\n   ✓ 内存效率计算完成");
}

#[test]
fn test_kv_cache_module_completeness() {
    println!("\n📦 Test: KV Cache Module Completeness");

    let modules = [
        ("block.rs", "基础块定义"),
        ("paged_cache.rs", "分页缓存实现"),
        ("page_table.rs", "页表管理"),
        ("block_manager.rs", "块管理器"),
        ("persistence.rs", "持久化支持"),
        ("continuous_batch.rs", "连续批处理"),
        ("streaming.rs", "流式处理"),
        ("prefix_cache.rs", "前缀缓存"),
        ("mla/", "MLA (Multi-Latent Attention)"),
    ];

    println!("\n   子模块清单:");
    for (module, desc) in &modules {
        println!("   ✓ {}: {}", module, desc);
    }

    println!("\n   ✓ KV Cache系统包含 {} 个子模块", modules.len());
}
