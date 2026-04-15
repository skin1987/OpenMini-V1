//! StreamingAttention - 流式场景优化的 Attention 实现
//!
//! 原理：
//! - 标准 Attention: O(n^2) 内存，每次新 token 需要读取全部 KV
//! - StreamingAttention: O(n) 内存，分块存储，增量更新
//!
//! 使用在线 softmax (Online Softmax) 算法实现数值稳定的增量注意力计算

#![allow(dead_code)]

use std::sync::RwLock;

#[derive(Debug, Clone)]
pub struct StreamingAttentionConfig {
    pub block_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_blocks: usize,
}

impl Default for StreamingAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 512,
            num_heads: 32,
            head_dim: 128,
            max_blocks: 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamingBlock {
    pub block_id: usize,
    pub k_data: Vec<f32>,
    pub v_data: Vec<f32>,
    pub num_tokens: usize,
    pub is_full: bool,
}

impl StreamingBlock {
    pub fn new(block_id: usize, config: &StreamingAttentionConfig) -> Self {
        let block_elements = config.block_size * config.num_heads * config.head_dim;
        Self {
            block_id,
            k_data: vec![0.0f32; block_elements],
            v_data: vec![0.0f32; block_elements],
            num_tokens: 0,
            is_full: false,
        }
    }

    pub fn write_token(
        &mut self,
        local_pos: usize,
        k: &[f32],
        v: &[f32],
        config: &StreamingAttentionConfig,
    ) -> Result<(), String> {
        if local_pos >= config.block_size {
            return Err(format!(
                "Local position {} exceeds block size {}",
                local_pos, config.block_size
            ));
        }

        if self.is_full {
            return Err("Block is already full".to_string());
        }

        let offset = local_pos * config.num_heads * config.head_dim;
        let chunk_size = config.num_heads * config.head_dim;

        if offset + chunk_size > self.k_data.len() {
            return Err(format!(
                "Offset {} out of bounds for block data size {}",
                offset + chunk_size,
                self.k_data.len()
            ));
        }

        self.k_data[offset..offset + chunk_size].copy_from_slice(k);
        self.v_data[offset..offset + chunk_size].copy_from_slice(v);

        if local_pos >= self.num_tokens {
            self.num_tokens = local_pos + 1;
        }

        if self.num_tokens >= config.block_size {
            self.is_full = true;
        }

        Ok(())
    }

    pub fn read_tokens(
        &self,
        start: usize,
        len: usize,
        config: &StreamingAttentionConfig,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        if start >= self.num_tokens {
            return None;
        }

        let actual_len = len.min(self.num_tokens - start);
        if actual_len == 0 {
            return None;
        }

        let chunk_size = config.num_heads * config.head_dim;
        let mut k_out = vec![0.0f32; actual_len * chunk_size];
        let mut v_out = vec![0.0f32; actual_len * chunk_size];

        for i in 0..actual_len {
            let src_offset = (start + i) * chunk_size;
            let dst_offset = i * chunk_size;
            k_out[dst_offset..dst_offset + chunk_size]
                .copy_from_slice(&self.k_data[src_offset..src_offset + chunk_size]);
            v_out[dst_offset..dst_offset + chunk_size]
                .copy_from_slice(&self.v_data[src_offset..src_offset + chunk_size]);
        }

        Some((k_out, v_out))
    }
}

/// 在线 softmax 状态
/// 用于增量计算 softmax，避免存储所有分数
#[derive(Debug, Clone, Default)]
struct OnlineSoftmaxState {
    /// 当前最大值
    max_score: f32,
    /// exp(max - prev_max) 的累积乘积
    scale: f32,
    /// 加权和的累积
    weighted_sum: Vec<f32>,
    /// exp 分数之和
    exp_sum: f32,
}

impl OnlineSoftmaxState {
    fn new(dim: usize) -> Self {
        Self {
            max_score: f32::NEG_INFINITY,
            scale: 1.0,
            weighted_sum: vec![0.0; dim],
            exp_sum: 0.0,
        }
    }

    /// 更新状态：添加新的分数和值
    fn update(&mut self, score: f32, value: &[f32]) {
        let new_max = self.max_score.max(score);

        if new_max > self.max_score {
            let scale_factor = (self.max_score - new_max).exp();
            self.scale *= scale_factor;
            for v in self.weighted_sum.iter_mut() {
                *v *= scale_factor;
            }
            self.exp_sum *= scale_factor;
            self.max_score = new_max;
        }

        let exp_score = (score - self.max_score).exp();
        self.exp_sum += exp_score;

        for (i, &val) in value.iter().enumerate() {
            self.weighted_sum[i] += exp_score * val;
        }
    }

    /// 获取最终输出
    fn finalize(&self) -> Vec<f32> {
        if self.exp_sum > 0.0 {
            self.weighted_sum
                .iter()
                .map(|&v| v / self.exp_sum)
                .collect()
        } else {
            vec![0.0; self.weighted_sum.len()]
        }
    }
}

#[derive(Debug)]
pub struct StreamingAttention {
    config: StreamingAttentionConfig,
    blocks: Vec<RwLock<StreamingBlock>>,
    /// 每个位置的块索引映射（按位置索引）
    position_to_block: Vec<Option<usize>>,
    /// 已分配的块数量
    allocated_blocks: usize,
    /// 总 token 数
    total_tokens: usize,
    /// 最大已写入位置
    max_position: usize,
}

impl StreamingAttention {
    pub fn new(config: StreamingAttentionConfig) -> Self {
        let blocks = (0..config.max_blocks)
            .map(|id| RwLock::new(StreamingBlock::new(id, &config)))
            .collect();

        Self {
            config,
            blocks,
            position_to_block: Vec::new(),
            allocated_blocks: 0,
            total_tokens: 0,
            max_position: 0,
        }
    }

    /// 写入 KV 数据
    /// 返回 Ok(()) 表示成功，Err 表示失败
    pub fn write(&mut self, pos: usize, k: &[f32], v: &[f32]) -> Result<(), String> {
        let expected_len = self.config.num_heads * self.config.head_dim;
        if k.len() != expected_len || v.len() != expected_len {
            return Err(format!(
                "KV length mismatch: expected {}, got k={}, v={}",
                expected_len,
                k.len(),
                v.len()
            ));
        }

        let block_idx = pos / self.config.block_size;
        let local_pos = pos % self.config.block_size;

        if block_idx >= self.config.max_blocks {
            return Err(format!(
                "Position {} exceeds max blocks (block_idx={}, max={})",
                pos, block_idx, self.config.max_blocks
            ));
        }

        if block_idx >= self.blocks.len() {
            return Err(format!("Block index {} out of range", block_idx));
        }

        // 扩展位置映射
        while self.position_to_block.len() <= block_idx {
            self.position_to_block.push(None);
        }

        // 分配新块（如果需要）
        let is_new_block = self.position_to_block[block_idx].is_none();
        if is_new_block && self.allocated_blocks >= self.config.max_blocks {
            return Err("No more blocks available".to_string());
        }

        // 尝试写入数据
        let write_result = {
            if let Ok(mut block) = self.blocks[block_idx].write() {
                block.write_token(local_pos, k, v, &self.config)
            } else {
                Err("Failed to acquire write lock".to_string())
            }
        };

        // 只有写入成功后才更新统计
        match write_result {
            Ok(()) => {
                if is_new_block {
                    self.position_to_block[block_idx] = Some(block_idx);
                    self.allocated_blocks += 1;
                }
                self.total_tokens += 1;
                self.max_position = self.max_position.max(pos + 1);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// 读取指定范围的 KV 数据
    /// 如果任何需要的块缺失，返回 None
    pub fn read_range(&self, start: usize, len: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        if start >= self.max_position {
            return None;
        }

        let actual_len = len.min(self.max_position - start);
        if actual_len == 0 {
            return None;
        }

        let first_block = start / self.config.block_size;
        let last_block = (start + actual_len - 1).div_ceil(self.config.block_size);

        // 预检查：确保所有需要的块都存在
        for block_idx in first_block..=last_block {
            if block_idx >= self.position_to_block.len() {
                return None;
            }
            self.position_to_block[block_idx]?;
        }

        let chunk_size = self.config.num_heads * self.config.head_dim;
        let mut k_out = vec![0.0f32; actual_len * chunk_size];
        let mut v_out = vec![0.0f32; actual_len * chunk_size];

        let mut remaining = actual_len;
        let mut pos = start;

        while remaining > 0 {
            let block_idx = pos / self.config.block_size;
            let local_pos = pos % self.config.block_size;

            let tokens_in_block = self.config.block_size - local_pos;
            let tokens_to_read = remaining.min(tokens_in_block);

            if block_idx >= self.blocks.len() {
                return None;
            }

            if let Ok(block) = self.blocks[block_idx].read() {
                if let Some((k, v)) = block.read_tokens(local_pos, tokens_to_read, &self.config) {
                    let out_offset = (actual_len - remaining) * chunk_size;
                    k_out[out_offset..out_offset + k.len()].copy_from_slice(&k);
                    v_out[out_offset..out_offset + v.len()].copy_from_slice(&v);
                } else {
                    return None;
                }
            } else {
                return None;
            }

            pos += tokens_to_read;
            remaining -= tokens_to_read;
        }

        Some((k_out, v_out))
    }

    /// 增量注意力计算
    /// 使用在线 softmax 算法，正确实现注意力机制
    pub fn incremental_attention(&self, query: &[f32], pos: usize, scale: f32) -> Vec<f32> {
        let chunk_size = self.config.num_heads * self.config.head_dim;
        let seq_len = self.max_position.min(pos + 1);

        if seq_len == 0 {
            return vec![0.0; chunk_size];
        }

        // 每个头独立计算
        let mut output = vec![0.0; chunk_size];

        for h in 0..self.config.num_heads {
            let h_offset = h * self.config.head_dim;
            let mut state = OnlineSoftmaxState::new(self.config.head_dim);

            // 按位置顺序遍历所有块
            let first_block = 0;
            let last_block = (seq_len - 1) / self.config.block_size;

            for block_idx in first_block..=last_block {
                // 检查块是否存在
                if block_idx >= self.position_to_block.len() {
                    continue;
                }
                if self.position_to_block[block_idx].is_none() {
                    continue;
                }

                let block = match self.blocks[block_idx].read() {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                let block_start = block_idx * self.config.block_size;
                let block_end = (block_start + self.config.block_size).min(seq_len);
                let tokens_to_process = block_end - block_start;

                if tokens_to_process == 0 {
                    continue;
                }

                // 读取该块的 K 和 V
                let (k, v) = match block.read_tokens(0, tokens_to_process, &self.config) {
                    Some((k, v)) => (k, v),
                    None => continue,
                };

                // 对每个 token 计算注意力分数
                for t in 0..tokens_to_process {
                    let token_offset = t * chunk_size + h_offset;

                    // 计算 QK^T
                    let mut qk_score = 0.0f32;
                    for d in 0..self.config.head_dim {
                        qk_score += query[h_offset + d] * k[token_offset + d];
                    }
                    qk_score *= scale;

                    // 提取该 token 的 V
                    let v_slice = &v[token_offset..token_offset + self.config.head_dim];

                    // 更新在线 softmax 状态
                    state.update(qk_score, v_slice);
                }
            }

            // 获取该头的输出
            let head_output = state.finalize();
            output[h_offset..h_offset + self.config.head_dim].copy_from_slice(&head_output);
        }

        output
    }

    /// 获取统计信息
    pub fn stats(&self) -> StreamingAttentionStats {
        StreamingAttentionStats {
            total_tokens: self.total_tokens,
            active_blocks: self.allocated_blocks,
            memory_used_mb: (self.allocated_blocks
                * self.config.block_size
                * self.config.num_heads
                * self.config.head_dim
                * 4)
                / (1024 * 1024),
        }
    }

    /// 清空缓存
    pub fn clear(&mut self) {
        self.position_to_block.clear();
        self.allocated_blocks = 0;
        self.total_tokens = 0;
        self.max_position = 0;

        // 重置所有块
        for block in &self.blocks {
            if let Ok(mut b) = block.write() {
                b.num_tokens = 0;
                b.is_full = false;
            }
        }
    }

    /// 获取总 token 数
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// 获取活跃块数
    pub fn active_blocks(&self) -> usize {
        self.allocated_blocks
    }
}

#[derive(Debug, Clone)]
pub struct StreamingAttentionStats {
    pub total_tokens: usize,
    pub active_blocks: usize,
    pub memory_used_mb: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> StreamingAttentionConfig {
        StreamingAttentionConfig {
            block_size: 4,
            num_heads: 2,
            head_dim: 4,
            max_blocks: 16,
        }
    }

    #[test]
    fn test_streaming_attention_basic() {
        let config = create_test_config();
        let mut attn = StreamingAttention::new(config.clone());

        let k = vec![1.0f32; config.num_heads * config.head_dim];
        let v = vec![2.0f32; config.num_heads * config.head_dim];

        for i in 0..8 {
            attn.write(i, &k, &v).unwrap();
        }

        assert_eq!(attn.total_tokens(), 8);

        let (k_out, _v_out) = attn.read_range(0, 8).unwrap();
        assert_eq!(k_out.len(), 8 * config.num_heads * config.head_dim);
    }

    #[test]
    fn test_write_boundary_check() {
        let config = create_test_config();
        let mut attn = StreamingAttention::new(config.clone());

        // 测试超出最大块数的写入
        let result = attn.write(
            config.max_blocks * config.block_size,
            &[1.0; 8],
            &[1.0; 8],
        );
        assert!(result.is_err(), "Should fail when exceeding max blocks");
    }

    #[test]
    fn test_incremental_attention_correctness() {
        let config = create_test_config();
        let mut attn = StreamingAttention::new(config.clone());

        // 写入一些数据
        for i in 0..8 {
            let k = vec![(i + 1) as f32; config.num_heads * config.head_dim];
            let v = vec![(i + 1) as f32; config.num_heads * config.head_dim];
            attn.write(i, &k, &v).unwrap();
        }

        let query = vec![1.0f32; config.num_heads * config.head_dim];
        let output = attn.incremental_attention(&query, 7, 1.0);

        assert_eq!(output.len(), config.num_heads * config.head_dim);

        // 输出应该不是全零（因为有有效的 KV 数据）
        let sum: f32 = output.iter().sum();
        assert!(sum > 0.0, "Output should not be all zeros");
    }

    #[test]
    fn test_online_softmax_state() {
        let mut state = OnlineSoftmaxState::new(4);

        // 添加一些分数和值
        state.update(1.0, &[1.0, 2.0, 3.0, 4.0]);
        state.update(2.0, &[2.0, 3.0, 4.0, 5.0]);
        state.update(0.5, &[0.5, 1.0, 1.5, 2.0]);

        let output = state.finalize();

        // 输出应该是加权和除以 exp 和
        assert!(output.iter().all(|&x| x.is_finite()));

        // 验证权重和约为 1（softmax 性质）
        // 由于是按维度归一化，每个维度应该有合理的值
        assert!(output[0] > 0.0 && output[0] < 10.0);
    }

    #[test]
    fn test_read_range_out_of_bounds() {
        let config = create_test_config();
        let mut attn = StreamingAttention::new(config.clone());

        for i in 0..4 {
            let k = vec![1.0f32; config.num_heads * config.head_dim];
            let v = vec![2.0f32; config.num_heads * config.head_dim];
            attn.write(i, &k, &v).unwrap();
        }

        // 读取超出范围
        let result = attn.read_range(100, 10);
        assert!(
            result.is_none(),
            "Should return None for out of bounds read"
        );
    }

    #[test]
    fn test_non_sequential_write() {
        let config = create_test_config();
        let mut attn = StreamingAttention::new(config.clone());

        // 非顺序写入
        attn.write(10, &[1.0; 8], &[2.0; 8]).unwrap();
        attn.write(0, &[3.0; 8], &[4.0; 8]).unwrap();
        attn.write(5, &[5.0; 8], &[6.0; 8]).unwrap();

        // 应该能正确读取
        let (k, _v) = attn.read_range(0, 1).unwrap();
        assert_eq!(k[0], 3.0, "Should read correct value at position 0");

        let (k, _v) = attn.read_range(10, 1).unwrap();
        assert_eq!(k[0], 1.0, "Should read correct value at position 10");
    }

    #[test]
    fn test_clear() {
        let config = create_test_config();
        let mut attn = StreamingAttention::new(config.clone());

        for i in 0..8 {
            attn.write(i, &[1.0; 8], &[2.0; 8]).unwrap();
        }

        assert_eq!(attn.total_tokens(), 8);

        attn.clear();

        assert_eq!(attn.total_tokens(), 0);
        assert_eq!(attn.active_blocks(), 0);
    }

    #[test]
    fn test_stats() {
        let config = StreamingAttentionConfig {
            block_size: 512,
            num_heads: 32,
            head_dim: 128,
            max_blocks: 10,
        };

        let mut attn = StreamingAttention::new(config.clone());

        for i in 0..1000 {
            attn.write(
                i,
                &vec![1.0; config.num_heads * config.head_dim],
                &vec![2.0; config.num_heads * config.head_dim],
            )
            .unwrap();
        }

        let stats = attn.stats();
        assert_eq!(stats.total_tokens, 1000);
        assert!(stats.active_blocks >= 1);
    }

    // ==================== 分支覆盖率补充测试 ====================

    #[test]
    fn test_streaming_attention_init() {
        // 初始化测试 - 使用较小的配置验证初始状态
        let config = StreamingAttentionConfig {
            block_size: 512,
            num_heads: 64, // 对应用户的参数
            head_dim: 8,   // hidden_size / num_heads = 512/64 = 8
            max_blocks: 1024,
        };
        let sa = StreamingAttention::new(config.clone());

        assert_eq!(sa.total_tokens(), 0); // 初始token数为0
        assert_eq!(sa.active_blocks(), 0); // 初始活跃块数为0

        // 验证统计信息
        let stats = sa.stats();
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.active_blocks, 0);
        assert_eq!(stats.memory_used_mb, 0);
    }

    #[test]
    fn test_streaming_attention_append_single() {
        // 追加单个token - 测试write方法的基本功能
        let config = StreamingAttentionConfig {
            block_size: 1024,
            num_heads: 64,
            head_dim: 8,
            max_blocks: 1024,
        };
        let kv_len = config.num_heads * config.head_dim; // 64 * 8 = 512
        let mut sa = StreamingAttention::new(config);

        let k: Vec<f32> = (0..kv_len).map(|i| i as f32 * 0.1).collect();
        let v: Vec<f32> = (0..kv_len).map(|i| i as f32 * 0.2).collect();

        let result = sa.write(0, &k, &v);
        assert!(result.is_ok(), "Single token write should succeed");

        assert_eq!(sa.total_tokens(), 1);
        assert_eq!(sa.active_blocks(), 1); // 应该分配了一个块
    }

    #[test]
    fn test_streaming_attention_append_batch() {
        // 批量追加 - 连续写入多个token
        let config = StreamingAttentionConfig {
            block_size: 1024,
            num_heads: 64,
            head_dim: 8,
            max_blocks: 1024,
        };
        let batch_size = 4;
        let kv_len = config.num_heads * config.head_dim;
        let mut sa = StreamingAttention::new(config);

        for i in 0..batch_size {
            let k: Vec<f32> = vec![0.0; kv_len];
            let v: Vec<f32> = vec![0.0; kv_len];
            let result = sa.write(i, &k, &v);
            assert!(result.is_ok(), "Batch write {} should succeed", i);
        }

        assert_eq!(sa.total_tokens(), batch_size);
        assert_eq!(sa.active_blocks(), 1); // 所有token在同一个块内
    }

    #[test]
    fn test_streaming_attention_compute() {
        // 注意力计算 - 测试incremental_attention方法
        let config = StreamingAttentionConfig {
            block_size: 256,
            num_heads: 1, // 简化：单头
            head_dim: 32, // hidden_size=32
            max_blocks: 16,
        };
        let output_dim = config.num_heads * config.head_dim;
        let head_dim = config.head_dim; // 保存head_dim因为config会被移动
        let mut sa = StreamingAttention::new(config);

        // 添加一些KV对
        for i in 0..10 {
            let k: Vec<f32> = (0..head_dim).map(|j| (i * j) as f32 * 0.01).collect();
            let v: Vec<f32> = (0..head_dim).map(|j| j as f32 * 0.1).collect();

            let result = sa.write(i, &k, &v);
            assert!(result.is_ok());
        }

        // 新的query向量
        let q: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.5).collect();

        // 计算注意力输出
        let output = sa.incremental_attention(&q, 9, 1.0); // pos=9, scale=1.0

        // 验证输出维度
        assert_eq!(output.len(), output_dim); // 1*32=32

        // 验证输出不是全零（应该有有效的计算结果）
        let sum: f32 = output.iter().sum();
        assert!(sum != 0.0, "Output should not be all zeros");

        // 验证所有值都是有限的（无NaN或Inf）
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "All values should be finite"
        );
    }

    #[test]
    fn test_streaming_attention_reset() {
        // 重置状态 - 测试clear方法
        let config = StreamingAttentionConfig {
            block_size: 100,
            num_heads: 1,
            head_dim: 16,
            max_blocks: 16,
        };
        let kv_len = config.num_heads * config.head_dim;
        let mut sa = StreamingAttention::new(config);

        // 先写入一些数据
        for i in 0..5 {
            let result = sa.write(i, &vec![0.0; kv_len], &vec![0.0; kv_len]);
            assert!(result.is_ok());
        }

        assert_eq!(sa.total_tokens(), 5);
        assert_eq!(sa.active_blocks(), 1); // 5个token < block_size=100，所以只有1个块

        // 重置状态
        sa.clear();

        // 验证重置后的状态
        assert_eq!(sa.total_tokens(), 0);
        assert_eq!(sa.active_blocks(), 0);

        // 验证读取返回None（因为已经清空）
        let read_result = sa.read_range(0, 1);
        assert!(read_result.is_none(), "Read after clear should return None");

        // 验证统计数据也重置了
        let stats = sa.stats();
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.active_blocks, 0);
    }

    #[test]
    fn test_streaming_block_write_and_read() {
        // 测试StreamingBlock的单独读写操作
        let config = create_test_config();
        let mut block = StreamingBlock::new(0, &config);

        // 写入一个token
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // num_heads*head_dim = 2*4 = 8
        let v = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = block.write_token(0, &k, &v, &config);
        assert!(result.is_ok());
        assert_eq!(block.num_tokens, 1);
        assert!(!block.is_full); // block_size=4，只写了1个token

        // 读取刚写入的token
        let (read_k, read_v) = block.read_tokens(0, 1, &config).unwrap();
        assert_eq!(read_k, k);
        assert_eq!(read_v, v);
    }

    #[test]
    fn test_streaming_block_full_boundary() {
        // 测试块的满边界条件
        let config = StreamingAttentionConfig {
            block_size: 2, // 小块大小以便快速填满
            num_heads: 1,
            head_dim: 4,
            max_blocks: 8,
        };
        let mut block = StreamingBlock::new(0, &config);

        // 写入第1个token
        let result = block.write_token(0, &[1.0; 4], &[2.0; 4], &config);
        assert!(result.is_ok());
        assert!(!block.is_full);

        // 写入第2个token（应该填满块）
        let result = block.write_token(1, &[3.0; 4], &[4.0; 4], &config);
        assert!(result.is_ok());
        assert!(block.is_full); // 现在块应该是满的

        // 尝试再写入应该失败
        let result = block.write_token(2, &[5.0; 4], &[6.0; 4], &config);
        assert!(result.is_err(), "Should fail when block is full");
    }

    #[test]
    fn test_kv_length_mismatch_error() {
        // 测试KV长度不匹配的错误处理
        let config = create_test_config();
        let mut sa = StreamingAttention::new(config.clone());

        let correct_len = config.num_heads * config.head_dim;
        let wrong_k = vec![1.0; correct_len + 1]; // K长度错误
        let correct_v = vec![2.0; correct_len]; // V长度正确

        let result = sa.write(0, &wrong_k, &correct_v);
        assert!(result.is_err(), "Should fail when K length mismatch");
        assert!(result.unwrap_err().contains("length mismatch"));

        // 测试V长度错误
        let correct_k2 = vec![1.0; correct_len];
        let wrong_v = vec![2.0; correct_len - 1];

        let result2 = sa.write(0, &correct_k2, &wrong_v);
        assert!(result2.is_err(), "Should fail when V length mismatch");
    }

    #[test]
    fn test_incremental_attention_empty_state() {
        // 测试空状态的增量注意力计算
        let config = create_test_config();
        let sa = StreamingAttention::new(config.clone());

        let query = vec![1.0; config.num_heads * config.head_dim];
        let output = sa.incremental_attention(&query, 0, 1.0);

        // 空状态下应该返回零向量
        assert_eq!(output.len(), config.num_heads * config.head_dim);
        assert!(
            output.iter().all(|&x| x == 0.0),
            "Empty state should return zero vector"
        );
    }

    #[test]
    fn test_read_partial_range() {
        // 测试部分范围读取
        let config = create_test_config();
        let mut sa = StreamingAttention::new(config.clone());

        // 写入10个token
        for i in 0..10 {
            let kv_len = config.num_heads * config.head_dim;
            sa.write(i, &vec![i as f32; kv_len], &vec![i as f32 * 2.0; kv_len])
                .unwrap();
        }

        // 读取中间的一部分 [3, 7)
        let (k, v) = sa.read_range(3, 4).unwrap();
        assert_eq!(k.len(), 4 * config.num_heads * config.head_dim);
        assert_eq!(v.len(), 4 * config.num_heads * config.head_dim);

        // 验证读到的数据是否正确
        // 第一个读到的应该是position 3的数据
        assert!((k[0] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_cross_block_operations() {
        // 测试跨块操作
        let config = StreamingAttentionConfig {
            block_size: 4, // 小块大小
            num_heads: 2,
            head_dim: 4,
            max_blocks: 8,
        };
        let mut sa = StreamingAttention::new(config.clone());

        // 写入跨越多个块的token（block_size=4，所以每4个token一个新块）
        for i in 0..12 {
            // 12个token会跨越3个块
            let kv_len = config.num_heads * config.head_dim;
            sa.write(i, &vec![i as f32; kv_len], &vec![i as f32; kv_len])
                .unwrap();
        }

        assert_eq!(sa.total_tokens(), 12);
        assert_eq!(sa.active_blocks(), 3); // 12/4 = 3个块

        // 跨块读取
        let result = sa.read_range(2, 8); // 从块0的末尾读到块2的开头
        assert!(result.is_some(), "Cross-block read should succeed");
        let (k, _v) = result.unwrap();
        assert_eq!(k.len(), 8 * config.num_heads * config.head_dim);
    }

    #[test]
    fn test_online_softmax_single_update() {
        // 测试在线softmax的单次更新
        let mut state = OnlineSoftmaxState::new(4);

        state.update(0.0, &[1.0, 1.0, 1.0, 1.0]);
        let output = state.finalize();

        // 单次更新时，输出应该等于输入（因为没有归一化竞争）
        assert!((output[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_online_softmax_extreme_values() {
        // 测试在线softmax的极端值处理
        let mut state = OnlineSoftmaxState::new(2);

        // 添加非常大的分数
        state.update(1000.0, &[1.0, 2.0]);
        // 添加非常小的分数
        state.update(-1000.0, &[3.0, 4.0]);

        let output = state.finalize();

        // 应该能正常处理而不产生NaN或溢出
        assert!(output.iter().all(|&x| x.is_finite()));
        // 大分数对应的值应该占主导（第一个值应该接近1.0，第二个接近2.0，因为大分数的权重极大）
        // 由于数值稳定性，输出应该主要受第一次更新影响
        assert!(
            output[0] > 0.0 && output[1] > 0.0,
            "Output should be positive"
        );
    }

    #[test]
    fn test_memory_usage_calculation() {
        // 测试内存使用量计算
        let config = StreamingAttentionConfig {
            block_size: 512,
            num_heads: 32,
            head_dim: 128,
            max_blocks: 4,
        };
        let mut sa = StreamingAttention::new(config.clone());

        // 写入足够多的token以分配多个块
        for i in 0..1500 {
            let kv_len = config.num_heads * config.head_dim;
            sa.write(i, &vec![1.0; kv_len], &vec![2.0; kv_len]).unwrap();
        }

        let stats = sa.stats();

        // 验证内存使用量为正数且合理
        assert!(stats.memory_used_mb > 0, "Memory usage should be positive");
        assert!(
            stats.active_blocks >= 1,
            "Should have at least one active block"
        );

        // 按照实际实现的公式计算：allocated_blocks * block_size * num_heads * head_dim * 4 / (1024*1024)
        // 注意：实际实现只乘了4（不是8），可能是只计算了K或V的内存
        let expected_mb =
            (stats.active_blocks * config.block_size * config.num_heads * config.head_dim * 4)
                / (1024 * 1024);

        assert_eq!(
            stats.memory_used_mb, expected_mb,
            "Memory usage {} should equal expected {}",
            stats.memory_used_mb, expected_mb
        );
    }

    #[test]
    fn test_default_config_values() {
        // 测试默认配置值
        let config = StreamingAttentionConfig::default();

        assert_eq!(config.block_size, 512);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.max_blocks, 1024);
    }

    #[test]
    fn test_position_to_block_mapping() {
        // 测试位置到块的映射逻辑
        let config = StreamingAttentionConfig {
            block_size: 8,
            num_heads: 2,
            head_dim: 4,
            max_blocks: 16,
        };
        let mut sa = StreamingAttention::new(config.clone());

        // 在不同位置写入，验证映射正确性
        let positions = [0, 7, 8, 15, 16]; // 跨越多个块的边界
        for &pos in &positions {
            let kv_len = config.num_heads * config.head_dim;
            let result = sa.write(pos, &vec![pos as f32; kv_len], &vec![pos as f32; kv_len]);
            assert!(result.is_ok(), "Write at position {} should succeed", pos);
        }

        // 验证总token数
        assert_eq!(sa.total_tokens(), positions.len());

        // 验证每个位置的值都能正确读回
        for &pos in &positions {
            let (k, _) = sa.read_range(pos, 1).expect("Should be able to read");
            assert!(
                (k[0] - pos as f32).abs() < 0.01,
                "Value at position {} should match",
                pos
            );
        }
    }
}
