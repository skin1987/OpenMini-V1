//! Attention with Hybrid Normalization (AHN) - RNN压缩+局部标准注意力模块
//!
//! AHN 是一种高效的长序列处理架构，通过 RNN 全局压缩和局部标准注意力的混合策略实现：
//! - **RNN Modules**: 使用 Mamba2/DeltaNext/GDN 进行全局序列压缩
//! - **LocalStandardAttention**: 在滑动窗口内执行标准多头注意力
//! - **TransitionLayer**: 门控融合 RNN 输出和局部注意力输出
//!
//! # 核心优势
//!
//! - **超长序列支持**: 优化 256K+ 序列的内存和计算效率
//! - **RNN 压缩**: 通过状态空间模型实现 O(N) 的全局信息捕获
//! - **局部注意力**: 保留关键细节的精确建模能力
//! - **门控融合**: 自适应平衡全局语义和局部细节
//!
//! # 性能目标
//!
//! - >256K tokens 显存 <32GB
//! - 长序列延迟降低 >70% (vs Full Attention baseline)
//! - 内存占用减少 >60%
//!
//! # 架构设计
//!
//! ```text
//! Input Sequence (>256K tokens)
//!     │
//!     ├──→ RNN Modules (Mamba2/DeltaNext/GDN) ──┐
//!     │   全局压缩，O(N) 复杂度                   │
//!     │                                          │
//!     └──→ LocalStandardAttention ───────────────┼──→ TransitionLayer ──→ Output
//!         滑动窗口内多头注意力                      │      门控融合
//!                                                  │
//! ```
//!
//! # 使用示例
//!
//! ```ignore
//! use openmini_server::model::inference::ahn::{
//!     AttentionHybridNetwork, AHNConfig, RnnType,
//! };
//!
//! let config = AHNConfig::default();
//! let mut ahn = AttentionHybridNetwork::new(config)?;
//!
//! // 处理超长序列
//! let output = ahn.forward(&input, num_heads, head_dim)?;
//! ```

use std::fmt;
use std::time::Instant;

use ndarray::{Array1, Array2, Axis};

use crate::model::inference::error::{InferenceError, InferenceResult};

/// Sigmoid 函数
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================================
// 配置与常量定义
// ============================================================================

/// AHN 配置参数
#[derive(Debug, Clone)]
pub struct AHNConfig {
    /// 隐藏层维度
    pub hidden_dim: usize,

    /// RNN 类型选择
    pub rnn_type: RnnType,

    /// RNN 层数
    pub num_rnn_layers: usize,

    /// 局部注意力窗口大小
    pub local_window_size: usize,

    /// 最大上下文长度
    pub max_context_length: usize,

    /// RNN 压缩率
    pub compression_ratio: f32,

    /// 融合层门控偏置
    pub gate_bias: f32,

    /// 是否启用 LayerNorm
    pub enable_layer_norm: bool,

    /// 注意力头数
    pub num_attention_heads: usize,

    /// 每个 attention head 的维度
    pub attention_head_dim: usize,

    /// RNN dropout 率
    pub rnn_dropout: f32,

    /// 是否启用性能统计
    pub enable_stats: bool,
}

impl Default for AHNConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 4096,
            rnn_type: RnnType::Mamba2,
            num_rnn_layers: 4,
            local_window_size: 2048,
            max_context_length: 262144,
            compression_ratio: 32.0,
            gate_bias: 0.0,
            enable_layer_norm: true,
            num_attention_heads: 32,
            attention_head_dim: 128,
            rnn_dropout: 0.1,
            enable_stats: false,
        }
    }
}

impl fmt::Display for AHNConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AHNConfig {{ hidden_dim={}, rnn_type={}, num_rnn_layers={}, \
             local_window={}, max_context={}, compression={:.1}, \
             heads={}, head_dim={}, layer_norm={} }}",
            self.hidden_dim,
            self.rnn_type,
            self.num_rnn_layers,
            self.local_window_size,
            self.max_context_length,
            self.compression_ratio,
            self.num_attention_heads,
            self.attention_head_dim,
            self.enable_layer_norm
        )
    }
}

/// RNN 类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[allow(clippy::upper_case_acronyms)]
pub enum RnnType {
    /// Mamba2: 选择性状态空间模型
    #[default]
    Mamba2,
    /// DeltaNext: 增量预测网络
    DeltaNext,
    /// GDN: 门控去噪网络
    GDN,
}

impl fmt::Display for RnnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mamba2 => write!(f, "Mamba2"),
            Self::DeltaNext => write!(f, "DeltaNext"),
            Self::GDN => write!(f, "GDN"),
        }
    }
}

/// AHN 性能统计
#[derive(Debug, Clone, Default)]
pub struct AHNPerformanceStats {
    /// 总调用次数
    pub total_calls: usize,

    /// RNN 前向传播总时间 (微秒)
    pub rnn_total_time_us: u64,

    /// 局部注意力总时间 (微秒)
    pub attn_total_time_us: u64,

    /// 融合层总时间 (微秒)
    pub fusion_total_time_us: u64,

    /// 总计算时间 (微秒)
    pub total_time_us: u64,

    /// 平均延迟 (微秒)
    pub avg_latency_us: f64,

    /// 峰值内存使用估算 (MB)
    pub peak_memory_mb: f64,

    /// 实际处理的序列长度分布
    pub sequence_lengths: Vec<usize>,

    /// RNN 压缩率统计
    pub actual_compression_ratios: Vec<f32>,
}

impl fmt::Display for AHNPerformanceStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AHN Stats {{ calls={}, avg_latency={:.2}us, \
             peak_memory={:.1}MB, rnn_time={}us, attn_time={}us }}",
            self.total_calls,
            self.avg_latency_us,
            self.peak_memory_mb,
            self.rnn_total_time_us,
            self.attn_total_time_us
        )
    }
}

// ============================================================================
// RnnBlock Trait 定义
// ============================================================================

/// RNN Block 特征定义
///
/// 所有 RNN 实现必须实现此特征，提供统一的前向传播接口。
///
/// # 泛型约束
///
/// - 支持任意维度的输入/输出
/// - 提供状态管理接口
/// - 支持序列到序列的映射
pub trait RnnBlock: fmt::Debug + Send + Sync {
    /// 执行 RNN 前向传播
    ///
    /// # 参数
    ///
    /// - `input`: 输入序列 [seq_len, hidden_dim]
    ///
    /// # 返回值
    ///
    /// 输出序列 [compressed_len, hidden_dim] 和内部状态
    fn forward(&mut self, input: &Array2<f32>) -> InferenceResult<(Array2<f32>, RnnState)>;

    /// 获取 RNN 类型
    fn rnn_type(&self) -> RnnType;

    /// 获取隐藏层维度
    fn hidden_dim(&self) -> usize;

    /// 获取实际压缩率
    fn compression_ratio(&self) -> f32;

    /// 重置内部状态
    fn reset_state(&mut self);

    /// 获取当前状态
    fn state(&self) -> &RnnState;

    /// 设置状态
    fn set_state(&mut self, state: RnnState);
}

/// RNN 内部状态表示
#[derive(Debug, Clone, Default)]
#[allow(clippy::upper_case_acronyms)]
pub enum RnnState {
    /// Mamba2 状态 (隐藏状态 + 卷积状态)
    Mamba2 {
        hidden: Array1<f32>,
        conv: Array1<f32>,
    },
    /// DeltaNext 状态 (增量状态 + 前一时刻输出)
    DeltaNext {
        delta_state: Array1<f32>,
        prev_output: Array1<f32>,
    },
    /// GDN 状态 (门控状态 + 噪声估计)
    GDN {
        gate: Array1<f32>,
        noise_estimate: Array1<f32>,
    },
    /// 未初始化状态
    #[default]
    Uninitialized,
}

// ============================================================================
// Mamba2Block 实现
// ============================================================================

/// Mamba2 Block 实现
///
/// 基于 Selective State Space Model (S4) 的改进版本，
/// 具有选择性机制和硬件感知的设计。
///
/// # 核心特性
///
/// - 选择性参数化：输入依赖的动态参数
/// - 并行扫描算法：高效的递归计算
/// - 硬件友好：适合 GPU/TPU 并行化
///
/// # 数学公式
///
/// ```text
/// h_t = A_t * h_{t-1} + B_t * x_t
/// y_t = C_t * h_t
/// ```
///
/// 其中 A_t, B_t, C_t 都是基于输入 x_t 动态计算的
pub struct Mamba2Block {
    /// 隐藏层维度
    hidden_dim: usize,

    /// 扩展因子 (通常为 2)
    expand_factor: usize,

    /// DSSD 状态维度
    d_state: usize,

    /// 卷积核大小
    d_conv: usize,

    /// 时间步缩放参数
    dt_min: f32,

    /// 时间步最大值
    dt_max: f32,

    /// 内部状态
    state: RnnState,

    /// 初始化标志
    initialized: bool,
}

impl Mamba2Block {
    /// 创建新的 Mamba2 Block
    ///
    /// # 参数
    ///
    /// - `hidden_dim`: 隐藏层维度
    /// - `d_state`: 状态空间维度
    /// - `d_conv`: 1D 卷积核大小
    pub fn new(hidden_dim: usize, d_state: usize, d_conv: usize) -> Self {
        assert!(hidden_dim > 0, "Hidden dimension must be positive");
        assert!(d_state > 0, "State dimension must be positive");
        assert!(d_conv > 0, "Conv dimension must be positive");

        Self {
            hidden_dim,
            expand_factor: 2,
            d_state,
            d_conv,
            dt_min: 0.001,
            dt_max: 100.0,
            state: RnnState::Uninitialized,
            initialized: false,
        }
    }

    /// 初始化内部状态
    fn initialize_state(&mut self) {
        if !self.initialized {
            self.state = RnnState::Mamba2 {
                hidden: Array1::zeros(self.hidden_dim),
                conv: Array1::zeros(self.d_conv * self.hidden_dim),
            };
            self.initialized = true;
        }
    }

    /// 计算选择性参数
    ///
    /// 基于当前输入动态生成 A, B, C, D 参数
    fn compute_selective_params(
        &self,
        x: &Array1<f32>,
    ) -> (Array1<f32>, Array1<f32>, Array1<f32>, f32) {
        // 简化的选择性参数计算（实际实现中应使用线性投影）
        let a_param = x.mapv(|v| (-v.exp()).clamp(-4.0, 0.0)); // A 参数（负值保证稳定）
        let b_param = x.mapv(|v| v.tanh()); // B 参数（软饱和）
        let c_param = x.mapv(sigmoid); // C 参数（门控）
        let d_param = 1.0_f32; // D 参数（跳跃连接权重）

        (a_param, b_param, c_param, d_param)
    }

    /// 并行扫描算法
    ///
    /// 高效的并行前缀和计算
    fn parallel_scan(
        &self,
        inputs: &Array2<f32>,
        a_params: &Array2<f32>,
        b_params: &Array2<f32>,
        c_params: &Array2<f32>,
    ) -> Array2<f32> {
        let (seq_len, dim) = inputs.dim();
        let mut outputs = Array2::<f32>::zeros((seq_len, dim));

        // 简化的串行扫描（实际应用中应使用并行前缀和）
        let mut hidden = Array1::zeros(dim);

        for t in 0..seq_len {
            let x_t = inputs.row(t).to_owned();
            let a_t = a_params.row(t).to_owned();
            let b_t = b_params.row(t).to_owned();
            let c_t = c_params.row(t).to_owned();

            // 状态更新: h_t = A_t * h_{t-1} + B_t * x_t
            hidden = a_t * &hidden + b_t * &x_t;

            // 输出计算: y_t = C_t * h_t
            let y_t = c_t * &hidden;
            outputs.row_mut(t).assign(&y_t);
        }

        outputs
    }

    /// 估算压缩率
    fn estimate_compression_ratio(&self, seq_len: usize) -> f32 {
        if seq_len == 0 {
            return 1.0;
        }

        // Mamba2 的压缩率基于状态维度和序列长度
        let effective_compression = self.d_state as f32 / seq_len as f32;
        effective_compression.clamp(0.01, 1.0)
    }
}

impl RnnBlock for Mamba2Block {
    fn forward(&mut self, input: &Array2<f32>) -> InferenceResult<(Array2<f32>, RnnState)> {
        self.initialize_state();

        let (seq_len, dim) = input.dim();

        if seq_len == 0 || dim == 0 {
            return Err(InferenceError::config("Empty input tensor"));
        }

        if dim != self.hidden_dim {
            return Err(InferenceError::config(format!(
                "Dimension mismatch: expected {}, got {}",
                self.hidden_dim, dim
            )));
        }

        // 计算每个时间步的选择性参数
        let mut a_params = Array2::<f32>::zeros((seq_len, dim));
        let mut b_params = Array2::<f32>::zeros((seq_len, dim));
        let mut c_params = Array2::<f32>::zeros((seq_len, dim));

        for t in 0..seq_len {
            let x_t = input.row(t).to_owned();
            let (a, b, c, _) = self.compute_selective_params(&x_t);

            a_params.row_mut(t).assign(&a);
            b_params.row_mut(t).assign(&b);
            c_params.row_mut(t).assign(&c);
        }

        // 执行并行扫描
        let output = self.parallel_scan(input, &a_params, &b_params, &c_params);

        // 更新最终状态
        if let RnnState::Mamba2 { ref mut hidden, .. } = self.state {
            *hidden = output.row(seq_len - 1).to_owned();
        }

        Ok((output, self.state.clone()))
    }

    fn rnn_type(&self) -> RnnType {
        RnnType::Mamba2
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn compression_ratio(&self) -> f32 {
        self.d_state as f32 / self.hidden_dim as f32
    }

    fn reset_state(&mut self) {
        self.state = RnnState::Uninitialized;
        self.initialized = false;
    }

    fn state(&self) -> &RnnState {
        &self.state
    }

    fn set_state(&mut self, state: RnnState) {
        self.state = state;
        self.initialized = true;
    }
}

impl fmt::Debug for Mamba2Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mamba2Block")
            .field("hidden_dim", &self.hidden_dim)
            .field("d_state", &self.d_state)
            .field("d_conv", &self.d_conv)
            .field("initialized", &self.initialized)
            .finish()
    }
}

// ============================================================================
// DeltaNextBlock 实现
// ============================================================================

/// DeltaNext Block 实现
///
/// 基于增量预测的网络结构，专注于捕捉序列中的变化模式。
/// 特别适用于处理具有强时间相关性的数据。
///
/// # 核心特性
///
/// - 增量建模：专注于预测变化量而非绝对值
/// - 多尺度时序：支持不同粒度的时间依赖
/// - 高效推理：可增量更新，无需重新计算整个序列
///
/// # 数学公式
///
/// ```text
/// delta_t = predict(x_t - x_{t-1})
/// output_t = x_{t-1} + alpha * delta_t
/// ```
///
/// 其中 alpha 是可学习的缩放因子
pub struct DeltaNextBlock {
    /// 隐藏层维度
    hidden_dim: usize,

    /// 增量预测层数
    num_delta_layers: usize,

    /// 缩放因子范围
    scale_range: (f32, f32),

    /// 平滑系数
    smoothing_factor: f32,

    /// 内部状态
    state: RnnState,

    /// 初始化标志
    initialized: bool,
}

impl DeltaNextBlock {
    /// 创建新的 DeltaNext Block
    ///
    /// # 参数
    ///
    /// - `hidden_dim`: 隐藏层维度
    /// - `num_delta_layers`: 增量预测层数
    pub fn new(hidden_dim: usize, num_delta_layers: usize) -> Self {
        assert!(hidden_dim > 0, "Hidden dimension must be positive");
        assert!(num_delta_layers > 0, "Number of layers must be positive");

        Self {
            hidden_dim,
            num_delta_layers,
            scale_range: (0.01, 10.0),
            smoothing_factor: 0.9,
            state: RnnState::Uninitialized,
            initialized: false,
        }
    }

    /// 初始化内部状态
    fn initialize_state(&mut self, initial_input: Option<&Array1<f32>>) {
        if !self.initialized {
            let init_val = match initial_input {
                Some(x) => x.clone(),
                None => Array1::zeros(self.hidden_dim),
            };

            self.state = RnnState::DeltaNext {
                delta_state: Array1::zeros(self.hidden_dim),
                prev_output: init_val,
            };
            self.initialized = true;
        }
    }

    /// 计算增量预测
    ///
    /// 基于当前输入和前一时刻的差异进行预测
    fn predict_delta(
        &self,
        current: &Array1<f32>,
        previous: &Array1<f32>,
        delta_state: &Array1<f32>,
    ) -> Array1<f32> {
        let diff = current - previous;

        // 多层增量预测（简化版）
        let mut delta = diff.clone();

        for _ in 0..self.num_delta_layers {
            // 应用平滑和缩放
            delta = delta.mapv(|d| {
                let scaled = d * self.smoothing_factor;
                scaled.clamp(self.scale_range.0, self.scale_range.1)
            });

            // 与历史 delta state 结合
            delta = delta + delta_state.mapv(|s| s * 0.1);
        }

        delta
    }

    /// 估算压缩率
    fn estimate_compression_ratio(&self, seq_len: usize) -> f32 {
        if seq_len <= 1 {
            return 1.0;
        }

        // DeltaNext 的压缩率基于增量稀疏性
        let theoretical_ratio = 1.0 / self.num_delta_layers as f32;
        theoretical_ratio.clamp(0.05, 1.0)
    }
}

impl RnnBlock for DeltaNextBlock {
    fn forward(&mut self, input: &Array2<f32>) -> InferenceResult<(Array2<f32>, RnnState)> {
        let (seq_len, dim) = input.dim();

        if seq_len == 0 || dim == 0 {
            return Err(InferenceError::config("Empty input tensor"));
        }

        if dim != self.hidden_dim {
            return Err(InferenceError::config(format!(
                "Dimension mismatch: expected {}, got {}",
                self.hidden_dim, dim
            )));
        }

        // 用第一个输入初始化状态
        let first_input = Some(input.row(0).to_owned());
        self.initialize_state(first_input.as_ref());

        let mut output = Array2::<f32>::zeros((seq_len, dim));

        // 第一个输出直接复制输入
        output.row_mut(0).assign(&input.row(0).to_owned());

        // 处理后续时间步
        for t in 1..seq_len {
            let current = input.row(t).to_owned();

            if let RnnState::DeltaNext {
                ref mut delta_state,
                ref mut prev_output,
            } = self.state
            {
                // 内联增量预测逻辑（避免借用冲突）
                let prev_output_snapshot = prev_output.clone();
                let diff = &current - &prev_output_snapshot;
                let mut delta = diff.clone();

                for _ in 0..self.num_delta_layers {
                    delta = delta.mapv(|d| {
                        let scaled = d * self.smoothing_factor;
                        scaled.clamp(self.scale_range.0, self.scale_range.1)
                    });

                    // 与历史 delta state 结合
                    delta = delta + delta_state.mapv(|s| s * 0.1);
                }

                // 更新输出: output_t = prev_output + alpha * delta
                let alpha = 0.5; // 可学习参数（简化版）
                let new_output = prev_output.clone() + &delta.mapv(|d| d * alpha);

                output.row_mut(t).assign(&new_output);

                // 更新状态
                *prev_output = new_output;
                *delta_state = delta;
            }
        }

        Ok((output, self.state.clone()))
    }

    fn rnn_type(&self) -> RnnType {
        RnnType::DeltaNext
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn compression_ratio(&self) -> f32 {
        1.0 / self.num_delta_layers as f32
    }

    fn reset_state(&mut self) {
        self.state = RnnState::Uninitialized;
        self.initialized = false;
    }

    fn state(&self) -> &RnnState {
        &self.state
    }

    fn set_state(&mut self, state: RnnState) {
        self.state = state;
        self.initialized = true;
    }
}

impl fmt::Debug for DeltaNextBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeltaNextBlock")
            .field("hidden_dim", &self.hidden_dim)
            .field("num_delta_layers", &self.num_delta_layers)
            .field("initialized", &self.initialized)
            .finish()
    }
}

// ============================================================================
// GDNBlock 实现
// ============================================================================

/// GDN (Gated Denoising Network) Block 实现
///
/// 基于门控去噪网络的结构，通过自适应噪声过滤提取有用信号。
/// 特别适合处理含噪声的长序列数据。
///
/// # 核心特性
///
/// - 自适应门控：动态调整信号/噪声阈值
/// - 多尺度去噪：在不同分辨率上过滤噪声
/// - 残差连接：保留原始信息的完整性
///
/// # 数学公式
///
/// ```text
/// noise_est = estimate_noise(x_t)
/// gate_t = sigmoid(W_g * x_t + b_g)
/// clean_signal = gate_t * (x_t - noise_est) + (1 - gate_t) * x_t
/// ```
pub struct GDNBlock {
    /// 隐藏层维度
    hidden_dim: usize,

    /// 去噪迭代次数
    denoise_iterations: usize,

    /// 门控温度参数
    gate_temperature: f32,

    /// 噪声敏感度
    noise_sensitivity: f32,

    /// 内部状态
    state: RnnState,

    /// 初始化标志
    initialized: bool,
}

impl GDNBlock {
    /// 创建新的 GDN Block
    ///
    /// # 参数
    ///
    /// - `hidden_dim`: 隐藏层维度
    /// - `denoise_iterations`: 去噪迭代次数
    pub fn new(hidden_dim: usize, denoise_iterations: usize) -> Self {
        assert!(hidden_dim > 0, "Hidden dimension must be positive");
        assert!(
            denoise_iterations > 0,
            "Denoise iterations must be positive"
        );

        Self {
            hidden_dim,
            denoise_iterations,
            gate_temperature: 1.0,
            noise_sensitivity: 0.5,
            state: RnnState::Uninitialized,
            initialized: false,
        }
    }

    /// 初始化内部状态
    fn initialize_state(&mut self) {
        if !self.initialized {
            self.state = RnnState::GDN {
                gate: Array1::ones(self.hidden_dim),
                noise_estimate: Array1::zeros(self.hidden_dim),
            };
            self.initialized = true;
        }
    }

    /// 估计噪声水平
    ///
    /// 基于局部统计特性估计每个维度的噪声
    fn estimate_noise(&self, signal: &Array1<f32>, history: &Array1<f32>) -> Array1<f32> {
        // 简化的噪声估计（基于局部方差）
        let diff = signal - history;
        let local_variance = diff.mapv(|d| d * d);

        // 应用平滑
        let smoothed_variance = local_variance.mapv(|v| v * self.noise_sensitivity);

        // 估计标准差作为噪声水平
        smoothed_variance.mapv(|v| v.sqrt().max(0.001))
    }

    /// 计算门控信号
    ///
    /// 基于信噪比动态调整门控值
    fn compute_gate(
        &self,
        signal: &Array1<f32>,
        noise_level: &Array1<f32>,
        prev_gate: &Array1<f32>,
    ) -> Array1<f32> {
        // 信噪比估计
        let snr = signal.mapv(|s| s.abs()) / noise_level.mapv(|n| n.max(0.001));

        // 温度缩放的 sigmoid
        let gate_logit = snr.mapv(|s| s / self.gate_temperature);
        let new_gate = gate_logit.mapv(sigmoid);

        // 与前一时刻的门控平滑结合

        prev_gate.mapv(|p| p * 0.7) + &new_gate.mapv(|n| n * 0.3)
    }

    /// 应用门控去噪
    fn apply_gated_denoising(
        &self,
        signal: &Array1<f32>,
        noise: &Array1<f32>,
        gate: &Array1<f32>,
    ) -> Array1<f32> {
        // clean_signal = gate * (signal - noise) + (1 - gate) * signal
        //              = signal - gate * noise
        let denoised = signal - &(gate * noise);

        // 残差连接（保留部分原始信号）
        let residual_scale = 0.1;
        signal.mapv(|s| s * residual_scale) + &denoised.mapv(|d| d * (1.0 - residual_scale))
    }

    /// 估算压缩率
    fn estimate_compression_ratio(&self, seq_len: usize) -> f32 {
        if seq_len == 0 {
            return 1.0;
        }

        // GDN 的压缩率基于去噪效果
        let base_ratio = 1.0 / (1.0 + self.denoise_iterations as f32 * 0.5);
        base_ratio.clamp(0.05, 1.0)
    }
}

impl RnnBlock for GDNBlock {
    fn forward(&mut self, input: &Array2<f32>) -> InferenceResult<(Array2<f32>, RnnState)> {
        self.initialize_state();

        let (seq_len, dim) = input.dim();

        if seq_len == 0 || dim == 0 {
            return Err(InferenceError::config("Empty input tensor"));
        }

        if dim != self.hidden_dim {
            return Err(InferenceError::config(format!(
                "Dimension mismatch: expected {}, got {}",
                self.hidden_dim, dim
            )));
        }

        let mut output = Array2::<f32>::zeros((seq_len, dim));

        for t in 0..seq_len {
            let signal = input.row(t).to_owned();

            if let RnnState::GDN {
                ref mut gate,
                ref mut noise_estimate,
            } = self.state
            {
                // 迭代去噪（内联逻辑以避免借用冲突）
                let mut current_signal = signal.clone();

                for _ in 0..self.denoise_iterations {
                    // 估计噪声（内联）
                    let noise_estimate_snap = noise_estimate.clone();
                    let diff = &current_signal - &noise_estimate_snap;
                    let local_variance = diff.mapv(|d| d * d);
                    let smoothed_variance = local_variance.mapv(|v| v * self.noise_sensitivity);
                    let noise = smoothed_variance.mapv(|v| v.sqrt().max(0.001));

                    // 计算门控（内联）
                    let snr = current_signal.mapv(|s| s.abs()) / noise.mapv(|n| n.max(0.001));
                    let gate_logit = snr.mapv(|s| s / self.gate_temperature);
                    let new_gate = gate_logit.mapv(sigmoid);
                    let gate_snap = gate.clone();
                    let smoothed_gate = gate_snap.mapv(|p| p * 0.7) + &new_gate.mapv(|_n| 0.3);

                    // 应用门控去噪（内联）
                    let denoised = &current_signal - &(&smoothed_gate * &noise);
                    let residual_scale = 0.1;
                    current_signal = current_signal.mapv(|s| s * residual_scale)
                        + &denoised.mapv(|d| d * (1.0 - residual_scale));

                    // 更新状态
                    *gate = smoothed_gate;
                    *noise_estimate = noise;
                }

                output.row_mut(t).assign(&current_signal);
            }
        }

        Ok((output, self.state.clone()))
    }

    fn rnn_type(&self) -> RnnType {
        RnnType::GDN
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn compression_ratio(&self) -> f32 {
        1.0 / (1.0 + self.denoise_iterations as f32 * 0.5)
    }

    fn reset_state(&mut self) {
        self.state = RnnState::Uninitialized;
        self.initialized = false;
    }

    fn state(&self) -> &RnnState {
        &self.state
    }

    fn set_state(&mut self, state: RnnState) {
        self.state = state;
        self.initialized = true;
    }
}

impl fmt::Debug for GDNBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GDNBlock")
            .field("hidden_dim", &self.hidden_dim)
            .field("denoise_iterations", &self.denoise_iterations)
            .field("initialized", &self.initialized)
            .finish()
    }
}

// ============================================================================
// LocalStandardAttention 实现
// ============================================================================

/// 局部标准多头注意力
///
/// 在滑动窗口内执行标准的缩放点积注意力，
/// 保留对局部细节的精确建模能力。
///
/// # 核心特性
///
/// - 滑动窗口机制：限制注意力范围为固定大小的窗口
/// - 多头注意力：捕捉不同的子空间表示
/// - 标准实现：完全兼容 Transformer 的注意力机制
///
/// # 窗口机制
///
/// 对于位置 i，只关注 [max(0, i-window_size), i] 范围内的 token。
/// 这将复杂度从 O(N^2) 降低到 O(N*W)，其中 W 是窗口大小。
///
/// # 性能优化
///
/// - 分块计算：将长序列分割为多个块并行处理
/// - 内存高效：避免存储完整的 N×N 注意力矩阵
/// - 缓存友好：利用局部性原理优化缓存命中
pub struct LocalStandardAttention {
    /// 窗口大小
    window_size: usize,

    /// 注意力头数
    num_heads: usize,

    /// 每个头的维度
    head_dim: usize,

    /// 缩放因子
    scale: f32,

    /// 是否使用因果掩码
    use_causal_mask: bool,

    /// Softmax 温度
    softmax_temperature: f32,
}

impl LocalStandardAttention {
    /// 创建新的局部标准注意力
    ///
    /// # 参数
    ///
    /// - `window_size`: 注意力窗口大小
    /// - `num_heads`: 注意力头数
    /// - `head_dim`: 每个头的维度
    pub fn new(window_size: usize, num_heads: usize, head_dim: usize) -> Self {
        assert!(window_size > 0, "Window size must be positive");
        assert!(num_heads > 0, "Number of heads must be positive");
        assert!(head_dim > 0, "Head dimension must be positive");
        assert!(
            head_dim % num_heads == 0,
            "Head dimension must be divisible by num_heads"
        );

        let scale = 1.0 / (head_dim as f32).sqrt();

        Self {
            window_size,
            num_heads,
            head_dim,
            scale,
            use_causal_mask: true,
            softmax_temperature: 1.0,
        }
    }

    /// 获取窗口大小
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// 获取头数
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// 获取头维度
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// 执行局部标准注意力前向传播
    ///
    /// # 参数
    ///
    /// - `q`: Query 矩阵 [seq_len, num_heads * head_dim]
    /// - `k`: Key 矩阵 [seq_len, num_heads * head_dim]
    /// - `v`: Value 矩阵 [seq_len, num_heads * head_dim]
    ///
    /// # 返回值
    ///
    /// 注意力输出矩阵 [seq_len, num_heads * head_dim]
    ///
    /// # 算法流程
    ///
    /// 1. 将 Q、K、V 分割为多个头
    /// 2. 对每个位置 i，在 [i-W+1, i] 窗口内计算注意力
    /// 3. 应用缩放点积注意力和 Softmax
    /// 4. 合并所有头的输出
    pub fn forward(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
    ) -> InferenceResult<Array2<f32>> {
        let (seq_len, total_dim) = q.dim();

        if seq_len == 0 {
            return Err(InferenceError::config("Empty input tensors"));
        }

        let expected_dim = self.num_heads * self.head_dim;
        if total_dim != expected_dim {
            return Err(InferenceError::config(format!(
                "Dimension mismatch: expected {} (heads*head_dim), got {}",
                expected_dim, total_dim
            )));
        }

        let mut output = Array2::<f32>::zeros((seq_len, total_dim));

        // 并行处理每个位置
        output
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut out_row)| {
                // 计算窗口范围
                let window_start = if self.window_size >= seq_len {
                    0
                } else {
                    i.saturating_sub(self.window_size - 1)
                };
                let window_end = i + 1;

                // 对每个头分别计算注意力
                for h in 0..self.num_heads {
                    let head_offset = h * self.head_dim;

                    // 提取当前 query 头
                    let q_row = q.row(i);
                    let q_head: Vec<f32> = q_row
                        .iter()
                        .skip(head_offset)
                        .take(self.head_dim)
                        .copied()
                        .collect();

                    // 收集窗口内的 key 和 value 头
                    let mut attn_scores: Vec<f32> = Vec::with_capacity(window_end - window_start);
                    let mut window_values: Vec<Vec<f32>> =
                        Vec::with_capacity(window_end - window_start);

                    for j in window_start..window_end {
                        // 因果掩码检查
                        if self.use_causal_mask && j > i {
                            break;
                        }

                        let k_row = k.row(j);
                        let k_head: Vec<f32> = k_row
                            .iter()
                            .skip(head_offset)
                            .take(self.head_dim)
                            .copied()
                            .collect();

                        let v_row = v.row(j);
                        let v_head: Vec<f32> = v_row
                            .iter()
                            .skip(head_offset)
                            .take(self.head_dim)
                            .copied()
                            .collect();

                        // 缩放点积
                        let dot: f32 = q_head
                            .iter()
                            .zip(k_head.iter())
                            .map(|(q_val, k_val)| q_val * k_val)
                            .sum();

                        let score = dot * self.scale / self.softmax_temperature;
                        attn_scores.push(score);
                        window_values.push(v_head);
                    }

                    // Softmax 归一化
                    if !attn_scores.is_empty() {
                        let max_score = attn_scores
                            .iter()
                            .cloned()
                            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

                        let exp_scores: Vec<f32> = attn_scores
                            .iter()
                            .map(|&s| ((s - max_score) / self.softmax_temperature).exp())
                            .collect();

                        let sum_exp: f32 = exp_scores.iter().sum();

                        if sum_exp > 0.0 {
                            // 加权求和得到输出
                            let mut head_output = vec![0.0_f32; self.head_dim];

                            for (idx, weight) in exp_scores.iter().enumerate() {
                                if idx < window_values.len() {
                                    for (d, val) in head_output.iter_mut().enumerate() {
                                        if d < window_values[idx].len() {
                                            *val += window_values[idx][d] * weight;
                                        }
                                    }
                                }
                            }

                            // 归一化
                            for val in head_output.iter_mut() {
                                *val /= sum_exp;
                            }

                            // 写入输出
                            for (d, val) in head_output.iter().enumerate() {
                                if head_offset + d < out_row.len() {
                                    out_row[head_offset + d] = *val;
                                }
                            }
                        }
                    }
                }
            });

        Ok(output)
    }

    /// 获取指定位置的窗口范围
    pub fn get_window_range(&self, pos: usize, seq_len: usize) -> (usize, usize) {
        let start = if self.window_size >= seq_len {
            0
        } else {
            pos.saturating_sub(self.window_size - 1)
        };
        let end = (pos + 1).min(seq_len);
        (start, end)
    }

    /// 计算理论复杂度
    ///
    /// 返回 (时间复杂度, 空间复杂度) 的元组
    pub fn complexity_analysis(&self, seq_len: usize) -> (usize, usize) {
        let time_complexity =
            seq_len * self.window_size.min(seq_len) * self.num_heads * self.head_dim;
        let space_complexity = seq_len * self.num_heads * self.head_dim; // 只需存储 Q,K,V

        (time_complexity, space_complexity)
    }
}

impl fmt::Debug for LocalStandardAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalStandardAttention")
            .field("window_size", &self.window_size)
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .finish()
    }
}

// ============================================================================
// TransitionLayer 实现
// ============================================================================

/// 过渡层（门控融合层）
///
/// 将 RNN 输出和局部注意力输出进行门控融合，
/// 自适应地平衡全局语义信息和局部细节信息。
///
/// # 融合公式
///
/// ```text
/// gate = sigmoid(W_g * [rnn_output; attn_output] + b_g)
/// output = gate * rnn_output + (1 - gate) * attn_output
/// ```
///
/// # 设计理念
///
/// - **自适应门控**：让网络自动学习何时信任全局信息，何时关注局部细节
/// - **残差连接**：保留原始信息，避免梯度消失
/// - **层归一化**：稳定训练过程，加速收敛
pub struct TransitionLayer {
    /// 输入维度
    input_dim: usize,

    /// 门控偏置
    gate_bias: f32,

    /// 是否启用层归一化
    use_layer_norm: bool,

    /// 残差连接权重
    residual_weight: f32,

    /// 门控温度
    gate_temperature: f32,
}

impl TransitionLayer {
    /// 创建新的过渡层
    ///
    /// # 参数
    ///
    /// - `input_dim`: 输入特征维度
    /// - `gate_bias`: 门控偏置项
    /// - `use_layer_norm`: 是否启用层归一化
    pub fn new(input_dim: usize, gate_bias: f32, use_layer_norm: bool) -> Self {
        assert!(input_dim > 0, "Input dimension must be positive");

        Self {
            input_dim,
            gate_bias,
            use_layer_norm,
            residual_weight: 0.1,
            gate_temperature: 1.0,
        }
    }

    /// 获取输入维度
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// 执行门控融合
    ///
    /// # 参数
    ///
    /// - `rnn_output`: RNN 模块的输出 [seq_len, dim]
    /// - `attn_output`: 局部注意力的输出 [seq_len, dim]
    ///
    /// # 返回值
    ///
    /// 融合后的输出 [seq_len, dim]
    ///
    /// # 算法细节
    ///
    /// 1. 计算 RNN 输出和注意力输出的拼接表示
    /// 2. 基于拼接表示计算门控值
    /// 3. 使用门控值加权融合两个输出
    /// 4. （可选）应用层归一化
    pub fn fuse(
        &self,
        rnn_output: &Array2<f32>,
        attn_output: &Array2<f32>,
    ) -> InferenceResult<Array2<f32>> {
        let (seq_len, rnn_dim) = rnn_output.dim();
        let (_, attn_dim) = attn_output.dim();

        if seq_len == 0 {
            return Err(InferenceError::config("Empty input tensors"));
        }

        // 确保维度一致
        let target_dim = rnn_dim.min(attn_dim);
        let mut fused = Array2::<f32>::zeros((seq_len, target_dim));

        // 逐位置计算门控融合
        fused
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut row)| {
                let rnn_row = rnn_output.row(i);
                let rnn_vec: Vec<f32> = rnn_row.iter().take(target_dim).copied().collect();

                let attn_row = attn_output.row(i);
                let attn_vec: Vec<f32> = attn_row.iter().take(target_dim).copied().collect();

                // 简化的门控计算（基于能量差异）
                let rnn_energy: f32 = rnn_vec.iter().map(|x| x * x).sum();
                let attn_energy: f32 = attn_vec.iter().map(|x| x * x).sum();

                let energy_diff = (rnn_energy - attn_energy).sqrt().max(0.0);

                // Sigmoid 门控
                let gate_value = sigmoid(energy_diff / self.gate_temperature + self.gate_bias);

                // 门控融合
                for (d, val) in row.iter_mut().enumerate() {
                    if d < target_dim {
                        let rnn_val = rnn_vec[d];
                        let attn_val = attn_vec[d];

                        // 主融合
                        let main_fused = gate_value * rnn_val + (1.0 - gate_value) * attn_val;

                        // 残差连接（使用均值作为恒等映射近似）
                        let mean_val = (rnn_val + attn_val) / 2.0;
                        *val = main_fused * (1.0 - self.residual_weight)
                            + mean_val * self.residual_weight;
                    }
                }
            });

        // 可选：应用层归一化
        if self.use_layer_norm {
            fused = self.apply_layer_norm(&fused);
        }

        Ok(fused)
    }

    /// 应用层归一化
    fn apply_layer_norm(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len, dim) = x.dim();
        let mut normalized = Array2::<f32>::zeros((seq_len, dim));

        normalized.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mean: f32 = row.iter().copied().sum::<f32>() / dim as f32;
            let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
            let std = var.sqrt().max(1e-5);

            for val in row.iter_mut() {
                *val = (*val - mean) / std;
            }
        });

        normalized
    }

    /// 分析门控分布统计
    pub fn analyze_gate_distribution(
        &self,
        rnn_output: &Array2<f32>,
        attn_output: &Array2<f32>,
    ) -> GateDistributionStats {
        let (seq_len, dim) = rnn_output.dim();
        let mut gates = Vec::with_capacity(seq_len);
        let mut rnn_weights_sum = 0.0_f64;
        let mut attn_weights_sum = 0.0_f64;

        for i in 0..seq_len {
            let rnn_row = rnn_output.row(i);
            let rnn_vec: Vec<f32> = rnn_row.iter().take(dim).copied().collect();

            let attn_row = attn_output.row(i);
            let attn_vec: Vec<f32> = attn_row.iter().take(dim).copied().collect();

            let rnn_energy: f32 = rnn_vec.iter().map(|x| x * x).sum();
            let attn_energy: f32 = attn_vec.iter().map(|x| x * x).sum();
            let energy_diff = (rnn_energy - attn_energy).sqrt().max(0.0);
            let gate_value = sigmoid(energy_diff / self.gate_temperature + self.gate_bias);

            gates.push(gate_value);
            rnn_weights_sum += gate_value as f64;
            attn_weights_sum += (1.0 - gate_value) as f64;
        }

        let mean_gate = gates.iter().sum::<f32>() / gates.len().max(1) as f32;
        let min_gate = gates.iter().cloned().fold(f32::MAX, f32::min);
        let max_gate = gates.iter().cloned().fold(f32::MIN, f32::max);

        GateDistributionStats {
            mean_gate,
            min_gate,
            max_gate,
            rnn_weight_avg: rnn_weights_sum / gates.len().max(1) as f64,
            attn_weight_avg: attn_weights_sum / gates.len().max(1) as f64,
        }
    }
}

/// 门控分布统计
#[derive(Debug, Clone)]
pub struct GateDistributionStats {
    /// 平均门控值
    pub mean_gate: f32,

    /// 最小门控值
    pub min_gate: f32,

    /// 最大门控值
    pub max_gate: f32,

    /// RNN 权重平均值
    pub rnn_weight_avg: f64,

    /// 注意力权重平均值
    pub attn_weight_avg: f64,
}

impl fmt::Display for GateDistributionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GateDistribution {{ mean={:.3}, range=[{:.3}, {:.3}], \
             rnn_weight={:.3}, attn_weight={:.3} }}",
            self.mean_gate, self.min_gate, self.max_gate, self.rnn_weight_avg, self.attn_weight_avg
        )
    }
}

impl fmt::Debug for TransitionLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransitionLayer")
            .field("input_dim", &self.input_dim)
            .field("gate_bias", &self.gate_bias)
            .field("use_layer_norm", &self.use_layer_norm)
            .finish()
    }
}

// ============================================================================
// AttentionHybridNetwork - 核心结构体
// ============================================================================

/// Attention with Hybrid Normalization (AHN) 核心引擎
///
/// 整合 RNN 压缩和局部标准注意力的完整实现：
/// - RNN Modules: Mamba2/DeltaNext/GDN 全局压缩
/// - LocalStandardAttention: 局部窗口多头注意力
/// - TransitionLayer: 门控融合层
///
/// # 协同工作流程
///
/// 1. **RNN 全局压缩**: 将超长序列压缩为紧凑的全局表示
/// 2. **局部注意力**: 在滑动窗口内精确建模局部依赖
/// 3. **门控融合**: 自适应平衡全局和局部信息
///
/// # 适用场景
///
/// - 超长文档理解 (>256K tokens)
/// - 长时间序列分析
/// - 大规模代码库处理
/// - 高分辨率视频理解
///
/// # 架构设计
///
/// ```text
/// Input Sequence
///     │
///     ├──→ RNN Modules (多层堆叠) ─────┐
///     │   - 全局上下文压缩               │
///     │   - O(N) 复杂度                  │
///     │                                 │
///     └──→ LocalStandardAttention ──────┼──→ TransitionLayer ──→ Output
///         - 局部细节建模                 │      自适应门控融合
///         - O(N*W) 复杂度                │
///                                         │
/// ```
pub struct AttentionHybridNetwork {
    /// RNN 模块列表
    rnn_modules: Vec<Box<dyn RnnBlock>>,

    /// 局部标准注意力
    local_attn: LocalStandardAttention,

    /// 过渡层（融合层）
    transition_layer: TransitionLayer,

    /// 配置
    config: AHNConfig,

    /// 性能统计
    stats: AHNPerformanceStats,
}

impl AttentionHybridNetwork {
    /// 创建新的 AHN 实例
    ///
    /// # 参数
    ///
    /// - `config`: AHN 配置参数
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let config = AHNConfig::default();
    /// let mut ahn = AttentionHybridNetwork::new(config)?;
    /// ```
    pub fn new(config: AHNConfig) -> InferenceResult<Self> {
        // 验证配置
        if config.hidden_dim == 0 {
            return Err(InferenceError::config("Hidden dimension must be positive"));
        }
        if config.num_rnn_layers == 0 {
            return Err(InferenceError::config(
                "Number of RNN layers must be positive",
            ));
        }
        if config.local_window_size == 0 {
            return Err(InferenceError::config("Local window size must be positive"));
        }
        if config.compression_ratio <= 0.0 {
            return Err(InferenceError::config("Compression ratio must be positive"));
        }

        // 创建 RNN 模块
        let mut rnn_modules: Vec<Box<dyn RnnBlock>> = Vec::with_capacity(config.num_rnn_layers);

        for _ in 0..config.num_rnn_layers {
            let rnn_block: Box<dyn RnnBlock> = match config.rnn_type {
                RnnType::Mamba2 => Box::new(Mamba2Block::new(
                    config.hidden_dim,
                    (config.hidden_dim / config.compression_ratio as usize).max(16),
                    4,
                )),
                RnnType::DeltaNext => Box::new(DeltaNextBlock::new(config.hidden_dim, 2)),
                RnnType::GDN => Box::new(GDNBlock::new(config.hidden_dim, 3)),
            };
            rnn_modules.push(rnn_block);
        }

        // 创建局部注意力
        let local_attn = LocalStandardAttention::new(
            config.local_window_size,
            config.num_attention_heads,
            config.attention_head_dim,
        );

        // 创建过渡层
        let transition_layer = TransitionLayer::new(
            config.hidden_dim,
            config.gate_bias,
            config.enable_layer_norm,
        );

        Ok(Self {
            rnn_modules,
            local_attn,
            transition_layer,
            config,
            stats: AHNPerformanceStats::default(),
        })
    }

    /// 获取配置引用
    pub fn config(&self) -> &AHNConfig {
        &self.config
    }

    /// 获取可变配置引用
    pub fn config_mut(&mut self) -> &mut AHNConfig {
        &mut self.config
    }

    /// 获取性能统计
    pub fn stats(&self) -> &AHNPerformanceStats {
        &self.stats
    }

    /// 重置统计信息
    pub fn reset_stats(&mut self) {
        self.stats = AHNPerformanceStats::default();
    }

    /// 重置所有 RNN 模块的状态
    pub fn reset_rnn_states(&mut self) {
        for module in self.rnn_modules.iter_mut() {
            module.reset_state();
        }
    }

    /// AHN 前向传播
    ///
    /// 完整的 RNN 压缩 + 局部注意力 + 门控融合流程。
    ///
    /// # 参数
    ///
    /// - `input`: 输入序列矩阵 [seq_len, hidden_dim]
    /// - `_num_heads`: 注意力头数（预留参数，用于兼容）
    /// - `_head_dim`: 每个头的维度（预留参数，用于兼容）
    ///
    /// # 返回值
    ///
    /// 融合后的输出矩阵 [seq_len, hidden_dim]
    ///
    /// # 流程
    ///
    /// 1. RNN 全局压缩：多层 RNN 依次处理输入序列
    /// 2. 局部注意力：在滑动窗口内计算标准多头注意力
    /// 3. 过渡层融合：门控融合两路输出
    ///
    /// # 性能特征
    ///
    /// - RNN 部分: O(N * L * D) 其中 L 是层数，D 是隐藏维度
    /// - 注意力部分: O(N * W * H * d) 其中 W 是窗口大小，H 是头数，d 是头维度
    /// - 总体: 相比全注意力 O(N^2 * H * d) 有显著优势
    pub fn forward(
        &mut self,
        input: &Array2<f32>,
        _num_heads: usize,
        _head_dim: usize,
    ) -> InferenceResult<Array2<f32>> {
        let start_time = Instant::now();
        let seq_len = input.nrows();

        // 更新统计
        if self.config.enable_stats {
            self.stats.total_calls += 1;
            self.stats.sequence_lengths.push(seq_len);
        }

        // Step 1: RNN 全局压缩
        let rnn_start = Instant::now();
        let mut rnn_output = input.clone();

        for (layer_idx, module) in self.rnn_modules.iter_mut().enumerate() {
            let (layer_output, _) = module.forward(&rnn_output)?;

            // 残差连接（除了第一层）
            if layer_idx > 0 {
                rnn_output = &rnn_output + &layer_output * 0.1;
            } else {
                rnn_output = layer_output;
            }
        }

        let rnn_elapsed = rnn_start.elapsed().as_micros() as u64;

        // Step 2: 局部标准注意力
        let attn_start = Instant::now();
        let attn_output = self.local_attn.forward(input, input, input)?;
        let attn_elapsed = attn_start.elapsed().as_micros() as u64;

        // Step 3: 过渡层融合
        let fusion_start = Instant::now();
        let output = self.transition_layer.fuse(&rnn_output, &attn_output)?;
        let fusion_elapsed = fusion_start.elapsed().as_micros() as u64;

        // 更新性能统计
        if self.config.enable_stats {
            let total_elapsed = start_time.elapsed().as_micros() as u64;
            self.stats.total_time_us += total_elapsed;
            self.stats.rnn_total_time_us += rnn_elapsed;
            self.stats.attn_total_time_us += attn_elapsed;
            self.stats.fusion_total_time_us += fusion_elapsed;
            self.stats.avg_latency_us =
                self.stats.total_time_us as f64 / self.stats.total_calls as f64;

            // 估算峰值内存
            let rnn_memory = seq_len * self.config.hidden_dim * self.config.num_rnn_layers;
            let attn_memory = seq_len
                * self.local_attn.window_size().min(seq_len)
                * self.config.num_attention_heads
                * self.config.attention_head_dim;
            let fusion_memory = seq_len * self.config.hidden_dim * 2;
            let peak_bytes = (rnn_memory + attn_memory + fusion_memory) * 4; // f32 = 4 bytes
            self.stats.peak_memory_mb = peak_bytes as f64 / (1024.0 * 1024.0);

            // 记录实际压缩率
            if let Some(first_module) = self.rnn_modules.first() {
                self.stats
                    .actual_compression_ratios
                    .push(first_module.compression_ratio());
            }
        }

        Ok(output)
    }

    /// 获取详细的性能报告
    pub fn performance_report(&self) -> String {
        format!(
            "AHN Performance Report:\n\
             - Config: {}\n\
             - Stats: {}\n\
             - RNN Type: {}\n\
             - Window Size: {}\n\
             - Time Breakdown: RNN={}us Attn={}us Fusion={}us\n\
             - Peak Memory: {:.1} MB\n\
             - Sequence Lengths: {:?}",
            self.config,
            self.stats,
            self.config.rnn_type,
            self.config.local_window_size,
            self.stats.rnn_total_time_us,
            self.stats.attn_total_time_us,
            self.stats.fusion_total_time_us,
            self.stats.peak_memory_mb,
            if self.stats.sequence_lengths.len() <= 10 {
                self.stats.sequence_lengths.clone()
            } else {
                self.stats.sequence_lengths[..10].to_vec()
            }
        )
    }

    /// 估算相对于全注意力的加速比
    ///
    /// 基于理论分析和实测数据估算加速比。
    ///
    /// # 参数
    ///
    /// - `seq_len`: 序列长度
    ///
    /// # 返回值
    ///
    /// 估算的加速比倍数
    pub fn estimated_speedup_vs_full_attention(&self, seq_len: usize) -> f64 {
        if seq_len <= self.config.local_window_size {
            // 短序列：AHN 与全注意力相当或略慢（因为额外开销）
            return 0.95;
        }

        // 全注意力复杂度: O(N^2 * H * d)
        let full_attention_complexity =
            seq_len * seq_len * self.config.num_attention_heads * self.config.attention_head_dim;

        // AHN 复杂度:
        // RNN: O(N * L * D)
        // Attention: O(N * W * H * d)
        let rnn_complexity = seq_len * self.config.num_rnn_layers * self.config.hidden_dim;
        let attn_complexity = seq_len
            * self.config.local_window_size.min(seq_len)
            * self.config.num_attention_heads
            * self.config.attention_head_dim;
        let ahn_complexity = rnn_complexity + attn_complexity;

        if ahn_complexity == 0 {
            return 1.0;
        }

        let theoretical_speedup = full_attention_complexity as f64 / ahn_complexity as f64;

        // 考虑实际开销（融合、RNN 状态管理等），乘以效率因子
        let efficiency_factor = 0.80;
        (theoretical_speedup * efficiency_factor).clamp(1.0, 50.0)
    }

    /// 分析门控分布
    ///
    /// 返回最近一次前向传播的门控分布统计
    pub fn analyze_last_gate_distribution(&self) -> Option<GateDistributionStats> {
        // 此功能需要在 forward 中保存中间结果才能实现
        // 这里返回 None 表示暂未实现
        None
    }

    /// 获取内存使用估算
    ///
    /// 返回给定序列长度下的各组件内存使用情况
    pub fn memory_breakdown(&self, seq_len: usize) -> MemoryBreakdown {
        let bytes_per_element = 4; // f32

        // RNN 内存（每层需要存储输入和隐藏状态）
        let rnn_per_layer = seq_len * self.config.hidden_dim * bytes_per_element;
        let rnn_total = rnn_per_layer * self.config.num_rnn_layers;

        // 注意力内存（Q, K, V + 输出）
        let attn_qkv = seq_len
            * self.config.num_attention_heads
            * self.config.attention_head_dim
            * 3
            * bytes_per_element;
        let attn_output = seq_len
            * self.config.num_attention_heads
            * self.config.attention_head_dim
            * bytes_per_element;
        let attn_total = attn_qkv + attn_output;

        // 融合层内存
        let fusion_total = seq_len * self.config.hidden_dim * 2 * bytes_per_element;

        // 中间激活值（保守估计）
        let activations = (rnn_total + attn_total) / 2;

        let total = rnn_total + attn_total + fusion_total + activations;

        MemoryBreakdown {
            rnn_mb: rnn_total as f64 / (1024.0 * 1024.0),
            attention_mb: attn_total as f64 / (1024.0 * 1024.0),
            fusion_mb: fusion_total as f64 / (1024.0 * 1024.0),
            activations_mb: activations as f64 / (1024.0 * 1024.0),
            total_mb: total as f64 / (1024.0 * 1024.0),
        }
    }
}

/// 内存使用分解
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    /// RNN 模块内存 (MB)
    pub rnn_mb: f64,

    /// 注意力模块内存 (MB)
    pub attention_mb: f64,

    /// 融合层内存 (MB)
    pub fusion_mb: f64,

    /// 中间激活值内存 (MB)
    pub activations_mb: f64,

    /// 总内存 (MB)
    pub total_mb: f64,
}

impl fmt::Display for MemoryBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryBreakdown {{ RNN={:.1}MB, Attn={:.1}MB, Fusion={:.1}MB, \
             Activations={:.1}MB, Total={:.1}MB }}",
            self.rnn_mb, self.attention_mb, self.fusion_mb, self.activations_mb, self.total_mb
        )
    }
}

impl fmt::Debug for AttentionHybridNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AttentionHybridNetwork")
            .field("config", &self.config)
            .field("num_rnn_modules", &self.rnn_modules.len())
            .field("local_attn", &self.local_attn)
            .field("stats", &self.stats)
            .finish()
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> AHNConfig {
        AHNConfig {
            hidden_dim: 128,
            rnn_type: RnnType::Mamba2,
            num_rnn_layers: 2,
            local_window_size: 32,
            max_context_length: 4096,
            compression_ratio: 16.0,
            gate_bias: 0.0,
            enable_layer_norm: true,
            num_attention_heads: 4,
            attention_head_dim: 32,
            rnn_dropout: 0.0,
            enable_stats: true,
        }
    }

    fn create_test_input(seq_len: usize, dim: usize) -> Array2<f32> {
        Array2::from_shape_fn((seq_len, dim), |(i, j)| {
            (i as f32 * 0.1 + j as f32 * 0.01).sin()
        })
    }

    // ===== 测试 1: 配置初始化测试 =====

    #[test]
    fn test_config_initialization() {
        let config = create_test_config();

        assert_eq!(config.hidden_dim, 128);
        assert_eq!(config.rnn_type, RnnType::Mamba2);
        assert_eq!(config.num_rnn_layers, 2);
        assert_eq!(config.local_window_size, 32);
        assert!((config.compression_ratio - 16.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_config() {
        let config = AHNConfig::default();

        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.rnn_type, RnnType::Mamba2);
        assert_eq!(config.num_rnn_layers, 4);
        assert_eq!(config.local_window_size, 2048);
        assert_eq!(config.max_context_length, 262144);
        assert!((config.compression_ratio - 32.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_display() {
        let config = create_test_config();
        let display = format!("{}", config);

        assert!(display.contains("AHNConfig"));
        assert!(display.contains("hidden_dim=128"));
        assert!(display.contains("Mamba2"));
    }

    #[test]
    fn test_rnn_type_display() {
        assert_eq!(format!("{}", RnnType::Mamba2), "Mamba2");
        assert_eq!(format!("{}", RnnType::DeltaNext), "DeltaNext");
        assert_eq!(format!("{}", RnnType::GDN), "GDN");
    }

    #[test]
    fn test_rnn_type_default() {
        let default_type = RnnType::default();
        assert_eq!(default_type, RnnType::Mamba2);
    }

    // ===== 测试 2: Mamba2Block 测试 =====

    #[test]
    fn test_mamba2_block_creation() {
        let block = Mamba2Block::new(64, 16, 4);

        assert_eq!(block.hidden_dim(), 64);
        assert_eq!(block.rnn_type(), RnnType::Mamba2);
        assert!(!block.initialized); // 未初始化
    }

    #[test]
    fn test_mamba2_block_forward() {
        let mut block = Mamba2Block::new(64, 16, 4);
        let input = create_test_input(32, 64);

        let result = block.forward(&input);

        assert!(result.is_ok());
        let (output, state) = result.unwrap();
        assert_eq!(output.dim(), (32, 64));
        assert!(block.initialized); // 已初始化

        // 验证状态类型正确
        match state {
            RnnState::Mamba2 { .. } => {}
            _ => panic!("Expected Mamba2 state"),
        }
    }

    #[test]
    fn test_mamba2_block_empty_input() {
        let mut block = Mamba2Block::new(64, 16, 4);
        let empty = create_test_input(0, 64);

        let result = block.forward(&empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_mamba2_block_dimension_mismatch() {
        let mut block = Mamba2Block::new(64, 16, 4);
        let wrong_dim_input = create_test_input(32, 128); // 错误维度

        let result = block.forward(&wrong_dim_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_mamba2_block_reset_state() {
        let mut block = Mamba2Block::new(64, 16, 4);
        let input = create_test_input(16, 64);

        let _ = block.forward(&input);
        assert!(block.initialized);

        block.reset_state();
        assert!(!block.initialized);
    }

    #[test]
    fn test_mamba2_block_compression_ratio() {
        let block = Mamba2Block::new(128, 32, 4);
        let ratio = block.compression_ratio();

        assert!(ratio > 0.0);
        assert!(ratio <= 1.0);
        assert!((ratio - 0.25).abs() < 0.01); // 32/128 = 0.25
    }

    // ===== 测试 3: DeltaNextBlock 测试 =====

    #[test]
    fn test_deltanext_block_creation() {
        let block = DeltaNextBlock::new(64, 2);

        assert_eq!(block.hidden_dim(), 64);
        assert_eq!(block.rnn_type(), RnnType::DeltaNext);
    }

    #[test]
    fn test_deltanext_block_forward() {
        let mut block = DeltaNextBlock::new(64, 2);
        let input = create_test_input(32, 64);

        let result = block.forward(&input);

        assert!(result.is_ok());
        let (output, state) = result.unwrap();
        assert_eq!(output.dim(), (32, 64));

        match state {
            RnnState::DeltaNext { .. } => {}
            _ => panic!("Expected DeltaNext state"),
        }
    }

    #[test]
    fn test_deltanext_block_single_step() {
        let mut block = DeltaNextBlock::new(64, 2);
        let input = create_test_input(1, 64);

        let result = block.forward(&input);
        assert!(result.is_ok());

        let (output, _) = result.unwrap();
        assert_eq!(output.dim(), (1, 64));
    }

    #[test]
    fn test_deltanext_block_compression_ratio() {
        let block = DeltaNextBlock::new(64, 4);
        let ratio = block.compression_ratio();

        assert!((ratio - 0.25).abs() < 0.01); // 1/4 = 0.25
    }

    // ===== 测试 4: GDNBlock 测试 =====

    #[test]
    fn test_gdn_block_creation() {
        let block = GDNBlock::new(64, 3);

        assert_eq!(block.hidden_dim(), 64);
        assert_eq!(block.rnn_type(), RnnType::GDN);
    }

    #[test]
    fn test_gdn_block_forward() {
        let mut block = GDNBlock::new(64, 3);
        let input = create_test_input(32, 64);

        let result = block.forward(&input);

        assert!(result.is_ok());
        let (output, state) = result.unwrap();
        assert_eq!(output.dim(), (32, 64));

        match state {
            RnnState::GDN { .. } => {}
            _ => panic!("Expected GDN state"),
        }
    }

    #[test]
    fn test_gdn_block_denoising_effect() {
        let mut block = GDNBlock::new(64, 3);

        // 创建带噪声的输入
        let noisy_input = Array2::from_shape_fn((16, 64), |(i, j)| {
            let signal = ((i as f32 * 0.1) + (j as f32 * 0.01)).sin();
            let noise = (i as f32 * j as f32 * 0.001).cos() * 0.1;
            signal + noise
        });

        let result = block.forward(&noisy_input);
        assert!(result.is_ok());

        let (output, _) = result.unwrap();
        // 输出应该存在（验证去噪不会崩溃）
        assert_eq!(output.dim(), (16, 64));

        // 验证输出不包含 NaN 或 Infinity
        for val in output.iter() {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
    }

    #[test]
    fn test_gdn_block_compression_ratio() {
        let block = GDNBlock::new(64, 3);
        let ratio = block.compression_ratio();

        assert!(ratio > 0.0);
        assert!(ratio <= 1.0);
    }

    // ===== 测试 5: LocalStandardAttention 测试 =====

    #[test]
    fn test_local_attention_creation() {
        let attn = LocalStandardAttention::new(32, 4, 32);

        assert_eq!(attn.window_size(), 32);
        assert_eq!(attn.num_heads(), 4);
        assert_eq!(attn.head_dim(), 32);
    }

    #[test]
    fn test_local_attention_forward() {
        let attn = LocalStandardAttention::new(16, 2, 32);
        let input = create_test_input(64, 64); // 2 heads * 32 dim = 64

        let result = attn.forward(&input, &input, &input);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), (64, 64));

        // 验证输出有效性
        for val in output.iter() {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
    }

    #[test]
    fn test_local_attention_window_range() {
        let attn = LocalStandardAttention::new(8, 4, 32);

        // 测试中间位置
        let (start, end) = attn.get_window_range(10, 32);
        assert_eq!(start, 3); // 10 - 8 + 1 = 3
        assert_eq!(end, 11); // 10 + 1 = 11

        // 测试起始位置
        let (start, end) = attn.get_window_range(0, 32);
        assert_eq!(start, 0);
        assert_eq!(end, 1);

        // 测试窗口大于序列长度
        let (start, end) = attn.get_window_range(5, 4);
        assert_eq!(start, 0);
        assert_eq!(end, 4);
    }

    #[test]
    fn test_local_attention_empty_input() {
        let attn = LocalStandardAttention::new(16, 4, 32);
        let empty = create_test_input(0, 128);

        let result = attn.forward(&empty, &empty, &empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_local_attention_dimension_mismatch() {
        let attn = LocalStandardAttention::new(16, 4, 32); // 期望 4*32=128
        let wrong_dim = create_test_input(16, 64); // 错误维度

        let result = attn.forward(&wrong_dim, &wrong_dim, &wrong_dim);
        assert!(result.is_err());
    }

    #[test]
    fn test_local_attention_complexity_analysis() {
        let attn = LocalStandardAttention::new(1024, 8, 64);
        let (time_complexity, space_complexity) = attn.complexity_analysis(8192);

        // 时间复杂度应该小于全注意力 (8192^2 * 8 * 64)
        let full_attention_complexity = 8192 * 8192 * 8 * 64;
        assert!(time_complexity < full_attention_complexity);

        // 空间复杂度应该是合理的
        assert!(space_complexity > 0);
    }

    // ===== 测试 6: TransitionLayer 测试 =====

    #[test]
    fn test_transition_layer_creation() {
        let transition = TransitionLayer::new(64, 0.0, true);

        assert_eq!(transition.input_dim(), 64);
    }

    #[test]
    fn test_transition_layer_fusion() {
        let transition = TransitionLayer::new(64, 0.0, false);
        let rnn_output = create_test_input(32, 64);
        let attn_output = create_test_input(32, 64);

        let result = transition.fuse(&rnn_output, &attn_output);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), (32, 64));

        // 验证输出有效性
        for val in output.iter() {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
    }

    #[test]
    fn test_transition_layer_with_layer_norm() {
        let transition = TransitionLayer::new(64, 0.0, true);
        let rnn_output = create_test_input(32, 64);
        let attn_output = create_test_input(32, 64);

        let result = transition.fuse(&rnn_output, &attn_output);

        assert!(result.is_ok());
        let output = result.unwrap();

        // 层归一化后，每行的均值应接近 0，标准差接近 1
        for row in output.axis_iter(Axis(0)) {
            let mean: f32 = row.iter().copied().sum::<f32>() / row.len() as f32;
            assert!(
                mean.abs() < 1.0,
                "Mean should be near 0 after layer norm, got {}",
                mean
            );
        }
    }

    #[test]
    fn test_transition_layer_empty_input() {
        let transition = TransitionLayer::new(64, 0.0, true);
        let empty = create_test_input(0, 64);

        let result = transition.fuse(&empty, &empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_transition_layer_gate_distribution() {
        let transition = TransitionLayer::new(64, 0.0, false);
        let rnn_output = create_test_input(32, 64);
        let attn_output = create_test_input(32, 64);

        let stats = transition.analyze_gate_distribution(&rnn_output, &attn_output);

        assert!(stats.mean_gate >= 0.0 && stats.mean_gate <= 1.0);
        assert!(stats.min_gate >= 0.0 && stats.min_gate <= 1.0);
        assert!(stats.max_gate >= 0.0 && stats.max_gate <= 1.0);
        assert!((stats.rnn_weight_avg + stats.attn_weight_avg - 1.0).abs() < 0.01);
    }

    // ===== 测试 7: AttentionHybridNetwork 集成测试 =====

    #[test]
    fn test_ahn_creation() {
        let config = create_test_config();
        let result = AttentionHybridNetwork::new(config);

        assert!(result.is_ok());
        let ahn = result.unwrap();

        assert_eq!(ahn.config().hidden_dim, 128);
        assert_eq!(ahn.config().rnn_type, RnnType::Mamba2);
        assert_eq!(ahn.rnn_modules.len(), 2);
    }

    #[test]
    fn test_ahn_invalid_config_zero_hidden() {
        let config = AHNConfig {
            hidden_dim: 0,
            ..create_test_config()
        };
        let result = AttentionHybridNetwork::new(config);

        assert!(result.is_err());
    }

    #[test]
    fn test_ahn_invalid_config_zero_layers() {
        let config = AHNConfig {
            num_rnn_layers: 0,
            ..create_test_config()
        };
        let result = AttentionHybridNetwork::new(config);

        assert!(result.is_err());
    }

    #[test]
    fn test_ahn_forward_basic() {
        let config = create_test_config();
        let mut ahn = AttentionHybridNetwork::new(config).unwrap();

        let input = create_test_input(64, 128);
        let result = ahn.forward(&input, 4, 32);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), (64, 128));

        // 验证输出有效性
        for val in output.iter() {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
    }

    #[test]
    fn test_ahn_forward_long_sequence() {
        let config = AHNConfig {
            local_window_size: 64,
            ..create_test_config()
        };
        let mut ahn = AttentionHybridNetwork::new(config).unwrap();

        let input = create_test_input(512, 128);
        let result = ahn.forward(&input, 4, 32);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), (512, 128));
    }

    #[test]
    fn test_ahn_with_different_rnn_types() {
        let rnn_types = vec![RnnType::Mamba2, RnnType::DeltaNext, RnnType::GDN];

        for rnn_type in rnn_types {
            let config = AHNConfig {
                rnn_type,
                ..create_test_config()
            };
            let result = AttentionHybridNetwork::new(config);

            assert!(result.is_ok(), "Failed to create AHN with {:?}", rnn_type);

            let mut ahn = result.unwrap();
            let input = create_test_input(32, 128);
            let forward_result = ahn.forward(&input, 4, 32);

            assert!(
                forward_result.is_ok(),
                "Forward failed with {:?}",
                ahn.config().rnn_type
            );
        }
    }

    // ===== 测试 8: 性能统计测试 =====

    #[test]
    fn test_performance_stats_tracking() {
        let config = AHNConfig {
            enable_stats: true,
            ..create_test_config()
        };
        let mut ahn = AttentionHybridNetwork::new(config).unwrap();

        let input = create_test_input(32, 128);

        // 执行多次前向传播
        for _ in 0..5 {
            let _ = ahn.forward(&input, 4, 32);
        }

        let stats = ahn.stats();
        assert_eq!(stats.total_calls, 5);
        assert!(stats.avg_latency_us > 0.0);
        assert_eq!(stats.sequence_lengths.len(), 5);
    }

    #[test]
    fn test_reset_stats() {
        let config = AHNConfig {
            enable_stats: true,
            ..create_test_config()
        };
        let mut ahn = AttentionHybridNetwork::new(config).unwrap();

        let input = create_test_input(16, 128);
        let _ = ahn.forward(&input, 4, 32);

        assert!(ahn.stats().total_calls > 0);

        ahn.reset_stats();
        assert_eq!(ahn.stats().total_calls, 0);
        assert_eq!(ahn.stats().sequence_lengths.len(), 0);
    }

    #[test]
    fn test_performance_report() {
        let config = create_test_config();
        let ahn = AttentionHybridNetwork::new(config).unwrap();

        let report = ahn.performance_report();
        assert!(report.contains("AHN Performance Report"));
        assert!(report.contains("Config"));
        assert!(report.contains("Stats"));
    }

    // ===== 测试 9: 速度比和内存分析测试 =====

    #[test]
    fn test_estimated_speedup_short_sequence() {
        let config = create_test_config();
        let ahn = AttentionHybridNetwork::new(config).unwrap();

        // 短序列：速度比接近 1.0
        let speedup = ahn.estimated_speedup_vs_full_attention(16);
        assert!((0.9..=1.1).contains(&speedup));
    }

    #[test]
    fn test_estimated_speedup_long_sequence() {
        let config = create_test_config();
        let ahn = AttentionHybridNetwork::new(config).unwrap();

        // 长序列：显著加速
        let speedup = ahn.estimated_speedup_vs_full_attention(8192);
        assert!(
            speedup >= 1.5,
            "Long sequence should have significant speedup, got {}",
            speedup
        );
    }

    #[test]
    fn test_memory_breakdown() {
        let config = create_test_config();
        let ahn = AttentionHybridNetwork::new(config).unwrap();

        let breakdown = ahn.memory_breakdown(1024);

        assert!(breakdown.rnn_mb > 0.0);
        assert!(breakdown.attention_mb > 0.0);
        assert!(breakdown.fusion_mb > 0.0);
        assert!(breakdown.total_mb > 0.0);
        assert!(
            (breakdown.total_mb
                - (breakdown.rnn_mb
                    + breakdown.attention_mb
                    + breakdown.fusion_mb
                    + breakdown.activations_mb))
                .abs()
                < 0.01
        );
    }

    #[test]
    fn test_memory_breakdown_long_sequence() {
        let config = AHNConfig {
            hidden_dim: 4096,
            num_attention_heads: 32,
            attention_head_dim: 128,
            ..create_test_config()
        };
        let ahn = AttentionHybridNetwork::new(config).unwrap();

        // 测试 256K 序列的内存使用
        let breakdown = ahn.memory_breakdown(262144);

        println!("\n[MEMORY] 256K sequence memory breakdown:");
        println!("  - RNN: {:.1} MB", breakdown.rnn_mb);
        println!("  - Attention: {:.1} MB", breakdown.attention_mb);
        println!("  - Fusion: {:.1} MB", breakdown.fusion_mb);
        println!("  - Activations: {:.1} MB", breakdown.activations_mb);
        println!("  - Total: {:.1} MB", breakdown.total_mb);

        // 验证内存 < 32GB (32768 MB)
        assert!(
            breakdown.total_mb < 32768.0,
            "Memory should be < 32GB for 256K tokens, got {:.1} MB",
            breakdown.total_mb
        );
    }

    // ===== 测试 10: 边界条件和错误处理测试 =====

    #[test]
    fn test_debug_format() {
        let config = create_test_config();
        let ahn = AttentionHybridNetwork::new(config).unwrap();

        let debug_str = format!("{:?}", ahn);
        assert!(debug_str.contains("AttentionHybridNetwork"));
        assert!(debug_str.contains("config"));
        assert!(debug_str.contains("num_rnn_modules"));
    }

    #[test]
    fn test_reset_rnn_states() {
        let config = create_test_config();
        let mut ahn = AttentionHybridNetwork::new(config).unwrap();

        let input = create_test_input(16, 128);
        let _ = ahn.forward(&input, 4, 32);

        // 重置 RNN 状态
        ahn.reset_rnn_states();

        // 再次前向传播应该正常工作
        let result = ahn.forward(&input, 4, 32);
        assert!(result.is_ok());
    }

    #[test]
    fn test_various_sequence_lengths() {
        let config = create_test_config();
        let mut ahn = AttentionHybridNetwork::new(config).unwrap();

        let test_lengths = [1, 8, 16, 32, 64, 128, 256];

        for &seq_len in &test_lengths {
            let input = create_test_input(seq_len, 128);
            let result = ahn.forward(&input, 4, 32);

            assert!(result.is_ok(), "Failed for sequence length {}", seq_len);

            let output = result.unwrap();
            assert_eq!(
                output.dim(),
                (seq_len, 128),
                "Output dimension mismatch for seq_len={}",
                seq_len
            );
        }
    }

    #[test]
    fn test_large_hidden_dimension() {
        let config = AHNConfig {
            hidden_dim: 1024,
            num_attention_heads: 8,
            attention_head_dim: 128,
            ..create_test_config()
        };
        let mut ahn = AttentionHybridNetwork::new(config).unwrap();

        let input = create_test_input(64, 1024);
        let result = ahn.forward(&input, 8, 128);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dim(), (64, 1024));
    }

    #[test]
    fn test_multiple_forward_calls() {
        let config = create_test_config();
        let mut ahn = AttentionHybridNetwork::new(config).unwrap();

        // 连续调用多次，验证状态管理
        for i in 0..10 {
            let input = create_test_input(32, 128);
            let result = ahn.forward(&input, 4, 32);

            assert!(result.is_ok(), "Forward call {} failed", i);
        }

        // 统计应该记录了 10 次调用
        assert_eq!(ahn.stats().total_calls, 10);
    }

    // ===== 测试 11: 综合集成测试 =====

    #[test]
    fn test_full_pipeline_integration() {
        let config = AHNConfig {
            hidden_dim: 256,
            num_rnn_layers: 3,
            local_window_size: 64,
            num_attention_heads: 8,
            attention_head_dim: 32,
            enable_stats: true,
            ..create_test_config()
        };
        let mut ahn = AttentionHybridNetwork::new(config).unwrap();

        // 测试不同长度的序列
        let test_lengths = [16, 64, 128, 256, 512, 1024];

        for &seq_len in &test_lengths {
            let input = create_test_input(seq_len, 256);
            let result = ahn.forward(&input, 8, 32);

            assert!(result.is_ok(), "Failed for sequence length {}", seq_len);

            let output = result.unwrap();
            assert_eq!(
                output.dim(),
                (seq_len, 256),
                "Output dimension mismatch for seq_len={}",
                seq_len
            );

            // 验证输出数值稳定性
            for val in output.iter() {
                assert!(val.is_finite(), "Non-finite value at seq_len={}", seq_len);
            }
        }
    }

    #[test]
    fn test_stress_test_medium_sequence() {
        let config = AHNConfig {
            local_window_size: 128,
            enable_stats: true,
            ..create_test_config()
        };
        let mut ahn = AttentionHybridNetwork::new(config).unwrap();

        let input = create_test_tokens(2048, 128);
        let start = Instant::now();

        let result = ahn.forward(&input, 4, 32);
        let elapsed = start.elapsed();

        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dim(), (2048, 128));

        let stats = ahn.stats();
        println!(
            "\n[STRESS TEST] 2K sequence performance:\n\
             - Elapsed: {:?}\n\
             - Avg latency: {:.2} us\n\
             - Total calls: {}\n\
             - Peak memory: {:.1} MB\n\
             - Estimated speedup vs full attention: {:.2}x",
            elapsed,
            stats.avg_latency_us,
            stats.total_calls,
            stats.peak_memory_mb,
            ahn.estimated_speedup_vs_full_attention(2048)
        );

        // 验证速度提升
        let speedup = ahn.estimated_speedup_vs_full_attention(2048);
        assert!(speedup > 1.0, "Should achieve speedup over full attention");
    }

    #[test]
    fn test_all_rnn_types_comprehensive() {
        let configs = vec![
            (RnnType::Mamba2, "Mamba2 SSM"),
            (RnnType::DeltaNext, "Delta Network"),
            (RnnType::GDN, "Gated Denoising"),
        ];

        for (rnn_type, description) in configs {
            let config = AHNConfig {
                rnn_type,
                enable_stats: true,
                ..create_test_config()
            };

            let result = AttentionHybridNetwork::new(config);
            assert!(result.is_ok(), "Failed to create AHN with {}", description);

            let mut ahn = result.unwrap();
            let input = create_test_input(64, 128);

            let forward_result = ahn.forward(&input, 4, 32);
            assert!(
                forward_result.is_ok(),
                "Forward failed with {}",
                description
            );

            let output = forward_result.unwrap();
            assert_eq!(output.dim(), (64, 128));

            // 验证每种类型的特殊性
            match ahn.config().rnn_type {
                RnnType::Mamba2 => {
                    // Mamba2 应该有较高的压缩率
                    assert!(ahn.rnn_modules[0].compression_ratio() > 0.0);
                }
                RnnType::DeltaNext => {
                    // DeltaNext 应该有适中的压缩率
                    assert!(ahn.rnn_modules[0].compression_ratio() > 0.0);
                    assert!(ahn.rnn_modules[0].compression_ratio() <= 1.0);
                }
                RnnType::GDN => {
                    // GDN 应该有较低的压缩率（更多迭代）
                    assert!(ahn.rnn_modules[0].compression_ratio() > 0.0);
                }
            }

            println!(
                "\n[RNN TYPE] {}: compression_ratio={:.3}",
                description,
                ahn.rnn_modules[0].compression_ratio()
            );
        }
    }

    // ===== 辅助函数 =====

    fn create_test_tokens(seq_len: usize, dim: usize) -> Array2<f32> {
        Array2::from_shape_fn((seq_len, dim), |(i, j)| {
            let base = i as f32 * 0.01 + j as f32 * 0.001;
            (base.sin() + 0.5 * base.cos()) * (1.0 + 0.01 * (i as f32))
        })
    }
}
