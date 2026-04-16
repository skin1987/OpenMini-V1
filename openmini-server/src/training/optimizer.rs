//! 优化器与学习率调度器
//!
//! 实现 AdamW、SGD with Momentum 优化器和多种学习率调度策略，
//! 支持模型训练的参数更新、状态保存与恢复。

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// 优化器错误类型
#[derive(Debug)]
pub enum OptimizerError {
    GradientMismatch { expected: usize, actual: usize },
    BufferNotInitialized,
    InvalidLearningRate(f64),
    Serialization(String),
    Other(String),
}

impl std::fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerError::GradientMismatch { expected, actual } => {
                write!(f, "梯度数量不匹配: 期望 {}, 实际 {}", expected, actual)
            }
            OptimizerError::BufferNotInitialized => write!(f, "优化器缓冲区未初始化"),
            OptimizerError::InvalidLearningRate(lr) => write!(f, "无效的学习率: {}", lr),
            OptimizerError::Serialization(msg) => write!(f, "序列化错误: {}", msg),
            OptimizerError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for OptimizerError {}

/// 参数状态（包含梯度存储）
#[derive(Debug, Clone)]
pub struct ParamState {
    pub data: ArrayD<f32>,
    pub grad: Option<ArrayD<f32>>,
}

/// 优化器 trait
pub trait Optimizer {
    /// 执行一步参数更新
    fn step(
        &mut self,
        params: &mut [ParamState],
        gradients: &[ArrayD<f32>],
    ) -> Result<f64, OptimizerError>;

    /// 清零所有梯度
    fn zero_grad(&self, params: &mut [ParamState]);

    /// 获取当前学习率
    fn learning_rate(&self) -> f64;

    /// 获取优化器状态（用于 checkpoint）
    fn state_dict(&self) -> OptimizerState;

    /// 加载优化器状态（用于恢复）
    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<(), OptimizerError>;

    /// 返回 self 作为 Any，用于 downcast
    ///
    /// 这允许在运行时检查优化器的具体类型，
    /// 例如：`optimizer.as_any().downcast_ref::<AdamW>()`
    fn as_any(&self) -> &dyn std::any::Any;

    /// 返回 mutable self 作为 Any，用于 downcast
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// AdamW 优化器
pub struct AdamW {
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    pub steps: u64,
    pub exp_avg: Vec<ArrayD<f64>>,
    pub exp_avg_sq: Vec<ArrayD<f64>>,
}

impl AdamW {
    pub fn new(lr: f64, betas: (f64, f64), eps: f64, weight_decay: f64) -> Self {
        assert!(lr > 0.0, "学习率必须为正数");
        assert!(eps > 0.0, "eps 必须为正数");

        Self {
            lr,
            betas,
            eps,
            weight_decay,
            steps: 0,
            exp_avg: Vec::new(),
            exp_avg_sq: Vec::new(),
        }
    }

    /// 初始化参数缓冲区
    pub fn init_buffers(&mut self, params: &[ParamState]) {
        self.exp_avg.clear();
        self.exp_avg_sq.clear();

        for param in params {
            let shape = param.data.shape().to_vec();
            self.exp_avg.push(ArrayD::<f64>::zeros(shape.clone()));
            self.exp_avg_sq.push(ArrayD::<f64>::zeros(shape));
        }
    }

    fn init_buffer_for_param(&mut self, param: &ParamState) {
        let shape = param.data.shape().to_vec();
        self.exp_avg.push(ArrayD::<f64>::zeros(shape.clone()));
        self.exp_avg_sq.push(ArrayD::<f64>::zeros(shape));
    }

    fn to_f64(arr: &ArrayD<f32>) -> ArrayD<f64> {
        arr.mapv(|x| x as f64)
    }

    fn to_f32(arr: &ArrayD<f64>) -> ArrayD<f32> {
        arr.mapv(|x| x as f32)
    }
}

impl Optimizer for AdamW {
    fn step(
        &mut self,
        params: &mut [ParamState],
        gradients: &[ArrayD<f32>],
    ) -> Result<f64, OptimizerError> {
        if params.len() != gradients.len() {
            return Err(OptimizerError::GradientMismatch {
                expected: params.len(),
                actual: gradients.len(),
            });
        }

        self.steps += 1;
        let mut grad_norm = 0.0f64;

        for (i, (param, grad)) in params.iter_mut().zip(gradients.iter()).enumerate() {
            if i >= self.exp_avg.len() {
                self.init_buffer_for_param(param);
            }

            // 将 f32 梯度转换为 f64 以进行计算
            let g_f64: ndarray::ArrayD<f64> = grad.mapv(|x| x as f64);

            // 更新一阶矩 m_t = β₁ * m_{t-1} + (1 - β₁) * g
            self.exp_avg[i] = &self.exp_avg[i] * self.betas.0 + &g_f64 * (1.0 - self.betas.0);

            // 更新二阶矩 v_t = β₂ * v_{t-1} + (1 - β₂) * g²
            self.exp_avg_sq[i] =
                &self.exp_avg_sq[i] * self.betas.1 + &g_f64.mapv(|x| x * x) * (1.0 - self.betas.1);

            // Bias correction
            let bias_correction1 = 1.0 - self.betas.0.powi(self.steps as i32);
            let bias_correction2 = 1.0 - self.betas.1.powi(self.steps as i32);

            let m_hat = &self.exp_avg[i] / bias_correction1;
            let v_hat = &self.exp_avg_sq[i] / bias_correction2;

            // 更新参数: θ = θ - lr * (m̂ / (√v̂ + ε) + λ * θ)
            let param_f64 = Self::to_f64(&param.data);
            let denom = v_hat.mapv(|x| x.sqrt() + self.eps);
            let update = &m_hat / &denom + &param_f64 * self.weight_decay;
            let new_param = &param_f64 - &update * self.lr;
            param.data = Self::to_f32(&new_param);

            // 计算梯度范数
            grad_norm += (&g_f64 * &g_f64).sum();
        }

        Ok(grad_norm.sqrt())
    }

    fn zero_grad(&self, params: &mut [ParamState]) {
        for param in params {
            param.grad = None;
        }
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn state_dict(&self) -> OptimizerState {
        let param_states: Vec<ParamOptimizerState> = (0..self.exp_avg.len())
            .map(|i| ParamOptimizerState {
                exp_avg: Some(self.exp_avg[i].iter().map(|&x| x as f32).collect()),
                exp_avg_sq: Some(self.exp_avg_sq[i].iter().map(|&x| x as f32).collect()),
                momentum_buf: None,
            })
            .collect();

        OptimizerState {
            optimizer_type: "AdamW".to_string(),
            steps: self.steps,
            param_states,
        }
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<(), OptimizerError> {
        if state.optimizer_type != "AdamW" {
            return Err(OptimizerError::Other(format!(
                "优化器类型不匹配: 期望 AdamW, 实际 {}",
                state.optimizer_type
            )));
        }

        self.steps = state.steps;
        self.exp_avg.clear();
        self.exp_avg_sq.clear();

        for ps in &state.param_states {
            let avg = ps
                .exp_avg
                .as_ref()
                .ok_or(OptimizerError::BufferNotInitialized)?;
            let avg_sq = ps
                .exp_avg_sq
                .as_ref()
                .ok_or(OptimizerError::BufferNotInitialized)?;

            self.exp_avg.push(
                ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[avg.len()]),
                    avg.iter().map(|&x| x as f64).collect(),
                )
                .expect("Invalid shape for exp_avg"),
            );
            self.exp_avg_sq.push(
                ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[avg_sq.len()]),
                    avg_sq.iter().map(|&x| x as f64).collect(),
                )
                .expect("Invalid shape for exp_avg_sq"),
            );
        }

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// SGD with Momentum 优化器
pub struct SGD {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub dampening: f64,
    pub nesterov: bool,
    pub steps: u64,
    pub buf: Vec<ArrayD<f64>>,
}

impl SGD {
    pub fn new(lr: f64, momentum: f64, weight_decay: f64, nesterov: bool) -> Self {
        assert!(lr > 0.0, "学习率必须为正数");

        Self {
            lr,
            momentum,
            weight_decay,
            dampening: if momentum > 0.0 { 0.0 } else { 1.0 },
            nesterov,
            steps: 0,
            buf: Vec::new(),
        }
    }

    /// 初始化动量缓冲区
    pub fn init_buffers(&mut self, params: &[ParamState]) {
        self.buf.clear();
        for param in params {
            let shape = param.data.shape().to_vec();
            self.buf.push(ArrayD::<f64>::zeros(shape));
        }
    }

    fn init_buffer_for_param(&mut self, param: &ParamState) {
        let shape = param.data.shape().to_vec();
        self.buf.push(ArrayD::<f64>::zeros(shape));
    }

    fn to_f64(arr: &ArrayD<f32>) -> ArrayD<f64> {
        arr.mapv(|x| x as f64)
    }

    fn to_f32(arr: &ArrayD<f64>) -> ArrayD<f32> {
        arr.mapv(|x| x as f32)
    }
}

impl Optimizer for SGD {
    fn step(
        &mut self,
        params: &mut [ParamState],
        gradients: &[ArrayD<f32>],
    ) -> Result<f64, OptimizerError> {
        if params.len() != gradients.len() {
            return Err(OptimizerError::GradientMismatch {
                expected: params.len(),
                actual: gradients.len(),
            });
        }

        self.steps += 1;
        let mut grad_norm = 0.0f64;

        for (i, (param, grad)) in params.iter_mut().zip(gradients.iter()).enumerate() {
            if i >= self.buf.len() {
                self.init_buffer_for_param(param);
            }

            let g = Self::to_f64(grad);
            grad_norm += (&g * &g).sum();

            let d_p = if self.weight_decay != 0.0 {
                let param_f64 = Self::to_f64(&param.data);
                &g + &param_f64 * self.weight_decay
            } else {
                g
            };

            if self.momentum != 0.0 {
                let buf_old = self.buf[i].clone();
                if self.nesterov {
                    self.buf[i] = &d_p + &buf_old * self.momentum;
                    let update = &d_p + &self.buf[i] * self.momentum;
                    let new_param = Self::to_f64(&param.data) - &update * self.lr;
                    param.data = Self::to_f32(&new_param);
                } else {
                    self.buf[i] = &buf_old * self.momentum + &d_p * (1.0 - self.dampening);
                    let new_param = Self::to_f64(&param.data) - &self.buf[i] * self.lr;
                    param.data = Self::to_f32(&new_param);
                }
            } else {
                let new_param = Self::to_f64(&param.data) - &d_p * self.lr;
                param.data = Self::to_f32(&new_param);
            }
        }

        Ok(grad_norm.sqrt())
    }

    fn zero_grad(&self, params: &mut [ParamState]) {
        for param in params {
            param.grad = None;
        }
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn state_dict(&self) -> OptimizerState {
        let param_states: Vec<ParamOptimizerState> = (0..self.buf.len())
            .map(|i| ParamOptimizerState {
                exp_avg: None,
                exp_avg_sq: None,
                momentum_buf: Some(self.buf[i].iter().map(|&x| x as f32).collect()),
            })
            .collect();

        OptimizerState {
            optimizer_type: "SGD".to_string(),
            steps: self.steps,
            param_states,
        }
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<(), OptimizerError> {
        if state.optimizer_type != "SGD" {
            return Err(OptimizerError::Other(format!(
                "优化器类型不匹配: 期望 SGD, 实际 {}",
                state.optimizer_type
            )));
        }

        self.steps = state.steps;
        self.buf.clear();

        for ps in &state.param_states {
            if let Some(mom_buf) = &ps.momentum_buf {
                self.buf.push(
                    ArrayD::from_shape_vec(
                        ndarray::IxDyn(&[mom_buf.len()]),
                        mom_buf.iter().map(|&x| x as f64).collect(),
                    )
                    .expect("Invalid shape for momentum buffer"),
                );
            } else {
                return Err(OptimizerError::BufferNotInitialized);
            }
        }

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// 优化器状态（用于序列化）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub optimizer_type: String,
    pub steps: u64,
    pub param_states: Vec<ParamOptimizerState>,
}

/// 单个参数的优化器状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamOptimizerState {
    #[serde(default)]
    pub exp_avg: Option<Vec<f32>>,
    #[serde(default)]
    pub exp_avg_sq: Option<Vec<f32>>,
    #[serde(default)]
    pub momentum_buf: Option<Vec<f32>>,
}

/// 学习率调度器 trait
pub trait LrScheduler {
    /// 获取指定步数的学习率
    fn get_lr(&self, step: u64) -> f64;

    /// 获取当前步数的学习率
    fn current_lr(&self) -> f64;

    /// 步进到下一步
    fn step(&mut self);

    /// 重置调度器
    fn reset(&mut self);
}

/// Linear Warmup + Cosine Decay 调度器
pub struct CosineWithWarmupScheduler {
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub min_lr: f64,
    pub target_lr: f64,
    current_step: u64,
}

impl CosineWithWarmupScheduler {
    pub fn new(warmup_steps: u64, total_steps: u64, min_lr: f64, target_lr: f64) -> Self {
        Self {
            warmup_steps,
            total_steps,
            min_lr,
            target_lr,
            current_step: 0,
        }
    }
}

impl LrScheduler for CosineWithWarmupScheduler {
    fn get_lr(&self, step: u64) -> f64 {
        if step < self.warmup_steps {
            self.target_lr * (step as f64 / self.warmup_steps.max(1) as f64)
        } else {
            let progress = (step - self.warmup_steps) as f64
                / (self.total_steps - self.warmup_steps).max(1) as f64;
            self.min_lr
                + 0.5
                    * (self.target_lr - self.min_lr)
                    * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }

    fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Linear Warmup + Linear Decay 调度器
pub struct LinearWithWarmupScheduler {
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub min_lr: f64,
    pub target_lr: f64,
    current_step: u64,
}

impl LinearWithWarmupScheduler {
    pub fn new(warmup_steps: u64, total_steps: u64, min_lr: f64, target_lr: f64) -> Self {
        Self {
            warmup_steps,
            total_steps,
            min_lr,
            target_lr,
            current_step: 0,
        }
    }
}

impl LrScheduler for LinearWithWarmupScheduler {
    fn get_lr(&self, step: u64) -> f64 {
        if step < self.warmup_steps {
            self.target_lr * (step as f64 / self.warmup_steps.max(1) as f64)
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            let progress =
                (step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;
            self.target_lr - (self.target_lr - self.min_lr) * progress
        }
    }

    fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Constant 调度器（调试用）
pub struct ConstantScheduler {
    pub lr: f64,
}

impl ConstantScheduler {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl LrScheduler for ConstantScheduler {
    fn get_lr(&self, _step: u64) -> f64 {
        self.lr
    }

    fn current_lr(&self) -> f64 {
        self.lr
    }

    fn step(&mut self) {}

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_param(value: f32) -> ParamState {
        ParamState {
            data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![value]).unwrap(),
            grad: Some(ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![1.0]).unwrap()),
        }
    }

    #[test]
    fn test_adamw_single_step() {
        let mut optimizer = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        let mut params = vec![create_test_param(1.0)];
        let gradients = vec![ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.1]).unwrap()];

        let result = optimizer.step(&mut params, &gradients);
        assert!(result.is_ok());

        let new_value = params[0].data[[0]];
        assert_ne!(new_value, 1.0, "参数值应该发生变化");
        assert!(new_value < 1.0, "参数值应该减小（因为梯度为正）");
    }

    #[test]
    fn test_adamw_bias_correction() {
        let mut optimizer = AdamW::new(0.01, (0.9, 0.999), 1e-8, 0.0);
        let mut params = vec![create_test_param(1.0)];
        let gradient = ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![1.0]).unwrap();

        // 第一步
        let _ = optimizer.step(&mut params, std::slice::from_ref(&gradient));
        let step1_value = params[0].data[[0]];

        // 执行多步后
        for _ in 0..100 {
            params[0].grad = Some(gradient.clone());
            let _ = optimizer.step(&mut params, std::slice::from_ref(&gradient));
        }

        let step101_value = params[0].data[[0]];

        // bias correction 在早期影响更大
        assert_ne!(step1_value, step101_value, "不同步数的更新幅度应不同");
    }

    #[test]
    fn test_adamw_weight_decay() {
        let gradient = ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.0]).unwrap(); // 梯度为零，只看权重衰减效果

        let mut optimizer_with_decay = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.1);
        let mut params_with_decay = vec![create_test_param(1.0)];
        let _ = optimizer_with_decay.step(&mut params_with_decay, std::slice::from_ref(&gradient));

        let mut optimizer_no_decay = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.0);
        let mut params_no_decay = vec![create_test_param(1.0)];
        let _ = optimizer_no_decay.step(&mut params_no_decay, std::slice::from_ref(&gradient));

        assert!(
            params_with_decay[0].data[[0]] < params_no_decay[0].data[[0]],
            "有权重衰减时参数应该更快趋向于 0"
        );
    }

    #[test]
    fn test_sgd_momentum() {
        let mut optimizer = SGD::new(0.01, 0.9, 0.0, false);
        let mut params = vec![create_test_param(1.0)];
        let gradient = ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.1]).unwrap();

        let _ = optimizer.step(&mut params, &[gradient.clone()]);
        let step1_value = params[0].data[[0]];

        params[0].grad = Some(gradient.clone());
        let _ = optimizer.step(&mut params, &[gradient.clone()]);
        let step2_value = params[0].data[[0]];

        // 有动量时，第二步的变化幅度应该更大
        let change1 = (1.0 - step1_value).abs();
        let change2 = (step1_value - step2_value).abs();
        assert!(change2 > change1, "动量应该使后续步子的变化幅度增大");
    }

    #[test]
    fn test_cosine_scheduler_shape() {
        let scheduler = CosineWithWarmupScheduler::new(100, 1000, 1e-6, 1e-3);

        // 验证 warmup 阶段线性增长
        let lr_start = scheduler.get_lr(0);
        let lr_mid_warmup = scheduler.get_lr(50);
        let lr_end_warmup = scheduler.get_lr(100); // warmup 结束

        assert_eq!(lr_start, 0.0, "warmup 起始学习率应为 0");
        assert!(
            lr_mid_warmup > lr_start && lr_mid_warmup < lr_end_warmup,
            "warmup 中间阶段应线性增长"
        );
        assert!(
            (lr_end_warmup - 1e-3).abs() < 1e-10,
            "warmup 结束时应达到目标学习率"
        );

        // 验证 cosine decay 形状
        let lr_early_decay = scheduler.get_lr(200);
        let lr_late_decay = scheduler.get_lr(900);
        assert!(lr_early_decay > lr_late_decay, "decay 阶段学习率应逐渐减小");

        // 验证最终收敛到 min_lr
        let lr_final = scheduler.get_lr(1000);
        assert!((lr_final - 1e-6).abs() < 1e-10, "最终学习率应为 min_lr");
    }

    #[test]
    fn test_linear_scheduler() {
        let scheduler = LinearWithWarmupScheduler::new(100, 1000, 1e-6, 1e-3);

        // 验证 warmup 阶段
        let lr_warmup_end = scheduler.get_lr(100);
        assert!((lr_warmup_end - 1e-3).abs() < 1e-10);

        // 验证线性衰减
        let lr_mid = scheduler.get_lr(550); // (100+1000)/2
        let expected_mid = (1e-3 + 1e-6) / 2.0;
        assert!(
            (lr_mid - expected_mid).abs() < 1e-6,
            "中间点应在目标值和最小值的平均位置"
        );

        // 验证终点
        let lr_end = scheduler.get_lr(1000);
        assert!((lr_end - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_constant_scheduler() {
        let scheduler = ConstantScheduler::new(1e-3);

        for step in 0..100 {
            let lr = scheduler.get_lr(step);
            assert!(
                (lr - 1e-3).abs() < 1e-10,
                "Constant 调度器应始终返回相同学习率"
            );
        }

        let mut mutable_scheduler = ConstantScheduler::new(1e-3);
        for _ in 0..50 {
            mutable_scheduler.step();
        }
        assert!(
            (mutable_scheduler.current_lr() - 1e-3).abs() < 1e-10,
            "步进后学习率不应改变"
        );
    }

    #[test]
    fn test_optimizer_state_save_load() {
        let mut optimizer = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        let mut params = vec![create_test_param(1.0)];
        let gradients = vec![ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.1]).unwrap()];

        // 执行几步更新
        for _ in 0..5 {
            let _ = optimizer.step(&mut params, &gradients);
        }

        // 保存状态
        let state = optimizer.state_dict();
        assert_eq!(state.optimizer_type, "AdamW");
        assert_eq!(state.steps, 5);
        assert_eq!(state.param_states.len(), 1);

        // 创建新优化器并加载状态
        let mut new_optimizer = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        let result = new_optimizer.load_state_dict(&state);
        assert!(result.is_ok());
        assert_eq!(new_optimizer.steps, 5);
        assert_eq!(new_optimizer.exp_avg.len(), 1);
    }

    #[test]
    fn test_zero_grad() {
        let optimizer = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        let mut params = vec![
            ParamState {
                data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![1.0]).unwrap(),
                grad: Some(ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.5]).unwrap()),
            },
            ParamState {
                data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![2.0]).unwrap(),
                grad: Some(ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.3]).unwrap()),
            },
        ];

        optimizer.zero_grad(&mut params);

        for param in &params {
            assert!(param.grad.is_none(), "梯度应被清空");
        }
    }

    #[test]
    fn test_gradient_mismatch_error() {
        let mut optimizer = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        let mut params = vec![create_test_param(1.0), create_test_param(2.0)];
        let gradients = vec![ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.1]).unwrap()]; // 只有 1 个梯度

        let result = optimizer.step(&mut params, &gradients);
        assert!(result.is_err());
        matches!(result.unwrap_err(), OptimizerError::GradientMismatch { .. });
    }

    #[test]
    fn test_sgd_nesterov() {
        let mut optimizer_nesterov = SGD::new(0.01, 0.9, 0.0, true);
        let mut params_nesterov = vec![create_test_param(1.0)];

        let mut optimizer_standard = SGD::new(0.01, 0.9, 0.0, false);
        let mut params_standard = vec![create_test_param(1.0)];

        let gradient = ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.1]).unwrap();

        let _ = optimizer_nesterov.step(&mut params_nesterov, std::slice::from_ref(&gradient));
        let _ = optimizer_standard.step(&mut params_standard, std::slice::from_ref(&gradient));

        // Nesterov 和标准动量应产生不同的结果
        assert_ne!(
            params_nesterov[0].data[[0]],
            params_standard[0].data[[0]],
            "Nesterov 和标准动量应产生不同的更新"
        );
    }

    #[test]
    fn test_sgd_state_save_load() {
        let mut optimizer = SGD::new(0.01, 0.9, 0.01, false);
        let mut params = vec![create_test_param(1.0)];
        let gradients = vec![ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.1]).unwrap()];

        for _ in 0..3 {
            let _ = optimizer.step(&mut params, &gradients);
        }

        let state = optimizer.state_dict();
        assert_eq!(state.optimizer_type, "SGD");
        assert_eq!(state.steps, 3);

        let mut new_optimizer = SGD::new(0.01, 0.9, 0.01, false);
        let result = new_optimizer.load_state_dict(&state);
        assert!(result.is_ok());
        assert_eq!(new_optimizer.steps, 3);
        assert_eq!(new_optimizer.buf.len(), 1);
    }

    #[test]
    fn test_scheduler_reset() {
        let mut scheduler = CosineWithWarmupScheduler::new(100, 1000, 1e-6, 1e-3);

        for _ in 0..500 {
            scheduler.step();
        }
        assert_eq!(scheduler.current_lr(), scheduler.get_lr(500));

        scheduler.reset();
        assert_eq!(scheduler.current_lr(), scheduler.get_lr(0));
        assert_eq!(scheduler.current_lr(), 0.0);
    }

    #[test]
    fn test_multiple_params_adamw() {
        let mut optimizer = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01);
        let mut params = vec![
            ParamState {
                data: ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
                grad: Some(ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.1, 0.2]).unwrap()),
            },
            ParamState {
                data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![3.0]).unwrap(),
                grad: Some(ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.3]).unwrap()),
            },
        ];
        let gradients = vec![
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.1, 0.2]).unwrap(),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![0.3]).unwrap(),
        ];

        let result = optimizer.step(&mut params, &gradients);
        assert!(result.is_ok());

        // 所有参数都应该改变
        assert_ne!(params[0].data[[0]], 1.0);
        assert_ne!(params[0].data[[1]], 2.0);
        assert_ne!(params[1].data[[0]], 3.0);
    }
}
