/**
 * OpenMini CUDA Kernels
 * 
 * 高性能GPU算子实现
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// ============================================================================
// 矩阵乘法 Kernel
// ============================================================================

/**
 * 矩阵乘法: C = A * B
 * 
 * A: M x K
 * B: K x N
 * C: M x N
 */
__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 使用共享内存优化
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    
    float sum = 0.0f;
    
    // 分块计算
    for (int t = 0; t < (K + 15) / 16; ++t) {
        // 加载A和B到共享内存
        if (row < M && t * 16 + tx < K) {
            As[ty][tx] = A[row * K + t * 16 + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * 16 + ty < K) {
            Bs[ty][tx] = B[(t * 16 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // 计算部分积
        for (int k = 0; k < 16; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Softmax Kernel
// ============================================================================

/**
 * Softmax: out = exp(x - max(x)) / sum(exp(x - max(x)))
 */
__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int len
) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // 找最大值
    float max_val = -INFINITY;
    for (int i = tid; i < len; i += blockDim.x) {
        int global_idx = blockIdx.x * len + i;
        if (global_idx < len) {
            max_val = fmaxf(max_val, input[global_idx]);
        }
    }
    
    shared[tid] = max_val;
    __syncthreads();
    
    // 归约找最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    
    max_val = shared[0];
    
    // 计算exp(x - max)
    float exp_sum = 0.0f;
    for (int i = tid; i < len; i += blockDim.x) {
        int global_idx = blockIdx.x * len + i;
        if (global_idx < len) {
            float exp_val = expf(input[global_idx] - max_val);
            output[global_idx] = exp_val;
            exp_sum += exp_val;
        }
    }
    
    shared[tid] = exp_sum;
    __syncthreads();
    
    // 归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    exp_sum = shared[0];
    
    // 归一化
    for (int i = tid; i < len; i += blockDim.x) {
        int global_idx = blockIdx.x * len + i;
        if (global_idx < len) {
            output[global_idx] /= exp_sum;
        }
    }
}

// ============================================================================
// LayerNorm Kernel
// ============================================================================

/**
 * LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
 */
__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int hidden_size,
    float eps
) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int idx = row * hidden_size + tid;
    
    // 计算均值
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += input[row * hidden_size + i];
    }
    
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    float mean = shared[0] / hidden_size;
    
    // 计算方差
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = input[row * hidden_size + i] - mean;
        var_sum += diff * diff;
    }
    
    shared[tid] = var_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    float var = shared[0] / hidden_size;
    float std = sqrtf(var + eps);
    
    // 归一化并应用权重和偏置
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = row * hidden_size + i;
        output[idx] = (input[idx] - mean) / std * weight[i] + bias[i];
    }
}

// ============================================================================
// FlashAttention Kernel (简化版)
// ============================================================================

/**
 * FlashAttention: out = softmax(Q * K^T / sqrt(d)) * V
 * 
 * Q: [seq_len, num_heads, head_dim]
 * K: [seq_len, num_heads, head_dim]
 * V: [seq_len, num_heads, head_dim]
 * out: [seq_len, num_heads, head_dim]
 */
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int seq_len,
    int num_heads,
    int head_dim,
    bool causal
) {
    int head_idx = blockIdx.x;
    int query_idx = blockIdx.y;
    
    if (head_idx >= num_heads || query_idx >= seq_len) return;
    
    extern __shared__ float shared[];
    float* scores = shared;
    float* k_cache = &shared[head_dim];
    
    // 加载Q
    float q[64]; // 假设head_dim <= 64
    for (int i = 0; i < head_dim; ++i) {
        q[i] = Q[query_idx * num_heads * head_dim + head_idx * head_dim + i];
    }
    
    // 计算注意力分数
    float max_score = -INFINITY;
    for (int key_idx = 0; key_idx < seq_len; ++key_idx) {
        // 因果掩码
        if (causal && key_idx > query_idx) continue;
        
        // 加载K
        for (int i = 0; i < head_dim; ++i) {
            k_cache[i] = K[key_idx * num_heads * head_dim + head_idx * head_dim + i];
        }
        
        // 计算Q * K^T
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += q[i] * k_cache[i];
        }
        score /= sqrtf((float)head_dim);
        
        scores[key_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Softmax
    float sum = 0.0f;
    for (int key_idx = 0; key_idx < seq_len; ++key_idx) {
        if (causal && key_idx > query_idx) continue;
        scores[key_idx] = expf(scores[key_idx] - max_score);
        sum += scores[key_idx];
    }
    
    // 加权求和
    for (int i = 0; i < head_dim; ++i) {
        float val = 0.0f;
        for (int key_idx = 0; key_idx < seq_len; ++key_idx) {
            if (causal && key_idx > query_idx) continue;
            float v = V[key_idx * num_heads * head_dim + head_idx * head_dim + i];
            val += scores[key_idx] / sum * v;
        }
        out[query_idx * num_heads * head_dim + head_idx * head_dim + i] = val;
    }
}

// ============================================================================
// RMSNorm Kernel
// ============================================================================

__global__ void rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // 计算平方和
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = input[row * hidden_size + i];
        sum_sq += val * val;
    }
    
    __shared__ float shared_sum;
    if (tid == 0) shared_sum = 0.0f;
    __syncthreads();
    
    atomicAdd(&shared_sum, sum_sq);
    __syncthreads();
    
    float rms = sqrtf(shared_sum / hidden_size + eps);
    
    // 归一化
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = row * hidden_size + i;
        output[idx] = (input[idx] / rms) * weight[i];
    }
}

// ============================================================================
// GELU Kernel
// ============================================================================

__global__ void gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[idx] = x * cdf;
}

// ============================================================================
// RoPE (Rotary Position Embedding) Kernel
// ============================================================================

__global__ void rope_kernel(
    float* __restrict__ x,
    const int* __restrict__ positions,
    int seq_len,
    int head_dim,
    float theta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * head_dim;
    
    if (idx >= total) return;
    
    int pos_idx = idx / head_dim;
    int dim_idx = idx % head_dim;
    
    if (dim_idx % 2 != 0) return; // 只处理偶数维度
    
    int pos = positions[pos_idx];
    float freq = 1.0f / powf(theta, dim_idx / (float)head_dim);
    float angle = pos * freq;
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);
    
    float x0 = x[idx];
    float x1 = x[idx + 1];
    
    x[idx] = x0 * cos_angle - x1 * sin_angle;
    x[idx + 1] = x0 * sin_angle + x1 * cos_angle;
}
