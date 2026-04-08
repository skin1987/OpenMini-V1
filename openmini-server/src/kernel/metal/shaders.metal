//
//  OpenMini Metal Shaders
//  
//  高性能Apple Silicon GPU算子实现
//

#include <metal_stdlib>
using namespace metal;

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
kernel void matmul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

/**
 * 分块矩阵乘法（优化版）
 */
kernel void matmul_tiled_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 16;
    
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];
    
    uint row = gid.y * TILE_SIZE + tid.y;
    uint col = gid.x * TILE_SIZE + tid.x;
    
    float sum = 0.0;
    
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < num_tiles; ++t) {
        // 加载A和B到共享内存
        uint a_col = t * TILE_SIZE + tid.x;
        uint b_row = t * TILE_SIZE + tid.y;
        
        if (row < M && a_col < K) {
            As[tid.y][tid.x] = A[row * K + a_col];
        } else {
            As[tid.y][tid.x] = 0.0;
        }
        
        if (b_row < K && col < N) {
            Bs[tid.y][tid.x] = B[b_row * N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 计算部分积
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
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
kernel void softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // 找最大值
    float max_val = -INFINITY;
    for (uint i = 0; i < len; ++i) {
        max_val = max(max_val, input[i]);
    }
    
    // 计算exp和sum
    float sum = 0.0;
    for (uint i = 0; i < len; ++i) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    
    // 归一化
    for (uint i = 0; i < len; ++i) {
        output[i] /= sum;
    }
}

// ============================================================================
// LayerNorm Kernel
// ============================================================================

/**
 * LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
 */
kernel void layernorm_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    
    // 计算均值
    float mean = 0.0;
    for (uint i = 0; i < hidden_size; ++i) {
        mean += input[row * hidden_size + i];
    }
    mean /= hidden_size;
    
    // 计算方差
    float var = 0.0;
    for (uint i = 0; i < hidden_size; ++i) {
        float diff = input[row * hidden_size + i] - mean;
        var += diff * diff;
    }
    var /= hidden_size;
    
    float std = sqrt(var + eps);
    
    // 归一化
    for (uint i = 0; i < hidden_size; ++i) {
        uint idx = row * hidden_size + i;
        output[idx] = (input[idx] - mean) / std * weight[i] + bias[i];
    }
}

// ============================================================================
// RMSNorm Kernel
// ============================================================================

kernel void rmsnorm_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    
    // 计算平方和
    float sum_sq = 0.0;
    for (uint i = 0; i < hidden_size; ++i) {
        float val = input[row * hidden_size + i];
        sum_sq += val * val;
    }
    
    float rms = sqrt(sum_sq / hidden_size + eps);
    
    // 归一化
    for (uint i = 0; i < hidden_size; ++i) {
        uint idx = row * hidden_size + i;
        output[idx] = (input[idx] / rms) * weight[i];
    }
}

// ============================================================================
// GELU Kernel
// ============================================================================

/**
 * GELU激活函数
 * gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
 */
kernel void gelu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = input[gid];
    float cdf = 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    output[gid] = x * cdf;
}

// ============================================================================
// SiLU Kernel
// ============================================================================

/**
 * SiLU激活函数
 * silu(x) = x * sigmoid(x)
 */
kernel void silu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = input[gid];
    output[gid] = x / (1.0 + exp(-x));
}

// ============================================================================
// RoPE (Rotary Position Embedding) Kernel
// ============================================================================

/**
 * RoPE位置编码
 */
kernel void rope_kernel(
    device float* x [[buffer(0)]],
    device const int* positions [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = seq_len * head_dim;
    if (gid >= total) return;
    
    uint pos_idx = gid / head_dim;
    uint dim_idx = gid % head_dim;
    
    if (dim_idx % 2 != 0) return; // 只处理偶数维度
    
    int pos = positions[pos_idx];
    float freq = 1.0 / pow(theta, dim_idx / float(head_dim));
    float angle = pos * freq;
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    
    float x0 = x[gid];
    float x1 = x[gid + 1];
    
    x[gid] = x0 * cos_angle - x1 * sin_angle;
    x[gid + 1] = x0 * sin_angle + x1 * cos_angle;
}

// ============================================================================
// FlashAttention Kernel (简化版)
// ============================================================================

/**
 * FlashAttention: out = softmax(Q * K^T / sqrt(d)) * V
 */
kernel void flash_attention_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant bool& causal [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint head_idx = gid.x;
    uint query_idx = gid.y;
    
    if (head_idx >= num_heads || query_idx >= seq_len) return;
    
    // 加载Q
    float q[64]; // 假设head_dim <= 64
    for (uint i = 0; i < head_dim; ++i) {
        q[i] = Q[query_idx * num_heads * head_dim + head_idx * head_dim + i];
    }
    
    // 计算注意力分数
    float scores[4096]; // 假设seq_len <= 4096
    float max_score = -INFINITY;
    
    for (uint key_idx = 0; key_idx < seq_len; ++key_idx) {
        // 因果掩码
        if (causal && key_idx > query_idx) continue;
        
        // 计算Q * K^T
        float score = 0.0;
        for (uint i = 0; i < head_dim; ++i) {
            float k = K[key_idx * num_heads * head_dim + head_idx * head_dim + i];
            score += q[i] * k;
        }
        score /= sqrt(float(head_dim));
        
        scores[key_idx] = score;
        max_score = max(max_score, score);
    }
    
    // Softmax
    float sum = 0.0;
    for (uint key_idx = 0; key_idx < seq_len; ++key_idx) {
        if (causal && key_idx > query_idx) continue;
        scores[key_idx] = exp(scores[key_idx] - max_score);
        sum += scores[key_idx];
    }
    
    // 加权求和
    for (uint i = 0; i < head_dim; ++i) {
        float val = 0.0;
        for (uint key_idx = 0; key_idx < seq_len; ++key_idx) {
            if (causal && key_idx > query_idx) continue;
            float v = V[key_idx * num_heads * head_dim + head_idx * head_dim + i];
            val += scores[key_idx] / sum * v;
        }
        out[query_idx * num_heads * head_dim + head_idx * head_dim + i] = val;
    }
}

// ============================================================================
// 向量运算 Kernels
// ============================================================================

/**
 * 向量加法
 */
kernel void add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    c[gid] = a[gid] + b[gid];
}

/**
 * 向量乘法
 */
kernel void mul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    c[gid] = a[gid] * b[gid];
}

/**
 * 向量缩放
 */
kernel void scale_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = input[gid] * scale;
}
