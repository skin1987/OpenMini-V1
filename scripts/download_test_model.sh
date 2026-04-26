#!/bin/bash
# OpenMini-V1 模型下载脚本
# 用于下载小型GGUF模型进行端到端验证

set -e

MODEL_DIR="${1:-./models}"
mkdir -p "$MODEL_DIR"

echo "=========================================="
echo "OpenMini-V1 模型下载工具"
echo "目标目录: $MODEL_DIR"
echo "=========================================="

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 下载函数
download_model() {
    local name="$1"
    local url="$2"
    local desc="$3"
    
    echo ""
    echo -e "${YELLOW}[$name]${NC} $desc"
    echo "URL: $url"
    
    if [ -f "$MODEL_DIR/$name" ]; then
        echo -e "${GREEN}✓ 已存在，跳过下载${NC}"
        return 0
    fi
    
    if command -v curl &> /dev/null; then
        curl -L -o "$MODEL_DIR/$name" "$url" --progress-bar || {
            echo -e "${RED}✗ 下载失败: $name${NC}"
            return 1
        }
    elif command -v wget &> /dev/null; then
        wget -O "$MODEL_DIR/$name" "$url" || {
            echo -e "${RED}✗ 下载失败: $name${NC}"
            return 1
        }
    else
        echo -e "${RED}✗ 未找到 curl 或 wget${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ 下载完成: $name${NC}"
}

# ============================================
# 推荐的测试模型 (按大小排序)
# ============================================

echo ""
echo "可选模型列表:"
echo "  [1] tiny-llama-1.1b (1.3GB) - 最小模型，快速测试"
echo "  [2] qwen2.5-0.5b (400MB) - 超小模型，适合CI"
echo "  [3] phi-2 (1.7GB) - Microsoft小模型"
echo "  [4] 全部下载"
echo "  [0] 退出"
echo ""

read -p "请选择 [0-4]: " choice

case $choice in
    1)
        # TinyLlama 1.1B Chat - Q4_K_M量化 (~1.3GB)
        download_model "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf" \
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf" \
            "TinyLlama 1.1B Chat (Q4_K_M, ~1.3GB)"
        ;;
        
    2)
        # Qwen2.5 0.5B - Q4_K_M量化 (~400MB) - 最小的实用模型
        download_model "qwen2.5-0.5b-instruct-q4_k_m.gguf" \
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
            "Qwen2.5 0.5B Instruct (Q4_K_M, ~400MB)"
        ;;
        
    3)
        # Phi-2 - Q4_K_M量化 (~1.7GB)
        download_model "phi-2.Q4_K_M.gguf" \
            "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf" \
            "Microsoft Phi-2 (Q4_K_M, ~1.7GB)"
        ;;
        
    4)
        # 下载所有推荐模型
        download_model "qwen2.5-0.5b-instruct-q4_k_m.gguf" \
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
            "Qwen2.5 0.5B Instruct (Q4_K_M, ~400MB)" || true
            
        download_model "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf" \
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf" \
            "TinyLlama 1.1B Chat (Q4_K_M, ~1.3GB)" || true
        ;;
        
    0|*)
        echo "退出"
        exit 0
        ;;
esac

# ============================================
# 验证下载结果
# ============================================
echo ""
echo "=========================================="
echo -e "${GREEN}下载完成！${NC}"
echo "=========================================="
echo ""
echo "已下载的模型:"
ls -lh "$MODEL_DIR"/*.gguf 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (无)"

echo ""
echo "使用方法:"
echo "  cargo test --package openmini-server --test e2e_validation_test -- --nocapture"
echo ""
echo "或运行自定义测试:"
echo "  OPENMINI_MODEL_PATH=$MODEL_DIR/<model>.gguf cargo test e2e_..."
