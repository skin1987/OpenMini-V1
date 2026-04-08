#!/bin/bash
# ============================================================
# OpenMini-V1 本地 CI/CD 测试脚本
# 使用 nektos/act 在本地运行 GitHub Actions
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
ACT_PATH="${HOME}/Downloads/act_Darwin_x86_64/act"

# 检查 act 是否存在
if [ ! -f "$ACT_PATH" ]; then
    echo -e "${RED}❌ 错误: 未找到 act 工具${NC}"
    echo -e "${YELLOW}请确保 act 位于: $ACT_PATH${NC}"
    echo ""
    echo -e "${BLUE}安装方法:${NC}"
    echo "  1. 下载: https://github.com/nektos/act/releases"
    echo "  2. 解压到: ~/Downloads/act_Darwin_x86_64/"
    echo "  3. 运行: xattr -d com.apple.quarantine ~/Downloads/act_Darwin_x86_64/act"
    exit 1
fi

echo -e "${BLUE}🚀 OpenMini-V1 本地 CI/CD 测试工具${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 显示帮助信息
show_help() {
    echo -e "${GREEN}用法: $0 [选项]${NC}"
    echo ""
    echo -e "${YELLOW}选项:${NC}"
    echo "  lint         运行 Lint 和格式检查"
    echo "  build        运行构建验证 (Ubuntu + macOS)"
    echo "  test         运行完整测试套件"
    echo "  security     运行安全审计"
    echo "  full         运行完整 CI 流水线"
    echo "  list         列出所有可用的 Jobs"
    echo "  dry-run      模拟运行（不实际执行）"
    echo "  help         显示帮助信息"
    echo ""
    echo -e "${BLUE}示例:${NC}"
    echo "  $0 lint          # 只运行 Lint 检查"
    echo "  $0 test          # 运行测试套件"
    echo "  $0 full          # 完整 CI 流水线"
    echo "  $0 list          # 查看可用任务"
    echo ""
}

# 运行指定 job
run_job() {
    local job_name=$1
    shift
    
    echo -e "${GREEN}▶ 正在运行 Job: ${job_name}${NC}"
    echo -e "${YELLOW}----------------------------------------${NC}"
    
    cd "$PROJECT_ROOT"
    
    if ! "$ACT_PATH" -j "$job_name" "$@"; then
        echo -e "${RED}❌ Job '${job_name}' 执行失败${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Job '${job_name}' 执行成功${NC}"
    echo ""
}

# 主逻辑
case "${1:-help}" in
    lint)
        run_job "lint"
        ;;
    build)
        echo -e "${YELLOW}运行 Ubuntu 构建...${NC}"
        run_job "build" --env PLATFORM=ubuntu-latest
        echo -e "${YELLOW}运行 macOS 构建...${NC}"
        run_job "build" --env PLATFORM=macos-latest
        ;;
    test)
        run_job "test"
        ;;
    security)
        run_job "security-audit"
        ;;
    full)
        echo -e "${GREEN}🎯 运行完整 CI/CD 流水线...${NC}"
        echo ""
        
        echo -e "${BLUE}[1/4] Lint & Format Check${NC}"
        run_job "lint" || exit 1
        
        echo -e "${BLUE}[2/4] Build Verification${NC}"
        run_job "build" || exit 1
        
        echo -e "${BLUE}[3/4] Test Suite${NC}"
        run_job "test" || exit 1
        
        echo -e "${BLUE}[4/4] Security Audit${NC}"
        run_job "security-audit" || true
        
        echo -e "${GREEN}🎉 所有 CI 任务完成！${NC}"
        ;;
    list)
        echo -e "${BLUE}📋 可用的 GitHub Actions Jobs:${NC}"
        echo ""
        cd "$PROJECT_ROOT"
        "$ACT_PATH" -l
        ;;
    dry-run)
        echo -e "${YELLOW}🔍 模拟模式（Dry Run）${NC}"
        echo -e "${YELLOW}不会执行任何操作，仅显示将运行的步骤${NC}"
        echo ""
        cd "$PROJECT_ROOT"
        "$ACT_PATH" -n -v
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ 未知选项: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
