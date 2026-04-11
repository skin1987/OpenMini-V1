#!/bin/bash
# ============================================================================
# OpenMini-V1 Performance Regression Test Suite
# ============================================================================
#
# 运行完整的回归测试套件，用于 CI/CD 和发布验证
#
# 使用方式:
#   ./scripts/run_regression_tests.sh              # 标准模式（不含基准测试）
#   ./scripts/run_regression_tests.sh --bench       # 包含基准测试
#   ./scripts/run_regression_tests.sh --quick       # 快速模式（跳过压力测试）
#   ./scripts/run_regression_tests.sh --full        # 完整测试（含长时间运行）
#
# 环境变量:
#   STRESS_TEST_DURATION  - 压力测试持续时间（秒），默认 60
#   MEMORY_TEST_DURATION  - 内存测试持续时间（秒），默认 300
#   CI                    - 设置为任意值启用 CI 模式（自动缩短时间）
#   RUST_LOG              - 日志级别，默认 warn
#
# 输出:
#   - 测试结果摘要
#   - JSON 格式的性能指标（可被 CI 解析）
#   - 失败时返回非零退出码

set -euo pipefail

# ============================================================================
# 颜色和格式化
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 配置
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="openmini-server"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="${PROJECT_ROOT}/target/regression_test_$(date '+%Y%m%d_%H%M%S').log"

# 解析命令行参数
RUN_BENCH=false
QUICK_MODE=false
FULL_MODE=false

for arg in "$@"; do
    case $arg in
        --bench)
            RUN_BENCH=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --full)
            FULL_MODE=true
            export STRESS_TEST_DURATION=3600
            export MEMORY_TEST_DURATION=3600
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# CI 环境自动配置
if [ "${CI:-}" = "true" ] || [ -n "${GITHUB_ACTIONS:-}" ]; then
    export CI=true
    QUICK_MODE=true
fi

# 快速模式配置
if [ "$QUICK_MODE" = true ]; then
    export STRESS_TEST_DURATION=${STRESS_TEST_DURATION:-30}
    export MEMORY_TEST_DURATION=${MEMORY_TEST_DURATION:-60}
else
    export STRESS_TEST_DURATION=${STRESS_TEST_DURATION:-60}
    export MEMORY_TEST_DURATION=${MEMORY_TEST_DURATION:-300}
fi

# ============================================================================
# 工具函数
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo "" | tee -a "$LOG_FILE"
}

# 记录测试结果
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0
METRICS_COLLECTED="[]"

record_result() {
    local test_name="$1"
    local status="$2"
    local duration="$3"

    case $status in
        "passed")
            ((TESTS_PASSED++))
            log_success "$test_name (${duration}s)"
            ;;
        "failed")
            ((TESTS_FAILED++))
            log_error "$test_name FAILED (${duration}s)"
            ;;
        "skipped")
            ((TESTS_SKIPPED++))
            log_warning "$test_name SKIPPED"
            ;;
    esac
}

# 运行单个测试并记录结果
run_test() {
    local test_name="$1"
    local test_command="$2"
    local timeout_duration="${3:-600}"

    log_info "Running: $test_name"
    log_info "Command: $test_command"

    local start_time=$(date +%s)

    if timeout "$timeout_duration" bash -c "$test_command" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        record_result "$test_name" "passed" "$duration"
        return 0
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        if [ $exit_code -eq 124 ]; then
            log_error "$test_name TIMEOUT after ${timeout_duration}s"
            record_result "$test_name" "failed" "$duration"
        else
            record_result "$test_name" "failed" "$duration"
        fi
        return 1
    fi
}

# 从日志中提取性能指标
extract_metrics() {
    log_info "Extracting performance metrics..."

    # 提取 METRICS: 开头的 JSON 行
    local metrics=$(grep "^METRICS:" "$LOG_FILE" | sed 's/^METRICS://' || true)
    local leak_reports=$(grep "^LEAK_REPORT:" "$LOG_FILE" | sed 's/^LEAK_REPORT://' || true)

    if [ -n "$metrics" ] || [ -n "$leak_reports" ]; then
        METRICS_COLLECTED=$(echo "[${metrics}${metrics:+,}${leak_reports}]" | \
            jq -s 'add' 2>/dev/null || echo "[]")
        log_info "Collected $(echo "$METRICS_COLLECTED" | jq 'length') metric entries"
    fi
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    cd "$PROJECT_ROOT"

    log_section "OpenMini-V1 Performance Regression Test Suite"
    echo "  Date:       $TIMESTAMP"
    echo "  Commit:      $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
    echo "  Branch:      $(git branch --show-current 2>/dev/null || echo 'N/A')"
    echo "  Rust version: $(rustc --version 2>/dev/null || echo 'N/A')"
    echo "  Mode:        $(if [ "$QUICK_MODE" = true ]; then echo 'Quick (CI)'; elif [ "$FULL_MODE" = true ]; then echo 'Full'; else echo 'Standard'; fi)"
    echo "  Benchmarks:  $(if [ "$RUN_BENCH" = true ]; then echo 'Yes'; else echo 'No'; fi)"
    echo ""

    # 创建日志目录
    mkdir -p target

    # ------------------------------------------------------------------
    # Phase 1: 单元测试
    # ------------------------------------------------------------------
    log_section "Phase 1: Unit Tests"

    run_test "Unit Tests (workspace)" \
        "cargo test --workspace --lib -- --quiet" \
        300

    # ------------------------------------------------------------------
    # Phase 2: 集成测试
    # ------------------------------------------------------------------
    log_section "Phase 2: Integration Tests"

    run_test "Integration Tests" \
        "cargo test --workspace --test integration_test -- --quiet" \
        300

    # ------------------------------------------------------------------
    # Phase 3: 内存泄漏检测
    # ------------------------------------------------------------------
    log_section "Phase 3: Memory Leak Detection"

    if [ "$QUICK_MODE" = false ] || [ "$FULL_MODE" = true ]; then
        run_test "Memory Leak Tests" \
            "cargo test --package ${PACKAGE} --test memory_leak_test -- --nocapture --quiet" \
            600
    else
        log_warning "Skipping memory leak tests in quick mode"
        record_result "Memory Leak Tests" "skipped" "0"
    fi

    # ------------------------------------------------------------------
    # Phase 4: 压力测试
    # ------------------------------------------------------------------
    log_section "Phase 4: Stress Tests"

    if [ "$QUICK_MODE" = false ] || [ "$FULL_MODE" = true ]; then
        run_test "Stress Tests" \
            "cargo test --package ${PACKAGE} --test stress_test -- --nocapture --quiet" \
            900
    else
        log_warning "Skipping stress tests in quick mode"
        record_result "Stress Tests" "skipped" "0"
    fi

    # ------------------------------------------------------------------
    # Phase 5: 性能基准测试（可选）
    # ------------------------------------------------------------------
    if [ "$RUN_BENCH" = true ]; then
        log_section "Phase 5: Performance Benchmarks"

        # 检查 nightly 是否可用
        if command -v rustup &> /dev/null && rustup toolchain list | grep -q "nightly"; then
            run_test "Criterion Benchmarks" \
                "cargo +nightly bench --package ${PACKAGE} --bench performance_bench 2>&1 | tail -50" \
                1200
        else
            log_warning "Nightly Rust not found, skipping benchmarks"
            log_info "Install with: rustup toolchain install nightly"
            record_result "Criterion Benchmarks" "skipped" "0"
        fi
    fi

    # ------------------------------------------------------------------
    # 结果汇总
    # ------------------------------------------------------------------
    log_section "Test Results Summary"

    extract_metrics

    local total=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))
    local exit_code=0

    echo ""
    echo "  Total tests:  $total"
    echo -e "  ${GREEN}Passed:${NC}       $TESTS_PASSED"
    echo -e "  ${RED}Failed:${NC}       $TESTS_FAILED"
    echo -e "  ${YELLOW}Skipped:${NC}      $TESTS_SKIPPED"
    echo ""

    if [ $TESTS_FAILED -gt 0 ]; then
        log_error "REGRESSION TEST SUITE FAILED"
        echo ""
        echo "  Log file: $LOG_FILE"
        exit_code=1
    else
        log_success "REGRESSION TEST SUITE PASSED"
        exit_code=0
    fi

    # 输出最终 JSON 报告
    cat > "${PROJECT_ROOT}/target/regression_report.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'N/A')",
    "results": {
        "total": $total,
        "passed": $TESTS_PASSED,
        "failed": $TESTS_FAILED,
        "skipped": $TESTS_SKIPPED
    },
    "mode": "$(if [ "$QUICK_MODE" = true ]; then echo 'quick'; elif [ "$FULL_MODE" = true ]; then echo 'full'; else echo 'standard'; fi)",
    "benchmarks_included": $RUN_BENCH,
    "metrics": $METRICS_COLLECTED
}
EOF

    log_info "Report saved to: target/regression_report.json"
    log_info "Log saved to: $LOG_FILE"

    exit $exit_code
}

# 执行主函数
main "$@"
