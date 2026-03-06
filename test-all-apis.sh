#!/bin/bash

# FundProphet API 测试脚本

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

total_tests=0
passed_tests=0
failed_tests=0
skipped_tests=0

report_file="api_test_report_$(date +%Y%m%d_%H%M%S).txt"

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
    passed_tests=$((passed_tests + 1))
}

failure() {
    echo -e "${RED}✗${NC} $1"
    failed_tests=$((failed_tests + 1))
}

skip() {
    echo -e "${YELLOW}○${NC} $1"
    skipped_tests=$((skipped_tests + 1))
}

header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

test_endpoint() {
    local name=$1
    local url=$2
    local method=${3:-GET}
    local timeout=${4:-10}
    
    total_tests=$((total_tests + 1))
    
    echo -n "[$method] $name... "
    
    local status_code
    local response
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" --max-time "$timeout" "$url" 2>&1)
    else
        response=$(curl -s -X "$method" -w "\n%{http_code}" --max-time "$timeout" "$url" 2>&1)
    fi
    
    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$status_code" -ge 200 ] && [ "$status_code" -lt 300 ]; then
        success "HTTP $status_code"
        echo "  $body" | head -c 80
        echo "" >> "$report_file"
        echo "[$method] $name" >> "$report_file"
        echo "URL: $url" >> "$report_file"
        echo "Status: $status_code" >> "$report_file"
        echo "Response: $body" >> "$report_file"
        echo "---" >> "$report_file"
    elif [ "$status_code" -eq 404 ]; then
        skip "HTTP 404 (端点可能不存在)"
    else
        failure "HTTP $status_code"
        echo "  错误: $body" | head -c 100
        echo "" >> "$report_file"
        echo "[$method] $name" >> "$report_file"
        echo "URL: $url" >> "$report_file"
        echo "Status: $status_code" >> "$report_file"
        echo "Error: $body" >> "$report_file"
        echo "---" >> "$report_file"
    fi
}

main() {
    header "FundProphet API 测试"
    
    info "测试报告将保存到: $report_file"
    echo "FundProphet API 测试报告" > "$report_file"
    echo "测试时间: $(date)" >> "$report_file"
    echo "========================================" >> "$report_file"
    echo "" >> "$report_file"
    
    header "Stock Service API"
    test_endpoint "股票列表" "http://localhost:8001/api/stock/list" GET
    test_endpoint "股票数据" "http://localhost:8001/api/stock/data?symbol=000001" GET 15
    test_endpoint "添加股票" "http://localhost:8001/api/stock/add" POST
    test_endpoint "股票分析" "http://localhost:8001/api/stock/analyze?symbol=000001" GET 15
    
    header "LSTM API (Stock Service)"
    test_endpoint "LSTM 训练" "http://localhost:8001/api/lstm/train?symbol=000001" POST 30
    test_endpoint "LSTM 预测" "http://localhost:8001/api/lstm/predict?symbol=000001" GET 10
    
    header "Fund Service API"
    test_endpoint "基金列表" "http://localhost:8002/api/fund/list" GET
    test_endpoint "基金详情" "http://localhost:8002/api/fund/000001" GET 15
    test_endpoint "基金净值" "http://localhost:8002/api/fund/nav/000001" GET 10
    test_endpoint "基金持仓" "http://localhost:8002/api/fund/holdings/000001" GET 15
    test_endpoint "基金指标" "http://localhost:8002/api/fund/indicators/000001" GET 15
    test_endpoint "基金预测" "http://localhost:8002/api/fund/predict/000001" GET 10
    test_endpoint "周期分析" "http://localhost:8002/api/fund/cycle/000001" GET 15
    
    header "Index API (Fund Service)"
    test_endpoint "指数列表" "http://localhost:8002/api/index/list" GET
    
    header "News Service API"
    test_endpoint "最新新闻" "http://localhost:8003/api/news/latest" GET 10
    test_endpoint "新闻列表" "http://localhost:8003/api/news/list?limit=10" GET 10
    test_endpoint "新闻详情" "http://localhost:8003/api/news/detail/1" GET 10
    test_endpoint "同步新闻" "http://localhost:8003/api/news/sync" POST 30
    test_endpoint "分析新闻" "http://localhost:8003/api/news/analyze" POST 20
    test_endpoint "最新分析" "http://localhost:8003/api/news/analysis/latest" GET 10
    
    header "Market Service API"
    test_endpoint "宏观数据" "http://localhost:8004/api/market/macro" GET 10
    test_endpoint "资金流向" "http://localhost:8004/api/market/money-flow" GET 10
    test_endpoint "市场情绪" "http://localhost:8004/api/market/sentiment" GET 10
    test_endpoint "全球宏观" "http://localhost:8004/api/market/global" GET 10
    test_endpoint "市场特征" "http://localhost:8004/api/market/features" GET 10
    test_endpoint "同步市场数据" "http://localhost:8004/api/market/sync" POST 30
    
    header "Fund Intel Service API"
    test_endpoint "基金行业分析" "http://localhost:8005/api/fund-industry/analyze/000001" POST 20
    test_endpoint "获取基金行业" "http://localhost:8005/api/fund-industry/000001" GET 10
    test_endpoint "主要行业" "http://localhost:8005/api/fund-industry/primary/000001" GET 10
    
    header "News Classification API (Fund Intel)"
    test_endpoint "分类新闻" "http://localhost:8005/api/news-classification/classify" POST 15
    test_endpoint "行业统计" "http://localhost:8005/api/news-classification/stats" GET 10
    test_endpoint "今日新闻" "http://localhost:8005/api/news-classification/today" GET 10
    
    header "Fund News Association API (Fund Intel)"
    test_endpoint "匹配新闻" "http://localhost:8005/api/fund-news/match/000001" GET 15
    test_endpoint "新闻摘要" "http://localhost:8005/api/fund-news/summary/000001" GET 10
    test_endpoint "基金列表" "http://localhost:8005/api/fund-news/list" GET 10
    
    header "Investment Advice API (Fund Intel)"
    test_endpoint "获取投资建议" "http://localhost:8005/api/investment-advice/000001" GET 20
    test_endpoint "批量投资建议" "http://localhost:8005/api/investment-advice/batch" POST 20
    
    header "LLM Service API"
    test_endpoint "LLM 聊天" "http://localhost:8006/api/llm/chat" POST 30
    test_endpoint "分析新闻" "http://localhost:8006/api/llm/analyze-news" POST 30
    test_endpoint "行业分类" "http://localhost:8006/api/llm/classify-industry" POST 30
    test_endpoint "投资建议" "http://localhost:8006/api/llm/investment-advice" POST 30
    
    header "测试报告汇总"
    
    echo "" >> "$report_file"
    echo "========================================" >> "$report_file"
    echo "测试汇总" >> "$report_file"
    echo "========================================" >> "$report_file"
    echo "总测试数: $total_tests" >> "$report_file"
    echo "通过数: $passed_tests" >> "$report_file"
    echo "失败数: $failed_tests" >> "$report_file"
    echo "跳过数: $skipped_tests" >> "$report_file"
    
    echo -e "${BLUE}测试汇总：${NC}"
    echo -e "  总测试数: ${BLUE}$total_tests${NC}"
    echo -e "  通过数:   ${GREEN}$passed_tests${NC}"
    echo -e "  失败数:   ${RED}$failed_tests${NC}"
    echo -e "  跳过数:   ${YELLOW}$skipped_tests${NC}"
    
    if [ $failed_tests -eq 0 ]; then
        echo ""
        success "所有测试通过！API 运行正常。"
    else
        echo ""
        warn "发现 $failed_tests 个失败，请检查 $report_file"
    fi
    
    info "详细报告已保存到: $report_file"
}

main "$@"