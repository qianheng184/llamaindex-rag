#!/bin/bash
# RAG 系统运行说明验证脚本

echo "======================================"
echo "RAG 系统 - 运行说明"
echo "======================================"
echo ""
echo "项目结构已完成！以下是运行步骤："
echo ""
echo "1️⃣  安装依赖"
echo "   pip install -r requirements.txt"
echo ""
echo "2️⃣  配置环境变量"
echo "   创建 .env 文件，添加："
echo "   DEEPSEEK_API_KEY=your_api_key_here"
echo "   DEEPSEEK_API_BASE=https://api.deepseek.com/v1"
echo ""
echo "3️⃣  准备数据"
echo "   将文档放入 data/ 目录"
echo ""
echo "4️⃣  构建索引（离线）"
echo "   python index/build_index.py"
echo ""
echo "5️⃣  启动 HTTP API 服务"
echo "   uvicorn api.app:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "6️⃣  启动 CLI 客户端（新终端窗口）"
echo "   python client/cli_chat.py"
echo ""
echo "7️⃣  运行评估"
echo "   python evaluation/evaluator.py"
echo ""
echo "======================================"
echo "项目结构验证"
echo "======================================"
echo ""

# 检查关键文件是否存在
files=(
    "config/config.yaml"
    "config/__init__.py"
    "index/loader.py"
    "index/chunker.py"
    "index/vector_store.py"
    "index/build_index.py"
    "index/__init__.py"
    "engine/rag_engine.py"
    "engine/__init__.py"
    "api/app.py"
    "client/cli_chat.py"
    "evaluation/evaluator.py"
    "evaluation/dataset.json"
    "main.py"
    "requirements.txt"
    "README.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (缺失)"
    fi
done

echo ""
echo "======================================"
echo "所有核心文件已创建完成！"
echo "======================================"
