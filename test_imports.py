#!/usr/bin/env python3
"""模块导入测试脚本"""
import sys

print("=" * 60)
print("RAG 系统模块导入测试")
print("=" * 60)
print()

# 测试配置模块
try:
    from config import load_config
    print("✓ config 模块导入成功")
except Exception as e:
    print(f"✗ config 模块导入失败：{e}")
    sys.exit(1)

# 测试 index 模块
try:
    from index.loader import DocumentLoader
    from index.chunker import TextChunker
    from index.vector_store import LocalVectorStore
    print("✓ index 模块导入成功")
except Exception as e:
    print(f"✗ index 模块导入失败：{e}")
    sys.exit(1)

# 测试 engine 模块
try:
    from engine.rag_engine import RAGEngine
    print("✓ engine 模块导入成功")
except Exception as e:
    print(f"✗ engine 模块导入失败：{e}")
    sys.exit(1)

# 测试 api 模块
try:
    from api.app import app
    print("✓ api 模块导入成功")
except Exception as e:
    print(f"✗ api 模块导入失败：{e}")
    sys.exit(1)

# 测试 client 模块（无需导入，直接检查文件）
try:
    with open('client/cli_chat.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if 'httpx' in content and 'chat_client' in content:
            print("✓ client 模块文件检查通过")
except Exception as e:
    print(f"✗ client 模块文件检查失败：{e}")
    sys.exit(1)

# 测试 evaluation 模块
try:
    with open('evaluation/evaluator.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if 'run_evaluation' in content and 'RAGEngine' in content:
            print("✓ evaluation 模块文件检查通过")
except Exception as e:
    print(f"✗ evaluation 模块文件检查失败：{e}")
    sys.exit(1)

print()
print("=" * 60)
print("所有模块导入测试通过！✅")
print("=" * 60)
print()
print("下一步操作：")
print("1. 创建 .env 文件并配置 DEEPSEEK_API_KEY")
print("2. 将文档放入 data/ 目录")
print("3. 运行：python index/build_index.py")
print("4. 启动 API: uvicorn api.app:app --reload")
print("5. 启动 CLI: python client/cli_chat.py")
