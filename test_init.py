#!/usr/bin/env python3
"""快速启动测试 - 仅验证组件初始化"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.config import load_config
from core.embedder import Embedder
from core.vector_store_chroma import ChromaVectorStore
from core.inference.llm_provider import create_llm_provider

def main():
    print("=" * 60)
    print("RAG System v2.0 - 组件初始化测试")
    print("=" * 60)
    
    try:
        # 1. 加载配置
        print("\n1. 加载配置...")
        config = load_config()
        print(f"   ✓ 配置加载成功：{config['system']['name']} v{config['system']['version']}")
        
        # 2. 初始化 Embedding
        print("\n2. 初始化 Embedding...")
        embedder = Embedder(
            model_name=config['embedding']['model'],
            dimension=config['embedding']['dimension']
        )
        print(f"   ✓ Embedding 初始化完成：{embedder.get_model_name()}")
        
        # 3. 初始化 ChromaDB
        print("\n3. 初始化 ChromaDB...")
        vector_store = ChromaVectorStore(
            persist_dir=config['database']['path']
        )
        print(f"   ✓ ChromaDB 初始化完成：文档数={vector_store.count()}")
        
        # 4. 初始化 LLM Provider
        print("\n4. 初始化 LLM Provider...")
        provider_type = config['inference']['provider']
        provider_config = config['inference']['providers'][provider_type]
        llm = create_llm_provider(provider_type, provider_config)
        print(f"   ✓ LLM Provider 初始化完成：{llm.get_model_name()}")
        
        # 5. 测试 Embedding
        print("\n5. 测试 Embedding 计算...")
        test_text = "这是一个测试"
        embedding = embedder.embed_query(test_text)
        print(f"   ✓ Embedding 计算成功：向量维度={len(embedding)}")
        
        print("\n" + "=" * 60)
        print("所有组件初始化成功 ✓")
        print("=" * 60)
        print("\n提示：可以运行 python main.py 启动完整服务")
        return 0
        
    except Exception as e:
        print(f"\n✗ 初始化失败：{e}")
        print("\n请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())
