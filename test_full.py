#!/usr/bin/env python3
"""RAG System v2.0 - 完整功能测试"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.config import load_config
from core.embedder import Embedder
from core.vector_store_chroma import ChromaVectorStore
from core.inference.llm_provider import create_llm_provider
from services.cache_service import CacheService
from services.retrieval_service import RetrievalService
from services.generation_service import GenerationService
from services.indexing_service import IndexingService
from orchestrator.rag_orchestrator import RAGOrchestrator


def test_all_components():
    """测试所有组件的初始化"""
    print("=" * 70)
    print("RAG System v2.0 - 完整功能测试")
    print("=" * 70)
    
    tests_passed = []
    tests_failed = []
    
    # Test 1: 配置加载
    print("\n[Test 1] 配置加载...")
    try:
        config = load_config()
        print(f"  ✓ 配置加载成功")
        print(f"    系统：{config['system']['name']} v{config['system']['version']}")
        tests_passed.append("配置加载")
    except Exception as e:
        print(f"  ✗ 配置加载失败：{e}")
        tests_failed.append("配置加载")
        return tests_passed, tests_failed
    
    # Test 2: Embedding
    print("\n[Test 2] Embedding 编码器...")
    try:
        embedder = Embedder(
            model_name=config['embedding']['model'],
            dimension=config['embedding']['dimension']
        )
        
        # 测试嵌入计算
        test_embedding = embedder.embed_query("测试文本")
        print(f"  ✓ Embedding 初始化完成，向量维度：{len(test_embedding)}")
        tests_passed.append("Embedding")
    except Exception as e:
        print(f"  ✗ Embedding 失败：{e}")
        tests_failed.append("Embedding")
    
    # Test 3: ChromaDB
    print("\n[Test 3] ChromaDB 向量存储...")
    try:
        vector_store = ChromaVectorStore(
            persist_dir=config['database']['path']
        )
        count = vector_store.count()
        print(f"  ✓ ChromaDB 初始化完成，当前文档数：{count}")
        tests_passed.append("ChromaDB")
    except Exception as e:
        print(f"  ✗ ChromaDB 失败：{e}")
        tests_failed.append("ChromaDB")
    
    # Test 4: LLM Provider
    print("\n[Test 4] LLM Provider...")
    try:
        provider_type = config['inference']['provider']
        provider_config = config['inference']['providers'][provider_type]
        llm = create_llm_provider(provider_type, provider_config)
        print(f"  ✓ LLM Provider 初始化完成：{llm.get_model_name()}")
        print(f"    可用状态：{'是' if llm.is_available() else '否'}")
        tests_passed.append("LLM Provider")
    except Exception as e:
        print(f"  ✗ LLM Provider 失败：{e}")
        tests_failed.append("LLM Provider")
    
    # Test 5: Cache Service
    print("\n[Test 5] 缓存服务...")
    try:
        cache = CacheService(ttl=config.get('cache', {}).get('ttl', 3600))
        
        # 测试缓存功能
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        
        if value == "test_value":
            print(f"  ✓ 缓存服务正常，TTL={cache.ttl}s")
            tests_passed.append("缓存服务")
        else:
            print(f"  ✗ 缓存功能异常")
            tests_failed.append("缓存服务")
    except Exception as e:
        print(f"  ✗ 缓存服务失败：{e}")
        tests_failed.append("缓存服务")
    
    # Test 6: Retrieval Service
    print("\n[Test 6] 检索服务...")
    try:
        retrieval = RetrievalService(
            embedder=embedder,
            vector_store=vector_store,
            config=config.get('retrieval', {})
        )
        stats = retrieval.get_retrieval_stats()
        print(f"  ✓ 检索服务初始化完成")
        print(f"    总文档数：{stats['total_documents']}")
        tests_passed.append("检索服务")
    except Exception as e:
        print(f"  ✗ 检索服务失败：{e}")
        tests_failed.append("检索服务")
    
    # Test 7: Generation Service
    print("\n[Test 7] 生成服务...")
    try:
        generation = GenerationService(
            llm_provider=llm,
            config=config.get('inference', {})
        )
        model_info = generation.get_model_info()
        print(f"  ✓ 生成服务初始化完成")
        print(f"    模型：{model_info['model_name']}")
        tests_passed.append("生成服务")
    except Exception as e:
        print(f"  ✗ 生成服务失败：{e}")
        tests_failed.append("生成服务")
    
    # Test 8: Indexing Service
    print("\n[Test 8] 索引服务...")
    try:
        indexing = IndexingService(
            embedder=embedder,
            vector_store=vector_store,
            config=config.get('indexing', {})
        )
        status = indexing.get_status()
        print(f"  ✓ 索引服务初始化完成")
        print(f"    当前状态：{status.status}")
        tests_passed.append("索引服务")
    except Exception as e:
        print(f"  ✗ 索引服务失败：{e}")
        tests_failed.append("索引服务")
    
    # Test 9: RAG Orchestrator
    print("\n[Test 9] RAG 编排器...")
    try:
        orchestrator = RAGOrchestrator(
            retrieval_service=retrieval,
            generation_service=generation,
            cache_service=cache
        )
        available = orchestrator.is_available()
        print(f"  ✓ RAG 编排器初始化完成")
        print(f"    可用状态：{'是' if available else '否'}")
        tests_passed.append("RAG 编排器")
    except Exception as e:
        print(f"  ✗ RAG 编排器失败：{e}")
        tests_failed.append("RAG 编排器")
    
    return tests_passed, tests_failed


def main():
    """主函数"""
    tests_passed, tests_failed = test_all_components()
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    print(f"\n通过 ({len(tests_passed)}):")
    for test in tests_passed:
        print(f"  ✓ {test}")
    
    if tests_failed:
        print(f"\n失败 ({len(tests_failed)}):")
        for test in tests_failed:
            print(f"  ✗ {test}")
    
    print("\n" + "=" * 70)
    
    if not tests_failed:
        print("✓ 所有测试通过！")
        print("\n下一步:")
        print("  1. 运行 'python cli_index.py build' 构建索引")
        print("  2. 运行 'python main.py' 启动 Web API 服务")
        print("  3. 运行 'python cli_query.py' 进入交互式查询")
        return 0
    else:
        print("✗ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
