#!/usr/bin/env python3
"""RAG System v2.0 - 快速测试脚本"""
import os
import sys

# 设置 HF_ENDPOINT
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.config import load_config
from core.embedder import Embedder
from core.inference.llm_provider import create_llm_provider

def test_config():
    """测试配置加载"""
    print("=" * 60)
    print("测试配置加载")
    print("=" * 60)
    
    try:
        config = load_config()
        print(f"✓ 配置加载成功")
        print(f"  系统名称：{config['system']['name']}")
        print(f"  版本：{config['system']['version']}")
        print(f"  Embedding 模型：{config['embedding']['model']}")
        print(f"  LLM Provider: {config['inference']['provider']}")
        return True
    except Exception as e:
        print(f"✗ 配置加载失败：{e}")
        return False


def test_embedder():
    """测试 Embedding"""
    print("\n" + "=" * 60)
    print("测试 Embedding")
    print("=" * 60)
    
    try:
        config = load_config()
        embedder = Embedder(
            model_name=config['embedding']['model'],
            dimension=config['embedding']['dimension']
        )
        
        # 测试嵌入计算
        test_texts = ["这是一个测试", "这是另一个测试"]
        embeddings = embedder.embed(test_texts)
        
        print(f"✓ Embedding 计算成功")
        print(f"  模型：{embedder.get_model_name()}")
        print(f"  维度：{embedder.dimension}")
        print(f"  输出形状：{embeddings.shape}")
        return True
    except Exception as e:
        print(f"✗ Embedding 测试失败：{e}")
        return False


def test_llm_provider():
    """测试 LLM Provider"""
    print("\n" + "=" * 60)
    print("测试 LLM Provider")
    print("=" * 60)
    
    try:
        config = load_config()
        provider_type = config['inference']['provider']
        provider_config = config['inference']['providers'][provider_type]
        
        llm = create_llm_provider(provider_type, provider_config)
        
        print(f"✓ LLM Provider 创建成功")
        print(f"  类型：{provider_type}")
        print(f"  模型：{llm.get_model_name()}")
        print(f"  可用：{llm.is_available()}")
        
        # 简单测试（不实际调用 API）
        if not llm.is_available():
            print(f"⚠ 注意：{provider_type} Provider 不可用，请检查 API Key")
        
        return True
    except Exception as e:
        print(f"✗ LLM Provider 测试失败：{e}")
        return False


def main():
    """运行所有测试"""
    print("\nRAG System v2.0 - 组件测试\n")
    
    results = []
    
    # 运行测试
    results.append(("配置加载", test_config()))
    results.append(("Embedding", test_embedder()))
    results.append(("LLM Provider", test_llm_provider()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过 ✓")
        print("\n提示：可以运行 main_v2.py 启动 Web API 服务")
    else:
        print("部分测试失败 ✗")
        print("\n请检查错误信息并修复")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
