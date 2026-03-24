#!/usr/bin/env python3
"""测试文档加载逻辑 - 验证多文件夹支持"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.config import load_config
from core.embedder import Embedder
from core.vector_store_chroma import ChromaVectorStore
from services.indexing_service import IndexingService

def test_document_loading():
    """测试文档加载逻辑"""
    print("=" * 70)
    print("测试多文件夹文档加载")
    print("=" * 70)
    
    try:
        # 加载配置
        print("\n1. 加载配置...")
        config = load_config()
        data_dir = config['indexing']['data_dir']
        print(f"   数据目录：{data_dir}")
        
        # 检查目录结构
        print("\n2. 检查目录结构...")
        if os.path.exists(data_dir):
            print(f"   ✓ 目录存在")
            
            # 列出所有子目录和文件
            for root, dirs, files in os.walk(data_dir):
                level = root.replace(data_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"   {indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    filepath = os.path.join(root, file)
                    size = os.path.getsize(filepath)
                    print(f"   {subindent}{file} ({size} bytes)")
        else:
            print(f"   ✗ 目录不存在")
            return False
        
        # 初始化服务
        print("\n3. 初始化索引服务...")
        embedder = Embedder(
            model_name=config['embedding']['model'],
            dimension=config['embedding']['dimension']
        )
        
        vector_store = ChromaVectorStore(
            persist_dir=config['database']['path']
        )
        
        indexing_service = IndexingService(
            embedder=embedder,
            vector_store=vector_store,
            config=config.get('indexing', {})
        )
        
        # 测试文档加载
        print("\n4. 测试文档加载...")
        documents = indexing_service._load_documents(data_dir)
        
        print(f"\n✓ 成功加载 {len(documents)} 个文档")
        
        # 显示文档信息
        if documents:
            print("\n5. 文档详情:")
            for i, doc in enumerate(documents[:5]):  # 只显示前 5 个
                metadata = doc.metadata
                print(f"   [{i+1}] {metadata.get('file_name', 'unknown')}")
                print(f"       路径：{metadata.get('file_path', 'unknown')}")
                print(f"       大小：{len(doc.text)} 字符")
            
            if len(documents) > 5:
                print(f"   ... 还有 {len(documents) - 5} 个文档")
        
        print("\n" + "=" * 70)
        print("测试通过 ✓")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        return False


if __name__ == "__main__":
    success = test_document_loading()
    sys.exit(0 if success else 1)
