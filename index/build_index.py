#!/usr/bin/env python3
"""离线索引构建脚本

使用方法:
    python index/build_index.py
"""
import os
import sys
import logging

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 设置 HF_ENDPOINT 加速国内模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import load_config
from index.loader import DocumentLoader
from index.chunker import TextChunker
from index.vector_store import LocalVectorStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_index():
    """执行索引构建流程"""
    logger.info("=" * 60)
    logger.info("开始构建 RAG 索引")
    logger.info("=" * 60)
    
    # 1. 加载配置
    logger.info("步骤 1: 加载配置文件...")
    config = load_config()
    
    data_path = config['data_path']
    vector_store_path = config['vector_store_path']
    embedding_model = config['embedding_model']
    chunk_size = config['chunk_size']
    chunk_overlap = config['chunk_overlap']
    
    logger.info(f"数据目录：{data_path}")
    logger.info(f"向量存储目录：{vector_store_path}")
    logger.info(f"Embedding 模型：{embedding_model}")
    logger.info(f"Chunk 大小：{chunk_size}, 重叠：{chunk_overlap}")
    
    # 2. 加载文档
    logger.info("\n步骤 2: 加载文档...")
    loader = DocumentLoader(data_path)
    documents = loader.load_documents()
    logger.info(f"成功加载 {len(documents)} 个文档")
    
    # 3. 文本切分
    logger.info("\n步骤 3: 文本切分...")
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(documents)
    logger.info(f"生成 {len(chunks)} 个文本块")
    
    # 4. 初始化向量存储
    logger.info("\n步骤 4: 初始化向量存储...")
    vector_store = LocalVectorStore(
        persist_dir=vector_store_path,
        embedding_model=embedding_model
    )
    
    # 5. 生成向量并保存
    logger.info("\n步骤 5: 生成向量并保存到向量存储...")
    texts = [chunk.get_content() for chunk in chunks]
    
    logger.info("正在计算文本向量...")
    vectors = vector_store.embedding_model.encode(texts).tolist()
    
    logger.info(f"已将 {len(vectors)} 个向量添加到存储...")
    vector_store.add_documents(texts=texts, vectors=vectors)
    
    # 6. 保存索引
    logger.info("\n步骤 6: 保存索引...")
    vector_store.save()
    
    logger.info("\n" + "=" * 60)
    logger.info("索引构建完成!")
    logger.info(f"向量存储位置：{vector_store_path}")
    logger.info(f"总文档数：{len(vector_store)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        logger.error(f"索引构建失败：{e}", exc_info=True)
        sys.exit(1)
