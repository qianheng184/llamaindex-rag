"""向量存储与检索模块"""
import os
import json
import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LocalVectorStore:
    """本地向量存储，使用 JSON 持久化"""
    
    def __init__(self, persist_dir: str, embedding_model: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化向量存储
        
        Args:
            persist_dir: 持久化目录
            embedding_model: Embedding 模型名称
        """
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        
        # 确保持久化目录存在
        os.makedirs(persist_dir, exist_ok=True)
        
        # 数据存储结构
        self.vectors: List[List[float]] = []
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        
        # 加载或初始化 embedding 模型
        logger.info(f"正在加载 Embedding 模型：{embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info("Embedding 模型加载完成")
        
        # 尝试加载已存在的索引
        self._load_index()
    
    def _get_index_path(self) -> str:
        """获取索引文件路径"""
        return os.path.join(self.persist_dir, "vector_index.json")
    
    def _load_index(self):
        """加载已保存的索引"""
        index_path = self._get_index_path()
        
        if os.path.exists(index_path):
            logger.info(f"正在从 {index_path} 加载向量索引...")
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.vectors = data.get('vectors', [])
                self.texts = data.get('texts', [])
                self.metadata = data.get('metadata', [])
            logger.info(f"成功加载 {len(self.texts)} 个向量")
        else:
            logger.info("未找到现有索引，将创建新索引")
    
    def add_documents(self, texts: List[str], vectors: List[List[float]], 
                     metadata: List[Dict[str, Any]] = None):
        """
        添加文档到向量存储
        
        Args:
            texts: 文本列表
            vectors: 向量列表
            metadata: 元数据列表
        """
        if metadata is None:
            metadata = [{} for _ in range(len(texts))]
        
        self.vectors.extend(vectors)
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        logger.info(f"添加了 {len(texts)} 个文档到向量存储")
    
    def save(self):
        """保存向量索引到磁盘"""
        index_path = self._get_index_path()
        
        logger.info(f"正在保存向量索引到 {index_path}...")
        
        data = {
            'vectors': self.vectors,
            'texts': self.texts,
            'metadata': self.metadata
        }
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"向量索引已保存，共 {len(self.texts)} 个向量")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        搜索最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回最相关的 K 个结果
            
        Returns:
            (文本，相似度分数) 列表
        """
        if len(self.vectors) == 0:
            logger.warning("向量存储为空，无法进行搜索")
            return []
        
        # 计算查询向量
        query_vector = self.embedding_model.encode(query)
        
        # 计算余弦相似度
        query_vector = np.array(query_vector)
        vectors_array = np.array(self.vectors)
        
        # 归一化
        query_norm = np.linalg.norm(query_vector)
        vectors_norm = np.linalg.norm(vectors_array, axis=1)
        
        # 余弦相似度计算
        similarities = np.dot(vectors_array, query_vector) / (vectors_norm * query_norm + 1e-8)
        
        # 获取 top_k 个最相似的文档
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.texts[idx], float(similarities[idx])))
        
        logger.info(f"搜索完成，返回 {len(results)} 个结果")
        
        return results
    
    def __len__(self) -> int:
        """返回向量存储中的文档数量"""
        return len(self.texts)
