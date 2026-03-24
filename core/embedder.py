"""Embedding 模块 - v2.0"""
import os
import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Embedding 编码器
    
    负责将文本转换为向量表示
    """
    
    def __init__(self, model_name: str, dimension: int = None):
        """
        初始化 Embedding 编码器
        
        Args:
            model_name: Embedding 模型名称
            dimension: 向量维度（可选，自动推断）
        """
        self.model_name = model_name
        self._model: SentenceTransformer = None
        
        # 设置 HF_ENDPOINT 加速国内模型下载
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        logger.info(f"正在加载 Embedding 模型：{model_name}")
        self._load_model()
        
        # 自动推断维度
        if dimension is None:
            test_embedding = self.embed(["test"])
            self.dimension = len(test_embedding[0])
            logger.info(f"自动检测到向量维度：{self.dimension}")
        else:
            self.dimension = dimension
        
        logger.info(f"Embedding 编码器初始化完成：model={model_name}, dimension={self.dimension}")
    
    def _load_model(self):
        """加载 Embedding 模型"""
        try:
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding 模型加载成功")
        except Exception as e:
            logger.error(f"Embedding 模型加载失败：{e}")
            raise
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        计算文本的向量表示
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            向量数组 (n x dimension)
        """
        # 确保输入是列表
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([]).reshape(0, self.dimension)
        
        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # 确保返回的是 2D 数组
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding 计算失败：{e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        计算查询文本的向量
        
        Args:
            query: 查询文本
            
        Returns:
            查询向量
        """
        embeddings = self.embed([query])
        return embeddings[0]
    
    def embed_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量计算文档的向量
        
        Args:
            documents: 文档列表
            batch_size: 批次大小
            
        Returns:
            向量数组
        """
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.debug(f"处理批次 {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")
            
            batch_embeddings = self.embed(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([]).reshape(0, self.dimension)
    
    def is_available(self) -> bool:
        """
        检查 Embedding 模型是否可用
        
        Returns:
            True 如果可用
        """
        return self._model is not None
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.model_name
