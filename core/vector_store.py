"""向量存储模块 - v2.0"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """向量存储抽象基类
    
    所有向量存储实现必须遵循此接口
    """
    
    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        添加文档到向量存储
        
        Args:
            ids: 文档 ID 列表
            embeddings: 向量数组 (n x dimension)
            documents: 文档内容列表
            metadatas: 元数据列表
        """
        pass
    
    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """
        查询最相似的文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filter_metadata: 元数据过滤条件
            
        Returns:
            [(id, content, score, metadata), ...]
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """
        删除文档
        
        Args:
            ids: 要删除的文档 ID 列表
            
        Returns:
            删除的文档数量
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        获取文档总数
        
        Returns:
            文档数量
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空向量存储"""
        pass
    
    @abstractmethod
    def save(self) -> None:
        """持久化向量存储（如果需要）"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查向量存储是否可用"""
        pass
