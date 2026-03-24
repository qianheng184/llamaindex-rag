"""ChromaDB 向量存储实现 - v2.0"""
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量存储实现
    
    支持持久化和高效的相似性搜索
    """
    
    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "documents",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 ChromaDB 向量存储
        
        Args:
            persist_dir: 持久化目录
            collection_name: 集合名称
            metadata: 集合元数据
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB 未安装，请运行：pip install chromadb"
            )
        
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        # ChromaDB 要求 metadata 不能为空字典
        self.collection_metadata = metadata if metadata else None
        
        # 确保持久化目录存在
        os.makedirs(persist_dir, exist_ok=True)
        
        logger.info(f"正在初始化 ChromaDB：persist_dir={persist_dir}")
        self._init_client()
        
        logger.info("ChromaDB 初始化完成")
    
    def _init_client(self):
        """初始化 ChromaDB 客户端"""
        try:
            # 创建持久化客户端
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 尝试获取现有集合，如果不存在则创建
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"集合 '{self.collection_name}' 已存在")
            except Exception:
                # 集合不存在，创建新集合
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata=self.collection_metadata
                )
                logger.info(f"创建新集合 '{self.collection_name}'")
            
            logger.info(f"集合 '{self.collection_name}' 已就绪，当前文档数：{self.collection.count()}")
            
        except Exception as e:
            logger.error(f"ChromaDB 初始化失败：{e}")
            raise
    
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
        if len(ids) != len(documents):
            raise ValueError("ids 和 documents 长度不一致")
        
        if embeddings.shape[0] != len(ids):
            raise ValueError("embeddings 数量与 ids 不一致")
        
        # 如果没有提供 metadatas，使用空字典
        if metadatas is None:
            metadatas = [{} for _ in range(len(ids))]
        
        try:
            # 转换为列表格式
            embeddings_list = embeddings.tolist()
            
            # 添加到集合
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"成功添加 {len(ids)} 个文档到向量存储")
            
        except Exception as e:
            logger.error(f"添加文档失败：{e}")
            raise
    
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
        if self.collection.count() == 0:
            logger.warning("向量存储为空，无法查询")
            return []
        
        try:
            # 转换为一维数组
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            
            # 执行查询
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "distances", "metadatas"]
            )
            
            # 解析结果
            formatted_results = []
            
            if results and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    content = results['documents'][0][i] if results['documents'] else ""
                    distance = results['distances'][0][i] if results['distances'] else 0.0
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    
                    # ChromaDB 返回的是距离，转换为相似度分数（余弦相似度）
                    # 距离越小，相似度越高
                    similarity = 1.0 / (1.0 + distance) if distance >= 0 else 1.0 - distance
                    
                    formatted_results.append((doc_id, content, similarity, metadata))
            
            logger.info(f"查询完成，返回 {len(formatted_results)} 个结果")
            return formatted_results
            
        except Exception as e:
            logger.error(f"查询失败：{e}")
            raise
    
    def delete(self, ids: List[str]) -> int:
        """
        删除文档
        
        Args:
            ids: 要删除的文档 ID 列表
            
        Returns:
            删除的文档数量
        """
        try:
            initial_count = self.collection.count()
            self.collection.delete(ids=ids)
            deleted_count = initial_count - self.collection.count()
            
            logger.info(f"成功删除 {deleted_count} 个文档")
            return deleted_count
            
        except Exception as e:
            logger.error(f"删除文档失败：{e}")
            raise
    
    def count(self) -> int:
        """获取文档总数"""
        return self.collection.count()
    
    def clear(self) -> None:
        """清空向量存储"""
        try:
            # 删除并重新创建集合
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata=self.collection_metadata
            )
            logger.info("向量存储已清空")
        except Exception as e:
            logger.error(f"清空向量存储失败：{e}")
            raise
    
    def save(self) -> None:
        """
        持久化向量存储
        
        ChromaDB 自动持久化，此方法不需要额外操作
        """
        logger.debug("ChromaDB 自动持久化，无需手动保存")
    
    def is_available(self) -> bool:
        """检查向量存储是否可用"""
        try:
            # 尝试访问集合
            _ = self.collection.count()
            return True
        except Exception:
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息
        
        Returns:
            集合信息字典
        """
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata,
            "persist_dir": self.persist_dir
        }
