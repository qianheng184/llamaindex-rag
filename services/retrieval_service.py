"""检索服务模块 - v2.0"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from models.schemas import Context
from core.embedder import Embedder
from core.vector_store import VectorStore
from core.reranker import Reranker

logger = logging.getLogger(__name__)


class RetrievalService:
    """检索服务
    
    负责从向量存储中检索相关文档
    """
    
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Optional[Reranker] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化检索服务
        
        Args:
            embedder: Embedding 编码器
            vector_store: 向量存储
            reranker: Rerank 模块（可选）
            config: 配置字典
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        
        # 配置参数
        self.config = config or {}
        self.top_k = self.config.get('top_k', 5)
        self.rerank_top_k = self.config.get('rerank_top_k', 10)
        self.use_rerank = self.config.get('use_rerank', False) and reranker is not None
        
        logger.info(
            f"检索服务初始化完成："
            f"top_k={self.top_k}, "
            f"use_rerank={self.use_rerank}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Context]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的结果数量，默认使用配置的 top_k
            
        Returns:
            Context 对象列表
        """
        if top_k is None:
            top_k = self.top_k
        
        logger.info(f"开始检索：query='{query[:50]}...', top_k={top_k}")
        
        # Step 1: 计算查询向量
        logger.debug("计算查询向量...")
        query_embedding = self.embedder.embed_query(query)
        
        # Step 2: 向量检索
        logger.debug(f"向量检索，rerank_top_k={self.rerank_top_k}...")
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.rerank_top_k if self.use_rerank else top_k
        )
        
        logger.debug(f"检索到 {len(results)} 个结果")
        
        # Step 3: Rerank（如果启用）
        if self.use_rerank and self.reranker:
            logger.debug("执行 Rerank...")
            results = self.reranker.rerank(query, results, top_k=top_k)
        
        # Step 4: 转换为 Context 对象
        contexts = []
        for doc_id, content, score, metadata in results:
            context = Context(
                id=doc_id,
                content=content,
                score=float(score),
                metadata=metadata
            )
            contexts.append(context)
        
        logger.info(f"检索完成，返回 {len(contexts)} 个上下文")
        return contexts
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[tuple]:
        """
        检索相关文档（带详细评分信息）
        
        Args:
            query: 查询文本
            top_k: 返回的结果数量
            
        Returns:
            [(Context, score_details), ...]
        """
        contexts = self.retrieve(query, top_k)
        
        # TODO: 添加详细的评分信息
        # 可以包括：原始相似度分数、Rerank 分数、最终分数等
        score_details = [(ctx, {"final_score": ctx.score}) for ctx in contexts]
        
        return score_details
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        获取检索统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "total_documents": self.vector_store.count(),
            "embedder_model": self.embedder.get_model_name(),
            "reranker_enabled": self.use_rerank,
            "top_k": self.top_k,
            "rerank_top_k": self.rerank_top_k
        }
    
    def is_available(self) -> bool:
        """检查检索服务是否可用"""
        return (
            self.embedder.is_available() and
            self.vector_store.is_available()
        )
