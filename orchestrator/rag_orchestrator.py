"""RAG 编排器模块 - v2.0"""
import time
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from models.schemas import QueryResponse, Context
from services.retrieval_service import RetrievalService
from services.generation_service import GenerationService
from services.cache_service import CacheService

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """RAG 编排器
    
    协调检索服务和生成服务，完成完整的 RAG 查询流程
    """
    
    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
        cache_service: Optional[CacheService] = None
    ):
        """
        初始化 RAG 编排器
        
        Args:
            retrieval_service: 检索服务
            generation_service: 生成服务
            cache_service: 缓存服务（可选）
        """
        self.retrieval = retrieval_service
        self.generation = generation_service
        self.cache = cache_service
        
        logger.info("RAG 编排器初始化完成")
    
    def query(
        self,
        user_query: str,
        top_k: Optional[int] = None,
        use_cache: bool = True
    ) -> QueryResponse:
        """
        执行 RAG 查询
        
        Args:
            user_query: 用户查询
            top_k: 返回的结果数量
            use_cache: 是否使用缓存
            
        Returns:
            查询响应
        """
        start_time = time.time()
        logger.info(f"收到查询：'{user_query[:50]}...'")
        
        # Step 1: 检查缓存
        if use_cache and self.cache:
            cached_response = self.cache.get(user_query)
            if cached_response:
                logger.info("缓存命中")
                return cached_response
        
        try:
            # Step 2: 检索相关文档
            logger.info("Step 2: 检索相关文档...")
            contexts = self.retrieval.retrieve(user_query, top_k=top_k)
            
            if not contexts:
                logger.warning("未检索到相关文档")
                response = QueryResponse(
                    answer="抱歉，没有找到相关的信息来回答您的问题。",
                    contexts=[],
                    sources=[],
                    latency_ms=(time.time() - start_time) * 1000
                )
                return response
            
            # Step 3: 提取上下文内容
            context_texts = [ctx.content for ctx in contexts]
            context_sources = [ctx.metadata for ctx in contexts]
            
            # Step 4: 构建 prompt 并生成回答
            logger.info("Step 4: 生成回答...")
            answer = self.generation.generate_with_context(
                query=user_query,
                contexts=context_texts
            )
            
            # Step 5: 构建响应
            response = QueryResponse(
                answer=answer,
                contexts=context_texts,
                sources=context_sources,
                latency_ms=(time.time() - start_time) * 1000
            )
            
            # Step 6: 缓存结果
            if self.cache and use_cache:
                logger.debug("缓存查询结果")
                self.cache.set(user_query, response)
            
            logger.info(f"查询完成，耗时：{response.latency_ms:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"查询处理失败：{e}", exc_info=True)
            raise
    
    async def stream_query(
        self,
        user_query: str,
        top_k: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        流式 RAG 查询
        
        Args:
            user_query: 用户查询
            top_k: 返回的结果数量
            
        Yields:
            生成的文本片段
        """
        logger.info(f"收到流式查询：'{user_query[:50]}...'")
        
        try:
            # Step 1: 检索相关文档
            logger.info("Step 1: 检索相关文档...")
            contexts = self.retrieval.retrieve(user_query, top_k=top_k)
            
            if not contexts:
                yield "抱歉，没有找到相关的信息来回答您的问题。"
                return
            
            # Step 2: 提取上下文内容
            context_texts = [ctx.content for ctx in contexts]
            
            # Step 3: 构建 prompt
            logger.info("Step 2: 构建 prompt...")
            prompt = self.generation.build_prompt(user_query, context_texts)
            
            # Step 4: 流式生成
            logger.info("Step 3: 流式生成回答...")
            async for chunk in self.generation.stream_generate(prompt):
                yield chunk
            
            logger.info("流式查询完成")
            
        except Exception as e:
            logger.error(f"流式查询失败：{e}", exc_info=True)
            yield f"\n\n[错误：{str(e)}]"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "retrieval": self.retrieval.get_retrieval_stats(),
            "generation": self.generation.get_model_info(),
        }
        
        if self.cache:
            stats["cache"] = self.cache.stats()
        
        return stats
    
    def is_available(self) -> bool:
        """检查编排器是否可用"""
        return (
            self.retrieval.is_available() and
            self.generation.is_available()
        )
