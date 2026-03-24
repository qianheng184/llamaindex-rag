#!/usr/bin/env python3
"""RAG System v2.0 - Web API 服务入口"""
import os
import sys
import logging
from contextlib import asynccontextmanager

# 设置 HF_ENDPOINT 加速国内模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.config import load_config
from core.embedder import Embedder
from core.vector_store_chroma import ChromaVectorStore
from core.reranker import SimpleReranker
from core.inference.llm_provider import create_llm_provider
from services.cache_service import CacheService
from services.retrieval_service import RetrievalService
from services.generation_service import GenerationService
from services.indexing_service import IndexingService
from orchestrator.rag_orchestrator import RAGOrchestrator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 全局状态
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("=" * 60)
    logger.info("RAG System v2.0 正在启动...")
    logger.info("=" * 60)
    
    try:
        # 加载配置
        config = load_config()
        logger.info(f"配置加载完成：{config['system']['name']} v{config['system']['version']}")
        
        # 初始化核心组件
        logger.info("初始化核心组件...")
        
        # 1. Embedding
        embedder = Embedder(
            model_name=config['embedding']['model'],
            dimension=config['embedding']['dimension']
        )
        
        # 2. Vector Store
        vector_store = ChromaVectorStore(
            persist_dir=config['database']['path']
        )
        
        # 3. LLM Provider
        provider_type = config['inference']['provider']
        provider_config = config['inference']['providers'][provider_type]
        llm_provider = create_llm_provider(provider_type, provider_config)
        
        # 4. Reranker
        reranker = SimpleReranker()
        
        # 5. Cache Service
        cache_enabled = config.get('cache', {}).get('enabled', True)
        cache = CacheService(
            ttl=config.get('cache', {}).get('ttl', 3600)
        ) if cache_enabled else None
        
        # 初始化服务层
        logger.info("初始化服务层...")
        
        retrieval_service = RetrievalService(
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,
            config=config.get('retrieval', {})
        )
        
        generation_service = GenerationService(
            llm_provider=llm_provider,
            config=config.get('inference', {})
        )
        
        indexing_service = IndexingService(
            embedder=embedder,
            vector_store=vector_store,
            config=config.get('indexing', {})
        )
        
        # 初始化编排层
        logger.info("初始化编排层...")
        
        orchestrator = RAGOrchestrator(
            retrieval_service=retrieval_service,
            generation_service=generation_service,
            cache_service=cache
        )
        
        # 存储到全局状态
        app_state.update({
            'config': config,
            'embedder': embedder,
            'vector_store': vector_store,
            'llm_provider': llm_provider,
            'reranker': reranker,
            'cache': cache,
            'retrieval_service': retrieval_service,
            'generation_service': generation_service,
            'indexing_service': indexing_service,
            'orchestrator': orchestrator
        })
        
        logger.info("RAG System v2.0 启动完成 ✓")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"启动失败：{e}", exc_info=True)
        raise
    
    finally:
        # 关闭时清理
        logger.info("正在关闭 RAG System...")
        
        # 关闭 HTTP 客户端
        if 'llm_provider' in app_state and hasattr(app_state['llm_provider'], 'close'):
            app_state['llm_provider'].close()
        
        logger.info("RAG System 已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="RAG System v2.0",
    description="基于 RAG 的智能问答系统",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    orchestrator = app_state.get('orchestrator')
    
    if not orchestrator:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "服务未初始化"}
        )
    
    is_healthy = orchestrator.is_available()
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "version": "2.0.0",
        "available": is_healthy
    }


@app.post("/query")
async def query(request: Request):
    """普通查询接口"""
    try:
        data = await request.json()
        query_text = data.get('query', '')
        top_k = data.get('top_k', 5)
        use_cache = data.get('use_cache', True)
        
        if not query_text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "查询文本不能为空"}
            )
        
        orchestrator = app_state.get('orchestrator')
        if not orchestrator:
            return JSONResponse(
                status_code=503,
                content={"error": "服务未初始化"}
            )
        
        response = orchestrator.query(query_text, top_k=top_k, use_cache=use_cache)
        
        return {
            "answer": response.answer,
            "contexts": response.contexts,
            "sources": response.sources,
            "latency_ms": response.latency_ms
        }
        
    except Exception as e:
        logger.error(f"查询失败：{e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/query/stream")
async def query_stream(request: Request):
    """流式查询接口"""
    from fastapi.responses import StreamingResponse
    import json
    
    try:
        data = await request.json()
        query_text = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query_text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "查询文本不能为空"}
            )
        
        orchestrator = app_state.get('orchestrator')
        if not orchestrator:
            return JSONResponse(
                status_code=503,
                content={"error": "服务未初始化"}
            )
        
        # 创建 SSE 流
        async def generate():
            try:
                async for chunk in orchestrator.stream_query(query_text, top_k=top_k):
                    # SSE 格式
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                
                # 结束标记
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                logger.error(f"流式生成失败：{e}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"流式查询失败：{e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    orchestrator = app_state.get('orchestrator')
    
    if not orchestrator:
        return JSONResponse(
            status_code=503,
            content={"error": "服务未初始化"}
        )
    
    stats = orchestrator.get_stats()
    return stats


if __name__ == "__main__":
    import uvicorn
    
    # 加载配置
    config = load_config()
    
    # 启动服务
    uvicorn.run(
        app,
        host=config.get('api', {}).get('host', '0.0.0.0'),
        port=config.get('api', {}).get('port', 8000)
    )
