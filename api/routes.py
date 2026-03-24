"""HTTP API 路由模块 - v2.0"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json

router = APIRouter()


@router.get("/health")
async def health_check():
    """健康检查接口"""
    # 注意：这个路由会被 main.py 中的同名路由覆盖
    # 这里仅作为文档说明
    return {
        "status": "healthy",
        "version": "2.0.0"
    }


@router.post("/query")
async def query(request: Request):
    """
    普通查询接口
    
    Args:
        request: 请求体，包含：
            - query: 用户问题
            - top_k: 返回的相关结果数量（默认 5）
            - use_cache: 是否使用缓存（默认 True）
    
    Returns:
        响应体，包含：
            - answer: LLM 生成的回答
            - contexts: 检索到的相关上下文
            - sources: 来源信息
            - latency_ms: 延迟（毫秒）
    """
    try:
        data = await request.json()
        query_text = data.get('query', '')
        top_k = data.get('top_k', 5)
        use_cache = data.get('use_cache', True)
        
        if not query_text.strip():
            raise HTTPException(status_code=400, detail="查询文本不能为空")
        
        # 从应用状态获取 orchestrator
        orchestrator = request.app.state.orchestrator
        
        if not orchestrator:
            raise HTTPException(status_code=503, detail="服务未初始化")
        
        # 执行查询
        response = orchestrator.query(query_text, top_k=top_k, use_cache=use_cache)
        
        return {
            "answer": response.answer,
            "contexts": response.contexts,
            "sources": response.sources,
            "latency_ms": response.latency_ms
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_stream(request: Request):
    """
    流式查询接口（SSE）
    
    Args:
        request: 请求体，包含：
            - query: 用户问题
            - top_k: 返回的相关结果数量（默认 5）
    
    Returns:
        SSE 流，格式：
            data: {"content": "文本片段"}
            data: {"done": true}
    """
    try:
        data = await request.json()
        query_text = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query_text.strip():
            raise HTTPException(status_code=400, detail="查询文本不能为空")
        
        orchestrator = request.app.state.orchestrator
        
        if not orchestrator:
            raise HTTPException(status_code=503, detail="服务未初始化")
        
        # 创建 SSE 流
        async def generate():
            try:
                async for chunk in orchestrator.stream_query(query_text, top_k=top_k):
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                
                # 结束标记
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats(request: Request):
    """
    获取统计信息
    
    Returns:
        统计信息，包含：
            - retrieval: 检索服务统计
            - generation: 生成服务统计
            - cache: 缓存统计（如果启用）
    """
    try:
        orchestrator = request.app.state.orchestrator
        
        if not orchestrator:
            raise HTTPException(status_code=503, detail="服务未初始化")
        
        stats = orchestrator.get_stats()
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 索引管理路由 ====================

@router.post("/index/build")
async def build_index(request: Request):
    """
    构建索引接口
    
    Args:
        request: 请求体，可选：
            - data_dir: 数据目录
            - force_rebuild: 是否强制重建
    """
    try:
        data = await request.json() or {}
        data_dir = data.get('data_dir')
        force_rebuild = data.get('force_rebuild', False)
        
        indexing_service = request.app.state.indexing_service
        
        if not indexing_service:
            raise HTTPException(status_code=503, detail="服务未初始化")
        
        # 异步执行构建任务
        from functools import partial
        import asyncio
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(indexing_service.build_index, data_dir or indexing_service.config.get('data_dir'), force_rebuild)
        )
        
        return {
            "success": result.success,
            "documents_processed": result.documents_processed,
            "chunks_created": result.chunks_created,
            "vectors_added": result.vectors_added,
            "message": result.message,
            "duration_seconds": result.duration_seconds
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/index")
async def delete_index(request: Request):
    """删除所有索引"""
    try:
        indexing_service = request.app.state.indexing_service
        
        if not indexing_service:
            raise HTTPException(status_code=503, detail="服务未初始化")
        
        result = indexing_service.delete_index()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/status")
async def index_status(request: Request):
    """获取索引状态"""
    try:
        indexing_service = request.app.state.indexing_service
        
        if not indexing_service:
            raise HTTPException(status_code=503, detail="服务未初始化")
        
        status = indexing_service.get_status()
        
        return {
            "total_documents": status.total_documents,
            "total_chunks": status.total_chunks,
            "vector_count": status.vector_count,
            "status": status.status,
            "last_updated": status.last_updated.isoformat() if status.last_updated else None,
            "error_message": status.error_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
