"""FastAPI HTTP 服务入口"""
import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 设置 HF_ENDPOINT 加速国内模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import load_config
from engine.rag_engine import RAGEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="RAG API",
    description="基于 RAG 的智能问答 API 服务",
    version="1.0.0"
)

# 全局变量存储引擎实例
rag_engine: RAGEngine = None


class QueryRequest(BaseModel):
    """查询请求体"""
    query: str


class QueryResponse(BaseModel):
    """查询响应体"""
    answer: str


class HealthResponse(BaseModel):
    """健康检查响应体"""
    status: str
    message: str


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化 RAG 引擎"""
    global rag_engine
    
    logger.info("正在启动 RAG API 服务...")
    
    try:
        # 加载配置
        config = load_config()
        
        # 初始化 RAG 引擎
        rag_engine = RAGEngine(config)
        
        logger.info("RAG API 服务启动完成")
    except Exception as e:
        logger.error(f"RAG API 服务启动失败：{e}", exc_info=True)
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        message="RAG API 服务运行正常"
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    RAG 查询接口
    
    Args:
        request: 查询请求，包含 query 字段
        
    Returns:
        查询结果，包含 answer 字段
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG 引擎未初始化")
    
    try:
        logger.info(f"收到查询请求：{request.query}")
        
        # 使用 RAG 引擎进行查询
        answer = rag_engine.query(request.query)
        
        return QueryResponse(answer=answer)
    
    except Exception as e:
        logger.error(f"查询处理失败：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询处理失败：{str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # 加载配置
    config = load_config()
    
    # 启动服务
    uvicorn.run(
        app,
        host=config.get('api_host', '0.0.0.0'),
        port=config.get('api_port', 8000)
    )
