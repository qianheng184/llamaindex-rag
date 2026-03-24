"""数据模型层 - Pydantic 模型和存储抽象"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


# ==================== 请求/响应模型 ====================

class QueryRequest(BaseModel):
    """查询请求体"""
    query: str = Field(..., description="用户查询文本", min_length=1)
    top_k: int = Field(default=5, description="返回的相关结果数量", ge=1, le=20)
    use_cache: bool = Field(default=True, description="是否使用缓存")


class QueryResponse(BaseModel):
    """查询响应体"""
    answer: str = Field(..., description="LLM 生成的回答")
    contexts: List[str] = Field(default=[], description="检索到的相关上下文")
    sources: List[Dict[str, Any]] = Field(default=[], description="来源信息")
    latency_ms: float = Field(default=0.0, description="延迟（毫秒）")


class StreamChunk(BaseModel):
    """流式响应片段"""
    content: str = Field(..., description="内容片段")
    is_last: bool = Field(default=False, description="是否为最后一个片段")


class StreamResponse(BaseModel):
    """流式响应元数据"""
    query_id: str = Field(..., description="查询 ID")
    status: str = Field(default="streaming", description="状态：streaming, completed, error")
    message: Optional[str] = Field(default=None, description="错误消息或完成消息")


# ==================== 文档和上下文模型 ====================

class Document(BaseModel):
    """文档元数据"""
    id: str = Field(..., description="文档唯一标识")
    content: str = Field(..., description="文档内容")
    metadata: Dict[str, Any] = Field(default={}, description="文档元数据")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")


class Context(BaseModel):
    """检索到的上下文片段"""
    id: str = Field(..., description="片段唯一标识")
    content: str = Field(..., description="片段内容")
    score: float = Field(..., description="相关性分数")
    metadata: Dict[str, Any] = Field(default={}, description="元数据信息")
    
    class Config:
        # 允许从任意类型创建，便于与 llama-index 集成
        arbitrary_types_allowed = True


class ChunkMetadata(BaseModel):
    """文本块元数据"""
    source_file: str = Field(..., description="源文件名")
    chunk_index: int = Field(..., description="块索引")
    start_char: int = Field(default=0, description="起始字符位置")
    end_char: int = Field(default=0, description="结束字符位置")
    page_number: Optional[int] = Field(default=None, description="页码（如果有）")


# ==================== 索引管理模型 ====================

class IndexingStatus(BaseModel):
    """索引状态信息"""
    total_documents: int = Field(..., description="文档总数")
    total_chunks: int = Field(..., description="文本块总数")
    vector_count: int = Field(..., description="向量数量")
    last_updated: Optional[datetime] = Field(default=None, description="最后更新时间")
    status: str = Field(default="ready", description="状态：ready, building, updating, error")
    error_message: Optional[str] = Field(default=None, description="错误消息")


class BuildIndexRequest(BaseModel):
    """构建索引请求"""
    data_dir: Optional[str] = Field(default=None, description="数据目录路径")
    force_rebuild: bool = Field(default=False, description="是否强制重建")


class BuildIndexResponse(BaseModel):
    """构建索引响应"""
    success: bool = Field(..., description="是否成功")
    documents_processed: int = Field(..., description="处理的文档数")
    chunks_created: int = Field(..., description="创建的文本块数")
    vectors_added: int = Field(..., description="添加的向量数")
    message: str = Field(..., description="结果消息")
    duration_seconds: float = Field(..., description="耗时（秒）")


class UpdateIndexRequest(BaseModel):
    """更新索引请求"""
    add_files: List[str] = Field(default=[], description="要添加的文件列表")
    remove_ids: List[str] = Field(default=[], description="要删除的文档 ID 列表")


class DeleteIndexResponse(BaseModel):
    """删除索引响应"""
    success: bool = Field(..., description="是否成功")
    deleted_count: int = Field(..., description="删除的文档数")
    message: str = Field(..., description="结果消息")


# ==================== 健康检查模型 ====================

class HealthStatus(BaseModel):
    """健康状态详情"""
    database: bool = Field(..., description="数据库状态")
    embedding_model: bool = Field(..., description="Embedding 模型状态")
    llm_provider: bool = Field(..., description="LLM Provider 状态")
    cache: bool = Field(..., description="缓存服务状态")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="整体状态：healthy, degraded, unhealthy")
    version: str = Field(..., description="系统版本")
    uptime_seconds: float = Field(..., description="运行时长（秒）")
    details: HealthStatus = Field(..., description="详细状态")


# ==================== 事件模型 ====================

class Event(BaseModel):
    """事件基类"""
    type: str = Field(..., description="事件类型")
    timestamp: datetime = Field(default_factory=datetime.now, description="事件时间戳")
    payload: Dict[str, Any] = Field(default={}, description="事件负载")


class IndexingEvent(Event):
    """索引事件"""
    type: str = "indexing"
    action: str = Field(..., description="操作类型：build, update, delete")
    status: str = Field(..., description="状态：started, progress, completed, failed")
    progress: float = Field(default=0.0, description="进度百分比")


class QueryEvent(Event):
    """查询事件"""
    type: str = "query"
    query_id: str = Field(..., description="查询 ID")
    action: str = Field(..., description="操作类型：start, retrieve, generate, complete, error")
    details: Optional[str] = Field(default=None, description="详细信息")
