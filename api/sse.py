"""SSE 流式处理模块 - v2.0"""
import json
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


async def create_sse_stream(
    orchestrator,
    query: str,
    top_k: int = 5
) -> AsyncGenerator[str, None]:
    """
    创建 SSE 流式响应
    
    Args:
        orchestrator: RAG 编排器
        query: 用户查询
        top_k: 返回的相关结果数量
        
    Yields:
        SSE 格式的数据行
    """
    try:
        logger.info(f"开始流式查询：'{query[:50]}...'")
        
        async for chunk in orchestrator.stream_query(query, top_k=top_k):
            # 发送内容片段
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        
        # 发送完成标记
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        
        logger.info("流式查询完成")
        
    except Exception as e:
        logger.error(f"流式查询失败：{e}", exc_info=True)
        # 发送错误信息
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


def parse_sse_line(line: str) -> dict:
    """
    解析 SSE 数据行
    
    Args:
        line: SSE 数据行（以 "data: " 开头）
        
    Returns:
        解析后的数据字典
    """
    if not line.strip():
        return None
    
    if not line.startswith("data: "):
        return None
    
    data_str = line[6:]  # 移除 "data: "
    
    if data_str.strip() == "[DONE]":
        return {"done": True}
    
    try:
        data = json.loads(data_str)
        return data
    except json.JSONDecodeError:
        logger.warning(f"无法解析 SSE 数据：{data_str}")
        return None


class SSEStreamHandler:
    """SSE 流处理器"""
    
    def __init__(self):
        self.buffer = ""
        self.completed = False
        self.error = None
    
    async def process_stream(self, response_text: str):
        """
        处理 SSE 流文本
        
        Args:
            response_text: SSE 流文本
        """
        lines = response_text.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            
            data = parse_sse_line(line)
            
            if data is None:
                continue
            
            if 'error' in data:
                self.error = data['error']
                self.completed = True
                break
            
            if 'content' in data:
                self.buffer += data['content']
            
            if data.get('done', False):
                self.completed = True
                break
    
    def get_result(self) -> str:
        """获取完整结果"""
        return self.buffer
    
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.completed
    
    def has_error(self) -> bool:
        """是否有错误"""
        return self.error is not None
