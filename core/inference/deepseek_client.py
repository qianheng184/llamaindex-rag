"""DeepSeek LLM Provider 实现 - v2.0"""
import os
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional
import httpx

from .llm_provider import LLMProvider, LLMProviderFactory

logger = logging.getLogger(__name__)


class DeepSeekClient(LLMProvider):
    """DeepSeek LLM Provider
    
    支持 DeepSeek API 的同步和流式调用
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float = 60.0,
        max_retries: int = 3
    ):
        """
        初始化 DeepSeek 客户端
        
        Args:
            api_key: DeepSeek API Key
            model: 模型名称
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # HTTP 客户端
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"DeepSeek 客户端初始化完成：model={model}, base_url={base_url}")
    
    def _get_sync_client(self) -> httpx.Client:
        """获取同步 HTTP 客户端"""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(timeout=self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._sync_client
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """获取异步 HTTP 客户端"""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(timeout=self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._async_client
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        同步生成完整回答
        
        Args:
            prompt: 输入提示词
            **kwargs: 额外的生成参数
            
        Returns:
            生成的完整文本
            
        Raises:
            RuntimeError: API 调用失败
        """
        client = self._get_sync_client()
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            **kwargs
        }
        
        try:
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            logger.debug(f"DeepSeek 生成完成，长度：{len(content)}")
            return content
            
        except httpx.HTTPError as e:
            error_msg = f"DeepSeek API 调用失败：{str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except (KeyError, IndexError) as e:
            error_msg = f"DeepSeek API 响应格式异常：{str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        流式生成回答
        
        Args:
            prompt: 输入提示词
            **kwargs: 额外的生成参数
            
        Yields:
            生成的文本片段
            
        Raises:
            RuntimeError: API 调用失败
        """
        client = self._get_async_client()
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            **kwargs
        }
        
        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    # 跳过空行
                    if not line.strip():
                        continue
                    
                    # 解析 SSE 数据
                    if line.startswith("data: "):
                        data = line[6:]  # 移除 "data: " 前缀
                        
                        # 检查结束标记
                        if data.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            choices = chunk_data.get("choices", [])
                            
                            if choices and len(choices) > 0:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    yield content
                                    logger.debug(f"DeepSeek 流式片段：{content[:50]}...")
                                    
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析 JSON 片段：{data}")
                            continue
                            
        except httpx.HTTPError as e:
            error_msg = f"DeepSeek 流式 API 调用失败：{str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"DeepSeek 流式处理出错：{str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_model_name(self) -> str:
        """获取当前使用的模型名称"""
        return self.model
    
    def is_available(self) -> bool:
        """检查 DeepSeek Provider 是否可用"""
        # 简单的检查：API Key 是否存在且非空
        return bool(self.api_key and self.api_key.strip())
    
    def close(self):
        """关闭 HTTP 客户端连接"""
        if self._sync_client:
            self._sync_client.close()
        # 异步客户端需要在异步上下文中关闭


# 注册 Provider 到工厂
LLMProviderFactory.register_provider("deepseek", DeepSeekClient)
