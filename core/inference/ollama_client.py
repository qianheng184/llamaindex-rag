"""Ollama LLM Provider 实现 - v2.0"""
import logging
from typing import AsyncGenerator, Dict, Any, Optional
import httpx

from .llm_provider import LLMProvider, LLMProviderFactory

logger = logging.getLogger(__name__)


class OllamaClient(LLMProvider):
    """Ollama LLM Provider
    
    支持本地 Ollama 模型的同步和流式调用
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b",
        timeout: float = 120.0,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 Ollama 客户端
        
        Args:
            base_url: Ollama API 地址
            model: 模型名称
            timeout: 请求超时时间（秒）
            options: 额外的生成选项
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.options = options or {}
        
        # HTTP 客户端
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"Ollama 客户端初始化完成：model={model}, base_url={base_url}")
    
    def _get_sync_client(self) -> httpx.Client:
        """获取同步 HTTP 客户端"""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(timeout=self.timeout),
                headers={"Content-Type": "application/json"}
            )
        return self._sync_client
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """获取异步 HTTP 客户端"""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(timeout=self.timeout),
                headers={"Content-Type": "application/json"}
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
            "prompt": prompt,
            "stream": False,
            "options": {**self.options, **kwargs}
        }
        
        try:
            response = client.post("/api/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("response", "")
            
            logger.debug(f"Ollama 生成完成，长度：{len(content)}")
            return content
            
        except httpx.HTTPError as e:
            error_msg = f"Ollama API 调用失败：{str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except KeyError as e:
            error_msg = f"Ollama API 响应格式异常：{str(e)}"
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
            "prompt": prompt,
            "stream": True,
            "options": {**self.options, **kwargs}
        }
        
        try:
            async with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    # 跳过空行
                    if not line.strip():
                        continue
                    
                    try:
                        chunk_data = json.loads(line)
                        
                        # Ollama 返回的字段
                        content = chunk_data.get("response", "")
                        done = chunk_data.get("done", False)
                        
                        if content:
                            yield content
                            logger.debug(f"Ollama 流式片段：{content[:50]}...")
                        
                        if done:
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析 JSON 片段：{line}")
                        continue
                        
        except httpx.HTTPError as e:
            error_msg = f"Ollama 流式 API 调用失败：{str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Ollama 流式处理出错：{str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_model_name(self) -> str:
        """获取当前使用的模型名称"""
        return self.model
    
    def is_available(self) -> bool:
        """检查 Ollama Provider 是否可用"""
        # 尝试连接 Ollama 服务
        try:
            client = self._get_sync_client()
            response = client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """关闭 HTTP 客户端连接"""
        if self._sync_client:
            self._sync_client.close()
        # 异步客户端需要在异步上下文中关闭


# 注册 Provider 到工厂
LLMProviderFactory.register_provider("ollama", OllamaClient)
