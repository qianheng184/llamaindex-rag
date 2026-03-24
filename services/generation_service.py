"""生成服务模块 - v2.0"""
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

from core.inference.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class GenerationService:
    """生成服务
    
    负责调用 LLM 生成回答
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化生成服务
        
        Args:
            llm_provider: LLM Provider
            config: 配置字典
        """
        self.llm = llm_provider
        self.config = config or {}
        
        # 配置参数
        self.temperature = self.config.get('temperature', 0.7)
        self.max_tokens = self.config.get('max_tokens', 2048)
        self.system_prompt = self.config.get(
            'system_prompt',
            "你是一个智能助手，请根据提供的信息回答问题。"
        )
        
        logger.info(f"生成服务初始化完成：model={self.llm.get_model_name()}")
    
    def build_prompt(
        self,
        query: str,
        contexts: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        构建 RAG Prompt
        
        Args:
            query: 用户查询
            contexts: 相关上下文列表
            system_prompt: 系统提示词（可选）
            
        Returns:
            构建好的 prompt
        """
        if system_prompt is None:
            system_prompt = self.system_prompt
        
        # 构建上下文文本
        if contexts:
            context_text = "\n\n".join([
                f"[相关信息 {i+1}]\n{ctx}"
                for i, ctx in enumerate(contexts)
            ])
            
            prompt = f"""{system_prompt}

请根据以下相关信息回答用户的问题：

{context_text}

用户问题：{query}

请基于上述信息提供准确、完整的回答。如果相关信息中没有答案，请直接说明你不知道。"""
        else:
            # 没有上下文时的处理
            prompt = f"""{system_prompt}

用户问题：{query}

抱歉，我没有找到相关的信息来回答这个问题。"""
        
        logger.debug(f"Prompt 构建完成，长度：{len(prompt)} 字符")
        return prompt
    
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        生成回答
        
        Args:
            prompt: 输入 prompt
            **kwargs: 额外的生成参数
            
        Returns:
            生成的回答
        """
        logger.info("正在调用 LLM 生成回答...")
        
        # 合并默认参数和自定义参数
        generation_kwargs = {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            **kwargs
        }
        
        try:
            # 调用 LLM Provider
            response = self.llm.complete(prompt, **generation_kwargs)
            
            logger.info(f"回答生成完成，长度：{len(response)} 字符")
            return response
            
        except Exception as e:
            logger.error(f"LLM 生成失败：{e}")
            raise RuntimeError(f"生成回答失败：{str(e)}")
    
    async def stream_generate(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成回答
        
        Args:
            prompt: 输入 prompt
            **kwargs: 额外的生成参数
            
        Yields:
            生成的文本片段
        """
        logger.info("正在流式调用 LLM 生成回答...")
        
        # 合并默认参数和自定义参数
        generation_kwargs = {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            **kwargs
        }
        
        try:
            # 流式调用 LLM Provider
            async for chunk in self.llm.stream(prompt, **generation_kwargs):
                yield chunk
                
            logger.info("流式生成完成")
            
        except Exception as e:
            logger.error(f"LLM 流式生成失败：{e}")
            raise RuntimeError(f"流式生成失败：{str(e)}")
    
    def generate_with_context(
        self,
        query: str,
        contexts: List[str],
        **kwargs
    ) -> str:
        """
        带上下文的生成
        
        Args:
            query: 用户查询
            contexts: 相关上下文列表
            **kwargs: 额外的生成参数
            
        Returns:
            生成的回答
        """
        # Step 1: 构建 prompt
        prompt = self.build_prompt(query, contexts)
        
        # Step 2: 生成回答
        return self.generate(prompt, **kwargs)
    
    async def stream_generate_with_context(
        self,
        query: str,
        contexts: List[str],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        带上下文的流式生成
        
        Args:
            query: 用户查询
            contexts: 相关上下文列表
            **kwargs: 额外的生成参数
            
        Yields:
            生成的文本片段
        """
        # Step 1: 构建 prompt
        prompt = self.build_prompt(query, contexts)
        
        # Step 2: 流式生成
        async for chunk in self.stream_generate(prompt, **kwargs):
            yield chunk
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.llm.get_model_name(),
            "context_window": self.llm.get_context_window(),
            "is_available": self.llm.is_available()
        }
    
    def is_available(self) -> bool:
        """检查生成服务是否可用"""
        return self.llm.is_available()
