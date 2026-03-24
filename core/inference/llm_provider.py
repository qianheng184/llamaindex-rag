"""LLM Provider 抽象基类 - v2.0"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional


class LLMProvider(ABC):
    """LLM 提供者抽象基类
    
    所有 LLM Provider 必须实现此接口，支持同步和异步流式调用
    """
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        同步生成完整回答
        
        Args:
            prompt: 输入提示词
            **kwargs: 额外的生成参数（如 temperature, max_tokens 等）
            
        Returns:
            生成的完整文本
        """
        pass
    
    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        流式生成回答
        
        Args:
            prompt: 输入提示词
            **kwargs: 额外的生成参数
            
        Yields:
            生成的文本片段
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        获取当前使用的模型名称
        
        Returns:
            模型名称
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查 LLM Provider 是否可用
        
        Returns:
            True 如果可用，False 否则
        """
        pass
    
    # ==================== 可选的辅助方法 ====================
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的 token 数量（可选实现）
        
        Args:
            text: 输入文本
            
        Returns:
            token 数量
        """
        # 默认简单估算：每 4 个字符约等于 1 个 token
        return len(text) // 4
    
    def get_context_window(self) -> int:
        """
        获取模型的上下文窗口大小（可选实现）
        
        Returns:
            上下文窗口大小（token 数）
        """
        # 默认返回一个保守值
        return 4096


class LLMProviderFactory:
    """LLM Provider 工厂类
    
    根据配置创建对应的 LLM Provider 实例
    """
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        注册 LLM Provider
        
        Args:
            name: Provider 名称
            provider_class: Provider 类
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create_provider(
        cls,
        provider_type: str,
        config: Dict[str, Any]
    ) -> LLMProvider:
        """
        创建 LLM Provider 实例
        
        Args:
            provider_type: Provider 类型（如 'deepseek', 'ollama'）
            config: Provider 配置
            
        Returns:
            LLM Provider 实例
            
        Raises:
            ValueError: 不支持的 Provider 类型
        """
        provider_key = provider_type.lower()
        
        if provider_key not in cls._providers:
            raise ValueError(f"不支持的 LLM Provider 类型：{provider_type}")
        
        provider_class = cls._providers[provider_key]
        return provider_class(**config)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        获取所有可用的 Provider 类型
        
        Returns:
            Provider 类型列表
        """
        return list(cls._providers.keys())


# ==================== 工具函数 ====================

def create_llm_provider(provider_type: str, config: Dict[str, Any]) -> LLMProvider:
    """
    便捷函数：创建 LLM Provider 实例
    
    Args:
        provider_type: Provider 类型
        config: Provider 配置
        
    Returns:
        LLM Provider 实例
    """
    return LLMProviderFactory.create_provider(provider_type, config)
