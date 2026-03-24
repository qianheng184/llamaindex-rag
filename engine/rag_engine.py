"""RAG 在线检索与生成引擎"""
import os
import logging
from typing import List
from dotenv import load_dotenv

# 设置 HF_ENDPOINT 加速国内模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llama_index.llms.deepseek import DeepSeek
from index.vector_store import LocalVectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 检索与生成引擎"""
    
    def __init__(self, config: dict):
        """
        初始化 RAG 引擎
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.top_k = config.get('top_k', 3)
        
        # 加载环境变量（API Key）
        load_dotenv()
        
        # 初始化向量存储
        logger.info("正在加载向量存储...")
        self.vector_store = LocalVectorStore(
            persist_dir=config['vector_store_path'],
            embedding_model=config['embedding_model']
        )
        logger.info(f"向量存储加载完成，共 {len(self.vector_store)} 个文档")
        
        # 初始化 LLM
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.error("未找到 DEEPSEEK_API_KEY，请检查 .env 文件")
            raise ValueError("DEEPSEEK_API_KEY is missing")
        
        logger.info("正在初始化 LLM...")
        self.llm = DeepSeek(
            model=config['llm_model'],
            api_key=api_key,
            api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        )
        logger.info("LLM 初始化完成")
    
    def retrieve(self, query: str) -> List[str]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            相关文本片段列表
        """
        logger.info(f"开始检索：{query}")
        
        # 使用向量存储进行搜索
        results = self.vector_store.search(query=query, top_k=self.top_k)
        
        # 提取文本内容
        contexts = [text for text, score in results]
        
        logger.info(f"检索到 {len(contexts)} 个相关片段")
        
        return contexts
    
    def build_prompt(self, query: str, contexts: List[str]) -> str:
        """
        构建 RAG prompt
        
        Args:
            query: 用户查询
            contexts: 相关上下文列表
            
        Returns:
            构建好的 prompt
        """
        logger.info("正在构建 prompt...")
        
        # 构建上下文文本
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        # 构建完整的 prompt
        prompt = f"""你是一个智能助手，请根据以下提供的信息回答用户的问题。

相关信息：
{context_text}

用户问题：{query}

请基于上述信息提供准确、完整的回答。如果相关信息中没有答案，请直接说明你不知道。"""
        
        logger.info(f"Prompt 构建完成，长度：{len(prompt)} 字符")
        
        return prompt
    
    def generate(self, prompt: str) -> str:
        """
        调用 LLM 生成回答
        
        Args:
            prompt: 输入 prompt
            
        Returns:
            LLM 生成的回答
        """
        logger.info("正在调用 LLM 生成回答...")
        
        # 使用 LLM 生成回答
        response = self.llm.complete(prompt)
        
        answer = str(response)
        
        logger.info(f"回答生成完成，长度：{len(answer)} 字符")
        
        return answer
    
    def query(self, query: str) -> str:
        """
        完整的 RAG 查询流程
        
        Args:
            query: 用户查询
            
        Returns:
            最终回答
        """
        logger.info("=" * 50)
        logger.info(f"收到查询：{query}")
        logger.info("=" * 50)
        
        # Step 1: 检索相关文档
        contexts = self.retrieve(query)
        
        if not contexts:
            logger.warning("未检索到相关文档")
            return "抱歉，没有找到相关的信息来回答您的问题。"
        
        # Step 2: 构建 prompt
        prompt = self.build_prompt(query, contexts)
        
        # Step 3: 生成回答
        answer = self.generate(prompt)
        
        logger.info("=" * 50)
        logger.info("查询处理完成")
        logger.info("=" * 50)
        
        return answer
