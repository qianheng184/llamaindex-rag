import os
import logging
import sys
from dotenv import load_dotenv

# 解决国内下载 HuggingFace 模型慢的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 配置日志：清晰查看 RAG 每一步进度
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, data_dir="./data", persist_dir="./storage"):
        load_dotenv()
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        
        # 1. 全局配置：LLM (DeepSeek)
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.error("未找到 DEEPSEEK_API_KEY，请检查 .env 文件")
            raise ValueError("API Key Missing")

        Settings.llm = DeepSeek(
            model="deepseek-chat", 
            api_key=api_key,
            api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        )
        
        # 2. 全局配置：本地 Embedding (使用 BAAI 针对中文优化的模型)
        # 第一次运行会自动下载约 100MB 的模型文件
        logger.info("正在初始化本地 Embedding 模型...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-zh-v1.5"
        )
        
        # 3. 初始化或加载索引
        self.index = self._get_index()

    def _get_index(self):
        # 核心修改：不仅检查文件夹存在，还要检查里面是否有索引文件
        index_files = ["docstore.json", "vector_store.json", "index_store.json"]
        is_persisted = os.path.exists(self.persist_dir) and all(
            os.path.exists(os.path.join(self.persist_dir, f)) for f in index_files
        )

        if is_persisted:
            logger.info(f"正在从 {self.persist_dir} 加载持久化索引...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            return load_index_from_storage(storage_context)
        else:
            # 如果文件夹存在但文件不全，或者文件夹根本不存在
            logger.warning("本地索引缺失或不完整，开始读取文档并构建...")
            
            # 检查数据目录
            if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
                logger.error(f"数据目录 {self.data_dir} 为空，请先放入文档！")
                raise FileNotFoundError(f"请在 {self.data_dir} 中放入一些 .txt 或 .pdf 文件")

            # 读取文档并构建
            documents = SimpleDirectoryReader(self.data_dir).load_data()
            logger.info(f"已加载 {len(documents)} 个文档片段，正在生成向量并保存...")
            
            # 确保目录存在
            os.makedirs(self.persist_dir, exist_ok=True)
            
            index = VectorStoreIndex.from_documents(documents)
            
            # 持久化保存
            index.storage_context.persist(persist_dir=self.persist_dir)
            logger.info(f"索引构建并保存至 {self.persist_dir} 完成。")
            return index

    def query(self, text: str):
        # 这里的 streaming=True 可以实现流式输出效果
        query_engine = self.index.as_query_engine(streaming=True)
        return query_engine.query(text)