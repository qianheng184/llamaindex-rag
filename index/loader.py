"""文档加载器模块"""
import os
import logging
from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """文档加载器，支持多种格式的文档"""
    
    def __init__(self, data_dir: str):
        """
        初始化文档加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在：{data_dir}")
        
        if not os.listdir(data_dir):
            raise ValueError(f"数据目录为空：{data_dir}")
    
    def load_documents(self) -> List[Document]:
        """
        加载数据目录中的所有文档
        
        Returns:
            文档列表
        """
        logger.info(f"正在从 {self.data_dir} 加载文档...")
        
        # 使用 llama-index 的 SimpleDirectoryReader 加载文档
        # 自动支持 PDF, TXT, DOCX 等格式
        reader = SimpleDirectoryReader(
            input_dir=self.data_dir,
            recursive=False
        )
        
        documents = reader.load_data()
        
        logger.info(f"成功加载 {len(documents)} 个文档")
        
        return documents
