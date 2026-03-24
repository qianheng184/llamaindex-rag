"""文本切分器模块"""
import logging
from typing import List
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)


class TextChunker:
    """文本切分器，将文档切分成合适大小的 chunk"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化文本切分器
        
        Args:
            chunk_size: 每个 chunk 的大小（字符数）
            chunk_overlap: chunk 之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 使用 llama-index 的 SentenceSplitter 进行智能切分
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"文本切分器初始化完成：chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档切分成 chunk
        
        Args:
            documents: 文档列表
            
        Returns:
            切分后的 chunk 列表
        """
        logger.info(f"开始切分 {len(documents)} 个文档...")
        
        nodes = self.splitter.get_nodes_from_documents(documents)
        
        logger.info(f"文档切分完成，共生成 {len(nodes)} 个 chunk")
        
        return nodes
