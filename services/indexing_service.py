"""索引服务模块 - v2.0"""
import os
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from core.embedder import Embedder
from core.vector_store import VectorStore
from models.schemas import BuildIndexResponse, IndexingStatus

logger = logging.getLogger(__name__)


class IndexingService:
    """索引服务
    
    负责构建、更新和管理文档索引
    """
    
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化索引服务
        
        Args:
            embedder: Embedding 编码器
            vector_store: 向量存储
            config: 配置字典
        """
        self.embedder = embedder
        self.vector_store = vector_store
        
        # 配置参数
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 500)
        self.chunk_overlap = self.config.get('chunk_overlap', 50)
        self.batch_size = self.config.get('batch_size', 32)
        
        # 文本切分器
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        logger.info(f"索引服务初始化完成：chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
    
    def build_index(self, data_dir: str, force_rebuild: bool = False) -> BuildIndexResponse:
        """
        构建文档索引
        
        Args:
            data_dir: 数据目录路径
            force_rebuild: 是否强制重建
            
        Returns:
            构建结果
        """
        start_time = time.time()
        logger.info(f"开始构建索引：data_dir={data_dir}, force_rebuild={force_rebuild}")
        
        try:
            # 检查是否需要清空现有索引
            if force_rebuild:
                logger.info("强制重建模式：清空现有索引...")
                self.vector_store.clear()
            
            # Step 1: 加载文档
            logger.info("Step 1: 加载文档...")
            documents = self._load_documents(data_dir)
            
            if not documents:
                return BuildIndexResponse(
                    success=False,
                    documents_processed=0,
                    chunks_created=0,
                    vectors_added=0,
                    message="未找到任何文档",
                    duration_seconds=time.time() - start_time
                )
            
            logger.info(f"成功加载 {len(documents)} 个文档")
            
            # Step 2: 切分文档
            logger.info("Step 2: 切分文档...")
            chunks = self._chunk_documents(documents)
            logger.info(f"切分完成，共 {len(chunks)} 个文本块")
            
            # Step 3: 计算嵌入
            logger.info("Step 3: 计算嵌入...")
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedder.embed_documents(texts, batch_size=self.batch_size)
            logger.info(f"嵌入计算完成，形状：{embeddings.shape}")
            
            # Step 4: 添加到向量存储
            logger.info("Step 4: 添加到向量存储...")
            ids = [f"chunk_{i}_{datetime.now().timestamp()}" for i in range(len(chunks))]
            metadatas = [self._extract_metadata(chunk) for chunk in chunks]
            
            self.vector_store.add(ids, embeddings, texts, metadatas)
            logger.info(f"成功添加 {len(ids)} 个向量到存储")
            
            # Step 5: 保存
            logger.info("Step 5: 持久化...")
            self.vector_store.save()
            
            duration = time.time() - start_time
            logger.info(f"索引构建完成，耗时：{duration:.2f}s")
            
            return BuildIndexResponse(
                success=True,
                documents_processed=len(documents),
                chunks_created=len(chunks),
                vectors_added=len(ids),
                message=f"索引构建完成，处理了 {len(documents)} 个文档，创建了 {len(chunks)} 个文本块",
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"索引构建失败：{e}", exc_info=True)
            raise
    
    def update_index(self, data_dir: str, file_patterns: Optional[List[str]] = None) -> BuildIndexResponse:
        """
        更新索引（仅添加新文档）
        
        Args:
            data_dir: 数据目录路径
            file_patterns: 文件匹配模式列表
            
        Returns:
            更新结果
        """
        logger.info(f"开始更新索引：data_dir={data_dir}")
        
        # TODO: 实现增量更新逻辑
        # 目前简单处理为全量构建
        return self.build_index(data_dir, force_rebuild=False)
    
    def delete_index(self) -> Dict[str, Any]:
        """
        删除索引
        
        Returns:
            删除结果
        """
        logger.info("开始删除索引...")
        
        try:
            initial_count = self.vector_store.count()
            self.vector_store.clear()
            self.vector_store.save()
            
            logger.info(f"索引已删除，共删除 {initial_count} 个文档")
            
            return {
                "success": True,
                "deleted_count": initial_count,
                "message": f"成功删除 {initial_count} 个文档"
            }
            
        except Exception as e:
            logger.error(f"删除索引失败：{e}", exc_info=True)
            raise
    
    def get_status(self) -> IndexingStatus:
        """
        获取索引状态
        
        Returns:
            索引状态信息
        """
        count = self.vector_store.count()
        
        return IndexingStatus(
            total_documents=count,  # 简化处理，实际应该统计文档数
            total_chunks=count,
            vector_count=count,
            last_updated=datetime.now(),
            status="ready",
            error_message=None
        )
    
    def _load_documents(self, data_dir: str):
        """
        加载文档（递归加载所有子文件夹）
        
        Args:
            data_dir: 数据目录
            
        Returns:
            文档列表
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在：{data_dir}")
        
        # 检查目录是否为空（包括子文件夹）
        all_files = []
        for root, dirs, files in os.walk(data_dir):
            all_files.extend(files)
        
        if not all_files:
            raise ValueError(f"数据目录为空：{data_dir}")
        
        logger.info(f"发现 {len(all_files)} 个文件")
        
        # 使用 llama-index 的 SimpleDirectoryReader
        # recursive=True 以递归加载所有子文件夹
        reader = SimpleDirectoryReader(
            input_dir=data_dir,
            recursive=True  # 修改为 True，递归加载子文件夹
        )
        
        documents = reader.load_data()
        logger.info(f"成功加载 {len(documents)} 个文档")
        
        return documents
    
    def _chunk_documents(self, documents):
        """
        切分文档
        
        Args:
            documents: 文档列表
            
        Returns:
            文本块列表
        """
        return self.splitter.get_nodes_from_documents(documents)
    
    def _extract_metadata(self, chunk) -> Dict[str, Any]:
        """
        从文本块提取元数据
        
        Args:
            chunk: 文本块
            
        Returns:
            元数据字典
        """
        metadata = {
            "source_file": getattr(chunk, 'ref_doc_id', 'unknown'),
            "chunk_index": getattr(chunk, 'index', 0),
        }
        
        # 如果有额外的元数据
        if hasattr(chunk, 'metadata'):
            metadata.update(chunk.metadata)
        
        return metadata
    
    def is_available(self) -> bool:
        """检查索引服务是否可用"""
        return (
            self.embedder.is_available() and
            self.vector_store.is_available()
        )
