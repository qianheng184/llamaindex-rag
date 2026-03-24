#!/usr/bin/env python3
"""RAG System v2.0 - 离线索引管理 CLI"""
import os
import sys
import typer
from typing import Optional

# 设置 HF_ENDPOINT
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.config import load_config
from core.embedder import Embedder
from core.vector_store_chroma import ChromaVectorStore
from services.indexing_service import IndexingService

# 创建 Typer 应用
app = typer.Typer(
    name="rag-index",
    help="RAG 系统离线索引管理工具",
    add_completion=False
)


@app.command("build")
def build_index(
    data_dir: Optional[str] = typer.Option(
        None, 
        "--data-dir", "-d",
        help="数据目录路径（默认使用 config.yaml 中的配置）"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="强制重建索引"
    )
):
    """构建文档索引"""
    print("=" * 60)
    print("RAG System v2.0 - 构建索引")
    print("=" * 60)
    
    try:
        # 加载配置
        print("\n正在加载配置...")
        config = load_config()
        
        # 确定数据目录
        if data_dir is None:
            data_dir = config['indexing']['data_dir']
        
        print(f"数据目录：{data_dir}")
        print(f"强制重建：{'是' if force else '否'}")
        
        # 初始化服务
        print("\n初始化索引服务...")
        embedder = Embedder(
            model_name=config['embedding']['model'],
            dimension=config['embedding']['dimension']
        )
        
        vector_store = ChromaVectorStore(
            persist_dir=config['database']['path']
        )
        
        indexing_service = IndexingService(
            embedder=embedder,
            vector_store=vector_store,
            config=config.get('indexing', {})
        )
        
        # 构建索引
        print("\n开始构建索引...")
        result = indexing_service.build_index(data_dir, force_rebuild=force)
        
        # 显示结果
        print("\n" + "=" * 60)
        print("索引构建完成 ✓")
        print("=" * 60)
        print(f"处理文档数：{result.documents_processed}")
        print(f"创建文本块：{result.chunks_created}")
        print(f"添加向量数：{result.vectors_added}")
        print(f"耗时：{result.duration_seconds:.2f}秒")
        print(f"消息：{result.message}")
        print("=" * 60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ 错误：{e}")
        print("\n请确保数据目录存在且包含文档文件。")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 索引构建失败：{e}")
        print("\n请检查错误信息并重试。")
        sys.exit(1)


@app.command("update")
def update_index(
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir", "-d",
        help="数据目录路径"
    )
):
    """更新索引（添加新文档）"""
    print("=" * 60)
    print("RAG System v2.0 - 更新索引")
    print("=" * 60)
    
    try:
        # 加载配置
        print("\n正在加载配置...")
        config = load_config()
        
        if data_dir is None:
            data_dir = config['indexing']['data_dir']
        
        print(f"数据目录：{data_dir}")
        
        # 初始化服务
        print("\n初始化索引服务...")
        embedder = Embedder(
            model_name=config['embedding']['model'],
            dimension=config['embedding']['dimension']
        )
        
        vector_store = ChromaVectorStore(
            persist_dir=config['database']['path']
        )
        
        indexing_service = IndexingService(
            embedder=embedder,
            vector_store=vector_store,
            config=config.get('indexing', {})
        )
        
        # 更新索引
        print("\n开始更新索引...")
        result = indexing_service.update_index(data_dir)
        
        # 显示结果
        print("\n" + "=" * 60)
        print("索引更新完成 ✓")
        print("=" * 60)
        print(f"处理文档数：{result.documents_processed}")
        print(f"创建文本块：{result.chunks_created}")
        print(f"添加向量数：{result.vectors_added}")
        print(f"耗时：{result.duration_seconds:.2f}秒")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ 索引更新失败：{e}")
        sys.exit(1)


@app.command("delete")
def delete_index(
    confirm: bool = typer.Option(
        False,
        "--yes", "-y",
        help="确认删除（跳过交互提示）"
    )
):
    """删除所有索引"""
    print("=" * 60)
    print("RAG System v2.0 - 删除索引")
    print("=" * 60)
    
    if not confirm:
        # 交互确认
        response = typer.prompt(
            "\n⚠️  警告：此操作将删除所有索引数据，是否继续？\n请输入 'yes' 确认",
            default=""
        )
        
        if response.lower() != 'yes':
            print("\n操作已取消。")
            sys.exit(0)
    
    try:
        # 加载配置
        print("\n正在加载配置...")
        config = load_config()
        
        # 初始化服务
        print("\n初始化索引服务...")
        embedder = Embedder(
            model_name=config['embedding']['model']
        )
        
        vector_store = ChromaVectorStore(
            persist_dir=config['database']['path']
        )
        
        indexing_service = IndexingService(
            embedder=embedder,
            vector_store=vector_store
        )
        
        # 删除索引
        print("\n正在删除索引...")
        result = indexing_service.delete_index()
        
        # 显示结果
        print("\n" + "=" * 60)
        print("索引删除完成 ✓")
        print("=" * 60)
        print(f"删除文档数：{result['deleted_count']}")
        print(f"消息：{result['message']}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ 索引删除失败：{e}")
        sys.exit(1)


@app.command("status")
def show_status():
    """显示索引状态"""
    print("=" * 60)
    print("RAG System v2.0 - 索引状态")
    print("=" * 60)
    
    try:
        # 加载配置
        config = load_config()
        
        # 初始化服务
        embedder = Embedder(
            model_name=config['embedding']['model']
        )
        
        vector_store = ChromaVectorStore(
            persist_dir=config['database']['path']
        )
        
        indexing_service = IndexingService(
            embedder=embedder,
            vector_store=vector_store
        )
        
        # 获取状态
        status = indexing_service.get_status()
        
        # 显示状态
        print(f"\n数据库路径：{config['database']['path']}")
        print(f"文档总数：{status.total_documents}")
        print(f"文本块总数：{status.total_chunks}")
        print(f"向量数量：{status.vector_count}")
        print(f"状态：{status.status}")
        if status.last_updated:
            print(f"最后更新：{status.last_updated}")
        if status.error_message:
            print(f"错误信息：{status.error_message}")
        
        print("\n" + "=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ 无法获取索引状态：{e}")
        sys.exit(1)


if __name__ == "__main__":
    app()
