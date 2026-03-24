#!/usr/bin/env python3
"""主程序入口 - 调试和测试使用"""
import os
import sys
import logging

# 设置 HF_ENDPOINT 加速国内模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import load_config
from engine.rag_engine import RAGEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数 - 用于快速测试 RAG 引擎"""
    logger.info("=" * 60)
    logger.info("RAG 系统调试模式")
    logger.info("=" * 60)
    
    try:
        # 加载配置
        config = load_config()
        
        # 初始化 RAG 引擎
        rag = RAGEngine(config)
        
        logger.info("\nRAG 引擎初始化完成，可以进行测试")
        logger.info("示例用法:")
        logger.info("  response = rag.query('你的问题')")
        logger.info("  print(response)")
        
        # 简单的交互测试（可选）
        print("\n" + "=" * 60)
        print("RAG 系统已就绪，请输入您的问题（输入 'q' 退出）")
        print("=" * 60)
        
        while True:
            question = input("\n问：").strip()
            
            if question.lower() in ['q', 'quit', 'exit', '退出']:
                break
            
            if not question:
                continue
            
            print("答：", end="", flush=True)
            response = rag.query(question)
            print(response)
        
    except Exception as e:
        logger.error(f"程序运行出错：{e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
