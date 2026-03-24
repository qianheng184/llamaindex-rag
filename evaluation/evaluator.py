#!/usr/bin/env python3
"""离线评估脚本

用于评估 RAG 系统的问答性能

使用方法:
    python evaluation/evaluator.py
"""
import os
import sys
import json
import logging
from typing import List, Dict, Any

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

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


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    加载测试数据集
    
    Args:
        dataset_path: 数据集文件路径
        
    Returns:
        测试样本列表
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    logger.info(f"成功加载 {len(dataset)} 条测试样本")
    return dataset


def evaluate_answer(predicted: str, reference: str) -> float:
    """
    简单评估答案质量（基于关键词匹配）
    
    Args:
        predicted: 预测答案
        reference: 参考答案
        
    Returns:
        相似度分数 (0-1)
    """
    # 简单的包含关系判断
    if reference in predicted:
        return 1.0
    
    # 检查关键词（对于中文，简单分词）
    ref_words = set(reference.lower())
    pred_words = set(predicted.lower())
    
    # 计算 Jaccard 相似度
    intersection = ref_words & pred_words
    union = ref_words | pred_words
    
    if len(union) == 0:
        return 0.0
    
    similarity = len(intersection) / len(union)
    return similarity


def run_evaluation():
    """执行评估流程"""
    logger.info("=" * 60)
    logger.info("开始 RAG 系统离线评估")
    logger.info("=" * 60)
    
    # 1. 加载配置
    logger.info("步骤 1: 加载配置文件...")
    config = load_config()
    
    # 2. 初始化 RAG 引擎
    logger.info("步骤 2: 初始化 RAG 引擎...")
    rag_engine = RAGEngine(config)
    logger.info("RAG 引擎初始化完成")
    
    # 3. 加载测试集
    logger.info("步骤 3: 加载测试数据集...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "dataset.json")
    
    if not os.path.exists(dataset_path):
        logger.error(f"测试集文件不存在：{dataset_path}")
        return
    
    dataset = load_dataset(dataset_path)
    
    # 4. 执行评估
    logger.info("步骤 4: 执行评估...")
    results = []
    total_score = 0
    
    for i, sample in enumerate(dataset, 1):
        question = sample['question']
        reference = sample['reference_answer']
        
        logger.info(f"\n[{i}/{len(dataset)}] 问题：{question}")
        
        try:
            # 获取预测答案
            predicted = rag_engine.query(question)
            
            # 评估答案
            score = evaluate_answer(predicted, reference)
            total_score += score
            
            result = {
                'id': sample.get('id', i),
                'question': question,
                'reference': reference,
                'predicted': predicted,
                'score': score
            }
            results.append(result)
            
            # 输出结果
            logger.info(f"参考答案：{reference}")
            logger.info(f"预测答案：{predicted}")
            logger.info(f"得分：{score:.2f}")
        
        except Exception as e:
            logger.error(f"处理失败：{e}", exc_info=True)
            results.append({
                'id': sample.get('id', i),
                'question': question,
                'reference': reference,
                'predicted': '[错误]',
                'score': 0.0
            })
    
    # 5. 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("评估结果汇总")
    logger.info("=" * 60)
    
    avg_score = total_score / len(results) if results else 0
    accuracy = sum(1 for r in results if r['score'] > 0.5) / len(results) if results else 0
    
    logger.info(f"总样本数：{len(results)}")
    logger.info(f"平均得分：{avg_score:.2f}")
    logger.info(f"准确率 (>0.5): {accuracy:.2%}")
    
    # 保存详细结果
    result_path = os.path.join(current_dir, "evaluation_result.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': len(results),
                'average_score': avg_score,
                'accuracy': accuracy
            },
            'details': results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n详细结果已保存到：{result_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        logger.error(f"评估失败：{e}", exc_info=True)
        sys.exit(1)
