"""配置加载工具模块"""
import os
import yaml
from typing import Any, Dict


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径，默认为 config/config.yaml
        
    Returns:
        配置字典
    """
    if config_path is None:
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, "config", "config.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 解析相对路径为绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    if 'vector_store_path' in config and not os.path.isabs(config['vector_store_path']):
        config['vector_store_path'] = os.path.join(project_root, config['vector_store_path'])
    
    if 'data_path' in config and not os.path.isabs(config['data_path']):
        config['data_path'] = os.path.join(project_root, config['data_path'])
    
    return config
