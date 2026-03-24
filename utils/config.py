"""配置加载工具模块 - v2.0"""
import os
import re
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# 自动加载 .env 文件
load_dotenv()


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载 YAML 配置文件，支持环境变量替换
    
    Args:
        config_path: 配置文件路径，默认为 config/config.yaml
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置验证失败
    """
    if config_path is None:
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, "config", "config.yaml")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    
    # 读取并解析 YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换环境变量 ${VAR} 格式
    content = _replace_env_vars(content)
    
    # 解析 YAML
    config = yaml.safe_load(content)
    
    # 验证必需字段
    _validate_config(config)
    
    # 转换相对路径为绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config = _resolve_paths(config, project_root)
    
    return config


def _replace_env_vars(content: str) -> str:
    """
    替换配置文件中的环境变量
    
    支持格式：${VAR_NAME} 或 ${VAR_NAME:default_value}
    
    Args:
        content: YAML 文件内容
        
    Returns:
        替换后的内容
    """
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
    
    def replacer(match):
        var_name = match.group(1)
        default_value = match.group(2)
        
        value = os.getenv(var_name)
        
        if value is None:
            if default_value is not None:
                return default_value
            else:
                raise ValueError(
                    f"环境变量 {var_name} 未设置，且没有默认值"
                )
        
        return value
    
    return re.sub(pattern, replacer, content)


def _validate_config(config: Dict[str, Any]) -> None:
    """
    验证配置文件的必需字段
    
    Args:
        config: 配置字典
        
    Raises:
        ValueError: 缺少必需字段
    """
    required_fields = [
        'system',
        'embedding',
        'inference',
        'database',
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"配置缺少必需字段：{field}")
    
    # 验证 system 配置
    if 'name' not in config['system'] or 'version' not in config['system']:
        raise ValueError("system 配置缺少 name 或 version 字段")
    
    # 验证 embedding 配置
    if 'model' not in config['embedding'] or 'dimension' not in config['embedding']:
        raise ValueError("embedding 配置缺少 model 或 dimension 字段")
    
    # 验证 inference 配置
    if 'provider' not in config['inference']:
        raise ValueError("inference 配置缺少 provider 字段")
    
    # 验证 database 配置
    if 'type' not in config['database'] or 'path' not in config['database']:
        raise ValueError("database 配置缺少 type 或 path 字段")


def _resolve_paths(config: Dict[str, Any], project_root: str) -> Dict[str, Any]:
    """
    将配置中的相对路径转换为绝对路径
    
    Args:
        config: 配置字典
        project_root: 项目根目录
        
    Returns:
        转换后的配置字典
    """
    # 需要转换路径的字段
    path_fields = [
        ('database', 'path'),
        ('indexing', 'data_dir'),
    ]
    
    for section, field in path_fields:
        if section in config and field in config[section]:
            path = config[section][field]
            if not os.path.isabs(path):
                config[section][field] = os.path.join(project_root, path)
    
    return config


def get_config_value(config: Dict[str, Any], *keys, default: Any = None) -> Any:
    """
    安全地获取嵌套配置值
    
    Args:
        config: 配置字典
        *keys: 键的路径，例如 get_config_value(config, 'inference', 'providers', 'deepseek')
        default: 默认值
        
    Returns:
        配置值或默认值
    """
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


# 全局配置缓存
_config_cache: Optional[Dict[str, Any]] = None


def get_cached_config() -> Dict[str, Any]:
    """
    获取缓存的配置，避免重复加载
    
    Returns:
        配置字典
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def reload_config() -> Dict[str, Any]:
    """
    重新加载配置（用于热更新）
    
    Returns:
        新的配置字典
    """
    global _config_cache
    _config_cache = load_config()
    return _config_cache
