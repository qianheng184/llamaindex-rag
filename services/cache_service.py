"""缓存服务模块 - v2.0"""
import time
import logging
from typing import Dict, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class CacheService:
    """内存缓存服务
    
    支持 TTL 过期机制的简单内存缓存
    """
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        """
        初始化缓存服务
        
        Args:
            ttl: 缓存生存时间（秒）
            max_size: 最大缓存条目数
        """
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, tuple] = {}
        self._lock = Lock()
        
        logger.info(f"缓存服务初始化完成：ttl={ttl}s, max_size={max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        从缓存中获取值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或已过期则返回 None
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            value, expiry = self._cache[key]
            
            # 检查是否过期
            if time.time() > expiry:
                del self._cache[key]
                logger.debug(f"缓存键 {key} 已过期")
                return None
            
            logger.debug(f"缓存命中：{key}")
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒），覆盖默认值
        """
        with self._lock:
            # 检查是否需要清理旧缓存
            if len(self._cache) >= self.max_size:
                self._cleanup_oldest(10)
            
            # 计算过期时间
            effective_ttl = ttl if ttl is not None else self.ttl
            expiry = time.time() + effective_ttl
            
            self._cache[key] = (value, expiry)
            logger.debug(f"缓存已设置：{key}, ttl={effective_ttl}s")
    
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            True 如果删除成功
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"缓存已删除：{key}")
                return True
            return False
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            logger.info("缓存已清空")
    
    def cleanup_expired(self) -> int:
        """
        清理所有过期的缓存
        
        Returns:
            清理的缓存数量
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if current_time > expiry
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info(f"清理了 {len(expired_keys)} 个过期缓存")
            
            return len(expired_keys)
    
    def _cleanup_oldest(self, count: int) -> None:
        """
        清理最旧的缓存条目
        
        Args:
            count: 要清理的数量
        """
        if not self._cache:
            return
        
        # 按过期时间排序
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: x[1][1]  # 按过期时间排序
        )
        
        # 删除最旧的
        for i in range(min(count, len(sorted_items))):
            key = sorted_items[i][0]
            del self._cache[key]
        
        logger.debug(f"清理了 {count} 个最旧的缓存")
    
    def stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for _, expiry in self._cache.values()
                if current_time > expiry
            )
            
            return {
                "total_entries": len(self._cache),
                "expired_entries": expired_count,
                "active_entries": len(self._cache) - expired_count,
                "max_size": self.max_size,
                "usage_percent": (len(self._cache) / self.max_size * 100) if self.max_size > 0 else 0
            }
    
    def is_available(self) -> bool:
        """检查缓存服务是否可用"""
        return True
