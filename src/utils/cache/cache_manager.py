# cache_manager.py
from dataclasses import dataclass
import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union
from functools import wraps
import hashlib
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目，包含值和过期时间"""
    value: Any
    created_at: float
    ttl: Optional[int] = None       # 过期时间，None默认永不过期

    def is_expired(self) -> bool:
        return self.ttl is not None and self.created_at + self.ttl < time.time()


class CacheBackend(ABC):
    """缓存后端抽象基类"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass


class FileCacheBackend(CacheBackend):
    """基于文件系统的缓存后端"""

    def __init__(self, cache_dir: Union[str, Path]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        cache_file = self._get_cache_path(key)
        if cache_file.exists():     # 缓存文件是否存在
            try:
                with open(cache_file, 'rb') as f:
                    entry: CacheEntry = pickle.load(f)
                    if not entry.is_expired():  # 缓存文件是否过期，如果没过期返回缓存中的值（文件内容）
                        return entry.value
                    else:
                        cache_file.unlink()     # 缓存过期情况下，删除文件或者链接
            except Exception as e:
                logger.warning(f"Failed to load cache {key}: {e}")
                return None
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        cache_file = self._get_cache_path(key)
        try:
            entry = CacheEntry(value, time.time(), ttl)
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {key}: {e}")

    def delete(self, key: str) -> None:
        cache_file = self._get_cache_path(key)
        if cache_file.exists():
            cache_file.unlink()

    def exists(self, key: str) -> bool:
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            return False
        try:
            with open(cache_file, 'rb') as f:
                entry: CacheEntry = pickle.load(f)
                if not entry.is_expired():
                    return True
                else:
                    cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to load cache {key}: {e}")
        return False


class MemoryCacheBackend(CacheBackend):
    """基于内存的缓存后端"""

    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self.max_size = max_size
        self._lock = Lock()     # 线程安全锁

    def _cleanup_expired_entries(self):
        """清理过期的缓存条目"""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]

    def _evict_lru(self):
        """缓存满时移除最不常用的缓存条目"""
        if len(self._cache) > self.max_size:
            oldest_keys = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_keys]

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._cleanup_expired_entries()
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():      # 这里还要检查缓存是否过期？
                    return entry.value
                else:
                    self.delete(key)
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            self._cleanup_expired_entries()
            self._evict_lru()
            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl
            )

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def exists(self, key: str) -> bool:
        with self._lock:
            self._cleanup_expired_entries()
            return key in self._cache and not self._cache[key].is_expired()


def get_cache_key(*args, **kwargs) -> str:
    """生成缓存键，使用pickle处理复杂对象"""
    try:
        # 使用pickle序列化，它可以处理更多类型的对象
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_bytes = pickle.dumps(key_data)
        return hashlib.md5(key_bytes).hexdigest()
    except Exception as e:
        # 如果pickle也失败，回退到简单的字符串哈希
        key_str = f"{hash(str(args))}_{hash(str(kwargs))}"
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()


class CacheManager:
    """通用缓存管理器（装饰器类）"""
    def __init__(self, backend: CacheBackend):
        self.backend = backend

    def cached(self, ttl: Optional[int] = None):
        """缓存装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = get_cache_key(*args, **kwargs)
                if self.backend.exists(cache_key):
                    logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                    return self.backend.get(cache_key)
                result = func(*args, **kwargs)
                self.backend.set(cache_key, result, ttl=ttl)
                return result
            return wrapper
        return decorator

    def get(self, key: str) -> Optional[Any]:
        return self.backend.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        return self.backend.set(key, value, ttl=ttl)

    def delete(self, key: str) -> None:
        return self.backend.delete(key)

    def exists(self, key: str) -> bool:
        return self.backend.exists(key)
