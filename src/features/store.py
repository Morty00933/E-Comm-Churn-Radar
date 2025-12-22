"""Lightweight Feature Store with Redis caching."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """Simple feature store with Redis caching.
    
    Features:
    - Cache computed features in Redis
    - TTL-based expiration
    - Batch retrieval
    - Fallback to computation on cache miss
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: int = 3600,  # 1 hour default
        prefix: str = "features",
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix
        self._client = None
    
    @property
    def client(self):
        """Lazy Redis client initialization."""
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self.redis_url, decode_responses=True)
                self._client.ping()
                logger.info(f"Connected to Redis: {self.redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self._client = None
        return self._client
    
    def _make_key(self, user_id: int, feature_set: str = "default") -> str:
        """Generate cache key for user features."""
        return f"{self.prefix}:{feature_set}:{user_id}"
    
    def _serialize(self, features: Dict[str, Any]) -> str:
        """Serialize features to JSON."""
        return json.dumps(features)
    
    def _deserialize(self, data: str) -> Dict[str, Any]:
        """Deserialize features from JSON."""
        return json.loads(data)
    
    def get(
        self,
        user_id: int,
        feature_set: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Get cached features for a user.
        
        Returns None on cache miss or Redis unavailable.
        """
        if self.client is None:
            return None
        
        try:
            key = self._make_key(user_id, feature_set)
            data = self.client.get(key)
            
            if data:
                logger.debug(f"Cache hit: {key}")
                return self._deserialize(data)
            
            logger.debug(f"Cache miss: {key}")
            return None
            
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None
    
    def set(
        self,
        user_id: int,
        features: Dict[str, Any],
        feature_set: str = "default",
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache features for a user.
        
        Returns True on success, False on failure.
        """
        if self.client is None:
            return False
        
        try:
            key = self._make_key(user_id, feature_set)
            data = self._serialize(features)
            ttl = ttl or self.ttl_seconds
            
            self.client.setex(key, ttl, data)
            logger.debug(f"Cached: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
            return False
    
    def get_many(
        self,
        user_ids: List[int],
        feature_set: str = "default",
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        """Get cached features for multiple users.
        
        Returns dict mapping user_id -> features (None for cache miss).
        """
        result = {uid: None for uid in user_ids}
        
        if self.client is None or not user_ids:
            return result
        
        try:
            keys = [self._make_key(uid, feature_set) for uid in user_ids]
            values = self.client.mget(keys)
            
            for uid, value in zip(user_ids, values):
                if value:
                    result[uid] = self._deserialize(value)
            
            hits = sum(1 for v in result.values() if v is not None)
            logger.debug(f"Batch get: {hits}/{len(user_ids)} hits")
            
            return result
            
        except Exception as e:
            logger.warning(f"Redis mget failed: {e}")
            return result
    
    def set_many(
        self,
        features_map: Dict[int, Dict[str, Any]],
        feature_set: str = "default",
        ttl: Optional[int] = None,
    ) -> int:
        """Cache features for multiple users.
        
        Returns number of successfully cached entries.
        """
        if self.client is None or not features_map:
            return 0
        
        try:
            ttl = ttl or self.ttl_seconds
            pipe = self.client.pipeline()
            
            for uid, features in features_map.items():
                key = self._make_key(uid, feature_set)
                data = self._serialize(features)
                pipe.setex(key, ttl, data)
            
            pipe.execute()
            logger.debug(f"Batch set: {len(features_map)} entries")
            return len(features_map)
            
        except Exception as e:
            logger.warning(f"Redis pipeline failed: {e}")
            return 0
    
    def delete(self, user_id: int, feature_set: str = "default") -> bool:
        """Delete cached features for a user."""
        if self.client is None:
            return False
        
        try:
            key = self._make_key(user_id, feature_set)
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
            return False
    
    def invalidate_all(self, feature_set: str = "default") -> int:
        """Invalidate all cached features for a feature set.
        
        Returns number of deleted keys.
        """
        if self.client is None:
            return 0
        
        try:
            pattern = f"{self.prefix}:{feature_set}:*"
            keys = list(self.client.scan_iter(match=pattern))
            
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Invalidated {deleted} keys matching {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.warning(f"Redis invalidate failed: {e}")
            return 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.client is None:
            return {"status": "disconnected"}
        
        try:
            info = self.client.info("memory")
            pattern = f"{self.prefix}:*"
            key_count = sum(1 for _ in self.client.scan_iter(match=pattern))
            
            return {
                "status": "connected",
                "key_count": key_count,
                "used_memory": info.get("used_memory_human", "N/A"),
                "max_memory": info.get("maxmemory_human", "N/A"),
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global feature store instance
_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get global feature store instance."""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store


def get_or_compute_features(
    user_ids: List[int],
    compute_fn,
    feature_set: str = "default",
    ttl: Optional[int] = None,
) -> pd.DataFrame:
    """Get features from cache or compute if missing.
    
    Args:
        user_ids: List of user IDs
        compute_fn: Function(user_ids) -> DataFrame with features
        feature_set: Feature set name
        ttl: Cache TTL in seconds
    
    Returns:
        DataFrame with features for all users
    """
    store = get_feature_store()
    
    # Try cache first
    cached = store.get_many(user_ids, feature_set)
    
    hits = {uid: feat for uid, feat in cached.items() if feat is not None}
    misses = [uid for uid, feat in cached.items() if feat is None]
    
    results = []
    
    # Add cached results
    if hits:
        results.append(pd.DataFrame.from_records(list(hits.values())))
    
    # Compute missing
    if misses:
        computed_df = compute_fn(misses)
        results.append(computed_df)
        
        # Cache computed results
        features_map = {
            int(row["user_id"]): row.to_dict()
            for _, row in computed_df.iterrows()
            if "user_id" in row
        }
        store.set_many(features_map, feature_set, ttl)
    
    if results:
        return pd.concat(results, ignore_index=True)
    
    return pd.DataFrame()
