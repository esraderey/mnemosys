"""
MNEME Storage Core: Módulo de almacenamiento seguro
Sistema de almacenamiento con validación y protección contra vulnerabilidades
"""

import os
import io
import time
import logging
import hashlib
import struct
from typing import Any, Dict, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import torch
import numpy as np
import lz4.frame
import xxhash
import json
import sqlite3
import tempfile
import shutil
import weakref
from threading import Lock, RLock
from collections import deque, OrderedDict
import safetensors
from safetensors.torch import save_file, load_file

from .mneme_security_core import SecurityManager, SecurityConfig, create_secure_config

logger = logging.getLogger(__name__)

# --- Enums de Almacenamiento ---

class StorageTier(Enum):
    """Niveles de almacenamiento"""
    MEMORY = "memory"
    SSD = "ssd"
    HDD = "hdd"
    ARCHIVE = "archive"

class CachePolicy(Enum):
    """Políticas de cache"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    LIFO = "lifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

class CompressionLevel(Enum):
    """Niveles de compresión"""
    ULTRA_FAST = 1
    FAST = 3
    BALANCED = 6
    HIGH = 9
    MAXIMUM = 12

# --- Clases de Almacenamiento ---

@dataclass
class StorageConfig:
    """Configuración de almacenamiento"""
    cache_size_mb: int = 1024
    cache_policy: CachePolicy = CachePolicy.ADAPTIVE
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    enable_compression: bool = True
    enable_deduplication: bool = True
    max_file_size_mb: int = 1024
    enable_encryption: bool = True
    storage_path: str = "./mneme_storage"
    backup_enabled: bool = True
    validate_integrity: bool = True

class SecureStorageBackend:
    """Backend de almacenamiento seguro"""
    
    def __init__(self, config: StorageConfig, security_config: SecurityConfig = None):
        self.config = config
        self.security_config = security_config or create_secure_config()
        self.security_manager = SecurityManager(self.security_config)
        self.lock = Lock()
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Inicializar base de datos para metadatos
        self.db_path = self.storage_path / "metadata.db"
        self._init_database()
        
        # Cache seguro
        self.cache = {}
        self.cache_lock = Lock()
        self.access_times = {}
        self.access_counts = {}
    
    def _init_database(self):
        """Inicializar base de datos de metadatos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS storage_metadata (
                        key TEXT PRIMARY KEY,
                        size INTEGER,
                        created_at REAL,
                        accessed_at REAL,
                        access_count INTEGER,
                        checksum TEXT,
                        compressed BOOLEAN,
                        tier TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _validate_key(self, key: str) -> bool:
        """Validar clave de almacenamiento"""
        if not isinstance(key, str):
            return False
        if len(key) > 255:  # Límite de longitud
            return False
        if not key.replace('_', '').replace('-', '').isalnum():
            return False
        return True
    
    def _validate_data(self, data: bytes) -> bool:
        """Validar datos de almacenamiento"""
        if not isinstance(data, bytes):
            return False
        if len(data) > self.config.max_file_size_mb * 1024 * 1024:
            return False
        return True
    
    def _compute_checksum(self, data: bytes) -> str:
        """Calcular checksum de datos"""
        return hashlib.sha256(data).hexdigest()
    
    def _compress_data(self, data: bytes) -> bytes:
        """Comprimir datos de forma segura"""
        if not self.config.enable_compression:
            return data
        
        try:
            compressed = lz4.frame.compress(
                data, 
                compression_level=self.config.compression_level.value
            )
            # Solo usar compresión si es beneficiosa
            if len(compressed) < len(data) * 0.9:
                return compressed
            return data
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Descomprimir datos de forma segura"""
        try:
            return lz4.frame.decompress(data)
        except:
            # Si falla la descompresión, asumir que no está comprimido
            return data
    
    def store(self, key: str, data: bytes, metadata: Dict[str, Any] = None) -> bool:
        """Almacenar datos de forma segura"""
        try:
            with self.lock:
                # Validar entrada
                if not self._validate_key(key):
                    raise ValueError(f"Invalid storage key: {key}")
                
                if not self._validate_data(data):
                    raise ValueError("Invalid data for storage")
                
                # Validar metadatos
                if metadata and not self.security_manager.validate_input(metadata, "metadata"):
                    raise ValueError("Invalid metadata")
                
                # Calcular checksum
                checksum = self._compute_checksum(data)
                
                # Comprimir si es necesario
                compressed_data = self._compress_data(data)
                is_compressed = len(compressed_data) < len(data)
                
                # Determinar nivel de almacenamiento
                tier = self._determine_tier(len(compressed_data))
                
                # Almacenar archivo
                file_path = self.storage_path / f"{key}.dat"
                with open(file_path, 'wb') as f:
                    f.write(compressed_data)
                
                # Actualizar metadatos en base de datos
                current_time = time.time()
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO storage_metadata 
                        (key, size, created_at, accessed_at, access_count, checksum, compressed, tier)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (key, len(compressed_data), current_time, current_time, 0, checksum, is_compressed, tier.value))
                    conn.commit()
                
                # Actualizar cache
                with self.cache_lock:
                    self.cache[key] = compressed_data
                    self.access_times[key] = current_time
                    self.access_counts[key] = 0
                
                logger.debug(f"Stored {key} ({len(compressed_data)} bytes, tier: {tier.value})")
                return True
                
        except Exception as e:
            logger.error(f"Storage failed for key {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[bytes]:
        """Recuperar datos de forma segura"""
        try:
            with self.lock:
                # Verificar cache primero
                with self.cache_lock:
                    if key in self.cache:
                        self.access_counts[key] = self.access_counts.get(key, 0) + 1
                        self.access_times[key] = time.time()
                        return self._decompress_data(self.cache[key])
                
                # Recuperar de almacenamiento
                file_path = self.storage_path / f"{key}.dat"
                if not file_path.exists():
                    return None
                
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Verificar integridad
                if self.config.validate_integrity:
                    if not self._verify_integrity(key, data):
                        logger.warning(f"Integrity check failed for key {key}")
                        return None
                
                # Actualizar metadatos
                current_time = time.time()
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE storage_metadata 
                        SET accessed_at = ?, access_count = access_count + 1
                        WHERE key = ?
                    """, (current_time, key))
                    conn.commit()
                
                # Actualizar cache
                with self.cache_lock:
                    self.cache[key] = data
                    self.access_times[key] = current_time
                    self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
                return self._decompress_data(data)
                
        except Exception as e:
            logger.error(f"Retrieval failed for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Eliminar datos de forma segura"""
        try:
            with self.lock:
                # Eliminar archivo
                file_path = self.storage_path / f"{key}.dat"
                if file_path.exists():
                    file_path.unlink()
                
                # Eliminar de base de datos
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM storage_metadata WHERE key = ?", (key,))
                    conn.commit()
                
                # Eliminar de cache
                with self.cache_lock:
                    self.cache.pop(key, None)
                    self.access_times.pop(key, None)
                    self.access_counts.pop(key, None)
                
                logger.debug(f"Deleted {key}")
                return True
                
        except Exception as e:
            logger.error(f"Deletion failed for key {key}: {e}")
            return False
    
    def _determine_tier(self, size: int) -> StorageTier:
        """Determinar nivel de almacenamiento basado en tamaño"""
        if size < 1024 * 1024:  # < 1MB
            return StorageTier.MEMORY
        elif size < 100 * 1024 * 1024:  # < 100MB
            return StorageTier.SSD
        elif size < 1024 * 1024 * 1024:  # < 1GB
            return StorageTier.HDD
        else:
            return StorageTier.ARCHIVE
    
    def _verify_integrity(self, key: str, data: bytes) -> bool:
        """Verificar integridad de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT checksum FROM storage_metadata WHERE key = ?", (key,)
                )
                result = cursor.fetchone()
                if not result:
                    return False
                
                stored_checksum = result[0]
                current_checksum = self._compute_checksum(data)
                return stored_checksum == current_checksum
                
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de almacenamiento"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_keys,
                        SUM(size) as total_size,
                        AVG(access_count) as avg_access_count,
                        COUNT(CASE WHEN compressed = 1 THEN 1 END) as compressed_count
                    FROM storage_metadata
                """)
                result = cursor.fetchone()
                
                return {
                    "total_keys": result[0] or 0,
                    "total_size_bytes": result[1] or 0,
                    "average_access_count": result[2] or 0,
                    "compressed_count": result[3] or 0,
                    "cache_size": len(self.cache),
                    "security_violations": self.security_manager.security_violations
                }
                
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {}
    
    def list_keys(self) -> List[str]:
        """Listar todas las claves almacenadas"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT key FROM storage_metadata")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return []
    
    def cleanup(self):
        """Limpiar recursos"""
        try:
            with self.cache_lock:
                self.cache.clear()
                self.access_times.clear()
                self.access_counts.clear()
            
            logger.info("Storage backend cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

class SecureCache:
    """Cache seguro con validación"""
    
    def __init__(self, max_size_bytes: int, policy: CachePolicy = CachePolicy.ADAPTIVE):
        self.max_size_bytes = max_size_bytes
        self.policy = policy
        self.cache = {}
        self.lock = Lock()
        self.current_size = 0
        self.access_times = {}
        self.access_counts = {}
        self.creation_times = {}
    
    def _estimate_size(self, value: Any) -> int:
        """Estimar tamaño de un valor"""
        if isinstance(value, bytes):
            return len(value)
        elif isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        elif isinstance(value, dict):
            return len(str(value))
        else:
            return len(str(value))
    
    def _evict_adaptive(self):
        """Evictar elemento usando estrategia adaptativa"""
        if not self.cache:
            return
        
        current_time = time.time()
        
        # Calcular scores para cada elemento
        scores = {}
        for key in self.cache.keys():
            access_time = self.access_times.get(key, self.creation_times.get(key, current_time))
            access_count = self.access_counts.get(key, 0)
            age = current_time - access_time
            
            # Score basado en frecuencia, recencia y tamaño
            freq_score = access_count / max(age, 1)
            recency_score = 1.0 / (age + 1)
            size_penalty = 1.0 / (self._estimate_size(self.cache[key]) + 1)
            
            scores[key] = freq_score * recency_score * size_penalty
        
        # Evictar elemento con menor score
        if scores:
            evict_key = min(scores.keys(), key=lambda k: scores[k])
            self._evict_key(evict_key)
    
    def _evict_key(self, key: str):
        """Evictar clave específica"""
        if key in self.cache:
            self.current_size -= self._estimate_size(self.cache[key])
            del self.cache[key]
            self.access_times.pop(key, None)
            self.access_counts.pop(key, None)
            self.creation_times.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener elemento del cache"""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Actualizar métricas de acceso
            current_time = time.time()
            self.access_times[key] = current_time
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> bool:
        """Almacenar elemento en cache"""
        with self.lock:
            value_size = self._estimate_size(value)
            
            # Verificar si cabe
            if value_size > self.max_size_bytes:
                return False
            
            # Evictar elementos si es necesario
            while self.current_size + value_size > self.max_size_bytes and self.cache:
                self._evict_adaptive()
            
            # Almacenar
            if key in self.cache:
                self.current_size -= self._estimate_size(self.cache[key])
            
            self.cache[key] = value
            self.current_size += value_size
            
            # Actualizar métricas
            current_time = time.time()
            self.access_times[key] = current_time
            self.access_counts[key] = 1
            self.creation_times[key] = current_time
            
            return True
    
    def delete(self, key: str) -> bool:
        """Eliminar elemento del cache"""
        with self.lock:
            if key in self.cache:
                self._evict_key(key)
                return True
            return False
    
    def clear(self):
        """Limpiar cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.creation_times.clear()
            self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        with self.lock:
            total_requests = sum(self.access_counts.values())
            hit_rate = (len(self.cache) / max(total_requests, 1)) * 100
            
            return {
                "size_bytes": self.current_size,
                "max_size_bytes": self.max_size_bytes,
                "usage_percent": (self.current_size / self.max_size_bytes) * 100,
                "entries": len(self.cache),
                "hit_rate": hit_rate,
                "policy": self.policy.value
            }

# --- Funciones de Utilidad ---

def create_secure_storage(config: StorageConfig = None) -> SecureStorageBackend:
    """Crear backend de almacenamiento seguro"""
    if config is None:
        config = StorageConfig()
    
    security_config = create_secure_config()
    return SecureStorageBackend(config, security_config)

def create_secure_cache(max_size_mb: int = 1024, policy: CachePolicy = CachePolicy.ADAPTIVE) -> SecureCache:
    """Crear cache seguro"""
    return SecureCache(max_size_mb * 1024 * 1024, policy)
