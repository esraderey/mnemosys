"""
MNEME Core: Motor de Memoria Neural Mórfica (Versión Segura)
Sistema avanzado de memoria computacional con síntesis determinista, verificación criptográfica robusta, 
aceleración de hardware y optimizaciones de rendimiento - SIN VULNERABILIDADES DE PICKLE
"""

import hashlib
import struct
import numpy as np
import torch
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from collections import deque, OrderedDict
from threading import Lock, RLock
from threading import local as ThreadLocal
import threading
from contextlib import contextmanager
import time
import lz4.frame
import tensorly as tl
from tensorly.decomposition import parafac, tucker, tensor_train
import xxhash
import io
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
import secrets
import hmac
import json
import logging
import gc
import psutil
import warnings
import msgpack
import zlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator
import threading
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import weakref
from functools import lru_cache, wraps
import sqlite3
import shutil
import tempfile
import sys
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file, load_file

# Importar módulos de seguridad y almacenamiento
from .mneme_security_core import (
    SecurityManager, SecurityConfig, create_secure_config,
    secure_tensor_serialize, secure_tensor_deserialize,
    validate_tensor_safe
)
from .mneme_storage_core import (
    SecureStorageBackend, SecureCache, StorageConfig,
    create_secure_storage, create_secure_cache
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar backend de TensorLy
try:
    tl.set_backend('pytorch')
except Exception as e:
    warnings.warn(f"Could not set TensorLy backend to PyTorch: {e}")

# --- Enums y Clases de Error ---

class LockType(Enum):
    """Tipos de locks granulares"""
    READ = "read"
    WRITE = "write"
    CACHE = "cache"
    STORAGE = "storage"
    SECURITY = "security"
    COMPRESSION = "compression"

class DecompType(Enum):
    TT = "tt"
    CP = "cp"
    TUCKER = "tucker"
    SVD = "svd"
    RAW = "raw"
    SPARSE = "sparse"
    QUANTIZED = "quantized"
    ADAPTIVE = "adaptive"

class CompressionLevel(Enum):
    ULTRA_FAST = 1
    FAST = 3
    BALANCED = 6
    HIGH = 9
    MAXIMUM = 12

class SerializationFormat(Enum):
    """Formatos de serialización soportados"""
    SAFETENSORS = "safetensors"  # Formato seguro por defecto
    TORCH = "torch"
    MSGPACK = "msgpack"
    JSON = "json"
    BINARY = "binary"
    HYBRID = "hybrid"

class SecurityLevel(Enum):
    """Niveles de seguridad para serialización"""
    NONE = "none"
    HMAC = "hmac"
    ENCRYPTED = "encrypted"
    SIGNED = "signed"
    SAFETENSORS = "safetensors"

class TensorEncryptionMode(Enum):
    AES_GCM = "aes_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    QUANTUM_SAFE = "quantum_safe"

class KeyRotationPolicy(Enum):
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    MANUAL = "manual"
    ADAPTIVE = "adaptive"

class StorageBackend(Enum):
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    HYBRID = "hybrid"

class CachePolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    LIFO = "lifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

class CompressionStrategy(Enum):
    LZ4 = "lz4"
    ZLIB = "zlib"
    LZMA = "lzma"
    ADAPTIVE = "adaptive"

class ContextSimilarityMethod(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    PEARSON = "pearson"

class ContextClusteringMethod(Enum):
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    SPECTRAL = "spectral"

# --- Clases de Error ---

class MnemeError(Exception):
    """Error base de MNEME"""
    pass

class SecurityError(MnemeError):
    """Error de seguridad"""
    pass

class ValidationError(MnemeError):
    """Error de validación"""
    pass

class StorageError(MnemeError):
    """Error de almacenamiento"""
    pass

# --- Clases de Configuración ---

@dataclass
class MnemeConfig:
    """Configuración principal de MNEME"""
    # Configuración básica
    cache_size_mb: int = 1024
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    security_level: SecurityLevel = SecurityLevel.SAFETENSORS
    serialization_format: SerializationFormat = SerializationFormat.SAFETENSORS
    
    # Configuración de seguridad
    secret_key: Optional[bytes] = None
    enable_encryption: bool = True
    enable_merkle: bool = True
    audit_log_file: Optional[str] = None
    
    # Configuración de almacenamiento
    storage_backend: StorageBackend = StorageBackend.HYBRID
    cache_policy: CachePolicy = CachePolicy.ADAPTIVE
    enable_distributed_cache: bool = False
    redis_url: Optional[str] = None
    
    # Configuración de GPU
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Configuración de paralelización
    max_workers: int = 4
    enable_parallel_processing: bool = True
    
    # Configuración de validación
    validate_inputs: bool = True
    max_tensor_size_mb: int = 1024
    max_batch_size: int = 1000

# --- Clases de Locks Granulares ---

class GranularLockManager:
    """Gestor de locks granulares para reemplazar RLock global"""
    
    def __init__(self):
        self._locks: Dict[str, RLock] = {}
        self._read_locks: Dict[str, int] = {}
        self._write_locks: Dict[str, bool] = {}
        self._lock_manager = Lock()
    
    def _get_lock(self, resource: str, lock_type: LockType) -> RLock:
        """Obtener o crear lock para un recurso específico"""
        with self._lock_manager:
            lock_key = f"{resource}_{lock_type.value}"
            if lock_key not in self._locks:
                self._locks[lock_key] = RLock()
            return self._locks[lock_key]
    
    @contextmanager
    def acquire_lock(self, resource: str, lock_type: LockType, timeout: float = 30.0):
        """Context manager para adquirir locks granulares"""
        lock = self._get_lock(resource, lock_type)
        acquired = False
        try:
            acquired = lock.acquire(timeout=timeout)
            if not acquired:
                raise TimeoutError(f"Could not acquire {lock_type.value} lock for {resource} within {timeout}s")
            yield lock
        finally:
            if acquired:
                lock.release()
    
    def get_lock_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de locks"""
        with self._lock_manager:
            return {
                "total_locks": len(self._locks),
                "active_readers": sum(self._read_locks.values()),
                "active_writers": sum(self._write_locks.values()),
                "lock_types": {lock_type.value: sum(1 for key in self._locks.keys() if lock_type.value in key) 
                             for lock_type in LockType}
            }

# --- Clases de Lazy Decompression ---

class LazyTensor:
    """Tensor con decompresión lazy para optimizar memoria"""
    
    def __init__(self, compressed_data: bytes, decompression_func: Callable, 
                 metadata: Dict[str, Any], device: torch.device = None):
        self.compressed_data = compressed_data
        self.decompression_func = decompression_func
        self.metadata = metadata
        self.device = device or torch.device('cpu')
        self._decompressed_tensor: Optional[torch.Tensor] = None
        self._lock = Lock()
    
    def decompress(self) -> torch.Tensor:
        """Decomprimir tensor solo cuando sea necesario"""
        with self._lock:
            if self._decompressed_tensor is None:
                self._decompressed_tensor = self.decompression_func(self.compressed_data)
                if self.device != torch.device('cpu'):
                    self._decompressed_tensor = self._decompressed_tensor.to(self.device)
            return self._decompressed_tensor
    
    def is_decompressed(self) -> bool:
        """Verificar si el tensor ya está decompressed"""
        return self._decompressed_tensor is not None
    
    def clear_decompressed(self):
        """Liberar memoria del tensor decompressed"""
        with self._lock:
            self._decompressed_tensor = None
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Obtener uso de memoria"""
        compressed_size = len(self.compressed_data)
        decompressed_size = 0
        if self._decompressed_tensor is not None:
            decompressed_size = self._decompressed_tensor.numel() * self._decompressed_tensor.element_size()
        
        return {
            "compressed_bytes": compressed_size,
            "decompressed_bytes": decompressed_size,
            "compression_ratio": compressed_size / max(decompressed_size, 1),
            "is_decompressed": self.is_decompressed()
        }

# --- Clases de Cache Adaptativo ---

class AdaptiveCache:
    """Cache adaptativo que reemplaza LRU con estrategias inteligentes"""
    
    def __init__(self, max_size_bytes: int, strategy: str = "adaptive"):
        self.max_size_bytes = max_size_bytes
        self.strategy = strategy
        self.current_size = 0
        
        self._lru_order = deque()
        self._lfu_counts = {}
        self._access_patterns = {}
        self._temporal_weights = {}
        
        self._cache = {}
        self._lock = Lock()
        self._access_times = {}
        self._access_frequencies = {}
        
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
    
    def _calculate_access_score(self, key: str, current_time: float) -> float:
        """Calcular score de acceso basado en múltiples factores"""
        if key not in self._access_times:
            return 0.0
        
        time_factor = 1.0 / (current_time - self._access_times[key] + 1)
        freq_factor = self._access_frequencies.get(key, 0)
        size_factor = 1.0 / max(len(str(self._cache.get(key, ""))), 1)
        
        compression_factor = 1.0
        if hasattr(self._cache.get(key), 'compressed_data'):
            compression_factor = 2.0
        
        return time_factor * freq_factor * size_factor * compression_factor
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener elemento del cache"""
        with self._lock:
            if key not in self._cache:
                self._miss_count += 1
                return None
            
            current_time = time.time()
            self._access_times[key] = current_time
            self._access_frequencies[key] = self._access_frequencies.get(key, 0) + 1
            
            if key in self._lru_order:
                self._lru_order.remove(key)
            self._lru_order.append(key)
            
            self._hit_count += 1
            return self._cache[key]
    
    def put(self, key: str, value: Any) -> bool:
        """Almacenar elemento en cache"""
        with self._lock:
            value_size = self._estimate_size(value)
            
            if value_size > self.max_size_bytes:
                return False
            
            while self.current_size + value_size > self.max_size_bytes and self._cache:
                self._evict_adaptive()
            
            if key in self._cache:
                self.current_size -= self._estimate_size(self._cache[key])
            
            self._cache[key] = value
            self.current_size += value_size
            
            current_time = time.time()
            self._access_times[key] = current_time
            self._access_frequencies[key] = 1
            
            if key in self._lru_order:
                self._lru_order.remove(key)
            self._lru_order.append(key)
            
            return True
    
    def _evict_adaptive(self):
        """Evictar elemento usando estrategia adaptativa"""
        if not self._cache:
            return
        
        current_time = time.time()
        
        if self.strategy == "adaptive":
            scores = {}
            for key in self._cache.keys():
                scores[key] = self._calculate_access_score(key, current_time)
            evict_key = min(scores.keys(), key=lambda k: scores[k])
        elif self.strategy == "lru":
            evict_key = self._lru_order[0]
        elif self.strategy == "lfu":
            evict_key = min(self._access_frequencies.keys(), 
                           key=lambda k: self._access_frequencies[k])
        else:
            evict_key = next(iter(self._cache.keys()))
        
        self._evict_key(evict_key)
    
    def _evict_key(self, key: str):
        """Evictar clave específica"""
        if key in self._cache:
            self.current_size -= self._estimate_size(self._cache[key])
            del self._cache[key]
            
            if key in self._access_times:
                del self._access_times[key]
            if key in self._access_frequencies:
                del self._access_frequencies[key]
            if key in self._lru_order:
                self._lru_order.remove(key)
            
            self._eviction_count += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimar tamaño de un valor"""
        if isinstance(value, (str, bytes)):
            return len(value)
        elif isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        elif hasattr(value, 'compressed_data'):
            return len(value.compressed_data)
        else:
            return len(str(value))
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        total_requests = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / max(total_requests, 1)) * 100
        
        return {
            "size_bytes": self.current_size,
            "max_size_bytes": self.max_size_bytes,
            "usage_percent": (self.current_size / self.max_size_bytes) * 100,
            "entries": len(self._cache),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self._eviction_count,
            "strategy": self.strategy
        }
    
    def clear(self):
        """Limpiar cache"""
        with self._lock:
            self._cache.clear()
            self._lru_order.clear()
            self._access_times.clear()
            self._access_frequencies.clear()
            self.current_size = 0

# --- Clases de Descriptores ---

@dataclass
class ZDescriptor:
    """Descriptor de tensor con validación de seguridad"""
    kind: str
    decomp_type: DecompType
    shape: Tuple[int, ...]
    ranks: Optional[Tuple[int, ...]] = None
    core_data: bytes = b""
    version: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    merkle_root: Optional[bytes] = None
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    delta_chain: Optional[bytes] = None
    lazy_tensor: Optional[LazyTensor] = None
    
    def __post_init__(self):
        """Validar descriptor después de la inicialización"""
        if not isinstance(self.core_data, bytes):
            raise ValidationError("core_data must be bytes")
        
        if len(self.core_data) == 0:
            raise ValidationError("core_data cannot be empty")
        
        if not isinstance(self.shape, tuple) or len(self.shape) == 0:
            raise ValidationError("shape must be a non-empty tuple")
        
        if any(s <= 0 for s in self.shape):
            raise ValidationError("shape dimensions must be positive")
    
    def verify_integrity(self) -> bool:
        """Verificar integridad del descriptor"""
        try:
            # Verificar que core_data no esté vacío
            if not self.core_data:
                return False
            
            # Verificar que shape sea válido
            if not self.shape or any(s <= 0 for s in self.shape):
                return False
            
            # Verificar merkle root si está presente
            if self.merkle_root:
                computed_root = self._compute_merkle_root()
                if computed_root != self.merkle_root:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False
    
    def _compute_merkle_root(self) -> bytes:
        """Calcular raíz de Merkle"""
        return hashlib.sha256(self.core_data).digest()

@dataclass
class ZAddr:
    """Dirección de descriptor con validación"""
    addr: bytes
    
    def __post_init__(self):
        """Validar dirección"""
        if not isinstance(self.addr, bytes):
            raise ValidationError("addr must be bytes")
        
        if len(self.addr) != 32:  # SHA-256
            raise ValidationError("addr must be 32 bytes")
    
    @classmethod
    def compute(cls, desc: ZDescriptor) -> 'ZAddr':
        """Calcular dirección de descriptor"""
        # Crear hash determinista del descriptor
        data = f"{desc.kind}:{desc.decomp_type.value}:{desc.shape}:{desc.version}".encode()
        addr = hashlib.sha256(data).digest()
        return cls(addr)
    
    def hex(self) -> str:
        """Obtener representación hexadecimal"""
        return self.addr.hex()

# --- Clase Principal ZSpace ---

class ZSpace:
    """Interfaz principal del runtime MNEME con seguridad mejorada"""
    
    def __init__(self, config: Optional[MnemeConfig] = None):
        self.config = config or MnemeConfig()
        
        # Configuración de dispositivo
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.config.use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Sistema de locks granulares
        self.lock_manager = GranularLockManager()
        
        # Configuración de seguridad
        if self.config.secret_key:
            if len(self.config.secret_key) < 32:
                raise ValueError("Secret key must be >= 32 bytes.")
        else:
            logger.warning("No secret key provided. Generating a transient secure key.")
            self.config = replace(self.config, secret_key=secrets.token_bytes(32))
        
        # Tablas de mapeo
        self.name_to_desc: Dict[str, ZDescriptor] = {}
        self.addr_to_desc: Dict[bytes, ZDescriptor] = {}
        self.version_graph: Dict[bytes, bytes] = {}
        
        # Cache adaptativo
        self.adaptive_cache = AdaptiveCache(
            max_size_bytes=self.config.cache_size_mb * 1024 * 1024,
            strategy="adaptive"
        )
        
        # Sistema de seguridad
        self.security_config = create_secure_config()
        self.security_manager = SecurityManager(self.security_config)
        
        # Sistema de almacenamiento seguro
        self.storage_config = StorageConfig(
            cache_size_mb=self.config.cache_size_mb,
            storage_path="./mneme_storage"
        )
        self.storage_backend = create_secure_storage(self.storage_config)
        
        # Métricas
        self.storage_metrics = {
            "read_operations": 0,
            "write_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_ratio": 0.0,
            "total_storage_bytes": 0
        }
        
        logger.info(f"ZSpace initialized with device: {self.device}")
        logger.info(f"Security level: {self.config.security_level.name}")
        logger.info(f"Using safetensors for secure serialization")
    
    def register(self, name: str, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """Registrar tensor con validación de seguridad"""
        # Validar entrada
        if not self.security_manager.validate_input(tensor, "tensor"):
            raise ValidationError("Tensor validation failed")
        
        # Usar lock granular para escritura
        with self.lock_manager.acquire_lock(name, LockType.WRITE):
            # Crear descriptor seguro
            desc = self._create_secure_descriptor(tensor, **kwargs)
            
            old_addr = None
            if name in self.name_to_desc:
                old_addr = ZAddr.compute(self.name_to_desc[name])
            
            addr = self._register_descriptor(name, desc, old_addr)
            
            # Almacenar en cache adaptativo
            self.adaptive_cache.put(f"desc_{name}", desc)
            
        logger.info(f"Registered '{name}'. Type: {desc.decomp_type.value}. Addr: {addr.hex()[:8]}")
        return desc
    
    def load(self, name: str) -> torch.Tensor:
        """Cargar tensor con lazy decompression"""
        # Usar lock granular para lectura
        with self.lock_manager.acquire_lock(name, LockType.READ):
            if name not in self.name_to_desc:
                raise KeyError(f"Unknown tensor: {name}")
            
            # Intentar obtener desde cache adaptativo
            cached_desc = self.adaptive_cache.get(f"desc_{name}")
            if cached_desc:
                desc = cached_desc
            else:
                desc = self.name_to_desc[name]
                self.adaptive_cache.put(f"desc_{name}", desc)
        
        # Usar lazy decompression si está disponible
        if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
            return desc.lazy_tensor.decompress()
        else:
            return self._synthesize_tensor(desc)
    
    def _create_secure_descriptor(self, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """Crear descriptor seguro"""
        # Serializar tensor de forma segura
        core_data = secure_tensor_serialize(tensor, kwargs.get('metadata', {}))
        
        # Comprimir datos
        compressed_data = lz4.frame.compress(
            core_data, 
            compression_level=self.config.compression_level.value
        )
        
        # Crear lazy tensor
        lazy_tensor = LazyTensor(
            compressed_data=compressed_data,
            decompression_func=self._decompress_tensor,
            metadata={"shape": tensor.shape, "dtype": str(tensor.dtype)},
            device=self.device
        )
        
        # Crear descriptor
        desc = ZDescriptor(
            kind="tensor",
            decomp_type=DecompType.RAW,  # Simplificado para seguridad
            shape=tuple(tensor.shape),
            core_data=compressed_data,
            version=0,
            meta={
                "compression_ratio": len(compressed_data) / max(len(core_data), 1),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device)
            },
            compression_level=self.config.compression_level,
            lazy_tensor=lazy_tensor
        )
        
        return desc
    
    def _decompress_tensor(self, compressed_data: bytes) -> torch.Tensor:
        """Decomprimir tensor de forma segura"""
        try:
            # Decomprimir datos
            core_data = lz4.frame.decompress(compressed_data)
            
            # Deserializar con safetensors
            tensor, metadata = secure_tensor_deserialize(core_data, self.device)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Tensor decompression failed: {e}")
            raise
    
    def _synthesize_tensor(self, desc: ZDescriptor) -> torch.Tensor:
        """Sintetizar tensor desde descriptor"""
        try:
            # Verificar integridad
            if not desc.verify_integrity():
                raise SecurityError("Descriptor integrity check failed")
            
            # Decomprimir datos
            core_data = lz4.frame.decompress(desc.core_data)
            
            # Deserializar con safetensors
            tensor, metadata = secure_tensor_deserialize(core_data, self.device)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Tensor synthesis failed: {e}")
            raise
    
    def _register_descriptor(self, name: str, desc: ZDescriptor, old_addr: Optional[ZAddr]) -> ZAddr:
        """Registrar descriptor en las tablas"""
        addr = ZAddr.compute(desc)
        
        self.name_to_desc[name] = desc
        self.addr_to_desc[addr.addr] = desc
        
        if old_addr:
            self.version_graph[addr.addr] = old_addr.addr
        
        return addr
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "device": str(self.device),
            "cache": self.adaptive_cache.get_stats(),
            "storage": self.storage_backend.get_stats(),
            "security": self.security_manager.get_security_stats(),
            "locks": self.lock_manager.get_lock_stats(),
            "metrics": self.storage_metrics
        }
    
    def cleanup(self):
        """Limpiar recursos con locks granulares"""
        logger.info("Cleaning up MNEME ZSpace...")
        
        # Limpiar cache adaptativo
        self.adaptive_cache.clear()
        
        # Limpiar lazy tensors
        for desc in self.name_to_desc.values():
            if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
                desc.lazy_tensor.clear_decompressed()
        
        # Limpiar almacenamiento
        self.storage_backend.cleanup()
        
        # Limpiar sistema
        gc.collect()
        if self.device.type.startswith('cuda'):
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
        
        # Obtener estadísticas de locks
        lock_stats = self.lock_manager.get_lock_stats()
        logger.info(f"Lock statistics: {lock_stats}")
        
        logger.info("Cleanup completed.")

# Alias para compatibilidad
Mneme = ZSpace
