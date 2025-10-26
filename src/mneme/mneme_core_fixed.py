"""
MNEME Core: Motor de Memoria Neural Mórfica (Versión Segura)
Sistema avanzado de memoria computacional con síntesis determinista, verificación criptográfica robusta, 
aceleración de hardware y optimizaciones de rendimiento - SIN VULNERABILIDADES DE PICKLE

Versión: 2.0.0
Autor: MNEME Development Team
Licencia: BSL 1.1
"""

# === IMPORTS ESTÁNDAR ===
import asyncio
import base64
import gc
import hashlib
import hmac
import io
import json
import logging
import multiprocessing as mp
import queue
import secrets
import shutil
import sqlite3
import struct
import sys
import tempfile
import threading
import time
import warnings
import weakref
import zlib
from collections import deque, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, as_completed
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from threading import Lock, RLock, local as ThreadLocal
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional, Tuple, Union

# === IMPORTS DE TERCEROS ===
import lz4.frame
import msgpack
import numpy as np
import psutil
import safetensors
import tensorly as tl
import torch
import xxhash
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tensorly.decomposition import parafac, tensor_train, tucker

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

# === SISTEMA DE ERRORES MEJORADO ===

class MnemeError(Exception):
    """Error base de MNEME con información contextual mejorada"""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "MNEME_ERROR"
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def __str__(self) -> str:
        base_msg = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        return base_msg

class SecurityError(MnemeError):
    """Error de seguridad con detalles específicos"""
    
    def __init__(self, message: str, security_level: str = None, threat_type: str = None, **kwargs):
        context = kwargs.get('context', {})
        if security_level:
            context['security_level'] = security_level
        if threat_type:
            context['threat_type'] = threat_type
        super().__init__(message, "SECURITY_ERROR", context)

class ValidationError(MnemeError):
    """Error de validación con información del campo problemático"""
    
    def __init__(self, message: str, field_name: str = None, expected_type: str = None, **kwargs):
        context = kwargs.get('context', {})
        if field_name:
            context['field_name'] = field_name
        if expected_type:
            context['expected_type'] = expected_type
        super().__init__(message, "VALIDATION_ERROR", context)

class StorageError(MnemeError):
    """Error de almacenamiento con información del backend"""
    
    def __init__(self, message: str, backend: str = None, operation: str = None, **kwargs):
        context = kwargs.get('context', {})
        if backend:
            context['backend'] = backend
        if operation:
            context['operation'] = operation
        super().__init__(message, "STORAGE_ERROR", context)

class CompressionError(MnemeError):
    """Error de compresión/descompresión"""
    
    def __init__(self, message: str, compression_type: str = None, **kwargs):
        context = kwargs.get('context', {})
        if compression_type:
            context['compression_type'] = compression_type
        super().__init__(message, "COMPRESSION_ERROR", context)

class MemoryError(MnemeError):
    """Error de memoria con información de uso"""
    
    def __init__(self, message: str, memory_usage: Dict[str, int] = None, **kwargs):
        context = kwargs.get('context', {})
        if memory_usage:
            context['memory_usage'] = memory_usage
        super().__init__(message, "MEMORY_ERROR", context)

class ConcurrencyError(MnemeError):
    """Error de concurrencia con información del lock"""
    
    def __init__(self, message: str, lock_type: str = None, resource: str = None, **kwargs):
        context = kwargs.get('context', {})
        if lock_type:
            context['lock_type'] = lock_type
        if resource:
            context['resource'] = resource
        super().__init__(message, "CONCURRENCY_ERROR", context)

# --- Clases de Configuración ---

@dataclass
class MnemeConfig:
    """
    Configuración principal de MNEME con validaciones mejoradas
    
    Esta clase maneja toda la configuración del sistema MNEME con validaciones
    automáticas y valores por defecto optimizados.
    """
    
    # === CONFIGURACIÓN BÁSICA ===
    cache_size_mb: int = 1024
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    security_level: SecurityLevel = SecurityLevel.SAFETENSORS
    serialization_format: SerializationFormat = SerializationFormat.SAFETENSORS
    
    # === CONFIGURACIÓN DE SEGURIDAD ===
    secret_key: Optional[bytes] = None
    enable_encryption: bool = True
    enable_merkle: bool = True
    audit_log_file: Optional[str] = None
    key_rotation_policy: KeyRotationPolicy = KeyRotationPolicy.ADAPTIVE
    encryption_mode: TensorEncryptionMode = TensorEncryptionMode.AES_GCM
    
    # === CONFIGURACIÓN DE ALMACENAMIENTO ===
    storage_backend: StorageBackend = StorageBackend.HYBRID
    cache_policy: CachePolicy = CachePolicy.ADAPTIVE
    enable_distributed_cache: bool = False
    redis_url: Optional[str] = None
    storage_path: str = "./mneme_storage"
    enable_compression: bool = True
    compression_strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE
    
    # === CONFIGURACIÓN DE GPU ===
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    gpu_memory_growth: bool = True
    mixed_precision: bool = False
    
    # === CONFIGURACIÓN DE PARALELIZACIÓN ===
    max_workers: int = 4
    enable_parallel_processing: bool = True
    thread_pool_size: int = 8
    process_pool_size: int = 2
    
    # === CONFIGURACIÓN DE VALIDACIÓN ===
    validate_inputs: bool = True
    max_tensor_size_mb: int = 1024
    max_batch_size: int = 1000
    strict_validation: bool = True
    
    # === CONFIGURACIÓN DE RENDIMIENTO ===
    enable_lazy_loading: bool = True
    enable_memory_mapping: bool = True
    enable_async_operations: bool = True
    batch_processing_size: int = 100
    
    # === CONFIGURACIÓN DE MONITOREO ===
    enable_metrics: bool = True
    metrics_interval: float = 60.0
    enable_profiling: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validar configuración después de la inicialización"""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validar todos los parámetros de configuración"""
        # Validar cache_size_mb
        if self.cache_size_mb <= 0:
            raise ValidationError(
                "cache_size_mb must be positive",
                field_name="cache_size_mb",
                expected_type="int > 0"
            )
        
        # Validar gpu_memory_fraction
        if not 0.0 < self.gpu_memory_fraction <= 1.0:
            raise ValidationError(
                "gpu_memory_fraction must be between 0.0 and 1.0",
                field_name="gpu_memory_fraction",
                expected_type="float in range (0.0, 1.0]"
            )
        
        # Validar max_workers
        if self.max_workers <= 0:
            raise ValidationError(
                "max_workers must be positive",
                field_name="max_workers",
                expected_type="int > 0"
            )
        
        # Validar max_tensor_size_mb
        if self.max_tensor_size_mb <= 0:
            raise ValidationError(
                "max_tensor_size_mb must be positive",
                field_name="max_tensor_size_mb",
                expected_type="int > 0"
            )
        
        # Validar max_batch_size
        if self.max_batch_size <= 0:
            raise ValidationError(
                "max_batch_size must be positive",
                field_name="max_batch_size",
                expected_type="int > 0"
            )
        
        # Validar secret_key si está presente
        if self.secret_key is not None and len(self.secret_key) < 32:
            raise ValidationError(
                "secret_key must be at least 32 bytes",
                field_name="secret_key",
                expected_type="bytes with len >= 32"
            )
        
        # Validar redis_url si distributed cache está habilitado
        if self.enable_distributed_cache and not self.redis_url:
            raise ValidationError(
                "redis_url is required when distributed cache is enabled",
                field_name="redis_url",
                expected_type="str (Redis URL)"
            )
        
        # Validar log_level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValidationError(
                f"log_level must be one of {valid_log_levels}",
                field_name="log_level",
                expected_type=f"str in {valid_log_levels}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                result[field_name] = field_value.value
            elif isinstance(field_value, bytes):
                result[field_name] = field_value.hex()
            else:
                result[field_name] = field_value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MnemeConfig':
        """Crear configuración desde diccionario"""
        # Convertir valores de enum de vuelta
        processed_dict = {}
        for key, value in config_dict.items():
            if key == 'compression_level' and isinstance(value, str):
                processed_dict[key] = CompressionLevel(value)
            elif key == 'security_level' and isinstance(value, str):
                processed_dict[key] = SecurityLevel(value)
            elif key == 'serialization_format' and isinstance(value, str):
                processed_dict[key] = SerializationFormat(value)
            elif key == 'storage_backend' and isinstance(value, str):
                processed_dict[key] = StorageBackend(value)
            elif key == 'cache_policy' and isinstance(value, str):
                processed_dict[key] = CachePolicy(value)
            elif key == 'key_rotation_policy' and isinstance(value, str):
                processed_dict[key] = KeyRotationPolicy(value)
            elif key == 'encryption_mode' and isinstance(value, str):
                processed_dict[key] = TensorEncryptionMode(value)
            elif key == 'compression_strategy' and isinstance(value, str):
                processed_dict[key] = CompressionStrategy(value)
            elif key == 'secret_key' and isinstance(value, str):
                processed_dict[key] = bytes.fromhex(value)
            else:
                processed_dict[key] = value
        
        return cls(**processed_dict)

# --- Clases de Locks Granulares ---

class GranularLockManager:
    """
    Gestor de locks granulares mejorado para reemplazar RLock global
    
    Proporciona un sistema de locks más eficiente con:
    - Locks granulares por recurso y tipo
    - Timeout configurables
    - Estadísticas detalladas
    - Prevención de deadlocks
    - Limpieza automática de locks no utilizados
    """
    
    def __init__(self, max_locks: int = 1000, cleanup_interval: float = 300.0):
        self._locks: Dict[str, RLock] = {}
        self._read_locks: Dict[str, int] = {}
        self._write_locks: Dict[str, bool] = {}
        self._lock_manager = Lock()
        self._max_locks = max_locks
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._lock_usage: Dict[str, Dict[str, Any]] = {}
        self._deadlock_detection = True
        self._lock_order: Dict[str, int] = {}
        self._next_order = 0
    
    def _get_lock(self, resource: str, lock_type: LockType) -> RLock:
        """Obtener o crear lock para un recurso específico con limpieza automática"""
        with self._lock_manager:
            # Limpieza periódica de locks no utilizados
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_unused_locks()
                self._last_cleanup = time.time()
            
            lock_key = f"{resource}_{lock_type.value}"
            
            if lock_key not in self._locks:
                # Verificar límite de locks
                if len(self._locks) >= self._max_locks:
                    self._cleanup_unused_locks()
                    if len(self._locks) >= self._max_locks:
                        raise ConcurrencyError(
                            f"Maximum number of locks ({self._max_locks}) exceeded",
                            lock_type=lock_type.value,
                            resource=resource
                        )
                
                self._locks[lock_key] = RLock()
                self._lock_usage[lock_key] = {
                    'created_at': time.time(),
                    'last_used': time.time(),
                    'usage_count': 0,
                    'resource': resource,
                    'lock_type': lock_type.value
                }
                self._lock_order[lock_key] = self._next_order
                self._next_order += 1
            
            # Actualizar estadísticas de uso
            self._lock_usage[lock_key]['last_used'] = time.time()
            self._lock_usage[lock_key]['usage_count'] += 1
            
            return self._locks[lock_key]
    
    def _cleanup_unused_locks(self) -> None:
        """Limpiar locks no utilizados recientemente"""
        current_time = time.time()
        locks_to_remove = []
        
        for lock_key, usage_info in self._lock_usage.items():
            # Remover locks no utilizados en los últimos 10 minutos
            if current_time - usage_info['last_used'] > 600:
                locks_to_remove.append(lock_key)
        
        for lock_key in locks_to_remove:
            if lock_key in self._locks:
                del self._locks[lock_key]
            if lock_key in self._lock_usage:
                del self._lock_usage[lock_key]
            if lock_key in self._lock_order:
                del self._lock_order[lock_key]
            if lock_key in self._read_locks:
                del self._read_locks[lock_key]
            if lock_key in self._write_locks:
                del self._write_locks[lock_key]
    
    def _check_deadlock(self, resource: str, lock_type: LockType) -> bool:
        """Verificar posible deadlock basado en orden de adquisición"""
        if not self._deadlock_detection:
            return False
        
        current_order = self._lock_order.get(f"{resource}_{lock_type.value}", 0)
        
        # Verificar si hay locks con orden mayor ya adquiridos
        for lock_key, order in self._lock_order.items():
            if order > current_order and lock_key in self._locks:
                if self._locks[lock_key].locked():
                    return True
        
        return False
    
    @contextmanager
    def acquire_lock(self, resource: str, lock_type: LockType, timeout: float = 30.0, 
                    priority: int = 0):
        """
        Context manager para adquirir locks granulares con prioridad
        
        Args:
            resource: Nombre del recurso a bloquear
            lock_type: Tipo de lock (READ, WRITE, etc.)
            timeout: Tiempo máximo de espera en segundos
            priority: Prioridad del lock (mayor = más prioridad)
        """
        lock = self._get_lock(resource, lock_type)
        acquired = False
        start_time = time.time()
        
        try:
            # Verificar deadlock potencial
            if self._check_deadlock(resource, lock_type):
                raise ConcurrencyError(
                    f"Potential deadlock detected for {resource}",
                    lock_type=lock_type.value,
                    resource=resource
                )
            
            # Intentar adquirir lock con timeout
            acquired = lock.acquire(timeout=timeout)
            if not acquired:
                elapsed = time.time() - start_time
                raise ConcurrencyError(
                    f"Could not acquire {lock_type.value} lock for {resource} within {timeout}s (elapsed: {elapsed:.2f}s)",
                    lock_type=lock_type.value,
                    resource=resource,
                    context={'timeout': timeout, 'elapsed': elapsed}
                )
            
            # Actualizar contadores de locks activos
            lock_key = f"{resource}_{lock_type.value}"
            if lock_type == LockType.READ:
                self._read_locks[lock_key] = self._read_locks.get(lock_key, 0) + 1
            elif lock_type == LockType.WRITE:
                self._write_locks[lock_key] = True
            
            yield lock
            
        finally:
            if acquired:
                # Actualizar contadores al liberar
                lock_key = f"{resource}_{lock_type.value}"
                if lock_type == LockType.READ:
                    if lock_key in self._read_locks:
                        self._read_locks[lock_key] -= 1
                        if self._read_locks[lock_key] <= 0:
                            del self._read_locks[lock_key]
                elif lock_type == LockType.WRITE:
                    if lock_key in self._write_locks:
                        del self._write_locks[lock_key]
                
                lock.release()
    
    def get_lock_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas detalladas de locks"""
        with self._lock_manager:
            current_time = time.time()
            
            # Calcular locks activos
            active_locks = sum(1 for lock in self._locks.values() if lock._is_owned())
            
            # Calcular uso promedio
            total_usage = sum(info['usage_count'] for info in self._lock_usage.values())
            avg_usage = total_usage / max(len(self._lock_usage), 1)
            
            # Locks más utilizados
            most_used = sorted(
                self._lock_usage.items(),
                key=lambda x: x[1]['usage_count'],
                reverse=True
            )[:5]
            
            return {
                "total_locks": len(self._locks),
                "active_locks": active_locks,
                "active_readers": sum(self._read_locks.values()),
                "active_writers": sum(self._write_locks.values()),
                "lock_types": {
                    lock_type.value: sum(1 for key in self._locks.keys() if lock_type.value in key) 
                    for lock_type in LockType
                },
                "usage_stats": {
                    "total_usage_count": total_usage,
                    "average_usage": avg_usage,
                    "most_used_locks": [
                        {
                            "lock_key": key,
                            "usage_count": info['usage_count'],
                            "last_used": info['last_used'],
                            "resource": info['resource'],
                            "lock_type": info['lock_type']
                        }
                        for key, info in most_used
                    ]
                },
                "cleanup_info": {
                    "last_cleanup": self._last_cleanup,
                    "cleanup_interval": self._cleanup_interval,
                    "max_locks": self._max_locks
                }
            }
    
    def force_cleanup(self) -> int:
        """Forzar limpieza de todos los locks no utilizados"""
        with self._lock_manager:
            initial_count = len(self._locks)
            self._cleanup_unused_locks()
            return initial_count - len(self._locks)
    
    def set_deadlock_detection(self, enabled: bool) -> None:
        """Habilitar/deshabilitar detección de deadlocks"""
        self._deadlock_detection = enabled

# --- Clases de Lazy Decompression ---

class LazyTensor:
    """
    Tensor con decompresión lazy mejorado para optimizar memoria
    
    Características mejoradas:
    - Decompresión bajo demanda
    - Gestión inteligente de memoria
    - Cache de metadatos
    - Compresión adaptativa
    - Monitoreo de uso de memoria
    - Limpieza automática
    """
    
    def __init__(self, compressed_data: bytes, decompression_func: Callable, 
                 metadata: Dict[str, Any], device: torch.device = None,
                 max_memory_mb: int = 512, auto_cleanup: bool = True):
        self.compressed_data = compressed_data
        self.decompression_func = decompression_func
        self.metadata = metadata
        self.device = device or torch.device('cpu')
        self.max_memory_mb = max_memory_mb
        self.auto_cleanup = auto_cleanup
        
        # Estado interno
        self._decompressed_tensor: Optional[torch.Tensor] = None
        self._lock = Lock()
        self._access_count = 0
        self._last_access = time.time()
        self._creation_time = time.time()
        self._memory_usage = 0
        
        # Cache de metadatos calculados
        self._cached_shape = None
        self._cached_dtype = None
        self._cached_size = None
        
        # Configuración de limpieza
        self._cleanup_threshold = 0.8  # 80% del límite de memoria
        self._idle_timeout = 300.0  # 5 minutos de inactividad
    
    def decompress(self, force: bool = False) -> torch.Tensor:
        """
        Decomprimir tensor solo cuando sea necesario
        
        Args:
            force: Forzar decompresión incluso si ya está en memoria
        """
        with self._lock:
            # Verificar si ya está decompressed y no se fuerza
            if not force and self._decompressed_tensor is not None:
                self._update_access_stats()
                return self._decompressed_tensor
            
            # Verificar límites de memoria antes de decompress
            if not self._check_memory_limits():
                if self.auto_cleanup:
                    self._cleanup_if_needed()
                else:
                    raise MemoryError(
                        "Insufficient memory for tensor decompression",
                        memory_usage=self.get_memory_usage()
                    )
            
            try:
                # Decomprimir tensor
                self._decompressed_tensor = self.decompression_func(self.compressed_data)
                
                # Mover a dispositivo correcto
                if self.device != torch.device('cpu'):
                    self._decompressed_tensor = self._decompressed_tensor.to(self.device)
                
                # Actualizar estadísticas
                self._update_access_stats()
                self._memory_usage = self._calculate_tensor_memory()
                
                return self._decompressed_tensor
                
            except Exception as e:
                logger.error(f"Failed to decompress tensor: {e}")
                raise CompressionError(
                    f"Tensor decompression failed: {e}",
                    compression_type="lz4",
                    context={'metadata': self.metadata}
                )
    
    def _check_memory_limits(self) -> bool:
        """Verificar si hay suficiente memoria disponible"""
        if self._decompressed_tensor is not None:
            return True  # Ya está en memoria
        
        # Estimar memoria necesaria
        estimated_memory = self._estimate_decompressed_size()
        max_memory_bytes = self.max_memory_mb * 1024 * 1024
        
        return estimated_memory <= max_memory_bytes
    
    def _estimate_decompressed_size(self) -> int:
        """Estimar tamaño del tensor decompressed"""
        if self._cached_size is not None:
            return self._cached_size
        
        # Calcular desde metadata si está disponible
        if 'shape' in self.metadata and 'dtype' in self.metadata:
            shape = self.metadata['shape']
            dtype_str = self.metadata['dtype']
            
            # Mapear tipos de datos a tamaños
            dtype_sizes = {
                'torch.float32': 4, 'torch.float64': 8,
                'torch.int32': 4, 'torch.int64': 8,
                'torch.uint8': 1, 'torch.int8': 1,
                'torch.float16': 2, 'torch.bfloat16': 2
            }
            
            element_size = dtype_sizes.get(dtype_str, 4)  # Default a 4 bytes
            total_elements = 1
            for dim in shape:
                total_elements *= dim
            
            self._cached_size = total_elements * element_size
            return self._cached_size
        
        # Fallback: estimar basado en ratio de compresión
        compression_ratio = self.metadata.get('compression_ratio', 0.1)
        estimated_size = int(len(self.compressed_data) / max(compression_ratio, 0.01))
        self._cached_size = estimated_size
        return estimated_size
    
    def _calculate_tensor_memory(self) -> int:
        """Calcular memoria real del tensor decompressed"""
        if self._decompressed_tensor is None:
            return 0
        return self._decompressed_tensor.numel() * self._decompressed_tensor.element_size()
    
    def _update_access_stats(self) -> None:
        """Actualizar estadísticas de acceso"""
        self._access_count += 1
        self._last_access = time.time()
    
    def _cleanup_if_needed(self) -> None:
        """Limpiar memoria si es necesario"""
        current_memory = self._calculate_tensor_memory()
        max_memory = self.max_memory_mb * 1024 * 1024
        
        if current_memory > max_memory * self._cleanup_threshold:
            self.clear_decompressed()
            logger.debug(f"Cleaned up tensor due to memory pressure: {current_memory} bytes")
    
    def is_decompressed(self) -> bool:
        """Verificar si el tensor ya está decompressed"""
        return self._decompressed_tensor is not None
    
    def clear_decompressed(self) -> None:
        """Liberar memoria del tensor decompressed"""
        with self._lock:
            if self._decompressed_tensor is not None:
                # Limpiar referencias
                del self._decompressed_tensor
                self._decompressed_tensor = None
                self._memory_usage = 0
                
                # Forzar garbage collection si es necesario
                if self.auto_cleanup:
                    gc.collect()
    
    def get_shape(self) -> Tuple[int, ...]:
        """Obtener forma del tensor sin decompress"""
        if self._cached_shape is not None:
            return self._cached_shape
        
        if 'shape' in self.metadata:
            self._cached_shape = tuple(self.metadata['shape'])
        else:
            # Fallback: decompress temporalmente para obtener forma
            temp_tensor = self.decompress()
            self._cached_shape = tuple(temp_tensor.shape)
            if self.auto_cleanup:
                self.clear_decompressed()
        
        return self._cached_shape
    
    def get_dtype(self) -> str:
        """Obtener tipo de datos del tensor sin decompress"""
        if self._cached_dtype is not None:
            return self._cached_dtype
        
        if 'dtype' in self.metadata:
            self._cached_dtype = self.metadata['dtype']
        else:
            # Fallback: decompress temporalmente para obtener dtype
            temp_tensor = self.decompress()
            self._cached_dtype = str(temp_tensor.dtype)
            if self.auto_cleanup:
                self.clear_decompressed()
        
        return self._cached_dtype
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Obtener uso detallado de memoria"""
        compressed_size = len(self.compressed_data)
        decompressed_size = self._calculate_tensor_memory()
        
        return {
            "compressed_bytes": compressed_size,
            "decompressed_bytes": decompressed_size,
            "compression_ratio": compressed_size / max(decompressed_size, 1) if decompressed_size > 0 else 0,
            "is_decompressed": self.is_decompressed(),
            "access_count": self._access_count,
            "last_access": self._last_access,
            "creation_time": self._creation_time,
            "memory_efficiency": decompressed_size / max(compressed_size, 1) if compressed_size > 0 else 0,
            "estimated_size": self._estimate_decompressed_size(),
            "memory_pressure": decompressed_size / (self.max_memory_mb * 1024 * 1024) if self.max_memory_mb > 0 else 0
        }
    
    def is_idle(self, timeout: float = None) -> bool:
        """Verificar si el tensor ha estado inactivo"""
        if timeout is None:
            timeout = self._idle_timeout
        return time.time() - self._last_access > timeout
    
    def should_cleanup(self) -> bool:
        """Determinar si el tensor debería ser limpiado"""
        if not self.auto_cleanup:
            return False
        
        # Limpiar si está inactivo
        if self.is_idle():
            return True
        
        # Limpiar si excede el límite de memoria
        current_memory = self._calculate_tensor_memory()
        max_memory = self.max_memory_mb * 1024 * 1024
        if current_memory > max_memory * self._cleanup_threshold:
            return True
        
        return False
    
    def __del__(self):
        """Destructor para limpiar recursos"""
        try:
            self.clear_decompressed()
        except:
            pass  # Ignorar errores en destructor

# --- Clases de Cache Adaptativo ---

class AdaptiveCache:
    """
    Cache adaptativo mejorado que reemplaza LRU con estrategias inteligentes
    
    Características:
    - Múltiples estrategias de evicción (LRU, LFU, Adaptive, TTL)
    - Análisis de patrones de acceso
    - Compresión automática de elementos grandes
    - Monitoreo de rendimiento en tiempo real
    - Limpieza automática de elementos expirados
    - Estadísticas detalladas
    """
    
    def __init__(self, max_size_bytes: int, strategy: str = "adaptive", 
                 ttl_seconds: float = 3600.0, compression_threshold: int = 1024):
        self.max_size_bytes = max_size_bytes
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        self.compression_threshold = compression_threshold
        self.current_size = 0
        
        # Estructuras de datos para diferentes estrategias
        self._lru_order = deque()
        self._lfu_counts = {}
        self._access_patterns = {}
        self._temporal_weights = {}
        self._ttl_expiry = {}
        
        # Cache principal
        self._cache = {}
        self._lock = Lock()
        self._access_times = {}
        self._access_frequencies = {}
        self._creation_times = {}
        
        # Métricas de rendimiento
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._compression_count = 0
        self._expiry_count = 0
        
        # Configuración de limpieza
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0  # Limpiar cada minuto
        self._max_entries = max_size_bytes // 1024  # Estimación de entradas máximas
    
    def _calculate_access_score(self, key: str, current_time: float) -> float:
        """Calcular score de acceso basado en múltiples factores mejorados"""
        if key not in self._access_times:
            return 0.0
        
        # Factor temporal (más reciente = mayor score)
        time_since_access = current_time - self._access_times[key]
        time_factor = 1.0 / (time_since_access + 1)
        
        # Factor de frecuencia (más accesos = mayor score)
        freq_factor = self._access_frequencies.get(key, 0)
        
        # Factor de tamaño (elementos más pequeños = mayor score)
        value = self._cache.get(key)
        size_factor = 1.0 / max(self._estimate_size(value), 1)
        
        # Factor de compresión (elementos comprimidos = mayor score)
        compression_factor = 1.0
        if hasattr(value, 'compressed_data'):
            compression_factor = 2.0
        elif hasattr(value, 'is_decompressed') and value.is_decompressed():
            compression_factor = 1.5
        
        # Factor de TTL (elementos que expiran pronto = menor score)
        ttl_factor = 1.0
        if key in self._ttl_expiry:
            time_to_expiry = self._ttl_expiry[key] - current_time
            if time_to_expiry < 0:
                ttl_factor = 0.0  # Ya expirado
            else:
                ttl_factor = min(1.0, time_to_expiry / self.ttl_seconds)
        
        # Factor de creación (elementos más nuevos = mayor score)
        creation_factor = 1.0
        if key in self._creation_times:
            age = current_time - self._creation_times[key]
            creation_factor = 1.0 / (age / 3600.0 + 1)  # Normalizar por horas
        
        # Score combinado con pesos
        score = (time_factor * 0.3 + 
                freq_factor * 0.25 + 
                size_factor * 0.2 + 
                compression_factor * 0.15 + 
                ttl_factor * 0.05 + 
                creation_factor * 0.05)
        
        return score
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener elemento del cache con limpieza automática"""
        with self._lock:
            # Limpieza periódica de elementos expirados
            self._cleanup_if_needed()
            
            if key not in self._cache:
                self._miss_count += 1
                return None
            
            # Verificar si el elemento ha expirado
            if self._is_expired(key):
                self._evict_key(key)
                self._expiry_count += 1
                self._miss_count += 1
                return None
            
            current_time = time.time()
            self._access_times[key] = current_time
            self._access_frequencies[key] = self._access_frequencies.get(key, 0) + 1
            
            # Actualizar orden LRU
            if key in self._lru_order:
                self._lru_order.remove(key)
            self._lru_order.append(key)
            
            # Actualizar contador LFU
            self._lfu_counts[key] = self._lfu_counts.get(key, 0) + 1
            
            self._hit_count += 1
            return self._cache[key]
    
    def put(self, key: str, value: Any, ttl: float = None) -> bool:
        """
        Almacenar elemento en cache con TTL opcional
        
        Args:
            key: Clave del elemento
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos (None = usar TTL por defecto)
        """
        with self._lock:
            # Limpieza periódica
            self._cleanup_if_needed()
            
            # Comprimir valor si es necesario
            compressed_value, value_size = self._maybe_compress_value(value)
            
            if value_size > self.max_size_bytes:
                return False
            
            # Evictar elementos hasta que haya espacio suficiente
            while self.current_size + value_size > self.max_size_bytes and self._cache:
                self._evict_adaptive()
            
            # Remover elemento existente si existe
            if key in self._cache:
                self.current_size -= self._estimate_size(self._cache[key])
                self._evict_key(key)
            
            # Almacenar nuevo elemento
            self._cache[key] = compressed_value
            self.current_size += value_size
            
            current_time = time.time()
            self._access_times[key] = current_time
            self._access_frequencies[key] = 1
            self._creation_times[key] = current_time
            
            # Configurar TTL
            if ttl is None:
                ttl = self.ttl_seconds
            self._ttl_expiry[key] = current_time + ttl
            
            # Actualizar orden LRU
            if key in self._lru_order:
                self._lru_order.remove(key)
            self._lru_order.append(key)
            
            # Inicializar contador LFU
            self._lfu_counts[key] = 1
            
            return True
    
    def _evict_adaptive(self):
        """Evictar elemento usando estrategia adaptativa mejorada"""
        if not self._cache:
            return
        
        current_time = time.time()
        
        if self.strategy == "adaptive":
            scores = {}
            for key in self._cache.keys():
                scores[key] = self._calculate_access_score(key, current_time)
            evict_key = min(scores.keys(), key=lambda k: scores[k])
        elif self.strategy == "lru":
            evict_key = self._lru_order[0] if self._lru_order else next(iter(self._cache.keys()))
        elif self.strategy == "lfu":
            if self._lfu_counts:
                evict_key = min(self._lfu_counts.keys(), 
                               key=lambda k: self._lfu_counts[k])
            else:
                evict_key = next(iter(self._cache.keys()))
        elif self.strategy == "ttl":
            # Evictar elemento que expira más pronto
            if self._ttl_expiry:
                evict_key = min(self._ttl_expiry.keys(), 
                               key=lambda k: self._ttl_expiry[k])
            else:
                evict_key = next(iter(self._cache.keys()))
        else:
            evict_key = next(iter(self._cache.keys()))
        
        self._evict_key(evict_key)
    
    def _maybe_compress_value(self, value: Any) -> Tuple[Any, int]:
        """Comprimir valor si es necesario y calcular tamaño"""
        original_size = self._estimate_size(value)
        
        # Solo comprimir si el valor es lo suficientemente grande
        if original_size < self.compression_threshold:
            return value, original_size
        
        # Intentar comprimir
        try:
            if isinstance(value, (str, bytes)):
                compressed = lz4.frame.compress(value.encode() if isinstance(value, str) else value)
                if len(compressed) < original_size * 0.8:  # Solo si la compresión es efectiva
                    self._compression_count += 1
                    return compressed, len(compressed)
        except Exception as e:
            logger.debug(f"Compression failed for value: {e}")
        
        return value, original_size
    
    def _is_expired(self, key: str) -> bool:
        """Verificar si un elemento ha expirado"""
        if key not in self._ttl_expiry:
            return False
        return time.time() > self._ttl_expiry[key]
    
    def _cleanup_if_needed(self):
        """Limpiar elementos expirados si es necesario"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = current_time
        expired_keys = []
        
        for key, expiry_time in self._ttl_expiry.items():
            if current_time > expiry_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._evict_key(key)
            self._expiry_count += 1
    
    def _evict_key(self, key: str):
        """Evictar clave específica con limpieza completa"""
        if key in self._cache:
            self.current_size -= self._estimate_size(self._cache[key])
            del self._cache[key]
            
            # Limpiar todas las estructuras de datos asociadas
            if key in self._access_times:
                del self._access_times[key]
            if key in self._access_frequencies:
                del self._access_frequencies[key]
            if key in self._creation_times:
                del self._creation_times[key]
            if key in self._ttl_expiry:
                del self._ttl_expiry[key]
            if key in self._lfu_counts:
                del self._lfu_counts[key]
            if key in self._lru_order:
                self._lru_order.remove(key)
            
            self._eviction_count += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimar tamaño de un valor con mayor precisión"""
        if isinstance(value, (str, bytes)):
            return len(value)
        elif isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        elif hasattr(value, 'compressed_data'):
            return len(value.compressed_data)
        elif hasattr(value, 'get_memory_usage'):
            # Para LazyTensor
            usage = value.get_memory_usage()
            return usage.get('compressed_bytes', 0)
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
        else:
            return len(str(value))
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas detalladas del cache"""
        total_requests = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / max(total_requests, 1)) * 100
        
        # Calcular estadísticas de TTL
        current_time = time.time()
        expired_count = sum(1 for expiry in self._ttl_expiry.values() if current_time > expiry)
        
        # Calcular estadísticas de compresión
        compression_ratio = 0.0
        if self._compression_count > 0:
            compression_ratio = self._compression_count / max(len(self._cache), 1)
        
        return {
            "size_bytes": self.current_size,
            "max_size_bytes": self.max_size_bytes,
            "usage_percent": (self.current_size / self.max_size_bytes) * 100,
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self._eviction_count,
            "expiry_count": self._expiry_count,
            "compression_count": self._compression_count,
            "compression_ratio": compression_ratio,
            "strategy": self.strategy,
            "ttl_seconds": self.ttl_seconds,
            "expired_entries": expired_count,
            "cleanup_info": {
                "last_cleanup": self._last_cleanup,
                "cleanup_interval": self._cleanup_interval
            },
            "access_patterns": {
                "avg_access_frequency": sum(self._access_frequencies.values()) / max(len(self._access_frequencies), 1),
                "most_accessed": sorted(
                    self._access_frequencies.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
        }
    
    def clear(self):
        """Limpiar cache completamente"""
        with self._lock:
            self._cache.clear()
            self._lru_order.clear()
            self._access_times.clear()
            self._access_frequencies.clear()
            self._creation_times.clear()
            self._ttl_expiry.clear()
            self._lfu_counts.clear()
            self.current_size = 0
            self._last_cleanup = time.time()
    
    def force_cleanup(self) -> int:
        """Forzar limpieza de elementos expirados"""
        with self._lock:
            initial_count = len(self._cache)
            self._cleanup_if_needed()
            return initial_count - len(self._cache)
    
    def set_strategy(self, strategy: str):
        """Cambiar estrategia de evicción"""
        valid_strategies = ["adaptive", "lru", "lfu", "ttl"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        self.strategy = strategy
    
    def set_ttl(self, ttl_seconds: float):
        """Cambiar TTL por defecto"""
        if ttl_seconds <= 0:
            raise ValueError("TTL must be positive")
        self.ttl_seconds = ttl_seconds

# --- Clases de Descriptores ---

@dataclass
class ZDescriptor:
    """
    Descriptor de tensor con validación de seguridad mejorada
    
    Representa un tensor comprimido y serializado con metadatos de seguridad,
    compresión y verificación de integridad.
    """
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
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    security_hash: Optional[bytes] = None
    
    def __post_init__(self):
        """Validar descriptor después de la inicialización"""
        if not isinstance(self.core_data, bytes):
            raise ValidationError(
                "core_data must be bytes",
                field_name="core_data",
                expected_type="bytes"
            )
        
        if len(self.core_data) == 0:
            raise ValidationError(
                "core_data cannot be empty",
                field_name="core_data",
                expected_type="non-empty bytes"
            )
        
        if not isinstance(self.shape, tuple) or len(self.shape) == 0:
            raise ValidationError(
                "shape must be a non-empty tuple",
                field_name="shape",
                expected_type="Tuple[int, ...]"
            )
        
        if any(s <= 0 for s in self.shape):
            raise ValidationError(
                "shape dimensions must be positive",
                field_name="shape",
                expected_type="Tuple[int, ...] with positive values"
            )
        
        # Validar ranks si está presente
        if self.ranks is not None:
            if not isinstance(self.ranks, tuple):
                raise ValidationError(
                    "ranks must be a tuple",
                    field_name="ranks",
                    expected_type="Tuple[int, ...]"
                )
            if any(r <= 0 for r in self.ranks):
                raise ValidationError(
                    "rank values must be positive",
                    field_name="ranks",
                    expected_type="Tuple[int, ...] with positive values"
                )
        
        # Calcular security hash si no está presente
        if self.security_hash is None:
            self.security_hash = self._compute_security_hash()
    
    def verify_integrity(self) -> bool:
        """
        Verificar integridad del descriptor con validaciones mejoradas
        
        Returns:
            bool: True si el descriptor es válido, False en caso contrario
        """
        try:
            # Verificar que core_data no esté vacío
            if not self.core_data:
                logger.warning("Descriptor has empty core_data")
                return False
            
            # Verificar que shape sea válido
            if not self.shape or any(s <= 0 for s in self.shape):
                logger.warning(f"Descriptor has invalid shape: {self.shape}")
                return False
            
            # Verificar que el tamaño de core_data sea razonable
            if len(self.core_data) > 1024 * 1024 * 1024:  # 1GB
                logger.warning(f"Descriptor core_data too large: {len(self.core_data)} bytes")
                return False

            # Verificar merkle root si está presente
            if self.merkle_root:
                computed_root = self._compute_merkle_root()
                if computed_root != self.merkle_root:
                    logger.warning("Descriptor merkle root mismatch")
                    return False

            # Verificar security hash
            if self.security_hash:
                computed_hash = self._compute_security_hash()
                if computed_hash != self.security_hash:
                    logger.warning("Descriptor security hash mismatch")
                    return False

            # Verificar que la versión sea válida
            if self.version < 0:
                logger.warning(f"Descriptor has negative version: {self.version}")
                return False

            return True
                
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False
    
    def _compute_merkle_root(self) -> bytes:
        """Calcular raíz de Merkle para verificación de integridad"""
        return hashlib.sha256(self.core_data).digest()
    
    def _compute_security_hash(self) -> bytes:
        """Calcular hash de seguridad para verificación adicional"""
        data = f"{self.kind}:{self.decomp_type.value}:{self.shape}:{self.version}:{self.core_data[:100]}".encode()
        return hashlib.sha256(data).digest()
    
    def update_access(self) -> None:
        """Actualizar estadísticas de acceso"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age_seconds(self) -> float:
        """Obtener edad del descriptor en segundos"""
        return time.time() - self.created_at
    
    def get_access_frequency(self) -> float:
        """Obtener frecuencia de acceso (accesos por segundo)"""
        age = self.get_age_seconds()
        if age <= 0:
            return 0.0
        return self.access_count / age
    
    def get_size_bytes(self) -> int:
        """Obtener tamaño total del descriptor en bytes"""
        size = len(self.core_data)
        if self.delta_chain:
            size += len(self.delta_chain)
        return size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir descriptor a diccionario para serialización"""
        return {
            'kind': self.kind,
            'decomp_type': self.decomp_type.value,
            'shape': list(self.shape) if self.shape else [],
            'ranks': self.ranks,
            'core_data': base64.b64encode(self.core_data).decode(),
            'version': self.version,
            'meta': self.meta,
            'merkle_root': base64.b64encode(self.merkle_root).decode() if self.merkle_root else None,
            'compression_level': self.compression_level.value,
            'delta_chain': base64.b64encode(self.delta_chain).decode() if self.delta_chain else None,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'security_hash': base64.b64encode(self.security_hash).decode() if self.security_hash else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ZDescriptor':
        """Crear descriptor desde diccionario"""
        # Decodificar datos binarios
        processed_data = data.copy()
        if 'core_data' in processed_data:
            processed_data['core_data'] = base64.b64decode(processed_data['core_data'])
        if 'merkle_root' in processed_data and processed_data['merkle_root']:
            processed_data['merkle_root'] = base64.b64decode(processed_data['merkle_root'])
        if 'delta_chain' in processed_data and processed_data['delta_chain']:
            processed_data['delta_chain'] = base64.b64decode(processed_data['delta_chain'])
        if 'security_hash' in processed_data and processed_data['security_hash']:
            processed_data['security_hash'] = base64.b64decode(processed_data['security_hash'])
        
        # Convertir enums
        if 'decomp_type' in processed_data:
            processed_data['decomp_type'] = DecompType(processed_data['decomp_type'])
        if 'compression_level' in processed_data:
            processed_data['compression_level'] = CompressionLevel(processed_data['compression_level'])
        
        # Convertir shape de vuelta a tupla
        if 'shape' in processed_data and isinstance(processed_data['shape'], list):
            processed_data['shape'] = tuple(processed_data['shape'])
        
        return cls(**processed_data)

@dataclass
class ZAddr:
    """
    Dirección de descriptor con validación mejorada
    
    Representa una dirección única para un descriptor de tensor,
    calculada de forma determinista basada en sus propiedades.
    """
    addr: bytes
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validar dirección con verificaciones mejoradas"""
        if not isinstance(self.addr, bytes):
            raise ValidationError(
                "addr must be bytes",
                field_name="addr",
                expected_type="bytes"
            )
        
        if len(self.addr) != 32:  # SHA-256
            raise ValidationError(
                f"addr must be 32 bytes, got {len(self.addr)}",
                field_name="addr",
                expected_type="bytes with length 32"
            )
        
        # Verificar que no sea todos ceros
        if all(b == 0 for b in self.addr):
            raise ValidationError(
                "addr cannot be all zeros",
                field_name="addr",
                expected_type="non-zero bytes"
            )
    
    @classmethod
    def compute(cls, desc: ZDescriptor) -> 'ZAddr':
        """
        Calcular dirección de descriptor de forma determinista
        
        Args:
            desc: Descriptor para el cual calcular la dirección
            
        Returns:
            ZAddr: Dirección calculada del descriptor
        """
        # Crear hash determinista del descriptor incluyendo más propiedades
        data_parts = [
            desc.kind,
            desc.decomp_type.value,
            str(desc.shape),
            str(desc.version),
            str(desc.compression_level.value)
        ]
        
        # Incluir ranks si están presentes
        if desc.ranks:
            data_parts.append(str(desc.ranks))
        
        # Incluir hash de seguridad si está presente
        if desc.security_hash:
            data_parts.append(desc.security_hash.hex())
        
        data = ":".join(data_parts).encode()
        addr = hashlib.sha256(data).digest()
        return cls(addr)
    
    @classmethod
    def from_hex(cls, hex_str: str) -> 'ZAddr':
        """
        Crear ZAddr desde representación hexadecimal
        
        Args:
            hex_str: Cadena hexadecimal de la dirección
            
        Returns:
            ZAddr: Dirección creada desde hex
        """
        try:
            addr_bytes = bytes.fromhex(hex_str)
            return cls(addr_bytes)
        except ValueError as e:
            raise ValidationError(
                f"Invalid hex string for address: {e}",
                field_name="hex_str",
                expected_type="valid hexadecimal string"
            )
    
    def hex(self) -> str:
        """Obtener representación hexadecimal de la dirección"""
        return self.addr.hex()
    
    def short_hex(self, length: int = 8) -> str:
        """
        Obtener representación hexadecimal corta
        
        Args:
            length: Longitud de la representación corta
            
        Returns:
            str: Representación hexadecimal corta
        """
        return self.addr.hex()[:length]
    
    def __str__(self) -> str:
        """Representación string de la dirección"""
        return f"ZAddr({self.short_hex()})"
    
    def __repr__(self) -> str:
        """Representación detallada de la dirección"""
        return f"ZAddr(addr={self.short_hex()}, created_at={self.created_at})"
    
    def __eq__(self, other) -> bool:
        """Comparación de igualdad"""
        if not isinstance(other, ZAddr):
            return False
        return self.addr == other.addr
    
    def __hash__(self) -> int:
        """Hash de la dirección para usar en conjuntos y diccionarios"""
        return hash(self.addr)
    
    def get_age_seconds(self) -> float:
        """Obtener edad de la dirección en segundos"""
        return time.time() - self.created_at

# --- Clase Principal ZSpace ---

class ZSpace:
    """
    Interfaz principal del runtime MNEME con seguridad mejorada
    
    ZSpace es el componente central del sistema MNEME que proporciona:
    - Almacenamiento seguro de tensores con compresión
    - Sistema de cache adaptativo inteligente
    - Gestión de memoria optimizada con lazy loading
    - Seguridad criptográfica robusta
    - Paralelización y concurrencia segura
    - Monitoreo y métricas en tiempo real
    
    Características principales:
    - Serialización segura con SafeTensors
    - Compresión adaptativa con LZ4
    - Locks granulares para concurrencia
    - Validación de integridad con Merkle trees
    - Gestión automática de memoria
    - Soporte para GPU/CPU con detección automática
    
    Ejemplo de uso:
        >>> config = MnemeConfig(cache_size_mb=512, use_gpu=True)
        >>> zspace = ZSpace(config)
        >>> tensor = torch.randn(100, 100)
        >>> desc = zspace.register("my_tensor", tensor)
        >>> loaded_tensor = zspace.load("my_tensor")
        >>> zspace.cleanup()
    """
    
    def __init__(self, config: Optional[MnemeConfig] = None):
        """
        Inicializar ZSpace con configuración opcional
        
        Args:
            config: Configuración personalizada de MNEME. Si es None,
                   se usará la configuración por defecto.
                   
        Raises:
            ValidationError: Si la configuración es inválida
            SecurityError: Si hay problemas con la configuración de seguridad
        """
        self.config = config or MnemeConfig()
        
        # Configuración de dispositivo con detección automática mejorada
        self.device = self._setup_device()
        
        # Sistema de locks granulares mejorado
        self.lock_manager = GranularLockManager(
            max_locks=self.config.max_workers * 10,
            cleanup_interval=300.0
        )
        
        # Configuración de seguridad mejorada
        self._setup_security()
        
        # Tablas de mapeo con métricas
        self.name_to_desc: Dict[str, ZDescriptor] = {}
        self.addr_to_desc: Dict[bytes, ZDescriptor] = {}
        self.version_graph: Dict[bytes, bytes] = {}
        
        # Cache adaptativo con configuración optimizada
        self.adaptive_cache = AdaptiveCache(
            max_size_bytes=self.config.cache_size_mb * 1024 * 1024,
            strategy=self.config.cache_policy.value,
            ttl_seconds=3600.0,
            compression_threshold=1024
        )
        
        # Sistema de seguridad
        self.security_config = create_secure_config()
        self.security_manager = SecurityManager(self.security_config)
        
        # Sistema de almacenamiento seguro
        self.storage_config = StorageConfig(
            cache_size_mb=self.config.cache_size_mb,
            storage_path=self.config.storage_path
        )
        self.storage_backend = create_secure_storage(self.storage_config)
        
        # Métricas mejoradas
        self.storage_metrics = {
            "read_operations": 0,
            "write_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "storage_loads": 0,
            "storage_stores": 0,
            "compression_ratio": 0.0,
            "total_storage_bytes": 0,
            "tensor_count": 0,
            "memory_usage_bytes": 0,
            "last_cleanup": time.time(),
            "uptime_seconds": 0
        }
        
        # Configurar logging
        self._setup_logging()
        
        # Inicialización completada
        self._log_initialization()
    
    def _setup_device(self) -> torch.device:
        """Configurar dispositivo de computación con detección automática"""
        if not self.config.use_gpu:
            return torch.device("cpu")
        
        # Verificar CUDA
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Configurar memoria GPU si es necesario
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            return device
        
        # Verificar MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        
        # Fallback a CPU
        logger.warning("GPU requested but not available, falling back to CPU")
        return torch.device("cpu")
    
    def _setup_security(self) -> None:
        """Configurar sistema de seguridad"""
        if self.config.secret_key:
            if len(self.config.secret_key) < 32:
                raise SecurityError(
                    "Secret key must be >= 32 bytes",
                    security_level=self.config.security_level.value,
                    context={'key_length': len(self.config.secret_key)}
                )
        else:
            logger.warning("No secret key provided. Generating a transient secure key.")
            self.config = replace(self.config, secret_key=secrets.token_bytes(32))
    
    def _setup_logging(self) -> None:
        """Configurar sistema de logging"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Configurar handler si no existe
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    def _log_initialization(self) -> None:
        """Registrar información de inicialización"""
        logger.info(f"ZSpace initialized successfully")
        logger.info(f"Device: {self.device}")
        logger.info(f"Security level: {self.config.security_level.name}")
        logger.info(f"Serialization format: {self.config.serialization_format.name}")
        logger.info(f"Cache size: {self.config.cache_size_mb} MB")
        logger.info(f"Compression level: {self.config.compression_level.name}")
        logger.info(f"Max workers: {self.config.max_workers}")
        logger.info(f"Using SafeTensors for secure serialization")
    
    def register(self, name: str, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """
        Registrar tensor con validación de seguridad mejorada
        
        Args:
            name: Nombre único para el tensor
            tensor: Tensor de PyTorch a registrar
            **kwargs: Argumentos adicionales para el descriptor
            
        Returns:
            ZDescriptor: Descriptor del tensor registrado
            
        Raises:
            ValidationError: Si el tensor o nombre no son válidos
            SecurityError: Si hay problemas de seguridad
            MemoryError: Si no hay suficiente memoria
        """
        # Validar nombre
        if not isinstance(name, str) or not name.strip():
            raise ValidationError(
                "Name must be a non-empty string",
                field_name="name",
                expected_type="str"
            )
        
        # Validar tensor
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(
                "Input must be a PyTorch tensor",
                field_name="tensor",
                expected_type="torch.Tensor"
            )
        
        # Validar tamaño del tensor
        tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        if tensor_size_mb > self.config.max_tensor_size_mb:
            raise MemoryError(
                f"Tensor size ({tensor_size_mb:.2f} MB) exceeds maximum allowed ({self.config.max_tensor_size_mb} MB)",
                memory_usage={'tensor_size_mb': tensor_size_mb, 'max_allowed_mb': self.config.max_tensor_size_mb}
            )
        
        # Validar entrada con sistema de seguridad
        if not self.security_manager.validate_input(tensor, "tensor"):
            raise SecurityError(
                "Tensor validation failed",
                security_level=self.config.security_level.value,
                context={'tensor_shape': tensor.shape, 'tensor_dtype': str(tensor.dtype)}
            )
        
        # Usar lock granular para escritura
        with self.lock_manager.acquire_lock(name, LockType.WRITE):
            # Crear descriptor seguro
            desc = self._create_secure_descriptor(tensor, **kwargs)
            
            # Verificar si ya existe un tensor con este nombre
            old_addr = None
            if name in self.name_to_desc:
                old_addr = ZAddr.compute(self.name_to_desc[name])
                logger.info(f"Replacing existing tensor '{name}'")
            
            # Registrar descriptor
            addr = self._register_descriptor(name, desc, old_addr)
            
            # Almacenar en cache adaptativo
            cache_success = self.adaptive_cache.put(f"desc_{name}", desc)
            if not cache_success:
                logger.warning(f"Failed to cache descriptor for '{name}'")
            
            # Actualizar métricas
            self.storage_metrics["write_operations"] += 1
            self.storage_metrics["tensor_count"] = len(self.name_to_desc)
            self.storage_metrics["total_storage_bytes"] += desc.get_size_bytes()
            
        logger.info(f"Registered '{name}'. Type: {desc.decomp_type.value}. Addr: {addr.short_hex()}")
        return desc
    
    def load(self, name: str) -> torch.Tensor:
        """
        Cargar tensor con lazy decompression mejorada
        
        Args:
            name: Nombre del tensor a cargar
            
        Returns:
            torch.Tensor: Tensor cargado y decompressed
            
        Raises:
            KeyError: Si el tensor no existe
            SecurityError: Si hay problemas de integridad
            MemoryError: Si no hay suficiente memoria
        """
        # Validar nombre
        if not isinstance(name, str) or not name.strip():
            raise ValidationError(
                "Name must be a non-empty string",
                field_name="name",
                expected_type="str"
            )
        
        # Usar lock granular para lectura
        with self.lock_manager.acquire_lock(name, LockType.READ):
            # Intentar obtener desde cache adaptativo primero
            cached_desc = self.adaptive_cache.get(f"desc_{name}")
            if cached_desc:
                desc = cached_desc
                self.storage_metrics["cache_hits"] += 1
            elif name in self.name_to_desc:
                # Obtener desde memoria
                desc = self.name_to_desc[name]
                self.adaptive_cache.put(f"desc_{name}", desc)
                self.storage_metrics["cache_misses"] += 1
            else:
                # Intentar cargar desde storage backend persistente
                try:
                    desc = self._load_from_storage(name)
                    if desc:
                        # Restaurar en memoria y cache
                        self.name_to_desc[name] = desc
                        self.addr_to_desc[ZAddr.compute(desc).addr] = desc
                        self.adaptive_cache.put(f"desc_{name}", desc)
                        self.storage_metrics["storage_loads"] += 1
                        logger.info(f"Loaded '{name}' from persistent storage")
                    else:
                        raise KeyError(f"Unknown tensor: {name}")
                except Exception as e:
                    logger.error(f"Failed to load '{name}' from storage: {e}")
                    raise KeyError(f"Unknown tensor: {name}")
            
            # Actualizar estadísticas de acceso del descriptor
            desc.update_access()
        
        # Actualizar métricas de lectura
        self.storage_metrics["read_operations"] += 1
        
        # Usar lazy decompression si está disponible
        if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
            try:
                return desc.lazy_tensor.decompress()
            except Exception as e:
                logger.error(f"Lazy decompression failed for '{name}': {e}")
                raise CompressionError(
                    f"Failed to decompress tensor '{name}'",
                    compression_type="lz4",
                    context={'tensor_name': name, 'error': str(e)}
                )
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
        """Registrar descriptor en las tablas y almacenamiento persistente"""
        addr = ZAddr.compute(desc)
        
        # Almacenar en memoria (tablas locales)
        self.name_to_desc[name] = desc
        self.addr_to_desc[addr.addr] = desc
        
        if old_addr:
            self.version_graph[addr.addr] = old_addr.addr
        
        # Almacenar en storage backend persistente
        try:
            # Serializar descriptor para almacenamiento
            desc_data = desc.to_dict()
            desc_bytes = json.dumps(desc_data).encode('utf-8')
            
            # Almacenar descriptor en storage backend
            self.storage_backend.store(
                key=f"desc_{name}",
                data=desc_bytes,
                metadata={
                    'tensor_name': name,
                    'addr': str(addr.addr),
                    'created_at': desc.created_at if isinstance(desc.created_at, str) else str(desc.created_at),
                    'size_bytes': desc.get_size_bytes(),
                    'decomp_type': desc.decomp_type.value
                }
            )
            
            # Si hay lazy tensor, almacenar también los datos comprimidos
            if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
                compressed_data = desc.lazy_tensor.compressed_data
                if compressed_data:
                    self.storage_backend.store(
                        key=f"data_{name}",
                        data=compressed_data,
                        metadata={
                            'tensor_name': name,
                            'addr': str(addr.addr),
                            'compression_type': 'lz4',
                            'original_size': getattr(desc.lazy_tensor, 'original_size', len(compressed_data)),
                            'compressed_size': len(compressed_data)
                        }
                    )
            
            logger.debug(f"Stored descriptor and data for '{name}' in persistent storage")
            self.storage_metrics["storage_stores"] += 1
            
        except Exception as e:
            logger.error(f"Failed to store '{name}' in persistent storage: {e}")
            # No fallar la operación, pero registrar el error
            # El tensor seguirá funcionando en memoria
        
        return addr
    
    def _load_from_storage(self, name: str) -> Optional[ZDescriptor]:
        """Cargar descriptor desde storage backend persistente"""
        try:
            # Cargar descriptor desde storage
            desc_data = self.storage_backend.retrieve(f"desc_{name}")
            if not desc_data:
                return None
            
            # Reconstruir descriptor desde datos serializados
            desc_dict = json.loads(desc_data.decode('utf-8'))
            desc = ZDescriptor.from_dict(desc_dict)
            
            # Cargar datos comprimidos si existen
            compressed_data = self.storage_backend.retrieve(f"data_{name}")
            if compressed_data and hasattr(desc, 'lazy_tensor'):
                # Reconstruir LazyTensor con datos comprimidos
                # Crear función de decompresión
                def decompression_func(data):
                    return self._decompress_tensor(data)
                
                desc.lazy_tensor = LazyTensor(
                    compressed_data=compressed_data,
                    decompression_func=decompression_func,
                    metadata={'shape': desc.shape, 'dtype': 'torch.float32'},
                    device=self.device,
                    max_memory_mb=getattr(self.config, 'lazy_tensor_memory_limit', 1024)  # 1GB por defecto
                )
            
            return desc
            
        except Exception as e:
            logger.error(f"Failed to load descriptor for '{name}' from storage: {e}")
            return None
    
    def list_tensors(self) -> Dict[str, Any]:
        """Listar todos los tensores disponibles (memoria + storage)"""
        memory_tensors = set(self.name_to_desc.keys())
        
        # Obtener tensores desde storage backend
        storage_tensors = set()
        try:
            storage_keys = self.storage_backend.list_keys()
            for key in storage_keys:
                if key.startswith("desc_"):
                    tensor_name = key[5:]  # Remove "desc_" prefix
                    storage_tensors.add(tensor_name)
        except Exception as e:
            logger.warning(f"Failed to list storage tensors: {e}")
        
        # Combinar y categorizar
        all_tensors = memory_tensors.union(storage_tensors)
        
        result = {
            "total_tensors": len(all_tensors),
            "memory_tensors": list(memory_tensors),
            "storage_tensors": list(storage_tensors),
            "memory_only": list(memory_tensors - storage_tensors),
            "storage_only": list(storage_tensors - memory_tensors),
            "both": list(memory_tensors.intersection(storage_tensors))
        }
        
        return result
    
    def delete_tensor(self, name: str) -> bool:
        """Eliminar tensor de memoria y storage"""
        success = True
        
        # Eliminar de memoria
        if name in self.name_to_desc:
            desc = self.name_to_desc[name]
            addr = ZAddr.compute(desc)
            
            # Eliminar de tablas
            del self.name_to_desc[name]
            if addr.addr in self.addr_to_desc:
                del self.addr_to_desc[addr.addr]
            
            # Eliminar de cache
            self.adaptive_cache.remove(f"desc_{name}")
            
            # Limpiar lazy tensor si existe
            if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
                desc.lazy_tensor.clear_decompressed()
        
        # Eliminar de storage
        try:
            self.storage_backend.delete(f"desc_{name}")
            self.storage_backend.delete(f"data_{name}")
            logger.info(f"Deleted tensor '{name}' from storage")
        except Exception as e:
            logger.error(f"Failed to delete '{name}' from storage: {e}")
            success = False
        
        return success
    
    def sync_to_storage(self) -> Dict[str, Any]:
        """Sincronizar todos los tensores en memoria con storage persistente"""
        sync_results = {
            "total_tensors": len(self.name_to_desc),
            "successful_syncs": 0,
            "failed_syncs": 0,
            "errors": []
        }
        
        for name, desc in self.name_to_desc.items():
            try:
                # Verificar si ya existe en storage
                existing_desc = self.storage_backend.retrieve(f"desc_{name}")
                if not existing_desc:
                    # No existe en storage, sincronizar
                    addr = ZAddr.compute(desc)
                    
                    # Almacenar descriptor
                    desc_data = desc.to_dict()
                    desc_bytes = json.dumps(desc_data).encode('utf-8')
                    self.storage_backend.store(
                        key=f"desc_{name}",
                        data=desc_bytes,
                        metadata={
                            'tensor_name': name,
                            'addr': str(addr.addr),
                            'created_at': desc.created_at if isinstance(desc.created_at, str) else str(desc.created_at),
                            'size_bytes': desc.get_size_bytes(),
                            'decomp_type': desc.decomp_type.value
                        }
                    )
                    
                    # Almacenar datos comprimidos si existen
                    if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
                        compressed_data = desc.lazy_tensor.compressed_data
                        if compressed_data:
                            self.storage_backend.store(
                                key=f"data_{name}",
                                data=compressed_data,
                                metadata={
                                    'tensor_name': name,
                                    'addr': str(addr.addr),
                                    'compression_type': 'lz4',
                                    'original_size': getattr(desc.lazy_tensor, 'original_size', len(compressed_data)),
                                    'compressed_size': len(compressed_data)
                                }
                            )
                    
                    sync_results["successful_syncs"] += 1
                    logger.debug(f"Synced '{name}' to storage")
                
            except Exception as e:
                sync_results["failed_syncs"] += 1
                sync_results["errors"].append(f"Failed to sync '{name}': {e}")
                logger.error(f"Failed to sync '{name}' to storage: {e}")
        
        logger.info(f"Storage sync completed: {sync_results['successful_syncs']} successful, {sync_results['failed_syncs']} failed")
        return sync_results
    
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
