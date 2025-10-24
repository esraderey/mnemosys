"""
MNEME Core: Motor de Memoria Neural Mórfica (Refactorizado y Mejorado)
Sistema avanzado de memoria computacional con síntesis determinista, verificación criptográfica robusta, 
aceleración de hardware y optimizaciones de rendimiento.
"""

import hashlib
import struct
import numpy as np
import torch
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from collections import deque, OrderedDict
from threading import Lock, RLock
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar backend de TensorLy
try:
    tl.set_backend('pytorch')
except Exception as e:
    warnings.warn(f"Could not set TensorLy backend to PyTorch: {e}")

# --- Enums y Clases de Error ---

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
    # Mapeo a niveles de compresión LZ4 (1-12)
    ULTRA_FAST = 1
    FAST = 3
    BALANCED = 6
    HIGH = 9
    MAXIMUM = 12

class SerializationFormat(Enum):
    """Formatos de serialización soportados"""
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

class TensorEncryptionMode(Enum):
    """Modos de cifrado específicos para tensores"""
    AES_GCM = "aes_gcm"  # Autenticado, recomendado
    AES_CBC = "aes_cbc"  # Más rápido, menos seguro
    CHACHA20 = "chacha20"  # Alternativa a AES
    BLOCK_CHAIN = "block_chain"  # Cifrado por bloques para tensores grandes

class KeyRotationPolicy(Enum):
    """Políticas de rotación de claves"""
    NEVER = "never"
    TIME_BASED = "time_based"  # Rotar cada X tiempo
    USAGE_BASED = "usage_based"  # Rotar después de X usos
    ADAPTIVE = "adaptive"  # Rotar basado en patrones de uso

class StorageBackend(Enum):
    """Backends de almacenamiento disponibles"""
    MEMORY = "memory"  # Solo en memoria
    DISK = "disk"  # Almacenamiento en disco
    REDIS = "redis"  # Redis para cache distribuido
    S3 = "s3"  # Amazon S3
    HDFS = "hdfs"  # Hadoop Distributed File System
    HYBRID = "hybrid"  # Combinación de múltiples backends

class CachePolicy(Enum):
    """Políticas de cache"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    ADAPTIVE = "adaptive"  # Política adaptativa
    TTL = "ttl"  # Time To Live

class CompressionStrategy(Enum):
    """Estrategias de compresión"""
    NONE = "none"  # Sin compresión
    LZ4 = "lz4"  # LZ4 rápido
    ZSTD = "zstd"  # Zstandard balanceado
    GZIP = "gzip"  # GZIP estándar
    ADAPTIVE = "adaptive"  # Selección automática

class ContextSimilarityMethod(Enum):
    """Métodos de similitud de contexto"""
    COSINE = "cosine"  # Similitud coseno
    EUCLIDEAN = "euclidean"  # Distancia euclidiana
    MANHATTAN = "manhattan"  # Distancia Manhattan
    JACCARD = "jaccard"  # Índice Jaccard
    SEMANTIC = "semantic"  # Análisis semántico
    HYBRID = "hybrid"  # Combinación de métodos

class ContextClusteringMethod(Enum):
    """Métodos de clustering de contexto"""
    KMEANS = "kmeans"  # K-means tradicional
    HIERARCHICAL = "hierarchical"  # Clustering jerárquico
    DBSCAN = "dbscan"  # DBSCAN basado en densidad
    SPECTRAL = "spectral"  # Clustering espectral
    ADAPTIVE = "adaptive"  # Selección automática

class SecurityError(Exception):
    """Error relacionado con la seguridad (e.g., fallo de verificación HMAC)"""
    pass

# --- Configuración ---

@dataclass
class MnemeConfig:
    """Configuración centralizada para el motor MNEME"""
    cache_size_bytes: int = 1 << 30  # 1 GB
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    use_gpu: bool = True
    secret_key: Optional[bytes] = None # Para firmado HMAC
    enable_merkle: bool = False # Merkle Tree complexity might outweigh benefits for single chunks
    delta_consolidation_threshold: int = 50 # Consolidar después de 50 deltas
    memory_pressure_threshold: float = 0.85 # 85% usage
    num_workers: int = max(4, psutil.cpu_count(logical=False) or 4)
    # Nuevas opciones de serialización avanzada
    serialization_format: SerializationFormat = SerializationFormat.HYBRID
    security_level: SecurityLevel = SecurityLevel.HMAC
    enable_encryption: bool = False
    encryption_password: Optional[str] = None
    enable_compression: bool = True
    enable_validation: bool = True
    
    # Opciones de cifrado de tensores
    tensor_encryption_mode: TensorEncryptionMode = TensorEncryptionMode.AES_GCM
    enable_tensor_encryption: bool = False
    tensor_encryption_key: Optional[bytes] = None
    
    # Rotación de claves
    key_rotation_policy: KeyRotationPolicy = KeyRotationPolicy.NEVER
    key_rotation_interval: timedelta = timedelta(days=30)  # Para TIME_BASED
    key_rotation_usage_count: int = 1000  # Para USAGE_BASED
    enable_key_versioning: bool = False
    
    # Contexto asíncrono
    enable_async_context: bool = False
    max_concurrent_operations: int = 10
    
    # Sistema de almacenamiento avanzado
    storage_backend: StorageBackend = StorageBackend.HYBRID
    storage_path: Optional[str] = None  # Ruta para almacenamiento en disco
    enable_persistent_storage: bool = True
    enable_deduplication: bool = True
    
    # Cache avanzado
    cache_policy: CachePolicy = CachePolicy.ADAPTIVE
    cache_size_mb: int = 1024  # Tamaño del cache en MB
    cache_ttl_seconds: int = 3600  # TTL por defecto
    enable_distributed_cache: bool = False
    redis_url: Optional[str] = None  # URL de Redis para cache distribuido
    
    # Compresión adaptativa
    compression_strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE
    enable_adaptive_compression: bool = True
    compression_threshold_mb: float = 1.0  # Umbral para compresión automática
    
    # Indexación y búsqueda
    enable_indexing: bool = True
    index_type: str = "btree"  # Tipo de índice
    enable_full_text_search: bool = False
    
    # Métricas y monitoreo
    enable_storage_metrics: bool = True
    metrics_interval_seconds: int = 60
    
    # Sistema de deduplicación de contexto
    enable_context_deduplication: bool = True
    context_similarity_method: ContextSimilarityMethod = ContextSimilarityMethod.HYBRID
    context_clustering_method: ContextClusteringMethod = ContextClusteringMethod.ADAPTIVE
    context_similarity_threshold: float = 0.8  # Umbral de similitud (0-1)
    context_cluster_size: int = 10  # Tamaño mínimo de cluster
    enable_semantic_analysis: bool = True
    context_compression_level: int = 6  # Nivel de compresión para contextos
    enable_context_caching: bool = True
    context_cache_size_mb: int = 256  # Tamaño del cache de contextos

# --- Utilidades ---

def deterministic_serialize(data: Any) -> bytes:
    """Serializa metadatos de forma determinista para hashing."""
    if isinstance(data, dict):
        # JSON con claves ordenadas garantiza determinismo
        return json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
    elif isinstance(data, Enum):
        return str(data.value).encode('utf-8')
    # Fallback para otros tipos básicos
    return str(data).encode('utf-8')

# --- Core Data Structures ---

@dataclass(frozen=True, slots=True)
class ZDescriptor:
    """Descriptor inmutable que identifica contenido sintetizable"""
    kind: str
    decomp_type: DecompType
    shape: Tuple[int, ...]
    # core_data contiene componentes base comprimidos y firmados
    core_data: bytes 
    version: int = 0
    ranks: Optional[Tuple[int, ...]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    # delta_chain contiene deltas acumulados comprimidos y firmados
    delta_chain: Optional[bytes] = None
    merkle_root: Optional[bytes] = None
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    # El checksum verifica la integridad de todo el descriptor (metadatos + contenido)
    checksum: bytes = field(init=False)

    def __post_init__(self):
        # Calcular checksum durante la inicialización. 
        # Usar object.__setattr__ porque la dataclass es frozen.
        object.__setattr__(self, 'checksum', self._compute_descriptor_checksum())

    def verify_integrity(self) -> bool:
        """Verificar integridad del descriptor completo."""
        computed = self._compute_descriptor_checksum()
        # Usar compare_digest para seguridad contra ataques de timing
        return secrets.compare_digest(computed, self.checksum)

    def _compute_content_hash(self) -> bytes:
        """Calcula el hash SHA256 del contenido real (core_data + deltas + merkle)."""
        h = hashlib.sha256()
        h.update(self.core_data)
        if self.delta_chain:
            h.update(self.delta_chain)
        if self.merkle_root:
            h.update(self.merkle_root)
        return h.digest()

    def _compute_descriptor_checksum(self) -> bytes:
        """
        [MEJORA] Calcula el checksum criptográfico (SHA256) del descriptor completo.
        """
        h = hashlib.sha256()
        
        # 1. Incluir metadatos estructurales de forma determinista
        h.update(deterministic_serialize(self.kind))
        h.update(deterministic_serialize(self.decomp_type))
        # Usar 'Q' (unsigned long long) para shapes potencialmente grandes
        h.update(struct.pack(f'!{len(self.shape)}Q', *self.shape))
        
        if self.ranks:
            h.update(struct.pack(f'!{len(self.ranks)}I', *self.ranks))
        
        h.update(struct.pack('!Q', self.version))
        h.update(deterministic_serialize(self.compression_level))
        h.update(deterministic_serialize(self.meta))

        # 2. Incluir el hash del contenido
        h.update(self._compute_content_hash())
        
        return h.digest()

class ZAddr:
    """Zero-address: direccionamiento basado en contenido"""
    
    @staticmethod
    def compute(desc: ZDescriptor) -> bytes:
        """
        Calcular dirección determinista. Usamos el checksum del descriptor, 
        ya que representa de forma única y segura el estado completo.
        Usamos XXH3-128 sobre el checksum SHA256 para una dirección rápida (128 bits).
        """
        return xxhash.xxh3_128(desc.checksum).digest()

class MerkleTree:
    """Árbol Merkle para verificación de integridad (Implementación básica)"""
    # (La implementación original era correcta. Se mantiene simplificada aquí)
    @staticmethod
    def compute_root(data_chunks: List[bytes]) -> bytes:
        if not data_chunks:
            return b''
        # En un sistema real, esto construiría el árbol. Aquí simplificamos el concepto.
        h = hashlib.sha256()
        for chunk in data_chunks:
             h.update(hashlib.sha256(chunk).digest())
        return h.digest()

class AdvancedCompressor:
    @staticmethod
    def compress(data: bytes, level: CompressionLevel) -> bytes:
        try:
            return lz4.frame.compress(data, compression_level=level.value, content_checksum=True)
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise
    
    @staticmethod
    def decompress(data: bytes) -> bytes:
        try:
            return lz4.frame.decompress(data)
        except Exception as e:
            logger.error(f"Decompression failed (data corruption likely): {e}")
            raise

# --- Sistema de Almacenamiento Avanzado ---

class StorageBackendInterface:
    """Interfaz para backends de almacenamiento."""
    
    def store(self, key: bytes, data: bytes) -> bool:
        """Almacenar datos."""
        raise NotImplementedError
    
    def retrieve(self, key: bytes) -> Optional[bytes]:
        """Recuperar datos."""
        raise NotImplementedError
    
    def delete(self, key: bytes) -> bool:
        """Eliminar datos."""
        raise NotImplementedError
    
    def exists(self, key: bytes) -> bool:
        """Verificar si existe."""
        raise NotImplementedError
    
    def list_keys(self) -> List[bytes]:
        """Listar todas las claves."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        raise NotImplementedError

class MemoryStorage(StorageBackendInterface):
    """Almacenamiento en memoria."""
    
    def __init__(self, max_size_mb: int = 1024):
        self.storage = {}
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.lock = Lock()
    
    def store(self, key: bytes, data: bytes) -> bool:
        with self.lock:
            if len(data) > self.max_size_bytes:
                return False
            
            # Verificar espacio disponible
            if key in self.storage:
                self.current_size -= len(self.storage[key])
            
            if self.current_size + len(data) > self.max_size_bytes:
                self._evict_oldest()
            
            self.storage[key] = data
            self.current_size += len(data)
            return True
    
    def retrieve(self, key: bytes) -> Optional[bytes]:
        with self.lock:
            return self.storage.get(key)
    
    def delete(self, key: bytes) -> bool:
        with self.lock:
            if key in self.storage:
                self.current_size -= len(self.storage[key])
                del self.storage[key]
                return True
            return False
    
    def exists(self, key: bytes) -> bool:
        with self.lock:
            return key in self.storage
    
    def list_keys(self) -> List[bytes]:
        with self.lock:
            return list(self.storage.keys())
    
    def _evict_oldest(self):
        """Eliminar el elemento más antiguo."""
        if self.storage:
            oldest_key = next(iter(self.storage))
            self.delete(oldest_key)
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "type": "memory",
                "entries": len(self.storage),
                "size_bytes": self.current_size,
                "max_size_bytes": self.max_size_bytes,
                "usage_percent": (self.current_size / self.max_size_bytes) * 100
            }

class DiskStorage(StorageBackendInterface):
    """Almacenamiento en disco."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self._init_metadata()
    
    def _init_metadata(self):
        """Inicializar metadatos."""
        self.metadata_file = self.storage_path / "metadata.json"
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"keys": {}, "stats": {"total_size": 0, "file_count": 0}}
    
    def _save_metadata(self):
        """Guardar metadatos."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def store(self, key: bytes, data: bytes) -> bool:
        try:
            with self.lock:
                key_hex = key.hex()
                file_path = self.storage_path / f"{key_hex}.dat"
                
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Actualizar metadatos
                self.metadata["keys"][key_hex] = {
                    "size": len(data),
                    "created": time.time()
                }
                self.metadata["stats"]["total_size"] += len(data)
                self.metadata["stats"]["file_count"] += 1
                self._save_metadata()
                
                return True
        except Exception as e:
            logger.error(f"Disk storage failed: {e}")
            return False
    
    def retrieve(self, key: bytes) -> Optional[bytes]:
        try:
            key_hex = key.hex()
            file_path = self.storage_path / f"{key_hex}.dat"
            
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return f.read()
            return None
        except Exception as e:
            logger.error(f"Disk retrieval failed: {e}")
            return None
    
    def delete(self, key: bytes) -> bool:
        try:
            with self.lock:
                key_hex = key.hex()
                file_path = self.storage_path / f"{key_hex}.dat"
                
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    
                    # Actualizar metadatos
                    if key_hex in self.metadata["keys"]:
                        del self.metadata["keys"][key_hex]
                    self.metadata["stats"]["total_size"] -= file_size
                    self.metadata["stats"]["file_count"] -= 1
                    self._save_metadata()
                    
                    return True
                return False
        except Exception as e:
            logger.error(f"Disk deletion failed: {e}")
            return False
    
    def exists(self, key: bytes) -> bool:
        key_hex = key.hex()
        file_path = self.storage_path / f"{key_hex}.dat"
        return file_path.exists()
    
    def list_keys(self) -> List[bytes]:
        try:
            keys = []
            for key_hex in self.metadata["keys"]:
                keys.append(bytes.fromhex(key_hex))
            return keys
        except Exception as e:
            logger.error(f"List keys failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "disk",
            "path": str(self.storage_path),
            "entries": len(self.metadata["keys"]),
            "total_size_bytes": self.metadata["stats"]["total_size"],
            "file_count": self.metadata["stats"]["file_count"]
        }

class HybridStorage(StorageBackendInterface):
    """Almacenamiento híbrido que combina múltiples backends."""
    
    def __init__(self, config: MnemeConfig):
        self.config = config
        self.backends = {}
        self._init_backends()
    
    def _init_backends(self):
        """Inicializar backends según configuración."""
        # Backend principal (disco)
        if self.config.storage_backend in [StorageBackend.DISK, StorageBackend.HYBRID]:
            storage_path = self.config.storage_path or "./mneme_storage"
            self.backends["disk"] = DiskStorage(storage_path)
        
        # Cache en memoria
        if self.config.storage_backend in [StorageBackend.MEMORY, StorageBackend.HYBRID]:
            self.backends["memory"] = MemoryStorage(self.config.cache_size_mb)
        
        # Redis para cache distribuido
        if self.config.enable_distributed_cache and self.config.redis_url:
            try:
                import redis
                self.backends["redis"] = RedisStorage(self.config.redis_url)
            except ImportError:
                logger.warning("Redis not available, skipping distributed cache")
    
    def store(self, key: bytes, data: bytes) -> bool:
        """Almacenar en múltiples backends."""
        success = False
        
        # Almacenar en disco (persistente)
        if "disk" in self.backends:
            if self.backends["disk"].store(key, data):
                success = True
        
        # Almacenar en memoria (cache)
        if "memory" in self.backends:
            self.backends["memory"].store(key, data)
        
        # Almacenar en Redis (distribuido)
        if "redis" in self.backends:
            self.backends["redis"].store(key, data)
        
        return success
    
    def retrieve(self, key: bytes) -> Optional[bytes]:
        """Recuperar desde backends en orden de prioridad."""
        # 1. Intentar memoria primero (más rápido)
        if "memory" in self.backends:
            data = self.backends["memory"].retrieve(key)
            if data is not None:
                return data
        
        # 2. Intentar Redis (cache distribuido)
        if "redis" in self.backends:
            data = self.backends["redis"].retrieve(key)
            if data is not None:
                # Cargar en memoria para futuras consultas
                if "memory" in self.backends:
                    self.backends["memory"].store(key, data)
                return data
        
        # 3. Intentar disco (persistente)
        if "disk" in self.backends:
            data = self.backends["disk"].retrieve(key)
            if data is not None:
                # Cargar en cache para futuras consultas
                if "memory" in self.backends:
                    self.backends["memory"].store(key, data)
                return data
        
        return None
    
    def delete(self, key: bytes) -> bool:
        """Eliminar de todos los backends."""
        success = False
        for backend in self.backends.values():
            if backend.delete(key):
                success = True
        return success
    
    def exists(self, key: bytes) -> bool:
        """Verificar existencia en cualquier backend."""
        for backend in self.backends.values():
            if backend.exists(key):
                return True
        return False
    
    def list_keys(self) -> List[bytes]:
        """Listar claves de todos los backends."""
        all_keys = set()
        for backend in self.backends.values():
            all_keys.update(backend.list_keys())
        return list(all_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de todos los backends."""
        stats = {"type": "hybrid", "backends": {}}
        for name, backend in self.backends.items():
            stats["backends"][name] = backend.get_stats()
        return stats

class RedisStorage(StorageBackendInterface):
    """Almacenamiento en Redis."""
    
    def __init__(self, redis_url: str):
        import redis
        self.redis_client = redis.from_url(redis_url)
        self.prefix = "mneme:"
    
    def store(self, key: bytes, data: bytes) -> bool:
        try:
            redis_key = self.prefix + key.hex()
            return self.redis_client.set(redis_key, data)
        except Exception as e:
            logger.error(f"Redis storage failed: {e}")
            return False
    
    def retrieve(self, key: bytes) -> Optional[bytes]:
        try:
            redis_key = self.prefix + key.hex()
            data = self.redis_client.get(redis_key)
            return data
        except Exception as e:
            logger.error(f"Redis retrieval failed: {e}")
            return None
    
    def delete(self, key: bytes) -> bool:
        try:
            redis_key = self.prefix + key.hex()
            return bool(self.redis_client.delete(redis_key))
        except Exception as e:
            logger.error(f"Redis deletion failed: {e}")
            return False
    
    def exists(self, key: bytes) -> bool:
        try:
            redis_key = self.prefix + key.hex()
            return bool(self.redis_client.exists(redis_key))
        except Exception as e:
            logger.error(f"Redis exists check failed: {e}")
            return False
    
    def list_keys(self) -> List[bytes]:
        try:
            pattern = self.prefix + "*"
            keys = self.redis_client.keys(pattern)
            return [bytes.fromhex(key.decode().replace(self.prefix, "")) for key in keys]
        except Exception as e:
            logger.error(f"Redis list keys failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            info = self.redis_client.info()
            return {
                "type": "redis",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except Exception as e:
            logger.error(f"Redis stats failed: {e}")
            return {"type": "redis", "error": str(e)}

class AdvancedCache:
    """Sistema de cache avanzado con múltiples políticas."""
    
    def __init__(self, config: MnemeConfig):
        self.config = config
        self.cache_policy = config.cache_policy
        self.cache_size_bytes = config.cache_size_mb * 1024 * 1024
        self.ttl_seconds = config.cache_ttl_seconds
        
        # Estructuras de datos para diferentes políticas
        self.cache = {}
        self.access_times = {}  # Para LRU
        self.access_counts = {}  # Para LFU
        self.insertion_order = deque()  # Para FIFO/LIFO
        self.expiration_times = {}  # Para TTL
        
        self.current_size = 0
        self.hits = 0
        self.misses = 0
        self.lock = Lock()
        
        # Inicializar métricas
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0,
            "entries": 0
        }
    
    def get(self, key: bytes) -> Optional[bytes]:
        """Obtener valor del cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                self.metrics["misses"] += 1
                return None
            
            # Verificar TTL
            if self.cache_policy == CachePolicy.TTL:
                if time.time() > self.expiration_times.get(key, 0):
                    self._evict_key(key)
                    self.misses += 1
                    self.metrics["misses"] += 1
                    return None
            
            # Actualizar estadísticas de acceso
            self._update_access_stats(key)
            
            self.hits += 1
            self.metrics["hits"] += 1
            return self.cache[key]
    
    def put(self, key: bytes, value: bytes) -> bool:
        """Almacenar valor en cache."""
        with self.lock:
            # Verificar si el valor cabe en el cache
            if len(value) > self.cache_size_bytes:
                return False
            
            # Evictar elementos si es necesario
            while self.current_size + len(value) > self.cache_size_bytes:
                if not self._evict_according_to_policy():
                    return False
            
            # Almacenar valor
            if key in self.cache:
                self.current_size -= len(self.cache[key])
            
            self.cache[key] = value
            self.current_size += len(value)
            
            # Actualizar estructuras de datos
            self._update_insertion_stats(key)
            
            # Actualizar métricas
            self.metrics["size_bytes"] = self.current_size
            self.metrics["entries"] = len(self.cache)
            
            return True
    
    def _update_access_stats(self, key: bytes):
        """Actualizar estadísticas de acceso."""
        current_time = time.time()
        
        # Para LRU
        if self.cache_policy == CachePolicy.LRU:
            self.access_times[key] = current_time
        
        # Para LFU
        if self.cache_policy == CachePolicy.LFU:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
    
    def _update_insertion_stats(self, key: bytes):
        """Actualizar estadísticas de inserción."""
        current_time = time.time()
        
        # Para FIFO/LIFO
        if self.cache_policy in [CachePolicy.FIFO, CachePolicy.LIFO]:
            if key not in self.cache:
                self.insertion_order.append(key)
        
        # Para TTL
        if self.cache_policy == CachePolicy.TTL:
            self.expiration_times[key] = current_time + self.ttl_seconds
    
    def _evict_according_to_policy(self) -> bool:
        """Evictar elemento según la política."""
        if not self.cache:
            return False
        
        if self.cache_policy == CachePolicy.LRU:
            return self._evict_lru()
        elif self.cache_policy == CachePolicy.LFU:
            return self._evict_lfu()
        elif self.cache_policy == CachePolicy.FIFO:
            return self._evict_fifo()
        elif self.cache_policy == CachePolicy.LIFO:
            return self._evict_lifo()
        elif self.cache_policy == CachePolicy.TTL:
            return self._evict_expired()
        elif self.cache_policy == CachePolicy.ADAPTIVE:
            return self._evict_adaptive()
        else:
            return self._evict_lru()  # Default
    
    def _evict_lru(self) -> bool:
        """Evictar Least Recently Used."""
        if not self.access_times:
            return self._evict_oldest()
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        return self._evict_key(oldest_key)
    
    def _evict_lfu(self) -> bool:
        """Evictar Least Frequently Used."""
        if not self.access_counts:
            return self._evict_oldest()
        
        least_frequent_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        return self._evict_key(least_frequent_key)
    
    def _evict_fifo(self) -> bool:
        """Evictar First In, First Out."""
        if not self.insertion_order:
            return self._evict_oldest()
        
        oldest_key = self.insertion_order.popleft()
        return self._evict_key(oldest_key)
    
    def _evict_lifo(self) -> bool:
        """Evictar Last In, First Out."""
        if not self.insertion_order:
            return self._evict_oldest()
        
        newest_key = self.insertion_order.pop()
        return self._evict_key(newest_key)
    
    def _evict_expired(self) -> bool:
        """Evictar elementos expirados."""
        current_time = time.time()
        expired_keys = [k for k, exp_time in self.expiration_times.items() if current_time > exp_time]
        
        if expired_keys:
            return self._evict_key(expired_keys[0])
        else:
            return self._evict_lru()  # Fallback a LRU
    
    def _evict_adaptive(self) -> bool:
        """Evictar usando política adaptativa."""
        # Combinar LRU y LFU basado en patrones de uso
        if len(self.cache) < 10:
            return self._evict_lru()
        
        # Analizar patrones de acceso
        recent_accesses = sum(1 for t in self.access_times.values() if time.time() - t < 300)  # 5 min
        total_accesses = len(self.access_times)
        
        if recent_accesses / max(total_accesses, 1) > 0.5:
            return self._evict_lru()  # Patrón temporal
        else:
            return self._evict_lfu()  # Patrón de frecuencia
    
    def _evict_oldest(self) -> bool:
        """Evictar el elemento más antiguo."""
        if self.cache:
            oldest_key = next(iter(self.cache))
            return self._evict_key(oldest_key)
        return False
    
    def _evict_key(self, key: bytes) -> bool:
        """Evictar clave específica."""
        if key in self.cache:
            self.current_size -= len(self.cache[key])
            del self.cache[key]
            
            # Limpiar estructuras de datos
            self.access_times.pop(key, None)
            self.access_counts.pop(key, None)
            self.expiration_times.pop(key, None)
            
            # Remover de insertion_order
            if key in self.insertion_order:
                self.insertion_order.remove(key)
            
            self.metrics["evictions"] += 1
            return True
        return False
    
    def delete(self, key: bytes) -> bool:
        """Eliminar clave del cache."""
        with self.lock:
            return self._evict_key(key)
    
    def exists(self, key: bytes) -> bool:
        """Verificar si existe en cache."""
        with self.lock:
            if key not in self.cache:
                return False
            
            # Verificar TTL
            if self.cache_policy == CachePolicy.TTL:
                if time.time() > self.expiration_times.get(key, 0):
                    self._evict_key(key)
                    return False
            
            return True
    
    def clear(self):
        """Limpiar todo el cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.insertion_order.clear()
            self.expiration_times.clear()
            self.current_size = 0
            self.metrics = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "size_bytes": 0,
                "entries": 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / max(total_requests, 1)) * 100
            
            return {
                "policy": self.cache_policy.value,
                "size_bytes": self.current_size,
                "max_size_bytes": self.cache_size_bytes,
                "usage_percent": (self.current_size / self.cache_size_bytes) * 100,
                "entries": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.metrics["evictions"]
            }

class DeduplicationEngine:
    """Motor de deduplicación para optimizar almacenamiento."""
    
    def __init__(self, config: MnemeConfig):
        self.config = config
        self.enabled = config.enable_deduplication
        self.content_hashes = {}  # hash -> key mapping
        self.reference_counts = {}  # key -> count
        self.lock = Lock()
    
    def should_deduplicate(self, data: bytes) -> Optional[bytes]:
        """Verificar si los datos ya existen."""
        if not self.enabled:
            return None
        
        with self.lock:
            content_hash = hashlib.sha256(data).digest()
            if content_hash in self.content_hashes:
                existing_key = self.content_hashes[content_hash]
                self.reference_counts[existing_key] += 1
                return existing_key
            return None
    
    def register_content(self, key: bytes, data: bytes) -> bytes:
        """Registrar nuevo contenido."""
        if not self.enabled:
            return key
        
        with self.lock:
            content_hash = hashlib.sha256(data).digest()
            self.content_hashes[content_hash] = key
            self.reference_counts[key] = 1
            return key
    
    def unregister_content(self, key: bytes) -> bool:
        """Desregistrar contenido."""
        if not self.enabled:
            return True
        
        with self.lock:
            if key in self.reference_counts:
                self.reference_counts[key] -= 1
                if self.reference_counts[key] <= 0:
                    del self.reference_counts[key]
                    # Limpiar hash si no hay más referencias
                    content_hash = next((h for h, k in self.content_hashes.items() if k == key), None)
                    if content_hash:
                        del self.content_hashes[content_hash]
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de deduplicación."""
        with self.lock:
            total_references = sum(self.reference_counts.values())
            unique_contents = len(self.content_hashes)
            
            return {
                "enabled": self.enabled,
                "unique_contents": unique_contents,
                "total_references": total_references,
                "deduplication_ratio": total_references / max(unique_contents, 1)
            }

class ContextAnalyzer:
    """Analizador de contexto para detectar similitudes semánticas."""
    
    def __init__(self, config: MnemeConfig):
        self.config = config
        self.similarity_method = config.context_similarity_method
        self.similarity_threshold = config.context_similarity_threshold
        self.enable_semantic = config.enable_semantic_analysis
        
        # Cache de análisis de contexto
        self.context_cache = {}
        self.similarity_cache = {}
        self.lock = Lock()
        
        # Métricas
        self.analysis_count = 0
        self.similarity_checks = 0
        self.cache_hits = 0
    
    def extract_context_features(self, tensor: torch.Tensor, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extraer características del contexto de un tensor."""
        try:
            # Características básicas del tensor
            features = {
                "shape": tensor.shape,
                "dtype": str(tensor.dtype),
                "size": tensor.numel(),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "sparsity": float((tensor == 0).float().mean().item()),
                "norm": float(tensor.norm().item())
            }
            
            # Características estadísticas avanzadas
            if tensor.numel() > 1:
                features.update({
                    "skewness": self._calculate_skewness(tensor),
                    "kurtosis": self._calculate_kurtosis(tensor),
                    "entropy": self._calculate_entropy(tensor),
                    "fractal_dimension": self._calculate_fractal_dimension(tensor)
                })
            
            # Características de distribución
            features["distribution_type"] = self._classify_distribution(tensor)
            features["pattern_type"] = self._detect_patterns(tensor)
            
            # Características semánticas si está habilitado
            if self.enable_semantic:
                features["semantic_features"] = self._extract_semantic_features(tensor)
            
            # Agregar metadatos si están disponibles
            if metadata:
                features["metadata"] = metadata
            
            return features
            
        except Exception as e:
            logger.error(f"Context feature extraction failed: {e}")
            return {"error": str(e)}
    
    def _calculate_skewness(self, tensor: torch.Tensor) -> float:
        """Calcular asimetría del tensor."""
        try:
            mean = tensor.mean()
            std = tensor.std()
            if std == 0:
                return 0.0
            skewness = ((tensor - mean) ** 3).mean() / (std ** 3)
            return float(skewness.item())
        except:
            return 0.0
    
    def _calculate_kurtosis(self, tensor: torch.Tensor) -> float:
        """Calcular curtosis del tensor."""
        try:
            mean = tensor.mean()
            std = tensor.std()
            if std == 0:
                return 0.0
            kurtosis = ((tensor - mean) ** 4).mean() / (std ** 4) - 3
            return float(kurtosis.item())
        except:
            return 0.0
    
    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """Calcular entropía del tensor."""
        try:
            # Discretizar tensor para calcular entropía
            tensor_flat = tensor.flatten()
            hist = torch.histc(tensor_flat, bins=min(256, tensor_flat.numel()))
            hist = hist[hist > 0]  # Remover bins vacíos
            prob = hist / hist.sum()
            entropy = -(prob * torch.log2(prob + 1e-10)).sum()
            return float(entropy.item())
        except:
            return 0.0
    
    def _calculate_fractal_dimension(self, tensor: torch.Tensor) -> float:
        """Calcular dimensión fractal aproximada."""
        try:
            # Implementación simplificada de box-counting
            if tensor.dim() < 2:
                return 1.0
            
            # Redimensionar a matriz cuadrada para análisis
            size = min(tensor.shape)
            tensor_square = tensor[:size, :size] if tensor.dim() == 2 else tensor[:size, :size, 0]
            
            # Box-counting simplificado
            scales = [2, 4, 8, 16]
            counts = []
            
            for scale in scales:
                if scale >= size:
                    continue
                boxes = (size // scale) ** 2
                occupied = 0
                for i in range(0, size - scale, scale):
                    for j in range(0, size - scale, scale):
                        box = tensor_square[i:i+scale, j:j+scale]
                        if box.any():
                            occupied += 1
                counts.append(occupied)
            
            if len(counts) < 2:
                return 1.0
            
            # Calcular pendiente (dimensión fractal)
            log_scales = np.log(scales[:len(counts)])
            log_counts = np.log(counts)
            slope = np.polyfit(log_scales, log_counts, 1)[0]
            return float(-slope)
            
        except:
            return 1.0
    
    def _classify_distribution(self, tensor: torch.Tensor) -> str:
        """Clasificar tipo de distribución del tensor."""
        try:
            mean = tensor.mean().item()
            std = tensor.std().item()
            skewness = self._calculate_skewness(tensor)
            kurtosis = self._calculate_kurtosis(tensor)
            
            if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
                return "normal"
            elif abs(skewness) > 1.0:
                return "skewed"
            elif kurtosis > 3.0:
                return "heavy_tailed"
            elif kurtosis < -1.0:
                return "light_tailed"
            else:
                return "mixed"
        except:
            return "unknown"
    
    def _detect_patterns(self, tensor: torch.Tensor) -> str:
        """Detectar patrones en el tensor."""
        try:
            if tensor.dim() < 2:
                return "1d"
            
            # Detectar patrones 2D
            if tensor.dim() == 2:
                # Detectar simetría
                if torch.allclose(tensor, tensor.T, atol=1e-6):
                    return "symmetric"
                
                # Detectar diagonalidad
                off_diagonal = tensor - torch.diag(torch.diag(tensor))
                if off_diagonal.norm() < tensor.norm() * 0.1:
                    return "diagonal"
                
                # Detectar sparse
                sparsity = (tensor == 0).float().mean().item()
                if sparsity > 0.9:
                    return "sparse"
                
                return "dense"
            
            # Patrones 3D+
            return "multidimensional"
            
        except:
            return "unknown"
    
    def _extract_semantic_features(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Extraer características semánticas del tensor."""
        try:
            features = {}
            
            # Análisis de componentes principales (PCA simplificado)
            if tensor.dim() >= 2 and tensor.numel() > 10:
                tensor_flat = tensor.flatten()
                if tensor_flat.std() > 0:
                    # Calcular varianza explicada por los primeros componentes
                    centered = tensor_flat - tensor_flat.mean()
                    cov_matrix = torch.outer(centered, centered)
                    eigenvals = torch.linalg.eigvals(cov_matrix).real
                    eigenvals = torch.sort(eigenvals, descending=True)[0]
                    
                    total_var = eigenvals.sum()
                    if total_var > 0:
                        explained_var = eigenvals[:min(3, len(eigenvals))].sum() / total_var
                        features["pca_variance_explained"] = float(explained_var.item())
            
            # Análisis de frecuencia (FFT simplificado)
            if tensor.numel() > 4:
                try:
                    fft = torch.fft.fft(tensor.flatten().float())
                    power_spectrum = torch.abs(fft) ** 2
                    dominant_freq = torch.argmax(power_spectrum[1:]) + 1
                    features["dominant_frequency"] = float(dominant_freq.item())
                    features["spectral_centroid"] = float(torch.sum(power_spectrum * torch.arange(len(power_spectrum))) / torch.sum(power_spectrum))
                except:
                    pass
            
            return features
            
        except Exception as e:
            logger.warning(f"Semantic feature extraction failed: {e}")
            return {}
    
    def calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calcular similitud entre dos conjuntos de características."""
        try:
            with self.lock:
                self.similarity_checks += 1
                
                # Crear clave de cache
                cache_key = (id(features1), id(features2))
                if cache_key in self.similarity_cache:
                    self.cache_hits += 1
                    return self.similarity_cache[cache_key]
            
            # Seleccionar método de similitud
            if self.similarity_method == ContextSimilarityMethod.COSINE:
                similarity = self._cosine_similarity(features1, features2)
            elif self.similarity_method == ContextSimilarityMethod.EUCLIDEAN:
                similarity = self._euclidean_similarity(features1, features2)
            elif self.similarity_method == ContextSimilarityMethod.MANHATTAN:
                similarity = self._manhattan_similarity(features1, features2)
            elif self.similarity_method == ContextSimilarityMethod.JACCARD:
                similarity = self._jaccard_similarity(features1, features2)
            elif self.similarity_method == ContextSimilarityMethod.SEMANTIC:
                similarity = self._semantic_similarity(features1, features2)
            elif self.similarity_method == ContextSimilarityMethod.HYBRID:
                similarity = self._hybrid_similarity(features1, features2)
            else:
                similarity = self._cosine_similarity(features1, features2)
            
            # Cachear resultado
            with self.lock:
                self.similarity_cache[cache_key] = similarity
                if len(self.similarity_cache) > 1000:  # Limitar tamaño del cache
                    # Eliminar entradas más antiguas
                    oldest_key = next(iter(self.similarity_cache))
                    del self.similarity_cache[oldest_key]
            
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _cosine_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calcular similitud coseno."""
        try:
            # Convertir características a vectores numéricos
            vec1 = self._features_to_vector(features1)
            vec2 = self._features_to_vector(features2)
            
            if len(vec1) == 0 or len(vec2) == 0:
                return 0.0
            
            # Normalizar vectores
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calcular similitud coseno
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(cosine_sim)
            
        except:
            return 0.0
    
    def _euclidean_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calcular similitud basada en distancia euclidiana."""
        try:
            vec1 = self._features_to_vector(features1)
            vec2 = self._features_to_vector(features2)
            
            if len(vec1) == 0 or len(vec2) == 0:
                return 0.0
            
            # Calcular distancia euclidiana
            distance = np.linalg.norm(vec1 - vec2)
            
            # Convertir a similitud (0-1)
            max_distance = np.linalg.norm(vec1) + np.linalg.norm(vec2)
            if max_distance == 0:
                return 1.0
            
            similarity = 1.0 - (distance / max_distance)
            return max(0.0, float(similarity))
            
        except:
            return 0.0
    
    def _manhattan_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calcular similitud basada en distancia Manhattan."""
        try:
            vec1 = self._features_to_vector(features1)
            vec2 = self._features_to_vector(features2)
            
            if len(vec1) == 0 or len(vec2) == 0:
                return 0.0
            
            # Calcular distancia Manhattan
            distance = np.sum(np.abs(vec1 - vec2))
            
            # Convertir a similitud
            max_distance = np.sum(np.abs(vec1)) + np.sum(np.abs(vec2))
            if max_distance == 0:
                return 1.0
            
            similarity = 1.0 - (distance / max_distance)
            return max(0.0, float(similarity))
            
        except:
            return 0.0
    
    def _jaccard_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calcular similitud Jaccard."""
        try:
            # Convertir características a conjuntos
            set1 = set(str(v) for v in features1.values() if isinstance(v, (int, float, str)))
            set2 = set(str(v) for v in features2.values() if isinstance(v, (int, float, str)))
            
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return float(intersection / union) if union > 0 else 0.0
            
        except:
            return 0.0
    
    def _semantic_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calcular similitud semántica."""
        try:
            # Combinar características semánticas
            semantic1 = features1.get("semantic_features", {})
            semantic2 = features2.get("semantic_features", {})
            
            if not semantic1 and not semantic2:
                return 0.5  # Neutral si no hay características semánticas
            
            # Calcular similitud basada en características semánticas
            similarities = []
            
            for key in set(semantic1.keys()).intersection(set(semantic2.keys())):
                val1 = semantic1[key]
                val2 = semantic2[key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Similitud numérica
                    if val1 == 0 and val2 == 0:
                        sim = 1.0
                    else:
                        sim = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2), 1e-10)
                    similarities.append(sim)
            
            return float(np.mean(similarities)) if similarities else 0.0
            
        except:
            return 0.0
    
    def _hybrid_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calcular similitud híbrida combinando múltiples métodos."""
        try:
            # Calcular similitudes con diferentes métodos
            cosine_sim = self._cosine_similarity(features1, features2)
            euclidean_sim = self._euclidean_similarity(features1, features2)
            semantic_sim = self._semantic_similarity(features1, features2)
            
            # Combinar con pesos
            weights = [0.4, 0.3, 0.3]  # Peso para coseno, euclidiano, semántico
            similarities = [cosine_sim, euclidean_sim, semantic_sim]
            
            weighted_sim = sum(w * s for w, s in zip(weights, similarities))
            return float(weighted_sim)
            
        except:
            return 0.0
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convertir características a vector numérico."""
        try:
            vector = []
            
            # Características numéricas básicas
            numeric_keys = ["mean", "std", "min", "max", "sparsity", "norm", 
                          "skewness", "kurtosis", "entropy", "fractal_dimension"]
            
            for key in numeric_keys:
                if key in features:
                    vector.append(float(features[key]))
                else:
                    vector.append(0.0)
            
            # Características categóricas (one-hot encoding)
            categorical_keys = ["distribution_type", "pattern_type"]
            categorical_values = {
                "distribution_type": ["normal", "skewed", "heavy_tailed", "light_tailed", "mixed", "unknown"],
                "pattern_type": ["1d", "symmetric", "diagonal", "sparse", "dense", "multidimensional", "unknown"]
            }
            
            for key in categorical_keys:
                if key in features:
                    value = features[key]
                    if value in categorical_values[key]:
                        # One-hot encoding
                        for cat_value in categorical_values[key]:
                            vector.append(1.0 if value == cat_value else 0.0)
                    else:
                        # Valor desconocido
                        vector.extend([0.0] * len(categorical_values[key]))
                else:
                    # Valor faltante
                    vector.extend([0.0] * len(categorical_values[key]))
            
            # Características semánticas
            semantic_features = features.get("semantic_features", {})
            for key in ["pca_variance_explained", "dominant_frequency", "spectral_centroid"]:
                if key in semantic_features:
                    vector.append(float(semantic_features[key]))
                else:
                    vector.append(0.0)
            
            return np.array(vector)
            
        except Exception as e:
            logger.warning(f"Feature vector conversion failed: {e}")
            return np.array([])
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del analizador."""
        with self.lock:
            return {
                "analysis_count": self.analysis_count,
                "similarity_checks": self.similarity_checks,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": (self.cache_hits / max(self.similarity_checks, 1)) * 100,
                "cache_size": len(self.similarity_cache),
                "method": self.similarity_method.value,
                "threshold": self.similarity_threshold
            }

class ContextClusterer:
    """Sistema de clustering para agrupar contextos similares."""
    
    def __init__(self, config: MnemeConfig):
        self.config = config
        self.clustering_method = config.context_clustering_method
        self.cluster_size = config.context_cluster_size
        self.similarity_threshold = config.context_similarity_threshold
        
        # Almacenamiento de clusters
        self.clusters = {}  # cluster_id -> {features, members, centroid}
        self.cluster_assignments = {}  # context_id -> cluster_id
        self.next_cluster_id = 0
        
        # Cache de similitudes
        self.similarity_matrix = {}
        self.lock = Lock()
        
        # Métricas
        self.clustering_operations = 0
        self.clusters_created = 0
        self.clusters_merged = 0
    
    def add_context(self, context_id: str, features: Dict[str, Any]) -> int:
        """Agregar contexto a un cluster."""
        try:
            with self.lock:
                self.clustering_operations += 1
                
                # Buscar cluster más similar
                best_cluster_id = self._find_best_cluster(features)
                
                if best_cluster_id is not None:
                    # Agregar a cluster existente
                    self.clusters[best_cluster_id]["members"].append(context_id)
                    self.cluster_assignments[context_id] = best_cluster_id
                    
                    # Actualizar centroide del cluster
                    self._update_cluster_centroid(best_cluster_id)
                    
                    return best_cluster_id
                else:
                    # Crear nuevo cluster
                    cluster_id = self._create_new_cluster(context_id, features)
                    return cluster_id
                    
        except Exception as e:
            logger.error(f"Failed to add context to cluster: {e}")
            return -1
    
    def _find_best_cluster(self, features: Dict[str, Any]) -> Optional[int]:
        """Encontrar el cluster más similar al contexto."""
        try:
            if not self.clusters:
                return None
            
            best_cluster_id = None
            best_similarity = 0.0
            
            for cluster_id, cluster_data in self.clusters.items():
                centroid = cluster_data["centroid"]
                
                # Calcular similitud con el centroide
                similarity = self._calculate_cluster_similarity(features, centroid)
                
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_id = cluster_id
            
            return best_cluster_id
            
        except Exception as e:
            logger.error(f"Failed to find best cluster: {e}")
            return None
    
    def _calculate_cluster_similarity(self, features: Dict[str, Any], centroid: Dict[str, Any]) -> float:
        """Calcular similitud entre características y centroide de cluster."""
        try:
            # Similitud coseno simplificada
            vec1 = self._features_to_vector(features)
            vec2 = self._features_to_vector(centroid)
            
            if len(vec1) == 0 or len(vec2) == 0:
                return 0.0
            
            # Normalizar vectores
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Similitud coseno
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except:
            return 0.0
    
    def _create_new_cluster(self, context_id: str, features: Dict[str, Any]) -> int:
        """Crear nuevo cluster."""
        try:
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            
            self.clusters[cluster_id] = {
                "features": features,
                "members": [context_id],
                "centroid": features.copy(),
                "created_at": time.time(),
                "size": 1
            }
            
            self.cluster_assignments[context_id] = cluster_id
            self.clusters_created += 1
            
            return cluster_id
            
        except Exception as e:
            logger.error(f"Failed to create new cluster: {e}")
            return -1
    
    def _update_cluster_centroid(self, cluster_id: int):
        """Actualizar centroide del cluster."""
        try:
            cluster_data = self.clusters[cluster_id]
            members = cluster_data["members"]
            
            if len(members) == 0:
                return
            
            # Calcular nuevo centroide como promedio de características
            if len(members) == 1:
                return  # No hay nada que promediar
            
            # Obtener características de todos los miembros
            member_features = []
            for member_id in members:
                if member_id in self.cluster_assignments:
                    # Aquí necesitaríamos acceso a las características originales
                    # Por simplicidad, usamos las características del cluster
                    member_features.append(cluster_data["features"])
            
            if not member_features:
                return
            
            # Calcular promedio de características numéricas
            numeric_keys = ["mean", "std", "min", "max", "sparsity", "norm", 
                          "skewness", "kurtosis", "entropy", "fractal_dimension"]
            
            new_centroid = {}
            for key in numeric_keys:
                values = [f.get(key, 0) for f in member_features if key in f]
                if values:
                    new_centroid[key] = np.mean(values)
            
            # Mantener características categóricas del primer miembro
            categorical_keys = ["distribution_type", "pattern_type", "shape", "dtype"]
            for key in categorical_keys:
                if key in cluster_data["features"]:
                    new_centroid[key] = cluster_data["features"][key]
            
            cluster_data["centroid"] = new_centroid
            cluster_data["size"] = len(members)
            
        except Exception as e:
            logger.error(f"Failed to update cluster centroid: {e}")
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convertir características a vector numérico."""
        try:
            vector = []
            
            # Características numéricas
            numeric_keys = ["mean", "std", "min", "max", "sparsity", "norm", 
                          "skewness", "kurtosis", "entropy", "fractal_dimension"]
            
            for key in numeric_keys:
                if key in features:
                    vector.append(float(features[key]))
                else:
                    vector.append(0.0)
            
            # Características semánticas
            semantic_features = features.get("semantic_features", {})
            for key in ["pca_variance_explained", "dominant_frequency", "spectral_centroid"]:
                if key in semantic_features:
                    vector.append(float(semantic_features[key]))
                else:
                    vector.append(0.0)
            
            return np.array(vector)
            
        except Exception as e:
            logger.warning(f"Feature vector conversion failed: {e}")
            return np.array([])
    
    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """Obtener miembros de un cluster."""
        with self.lock:
            return self.clusters.get(cluster_id, {}).get("members", [])
    
    def get_context_cluster(self, context_id: str) -> Optional[int]:
        """Obtener cluster de un contexto."""
        with self.lock:
            return self.cluster_assignments.get(context_id)
    
    def merge_similar_clusters(self, similarity_threshold: float = 0.9):
        """Fusionar clusters similares."""
        try:
            with self.lock:
                clusters_to_merge = []
                cluster_ids = list(self.clusters.keys())
                
                for i, cluster_id1 in enumerate(cluster_ids):
                    for cluster_id2 in cluster_ids[i+1:]:
                        similarity = self._calculate_cluster_similarity(
                            self.clusters[cluster_id1]["centroid"],
                            self.clusters[cluster_id2]["centroid"]
                        )
                        
                        if similarity > similarity_threshold:
                            clusters_to_merge.append((cluster_id1, cluster_id2))
                
                # Fusionar clusters
                for cluster_id1, cluster_id2 in clusters_to_merge:
                    if cluster_id1 in self.clusters and cluster_id2 in self.clusters:
                        self._merge_clusters(cluster_id1, cluster_id2)
                        self.clusters_merged += 1
                        
        except Exception as e:
            logger.error(f"Failed to merge similar clusters: {e}")
    
    def _merge_clusters(self, cluster_id1: int, cluster_id2: int):
        """Fusionar dos clusters."""
        try:
            cluster1 = self.clusters[cluster_id1]
            cluster2 = self.clusters[cluster_id2]
            
            # Combinar miembros
            all_members = cluster1["members"] + cluster2["members"]
            
            # Actualizar asignaciones
            for member in cluster2["members"]:
                self.cluster_assignments[member] = cluster_id1
            
            # Actualizar cluster1
            cluster1["members"] = all_members
            cluster1["size"] = len(all_members)
            
            # Eliminar cluster2
            del self.clusters[cluster_id2]
            
        except Exception as e:
            logger.error(f"Failed to merge clusters: {e}")
    
    def optimize_clusters(self):
        """Optimizar clusters eliminando clusters pequeños y fusionando similares."""
        try:
            with self.lock:
                # Eliminar clusters pequeños
                small_clusters = [
                    cluster_id for cluster_id, data in self.clusters.items()
                    if data["size"] < self.cluster_size
                ]
                
                for cluster_id in small_clusters:
                    self._remove_cluster(cluster_id)
                
                # Fusionar clusters similares
                self.merge_similar_clusters()
                
        except Exception as e:
            logger.error(f"Failed to optimize clusters: {e}")
    
    def _remove_cluster(self, cluster_id: int):
        """Eliminar cluster."""
        try:
            if cluster_id in self.clusters:
                # Reasignar miembros a clusters existentes o crear nuevos
                members = self.clusters[cluster_id]["members"]
                
                for member in members:
                    if member in self.cluster_assignments:
                        del self.cluster_assignments[member]
                
                del self.clusters[cluster_id]
                
        except Exception as e:
            logger.error(f"Failed to remove cluster: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del clusterer."""
        with self.lock:
            total_members = sum(data["size"] for data in self.clusters.values())
            avg_cluster_size = total_members / max(len(self.clusters), 1)
            
            return {
                "total_clusters": len(self.clusters),
                "total_members": total_members,
                "avg_cluster_size": avg_cluster_size,
                "clustering_operations": self.clustering_operations,
                "clusters_created": self.clusters_created,
                "clusters_merged": self.clusters_merged,
                "method": self.clustering_method.value,
                "threshold": self.similarity_threshold
            }

class ContextDeduplicationEngine:
    """Motor principal de deduplicación de contexto."""
    
    def __init__(self, config: MnemeConfig):
        self.config = config
        self.enabled = config.enable_context_deduplication
        
        # Componentes del sistema
        self.analyzer = ContextAnalyzer(config)
        self.clusterer = ContextClusterer(config)
        
        # Almacenamiento de contextos
        self.context_features = {}  # context_id -> features
        self.context_clusters = {}  # context_id -> cluster_id
        self.compressed_contexts = {}  # context_id -> compressed_data
        
        # Cache de contexto
        self.context_cache = {}
        self.cache_size_bytes = config.context_cache_size_mb * 1024 * 1024
        self.current_cache_size = 0
        
        # Métricas
        self.deduplication_saves = 0
        self.compression_ratios = []
        self.similarity_checks = 0
        self.lock = Lock()
    
    def process_context(self, context_id: str, tensor: torch.Tensor, 
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Procesar contexto para deduplicación."""
        try:
            if not self.enabled:
                return {"context_id": context_id, "deduplicated": False}
            
            with self.lock:
                # Extraer características del contexto
                features = self.analyzer.extract_context_features(tensor, metadata)
                self.context_features[context_id] = features
                
                # Buscar contexto similar
                similar_context = self._find_similar_context(features)
                
                if similar_context:
                    # Contexto duplicado encontrado
                    self.deduplication_saves += 1
                    self.context_clusters[context_id] = similar_context
                    
                    return {
                        "context_id": context_id,
                        "deduplicated": True,
                        "similar_context": similar_context,
                        "similarity": self._calculate_similarity_score(features, similar_context)
                    }
                else:
                    # Nuevo contexto, agregar a cluster
                    cluster_id = self.clusterer.add_context(context_id, features)
                    self.context_clusters[context_id] = cluster_id
                    
                    # Comprimir contexto
                    compressed_data = self._compress_context(tensor, features)
                    self.compressed_contexts[context_id] = compressed_data
                    
                    return {
                        "context_id": context_id,
                        "deduplicated": False,
                        "cluster_id": cluster_id,
                        "compression_ratio": compressed_data.get("compression_ratio", 1.0)
                    }
                    
        except Exception as e:
            logger.error(f"Context processing failed: {e}")
            return {"context_id": context_id, "error": str(e)}
    
    def _find_similar_context(self, features: Dict[str, Any]) -> Optional[str]:
        """Buscar contexto similar."""
        try:
            best_similarity = 0.0
            best_context = None
            
            for context_id, stored_features in self.context_features.items():
                similarity = self.analyzer.calculate_similarity(features, stored_features)
                
                if similarity > self.config.context_similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_context = context_id
                
                self.similarity_checks += 1
            
            return best_context
            
        except Exception as e:
            logger.error(f"Failed to find similar context: {e}")
            return None
    
    def _calculate_similarity_score(self, features1: Dict[str, Any], context_id: str) -> float:
        """Calcular puntuación de similitud."""
        try:
            if context_id in self.context_features:
                features2 = self.context_features[context_id]
                return self.analyzer.calculate_similarity(features1, features2)
            return 0.0
        except:
            return 0.0
    
    def _compress_context(self, tensor: torch.Tensor, features: Dict[str, Any]) -> Dict[str, Any]:
        """Comprimir contexto."""
        try:
            # Serializar tensor
            tensor_bytes = self.analyzer._features_to_vector(features).tobytes()
            
            # Comprimir usando LZ4
            compressed_bytes = lz4.frame.compress(
                tensor_bytes, 
                compression_level=self.config.context_compression_level
            )
            
            compression_ratio = len(compressed_bytes) / max(len(tensor_bytes), 1)
            self.compression_ratios.append(compression_ratio)
            
            return {
                "compressed_data": compressed_bytes,
                "compression_ratio": compression_ratio,
                "original_size": len(tensor_bytes),
                "compressed_size": len(compressed_bytes)
            }
            
        except Exception as e:
            logger.error(f"Context compression failed: {e}")
            return {"error": str(e)}
    
    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Obtener contexto procesado."""
        try:
            with self.lock:
                if context_id in self.context_clusters:
                    cluster_id = self.context_clusters[context_id]
                    
                    # Si está en cache, devolver desde cache
                    if context_id in self.context_cache:
                        return self.context_cache[context_id]
                    
                    # Buscar contexto similar en el cluster
                    cluster_members = self.clusterer.get_cluster_members(cluster_id)
                    for member in cluster_members:
                        if member != context_id and member in self.context_features:
                            # Reconstruir contexto desde características similares
                            similar_features = self.context_features[member]
                            reconstructed_context = self._reconstruct_context(similar_features)
                            
                            # Cachear resultado
                            self._cache_context(context_id, reconstructed_context)
                            
                            return reconstructed_context
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return None
    
    def _reconstruct_context(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruir contexto desde características."""
        try:
            # Esta es una implementación simplificada
            # En un sistema real, se usarían técnicas más sofisticadas
            return {
                "reconstructed": True,
                "features": features,
                "confidence": 0.8  # Confianza en la reconstrucción
            }
        except:
            return {"reconstructed": False}
    
    def _cache_context(self, context_id: str, context_data: Dict[str, Any]):
        """Cachear contexto."""
        try:
            if self.current_cache_size < self.cache_size_bytes:
                self.context_cache[context_id] = context_data
                self.current_cache_size += len(str(context_data))
        except:
            pass
    
    def optimize_contexts(self):
        """Optimizar contextos."""
        try:
            with self.lock:
                # Optimizar clusters
                self.clusterer.optimize_clusters()
                
                # Limpiar cache si es necesario
                if self.current_cache_size > self.cache_size_bytes:
                    self._cleanup_cache()
                
                # Limpiar contextos antiguos
                self._cleanup_old_contexts()
                
        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
    
    def _cleanup_cache(self):
        """Limpiar cache de contextos."""
        try:
            # Eliminar entradas más antiguas
            if len(self.context_cache) > 100:  # Límite de entradas
                oldest_keys = list(self.context_cache.keys())[:50]
                for key in oldest_keys:
                    del self.context_cache[key]
                    self.current_cache_size = max(0, self.current_cache_size - 1000)
        except:
            pass
    
    def _cleanup_old_contexts(self):
        """Limpiar contextos antiguos."""
        try:
            # Implementación simplificada - en producción sería más sofisticada
            if len(self.context_features) > 1000:
                keys_to_remove = list(self.context_features.keys())[:100]
                for key in keys_to_remove:
                    if key in self.context_features:
                        del self.context_features[key]
                    if key in self.context_clusters:
                        del self.context_clusters[key]
                    if key in self.compressed_contexts:
                        del self.compressed_contexts[key]
        except:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de deduplicación de contexto."""
        try:
            with self.lock:
                total_contexts = len(self.context_features)
                deduplicated_contexts = len(self.context_clusters)
                deduplication_rate = deduplicated_contexts / max(total_contexts, 1)
                
                avg_compression = np.mean(self.compression_ratios) if self.compression_ratios else 1.0
                
                return {
                    "enabled": self.enabled,
                    "total_contexts": total_contexts,
                    "deduplicated_contexts": deduplicated_contexts,
                    "deduplication_rate": deduplication_rate,
                    "deduplication_saves": self.deduplication_saves,
                    "similarity_checks": self.similarity_checks,
                    "avg_compression_ratio": avg_compression,
                    "cache_size_bytes": self.current_cache_size,
                    "cache_entries": len(self.context_cache),
                    "analyzer_stats": self.analyzer.get_stats(),
                    "clusterer_stats": self.clusterer.get_stats()
                }
                
        except Exception as e:
            logger.error(f"Failed to get context deduplication stats: {e}")
            return {"error": str(e)}

# --- Cifrado de Tensores y Gestión de Claves ---

class TensorEncryptor:
    """Cifrado especializado para tensores con optimizaciones específicas."""
    
    def __init__(self, config: MnemeConfig):
        self.config = config
        self.encryption_mode = config.tensor_encryption_mode
        self.encryption_key = config.tensor_encryption_key
        self.backend = default_backend()
        
        if not self.encryption_key:
            self.encryption_key = self._generate_key()
    
    def _generate_key(self) -> bytes:
        """Generar clave de cifrado segura."""
        if self.encryption_mode == TensorEncryptionMode.AES_GCM:
            return secrets.token_bytes(32)  # 256 bits
        elif self.encryption_mode == TensorEncryptionMode.AES_CBC:
            return secrets.token_bytes(32)  # 256 bits
        elif self.encryption_mode == TensorEncryptionMode.CHACHA20:
            return secrets.token_bytes(32)  # 256 bits
        else:
            return secrets.token_bytes(32)
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Cifrar tensor usando el modo configurado."""
        try:
            # Convertir tensor a bytes
            tensor_bytes = self._tensor_to_bytes(tensor)
            
            if self.encryption_mode == TensorEncryptionMode.AES_GCM:
                return self._encrypt_aes_gcm(tensor_bytes, tensor.shape, tensor.dtype)
            elif self.encryption_mode == TensorEncryptionMode.AES_CBC:
                return self._encrypt_aes_cbc(tensor_bytes, tensor.shape, tensor.dtype)
            elif self.encryption_mode == TensorEncryptionMode.CHACHA20:
                return self._encrypt_chacha20(tensor_bytes, tensor.shape, tensor.dtype)
            elif self.encryption_mode == TensorEncryptionMode.BLOCK_CHAIN:
                return self._encrypt_block_chain(tensor_bytes, tensor.shape, tensor.dtype)
            else:
                raise ValueError(f"Unsupported encryption mode: {self.encryption_mode}")
                
        except Exception as e:
            logger.error(f"Tensor encryption failed: {e}")
            raise
    
    def decrypt_tensor(self, encrypted_data: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """Descifrar tensor."""
        try:
            encryption_mode = encrypted_data.get('mode', self.encryption_mode.value)
            
            if encryption_mode == TensorEncryptionMode.AES_GCM.value:
                tensor_bytes = self._decrypt_aes_gcm(encrypted_data)
            elif encryption_mode == TensorEncryptionMode.AES_CBC.value:
                tensor_bytes = self._decrypt_aes_cbc(encrypted_data)
            elif encryption_mode == TensorEncryptionMode.CHACHA20.value:
                tensor_bytes = self._decrypt_chacha20(encrypted_data)
            elif encryption_mode == TensorEncryptionMode.BLOCK_CHAIN.value:
                tensor_bytes = self._decrypt_block_chain(encrypted_data)
            else:
                raise ValueError(f"Unsupported encryption mode: {encryption_mode}")
            
            # Reconstruir tensor
            return self._bytes_to_tensor(tensor_bytes, encrypted_data['shape'], 
                                       encrypted_data['dtype'], device)
            
        except Exception as e:
            logger.error(f"Tensor decryption failed: {e}")
            raise
    
    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convertir tensor a bytes de forma eficiente."""
        # Usar numpy para conversión más rápida
        numpy_array = tensor.detach().cpu().numpy()
        return numpy_array.tobytes()
    
    def _bytes_to_tensor(self, data: bytes, shape: Tuple[int, ...], 
                        dtype: str, device: torch.device) -> torch.Tensor:
        """Reconstruir tensor desde bytes."""
        numpy_array = np.frombuffer(data, dtype=dtype).reshape(shape)
        return torch.from_numpy(numpy_array).to(device)
    
    def _encrypt_aes_gcm(self, data: bytes, shape: Tuple[int, ...], dtype: str) -> Dict[str, Any]:
        """Cifrado AES-GCM (autenticado)."""
        # Generar IV aleatorio
        iv = secrets.token_bytes(12)  # 96 bits para GCM
        
        # Crear cipher
        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.GCM(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        # Cifrar datos
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return {
            'mode': TensorEncryptionMode.AES_GCM.value,
            'ciphertext': ciphertext,
            'tag': encryptor.tag,
            'iv': iv,
            'shape': shape,
            'dtype': dtype
        }
    
    def _decrypt_aes_gcm(self, encrypted_data: Dict[str, Any]) -> bytes:
        """Descifrado AES-GCM."""
        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.GCM(encrypted_data['iv'], encrypted_data['tag']),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        return decryptor.update(encrypted_data['ciphertext']) + decryptor.finalize()
    
    def _encrypt_aes_cbc(self, data: bytes, shape: Tuple[int, ...], dtype: str) -> Dict[str, Any]:
        """Cifrado AES-CBC."""
        # Padding para CBC
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        # Generar IV aleatorio
        iv = secrets.token_bytes(16)
        
        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return {
            'mode': TensorEncryptionMode.AES_CBC.value,
            'ciphertext': ciphertext,
            'iv': iv,
            'shape': shape,
            'dtype': dtype
        }
    
    def _decrypt_aes_cbc(self, encrypted_data: Dict[str, Any]) -> bytes:
        """Descifrado AES-CBC."""
        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.CBC(encrypted_data['iv']),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(encrypted_data['ciphertext']) + decryptor.finalize()
        
        # Remover padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def _encrypt_chacha20(self, data: bytes, shape: Tuple[int, ...], dtype: str) -> Dict[str, Any]:
        """Cifrado ChaCha20."""
        # Generar nonce aleatorio
        nonce = secrets.token_bytes(12)
        
        cipher = Cipher(
            algorithms.ChaCha20(self.encryption_key, nonce),
            mode=None,
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return {
            'mode': TensorEncryptionMode.CHACHA20.value,
            'ciphertext': ciphertext,
            'nonce': nonce,
            'shape': shape,
            'dtype': dtype
        }
    
    def _decrypt_chacha20(self, encrypted_data: Dict[str, Any]) -> bytes:
        """Descifrado ChaCha20."""
        cipher = Cipher(
            algorithms.ChaCha20(self.encryption_key, encrypted_data['nonce']),
            mode=None,
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        return decryptor.update(encrypted_data['ciphertext']) + decryptor.finalize()
    
    def _encrypt_block_chain(self, data: bytes, shape: Tuple[int, ...], dtype: str) -> Dict[str, Any]:
        """Cifrado por bloques para tensores grandes."""
        block_size = 1024 * 1024  # 1MB por bloque
        blocks = []
        
        for i in range(0, len(data), block_size):
            block = data[i:i + block_size]
            
            # Usar AES-GCM para cada bloque
            iv = secrets.token_bytes(12)
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.GCM(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(block) + encryptor.finalize()
            
            blocks.append({
                'ciphertext': ciphertext,
                'tag': encryptor.tag,
                'iv': iv
            })
        
        return {
            'mode': TensorEncryptionMode.BLOCK_CHAIN.value,
            'blocks': blocks,
            'shape': shape,
            'dtype': dtype
        }
    
    def _decrypt_block_chain(self, encrypted_data: Dict[str, Any]) -> bytes:
        """Descifrado por bloques."""
        decrypted_blocks = []
        
        for block_data in encrypted_data['blocks']:
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.GCM(block_data['iv'], block_data['tag']),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            decrypted_block = decryptor.update(block_data['ciphertext']) + decryptor.finalize()
            decrypted_blocks.append(decrypted_block)
        
        return b''.join(decrypted_blocks)

class KeyManager:
    """Gestión avanzada de claves con rotación automática."""
    
    def __init__(self, config: MnemeConfig):
        self.config = config
        self.current_key = config.secret_key
        self.key_version = 1
        self.key_history: Dict[int, bytes] = {}
        self.usage_count = 0
        self.last_rotation = datetime.now()
        self.rotation_lock = Lock()
        
        if self.current_key:
            self.key_history[self.key_version] = self.current_key
    
    def get_current_key(self) -> bytes:
        """Obtener clave actual."""
        if not self.current_key:
            self._generate_new_key()
        return self.current_key
    
    def _generate_new_key(self) -> bytes:
        """Generar nueva clave."""
        new_key = secrets.token_bytes(32)
        with self.rotation_lock:
            self.current_key = new_key
            self.key_version += 1
            self.key_history[self.key_version] = new_key
            self.last_rotation = datetime.now()
            self.usage_count = 0
        return new_key
    
    def should_rotate_key(self) -> bool:
        """Determinar si se debe rotar la clave."""
        if self.config.key_rotation_policy == KeyRotationPolicy.NEVER:
            return False
        elif self.config.key_rotation_policy == KeyRotationPolicy.TIME_BASED:
            return datetime.now() - self.last_rotation > self.config.key_rotation_interval
        elif self.config.key_rotation_policy == KeyRotationPolicy.USAGE_BASED:
            return self.usage_count >= self.config.key_rotation_usage_count
        elif self.config.key_rotation_policy == KeyRotationPolicy.ADAPTIVE:
            # Lógica adaptativa basada en patrones de uso
            return self._adaptive_rotation_check()
        return False
    
    def _adaptive_rotation_check(self) -> bool:
        """Verificación adaptativa de rotación."""
        # Implementar lógica adaptativa basada en patrones de uso
        # Por ahora, usar una combinación de tiempo y uso
        time_based = datetime.now() - self.last_rotation > self.config.key_rotation_interval
        usage_based = self.usage_count >= self.config.key_rotation_usage_count
        return time_based or usage_based
    
    def rotate_key(self) -> bytes:
        """Rotar clave manualmente."""
        with self.rotation_lock:
            old_key = self.current_key
            new_key = self._generate_new_key()
            logger.info(f"Key rotated from version {self.key_version - 1} to {self.key_version}")
            return new_key
    
    def increment_usage(self):
        """Incrementar contador de uso."""
        with self.rotation_lock:
            self.usage_count += 1
            
            # Rotación automática si es necesario
            if self.should_rotate_key():
                self.rotate_key()
    
    def get_key_by_version(self, version: int) -> Optional[bytes]:
        """Obtener clave por versión."""
        return self.key_history.get(version)
    
    def cleanup_old_keys(self, keep_versions: int = 5):
        """Limpiar claves antiguas."""
        with self.rotation_lock:
            if len(self.key_history) > keep_versions:
                oldest_version = min(self.key_history.keys())
                del self.key_history[oldest_version]
                logger.info(f"Cleaned up old key version {oldest_version}")

# --- Serialización Avanzada y Segura ---

class AdvancedSerializer:
    """
    Sistema de serialización avanzado y seguro que reemplaza pickle.
    Soporta múltiples formatos, compresión, cifrado y validación de integridad.
    """

    def __init__(self, config: MnemeConfig):
        self.config = config
        self.secret_key = config.secret_key
        self.security_level = config.security_level
        self.serialization_format = config.serialization_format
        self.enable_encryption = config.enable_encryption
        self.encryption_password = config.encryption_password
        self.enable_compression = config.enable_compression
        self.enable_validation = config.enable_validation
        
        # Inicializar cifrado si está habilitado
        self._fernet = None
        if self.enable_encryption and self.encryption_password:
            self._fernet = self._create_fernet_key()
        
        # Inicializar cifrado de tensores
        self.tensor_encryptor = None
        if config.enable_tensor_encryption:
            self.tensor_encryptor = TensorEncryptor(config)
        
        # Inicializar gestión de claves
        self.key_manager = KeyManager(config)

    def _create_fernet_key(self) -> Fernet:
        """Crear clave Fernet para cifrado simétrico."""
        password = self.encryption_password.encode()
        salt = b'mneme_salt_2024'  # Salt fijo para consistencia
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)

    def serialize(self, data: Any) -> bytes:
        """
        Serializa datos usando el formato y nivel de seguridad configurados.
        """
        try:
            # 1. Verificar rotación de claves
            if hasattr(self, 'key_manager'):
                self.key_manager.increment_usage()
            
            # 2. Preparar datos (mover a CPU)
            prepared_data = self._prepare_data(data)
            
            # 3. Serializar según el formato
            serialized_data = self._serialize_by_format(prepared_data)
            
            # 4. Comprimir si está habilitado
            if self.enable_compression:
                serialized_data = self._compress_data(serialized_data)
            
            # 5. Aplicar seguridad
            secured_data = self._apply_security(serialized_data)
            
            # 6. Agregar metadatos de validación
            if self.enable_validation:
                secured_data = self._add_validation_metadata(secured_data, data)
            
            return secured_data
            
        except Exception as e:
            logger.error(f"Advanced serialization failed: {e}")
            raise
        
    def deserialize(self, data: bytes, device: torch.device) -> Any:
        """
        Deserializa datos verificando integridad y aplicando transformaciones inversas.
        """
        try:
            # 1. Verificar metadatos de validación
            if self.enable_validation:
                data, original_type = self._verify_validation_metadata(data)
            
            # 2. Aplicar seguridad inversa
            serialized_data = self._remove_security(data)
            
            # 3. Descomprimir si fue comprimido
            if self.enable_compression:
                serialized_data = self._decompress_data(serialized_data)
            
            # 4. Deserializar según el formato
            deserialized_data = self._deserialize_by_format(serialized_data, device)
            
            # 5. Validar tipo si está habilitado
            if self.enable_validation:
                self._validate_data_type(deserialized_data, original_type)
            
            return deserialized_data
            
        except Exception as e:
            logger.error(f"Advanced deserialization failed: {e}")
            raise

    def _prepare_data(self, data: Any) -> Any:
        """Preparar datos moviendo tensores a CPU y optimizando estructura."""
        return self._move_to_cpu(data)

    def _move_to_cpu(self, data: Any) -> Any:
        """Mueve recursivamente tensores a CPU y aplica cifrado si está habilitado."""
        if isinstance(data, torch.Tensor):
            tensor_cpu = data.detach().cpu()
            
            # Aplicar cifrado de tensor si está habilitado
            if self.tensor_encryptor:
                try:
                    encrypted_tensor = self.tensor_encryptor.encrypt_tensor(tensor_cpu)
                    # Marcar como cifrado para identificación
                    encrypted_tensor['_encrypted_tensor'] = True
                    return encrypted_tensor
                except Exception as e:
                    logger.warning(f"Tensor encryption failed, using plain tensor: {e}")
                    return tensor_cpu
            
            return tensor_cpu
        elif isinstance(data, dict):
            return {k: self._move_to_cpu(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._move_to_cpu(v) for v in data)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).cpu()
        return data

    def _serialize_by_format(self, data: Any) -> bytes:
        """Serializar según el formato configurado."""
        if self.serialization_format == SerializationFormat.TORCH:
            return self._serialize_torch(data)
        elif self.serialization_format == SerializationFormat.MSGPACK:
            return self._serialize_msgpack(data)
        elif self.serialization_format == SerializationFormat.JSON:
            return self._serialize_json(data)
        elif self.serialization_format == SerializationFormat.BINARY:
            return self._serialize_binary(data)
        elif self.serialization_format == SerializationFormat.HYBRID:
            return self._serialize_hybrid(data)
        else:
            raise ValueError(f"Unsupported serialization format: {self.serialization_format}")

    def _deserialize_by_format(self, data: bytes, device: torch.device) -> Any:
        """Deserializar según el formato."""
        if self.serialization_format == SerializationFormat.TORCH:
            return self._deserialize_torch(data, device)
        elif self.serialization_format == SerializationFormat.MSGPACK:
            return self._deserialize_msgpack(data, device)
        elif self.serialization_format == SerializationFormat.JSON:
            return self._deserialize_json(data, device)
        elif self.serialization_format == SerializationFormat.BINARY:
            return self._deserialize_binary(data, device)
        elif self.serialization_format == SerializationFormat.HYBRID:
            return self._deserialize_hybrid(data, device)
        else:
            raise ValueError(f"Unsupported serialization format: {self.serialization_format}")

    def _serialize_torch(self, data: Any) -> bytes:
        """Serialización usando torch.save (más segura que pickle)."""
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    def _deserialize_torch(self, data: bytes, device: torch.device) -> Any:
        """Deserialización usando torch.load."""
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location=device)

    def _serialize_msgpack(self, data: Any) -> bytes:
        """Serialización usando MessagePack (más rápida que JSON)."""
        # Convertir tensores a arrays numpy para msgpack
        converted_data = self._convert_tensors_for_msgpack(data)
        return msgpack.packb(converted_data, use_bin_type=True)

    def _deserialize_msgpack(self, data: bytes, device: torch.device) -> Any:
        """Deserialización usando MessagePack."""
        unpacked = msgpack.unpackb(data, raw=False)
        return self._convert_arrays_to_tensors(unpacked, device)

    def _serialize_json(self, data: Any) -> bytes:
        """Serialización usando JSON (para datos simples)."""
        converted_data = self._convert_for_json(data)
        return json.dumps(converted_data, separators=(',', ':')).encode('utf-8')

    def _deserialize_json(self, data: bytes, device: torch.device) -> Any:
        """Deserialización usando JSON."""
        json_data = json.loads(data.decode('utf-8'))
        return self._convert_from_json(json_data, device)

    def _serialize_binary(self, data: Any) -> bytes:
        """Serialización binaria personalizada."""
        return self._convert_to_binary(data)

    def _deserialize_binary(self, data: bytes, device: torch.device) -> Any:
        """Deserialización binaria personalizada."""
        return self._convert_from_binary(data, device)

    def _serialize_hybrid(self, data: Any) -> bytes:
        """Serialización híbrida que elige el mejor formato automáticamente."""
        # Analizar el tipo de datos y elegir el formato óptimo
        if self._is_tensor_heavy(data):
            return self._serialize_torch(data)
        elif self._is_simple_data(data):
            return self._serialize_msgpack(data)
        else:
            return self._serialize_torch(data)

    def _deserialize_hybrid(self, data: bytes, device: torch.device) -> Any:
        """Deserialización híbrida."""
        # Detectar el formato usado y deserializar apropiadamente
        if data.startswith(b'PK'):  # Formato torch
            return self._deserialize_torch(data, device)
        else:
            return self._deserialize_msgpack(data, device)

    def _is_tensor_heavy(self, data: Any) -> bool:
        """Detectar si los datos contienen principalmente tensores."""
        if isinstance(data, torch.Tensor):
            return True
        elif isinstance(data, dict):
            return any(self._is_tensor_heavy(v) for v in data.values())
        elif isinstance(data, (list, tuple)):
            return any(self._is_tensor_heavy(v) for v in data)
        return False

    def _is_simple_data(self, data: Any) -> bool:
        """Detectar si los datos son simples (números, strings, etc.)."""
        if isinstance(data, (int, float, str, bool, type(None))):
            return True
        elif isinstance(data, (list, tuple)):
            return all(self._is_simple_data(v) for v in data)
        elif isinstance(data, dict):
            return all(isinstance(k, str) and self._is_simple_data(v) for k, v in data.items())
        return False

    def _convert_tensors_for_msgpack(self, data: Any) -> Any:
        """Convertir tensores a formato compatible con msgpack."""
        if isinstance(data, torch.Tensor):
            return {
                '_tensor': True,
                'data': data.numpy().tolist(),
                'shape': list(data.shape),
                'dtype': str(data.dtype)
            }
        elif isinstance(data, dict):
            return {k: self._convert_tensors_for_msgpack(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._convert_tensors_for_msgpack(v) for v in data)
        return data

    def _convert_arrays_to_tensors(self, data: Any, device: torch.device) -> Any:
        """Convertir arrays de msgpack de vuelta a tensores."""
        if isinstance(data, dict) and data.get('_tensor'):
            tensor_data = np.array(data['data'], dtype=data.get('dtype', 'float32'))
            return torch.from_numpy(tensor_data).to(device)
        elif isinstance(data, dict) and data.get('_encrypted_tensor'):
            # Descifrar tensor si está cifrado
            if self.tensor_encryptor:
                try:
                    return self.tensor_encryptor.decrypt_tensor(data, device)
                except Exception as e:
                    logger.warning(f"Tensor decryption failed: {e}")
                    # Fallback a tensor plano si el descifrado falla
                    if 'shape' in data and 'dtype' in data:
                        tensor_data = np.array(data.get('data', []), dtype=data.get('dtype', 'float32'))
                        return torch.from_numpy(tensor_data).to(device)
            return data
        elif isinstance(data, dict):
            return {k: self._convert_arrays_to_tensors(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._convert_arrays_to_tensors(v, device) for v in data)
        return data

    def _convert_for_json(self, data: Any) -> Any:
        """Convertir datos para serialización JSON."""
        if isinstance(data, torch.Tensor):
            return {
                '_tensor': True,
                'data': data.numpy().tolist(),
                'shape': list(data.shape),
                'dtype': str(data.dtype)
            }
        elif isinstance(data, np.ndarray):
            return {
                '_ndarray': True,
                'data': data.tolist(),
                'shape': list(data.shape),
                'dtype': str(data.dtype)
            }
        elif isinstance(data, dict):
            return {k: self._convert_for_json(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._convert_for_json(v) for v in data)
        return data

    def _convert_from_json(self, data: Any, device: torch.device) -> Any:
        """Convertir datos desde JSON."""
        if isinstance(data, dict):
            if data.get('_tensor'):
                tensor_data = np.array(data['data'], dtype=data.get('dtype', 'float32'))
                return torch.from_numpy(tensor_data).to(device)
            elif data.get('_ndarray'):
                return np.array(data['data'], dtype=data.get('dtype', 'float32'))
            else:
                return {k: self._convert_from_json(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._convert_from_json(v, device) for v in data)
        return data

    def _convert_to_binary(self, data: Any) -> bytes:
        """Conversión binaria personalizada."""
        # Implementación simplificada - en producción sería más robusta
        if isinstance(data, torch.Tensor):
            return data.numpy().tobytes()
        elif isinstance(data, np.ndarray):
            return data.tobytes()
        else:
            return str(data).encode('utf-8')

    def _convert_from_binary(self, data: bytes, device: torch.device) -> Any:
        """Conversión desde binario."""
        # Implementación simplificada
        try:
            array = np.frombuffer(data, dtype=np.float32)
            return torch.from_numpy(array).to(device)
        except:
            return data.decode('utf-8')

    def _compress_data(self, data: bytes) -> bytes:
        """Comprimir datos usando LZ4."""
        try:
            return lz4.frame.compress(data, compression_level=self.config.compression_level.value)
        except Exception as e:
            logger.warning(f"Compression failed, using uncompressed data: {e}")
            return data

    def _decompress_data(self, data: bytes) -> bytes:
        """Descomprimir datos."""
        try:
            return lz4.frame.decompress(data)
        except Exception as e:
            logger.warning(f"Decompression failed, assuming uncompressed data: {e}")
            return data

    def _apply_security(self, data: bytes) -> bytes:
        """Aplicar medidas de seguridad según el nivel configurado."""
        if self.security_level == SecurityLevel.NONE:
            return data
        elif self.security_level == SecurityLevel.HMAC:
            return self._sign_data(data)
        elif self.security_level == SecurityLevel.ENCRYPTED:
            return self._encrypt_data(data)
        elif self.security_level == SecurityLevel.SIGNED:
            return self._sign_data(data)  # Similar a HMAC pero con firma digital
        else:
            return data

    def _remove_security(self, data: bytes) -> bytes:
        """Remover medidas de seguridad."""
        if self.security_level == SecurityLevel.NONE:
            return data
        elif self.security_level == SecurityLevel.HMAC:
            return self._verify_and_extract_data(data)
        elif self.security_level == SecurityLevel.ENCRYPTED:
            return self._decrypt_data(data)
        elif self.security_level == SecurityLevel.SIGNED:
            return self._verify_and_extract_data(data)
        else:
            return data

    def _sign_data(self, data: bytes) -> bytes:
        """Firmar datos usando HMAC-SHA256."""
        if not self.secret_key:
            raise SecurityError("Secret key required for HMAC signing")
        signature = hmac.new(self.secret_key, data, hashlib.sha256).digest()
        return signature + data

    def _verify_and_extract_data(self, signed_data: bytes) -> bytes:
        """Verificar firma HMAC y extraer datos."""
        if not self.secret_key:
            raise SecurityError("Secret key required for HMAC verification")
        if len(signed_data) < 32:
            raise SecurityError("Invalid signed data format")
            
        signature = signed_data[:32]
        data = signed_data[32:]
        
        expected_signature = hmac.new(self.secret_key, data, hashlib.sha256).digest()
        
        if not hmac.compare_digest(signature, expected_signature):
            raise SecurityError("Data integrity verification failed (HMAC mismatch)")
            
        return data

    def _encrypt_data(self, data: bytes) -> bytes:
        """Cifrar datos usando Fernet."""
        if not self._fernet:
            raise SecurityError("Encryption not properly configured")
        return self._fernet.encrypt(data)

    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Descifrar datos usando Fernet."""
        if not self._fernet:
            raise SecurityError("Decryption not properly configured")
        return self._fernet.decrypt(encrypted_data)

    def _add_validation_metadata(self, data: bytes, original_data: Any) -> bytes:
        """Agregar metadatos de validación."""
        metadata = {
            'type': type(original_data).__name__,
            'timestamp': time.time(),
            'size': len(data)
        }
        metadata_bytes = json.dumps(metadata).encode('utf-8')
        return len(metadata_bytes).to_bytes(4, 'big') + metadata_bytes + data

    def _verify_validation_metadata(self, data: bytes) -> Tuple[bytes, str]:
        """Verificar metadatos de validación."""
        if len(data) < 4:
            raise SecurityError("Invalid validation metadata")
        
        metadata_size = int.from_bytes(data[:4], 'big')
        metadata_bytes = data[4:4+metadata_size]
        actual_data = data[4+metadata_size:]
        
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        return actual_data, metadata['type']

    def _validate_data_type(self, data: Any, expected_type: str) -> None:
        """Validar que el tipo de datos deserializado coincida con el esperado."""
        actual_type = type(data).__name__
        if actual_type != expected_type:
            logger.warning(f"Type mismatch: expected {expected_type}, got {actual_type}")

# Alias para compatibilidad hacia atrás
Serializer = AdvancedSerializer

# --- Contexto Asíncrono Mejorado ---

class AsyncMnemeContext:
    """Contexto asíncrono para operaciones MNEME con gestión avanzada de recursos."""
    
    def __init__(self, config: MnemeConfig):
        self.config = config
        self.zspace = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent_operations)
        self.active_operations = 0
        self.operation_lock = asyncio.Lock()
        
    async def __aenter__(self):
        """Entrar al contexto asíncrono."""
        self.zspace = ZSpace(self.config)
        logger.info("Async MNEME context initialized")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Salir del contexto asíncrono."""
        if self.zspace:
            self.zspace.cleanup()
        logger.info("Async MNEME context cleaned up")
    
    async def register_async(self, name: str, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """Registrar tensor de forma asíncrona."""
        async with self.semaphore:
            async with self.operation_lock:
                self.active_operations += 1
            
            try:
                # Ejecutar en thread pool para evitar bloqueo
                loop = asyncio.get_event_loop()
                desc = await loop.run_in_executor(
                    None, 
                    self.zspace.register, 
                    name, 
                    tensor, 
                    **kwargs
                )
                return desc
            finally:
                async with self.operation_lock:
                    self.active_operations -= 1
    
    async def load_async(self, name: str) -> torch.Tensor:
        """Cargar tensor de forma asíncrona."""
        async with self.semaphore:
            async with self.operation_lock:
                self.active_operations += 1
            
            try:
                loop = asyncio.get_event_loop()
                tensor = await loop.run_in_executor(
                    None,
                    self.zspace.load,
                    name
                )
                return tensor
            finally:
                async with self.operation_lock:
                    self.active_operations -= 1
    
    async def update_async(self, name: str, delta_op: Dict) -> ZDescriptor:
        """Actualizar tensor de forma asíncrona."""
        async with self.semaphore:
            async with self.operation_lock:
                self.active_operations += 1
            
            try:
                loop = asyncio.get_event_loop()
                desc = await loop.run_in_executor(
                    None,
                    self.zspace.update,
                    name,
                    delta_op
                )
                return desc
            finally:
                async with self.operation_lock:
                    self.active_operations -= 1
    
    async def get_stats_async(self) -> Dict:
        """Obtener estadísticas de forma asíncrona."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.zspace.get_stats)
    
    @property
    async def active_operations_count(self) -> int:
        """Obtener número de operaciones activas."""
        async with self.operation_lock:
            return self.active_operations

class MnemeContextManager:
    """Gestor de contexto mejorado con funcionalidades avanzadas."""
    
    def __init__(self, config: MnemeConfig):
        self.config = config
        self.zspace = None
        self.context_stack = []
        self.resource_monitor = None
        
    def __enter__(self):
        """Entrar al contexto síncrono."""
        self.zspace = ZSpace(self.config)
        self.context_stack.append(self.zspace)
        
        # Inicializar monitoreo de recursos si está habilitado
        if self.config.enable_async_context:
            self._start_resource_monitoring()
        
        logger.info("MNEME context initialized")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Salir del contexto síncrono."""
        if self.zspace:
            self.zspace.cleanup()
        
        if self.resource_monitor:
            self._stop_resource_monitoring()
        
        self.context_stack.clear()
        logger.info("MNEME context cleaned up")
    
    def _start_resource_monitoring(self):
        """Iniciar monitoreo de recursos."""
        self.resource_monitor = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self.resource_monitor.start()
    
    def _stop_resource_monitoring(self):
        """Detener monitoreo de recursos."""
        if self.resource_monitor:
            self.resource_monitor.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Monitorear recursos del sistema."""
        while self.zspace:
            try:
                # Monitorear uso de memoria
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > self.config.memory_pressure_threshold * 100:
                    logger.warning(f"High memory usage detected: {memory_usage:.1f}%")
                
                # Monitorear uso de GPU si está disponible
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    if gpu_memory > 0.9:  # 90% de uso
                        logger.warning(f"High GPU memory usage: {gpu_memory:.1%}")
                
                time.sleep(5)  # Verificar cada 5 segundos
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                break
    
    def register(self, name: str, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """Registrar tensor."""
        return self.zspace.register(name, tensor, **kwargs)
    
    def load(self, name: str) -> torch.Tensor:
        """Cargar tensor."""
        return self.zspace.load(name)
    
    def update(self, name: str, delta_op: Dict) -> ZDescriptor:
        """Actualizar tensor."""
        return self.zspace.update(name, delta_op)
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas."""
        return self.zspace.get_stats()
    
    def rotate_keys(self) -> bool:
        """Rotar claves manualmente."""
        if hasattr(self.zspace, 'gen') and hasattr(self.zspace.gen, 'serializer'):
            serializer = self.zspace.gen.serializer
            if hasattr(serializer, 'key_manager'):
                serializer.key_manager.rotate_key()
                return True
        return False
    
    def get_key_info(self) -> Dict:
        """Obtener información de claves."""
        if hasattr(self.zspace, 'gen') and hasattr(self.zspace.gen, 'serializer'):
            serializer = self.zspace.gen.serializer
            if hasattr(serializer, 'key_manager'):
                return {
                    'current_version': serializer.key_manager.key_version,
                    'usage_count': serializer.key_manager.usage_count,
                    'last_rotation': serializer.key_manager.last_rotation.isoformat(),
                    'rotation_policy': serializer.key_manager.config.key_rotation_policy.value
                }
        return {}

# Funciones de utilidad para contexto
@contextmanager
def mneme_context(config: Optional[MnemeConfig] = None) -> Generator[MnemeContextManager, None, None]:
    """Context manager síncrono para MNEME."""
    config = config or MnemeConfig()
    with MnemeContextManager(config) as context:
        yield context

@asynccontextmanager
async def async_mneme_context(config: Optional[MnemeConfig] = None) -> AsyncGenerator[AsyncMnemeContext, None]:
    """Context manager asíncrono para MNEME."""
    config = config or MnemeConfig()
    async with AsyncMnemeContext(config) as context:
        yield context

# --- Tensor Operations (Quantizer, Decomposer) ---

class Quantizer:
    # (Implementación similar a la original, robusta y funcional)
    @staticmethod
    def quantize(tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, float]:
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        if min_val == max_val:
            dtype = torch.uint8 if bits <= 8 else torch.int16
            return torch.zeros_like(tensor, dtype=dtype), min_val, 1.0
        
        scale = (2**bits - 1) / (max_val - min_val)
        offset = min_val
        
        quantized = torch.clamp(torch.round((tensor - offset) * scale), 0, 2**bits - 1)
        
        if bits <= 8:
            quantized = quantized.to(torch.uint8)
        elif bits <= 16:
            quantized = quantized.to(torch.int16) # PyTorch prefiere int16 sobre uint16
        else:
            quantized = quantized.to(torch.int32)
        
        return quantized, offset, scale
    
    @staticmethod
    def dequantize(quantized: torch.Tensor, offset: float, scale: float) -> torch.Tensor:
        return quantized.float() / scale + offset

class TensorDecomposer:
    """Descompositor de tensores con heurísticas mejoradas y soporte de dispositivo."""
    
    @staticmethod
    def auto_select(tensor: torch.Tensor, 
                   target_ratio: float = 0.1) -> Tuple[DecompType, Dict[str, Any]]:
        # (Heurísticas refinadas basadas en la implementación original y mejoras comunes)
        shape = tensor.shape
        numel = tensor.numel()
        
        # 1. Análisis de esparsidad
        sparsity = (tensor == 0).float().mean().item()
        if sparsity > 0.90:
            return DecompType.SPARSE, {}
        
        # 2. Análisis de Dimensionalidad
        if len(shape) == 2:
            # SVD para matrices. Heurística refinada para respetar target_ratio.
            M, N = shape
            # Ratio = K*(M+N+1) / (M*N). Resolver para K (rank).
            target_rank = int((target_ratio * M * N) / (M + N + 1))
            max_rank = min(M, N)
            rank = max(1, min(target_rank, max_rank))

            if rank < 5 and max_rank > 100:
                return DecompType.QUANTIZED, {"bits": 8}
            return DecompType.SVD, {"rank": rank}
        
        if len(shape) >= 3:
            # TT preferido para alta dimensión por su robustez
            avg_dim = np.prod(shape)**(1/len(shape))
            # Heurística simplificada para rango TT
            scaling_factor = target_ratio**(1/(len(shape)-1))
            target_rank = max(1, int(avg_dim * scaling_factor))
            tt_ranks = tuple([target_rank] * (len(shape) - 1))
            return DecompType.TT, {"ranks": tt_ranks}
        
        # Default (e.g., 1D tensors)
        return DecompType.QUANTIZED, {"bits": 8}
    
    @staticmethod
    def decompose(tensor: torch.Tensor, decomp_type: DecompType, device: torch.device, **params) -> Dict[str, Any]:
        # Mover tensor al dispositivo de cómputo
        tensor = tensor.to(device)

        try:
            if decomp_type == DecompType.TT:
                ranks = params.get('ranks')
                if not ranks:
                     # Fallback
                    avg_dim = np.prod(tensor.shape)**(1/len(tensor.shape))
                    default_rank = max(1, int(avg_dim * 0.1))
                    ranks = [default_rank] * (len(tensor.shape) - 1)
                
                factors = tensor_train(tensor, rank=ranks)
                # Extraer los factores (tensores) del objeto TensorTrain
                return {"factors": list(factors), "type": "tt", "ranks": ranks}
                
            elif decomp_type == DecompType.CP:
                rank = params.get('rank', 10)
                weights, factors = parafac(tensor, rank=rank, init='svd', n_iter_max=300, tol=1e-8, linesearch=True)
                return {"weights": weights, "factors": factors, "type": "cp", "rank": rank}
                
            elif decomp_type == DecompType.TUCKER:
                ranks = params.get('ranks', [min(s, 20) for s in tensor.shape])
                core, factors = tucker(tensor, rank=ranks, n_iter_max=200, tol=1e-8)
                return {"core": core, "factors": factors, "type": "tucker", "ranks": ranks}

            elif decomp_type == DecompType.SVD:
                rank = params.get('rank', min(tensor.shape) // 4)
                U, S, V = torch.svd_lowrank(tensor, q=rank, niter=10)
                return {"U": U, "S": S, "V": V, "type": "svd", "rank": rank}

            elif decomp_type == DecompType.SPARSE:
                sparse_tensor = tensor.to_sparse()
                return {"indices": sparse_tensor.indices(), "values": sparse_tensor.values(), "shape": tensor.shape, "type": "sparse"}

            elif decomp_type == DecompType.QUANTIZED:
                bits = params.get('bits', 8)
                quantized, offset, scale = Quantizer.quantize(tensor, bits)
                return {"quantized": quantized, "offset": offset, "scale": scale, "type": "quantized", "bits": bits}

            else:  # RAW
                return {"tensor": tensor, "type": "raw"}
                
        except Exception as e:
            logger.warning(f"Decomposition {decomp_type.value} failed on device {device}: {e}. Falling back to RAW.")
            return {"tensor": tensor, "type": "raw"}
    
    @staticmethod
    def reconstruct(components: Dict[str, Any], device: torch.device) -> torch.Tensor:
        # Los componentes ya deberían estar en el dispositivo correcto gracias al Serializer.deserialize(map_location=device)
        comp_type = components["type"]
        
        try:
            if comp_type == "tt":
                factors = components["factors"]
                return tl.tt_to_tensor(factors)
            elif comp_type == "cp":
                weights = components.get("weights")
                factors = components["factors"]
                return tl.cp_to_tensor((weights, factors))
            elif comp_type == "tucker":
                core = components["core"]
                factors = components["factors"]
                return tl.tucker_to_tensor((core, factors))
            elif comp_type == "svd":
                U, S, V = components["U"], components["S"], components["V"]
                return U @ torch.diag(S) @ V.T
            elif comp_type == "sparse":
                shape = components["shape"]
                indices = components["indices"]
                values = components["values"]
                return torch.sparse_coo_tensor(indices, values, shape, device=device).to_dense()
            elif comp_type == "quantized":
                quantized = components["quantized"]
                offset = components["offset"]
                scale = components["scale"]
                return Quantizer.dequantize(quantized, offset, scale)
            else:  # raw
                return components["tensor"]
                
        except Exception as e:
            logger.error(f"Reconstruction failed for type {comp_type} on device {device}: {e}")
            raise

# --- Caching and Prefetching Systems ---

class ZCache:
    """
    [MEJORA] Cache LRU con gestión inteligente de memoria. Almacena en CPU para ahorrar VRAM.
    """
    
    def __init__(self, config: MnemeConfig):
        self.capacity = config.cache_size_bytes
        self.memory_threshold = config.memory_pressure_threshold
        self.used = 0
        self.cache = OrderedDict()
        self.lock = RLock()
        
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "memory_pressure_events": 0}
        
    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        return tensor.element_size() * tensor.nelement()

    def get(self, key: bytes, target_device: torch.device) -> Optional[torch.Tensor]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                # Mover al dispositivo solicitado solo en la lectura
                return self.cache[key].to(target_device)
            
            self.stats["misses"] += 1
            return None
    
    def put(self, key: bytes, tensor: torch.Tensor):
        with self.lock:
            if key in self.cache:
                # Si ya existe, no hacer nada (asumiendo inmutabilidad del contenido)
                return

            # Asegurar que el tensor esté en CPU antes de cachear
            tensor_cpu = tensor.cpu()
            tensor_bytes = self._get_tensor_size(tensor_cpu)
            
            if tensor_bytes > self.capacity:
                return

            if self._check_system_memory_pressure():
                self._aggressive_eviction()
            
            while self.used + tensor_bytes > self.capacity and self.cache:
                self._evict_one()
            
            self.cache[key] = tensor_cpu
            self.used += tensor_bytes
            self.cache.move_to_end(key)

    def _check_system_memory_pressure(self) -> bool:
        """Verificar presión de memoria del sistema (RAM)."""
        try:
            vm = psutil.virtual_memory()
            if vm.percent / 100.0 > self.memory_threshold:
                self.stats["memory_pressure_events"] += 1
                return True
        except Exception:
            pass
        return False

    def _aggressive_eviction(self):
        """Liberar 20% del cache."""
        logger.warning("System memory pressure detected. Initiating aggressive eviction.")
        target_usage = int(self.capacity * 0.8)
        while self.used > target_usage and self.cache:
            self._evict_one()
        gc.collect()
    
    def _evict_one(self):
        if not self.cache: return
        _, tensor = self.cache.popitem(last=False) # LRU
        self.used -= self._get_tensor_size(tensor)
        self.stats["evictions"] += 1

class MarkovPrefetcher:
    """Prefetcher Markov de 2do orden."""
    
    def __init__(self):
        self.history = deque(maxlen=1000)
        self.transitions = {}  # (prev, curr) -> {next: count}
        self.confidence_threshold = 0.3
        self.lock = Lock()
        
    def record_access(self, addr: bytes):
        with self.lock:
            self.history.append(addr)
            if len(self.history) >= 3:
                A, B, C = self.history[-3], self.history[-2], self.history[-1]
                key = (A, B)
                if key not in self.transitions:
                    self.transitions[key] = {}
                self.transitions[key][C] = self.transitions[key].get(C, 0) + 1
    
    def predict_next(self, curr: bytes) -> List[bytes]:
        with self.lock:
            if len(self.history) < 2: return []
            
            prev = self.history[-2]
            key = (prev, curr)

            if key in self.transitions:
                transitions = self.transitions[key]
                total = sum(transitions.values())
                predictions = []
                for next_addr, count in transitions.items():
                    confidence = count / total
                    if confidence >= self.confidence_threshold:
                        predictions.append((next_addr, confidence))
                
                predictions.sort(key=lambda x: x[1], reverse=True)
                return [addr for addr, conf in predictions[:3]] # Top 3
            return []

# --- Synthesis Engine (ZGen) ---

class ZGen:
    """Motor de síntesis."""
    
    def __init__(self, config: MnemeConfig, 
                 device: torch.device,
                 descriptor_lookup_fn: Callable[[bytes], Optional[ZDescriptor]]):
        self.config = config
        self.device = device
        self.cache = ZCache(config)
        self.prefetcher = MarkovPrefetcher()
        self.serializer = AdvancedSerializer(config)
        
        # Integrar gestión de claves
        if hasattr(self.serializer, 'key_manager'):
            self.key_manager = self.serializer.key_manager
        
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self.pending_synthesis: Dict[bytes, Future] = {} # addr -> Future
        self.lock = Lock()
        self.descriptor_lookup_fn = descriptor_lookup_fn
        
        self.stats = {"total_synthesis_time": 0.0, "synthesis_count": 0, "compression_ratios": []}

    def synthesize(self, desc: ZDescriptor) -> torch.Tensor:
        """Pipeline de síntesis principal."""
        start_time = time.time()
        
        try:
            # 1. Verificación de Integridad del Descriptor
            if not desc.verify_integrity():
                raise SecurityError("Descriptor integrity check failed.")
            
            # 2. Procesar Core Data (Descomprimir, Verificar Firma, Deserializar)
            core_data_bytes = AdvancedCompressor.decompress(desc.core_data)
            # Carga los componentes en el dispositivo de cómputo
            components = self.serializer.deserialize(core_data_bytes, device=self.device)
            
            # 3. Reconstrucción del Tensor Base
            tensor = TensorDecomposer.reconstruct(components, self.device)
            
            # 4. Aplicación de Cadena de Deltas
            if desc.delta_chain:
                deltas_bytes = AdvancedCompressor.decompress(desc.delta_chain)
                # Carga los deltas en el dispositivo de cómputo
                deltas = self.serializer.deserialize(deltas_bytes, device=self.device)
                
                for delta_op in deltas:
                    tensor = self._apply_delta(tensor, delta_op)
            
            # 5. Verificación Final de Forma
            if tensor.shape != desc.shape:
                try:
                    tensor = tensor.reshape(desc.shape)
                except RuntimeError:
                    raise ValueError(f"Synthesized tensor shape mismatch. Expected {desc.shape}, got {tensor.shape}.")
            
            # Actualizar estadísticas
            self.stats["total_synthesis_time"] += time.time() - start_time
            self.stats["synthesis_count"] += 1
            
            return tensor
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise
    
    def _apply_delta(self, tensor: torch.Tensor, delta_op: Dict) -> torch.Tensor:
        """Aplica operación delta (en el dispositivo)."""
        op_type = delta_op["type"]
        # Los valores ya están en el dispositivo correcto gracias al Serializer
        
        if op_type == "add":
            return tensor + delta_op["value"]
        elif op_type == "mul":
            return tensor * delta_op["value"]
        elif op_type == "sparse_update":
            indices = delta_op["indices"]
            values = delta_op["values"]
            tensor = tensor.clone()
            # Manejo de índices multidimensionales requiere tupla
            if indices.dim() > 1:
                tensor[tuple(indices)] = values
            else:
                tensor[indices] = values
            return tensor
        elif op_type == "quantized_update":
            indices = delta_op["indices"]
            quantized_values = delta_op["quantized_values"]
            offset = delta_op["offset"]
            scale = delta_op["scale"]
            values = Quantizer.dequantize(quantized_values, offset, scale)
            tensor = tensor.clone()
            if indices.dim() > 1:
                tensor[tuple(indices)] = values
            else:
                tensor[indices] = values
            return tensor
        else:
            raise ValueError(f"Unknown delta op: {op_type}")
    
    def load(self, desc: ZDescriptor) -> torch.Tensor:
        """Cargar con cache, manejo de concurrencia y prefetching."""
        addr = ZAddr.compute(desc)
        
        # 1. Verificar Cache
        tensor = self.cache.get(addr, self.device)
        if tensor is not None:
            self.prefetcher.record_access(addr)
            self._trigger_prefetch(addr)
            return tensor
        
        # 2. Manejo de Síntesis Concurrente (Thundering Herd Protection)
        with self.lock:
            if addr in self.pending_synthesis:
                future = self.pending_synthesis[addr]
            else:
                future = self.executor.submit(self.synthesize, desc)
                self.pending_synthesis[addr] = future
        
        # 3. Esperar Resultado y Cachear
        try:
            tensor = future.result()
            # ZCache se encarga de mover a CPU antes de almacenar
            self.cache.put(addr, tensor)
        finally:
            with self.lock:
                # Asegurar que eliminamos la future correcta
                if self.pending_synthesis.get(addr) == future:
                   self.pending_synthesis.pop(addr, None)
        
        # 4. Prefetching
        self.prefetcher.record_access(addr)
        self._trigger_prefetch(addr)
        
        return tensor
    
    def _trigger_prefetch(self, current_addr: bytes):
        """[MEJORA] Activar prefetching especulativo funcional."""
        predicted_addrs = self.prefetcher.predict_next(current_addr)
        
        for addr in predicted_addrs:
            if addr not in self.cache.cache:
                with self.lock:
                    if addr not in self.pending_synthesis:
                        # Buscar el descriptor correspondiente usando el callback
                        desc = self.descriptor_lookup_fn(addr)
                        if desc:
                            logger.debug(f"Prefetching {addr.hex()[:8]}...")
                            future = self.executor.submit(self.synthesize, desc)
                            self.pending_synthesis[addr] = future
                            # Añadir callback para cachear cuando termine
                            future.add_done_callback(lambda f: self._handle_prefetch_result(addr, f))

    def _handle_prefetch_result(self, addr: bytes, future: Future):
        """Callback para resultados de prefetch."""
        with self.lock:
             if self.pending_synthesis.get(addr) == future:
                self.pending_synthesis.pop(addr, None)
        try:
            tensor = future.result()
            self.cache.put(addr, tensor)
        except Exception:
            pass

    def store(self, tensor: torch.Tensor, 
             target_ratio: float = 0.1,
             decomp_type: Optional[DecompType] = None) -> ZDescriptor:
        """Almacenar tensor: descomponer, serializar (seguro), firmar, comprimir y crear descriptor."""
        
        # 1. Selección y Descomposición (en el dispositivo de cómputo)
        if decomp_type is None or decomp_type == DecompType.ADAPTIVE:
            decomp_type, params = TensorDecomposer.auto_select(tensor, target_ratio)
        else:
            params = {}
        
        components = TensorDecomposer.decompose(tensor, decomp_type, self.device, **params)
        
        # 2. Serialización (incluye mover a CPU y firmado) y Compresión
        components_bytes = self.serializer.serialize(components)
        compressed_core_data = AdvancedCompressor.compress(components_bytes, self.config.compression_level)
        
        # 3. Creación de Árbol Merkle (Opcional)
        merkle_root = None
        if self.config.enable_merkle:
            merkle_root = MerkleTree.compute_root([compressed_core_data])
        
        # 4. Creación del Descriptor (El checksum se calcula automáticamente en __post_init__)
        original_size = tensor.nelement() * tensor.element_size() + 1e-9
        ratio = len(compressed_core_data) / original_size

        desc = ZDescriptor(
            kind="tensor",
            decomp_type=decomp_type,
            shape=tuple(tensor.shape),
            ranks=params.get("ranks") or (params.get("rank"),) if "rank" in params else None,
            core_data=compressed_core_data,
            version=0,
            meta={"compression_ratio": ratio, "decomp_params": params, "dtype": str(tensor.dtype)},
            merkle_root=merkle_root,
            compression_level=self.config.compression_level
        )
        
        self.stats["compression_ratios"].append(ratio)
        return desc
    
    def shutdown(self):
        self.executor.shutdown(wait=True)

# --- Main Runtime Interface (ZSpace) ---

class ZSpace:
    """Interfaz principal del runtime MNEME. Gestiona descriptores, versiones y consolidación."""
    
    def __init__(self, config: Optional[MnemeConfig] = None):
        self.config = config or MnemeConfig()
        
        # 1. Configuración de Dispositivo
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.config.use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # 2. Configuración de Seguridad (Clave Secreta)
        if self.config.secret_key:
            if len(self.config.secret_key) < 32: 
                raise ValueError("Secret key must be >= 32 bytes.")
        else:
            logger.warning("No secret key provided. Generating a transient secure key. Persistence requires a fixed key.")
            # Generar una clave transitoria segura si no se proporciona
            self.config = replace(self.config, secret_key=secrets.token_bytes(32))

        # 3. Tablas de mapeo
        self.name_to_desc: Dict[str, ZDescriptor] = {}
        # [MEJORA] Índice inverso para prefetching
        self.addr_to_desc: Dict[bytes, ZDescriptor] = {}
        self.version_graph: Dict[bytes, bytes] = {} # new_addr -> old_addr
        self.lock = RLock()
        
        # 4. Inicializar sistema de almacenamiento avanzado
        self._init_storage_system()
        
        # 5. Inicializar sistema de deduplicación de contexto
        self._init_context_deduplication()
        
        # 6. Inicializar ZGen
        self.gen = ZGen(self.config, self.device, self._lookup_descriptor_by_addr)

        logger.info(f"MNEME ZSpace initialized on device {self.device}.")
    
    def _init_storage_system(self):
        """Inicializar sistema de almacenamiento avanzado."""
        # Inicializar backend de almacenamiento
        if self.config.storage_backend == StorageBackend.MEMORY:
            self.storage_backend = MemoryStorage(self.config.cache_size_mb)
        elif self.config.storage_backend == StorageBackend.DISK:
            storage_path = self.config.storage_path or "./mneme_storage"
            self.storage_backend = DiskStorage(storage_path)
        elif self.config.storage_backend == StorageBackend.HYBRID:
            self.storage_backend = HybridStorage(self.config)
        else:
            # Default a memoria
            self.storage_backend = MemoryStorage(self.config.cache_size_mb)
        
        # Inicializar cache avanzado
        self.advanced_cache = AdvancedCache(self.config)
        
        # Inicializar motor de deduplicación
        self.deduplication_engine = DeduplicationEngine(self.config)
        
        # Inicializar métricas de almacenamiento
        self.storage_metrics = {
            "total_stores": 0,
            "total_retrieves": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "deduplication_saves": 0,
            "total_size_bytes": 0
        }
    
    def _init_context_deduplication(self):
        """Inicializar sistema de deduplicación de contexto."""
        try:
            if self.config.enable_context_deduplication:
                self.context_deduplication_engine = ContextDeduplicationEngine(self.config)
                logger.info("Context deduplication system initialized")
            else:
                self.context_deduplication_engine = None
                logger.info("Context deduplication disabled")
        except Exception as e:
            logger.error(f"Failed to initialize context deduplication: {e}")
            self.context_deduplication_engine = None

    # [MEJORA] Implementar Context Manager para gestión de recursos
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _lookup_descriptor_by_addr(self, addr: bytes) -> Optional[ZDescriptor]:
        """Callback para ZGen."""
        with self.lock:
            return self.addr_to_desc.get(addr)

    def _register_descriptor(self, name: str, desc: ZDescriptor, old_addr: Optional[bytes] = None):
        """Helper para registrar descriptor en todas las tablas y manejar linaje."""
        addr = ZAddr.compute(desc)
        self.name_to_desc[name] = desc
        self.addr_to_desc[addr] = desc
        if old_addr:
            self.version_graph[addr] = old_addr
        return addr
    
    def register(self, name: str, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """Registrar tensor."""
        
        desc = self.gen.store(tensor, **kwargs)
        
        with self.lock:
            old_addr = None
            if name in self.name_to_desc:
                old_addr = ZAddr.compute(self.name_to_desc[name])

            addr = self._register_descriptor(name, desc, old_addr)
            
        logger.info(f"Registered '{name}'. Type: {desc.decomp_type.value}. Ratio: {desc.meta['compression_ratio']:.3f}. Addr: {addr.hex()[:8]}")
        return desc
    
    def load(self, name: str) -> torch.Tensor:
        """Cargar tensor por nombre."""
        with self.lock:
            if name not in self.name_to_desc:
                raise KeyError(f"Unknown tensor: {name}")
            desc = self.name_to_desc[name]
        return self.gen.load(desc)
    
    def update(self, name: str, delta_op: Dict) -> ZDescriptor:
        """
        [MEJORA] Actualizar tensor con operación delta y manejar consolidación automática.
        """
        with self.lock:
            if name not in self.name_to_desc:
                raise KeyError(f"Unknown tensor: {name}")
                
            old_desc = self.name_to_desc[name]
            
            # 1. Cargar y Deserializar Cadena Delta Existente (Seguro)
            if old_desc.delta_chain:
                try:
                    deltas_bytes = AdvancedCompressor.decompress(old_desc.delta_chain)
                    # Deserializar a CPU para manipulación de la estructura
                    deltas = self.gen.serializer.deserialize(deltas_bytes, device=torch.device('cpu'))
                except Exception as e:
                    logger.error(f"Failed to load delta chain for '{name}': {e}. Resetting chain.")
                    deltas = []
            else:
                deltas = []
            
            # 2. Añadir Nueva Operación (Asegurar que tensores en delta_op estén en CPU)
            delta_op_cpu = self.gen.serializer._move_to_cpu(delta_op)
            deltas.append(delta_op_cpu)
            
            # 3. Verificar Umbral de Consolidación Automática
            if len(deltas) >= self.config.delta_consolidation_threshold:
                logger.info(f"Delta chain threshold reached for '{name}'. Initiating automatic consolidation.")
                return self._consolidate(name, old_desc)

            # 4. Serializar (y firmar) y Comprimir Nueva Cadena Delta
            new_deltas_bytes = self.gen.serializer.serialize(deltas)
            compressed_deltas = AdvancedCompressor.compress(new_deltas_bytes, self.config.compression_level)
            
            # 5. Crear Nuevo Descriptor (Versión Incremental)
            # Usamos replace; el checksum se recalcula automáticamente en __post_init__
            new_desc = replace(
                old_desc,
                version=old_desc.version + 1,
                delta_chain=compressed_deltas,
                meta={**old_desc.meta, 'delta_count': len(deltas)}
            )
            
            # 6. Actualizar Tablas y Linaje
            old_addr = ZAddr.compute(old_desc)
            self._register_descriptor(name, new_desc, old_addr)
            
            logger.info(f"Updated '{name}' to version {new_desc.version} (Delta applied).")
            return new_desc

    def _consolidate(self, name: str, desc: ZDescriptor) -> ZDescriptor:
        """
        Consolida la cadena delta sintetizando el estado actual y almacenándolo como una nueva base.
        Debe llamarse dentro de un lock de ZSpace.
        """
        logger.info(f"Consolidating '{name}' (Version {desc.version})...")
        try:
            # 1. Sintetizar el estado actual (requiere cargar fuera del lock si gen.load no es reentrante, 
            # pero aquí asumimos que gen.load maneja su propia concurrencia)
            current_tensor = self.gen.load(desc)
        except Exception as e:
            logger.error(f"Consolidation failed during synthesis for '{name}': {e}")
            return desc # Si falla, retornar el descriptor antiguo

        # 2. Almacenar el tensor sintetizado como nueva base
        # Usar parámetros de descomposición anteriores si están disponibles, o default
        target_ratio = desc.meta.get('decomp_params', {}).get('target_ratio', 0.1)
        new_base_desc = self.gen.store(current_tensor, target_ratio=target_ratio)
        
        # 3. Actualizar Descriptor para reflejar la nueva base (manteniendo versión incremental)
        new_desc = replace(
            new_base_desc,
            version=desc.version + 1,
            meta={**new_base_desc.meta, 'consolidated_from_v': desc.version}
        )

        # 4. Actualizar Tablas y Linaje
        old_addr = ZAddr.compute(desc)
        self._register_descriptor(name, new_desc, old_addr)

        logger.info(f"Consolidated '{name}' to new base version {new_desc.version}.")
        return new_desc

    def get_stats(self) -> Dict:
        """Obtener estadísticas del runtime"""
        return {
            "config": self.config,
            "device": str(self.device),
            "descriptors": len(self.name_to_desc),
            "unique_addresses": len(self.addr_to_desc),
            "performance": self.gen.stats,
            "storage": self.get_storage_stats(),
            "cache": self.advanced_cache.get_stats(),
            "deduplication": self.deduplication_engine.get_stats(),
            "context_deduplication": self.get_context_deduplication_stats()
        }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de almacenamiento."""
        backend_stats = self.storage_backend.get_stats()
        return {
            **backend_stats,
            "metrics": self.storage_metrics
        }
    
    def store_descriptor(self, addr: bytes, desc: ZDescriptor) -> bool:
        """Almacenar descriptor usando el sistema de almacenamiento avanzado."""
        try:
            # Serializar descriptor
            desc_data = self.gen.serializer.serialize(desc)
            
            # Verificar deduplicación
            existing_key = self.deduplication_engine.should_deduplicate(desc_data)
            if existing_key:
                self.storage_metrics["deduplication_saves"] += 1
                logger.debug(f"Content deduplicated for address {addr.hex()[:8]}")
                return True
            
            # Almacenar en cache primero
            self.advanced_cache.put(addr, desc_data)
            
            # Almacenar en backend persistente
            success = self.storage_backend.store(addr, desc_data)
            
            if success:
                # Registrar contenido para deduplicación
                self.deduplication_engine.register_content(addr, desc_data)
                self.storage_metrics["total_stores"] += 1
                self.storage_metrics["total_size_bytes"] += len(desc_data)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store descriptor {addr.hex()[:8]}: {e}")
            return False
    
    def retrieve_descriptor(self, addr: bytes) -> Optional[ZDescriptor]:
        """Recuperar descriptor usando el sistema de almacenamiento avanzado."""
        try:
            # 1. Intentar cache primero
            desc_data = self.advanced_cache.get(addr)
            if desc_data is not None:
                self.storage_metrics["cache_hits"] += 1
                self.storage_metrics["total_retrieves"] += 1
                return self.gen.serializer.deserialize(desc_data, torch.device('cpu'))
            
            # 2. Intentar backend persistente
            desc_data = self.storage_backend.retrieve(addr)
            if desc_data is not None:
                # Cargar en cache para futuras consultas
                self.advanced_cache.put(addr, desc_data)
                self.storage_metrics["cache_misses"] += 1
                self.storage_metrics["total_retrieves"] += 1
                return self.gen.serializer.deserialize(desc_data, torch.device('cpu'))
            
            self.storage_metrics["cache_misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve descriptor {addr.hex()[:8]}: {e}")
            return None
    
    def delete_descriptor(self, addr: bytes) -> bool:
        """Eliminar descriptor del sistema de almacenamiento."""
        try:
            # Eliminar de cache
            self.advanced_cache.delete(addr)
            
            # Eliminar de backend persistente
            success = self.storage_backend.delete(addr)
            
            # Desregistrar de deduplicación
            self.deduplication_engine.unregister_content(addr)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete descriptor {addr.hex()[:8]}: {e}")
            return False
    
    def optimize_storage(self) -> Dict[str, Any]:
        """Optimizar almacenamiento ejecutando limpieza y compresión."""
        try:
            optimization_results = {
                "cache_cleaned": False,
                "deduplication_optimized": False,
                "storage_compressed": False,
                "space_saved_bytes": 0
            }
            
            # Limpiar cache expirado
            if hasattr(self.advanced_cache, 'clear_expired'):
                self.advanced_cache.clear_expired()
                optimization_results["cache_cleaned"] = True
            
            # Optimizar deduplicación
            if self.deduplication_engine.enabled:
                # Limpiar referencias huérfanas
                self.deduplication_engine.cleanup_orphaned_references()
                optimization_results["deduplication_optimized"] = True
            
            # Comprimir almacenamiento si es posible
            if hasattr(self.storage_backend, 'compress'):
                space_saved = self.storage_backend.compress()
                optimization_results["space_saved_bytes"] = space_saved
                optimization_results["storage_compressed"] = True
            
            logger.info(f"Storage optimization completed: {optimization_results}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Storage optimization failed: {e}")
            return {"error": str(e)}
    
    def get_storage_health(self) -> Dict[str, Any]:
        """Obtener estado de salud del sistema de almacenamiento."""
        try:
            cache_stats = self.advanced_cache.get_stats()
            storage_stats = self.get_storage_stats()
            dedup_stats = self.deduplication_engine.get_stats()
            
            # Calcular métricas de salud
            cache_usage = cache_stats.get("usage_percent", 0)
            hit_rate = cache_stats.get("hit_rate", 0)
            
            health_score = 100
            warnings = []
            
            # Evaluar salud del cache
            if cache_usage > 90:
                health_score -= 20
                warnings.append("Cache usage > 90%")
            
            if hit_rate < 50:
                health_score -= 15
                warnings.append("Cache hit rate < 50%")
            
            # Evaluar salud del almacenamiento
            if storage_stats.get("error"):
                health_score -= 30
                warnings.append("Storage backend error")
            
            # Evaluar deduplicación
            dedup_ratio = dedup_stats.get("deduplication_ratio", 1)
            if dedup_ratio > 2:
                health_score += 10  # Bonus por buena deduplicación
            
            return {
                "health_score": max(0, min(100, health_score)),
                "cache_usage_percent": cache_usage,
                "cache_hit_rate": hit_rate,
                "deduplication_ratio": dedup_ratio,
                "total_storage_bytes": storage_stats.get("total_size_bytes", 0),
                "warnings": warnings,
                "status": "healthy" if health_score >= 80 else "warning" if health_score >= 60 else "critical"
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage health: {e}")
            return {"error": str(e), "health_score": 0, "status": "error"}
    
    def process_context_for_deduplication(self, context_id: str, tensor: torch.Tensor, 
                                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Procesar contexto para deduplicación."""
        try:
            if self.context_deduplication_engine:
                return self.context_deduplication_engine.process_context(context_id, tensor, metadata)
            else:
                return {"context_id": context_id, "deduplicated": False, "reason": "disabled"}
        except Exception as e:
            logger.error(f"Context deduplication processing failed: {e}")
            return {"context_id": context_id, "error": str(e)}
    
    def get_context_deduplication_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de deduplicación de contexto."""
        try:
            if self.context_deduplication_engine:
                return self.context_deduplication_engine.get_stats()
            else:
                return {"enabled": False, "reason": "Context deduplication disabled"}
        except Exception as e:
            logger.error(f"Failed to get context deduplication stats: {e}")
            return {"error": str(e)}
    
    def optimize_context_deduplication(self):
        """Optimizar sistema de deduplicación de contexto."""
        try:
            if self.context_deduplication_engine:
                self.context_deduplication_engine.optimize_contexts()
                logger.info("Context deduplication optimization completed")
            else:
                logger.warning("Context deduplication not enabled")
        except Exception as e:
            logger.error(f"Context deduplication optimization failed: {e}")
    
    def get_context_cluster_info(self, context_id: str) -> Dict[str, Any]:
        """Obtener información del cluster de un contexto."""
        try:
            if self.context_deduplication_engine:
                cluster_id = self.context_deduplication_engine.clusterer.get_context_cluster(context_id)
                if cluster_id is not None:
                    cluster_members = self.context_deduplication_engine.clusterer.get_cluster_members(cluster_id)
                    return {
                        "context_id": context_id,
                        "cluster_id": cluster_id,
                        "cluster_members": cluster_members,
                        "cluster_size": len(cluster_members)
                    }
                else:
                    return {"context_id": context_id, "cluster_id": None}
            else:
                return {"context_id": context_id, "error": "Context deduplication disabled"}
        except Exception as e:
            logger.error(f"Failed to get context cluster info: {e}")
            return {"context_id": context_id, "error": str(e)}
    
    def cleanup(self):
        """Limpiar recursos."""
        logger.info("Cleaning up MNEME ZSpace...")
        self.gen.shutdown()
        gc.collect()
        if self.device.type.startswith('cuda'):
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
        logger.info("Cleanup completed.")

# Alias para compatibilidad
Mneme = ZSpace
