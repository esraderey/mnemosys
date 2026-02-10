"""
MNEME Core V3.0: Motor de Memoria Neural Mórfica - Enterprise Architecture
Sistema avanzado de memoria computacional con arquitectura de plugins,
circuit breaker nativo, observabilidad distribuida y API unificada.

Esta versión actúa como "cabeza" del sistema MNEME, proporcionando:
- Arquitectura de plugins extensible
- Circuit breaker para resiliencia
- Pipeline de procesamiento con hooks
- Métricas OpenTelemetry-compatible
- Integración nativa con MNEMEOptimizer
- Memory-aware operations
- API Facade unificada

Versión: 3.0.0
Autor: MNEME Development Team
Licencia: BSL 1.1
"""

from __future__ import annotations

# === IMPORTS ESTÁNDAR ===
import asyncio
import base64
import contextlib
import functools
import gc
import hashlib
import hmac
import io
import json
import logging
import multiprocessing as mp
import os
import queue
import secrets
import shutil
import sqlite3
import struct
import sys
import tempfile
import threading
import time
import traceback
import uuid
import warnings
import weakref
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timedelta
from enum import Enum, IntEnum, auto
from functools import lru_cache, wraps, partial
from pathlib import Path
from threading import Lock, RLock, local as ThreadLocal, Condition, Event, Semaphore
from typing import (
    Any, AsyncGenerator, Awaitable, Callable, ClassVar, Coroutine,
    Dict, Final, Generator, Generic, Iterable, Iterator, List, Literal,
    Mapping, NamedTuple, Optional, Protocol, Sequence, Set, Tuple,
    Type, TypeVar, Union, cast, overload, runtime_checkable,
)

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

# === CONFIGURACIÓN DE LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# === CONSTANTES GLOBALES ===
__version__: Final[str] = "3.0.0"
__author__: Final[str] = "MNEME Development Team"

# Tamaños de memoria
KB: Final[int] = 1024
MB: Final[int] = KB * 1024
GB: Final[int] = MB * 1024

# Thresholds por defecto
DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5
DEFAULT_CIRCUIT_BREAKER_TIMEOUT: Final[float] = 60.0
DEFAULT_CACHE_SIZE_MB: Final[int] = 1024
DEFAULT_MAX_TENSOR_SIZE_MB: Final[int] = 2048
DEFAULT_LAZY_MEMORY_LIMIT_MB: Final[int] = 512

# Configurar backend de TensorLy
try:
    tl.set_backend('pytorch')
except Exception as e:
    warnings.warn(f"Could not set TensorLy backend to PyTorch: {e}")

# === TYPE VARIABLES ===
T = TypeVar('T')
TensorType = TypeVar('TensorType', bound=torch.Tensor)
ConfigType = TypeVar('ConfigType', bound='MnemeConfig')


# ============================================================================
# ENUMS Y TIPOS
# ============================================================================

class LockType(Enum):
    """Tipos de locks granulares para control de concurrencia"""
    READ = "read"
    WRITE = "write"
    CACHE = "cache"
    STORAGE = "storage"
    SECURITY = "security"
    COMPRESSION = "compression"
    PIPELINE = "pipeline"


class DecompType(Enum):
    """Tipos de descomposición tensorial soportados"""
    TT = "tt"           # Tensor Train
    CP = "cp"           # CANDECOMP/PARAFAC
    TUCKER = "tucker"   # Tucker decomposition
    SVD = "svd"         # Singular Value Decomposition
    RAW = "raw"         # Sin descomposición
    SPARSE = "sparse"   # Representación sparse
    QUANTIZED = "quantized"  # Cuantizado
    ADAPTIVE = "adaptive"    # Selección adaptativa


class CompressionLevel(IntEnum):
    """Niveles de compresión con valores numéricos"""
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
    """Niveles de seguridad para operaciones"""
    NONE = "none"
    HMAC = "hmac"
    ENCRYPTED = "encrypted"
    SIGNED = "signed"
    SAFETENSORS = "safetensors"


class TensorEncryptionMode(Enum):
    """Modos de encriptación de tensores"""
    AES_GCM = "aes_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    QUANTUM_SAFE = "quantum_safe"


class KeyRotationPolicy(Enum):
    """Políticas de rotación de llaves"""
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    MANUAL = "manual"
    ADAPTIVE = "adaptive"


class StorageBackend(Enum):
    """Backends de almacenamiento"""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    HYBRID = "hybrid"
    S3 = "s3"


class CachePolicy(Enum):
    """Políticas de cache"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    LIFO = "lifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    ARC = "arc"  # Adaptive Replacement Cache


class CompressionStrategy(Enum):
    """Estrategias de compresión"""
    LZ4 = "lz4"
    ZLIB = "zlib"
    LZMA = "lzma"
    ZSTD = "zstd"
    ADAPTIVE = "adaptive"


class CircuitState(Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"      # Operación normal
    OPEN = "open"          # Rechazando requests
    HALF_OPEN = "half_open"  # Probando recuperación


class HealthStatus(Enum):
    """Estados de salud del sistema"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


class PipelineStage(Enum):
    """Etapas del pipeline de procesamiento"""
    PRE_VALIDATE = "pre_validate"
    TRANSFORM = "transform"
    COMPRESS = "compress"
    ENCRYPT = "encrypt"
    SERIALIZE = "serialize"
    STORE = "store"
    POST_PROCESS = "post_process"


class EventType(Enum):
    """Tipos de eventos del sistema"""
    TENSOR_STORED = "tensor_stored"
    TENSOR_LOADED = "tensor_loaded"
    TENSOR_DELETED = "tensor_deleted"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_CLOSE = "circuit_close"
    ERROR = "error"
    WARNING = "warning"
    METRIC = "metric"


# ============================================================================
# PROTOCOLOS Y TIPOS ABSTRACTOS
# ============================================================================

@runtime_checkable
class Plugin(Protocol):
    """Protocolo base para plugins de MNEME"""
    
    @property
    def name(self) -> str:
        """Nombre único del plugin"""
        ...
    
    @property
    def version(self) -> str:
        """Versión del plugin"""
        ...
    
    def initialize(self, context: 'MnemeContext') -> None:
        """Inicializar el plugin con el contexto de MNEME"""
        ...
    
    def cleanup(self) -> None:
        """Limpiar recursos del plugin"""
        ...


@runtime_checkable
class TensorTransformer(Protocol):
    """Protocolo para transformadores de tensores"""
    
    def transform(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Transformar un tensor"""
        ...
    
    def inverse_transform(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Transformación inversa"""
        ...


@runtime_checkable
class MetricsCollector(Protocol):
    """Protocolo para colectores de métricas"""
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Registrar una métrica"""
        ...
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Registrar en histograma"""
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener todas las métricas"""
        ...


@runtime_checkable
class EventHandler(Protocol):
    """Protocolo para manejadores de eventos"""
    
    def handle(self, event: 'MnemeEvent') -> None:
        """Manejar un evento"""
        ...


# ============================================================================
# SISTEMA DE ERRORES MEJORADO
# ============================================================================

class MnemeError(Exception):
    """Error base de MNEME con información contextual y tracing"""
    
    def __init__(
        self,
        message: str,
        error_code: str = None,
        context: Dict[str, Any] = None,
        cause: Exception = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "MNEME_ERROR"
        self.context = context or {}
        self.cause = cause
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.trace_id = str(uuid.uuid4())[:8]
        self.stack_trace = traceback.format_exc() if cause else None
    
    def __str__(self) -> str:
        base_msg = f"[{self.error_code}:{self.trace_id}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        return base_msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir error a diccionario para logging/serialización"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "recoverable": self.recoverable,
            "stack_trace": self.stack_trace
        }


class SecurityError(MnemeError):
    """Error de seguridad con detalles de amenaza"""
    
    def __init__(self, message: str, security_level: str = None, 
                 threat_type: str = None, **kwargs):
        context = kwargs.pop('context', {})
        if security_level:
            context['security_level'] = security_level
        if threat_type:
            context['threat_type'] = threat_type
        super().__init__(message, "SECURITY_ERROR", context, 
                        recoverable=False, **kwargs)


class ValidationError(MnemeError):
    """Error de validación con información del campo"""
    
    def __init__(self, message: str, field_name: str = None, 
                 expected_type: str = None, actual_value: Any = None, **kwargs):
        context = kwargs.pop('context', {})
        if field_name:
            context['field_name'] = field_name
        if expected_type:
            context['expected_type'] = expected_type
        if actual_value is not None:
            context['actual_value'] = str(actual_value)[:100]
        super().__init__(message, "VALIDATION_ERROR", context, **kwargs)


class StorageError(MnemeError):
    """Error de almacenamiento con información del backend"""
    
    def __init__(self, message: str, backend: str = None, 
                 operation: str = None, **kwargs):
        context = kwargs.pop('context', {})
        if backend:
            context['backend'] = backend
        if operation:
            context['operation'] = operation
        super().__init__(message, "STORAGE_ERROR", context, **kwargs)


class CompressionError(MnemeError):
    """Error de compresión/descompresión"""
    
    def __init__(self, message: str, compression_type: str = None, **kwargs):
        context = kwargs.pop('context', {})
        if compression_type:
            context['compression_type'] = compression_type
        super().__init__(message, "COMPRESSION_ERROR", context, **kwargs)


class CircuitBreakerError(MnemeError):
    """Error cuando el circuit breaker está abierto"""
    
    def __init__(self, message: str, circuit_name: str = None, 
                 failure_count: int = None, **kwargs):
        context = kwargs.pop('context', {})
        if circuit_name:
            context['circuit_name'] = circuit_name
        if failure_count:
            context['failure_count'] = failure_count
        super().__init__(message, "CIRCUIT_BREAKER_ERROR", context, 
                        recoverable=True, **kwargs)


class PipelineError(MnemeError):
    """Error en el pipeline de procesamiento"""
    
    def __init__(self, message: str, stage: PipelineStage = None, **kwargs):
        context = kwargs.pop('context', {})
        if stage:
            context['pipeline_stage'] = stage.value
        super().__init__(message, "PIPELINE_ERROR", context, **kwargs)


class ResourceError(MnemeError):
    """Error de recursos (memoria, CPU, etc.)"""
    
    def __init__(self, message: str, resource_type: str = None, 
                 usage: float = None, limit: float = None, **kwargs):
        context = kwargs.pop('context', {})
        if resource_type:
            context['resource_type'] = resource_type
        if usage is not None:
            context['usage'] = usage
        if limit is not None:
            context['limit'] = limit
        super().__init__(message, "RESOURCE_ERROR", context, **kwargs)


# ============================================================================
# DATACLASSES DE EVENTOS Y MÉTRICAS
# ============================================================================

@dataclass(frozen=True)
class MnemeEvent:
    """Evento del sistema MNEME"""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "mneme_core"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "data": self.data,
            "source": self.source
        }


@dataclass
class MetricPoint:
    """Punto de métrica con timestamp y tags"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class LatencyHistogram:
    """Histograma de latencias con percentiles"""
    name: str
    values: List[float] = field(default_factory=list)
    max_size: int = 10000
    
    def record(self, value: float) -> None:
        self.values.append(value)
        if len(self.values) > self.max_size:
            self.values = self.values[-self.max_size:]
    
    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]
    
    def get_stats(self) -> Dict[str, float]:
        if not self.values:
            return {"p50": 0, "p95": 0, "p99": 0, "avg": 0, "min": 0, "max": 0}
        return {
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "avg": sum(self.values) / len(self.values),
            "min": min(self.values),
            "max": max(self.values),
            "count": len(self.values)
        }


# ============================================================================
# CIRCUIT BREAKER NATIVO
# ============================================================================

class CircuitBreaker:
    """
    Circuit Breaker para protección contra fallos en cascada.
    
    Estados:
    - CLOSED: Operación normal, contando fallos
    - OPEN: Rechazando requests, esperando timeout
    - HALF_OPEN: Permitiendo requests de prueba
    """
    
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        reset_timeout: float = DEFAULT_CIRCUIT_BREAKER_TIMEOUT,
        half_open_max_calls: int = 3,
        success_threshold: int = 2
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = Lock()
        
        # Métricas
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._state_changes: List[Tuple[float, CircuitState]] = []
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._check_state_transition()
            return self._state
    
    def _check_state_transition(self) -> None:
        """Verificar y realizar transiciones de estado"""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               time.time() - self._last_failure_time >= self.reset_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transicionar a nuevo estado"""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self._state_changes.append((time.time(), new_state))
            
            # Reset contadores según el nuevo estado
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._success_count = 0
            elif new_state == CircuitState.CLOSED:
                self._failure_count = 0
            
            logger.info(f"Circuit '{self.name}' transitioned: {old_state.value} -> {new_state.value}")
    
    def record_success(self) -> None:
        """Registrar operación exitosa"""
        with self._lock:
            self._total_calls += 1
            self._total_successes += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    def record_failure(self, error: Exception = None) -> None:
        """Registrar fallo"""
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def allow_request(self) -> bool:
        """Verificar si se permite una request"""
        with self._lock:
            self._check_state_transition()
            
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                return False
            else:  # HALF_OPEN
                self._half_open_calls += 1
                return self._half_open_calls <= self.half_open_max_calls
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del circuit breaker"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "reset_timeout": self.reset_timeout,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "failure_rate": self._total_failures / max(self._total_calls, 1),
                "last_failure_time": self._last_failure_time,
                "state_changes": len(self._state_changes)
            }
    
    def reset(self) -> None:
        """Reset manual del circuit breaker"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            logger.info(f"Circuit '{self.name}' manually reset")
    
    @contextmanager
    def protect(self):
        """Context manager para proteger operaciones"""
        if not self.allow_request():
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is open",
                circuit_name=self.name,
                failure_count=self._failure_count
            )
        
        try:
            yield
            self.record_success()
        except Exception as e:
            self.record_failure(e)
            raise


def circuit_breaker_decorator(
    breaker: CircuitBreaker = None,
    name: str = "default",
    failure_threshold: int = 5,
    reset_timeout: float = 60.0
):
    """Decorador para aplicar circuit breaker a funciones"""
    def decorator(func: Callable) -> Callable:
        nonlocal breaker
        if breaker is None:
            breaker = CircuitBreaker(
                name=name or func.__name__,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout
            )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with breaker.protect():
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with breaker.protect():
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


# ============================================================================
# SISTEMA DE MÉTRICAS Y OBSERVABILIDAD
# ============================================================================

class MetricsRegistry:
    """
    Registro central de métricas compatible con OpenTelemetry.
    
    Proporciona:
    - Contadores
    - Gauges
    - Histogramas
    - Tags/Labels
    - Exportación a múltiples formatos
    """
    
    def __init__(self, namespace: str = "mneme"):
        self.namespace = namespace
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, LatencyHistogram] = {}
        self._lock = Lock()
        self._start_time = time.time()
        self._metric_history: deque = deque(maxlen=10000)
    
    def counter(self, name: str, value: int = 1, tags: Dict[str, str] = None) -> None:
        """Incrementar un contador"""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value
            self._record_history(name, self._counters[key], tags, "counter")
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Establecer un gauge"""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value
            self._record_history(name, value, tags, "gauge")
    
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Registrar en histograma"""
        key = self._make_key(name, tags)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = LatencyHistogram(name=key)
            self._histograms[key].record(value)
            self._record_history(name, value, tags, "histogram")
    
    def _make_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Crear clave única para métrica"""
        base = f"{self.namespace}.{name}"
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{base}{{{tag_str}}}"
        return base
    
    def _record_history(self, name: str, value: float, 
                       tags: Dict[str, str], metric_type: str) -> None:
        """Registrar en historial"""
        self._metric_history.append(MetricPoint(
            name=f"{self.namespace}.{name}",
            value=value,
            tags=tags or {},
            unit=metric_type
        ))
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Obtener todas las métricas"""
        with self._lock:
            return {
                "namespace": self.namespace,
                "uptime_seconds": time.time() - self._start_time,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: hist.get_stats() 
                    for name, hist in self._histograms.items()
                },
                "total_metrics_recorded": len(self._metric_history)
            }
    
    def export_prometheus(self) -> str:
        """Exportar métricas en formato Prometheus"""
        lines = []
        with self._lock:
            for name, value in self._counters.items():
                lines.append(f"{name.replace('.', '_')}_total {value}")
            for name, value in self._gauges.items():
                lines.append(f"{name.replace('.', '_')} {value}")
            for name, hist in self._histograms.items():
                stats = hist.get_stats()
                base_name = name.replace('.', '_')
                lines.append(f"{base_name}_p50 {stats['p50']}")
                lines.append(f"{base_name}_p95 {stats['p95']}")
                lines.append(f"{base_name}_p99 {stats['p99']}")
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Resetear todas las métricas"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._metric_history.clear()
            self._start_time = time.time()


# ============================================================================
# SISTEMA DE EVENTOS
# ============================================================================

class EventBus:
    """
    Bus de eventos para comunicación desacoplada entre componentes.
    
    Soporta:
    - Suscripción por tipo de evento
    - Handlers síncronos y asíncronos
    - Prioridades de handlers
    - Filtros de eventos
    """
    
    def __init__(self):
        self._handlers: Dict[EventType, List[Tuple[int, EventHandler]]] = defaultdict(list)
        self._async_handlers: Dict[EventType, List[Tuple[int, Callable]]] = defaultdict(list)
        self._lock = Lock()
        self._event_history: deque = deque(maxlen=1000)
        self._stats = defaultdict(int)
    
    def subscribe(
        self,
        event_type: EventType,
        handler: Union[EventHandler, Callable],
        priority: int = 0,
        is_async: bool = False
    ) -> None:
        """Suscribir handler a tipo de evento"""
        with self._lock:
            if is_async:
                self._async_handlers[event_type].append((priority, handler))
                self._async_handlers[event_type].sort(key=lambda x: -x[0])
            else:
                self._handlers[event_type].append((priority, handler))
                self._handlers[event_type].sort(key=lambda x: -x[0])
    
    def unsubscribe(self, event_type: EventType, handler: Any) -> bool:
        """Desuscribir handler"""
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            for i, (_, h) in enumerate(handlers):
                if h == handler:
                    handlers.pop(i)
                    return True
            
            async_handlers = self._async_handlers.get(event_type, [])
            for i, (_, h) in enumerate(async_handlers):
                if h == handler:
                    async_handlers.pop(i)
                    return True
        return False
    
    def emit(self, event: MnemeEvent) -> None:
        """Emitir evento a todos los handlers suscritos"""
        self._event_history.append(event)
        self._stats[event.event_type.value] += 1
        
        with self._lock:
            handlers = self._handlers.get(event.event_type, [])
        
        for _, handler in handlers:
            try:
                if hasattr(handler, 'handle'):
                    handler.handle(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    async def emit_async(self, event: MnemeEvent) -> None:
        """Emitir evento de forma asíncrona"""
        self._event_history.append(event)
        self._stats[event.event_type.value] += 1
        
        with self._lock:
            handlers = self._async_handlers.get(event.event_type, [])
        
        tasks = []
        for _, handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                elif hasattr(handler, 'handle'):
                    handler.handle(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in async event handler: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del event bus"""
        return {
            "events_emitted": dict(self._stats),
            "handlers_registered": {
                et.value: len(self._handlers.get(et, [])) + len(self._async_handlers.get(et, []))
                for et in EventType
            },
            "history_size": len(self._event_history)
        }


# ============================================================================
# SISTEMA DE PLUGINS
# ============================================================================

class PluginRegistry:
    """
    Registro de plugins para extensibilidad.
    
    Permite:
    - Registro de plugins por categoría
    - Lifecycle management (init/cleanup)
    - Descubrimiento de plugins
    - Dependencias entre plugins
    """
    
    def __init__(self, context: 'MnemeContext' = None):
        self._plugins: Dict[str, Plugin] = {}
        self._categories: Dict[str, List[str]] = defaultdict(list)
        self._lock = Lock()
        self._context = context
        self._initialized = False
    
    def register(self, plugin: Plugin, category: str = "general") -> bool:
        """Registrar un plugin"""
        with self._lock:
            if plugin.name in self._plugins:
                logger.warning(f"Plugin '{plugin.name}' already registered")
                return False
            
            self._plugins[plugin.name] = plugin
            self._categories[category].append(plugin.name)
            
            if self._initialized and self._context:
                try:
                    plugin.initialize(self._context)
                except Exception as e:
                    logger.error(f"Failed to initialize plugin '{plugin.name}': {e}")
                    del self._plugins[plugin.name]
                    self._categories[category].remove(plugin.name)
                    return False
            
            logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
            return True
    
    def unregister(self, name: str) -> bool:
        """Desregistrar un plugin"""
        with self._lock:
            if name not in self._plugins:
                return False
            
            plugin = self._plugins[name]
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin '{name}': {e}")
            
            del self._plugins[name]
            for category, plugins in self._categories.items():
                if name in plugins:
                    plugins.remove(name)
            
            logger.info(f"Unregistered plugin: {name}")
            return True
    
    def get(self, name: str) -> Optional[Plugin]:
        """Obtener plugin por nombre"""
        return self._plugins.get(name)
    
    def get_by_category(self, category: str) -> List[Plugin]:
        """Obtener plugins por categoría"""
        names = self._categories.get(category, [])
        return [self._plugins[name] for name in names if name in self._plugins]
    
    def initialize_all(self, context: 'MnemeContext') -> None:
        """Inicializar todos los plugins"""
        self._context = context
        with self._lock:
            for name, plugin in self._plugins.items():
                try:
                    plugin.initialize(context)
                    logger.debug(f"Initialized plugin: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize plugin '{name}': {e}")
            self._initialized = True
    
    def cleanup_all(self) -> None:
        """Limpiar todos los plugins"""
        with self._lock:
            for name, plugin in self._plugins.items():
                try:
                    plugin.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up plugin '{name}': {e}")
            self._initialized = False
    
    def list_plugins(self) -> Dict[str, Any]:
        """Listar todos los plugins registrados"""
        return {
            "plugins": {
                name: {"version": p.version, "category": self._get_category(name)}
                for name, p in self._plugins.items()
            },
            "categories": dict(self._categories)
        }
    
    def _get_category(self, plugin_name: str) -> str:
        for category, plugins in self._categories.items():
            if plugin_name in plugins:
                return category
        return "unknown"


# ============================================================================
# PIPELINE DE PROCESAMIENTO
# ============================================================================

@dataclass
class PipelineContext:
    """Contexto que fluye a través del pipeline"""
    tensor: torch.Tensor
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage: PipelineStage = PipelineStage.PRE_VALIDATE
    errors: List[Exception] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value
    
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


class PipelineStageHandler(ABC):
    """Handler abstracto para etapas del pipeline"""
    
    @property
    @abstractmethod
    def stage(self) -> PipelineStage:
        """Etapa que maneja este handler"""
        pass
    
    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Procesar el contexto del pipeline"""
        pass
    
    def can_skip(self, context: PipelineContext) -> bool:
        """Determinar si se puede saltar esta etapa"""
        return False


class ProcessingPipeline:
    """
    Pipeline de procesamiento extensible.
    
    Permite:
    - Añadir/remover etapas dinámicamente
    - Hooks pre/post para cada etapa
    - Manejo de errores por etapa
    - Métricas por etapa
    """
    
    def __init__(self, metrics: MetricsRegistry = None):
        self._stages: Dict[PipelineStage, List[PipelineStageHandler]] = defaultdict(list)
        self._pre_hooks: Dict[PipelineStage, List[Callable]] = defaultdict(list)
        self._post_hooks: Dict[PipelineStage, List[Callable]] = defaultdict(list)
        self._metrics = metrics or MetricsRegistry()
        self._lock = Lock()
    
    def add_stage_handler(self, handler: PipelineStageHandler) -> None:
        """Añadir handler a una etapa"""
        with self._lock:
            self._stages[handler.stage].append(handler)
    
    def remove_stage_handler(self, handler: PipelineStageHandler) -> bool:
        """Remover handler de una etapa"""
        with self._lock:
            if handler in self._stages[handler.stage]:
                self._stages[handler.stage].remove(handler)
                return True
            return False
    
    def add_pre_hook(self, stage: PipelineStage, hook: Callable) -> None:
        """Añadir hook pre-procesamiento"""
        self._pre_hooks[stage].append(hook)
    
    def add_post_hook(self, stage: PipelineStage, hook: Callable) -> None:
        """Añadir hook post-procesamiento"""
        self._post_hooks[stage].append(hook)
    
    def process(self, context: PipelineContext) -> PipelineContext:
        """Ejecutar el pipeline completo"""
        stages_order = [
            PipelineStage.PRE_VALIDATE,
            PipelineStage.TRANSFORM,
            PipelineStage.COMPRESS,
            PipelineStage.ENCRYPT,
            PipelineStage.SERIALIZE,
            PipelineStage.STORE,
            PipelineStage.POST_PROCESS,
        ]
        
        for stage in stages_order:
            context.stage = stage
            stage_start = time.time()
            
            try:
                # Pre-hooks
                for hook in self._pre_hooks.get(stage, []):
                    context = hook(context) or context
                
                # Handlers de la etapa
                for handler in self._stages.get(stage, []):
                    if not handler.can_skip(context):
                        context = handler.process(context)
                
                # Post-hooks
                for hook in self._post_hooks.get(stage, []):
                    context = hook(context) or context
                
                # Métricas
                stage_time = (time.time() - stage_start) * 1000
                self._metrics.histogram(
                    f"pipeline.stage.{stage.value}.duration_ms",
                    stage_time,
                    {"tensor_name": context.name}
                )
                
            except Exception as e:
                context.errors.append(e)
                logger.error(f"Pipeline error at stage {stage.value}: {e}")
                raise PipelineError(f"Pipeline failed at {stage.value}", stage=stage, cause=e)
        
        # Métricas totales
        self._metrics.histogram(
            "pipeline.total.duration_ms",
            context.elapsed_ms(),
            {"tensor_name": context.name}
        )
        
        return context
    
    async def process_async(self, context: PipelineContext) -> PipelineContext:
        """Ejecutar pipeline de forma asíncrona"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.process, context
        )


# ============================================================================
# CONFIGURACIÓN PRINCIPAL
# ============================================================================

@dataclass
class MnemeConfig:
    """
    Configuración principal de MNEME con validaciones y defaults inteligentes.
    """
    
    # === CONFIGURACIÓN BÁSICA ===
    cache_size_mb: int = DEFAULT_CACHE_SIZE_MB
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
    preferred_device: Optional[str] = None
    
    # === CONFIGURACIÓN DE PARALELIZACIÓN ===
    max_workers: int = 4
    enable_parallel_processing: bool = True
    thread_pool_size: int = 8
    process_pool_size: int = 2
    
    # === CONFIGURACIÓN DE VALIDACIÓN ===
    validate_inputs: bool = True
    max_tensor_size_mb: int = DEFAULT_MAX_TENSOR_SIZE_MB
    max_batch_size: int = 1000
    strict_validation: bool = True
    
    # === CONFIGURACIÓN DE RENDIMIENTO ===
    enable_lazy_loading: bool = True
    enable_memory_mapping: bool = True
    enable_async_operations: bool = True
    batch_processing_size: int = 100
    lazy_tensor_memory_limit: int = DEFAULT_LAZY_MEMORY_LIMIT_MB
    
    # === CONFIGURACIÓN DE MONITOREO ===
    enable_metrics: bool = True
    metrics_interval: float = 60.0
    enable_profiling: bool = False
    log_level: str = "INFO"
    
    # === CONFIGURACIÓN DE RESILIENCIA ===
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD
    circuit_breaker_timeout: float = DEFAULT_CIRCUIT_BREAKER_TIMEOUT
    enable_retry: bool = True
    max_retries: int = 3
    retry_backoff: float = 1.0
    
    # === CONFIGURACIÓN DE PLUGINS ===
    enable_plugins: bool = True
    plugin_directories: List[str] = field(default_factory=list)
    auto_load_plugins: bool = True
    
    def __post_init__(self):
        """Validar configuración después de inicialización"""
        self._validate_config()
        self._apply_defaults()
    
    def _validate_config(self) -> None:
        """Validar todos los parámetros"""
        if self.cache_size_mb <= 0:
            raise ValidationError(
                "cache_size_mb must be positive",
                field_name="cache_size_mb",
                expected_type="int > 0"
            )
        
        if not 0.0 < self.gpu_memory_fraction <= 1.0:
            raise ValidationError(
                "gpu_memory_fraction must be between 0.0 and 1.0",
                field_name="gpu_memory_fraction",
                expected_type="float in range (0.0, 1.0]"
            )
        
        if self.max_workers <= 0:
            raise ValidationError(
                "max_workers must be positive",
                field_name="max_workers",
                expected_type="int > 0"
            )
        
        if self.secret_key is not None and len(self.secret_key) < 32:
            raise ValidationError(
                "secret_key must be at least 32 bytes",
                field_name="secret_key",
                expected_type="bytes with len >= 32"
            )
        
        if self.enable_distributed_cache and not self.redis_url:
            raise ValidationError(
                "redis_url is required when distributed cache is enabled",
                field_name="redis_url"
            )
    
    def _apply_defaults(self) -> None:
        """Aplicar valores por defecto inteligentes"""
        # Auto-detectar workers óptimos
        if self.max_workers == 4:
            cpu_count = mp.cpu_count() or 4
            self.max_workers = min(32, cpu_count)
        
        # Generar secret_key si no se proporciona
        if self.secret_key is None and self.enable_encryption:
            self.secret_key = secrets.token_bytes(32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, Enum):
                result[field_name] = value.value
            elif isinstance(value, bytes):
                result[field_name] = base64.b64encode(value).decode()
            else:
                result[field_name] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MnemeConfig':
        """Crear desde diccionario"""
        processed = {}
        enum_mappings = {
            'compression_level': CompressionLevel,
            'security_level': SecurityLevel,
            'serialization_format': SerializationFormat,
            'storage_backend': StorageBackend,
            'cache_policy': CachePolicy,
            'key_rotation_policy': KeyRotationPolicy,
            'encryption_mode': TensorEncryptionMode,
            'compression_strategy': CompressionStrategy,
        }
        
        for key, value in data.items():
            if key in enum_mappings and isinstance(value, (str, int)):
                processed[key] = enum_mappings[key](value)
            elif key == 'secret_key' and isinstance(value, str):
                processed[key] = base64.b64decode(value)
            else:
                processed[key] = value
        
        return cls(**processed)
    
    @classmethod
    def production(cls) -> 'MnemeConfig':
        """Configuración optimizada para producción"""
        return cls(
            cache_size_mb=4096,
            compression_level=CompressionLevel.HIGH,
            security_level=SecurityLevel.ENCRYPTED,
            enable_metrics=True,
            enable_profiling=False,
            enable_circuit_breaker=True,
            log_level="WARNING"
        )
    
    @classmethod
    def development(cls) -> 'MnemeConfig':
        """Configuración para desarrollo"""
        return cls(
            cache_size_mb=512,
            compression_level=CompressionLevel.FAST,
            security_level=SecurityLevel.HMAC,
            enable_metrics=True,
            enable_profiling=True,
            log_level="DEBUG"
        )


# ============================================================================
# CONTEXTO DE MNEME
# ============================================================================

@dataclass
class MnemeContext:
    """
    Contexto compartido entre componentes de MNEME.
    
    Proporciona acceso centralizado a:
    - Configuración
    - Métricas
    - Event Bus
    - Plugin Registry
    - Circuit Breakers
    """
    config: MnemeConfig
    metrics: MetricsRegistry = field(default_factory=lambda: MetricsRegistry())
    event_bus: EventBus = field(default_factory=EventBus)
    plugins: PluginRegistry = field(default=None)
    device: torch.device = field(default=None)
    
    def __post_init__(self):
        if self.plugins is None:
            self.plugins = PluginRegistry(self)
        if self.device is None:
            self.device = self._detect_device()
    
    def _detect_device(self) -> torch.device:
        """Detectar dispositivo óptimo"""
        if self.config.preferred_device:
            return torch.device(self.config.preferred_device)
        
        if self.config.use_gpu:
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
        
        return torch.device('cpu')
    
    def emit_event(self, event_type: EventType, data: Dict[str, Any] = None) -> None:
        """Emitir evento a través del bus"""
        event = MnemeEvent(event_type=event_type, data=data or {})
        self.event_bus.emit(event)
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Registrar métrica"""
        if self.config.enable_metrics:
            self.metrics.gauge(name, value, tags)


# ============================================================================
# GESTOR DE LOCKS GRANULARES
# ============================================================================

class GranularLockManager:
    """
    Gestor de locks con granularidad fina y detección de deadlocks.
    """
    
    def __init__(self, max_locks: int = 1000, cleanup_interval: float = 300.0):
        self._locks: Dict[str, Lock] = {}
        self._rw_locks: Dict[str, 'RWLock'] = {}
        self._lock_usage: Dict[str, Dict[str, Any]] = {}
        self._lock_manager = Lock()
        self._read_locks: Dict[str, int] = defaultdict(int)
        self._write_locks: Dict[str, int] = defaultdict(int)
        
        self._max_locks = max_locks
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._deadlock_detection = True
    
    def _get_lock_key(self, resource: str, lock_type: LockType) -> str:
        return f"{lock_type.value}:{resource}"
    
    @contextmanager
    def acquire_lock(
        self,
        resource: str,
        lock_type: LockType,
        timeout: float = 30.0,
        blocking: bool = True
    ) -> Generator[None, None, None]:
        """Adquirir lock con timeout y manejo de errores"""
        lock_key = self._get_lock_key(resource, lock_type)
        
        with self._lock_manager:
            if lock_key not in self._locks:
                self._locks[lock_key] = Lock()
                self._lock_usage[lock_key] = {
                    'resource': resource,
                    'lock_type': lock_type.value,
                    'created_at': time.time(),
                    'usage_count': 0,
                    'last_used': time.time()
                }
            
            lock = self._locks[lock_key]
            self._lock_usage[lock_key]['usage_count'] += 1
            self._lock_usage[lock_key]['last_used'] = time.time()
        
        acquired = lock.acquire(blocking=blocking, timeout=timeout if blocking else -1)
        if not acquired:
            raise ConcurrencyError(
                f"Could not acquire lock for '{resource}'",
                lock_type=lock_type.value,
                resource=resource
            )
        
        try:
            yield
        finally:
            lock.release()
    
    @contextmanager
    def read_lock(self, resource: str, timeout: float = 30.0):
        """Lock de lectura (permite múltiples lectores)"""
        lock_key = f"rw:{resource}"
        
        with self._lock_manager:
            if lock_key not in self._rw_locks:
                self._rw_locks[lock_key] = RWLock()
        
        rw_lock = self._rw_locks[lock_key]
        with rw_lock.read_lock(timeout):
            self._read_locks[resource] += 1
            try:
                yield
            finally:
                self._read_locks[resource] -= 1
    
    @contextmanager
    def write_lock(self, resource: str, timeout: float = 30.0):
        """Lock de escritura (exclusivo)"""
        lock_key = f"rw:{resource}"
        
        with self._lock_manager:
            if lock_key not in self._rw_locks:
                self._rw_locks[lock_key] = RWLock()
        
        rw_lock = self._rw_locks[lock_key]
        with rw_lock.write_lock(timeout):
            self._write_locks[resource] += 1
            try:
                yield
            finally:
                self._write_locks[resource] -= 1
    
    def get_lock_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de locks"""
        with self._lock_manager:
            active_locks = sum(1 for lock in self._locks.values() if lock.locked())
            return {
                "total_locks": len(self._locks),
                "active_locks": active_locks,
                "active_readers": sum(self._read_locks.values()),
                "active_writers": sum(self._write_locks.values()),
                "lock_usage": {
                    k: v['usage_count'] for k, v in self._lock_usage.items()
                }
            }


class RWLock:
    """Read-Write Lock implementation"""
    
    def __init__(self):
        self._read_ready = Condition(Lock())
        self._readers = 0
    
    @contextmanager
    def read_lock(self, timeout: float = 30.0):
        """Adquirir lock de lectura"""
        with self._read_ready:
            self._readers += 1
        try:
            yield
        finally:
            with self._read_ready:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.notify_all()
    
    @contextmanager
    def write_lock(self, timeout: float = 30.0):
        """Adquirir lock de escritura"""
        with self._read_ready:
            while self._readers > 0:
                if not self._read_ready.wait(timeout):
                    raise TimeoutError("Write lock timeout")
            yield


# ============================================================================
# LAZY TENSOR MEJORADO
# ============================================================================

class LazyTensor:
    """
    Tensor con decompresión lazy y gestión inteligente de memoria.
    """
    
    def __init__(
        self,
        compressed_data: bytes,
        decompression_func: Callable,
        metadata: Dict[str, Any],
        device: torch.device = None,
        max_memory_mb: int = DEFAULT_LAZY_MEMORY_LIMIT_MB,
        auto_cleanup: bool = True
    ):
        self.compressed_data = compressed_data
        self.decompression_func = decompression_func
        self.metadata = metadata
        self.device = device or torch.device('cpu')
        self.max_memory_mb = max_memory_mb
        self.auto_cleanup = auto_cleanup
        
        self._decompressed_tensor: Optional[torch.Tensor] = None
        self._lock = Lock()
        self._access_count = 0
        self._last_access = time.time()
        self._creation_time = time.time()
        self._memory_usage = 0
        
        self._cached_shape = None
        self._cached_dtype = None
        self._cached_size = None
        
        self._cleanup_threshold = 0.8
        self._idle_timeout = 300.0
    
    def decompress(self, force: bool = False) -> torch.Tensor:
        """Decomprimir tensor bajo demanda"""
        with self._lock:
            if not force and self._decompressed_tensor is not None:
                self._update_access_stats()
                return self._decompressed_tensor
            
            if not self._check_memory_limits():
                if self.auto_cleanup:
                    self._cleanup_if_needed()
                else:
                    raise ResourceError(
                        "Insufficient memory for tensor decompression",
                        resource_type="memory",
                        usage=self._memory_usage,
                        limit=self.max_memory_mb * MB
                    )
            
            try:
                self._decompressed_tensor = self.decompression_func(self.compressed_data)
                
                if self.device != torch.device('cpu'):
                    self._decompressed_tensor = self._decompressed_tensor.to(self.device)
                
                self._update_access_stats()
                self._memory_usage = self._calculate_tensor_memory()
                
                return self._decompressed_tensor
                
            except Exception as e:
                logger.error(f"Failed to decompress tensor: {e}")
                raise CompressionError(
                    f"Tensor decompression failed: {e}",
                    compression_type="lz4"
                )
    
    def _check_memory_limits(self) -> bool:
        if self._decompressed_tensor is not None:
            return True
        
        estimated_memory = self._estimate_decompressed_size()
        max_memory_bytes = self.max_memory_mb * MB
        return estimated_memory <= max_memory_bytes
    
    def _estimate_decompressed_size(self) -> int:
        if self._cached_size is not None:
            return self._cached_size
        
        if 'shape' in self.metadata and 'dtype' in self.metadata:
            shape = self.metadata['shape']
            dtype_str = self.metadata['dtype']
            
            dtype_sizes = {
                'torch.float32': 4, 'torch.float64': 8,
                'torch.int32': 4, 'torch.int64': 8,
                'torch.uint8': 1, 'torch.int8': 1,
                'torch.float16': 2, 'torch.bfloat16': 2
            }
            
            element_size = dtype_sizes.get(dtype_str, 4)
            total_elements = 1
            for dim in shape:
                total_elements *= dim
            
            self._cached_size = total_elements * element_size
            return self._cached_size
        
        compression_ratio = self.metadata.get('compression_ratio', 0.1)
        estimated_size = int(len(self.compressed_data) / max(compression_ratio, 0.01))
        self._cached_size = estimated_size
        return estimated_size
    
    def _calculate_tensor_memory(self) -> int:
        if self._decompressed_tensor is None:
            return 0
        return self._decompressed_tensor.numel() * self._decompressed_tensor.element_size()
    
    def _update_access_stats(self) -> None:
        self._access_count += 1
        self._last_access = time.time()
    
    def _cleanup_if_needed(self) -> None:
        current_memory = self._calculate_tensor_memory()
        max_memory = self.max_memory_mb * MB
        
        if current_memory > max_memory * self._cleanup_threshold:
            self.clear_decompressed()
    
    def is_decompressed(self) -> bool:
        return self._decompressed_tensor is not None
    
    def clear_decompressed(self) -> None:
        with self._lock:
            if self._decompressed_tensor is not None:
                del self._decompressed_tensor
                self._decompressed_tensor = None
                self._memory_usage = 0
                
                if self.auto_cleanup:
                    gc.collect()
    
    def get_shape(self) -> Tuple[int, ...]:
        if self._cached_shape is not None:
            return self._cached_shape
        
        if 'shape' in self.metadata:
            self._cached_shape = tuple(self.metadata['shape'])
        else:
            temp_tensor = self.decompress()
            self._cached_shape = tuple(temp_tensor.shape)
            if self.auto_cleanup:
                self.clear_decompressed()
        
        return self._cached_shape
    
    def get_memory_usage(self) -> Dict[str, Any]:
        compressed_size = len(self.compressed_data)
        decompressed_size = self._calculate_tensor_memory()
        
        return {
            "compressed_bytes": compressed_size,
            "decompressed_bytes": decompressed_size,
            "compression_ratio": compressed_size / max(decompressed_size, 1) if decompressed_size > 0 else 0,
            "is_decompressed": self.is_decompressed(),
            "access_count": self._access_count,
            "last_access": self._last_access,
            "memory_pressure": decompressed_size / (self.max_memory_mb * MB) if self.max_memory_mb > 0 else 0
        }
    
    def is_idle(self, timeout: float = None) -> bool:
        if timeout is None:
            timeout = self._idle_timeout
        return time.time() - self._last_access > timeout
    
    def __del__(self):
        try:
            self.clear_decompressed()
        except:
            pass


# ============================================================================
# CACHE ADAPTATIVO MEJORADO
# ============================================================================

class AdaptiveCache:
    """
    Cache adaptativo con múltiples estrategias y métricas detalladas.
    """
    
    def __init__(
        self,
        max_size_bytes: int,
        strategy: str = "adaptive",
        ttl_seconds: float = 3600.0,
        compression_threshold: int = 1024
    ):
        self.max_size_bytes = max_size_bytes
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        self.compression_threshold = compression_threshold
        self.current_size = 0
        
        self._lru_order = deque()
        self._lfu_counts: Dict[str, int] = {}
        self._ttl_expiry: Dict[str, float] = {}
        
        self._cache: Dict[str, Any] = {}
        self._lock = Lock()
        self._access_times: Dict[str, float] = {}
        self._access_frequencies: Dict[str, int] = defaultdict(int)
        self._creation_times: Dict[str, float] = {}
        
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener elemento del cache"""
        with self._lock:
            self._cleanup_if_needed()
            
            if key not in self._cache:
                self._miss_count += 1
                return None
            
            # Verificar TTL
            if key in self._ttl_expiry:
                if time.time() > self._ttl_expiry[key]:
                    self._remove_entry(key)
                    self._miss_count += 1
                    return None
            
            self._hit_count += 1
            self._access_times[key] = time.time()
            self._access_frequencies[key] += 1
            
            # Actualizar LRU
            if key in self._lru_order:
                self._lru_order.remove(key)
            self._lru_order.append(key)
            
            return self._cache[key]
    
    def put(self, key: str, value: Any, ttl: float = None) -> bool:
        """Almacenar elemento en cache"""
        with self._lock:
            size = self._estimate_size(value)
            
            # Evictar si es necesario
            while self.current_size + size > self.max_size_bytes and self._cache:
                self._evict_one()
            
            if self.current_size + size > self.max_size_bytes:
                return False
            
            # Remover entrada existente si hay
            if key in self._cache:
                self._remove_entry(key)
            
            self._cache[key] = value
            self.current_size += size
            self._access_times[key] = time.time()
            self._creation_times[key] = time.time()
            self._access_frequencies[key] = 1
            self._lru_order.append(key)
            
            if ttl is not None:
                self._ttl_expiry[key] = time.time() + ttl
            elif self.ttl_seconds > 0:
                self._ttl_expiry[key] = time.time() + self.ttl_seconds
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remover elemento del cache"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def _remove_entry(self, key: str) -> None:
        """Remover entrada y limpiar metadatos"""
        if key in self._cache:
            size = self._estimate_size(self._cache[key])
            del self._cache[key]
            self.current_size -= size
        
        self._access_times.pop(key, None)
        self._access_frequencies.pop(key, None)
        self._creation_times.pop(key, None)
        self._ttl_expiry.pop(key, None)
        self._lfu_counts.pop(key, None)
        
        if key in self._lru_order:
            self._lru_order.remove(key)
    
    def _evict_one(self) -> None:
        """Evictar un elemento según la estrategia"""
        if not self._cache:
            return
        
        key_to_evict = None
        
        if self.strategy == "lru" and self._lru_order:
            key_to_evict = self._lru_order[0]
        elif self.strategy == "lfu":
            key_to_evict = min(
                self._cache.keys(),
                key=lambda k: self._access_frequencies.get(k, 0)
            )
        elif self.strategy == "adaptive":
            key_to_evict = self._select_adaptive_eviction()
        else:
            key_to_evict = next(iter(self._cache))
        
        if key_to_evict:
            self._remove_entry(key_to_evict)
            self._eviction_count += 1
    
    def _select_adaptive_eviction(self) -> Optional[str]:
        """Selección adaptativa de elemento a evictar"""
        if not self._cache:
            return None
        
        current_time = time.time()
        scores = {}
        
        for key in self._cache:
            time_factor = 1.0 / (current_time - self._access_times.get(key, current_time) + 1)
            freq_factor = self._access_frequencies.get(key, 0)
            size_factor = 1.0 / max(self._estimate_size(self._cache[key]), 1)
            
            scores[key] = time_factor * 0.4 + freq_factor * 0.4 + size_factor * 0.2
        
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _estimate_size(self, value: Any) -> int:
        """Estimar tamaño de un valor"""
        if value is None:
            return 0
        if hasattr(value, 'get_size_bytes'):
            return value.get_size_bytes()
        if isinstance(value, bytes):
            return len(value)
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        return sys.getsizeof(value)
    
    def _cleanup_if_needed(self) -> None:
        """Limpiar entradas expiradas"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        expired_keys = [
            key for key, expiry in self._ttl_expiry.items()
            if current_time > expiry
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
        
        self._last_cleanup = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / max(total_requests, 1)
        
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
    
    def clear(self) -> None:
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


# ============================================================================
# DESCRIPTORES DE TENSOR
# ============================================================================

@dataclass
class ZDescriptor:
    """
    Descriptor de tensor con validación y verificación de integridad.
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
        if not isinstance(self.core_data, bytes):
            raise ValidationError(
                "core_data must be bytes",
                field_name="core_data"
            )
        
        if len(self.core_data) == 0:
            raise ValidationError(
                "core_data cannot be empty",
                field_name="core_data"
            )
        
        if not isinstance(self.shape, tuple) or len(self.shape) == 0:
            raise ValidationError(
                "shape must be a non-empty tuple",
                field_name="shape"
            )
        
        if any(s <= 0 for s in self.shape):
            raise ValidationError(
                "shape dimensions must be positive",
                field_name="shape"
            )
        
        if self.security_hash is None:
            self.security_hash = self._compute_security_hash()
    
    def verify_integrity(self) -> bool:
        """Verificar integridad del descriptor"""
        try:
            if not self.core_data:
                return False
            
            if not self.shape or any(s <= 0 for s in self.shape):
                return False
            
            if len(self.core_data) > GB:
                return False
            
            if self.merkle_root:
                computed_root = self._compute_merkle_root()
                if computed_root != self.merkle_root:
                    return False
            
            if self.security_hash:
                computed_hash = self._compute_security_hash()
                if computed_hash != self.security_hash:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False
    
    def _compute_merkle_root(self) -> bytes:
        return hashlib.sha256(self.core_data).digest()
    
    def _compute_security_hash(self) -> bytes:
        data = f"{self.kind}:{self.decomp_type.value}:{self.shape}:{self.version}".encode()
        return hashlib.sha256(data).digest()
    
    def update_access(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_size_bytes(self) -> int:
        size = len(self.core_data)
        if self.delta_chain:
            size += len(self.delta_chain)
        return size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'kind': self.kind,
            'decomp_type': self.decomp_type.value,
            'shape': list(self.shape),
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
        processed = data.copy()
        
        if 'core_data' in processed:
            processed['core_data'] = base64.b64decode(processed['core_data'])
        if 'merkle_root' in processed and processed['merkle_root']:
            processed['merkle_root'] = base64.b64decode(processed['merkle_root'])
        if 'delta_chain' in processed and processed['delta_chain']:
            processed['delta_chain'] = base64.b64decode(processed['delta_chain'])
        if 'security_hash' in processed and processed['security_hash']:
            processed['security_hash'] = base64.b64decode(processed['security_hash'])
        
        if 'decomp_type' in processed:
            processed['decomp_type'] = DecompType(processed['decomp_type'])
        if 'compression_level' in processed:
            processed['compression_level'] = CompressionLevel(processed['compression_level'])
        if 'shape' in processed and isinstance(processed['shape'], list):
            processed['shape'] = tuple(processed['shape'])
        
        return cls(**processed)


@dataclass
class ZAddr:
    """Dirección de descriptor con validación"""
    addr: bytes
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not isinstance(self.addr, bytes):
            raise ValidationError("addr must be bytes", field_name="addr")
        
        if len(self.addr) != 32:
            raise ValidationError(
                f"addr must be 32 bytes, got {len(self.addr)}",
                field_name="addr"
            )
    
    @classmethod
    def compute(cls, desc: ZDescriptor) -> 'ZAddr':
        data_parts = [
            desc.kind,
            desc.decomp_type.value,
            str(desc.shape),
            str(desc.version),
            str(desc.compression_level.value)
        ]
        
        if desc.ranks:
            data_parts.append(str(desc.ranks))
        if desc.security_hash:
            data_parts.append(desc.security_hash.hex())
        
        data = ":".join(data_parts).encode()
        addr = hashlib.sha256(data).digest()
        return cls(addr)
    
    @classmethod
    def from_hex(cls, hex_str: str) -> 'ZAddr':
        try:
            addr_bytes = bytes.fromhex(hex_str)
            return cls(addr_bytes)
        except ValueError as e:
            raise ValidationError(f"Invalid hex string: {e}", field_name="hex_str")
    
    def hex(self) -> str:
        return self.addr.hex()
    
    def short_hex(self, length: int = 8) -> str:
        return self.addr.hex()[:length]
    
    def __str__(self) -> str:
        return f"ZAddr({self.short_hex()})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ZAddr):
            return False
        return self.addr == other.addr
    
    def __hash__(self) -> int:
        return hash(self.addr)


# ============================================================================
# CLASE PRINCIPAL: ZSPACE
# ============================================================================

class ZSpace:
    """
    Interfaz principal del runtime MNEME V3.
    
    Características:
    - Arquitectura de plugins
    - Circuit breaker integrado
    - Pipeline de procesamiento extensible
    - Métricas OpenTelemetry-compatible
    - Integración con MNEMEOptimizer
    - API unificada
    """
    
    def __init__(
        self,
        config: MnemeConfig = None,
        context: MnemeContext = None,
        auto_init: bool = True
    ):
        # Configuración
        self.config = config or MnemeConfig()
        
        # Contexto compartido
        if context:
            self.context = context
        else:
            self.context = MnemeContext(config=self.config)
        
        self.device = self.context.device
        
        # Métricas y eventos
        self.metrics = self.context.metrics
        self.event_bus = self.context.event_bus
        
        # Componentes core
        self.lock_manager = GranularLockManager()
        self.adaptive_cache = AdaptiveCache(
            max_size_bytes=self.config.cache_size_mb * MB,
            strategy="adaptive" if self.config.cache_policy == CachePolicy.ADAPTIVE else self.config.cache_policy.value,
            ttl_seconds=3600.0
        )
        
        # Circuit breakers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        if self.config.enable_circuit_breaker:
            self._circuit_breakers['storage'] = CircuitBreaker(
                name="storage",
                failure_threshold=self.config.circuit_breaker_threshold,
                reset_timeout=self.config.circuit_breaker_timeout
            )
            self._circuit_breakers['compression'] = CircuitBreaker(
                name="compression",
                failure_threshold=self.config.circuit_breaker_threshold,
                reset_timeout=self.config.circuit_breaker_timeout
            )
        
        # Pipeline de procesamiento
        self.pipeline = ProcessingPipeline(metrics=self.metrics)
        
        # Storage backend
        self.storage_backend = create_secure_storage(
            StorageConfig(
                backend=self.config.storage_backend.value,
                path=self.config.storage_path
            )
        )
        
        # Security manager
        self.security_manager = SecurityManager(
            create_secure_config(self.config.secret_key)
        )
        
        # Tablas de descriptores
        self.name_to_desc: Dict[str, ZDescriptor] = {}
        self.addr_to_desc: Dict[bytes, ZDescriptor] = {}
        self.version_graph: Dict[bytes, bytes] = {}
        
        # Métricas de storage
        self.storage_metrics = {
            "read_operations": 0,
            "write_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "storage_loads": 0,
            "storage_stores": 0,
            "tensor_count": 0,
            "total_storage_bytes": 0
        }
        
        # Hooks para integración con optimizador
        self._optimizer_hooks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Plugins
        self.plugins = self.context.plugins
        
        if auto_init:
            self._initialize()
    
    def _initialize(self) -> None:
        """Inicializar componentes"""
        logger.info(f"Initializing MNEME ZSpace V{__version__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Config: cache={self.config.cache_size_mb}MB, "
                   f"compression={self.config.compression_level.name}")
        
        # Inicializar plugins
        if self.config.enable_plugins:
            self.plugins.initialize_all(self.context)
        
        # Emitir evento de inicio
        self.context.emit_event(
            EventType.METRIC,
            {"event": "zspace_initialized", "device": str(self.device)}
        )
    
    # === API PRINCIPAL ===
    
    def store(self, name: str, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """
        Almacenar tensor con nombre.
        
        Args:
            name: Nombre único del tensor
            tensor: Tensor a almacenar
            **kwargs: Metadata adicional
            
        Returns:
            ZDescriptor del tensor almacenado
        """
        # Validaciones
        if not isinstance(name, str) or not name.strip():
            raise ValidationError(
                "Name must be a non-empty string",
                field_name="name"
            )
        
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(
                "Must provide a torch.Tensor",
                field_name="tensor",
                expected_type="torch.Tensor"
            )
        
        # Circuit breaker
        circuit = self._circuit_breakers.get('storage')
        if circuit and not circuit.allow_request():
            raise CircuitBreakerError(
                "Storage circuit breaker is open",
                circuit_name="storage"
            )
        
        start_time = time.time()
        
        try:
            with self.lock_manager.write_lock(name):
                # Ejecutar hooks pre-store
                tensor = self._run_hooks('pre_store', tensor, name=name)
                
                # Crear descriptor
                desc = self._create_secure_descriptor(tensor, **kwargs)
                
                # Verificar si existe
                old_addr = None
                if name in self.name_to_desc:
                    old_addr = ZAddr.compute(self.name_to_desc[name])
                    logger.info(f"Replacing existing tensor '{name}'")
                
                # Registrar
                addr = self._register_descriptor(name, desc, old_addr)
                
                # Cache
                self.adaptive_cache.put(f"desc_{name}", desc)
                
                # Métricas
                self.storage_metrics["write_operations"] += 1
                self.storage_metrics["tensor_count"] = len(self.name_to_desc)
                self.storage_metrics["total_storage_bytes"] += desc.get_size_bytes()
                
                # Hooks post-store
                self._run_hooks('post_store', desc, name=name)
                
                # Evento
                self.context.emit_event(
                    EventType.TENSOR_STORED,
                    {"name": name, "shape": desc.shape, "size": desc.get_size_bytes()}
                )
            
            if circuit:
                circuit.record_success()
            
            # Métrica de latencia
            self.metrics.histogram(
                "tensor.store.duration_ms",
                (time.time() - start_time) * 1000,
                {"tensor_name": name}
            )
            
            logger.info(f"Stored '{name}'. Type: {desc.decomp_type.value}. Addr: {addr.short_hex()}")
            return desc
            
        except Exception as e:
            if circuit:
                circuit.record_failure(e)
            raise
    
    def load(self, name: str) -> torch.Tensor:
        """
        Cargar tensor por nombre.
        
        Args:
            name: Nombre del tensor
            
        Returns:
            torch.Tensor cargado
        """
        if not isinstance(name, str) or not name.strip():
            raise ValidationError(
                "Name must be a non-empty string",
                field_name="name"
            )
        
        start_time = time.time()
        
        with self.lock_manager.read_lock(name):
            # Cache hit?
            cached_desc = self.adaptive_cache.get(f"desc_{name}")
            if cached_desc:
                desc = cached_desc
                self.storage_metrics["cache_hits"] += 1
                self.context.emit_event(EventType.CACHE_HIT, {"name": name})
            elif name in self.name_to_desc:
                desc = self.name_to_desc[name]
                self.adaptive_cache.put(f"desc_{name}", desc)
                self.storage_metrics["cache_misses"] += 1
                self.context.emit_event(EventType.CACHE_MISS, {"name": name})
            else:
                # Intentar cargar desde storage
                desc = self._load_from_storage(name)
                if desc:
                    self.name_to_desc[name] = desc
                    self.addr_to_desc[ZAddr.compute(desc).addr] = desc
                    self.adaptive_cache.put(f"desc_{name}", desc)
                    self.storage_metrics["storage_loads"] += 1
                else:
                    raise KeyError(f"Unknown tensor: {name}")
            
            desc.update_access()
        
        self.storage_metrics["read_operations"] += 1
        
        # Decomprimir
        if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
            tensor = desc.lazy_tensor.decompress()
        else:
            tensor = self._synthesize_tensor(desc)
        
        # Evento
        self.context.emit_event(
            EventType.TENSOR_LOADED,
            {"name": name, "shape": tuple(tensor.shape)}
        )
        
        # Métrica
        self.metrics.histogram(
            "tensor.load.duration_ms",
            (time.time() - start_time) * 1000,
            {"tensor_name": name}
        )
        
        return tensor
    
    def delete(self, name: str) -> bool:
        """Eliminar tensor"""
        success = True
        
        with self.lock_manager.write_lock(name):
            if name in self.name_to_desc:
                desc = self.name_to_desc[name]
                addr = ZAddr.compute(desc)
                
                del self.name_to_desc[name]
                if addr.addr in self.addr_to_desc:
                    del self.addr_to_desc[addr.addr]
                
                self.adaptive_cache.remove(f"desc_{name}")
                
                if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
                    desc.lazy_tensor.clear_decompressed()
            
            try:
                self.storage_backend.delete(f"desc_{name}")
                self.storage_backend.delete(f"data_{name}")
            except Exception as e:
                logger.error(f"Failed to delete '{name}' from storage: {e}")
                success = False
        
        if success:
            self.context.emit_event(EventType.TENSOR_DELETED, {"name": name})
        
        return success
    
    def exists(self, name: str) -> bool:
        """Verificar si existe un tensor"""
        if name in self.name_to_desc:
            return True
        
        try:
            desc_data = self.storage_backend.retrieve(f"desc_{name}")
            return desc_data is not None
        except:
            return False
    
    def list_tensors(self) -> Dict[str, Any]:
        """Listar todos los tensores"""
        memory_tensors = set(self.name_to_desc.keys())
        
        storage_tensors = set()
        try:
            storage_keys = self.storage_backend.list_keys()
            for key in storage_keys:
                if key.startswith("desc_"):
                    storage_tensors.add(key[5:])
        except:
            pass
        
        all_tensors = memory_tensors.union(storage_tensors)
        
        return {
            "total_tensors": len(all_tensors),
            "memory_tensors": list(memory_tensors),
            "storage_tensors": list(storage_tensors),
            "memory_only": list(memory_tensors - storage_tensors),
            "storage_only": list(storage_tensors - memory_tensors),
            "both": list(memory_tensors.intersection(storage_tensors))
        }
    
    # === INTEGRACIÓN CON OPTIMIZADOR ===
    
    def register_optimizer_hook(self, hook_name: str, callback: Callable) -> None:
        """Registrar hook para integración con MNEMEOptimizer"""
        self._optimizer_hooks[hook_name].append(callback)
    
    def _run_hooks(self, hook_name: str, data: Any, **kwargs) -> Any:
        """Ejecutar hooks registrados"""
        for hook in self._optimizer_hooks.get(hook_name, []):
            try:
                result = hook(data, **kwargs)
                if result is not None:
                    data = result
            except Exception as e:
                logger.error(f"Hook '{hook_name}' error: {e}")
        return data
    
    def get_optimization_context(self) -> Dict[str, Any]:
        """Obtener contexto para el optimizador"""
        return {
            "device": str(self.device),
            "config": self.config.to_dict(),
            "tensor_count": len(self.name_to_desc),
            "cache_stats": self.adaptive_cache.get_stats(),
            "storage_metrics": self.storage_metrics,
            "circuit_breakers": {
                name: cb.get_stats() for name, cb in self._circuit_breakers.items()
            }
        }
    
    # === MÉTODOS INTERNOS ===
    
    def _create_secure_descriptor(self, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """Crear descriptor seguro"""
        # Serializar
        core_data = secure_tensor_serialize(tensor, kwargs.get('metadata', {}))
        
        # Comprimir
        compressed_data = lz4.frame.compress(
            core_data,
            compression_level=self.config.compression_level.value
        )
        
        # Lazy tensor
        lazy_tensor = LazyTensor(
            compressed_data=compressed_data,
            decompression_func=self._decompress_tensor,
            metadata={"shape": tuple(tensor.shape), "dtype": str(tensor.dtype)},
            device=self.device,
            max_memory_mb=self.config.lazy_tensor_memory_limit
        )
        
        # Descriptor
        desc = ZDescriptor(
            kind="tensor",
            decomp_type=DecompType.RAW,
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
        """Decomprimir tensor"""
        core_data = lz4.frame.decompress(compressed_data)
        tensor, _ = secure_tensor_deserialize(core_data, self.device)
        return tensor
    
    def _synthesize_tensor(self, desc: ZDescriptor) -> torch.Tensor:
        """Sintetizar tensor desde descriptor"""
        if not desc.verify_integrity():
            raise SecurityError("Descriptor integrity check failed")
        
        core_data = lz4.frame.decompress(desc.core_data)
        tensor, _ = secure_tensor_deserialize(core_data, self.device)
        return tensor
    
    def _register_descriptor(self, name: str, desc: ZDescriptor, 
                            old_addr: Optional[ZAddr]) -> ZAddr:
        """Registrar descriptor"""
        addr = ZAddr.compute(desc)
        
        self.name_to_desc[name] = desc
        self.addr_to_desc[addr.addr] = desc
        
        if old_addr:
            self.version_graph[addr.addr] = old_addr.addr
        
        # Persistir
        try:
            desc_bytes = json.dumps(desc.to_dict()).encode('utf-8')
            self.storage_backend.store(
                key=f"desc_{name}",
                data=desc_bytes,
                metadata={'tensor_name': name, 'addr': str(addr.addr)}
            )
            
            if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
                self.storage_backend.store(
                    key=f"data_{name}",
                    data=desc.lazy_tensor.compressed_data,
                    metadata={'tensor_name': name}
                )
            
            self.storage_metrics["storage_stores"] += 1
        except Exception as e:
            logger.error(f"Failed to persist '{name}': {e}")
        
        return addr
    
    def _load_from_storage(self, name: str) -> Optional[ZDescriptor]:
        """Cargar desde storage"""
        try:
            desc_data = self.storage_backend.retrieve(f"desc_{name}")
            if not desc_data:
                return None
            
            desc_dict = json.loads(desc_data.decode('utf-8'))
            desc = ZDescriptor.from_dict(desc_dict)
            
            compressed_data = self.storage_backend.retrieve(f"data_{name}")
            if compressed_data:
                desc.lazy_tensor = LazyTensor(
                    compressed_data=compressed_data,
                    decompression_func=self._decompress_tensor,
                    metadata={'shape': desc.shape, 'dtype': 'torch.float32'},
                    device=self.device,
                    max_memory_mb=self.config.lazy_tensor_memory_limit
                )
            
            return desc
        except Exception as e:
            logger.error(f"Failed to load '{name}' from storage: {e}")
            return None
    
    # === ESTADÍSTICAS Y MONITOREO ===
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas"""
        return {
            "version": __version__,
            "device": str(self.device),
            "health": self.get_health_status().value,
            "cache": self.adaptive_cache.get_stats(),
            "storage": self.storage_backend.get_stats(),
            "security": self.security_manager.get_security_stats(),
            "locks": self.lock_manager.get_lock_stats(),
            "metrics": self.storage_metrics,
            "circuit_breakers": {
                name: cb.get_stats() for name, cb in self._circuit_breakers.items()
            },
            "plugins": self.plugins.list_plugins(),
            "events": self.event_bus.get_stats()
        }
    
    def get_health_status(self) -> HealthStatus:
        """Obtener estado de salud"""
        # Verificar circuit breakers
        for cb in self._circuit_breakers.values():
            if cb.state == CircuitState.OPEN:
                return HealthStatus.CRITICAL
            if cb.state == CircuitState.HALF_OPEN:
                return HealthStatus.RECOVERING
        
        # Verificar uso de cache
        cache_stats = self.adaptive_cache.get_stats()
        if cache_stats['usage_percent'] > 95:
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    def get_metrics_prometheus(self) -> str:
        """Exportar métricas en formato Prometheus"""
        return self.metrics.export_prometheus()
    
    # === LIFECYCLE ===
    
    def sync_to_storage(self) -> Dict[str, Any]:
        """Sincronizar todos los tensores a storage"""
        results = {
            "total": len(self.name_to_desc),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        for name, desc in self.name_to_desc.items():
            try:
                existing = self.storage_backend.retrieve(f"desc_{name}")
                if not existing:
                    addr = ZAddr.compute(desc)
                    desc_bytes = json.dumps(desc.to_dict()).encode('utf-8')
                    self.storage_backend.store(
                        key=f"desc_{name}",
                        data=desc_bytes,
                        metadata={'tensor_name': name}
                    )
                    
                    if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
                        self.storage_backend.store(
                            key=f"data_{name}",
                            data=desc.lazy_tensor.compressed_data,
                            metadata={'tensor_name': name}
                        )
                    
                    results["successful"] += 1
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{name}: {e}")
        
        return results
    
    def cleanup(self) -> None:
        """Limpiar recursos"""
        logger.info("Cleaning up MNEME ZSpace...")
        
        # Limpiar cache
        self.adaptive_cache.clear()
        
        # Limpiar lazy tensors
        for desc in self.name_to_desc.values():
            if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
                desc.lazy_tensor.clear_decompressed()
        
        # Limpiar storage
        self.storage_backend.cleanup()
        
        # Limpiar plugins
        self.plugins.cleanup_all()
        
        # GC
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
        
        logger.info("Cleanup completed")
    
    def __enter__(self) -> 'ZSpace':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


# ============================================================================
# FACADE PARA API SIMPLIFICADA
# ============================================================================

class MnemeFacade:
    """
    Facade que proporciona una API simplificada para MNEME.
    
    Uso:
        mneme = MnemeFacade()
        mneme.store("weights", tensor)
        tensor = mneme.load("weights")
    """
    
    _instance: Optional['MnemeFacade'] = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        config: MnemeConfig = None,
        auto_init: bool = True
    ):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.config = config or MnemeConfig()
        self.context = MnemeContext(config=self.config)
        self._zspace: Optional[ZSpace] = None
        
        if auto_init:
            self._zspace = ZSpace(config=self.config, context=self.context)
        
        self._initialized = True
    
    @property
    def zspace(self) -> ZSpace:
        """Obtener instancia de ZSpace"""
        if self._zspace is None:
            self._zspace = ZSpace(config=self.config, context=self.context)
        return self._zspace
    
    def store(self, name: str, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """Almacenar tensor"""
        return self.zspace.store(name, tensor, **kwargs)
    
    def load(self, name: str) -> torch.Tensor:
        """Cargar tensor"""
        return self.zspace.load(name)
    
    def delete(self, name: str) -> bool:
        """Eliminar tensor"""
        return self.zspace.delete(name)
    
    def exists(self, name: str) -> bool:
        """Verificar existencia"""
        return self.zspace.exists(name)
    
    def list(self) -> List[str]:
        """Listar tensores"""
        return self.zspace.list_tensors()['memory_tensors']
    
    def stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        return self.zspace.get_stats()
    
    def health(self) -> str:
        """Obtener estado de salud"""
        return self.zspace.get_health_status().value
    
    def cleanup(self) -> None:
        """Limpiar recursos"""
        if self._zspace:
            self._zspace.cleanup()
    
    @classmethod
    def reset_instance(cls) -> None:
        """Resetear singleton (para testing)"""
        with cls._lock:
            if cls._instance and cls._instance._zspace:
                cls._instance._zspace.cleanup()
            cls._instance = None


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def create_mneme(
    config: MnemeConfig = None,
    production: bool = False,
    development: bool = False
) -> ZSpace:
    """Factory function para crear instancia de MNEME"""
    if production:
        config = MnemeConfig.production()
    elif development:
        config = MnemeConfig.development()
    elif config is None:
        config = MnemeConfig()
    
    return ZSpace(config=config)


def get_default_device() -> torch.device:
    """Obtener dispositivo por defecto"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_system_info() -> Dict[str, Any]:
    """Obtener información del sistema"""
    info = {
        "version": __version__,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cpu_count": mp.cpu_count(),
        "default_device": str(get_default_device())
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
    
    try:
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = mem.total / GB
        info["memory_available_gb"] = mem.available / GB
    except:
        pass
    
    return info


# ============================================================================
# ALIASES PARA COMPATIBILIDAD
# ============================================================================

Mneme = ZSpace
MnemeCore = ZSpace


# ============================================================================
# EXPORTACIONES
# ============================================================================

__all__ = [
    # Versión
    '__version__',
    
    # Clases principales
    'ZSpace',
    'Mneme',
    'MnemeCore',
    'MnemeFacade',
    'MnemeConfig',
    'MnemeContext',
    
    # Descriptores
    'ZDescriptor',
    'ZAddr',
    
    # Componentes
    'LazyTensor',
    'AdaptiveCache',
    'GranularLockManager',
    'CircuitBreaker',
    'ProcessingPipeline',
    
    # Sistemas
    'MetricsRegistry',
    'EventBus',
    'PluginRegistry',
    
    # Enums
    'LockType',
    'DecompType',
    'CompressionLevel',
    'SerializationFormat',
    'SecurityLevel',
    'TensorEncryptionMode',
    'KeyRotationPolicy',
    'StorageBackend',
    'CachePolicy',
    'CompressionStrategy',
    'CircuitState',
    'HealthStatus',
    'PipelineStage',
    'EventType',
    
    # Errores
    'MnemeError',
    'SecurityError',
    'ValidationError',
    'StorageError',
    'CompressionError',
    'CircuitBreakerError',
    'PipelineError',
    'ResourceError',
    
    # Dataclasses
    'MnemeEvent',
    'MetricPoint',
    'LatencyHistogram',
    'PipelineContext',
    
    # Protocolos
    'Plugin',
    'TensorTransformer',
    'MetricsCollector',
    'EventHandler',
    
    # Decoradores
    'circuit_breaker_decorator',
    
    # Funciones de utilidad
    'create_mneme',
    'get_default_device',
    'get_system_info',
]