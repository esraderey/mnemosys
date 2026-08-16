"""
MNEME Core V3.0: Motor de Memoria Neural Mórfica - Enterprise Architecture
Sistema avanzado de memoria computacional con circuit breaker nativo,
métricas y API unificada.

Esta versión actúa como "cabeza" del sistema MNEME, proporcionando:
- Circuit breaker para resiliencia
- Métricas OpenTelemetry-compatible
- Integración nativa con MNEMEOptimizer
- Memory-aware operations

Versión: 3.0.0
Autor: MNEME Development Team
Licencia: BSL 1.1
"""

from __future__ import annotations

# === IMPORTS ESTÁNDAR ===
import asyncio
import base64
import functools
import gc
import hashlib
import hmac
import json
import logging
import multiprocessing as mp
import os
import struct
import sys
import time
import traceback
import uuid
import warnings
from collections import Counter as _Counter
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Generator
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from threading import Condition, Lock
from typing import (
    Any,
    Final,
    TypeVar,
)

# === IMPORTS DE TERCEROS ===
import lz4.frame
import msgpack
import numpy as np
import psutil
import safetensors
import tensorly as tl
import torch
from tensorly.decomposition import parafac, tensor_train, tucker

# Importar módulos de seguridad y almacenamiento
from .mneme_security_core import IntegrityError, SecurityLevel, SecurityManager, create_secure_config
from .mneme_storage_core import StorageAuthenticationError, StorageConfig, StorageFormatError, create_secure_storage

# === CONFIGURACIÓN DE LOGGING ===
logger = logging.getLogger(__name__)

# === CONSTANTES GLOBALES ===
__version__: Final[str] = "3.0.0"
__author__: Final[str] = "MNEME Development Team"

# Tamaños de memoria
KB: Final[int] = 1024
MB: Final[int] = KB * 1024
GB: Final[int] = MB * 1024

# Formato del payload de cuantización agrupada.
# v1 codificaba el offset como zero-point entero recortado a [0, qmax], que no puede
# representar un g_min positivo: todo grupo que no cruzara el cero perdía su offset al
# reconstruir. v2 guarda g_min en float32. Los payloads v1 no se leen: sus valores ya
# son incorrectos y devolverlos en silencio sería propagar la corrupción.
QUANT_FORMAT_VERSION: Final[int] = 2


def _clave_integridad_descriptores() -> bytes | None:
    """Clave de integridad de los descriptores, o None si no hay ninguna estable.

    Debe ser ESTABLE entre procesos: un descriptor persistido se verifica al
    releerlo, así que una clave efímera haría que todo lo guardado pareciera
    corrupto tras reiniciar. Por eso solo se aceptan claves del entorno, y la
    ausencia se resuelve degradando a hash sin clave, no inventando una.
    """
    for variable in ("MNEME_SIGNING_KEY", "MNEME_SECRET_KEY"):
        valor = os.environ.get(variable)
        if valor:
            return valor.encode("utf-8")
    return None

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
    warnings.warn(f"Could not set TensorLy backend to PyTorch: {e}", stacklevel=2)

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


# SecurityLevel importado de mneme_security_core (fuente única de verdad)
# TensorEncryptionMode y KeyRotationPolicy: eliminados (sin implementación real)
# SerializationFormat, StorageBackend, CompressionStrategy: eliminados (siempre safetensors/SQLite/LZ4)
# ContextSimilarityMethod, ContextClusteringMethod: eliminados (features no implementadas)


class CachePolicy(Enum):
    """Políticas de cache"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    LIFO = "lifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    ARC = "arc"  # Adaptive Replacement Cache


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


# ============================================================================
# SISTEMA DE ERRORES MEJORADO
# ============================================================================

class MnemeError(Exception):
    """Error base de MNEME con información contextual y tracing"""

    def __init__(
        self,
        message: str,
        error_code: str = None,
        context: dict[str, Any] = None,
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

    def to_dict(self) -> dict[str, Any]:
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


class PerformanceError(MnemeError):
    """Error de rendimiento (timeouts, degradación, etc.)"""

    def __init__(self, message: str, metric: str = None,
                 value: float = None, threshold: float = None, **kwargs):
        context = kwargs.pop('context', {})
        if metric:
            context['metric'] = metric
        if value is not None:
            context['value'] = value
        if threshold is not None:
            context['threshold'] = threshold
        super().__init__(message, "PERFORMANCE_ERROR", context, **kwargs)


class ConcurrencyError(MnemeError):
    """Error de concurrencia (deadlocks, timeouts de lock, etc.)"""
    def __init__(self, message: str, lock_type: str = None, resource: str = None, **kwargs):
        context = kwargs.pop('context', {})
        if lock_type:
            context['lock_type'] = lock_type
        if resource:
            context['resource'] = resource
        super().__init__(message, "CONCURRENCY_ERROR", context, **kwargs)


def _coerce_decomp_type(value: Any) -> DecompType | None:
    """Coaccionar un decomp_type de kwargs a DecompType (None si se omitió).

    decomp_type viaja por **kwargs, sin anotación que lo proteja: un string
    ("svd") es truthy, no iguala a ningún miembro del enum y se colaba hasta
    el camino forzado del routing, donde decompose() no lo reconocía y
    decomp_type.value reventaba con AttributeError — otra vez dentro del log
    del manejador de fallback, enmascarando la causa. Coaccionar o rechazar
    en la frontera, antes de cualquier routing.
    """
    if value is None or isinstance(value, DecompType):
        return value
    try:
        return DecompType(value)
    except ValueError as exc:
        raise ValidationError(
            f"decomp_type inválido: {value!r}; usa un miembro de DecompType "
            f"o uno de sus valores ({', '.join(m.value for m in DecompType)})",
            field_name="decomp_type",
            expected_type="DecompType",
            actual_value=value,
        ) from exc


def _validar_sparsity_estructurada(kwargs: dict[str, Any]) -> None:
    """Rechazar enable_structured_sparsity sin quantization_type.

    El pre-pass 2:4 está definido como sparsificar-luego-cuantizar: la máscara
    viaja dentro del payload cuantizado y se reaplica al decuantizar. Las rutas
    sin cuantización no tienen dónde conservarla (SVD/TT ni siquiera mantienen
    los ceros), y aceptar el flag sin efecto es exactamente el defecto que
    motivó este cambio.
    """
    if kwargs.get("enable_structured_sparsity") and kwargs.get("quantization_type") is None:
        raise ValidationError(
            "enable_structured_sparsity requiere quantization_type: el pre-pass "
            "2:4 sparsifica y luego cuantiza, y la máscara viaja en el payload "
            "cuantizado; sin cuantización el patrón no se conservaría",
            field_name="enable_structured_sparsity",
        )


def _dequantize_group_payload(info: dict[str, Any]) -> torch.Tensor:
    """Reconstruir un tensor desde un payload de cuantización agrupada.

    Único punto de decuantización: antes existían dos copias de esta fórmula y
    ambas compartían el mismo defecto.
    """
    formato = info.get("quant_format")
    if formato != QUANT_FORMAT_VERSION:
        raise CompressionError(
            f"payload de cuantización en formato {formato!r}, se esperaba "
            f"{QUANT_FORMAT_VERSION}. Los payloads anteriores se escribieron con un "
            f"codificador que perdía el offset de los grupos que no cruzan el cero, "
            f"así que sus valores son incorrectos: hay que regenerarlos desde el "
            f"tensor original.",
            compression_type="quantized",
        )

    q = np.frombuffer(info["quantized"], dtype=np.uint8).reshape(
        info["n_groups"], info["group_size"]
    )
    sc = np.frombuffer(info["scale"], dtype=np.float32)
    g_min = np.frombuffer(info["g_min"], dtype=np.float32)
    dequant = q.astype(np.float32) * sc[:, None] + g_min[:, None]
    flat = torch.from_numpy(dequant.reshape(-1)[: info["original_numel"]].copy())
    tensor = flat.reshape(info["shape"])

    # Máscara del pre-pass 2:4: los ceros podados cuantizan al escalón más
    # cercano a cero de su grupo, no a cero exacto, así que sin reaplicarla el
    # patrón 2:4 prometido no sobrevive al roundtrip.
    bits_mascara = info.get("sparsity_mask")
    if bits_mascara is not None:
        mascara = np.unpackbits(
            np.frombuffer(bits_mascara, dtype=np.uint8),
            count=info["original_numel"],
        ).astype(bool).reshape(info["shape"])
        tensor = tensor * torch.from_numpy(mascara)

    # El payload ya guardaba el dtype original y nadie lo leía: un tensor float16 o
    # float64 volvía como float32.
    nombre_dtype = str(info.get("dtype", "")).removeprefix("torch.")
    dtype_original = getattr(torch, nombre_dtype, None)
    if isinstance(dtype_original, torch.dtype) and dtype_original.is_floating_point:
        tensor = tensor.to(dtype_original)
    return tensor


# ============================================================================
# DATACLASSES DE EVENTOS Y MÉTRICAS
# ============================================================================


@dataclass
class LatencyHistogram:
    """Histograma de latencias con percentiles"""
    name: str
    values: list[float] = field(default_factory=list)
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

    def get_stats(self) -> dict[str, float]:
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
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = Lock()

        # Métricas
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._state_changes: list[tuple[float, CircuitState]] = []

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

    def get_stats(self) -> dict[str, Any]:
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
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, LatencyHistogram] = {}
        self._lock = Lock()
        self._start_time = time.time()
        self._metric_history: deque = deque(maxlen=10000)

    def counter(self, name: str, value: int = 1, tags: dict[str, str] = None) -> None:
        """Incrementar un contador"""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value
            self._record_history(name, self._counters[key], tags, "counter")

    def gauge(self, name: str, value: float, tags: dict[str, str] = None) -> None:
        """Establecer un gauge"""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value
            self._record_history(name, value, tags, "gauge")

    def histogram(self, name: str, value: float, tags: dict[str, str] = None) -> None:
        """Registrar en histograma"""
        key = self._make_key(name, tags)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = LatencyHistogram(name=key)
            self._histograms[key].record(value)
            self._record_history(name, value, tags, "histogram")

    def _make_key(self, name: str, tags: dict[str, str] = None) -> str:
        """Crear clave única para métrica"""
        base = f"{self.namespace}.{name}"
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{base}{{{tag_str}}}"
        return base

    def _record_history(self, name: str, value: float,
                       tags: dict[str, str], metric_type: str) -> None:
        """Registrar en historial"""
        self._metric_history.append({
            "name": f"{self.namespace}.{name}",
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {},
            "unit": metric_type,
        })

    def get_all_metrics(self) -> dict[str, Any]:
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

    # === CONFIGURACIÓN DE SEGURIDAD ===
    # repr=False: la clave no debe salir por logs que impriman la config. El
    # volcado sancionado es to_dict(), que la redacta. dataclasses.asdict() y
    # pickle SÍ la transportan a propósito: la config es la portadora legítima
    # de la clave (p. ej. hacia workers de otro proceso).
    secret_key: bytes | None = field(default=None, repr=False)
    enable_encryption: bool = True
    enable_merkle: bool = True
    audit_log_file: str | None = None
    key_rotation_policy: str = "adaptive"
    encryption_mode: str = "aes_gcm"

    # === CONFIGURACIÓN DE ALMACENAMIENTO ===
    cache_policy: CachePolicy = CachePolicy.ADAPTIVE
    storage_path: str = "./mneme_storage"
    enable_compression: bool = True

    # === CONFIGURACIÓN DE GPU ===
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    gpu_memory_growth: bool = True
    mixed_precision: bool = False
    preferred_device: str | None = None

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

    def _apply_defaults(self) -> None:
        """Aplicar valores por defecto inteligentes"""
        # Auto-detectar workers óptimos
        if self.max_workers == 4:
            cpu_count = mp.cpu_count() or 4
            self.max_workers = min(32, cpu_count)

        # Clave de cifrado en reposo.
        #
        # NO se genera una clave aleatoria por proceso: el cifrado ahora es real, así
        # que una clave efímera haría irrecuperable todo lo persistido en cuanto el
        # proceso se reinicia, y de forma silenciosa. La clave debe venir del usuario
        # (parámetro) o del entorno; sin clave, se desactiva el cifrado y se avisa,
        # que es honesto y recuperable, en vez de cifrar contra una clave que se
        # pierde al salir.
        if self.secret_key is None and self.enable_encryption:
            del_entorno = os.environ.get("MNEME_SECRET_KEY")
            if del_entorno:
                clave = del_entorno.encode("utf-8")
                # La validación de longitud de _validate_config() ya pasó cuando
                # llegamos aquí, así que la clave del entorno hay que comprobarla
                # ahora: sin esto, un MNEME_SECRET_KEY de 4 bytes activaba cifrado
                # real y se derivaba sin una sola queja.
                if len(clave) < 32:
                    raise ValidationError(
                        f"MNEME_SECRET_KEY debe tener al menos 32 bytes, tiene "
                        f"{len(clave)}",
                        field_name="secret_key",
                        expected_type="bytes with len >= 32",
                    )
                self.secret_key = clave
            else:
                self.enable_encryption = False
                warnings.warn(
                    "enable_encryption estaba activo pero no hay secret_key: ni por "
                    "parámetro ni en la variable de entorno MNEME_SECRET_KEY. El "
                    "cifrado en reposo queda DESACTIVADO para esta instancia. Cifrar "
                    "con una clave generada al vuelo haría irrecuperable todo lo "
                    "almacenado en cuanto termine el proceso.",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def to_dict(self) -> dict[str, Any]:
        """Convertir a diccionario, sin material de clave.

        Un volcado de configuración acaba en logs, checkpoints o en el
        contexto de optimización (`ZSpace.get_optimization_context()`), y
        ninguno de esos canales debe transportar la clave: `secret_key` se
        sustituye por el marcador "<redactada>" (None si no hay clave), y
        ningún otro campo bytes sale del volcado. `from_dict()` tampoco la
        acepta de vuelta: el par to_dict/from_dict restaura todo MENOS la
        clave, que se aprovisiona por el constructor o por MNEME_SECRET_KEY.
        """
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if field_name == 'secret_key':
                result[field_name] = '<redactada>' if value is not None else None
            elif isinstance(value, Enum):
                result[field_name] = value.value
            elif isinstance(value, bytes):
                # Esta rama en base64 fue la que fugó secret_key; un campo
                # bytes futuro (un salt, otra clave) la reactivaría igual.
                result[field_name] = '<redactada>'
            else:
                result[field_name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MnemeConfig:
        """Crear desde diccionario, ignorando cualquier `secret_key`.

        Una clave jamás se reconstruye desde un volcado: ni el marcador de
        `to_dict()`, ni el base64 de volcados antiguos, ni bytes crudos. La
        instancia se crea sin clave y `__post_init__` aplica la regla de
        siempre (MNEME_SECRET_KEY del entorno o cifrado desactivado con
        aviso). Para restaurar una configuración cifrada, pásese la clave
        por el parámetro `secret_key` del constructor.
        """
        processed = {}
        enum_mappings = {
            'compression_level': CompressionLevel,
            'security_level': SecurityLevel,
            'cache_policy': CachePolicy,
        }

        for key, value in data.items():
            if key == 'secret_key':
                continue
            if key in enum_mappings and isinstance(value, (str, int)):
                processed[key] = enum_mappings[key](value)
            else:
                processed[key] = value

        return cls(**processed)

    @classmethod
    def production(cls) -> MnemeConfig:
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
    def development(cls) -> MnemeConfig:
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
    - Circuit Breakers
    """
    config: MnemeConfig
    metrics: MetricsRegistry = field(default_factory=lambda: MetricsRegistry())
    device: torch.device = field(default=None)

    def __post_init__(self):
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

    def record_metric(self, name: str, value: float, tags: dict[str, str] = None) -> None:
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
        self._locks: dict[str, Lock] = {}
        self._rw_locks: dict[str, RWLock] = {}
        self._lock_usage: dict[str, dict[str, Any]] = {}
        self._lock_manager = Lock()
        self._read_locks: dict[str, int] = defaultdict(int)
        self._write_locks: dict[str, int] = defaultdict(int)

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

        acquired = lock.acquire(blocking=blocking, timeout=timeout if blocking else 0)
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

    def get_lock_stats(self) -> dict[str, Any]:
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
        metadata: dict[str, Any],
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

        self._decompressed_tensor: torch.Tensor | None = None
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

            except (MnemeError, IntegrityError):
                # CompressionError de cuantización, SecurityError, IntegrityError
                # del marco firmado…: ya dicen exactamente qué pasó. Envolverlos
                # en un CompressionError "lz4" disfrazaba la manipulación
                # detectada de corrupción accidental, indistinguibles para quien
                # opera el sistema.
                raise
            except Exception as e:
                logger.error(f"Failed to decompress tensor: {e}")
                raise CompressionError(
                    f"Tensor decompression failed: {e}",
                    compression_type="lz4"
                ) from e

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

    def get_shape(self) -> tuple[int, ...]:
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

    def get_memory_usage(self) -> dict[str, Any]:
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
        except Exception:
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

        self._lru_order: OrderedDict[str, None] = OrderedDict()
        self._lfu_counts: dict[str, int] = {}
        self._ttl_expiry: dict[str, float] = {}

        self._cache: dict[str, Any] = {}
        self._lock = Lock()
        self._access_times: dict[str, float] = {}
        self._access_frequencies: dict[str, int] = defaultdict(int)
        self._creation_times: dict[str, float] = {}

        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0

    def get(self, key: str) -> Any | None:
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

            # Actualizar LRU — O(1) con OrderedDict
            self._lru_order.move_to_end(key)

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
            self._lru_order[key] = None

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

        self._lru_order.pop(key, None)

    def _evict_one(self) -> None:
        """Evictar un elemento según la estrategia"""
        if not self._cache:
            return

        key_to_evict = None

        if self.strategy == "lru" and self._lru_order:
            key_to_evict = next(iter(self._lru_order))
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

    def _select_adaptive_eviction(self) -> str | None:
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

    def get_stats(self) -> dict[str, Any]:
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
# DESCOMPOSICIÓN TENSORIAL
# ============================================================================

class TensorDecomposer:
    """
    Descomposición tensorial real con auto-selección, descomposición y
    reconstrucción.  Soporta TT, CP, Tucker, SVD y sparse.
    """

    @staticmethod
    def auto_select(
        tensor: torch.Tensor,
        target_ratio: float = 0.1,
    ) -> tuple[DecompType, dict[str, Any]]:
        """Seleccionar automáticamente la mejor descomposición."""
        shape = tensor.shape
        numel = tensor.numel()

        # Sparse check
        sparsity = (tensor == 0).float().mean().item()
        if sparsity > 0.9:
            return DecompType.SPARSE, {"sparsity": sparsity}

        # 2-D → SVD
        if len(shape) == 2:
            rank = min(int(min(shape) * target_ratio), min(shape) // 2)
            rank = max(rank, 1)
            return DecompType.SVD, {"rank": rank}

        # ≥3-D → Tensor-Train
        if len(shape) >= 3:
            tt_ranks: list[int] = []
            cum_left = 1
            cum_right = numel
            for i in range(len(shape) - 1):
                cum_left *= shape[i]
                cum_right //= shape[i]
                r = min(cum_left, cum_right)
                r = min(r, max(1, int(r * target_ratio)))
                tt_ranks.append(max(r, 1))
            # tensorly.tensor_train exige un vector de rangos de longitud
            # ndim+1 con extremos de frontera fijos en 1 (rank[0] ==
            # rank[ndim] == 1). El bucle de arriba solo calcula los ndim-1
            # rangos internos (uno por corte entre ejes consecutivos), así
            # que hay que envolverlo con esos bordes antes de devolverlo.
            return DecompType.TT, {"ranks": (1,) + tuple(tt_ranks) + (1,)}

        # Fallback → CP para 1-D (raro), o RAW
        if len(shape) == 1:
            return DecompType.RAW, {}

        rank = max(1, int(min(shape) * target_ratio))
        return DecompType.CP, {"rank": rank}

    @staticmethod
    def decompose(
        tensor: torch.Tensor,
        decomp_type: DecompType,
        **params,
    ) -> dict[str, Any]:
        """Ejecutar descomposición tensorial real."""

        if decomp_type == DecompType.TT:
            ranks = params.get("ranks")
            tt_tensor = tensor_train(tensor, rank=ranks)
            # tensor_train() devuelve un TTTensor: un Mapping de tensorly
            # (no list/tuple), que _serialize_components no sabe empaquetar
            # a nivel de campo de nivel superior (solo reconoce un
            # torch.Tensor suelto o una list/tuple de tensores). Se
            # desempaqueta a la lista plana de cores — tl.tt_to_tensor()
            # acepta esa misma lista igual que el TTTensor original.
            return {"factors": list(tt_tensor.factors), "type": "tt"}

        elif decomp_type == DecompType.CP:
            rank = params.get("rank", 10)
            cp_tensor = parafac(tensor, rank=rank, init="random",
                                n_iter_max=100, tol=1e-6)
            # parafac() devuelve un CPTensor (weights, factors), también un
            # Mapping de tensorly que msgpack no puede serializar de
            # nivel superior (era el TypeError "can not serialize
            # 'CPTensor' object"). Se desempaqueta en sus dos piezas
            # planas — un tensor de pesos y una lista de factores — que
            # _serialize_components ya sabe empaquetar cada una por su
            # propia rama existente.
            weights, factors = cp_tensor
            return {"weights": weights, "factors": factors, "type": "cp"}

        elif decomp_type == DecompType.TUCKER:
            ranks = params.get("ranks", [min(s, 10) for s in tensor.shape])
            core, factors = tucker(tensor, rank=ranks)
            return {"core": core, "factors": factors, "type": "tucker"}

        elif decomp_type == DecompType.SVD:
            rank = params.get("rank", min(tensor.shape) // 2)
            rank = max(rank, 1)
            U, S, V = torch.svd_lowrank(tensor.float(), q=rank)
            return {"U": U, "S": S, "V": V, "type": "svd"}

        elif decomp_type == DecompType.SPARSE:
            indices = torch.nonzero(tensor)
            values = tensor[indices.split(1, dim=1)].squeeze()
            return {
                "indices": indices,
                "values": values,
                "shape": tuple(tensor.shape),
                "type": "sparse",
            }

        # Sin rama implementada: RAW, QUANTIZED, ADAPTIVE o un miembro futuro
        # del enum. El fallback histórico devolvía {"type": "raw"}, un payload
        # sin los datos del tensor que reconstruct() no sabe leer: pérdida
        # silenciosa en el momento de escribir, descubierta recién al leer.
        # Fallar aquí, ruidoso y temprano; esos tipos se resuelven en el
        # routing de ZSpace, no en este nivel.
        raise ValidationError(
            f"decompose() no implementa {decomp_type!r}; tipos con rama: "
            f"TT, CP, TUCKER, SVD, SPARSE. RAW, QUANTIZED y ADAPTIVE se "
            f"resuelven en el routing de ZSpace (register()).",
            field_name="decomp_type",
            actual_value=decomp_type,
        )

    @staticmethod
    def reconstruct(components: dict[str, Any]) -> torch.Tensor:
        """Reconstruir tensor desde componentes descompuestos."""
        comp_type = components["type"]

        if comp_type == "tt":
            return tl.tt_to_tensor(components["factors"])

        elif comp_type == "cp":
            return tl.cp_to_tensor((components["weights"], components["factors"]))

        elif comp_type == "tucker":
            return tl.tucker_to_tensor(
                (components["core"], components["factors"])
            )

        elif comp_type == "svd":
            U, S, V = components["U"], components["S"], components["V"]
            return U @ torch.diag(S) @ V.T

        elif comp_type == "sparse":
            shape = components["shape"]
            indices = components["indices"]
            values = components["values"]
            tensor = torch.zeros(shape)
            for idx, val in zip(indices, values, strict=False):
                tensor[tuple(idx)] = val
            return tensor

        raise ValueError(f"Unknown component type: {comp_type}")


# ============================================================================
# PREFETCHER MARKOV DE SEGUNDO ORDEN
# ============================================================================

class MarkovPrefetcher:
    """
    Prefetcher basado en cadena de Markov de 2.º orden.
    Registra patrones de acceso (addr₁→addr₂→addr₃) y predice los
    siguientes descriptores más probables.
    """

    def __init__(self, history_size: int = 1000):
        self._history: deque = deque(maxlen=history_size)
        self._transitions: dict[tuple[bytes, bytes], list[bytes]] = {}
        self._lock = Lock()

    def record_access(self, addr: bytes) -> None:
        """Registrar un acceso para actualizar las transiciones."""
        with self._lock:
            self._history.append(addr)
            if len(self._history) >= 3:
                prev = self._history[-3]
                curr = self._history[-2]
                nxt = self._history[-1]
                key = (prev, curr)
                if key not in self._transitions:
                    self._transitions[key] = []
                self._transitions[key].append(nxt)

    def predict_next(
        self,
        curr: bytes,
        prev: bytes | None = None,
        top_k: int = 3,
    ) -> list[bytes]:
        """Predecir las siguientes *top_k* direcciones más probables."""
        with self._lock:
            if prev and (prev, curr) in self._transitions:
                next_addrs = self._transitions[(prev, curr)]
                if next_addrs:
                    return [
                        addr for addr, _ in _Counter(next_addrs).most_common(top_k)
                    ]
            return []

    def get_stats(self) -> dict[str, Any]:
        """Estadísticas del prefetcher."""
        with self._lock:
            return {
                "history_len": len(self._history),
                "transition_entries": len(self._transitions),
            }


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
    shape: tuple[int, ...]
    ranks: tuple[int, ...] | None = None
    core_data: bytes = b""
    version: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    merkle_root: bytes | None = None
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    delta_chain: bytes | None = None
    lazy_tensor: LazyTensor | None = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    security_hash: bytes | None = None
    # Clave HMAC del hash de integridad, inyectada por ZSpace desde
    # MnemeConfig.secret_key. Es InitVar y se guarda en `_clave_integridad`,
    # que NO es un field del dataclass: queda fuera de to_dict()/from_dict(),
    # de repr() y también de dataclasses.asdict(); ZSpace la reinyecta al
    # recargar desde storage. Con None se cae a las variables de entorno
    # (_clave_integridad_descriptores). Ojo: dataclasses.replace() no la
    # arrastra — quien clone un descriptor debe volver a pasarla; pickle y
    # copy.deepcopy() la despojan vía __getstate__. Límite conocido:
    # vars()/__dict__ sí ve _clave_integridad (introspección de instancia,
    # inevitable para cualquier atributo), y el nombre 'clave_integridad'
    # pervive como atributo de clase con valor None — para leer la clave
    # efectiva úsese _clave_integridad, nunca el nombre del InitVar.
    clave_integridad: InitVar[bytes | None] = None

    def __post_init__(self, clave_integridad: bytes | None = None):
        self._clave_integridad: bytes | None = clave_integridad
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

    def __getstate__(self):
        """El estado picklable/copiable jamás transporta la clave de integridad.

        pickle está vetado por convención del proyecto, pero si un tercero
        picklea (o deepcopy-a) un descriptor, la clave no debe viajar con él:
        el clon queda con `_clave_integridad = None` y verify_integrity()
        degrada al fallback de entorno.
        """
        estado = self.__dict__.copy()
        estado['_clave_integridad'] = None
        return estado

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
                if not hmac.compare_digest(computed_hash, self.security_hash):
                    return False

            return True

        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False

    def _compute_merkle_root(self) -> bytes:
        return hashlib.sha256(self.core_data).digest()

    def _compute_security_hash(self) -> bytes:
        """Hash de integridad sobre metadatos Y payload.

        Antes era un SHA-256 sin clave sobre SOLO los metadatos: no cubría
        `core_data`, así que un tercero podía sustituir el payload entero y
        recalcular el mismo hash. Ahora el payload siempre está cubierto.

        Con una clave estable configurada (`MnemeConfig.secret_key` inyectada en
        `clave_integridad`, o `MNEME_SIGNING_KEY`/`MNEME_SECRET_KEY` del entorno)
        es un HMAC y protege también contra un falsificador deliberado. Sin clave
        degrada a SHA-256: sigue detectando corrupción y sustitución del payload,
        pero no a quien recalcule el hash a propósito.

        Cobertura exacta, para no prometer de más: cubre `kind`, `decomp_type`,
        `shape`, `version` y `core_data`. NO cubre `meta`, `ranks`, `delta_chain`,
        `compression_level` ni `merkle_root`; alterar solo esos campos no se
        detecta aquí.
        """
        # Framing con longitudes explícitas: concatenar con ':' permitiría que dos
        # tuplas de metadatos distintas produjeran la misma cabecera.
        campos = (
            str(self.kind).encode(),
            str(self.decomp_type.value).encode(),
            str(list(self.shape)).encode(),
            str(self.version).encode(),
            self.core_data,
        )
        cubierto = b"".join(len(c).to_bytes(8, "big") + c for c in campos)

        clave = (self._clave_integridad if self._clave_integridad is not None
                 else _clave_integridad_descriptores())
        if clave is not None:
            return hmac.new(clave, cubierto, hashlib.sha256).digest()
        return hashlib.sha256(cubierto).digest()

    def update_access(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1

    def get_size_bytes(self) -> int:
        size = len(self.core_data)
        if self.delta_chain:
            size += len(self.delta_chain)
        return size

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> ZDescriptor:
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
        if 'ranks' in processed and isinstance(processed['ranks'], list):
            # ranks nace tupla; dejarlo como lista JSON hace que ZAddr.compute
            # (usa str(ranks)) dé otra dirección para el mismo descriptor
            # tras rehidratar: "[30]" vs "(30,)".
            processed['ranks'] = tuple(processed['ranks'])

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
    def compute(cls, desc: ZDescriptor) -> ZAddr:
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
    def from_hex(cls, hex_str: str) -> ZAddr:
        try:
            addr_bytes = bytes.fromhex(hex_str)
            return cls(addr_bytes)
        except ValueError as e:
            raise ValidationError(f"Invalid hex string: {e}", field_name="hex_str") from e

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
    - Circuit breaker integrado
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

        # Métricas
        self.metrics = self.context.metrics

        # Componentes core
        self.lock_manager = GranularLockManager()
        self.adaptive_cache = AdaptiveCache(
            max_size_bytes=self.config.cache_size_mb * MB,
            strategy="adaptive" if self.config.cache_policy == CachePolicy.ADAPTIVE else self.config.cache_policy.value,
            ttl_seconds=3600.0
        )

        # Circuit breakers
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
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

        # Storage backend
        # La clave se propaga al backend: antes MnemeConfig generaba una secret_key
        # que nunca llegaba al almacenamiento, de modo que el cifrado en reposo no
        # tenía con qué cifrar.
        self.storage_backend = create_secure_storage(
            StorageConfig(
                storage_path=self.config.storage_path,
                enable_encryption=self.config.enable_encryption,
                secret_key=self.config.secret_key,
            )
        )

        # Security manager. La clave de MnemeConfig alimenta también la firma
        # HMAC de los marcos serializados: sin esto, un secret_key pasado por
        # código cifraba el almacenamiento pero los artefactos salían sin firmar,
        # porque el serializador solo miraba el entorno.
        self.security_manager = SecurityManager(
            create_secure_config(signing_key=self.config.secret_key)
        )

        # Tablas de descriptores
        self.name_to_desc: dict[str, ZDescriptor] = {}
        self.addr_to_desc: dict[bytes, ZDescriptor] = {}
        self.version_graph: dict[bytes, bytes] = {}

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
        self._optimizer_hooks: dict[str, list[Callable]] = defaultdict(list)

        # Entrega FIFO por nombre de los hooks post-store: se encola bajo el
        # write_lock y se drena fuera de él con un único drenador activo por
        # nombre, de modo que el orden de entrega es el orden real de registro
        # aunque dos hilos escriban el mismo nombre a la vez.
        self._post_store_pendientes: dict[str, deque] = defaultdict(deque)
        self._post_store_drenando: set = set()
        self._post_store_lock = Lock()

        # Prefetcher Markov de 2.º orden
        self.prefetcher = MarkovPrefetcher()

        # Deduplicación de síntesis concurrentes
        self._pending_synthesis: dict[bytes, Future] = {}
        self._synthesis_lock = Lock()
        self._synthesis_executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers
        )

        if auto_init:
            self._initialize()

    def _initialize(self) -> None:
        """Inicializar componentes"""
        logger.info(f"Initializing MNEME ZSpace V{__version__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Config: cache={self.config.cache_size_mb}MB, "
                   f"compression={self.config.compression_level.name}")

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

        Nota sobre hooks: los post-store se entregan en orden de registro por
        nombre pero fuera del write_lock; bajo contención puede ejecutarlos el
        hilo drenador de otro store(), y este método puede devolver antes de
        que el hook de SU escritura haya corrido.
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

                # Encolado bajo el lock: el orden de la cola es el orden real de
                # registro del nombre.
                self._encolar_post_store(name, desc)

            # Hooks post-store FUERA del write_lock: el lock subyacente no es reentrante
            # y se retiene durante todo el cuerpo del `with`, así que un hook que vuelva
            # a entrar en el espacio (p. ej. load(name) para comprobar lo escrito) se
            # autobloqueaba de forma permanente y sin timeout. La entrega va por la
            # cola FIFO del nombre: sin ella, otro hilo podía escribir el mismo nombre
            # entre soltar el lock y ejecutar el hook, y el observador recibía como
            # "último" un descriptor ya obsoleto.
            self._drenar_post_store(name)

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

    def register(self, name: str, tensor: torch.Tensor, **kwargs) -> ZDescriptor:
        """
        Registrar tensor con compresión inteligente (routing SVD/INT8/RAW).

        Alias enriquecido de ``store()`` que aplica ``_create_smart_descriptor``
        en lugar de ``_create_secure_descriptor`` cuando el tensor es grande
        y se beneficia de descomposición tensorial o cuantización.

        Args:
            name: Nombre único del tensor
            tensor: Tensor a registrar
            **kwargs: target_ratio, decomp_type, quantization_type,
                      group_size, gptq_metadata, enable_structured_sparsity, etc.
                      ``decomp_type=DecompType.ADAPTIVE`` equivale a omitirlo:
                      routing automático y tipo concreto en el descriptor.
                      ``enable_structured_sparsity=True`` aplica el pre-pass
                      2:4 antes de cuantizar (exige ``quantization_type``); la
                      máscara viaja en el payload y el patrón se restaura
                      exacto al cargar.

        Returns:
            ZDescriptor del tensor registrado

        Nota sobre hooks: misma semántica de entrega que store() — FIFO por
        nombre, fuera del write_lock, posiblemente en el hilo de otro store().
        """
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Name must be a non-empty string",
                                  field_name="name")
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError("Must provide a torch.Tensor",
                                  field_name="tensor",
                                  expected_type="torch.Tensor")
        # Validación de entrada, como name/tensor: ANTES de la puerta del
        # breaker. Dentro del try, cada typo contaría como fallo de storage y
        # 5 acumulados abrían el breaker, bloqueando también los register()
        # válidos durante el reset_timeout.
        if "decomp_type" in kwargs:
            kwargs["decomp_type"] = _coerce_decomp_type(kwargs["decomp_type"])
        # Misma frontera para el pre-pass 2:4: pedirlo sin cuantización es error
        # del caller, no un fallo de storage que deba contar para el breaker.
        _validar_sparsity_estructurada(kwargs)

        circuit = self._circuit_breakers.get('storage')
        if circuit and not circuit.allow_request():
            raise CircuitBreakerError("Storage circuit breaker is open",
                                      circuit_name="storage")

        start_time = time.time()

        try:
            with self.lock_manager.write_lock(name):
                tensor = self._run_hooks('pre_store', tensor, name=name)

                desc = self._create_smart_descriptor(tensor, **kwargs)

                old_addr = None
                if name in self.name_to_desc:
                    old_addr = ZAddr.compute(self.name_to_desc[name])

                addr = self._register_descriptor(name, desc, old_addr)

                self.adaptive_cache.put(f"desc_{name}", desc)

                self.storage_metrics["write_operations"] += 1
                self.storage_metrics["tensor_count"] = len(self.name_to_desc)
                self.storage_metrics["total_storage_bytes"] += desc.get_size_bytes()

                self._encolar_post_store(name, desc)

            # Hooks post-store FUERA del write_lock: el lock subyacente no es reentrante
            # y se retiene durante todo el cuerpo del `with`, así que un hook que vuelva
            # a entrar en el espacio (p. ej. load(name)) se autobloqueaba sin timeout.
            # La cola FIFO por nombre conserva el orden de registro bajo contención.
            self._drenar_post_store(name)

            if circuit:
                circuit.record_success()

            self.metrics.histogram(
                "tensor.register.duration_ms",
                (time.time() - start_time) * 1000,
                {"tensor_name": name}
            )

            logger.info(
                f"Registered '{name}'. Decomp: {desc.decomp_type.value}. "
                f"Addr: {addr.short_hex()}"
            )
            return desc

        except Exception as e:
            if circuit:
                circuit.record_failure(e)
            raise

    def update(self, name: str, delta_op: dict) -> ZDescriptor:
        """
        Actualizar tensor con operación delta reversible.

        Operaciones soportadas:
            {"type": "add", "value": <tensor_or_scalar>}
            {"type": "mul", "value": <tensor_or_scalar>}
            {"type": "sparse_update", "indices": <idx>, "values": <vals>}

        Args:
            name: Nombre del tensor existente
            delta_op: Diccionario describiendo la operación delta

        Returns:
            Nuevo ZDescriptor (versión incrementada)
        """
        if name not in self.name_to_desc:
            raise KeyError(f"Unknown tensor: {name}")

        with self.lock_manager.write_lock(name):
            old_desc = self.name_to_desc[name]

            # Cargar tensor directamente (sin pasar por load() para evitar deadlock)
            if hasattr(old_desc, 'lazy_tensor') and old_desc.lazy_tensor:
                tensor = old_desc.lazy_tensor.decompress()
            else:
                tensor = self._do_synthesize(old_desc)
            tensor = self._apply_delta(tensor, delta_op)

            # Crear nuevo descriptor con la versión incrementada
            new_desc = self._create_smart_descriptor(tensor)

            # Preservar delta chain comprimida para auditoría
            if old_desc.delta_chain:
                prev_deltas = json.loads(
                    lz4.frame.decompress(old_desc.delta_chain).decode('utf-8')
                )
            else:
                prev_deltas = []

            # Serializar delta_op de forma segura (convertir tensores)
            safe_op = {}
            for k, v in delta_op.items():
                if isinstance(v, torch.Tensor):
                    safe_op[k] = v.tolist()
                else:
                    safe_op[k] = v
            prev_deltas.append(safe_op)

            compressed_chain = lz4.frame.compress(
                json.dumps(prev_deltas).encode('utf-8')
            )

            # Crear descriptor con versión incrementada y delta chain
            new_desc = ZDescriptor(
                kind=new_desc.kind,
                decomp_type=new_desc.decomp_type,
                shape=new_desc.shape,
                core_data=new_desc.core_data,
                version=old_desc.version + 1,
                meta={**new_desc.meta, "delta_applied": safe_op["type"]},
                compression_level=new_desc.compression_level,
                lazy_tensor=new_desc.lazy_tensor,
                delta_chain=compressed_chain,
                clave_integridad=self.config.secret_key,
            )

            old_addr = ZAddr.compute(old_desc)
            self._register_descriptor(name, new_desc, old_addr)

            # Invalidar cache
            self.adaptive_cache.remove(f"desc_{name}")
            self.adaptive_cache.put(f"desc_{name}", new_desc)

            logger.info(
                f"Updated '{name}' v{old_desc.version}→v{new_desc.version}. "
                f"Delta: {delta_op.get('type')}"
            )
            return new_desc

    @staticmethod
    def _apply_delta(tensor: torch.Tensor, delta_op: dict) -> torch.Tensor:
        """Aplicar operación delta reversible a un tensor."""
        op_type = delta_op.get("type")
        if op_type == "add":
            val = delta_op["value"]
            if isinstance(val, list):
                val = torch.tensor(val, dtype=tensor.dtype, device=tensor.device)
            return tensor + val
        elif op_type == "mul":
            val = delta_op["value"]
            if isinstance(val, list):
                val = torch.tensor(val, dtype=tensor.dtype, device=tensor.device)
            return tensor * val
        elif op_type == "sparse_update":
            tensor = tensor.clone()
            indices = delta_op["indices"]
            values = delta_op["values"]
            if isinstance(values, list):
                values = torch.tensor(values, dtype=tensor.dtype,
                                      device=tensor.device)
            tensor[indices] = values
            return tensor
        else:
            raise ValueError(f"Unknown delta operation: {op_type}")

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
            elif name in self.name_to_desc:
                desc = self.name_to_desc[name]
                self.adaptive_cache.put(f"desc_{name}", desc)
                self.storage_metrics["cache_misses"] += 1
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

        # Verificar integridad ANTES de reconstruir. El camino real de lectura pasa
        # por lazy_tensor, así que comprobar solo en _synthesize_tensor dejaba
        # verify_integrity() como código muerto: un payload alterado se reconstruía
        # sin una sola excepción.
        if not desc.verify_integrity():
            raise SecurityError(
                f"integridad del descriptor '{name}' no verificada: los datos fueron "
                f"alterados o se escribieron con otra clave",
                threat_type="integrity",
            )

        # Decomprimir
        if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
            tensor = desc.lazy_tensor.decompress()
        else:
            tensor = self._synthesize_tensor(desc)

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

        return success

    def exists(self, name: str) -> bool:
        """Verificar si existe un tensor"""
        if name in self.name_to_desc:
            return True

        try:
            desc_data = self.storage_backend.retrieve(f"desc_{name}")
            return desc_data is not None
        except Exception:
            return False

    def list_tensors(self) -> dict[str, Any]:
        """Listar todos los tensores"""
        memory_tensors = set(self.name_to_desc.keys())

        storage_tensors = set()
        try:
            storage_keys = self.storage_backend.list_keys()
            for key in storage_keys:
                if key.startswith("desc_"):
                    storage_tensors.add(key[5:])
        except Exception:
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
        """Registrar hook para integración con MNEMEOptimizer.

        Contrato de entrega de 'post_store': FIFO por nombre (el orden real de
        registro), sin ningún lock del espacio tomado — el hook puede releer o
        re-almacenar sin bloquearse — y, bajo contención, posiblemente en el
        hilo drenador de otro store(). 'pre_store' corre bajo el write_lock del
        nombre: un hook pre_store que reentre en el espacio se bloqueará.
        """
        self._optimizer_hooks[hook_name].append(callback)

    def _encolar_post_store(self, name: str, desc: ZDescriptor) -> None:
        """Encolar la entrega del hook post-store. Llamar bajo el write_lock."""
        with self._post_store_lock:
            self._post_store_pendientes[name].append(desc)

    def _drenar_post_store(self, name: str) -> None:
        """Entregar los hooks post-store pendientes de `name` en orden de registro.

        Un único drenador activo por nombre: quien llega mientras otro hilo drena
        se va sin esperar — su evento ya está encolado y el drenador activo lo
        entregará al terminar el hook en curso. Los hooks corren sin ningún lock
        del espacio, así que pueden releer o re-almacenar sin bloquearse; si
        re-almacenan el mismo nombre, el evento nuevo se entrega a continuación,
        no de forma recursiva. Bajo contención, el hook de un store puede
        ejecutarlo el hilo drenador de otro store; lo que se garantiza es el
        orden por nombre, no el hilo.
        """
        while True:
            with self._post_store_lock:
                if name in self._post_store_drenando:
                    return
                cola = self._post_store_pendientes.get(name)
                if not cola:
                    if cola is not None:
                        del self._post_store_pendientes[name]
                    return
                desc = cola.popleft()
                self._post_store_drenando.add(name)
            try:
                self._run_hooks('post_store', desc, name=name)
            finally:
                with self._post_store_lock:
                    self._post_store_drenando.discard(name)

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

    def get_optimization_context(self) -> dict[str, Any]:
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
        # Serializar con el serializador del espacio: lleva la clave de firma de
        # MnemeConfig, no solo la del entorno.
        core_data = self.security_manager.serializer.serialize_tensor(
            tensor, kwargs.get('metadata', {})
        )

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
            lazy_tensor=lazy_tensor,
            clave_integridad=self.config.secret_key,
        )

        return desc

    def _create_smart_descriptor(
        self, tensor: torch.Tensor, **kwargs
    ) -> ZDescriptor:
        """
        Crear descriptor con routing inteligente de compresión.

        Routing:
            1. Si kwarg ``quantization_type`` presente → QUANTIZED
               (INT4/INT8/GPTQ), con pre-pass 2:4 si ``enable_structured_sparsity``
            2. 2-D con ≥10 000 elementos → SVD truncado
            3. ≥3-D → Tensor-Train
            4. ≥1 000 elementos → safetensors + LZ4 (RAW comprimido)
            5. < 1 000 elementos → RAW sin compresión adicional

        ``decomp_type`` fuerza un tipo de descomposición implementado (TT,
        CP, TUCKER, SVD, SPARSE); acepta el miembro del enum o su valor
        string (``"svd"``) y rechaza con ``ValidationError`` cualquier valor
        no coaccionable a ``DecompType``. ``DecompType.ADAPTIVE`` equivale a
        omitirlo: se aplica este mismo routing y el descriptor guarda el tipo
        concreto resuelto. ``RAW`` y ``QUANTIZED`` no son forzables por esta
        vía: se ignoran y decide el routing (para cuantizar, usar
        ``quantization_type``).

        Args:
            tensor: Tensor a almacenar
            **kwargs: target_ratio, decomp_type, quantization_type,
                      group_size, gptq_metadata, enable_structured_sparsity
        """
        target_ratio = kwargs.get("target_ratio", 0.1)
        forced_decomp = kwargs.get("decomp_type")
        quant_type = kwargs.get("quantization_type")
        numel = tensor.numel()

        # register() ya coacciona decomp_type antes de la puerta del circuit
        # breaker (un typo es error del caller, no fallo de storage); esta
        # llamada repite la misma defensa para entradas que no pasan por
        # register().
        forced_decomp = _coerce_decomp_type(forced_decomp)

        # ADAPTIVE no es un tipo forzable: es pedir explícitamente el routing
        # automático de abajo, igual que omitir decomp_type. decompose() solo
        # implementa tipos concretos (TT/CP/Tucker/SVD/Sparse) y rechaza el
        # resto con ValidationError; sin esta normalización, forzar ADAPTIVE
        # acabaría degradado al RAW del manejador de fallback en vez de en el
        # tipo concreto que el routing habría elegido.
        if forced_decomp == DecompType.ADAPTIVE:
            forced_decomp = None

        # Defensa repetida de register(), como la coacción de decomp_type de
        # arriba: el pre-pass 2:4 solo existe en la ruta cuantizada.
        _validar_sparsity_estructurada(kwargs)

        # --- Path 1: Cuantización explícita (INT4/INT8/GPTQ) ---
        if quant_type is not None:
            return self._create_quantized_descriptor(tensor, **kwargs)

        # --- Path 2: Descomposición forzada o auto-seleccionada ---
        if forced_decomp and forced_decomp not in (DecompType.RAW, DecompType.QUANTIZED):
            decomp_type = forced_decomp
            rank = max(1, int(min(tensor.shape) * target_ratio))
            if decomp_type == DecompType.TT:
                # decompose() para TT lee params["ranks"] (plural: vector de
                # longitud ndim+1 con extremos 1), no params["rank"]
                # (singular) — con solo "rank" ranks=None y
                # tensor_train(..., rank=None) revienta. Repetir el mismo
                # rank en cada corte interno es válido: para cualquier
                # tensor con todas las dims >=1, cum_left/cum_right en
                # cualquier corte son >= min(tensor.shape) >= rank (ya que
                # target_ratio <= 1), así que nunca excede el límite de
                # tensorly en ningún corte.
                ndim = len(tensor.shape)
                params = {"ranks": (1,) + tuple([rank] * (ndim - 1)) + (1,)}
            else:
                params = {"rank": rank}
        elif len(tensor.shape) == 2 and numel >= 10_000:
            decomp_type, params = TensorDecomposer.auto_select(tensor, target_ratio)
        elif len(tensor.shape) >= 3:
            decomp_type, params = TensorDecomposer.auto_select(tensor, target_ratio)
        elif numel >= 1_000:
            # Safetensors + LZ4 (RAW comprimido)
            return self._create_secure_descriptor(tensor, **kwargs)
        else:
            # Pequeño: RAW sin overhead
            return self._create_secure_descriptor(tensor, **kwargs)

        # Ejecutar descomposición
        try:
            components = TensorDecomposer.decompose(tensor, decomp_type, **params)

            # Serializar componentes con safetensors cuando es posible,
            # fallback a msgpack para estructuras complejas
            components_meta = self._serialize_components(components)
            compressed = lz4.frame.compress(
                components_meta,
                compression_level=self.config.compression_level.value,
            )

            # Función de reconstrucción para LazyTensor
            def _reconstruct_fn(data: bytes) -> torch.Tensor:
                raw = lz4.frame.decompress(data)
                comps = self._deserialize_components(raw)
                return TensorDecomposer.reconstruct(comps).to(
                    dtype=tensor.dtype, device=self.device
                )

            lazy = LazyTensor(
                compressed_data=compressed,
                decompression_func=_reconstruct_fn,
                metadata={"shape": tuple(tensor.shape),
                          "dtype": str(tensor.dtype),
                          "decomp_type": decomp_type.value},
                device=self.device,
                max_memory_mb=self.config.lazy_tensor_memory_limit,
            )

            original_bytes = tensor.nelement() * tensor.element_size()
            ratio = len(compressed) / max(original_bytes, 1)

            desc = ZDescriptor(
                kind="tensor",
                decomp_type=decomp_type,
                shape=tuple(tensor.shape),
                ranks=params.get("ranks") or (
                    (params["rank"],) if "rank" in params else None
                ),
                core_data=compressed,
                version=0,
                meta={
                    "compression_ratio": ratio,
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "decomp_params": {
                        k: v if not isinstance(v, tuple) else list(v)
                        for k, v in params.items()
                    },
                },
                compression_level=self.config.compression_level,
                lazy_tensor=lazy,
                clave_integridad=self.config.secret_key,
            )
            return desc

        except Exception as e:
            logger.warning(
                f"Decomposition {decomp_type.value} failed ({e}), "
                f"falling back to RAW"
            )
            return self._create_secure_descriptor(tensor, **kwargs)

    def _create_quantized_descriptor(
        self, tensor: torch.Tensor, **kwargs
    ) -> ZDescriptor:
        """Crear descriptor con cuantización (INT4/INT8/GPTQ group-wise).

        Con ``enable_structured_sparsity`` aplica primero el pre-pass 2:4
        (StructuredSparsifier) y conserva la máscara en el payload para que
        la decuantización restaure el patrón exacto.
        """
        quant_type = kwargs.get("quantization_type", "int8")
        group_size = kwargs.get("group_size", 128)
        gptq_meta = kwargs.get("gptq_metadata")

        # Pre-pass 2:4: sparsificar ANTES de cuantizar; la máscara viaja en el
        # payload porque los ceros podados no decuantizan a cero exacto.
        sparsity_mask = None
        if kwargs.get("enable_structured_sparsity"):
            # Import perezoso: mneme_optimization importa de este módulo.
            from .mneme_optimization import StructuredSparsifier
            tensor, sparsity_mask = StructuredSparsifier.apply_2_4_sparsity(tensor)

        # Cuantizar — operamos en CPU para compatibilidad con numpy/msgpack
        original = tensor.float().cpu()
        if "int4" in quant_type:
            n_bits = 4
        else:
            n_bits = 8

        # Group-wise quantization. División hacia arriba: con división entera hacia
        # abajo el último grupo parcial quedaba fuera y el reshape fallaba.
        flat = original.reshape(-1)
        n_groups = max(1, -(-flat.numel() // group_size))
        padded_len = n_groups * group_size
        if padded_len > flat.numel():
            # Rellenar repitiendo el último valor real, no con ceros: un cero fuera del
            # rango del grupo ensancha su min/max y malgasta la escala del último grupo.
            relleno = flat[-1] if flat.numel() else torch.zeros((), dtype=flat.dtype)
            flat = torch.cat([
                flat,
                relleno.expand(padded_len - flat.numel()),
            ])
        groups = flat.reshape(n_groups, group_size)

        # Per-group scale + offset. El offset se guarda como g_min en float32: un
        # zero-point entero recortado a [0, qmax] no puede representar un g_min
        # positivo, y el recorte perdía el offset de todo grupo que no cruzara cero.
        g_min = groups.min(dim=1).values
        g_max = groups.max(dim=1).values
        qmax = (1 << n_bits) - 1
        scale = (g_max - g_min) / max(qmax, 1)
        scale = scale.clamp(min=1e-10)

        quantized = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round()
        quantized = quantized.clamp(0, qmax).to(torch.uint8)

        # Serializar meta de cuantización
        quant_payload = {
            "quant_format": QUANT_FORMAT_VERSION,
            "quantized": quantized.numpy().tobytes(),
            "scale": scale.numpy().tobytes(),
            "g_min": g_min.numpy().astype(np.float32).tobytes(),
            "n_bits": n_bits,
            "group_size": group_size,
            "n_groups": n_groups,
            "original_numel": tensor.numel(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        if sparsity_mask is not None:
            # Bits empaquetados (numel/8 bytes); True = valor conservado. Clave
            # aditiva dentro del formato 2 a propósito (subir la versión dejaría
            # ilegibles los checkpoints previos): un lector pre-máscara la ignora
            # y devuelve los podados como ~0 (error acotado por el paso del
            # grupo), no como cero exacto ni como fallo.
            quant_payload["sparsity_mask"] = np.packbits(
                sparsity_mask.cpu().numpy()
            ).tobytes()
        if gptq_meta:
            quant_payload["gptq"] = True

        payload_bytes = msgpack.packb(quant_payload, use_bin_type=True)
        compressed = lz4.frame.compress(
            payload_bytes,
            compression_level=self.config.compression_level.value,
        )

        def _dequant_fn(data: bytes) -> torch.Tensor:
            raw = lz4.frame.decompress(data)
            info = msgpack.unpackb(raw, raw=False)
            return _dequantize_group_payload(info)

        lazy = LazyTensor(
            compressed_data=compressed,
            decompression_func=_dequant_fn,
            metadata={"shape": tuple(tensor.shape),
                      "dtype": str(tensor.dtype),
                      "quant_type": quant_type},
            device=self.device,
            max_memory_mb=self.config.lazy_tensor_memory_limit,
        )

        original_bytes = tensor.nelement() * tensor.element_size()
        desc = ZDescriptor(
            kind="tensor",
            decomp_type=DecompType.QUANTIZED,
            shape=tuple(tensor.shape),
            core_data=compressed,
            version=0,
            meta={
                "compression_ratio": len(compressed) / max(original_bytes, 1),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "quantization_type": quant_type,
                "group_size": group_size,
                "n_bits": n_bits,
                "gptq": gptq_meta is not None,
                "structured_sparsity": sparsity_mask is not None,
            },
            compression_level=self.config.compression_level,
            lazy_tensor=lazy,
            clave_integridad=self.config.secret_key,
        )
        return desc

    def _serialize_components(self, components: dict[str, Any]) -> bytes:
        """Serializar componentes de descomposición con msgpack + safetensors."""
        tensors = {}
        meta = {"type": components["type"]}
        idx = 0

        for key, val in components.items():
            if key == "type":
                continue
            if isinstance(val, torch.Tensor):
                tname = f"t_{idx}"
                tensors[tname] = val.contiguous().cpu()
                meta[key] = {"__tensor__": tname}
                idx += 1
            elif isinstance(val, (list, tuple)):
                # Lista de factores (TT / CP)
                tensor_names = []
                for _i, t in enumerate(val):
                    if isinstance(t, torch.Tensor):
                        tname = f"t_{idx}"
                        tensors[tname] = t.contiguous().cpu()
                        tensor_names.append(tname)
                        idx += 1
                    elif hasattr(t, 'factors'):
                        # CPTensor-like objects from tensorly
                        for _j, f in enumerate(t):
                            tname = f"t_{idx}"
                            tensors[tname] = (
                                f.contiguous().cpu() if isinstance(f, torch.Tensor)
                                else torch.tensor(f)
                            )
                            tensor_names.append(tname)
                            idx += 1
                    else:
                        tname = f"t_{idx}"
                        tensors[tname] = torch.tensor(t) if not isinstance(t, torch.Tensor) else t.contiguous().cpu()
                        tensor_names.append(tname)
                        idx += 1
                meta[key] = {"__tensor_list__": tensor_names}
            elif isinstance(val, tuple):
                meta[key] = list(val)
            else:
                meta[key] = val

        # Serializar tensores con safetensors (in-memory)
        if tensors:
            tensor_data = safetensors.torch.save(tensors)
        else:
            tensor_data = b""

        meta_bytes = msgpack.packb(meta, use_bin_type=True)
        # Format: [4 bytes meta_len][meta_bytes][tensor_data]
        result = struct.pack('I', len(meta_bytes)) + meta_bytes + tensor_data
        return result

    def _deserialize_components(self, data: bytes) -> dict[str, Any]:
        """Deserializar componentes de descomposición."""
        meta_len = struct.unpack('I', data[:4])[0]
        meta = msgpack.unpackb(data[4:4 + meta_len], raw=False)
        tensor_data = data[4 + meta_len:]

        tensors = {}
        if tensor_data:
            tensors = safetensors.torch.load(tensor_data)

        result = {"type": meta["type"]}
        for key, val in meta.items():
            if key == "type":
                continue
            if isinstance(val, dict) and "__tensor__" in val:
                result[key] = tensors[val["__tensor__"]]
            elif isinstance(val, dict) and "__tensor_list__" in val:
                result[key] = [tensors[n] for n in val["__tensor_list__"]]
            else:
                result[key] = val

        return result

    def _decompress_tensor(self, compressed_data: bytes) -> torch.Tensor:
        """Decomprimir tensor (RAW path)."""
        core_data = lz4.frame.decompress(compressed_data)
        tensor, _ = self.security_manager.serializer.deserialize_tensor(
            core_data, self.device
        )
        return tensor

    def _synthesize_tensor(self, desc: ZDescriptor) -> torch.Tensor:
        """
        Sintetizar tensor desde descriptor con deduplicación concurrente.

        Si otra corrutina ya está sintetizando el mismo descriptor,
        reutiliza el Future en lugar de duplicar trabajo.
        """
        if not desc.verify_integrity():
            raise SecurityError("Descriptor integrity check failed")

        addr = ZAddr.compute(desc).addr

        # Deduplicación concurrente
        with self._synthesis_lock:
            if addr in self._pending_synthesis:
                future = self._pending_synthesis[addr]
                return future.result()

            future = self._synthesis_executor.submit(
                self._do_synthesize, desc
            )
            self._pending_synthesis[addr] = future

        try:
            tensor = future.result()
        finally:
            with self._synthesis_lock:
                self._pending_synthesis.pop(addr, None)

        # Registrar acceso para prefetching
        self.prefetcher.record_access(addr)

        return tensor

    def _do_synthesize(self, desc: ZDescriptor) -> torch.Tensor:
        """Ejecución real de síntesis (llamada dentro del executor)."""
        if desc.lazy_tensor is not None:
            return desc.lazy_tensor.decompress()

        # Fallback: descompresión directa de core_data
        if desc.decomp_type == DecompType.RAW:
            core_data = lz4.frame.decompress(desc.core_data)
            tensor, _ = self.security_manager.serializer.deserialize_tensor(
            core_data, self.device
        )
            return tensor
        elif desc.decomp_type == DecompType.QUANTIZED:
            # Debería tener lazy_tensor, pero fallback
            raw = lz4.frame.decompress(desc.core_data)
            info = msgpack.unpackb(raw, raw=False)
            return _dequantize_group_payload(info).to(self.device)
        else:
            # Descomposición tensorial
            raw = lz4.frame.decompress(desc.core_data)
            comps = self._deserialize_components(raw)
            tensor = TensorDecomposer.reconstruct(comps)
            # Mismo contrato que el closure de _load_from_storage: los
            # componentes reconstruyen en su dtype de cómputo, no en el del
            # tensor registrado; el original viaja en meta['dtype'].
            nombre_dtype = str(
                desc.meta.get('dtype', 'torch.float32')
            ).removeprefix('torch.')
            dtype_original = getattr(torch, nombre_dtype, None)
            if not isinstance(dtype_original, torch.dtype):
                dtype_original = torch.float32
            return tensor.to(dtype=dtype_original, device=self.device)

    def _register_descriptor(self, name: str, desc: ZDescriptor,
                            old_addr: ZAddr | None) -> ZAddr:
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

    def _load_from_storage(self, name: str) -> ZDescriptor | None:
        """Cargar desde storage"""
        try:
            desc_data = self.storage_backend.retrieve(f"desc_{name}")
            if not desc_data:
                return None

            desc_dict = json.loads(desc_data.decode('utf-8'))
            desc = ZDescriptor.from_dict(desc_dict)
            # La clave de integridad no viaja en el dict persistido: se reinyecta
            # desde la config para que verify_integrity() use la misma clave con
            # la que este espacio escribe.
            desc._clave_integridad = self.config.secret_key

            compressed_data = self.storage_backend.retrieve(f"data_{name}")
            # `desc_{name}` y `data_{name}` son dos blobs distintos que duplican los
            # mismos bytes. Si divergen, alguien tocó uno de los dos: el hash del
            # descriptor cubre core_data, así que sin esta comprobación un cambio en
            # `data_{name}` entraría sin ser cubierto por ninguna verificación.
            if compressed_data and compressed_data != desc.core_data:
                raise SecurityError(
                    f"el payload almacenado de '{name}' no coincide con el descriptor: "
                    f"uno de los dos fue alterado",
                    threat_type="integrity",
                )
            if compressed_data:
                # El closure debe corresponder al formato del payload: RAW viaja
                # como lz4+safetensors, QUANTIZED como lz4+msgpack y las
                # descomposiciones como lz4+componentes. Cablear siempre la ruta
                # RAW dejaba todo tensor no-RAW irrecuperable tras un reinicio:
                # IntegrityError al exigirle la cabecera MNEME a bytes msgpack.
                if desc.decomp_type == DecompType.QUANTIZED:
                    def decompression_func(data: bytes) -> torch.Tensor:
                        raw = lz4.frame.decompress(data)
                        info = msgpack.unpackb(raw, raw=False)
                        return _dequantize_group_payload(info)
                elif desc.decomp_type != DecompType.RAW:
                    # Mismo contrato que el closure de _create_smart_descriptor:
                    # la reconstrucción vuelve en el dtype original registrado.
                    nombre_dtype = str(
                        desc.meta.get('dtype', 'torch.float32')
                    ).removeprefix('torch.')
                    dtype_original = getattr(torch, nombre_dtype, None)
                    if not isinstance(dtype_original, torch.dtype):
                        dtype_original = torch.float32

                    def decompression_func(data: bytes) -> torch.Tensor:
                        raw = lz4.frame.decompress(data)
                        comps = self._deserialize_components(raw)
                        return TensorDecomposer.reconstruct(comps).to(
                            dtype=dtype_original, device=self.device
                        )
                else:
                    decompression_func = self._decompress_tensor

                desc.lazy_tensor = LazyTensor(
                    compressed_data=compressed_data,
                    decompression_func=decompression_func,
                    metadata={'shape': desc.shape, 'dtype': desc.meta.get('dtype', 'torch.float32')},
                    device=self.device,
                    max_memory_mb=self.config.lazy_tensor_memory_limit
                )

            return desc
        except (StorageAuthenticationError, StorageFormatError, SecurityError):
            # No se degradan a None: devolverlo aquí convertiría "el blob fue alterado
            # o la clave no corresponde" en "este tensor nunca existió", que es
            # justamente la confusión que la capa de almacenamiento evita a propósito.
            raise
        except Exception as e:
            logger.error(f"Failed to load '{name}' from storage: {e}")
            return None

    # === ESTADÍSTICAS Y MONITOREO ===

    def get_stats(self) -> dict[str, Any]:
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

    def get_performance_metrics(self) -> dict[str, Any]:
        """Obtener métricas de rendimiento del sistema completo.

        Unifica estadísticas de cache, storage, circuit breakers,
        prefetcher y métricas de latencia.
        """
        stats = self.get_stats()
        stats["prefetcher"] = self.prefetcher.get_stats()
        stats["pending_synthesis"] = len(self._pending_synthesis)
        stats["metrics_summary"] = self.metrics.get_all_metrics()
        return stats

    def optimize_system(self) -> dict[str, Any]:
        """Optimización proactiva del sistema MNEME.

        - Limpia cache de entradas expiradas
        - Compacta descriptores huérfanos
        - Fuerza GC de tensores lazy no accedidos
        """
        results: dict[str, Any] = {"actions": []}

        # 1. Limpiar entradas TTL expiradas del cache
        pre_cache = self.adaptive_cache.get_stats()
        pre_size = pre_cache.get("size", pre_cache.get("current_size", 0))
        # Evictar entradas expiradas (TTL) sin vaciar todo
        if hasattr(self.adaptive_cache, '_evict_expired'):
            self.adaptive_cache._evict_expired()
        post_cache = self.adaptive_cache.get_stats()
        post_size = post_cache.get("size", post_cache.get("current_size", 0))
        freed = pre_size - post_size
        results["cache_freed_bytes"] = freed
        results["actions"].append(f"cache_cleanup: freed {freed} bytes")

        # 2. Liberar tensores lazy no recientes
        cleared_count = 0
        for _name, desc in self.name_to_desc.items():
            if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
                lt = desc.lazy_tensor
                if (lt._decompressed_tensor is not None
                        and time.time() - lt._last_access > lt._idle_timeout):
                    lt.clear_decompressed()
                    cleared_count += 1
        results["lazy_tensors_cleared"] = cleared_count
        results["actions"].append(f"lazy_gc: cleared {cleared_count} tensors")

        # 3. GC de Python
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results["actions"].append("gc_collect + cuda_empty_cache")

        return results

    def get_metrics_prometheus(self) -> str:
        """Exportar métricas en formato Prometheus"""
        return self.metrics.export_prometheus()

    # === LIFECYCLE ===

    def sync_to_storage(self) -> dict[str, Any]:
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

        # GC
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()

        logger.info("Cleanup completed")

    def __enter__(self) -> ZSpace:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


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


def get_system_info() -> dict[str, Any]:
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
    except Exception:
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

    # Sistemas
    'MetricsRegistry',

    # Enums
    'LockType',
    'DecompType',
    'CompressionLevel',
    'SecurityLevel',
    'CachePolicy',
    'CircuitState',
    'HealthStatus',

    # Errores
    'MnemeError',
    'SecurityError',
    'ValidationError',
    'StorageError',
    'CompressionError',
    'CircuitBreakerError',
    'ResourceError',

    # Dataclasses
    'LatencyHistogram',


    # Decoradores
    'circuit_breaker_decorator',

    # Funciones de utilidad
    'create_mneme',
    'get_default_device',
    'get_system_info',
]
