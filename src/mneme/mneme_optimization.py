"""
MNEME Optimization Module V3.0 - Enterprise Edition
Módulo de optimización avanzado con resiliencia, observabilidad distribuida,
optimización tensorial real y procesamiento paralelo adaptativo.

Versión: 3.0.0
Autor: MNEME Development Team
Licencia: BSL 1.1

Changelog V3.0:
- Circuit Breaker pattern para resiliencia ante fallos
- Backpressure mechanism con rate limiting adaptativo
- Sistema de checkpointing/recovery robusto
- Observabilidad distribuida (OpenTelemetry-compatible)
- Tensor pooling para reducir allocaciones
- Compresión real multi-algoritmo (LZ4/ZSTD/Blosc)
- Mixed precision automático con fallback inteligente
- Gradient checkpointing con memory-aware scheduling
- Histogramas con percentiles (P50/P95/P99)
- Work-stealing scheduler para paralelización
- Pipeline parallelism para transformaciones
- Implementación completa CP/Tucker/TT decomposition
- Pruning dinámico y quantization automática
- Sparsification con soporte CSR/COO
"""

from __future__ import annotations

import asyncio
import gc
import gzip
import hashlib
import logging
import lzma
import multiprocessing as mp
import tempfile
import threading
import time
import uuid
import weakref
import zlib
from collections import OrderedDict, defaultdict, deque
from collections.abc import Awaitable, Callable
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from threading import Event, Lock
from typing import (
    Any,
    Final,
    TypeVar,
)

import mscs
import numpy as np
import torch
import torch.nn as nn

# Intentar importar bibliotecas opcionales de compresión
try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import blosc2
    HAS_BLOSC = True
except ImportError:
    HAS_BLOSC = False

try:
    import tensorly as tl
    from tensorly.decomposition import parafac, tensor_train, tucker
    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Importar desde el core de MNEME
from .mneme_core import (
    CircuitBreaker,
    CircuitState,
    CompressionLevel,
    HealthStatus,
    MnemeConfig,
)

logger = logging.getLogger(__name__)

# Type Variables
T = TypeVar('T')
TensorType = TypeVar('TensorType', bound=torch.Tensor)

# ============================================================================
# CONSTANTES Y CONFIGURACIONES GLOBALES
# ============================================================================

# Versión del módulo
__version__: Final[str] = "3.0.0"

# Límites por defecto
DEFAULT_MAX_WORKERS: Final[int] = min(32, (mp.cpu_count() or 1) * 2)
DEFAULT_QUEUE_SIZE: Final[int] = 10000
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5
DEFAULT_CIRCUIT_BREAKER_RESET_SECONDS: Final[float] = 60.0
DEFAULT_RATE_LIMIT_OPS_PER_SEC: Final[int] = 1000
DEFAULT_CHECKPOINT_INTERVAL_SECONDS: Final[float] = 300.0
DEFAULT_TENSOR_POOL_SIZE: Final[int] = 100

# Tamaños de memoria
KB: Final[int] = 1024
MB: Final[int] = KB * 1024
GB: Final[int] = MB * 1024

# Thresholds
SMALL_TENSOR_THRESHOLD: Final[int] = 1000  # elementos
LARGE_TENSOR_THRESHOLD: Final[int] = 10_000_000  # 10M elementos
COMPRESSION_THRESHOLD: Final[int] = 100_000  # 100K elementos

# ============================================================================
# ENUMS EXTENDIDOS
# ============================================================================

class OptimizationLevel(IntEnum):
    """Niveles de optimización con valores numéricos para comparación"""
    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    MAXIMUM = 3
    ADAPTIVE = 4
    EXTREME = 5  # Nuevo: optimización extrema con trade-offs

class ResourceType(Enum):
    """Tipos de recursos del sistema"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    VRAM = "vram"  # Nuevo: memoria de video específica
    SWAP = "swap"  # Nuevo: memoria swap

# HealthStatus y CircuitState importados de mneme_core (fuente única de verdad)
# OptimizationStrategy: eliminado (nunca referenciado por ningún código)

class CompressionAlgorithm(Enum):
    """Algoritmos de compresión disponibles"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZMA = "lzma"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BLOSC = "blosc"
    AUTO = "auto"  # Selección automática

class DecompositionMethod(Enum):
    """Métodos de descomposición tensorial"""
    CP = "cp"  # CANDECOMP/PARAFAC
    TUCKER = "tucker"
    TENSOR_TRAIN = "tensor_train"
    SVD = "svd"
    HOSVD = "hosvd"
    RANDOMIZED_SVD = "randomized_svd"

class QuantizationType(Enum):
    """Tipos de cuantización"""
    NONE = "none"
    INT8 = "int8"
    FP16 = "fp16"
    BF16 = "bf16"
    INT4 = "int4"
    DYNAMIC = "dynamic"
    INT4_GROUP = "int4_group"   # Group-wise INT4 (group_size=128)
    INT8_GROUP = "int8_group"   # Group-wise INT8 (group_size=128)
    GPTQ_INT4 = "gptq_int4"    # GPTQ-calibrated INT4
    GPTQ_INT8 = "gptq_int8"    # GPTQ-calibrated INT8

class SparsityFormat(Enum):
    """Formatos de matrices sparse"""
    DENSE = "dense"
    CSR = "csr"  # Compressed Sparse Row
    CSC = "csc"  # Compressed Sparse Column
    COO = "coo"  # Coordinate format
    BSR = "bsr"  # Block Sparse Row

# ============================================================================
# PROTOCOLOS Y TIPOS ABSTRACTOS
# ============================================================================

# Optimizable, Compressible, MetricsExporter: eliminados (nunca implementados/subclaseados)

# ============================================================================
# DATACLASSES MEJORADAS
# ============================================================================

@dataclass(frozen=True, slots=True)
class TensorMetadata:
    """Metadatos inmutables de un tensor"""
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: str
    numel: int
    memory_bytes: int
    is_contiguous: bool
    requires_grad: bool
    hash_value: str
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorMetadata:
        """Crear metadatos desde un tensor"""
        return cls(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=str(tensor.device),
            numel=tensor.numel(),
            memory_bytes=tensor.element_size() * tensor.numel(),
            is_contiguous=tensor.is_contiguous(),
            requires_grad=tensor.requires_grad,
            hash_value=hashlib.md5(
                tensor.cpu().numpy().tobytes()[:1024]
            ).hexdigest()[:16]
        )

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del sistema con histogramas"""
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    cache_hit_rate: float = 0.0
    compression_ratio: float = 1.0
    avg_operation_time_ms: float = 0.0
    total_operations: int = 0
    failed_operations: int = 0

    # Nuevas métricas V3
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    memory_fragmentation: float = 0.0
    gc_pause_time_ms: float = 0.0
    tensor_pool_utilization: float = 0.0
    circuit_breaker_trips: int = 0
    backpressure_events: int = 0

    timestamp: datetime = field(default_factory=datetime.now)

    def success_rate(self) -> float:
        """Calcular tasa de éxito"""
        if self.total_operations == 0:
            return 1.0
        return (self.total_operations - self.failed_operations) / self.total_operations

    def to_dict(self) -> dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "memory_usage_percent": round(self.memory_usage_percent, 2),
            "cpu_usage_percent": round(self.cpu_usage_percent, 2),
            "gpu_usage_percent": round(self.gpu_usage_percent, 2),
            "gpu_memory_mb": round(self.gpu_memory_mb, 2),
            "gpu_memory_percent": round(self.gpu_memory_percent, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "compression_ratio": round(self.compression_ratio, 3),
            "avg_operation_time_ms": round(self.avg_operation_time_ms, 3),
            "total_operations": self.total_operations,
            "failed_operations": self.failed_operations,
            "success_rate": round(self.success_rate(), 4),
            "latency": {
                "p50_ms": round(self.p50_latency_ms, 3),
                "p95_ms": round(self.p95_latency_ms, 3),
                "p99_ms": round(self.p99_latency_ms, 3),
            },
            "throughput_ops_per_sec": round(self.throughput_ops_per_sec, 2),
            "memory_fragmentation": round(self.memory_fragmentation, 4),
            "gc_pause_time_ms": round(self.gc_pause_time_ms, 3),
            "tensor_pool_utilization": round(self.tensor_pool_utilization, 4),
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "backpressure_events": self.backpressure_events,
            "timestamp": self.timestamp.isoformat()
        }

    def to_prometheus_format(self) -> str:
        """Exportar en formato Prometheus"""
        lines = []
        prefix = "mneme"

        lines.append(f"# HELP {prefix}_memory_usage_bytes Memory usage in bytes")
        lines.append(f"# TYPE {prefix}_memory_usage_bytes gauge")
        lines.append(f"{prefix}_memory_usage_bytes {self.memory_usage_mb * MB}")

        lines.append(f"# HELP {prefix}_cpu_usage_percent CPU usage percentage")
        lines.append(f"# TYPE {prefix}_cpu_usage_percent gauge")
        lines.append(f"{prefix}_cpu_usage_percent {self.cpu_usage_percent}")

        lines.append(f"# HELP {prefix}_operations_total Total operations")
        lines.append(f"# TYPE {prefix}_operations_total counter")
        lines.append(f'{prefix}_operations_total{{status="success"}} {self.total_operations - self.failed_operations}')
        lines.append(f'{prefix}_operations_total{{status="failed"}} {self.failed_operations}')

        lines.append(f"# HELP {prefix}_latency_seconds Operation latency")
        lines.append(f"# TYPE {prefix}_latency_seconds summary")
        lines.append(f'{prefix}_latency_seconds{{quantile="0.5"}} {self.p50_latency_ms / 1000}')
        lines.append(f'{prefix}_latency_seconds{{quantile="0.95"}} {self.p95_latency_ms / 1000}')
        lines.append(f'{prefix}_latency_seconds{{quantile="0.99"}} {self.p99_latency_ms / 1000}')

        return "\n".join(lines)

@dataclass
class ResourceMetrics:
    """Métricas específicas de recursos"""
    resource_type: ResourceType
    current_usage: float
    peak_usage: float
    average_usage: float
    available: float
    total: float
    threshold_warning: float
    threshold_critical: float

    # Nuevos campos V3
    trend: float = 0.0  # Tendencia de uso (positivo = creciendo)
    predicted_exhaustion_seconds: float | None = None
    fragmentation: float = 0.0

    def is_warning(self) -> bool:
        """Verificar si está en nivel de advertencia.

        Los umbrales (threshold_warning/threshold_critical) están definidos
        como PORCENTAJE de uso, así que la comparación debe hacerse en esa
        misma unidad vía usage_percent() — no contra current_usage, que para
        MEMORY/GPU/VRAM es un valor absoluto en MB. usage_percent() ya
        protege total==0 devolviendo 0.0, así que con métricas vacías
        (total no disponible) esto es False sin dividir por cero.
        """
        return self.usage_percent() >= self.threshold_warning

    def is_critical(self) -> bool:
        """Verificar si está en nivel crítico (misma unidad que is_warning: ver su docstring)."""
        return self.usage_percent() >= self.threshold_critical

    def usage_percent(self) -> float:
        """Calcular porcentaje de uso"""
        if self.total == 0:
            return 0.0
        return (self.current_usage / self.total) * 100

    def headroom_percent(self) -> float:
        """Calcular margen disponible"""
        return 100.0 - self.usage_percent()

@dataclass
class OptimizationRecommendation:
    """Recomendación de optimización con metadatos extendidos"""
    priority: int  # 1=crítico, 2=alto, 3=medio, 4=bajo
    category: str
    title: str
    description: str
    estimated_improvement: str
    actions: list[str]

    # Nuevos campos V3
    confidence: float = 0.8  # Confianza en la recomendación
    auto_applicable: bool = False  # ¿Se puede aplicar automáticamente?
    prerequisites: list[str] = field(default_factory=list)
    estimated_duration_seconds: float = 0.0
    risk_level: str = "low"  # low, medium, high

    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CheckpointData:
    """Datos de checkpoint para recovery"""
    checkpoint_id: str
    created_at: datetime
    optimizer_state: dict[str, Any]
    metrics_snapshot: PerformanceMetrics
    resource_state: dict[str, Any]
    tensor_pool_state: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serializar checkpoint"""
        # mscs, no pickle: el módulo ya migró a mscs y estas dos llamadas quedaron
        # atrás. `pickle` no se importa en el paquete, así que además levantaban
        # NameError. Reintroducir pickle aquí abriría ejecución arbitraria al cargar
        # un checkpoint escrito por otro usuario en el directorio temporal compartido.
        return mscs.dumps(asdict(self))

    @classmethod
    def from_bytes(cls, data: bytes) -> CheckpointData:
        """Deserializar checkpoint"""
        d = mscs.loads(data)
        d['metrics_snapshot'] = PerformanceMetrics(**d['metrics_snapshot'])
        return cls(**d)

@dataclass
class CompressionResult:
    """Resultado de una operación de compresión"""
    original_size: int
    compressed_size: int
    algorithm: CompressionAlgorithm
    compression_ratio: float
    compression_time_ms: float
    data: bytes
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def space_saved_percent(self) -> float:
        """Porcentaje de espacio ahorrado"""
        if self.original_size == 0:
            return 0.0
        return (1 - self.compressed_size / self.original_size) * 100

@dataclass
class DecompositionResult:
    """Resultado de una descomposición tensorial"""
    method: DecompositionMethod
    factors: list[torch.Tensor]
    core: torch.Tensor | None
    original_shape: tuple[int, ...]
    rank: int | tuple[int, ...]
    reconstruction_error: float
    compression_ratio: float
    decomposition_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def reconstruct(self) -> torch.Tensor:
        """Reconstruir tensor desde factores"""
        if self.method == DecompositionMethod.CP:
            # Reconstrucción CP usando khatri-rao products
            result = None
            for _i, factor in enumerate(self.factors):
                if result is None:
                    result = factor
                else:
                    result = torch.einsum('ir,jr->ijr', result, factor)
            if result is not None:
                result = result.sum(dim=-1)
            return result
        elif self.method == DecompositionMethod.TUCKER:
            # Reconstrucción Tucker: G x_1 A x_2 B x_3 C ...
            result = self.core
            for _mode, factor in enumerate(self.factors):
                result = torch.tensordot(result, factor, dims=([0], [1]))
            return result
        else:
            raise NotImplementedError(f"Reconstruction not implemented for {self.method}")

# CircuitBreaker importado de mneme_core (fuente única de verdad)

# ============================================================================
# RATE LIMITER Y BACKPRESSURE
# ============================================================================

class TokenBucket:
    """
    Implementación de Token Bucket para rate limiting.

    Permite ráfagas controladas mientras mantiene un rate promedio.
    """

    def __init__(
        self,
        rate: float,  # tokens por segundo
        capacity: int,  # capacidad máxima del bucket
    ):
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_update = time.monotonic()
        self._lock = Lock()

    def _refill(self) -> None:
        """Rellenar tokens basado en tiempo transcurrido"""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now

    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: float = None) -> bool:
        """
        Adquirir tokens del bucket.

        Args:
            tokens: Número de tokens a adquirir
            blocking: Si True, espera hasta que haya tokens disponibles
            timeout: Tiempo máximo de espera (None = sin límite)

        Returns:
            True si se adquirieron los tokens, False si no
        """
        deadline = time.monotonic() + timeout if timeout else None

        while True:
            with self._lock:
                self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                if not blocking:
                    return False

                if deadline and time.monotonic() >= deadline:
                    return False

                # Calcular tiempo de espera
                needed = tokens - self._tokens
                wait_time = needed / self.rate

            # Esperar fuera del lock
            time.sleep(min(wait_time, 0.1))

    def available(self) -> float:
        """Obtener tokens disponibles"""
        with self._lock:
            self._refill()
            return self._tokens

class AdaptiveBackpressure:
    """
    Sistema de backpressure adaptativo basado en métricas del sistema.

    Ajusta automáticamente la presión según la carga observada.
    """

    def __init__(
        self,
        initial_rate: float = DEFAULT_RATE_LIMIT_OPS_PER_SEC,
        min_rate: float = 10.0,
        max_rate: float = 10000.0,
        adjustment_factor: float = 0.1,
        target_latency_ms: float = 100.0,
    ):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adjustment_factor = adjustment_factor
        self.target_latency_ms = target_latency_ms

        self._current_rate = initial_rate
        self._token_bucket = TokenBucket(initial_rate, int(initial_rate * 2))
        self._lock = Lock()

        # Métricas para adaptación
        self._latency_window: deque = deque(maxlen=100)
        self._rejection_count = 0
        self._total_requests = 0
        self._last_adjustment = time.time()
        self._adjustment_interval = 5.0  # segundos

        logger.debug(f"AdaptiveBackpressure initialized with rate {initial_rate}")

    def acquire(self, blocking: bool = True, timeout: float = 1.0) -> bool:
        """Intentar adquirir permiso para una operación"""
        self._total_requests += 1

        if self._token_bucket.acquire(blocking=blocking, timeout=timeout):
            return True

        self._rejection_count += 1
        return False

    def record_latency(self, latency_ms: float) -> None:
        """Registrar latencia de una operación"""
        with self._lock:
            self._latency_window.append(latency_ms)
            self._maybe_adjust_rate()

    def _maybe_adjust_rate(self) -> None:
        """Ajustar rate si es necesario"""
        now = time.time()
        if now - self._last_adjustment < self._adjustment_interval:
            return

        if len(self._latency_window) < 10:
            return

        self._last_adjustment = now

        # Calcular latencia promedio
        avg_latency = np.mean(list(self._latency_window))

        # Ajustar rate
        if avg_latency > self.target_latency_ms * 1.5:
            # Latencia muy alta, reducir rate
            new_rate = self._current_rate * (1 - self.adjustment_factor)
        elif avg_latency < self.target_latency_ms * 0.5:
            # Latencia baja, aumentar rate
            new_rate = self._current_rate * (1 + self.adjustment_factor)
        else:
            return  # Dentro del rango objetivo

        # Aplicar límites
        new_rate = max(self.min_rate, min(self.max_rate, new_rate))

        if abs(new_rate - self._current_rate) / self._current_rate > 0.05:
            self._current_rate = new_rate
            self._token_bucket = TokenBucket(new_rate, int(new_rate * 2))
            logger.debug(f"Backpressure rate adjusted to {new_rate:.1f} ops/sec")

    def get_stats(self) -> dict[str, Any]:
        """Obtener estadísticas"""
        with self._lock:
            return {
                "current_rate": self._current_rate,
                "available_tokens": self._token_bucket.available(),
                "total_requests": self._total_requests,
                "rejections": self._rejection_count,
                "rejection_rate": self._rejection_count / max(1, self._total_requests),
                "avg_latency_ms": np.mean(list(self._latency_window)) if self._latency_window else 0,
                "target_latency_ms": self.target_latency_ms,
            }

# ============================================================================
# HISTOGRAMA DE LATENCIAS
# ============================================================================

class LatencyHistogram:
    """
    Histograma de latencias con soporte para percentiles.

    Usa una estructura de buckets exponenciales para eficiencia.
    """

    def __init__(
        self,
        buckets: list[float] | None = None,
        max_samples: int = 10000,
    ):
        if buckets is None:
            buckets = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

        self.buckets = sorted(buckets)
        self.max_samples = max_samples

        self._samples: deque = deque(maxlen=max_samples)
        self._bucket_counts: dict[float, int] = {b: 0 for b in self.buckets}
        self._bucket_counts[float('inf')] = 0
        self._total_count = 0
        self._sum = 0.0
        self._lock = Lock()

    def record(self, value_ms: float) -> None:
        """Registrar un valor de latencia"""
        with self._lock:
            self._samples.append(value_ms)
            self._total_count += 1
            self._sum += value_ms

            for bucket in self.buckets:
                if value_ms <= bucket:
                    self._bucket_counts[bucket] += 1
                    return
            self._bucket_counts[float('inf')] += 1

    def percentile(self, p: float) -> float:
        """Calcular percentil (0-100)"""
        with self._lock:
            if not self._samples:
                return 0.0
            sorted_samples = sorted(self._samples)
            idx = min(int(len(sorted_samples) * p / 100), len(sorted_samples) - 1)
            return sorted_samples[idx]

    def mean(self) -> float:
        """Calcular media"""
        with self._lock:
            if self._total_count == 0:
                return 0.0
            return self._sum / self._total_count

    def get_percentiles(self) -> dict[str, float]:
        """Obtener percentiles comunes"""
        return {
            "p50": self.percentile(50),
            "p75": self.percentile(75),
            "p90": self.percentile(90),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "p999": self.percentile(99.9),
        }

    def get_histogram(self) -> dict[str, Any]:
        """Obtener histograma completo"""
        with self._lock:
            return {
                "buckets": dict(self._bucket_counts),
                "total_count": self._total_count,
                "sum_ms": self._sum,
                "mean_ms": self.mean(),
                "percentiles": self.get_percentiles(),
            }

# ============================================================================
# TENSOR POOL
# ============================================================================

class TensorPool:
    """
    Pool de tensores reutilizables para reducir allocaciones.

    Implementa un sistema de pooling con diferentes tamaños y shapes
    para maximizar la reutilización.
    """

    def __init__(
        self,
        max_tensors: int = DEFAULT_TENSOR_POOL_SIZE,
        max_memory_mb: float = 1024.0,
        device: torch.device | None = None,
    ):
        self.max_tensors = max_tensors
        self.max_memory_bytes = int(max_memory_mb * MB)
        self.device = device or torch.device('cpu')

        # Pool organizado por (shape, dtype)
        self._pool: dict[tuple[tuple[int, ...], torch.dtype], deque] = defaultdict(
            lambda: deque(maxlen=10)
        )
        self._in_use: weakref.WeakSet = weakref.WeakSet()
        self._current_memory = 0
        self._lock = Lock()

        # Estadísticas
        self._hits = 0
        self._misses = 0
        self._allocations = 0
        self._deallocations = 0

        logger.debug(f"TensorPool initialized: max_tensors={max_tensors}, max_memory={max_memory_mb}MB")

    def acquire(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        zero_fill: bool = True,
    ) -> torch.Tensor:
        """
        Adquirir un tensor del pool o crear uno nuevo.

        Args:
            shape: Shape del tensor requerido
            dtype: Tipo de datos
            zero_fill: Si True, llenar con ceros

        Returns:
            Tensor del pool o nuevo
        """
        key = (shape, dtype)

        with self._lock:
            pool_queue = self._pool[key]

            if pool_queue:
                tensor = pool_queue.popleft()
                self._hits += 1

                if zero_fill:
                    tensor.zero_()

                self._in_use.add(tensor)
                return tensor

            self._misses += 1

        # Crear nuevo tensor fuera del lock
        tensor = self._allocate_tensor(shape, dtype)

        with self._lock:
            self._in_use.add(tensor)

        return tensor

    def _allocate_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocar nuevo tensor"""
        tensor = torch.zeros(shape, dtype=dtype, device=self.device)
        tensor_size = tensor.element_size() * tensor.numel()

        with self._lock:
            self._current_memory += tensor_size
            self._allocations += 1

        return tensor

    def release(self, tensor: torch.Tensor) -> None:
        """
        Devolver tensor al pool para reutilización.

        Args:
            tensor: Tensor a devolver
        """
        if tensor.device != self.device:
            return  # No poolear tensores de otros devices

        key = (tuple(tensor.shape), tensor.dtype)

        with self._lock:
            # Verificar límites
            if len(self._pool[key]) >= 10:
                self._deallocations += 1
                return

            if self._current_memory > self.max_memory_bytes:
                self._evict_oldest()

            # Añadir al pool
            self._pool[key].append(tensor.detach())

            if tensor in self._in_use:
                # weakref.WeakSet no tiene discard, ignorar si no está
                pass

    def _evict_oldest(self) -> None:
        """Evictar tensores más antiguos"""
        evicted = 0
        target_evictions = max(1, len(self._pool) // 4)

        for key in list(self._pool.keys()):
            if evicted >= target_evictions:
                break

            pool_queue = self._pool[key]
            while pool_queue and evicted < target_evictions:
                tensor = pool_queue.popleft()
                tensor_size = tensor.element_size() * tensor.numel()
                self._current_memory -= tensor_size
                self._deallocations += 1
                evicted += 1

    def clear(self) -> None:
        """Limpiar todo el pool"""
        with self._lock:
            self._pool.clear()
            self._current_memory = 0
            logger.debug("TensorPool cleared")

    def get_stats(self) -> dict[str, Any]:
        """Obtener estadísticas del pool"""
        with self._lock:
            total_pooled = sum(len(q) for q in self._pool.values())

            return {
                "pooled_tensors": total_pooled,
                "unique_shapes": len(self._pool),
                "memory_mb": self._current_memory / MB,
                "max_memory_mb": self.max_memory_bytes / MB,
                "utilization": self._current_memory / self.max_memory_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(1, self._hits + self._misses),
                "allocations": self._allocations,
                "deallocations": self._deallocations,
            }

# ============================================================================
# COMPRESIÓN MULTI-ALGORITMO
# ============================================================================

class TensorCompressor:
    """
    Compresor de tensores con soporte para múltiples algoritmos.

    Selecciona automáticamente el mejor algoritmo según el tensor
    y las preferencias de velocidad/compresión.
    """

    ALGORITHM_PRIORITY = [
        CompressionAlgorithm.LZ4,
        CompressionAlgorithm.ZSTD,
        CompressionAlgorithm.BLOSC,
        CompressionAlgorithm.GZIP,
        CompressionAlgorithm.ZLIB,
        CompressionAlgorithm.LZMA,
    ]

    def __init__(
        self,
        default_algorithm: CompressionAlgorithm = CompressionAlgorithm.AUTO,
        compression_level: int = 6,
    ):
        self.default_algorithm = default_algorithm
        self.compression_level = compression_level

        # Verificar algoritmos disponibles
        self._available_algorithms = {CompressionAlgorithm.NONE, CompressionAlgorithm.GZIP,
                                       CompressionAlgorithm.ZLIB, CompressionAlgorithm.LZMA}
        if HAS_LZ4:
            self._available_algorithms.add(CompressionAlgorithm.LZ4)
        if HAS_ZSTD:
            self._available_algorithms.add(CompressionAlgorithm.ZSTD)
        if HAS_BLOSC:
            self._available_algorithms.add(CompressionAlgorithm.BLOSC)

        logger.debug(f"TensorCompressor initialized with algorithms: {self._available_algorithms}")

    def _select_algorithm(
        self,
        tensor: torch.Tensor,
        prefer_speed: bool = False,
    ) -> CompressionAlgorithm:
        """Seleccionar mejor algoritmo para el tensor"""
        if self.default_algorithm != CompressionAlgorithm.AUTO:
            if self.default_algorithm in self._available_algorithms:
                return self.default_algorithm

        numel = tensor.numel()

        # Para tensores pequeños, usar algo rápido
        if numel < SMALL_TENSOR_THRESHOLD:
            if CompressionAlgorithm.LZ4 in self._available_algorithms:
                return CompressionAlgorithm.LZ4
            return CompressionAlgorithm.ZLIB

        # Para tensores grandes, priorizar ratio de compresión
        if numel > LARGE_TENSOR_THRESHOLD and not prefer_speed:
            if CompressionAlgorithm.ZSTD in self._available_algorithms:
                return CompressionAlgorithm.ZSTD
            return CompressionAlgorithm.GZIP

        # Default: balance velocidad/compresión
        if CompressionAlgorithm.LZ4 in self._available_algorithms:
            return CompressionAlgorithm.LZ4
        if CompressionAlgorithm.ZSTD in self._available_algorithms:
            return CompressionAlgorithm.ZSTD

        return CompressionAlgorithm.GZIP

    def compress(
        self,
        tensor: torch.Tensor,
        algorithm: CompressionAlgorithm | None = None,
    ) -> CompressionResult:
        """
        Comprimir un tensor.

        Args:
            tensor: Tensor a comprimir
            algorithm: Algoritmo específico o None para auto

        Returns:
            CompressionResult con datos comprimidos y metadatos
        """
        if algorithm is None or algorithm == CompressionAlgorithm.AUTO:
            algorithm = self._select_algorithm(tensor)

        start_time = time.time()

        # Serializar tensor
        raw_data = mscs.dumps({
            'data': tensor.cpu(),
            'shape': tensor.shape,
            'dtype': tensor.dtype,
        })
        original_size = len(raw_data)

        # Comprimir
        if algorithm == CompressionAlgorithm.NONE:
            compressed_data = raw_data
        elif algorithm == CompressionAlgorithm.GZIP:
            compressed_data = gzip.compress(raw_data, compresslevel=self.compression_level)
        elif algorithm == CompressionAlgorithm.ZLIB:
            compressed_data = zlib.compress(raw_data, level=self.compression_level)
        elif algorithm == CompressionAlgorithm.LZMA:
            compressed_data = lzma.compress(raw_data, preset=min(self.compression_level, 9))
        elif algorithm == CompressionAlgorithm.LZ4 and HAS_LZ4:
            compressed_data = lz4.compress(raw_data, compression_level=self.compression_level)
        elif algorithm == CompressionAlgorithm.ZSTD and HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=self.compression_level)
            compressed_data = cctx.compress(raw_data)
        elif algorithm == CompressionAlgorithm.BLOSC and HAS_BLOSC:
            compressed_data = blosc2.compress(
                raw_data,
                clevel=self.compression_level,
                shuffle=blosc2.SHUFFLE
            )
        else:
            # Fallback
            compressed_data = gzip.compress(raw_data, compresslevel=self.compression_level)
            algorithm = CompressionAlgorithm.GZIP

        compression_time_ms = (time.time() - start_time) * 1000
        compressed_size = len(compressed_data)

        return CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            algorithm=algorithm,
            compression_ratio=original_size / max(1, compressed_size),
            compression_time_ms=compression_time_ms,
            data=compressed_data,
            metadata={
                'tensor_shape': list(tensor.shape),
                'tensor_dtype': str(tensor.dtype),
            }
        )

    def decompress(
        self,
        result: CompressionResult,
    ) -> torch.Tensor:
        """
        Descomprimir un tensor.

        Args:
            result: CompressionResult con datos comprimidos

        Returns:
            Tensor descomprimido
        """
        algorithm = result.algorithm
        data = result.data

        # Descomprimir
        if algorithm == CompressionAlgorithm.NONE:
            raw_data = data
        elif algorithm == CompressionAlgorithm.GZIP:
            raw_data = gzip.decompress(data)
        elif algorithm == CompressionAlgorithm.ZLIB:
            raw_data = zlib.decompress(data)
        elif algorithm == CompressionAlgorithm.LZMA:
            raw_data = lzma.decompress(data)
        elif algorithm == CompressionAlgorithm.LZ4 and HAS_LZ4:
            raw_data = lz4.decompress(data)
        elif algorithm == CompressionAlgorithm.ZSTD and HAS_ZSTD:
            dctx = zstd.ZstdDecompressor()
            raw_data = dctx.decompress(data)
        elif algorithm == CompressionAlgorithm.BLOSC and HAS_BLOSC:
            raw_data = blosc2.decompress(data)
        else:
            raw_data = gzip.decompress(data)

        # Deserializar
        loaded = mscs.loads(raw_data)

        return loaded['data']

    def benchmark_algorithms(
        self,
        tensor: torch.Tensor,
    ) -> dict[str, dict[str, float]]:
        """
        Benchmark todos los algoritmos disponibles para un tensor.

        Returns:
            Dict con métricas por algoritmo
        """
        results = {}

        for algo in self._available_algorithms:
            if algo in (CompressionAlgorithm.NONE, CompressionAlgorithm.AUTO):
                continue

            try:
                result = self.compress(tensor, algorithm=algo)

                # Test decompression
                start = time.time()
                _ = self.decompress(result)
                decomp_time_ms = (time.time() - start) * 1000

                results[algo.value] = {
                    'compression_ratio': result.compression_ratio,
                    'compression_time_ms': result.compression_time_ms,
                    'decompression_time_ms': decomp_time_ms,
                    'total_time_ms': result.compression_time_ms + decomp_time_ms,
                    'compressed_size_bytes': result.compressed_size,
                }
            except Exception as e:
                results[algo.value] = {'error': str(e)}

        return results

# ============================================================================
# DESCOMPOSICIÓN TENSORIAL
# ============================================================================

class TensorDecomposer:
    """
    Descomposición tensorial con múltiples métodos.

    Implementa CP, Tucker, y Tensor-Train decomposition para
    compresión y aproximación de bajo rango.
    """

    def __init__(
        self,
        default_method: DecompositionMethod = DecompositionMethod.CP,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        self.default_method = default_method
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        if not HAS_TENSORLY:
            logger.warning("TensorLy not available - decomposition will use fallback methods")

    def decompose(
        self,
        tensor: torch.Tensor,
        rank: int | tuple[int, ...],
        method: DecompositionMethod | None = None,
    ) -> DecompositionResult:
        """
        Descomponer tensor usando el método especificado.

        Args:
            tensor: Tensor a descomponer
            rank: Rango de la descomposición
            method: Método de descomposición

        Returns:
            DecompositionResult con factores y metadatos
        """
        method = method or self.default_method
        start_time = time.time()

        original_shape = tuple(tensor.shape)
        original_numel = tensor.numel()

        if method == DecompositionMethod.CP:
            factors, core = self._cp_decomposition(tensor, rank)
        elif method == DecompositionMethod.TUCKER:
            factors, core = self._tucker_decomposition(tensor, rank)
        elif method == DecompositionMethod.TENSOR_TRAIN:
            factors, core = self._tt_decomposition(tensor, rank)
        elif method == DecompositionMethod.SVD:
            factors, core = self._svd_decomposition(tensor, rank)
        else:
            raise ValueError(f"Unknown decomposition method: {method}")

        # Calcular error de reconstrucción
        result = DecompositionResult(
            method=method,
            factors=factors,
            core=core,
            original_shape=original_shape,
            rank=rank,
            reconstruction_error=0.0,  # Se calcula abajo
            compression_ratio=1.0,
            decomposition_time_ms=(time.time() - start_time) * 1000,
        )

        # Calcular métricas
        try:
            reconstructed = result.reconstruct()
            result.reconstruction_error = float(
                torch.norm(tensor - reconstructed) / torch.norm(tensor)
            )
        except Exception:
            result.reconstruction_error = float('inf')

        # Calcular ratio de compresión
        factor_numel = sum(f.numel() for f in factors)
        if core is not None:
            factor_numel += core.numel()
        result.compression_ratio = original_numel / max(1, factor_numel)

        return result

    def _cp_decomposition(
        self,
        tensor: torch.Tensor,
        rank: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Descomposición CP (CANDECOMP/PARAFAC)"""
        if HAS_TENSORLY:
            tl.set_backend('pytorch')
            factors = parafac(
                tensor,
                rank=rank,
                n_iter_max=self.max_iterations,
                tol=self.tolerance,
            )
            # parafac devuelve (weights, factors)
            if isinstance(factors, tuple):
                weights, factor_matrices = factors
                # Absorber weights en el primer factor
                factor_matrices[0] = factor_matrices[0] * weights.unsqueeze(0)
                return list(factor_matrices), None
            return list(factors), None
        else:
            # Fallback: usar SVD iterativo
            return self._fallback_cp(tensor, rank)

    def _fallback_cp(
        self,
        tensor: torch.Tensor,
        rank: int,
    ) -> tuple[list[torch.Tensor], None]:
        """Fallback CP usando ALS simplificado"""
        ndim = tensor.ndim
        factors = []

        # Inicialización aleatoria
        for mode in range(ndim):
            size = tensor.shape[mode]
            factor = torch.randn(size, rank, dtype=tensor.dtype, device=tensor.device)
            factors.append(factor)

        # ALS iterations (simplificado)
        for _ in range(min(10, self.max_iterations)):
            for mode in range(ndim):
                # Matricizar tensor
                unfolded = self._unfold(tensor, mode)

                # Khatri-Rao product de los otros factores
                kr_product = self._khatri_rao_product(
                    [f for i, f in enumerate(factors) if i != mode]
                )

                # Actualizar factor
                if kr_product.shape[0] > 0:
                    factors[mode] = torch.linalg.lstsq(
                        kr_product, unfolded.T
                    ).solution.T

        return factors, None

    def _tucker_decomposition(
        self,
        tensor: torch.Tensor,
        ranks: int | tuple[int, ...],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Descomposición Tucker"""
        if isinstance(ranks, int):
            ranks = tuple(min(ranks, s) for s in tensor.shape)

        if HAS_TENSORLY:
            tl.set_backend('pytorch')
            core, factors = tucker(
                tensor,
                rank=ranks,
                n_iter_max=self.max_iterations,
                tol=self.tolerance,
            )
            return list(factors), core
        else:
            return self._fallback_tucker(tensor, ranks)

    def _fallback_tucker(
        self,
        tensor: torch.Tensor,
        ranks: tuple[int, ...],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Fallback Tucker usando HOSVD"""
        factors = []
        core = tensor

        for mode, rank in enumerate(ranks):
            # Matricizar
            unfolded = self._unfold(core, mode)

            # SVD truncado
            U, S, Vh = torch.linalg.svd(unfolded, full_matrices=False)
            U = U[:, :rank]

            factors.append(U)

            # Actualizar core
            core = torch.tensordot(core, U, dims=([mode], [0]))
            # Mover dimensión al final y luego de vuelta
            core = core.movedim(-1, mode)

        return factors, core

    def _tt_decomposition(
        self,
        tensor: torch.Tensor,
        rank: int,
    ) -> tuple[list[torch.Tensor], None]:
        """Descomposición Tensor-Train"""
        if HAS_TENSORLY:
            tl.set_backend('pytorch')
            factors = tensor_train(tensor, rank=rank)
            return list(factors), None
        else:
            return self._fallback_tt(tensor, rank)

    def _fallback_tt(
        self,
        tensor: torch.Tensor,
        rank: int,
    ) -> tuple[list[torch.Tensor], None]:
        """Fallback TT usando SVD secuencial"""
        cores = []
        remaining = tensor.reshape(-1)

        shape = tensor.shape
        n_dims = len(shape)

        r_prev = 1
        for k in range(n_dims - 1):
            n_k = shape[k]
            remaining = remaining.reshape(r_prev * n_k, -1)

            U, S, Vh = torch.linalg.svd(remaining, full_matrices=False)
            r_k = min(rank, U.shape[1])

            U = U[:, :r_k]
            S = S[:r_k]
            Vh = Vh[:r_k, :]

            cores.append(U.reshape(r_prev, n_k, r_k))
            remaining = torch.diag(S) @ Vh
            r_prev = r_k

        # Último core
        cores.append(remaining.reshape(r_prev, shape[-1], 1))

        return cores, None

    def _svd_decomposition(
        self,
        tensor: torch.Tensor,
        rank: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Descomposición SVD para tensores 2D"""
        if tensor.ndim != 2:
            # Matricizar para tensores de mayor dimensión
            tensor = tensor.reshape(tensor.shape[0], -1)

        U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)

        rank = min(rank, len(S))
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        return [U, Vh.T], torch.diag(S)

    @staticmethod
    def _unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
        """Matricizar tensor a lo largo de un modo"""
        return tensor.movedim(mode, 0).reshape(tensor.shape[mode], -1)

    @staticmethod
    def _khatri_rao_product(matrices: list[torch.Tensor]) -> torch.Tensor:
        """Producto Khatri-Rao de una lista de matrices"""
        if not matrices:
            return torch.tensor([])

        result = matrices[0]
        for mat in matrices[1:]:
            # Khatri-Rao: column-wise Kronecker
            n1, r = result.shape
            n2, _ = mat.shape
            result = (result.unsqueeze(1) * mat.unsqueeze(0)).reshape(n1 * n2, r)

        return result

# ============================================================================
# QUANTIZACIÓN
# ============================================================================

class TensorQuantizer:
    """
    Cuantización de tensores para reducción de memoria.

    Soporta INT8, FP16, BF16 con escalado dinámico.
    """

    def __init__(self, default_type: QuantizationType = QuantizationType.DYNAMIC):
        self.default_type = default_type

    def quantize(
        self,
        tensor: torch.Tensor,
        quant_type: QuantizationType | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Cuantizar tensor.

        Returns:
            Tuple de (tensor cuantizado, metadatos para descuantización)
        """
        quant_type = quant_type or self.default_type

        if quant_type == QuantizationType.NONE:
            return tensor, {}

        if quant_type == QuantizationType.FP16:
            return tensor.half(), {'dtype': tensor.dtype}

        if quant_type == QuantizationType.BF16:
            return tensor.bfloat16(), {'dtype': tensor.dtype}

        if quant_type == QuantizationType.INT8:
            return self._quantize_int8(tensor)

        if quant_type == QuantizationType.INT4:
            return self._quantize_int4(tensor)

        if quant_type == QuantizationType.INT4_GROUP:
            return self._quantize_int4_group(tensor)

        if quant_type == QuantizationType.INT8_GROUP:
            return self._quantize_int8_group(tensor)

        if quant_type == QuantizationType.DYNAMIC:
            return self._quantize_dynamic(tensor)

        return tensor, {}

    def _quantize_int8(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Cuantización INT8 con escalado simétrico"""
        abs_max = tensor.abs().max()
        scale = abs_max / 127.0 if abs_max > 0 else 1.0

        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)

        return quantized, {
            'scale': scale.item(),
            'dtype': tensor.dtype,
            'quant_type': 'int8',
        }

    def _quantize_int4(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Cuantización INT4 (empaquetado en INT8)"""
        abs_max = tensor.abs().max()
        scale = abs_max / 7.0 if abs_max > 0 else 1.0

        quantized = (tensor / scale).round().clamp(-8, 7).to(torch.int8)

        # Empaquetar dos valores INT4 en un INT8
        flat = quantized.flatten()
        if flat.numel() % 2 == 1:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8, device=flat.device)])

        packed = (flat[::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)

        return packed, {
            'scale': scale.item(),
            'dtype': tensor.dtype,
            'shape': list(tensor.shape),
            'quant_type': 'int4',
        }

    def _quantize_dynamic(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Cuantización dinámica basada en contenido"""
        numel = tensor.numel()

        # Elegir tipo basado en tamaño y rango
        if numel < 1000:
            return tensor, {}  # No cuantizar tensores pequeños

        abs_max = tensor.abs().max()

        if abs_max < 1e4:
            # Rango moderado -> FP16
            return tensor.half(), {'dtype': tensor.dtype, 'quant_type': 'fp16'}
        else:
            # Rango amplio -> INT8 con escala
            return self._quantize_int8(tensor)

    def _quantize_int4_group(
        self,
        tensor: torch.Tensor,
        group_size: int = 128,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Group-wise asymmetric INT4 quantization.

        Each group of ``group_size`` elements along the last dimension gets its
        own scale and zero_point, significantly reducing quantization error
        compared to per-tensor quantization.
        """
        original_shape = tensor.shape
        # Flatten all dims except last, then group along last dim
        flat = tensor.reshape(-1, tensor.shape[-1]).float()
        rows, cols = flat.shape

        # Pad last dim to multiple of group_size
        pad_cols = (group_size - cols % group_size) % group_size
        if pad_cols > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_cols))

        num_groups = flat.shape[-1] // group_size
        grouped = flat.reshape(rows, num_groups, group_size)

        # Per-group min / max → asymmetric [0, 15]. El offset se propaga como g_min en
        # float: un zero-point entero recortado a [0, 15] no puede representar un g_min
        # positivo, y el recorte perdía el offset de todo grupo que no cruzara cero.
        g_min = grouped.amin(dim=-1, keepdim=True)
        g_max = grouped.amax(dim=-1, keepdim=True)
        scale = (g_max - g_min) / 15.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        q = ((grouped - g_min) / scale).round().clamp(0, 15).to(torch.uint8)

        # Pack two uint4 into one int8
        q_flat = q.reshape(rows, -1)
        if q_flat.shape[-1] % 2 == 1:
            q_flat = torch.cat([q_flat, torch.zeros(rows, 1, dtype=torch.uint8, device=q_flat.device)], dim=-1)
        packed = (q_flat[:, ::2]) | (q_flat[:, 1::2] << 4)

        return packed.to(torch.int8), {
            'quant_type': 'int4_group',
            'scales': scale.squeeze(-1).contiguous(),
            'g_min': g_min.squeeze(-1).contiguous(),
            'original_shape': list(original_shape),
            'group_size': group_size,
            'pad_cols': pad_cols,
            'dtype': tensor.dtype,
            'rows': rows,
            'num_groups': num_groups,
        }

    def _quantize_int8_group(
        self,
        tensor: torch.Tensor,
        group_size: int = 128,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Group-wise symmetric INT8 quantization.

        Each group of ``group_size`` elements gets its own scale factor.
        """
        original_shape = tensor.shape
        flat = tensor.reshape(-1, tensor.shape[-1]).float()
        rows, cols = flat.shape

        pad_cols = (group_size - cols % group_size) % group_size
        if pad_cols > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_cols))

        num_groups = flat.shape[-1] // group_size
        grouped = flat.reshape(rows, num_groups, group_size)

        # Per-group symmetric scale
        abs_max = grouped.abs().amax(dim=-1, keepdim=True)
        scale = abs_max / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        q = (grouped / scale).round().clamp(-128, 127).to(torch.int8)

        return q.reshape(rows, -1).contiguous(), {
            'quant_type': 'int8_group',
            'scales': scale.squeeze(-1).contiguous(),
            'original_shape': list(original_shape),
            'group_size': group_size,
            'pad_cols': pad_cols,
            'dtype': tensor.dtype,
            'rows': rows,
            'num_groups': num_groups,
        }

    def dequantize(
        self,
        tensor: torch.Tensor,
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        """Descuantizar tensor"""
        if not metadata:
            return tensor

        quant_type = metadata.get('quant_type', '')
        original_dtype = metadata.get('dtype', torch.float32)

        if quant_type == 'fp16':
            return tensor.to(original_dtype)

        if quant_type == 'int8':
            scale = metadata['scale']
            return tensor.to(original_dtype) * scale

        if quant_type == 'int4':
            scale = metadata['scale']
            shape = metadata['shape']

            # Desempaquetar
            low = tensor & 0x0F
            high = (tensor >> 4) & 0x0F

            # Extender signo
            low = torch.where(low > 7, low - 16, low)
            high = torch.where(high > 7, high - 16, high)

            unpacked = torch.stack([low, high], dim=-1).flatten()
            unpacked = unpacked[:np.prod(shape)]

            return (unpacked.to(original_dtype) * scale).reshape(shape)

        if quant_type == 'int4_group':
            return self._dequantize_int4_group(tensor, metadata)

        if quant_type == 'int8_group':
            return self._dequantize_int8_group(tensor, metadata)

        if quant_type in ('gptq_int4', 'gptq_int8'):
            # GPTQ uses group dequantization with same format
            bits = 4 if 'int4' in quant_type else 8
            if bits == 4:
                return self._dequantize_int4_group(tensor, metadata)
            return self._dequantize_int8_group(tensor, metadata)

        return tensor.to(original_dtype) if 'dtype' in metadata else tensor

    def _dequantize_int4_group(
        self,
        packed: torch.Tensor,
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        """Dequantize group-wise asymmetric INT4."""
        scales = metadata['scales']          # (rows, num_groups)
        if 'g_min' not in metadata:
            raise ValueError(
                "metadata de cuantización sin 'g_min': fue producida por el "
                "codificador anterior, que recortaba el offset a un entero sin signo "
                "y perdía el de los grupos que no cruzan el cero. Recuantizar."
            )
        g_min = metadata['g_min']            # (rows, num_groups)
        original_shape = metadata['original_shape']
        group_size = metadata['group_size']
        original_dtype = metadata.get('dtype', torch.float32)
        rows = metadata['rows']
        num_groups = metadata['num_groups']

        # Unpack: low nibble and high nibble (unsigned)
        low = (packed & 0x0F).to(torch.float32)
        high = ((packed >> 4) & 0x0F).to(torch.float32)
        unpacked = torch.stack([low, high], dim=-1).reshape(rows, -1)

        # Trim to actual padded size
        total_padded = num_groups * group_size
        unpacked = unpacked[:, :total_padded]

        # Reshape into groups
        grouped = unpacked.reshape(rows, num_groups, group_size)

        # Dequantize: x = q * scale + g_min
        gm = g_min.float().unsqueeze(-1)           # (rows, num_groups, 1)
        sc = scales.float().unsqueeze(-1)          # (rows, num_groups, 1)
        dequantized = grouped * sc + gm

        # Flatten and trim padding
        flat = dequantized.reshape(rows, -1)
        cols = original_shape[-1]
        flat = flat[:, :cols]

        return flat.reshape(original_shape).to(original_dtype)

    def _dequantize_int8_group(
        self,
        q_tensor: torch.Tensor,
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        """Dequantize group-wise symmetric INT8."""
        scales = metadata['scales']  # (rows, num_groups)
        original_shape = metadata['original_shape']
        group_size = metadata['group_size']
        original_dtype = metadata.get('dtype', torch.float32)
        rows = metadata['rows']
        num_groups = metadata['num_groups']

        flat = q_tensor.reshape(rows, -1).float()
        total_padded = num_groups * group_size
        flat = flat[:, :total_padded]

        grouped = flat.reshape(rows, num_groups, group_size)
        sc = scales.float().unsqueeze(-1)
        dequantized = grouped * sc

        flat_out = dequantized.reshape(rows, -1)
        cols = original_shape[-1]
        flat_out = flat_out[:, :cols]

        return flat_out.reshape(original_shape).to(original_dtype)

# ============================================================================
# SPARSIFICATION
# ============================================================================

class TensorSparsifier:
    """
    Sparsificación de tensores con múltiples formatos.

    Convierte tensores densos a formatos sparse cuando es eficiente.
    """

    def __init__(
        self,
        default_format: SparsityFormat = SparsityFormat.CSR,
        threshold: float = 0.5,  # Sparsificar si >50% zeros
    ):
        self.default_format = default_format
        self.threshold = threshold

    def should_sparsify(self, tensor: torch.Tensor) -> bool:
        """Determinar si vale la pena sparsificar"""
        if tensor.ndim > 2:
            return False  # Solo tensores 2D por ahora

        sparsity = (tensor == 0).float().mean().item()
        return sparsity > self.threshold

    def sparsify(
        self,
        tensor: torch.Tensor,
        format: SparsityFormat | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Convertir tensor a formato sparse.

        Returns:
            Tuple de (tensor sparse, metadatos)
        """
        format = format or self.default_format

        if format == SparsityFormat.DENSE or not self.should_sparsify(tensor):
            return tensor, {'format': 'dense'}

        if tensor.ndim != 2:
            return tensor, {'format': 'dense'}

        if format == SparsityFormat.COO:
            sparse = tensor.to_sparse_coo()
        elif format == SparsityFormat.CSR:
            sparse = tensor.to_sparse_csr()
        elif format == SparsityFormat.CSC:
            sparse = tensor.to_sparse_csc()
        else:
            return tensor, {'format': 'dense'}

        # Calcular ratio de compresión
        dense_size = tensor.numel() * tensor.element_size()
        sparse_size = self._estimate_sparse_size(sparse)

        return sparse, {
            'format': format.value,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'compression_ratio': dense_size / max(1, sparse_size),
            'sparsity': (tensor == 0).float().mean().item(),
        }

    def _estimate_sparse_size(self, sparse: torch.Tensor) -> int:
        """Estimar tamaño en memoria de tensor sparse"""
        if sparse.layout == torch.sparse_coo:
            indices_size = sparse._indices().numel() * sparse._indices().element_size()
            values_size = sparse._values().numel() * sparse._values().element_size()
            return indices_size + values_size
        elif sparse.layout == torch.sparse_csr:
            return (
                sparse.crow_indices().numel() * 4 +
                sparse.col_indices().numel() * 4 +
                sparse.values().numel() * sparse.values().element_size()
            )
        return 0

    def densify(
        self,
        sparse: Any,
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        """Convertir tensor sparse de vuelta a denso"""
        if metadata.get('format') == 'dense':
            return sparse

        return sparse.to_dense()

# ============================================================================
# 2:4 STRUCTURED SPARSITY
# ============================================================================

class StructuredSparsifier:
    """2:4 structured sparsity: for every group of 4 consecutive values,
    keep only the 2 with largest magnitude, zero the rest.

    Compatible with NVIDIA Ampere sparse tensor cores for 2x speedup.
    Can be combined with quantization (sparsify then quantize).
    """

    @staticmethod
    def apply_2_4_sparsity(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 2:4 structured sparsity pattern.

        Returns:
            (sparse_tensor, boolean_mask) — mask is True where values are kept.
        """
        original_shape = tensor.shape
        numel = tensor.numel()

        # Pad to multiple of 4
        pad = (4 - numel % 4) % 4
        flat = tensor.flatten()
        if pad > 0:
            flat = torch.cat([flat, torch.zeros(pad, device=tensor.device, dtype=tensor.dtype)])

        groups = flat.reshape(-1, 4)

        # Top-2 per group of 4 by magnitude
        _, indices = groups.abs().topk(2, dim=-1)
        mask = torch.zeros_like(groups, dtype=torch.bool)
        mask.scatter_(1, indices, True)

        sparse = groups * mask.to(groups.dtype)

        # Trim padding
        sparse_out = sparse.flatten()[:numel].reshape(original_shape)
        mask_out = mask.flatten()[:numel].reshape(original_shape)

        return sparse_out, mask_out

    @staticmethod
    def compress_sparse_2_4(
        tensor: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact form: store only non-zero values (50%) + 2-bit position indices.

        Returns:
            (values, indices) where values has 50% of elements, indices encodes positions.
        """
        flat = tensor.flatten()
        mask_flat = mask.flatten()
        values = flat[mask_flat]

        # Encode positions per group of 4
        numel = flat.numel()
        pad = (4 - numel % 4) % 4
        if pad > 0:
            mask_flat = torch.cat([mask_flat, torch.zeros(pad, dtype=torch.bool, device=mask.device)])
        mask_groups = mask_flat.reshape(-1, 4)
        indices = torch.zeros(mask_groups.shape[0], 2, dtype=torch.uint8, device=tensor.device)
        for i in range(mask_groups.shape[0]):
            positions = mask_groups[i].nonzero(as_tuple=True)[0]
            if len(positions) >= 2:
                indices[i, 0] = positions[0].to(torch.uint8)
                indices[i, 1] = positions[1].to(torch.uint8)

        return values, indices

    @staticmethod
    def decompress_sparse_2_4(
        values: torch.Tensor, indices: torch.Tensor,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Reconstruct dense tensor from compressed 2:4 sparse representation."""
        numel = 1
        for s in original_shape:
            numel *= s

        num_groups = indices.shape[0]
        flat = torch.zeros(num_groups * 4, dtype=values.dtype, device=values.device)

        vi = 0
        for g in range(num_groups):
            pos0 = int(indices[g, 0].item())
            pos1 = int(indices[g, 1].item())
            flat[g * 4 + pos0] = values[vi]
            flat[g * 4 + pos1] = values[vi + 1]
            vi += 2

        return flat[:numel].reshape(original_shape)


# ============================================================================
# GPTQ CALIBRATED QUANTIZATION
# ============================================================================

class GPTQCalibrator:
    """GPTQ-style calibrated quantization for linear layers.

    Collects Hessian information (X^T X) from calibration data, then applies
    optimal quantization with Hessian-weighted error compensation — the key
    algorithm from Frantar et al. (2022).

    Usage::

        calibrator = GPTQCalibrator(bits=4, group_size=128)
        hessians = calibrator.collect_hessian(model, cal_data, num_samples=128)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in hessians:
                packed, meta = calibrator.quantize_layer(module.weight.data, hessians[name])
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        damp_percent: float = 0.01,
        block_size: int = 128,
    ):
        self.bits = bits
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.block_size = block_size

    def collect_hessian(
        self,
        model: torch.nn.Module,
        calibration_data,
        num_samples: int = 128,
    ) -> dict[str, torch.Tensor]:
        """Run calibration data through model, collecting X^T X for each Linear layer.

        Args:
            model: The model to calibrate.
            calibration_data: Iterable yielding input tensors (or batches).
            num_samples: Maximum samples to process.

        Returns:
            Dict mapping layer_name -> Hessian tensor (in_features, in_features).
        """
        hessians: dict[str, torch.Tensor] = {}
        sample_counts: dict[str, int] = {}
        hooks = []

        def _make_hook(name: str):
            def _hook(module, inp, out):
                x = inp[0].detach().float()   # FP32 to avoid FP16 overflow in x^T x
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                elif x.dim() == 1:
                    x = x.unsqueeze(0)
                H = x.t() @ x
                # Sanitize: replace any NaN/Inf with 0
                H = torch.where(torch.isfinite(H), H, torch.zeros_like(H))
                if name in hessians:
                    hessians[name] += H
                else:
                    hessians[name] = H.clone()
                sample_counts[name] = sample_counts.get(name, 0) + x.shape[0]
            return _hook

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(_make_hook(name)))

        model.eval()
        count = 0
        with torch.no_grad():
            for batch in calibration_data:
                if count >= num_samples:
                    break
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                if isinstance(batch, dict):
                    model(**batch)
                else:
                    model(batch)
                count += batch.shape[0] if hasattr(batch, 'shape') else 1

        for h in hooks:
            h.remove()

        # Normalize
        for name in hessians:
            n = sample_counts.get(name, 1)
            hessians[name] /= max(n, 1)

        return hessians

    def quantize_layer(
        self,
        weight: torch.Tensor,
        hessian: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply GPTQ quantization to a weight matrix using its Hessian.

        Core algorithm:
        1. Compute damped inverse Hessian via Cholesky
        2. Process columns in blocks: quantize each column, compensate error
           on remaining columns weighted by H^{-1}

        Returns:
            (packed_quantized_weights, metadata_dict)
        """
        W = weight.clone().float()
        rows, cols = W.shape
        device = W.device

        maxq = (2 ** self.bits) - 1

        # Sanitize Hessian: replace NaN/Inf with 0
        hessian = torch.where(torch.isfinite(hessian), hessian, torch.zeros_like(hessian))

        # Damping
        diag_mean = torch.diag(hessian).mean()
        if diag_mean == 0:
            diag_mean = torch.tensor(1.0, device=device)
        damp = self.damp_percent * diag_mean
        H = hessian.float() + damp * torch.eye(cols, device=device)

        # Cholesky-based inverse (preferred) with fallback to pseudo-inverse
        self._cholesky_ok = True
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except Exception:
            self._cholesky_ok = False
            H_inv = torch.linalg.pinv(H)

        # Prepare group quantization parameters
        pad_cols = (self.group_size - cols % self.group_size) % self.group_size
        total_cols = cols + pad_cols
        num_groups = total_cols // self.group_size

        scales = torch.zeros(rows, num_groups, device=device)
        zeros = torch.zeros(rows, num_groups, dtype=torch.int8, device=device)
        Q = torch.zeros_like(W)

        # Pad weight and H_inv if needed
        if pad_cols > 0:
            W = torch.nn.functional.pad(W, (0, pad_cols))
            H_inv = torch.nn.functional.pad(H_inv, (0, pad_cols, 0, pad_cols))
            for i in range(cols, cols + pad_cols):
                H_inv[i, i] = 1.0  # identity for padded dims
            Q = torch.nn.functional.pad(Q, (0, pad_cols))

        # Process in blocks
        for block_start in range(0, total_cols, self.block_size):
            block_end = min(block_start + self.block_size, total_cols)
            block_len = block_end - block_start

            W_block = W[:, block_start:block_end].clone()
            Q_block = torch.zeros_like(W_block)
            Err = torch.zeros_like(W_block)
            H_inv_block_diag = torch.diag(H_inv[block_start:block_end, block_start:block_end])

            for j in range(block_len):
                col_idx = block_start + j
                group_idx = col_idx // self.group_size

                # Compute group scale at group boundary
                if col_idx % self.group_size == 0:
                    g_start = col_idx
                    g_end = min(col_idx + self.group_size, total_cols)
                    w_group = W[:, g_start:g_end]
                    g_min = w_group.amin(dim=-1)
                    g_max = w_group.amax(dim=-1)
                    s = (g_max - g_min) / maxq
                    s = torch.where(s == 0, torch.ones_like(s), s)
                    zp = (-g_min / s).round().clamp(0, maxq).to(torch.int8)
                    scales[:, group_idx] = s
                    zeros[:, group_idx] = zp

                s = scales[:, group_idx]
                zp = zeros[:, group_idx].float()

                w = W_block[:, j]
                d = H_inv_block_diag[j].clamp(min=1e-8)

                # Quantize
                q_val = ((w / s) + zp).round().clamp(0, maxq)
                Q_block[:, j] = (q_val - zp) * s

                # Error and compensation
                err = (w - Q_block[:, j]) / d
                Err[:, j] = err

                # Update remaining columns in block
                if j + 1 < block_len:
                    h_row = H_inv[col_idx, block_start + j + 1:block_end]
                    W_block[:, j + 1:] -= err.unsqueeze(1) * h_row.unsqueeze(0)

            Q[:, block_start:block_end] = Q_block

            # Compensate remaining weights after block
            if block_end < total_cols:
                h_cross = H_inv[block_start:block_end, block_end:total_cols]
                W[:, block_end:] -= Err @ h_cross

        # Trim padding from Q
        Q = Q[:, :cols]

        # Pack Q into INT4 / INT8 using group quantization
        quantizer = TensorQuantizer()
        if self.bits == 4:
            packed, pack_meta = quantizer._quantize_int4_group(Q, self.group_size)
        else:
            packed, pack_meta = quantizer._quantize_int8_group(Q, self.group_size)

        # Use pack_meta as base (has the 'g_min' offset key for dequantize)
        # and add GPTQ-specific fields
        meta = {**pack_meta}
        meta['quant_type'] = f'gptq_int{self.bits}'
        meta['bits'] = self.bits
        meta['gptq_applied'] = True
        meta['gptq_scales'] = scales      # original GPTQ per-group scales
        meta['gptq_zeros'] = zeros        # original GPTQ per-group zeros

        return packed, meta


# ============================================================================
# PERFORMANCE MONITOR V3
# ============================================================================

class PerformanceMonitor:
    """
    Monitor de rendimiento del sistema con métricas avanzadas.

    Incluye histogramas de latencia, detección de anomalías y
    exportación de métricas.
    """

    def __init__(
        self,
        config: MnemeConfig,
        history_size: int = 1000,
        enable_gpu_monitoring: bool = True,
    ):
        self.config = config
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.lock = Lock()

        # Historial de métricas
        self.metrics_history: deque = deque(maxlen=history_size)
        self.operation_times: dict[str, LatencyHistogram] = defaultdict(LatencyHistogram)
        self.operation_counts: dict[str, int] = defaultdict(int)
        self.operation_failures: dict[str, int] = defaultdict(int)

        # Estado actual
        self.current_metrics = PerformanceMetrics()
        self.start_time = time.time()

        # Circuit breakers por operación
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Backpressure
        self.backpressure = AdaptiveBackpressure()

        # Thread para monitoreo continuo
        self.monitoring_active = False
        self.monitor_thread: threading.Thread | None = None
        self._stop_event = Event()

        # GC tracking
        self._gc_callback_id: int | None = None
        self._gc_pause_times: deque = deque(maxlen=100)

        logger.info("PerformanceMonitor V3 initialized")

    def start_monitoring(self, interval: float = 1.0):
        """Iniciar monitoreo continuo"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._stop_event.clear()

        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True,
            name="MNEME-Monitor"
        )
        self.monitor_thread.start()

        # Registrar callback de GC
        self._setup_gc_monitoring()

        logger.info(f"Started continuous monitoring with {interval}s interval")

    def _setup_gc_monitoring(self):
        """Configurar monitoreo de garbage collection"""
        def gc_callback(phase: str, info: dict):
            if phase == 'stop':
                # GC terminó
                generation = info.get('generation', 0)
                if generation == 2:  # Full GC
                    elapsed = info.get('elapsed', 0) * 1000  # ms
                    self._gc_pause_times.append(elapsed)

        try:
            gc.callbacks.append(gc_callback)
            self._gc_callback_id = len(gc.callbacks) - 1
        except Exception as e:
            logger.debug(f"Could not setup GC monitoring: {e}")

    def stop_monitoring(self):
        """Detener monitoreo continuo"""
        self.monitoring_active = False
        self._stop_event.set()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        # Remover callback de GC
        if self._gc_callback_id is not None:
            try:
                gc.callbacks.pop(self._gc_callback_id)
            except (IndexError, AttributeError):
                pass

        logger.info("Stopped monitoring")

    def _monitoring_loop(self, interval: float):
        """Loop de monitoreo continuo"""
        while not self._stop_event.wait(timeout=interval):
            try:
                self.update_metrics()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def update_metrics(self):
        """Actualizar métricas del sistema"""
        with self.lock:
            # Métricas de memoria
            if HAS_PSUTIL:
                process = psutil.Process()
                memory_info = process.memory_info()
                self.current_metrics.memory_usage_mb = memory_info.rss / MB

                virtual_memory = psutil.virtual_memory()
                self.current_metrics.memory_usage_percent = virtual_memory.percent

                # CPU
                self.current_metrics.cpu_usage_percent = process.cpu_percent(interval=0.1)

            # GPU metrics
            if self.enable_gpu_monitoring:
                try:
                    self.current_metrics.gpu_memory_mb = torch.cuda.memory_allocated() / MB
                    total_gpu = torch.cuda.get_device_properties(0).total_memory
                    self.current_metrics.gpu_memory_percent = (
                        torch.cuda.memory_allocated() / total_gpu * 100
                    )

                    # Utilization si está disponible
                    if hasattr(torch.cuda, 'utilization'):
                        self.current_metrics.gpu_usage_percent = torch.cuda.utilization()
                except Exception as e:
                    logger.debug(f"Could not get GPU metrics: {e}")

            # Calcular latencias agregadas
            all_histograms = list(self.operation_times.values())
            if all_histograms:
                p50s = [h.percentile(50) for h in all_histograms]
                p95s = [h.percentile(95) for h in all_histograms]
                p99s = [h.percentile(99) for h in all_histograms]

                self.current_metrics.p50_latency_ms = np.mean(p50s) if p50s else 0
                self.current_metrics.p95_latency_ms = np.mean(p95s) if p95s else 0
                self.current_metrics.p99_latency_ms = np.mean(p99s) if p99s else 0
                self.current_metrics.avg_operation_time_ms = np.mean([h.mean() for h in all_histograms])

            # Contadores
            self.current_metrics.total_operations = sum(self.operation_counts.values())
            self.current_metrics.failed_operations = sum(self.operation_failures.values())

            # Throughput
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.current_metrics.throughput_ops_per_sec = (
                    self.current_metrics.total_operations / elapsed
                )

            # GC pause time
            if self._gc_pause_times:
                self.current_metrics.gc_pause_time_ms = np.mean(list(self._gc_pause_times))

            # Circuit breaker stats
            self.current_metrics.circuit_breaker_trips = sum(
                1 for cb in self.circuit_breakers.values()
                if cb.state == CircuitState.OPEN
            )

            # Backpressure stats
            bp_stats = self.backpressure.get_stats()
            self.current_metrics.backpressure_events = bp_stats['rejections']

            # Update timestamp
            self.current_metrics.timestamp = datetime.now()

            # Agregar al historial
            self.metrics_history.append(PerformanceMetrics(**asdict(self.current_metrics)))

    def record_operation(
        self,
        operation_name: str,
        duration: float,
        success: bool = True,
    ):
        """Registrar una operación"""
        with self.lock:
            self.operation_times[operation_name].record(duration * 1000)
            self.operation_counts[operation_name] += 1

            if not success:
                self.operation_failures[operation_name] += 1

            # Actualizar backpressure
            self.backpressure.record_latency(duration * 1000)

    @contextmanager
    def measure_operation(self, operation_name: str):
        """Context manager para medir operaciones con circuit breaker"""
        # Obtener o crear circuit breaker
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker(operation_name)

        cb = self.circuit_breakers[operation_name]

        start_time = time.time()
        success = True

        try:
            with cb.protect():
                yield
        except Exception:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            self.record_operation(operation_name, duration, success)

    def get_performance_report(self) -> dict[str, Any]:
        """Obtener reporte de rendimiento completo"""
        # Se actualiza ANTES de tomar el lock: `self.lock` es un `Lock` no reentrante
        # y `update_metrics()` lo toma por su cuenta, así que llamarlo desde dentro
        # del `with` autobloqueaba el hilo de forma determinista.
        self.update_metrics()

        with self.lock:
            uptime = time.time() - self.start_time

            # Operaciones más lentas
            slow_operations = {}
            for op_name, histogram in self.operation_times.items():
                slow_operations[op_name] = {
                    "avg_time_ms": round(histogram.mean(), 3),
                    "p50_ms": round(histogram.percentile(50), 3),
                    "p95_ms": round(histogram.percentile(95), 3),
                    "p99_ms": round(histogram.percentile(99), 3),
                    "count": self.operation_counts[op_name],
                    "failures": self.operation_failures[op_name],
                }

            sorted_ops = sorted(
                slow_operations.items(),
                key=lambda x: x[1]["p99_ms"],
                reverse=True
            )[:10]

            # Circuit breaker stats
            cb_stats = {
                name: cb.get_stats()
                for name, cb in self.circuit_breakers.items()
            }

            return {
                "current_metrics": self.current_metrics.to_dict(),
                "uptime_seconds": round(uptime, 2),
                "operations": {
                    "total": sum(self.operation_counts.values()),
                    "failed": sum(self.operation_failures.values()),
                    "success_rate": round(self.current_metrics.success_rate(), 4),
                    "unique_operations": len(self.operation_counts),
                },
                "slowest_operations": dict(sorted_ops),
                "circuit_breakers": cb_stats,
                "backpressure": self.backpressure.get_stats(),
                "history_size": len(self.metrics_history),
            }

    def get_health_status(self) -> str:
        """Determinar estado de salud del sistema"""
        self.update_metrics()

        # Criterios de salud
        memory_critical = self.current_metrics.memory_usage_percent > 90
        memory_warning = self.current_metrics.memory_usage_percent > 75

        cpu_critical = self.current_metrics.cpu_usage_percent > 95
        cpu_warning = self.current_metrics.cpu_usage_percent > 80

        failure_rate = 1 - self.current_metrics.success_rate()
        failure_critical = failure_rate > 0.1
        failure_warning = failure_rate > 0.05

        # Latencia
        latency_critical = self.current_metrics.p99_latency_ms > 1000
        latency_warning = self.current_metrics.p95_latency_ms > 500

        # Circuit breakers
        open_breakers = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.OPEN
        )
        breaker_critical = open_breakers > len(self.circuit_breakers) * 0.5

        if memory_critical or cpu_critical or failure_critical or latency_critical or breaker_critical:
            return HealthStatus.CRITICAL.value

        if any(cb.state == CircuitState.HALF_OPEN for cb in self.circuit_breakers.values()):
            return HealthStatus.RECOVERING.value

        if memory_warning or cpu_warning or failure_warning or latency_warning:
            return HealthStatus.WARNING.value

        return HealthStatus.HEALTHY.value

    def export_prometheus_metrics(self) -> str:
        """Exportar métricas en formato Prometheus"""
        return self.current_metrics.to_prometheus_format()

    def cleanup(self):
        """Limpiar recursos"""
        self.stop_monitoring()

        with self.lock:
            self.metrics_history.clear()
            self.operation_times.clear()
            self.operation_counts.clear()
            self.operation_failures.clear()
            self.circuit_breakers.clear()

        logger.info("PerformanceMonitor cleaned up")

# ============================================================================
# RESOURCE OPTIMIZER V3
# ============================================================================

class ResourceOptimizer:
    """
    Optimizador de recursos del sistema con predicción y auto-scaling.
    """

    def __init__(self, config: MnemeConfig):
        self.config = config
        self.lock = Lock()

        # Umbrales de recursos
        self.thresholds = {
            ResourceType.MEMORY: {"warning": 75.0, "critical": 90.0},
            ResourceType.CPU: {"warning": 80.0, "critical": 95.0},
            ResourceType.GPU: {"warning": 85.0, "critical": 95.0},
            ResourceType.VRAM: {"warning": 80.0, "critical": 92.0},
        }

        # Historial para predicción
        self._memory_history: deque = deque(maxlen=60)
        self._cpu_history: deque = deque(maxlen=60)

        # Cache de métricas
        self.resource_cache: dict[ResourceType, ResourceMetrics] = {}
        self.last_optimization = None

        logger.info("ResourceOptimizer V3 initialized")

    def get_resource_metrics(self, resource_type: ResourceType) -> ResourceMetrics:
        """Obtener métricas de un recurso específico"""
        if resource_type == ResourceType.MEMORY:
            return self._get_memory_metrics()
        elif resource_type == ResourceType.CPU:
            return self._get_cpu_metrics()
        elif resource_type == ResourceType.GPU:
            return self._get_gpu_metrics()
        elif resource_type == ResourceType.VRAM:
            return self._get_vram_metrics()
        else:
            raise ValueError(f"Unsupported resource type: {resource_type}")

    def _get_memory_metrics(self) -> ResourceMetrics:
        """Obtener métricas de memoria con predicción"""
        if not HAS_PSUTIL:
            return self._empty_metrics(ResourceType.MEMORY)

        vm = psutil.virtual_memory()
        process = psutil.Process()
        process_mem = process.memory_info().rss

        current_usage = process_mem / MB
        self._memory_history.append(current_usage)

        # Calcular tendencia
        trend = 0.0
        if len(self._memory_history) >= 5:
            recent = list(self._memory_history)[-5:]
            trend = (recent[-1] - recent[0]) / len(recent)

        # Predecir agotamiento
        predicted_exhaustion = None
        if trend > 0 and vm.available > 0:
            time_to_exhaustion = (vm.available / MB) / trend
            if time_to_exhaustion > 0:
                predicted_exhaustion = time_to_exhaustion

        return ResourceMetrics(
            resource_type=ResourceType.MEMORY,
            current_usage=current_usage,
            peak_usage=process.memory_info().peak_wset / MB if hasattr(process.memory_info(), 'peak_wset') else current_usage,
            average_usage=np.mean(list(self._memory_history)) if self._memory_history else current_usage,
            available=vm.available / MB,
            total=vm.total / MB,
            threshold_warning=self.thresholds[ResourceType.MEMORY]["warning"],
            threshold_critical=self.thresholds[ResourceType.MEMORY]["critical"],
            trend=trend,
            predicted_exhaustion_seconds=predicted_exhaustion,
        )

    def _get_cpu_metrics(self) -> ResourceMetrics:
        """Obtener métricas de CPU"""
        if not HAS_PSUTIL:
            return self._empty_metrics(ResourceType.CPU)

        cpu_percent = psutil.cpu_percent(interval=0.1)
        self._cpu_history.append(cpu_percent)

        # Calcular tendencia
        trend = 0.0
        if len(self._cpu_history) >= 5:
            recent = list(self._cpu_history)[-5:]
            trend = (recent[-1] - recent[0]) / len(recent)

        return ResourceMetrics(
            resource_type=ResourceType.CPU,
            current_usage=cpu_percent,
            peak_usage=max(self._cpu_history) if self._cpu_history else cpu_percent,
            average_usage=np.mean(list(self._cpu_history)) if self._cpu_history else cpu_percent,
            available=100.0 - cpu_percent,
            total=100.0,
            threshold_warning=self.thresholds[ResourceType.CPU]["warning"],
            threshold_critical=self.thresholds[ResourceType.CPU]["critical"],
            trend=trend,
        )

    def _get_gpu_metrics(self) -> ResourceMetrics:
        """Obtener métricas de GPU"""
        if not torch.cuda.is_available():
            return self._empty_metrics(ResourceType.GPU)

        try:
            allocated = torch.cuda.memory_allocated() / MB
            total = torch.cuda.get_device_properties(0).total_memory / MB

            return ResourceMetrics(
                resource_type=ResourceType.GPU,
                current_usage=allocated,
                peak_usage=torch.cuda.max_memory_allocated() / MB,
                average_usage=allocated,
                available=total - allocated,
                total=total,
                threshold_warning=self.thresholds[ResourceType.GPU]["warning"],
                threshold_critical=self.thresholds[ResourceType.GPU]["critical"],
                fragmentation=self._calculate_gpu_fragmentation(),
            )
        except Exception as e:
            logger.warning(f"Could not get GPU metrics: {e}")
            return self._empty_metrics(ResourceType.GPU)

    def _get_vram_metrics(self) -> ResourceMetrics:
        """Obtener métricas de VRAM específicas"""
        if not torch.cuda.is_available():
            return self._empty_metrics(ResourceType.VRAM)

        try:
            reserved = torch.cuda.memory_reserved() / MB
            allocated = torch.cuda.memory_allocated() / MB
            total = torch.cuda.get_device_properties(0).total_memory / MB

            return ResourceMetrics(
                resource_type=ResourceType.VRAM,
                current_usage=reserved,
                peak_usage=torch.cuda.max_memory_reserved() / MB,
                average_usage=reserved,
                available=total - reserved,
                total=total,
                threshold_warning=self.thresholds[ResourceType.VRAM]["warning"],
                threshold_critical=self.thresholds[ResourceType.VRAM]["critical"],
                fragmentation=(reserved - allocated) / max(1, reserved),
            )
        except Exception:
            return self._empty_metrics(ResourceType.VRAM)

    def _calculate_gpu_fragmentation(self) -> float:
        """Calcular fragmentación de memoria GPU"""
        if not torch.cuda.is_available():
            return 0.0

        try:
            reserved = torch.cuda.memory_reserved()
            allocated = torch.cuda.memory_allocated()

            if reserved == 0:
                return 0.0

            return (reserved - allocated) / reserved
        except Exception:
            return 0.0

    def _empty_metrics(self, resource_type: ResourceType) -> ResourceMetrics:
        """Crear métricas vacías"""
        return ResourceMetrics(
            resource_type=resource_type,
            current_usage=0.0,
            peak_usage=0.0,
            average_usage=0.0,
            available=0.0,
            total=0.0,
            threshold_warning=75.0,
            threshold_critical=90.0,
        )

    def optimize_resources(self) -> dict[str, Any]:
        """Optimizar recursos del sistema"""
        with self.lock:
            optimizations = {
                "timestamp": datetime.now().isoformat(),
                "actions_taken": [],
                "resources": {},
            }

            # Optimizar memoria
            memory_metrics = self.get_resource_metrics(ResourceType.MEMORY)
            if memory_metrics.is_critical():
                actions = self._optimize_memory_critical()
                optimizations["actions_taken"].extend(actions)
            elif memory_metrics.is_warning():
                actions = self._optimize_memory_warning()
                optimizations["actions_taken"].extend(actions)

            optimizations["resources"]["memory"] = {
                "usage_mb": round(memory_metrics.current_usage, 2),
                "usage_percent": round(memory_metrics.usage_percent(), 2),
                "trend": round(memory_metrics.trend, 4),
                "predicted_exhaustion_seconds": memory_metrics.predicted_exhaustion_seconds,
                "status": "critical" if memory_metrics.is_critical() else "warning" if memory_metrics.is_warning() else "ok",
            }

            # Optimizar CPU
            cpu_metrics = self.get_resource_metrics(ResourceType.CPU)
            optimizations["resources"]["cpu"] = {
                "usage_percent": round(cpu_metrics.current_usage, 2),
                "trend": round(cpu_metrics.trend, 4),
                "status": "critical" if cpu_metrics.is_critical() else "warning" if cpu_metrics.is_warning() else "ok",
            }

            # Optimizar GPU si está disponible
            if torch.cuda.is_available():
                gpu_metrics = self.get_resource_metrics(ResourceType.GPU)
                if gpu_metrics.is_critical():
                    actions = self._optimize_gpu_critical()
                    optimizations["actions_taken"].extend(actions)

                optimizations["resources"]["gpu"] = {
                    "memory_mb": round(gpu_metrics.current_usage, 2),
                    "usage_percent": round(gpu_metrics.usage_percent(), 2),
                    "fragmentation": round(gpu_metrics.fragmentation, 4),
                    "status": "critical" if gpu_metrics.is_critical() else "warning" if gpu_metrics.is_warning() else "ok",
                }

            self.last_optimization = datetime.now()
            return optimizations

    def _optimize_memory_critical(self) -> list[str]:
        """Optimización crítica de memoria"""
        actions = []

        # Garbage collection agresivo
        gc.collect(generation=2)
        actions.append("Executed full garbage collection")

        # Limpiar caches de PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            actions.append("Cleared CUDA cache and synchronized")

        if (
            hasattr(torch.backends, 'mps')
            and torch.backends.mps.is_available()
            and hasattr(torch, 'mps')
            and hasattr(torch.mps, 'empty_cache')
        ):
            torch.mps.empty_cache()
            actions.append("Cleared MPS cache")

        # Compactar arena de memoria (si está disponible)
        if hasattr(gc, 'freeze'):
            gc.freeze()
            gc.collect()
            gc.unfreeze()
            actions.append("Compacted memory arena")

        return actions

    def _optimize_memory_warning(self) -> list[str]:
        """Optimización de advertencia de memoria"""
        actions = []

        # Garbage collection selectivo
        gc.collect(generation=1)
        actions.append("Executed partial garbage collection")

        return actions

    def _optimize_gpu_critical(self) -> list[str]:
        """Optimización crítica de GPU"""
        actions = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            actions.append("Synchronized and cleared GPU cache")

            # Resetear peak memory stats
            torch.cuda.reset_peak_memory_stats()
            actions.append("Reset GPU peak memory stats")

        return actions

    def get_optimization_recommendations(self) -> list[OptimizationRecommendation]:
        """Obtener recomendaciones de optimización"""
        recommendations = []

        # Analizar memoria
        memory_metrics = self.get_resource_metrics(ResourceType.MEMORY)
        if memory_metrics.is_critical():
            recommendations.append(OptimizationRecommendation(
                priority=1,
                category="Memory",
                title="Critical Memory Usage",
                description=f"Memory usage at {memory_metrics.usage_percent():.1f}% "
                           f"(trend: {memory_metrics.trend:+.2f} MB/sample)",
                estimated_improvement="30-50% memory reduction",
                actions=[
                    "Enable aggressive compression",
                    "Reduce batch size",
                    "Clear unused tensors",
                    "Enable lazy loading",
                    "Use memory-mapped tensors for large data",
                ],
                confidence=0.9,
                auto_applicable=True,
                estimated_duration_seconds=5.0,
                risk_level="low",
            ))
        elif memory_metrics.is_warning():
            recommendations.append(OptimizationRecommendation(
                priority=2,
                category="Memory",
                title="High Memory Usage",
                description=f"Memory usage at {memory_metrics.usage_percent():.1f}%",
                estimated_improvement="15-30% memory reduction",
                actions=[
                    "Enable compression",
                    "Optimize cache size",
                    "Use tensor pooling",
                ],
                confidence=0.8,
                auto_applicable=True,
                estimated_duration_seconds=2.0,
            ))

        # Predicción de agotamiento
        if memory_metrics.predicted_exhaustion_seconds:
            if memory_metrics.predicted_exhaustion_seconds < 300:  # < 5 minutos
                recommendations.append(OptimizationRecommendation(
                    priority=1,
                    category="Memory",
                    title="Memory Exhaustion Predicted",
                    description=f"At current rate, memory will be exhausted in "
                               f"{memory_metrics.predicted_exhaustion_seconds:.0f} seconds",
                    estimated_improvement="Prevent OOM crash",
                    actions=[
                        "Immediately reduce memory usage",
                        "Consider restarting with larger memory allocation",
                        "Enable aggressive garbage collection",
                    ],
                    confidence=0.7,
                    risk_level="high",
                ))

        # Analizar GPU
        if torch.cuda.is_available():
            gpu_metrics = self.get_resource_metrics(ResourceType.GPU)
            if gpu_metrics.is_critical():
                recommendations.append(OptimizationRecommendation(
                    priority=1,
                    category="GPU",
                    title="Critical GPU Memory",
                    description=f"GPU memory at {gpu_metrics.usage_percent():.1f}% "
                               f"(fragmentation: {gpu_metrics.fragmentation*100:.1f}%)",
                    estimated_improvement="20-40% GPU memory reduction",
                    actions=[
                        "Use gradient checkpointing",
                        "Reduce model/batch size",
                        "Enable mixed precision (FP16/BF16)",
                        "Clear GPU cache regularly",
                        "Defragment GPU memory",
                    ],
                    confidence=0.85,
                    auto_applicable=True,
                    estimated_duration_seconds=3.0,
                ))

            # Fragmentación alta
            if gpu_metrics.fragmentation > 0.3:
                recommendations.append(OptimizationRecommendation(
                    priority=3,
                    category="GPU",
                    title="High GPU Memory Fragmentation",
                    description=f"GPU memory fragmentation at {gpu_metrics.fragmentation*100:.1f}%",
                    estimated_improvement="10-20% effective GPU memory increase",
                    actions=[
                        "Clear and reallocate GPU tensors",
                        "Use memory pooling",
                        "Consolidate tensor allocations",
                    ],
                    confidence=0.75,
                ))

        # Ordenar por prioridad
        recommendations.sort(key=lambda x: (x.priority, -x.confidence))

        return recommendations

# ============================================================================
# PARALLEL EXECUTOR V3
# ============================================================================

class ParallelExecutor:
    """
    Executor para operaciones paralelas con work-stealing y adaptive scaling.
    """

    def __init__(
        self,
        config: MnemeConfig,
        max_workers: int | None = None,
    ):
        self.config = config
        self.max_workers = max_workers or min(DEFAULT_MAX_WORKERS, (mp.cpu_count() or 1) * 2)

        self.thread_executor: ThreadPoolExecutor | None = None
        self.process_executor: ProcessPoolExecutor | None = None
        self.lock = Lock()

        # Estadísticas
        self._total_tasks = 0
        self._completed_tasks = 0

        # Work-stealing: no implementado todavía (get_stats() lo reporta).
        self.enable_work_stealing = False
        self._stolen_tasks = 0

        logger.info(f"ParallelExecutor initialized with {self.max_workers} workers")

    def _get_executor(self, use_processes: bool) -> ThreadPoolExecutor | ProcessPoolExecutor:
        """Obtener executor apropiado"""
        with self.lock:
            if use_processes:
                if self.process_executor is None:
                    self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
                return self.process_executor
            else:
                if self.thread_executor is None:
                    self.thread_executor = ThreadPoolExecutor(
                        max_workers=self.max_workers,
                        thread_name_prefix="MNEME-Worker"
                    )
                return self.thread_executor

    def execute_parallel(
        self,
        func: Callable[[T], Any],
        items: list[T],
        use_processes: bool = False,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        ordered: bool = False,
    ) -> list[Any]:
        """
        Ejecutar función en paralelo sobre items.

        Args:
            func: Función a ejecutar
            items: Items a procesar
            use_processes: Usar procesos en lugar de threads
            timeout: Timeout por operación
            ordered: Mantener orden de resultados

        Returns:
            Lista de resultados
        """
        if len(items) == 0:
            return []

        if len(items) == 1:
            return [func(items[0])]

        executor = self._get_executor(use_processes)
        self._total_tasks += len(items)

        if ordered:
            futures = [executor.submit(func, item) for item in items]
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                    self._completed_tasks += 1
                except Exception as e:
                    logger.error(f"Parallel execution failed: {e}")
                    results.append(None)
            return results
        else:
            futures = {executor.submit(func, item): i for i, item in enumerate(items)}
            results = [None] * len(items)

            for future in as_completed(futures, timeout=timeout * len(items)):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=timeout)
                    self._completed_tasks += 1
                except Exception as e:
                    logger.error(f"Parallel execution failed for item {idx}: {e}")

            return results

    async def execute_async(
        self,
        func: Callable[[T], Awaitable[Any]],
        items: list[T],
        max_concurrent: int | None = None,
    ) -> list[Any]:
        """
        Ejecutar función async en paralelo con límite de concurrencia.
        """
        max_concurrent = max_concurrent or self.max_workers
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_task(item: T) -> Any:
            async with semaphore:
                return await func(item)

        tasks = [bounded_task(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def map_reduce(
        self,
        map_func: Callable[[T], Any],
        reduce_func: Callable[[list[Any]], Any],
        items: list[T],
        chunk_size: int | None = None,
    ) -> Any:
        """
        Ejecutar map-reduce pattern.

        Args:
            map_func: Función de mapeo
            reduce_func: Función de reducción
            items: Items a procesar
            chunk_size: Tamaño de chunks (auto si None)

        Returns:
            Resultado reducido
        """
        if chunk_size is None:
            chunk_size = max(1, len(items) // self.max_workers)

        # Map phase
        mapped = self.execute_parallel(map_func, items, ordered=True)

        # Reduce phase
        return reduce_func(mapped)

    def pipeline(
        self,
        stages: list[Callable],
        items: list[Any],
    ) -> list[Any]:
        """
        Ejecutar pipeline de transformaciones.

        Args:
            stages: Lista de funciones de transformación
            items: Items iniciales

        Returns:
            Items transformados
        """
        current = items

        for stage in stages:
            current = self.execute_parallel(stage, current, ordered=True)

        return current

    def get_stats(self) -> dict[str, Any]:
        """Obtener estadísticas del executor"""
        return {
            "max_workers": self.max_workers,
            "total_tasks": self._total_tasks,
            "completed_tasks": self._completed_tasks,
            "completion_rate": self._completed_tasks / max(1, self._total_tasks),
            "work_stealing_enabled": self.enable_work_stealing,
            "stolen_tasks": self._stolen_tasks,
            "thread_executor_active": self.thread_executor is not None,
            "process_executor_active": self.process_executor is not None,
        }

    def cleanup(self):
        """Limpiar recursos"""
        with self.lock:
            if self.thread_executor:
                self.thread_executor.shutdown(wait=True, cancel_futures=True)
                self.thread_executor = None

            if self.process_executor:
                self.process_executor.shutdown(wait=True, cancel_futures=True)
                self.process_executor = None

        logger.info("ParallelExecutor V3 cleaned up")

# ============================================================================
# PARALLEL TENSOR PROCESSOR V3
# ============================================================================

class ParallelTensorProcessor:
    """
    Procesador paralelo de operaciones con tensores.

    Incluye batch coalescing, pipeline parallelism y optimizaciones.
    """

    def __init__(
        self,
        config: MnemeConfig,
        max_workers: int | None = None,
        tensor_pool: TensorPool | None = None,
    ):
        self.config = config
        self.max_workers = max_workers or min(8, (mp.cpu_count() or 1) + 4)
        self.tensor_pool = tensor_pool or TensorPool()

        self.executor: ThreadPoolExecutor | None = None
        self.compressor = TensorCompressor()
        self.decomposer = TensorDecomposer()
        self.quantizer = TensorQuantizer()
        self.sparsifier = TensorSparsifier()

        self.lock = Lock()

        logger.info(f"ParallelTensorProcessor initialized with {self.max_workers} workers")

    def _get_executor(self) -> ThreadPoolExecutor:
        """Obtener executor, crear si no existe"""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="MNEME-Tensor"
            )
        return self.executor

    def parallel_decomposition(
        self,
        tensors: list[torch.Tensor],
        method: DecompositionMethod = DecompositionMethod.CP,
        rank: int = 10,
    ) -> list[DecompositionResult]:
        """Descomposición paralela de tensores"""
        if len(tensors) <= 1:
            return [self.decomposer.decompose(t, rank, method) for t in tensors]

        executor = self._get_executor()
        futures = []

        for tensor in tensors:
            future = executor.submit(self.decomposer.decompose, tensor, rank, method)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=DEFAULT_TIMEOUT_SECONDS)
                results.append(result)
            except Exception as e:
                logger.error(f"Tensor decomposition failed: {e}")
                results.append(None)

        return results

    def parallel_compression(
        self,
        tensors: list[torch.Tensor],
        algorithm: CompressionAlgorithm | None = None,
    ) -> list[CompressionResult]:
        """Compresión paralela de tensores"""
        executor = self._get_executor()
        futures = []

        for tensor in tensors:
            future = executor.submit(self.compressor.compress, tensor, algorithm)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=DEFAULT_TIMEOUT_SECONDS)
                results.append(result)
            except Exception as e:
                logger.error(f"Tensor compression failed: {e}")
                results.append(None)

        return results

    def parallel_quantization(
        self,
        tensors: list[torch.Tensor],
        quant_type: QuantizationType = QuantizationType.DYNAMIC,
    ) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        """Cuantización paralela de tensores"""
        executor = self._get_executor()
        future_to_idx = {}

        for idx, tensor in enumerate(tensors):
            future = executor.submit(self.quantizer.quantize, tensor, quant_type)
            future_to_idx[future] = idx

        results = [None] * len(tensors)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result(timeout=DEFAULT_TIMEOUT_SECONDS)
                results[idx] = result
            except Exception as e:
                logger.error(f"Tensor quantization failed: {e}")
                results[idx] = (tensors[idx], {})

        return results

    def batch_process(
        self,
        tensors: list[torch.Tensor],
        operation: Callable[[torch.Tensor], torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Procesar tensores en batch con coalescing.

        Agrupa operaciones pequeñas para mejor eficiencia.
        """
        # Separar por tamaño
        small_tensors = []
        large_tensors = []
        small_indices = []
        large_indices = []

        for i, tensor in enumerate(tensors):
            if tensor.numel() < SMALL_TENSOR_THRESHOLD:
                small_tensors.append(tensor)
                small_indices.append(i)
            else:
                large_tensors.append(tensor)
                large_indices.append(i)

        results = [None] * len(tensors)

        # Procesar tensores pequeños en batch
        if small_tensors:
            # Intentar apilar si tienen la misma forma
            try:
                if all(t.shape == small_tensors[0].shape for t in small_tensors):
                    stacked = torch.stack(small_tensors)
                    processed = operation(stacked)
                    for i, idx in enumerate(small_indices):
                        results[idx] = processed[i]
                else:
                    raise ValueError("Mixed shapes")
            except Exception:
                # Fallback a procesamiento individual
                for tensor, idx in zip(small_tensors, small_indices, strict=False):
                    results[idx] = operation(tensor)

        # Procesar tensores grandes en paralelo
        if large_tensors:
            executor = self._get_executor()
            futures = {
                executor.submit(operation, t): idx
                for t, idx in zip(large_tensors, large_indices, strict=False)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=DEFAULT_TIMEOUT_SECONDS)
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    results[idx] = tensors[idx]

        return results

    def pipeline_transform(
        self,
        tensors: list[torch.Tensor],
        transforms: list[Callable[[torch.Tensor], torch.Tensor]],
    ) -> list[torch.Tensor]:
        """
        Aplicar pipeline de transformaciones con paralelismo.
        """
        current = tensors

        for transform in transforms:
            current = self.batch_process(current, transform)

        return current

    def optimize_tensor(
        self,
        tensor: torch.Tensor,
        optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
    ) -> torch.Tensor:
        """
        Optimizar un tensor según el nivel especificado.
        """
        if optimization_level == OptimizationLevel.NONE:
            return tensor

        # Básico: contiguidad
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if optimization_level == OptimizationLevel.BASIC:
            return tensor

        # Agresivo: pin memory para CPU tensors
        if optimization_level >= OptimizationLevel.AGGRESSIVE:
            if torch.cuda.is_available() and tensor.device.type == 'cpu':
                try:
                    if not tensor.is_pinned():
                        tensor = tensor.pin_memory()
                except Exception as e:
                    logger.debug(f"Could not pin memory: {e}")

        # Máximo: considerar cuantización para tensores grandes
        if optimization_level >= OptimizationLevel.MAXIMUM:
            if tensor.numel() > LARGE_TENSOR_THRESHOLD:
                quantized, metadata = self.quantizer.quantize(tensor, QuantizationType.FP16)
                return quantized

        # Extremo: sparsificación si aplica
        if optimization_level == OptimizationLevel.EXTREME:
            if self.sparsifier.should_sparsify(tensor):
                sparse, _ = self.sparsifier.sparsify(tensor)
                return sparse

        return tensor

    def cleanup(self):
        """Limpiar recursos"""
        with self.lock:
            if self.executor:
                self.executor.shutdown(wait=True, cancel_futures=True)
                self.executor = None

        self.tensor_pool.clear()
        logger.info("ParallelTensorProcessor V3 cleaned up")

# ============================================================================
# CHECKPOINTING Y RECOVERY
# ============================================================================

class CheckpointManager:
    """
    Gestor de checkpoints para recovery y persistencia.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        max_checkpoints: int = 5,
        auto_checkpoint_interval: float = DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
    ):
        self.checkpoint_dir = checkpoint_dir or Path(tempfile.gettempdir()) / "mneme_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.auto_checkpoint_interval = auto_checkpoint_interval

        self._checkpoints: OrderedDict[str, Path] = OrderedDict()
        self._last_checkpoint_time: float | None = None
        self._lock = Lock()

        # Cargar checkpoints existentes
        self._load_existing_checkpoints()

        logger.info(f"CheckpointManager initialized at {self.checkpoint_dir}")

    def _load_existing_checkpoints(self) -> None:
        """Cargar checkpoints existentes del directorio"""
        if not self.checkpoint_dir.exists():
            return

        for path in sorted(self.checkpoint_dir.glob("checkpoint_*.pkl")):
            checkpoint_id = path.stem.replace("checkpoint_", "")
            self._checkpoints[checkpoint_id] = path

        # Mantener solo los más recientes
        while len(self._checkpoints) > self.max_checkpoints:
            _, path = self._checkpoints.popitem(last=False)
            path.unlink(missing_ok=True)

    def create_checkpoint(
        self,
        optimizer_state: dict[str, Any],
        metrics: PerformanceMetrics,
        resource_state: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Crear un nuevo checkpoint.

        Returns:
            ID del checkpoint creado
        """
        with self._lock:
            checkpoint_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            checkpoint = CheckpointData(
                checkpoint_id=checkpoint_id,
                created_at=datetime.now(),
                optimizer_state=optimizer_state,
                metrics_snapshot=metrics,
                resource_state=resource_state,
                metadata=metadata or {},
            )

            # Guardar
            path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
            with open(path, 'wb') as f:
                f.write(checkpoint.to_bytes())

            self._checkpoints[checkpoint_id] = path
            self._last_checkpoint_time = time.time()

            # Limpiar checkpoints antiguos
            while len(self._checkpoints) > self.max_checkpoints:
                old_id, old_path = self._checkpoints.popitem(last=False)
                old_path.unlink(missing_ok=True)
                logger.debug(f"Removed old checkpoint: {old_id}")

            logger.info(f"Created checkpoint: {checkpoint_id}")
            return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str | None = None) -> CheckpointData | None:
        """
        Cargar un checkpoint.

        Args:
            checkpoint_id: ID específico o None para el más reciente

        Returns:
            CheckpointData o None si no existe
        """
        with self._lock:
            if checkpoint_id is None:
                if not self._checkpoints:
                    return None
                checkpoint_id = list(self._checkpoints.keys())[-1]

            path = self._checkpoints.get(checkpoint_id)
            if path is None or not path.exists():
                return None

            try:
                with open(path, 'rb') as f:
                    return CheckpointData.from_bytes(f.read())
            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
                return None

    def should_checkpoint(self) -> bool:
        """Verificar si es tiempo de crear checkpoint automático"""
        if self._last_checkpoint_time is None:
            return True
        return time.time() - self._last_checkpoint_time >= self.auto_checkpoint_interval

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Listar checkpoints disponibles"""
        with self._lock:
            result = []
            for checkpoint_id, path in self._checkpoints.items():
                result.append({
                    "id": checkpoint_id,
                    "path": str(path),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                    "exists": path.exists(),
                })
            return result

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Eliminar un checkpoint"""
        with self._lock:
            path = self._checkpoints.pop(checkpoint_id, None)
            if path:
                path.unlink(missing_ok=True)
                return True
            return False

    def cleanup(self) -> None:
        """Limpiar todos los checkpoints"""
        with self._lock:
            for path in self._checkpoints.values():
                path.unlink(missing_ok=True)
            self._checkpoints.clear()
        logger.info("CheckpointManager cleaned up")

# ============================================================================
# MNEME OPTIMIZER V3 - CLASE PRINCIPAL
# ============================================================================

class MNEMEOptimizer:
    """
    Optimizador principal de MNEME V3 con todas las capacidades integradas.

    Características principales:
    - Circuit breaker pattern para resiliencia
    - Backpressure adaptativo
    - Checkpointing y recovery
    - Compresión multi-algoritmo
    - Descomposición tensorial (CP/Tucker/TT)
    - Cuantización automática
    - Sparsificación
    - Tensor pooling
    - Pipeline parallelism
    - Métricas con percentiles
    """

    def __init__(
        self,
        config: MnemeConfig | None = None,
        optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
        enable_profiling: bool = True,
        enable_parallel_processing: bool = True,
        enable_auto_optimization: bool = False,
        enable_checkpointing: bool = False,
        checkpoint_dir: Path | None = None,
    ):
        self.config = config or MnemeConfig()
        self.optimization_level = optimization_level
        self.enable_profiling = enable_profiling
        self.enable_parallel_processing = enable_parallel_processing
        self.enable_auto_optimization = enable_auto_optimization
        self.enable_checkpointing = enable_checkpointing

        # Inicializar componentes
        self.tensor_pool = TensorPool()
        self.performance_monitor = PerformanceMonitor(self.config)
        self.resource_optimizer = ResourceOptimizer(self.config)
        self.parallel_executor = ParallelExecutor(self.config)
        self.tensor_processor = ParallelTensorProcessor(
            self.config,
            tensor_pool=self.tensor_pool
        )

        # Checkpointing
        self.checkpoint_manager: CheckpointManager | None = None
        if enable_checkpointing:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        # Compresión y descomposición
        self.compressor = TensorCompressor()
        self.decomposer = TensorDecomposer()
        self.quantizer = TensorQuantizer()
        self.sparsifier = TensorSparsifier()

        # Circuit breaker global
        self.global_circuit_breaker = CircuitBreaker(
            "global",
            failure_threshold=10,
            reset_timeout=120.0
        )

        # Backpressure
        self.backpressure = AdaptiveBackpressure()

        # Configurar nivel de optimización
        self._configure_optimization_level()

        # Iniciar monitoreo si está habilitado
        if enable_profiling:
            self.performance_monitor.start_monitoring(interval=2.0)

        # Thread para auto-optimización
        self.auto_optimization_thread: threading.Thread | None = None
        self._stop_event = Event()
        if enable_auto_optimization:
            self._start_auto_optimization()

        logger.info(f"MNEMEOptimizer V3 initialized with level: {optimization_level.name}")

    def _configure_optimization_level(self):
        """Configurar parámetros según nivel de optimización"""
        if self.optimization_level == OptimizationLevel.NONE:
            return

        elif self.optimization_level == OptimizationLevel.BASIC:
            self.config.enable_compression = True
            self.config.compression_level = CompressionLevel.FAST

        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            self.config.enable_compression = True
            self.config.compression_level = CompressionLevel.HIGH
            self.config.memory_pressure_threshold = 0.7
            self.config.enable_adaptive_compression = True

        elif self.optimization_level == OptimizationLevel.MAXIMUM:
            self.config.enable_compression = True
            self.config.compression_level = CompressionLevel.MAXIMUM
            self.config.memory_pressure_threshold = 0.6
            self.config.enable_adaptive_compression = True
            self.config.lazy_loading = True

        elif self.optimization_level == OptimizationLevel.ADAPTIVE:
            self.config.enable_compression = True
            self.config.compression_level = CompressionLevel.BALANCED
            self.config.enable_adaptive_compression = True
            self.config.memory_pressure_threshold = 0.75

        elif self.optimization_level == OptimizationLevel.EXTREME:
            self.config.enable_compression = True
            self.config.compression_level = CompressionLevel.MAXIMUM
            self.config.memory_pressure_threshold = 0.5
            self.config.enable_adaptive_compression = True
            self.config.lazy_loading = True
            # Habilitar todas las optimizaciones

    def _start_auto_optimization(self):
        """Iniciar thread de auto-optimización"""
        def auto_optimize_loop():
            while not self._stop_event.wait(timeout=30):
                try:
                    health = self.get_health_status()

                    if health in [HealthStatus.WARNING.value, HealthStatus.CRITICAL.value]:
                        logger.info(f"Auto-optimization triggered (health: {health})")
                        self.optimize_system()

                    # Auto-checkpoint si está habilitado
                    if self.checkpoint_manager and self.checkpoint_manager.should_checkpoint():
                        self._create_auto_checkpoint()

                except Exception as e:
                    logger.error(f"Error in auto-optimization: {e}")

        self.auto_optimization_thread = threading.Thread(
            target=auto_optimize_loop,
            daemon=True,
            name="MNEME-AutoOptimizer"
        )
        self.auto_optimization_thread.start()
        logger.info("Auto-optimization thread started")

    def _create_auto_checkpoint(self):
        """Crear checkpoint automático"""
        if not self.checkpoint_manager:
            return

        try:
            self.checkpoint_manager.create_checkpoint(
                optimizer_state=self._get_state_dict(),
                metrics=self.performance_monitor.current_metrics,
                resource_state=self.resource_optimizer.optimize_resources(),
                metadata={"auto": True, "level": self.optimization_level.name}
            )
        except Exception as e:
            logger.error(f"Failed to create auto-checkpoint: {e}")

    def _get_state_dict(self) -> dict[str, Any]:
        """Obtener estado del optimizador"""
        return {
            "optimization_level": self.optimization_level.value,
            "enable_profiling": self.enable_profiling,
            "enable_parallel_processing": self.enable_parallel_processing,
            "tensor_pool_stats": self.tensor_pool.get_stats(),
            "circuit_breaker_state": self.global_circuit_breaker.get_stats(),
        }

    def optimize_tensor_operations(
        self,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Optimizar operaciones con tensores"""
        if not tensors:
            return []

        # Verificar backpressure
        if not self.backpressure.acquire(blocking=True, timeout=5.0):
            logger.warning("Backpressure: request delayed")

        with self.performance_monitor.measure_operation("tensor_optimization"):
            with self.global_circuit_breaker.protect():
                if self.enable_parallel_processing and len(tensors) > 1:
                    return self._optimize_tensors_parallel(tensors)
                else:
                    return self._optimize_tensors_sequential(tensors)

    def _optimize_tensors_parallel(
        self,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Optimización paralela de tensores"""
        return self.tensor_processor.batch_process(
            tensors,
            lambda t: self.tensor_processor.optimize_tensor(t, self.optimization_level)
        )

    def _optimize_tensors_sequential(
        self,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Optimización secuencial de tensores"""
        return [
            self.tensor_processor.optimize_tensor(t, self.optimization_level)
            for t in tensors
        ]

    def compress_tensors(
        self,
        tensors: list[torch.Tensor],
        algorithm: CompressionAlgorithm | None = None,
    ) -> list[CompressionResult]:
        """Comprimir lista de tensores"""
        with self.performance_monitor.measure_operation("tensor_compression"):
            if self.enable_parallel_processing and len(tensors) > 1:
                return self.tensor_processor.parallel_compression(tensors, algorithm)
            else:
                return [self.compressor.compress(t, algorithm) for t in tensors]

    def decompose_tensors(
        self,
        tensors: list[torch.Tensor],
        rank: int = 10,
        method: DecompositionMethod = DecompositionMethod.CP,
    ) -> list[DecompositionResult]:
        """Descomponer lista de tensores"""
        with self.performance_monitor.measure_operation("tensor_decomposition"):
            if self.enable_parallel_processing and len(tensors) > 1:
                return self.tensor_processor.parallel_decomposition(tensors, method, rank)
            else:
                return [self.decomposer.decompose(t, rank, method) for t in tensors]

    def quantize_tensors(
        self,
        tensors: list[torch.Tensor],
        quant_type: QuantizationType = QuantizationType.DYNAMIC,
    ) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        """Cuantizar lista de tensors"""
        with self.performance_monitor.measure_operation("tensor_quantization"):
            if self.enable_parallel_processing and len(tensors) > 1:
                return self.tensor_processor.parallel_quantization(tensors, quant_type)
            else:
                return [self.quantizer.quantize(t, quant_type) for t in tensors]

    def optimize_model(
        self,
        model: nn.Module,
        use_mixed_precision: bool = True,
        use_gradient_checkpointing: bool = False,
    ) -> nn.Module:
        """
        Optimizar modelo PyTorch completo.

        Args:
            model: Modelo a optimizar
            use_mixed_precision: Habilitar mixed precision
            use_gradient_checkpointing: Habilitar gradient checkpointing

        Returns:
            Modelo optimizado
        """
        with self.performance_monitor.measure_operation("model_optimization"):
            # Optimizar parámetros
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.data = self.tensor_processor.optimize_tensor(
                            param.data,
                            self.optimization_level
                        )

            # Mixed precision
            if use_mixed_precision and torch.cuda.is_available():
                try:
                    model = model.half()
                    logger.info("Enabled FP16 for model")
                except Exception as e:
                    logger.warning(f"Could not enable mixed precision: {e}")

            # Gradient checkpointing para módulos secuenciales
            if use_gradient_checkpointing:
                self._enable_gradient_checkpointing(model)

            return model

    def _enable_gradient_checkpointing(self, model: nn.Module) -> None:
        """Habilitar gradient checkpointing en el modelo"""
        for name, module in model.named_modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
                logger.debug(f"Enabled gradient checkpointing for {name}")

    def optimize_system(self) -> dict[str, Any]:
        """Optimizar sistema completo"""
        with self.performance_monitor.measure_operation("system_optimization"):
            result = self.resource_optimizer.optimize_resources()

            # También limpiar tensor pool si está muy lleno
            pool_stats = self.tensor_pool.get_stats()
            if pool_stats['utilization'] > 0.9:
                self.tensor_pool.clear()
                result['actions_taken'].append("Cleared tensor pool (>90% utilization)")

            return result

    def get_optimization_report(self) -> dict[str, Any]:
        """Obtener reporte completo de optimización"""
        return {
            "version": __version__,
            "timestamp": datetime.now().isoformat(),
            "optimization_level": self.optimization_level.name,
            "performance": self.performance_monitor.get_performance_report(),
            "resources": self.resource_optimizer.optimize_resources(),
            "recommendations": [
                asdict(rec) for rec in self.resource_optimizer.get_optimization_recommendations()
            ],
            "health_status": self.get_health_status(),
            "tensor_pool": self.tensor_pool.get_stats(),
            "circuit_breaker": self.global_circuit_breaker.get_stats(),
            "backpressure": self.backpressure.get_stats(),
            "parallel_executor": self.parallel_executor.get_stats(),
            "checkpoints": self.checkpoint_manager.list_checkpoints() if self.checkpoint_manager else [],
        }

    def get_health_status(self) -> str:
        """Obtener estado de salud"""
        # Verificar circuit breaker global
        if self.global_circuit_breaker.state == CircuitState.OPEN:
            return HealthStatus.CRITICAL.value

        if self.global_circuit_breaker.state == CircuitState.HALF_OPEN:
            return HealthStatus.RECOVERING.value

        return self.performance_monitor.get_health_status()

    def create_checkpoint(self, metadata: dict[str, Any] | None = None) -> str | None:
        """Crear checkpoint manual"""
        if not self.checkpoint_manager:
            logger.warning("Checkpointing not enabled")
            return None

        return self.checkpoint_manager.create_checkpoint(
            optimizer_state=self._get_state_dict(),
            metrics=self.performance_monitor.current_metrics,
            resource_state=self.resource_optimizer.optimize_resources(),
            metadata=metadata or {"manual": True}
        )

    def restore_from_checkpoint(self, checkpoint_id: str | None = None) -> bool:
        """Restaurar desde checkpoint"""
        if not self.checkpoint_manager:
            logger.warning("Checkpointing not enabled")
            return False

        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        if checkpoint is None:
            logger.error("No checkpoint found")
            return False

        try:
            # Restaurar estado
            state = checkpoint.optimizer_state
            self.optimization_level = OptimizationLevel(state.get('optimization_level', 1))
            self._configure_optimization_level()

            logger.info(f"Restored from checkpoint: {checkpoint.checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False

    def benchmark(
        self,
        tensors: list[torch.Tensor],
        iterations: int = 10,
    ) -> dict[str, Any]:
        """
        Ejecutar benchmark de optimización.

        Args:
            tensors: Tensores para benchmark
            iterations: Número de iteraciones

        Returns:
            Resultados del benchmark
        """
        results = {
            "compression": {},
            "decomposition": {},
            "quantization": {},
            "optimization": {},
        }

        # Benchmark compression
        comp_times = []
        for _ in range(iterations):
            start = time.time()
            self.compress_tensors(tensors)
            comp_times.append(time.time() - start)

        results["compression"] = {
            "avg_time_ms": np.mean(comp_times) * 1000,
            "std_time_ms": np.std(comp_times) * 1000,
            "min_time_ms": np.min(comp_times) * 1000,
            "max_time_ms": np.max(comp_times) * 1000,
        }

        # Benchmark optimization
        opt_times = []
        for _ in range(iterations):
            start = time.time()
            self.optimize_tensor_operations(tensors)
            opt_times.append(time.time() - start)

        results["optimization"] = {
            "avg_time_ms": np.mean(opt_times) * 1000,
            "std_time_ms": np.std(opt_times) * 1000,
            "min_time_ms": np.min(opt_times) * 1000,
            "max_time_ms": np.max(opt_times) * 1000,
        }

        return results

    def cleanup(self):
        """Limpiar todos los recursos"""
        logger.info("Cleaning up MNEMEOptimizer V3...")

        # Detener auto-optimización
        self.enable_auto_optimization = False
        self._stop_event.set()
        if self.auto_optimization_thread:
            self.auto_optimization_thread.join(timeout=2.0)

        # Limpiar componentes
        self.performance_monitor.cleanup()
        self.parallel_executor.cleanup()
        self.tensor_processor.cleanup()
        self.tensor_pool.clear()

        if self.checkpoint_manager:
            # No limpiar checkpoints, solo el manager
            pass

        logger.info("MNEMEOptimizer V3 cleanup completed")

    def __enter__(self) -> MNEMEOptimizer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def create_optimizer(
    config: MnemeConfig | None = None,
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
    enable_profiling: bool = True,
    enable_parallel: bool = True,
    enable_checkpointing: bool = False,
) -> MNEMEOptimizer:
    """Crear optimizador con configuración específica"""
    return MNEMEOptimizer(
        config=config,
        optimization_level=optimization_level,
        enable_profiling=enable_profiling,
        enable_parallel_processing=enable_parallel,
        enable_checkpointing=enable_checkpointing,
    )

def optimize_model(
    model: nn.Module,
    config: MnemeConfig | None = None,
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
    use_mixed_precision: bool = True,
) -> nn.Module:
    """Optimizar modelo PyTorch usando MNEME"""
    with MNEMEOptimizer(config, optimization_level) as optimizer:
        return optimizer.optimize_model(model, use_mixed_precision=use_mixed_precision)

def get_system_metrics() -> dict[str, Any]:
    """Obtener métricas del sistema"""
    config = MnemeConfig()
    monitor = PerformanceMonitor(config)

    try:
        monitor.update_metrics()
        return monitor.get_performance_report()
    finally:
        monitor.cleanup()

def benchmark_optimization(
    tensors: list[torch.Tensor],
    optimization_levels: list[OptimizationLevel] | None = None,
) -> dict[str, Any]:
    """Benchmark de diferentes niveles de optimización"""
    if optimization_levels is None:
        optimization_levels = list(OptimizationLevel)

    results = {}

    for level in optimization_levels:
        with MNEMEOptimizer(optimization_level=level, enable_profiling=True) as optimizer:
            try:
                results[level.name] = optimizer.benchmark(tensors)
                results[level.name]["success"] = True
            except Exception as e:
                results[level.name] = {
                    "error": str(e),
                    "success": False,
                }

    return results

def benchmark_compression(
    tensor: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Benchmark de algoritmos de compresión para un tensor"""
    compressor = TensorCompressor()
    return compressor.benchmark_algorithms(tensor)

# ============================================================================
# EXPORTACIONES
# ============================================================================

__all__ = [
    # Versión
    '__version__',

    # Clases principales
    'MNEMEOptimizer',
    'PerformanceMonitor',
    'ResourceOptimizer',
    'ParallelTensorProcessor',
    'ParallelExecutor',
    'CheckpointManager',

    # Componentes de optimización
    'TensorPool',
    'TensorCompressor',
    'TensorDecomposer',
    'TensorQuantizer',
    'TensorSparsifier',

    # Resiliencia
    'CircuitBreaker',
    'AdaptiveBackpressure',
    'TokenBucket',

    # Métricas
    'LatencyHistogram',

    # Enums
    'OptimizationLevel',
    'ResourceType',
    'HealthStatus',
    'CircuitState',
    'CompressionAlgorithm',
    'DecompositionMethod',
    'QuantizationType',
    'SparsityFormat',

    # Dataclasses
    'PerformanceMetrics',
    'ResourceMetrics',
    'OptimizationRecommendation',
    'TensorMetadata',
    'CheckpointData',
    'CompressionResult',
    'DecompositionResult',

    # Funciones de utilidad
    'create_optimizer',
    'optimize_model',
    'get_system_metrics',
    'benchmark_optimization',
    'benchmark_compression',
]
