"""
MNEME Optimization Module - Versión Completa y Actualizada
Módulo de optimización avanzado con monitoreo de rendimiento, 
optimización de recursos y procesamiento paralelo

Versión: 2.0.0
Autor: MNEME Development Team
Licencia: BSL 1.1
"""

import asyncio
import gc
import logging
import multiprocessing as mp
import psutil
import queue
import threading
import time
import warnings
import weakref
from collections import deque, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, as_completed
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps, lru_cache
from pathlib import Path
from threading import Lock, RLock, Event
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
import tensorly as tl

# Importar desde el core de MNEME
from .mneme_core import (
    ZSpace, MnemeConfig, DecompType, CompressionLevel,
    MnemeError, ValidationError, PerformanceError
)

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS Y CONFIGURACIONES
# ============================================================================

class OptimizationLevel(Enum):
    """Niveles de optimización disponibles"""
    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    MAXIMUM = 3
    ADAPTIVE = 4

class ResourceType(Enum):
    """Tipos de recursos del sistema"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"

class OptimizationStrategy(Enum):
    """Estrategias de optimización"""
    MEMORY_FIRST = "memory_first"
    SPEED_FIRST = "speed_first"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"

class HealthStatus(Enum):
    """Estados de salud del sistema"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"

# ============================================================================
# DATACLASSES PARA MÉTRICAS
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del sistema"""
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    cache_hit_rate: float = 0.0
    compression_ratio: float = 1.0
    avg_operation_time_ms: float = 0.0
    total_operations: int = 0
    failed_operations: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def success_rate(self) -> float:
        """Calcular tasa de éxito"""
        if self.total_operations == 0:
            return 1.0
        return (self.total_operations - self.failed_operations) / self.total_operations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "memory_usage_mb": self.memory_usage_mb,
            "memory_usage_percent": self.memory_usage_percent,
            "cpu_usage_percent": self.cpu_usage_percent,
            "gpu_usage_percent": self.gpu_usage_percent,
            "gpu_memory_mb": self.gpu_memory_mb,
            "cache_hit_rate": self.cache_hit_rate,
            "compression_ratio": self.compression_ratio,
            "avg_operation_time_ms": self.avg_operation_time_ms,
            "total_operations": self.total_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.success_rate(),
            "timestamp": self.timestamp.isoformat()
        }

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
    
    def is_warning(self) -> bool:
        """Verificar si está en nivel de advertencia"""
        return self.current_usage >= self.threshold_warning
    
    def is_critical(self) -> bool:
        """Verificar si está en nivel crítico"""
        return self.current_usage >= self.threshold_critical
    
    def usage_percent(self) -> float:
        """Calcular porcentaje de uso"""
        if self.total == 0:
            return 0.0
        return (self.current_usage / self.total) * 100

@dataclass
class OptimizationRecommendation:
    """Recomendación de optimización"""
    priority: int  # 1=crítico, 2=alto, 3=medio, 4=bajo
    category: str
    title: str
    description: str
    estimated_improvement: str
    actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

# ============================================================================
# MONITOR DE RENDIMIENTO
# ============================================================================

class PerformanceMonitor:
    """Monitor de rendimiento del sistema con métricas detalladas"""
    
    def __init__(self, config: MnemeConfig, history_size: int = 100):
        self.config = config
        self.history_size = history_size
        self.lock = Lock()
        
        # Historial de métricas
        self.metrics_history: deque = deque(maxlen=history_size)
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.operation_failures: Dict[str, int] = defaultdict(int)
        
        # Estado actual
        self.current_metrics = PerformanceMetrics()
        self.start_time = time.time()
        
        # Thread para monitoreo continuo
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        logger.info("PerformanceMonitor initialized")
    
    def start_monitoring(self, interval: float = 1.0):
        """Iniciar monitoreo continuo"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started continuous monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Detener monitoreo continuo"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Loop de monitoreo continuo"""
        while self.monitoring_active:
            try:
                self.update_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def update_metrics(self):
        """Actualizar métricas del sistema"""
        with self.lock:
            # Métricas de memoria
            process = psutil.Process()
            memory_info = process.memory_info()
            self.current_metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)
            
            virtual_memory = psutil.virtual_memory()
            self.current_metrics.memory_usage_percent = virtual_memory.percent
            
            # Métricas de CPU
            self.current_metrics.cpu_usage_percent = process.cpu_percent(interval=0.1)
            
            # Métricas de GPU si está disponible
            if torch.cuda.is_available():
                try:
                    self.current_metrics.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    self.current_metrics.gpu_usage_percent = torch.cuda.utilization()
                except Exception as e:
                    logger.debug(f"Could not get GPU metrics: {e}")
            
            # Calcular métricas agregadas
            if self.operation_times:
                all_times = []
                for times in self.operation_times.values():
                    all_times.extend(times)
                if all_times:
                    self.current_metrics.avg_operation_time_ms = np.mean(all_times) * 1000
            
            self.current_metrics.total_operations = sum(self.operation_counts.values())
            self.current_metrics.failed_operations = sum(self.operation_failures.values())
            
            # Agregar al historial
            self.metrics_history.append(self.current_metrics)
    
    def record_operation(self, operation_name: str, duration: float, success: bool = True):
        """Registrar una operación"""
        with self.lock:
            self.operation_times[operation_name].append(duration)
            self.operation_counts[operation_name] += 1
            
            if not success:
                self.operation_failures[operation_name] += 1
            
            # Limitar historial de tiempos
            if len(self.operation_times[operation_name]) > 1000:
                self.operation_times[operation_name] = self.operation_times[operation_name][-1000:]
    
    @contextmanager
    def measure_operation(self, operation_name: str):
        """Context manager para medir operaciones"""
        start_time = time.time()
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            self.record_operation(operation_name, duration, success)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Obtener reporte de rendimiento completo"""
        with self.lock:
            # Actualizar métricas actuales
            self.update_metrics()
            
            # Calcular estadísticas
            uptime = time.time() - self.start_time
            
            # Top operaciones más lentas
            slow_operations = {}
            for op_name, times in self.operation_times.items():
                if times:
                    avg_time = np.mean(times) * 1000
                    slow_operations[op_name] = {
                        "avg_time_ms": avg_time,
                        "count": self.operation_counts[op_name],
                        "failures": self.operation_failures[op_name]
                    }
            
            sorted_ops = sorted(
                slow_operations.items(),
                key=lambda x: x[1]["avg_time_ms"],
                reverse=True
            )[:10]
            
            return {
                "current_metrics": self.current_metrics.to_dict(),
                "uptime_seconds": uptime,
                "operations": {
                    "total": sum(self.operation_counts.values()),
                    "failed": sum(self.operation_failures.values()),
                    "success_rate": self.current_metrics.success_rate(),
                    "unique_operations": len(self.operation_counts)
                },
                "slowest_operations": dict(sorted_ops),
                "history_size": len(self.metrics_history)
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
        
        if memory_critical or cpu_critical or failure_critical:
            return HealthStatus.CRITICAL.value
        elif memory_warning or cpu_warning or failure_warning:
            return HealthStatus.WARNING.value
        else:
            return HealthStatus.HEALTHY.value
    
    def cleanup(self):
        """Limpiar recursos"""
        self.stop_monitoring()
        with self.lock:
            self.metrics_history.clear()
            self.operation_times.clear()
            self.operation_counts.clear()
            self.operation_failures.clear()
        logger.info("PerformanceMonitor cleaned up")

# ============================================================================
# OPTIMIZADOR DE RECURSOS
# ============================================================================

class ResourceOptimizer:
    """Optimizador de recursos del sistema"""
    
    def __init__(self, config: MnemeConfig):
        self.config = config
        self.lock = Lock()
        
        # Umbrales de recursos
        self.thresholds = {
            ResourceType.MEMORY: {"warning": 75.0, "critical": 90.0},
            ResourceType.CPU: {"warning": 80.0, "critical": 95.0},
            ResourceType.GPU: {"warning": 85.0, "critical": 95.0}
        }
        
        # Cache de métricas
        self.resource_cache: Dict[ResourceType, ResourceMetrics] = {}
        self.last_optimization = None
        
        logger.info("ResourceOptimizer initialized")
    
    def get_resource_metrics(self, resource_type: ResourceType) -> ResourceMetrics:
        """Obtener métricas de un recurso específico"""
        if resource_type == ResourceType.MEMORY:
            return self._get_memory_metrics()
        elif resource_type == ResourceType.CPU:
            return self._get_cpu_metrics()
        elif resource_type == ResourceType.GPU:
            return self._get_gpu_metrics()
        else:
            raise ValueError(f"Unsupported resource type: {resource_type}")
    
    def _get_memory_metrics(self) -> ResourceMetrics:
        """Obtener métricas de memoria"""
        vm = psutil.virtual_memory()
        process = psutil.Process()
        process_mem = process.memory_info().rss
        
        return ResourceMetrics(
            resource_type=ResourceType.MEMORY,
            current_usage=process_mem / (1024 * 1024),  # MB
            peak_usage=process.memory_info().rss / (1024 * 1024),
            average_usage=vm.used / (1024 * 1024),
            available=vm.available / (1024 * 1024),
            total=vm.total / (1024 * 1024),
            threshold_warning=self.thresholds[ResourceType.MEMORY]["warning"],
            threshold_critical=self.thresholds[ResourceType.MEMORY]["critical"]
        )
    
    def _get_cpu_metrics(self) -> ResourceMetrics:
        """Obtener métricas de CPU"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        return ResourceMetrics(
            resource_type=ResourceType.CPU,
            current_usage=cpu_percent,
            peak_usage=100.0,
            average_usage=cpu_percent,
            available=100.0 - cpu_percent,
            total=100.0,
            threshold_warning=self.thresholds[ResourceType.CPU]["warning"],
            threshold_critical=self.thresholds[ResourceType.CPU]["critical"]
        )
    
    def _get_gpu_metrics(self) -> ResourceMetrics:
        """Obtener métricas de GPU"""
        if not torch.cuda.is_available():
            return ResourceMetrics(
                resource_type=ResourceType.GPU,
                current_usage=0.0,
                peak_usage=0.0,
                average_usage=0.0,
                available=0.0,
                total=0.0,
                threshold_warning=85.0,
                threshold_critical=95.0
            )
        
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            
            return ResourceMetrics(
                resource_type=ResourceType.GPU,
                current_usage=gpu_memory_allocated,
                peak_usage=gpu_memory_reserved,
                average_usage=gpu_memory_allocated,
                available=gpu_memory_total - gpu_memory_allocated,
                total=gpu_memory_total,
                threshold_warning=self.thresholds[ResourceType.GPU]["warning"],
                threshold_critical=self.thresholds[ResourceType.GPU]["critical"]
            )
        except Exception as e:
            logger.warning(f"Could not get GPU metrics: {e}")
            return ResourceMetrics(
                resource_type=ResourceType.GPU,
                current_usage=0.0,
                peak_usage=0.0,
                average_usage=0.0,
                available=0.0,
                total=0.0,
                threshold_warning=85.0,
                threshold_critical=95.0
            )
    
    def optimize_resources(self) -> Dict[str, Any]:
        """Optimizar recursos del sistema"""
        with self.lock:
            optimizations = {
                "timestamp": datetime.now().isoformat(),
                "actions_taken": [],
                "resources": {}
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
                "usage_mb": memory_metrics.current_usage,
                "usage_percent": memory_metrics.usage_percent(),
                "status": "critical" if memory_metrics.is_critical() else "warning" if memory_metrics.is_warning() else "ok"
            }
            
            # Optimizar CPU
            cpu_metrics = self.get_resource_metrics(ResourceType.CPU)
            optimizations["resources"]["cpu"] = {
                "usage_percent": cpu_metrics.current_usage,
                "status": "critical" if cpu_metrics.is_critical() else "warning" if cpu_metrics.is_warning() else "ok"
            }
            
            # Optimizar GPU si está disponible
            if torch.cuda.is_available():
                gpu_metrics = self.get_resource_metrics(ResourceType.GPU)
                if gpu_metrics.is_critical():
                    actions = self._optimize_gpu_critical()
                    optimizations["actions_taken"].extend(actions)
                
                optimizations["resources"]["gpu"] = {
                    "memory_mb": gpu_metrics.current_usage,
                    "usage_percent": gpu_metrics.usage_percent(),
                    "status": "critical" if gpu_metrics.is_critical() else "warning" if gpu_metrics.is_warning() else "ok"
                }
            
            self.last_optimization = datetime.now()
            return optimizations
    
    def _optimize_memory_critical(self) -> List[str]:
        """Optimización crítica de memoria"""
        actions = []
        
        # Garbage collection agresivo
        gc.collect()
        actions.append("Executed aggressive garbage collection")
        
        # Limpiar cache de PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            actions.append("Cleared CUDA cache")
        
        if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()
            actions.append("Cleared MPS cache")
        
        return actions
    
    def _optimize_memory_warning(self) -> List[str]:
        """Optimización de advertencia de memoria"""
        actions = []
        
        # Garbage collection normal
        gc.collect()
        actions.append("Executed garbage collection")
        
        return actions
    
    def _optimize_gpu_critical(self) -> List[str]:
        """Optimización crítica de GPU"""
        actions = []
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            actions.append("Synchronized and cleared GPU cache")
        
        return actions
    
    def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Obtener recomendaciones de optimización"""
        recommendations = []
        
        # Analizar memoria
        memory_metrics = self.get_resource_metrics(ResourceType.MEMORY)
        if memory_metrics.is_critical():
            recommendations.append(OptimizationRecommendation(
                priority=1,
                category="Memory",
                title="Critical Memory Usage",
                description=f"Memory usage at {memory_metrics.usage_percent():.1f}%",
                estimated_improvement="30-50% memory reduction",
                actions=[
                    "Enable aggressive compression",
                    "Reduce batch size",
                    "Clear unused tensors",
                    "Enable lazy loading"
                ]
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
                    "Use tensor pooling"
                ]
            ))
        
        # Analizar GPU
        if torch.cuda.is_available():
            gpu_metrics = self.get_resource_metrics(ResourceType.GPU)
            if gpu_metrics.is_critical():
                recommendations.append(OptimizationRecommendation(
                    priority=1,
                    category="GPU",
                    title="Critical GPU Memory",
                    description=f"GPU memory at {gpu_metrics.usage_percent():.1f}%",
                    estimated_improvement="20-40% GPU memory reduction",
                    actions=[
                        "Use gradient checkpointing",
                        "Reduce model size",
                        "Enable mixed precision",
                        "Clear GPU cache regularly"
                    ]
                ))
        
        # Ordenar por prioridad
        recommendations.sort(key=lambda x: x.priority)
        
        return recommendations

# ============================================================================
# PROCESADOR PARALELO DE TENSORES
# ============================================================================

class ParallelTensorProcessor:
    """Procesador paralelo de operaciones con tensores"""
    
    def __init__(self, config: MnemeConfig, max_workers: Optional[int] = None):
        self.config = config
        self.max_workers = max_workers or min(8, (mp.cpu_count() or 1) + 4)
        
        self.executor: Optional[ThreadPoolExecutor] = None
        self.lock = Lock()
        
        logger.info(f"ParallelTensorProcessor initialized with {self.max_workers} workers")
    
    def _get_executor(self) -> ThreadPoolExecutor:
        """Obtener executor, crear si no existe"""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self.executor
    
    def parallel_decomposition(
        self,
        tensors: List[torch.Tensor],
        decomp_type: DecompType
    ) -> List[Dict[str, Any]]:
        """Decomposición paralela de tensores"""
        if len(tensors) <= 1:
            # No vale la pena paralelizar
            return [{"tensor": t, "decomp_type": decomp_type.value} for t in tensors]
        
        executor = self._get_executor()
        futures = []
        
        for tensor in tensors:
            future = executor.submit(self._decompose_tensor, tensor, decomp_type)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                logger.error(f"Tensor decomposition failed: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def _decompose_tensor(
        self,
        tensor: torch.Tensor,
        decomp_type: DecompType
    ) -> Dict[str, Any]:
        """Descomponer un tensor individual"""
        try:
            # Placeholder para implementación real de descomposición
            # Aquí se implementaría la lógica específica según decomp_type
            return {
                "tensor": tensor,
                "decomp_type": decomp_type.value,
                "success": True
            }
        except Exception as e:
            return {
                "tensor": tensor,
                "decomp_type": decomp_type.value,
                "success": False,
                "error": str(e)
            }
    
    def parallel_compression(
        self,
        tensors: List[torch.Tensor],
        compression_level: CompressionLevel
    ) -> List[bytes]:
        """Compresión paralela de tensores"""
        executor = self._get_executor()
        futures = []
        
        for tensor in tensors:
            future = executor.submit(self._compress_tensor, tensor, compression_level)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                logger.error(f"Tensor compression failed: {e}")
                results.append(b"")
        
        return results
    
    def _compress_tensor(
        self,
        tensor: torch.Tensor,
        compression_level: CompressionLevel
    ) -> bytes:
        """Comprimir un tensor individual"""
        # Implementación simplificada
        return b""
    
    def cleanup(self):
        """Limpiar recursos"""
        with self.lock:
            if self.executor:
                self.executor.shutdown(wait=True, cancel_futures=True)
                self.executor = None
        logger.info("ParallelTensorProcessor cleaned up")

# ============================================================================
# EXECUTOR PARALELO
# ============================================================================

class ParallelExecutor:
    """Executor para operaciones paralelas genéricas"""
    
    def __init__(self, config: MnemeConfig, max_workers: Optional[int] = None):
        self.config = config
        self.max_workers = max_workers or min(8, (mp.cpu_count() or 1) + 4)
        
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        self.lock = Lock()
        
        logger.info(f"ParallelExecutor initialized with {self.max_workers} workers")
    
    def execute_parallel(
        self,
        func: Callable,
        items: List[Any],
        use_processes: bool = False
    ) -> List[Any]:
        """Ejecutar función en paralelo sobre items"""
        if len(items) == 0:
            return []
        
        if len(items) == 1:
            # No paralelizar para un solo item
            return [func(items[0])]
        
        executor = self._get_executor(use_processes)
        futures = [executor.submit(func, item) for item in items]
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel execution failed: {e}")
                results.append(None)
        
        return results
    
    def _get_executor(self, use_processes: bool) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        """Obtener executor apropiado"""
        with self.lock:
            if use_processes:
                if self.process_executor is None:
                    self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
                return self.process_executor
            else:
                if self.thread_executor is None:
                    self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
                return self.thread_executor
    
    def cleanup(self):
        """Limpiar recursos"""
        with self.lock:
            if self.thread_executor:
                self.thread_executor.shutdown(wait=True, cancel_futures=True)
                self.thread_executor = None
            
            if self.process_executor:
                self.process_executor.shutdown(wait=True, cancel_futures=True)
                self.process_executor = None
        
        logger.info("ParallelExecutor cleaned up")

# ============================================================================
# OPTIMIZADOR PRINCIPAL
# ============================================================================

class MNEMEOptimizer:
    """Optimizador principal de MNEME con todas las capacidades integradas"""
    
    def __init__(
        self,
        config: Optional[MnemeConfig] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
        enable_profiling: bool = True,
        enable_parallel_processing: bool = True,
        enable_auto_optimization: bool = False
    ):
        self.config = config or MnemeConfig()
        self.optimization_level = optimization_level
        self.enable_profiling = enable_profiling
        self.enable_parallel_processing = enable_parallel_processing
        self.enable_auto_optimization = enable_auto_optimization
        
        # Inicializar componentes
        self.performance_monitor = PerformanceMonitor(self.config)
        self.resource_optimizer = ResourceOptimizer(self.config)
        self.parallel_executor = ParallelExecutor(self.config)
        self.tensor_processor = ParallelTensorProcessor(self.config)
        
        # Configurar nivel de optimización
        self._configure_optimization_level()
        
        # Iniciar monitoreo si está habilitado
        if enable_profiling:
            self.performance_monitor.start_monitoring(interval=2.0)
        
        # Thread para auto-optimización
        self.auto_optimization_thread: Optional[threading.Thread] = None
        if enable_auto_optimization:
            self._start_auto_optimization()
        
        logger.info(f"MNEMEOptimizer initialized with level: {optimization_level.name}")
    
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
    
    def _start_auto_optimization(self):
        """Iniciar thread de auto-optimización"""
        def auto_optimize_loop():
            while self.enable_auto_optimization:
                try:
                    time.sleep(30)  # Optimizar cada 30 segundos
                    health = self.get_health_status()
                    
                    if health in [HealthStatus.WARNING.value, HealthStatus.CRITICAL.value]:
                        logger.info(f"Auto-optimization triggered (health: {health})")
                        self.optimize_system()
                
                except Exception as e:
                    logger.error(f"Error in auto-optimization: {e}")
        
        self.auto_optimization_thread = threading.Thread(
            target=auto_optimize_loop,
            daemon=True
        )
        self.auto_optimization_thread.start()
        logger.info("Auto-optimization thread started")
    
    def optimize_tensor_operations(
        self,
        tensors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Optimizar operaciones con tensores"""
        if not tensors:
            return []
        
        with self.performance_monitor.measure_operation("tensor_optimization"):
            if self.enable_parallel_processing and len(tensors) > 1:
                return self._optimize_tensors_parallel(tensors)
            else:
                return self._optimize_tensors_sequential(tensors)
    
    def _optimize_tensors_parallel(
        self,
        tensors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Optimización paralela de tensores"""
        results = self.tensor_processor.parallel_decomposition(
            tensors,
            self.config.decomp_type
        )
        
        optimized = []
        for result, original in zip(results, tensors):
            if result.get('success', False):
                optimized.append(result.get('tensor', original))
            else:
                optimized.append(original)
        
        return optimized
    
    def _optimize_tensors_sequential(
        self,
        tensors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Optimización secuencial de tensores"""
        optimized = []
        
        for tensor in tensors:
            if self.optimization_level == OptimizationLevel.BASIC:
                opt_tensor = self._basic_optimization(tensor)
            elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
                opt_tensor = self._aggressive_optimization(tensor)
            elif self.optimization_level == OptimizationLevel.MAXIMUM:
                opt_tensor = self._maximum_optimization(tensor)
            else:
                opt_tensor = tensor
            
            optimized.append(opt_tensor)
        
        return optimized
    
    def _basic_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimización básica"""
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor
    
    def _aggressive_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimización agresiva"""
        tensor = self._basic_optimization(tensor)
        
        # Pin memory para transferencias CPU-GPU más rápidas
        if torch.cuda.is_available() and tensor.device.type == 'cpu':
            try:
                tensor = tensor.pin_memory()
            except Exception as e:
                logger.debug(f"Could not pin memory: {e}")
        
        return tensor
    
    def _maximum_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimización máxima"""
        tensor = self._aggressive_optimization(tensor)
        
        # Aplicar compresión para tensores grandes
        if tensor.numel() > 1000000:  # > 1M elementos
            # Aquí se aplicaría compresión real
            pass
        
        return tensor
    
    def optimize_system(self) -> Dict[str, Any]:
        """Optimizar sistema completo"""
        with self.performance_monitor.measure_operation("system_optimization"):
            return self.resource_optimizer.optimize_resources()
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Obtener reporte completo de optimización"""
        return {
            "timestamp": datetime.now().isoformat(),
            "optimization_level": self.optimization_level.value,
            "performance": self.performance_monitor.get_performance_report(),
            "resources": self.resource_optimizer.optimize_resources(),
            "recommendations": [
                rec.__dict__ for rec in self.resource_optimizer.get_optimization_recommendations()
            ],
            "health_status": self.get_health_status()
        }
    
    def get_health_status(self) -> str:
        """Obtener estado de salud"""
        return self.performance_monitor.get_health_status()
    
    def cleanup(self):
        """Limpiar todos los recursos"""
        logger.info("Cleaning up MNEMEOptimizer...")
        
        self.enable_auto_optimization = False
        if self.auto_optimization_thread:
            self.auto_optimization_thread.join(timeout=2.0)
        
        self.performance_monitor.cleanup()
        self.parallel_executor.cleanup()
        self.tensor_processor.cleanup()
        
        logger.info("MNEMEOptimizer cleanup completed")

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def create_optimizer(
    config: Optional[MnemeConfig] = None,
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
    enable_profiling: bool = True,
    enable_parallel: bool = True
) -> MNEMEOptimizer:
    """Crear optimizador con configuración específica"""
    return MNEMEOptimizer(
        config=config,
        optimization_level=optimization_level,
        enable_profiling=enable_profiling,
        enable_parallel_processing=enable_parallel
    )

def optimize_model(
    model: torch.nn.Module,
    config: Optional[MnemeConfig] = None,
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
) -> torch.nn.Module:
    """Optimizar modelo PyTorch usando MNEME"""
    optimizer = MNEMEOptimizer(config, optimization_level)
    
    # Optimizar parámetros del modelo
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.data = optimizer._basic_optimization(param.data)
    
    return model

def get_system_metrics() -> Dict[str, Any]:
    """Obtener métricas del sistema"""
    config = MnemeConfig()
    monitor = PerformanceMonitor(config)
    
    try:
        return monitor.get_performance_report()
    finally:
        monitor.cleanup()

def benchmark_optimization(
    tensors: List[torch.Tensor],
    optimization_levels: List[OptimizationLevel] = None
) -> Dict[str, Any]:
    """Benchmark de diferentes niveles de optimización"""
    if optimization_levels is None:
        optimization_levels = list(OptimizationLevel)
    
    results = {}
    
    for level in optimization_levels:
        optimizer = MNEMEOptimizer(optimization_level=level, enable_profiling=True)
        
        try:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            optimized = optimizer.optimize_tensor_operations(tensors)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            results[level.name] = {
                "time_seconds": end_time - start_time,
                "memory_delta_mb": end_memory - start_memory,
                "tensors_processed": len(tensors),
                "success": True
            }
        
        except Exception as e:
            results[level.name] = {
                "error": str(e),
                "success": False
            }
        
        finally:
            optimizer.cleanup()
    
    return results

# ============================================================================
# EXPORTACIONES
# ============================================================================

__all__ = [
    # Clases principales
    'MNEMEOptimizer',
    'PerformanceMonitor',
    'ResourceOptimizer',
    'ParallelTensorProcessor',
    'ParallelExecutor',
    
    # Enums
    'OptimizationLevel',
    'ResourceType',
    'OptimizationStrategy',
    'HealthStatus',
    
    # Dataclasses
    'PerformanceMetrics',
    'ResourceMetrics',
    'OptimizationRecommendation',
    
    # Funciones de utilidad
    'create_optimizer',
    'optimize_model',
    'get_system_metrics',
    'benchmark_optimization'
]