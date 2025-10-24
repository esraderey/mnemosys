"""
MNEME Optimization Module - Actualizado
Módulo de optimización que utiliza las funcionalidades del core
"""

from .mneme_core import (
    ZSpace, MnemeConfig
)
import torch
import numpy as np
import time
import gc
import psutil
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import contextmanager
import queue
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Niveles de optimización"""
    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    MAXIMUM = 3

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento simplificadas"""
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    cache_hit_rate: float = 0.0
    compression_ratio: float = 1.0
    synthesis_time: float = 0.0
    total_operations: int = 0
    failed_operations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_usage_mb": self.memory_usage,
            "cpu_usage_percent": self.cpu_usage,
            "gpu_usage_percent": self.gpu_usage,
            "cache_hit_rate": self.cache_hit_rate,
            "compression_ratio": self.compression_ratio,
            "synthesis_time_ms": self.synthesis_time * 1000,
            "total_operations": self.total_operations,
            "failed_operations": self.failed_operations,
            "success_rate": (self.total_operations - self.failed_operations) / max(1, self.total_operations)
        }

class MNEMEOptimizer:
    """Optimizador principal de MNEME que utiliza las funcionalidades del core"""
    
    def __init__(self, 
                 config: MnemeConfig = None,
                 optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
                 enable_profiling: bool = True,
                 enable_parallel_processing: bool = True):
        
        self.config = config or MnemeConfig()
        self.optimization_level = optimization_level
        self.enable_profiling = enable_profiling
        self.enable_parallel_processing = enable_parallel_processing
        
        # Usar componentes del core
        self.performance_monitor = PerformanceMonitor(self.config)
        self.resource_optimizer = ResourceOptimizer(self.config)
        self.parallel_executor = ParallelExecutor(self.config)
        self.tensor_processor = ParallelTensorProcessor(self.config)
        
        # Configurar nivel de optimización
        self._configure_optimization_level()
        
        logger.info(f"MNEME Optimizer initialized with level: {optimization_level.name}")
    
    def _configure_optimization_level(self):
        """Configurar nivel de optimización en el config"""
        if self.optimization_level == OptimizationLevel.AGGRESSIVE:
            self.config.memory_pressure_threshold = 0.7
            self.config.compression_level = self.config.compression_level.MAXIMUM
        elif self.optimization_level == OptimizationLevel.MAXIMUM:
            self.config.memory_pressure_threshold = 0.6
            self.config.compression_level = self.config.compression_level.MAXIMUM
            self.config.enable_adaptive_compression = True
    
    def optimize_tensor_operations(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Optimizar operaciones con tensores usando procesamiento paralelo"""
        if self.enable_profiling:
            with self._profile_operation("tensor_optimization"):
                return self._optimize_tensors_parallel(tensors)
        else:
            return self._optimize_tensors_parallel(tensors)
    
    def _optimize_tensors_parallel(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Optimizar tensores usando procesamiento paralelo"""
        if self.enable_parallel_processing and len(tensors) > 1:
            # Usar procesador paralelo del core
            results = self.tensor_processor.parallel_decomposition(tensors, self.config.decomp_type)
            return [result.get('tensor', tensor) for result, tensor in zip(results, tensors)]
        else:
            return self._optimize_tensors_sequential(tensors)
    
    def _optimize_tensors_sequential(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Optimizar tensores secuencialmente"""
        optimized = []
        for tensor in tensors:
            if self.optimization_level == OptimizationLevel.BASIC:
                optimized_tensor = self._basic_optimization(tensor)
            elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
                optimized_tensor = self._aggressive_optimization(tensor)
            elif self.optimization_level == OptimizationLevel.MAXIMUM:
                optimized_tensor = self._maximum_optimization(tensor)
            else:
                optimized_tensor = tensor
            
            optimized.append(optimized_tensor)
        
        return optimized
    
    def _basic_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimización básica"""
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor
    
    def _aggressive_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimización agresiva"""
        tensor = self._basic_optimization(tensor)
        if torch.cuda.is_available():
            tensor = tensor.pin_memory()
        return tensor
    
    def _maximum_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimización máxima"""
        tensor = self._aggressive_optimization(tensor)
        if tensor.numel() > 1000000:  # 1M elementos
            tensor = self._apply_memory_compression(tensor)
        return tensor
    
    def _apply_memory_compression(self, tensor: torch.Tensor) -> torch.Tensor:
        """Aplicar compresión temporal en memoria"""
        # Implementación simplificada
        return tensor
    
    @contextmanager
    def _profile_operation(self, operation_name: str):
        """Context manager para perfilar operaciones"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated()
        
        try:
            yield
            self.performance_monitor.record_operation(operation_name, time.time() - start_time, True)
        except Exception as e:
            self.performance_monitor.record_operation(operation_name, time.time() - start_time, False)
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            operation_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated()
                gpu_memory_delta = end_gpu_memory - start_gpu_memory
            
            logger.debug(f"Operation {operation_name}: {operation_time*1000:.2f}ms, Memory: {memory_delta:+.1f}MB")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Obtener reporte de optimización usando métricas del core"""
        report = {
            "optimization_level": self.optimization_level.value,
            "performance_metrics": self.performance_monitor.get_performance_report(),
            "resource_optimization": self.resource_optimizer.optimize_resources(),
            "recommendations": self.resource_optimizer.get_optimization_recommendations()
        }
        
        return report
    
    def optimize_system(self) -> Dict[str, Any]:
        """Optimizar sistema completo usando funcionalidades del core"""
        return self.resource_optimizer.optimize_resources()
    
    def get_health_status(self) -> str:
        """Obtener estado de salud del sistema"""
        return self.performance_monitor.get_health_status()
    
    def cleanup(self):
        """Limpiar todos los recursos"""
        self.performance_monitor.cleanup()
        self.parallel_executor.cleanup()
        self.tensor_processor.cleanup()
        logger.info("MNEME Optimizer cleanup completed")

# Funciones de utilidad para compatibilidad
def create_optimizer(config: MnemeConfig = None, 
                    optimization_level: OptimizationLevel = OptimizationLevel.BASIC) -> MNEMEOptimizer:
    """Crear optimizador con configuración específica"""
    return MNEMEOptimizer(config, optimization_level)

def optimize_model(model: torch.nn.Module, 
                  config: MnemeConfig = None,
                  optimization_level: OptimizationLevel = OptimizationLevel.BASIC) -> torch.nn.Module:
    """Optimizar modelo usando MNEME"""
    optimizer = MNEMEOptimizer(config, optimization_level)
    
    # Aplicar optimizaciones a los parámetros del modelo
    for param in model.parameters():
        if param.requires_grad:
            param.data = optimizer._basic_optimization(param.data)
    
    return model

def get_system_metrics() -> Dict[str, Any]:
    """Obtener métricas del sistema usando el core"""
    config = MnemeConfig()
    monitor = PerformanceMonitor(config)
    return monitor.get_performance_report()