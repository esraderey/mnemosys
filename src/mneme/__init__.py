"""
MNEME - Motor de Memoria Neural Mórfica

Sistema avanzado de memoria computacional con síntesis determinista.
"""

from .mneme_core import (
    ZSpace,
    ZDescriptor,
    ZGen,
    MnemeConfig,
    CompressionLevel,
    SecurityLevel,
    SerializationFormat,
    SecurityLevel as SecurityLevel,
    TensorEncryptionMode,
    KeyRotationPolicy,
    StorageBackend,
    CachePolicy,
    CompressionStrategy,
    ContextSimilarityMethod,
    ContextClusteringMethod,
    SecurityError,
    MnemeError
)

from .mneme_torch import (
    ZLinear,
    ZConv2d,
    ZAttention,
    ZTransformerBlock,
    compress_model,
    get_compression_stats,
    CompressionConfig
)

from .mneme_security import (
    SecurityManager,
    SecurityLevel as SecurityLevelEnum,
    SecureDescriptor
)

from .mneme_optimization import (
    MNEMEOptimizer,
    OptimizationLevel,
    PerformanceProfiler
)

__version__ = "2.0.0"
__author__ = "Esraderey and Raul Cruz Acosta"
__email__ = "msc.framework@gmail.com"

__all__ = [
    # Core classes
    "ZSpace",
    "ZDescriptor", 
    "ZGen",
    "MnemeConfig",
    
    # Enums
    "CompressionLevel",
    "SecurityLevel",
    "SerializationFormat",
    "TensorEncryptionMode",
    "KeyRotationPolicy",
    "StorageBackend",
    "CachePolicy",
    "CompressionStrategy",
    "ContextSimilarityMethod",
    "ContextClusteringMethod",
    
    # PyTorch integration
    "ZLinear",
    "ZConv2d", 
    "ZAttention",
    "ZTransformerBlock",
    "compress_model",
    "get_compression_stats",
    "CompressionConfig",
    
    # Security
    "SecurityManager",
    "SecurityLevelEnum",
    "SecureDescriptor",
    
    # Optimization
    "MNEMEOptimizer",
    "OptimizationLevel",
    "PerformanceProfiler",
    
    # Exceptions
    "SecurityError",
    "MnemeError"
]
