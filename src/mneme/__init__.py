"""
MNEME - Motor de Memoria Neural Mórfica v2.0

Sistema avanzado de memoria computacional con síntesis determinista,
optimización en paralelo, seguridad avanzada y monitoreo de rendimiento.
"""

from .mneme_core import (
    AdaptiveCache,
    CachePolicy,
    CompressionLevel,
    DecompType,
    # Funcionalidades avanzadas
    GranularLockManager,
    LazyTensor,
    MarkovPrefetcher,
    MnemeConfig,
    MnemeError,
    SecurityError,
    SecurityLevel,
    StorageError,
    TensorDecomposer,
    ValidationError,
    ZAddr,
    ZDescriptor,
    ZSpace,
)
from .mneme_lazy import ZLinearTurbo, compress_model_turbo
from .mneme_optimization import (
    GPTQCalibrator,
    MNEMEOptimizer,
    OptimizationLevel,
    QuantizationType,
    StructuredSparsifier,
    TensorQuantizer,
)
from .mneme_security_core import (
    InputValidator,
    SecureSerializer,
    SecurityConfig,
    SecurityManager,
    create_secure_config,
    secure_tensor_deserialize,
    secure_tensor_serialize,
    validate_tensor_safe,
)
from .mneme_storage_core import SecureCache, SecureStorageBackend, StorageConfig, create_secure_cache, create_secure_storage
from .mneme_torch import (
    CompressionConfig,
    LayerPrecisionPolicy,
    QuantizedKVCache,
    ZAttention,
    ZConv2d,
    ZLinear,
    ZParameter,
    ZTransformerBlock,
    compress_model,
    compress_model_calibrated,
    get_compression_stats,
    get_health_status,
    get_model_performance_stats,
    get_system_metrics,
    optimize_model_memory,
    optimize_system,
)

__version__ = "1.0.1"
__author__ = "Esraderey and Raul Cruz Acosta"
__email__ = "msc.framework@gmail.com"

__all__ = [
    # Core classes
    "ZSpace",
    "ZDescriptor",
    "ZAddr",
    "MnemeConfig",

    # Enums
    "DecompType",
    "CompressionLevel",
    "SecurityLevel",
    "CachePolicy",

    # PyTorch integration
    "ZLinear",
    "ZConv2d",
    "ZAttention",
    "ZTransformerBlock",
    "ZParameter",
    "compress_model",
    "compress_model_calibrated",
    "get_compression_stats",
    "get_model_performance_stats",
    "optimize_model_memory",
    "get_system_metrics",
    "get_health_status",
    "optimize_system",
    "CompressionConfig",
    "LayerPrecisionPolicy",
    "QuantizedKVCache",
    "ZLinearTurbo",
    "compress_model_turbo",

    # Optimization
    "MNEMEOptimizer",
    "OptimizationLevel",
    "GPTQCalibrator",
    "StructuredSparsifier",
    "QuantizationType",
    "TensorQuantizer",

    # Security
    "SecurityManager",
    "SecurityConfig",
    "InputValidator",
    "SecureSerializer",
    "create_secure_config",
    "validate_tensor_safe",
    "secure_tensor_serialize",
    "secure_tensor_deserialize",

    # Storage
    "SecureStorageBackend",
    "SecureCache",
    "StorageConfig",
    "create_secure_storage",
    "create_secure_cache",

    # Advanced features
    "GranularLockManager",
    "LazyTensor",
    "AdaptiveCache",
    "TensorDecomposer",
    "MarkovPrefetcher",

    # Exceptions
    "SecurityError",
    "ValidationError",
    "StorageError",
    "MnemeError",
]
