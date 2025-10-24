"""
MNEME - Motor de Memoria Neural Mórfica v2.0

Sistema avanzado de memoria computacional con síntesis determinista,
optimización en paralelo, seguridad avanzada y monitoreo de rendimiento.
"""

from .mneme_core import (
    ZSpace,
    ZDescriptor,
    ZAddr,
    MnemeConfig,
    CompressionLevel,
    SecurityLevel,
    SerializationFormat,
    TensorEncryptionMode,
    KeyRotationPolicy,
    StorageBackend,
    CachePolicy,
    CompressionStrategy,
    ContextSimilarityMethod,
    ContextClusteringMethod,
    SecurityError,
    ValidationError,
    StorageError,
    MnemeError,
    # Nuevas funcionalidades de seguridad
    GranularLockManager,
    LazyTensor,
    AdaptiveCache
)

from .mneme_torch import (
    ZLinear,
    ZConv2d,
    ZAttention,
    ZTransformerBlock,
    ZParameter,
    compress_model,
    get_compression_stats,
    get_model_performance_stats,
    optimize_model_memory,
    get_system_metrics,
    get_health_status,
    optimize_system,
    CompressionConfig
)

from .mneme_security_core import (
    SecurityManager,
    SecurityConfig,
    InputValidator,
    SecureSerializer,
    create_secure_config,
    validate_tensor_safe,
    secure_tensor_serialize,
    secure_tensor_deserialize
)

from .mneme_storage_core import (
    SecureStorageBackend,
    SecureCache,
    StorageConfig,
    create_secure_storage,
    create_secure_cache
)

__version__ = "2.0.0"
__author__ = "Esraderey and Raul Cruz Acosta"
__email__ = "msc.framework@gmail.com"

__all__ = [
    # Core classes
    "ZSpace",
    "ZDescriptor", 
    "ZAddr",
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
    "ZParameter",
    "compress_model",
    "get_compression_stats",
    "get_model_performance_stats",
    "optimize_model_memory",
    "get_system_metrics",
    "get_health_status",
    "optimize_system",
    "CompressionConfig",
    
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
    
    # Exceptions
    "SecurityError",
    "ValidationError",
    "StorageError",
    "MnemeError"
]