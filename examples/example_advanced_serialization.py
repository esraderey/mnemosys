#!/usr/bin/env python3
"""
Ejemplo de uso de la serialización avanzada en MNEME
Demuestra las nuevas funcionalidades de seguridad y múltiples formatos.
"""

import torch
import numpy as np
import time
from mneme_core import (
    MnemeConfig, 
    ZSpace, 
    SerializationFormat, 
    SecurityLevel,
    CompressionLevel
)

def demo_serialization_formats():
    """Demostrar diferentes formatos de serialización."""
    print("=== Demostración de Formatos de Serialización ===")
    
    # Crear datos de prueba
    tensor_data = torch.randn(100, 50)
    simple_data = {"numbers": [1, 2, 3, 4, 5], "text": "Hello MNEME"}
    mixed_data = {
        "tensor": tensor_data,
        "metadata": simple_data,
        "numpy_array": np.random.rand(10, 10)
    }
    
    formats = [
        SerializationFormat.TORCH,
        SerializationFormat.MSGPACK,
        SerializationFormat.JSON,
        SerializationFormat.HYBRID
    ]
    
    for format_type in formats:
        print(f"\n--- Formato: {format_type.value} ---")
        
        # Configurar MNEME con formato específico
        config = MnemeConfig(
            serialization_format=format_type,
            security_level=SecurityLevel.NONE,  # Sin seguridad para comparar tamaños
            enable_compression=False,
            enable_validation=False
        )
        
        with ZSpace(config) as mneme:
            # Medir tiempo de serialización
            start_time = time.time()
            desc = mneme.register("test_data", mixed_data)
            serialization_time = time.time() - start_time
            
            # Medir tiempo de deserialización
            start_time = time.time()
            loaded_data = mneme.load("test_data")
            deserialization_time = time.time() - start_time
            
            print(f"Tiempo de serialización: {serialization_time:.4f}s")
            print(f"Tiempo de deserialización: {deserialization_time:.4f}s")
            print(f"Tamaño comprimido: {len(desc.core_data)} bytes")
            print(f"Ratio de compresión: {desc.meta.get('compression_ratio', 'N/A')}")

def demo_security_levels():
    """Demostrar diferentes niveles de seguridad."""
    print("\n=== Demostración de Niveles de Seguridad ===")
    
    # Datos sensibles de prueba
    sensitive_data = {
        "model_weights": torch.randn(1000, 1000),
        "api_keys": ["secret_key_1", "secret_key_2"],
        "user_data": {"id": 12345, "name": "Usuario Test"}
    }
    
    security_levels = [
        (SecurityLevel.NONE, "Sin seguridad"),
        (SecurityLevel.HMAC, "Firma HMAC"),
        (SecurityLevel.ENCRYPTED, "Cifrado completo")
    ]
    
    for security_level, description in security_levels:
        print(f"\n--- {description} ---")
        
        config = MnemeConfig(
            serialization_format=SerializationFormat.HYBRID,
            security_level=security_level,
            secret_key=b"test_secret_key_32_bytes_long_12345" if security_level != SecurityLevel.NONE else None,
            enable_encryption=(security_level == SecurityLevel.ENCRYPTED),
            encryption_password="test_password_123" if security_level == SecurityLevel.ENCRYPTED else None,
            enable_compression=True,
            enable_validation=True
        )
        
        with ZSpace(config) as mneme:
            # Medir tiempo de serialización con seguridad
            start_time = time.time()
            desc = mneme.register("sensitive_data", sensitive_data)
            serialization_time = time.time() - start_time
            
            # Medir tiempo de deserialización
            start_time = time.time()
            loaded_data = mneme.load("sensitive_data")
            deserialization_time = time.time() - start_time
            
            print(f"Tiempo de serialización: {serialization_time:.4f}s")
            print(f"Tiempo de deserialización: {deserialization_time:.4f}s")
            print(f"Tamaño final: {len(desc.core_data)} bytes")
            
            # Verificar integridad
            if desc.verify_integrity():
                print("✓ Verificación de integridad: EXITOSA")
            else:
                print("✗ Verificación de integridad: FALLÓ")

def demo_compression_levels():
    """Demostrar diferentes niveles de compresión."""
    print("\n=== Demostración de Niveles de Compresión ===")
    
    # Datos con diferentes patrones de compresión
    random_data = torch.randn(500, 500)  # Datos aleatorios (baja compresión)
    sparse_data = torch.zeros(500, 500)  # Datos dispersos (alta compresión)
    sparse_data[::10, ::10] = 1.0
    
    test_cases = [
        ("Datos aleatorios", random_data),
        ("Datos dispersos", sparse_data)
    ]
    
    compression_levels = [
        CompressionLevel.ULTRA_FAST,
        CompressionLevel.BALANCED,
        CompressionLevel.MAXIMUM
    ]
    
    for data_name, data in test_cases:
        print(f"\n--- {data_name} ---")
        
        for comp_level in compression_levels:
            config = MnemeConfig(
                serialization_format=SerializationFormat.TORCH,
                compression_level=comp_level,
                security_level=SecurityLevel.HMAC,
                secret_key=b"test_secret_key_32_bytes_long_12345",
                enable_compression=True
            )
            
            with ZSpace(config) as mneme:
                start_time = time.time()
                desc = mneme.register(f"test_{comp_level.value}", data)
                serialization_time = time.time() - start_time
                
                print(f"  {comp_level.value}: {len(desc.core_data)} bytes "
                      f"(ratio: {desc.meta.get('compression_ratio', 'N/A'):.3f}, "
                      f"tiempo: {serialization_time:.4f}s)")

def demo_validation_features():
    """Demostrar características de validación."""
    print("\n=== Demostración de Validación ===")
    
    config = MnemeConfig(
        serialization_format=SerializationFormat.HYBRID,
        security_level=SecurityLevel.HMAC,
        secret_key=b"test_secret_key_32_bytes_long_12345",
        enable_validation=True,
        enable_compression=True
    )
    
    with ZSpace(config) as mneme:
        # Datos de prueba con diferentes tipos
        test_cases = [
            ("tensor_simple", torch.randn(10, 10)),
            ("tensor_complejo", torch.randn(100, 100, 3)),
            ("datos_mixtos", {
                "tensor": torch.randn(50, 50),
                "lista": [1, 2, 3, 4, 5],
                "dict": {"a": 1, "b": 2}
            }),
            ("numpy_data", np.random.rand(20, 20))
        ]
        
        for name, data in test_cases:
            print(f"\n--- {name} ---")
            
            # Serializar
            desc = mneme.register(name, data)
            print(f"Tipo original: {type(data).__name__}")
            print(f"Tamaño serializado: {len(desc.core_data)} bytes")
            
            # Deserializar y verificar
            loaded_data = mneme.load(name)
            print(f"Tipo cargado: {type(loaded_data).__name__}")
            
            # Verificar integridad del descriptor
            if desc.verify_integrity():
                print("✓ Integridad verificada")
            else:
                print("✗ Fallo en verificación de integridad")

def demo_performance_comparison():
    """Comparar rendimiento entre diferentes configuraciones."""
    print("\n=== Comparación de Rendimiento ===")
    
    # Datos de prueba grandes
    large_tensor = torch.randn(1000, 1000)
    
    configurations = [
        ("Básico (Torch)", MnemeConfig(
            serialization_format=SerializationFormat.TORCH,
            security_level=SecurityLevel.NONE,
            enable_compression=False,
            enable_validation=False
        )),
        ("Con compresión", MnemeConfig(
            serialization_format=SerializationFormat.TORCH,
            security_level=SecurityLevel.NONE,
            enable_compression=True,
            compression_level=CompressionLevel.BALANCED
        )),
        ("Con seguridad", MnemeConfig(
            serialization_format=SerializationFormat.TORCH,
            security_level=SecurityLevel.HMAC,
            secret_key=b"test_secret_key_32_bytes_long_12345",
            enable_compression=True
        )),
        ("Híbrido completo", MnemeConfig(
            serialization_format=SerializationFormat.HYBRID,
            security_level=SecurityLevel.HMAC,
            secret_key=b"test_secret_key_32_bytes_long_12345",
            enable_compression=True,
            enable_validation=True
        ))
    ]
    
    for config_name, config in configurations:
        print(f"\n--- {config_name} ---")
        
        with ZSpace(config) as mneme:
            # Medir serialización
            start_time = time.time()
            desc = mneme.register("perf_test", large_tensor)
            serialization_time = time.time() - start_time
            
            # Medir deserialización
            start_time = time.time()
            loaded_tensor = mneme.load("perf_test")
            deserialization_time = time.time() - start_time
            
            # Verificar que los datos son correctos
            data_correct = torch.allclose(large_tensor, loaded_tensor)
            
            print(f"Serialización: {serialization_time:.4f}s")
            print(f"Deserialización: {deserialization_time:.4f}s")
            print(f"Tamaño: {len(desc.core_data)} bytes")
            print(f"Datos correctos: {'✓' if data_correct else '✗'}")

if __name__ == "__main__":
    print("MNEME - Demostración de Serialización Avanzada")
    print("=" * 50)
    
    try:
        demo_serialization_formats()
        demo_security_levels()
        demo_compression_levels()
        demo_validation_features()
        demo_performance_comparison()
        
        print("\n" + "=" * 50)
        print("✓ Demostración completada exitosamente")
        print("\nCaracterísticas implementadas:")
        print("- Múltiples formatos de serialización (Torch, MessagePack, JSON, Híbrido)")
        print("- Niveles de seguridad (HMAC, Cifrado)")
        print("- Compresión LZ4 configurable")
        print("- Validación de integridad y tipos")
        print("- Reemplazo completo de pickle por métodos seguros")
        
    except Exception as e:
        print(f"\n✗ Error durante la demostración: {e}")
        import traceback
        traceback.print_exc()
