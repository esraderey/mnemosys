#!/usr/bin/env python3
"""
Ejemplo de uso de cifrado de tensores, rotación de claves y contexto mejorado en MNEME
"""

import torch
import numpy as np
import asyncio
import time
from datetime import timedelta
from mneme_core import (
    MnemeConfig, 
    ZSpace, 
    TensorEncryptionMode,
    KeyRotationPolicy,
    SecurityLevel,
    SerializationFormat,
    mneme_context,
    async_mneme_context
)

def demo_tensor_encryption():
    """Demostrar cifrado específico de tensores."""
    print("=== Demostración de Cifrado de Tensores ===")
    
    # Crear tensores de prueba
    test_tensors = {
        "small_tensor": torch.randn(100, 100),
        "large_tensor": torch.randn(1000, 1000),
        "sparse_tensor": torch.zeros(500, 500),
        "mixed_data": {
            "weights": torch.randn(200, 200),
            "bias": torch.randn(200),
            "metadata": {"epoch": 10, "loss": 0.5}
        }
    }
    
    encryption_modes = [
        TensorEncryptionMode.AES_GCM,
        TensorEncryptionMode.AES_CBC,
        TensorEncryptionMode.CHACHA20,
        TensorEncryptionMode.BLOCK_CHAIN
    ]
    
    for mode in encryption_modes:
        print(f"\n--- Modo de Cifrado: {mode.value} ---")
        
        config = MnemeConfig(
            enable_tensor_encryption=True,
            tensor_encryption_mode=mode,
            tensor_encryption_key=b"test_tensor_key_32_bytes_long_12345",
            serialization_format=SerializationFormat.TORCH,
            security_level=SecurityLevel.HMAC,
            secret_key=b"test_secret_key_32_bytes_long_12345"
        )
        
        with mneme_context(config) as mneme:
            for name, tensor in test_tensors.items():
                print(f"\n  Procesando: {name}")
                
                # Medir tiempo de cifrado
                start_time = time.time()
                desc = mneme.register(name, tensor)
                encryption_time = time.time() - start_time
                
                # Medir tiempo de descifrado
                start_time = time.time()
                loaded_tensor = mneme.load(name)
                decryption_time = time.time() - start_time
                
                # Verificar integridad
                if isinstance(tensor, torch.Tensor):
                    is_correct = torch.allclose(tensor, loaded_tensor, atol=1e-6)
                else:
                    # Para datos mixtos, verificar componentes
                    is_correct = True
                    if isinstance(tensor, dict) and isinstance(loaded_tensor, dict):
                        for key in tensor:
                            if isinstance(tensor[key], torch.Tensor):
                                is_correct &= torch.allclose(tensor[key], loaded_tensor[key], atol=1e-6)
                
                print(f"    Tiempo de cifrado: {encryption_time:.4f}s")
                print(f"    Tiempo de descifrado: {decryption_time:.4f}s")
                print(f"    Tamaño cifrado: {len(desc.core_data)} bytes")
                print(f"    Integridad: {'✓' if is_correct else '✗'}")

def demo_key_rotation():
    """Demostrar rotación automática de claves."""
    print("\n=== Demostración de Rotación de Claves ===")
    
    rotation_policies = [
        (KeyRotationPolicy.TIME_BASED, "Basada en tiempo (5 segundos)"),
        (KeyRotationPolicy.USAGE_BASED, "Basada en uso (5 operaciones)"),
        (KeyRotationPolicy.ADAPTIVE, "Adaptativa")
    ]
    
    for policy, description in rotation_policies:
        print(f"\n--- {description} ---")
        
        config = MnemeConfig(
            key_rotation_policy=policy,
            key_rotation_interval=timedelta(seconds=5) if policy == KeyRotationPolicy.TIME_BASED else timedelta(days=30),
            key_rotation_usage_count=5 if policy == KeyRotationPolicy.USAGE_BASED else 1000,
            enable_key_versioning=True,
            secret_key=b"initial_key_32_bytes_long_12345"
        )
        
        with mneme_context(config) as mneme:
            # Obtener información inicial de claves
            initial_key_info = mneme.get_key_info()
            print(f"  Clave inicial: versión {initial_key_info.get('current_version', 'N/A')}")
            
            # Realizar operaciones para activar rotación
            tensor = torch.randn(50, 50)
            
            for i in range(10):
                name = f"test_tensor_{i}"
                desc = mneme.register(name, tensor)
                
                # Obtener información de claves después de cada operación
                key_info = mneme.get_key_info()
                print(f"  Operación {i+1}: versión {key_info.get('current_version', 'N/A')}, "
                      f"uso {key_info.get('usage_count', 'N/A')}")
                
                # Rotar manualmente si es necesario
                if i == 5:
                    rotated = mneme.rotate_keys()
                    if rotated:
                        print(f"    ✓ Rotación manual activada")
                
                time.sleep(0.1)  # Pequeña pausa para políticas de tiempo
            
            # Información final
            final_key_info = mneme.get_key_info()
            print(f"  Clave final: versión {final_key_info.get('current_version', 'N/A')}")
            print(f"  Total de usos: {final_key_info.get('usage_count', 'N/A')}")

def demo_async_context():
    """Demostrar contexto asíncrono mejorado."""
    print("\n=== Demostración de Contexto Asíncrono ===")
    
    async def async_operations():
        config = MnemeConfig(
            enable_async_context=True,
            max_concurrent_operations=3,
            enable_tensor_encryption=True,
            tensor_encryption_mode=TensorEncryptionMode.AES_GCM,
            key_rotation_policy=KeyRotationPolicy.USAGE_BASED,
            key_rotation_usage_count=3
        )
        
        async with async_mneme_context(config) as mneme:
            print("  Contexto asíncrono inicializado")
            
            # Crear tareas concurrentes
            tasks = []
            for i in range(5):
                tensor = torch.randn(100, 100)
                task = mneme.register_async(f"async_tensor_{i}", tensor)
                tasks.append(task)
            
            # Ejecutar todas las tareas
            print("  Ejecutando operaciones concurrentes...")
            start_time = time.time()
            descriptors = await asyncio.gather(*tasks)
            registration_time = time.time() - start_time
            
            print(f"  Tiempo de registro concurrente: {registration_time:.4f}s")
            print(f"  Operaciones activas: {await mneme.active_operations_count}")
            
            # Cargar tensores de forma asíncrona
            load_tasks = []
            for i in range(5):
                task = mneme.load_async(f"async_tensor_{i}")
                load_tasks.append(task)
            
            start_time = time.time()
            loaded_tensors = await asyncio.gather(*load_tasks)
            load_time = time.time() - start_time
            
            print(f"  Tiempo de carga concurrente: {load_time:.4f}s")
            
            # Obtener estadísticas
            stats = await mneme.get_stats_async()
            print(f"  Estadísticas: {len(stats.get('descriptors', []))} descriptores")
    
    # Ejecutar operaciones asíncronas
    asyncio.run(async_operations())

def demo_resource_monitoring():
    """Demostrar monitoreo de recursos."""
    print("\n=== Demostración de Monitoreo de Recursos ===")
    
    config = MnemeConfig(
        enable_async_context=True,
        memory_pressure_threshold=0.5,  # 50% para demostración
        enable_tensor_encryption=True,
        tensor_encryption_mode=TensorEncryptionMode.BLOCK_CHAIN
    )
    
    with mneme_context(config) as mneme:
        print("  Monitoreo de recursos activado")
        
        # Crear tensores grandes para probar el monitoreo
        large_tensors = []
        for i in range(3):
            tensor = torch.randn(1000, 1000)  # ~4MB cada uno
            name = f"large_tensor_{i}"
            desc = mneme.register(name, tensor)
            large_tensors.append((name, desc))
            print(f"  Registrado tensor {i+1}: {len(desc.core_data)} bytes")
        
        # Simular operaciones intensivas
        print("  Simulando operaciones intensivas...")
        for i in range(5):
            # Actualizar tensores
            delta_op = {
                "type": "add",
                "value": torch.randn(1000, 1000) * 0.1
            }
            mneme.update(f"large_tensor_{i % 3}", delta_op)
            time.sleep(0.5)
        
        # Obtener estadísticas finales
        stats = mneme.get_stats()
        print(f"  Estadísticas finales: {stats}")

def demo_advanced_security():
    """Demostrar características de seguridad avanzadas."""
    print("\n=== Demostración de Seguridad Avanzada ===")
    
    # Datos sensibles
    sensitive_data = {
        "model_weights": torch.randn(500, 500),
        "user_embeddings": torch.randn(1000, 128),
        "api_keys": ["secret_key_1", "secret_key_2"],
        "metadata": {
            "user_id": 12345,
            "session_token": "abc123def456",
            "permissions": ["read", "write", "admin"]
        }
    }
    
    config = MnemeConfig(
        enable_tensor_encryption=True,
        tensor_encryption_mode=TensorEncryptionMode.AES_GCM,
        security_level=SecurityLevel.ENCRYPTED,
        enable_encryption=True,
        encryption_password="strong_password_123",
        enable_validation=True,
        key_rotation_policy=KeyRotationPolicy.ADAPTIVE,
        enable_key_versioning=True
    )
    
    with mneme_context(config) as mneme:
        print("  Configuración de seguridad máxima activada")
        
        # Registrar datos sensibles
        for name, data in sensitive_data.items():
            desc = mneme.register(name, data)
            print(f"  Registrado {name}: {len(desc.core_data)} bytes")
            
            # Verificar integridad
            if desc.verify_integrity():
                print(f"    ✓ Integridad verificada")
            else:
                print(f"    ✗ Fallo en verificación de integridad")
        
        # Información de claves
        key_info = mneme.get_key_info()
        print(f"  Información de claves: {key_info}")
        
        # Rotar claves manualmente
        rotated = mneme.rotate_keys()
        if rotated:
            print("  ✓ Rotación de claves exitosa")
        
        # Verificar que los datos siguen siendo accesibles
        print("  Verificando acceso a datos después de rotación...")
        for name in sensitive_data.keys():
            try:
                loaded_data = mneme.load(name)
                print(f"    ✓ {name} cargado exitosamente")
            except Exception as e:
                print(f"    ✗ Error cargando {name}: {e}")

if __name__ == "__main__":
    print("MNEME - Demostración de Cifrado Avanzado y Gestión de Claves")
    print("=" * 70)
    
    try:
        demo_tensor_encryption()
        demo_key_rotation()
        demo_async_context()
        demo_resource_monitoring()
        demo_advanced_security()
        
        print("\n" + "=" * 70)
        print("✓ Demostración completada exitosamente")
        print("\nNuevas características implementadas:")
        print("- Cifrado específico para tensores (AES-GCM, AES-CBC, ChaCha20, Block-Chain)")
        print("- Rotación automática de claves (tiempo, uso, adaptativa)")
        print("- Contexto asíncrono con gestión de recursos")
        print("- Monitoreo de memoria y GPU en tiempo real")
        print("- Gestión avanzada de claves con versionado")
        print("- Operaciones concurrentes con semáforos")
        
    except Exception as e:
        print(f"\n✗ Error durante la demostración: {e}")
        import traceback
        traceback.print_exc()
