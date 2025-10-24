#!/usr/bin/env python3
"""
Ejemplo de uso del sistema de almacenamiento avanzado y cache en MNEME
"""

import torch
import numpy as np
import time
import asyncio
from datetime import timedelta
from mneme_core import (
    MnemeConfig, 
    ZSpace, 
    StorageBackend,
    CachePolicy,
    CompressionStrategy,
    mneme_context,
    async_mneme_context
)

def demo_storage_backends():
    """Demostrar diferentes backends de almacenamiento."""
    print("=== Demostración de Backends de Almacenamiento ===")
    
    backends = [
        (StorageBackend.MEMORY, "Solo en memoria"),
        (StorageBackend.DISK, "Almacenamiento en disco"),
        (StorageBackend.HYBRID, "Híbrido (memoria + disco)")
    ]
    
    # Crear datos de prueba
    test_data = {
        "small_tensor": torch.randn(100, 100),
        "medium_tensor": torch.randn(500, 500),
        "large_tensor": torch.randn(1000, 1000),
        "sparse_tensor": torch.zeros(800, 800),
        "mixed_data": {
            "weights": torch.randn(200, 200),
            "metadata": {"epoch": 10, "loss": 0.5}
        }
    }
    
    for backend, description in backends:
        print(f"\n--- {description} ---")
        
        config = MnemeConfig(
            storage_backend=backend,
            storage_path="./test_storage" if backend != StorageBackend.MEMORY else None,
            cache_size_mb=512,
            enable_persistent_storage=True
        )
        
        with mneme_context(config) as mneme:
            # Medir tiempo de almacenamiento
            start_time = time.time()
            descriptors = {}
            
            for name, data in test_data.items():
                desc = mneme.register(name, data)
                descriptors[name] = desc
                print(f"  Registrado {name}: {len(desc.core_data)} bytes")
            
            storage_time = time.time() - start_time
            
            # Medir tiempo de carga
            start_time = time.time()
            for name in test_data.keys():
                loaded_data = mneme.load(name)
            
            load_time = time.time() - start_time
            
            # Obtener estadísticas
            stats = mneme.get_storage_stats()
            print(f"  Tiempo de almacenamiento: {storage_time:.4f}s")
            print(f"  Tiempo de carga: {load_time:.4f}s")
            print(f"  Entradas almacenadas: {stats.get('entries', 'N/A')}")
            print(f"  Tamaño total: {stats.get('total_size_bytes', 'N/A')} bytes")

def demo_cache_policies():
    """Demostrar diferentes políticas de cache."""
    print("\n=== Demostración de Políticas de Cache ===")
    
    policies = [
        (CachePolicy.LRU, "Least Recently Used"),
        (CachePolicy.LFU, "Least Frequently Used"),
        (CachePolicy.FIFO, "First In, First Out"),
        (CachePolicy.TTL, "Time To Live"),
        (CachePolicy.ADAPTIVE, "Adaptativa")
    ]
    
    # Crear datos de prueba con patrones de acceso diferentes
    test_tensors = {
        f"tensor_{i}": torch.randn(200, 200) for i in range(20)
    }
    
    for policy, description in policies:
        print(f"\n--- {description} ---")
        
        config = MnemeConfig(
            storage_backend=StorageBackend.MEMORY,
            cache_policy=policy,
            cache_size_mb=100,  # Cache pequeño para forzar evicción
            cache_ttl_seconds=5 if policy == CachePolicy.TTL else 3600
        )
        
        with mneme_context(config) as mneme:
            # Registrar todos los tensores
            for name, tensor in test_tensors.items():
                mneme.register(name, tensor)
            
            # Simular patrones de acceso
            access_patterns = [
                # Patrón 1: Acceso secuencial
                list(range(20)),
                # Patrón 2: Acceso a elementos específicos repetidamente
                [0, 1, 2] * 5,
                # Patrón 3: Acceso aleatorio
                np.random.randint(0, 20, 15).tolist()
            ]
            
            for pattern_name, pattern in [("Secuencial", access_patterns[0]), 
                                       ("Repetitivo", access_patterns[1]), 
                                       ("Aleatorio", access_patterns[2])]:
                
                # Limpiar cache para cada patrón
                mneme.advanced_cache.clear()
                
                start_time = time.time()
                for idx in pattern:
                    name = f"tensor_{idx}"
                    mneme.load(name)
                
                access_time = time.time() - start_time
                
                # Obtener estadísticas del cache
                cache_stats = mneme.advanced_cache.get_stats()
                
                print(f"    {pattern_name}: {access_time:.4f}s, "
                      f"Hit rate: {cache_stats.get('hit_rate', 0):.1f}%, "
                      f"Evicciones: {cache_stats.get('evictions', 0)}")

def demo_deduplication():
    """Demostrar sistema de deduplicación."""
    print("\n=== Demostración de Deduplicación ===")
    
    config = MnemeConfig(
        storage_backend=StorageBackend.DISK,
        storage_path="./test_deduplication",
        enable_deduplication=True,
        cache_size_mb=256
    )
    
    with mneme_context(config) as mneme:
        # Crear datos con contenido duplicado
        base_tensor = torch.randn(100, 100)
        
        # Registrar el mismo tensor con diferentes nombres
        print("  Registrando mismo tensor con diferentes nombres...")
        for i in range(5):
            name = f"duplicate_{i}"
            desc = mneme.register(name, base_tensor)
            print(f"    {name}: {len(desc.core_data)} bytes")
        
        # Registrar tensor similar (pequeña variación)
        similar_tensor = base_tensor + torch.randn(100, 100) * 0.01
        desc_similar = mneme.register("similar_tensor", similar_tensor)
        print(f"    similar_tensor: {len(desc_similar.core_data)} bytes")
        
        # Obtener estadísticas de deduplicación
        dedup_stats = mneme.deduplication_engine.get_stats()
        print(f"  Contenidos únicos: {dedup_stats.get('unique_contents', 0)}")
        print(f"  Referencias totales: {dedup_stats.get('total_references', 0)}")
        print(f"  Ratio de deduplicación: {dedup_stats.get('deduplication_ratio', 1):.2f}")
        
        # Verificar que todos los tensores se pueden cargar correctamente
        print("  Verificando acceso a datos...")
        for i in range(5):
            name = f"duplicate_{i}"
            loaded_tensor = mneme.load(name)
            is_correct = torch.allclose(base_tensor, loaded_tensor, atol=1e-6)
            print(f"    {name}: {'✓' if is_correct else '✗'}")

def demo_compression_strategies():
    """Demostrar diferentes estrategias de compresión."""
    print("\n=== Demostración de Estrategias de Compresión ===")
    
    strategies = [
        (CompressionStrategy.NONE, "Sin compresión"),
        (CompressionStrategy.LZ4, "LZ4 rápido"),
        (CompressionStrategy.ADAPTIVE, "Adaptativa")
    ]
    
    # Crear datos con diferentes características de compresión
    test_cases = [
        ("Datos aleatorios", torch.randn(500, 500)),
        ("Datos con patrones", torch.ones(500, 500) + torch.randn(500, 500) * 0.1),
        ("Datos dispersos", torch.zeros(500, 500)),
        ("Datos repetitivos", torch.tile(torch.randn(50, 50), (10, 10)))
    ]
    
    for strategy, description in strategies:
        print(f"\n--- {description} ---")
        
        config = MnemeConfig(
            storage_backend=StorageBackend.DISK,
            storage_path=f"./test_compression_{strategy.value}",
            compression_strategy=strategy,
            enable_adaptive_compression=(strategy == CompressionStrategy.ADAPTIVE)
        )
        
        with mneme_context(config) as mneme:
            for case_name, tensor in test_cases:
                # Medir tiempo de compresión
                start_time = time.time()
                desc = mneme.register(case_name, tensor)
                compression_time = time.time() - start_time
                
                # Calcular ratio de compresión
                original_size = tensor.nelement() * tensor.element_size()
                compressed_size = len(desc.core_data)
                compression_ratio = compressed_size / original_size
                
                print(f"  {case_name}:")
                print(f"    Tiempo: {compression_time:.4f}s")
                print(f"    Tamaño original: {original_size} bytes")
                print(f"    Tamaño comprimido: {compressed_size} bytes")
                print(f"    Ratio: {compression_ratio:.3f}")

def demo_storage_optimization():
    """Demostrar optimización de almacenamiento."""
    print("\n=== Demostración de Optimización de Almacenamiento ===")
    
    config = MnemeConfig(
        storage_backend=StorageBackend.HYBRID,
        storage_path="./test_optimization",
        cache_policy=CachePolicy.ADAPTIVE,
        enable_deduplication=True,
        cache_size_mb=128
    )
    
    with mneme_context(config) as mneme:
        # Crear datos de prueba
        print("  Creando datos de prueba...")
        for i in range(50):
            tensor = torch.randn(100, 100)
            mneme.register(f"tensor_{i}", tensor)
        
        # Obtener estadísticas antes de optimización
        stats_before = mneme.get_storage_stats()
        health_before = mneme.get_storage_health()
        
        print(f"  Antes de optimización:")
        print(f"    Entradas: {stats_before.get('entries', 0)}")
        print(f"    Tamaño: {stats_before.get('total_size_bytes', 0)} bytes")
        print(f"    Salud: {health_before.get('health_score', 0)}/100")
        
        # Ejecutar optimización
        print("  Ejecutando optimización...")
        optimization_results = mneme.optimize_storage()
        
        # Obtener estadísticas después de optimización
        stats_after = mneme.get_storage_stats()
        health_after = mneme.get_storage_health()
        
        print(f"  Después de optimización:")
        print(f"    Entradas: {stats_after.get('entries', 0)}")
        print(f"    Tamaño: {stats_after.get('total_size_bytes', 0)} bytes")
        print(f"    Salud: {health_after.get('health_score', 0)}/100")
        print(f"    Espacio ahorrado: {optimization_results.get('space_saved_bytes', 0)} bytes")
        
        # Mostrar resultados de optimización
        print("  Resultados de optimización:")
        for key, value in optimization_results.items():
            if isinstance(value, bool):
                print(f"    {key}: {'✓' if value else '✗'}")
            else:
                print(f"    {key}: {value}")

def demo_async_storage():
    """Demostrar almacenamiento asíncrono."""
    print("\n=== Demostración de Almacenamiento Asíncrono ===")
    
    async def async_operations():
        config = MnemeConfig(
            storage_backend=StorageBackend.HYBRID,
            storage_path="./test_async",
            enable_async_context=True,
            max_concurrent_operations=5,
            cache_policy=CachePolicy.ADAPTIVE
        )
        
        async with async_mneme_context(config) as mneme:
            print("  Contexto asíncrono inicializado")
            
            # Crear tareas concurrentes de almacenamiento
            tasks = []
            for i in range(10):
                tensor = torch.randn(200, 200)
                task = mneme.register_async(f"async_tensor_{i}", tensor)
                tasks.append(task)
            
            # Ejecutar todas las tareas
            print("  Ejecutando operaciones concurrentes...")
            start_time = time.time()
            descriptors = await asyncio.gather(*tasks)
            storage_time = time.time() - start_time
            
            print(f"  Tiempo de almacenamiento concurrente: {storage_time:.4f}s")
            print(f"  Operaciones activas: {await mneme.active_operations_count}")
            
            # Cargar tensores de forma asíncrona
            load_tasks = []
            for i in range(10):
                task = mneme.load_async(f"async_tensor_{i}")
                load_tasks.append(task)
            
            start_time = time.time()
            loaded_tensors = await asyncio.gather(*load_tasks)
            load_time = time.time() - start_time
            
            print(f"  Tiempo de carga concurrente: {load_time:.4f}s")
            
            # Obtener estadísticas asíncronas
            stats = await mneme.get_stats_async()
            print(f"  Estadísticas: {len(stats.get('descriptors', []))} descriptores")
    
    # Ejecutar operaciones asíncronas
    asyncio.run(async_operations())

def demo_storage_health_monitoring():
    """Demostrar monitoreo de salud del almacenamiento."""
    print("\n=== Demostración de Monitoreo de Salud ===")
    
    config = MnemeConfig(
        storage_backend=StorageBackend.HYBRID,
        storage_path="./test_health",
        cache_policy=CachePolicy.ADAPTIVE,
        enable_deduplication=True,
        cache_size_mb=64  # Cache pequeño para simular presión
    )
    
    with mneme_context(config) as mneme:
        # Simular diferentes condiciones de salud
        scenarios = [
            ("Condición normal", 20),
            ("Alta carga", 100),
            ("Cache saturado", 200)
        ]
        
        for scenario_name, num_tensors in scenarios:
            print(f"\n  --- {scenario_name} ---")
            
            # Limpiar cache para cada escenario
            mneme.advanced_cache.clear()
            
            # Crear tensores
            for i in range(num_tensors):
                tensor = torch.randn(50, 50)
                mneme.register(f"health_tensor_{i}", tensor)
            
            # Obtener métricas de salud
            health = mneme.get_storage_health()
            
            print(f"    Puntuación de salud: {health.get('health_score', 0)}/100")
            print(f"    Estado: {health.get('status', 'unknown')}")
            print(f"    Uso de cache: {health.get('cache_usage_percent', 0):.1f}%")
            print(f"    Tasa de aciertos: {health.get('cache_hit_rate', 0):.1f}%")
            print(f"    Ratio de deduplicación: {health.get('deduplication_ratio', 1):.2f}")
            
            if health.get('warnings'):
                print(f"    Advertencias: {', '.join(health['warnings'])}")

if __name__ == "__main__":
    print("MNEME - Demostración de Sistema de Almacenamiento Avanzado")
    print("=" * 70)
    
    try:
        demo_storage_backends()
        demo_cache_policies()
        demo_deduplication()
        demo_compression_strategies()
        demo_storage_optimization()
        demo_async_storage()
        demo_storage_health_monitoring()
        
        print("\n" + "=" * 70)
        print("✓ Demostración completada exitosamente")
        print("\nCaracterísticas del sistema de almacenamiento implementadas:")
        print("- Múltiples backends (Memoria, Disco, Híbrido, Redis)")
        print("- Políticas de cache avanzadas (LRU, LFU, FIFO, TTL, Adaptativa)")
        print("- Sistema de deduplicación automática")
        print("- Estrategias de compresión adaptativa")
        print("- Optimización automática de almacenamiento")
        print("- Monitoreo de salud en tiempo real")
        print("- Operaciones asíncronas concurrentes")
        
    except Exception as e:
        print(f"\n✗ Error durante la demostración: {e}")
        import traceback
        traceback.print_exc()
