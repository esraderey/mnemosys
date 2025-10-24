#!/usr/bin/env python3
"""
Ejemplo de MNEME v2.0 con Locks Granulares, Safetensors, Lazy Decompression y Cache Adaptativo
"""

import torch
import time
import numpy as np
from mneme import ZSpace, MnemeConfig, CompressionLevel, SecurityLevel, LockType, GranularLockManager, LazyTensor, AdaptiveCache

def demonstrate_granular_locks():
    """Demostrar locks granulares"""
    print("🔒 Demostrando Locks Granulares")
    print("=" * 50)
    
    config = MnemeConfig(
        cache_size_mb=512,
        security_level=SecurityLevel.HIGH,
        compression_level=CompressionLevel.BALANCED
    )
    
    mneme = ZSpace(config)
    
    # Crear tensores para demostrar locks granulares
    tensors = {
        'tensor1': torch.randn(1000, 1000),
        'tensor2': torch.randn(1000, 1000),
        'tensor3': torch.randn(1000, 1000)
    }
    
    # Registrar tensores con locks granulares
    start_time = time.time()
    for name, tensor in tensors.items():
        desc = mneme.register(name, tensor)
        print(f"✅ Registrado {name} con ratio {desc.meta['compression_ratio']:.3f}")
    
    registration_time = time.time() - start_time
    print(f"⏱️ Tiempo de registro: {registration_time:.3f}s")
    
    # Obtener estadísticas de locks
    lock_stats = mneme.lock_manager.get_lock_stats()
    print(f"📊 Estadísticas de locks: {lock_stats}")
    
    return mneme

def demonstrate_safetensors():
    """Demostrar serialización con safetensors"""
    print("\n🛡️ Demostrando Safetensors")
    print("=" * 50)
    
    # Crear tensor de prueba
    tensor = torch.randn(500, 500)
    
    # Simular serialización con safetensors
    import io
    from safetensors.torch import save_file, load_file
    
    # Serializar con safetensors
    start_time = time.time()
    buffer = io.BytesIO()
    save_file({"tensor": tensor}, buffer)
    safetensors_data = buffer.getvalue()
    serialize_time = time.time() - start_time
    
    # Deserializar con safetensors
    start_time = time.time()
    buffer = io.BytesIO(safetensors_data)
    loaded_data = load_file(buffer)
    deserialize_time = time.time() - start_time
    
    print(f"📦 Tamaño serializado: {len(safetensors_data)} bytes")
    print(f"⏱️ Tiempo de serialización: {serialize_time*1000:.2f}ms")
    print(f"⏱️ Tiempo de deserialización: {deserialize_time*1000:.2f}ms")
    print(f"✅ Verificación: {torch.allclose(tensor, loaded_data['tensor'])}")

def demonstrate_lazy_decompression(mneme):
    """Demostrar lazy decompression"""
    print("\n🔄 Demostrando Lazy Decompression")
    print("=" * 50)
    
    # Crear tensor grande
    large_tensor = torch.randn(2000, 2000)
    
    # Registrar con lazy decompression
    start_time = time.time()
    desc = mneme.register("large_tensor", large_tensor)
    registration_time = time.time() - start_time
    
    print(f"📝 Registrado en {registration_time:.3f}s")
    print(f"🗜️ Ratio de compresión: {desc.meta['compression_ratio']:.3f}")
    
    # Verificar si tiene lazy tensor
    if hasattr(desc, 'lazy_tensor') and desc.lazy_tensor:
        print("✅ Lazy tensor disponible")
        
        # Obtener estadísticas de memoria
        memory_stats = desc.lazy_tensor.get_memory_usage()
        print(f"💾 Memoria comprimida: {memory_stats['compressed_bytes']/1024/1024:.2f}MB")
        print(f"💾 Ratio de compresión: {memory_stats['compression_ratio']:.3f}")
        print(f"🔄 Decompressed: {memory_stats['is_decompressed']}")
        
        # Decomprimir solo cuando sea necesario
        start_time = time.time()
        decompressed = desc.lazy_tensor.decompress()
        decompression_time = time.time() - start_time
        
        print(f"⏱️ Tiempo de decompresión: {decompression_time:.3f}s")
        print(f"✅ Verificación: {torch.allclose(large_tensor, decompressed)}")
        
        # Limpiar memoria decompressed
        desc.lazy_tensor.clear_decompressed()
        print("🧹 Memoria decompressed liberada")

def demonstrate_adaptive_cache(mneme):
    """Demostrar cache adaptativo"""
    print("\n🧠 Demostrando Cache Adaptativo")
    print("=" * 50)
    
    # Obtener estadísticas del cache
    cache_stats = mneme.adaptive_cache.get_stats()
    print(f"📊 Estadísticas del cache:")
    print(f"  - Tamaño: {cache_stats['size_bytes']/1024/1024:.2f}MB")
    print(f"  - Uso: {cache_stats['usage_percent']:.1f}%")
    print(f"  - Entradas: {cache_stats['entries']}")
    print(f"  - Hit rate: {cache_stats['hit_rate']:.1f}%")
    print(f"  - Estrategia: {cache_stats['strategy']}")
    
    # Cargar tensores para probar cache
    start_time = time.time()
    for i in range(5):
        tensor = mneme.load(f"tensor{(i % 3) + 1}")
    load_time = time.time() - start_time
    
    print(f"⏱️ Tiempo de carga (5 iteraciones): {load_time:.3f}s")
    
    # Obtener estadísticas actualizadas
    updated_stats = mneme.adaptive_cache.get_stats()
    print(f"📈 Hit rate actualizado: {updated_stats['hit_rate']:.1f}%")
    print(f"📈 Hits: {updated_stats['hit_count']}")
    print(f"📈 Misses: {updated_stats['miss_count']}")

def demonstrate_performance_comparison():
    """Comparar rendimiento con y sin las mejoras"""
    print("\n⚡ Comparación de Rendimiento")
    print("=" * 50)
    
    # Configuración estándar
    config_standard = MnemeConfig(
        cache_size_mb=256,
        security_level=SecurityLevel.STANDARD,
        compression_level=CompressionLevel.BALANCED
    )
    
    # Configuración optimizada
    config_optimized = MnemeConfig(
        cache_size_mb=512,
        security_level=SecurityLevel.HIGH,
        compression_level=CompressionLevel.HIGH
    )
    
    # Crear tensores de prueba
    test_tensors = [torch.randn(1000, 1000) for _ in range(10)]
    
    # Probar configuración estándar
    print("🔧 Configuración Estándar:")
    mneme_std = ZSpace(config_standard)
    
    start_time = time.time()
    for i, tensor in enumerate(test_tensors):
        mneme_std.register(f"std_tensor_{i}", tensor)
    std_registration_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(10):
        mneme_std.load(f"std_tensor_{i % 10}")
    std_load_time = time.time() - start_time
    
    print(f"  - Registro: {std_registration_time:.3f}s")
    print(f"  - Carga: {std_load_time:.3f}s")
    
    # Probar configuración optimizada
    print("\n🚀 Configuración Optimizada:")
    mneme_opt = ZSpace(config_optimized)
    
    start_time = time.time()
    for i, tensor in enumerate(test_tensors):
        mneme_opt.register(f"opt_tensor_{i}", tensor)
    opt_registration_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(10):
        mneme_opt.load(f"opt_tensor_{i % 10}")
    opt_load_time = time.time() - start_time
    
    print(f"  - Registro: {opt_registration_time:.3f}s")
    print(f"  - Carga: {opt_load_time:.3f}s")
    
    # Calcular mejoras
    reg_improvement = (std_registration_time - opt_registration_time) / std_registration_time * 100
    load_improvement = (std_load_time - opt_load_time) / std_load_time * 100
    
    print(f"\n📈 Mejoras:")
    print(f"  - Registro: {reg_improvement:.1f}% más rápido")
    print(f"  - Carga: {load_improvement:.1f}% más rápido")
    
    # Limpiar
    mneme_std.cleanup()
    mneme_opt.cleanup()

def main():
    """Función principal"""
    print("🧠 MNEME v2.0 - Funcionalidades Avanzadas")
    print("=" * 60)
    print("Demostrando: Locks Granulares, Safetensors, Lazy Decompression, Cache Adaptativo")
    print("=" * 60)
    
    try:
        # Demostrar locks granulares
        mneme = demonstrate_granular_locks()
        
        # Demostrar safetensors
        demonstrate_safetensors()
        
        # Demostrar lazy decompression
        demonstrate_lazy_decompression(mneme)
        
        # Demostrar cache adaptativo
        demonstrate_adaptive_cache(mneme)
        
        # Comparación de rendimiento
        demonstrate_performance_comparison()
        
        # Limpiar recursos
        mneme.cleanup()
        
        print("\n✅ Todas las demostraciones completadas exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante la demostración: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
