"""
MNEME Usage Examples v2.0
Ejemplos prácticos del sistema MNEME con todas las funcionalidades avanzadas
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, Any, List

# Importar módulos MNEME actualizados
from mneme import (
    ZSpace, DecompType, CompressionLevel, SecurityLevel,
    ZLinear, ZConv2d, ZAttention, ZTransformerBlock, 
    compress_model, get_compression_stats, get_model_performance_stats,
    CompressionConfig, optimize_model_memory, get_system_metrics,
    get_health_status, optimize_system, MNEMEOptimizer, OptimizationLevel,
    SecurityManager, encrypt_data, decrypt_data
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Uso básico de MNEME con todas las funcionalidades"""
    print("="*60)
    print("EJEMPLO BÁSICO DE MNEME v2.0")
    print("="*60)
    
    # Inicializar MNEME con configuración avanzada
    mneme = ZSpace()
    
    # Crear tensor de ejemplo
    tensor = torch.randn(1024, 1024)
    print(f"Tensor original: {tensor.shape}, {tensor.numel() * 4 / 1024:.1f}KB")
    
    # Registrar con diferentes configuraciones de compresión
    configs = [
        {"target_ratio": 0.1, "decomp_type": DecompType.TT},
        {"target_ratio": 0.05, "decomp_type": DecompType.SVD},
        {"target_ratio": 0.2, "decomp_type": DecompType.CP}
    ]
    
    for i, config in enumerate(configs):
        desc = mneme.register(f"tensor_{i}", tensor, **config)
        loaded = mneme.load(f"tensor_{i}")
        
        error = torch.norm(tensor - loaded) / torch.norm(tensor)
        ratio = desc.meta.get('compression_ratio', 1.0)
        
        print(f"Config {i+1}: Ratio={ratio:.3f}, Error={error:.6f}, "
              f"Tipo={desc.decomp_type.value}")
    
    # Estadísticas del sistema
    stats = mneme.get_stats()
    print(f"\nEstadísticas del sistema:")
    print(f"- Descriptores: {stats['descriptors']}")
    print(f"- Versiones: {stats['versions']}")
    print(f"- Cache hits: {stats['performance']['cache']['hits']}")
    print(f"- Cache misses: {stats['performance']['cache']['misses']}")

def example_parallel_processing():
    """Ejemplo de procesamiento en paralelo"""
    print("\n" + "="*60)
    print("PROCESAMIENTO EN PARALELO")
    print("="*60)
    
    mneme = ZSpace()
    
    # Crear múltiples tensores para procesamiento paralelo
    tensors = [torch.randn(500, 500) for _ in range(8)]
    print(f"Procesando {len(tensors)} tensores en paralelo...")
    
    # Registrar con procesamiento paralelo
    start_time = time.time()
    descriptors = []
    for i, tensor in enumerate(tensors):
        desc = mneme.register_parallel(f"parallel_tensor_{i}", tensor, 
                                       target_ratio=0.1, decomp_type=DecompType.TT)
        descriptors.append(desc)
    parallel_time = time.time() - start_time
    
    # Cargar con optimizaciones paralelas
    start_time = time.time()
    loaded_tensors = []
    for i in range(len(tensors)):
        loaded = mneme.load_parallel(f"parallel_tensor_{i}")
        loaded_tensors.append(loaded)
    load_time = time.time() - start_time
    
    # Verificar precisión
    total_error = 0
    for i, (original, loaded) in enumerate(zip(tensors, loaded_tensors)):
        error = torch.norm(original - loaded) / torch.norm(original)
        total_error += error
    
    avg_error = total_error / len(tensors)
    avg_ratio = sum(desc.meta.get('compression_ratio', 1.0) for desc in descriptors) / len(descriptors)
    
    print(f"Resultados del procesamiento paralelo:")
    print(f"- Tiempo de registro: {parallel_time:.3f}s")
    print(f"- Tiempo de carga: {load_time:.3f}s")
    print(f"- Ratio promedio: {avg_ratio:.3f}")
    print(f"- Error promedio: {avg_error:.6f}")
    
    # Métricas de paralelización
    parallel_metrics = mneme.get_parallel_metrics()
    print(f"- Operaciones paralelas: {parallel_metrics['thread_operations'] + parallel_metrics['process_operations']}")
    print(f"- Eficiencia: {parallel_metrics['parallel_efficiency']:.2%}")

def example_advanced_security():
    """Ejemplo de seguridad avanzada"""
    print("\n" + "="*60)
    print("SEGURIDAD AVANZADA")
    print("="*60)
    
    # Crear tensor sensible
    sensitive_tensor = torch.randn(100, 100)
    print(f"Tensor sensible: {sensitive_tensor.shape}")
    
    # Cifrar tensor
    encrypted_data, metadata = mneme.encrypt_tensor(sensitive_tensor)
    print(f"Datos cifrados: {len(encrypted_data)} bytes")
    print(f"Metadata: {list(metadata.keys())}")
    
    # Descifrar tensor
    decrypted_tensor = mneme.decrypt_tensor(encrypted_data, metadata)
    print(f"Tensor descifrado: {decrypted_tensor.shape}")
    
    # Verificar integridad
    integrity_ok = torch.allclose(sensitive_tensor, decrypted_tensor, atol=1e-6)
    print(f"Integridad verificada: {'✓' if integrity_ok else '✗'}")
    
    # Autenticación de usuario
    credentials = {"username": "test_user", "password": "test_pass"}
    session_id = mneme.authenticate_user(credentials)
    print(f"Sesión autenticada: {session_id[:8]}...")
    
    # Métricas de seguridad
    security_metrics = mneme.get_security_metrics()
    print(f"Métricas de seguridad:")
    print(f"- Operaciones de cifrado: {security_metrics['encryption_operations']}")
    print(f"- Intentos de autenticación: {security_metrics['authentication_attempts']}")
    print(f"- Rotaciones de clave: {security_metrics['key_rotations']}")

def example_advanced_storage():
    """Ejemplo de almacenamiento avanzado"""
    print("\n" + "="*60)
    print("ALMACENAMIENTO AVANZADO")
    print("="*60)
    
    # Crear tensores de diferentes tamaños
    small_tensor = torch.randn(100, 100)  # Memoria
    medium_tensor = torch.randn(1000, 1000)  # SSD
    large_tensor = torch.randn(5000, 5000)  # HDD
    
    tensors = {
        "small": small_tensor,
        "medium": medium_tensor,
        "large": large_tensor
    }
    
    for name, tensor in tensors.items():
        print(f"\nProcesando tensor {name}: {tensor.shape}")
        
        # Registrar con almacenamiento automático
        desc = mneme.register(name, tensor, target_ratio=0.1)
        
        # Simular accesos para migración de niveles
        for _ in range(5):
            loaded = mneme.load(name)
        
        print(f"  Ratio de compresión: {desc.meta.get('compression_ratio', 1.0):.3f}")
        print(f"  Tamaño comprimido: {len(desc.core_data)/1024:.1f}KB")
    
    # Métricas de almacenamiento
    storage_metrics = mneme.get_storage_metrics()
    print(f"\nMétricas de almacenamiento:")
    print(f"- Operaciones de lectura: {storage_metrics.get('read_operations', 0)}")
    print(f"- Operaciones de escritura: {storage_metrics.get('write_operations', 0)}")
    print(f"- Cache hits: {storage_metrics.get('cache_hits', 0)}")
    print(f"- Cache misses: {storage_metrics.get('cache_misses', 0)}")

def example_model_compression():
    """Compresión de modelos con MNEME v2.0"""
    print("\n" + "="*60)
    print("COMPRESIÓN DE MODELOS v2.0")
    print("="*60)
    
    # Crear modelo complejo
    class ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    model = ComplexModel()
    print(f"Modelo original: {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Configuración de compresión avanzada
    config = CompressionConfig(
        target_ratio=0.1,
        compression_level=CompressionLevel.HIGH,
        memory_limit=50 * 1024 * 1024,  # 50MB
        use_parallel_processing=True,
        enable_security=True
    )
    
    # Comprimir modelo
    compressed_model = compress_model(model, config=config, min_params=1000)
    
    # Estadísticas de compresión
    stats = get_compression_stats(compressed_model)
    print(f"\nEstadísticas de compresión:")
    print(f"- Parámetros originales: {stats['original_params']:,.0f}")
    print(f"- Parámetros comprimidos: {stats['compressed_params']:,.0f}")
    print(f"- Ratio general: {stats['overall_ratio']:.3f}")
    print(f"- Capas comprimidas: {stats['compressed_layers']}/{stats['total_layers']}")
    print(f"- Ratio promedio: {stats['avg_compression_ratio']:.3f}")
    
    # Probar inferencia
    x = torch.randn(4, 3, 32, 32)
    
    with torch.no_grad():
        start_time = time.time()
        original_out = model(x)
        original_time = time.time() - start_time
        
        start_time = time.time()
        compressed_out = compressed_model(x)
        compressed_time = time.time() - start_time
    
    diff = torch.norm(original_out - compressed_out) / torch.norm(original_out)
    print(f"\nInferencia:")
    print(f"- Tiempo original: {original_time:.4f}s")
    print(f"- Tiempo comprimido: {compressed_time:.4f}s")
    print(f"- Diferencia de salida: {diff:.6f}")
    
    # Estadísticas de rendimiento del modelo
    perf_stats = get_model_performance_stats(compressed_model)
    print(f"\nEstadísticas de rendimiento:")
    print(f"- Tiempo total de forward: {perf_stats['total_forward_time']:.4f}s")
    print(f"- Conteo total de forward: {perf_stats['total_forward_count']}")
    print(f"- Tiempo promedio: {perf_stats['avg_forward_time']:.4f}s")

def example_transformer_compression():
    """Compresión de modelo Transformer con MNEME v2.0"""
    print("\n" + "="*60)
    print("COMPRESIÓN DE TRANSFORMER v2.0")
    print("="*60)
    
    # Crear Transformer simple con capas MNEME
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=1000, embed_dim=512, num_heads=8, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
            
            # Configuración de compresión para capas
            config = CompressionConfig(
                target_ratio=0.1,
                use_parallel_processing=True,
                enable_security=True
            )
            
            self.layers = nn.ModuleList([
                ZTransformerBlock(embed_dim, num_heads, config=config)
                for _ in range(num_layers)
            ])
            
            self.norm = nn.LayerNorm(embed_dim)
            self.output = ZLinear(embed_dim, vocab_size, config=config)
        
        def forward(self, x):
            x = self.embedding(x) + self.pos_encoding[:x.size(1)]
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.output(x)
    
    model = SimpleTransformer()
    print(f"Transformer original: {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Estadísticas de compresión
    stats = get_compression_stats(model)
    print(f"Estadísticas de compresión:")
    print(f"- Parámetros originales: {stats['original_params']:,.0f}")
    print(f"- Parámetros comprimidos: {stats['compressed_params']:,.0f}")
    print(f"- Ratio general: {stats['overall_ratio']:.3f}")
    
    # Probar inferencia
    x = torch.randint(0, 1000, (4, 128))
    
    with torch.no_grad():
        start_time = time.time()
        output = model(x)
        inference_time = time.time() - start_time
    
    print(f"Inferencia: {inference_time:.4f}s")
    print(f"Salida: {output.shape}")

def example_optimization_system():
    """Ejemplo de sistema de optimización"""
    print("\n" + "="*60)
    print("SISTEMA DE OPTIMIZACIÓN")
    print("="*60)
    
    # Crear optimizador
    optimizer = MNEMEOptimizer(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_profiling=True,
        enable_parallel_processing=True
    )
    
    # Crear tensores para optimización
    tensors = [torch.randn(1000, 1000) for _ in range(5)]
    print(f"Optimizando {len(tensors)} tensores...")
    
    # Optimizar tensores
    start_time = time.time()
    optimized_tensors = optimizer.optimize_tensor_operations(tensors)
    optimization_time = time.time() - start_time
    
    print(f"Tiempo de optimización: {optimization_time:.3f}s")
    
    # Reporte de optimización
    report = optimizer.get_optimization_report()
    print(f"\nReporte de optimización:")
    print(f"- Nivel: {report['optimization_level']}")
    print(f"- Métricas de rendimiento: {len(report['performance_metrics'])} categorías")
    print(f"- Optimizaciones de recursos: {len(report['resource_optimization'])} estrategias")
    print(f"- Recomendaciones: {len(report['recommendations'])}")
    
    # Estado de salud del sistema
    health = optimizer.get_health_status()
    print(f"Estado de salud: {health}")
    
    # Optimizar sistema completo
    system_optimization = optimizer.optimize_system()
    print(f"\nOptimizaciones del sistema:")
    for strategy, result in system_optimization.items():
        if 'optimizations' in result:
            print(f"- {strategy}: {len(result['optimizations'])} optimizaciones")

def example_performance_monitoring():
    """Ejemplo de monitoreo de rendimiento"""
    print("\n" + "="*60)
    print("MONITOREO DE RENDIMIENTO")
    print("="*60)
    
    # Obtener métricas del sistema
    metrics = get_system_metrics()
    print(f"Métricas del sistema:")
    print(f"- Operaciones totales: {metrics['metrics']['operations']['total']}")
    print(f"- Operaciones exitosas: {metrics['metrics']['operations']['successful']}")
    print(f"- Operaciones fallidas: {metrics['metrics']['operations']['failed']}")
    print(f"- Uso de memoria: {metrics['metrics']['memory']['current_usage']/1024/1024:.1f}MB")
    print(f"- Presión de memoria: {metrics['metrics']['memory']['memory_pressure']:.1%}")
    
    # Estado de salud
    health = get_health_status()
    print(f"Estado de salud: {health}")
    
    # Optimizar sistema
    optimization_result = optimize_system()
    print(f"\nResultado de optimización:")
    for strategy, result in optimization_result.items():
        if isinstance(result, dict) and 'optimizations' in result:
            print(f"- {strategy}: {len(result['optimizations'])} optimizaciones aplicadas")

def example_memory_optimization():
    """Ejemplo de optimización de memoria"""
    print("\n" + "="*60)
    print("OPTIMIZACIÓN DE MEMORIA")
    print("="*60)
    
    # Crear modelo grande
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(1000, 1000) for _ in range(10)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            return x
    
    model = LargeModel()
    original_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Memoria original: {original_memory:.1f}MB")
    
    # Optimizar para memoria específica
    target_memory = 50  # 50MB
    optimized_model = optimize_model_memory(model, target_memory_mb=target_memory)
    
    new_memory = sum(p.numel() * p.element_size() for p in optimized_model.parameters()) / (1024 * 1024)
    print(f"Memoria optimizada: {new_memory:.1f}MB")
    print(f"Reducción: {(1 - new_memory/original_memory)*100:.1f}%")
    
    # Estadísticas de rendimiento
    x = torch.randn(32, 1000)
    
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):
            _ = optimized_model(x)
        avg_time = (time.time() - start_time) / 100
    
    print(f"Tiempo promedio de inferencia: {avg_time*1000:.2f}ms")

def example_incremental_updates():
    """Ejemplo de actualizaciones incrementales"""
    print("\n" + "="*60)
    print("ACTUALIZACIONES INCREMENTALES")
    print("="*60)
    
    mneme = ZSpace()
    
    # Estado inicial
    state = torch.zeros(100, 100)
    desc = mneme.register("game_state", state)
    print(f"Estado inicial: versión {desc.version}")
    
    # Simular actualizaciones del juego
    for step in range(10):
        # Actualización dispersa
        indices = torch.randint(0, 100, (20, 2))
        values = torch.randn(20)
        
        delta = {
            "type": "sparse_update",
            "indices": indices,
            "values": values
        }
        
        desc = mneme.update("game_state", delta)
        print(f"Paso {step+1}: versión {desc.version}, {len(indices)} actualizaciones")
    
    # Cargar estado final
    final_state = mneme.load("game_state")
    non_zero = (final_state != 0).sum().item()
    print(f"\nEstado final: {non_zero} elementos no cero")
    
    # Estadísticas de versiones
    stats = mneme.get_stats()
    print(f"Versiones creadas: {stats['versions']}")

def example_performance_benchmark():
    """Benchmark de rendimiento"""
    print("\n" + "="*60)
    print("BENCHMARK DE RENDIMIENTO")
    print("="*60)
    
    mneme = ZSpace()
    
    # Diferentes tamaños de tensor
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for size in sizes:
        print(f"\nTamaño: {size}")
        
        # Crear tensor
        tensor = torch.randn(size)
        original_size = tensor.numel() * 4 / 1024  # KB
        
        # Medir tiempo de almacenamiento
        start_time = time.time()
        desc = mneme.register(f"tensor_{size}", tensor)
        store_time = time.time() - start_time
        
        # Medir tiempo de carga
        start_time = time.time()
        loaded = mneme.load(f"tensor_{size}")
        load_time = time.time() - start_time
        
        # Verificar precisión
        error = torch.norm(tensor - loaded) / torch.norm(tensor)
        ratio = desc.meta.get('compression_ratio', 1.0)
        
        print(f"  Tamaño original: {original_size:.1f}KB")
        print(f"  Tamaño comprimido: {len(desc.core_data)/1024:.1f}KB")
        print(f"  Ratio: {ratio:.3f}")
        print(f"  Error: {error:.6f}")
        print(f"  Tiempo almacenamiento: {store_time*1000:.2f}ms")
        print(f"  Tiempo carga: {load_time*1000:.2f}ms")

def main():
    """Función principal con todos los ejemplos"""
    print("🧠 MNEME v2.0 - Motor de Memoria Neural Mórfica")
    print("Ejemplos de uso avanzado con nuevas funcionalidades\n")
    
    try:
        example_basic_usage()
        example_parallel_processing()
        example_advanced_security()
        example_advanced_storage()
        example_model_compression()
        example_transformer_compression()
        example_optimization_system()
        example_performance_monitoring()
        example_memory_optimization()
        example_incremental_updates()
        example_performance_benchmark()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS EJEMPLOS COMPLETADOS")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error en ejemplos: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()