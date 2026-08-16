"""
MNEME Usage Examples v2.0
Ejemplos prácticos del sistema MNEME con todas las funcionalidades avanzadas
"""

import logging
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn

# Importar módulos MNEME actualizados
from mneme import (
    CompressionConfig,
    DecompType,
    MnemeConfig,
    MNEMEOptimizer,
    OptimizationLevel,
    SecureSerializer,
    SecurityLevel,
    ZLinear,
    ZSpace,
    ZTransformerBlock,
    compress_model,
    create_secure_config,
    get_compression_stats,
    get_health_status,
    get_model_performance_stats,
    get_system_metrics,
    optimize_model_memory,
    optimize_system,
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
        loaded = mneme.load(f"tensor_{i}").cpu()

        error = torch.norm(tensor - loaded) / torch.norm(tensor)
        ratio = desc.meta.get('compression_ratio', 1.0)

        print(f"Config {i+1}: Ratio={ratio:.3f}, Error={error:.6f}, "
              f"Tipo={desc.decomp_type.value}")

    # Estadísticas del sistema
    stats = mneme.get_stats()
    cache_stats = stats["cache"]
    print("\nEstadísticas del sistema:")
    print(f"- Salud: {stats['health']}")
    print(f"- Tensores registrados: {stats['metrics']['tensor_count']}")
    print(f"- Cache hits: {cache_stats['hit_count']}")
    print(f"- Cache misses: {cache_stats['miss_count']}")

def example_parallel_processing():
    """Ejemplo de procesamiento en paralelo"""
    print("\n" + "="*60)
    print("PROCESAMIENTO EN PARALELO")
    print("="*60)

    mneme = ZSpace()

    # Crear múltiples tensores para procesamiento concurrente
    tensors = [torch.randn(500, 500) for _ in range(8)]
    print(f"Procesando {len(tensors)} tensores en paralelo...")

    # Registrar concurrentemente: ZSpace es thread-safe (locks granulares por nombre)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        descriptors = list(pool.map(
            lambda item: mneme.register(f"parallel_tensor_{item[0]}", item[1],
                                        target_ratio=0.1),
            enumerate(tensors),
        ))
    parallel_time = time.time() - start_time

    # Cargar concurrentemente
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded_tensors = list(pool.map(
            lambda i: mneme.load(f"parallel_tensor_{i}"),
            range(len(tensors)),
        ))
    load_time = time.time() - start_time

    # Verificar precisión
    total_error = 0
    for original, loaded in zip(tensors, loaded_tensors, strict=True):
        error = torch.norm(original - loaded.cpu()) / torch.norm(original)
        total_error += error

    avg_error = total_error / len(tensors)
    avg_ratio = sum(desc.meta.get('compression_ratio', 1.0) for desc in descriptors) / len(descriptors)

    print("Resultados del procesamiento paralelo:")
    print(f"- Tiempo de registro: {parallel_time:.3f}s")
    print(f"- Tiempo de carga: {load_time:.3f}s")
    print(f"- Ratio promedio: {avg_ratio:.3f}")
    print(f"- Error promedio: {avg_error:.6f}")

    # Métricas de concurrencia (locks granulares por tensor)
    lock_stats = mneme.get_stats()["locks"]
    print(f"- Locks granulares creados: {lock_stats['total_locks']}")
    print(f"- Escritores activos al finalizar: {lock_stats['active_writers']}")

def example_advanced_security():
    """Ejemplo de seguridad avanzada"""
    print("\n" + "="*60)
    print("SEGURIDAD AVANZADA")
    print("="*60)

    # Crear tensor sensible (2-D pequeño: el routing lo serializa RAW, sin pérdida)
    sensitive_tensor = torch.randn(64, 64)
    print(f"Tensor sensible: {sensitive_tensor.shape}")

    # Serialización firmada (HMAC) con el marco seguro de MNEME
    secret_key = b"clave_de_ejemplo_de_32_bytes____"
    serializer = SecureSerializer(create_secure_config(
        security_level=SecurityLevel.HMAC,
        signing_key=secret_key,
    ))

    signed_data = serializer.serialize_tensor(sensitive_tensor)
    print(f"Datos firmados: {len(signed_data)} bytes")

    restored_tensor, _metadata = serializer.deserialize_tensor(signed_data)
    print(f"Tensor restaurado: {restored_tensor.shape}")

    # Verificar integridad
    integrity_ok = torch.allclose(sensitive_tensor, restored_tensor, atol=1e-6)
    print(f"Integridad verificada: {'✓' if integrity_ok else '✗'}")

    # Cifrado en reposo dentro de ZSpace (secret_key habilita el cifrado)
    mneme = ZSpace(MnemeConfig(secret_key=secret_key))
    desc = mneme.register("tensor_cifrado", sensitive_tensor)
    decrypted_tensor = mneme.load("tensor_cifrado").cpu()
    roundtrip_ok = torch.allclose(sensitive_tensor, decrypted_tensor, atol=1e-6)
    print(f"Roundtrip cifrado ({desc.decomp_type.value}): {'✓' if roundtrip_ok else '✗'}")

    # Métricas de seguridad
    security_stats = mneme.get_stats()["security"]
    print("Métricas de seguridad:")
    print(f"- Violaciones de seguridad: {security_stats['security_violations']}")
    print(f"- Eventos de auditoría: {security_stats['audit_events']}")
    print(f"- Nivel de seguridad: {security_stats['config']['security_level']}")

def example_advanced_storage():
    """Ejemplo de almacenamiento avanzado"""
    print("\n" + "="*60)
    print("ALMACENAMIENTO AVANZADO")
    print("="*60)

    # Almacén persistente propio para la demo
    storage_dir = tempfile.mkdtemp(prefix="mneme_storage_demo_")
    mneme = ZSpace(MnemeConfig(storage_path=storage_dir))

    # Crear tensores de diferentes tamaños
    tensors = {
        "small": torch.randn(100, 100),
        "medium": torch.randn(1000, 1000),
        "large": torch.randn(2000, 2000),
    }

    for name, tensor in tensors.items():
        print(f"\nProcesando tensor {name}: {tensor.shape}")

        # Registrar con almacenamiento automático
        desc = mneme.register(name, tensor, target_ratio=0.1)

        # Simular accesos repetidos (alimentan cache y prefetcher)
        for _ in range(5):
            mneme.load(name)

        print(f"  Ratio de compresión: {desc.meta.get('compression_ratio', 1.0):.3f}")
        print(f"  Tamaño comprimido: {len(desc.core_data)/1024:.1f}KB")

    # Métricas de almacenamiento de la instancia activa
    storage_metrics = mneme.get_stats()["metrics"]
    print("\nMétricas de almacenamiento:")
    print(f"- Operaciones de lectura: {storage_metrics['read_operations']}")
    print(f"- Operaciones de escritura: {storage_metrics['write_operations']}")
    print(f"- Cache hits: {storage_metrics['cache_hits']}")
    print(f"- Cache misses: {storage_metrics['cache_misses']}")

    # Rehidratación: una instancia nueva reconstruye desde el almacén en disco
    reference = mneme.load("medium").cpu()
    mneme.cleanup()

    mneme2 = ZSpace(MnemeConfig(storage_path=storage_dir))
    rehydrated = mneme2.load("medium").cpu()
    rehydration_ok = torch.allclose(reference, rehydrated)
    print(f"\nRehidratación desde disco: {'✓' if rehydration_ok else '✗'}")

    mneme2.cleanup()
    shutil.rmtree(storage_dir, ignore_errors=True)

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

    # Configuración de compresión
    config = CompressionConfig(target_ratio=0.1)

    # Comprimir modelo
    compressed_model = compress_model(model, config=config, min_params=1000)

    # Estadísticas de compresión
    stats = get_compression_stats(compressed_model)
    print("\nEstadísticas de compresión:")
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
    print("\nInferencia:")
    print(f"- Tiempo original: {original_time:.4f}s")
    print(f"- Tiempo comprimido: {compressed_time:.4f}s")
    print(f"- Diferencia de salida: {diff:.6f}")

    # Estadísticas de rendimiento del modelo
    perf_stats = get_model_performance_stats(compressed_model)
    print("\nEstadísticas de rendimiento:")
    print(f"- Tiempo total de forward: {perf_stats['total_forward_time']:.4f}s")
    print(f"- Conteo total de forward: {perf_stats['total_forward_count']}")
    print(f"- Tiempo promedio: {perf_stats.get('avg_forward_time', 0.0):.4f}s")

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
            config = CompressionConfig(target_ratio=0.1)

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
    print("Estadísticas de compresión:")
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
    print(f"Tensores optimizados: {len(optimized_tensors)}")

    # Reporte de optimización
    report = optimizer.get_optimization_report()
    print("\nReporte de optimización:")
    print(f"- Nivel: {report['optimization_level']}")
    print(f"- Métricas de rendimiento: {len(report['performance'])} categorías")
    print(f"- Optimizaciones de recursos: {len(report['resources'])} estrategias")
    print(f"- Recomendaciones: {len(report['recommendations'])}")

    # Estado de salud del sistema
    health = optimizer.get_health_status()
    print(f"Estado de salud: {health}")

    # Optimizar sistema completo
    system_optimization = optimizer.optimize_system()
    print("\nOptimizaciones del sistema:")
    for strategy in system_optimization:
        print(f"- {strategy}")

def example_performance_monitoring():
    """Ejemplo de monitoreo de rendimiento"""
    print("\n" + "="*60)
    print("MONITOREO DE RENDIMIENTO")
    print("="*60)

    # Obtener métricas del sistema (ZSpace global de las capas Z*)
    metrics = get_system_metrics()
    storage_metrics = metrics["metrics"]
    print("Métricas del sistema:")
    print(f"- Operaciones de lectura: {storage_metrics['read_operations']}")
    print(f"- Operaciones de escritura: {storage_metrics['write_operations']}")
    print(f"- Tensores almacenados: {storage_metrics['tensor_count']}")
    print(f"- Bytes almacenados: {storage_metrics['total_storage_bytes']/1024/1024:.1f}MB")

    # Estado de salud
    health = get_health_status()
    print(f"Estado de salud: {health}")

    # Optimizar sistema
    optimization_result = optimize_system()
    print("\nResultado de optimización:")
    for action in optimization_result.get("actions", []):
        print(f"- {action}")

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
        # Actualización dispersa: pares (fila, columna) como tupla de índices
        rows = torch.randint(0, 100, (20,)).tolist()
        cols = torch.randint(0, 100, (20,)).tolist()
        values = torch.randn(20).tolist()

        delta = {
            "type": "sparse_update",
            "indices": (rows, cols),
            "values": values
        }

        desc = mneme.update("game_state", delta)
        print(f"Paso {step+1}: versión {desc.version}, {len(rows)} actualizaciones")

    # Cargar estado final
    final_state = mneme.load("game_state")
    non_zero = (final_state != 0).sum().item()
    print(f"\nEstado final: {non_zero} elementos no cero")

    # Última versión creada por la cadena de deltas
    print(f"Versión final del descriptor: {desc.version}")

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
        error = torch.norm(tensor - loaded.cpu()) / torch.norm(tensor)
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
