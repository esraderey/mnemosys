"""
Script de Prueba y Demostración - MNEME Optimization Module
Demuestra las capacidades del sistema de optimización
"""

import torch
import numpy as np
import time
from typing import List

# Importar desde el módulo de optimización
# (Asumiendo estructura de paquete apropiada)
# from mneme_optimization_updated import (
#     MNEMEOptimizer, OptimizationLevel, create_optimizer,
#     get_system_metrics, benchmark_optimization, PerformanceMonitor,
#     ResourceOptimizer
# )

def create_test_tensors(num_tensors: int = 10, size: tuple = (100, 100)) -> List[torch.Tensor]:
    """Crear tensores de prueba"""
    tensors = []
    for i in range(num_tensors):
        tensor = torch.randn(*size)
        tensors.append(tensor)
    return tensors

def demo_basic_optimization():
    """Demo 1: Optimización básica"""
    print("\n" + "="*80)
    print("DEMO 1: Optimización Básica")
    print("="*80)
    
    # Crear optimizador básico
    from mneme_optimization_updated import MNEMEOptimizer, OptimizationLevel
    
    optimizer = MNEMEOptimizer(
        optimization_level=OptimizationLevel.BASIC,
        enable_profiling=True,
        enable_parallel_processing=False
    )
    
    # Crear tensores de prueba
    print("\n1. Creando tensores de prueba...")
    tensors = create_test_tensors(num_tensors=5, size=(200, 200))
    print(f"   Creados {len(tensors)} tensores de forma {tensors[0].shape}")
    
    # Optimizar tensores
    print("\n2. Optimizando tensores...")
    start_time = time.time()
    optimized_tensors = optimizer.optimize_tensor_operations(tensors)
    optimization_time = time.time() - start_time
    
    print(f"   Tiempo de optimización: {optimization_time*1000:.2f}ms")
    print(f"   Tensores optimizados: {len(optimized_tensors)}")
    
    # Obtener reporte
    print("\n3. Reporte de optimización:")
    report = optimizer.get_optimization_report()
    print(f"   Estado de salud: {report['health_status']}")
    print(f"   Nivel de optimización: {report['optimization_level']}")
    
    if 'performance' in report:
        perf = report['performance']
        if 'current_metrics' in perf:
            metrics = perf['current_metrics']
            print(f"   Uso de memoria: {metrics.get('memory_usage_mb', 0):.1f} MB")
            print(f"   Uso de CPU: {metrics.get('cpu_usage_percent', 0):.1f}%")
    
    # Cleanup
    optimizer.cleanup()
    print("\n✓ Demo completado exitosamente")

def demo_performance_monitoring():
    """Demo 2: Monitoreo de rendimiento"""
    print("\n" + "="*80)
    print("DEMO 2: Monitoreo de Rendimiento")
    print("="*80)
    
    from mneme_optimization_updated import PerformanceMonitor, MnemeConfig
    
    # Crear monitor
    config = MnemeConfig()
    monitor = PerformanceMonitor(config, history_size=50)
    
    print("\n1. Iniciando monitoreo continuo...")
    monitor.start_monitoring(interval=1.0)
    
    # Simular operaciones
    print("\n2. Ejecutando operaciones de prueba...")
    for i in range(5):
        with monitor.measure_operation(f"operation_{i}"):
            # Simular trabajo
            tensor = torch.randn(500, 500)
            result = torch.matmul(tensor, tensor.t())
            time.sleep(0.1)
        print(f"   Operación {i+1} completada")
    
    # Esperar un poco para recolectar métricas
    time.sleep(2)
    
    # Obtener reporte
    print("\n3. Reporte de rendimiento:")
    report = monitor.get_performance_report()
    
    print(f"   Tiempo activo: {report['uptime_seconds']:.1f}s")
    print(f"   Operaciones totales: {report['operations']['total']}")
    print(f"   Tasa de éxito: {report['operations']['success_rate']*100:.1f}%")
    
    if 'current_metrics' in report:
        metrics = report['current_metrics']
        print(f"   Memoria actual: {metrics.get('memory_usage_mb', 0):.1f} MB")
        print(f"   CPU: {metrics.get('cpu_usage_percent', 0):.1f}%")
    
    # Estado de salud
    health = monitor.get_health_status()
    print(f"   Estado de salud: {health}")
    
    # Cleanup
    monitor.cleanup()
    print("\n✓ Demo completado exitosamente")

def demo_resource_optimization():
    """Demo 3: Optimización de recursos"""
    print("\n" + "="*80)
    print("DEMO 3: Optimización de Recursos")
    print("="*80)
    
    from mneme_optimization_updated import ResourceOptimizer, MnemeConfig, ResourceType
    
    # Crear optimizador de recursos
    config = MnemeConfig()
    resource_optimizer = ResourceOptimizer(config)
    
    # Obtener métricas de recursos
    print("\n1. Métricas de recursos del sistema:")
    
    memory_metrics = resource_optimizer.get_resource_metrics(ResourceType.MEMORY)
    print(f"\n   MEMORIA:")
    print(f"   - Uso actual: {memory_metrics.current_usage:.1f} MB")
    print(f"   - Disponible: {memory_metrics.available:.1f} MB")
    print(f"   - Porcentaje: {memory_metrics.usage_percent():.1f}%")
    print(f"   - Estado: {'⚠️ Advertencia' if memory_metrics.is_warning() else '✓ Normal'}")
    
    cpu_metrics = resource_optimizer.get_resource_metrics(ResourceType.CPU)
    print(f"\n   CPU:")
    print(f"   - Uso: {cpu_metrics.current_usage:.1f}%")
    print(f"   - Estado: {'⚠️ Advertencia' if cpu_metrics.is_warning() else '✓ Normal'}")
    
    if torch.cuda.is_available():
        gpu_metrics = resource_optimizer.get_resource_metrics(ResourceType.GPU)
        print(f"\n   GPU:")
        print(f"   - Memoria: {gpu_metrics.current_usage:.1f} MB")
        print(f"   - Disponible: {gpu_metrics.available:.1f} MB")
        print(f"   - Porcentaje: {gpu_metrics.usage_percent():.1f}%")
    
    # Optimizar recursos
    print("\n2. Ejecutando optimización de recursos...")
    optimization_result = resource_optimizer.optimize_resources()
    
    print(f"   Acciones tomadas: {len(optimization_result['actions_taken'])}")
    for action in optimization_result['actions_taken']:
        print(f"   - {action}")
    
    # Obtener recomendaciones
    print("\n3. Recomendaciones de optimización:")
    recommendations = resource_optimizer.get_optimization_recommendations()
    
    if recommendations:
        for i, rec in enumerate(recommendations[:3], 1):  # Mostrar top 3
            print(f"\n   Recomendación {i} (Prioridad {rec.priority}):")
            print(f"   - Categoría: {rec.category}")
            print(f"   - Título: {rec.title}")
            print(f"   - Descripción: {rec.description}")
            print(f"   - Mejora estimada: {rec.estimated_improvement}")
            print(f"   - Acciones sugeridas:")
            for action in rec.actions[:2]:  # Mostrar primeras 2 acciones
                print(f"     • {action}")
    else:
        print("   ✓ No hay recomendaciones - sistema operando óptimamente")
    
    print("\n✓ Demo completado exitosamente")

def demo_parallel_processing():
    """Demo 4: Procesamiento paralelo"""
    print("\n" + "="*80)
    print("DEMO 4: Procesamiento Paralelo")
    print("="*80)
    
    from mneme_optimization_updated import MNEMEOptimizer, OptimizationLevel
    
    # Crear tensores más grandes
    print("\n1. Creando conjunto grande de tensores...")
    num_tensors = 20
    tensors = create_test_tensors(num_tensors=num_tensors, size=(300, 300))
    print(f"   Creados {num_tensors} tensores de {tensors[0].numel():,} elementos cada uno")
    
    # Optimización secuencial
    print("\n2. Procesamiento SECUENCIAL:")
    optimizer_seq = MNEMEOptimizer(
        optimization_level=OptimizationLevel.BASIC,
        enable_parallel_processing=False
    )
    
    start_time = time.time()
    result_seq = optimizer_seq.optimize_tensor_operations(tensors)
    time_seq = time.time() - start_time
    
    print(f"   Tiempo: {time_seq*1000:.2f}ms")
    print(f"   Tensores procesados: {len(result_seq)}")
    optimizer_seq.cleanup()
    
    # Optimización paralela
    print("\n3. Procesamiento PARALELO:")
    optimizer_par = MNEMEOptimizer(
        optimization_level=OptimizationLevel.BASIC,
        enable_parallel_processing=True
    )
    
    start_time = time.time()
    result_par = optimizer_par.optimize_tensor_operations(tensors)
    time_par = time.time() - start_time
    
    print(f"   Tiempo: {time_par*1000:.2f}ms")
    print(f"   Tensores procesados: {len(result_par)}")
    
    # Comparación
    if time_seq > time_par:
        speedup = time_seq / time_par
        print(f"\n4. Resultado:")
        print(f"   ⚡ Aceleración: {speedup:.2f}x más rápido con procesamiento paralelo")
    else:
        print(f"\n4. Resultado:")
        print(f"   ℹ️  Para este conjunto de datos, el overhead de paralelización")
        print(f"      supera los beneficios (prueba con tensores más grandes)")
    
    optimizer_par.cleanup()
    print("\n✓ Demo completado exitosamente")

def demo_optimization_levels():
    """Demo 5: Comparación de niveles de optimización"""
    print("\n" + "="*80)
    print("DEMO 5: Comparación de Niveles de Optimización")
    print("="*80)
    
    from mneme_optimization_updated import benchmark_optimization, OptimizationLevel
    
    # Crear tensores de prueba
    print("\n1. Preparando tensores de prueba...")
    tensors = create_test_tensors(num_tensors=10, size=(400, 400))
    total_size = sum(t.numel() * t.element_size() for t in tensors) / (1024 * 1024)
    print(f"   {len(tensors)} tensores, tamaño total: {total_size:.2f} MB")
    
    # Ejecutar benchmark
    print("\n2. Ejecutando benchmark de optimización...")
    print("   (esto puede tomar unos momentos...)")
    
    levels = [
        OptimizationLevel.NONE,
        OptimizationLevel.BASIC,
        OptimizationLevel.AGGRESSIVE,
        OptimizationLevel.MAXIMUM
    ]
    
    results = benchmark_optimization(tensors, levels)
    
    # Mostrar resultados
    print("\n3. Resultados del benchmark:")
    print("\n   " + "-"*70)
    print(f"   {'Nivel':<20} {'Tiempo':<15} {'Mem Delta':<15} {'Estado':<15}")
    print("   " + "-"*70)
    
    for level_name, result in results.items():
        if result.get('success'):
            time_val = f"{result['time_seconds']*1000:.2f}ms"
            mem_val = f"{result['memory_delta_mb']:+.1f}MB"
            status = "✓ OK"
        else:
            time_val = "N/A"
            mem_val = "N/A"
            status = "✗ Error"
        
        print(f"   {level_name:<20} {time_val:<15} {mem_val:<15} {status:<15}")
    
    print("   " + "-"*70)
    print("\n✓ Demo completado exitosamente")

def demo_auto_optimization():
    """Demo 6: Auto-optimización"""
    print("\n" + "="*80)
    print("DEMO 6: Auto-Optimización")
    print("="*80)
    
    from mneme_optimization_updated import MNEMEOptimizer, OptimizationLevel
    
    print("\n1. Creando optimizador con auto-optimización...")
    optimizer = MNEMEOptimizer(
        optimization_level=OptimizationLevel.ADAPTIVE,
        enable_profiling=True,
        enable_auto_optimization=True
    )
    
    print("   ✓ Auto-optimización habilitada")
    print("   ℹ️  El sistema se optimizará automáticamente cada 30 segundos")
    
    # Simular carga de trabajo
    print("\n2. Simulando carga de trabajo durante 5 segundos...")
    start_time = time.time()
    
    iteration = 0
    while time.time() - start_time < 5:
        # Crear y procesar tensores
        tensors = create_test_tensors(num_tensors=3, size=(200, 200))
        optimized = optimizer.optimize_tensor_operations(tensors)
        
        iteration += 1
        if iteration % 5 == 0:
            health = optimizer.get_health_status()
            print(f"   Iteración {iteration}: Estado = {health}")
        
        time.sleep(0.2)
    
    # Obtener reporte final
    print("\n3. Reporte final:")
    report = optimizer.get_optimization_report()
    
    print(f"   Estado de salud: {report['health_status']}")
    
    if 'performance' in report:
        perf = report['performance']
        if 'operations' in perf:
            ops = perf['operations']
            print(f"   Operaciones totales: {ops.get('total', 0)}")
            print(f"   Tasa de éxito: {ops.get('success_rate', 0)*100:.1f}%")
    
    # Cleanup
    optimizer.cleanup()
    print("\n✓ Demo completado exitosamente")

def demo_system_metrics():
    """Demo 7: Métricas del sistema"""
    print("\n" + "="*80)
    print("DEMO 7: Métricas del Sistema")
    print("="*80)
    
    from mneme_optimization_updated import get_system_metrics
    
    print("\n1. Obteniendo métricas del sistema...")
    metrics = get_system_metrics()
    
    print("\n2. Métricas actuales:")
    
    if 'current_metrics' in metrics:
        current = metrics['current_metrics']
        print(f"\n   RECURSOS:")
        print(f"   - Memoria: {current.get('memory_usage_mb', 0):.1f} MB ({current.get('memory_usage_percent', 0):.1f}%)")
        print(f"   - CPU: {current.get('cpu_usage_percent', 0):.1f}%")
        
        if current.get('gpu_usage_percent', 0) > 0:
            print(f"   - GPU: {current.get('gpu_memory_mb', 0):.1f} MB ({current.get('gpu_usage_percent', 0):.1f}%)")
    
    if 'operations' in metrics:
        ops = metrics['operations']
        print(f"\n   OPERACIONES:")
        print(f"   - Total: {ops.get('total', 0)}")
        print(f"   - Fallidas: {ops.get('failed', 0)}")
        print(f"   - Tasa de éxito: {ops.get('success_rate', 0)*100:.1f}%")
        print(f"   - Operaciones únicas: {ops.get('unique_operations', 0)}")
    
    print("\n✓ Demo completado exitosamente")

def main():
    """Ejecutar todas las demos"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "MNEME OPTIMIZATION MODULE DEMO" + " "*28 + "║")
    print("║" + " "*25 + "Versión 2.0.0" + " "*40 + "║")
    print("╚" + "="*78 + "╝")
    
    demos = [
        ("Optimización Básica", demo_basic_optimization),
        ("Monitoreo de Rendimiento", demo_performance_monitoring),
        ("Optimización de Recursos", demo_resource_optimization),
        ("Procesamiento Paralelo", demo_parallel_processing),
        ("Niveles de Optimización", demo_optimization_levels),
        ("Auto-Optimización", demo_auto_optimization),
        ("Métricas del Sistema", demo_system_metrics)
    ]
    
    print("\nDemostraciones disponibles:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "-"*80)
    
    try:
        # Ejecutar demos seleccionadas
        # Por defecto ejecutar las 3 primeras
        for i in range(min(3, len(demos))):
            name, demo_func = demos[i]
            try:
                demo_func()
            except Exception as e:
                print(f"\n✗ Error en demo '{name}': {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*80)
        print("TODAS LAS DEMOS COMPLETADAS")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n✗ Demos interrumpidas por el usuario")
    except Exception as e:
        print(f"\n\n✗ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
