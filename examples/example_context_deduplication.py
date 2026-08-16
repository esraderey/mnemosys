#!/usr/bin/env python3
"""
Ejemplo de uso del sistema de deduplicación de contexto de MNEME.

Este ejemplo demuestra:
1. Configuración del sistema de deduplicación de contexto
2. Procesamiento de contextos similares
3. Análisis de similitud semántica
4. Clustering automático de contextos
5. Optimización de almacenamiento
6. Métricas de ahorro de espacio
"""

import torch
import numpy as np
import time
import logging
from mneme_core import MnemeConfig, ZSpace, ContextSimilarityMethod, ContextClusteringMethod

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_similar_tensors():
    """Crear tensores similares para demostrar deduplicación."""
    # Tensor base
    base_tensor = torch.randn(100, 100)
    
    # Tensores similares (variaciones del base)
    similar_tensors = []
    
    # 1. Tensor con pequeñas variaciones
    similar_tensors.append(base_tensor + torch.randn(100, 100) * 0.1)
    
    # 2. Tensor con ruido gaussiano
    similar_tensors.append(base_tensor + torch.randn(100, 100) * 0.05)
    
    # 3. Tensor escalado
    similar_tensors.append(base_tensor * 1.1)
    
    # 4. Tensor con rotación (simulada)
    similar_tensors.append(torch.rot90(base_tensor, 1))
    
    # 5. Tensor completamente diferente
    different_tensor = torch.randn(100, 100)
    
    return base_tensor, similar_tensors, different_tensor

def demonstrate_context_analysis():
    """Demostrar análisis de contexto."""
    print("🔍 Análisis de Contexto")
    print("=" * 50)
    
    # Crear configuración con deduplicación de contexto
    config = MnemeConfig(
        enable_context_deduplication=True,
        context_similarity_method=ContextSimilarityMethod.HYBRID,
        context_clustering_method=ContextClusteringMethod.ADAPTIVE,
        context_similarity_threshold=0.7,
        context_cluster_size=5,
        enable_semantic_analysis=True,
        context_compression_level=6,
        enable_context_caching=True,
        context_cache_size_mb=128
    )
    
    # Crear ZSpace
    zspace = ZSpace(config)
    
    # Crear tensores similares
    base_tensor, similar_tensors, different_tensor = create_similar_tensors()
    
    print(f"📊 Procesando {len(similar_tensors) + 2} contextos...")
    
    # Procesar contextos
    results = []
    
    # Procesar tensor base
    result = zspace.process_context_for_deduplication(
        "base_context", base_tensor, 
        {"type": "base", "description": "Tensor base"}
    )
    results.append(result)
    print(f"✅ Contexto base: {result}")
    
    # Procesar tensores similares
    for i, tensor in enumerate(similar_tensors):
        result = zspace.process_context_for_deduplication(
            f"similar_context_{i}", tensor,
            {"type": "similar", "description": f"Tensor similar {i}"}
        )
        results.append(result)
        print(f"✅ Contexto similar {i}: {result}")
    
    # Procesar tensor diferente
    result = zspace.process_context_for_deduplication(
        "different_context", different_tensor,
        {"type": "different", "description": "Tensor diferente"}
    )
    results.append(result)
    print(f"✅ Contexto diferente: {result}")
    
    return zspace, results

def demonstrate_similarity_analysis():
    """Demostrar análisis de similitud."""
    print("\n🔬 Análisis de Similitud")
    print("=" * 50)
    
    # Crear configuración con diferentes métodos de similitud
    methods = [
        ContextSimilarityMethod.COSINE,
        ContextSimilarityMethod.EUCLIDEAN,
        ContextSimilarityMethod.SEMANTIC,
        ContextSimilarityMethod.HYBRID
    ]
    
    base_tensor = torch.randn(50, 50)
    similar_tensor = base_tensor + torch.randn(50, 50) * 0.1
    different_tensor = torch.randn(50, 50)
    
    for method in methods:
        config = MnemeConfig(
            enable_context_deduplication=True,
            context_similarity_method=method,
            context_similarity_threshold=0.5
        )
        
        zspace = ZSpace(config)
        
        # Procesar contextos
        result1 = zspace.process_context_for_deduplication("base", base_tensor)
        result2 = zspace.process_context_for_deduplication("similar", similar_tensor)
        result3 = zspace.process_context_for_deduplication("different", different_tensor)
        
        print(f"📊 Método {method.value}:")
        print(f"   - Base vs Similar: {result2.get('similarity', 'N/A')}")
        print(f"   - Base vs Different: {result3.get('similarity', 'N/A')}")

def demonstrate_clustering():
    """Demostrar clustering de contextos."""
    print("\n🎯 Clustering de Contextos")
    print("=" * 50)
    
    config = MnemeConfig(
        enable_context_deduplication=True,
        context_clustering_method=ContextClusteringMethod.ADAPTIVE,
        context_similarity_threshold=0.6,
        context_cluster_size=3
    )
    
    zspace = ZSpace(config)
    
    # Crear múltiples contextos
    contexts = []
    for i in range(10):
        # Crear grupos de contextos similares
        if i < 3:
            # Grupo 1: Tensores con patrón similar
            tensor = torch.randn(30, 30) + i * 0.1
        elif i < 6:
            # Grupo 2: Tensores con patrón diferente
            tensor = torch.ones(30, 30) * i
        else:
            # Grupo 3: Tensores únicos
            tensor = torch.randn(30, 30) * (i + 1)
        
        context_id = f"context_{i}"
        result = zspace.process_context_for_deduplication(
            context_id, tensor, {"group": i // 3}
        )
        contexts.append((context_id, result))
    
    # Obtener información de clusters
    print("📊 Información de Clusters:")
    for context_id, result in contexts:
        cluster_info = zspace.get_context_cluster_info(context_id)
        print(f"   - {context_id}: Cluster {cluster_info.get('cluster_id', 'N/A')}, "
              f"Tamaño: {cluster_info.get('cluster_size', 0)}")

def demonstrate_compression_analysis():
    """Demostrar análisis de compresión."""
    print("\n🗜️ Análisis de Compresión")
    print("=" * 50)
    
    config = MnemeConfig(
        enable_context_deduplication=True,
        context_compression_level=6,
        enable_context_caching=True,
        context_cache_size_mb=64
    )
    
    zspace = ZSpace(config)
    
    # Crear tensores con diferentes características
    tensors = {
        "sparse": torch.zeros(100, 100),
        "dense": torch.randn(100, 100),
        "patterned": torch.eye(100),
        "repeated": torch.ones(100, 100) * 0.5
    }
    
    # Hacer algunos tensores sparse
    tensors["sparse"][::10, ::10] = 1.0
    
    print("📊 Análisis de Compresión por Tipo de Tensor:")
    for name, tensor in tensors.items():
        result = zspace.process_context_for_deduplication(
            f"tensor_{name}", tensor, {"type": name}
        )
        
        compression_ratio = result.get("compression_ratio", 1.0)
        print(f"   - {name}: Ratio de compresión = {compression_ratio:.3f}")

def demonstrate_optimization():
    """Demostrar optimización del sistema."""
    print("\n⚡ Optimización del Sistema")
    print("=" * 50)
    
    config = MnemeConfig(
        enable_context_deduplication=True,
        context_similarity_threshold=0.8,
        context_cluster_size=5
    )
    
    zspace = ZSpace(config)
    
    # Crear muchos contextos similares
    print("📊 Creando contextos similares...")
    for i in range(20):
        # Crear contextos con patrones similares
        base = torch.randn(20, 20)
        tensor = base + torch.randn(20, 20) * 0.05
        
        zspace.process_context_for_deduplication(
            f"optimization_context_{i}", tensor,
            {"batch": i // 5}
        )
    
    # Obtener estadísticas antes de la optimización
    stats_before = zspace.get_context_deduplication_stats()
    print(f"📈 Estadísticas antes de la optimización:")
    print(f"   - Total contextos: {stats_before.get('total_contexts', 0)}")
    print(f"   - Contextos deduplicados: {stats_before.get('deduplicated_contexts', 0)}")
    print(f"   - Tasa de deduplicación: {stats_before.get('deduplication_rate', 0):.2%}")
    
    # Optimizar sistema
    print("\n🔧 Optimizando sistema...")
    zspace.optimize_context_deduplication()
    
    # Obtener estadísticas después de la optimización
    stats_after = zspace.get_context_deduplication_stats()
    print(f"📈 Estadísticas después de la optimización:")
    print(f"   - Total contextos: {stats_after.get('total_contexts', 0)}")
    print(f"   - Contextos deduplicados: {stats_after.get('deduplicated_contexts', 0)}")
    print(f"   - Tasa de deduplicación: {stats_after.get('deduplication_rate', 0):.2%}")
    
    # Mostrar estadísticas del clusterer
    clusterer_stats = stats_after.get("clusterer_stats", {})
    print(f"   - Total clusters: {clusterer_stats.get('total_clusters', 0)}")
    print(f"   - Tamaño promedio de cluster: {clusterer_stats.get('avg_cluster_size', 0):.2f}")

def demonstrate_performance_metrics():
    """Demostrar métricas de rendimiento."""
    print("\n📊 Métricas de Rendimiento")
    print("=" * 50)
    
    config = MnemeConfig(
        enable_context_deduplication=True,
        context_similarity_method=ContextSimilarityMethod.HYBRID,
        context_clustering_method=ContextClusteringMethod.ADAPTIVE,
        enable_semantic_analysis=True,
        context_compression_level=6,
        enable_context_caching=True
    )
    
    zspace = ZSpace(config)
    
    # Crear contextos de prueba
    print("📊 Procesando contextos de prueba...")
    start_time = time.time()
    
    for i in range(50):
        # Crear tensores con diferentes características
        if i % 5 == 0:
            tensor = torch.randn(50, 50)  # Tensores únicos
        else:
            tensor = torch.randn(50, 50) + torch.randn(50, 50) * 0.1  # Tensores similares
        
        zspace.process_context_for_deduplication(
            f"perf_context_{i}", tensor,
            {"iteration": i, "type": "unique" if i % 5 == 0 else "similar"}
        )
    
    processing_time = time.time() - start_time
    
    # Obtener estadísticas finales
    stats = zspace.get_context_deduplication_stats()
    
    print(f"⏱️ Tiempo de procesamiento: {processing_time:.3f} segundos")
    print(f"📈 Estadísticas del sistema:")
    print(f"   - Total contextos: {stats.get('total_contexts', 0)}")
    print(f"   - Contextos deduplicados: {stats.get('deduplicated_contexts', 0)}")
    print(f"   - Ahorros de deduplicación: {stats.get('deduplication_saves', 0)}")
    print(f"   - Verificaciones de similitud: {stats.get('similarity_checks', 0)}")
    print(f"   - Ratio de compresión promedio: {stats.get('avg_compression_ratio', 1.0):.3f}")
    
    # Estadísticas del analizador
    analyzer_stats = stats.get("analyzer_stats", {})
    print(f"   - Análisis realizados: {analyzer_stats.get('analysis_count', 0)}")
    print(f"   - Tasa de acierto del cache: {analyzer_stats.get('cache_hit_rate', 0):.1f}%")
    
    # Estadísticas del clusterer
    clusterer_stats = stats.get("clusterer_stats", {})
    print(f"   - Clusters creados: {clusterer_stats.get('clusters_created', 0)}")
    print(f"   - Clusters fusionados: {clusterer_stats.get('clusters_merged', 0)}")
    print(f"   - Tamaño promedio de cluster: {clusterer_stats.get('avg_cluster_size', 0):.2f}")

def demonstrate_memory_savings():
    """Demostrar ahorros de memoria."""
    print("\n💾 Ahorros de Memoria")
    print("=" * 50)
    
    config = MnemeConfig(
        enable_context_deduplication=True,
        context_similarity_threshold=0.8,
        context_compression_level=6
    )
    
    zspace = ZSpace(config)
    
    # Crear contextos con diferentes niveles de similitud
    base_tensor = torch.randn(100, 100)
    
    print("📊 Creando contextos con diferentes niveles de similitud...")
    
    # Contextos muy similares (deberían ser deduplicados)
    similar_contexts = []
    for i in range(10):
        tensor = base_tensor + torch.randn(100, 100) * 0.01  # Muy similar
        result = zspace.process_context_for_deduplication(
            f"similar_{i}", tensor, {"similarity_level": "high"}
        )
        similar_contexts.append(result)
    
    # Contextos moderadamente similares
    moderate_contexts = []
    for i in range(5):
        tensor = base_tensor + torch.randn(100, 100) * 0.1  # Moderadamente similar
        result = zspace.process_context_for_deduplication(
            f"moderate_{i}", tensor, {"similarity_level": "medium"}
        )
        moderate_contexts.append(result)
    
    # Contextos únicos
    unique_contexts = []
    for i in range(3):
        tensor = torch.randn(100, 100)  # Único
        result = zspace.process_context_for_deduplication(
            f"unique_{i}", tensor, {"similarity_level": "low"}
        )
        unique_contexts.append(result)
    
    # Analizar resultados
    print("📈 Análisis de Deduplicación:")
    
    similar_deduplicated = sum(1 for r in similar_contexts if r.get("deduplicated", False))
    moderate_deduplicated = sum(1 for r in moderate_contexts if r.get("deduplicated", False))
    unique_deduplicated = sum(1 for r in unique_contexts if r.get("deduplicated", False))
    
    print(f"   - Contextos muy similares: {similar_deduplicated}/{len(similar_contexts)} deduplicados")
    print(f"   - Contextos moderadamente similares: {moderate_deduplicated}/{len(moderate_contexts)} deduplicados")
    print(f"   - Contextos únicos: {unique_deduplicated}/{len(unique_contexts)} deduplicados")
    
    # Obtener estadísticas finales
    stats = zspace.get_context_deduplication_stats()
    
    total_contexts = stats.get("total_contexts", 0)
    deduplicated_contexts = stats.get("deduplicated_contexts", 0)
    deduplication_rate = stats.get("deduplication_rate", 0)
    
    print(f"\n💾 Ahorros de Memoria:")
    print(f"   - Total contextos procesados: {total_contexts}")
    print(f"   - Contextos deduplicados: {deduplicated_contexts}")
    print(f"   - Tasa de deduplicación: {deduplication_rate:.2%}")
    print(f"   - Ahorro de almacenamiento: {deduplication_rate:.1%}")
    
    # Calcular ahorro estimado
    tensor_size = 100 * 100 * 4  # 100x100 float32
    total_size = total_contexts * tensor_size
    saved_size = deduplicated_contexts * tensor_size
    compression_ratio = stats.get("avg_compression_ratio", 1.0)
    
    print(f"   - Tamaño total sin deduplicación: {total_size / 1024 / 1024:.2f} MB")
    print(f"   - Tamaño ahorrado: {saved_size / 1024 / 1024:.2f} MB")
    print(f"   - Ratio de compresión promedio: {compression_ratio:.3f}")

def main():
    """Función principal."""
    print("🚀 Sistema de Deduplicación de Contexto MNEME")
    print("=" * 60)
    
    try:
        # 1. Análisis de contexto
        zspace, results = demonstrate_context_analysis()
        
        # 2. Análisis de similitud
        demonstrate_similarity_analysis()
        
        # 3. Clustering
        demonstrate_clustering()
        
        # 4. Análisis de compresión
        demonstrate_compression_analysis()
        
        # 5. Optimización
        demonstrate_optimization()
        
        # 6. Métricas de rendimiento
        demonstrate_performance_metrics()
        
        # 7. Ahorros de memoria
        demonstrate_memory_savings()
        
        print("\n✅ Demostración completada exitosamente!")
        print("\n📊 Estadísticas finales del sistema:")
        
        # Obtener estadísticas finales
        final_stats = zspace.get_stats()
        context_stats = final_stats.get("context_deduplication", {})
        
        print(f"   - Sistema habilitado: {context_stats.get('enabled', False)}")
        print(f"   - Total contextos: {context_stats.get('total_contexts', 0)}")
        print(f"   - Tasa de deduplicación: {context_stats.get('deduplication_rate', 0):.2%}")
        print(f"   - Ahorros de deduplicación: {context_stats.get('deduplication_saves', 0)}")
        print(f"   - Verificaciones de similitud: {context_stats.get('similarity_checks', 0)}")
        
    except Exception as e:
        logger.error(f"Error en la demostración: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
