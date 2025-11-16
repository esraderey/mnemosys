"""
MNEME Test Suite v2.0
Suite de pruebas para verificar el funcionamiento de MNEME con nuevas funcionalidades
"""

import torch
import numpy as np
import time
import logging
import unittest
import io

# Importar módulos MNEME actualizados
from mneme import (
    ZSpace, DecompType, CompressionLevel, SecurityLevel,
    ZLinear, ZConv2d, ZAttention, ZTransformerBlock, ZParameter,
    compress_model, get_compression_stats, get_model_performance_stats,
    CompressionConfig, optimize_model_memory, get_system_metrics,
    get_health_status, optimize_system, MNEMEOptimizer, OptimizationLevel,
    SecurityManager, encrypt_data, decrypt_data, ParallelExecutionMode
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMNEMECore(unittest.TestCase):
    """Pruebas del núcleo de MNEME v2.0"""
    
    def setUp(self):
        """Configurar para cada prueba"""
        self.mneme = ZSpace()
    
    def test_basic_tensor_operations(self):
        """Probar operaciones básicas con tensores"""
        # Crear tensor de prueba
        tensor = torch.randn(100, 100)
        
        # Registrar tensor
        desc = self.mneme.register("test_tensor", tensor, target_ratio=0.1)
        self.assertIsNotNone(desc)
        self.assertEqual(desc.shape, tensor.shape)
        
        # Cargar tensor
        loaded = self.mneme.load("test_tensor")
        self.assertEqual(loaded.shape, tensor.shape)
        
        # Verificar precisión
        error = torch.norm(tensor - loaded) / torch.norm(tensor)
        self.assertLess(error, 0.01)  # Error < 1%
    
    def test_parallel_processing(self):
        """Probar procesamiento en paralelo"""
        # Crear múltiples tensores
        tensors = [torch.randn(100, 100) for _ in range(4)]
        
        # Registrar con procesamiento paralelo
        descriptors = []
        for i, tensor in enumerate(tensors):
            desc = self.mneme.register_parallel(f"parallel_tensor_{i}", tensor, 
                                               target_ratio=0.1, decomp_type=DecompType.TT)
            descriptors.append(desc)
        
        # Verificar que se crearon los descriptores
        self.assertEqual(len(descriptors), len(tensors))
        
        # Cargar con optimizaciones paralelas
        loaded_tensors = []
        for i in range(len(tensors)):
            loaded = self.mneme.load_parallel(f"parallel_tensor_{i}")
            loaded_tensors.append(loaded)
        
        # Verificar precisión
        for i, (original, loaded) in enumerate(zip(tensors, loaded_tensors)):
            error = torch.norm(original - loaded) / torch.norm(original)
            self.assertLess(error, 0.1, f"Error en tensor {i}: {error}")
    
    def test_advanced_security(self):
        """Probar funcionalidades de seguridad avanzada"""
        # Crear tensor sensible
        sensitive_tensor = torch.randn(50, 50)
        
        # Cifrar tensor
        encrypted_data, metadata = self.mneme.encrypt_tensor(sensitive_tensor)
        self.assertIsInstance(encrypted_data, bytes)
        self.assertIsInstance(metadata, dict)
        self.assertGreater(len(encrypted_data), 0)
        
        # Descifrar tensor
        decrypted_tensor = self.mneme.decrypt_tensor(encrypted_data, metadata)
        self.assertEqual(decrypted_tensor.shape, sensitive_tensor.shape)
        
        # Verificar integridad
        integrity_ok = torch.allclose(sensitive_tensor, decrypted_tensor, atol=1e-6)
        self.assertTrue(integrity_ok)
        
        # Autenticación de usuario
        credentials = {"username": "test_user", "password": "test_pass"}
        session_id = self.mneme.authenticate_user(credentials)
        self.assertIsInstance(session_id, str)
        self.assertGreater(len(session_id), 0)
    
    def test_advanced_storage(self):
        """Probar almacenamiento avanzado"""
        # Crear tensores de diferentes tamaños
        small_tensor = torch.randn(50, 50)
        large_tensor = torch.randn(500, 500)
        
        # Registrar tensores
        desc1 = self.mneme.register("small_tensor", small_tensor, target_ratio=0.1)
        desc2 = self.mneme.register("large_tensor", large_tensor, target_ratio=0.1)
        
        # Verificar que se crearon los descriptores
        self.assertIsNotNone(desc1)
        self.assertIsNotNone(desc2)
        
        # Cargar tensores
        loaded1 = self.mneme.load("small_tensor")
        loaded2 = self.mneme.load("large_tensor")
        
        # Verificar precisión
        error1 = torch.norm(small_tensor - loaded1) / torch.norm(small_tensor)
        error2 = torch.norm(large_tensor - loaded2) / torch.norm(large_tensor)
        
        self.assertLess(error1, 0.1)
        self.assertLess(error2, 0.1)
    
    def test_performance_monitoring(self):
        """Probar monitoreo de rendimiento"""
        # Obtener métricas del sistema
        metrics = self.mneme.get_performance_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('metrics', metrics)
        self.assertIn('timestamp', metrics)
        
        # Estado de salud
        health = self.mneme.get_health_status()
        self.assertIn(health, ['excellent', 'good', 'fair', 'poor'])
        
        # Optimizar sistema
        optimization_result = self.mneme.optimize_system()
        self.assertIsInstance(optimization_result, dict)
    
    def test_different_decomp_types(self):
        """Probar diferentes tipos de descomposición"""
        tensor = torch.randn(50, 50)
        
        decomp_types = [DecompType.TT, DecompType.CP, DecompType.SVD]
        
        for decomp_type in decomp_types:
            with self.subTest(decomp_type=decomp_type):
                desc = self.mneme.register(
                    f"test_{decomp_type.value}", 
                    tensor, 
                    decomp_type=decomp_type,
                    target_ratio=0.1
                )
                
                loaded = self.mneme.load(f"test_{decomp_type.value}")
                error = torch.norm(tensor - loaded) / torch.norm(tensor)
                
                self.assertLess(error, 0.1)  # Error < 10%

class TestMNEMETorch(unittest.TestCase):
    """Pruebas de integración con PyTorch v2.0"""
    
    def test_zlinear_layer(self):
        """Probar capa ZLinear"""
        config = CompressionConfig(target_ratio=0.1, use_parallel_processing=True)
        layer = ZLinear(100, 50, config=config)
        
        # Forward pass
        x = torch.randn(32, 100)
        output = layer(x)
        
        self.assertEqual(output.shape, (32, 50))
        
        # Verificar estadísticas de rendimiento
        stats = layer.get_performance_stats()
        self.assertIn('forward_count', stats)
        self.assertIn('avg_forward_time', stats)
        self.assertIn('compression', stats)
    
    def test_zconv2d_layer(self):
        """Probar capa ZConv2d"""
        config = CompressionConfig(target_ratio=0.1, use_parallel_processing=True)
        layer = ZConv2d(3, 64, 3, config=config)
        
        # Forward pass
        x = torch.randn(32, 3, 32, 32)
        output = layer(x)
        
        self.assertEqual(output.shape, (32, 64, 32, 32))
    
    def test_zattention_layer(self):
        """Probar capa ZAttention"""
        config = CompressionConfig(target_ratio=0.1, use_parallel_processing=True)
        layer = ZAttention(512, 8, config=config)
        
        # Forward pass
        x = torch.randn(32, 128, 512)
        output = layer(x)
        
        self.assertEqual(output.shape, (32, 128, 512))
    
    def test_ztransformer_block(self):
        """Probar bloque ZTransformerBlock"""
        config = CompressionConfig(target_ratio=0.1, use_parallel_processing=True)
        block = ZTransformerBlock(512, 8, config=config)
        
        # Forward pass
        x = torch.randn(32, 128, 512)
        output = block(x)
        
        self.assertEqual(output.shape, (32, 128, 512))
    
    def test_zparameter(self):
        """Probar ZParameter"""
        tensor = torch.randn(100, 100)
        config = CompressionConfig(target_ratio=0.1, use_parallel_processing=True)
        
        param = ZParameter.from_tensor(tensor, "test_param", config)
        
        self.assertEqual(param.shape, tensor.shape)
        
        # Verificar estadísticas de compresión
        stats = param.get_compression_stats()
        self.assertIn('compression_ratio', stats)
        self.assertIn('decomp_type', stats)
        self.assertIn('version', stats)
    
    def test_model_compression(self):
        """Probar compresión de modelo"""
        # Crear modelo simple
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )
        
        # Comprimir modelo
        config = CompressionConfig(target_ratio=0.1, use_parallel_processing=True)
        compressed_model = compress_model(model, config=config)
        
        # Verificar que la compresión funcionó
        stats = get_compression_stats(compressed_model)
        self.assertGreater(stats["compressed_layers"], 0)
        self.assertLess(stats["overall_ratio"], 1.0)
        
        # Verificar estadísticas de rendimiento
        perf_stats = get_model_performance_stats(compressed_model)
        self.assertIn('total_forward_time', perf_stats)
        self.assertIn('total_forward_count', perf_stats)
        self.assertIn('layers', perf_stats)
    
    def test_memory_optimization(self):
        """Probar optimización de memoria"""
        # Crear modelo grande
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 10)
        )
        
        # Optimizar para memoria específica
        optimized_model = optimize_model_memory(model, target_memory_mb=50)
        
        # Verificar que el modelo se optimizó
        self.assertIsNotNone(optimized_model)
        
        # Probar inferencia
        x = torch.randn(32, 1000)
        with torch.no_grad():
            output = optimized_model(x)
        
        self.assertEqual(output.shape, (32, 10))

class TestMNEMEOptimization(unittest.TestCase):
    """Pruebas del sistema de optimización"""
    
    def test_optimizer_creation(self):
        """Probar creación de optimizador"""
        optimizer = MNEMEOptimizer(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            enable_profiling=True,
            enable_parallel_processing=True
        )
        
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.optimization_level, OptimizationLevel.AGGRESSIVE)
        self.assertTrue(optimizer.enable_profiling)
        self.assertTrue(optimizer.enable_parallel_processing)
    
    def test_tensor_optimization(self):
        """Probar optimización de tensores"""
        optimizer = MNEMEOptimizer(optimization_level=OptimizationLevel.BASIC)
        
        # Crear tensores para optimización
        tensors = [torch.randn(100, 100) for _ in range(3)]
        
        # Optimizar tensores
        optimized_tensors = optimizer.optimize_tensor_operations(tensors)
        
        self.assertEqual(len(optimized_tensors), len(tensors))
        
        # Verificar que los tensores se optimizaron
        for i, (original, optimized) in enumerate(zip(tensors, optimized_tensors)):
            self.assertEqual(optimized.shape, original.shape)
    
    def test_optimization_report(self):
        """Probar reporte de optimización"""
        optimizer = MNEMEOptimizer(optimization_level=OptimizationLevel.BASIC)
        
        # Obtener reporte
        report = optimizer.get_optimization_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('optimization_level', report)
        self.assertIn('performance_metrics', report)
        self.assertIn('resource_optimization', report)
        self.assertIn('recommendations', report)
    
    def test_health_status(self):
        """Probar estado de salud"""
        optimizer = MNEMEOptimizer()
        
        health = optimizer.get_health_status()
        self.assertIn(health, ['excellent', 'good', 'fair', 'poor'])
    
    def test_system_optimization(self):
        """Probar optimización del sistema"""
        optimizer = MNEMEOptimizer()
        
        result = optimizer.optimize_system()
        self.assertIsInstance(result, dict)

class TestMNEMESecurity(unittest.TestCase):
    """Pruebas del sistema de seguridad"""
    
    def test_security_manager_creation(self):
        """Probar creación de gestor de seguridad"""
        security_manager = SecurityManager(security_level=SecurityLevel.HIGH)
        
        self.assertIsNotNone(security_manager)
        self.assertEqual(security_manager.security_level, SecurityLevel.HIGH)
    
    def test_encryption_decryption(self):
        """Probar cifrado y descifrado"""
        data = torch.randn(50, 50).numpy().tobytes()
        
        # Cifrar datos
        encrypted_data, metadata = encrypt_data(data, SecurityLevel.STANDARD)
        
        self.assertIsInstance(encrypted_data, bytes)
        self.assertIsInstance(metadata, dict)
        self.assertGreater(len(encrypted_data), 0)
        
        # Descifrar datos
        decrypted_data = decrypt_data(encrypted_data, metadata)
        
        self.assertEqual(len(decrypted_data), len(data))
        self.assertEqual(decrypted_data, data)
    
    def test_secure_descriptor(self):
        """Probar descriptor seguro"""
        security_manager = SecurityManager(security_level=SecurityLevel.HIGH)
        
        data = torch.randn(50, 50).numpy().tobytes()
        secure_desc = security_manager.create_secure_descriptor(data, "test_resource")
        
        self.assertIsNotNone(secure_desc)
        self.assertIsNotNone(secure_desc.checksum)
        self.assertIsNotNone(secure_desc.signature)
        
        # Verificar integridad
        integrity_ok = secure_desc.verify_integrity()
        self.assertTrue(integrity_ok)
        
        # Verificar firma
        signature_ok = secure_desc.verify_signature()
        self.assertTrue(signature_ok)
    
    def test_security_status(self):
        """Probar estado de seguridad"""
        security_manager = SecurityManager(security_level=SecurityLevel.STANDARD)
        
        status = security_manager.get_security_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('security_level', status)
        self.assertIn('policies', status)
        self.assertIn('audit_report', status)

class TestSystemMetrics(unittest.TestCase):
    """Pruebas de métricas del sistema"""
    
    def test_system_metrics(self):
        """Probar métricas del sistema"""
        metrics = get_system_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('metrics', metrics)
        self.assertIn('timestamp', metrics)
    
    def test_health_status(self):
        """Probar estado de salud"""
        health = get_health_status()
        
        self.assertIn(health, ['excellent', 'good', 'fair', 'poor'])
    
    def test_system_optimization(self):
        """Probar optimización del sistema"""
        result = optimize_system()
        
        self.assertIsInstance(result, dict)

def run_performance_benchmark():
    """Ejecutar benchmark de rendimiento"""
    print("\n" + "="*60)
    print("BENCHMARK DE RENDIMIENTO v2.0")
    print("="*60)
    
    # Configurar MNEME
    mneme = ZSpace()
    
    # Diferentes tamaños de tensor
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    
    for size in sizes:
        print(f"\nTamaño: {size}")
        
        # Crear tensor
        tensor = torch.randn(size)
        original_size = tensor.numel() * 4 / 1024  # KB
        
        # Medir tiempo de almacenamiento
        start_time = time.time()
        desc = mneme.register(f"benchmark_{size}", tensor)
        store_time = time.time() - start_time
        
        # Medir tiempo de carga
        start_time = time.time()
        loaded = mneme.load(f"benchmark_{size}")
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
    
    # Probar procesamiento paralelo
    print(f"\nProcesamiento paralelo:")
    tensors = [torch.randn(200, 200) for _ in range(4)]
    
    start_time = time.time()
    for i, tensor in enumerate(tensors):
        mneme.register_parallel(f"parallel_{i}", tensor, target_ratio=0.1)
    parallel_time = time.time() - start_time
    
    print(f"  Tiempo registro paralelo: {parallel_time:.3f}s")
    
    # Probar seguridad
    print(f"\nSeguridad:")
    sensitive_tensor = torch.randn(100, 100)
    
    start_time = time.time()
    encrypted_data, metadata = mneme.encrypt_tensor(sensitive_tensor)
    encrypt_time = time.time() - start_time
    
    start_time = time.time()
    decrypted_tensor = mneme.decrypt_tensor(encrypted_data, metadata)
    decrypt_time = time.time() - start_time
    
    print(f"  Tiempo cifrado: {encrypt_time*1000:.2f}ms")
    print(f"  Tiempo descifrado: {decrypt_time*1000:.2f}ms")
    print(f"  Integridad: {'✓' if torch.allclose(sensitive_tensor, decrypted_tensor) else '✗'}")
    
    # Métricas del sistema
    print(f"\nMétricas del sistema:")
    metrics = mneme.get_performance_metrics()
    print(f"  Operaciones totales: {metrics['metrics']['operations']['total']}")
    print(f"  Uso de memoria: {metrics['metrics']['memory']['current_usage']/1024/1024:.1f}MB")
    print(f"  Estado de salud: {mneme.get_health_status()}")

def main():
    """Función principal de pruebas"""
    print("🧠 MNEME v2.0 - Suite de Pruebas")
    print("="*60)
    
    # Ejecutar pruebas unitarias
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Ejecutar benchmark de rendimiento
    run_performance_benchmark()
    
    print("\n✅ Todas las pruebas completadas")

if __name__ == "__main__":
    main()