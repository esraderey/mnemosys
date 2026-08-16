---
name: ⚡ Performance Issue
about: Reportar un problema de rendimiento en MNEME v2.0
title: '[PERFORMANCE] '
labels: ['performance', 'needs-triage']
assignees: ''
---

## ⚡ Descripción del Problema de Rendimiento
[Descripción clara del problema de rendimiento que estás experimentando]

## 📊 Métricas Actuales
- **Tiempo de registro**: [ej. 5.2s]
- **Tiempo de carga**: [ej. 3.1s]
- **Uso de memoria**: [ej. 2.1GB]
- **Uso de CPU**: [ej. 85%]
- **Ratio de compresión**: [ej. 8.5x]
- **Error de reconstrucción**: [ej. 0.001]

## 🎯 Métricas Esperadas
- **Tiempo de registro**: [ej. <1s]
- **Tiempo de carga**: [ej. <0.5s]
- **Uso de memoria**: [ej. <1GB]
- **Uso de CPU**: [ej. <50%]
- **Ratio de compresión**: [ej. >10x]
- **Error de reconstrucción**: [ej. <0.0001]

## 🔄 Pasos para Reproducir
1. [Paso 1]
2. [Paso 2]
3. [Paso 3]
4. [Ver problema de rendimiento]

## 🖥️ Información del Sistema
- **OS**: [ej. Windows 10, macOS 12, Ubuntu 20.04]
- **Python**: [ej. 3.8.10, 3.9.7, 3.10.0]
- **PyTorch**: [ej. 1.12.0, 2.0.0]
- **MNEME**: [ej. 2.0.0]
- **Safetensors**: [ej. 0.3.0+]
- **RAM**: [ej. 8GB, 16GB, 32GB]
- **CPU**: [ej. Intel i7, AMD Ryzen 7]
- **GPU**: [ej. NVIDIA RTX 3080, AMD RX 6800, CPU only]

## 📋 Configuración
```env
# Configuración relevante
MNEME_CACHE_SIZE_MB=1024
MNEME_COMPRESSION_LEVEL=6
MNEME_ENABLE_PARALLEL_PROCESSING=true
MNEME_MAX_WORKERS=4
MNEME_ENABLE_PROFILING=true
```

## 🔍 Código de Reproducción
```python
import time
import torch
from mneme import ZSpace, CompressionConfig

# Código que reproduce el problema de rendimiento
mneme = ZSpace()
tensor = torch.randn(1000, 1000)

# Medir tiempo de registro
start = time.time()
desc = mneme.register("test", tensor)
reg_time = time.time() - start
print(f"Registration time: {reg_time:.3f}s")

# Medir tiempo de carga
start = time.time()
loaded = mneme.load("test")
load_time = time.time() - start
print(f"Loading time: {load_time:.3f}s")
```

## 📊 Benchmarks
```python
# Si tienes benchmarks específicos
import time
import torch
from mneme import ZSpace

# Benchmark code here
```

## 🏷️ Componente Afectado
- [ ] Procesamiento Paralelo
- [ ] Seguridad Cuántica
- [ ] Almacenamiento Inteligente
- [ ] Monitoreo de Rendimiento
- [ ] Integración PyTorch
- [ ] Optimización Automática
- [ ] Cache
- [ ] Compresión
- [ ] Otro

## 📈 Tipo de Problema
- [ ] Latencia alta
- [ ] Uso excesivo de memoria
- [ ] Uso excesivo de CPU
- [ ] Baja compresión
- [ ] Error de reconstrucción alto
- [ ] Problema de escalabilidad
- [ ] Otro

## 🔧 Solución Propuesta
[Si tienes una idea de cómo solucionar el problema de rendimiento]

## 📚 Referencias
[Enlaces a documentación, papers, o implementaciones similares]

## 🏷️ Etiquetas Adicionales
- [ ] Critical
- [ ] High
- [ ] Medium
- [ ] Low
- [ ] Memory
- [ ] CPU
- [ ] GPU
- [ ] Network

## 📊 Impacto
- [ ] Crítico (sistema inutilizable)
- [ ] Alto (rendimiento significativamente degradado)
- [ ] Medio (rendimiento moderadamente afectado)
- [ ] Bajo (rendimiento ligeramente afectado)

## 📚 Contexto Adicional
[Cualquier otra información relevante sobre el problema de rendimiento]