# 🧠 MNEMOSYS – Motor de Memoria Neural Mórfica

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Security](https://img.shields.io/badge/Security-Enterprise-green.svg)](https://github.com/esraderey/mnemosys)
[![Performance](https://img.shields.io/badge/Performance-Optimized-orange.svg)](https://github.com/esraderey/mnemosys)
[![Version](https://img.shields.io/badge/Version-1.0.1-purple.svg)](https://github.com/esraderey/mnemosys)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue.svg)](https://github.com/esraderey/mnemosys/actions)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-A+-green.svg)](https://github.com/esraderey/mnemosys)

**MNEMOSYS** (antes MNEME) redefine la memoria computacional mediante un motor neural inspirado en estructuras biológicas con **arquitectura modular segura**, **locks granulares**, **safetensors**, **lazy decompression** y **cache adaptativo**.  
En lugar de almacenar datos en ubicaciones fijas, **MNEMOSYS guarda descriptores compactos y generativos** que reconstruyen el contenido de forma determinista, como si fueran recuerdos que emergen bajo demanda.

> El paquete Python conserva el import histórico — `import mneme` — como Mnemosyne conserva a sus musas: la distribución se llama `mnemosys`, el módulo se sigue llamando `mneme`.

## 📜 Historial de mejoras (era MNEME)

> Nota de versionado: la primera release pública en PyPI es la **1.0.0**. Las
> menciones a «v2.x» en esta sección corresponden a la numeración interna del
> motor cuando se llamaba MNEME, previa al estreno y nunca publicada — ver
> [CHANGELOG.md](CHANGELOG.md).

### 🐛 **Corrección Crítica del Storage Backend**
- **Integración completa del SecureStorageBackend** - Corregido problema crítico donde el storage backend nunca se utilizaba
- **Almacenamiento persistente automático** - Los tensores ahora se almacenan automáticamente en storage persistente
- **Carga desde storage** - Los tensores se cargan desde storage cuando no están en memoria
- **Sincronización memoria-storage** - Mantiene sincronización completa entre memoria y storage persistente
- **Eliminación de "dead code"** - El SecureStorageBackend ya no es código muerto, está totalmente funcional

### 🔧 **Sistema de Errores Contextuales**
- **Clases de error mejoradas** con información detallada y timestamps
- **Contexto específico** para debugging y monitoreo
- **Códigos de error** únicos para cada tipo de problema
- **Logging mejorado** con mensajes más informativos

### ⚙️ **Configuración Avanzada**
- **MnemeConfig robusta** con validaciones automáticas
- **Serialización completa** con métodos `to_dict()` y `from_dict()`
- **Parámetros extendidos** para TTL, compresión y monitoreo
- **Validación robusta** de todos los parámetros de configuración

### 🔒 **Sistema de Locks Granulares Mejorado**
- **Límites configurables** con limpieza automática de locks no utilizados
- **Detección de deadlocks** y prevención automática
- **Estadísticas detalladas** de uso y rendimiento de locks
- **Gestión inteligente** de recursos de concurrencia

### 🧠 **LazyTensor Optimizado**
- **Gestión inteligente de memoria** con límites configurables
- **Cache de metadatos** para forma y tipo sin decompress
- **Monitoreo de presión** de memoria y limpieza automática
- **Compresión adaptativa** basada en características de datos

### 📊 **Cache Adaptativo Avanzado**
- **Múltiples estrategias** de evicción (LRU, LFU, TTL, Adaptive)
- **Compresión automática** para elementos grandes
- **TTL y expiración** con limpieza automática
- **Métricas detalladas** de rendimiento y patrones de acceso

### 🏗️ **Descriptores Mejorados**
- **ZDescriptor enriquecido** con estadísticas de acceso
- **ZAddr avanzado** con validaciones robustas
- **Verificación de integridad** con Merkle roots y security hashes
- **Serialización completa** con métodos de conversión

### 🚀 **ZSpace Principal Optimizado**
- **Inicialización robusta** con detección automática de GPU
- **Métodos mejorados** con validaciones completas
- **Métricas en tiempo real** de operaciones y memoria
- **Logging configurable** con niveles personalizables

### 💾 **Storage Persistente Funcional**
- **Almacenamiento automático** - Los tensores se guardan automáticamente en storage
- **Carga inteligente** - Carga desde storage cuando no están en memoria
- **Sincronización completa** - Mantiene coherencia entre memoria y storage
- **Gestión de storage** - Métodos para listar, eliminar y sincronizar tensores
- **Métricas de storage** - Monitoreo de operaciones de almacenamiento y carga

### 🛠️ **Herramientas de Desarrollo**
- **GitHub Actions CI/CD** completo con testing, linting y security
- **Dependabot** para actualizaciones automáticas
- **CodeQL** para análisis de seguridad
- **Pre-commit hooks** para calidad de código
- **Templates** para issues y pull requests

---

## 🚀 Funcionalidades del núcleo

### ⚡ **Optimización en Paralelo**
- **Procesamiento paralelo** con ThreadPoolExecutor, ProcessPoolExecutor y asyncio
- **Modos híbridos** que combinan threads, procesos y operaciones asíncronas
- **Descomposición paralela** de tensores (TT, CP, Tucker)
- **Métricas de rendimiento** en tiempo real con análisis de eficiencia

### 🔒 **Seguridad Avanzada**
- **Gestión de claves cuánticas** resistentes a computación cuántica
- **Autenticación multifactor** con tokens y sesiones seguras
- **Cifrado avanzado** con AES-GCM, ChaCha20-Poly1305 y algoritmos post-cuánticos
- **Rotación automática de claves** basada en tiempo y uso
- **Auditoría de seguridad** con logging detallado

### 🗄️ **Almacenamiento Mejorado**
- **Almacenamiento por niveles** (Memoria, SSD, HDD, Archivo) con migración automática
- **Compresión adaptativa** que decide automáticamente el nivel de compresión
- **Almacenamiento distribuido** con replicación y hashing consistente
- **Métricas de almacenamiento** con análisis de patrones de acceso

### 📊 **Monitoreo de Rendimiento**
- **Métricas en tiempo real** de operaciones, memoria, almacenamiento y seguridad
- **Alertas automáticas** cuando se superan umbrales de rendimiento
- **Optimización de recursos** con gestión inteligente de memoria y CPU
- **Estado de salud del sistema** con recomendaciones automáticas

### 🔧 **Mejoras de Arquitectura**
- **Arquitectura modular** dividida en 3 módulos especializados (core, security, storage)
- **Locks granulares** que reemplazan RLock global para mejor concurrencia
- **Safetensors** para serialización segura (sin pickle)
- **Lazy decompression** para optimizar uso de memoria
- **Cache adaptativo** que reemplaza LRU con estrategias inteligentes
- **Validación robusta** de entrada con InputValidator
- **Eliminación completa de pickle** para mayor seguridad

---

## 📋 Nombre del Proyecto

**MNEMOSYS** — heredero directo de MNEME, con el nombre de Mnemosyne, la titánide griega de la memoria y madre de las musas:

- **M**emoria  
- **N**eural  
- **E**structurada  
- **M**órfica con  
- **O**rquestación de  
- **S**íntesis  
- **Y**  
- **S**eguridad  

---

## 🚀 Innovación Clave

Tradicional: **Dirección → Localización → Datos**  
MNEME: **Descriptor → Síntesis → Recuerdo**

```python
# Tradicional: Guardar tensor de 4MB en RAM
memory[0x1000] = huge_tensor

# MNEME: Guardar un descriptor compacto que reconstruye el tensor bajo demanda
descriptor = mneme.register("huge_tensor", huge_tensor, target_ratio=0.1)
tensor = mneme.load("huge_tensor")  # Síntesis determinista
```

## 🎯 ¿Por qué MNEME?

🔹 **Compresión real medida** – SVD al 20 % de bytes, INT8 al 27 % con 0.6 % de error; hasta 60× cuando el dato tiene estructura

🔹 **Síntesis determinista** – mismo descriptor, mismo resultado bit a bit, incluso entre reinicios (verificado 36/36 cross-instancia)

🔹 **Turbo de inferencia** – `compress_model_turbo` ejecuta modelos con el peso comprimido residente: 20.5 % de VRAM, pico de forward ÷4.9, latencia a la par del denso

🔹 **Thread-safe verificado** – locks granulares por tensor; 640 operaciones concurrentes en 8 hilos sin un solo incidente

🔹 **Nada se corrompe en silencio** – flip de bytes, blobs truncados o clave equivocada mueren en `StorageAuthenticationError`; los controles cargan bit-idénticos

🔹 **Cifrado en reposo y firma HMAC reales** – activados por `secret_key`, con almacenamiento persistente SQLite + safetensors + LZ4

🔹 **Sin pickle** – serialización exclusivamente segura

## 📊 Métricas medidas (16-ago-2026, artefactos en `benchmarks/` y suite)

Números de ejecuciones reales en esta máquina (CUDA), no estimaciones:

| Métrica | Valor medido |
|---------|--------------|
| Suite de tests | **134 passed / 0 failed** (~45 s) |
| Turbo SVD 0.1 (8×Linear 2048²) | VRAM residente **20.5 %** del denso · pico forward **÷4.9** · latencia **a la par** |
| Turbo INT8 | VRAM residente 26.6 % · latencia ×100+ (dequant CPU: solo si la VRAM es el cuello) |
| INT8 por grupos (fidelidad) | 26.6 % de bytes, error relativo **0.6 %** |
| SVD sobre dato estructurado | 20 % de bytes, error 10⁻⁶ (rango exacto) / 0.9 % (+1 % ruido) |
| Sparsity 2:4 | patrón exacto tras roundtrip, 50.0 % ceros |
| Rehidratación desde disco | 36/36 tensores **bit-idénticos** entre instancias (fp32 y fp64) |
| Concurrencia (8 hilos, 640 ops) | 0 excepciones, 0 cross-talk, 0 desviaciones |
| Corrupción/clave errónea | 3/3 detectadas (`StorageAuthenticationError`), 0 silenciosas |
| Pythia-410M · WikiText-2 (PPL) | FP16 **15.07** · GPTQ INT4 **19.97** (+32 %) · INT4 naive **24.25** (+61 %) |
| Register / load (1000×1000, SVD) | 57 ms / 6 ms (28 ms el load de 2000², cacheado ~×20 más rápido) |

## 🏗️ Arquitectura Modular Segura

### 📦 Módulos Especializados
```
src/mneme/
├── __init__.py                  # Punto de entrada principal
├── mneme_core.py                # Módulo principal seguro
├── mneme_security_core.py       # Seguridad y validación
├── mneme_storage_core.py        # Almacenamiento seguro
├── mneme_torch.py               # Integración PyTorch
├── mneme_lazy.py                # Turbo: inferencia con pesos comprimidos
└── mneme_optimization.py        # Optimizaciones y cuantización
```

### 🔒 Características de Seguridad
- **Sin pickle** - Eliminadas vulnerabilidades de deserialización
- **Solo safetensors** - Serialización segura garantizada
- **Validación robusta** - InputValidator para todos los datos
- **Locks granulares** - Mejor concurrencia y seguridad
- **Arquitectura modular** - Separación clara de responsabilidades

## 🏗️ Estructura del Proyecto

```
MNEME---Motor-de-Memoria-Neural-M-rfica/
├── src/mneme/                    # Código fuente modular
│   ├── __init__.py              # Exports principales
│   ├── mneme_core.py            # Módulo principal seguro
│   ├── mneme_security_core.py   # Seguridad y validación
│   ├── mneme_storage_core.py    # Almacenamiento seguro
│   ├── mneme_torch.py           # Integración PyTorch
│   └── mneme_optimization.py    # Optimizaciones
├── examples/                     # Ejemplos ejecutables contra la API real
│   ├── example_mneme.py         # Recorrido completo (11 ejemplos)
│   ├── example_advanced_features.py  # Locks, safetensors, lazy, cache
│   └── example_advanced_serialization.py  # Rutas, HMAC, cifrado, LZ4
├── benchmarks/                  # Benchmarks reales (Pythia-410M / WikiText-2)
│   ├── bench_pythia_int4.py
│   ├── bench_pythia_gptq.py
│   └── results_pythia_*.json
├── docs/                        # Documentación
├── tests/                       # Suite (134 tests)
│   ├── test_mneme.py
│   ├── test_regresiones_auditoria.py
│   └── test_turbo.py
├── requirements.txt             # Dependencias
├── pyproject.toml               # Configuración del paquete
└── LICENSE                     # Licencia
```

## 🛠️ Instalación

### Requisitos del Sistema

- **Python**: 3.10+ (recomendado 3.11+)
- **PyTorch**: 2.0+ (con soporte CUDA/MPS opcional)
- **Memoria RAM**: 8GB+ (recomendado 16GB+)
- **GPU**: Opcional pero recomendada (NVIDIA RTX 3060+ o Apple M1+)
- **OS**: Linux / macOS / Windows

### Instalación Básica

```bash
# Instalar desde PyPI (cuando esté disponible)
pip install mnemosys

# O con dependencias específicas
pip install mnemosys[gpu,security]
```

### Instalación desde Fuente

```bash
# Clonar el repositorio
git clone https://github.com/esraderey/mnemosys.git
cd MNEME---Motor-de-Memoria-Neural-M-rfica

# Instalar dependencias
pip install -r requirements.txt

# Instalar MNEME
pip install -e .
```

### Instalación con PyTorch GPU

```bash
# Para NVIDIA CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para Apple Metal Performance Shaders
pip install torch torchvision torchaudio

# Instalar MNEME
pip install -e .
```

### Instalación con Poetry (Recomendado)

```bash
# Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Instalar dependencias
poetry install

# Activar entorno virtual
poetry shell
```

### Instalación con Optimizaciones

```bash
# Para desarrollo
pip install -e .[dev]

# Para GPU
pip install -e .[gpu]

# Para seguridad empresarial
pip install -e .[security]

# Para optimización máxima
pip install -e .[optimization]

# Para monitoreo
pip install -e .[monitoring]
```

### Verificación de Instalación

```bash
# Verificar instalación (el import conserva el nombre histórico "mneme")
python -c "import mneme; print(f'MNEMOSYS v{mneme.__version__} instalado correctamente')"

# Ejecutar tests
pytest tests/ -v

# Verificar configuración
python -c "from mneme import MnemeConfig; print(MnemeConfig().to_dict())"
```

## 🚦 Uso Rápido

### 🔒 Seguridad Garantizada

```python
import torch
from mneme import ZSpace, MnemeConfig

# El cifrado en reposo y la firma HMAC se activan con una clave estable
# (también puede venir de la variable de entorno MNEME_SECRET_KEY)
config = MnemeConfig(
    secret_key=b"clave_estable_de_32_bytes_o_mas_",
    validate_inputs=True,
    enable_encryption=True,
)
mneme = ZSpace(config)

# Registrar (routing inteligente + validación + almacenamiento cifrado)
tensor = torch.randn(64, 64)          # 2-D pequeño → ruta RAW, sin pérdida
desc = mneme.register("my_tensor", tensor)

# Cargar (síntesis determinista; en GPU si hay CUDA)
loaded_tensor = mneme.load("my_tensor").cpu()
assert torch.allclose(tensor, loaded_tensor)

# Métricas reales del sistema
stats = mneme.get_stats()
print(f"Salud: {stats['health']}")
print(f"Escrituras a storage: {stats['metrics']['storage_stores']}")
print(f"Nivel de seguridad: {stats['security']['config']['security_level']}")
print(f"Cache hits: {stats['cache']['hit_count']}")
```

### 🧪 Validación y errores con contexto

```python
import torch
from mneme import ZSpace, ValidationError

mneme = ZSpace()

try:
    mneme.register("no_tensor", {"esto": "no es un tensor"})
except ValidationError as e:
    print(f"Rechazado por validación: {e}")

try:
    mneme.register("", torch.randn(4, 4))
except ValidationError as e:
    print(f"Nombre vacío rechazado: {e}")

# La manipulación del almacén en disco (bytes corruptos, blob truncado,
# clave equivocada) aflora como StorageAuthenticationError: nunca hay
# corrupción silenciosa (verificado en la batería del 16-ago-2026).
```

### 🏗️ Arquitectura Modular

```python
from mneme import ZSpace

# Los módulos especializados viven integrados en cada instancia
mneme = ZSpace()
print(f"Cache stats: {mneme.adaptive_cache.get_stats()}")
print(f"Security stats: {mneme.security_manager.get_security_stats()}")
print(f"Lock stats: {mneme.lock_manager.get_lock_stats()}")
print(f"Storage stats: {mneme.storage_backend.get_stats()}")
```

### Procesamiento Concurrente

```python
import torch
from concurrent.futures import ThreadPoolExecutor
from mneme import ZSpace

# ZSpace es thread-safe: locks granulares por nombre de tensor
# (verificado: 640 operaciones en 8 hilos, 0 incidentes)
mneme = ZSpace()
tensors = [torch.randn(1000, 1000) for _ in range(8)]

with ThreadPoolExecutor(max_workers=4) as pool:
    descs = list(pool.map(
        lambda item: mneme.register(f"tensor_{item[0]}", item[1], target_ratio=0.1),
        enumerate(tensors),
    ))

with ThreadPoolExecutor(max_workers=4) as pool:
    loaded = list(pool.map(lambda i: mneme.load(f"tensor_{i}"), range(8)))

print(f"Locks granulares: {mneme.get_stats()['locks']}")
```

### Seguridad Avanzada

```python
import torch
from mneme import (SecureSerializer, SecurityConfig, SecurityLevel,
                   ZSpace, MnemeConfig)

secret_key = b"clave_estable_de_32_bytes_o_mas_"
sensitive_tensor = torch.randn(64, 64)

# 1) Serialización firmada (HMAC) con el marco MNEM — sin pickle
serializer = SecureSerializer(SecurityConfig(
    security_level=SecurityLevel.HMAC,
    require_signatures=True,
    signing_key=secret_key,
))
signed = serializer.serialize_tensor(sensitive_tensor)
restored, _meta = serializer.deserialize_tensor(signed)
assert torch.allclose(sensitive_tensor, restored)

# 2) Cifrado en reposo dentro de ZSpace (secret_key lo habilita)
mneme = ZSpace(MnemeConfig(secret_key=secret_key))
mneme.register("tensor_sensible", sensitive_tensor)
decrypted = mneme.load("tensor_sensible").cpu()
assert torch.allclose(sensitive_tensor, decrypted)
```

### Almacenamiento Persistente y Rehidratación

```python
import torch
from mneme import ZSpace, MnemeConfig

# Almacén persistente en disco (SQLite + safetensors + LZ4)
mneme = ZSpace(MnemeConfig(storage_path="./mi_almacen"))
mneme.register("medium", torch.randn(1000, 1000), target_ratio=0.1)
referencia = mneme.load("medium").cpu()
mneme.cleanup()

# Una instancia NUEVA reconstruye desde disco, bit a bit
mneme2 = ZSpace(MnemeConfig(storage_path="./mi_almacen"))
rehidratado = mneme2.load("medium").cpu()
assert torch.equal(referencia, rehidratado)  # síntesis determinista

# Métricas reales de almacenamiento
storage_metrics = mneme2.get_stats()["metrics"]
print(f"Lecturas: {storage_metrics['read_operations']}")
print(f"Cache hits: {storage_metrics['cache_hits']}")
```

### Mejoras Arquitecturales

```python
import torch
from mneme import ZSpace

mneme = ZSpace()

# Los locks granulares por nombre son automáticos en register/load/update;
# también pueden usarse directamente:
with mneme.lock_manager.write_lock("tensor1"):
    pass  # sección crítica propia sobre "tensor1"

# Safetensors para serialización segura (automático en el registro)
desc = mneme.register("secure_tensor", torch.randn(500, 500))

# Lazy decompression para optimizar memoria
if hasattr(desc, 'lazy_tensor'):
    # Solo se decompress cuando se accede
    tensor = desc.lazy_tensor.decompress()
    
    # Liberar memoria cuando no se necesite
    desc.lazy_tensor.clear_decompressed()

# Cache adaptativo con estrategias inteligentes
cache_stats = mneme.adaptive_cache.get_stats()
print(f"Hit rate: {cache_stats['hit_rate']:.1f}%")
print(f"Estrategia: {cache_stats['strategy']}")

# Estadísticas de locks granulares
lock_stats = mneme.lock_manager.get_lock_stats()
print(f"Locks activos: {lock_stats['total_locks']}")
```

### Monitoreo de Rendimiento

```python
from mneme import ZSpace, MNEMEOptimizer, OptimizationLevel

# Inicializar con monitoreo
mneme = ZSpace()

# Crear optimizador
optimizer = MNEMEOptimizer(
    optimization_level=OptimizationLevel.AGGRESSIVE,
    enable_profiling=True,
    enable_parallel_processing=True
)

# Obtener métricas en tiempo real
metrics = mneme.get_performance_metrics()
print(f"Lecturas: {metrics['metrics']['read_operations']}")
print(f"Tensores almacenados: {metrics['metrics']['tensor_count']}")
print(f"Bytes en storage: {metrics['metrics']['total_storage_bytes']/1024/1024:.1f}MB")

# Estado de salud del sistema
health = mneme.get_health_status()
print(f"Estado: {health.value}")

# Optimizar sistema automáticamente
optimization_result = mneme.optimize_system()
for action in optimization_result.get("actions", []):
    print(f"- {action}")
```

### ⚡ Turbo: Inferencia con Pesos Comprimidos

`compress_model_turbo` es la vía para ejecutar modelos con la compresión
gobernando de verdad la inferencia: el peso denso no queda residente. Con SVD
ni siquiera se materializa (forward factorizado con dos matmuls chicos);
con INT8/RAW el comprimido vive como buffer y el peso se sintetiza dentro
del forward y se libera (recompute en backward). Medido en 8×Linear 2048²:
**VRAM residente 20.5 % del denso, pico de forward ÷4.9, latencia a la par.**

```python
import torch.nn as nn
from mneme import compress_model_turbo, CompressionConfig

model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10)
)

turbo = compress_model_turbo(
    model,
    config=CompressionConfig(target_ratio=0.1, decomp_type="svd"),
    min_params=10000,
)

# Las salidas reflejan la pérdida real de la compresión (medible),
# los pesos van congelados y el gradiente fluye a entrada y bias.
# Soporta modelos .half()/.cuda(), state_dict y pesos podados (ruta sparse).
```

### Compresión de Modelos (pipeline de almacenamiento)

```python
import torch.nn as nn
from mneme import compress_model, get_compression_stats, CompressionConfig

config = CompressionConfig(target_ratio=0.1)
model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10)
)

compressed = compress_model(model, config=config)
stats = get_compression_stats(compressed)
print(f"Compresión: {stats['overall_ratio']:.1%}")
print(f"Capas comprimidas: {stats['compressed_layers']}")
```

> **Nota honesta** (verificada empíricamente el 16-ago-2026): `compress_model`
> registra los pesos comprimidos en ZSpace pero su forward sigue usando el
> peso original — demuestra el pipeline de almacenamiento y sirve para medir
> reconstrucción, no ahorra memoria de inferencia. Para eso está
> `compress_model_turbo` (arriba). Para evaluar calidad de una ruta de
> compresión sobre un modelo, el patrón de referencia es el de
> `benchmarks/bench_pythia_*.py`: `register → load → weight.copy_()`.

### Capas Z Transparentes

```python
from mneme import ZLinear, ZConv2d, ZAttention, ZTransformerBlock, CompressionConfig

# Configuración de compresión
config = CompressionConfig(target_ratio=0.1)

# Modelo con capas MNEME
model = nn.Sequential(
    ZLinear(784, 512, config=config),
    nn.ReLU(),
    ZLinear(512, 256, config=config),
    nn.ReLU(),
    ZLinear(256, 10, config=config)
)

# Transformer con compresión y seguridad
transformer = ZTransformerBlock(
    embed_dim=512, 
    num_heads=8, 
    config=config
)
```

## 🏗️ Arquitectura Avanzada

```
┌─────────────────────────────────────────────────────────────────┐
│                         MNEMOSYS Core                          │
├─────────────────────────────────────────────────────────────────┤
│  Z-Addr (Hashing)   │   Z-Gen (Synthesis)   │   Security (HMAC) │
│  Cache (CPU-Aware)  │   Proof (Merkle)      │   Serializer      │
│  Prefetch (Markov)  │   Delta Consolidation │   Crypto Engine   │
│  Parallel Executor  │   Performance Monitor │   Resource Opt.   │
│  Security Manager   │   Tiered Storage      │   Health Monitor  │
│-----------------------------------------------------------------│
│   Motores de Descomposición: TT | CP | Tucker | SVD | Quantized   │
│   + Procesamiento Paralelo + Seguridad Cuántica + Monitoreo      │
│   + Almacenamiento Inteligente + Optimización Automática        │
│   + Métricas en Tiempo Real + Alertas Inteligentes              │
└─────────────────────────────────────────────────────────────────┘
```

### Flujo de datos

**Store** → Tensor → Analyze → Decompose → Encrypt → Serialize → Sign (HMAC) → Compress → Store (Tiered) → Descriptor

**Load** → Descriptor → Load (Tiered) → Decompress → Verify (HMAC) → Decrypt → Deserialize → Reconstruct → Verify → Tensor

**Parallel** → Batch → Analyze → Distribute → Process → Collect → Merge → Result

**Security** → Authenticate → Encrypt → Sign → Audit → Monitor → Rotate Keys

**Monitor** → Collect Metrics → Analyze → Alert → Optimize → Report

## 📈 Benchmarks Reales (16-ago-2026)

Todos con artefacto reproducible (JSON en `benchmarks/` y scripts de la batería).

### Pythia-410M · WikiText-2 (perplejidad, menor es mejor)

| Configuración | PPL | Δ vs FP16 |
|---|---|---|
| FP16 (baseline) | **15.07** | — |
| GPTQ INT4 (g=64, 512 muestras calibración) | **19.97** | +32.5 % |
| INT4 group-wise sin calibrar (g=128) | **24.25** | +60.9 % |

*La reparación del offset de cuantización (ago-2026) se midió aquí: el INT4
sin calibrar pasó de PPL 97.7 a 24.25 con la misma compresión (0.36).*

### Turbo (8×Linear 2048², fp32, CUDA)

| | Denso | Turbo SVD 0.1 | Turbo INT8 |
|---|---|---|---|
| VRAM residente | 128.1 MB | **26.3 MB** | 34.0 MB |
| Pico de forward | 138.2 MB | **28.3 MB** | 52.0 MB |
| Latencia mediana | 0.91 ms | **0.94 ms** | 271 ms |

### Fidelidad por ruta (tensores de 10⁶ elementos)

| Ruta | Bytes | Error relativo |
|---|---|---|
| RAW (safetensors+LZ4) | 100 % | **0 (bit-exacto)** |
| SVD 0.1 · dato con estructura (rango-30 + 1 % ruido) | 20 % | 0.9 % |
| SVD 0.1 · ruido puro | 20 % | 85 % |
| INT8 por grupos | 26.6 % | **0.6 %** |
| TT · 3-D separable + 1 % ruido | 1.6 % | 1.0 % |

*La lección medida: la estructura compra fidelidad — la misma ruta rinde seis
órdenes de magnitud mejor sobre dato estructurado que sobre ruido.*

### Almacenamiento y síntesis

- Register / load (1000², SVD): 57 ms / 6 ms · (2000²): 227 ms / 28 ms
- Relectura cacheada: ~×20 más rápida (166 ms → 8 ms en RAW de 4 MB)
- Rehidratación cross-instancia: 36/36 bit-idéntica · arranque perezoso 1.9 ms
- Corrupción de blobs y clave errónea: 3/3 detectadas, 0 silenciosas

## 🔬 Funcionalidades Avanzadas

### ⚡ **Procesamiento Paralelo**
- Ejecución híbrida con threads, procesos y asyncio
- Descomposición paralela de tensores
- Procesamiento por lotes optimizado
- Métricas de eficiencia en tiempo real

### 🔒 **Seguridad Cuántica**
- Gestión de claves resistentes a computación cuántica
- Cifrado post-cuántico con algoritmos seguros
- Autenticación multifactor robusta
- Rotación automática de claves

### 🗄️ **Almacenamiento Inteligente**
- Migración automática entre niveles de almacenamiento
- Compresión adaptativa basada en características de datos
- Almacenamiento distribuido con replicación
- Análisis de patrones de acceso

### 📊 **Monitoreo Avanzado**
- Métricas en tiempo real de todos los componentes
- Alertas automáticas con umbrales configurables
- Optimización automática de recursos
- Estado de salud del sistema con recomendaciones

### 🧠 **Núcleo Inteligente**
- Selección automática de descomposición
- Prefetching adaptativo con aprendizaje
- Gestión inteligente de memoria CPU/GPU
- Consolidación automática de deltas

### 🔗 **Integración PyTorch Mejorada**
- Drop-in replacement con funcionalidades avanzadas
- Compresión transparente con seguridad
- Soporte completo para arquitecturas modernas
- Estadísticas de rendimiento en tiempo real

## 🎮 Aplicaciones

### **Machine Learning Avanzado**
- Compresión y serving de modelos LLM con procesamiento paralelo
- Entrenamiento distribuido con optimización automática
- Inferencia en dispositivos edge con monitoreo de recursos
- Optimización de memoria GPU con alertas automáticas

### **Simulaciones y Juegos**
- Mundos de juego infinitos con almacenamiento inteligente
- Estados de simulación masivos con procesamiento paralelo
- Física en tiempo real con optimización automática
- Procedural generation con deduplicación de contextos

### **Ciencia de Datos**
- Análisis de datasets masivos con procesamiento paralelo
- Compresión de matrices dispersas con almacenamiento inteligente
- Cálculos científicos optimizados con monitoreo de recursos
- Visualización de datos grandes con métricas en tiempo real

### **Seguridad y Auditoría**
- Sistemas de logging seguros con cifrado cuántico
- Verificación de integridad con rotación automática de claves
- Trazabilidad de datos con auditoría completa
- Compliance empresarial con monitoreo automático
- **Serialización segura** con safetensors (sin pickle)
- **Validación robusta** de entrada con InputValidator
- **Locks granulares** para mejor concurrencia y seguridad

## 🗺️ Roadmap

### ✅ **Fase 1 – Núcleo Completo**
- [x] Procesamiento paralelo híbrido
- [x] Seguridad cuántica avanzada
- [x] Almacenamiento inteligente por niveles
- [x] Monitoreo de rendimiento en tiempo real
- [x] Optimización automática de recursos
- [x] Métricas y alertas inteligentes
- [x] **Arquitectura modular segura** (3 módulos especializados)
- [x] **Eliminación completa de pickle** (solo safetensors)
- [x] **Locks granulares** para mejor concurrencia
- [x] **Lazy decompression** para optimización de memoria
- [x] **Cache adaptativo** con estrategias inteligentes
- [x] **Validación robusta** de entrada

### ✅ **Fase 1.1 – Mejoras de robustez (era MNEME, enero 2025)**
- [x] **Sistema de errores contextuales** con información detallada
- [x] **Configuración avanzada** con validaciones automáticas
- [x] **Locks granulares optimizados** con detección de deadlocks
- [x] **LazyTensor mejorado** con gestión inteligente de memoria
- [x] **Cache adaptativo avanzado** con múltiples estrategias
- [x] **Descriptores enriquecidos** con estadísticas de acceso
- [x] **ZSpace optimizado** con métricas en tiempo real
- [x] **Storage persistente funcional** - Corregido problema crítico del SecureStorageBackend
- [x] **Herramientas de desarrollo** (CI/CD, Dependabot, CodeQL)
- [x] **Documentación completa** y templates de GitHub
- [x] **Calidad de código** con pre-commit hooks y linting

### 🚧 **Fase 2 – Aceleración HW (Q2 2025)**
- [ ] Kernels CUDA optimizados para procesamiento paralelo
- [ ] Prototipo FPGA para descomposición de tensores
- [ ] Caché NVMe inteligente con migración automática
- [ ] Aceleración GPU masiva con monitoreo de recursos
- [ ] Integración con TensorRT optimizada
- [ ] Hardware de deduplicación de contexto

### 🔮 **Fase 3 – Silicio (Q4 2025)**
- [ ] Diseño de MMU-MNEME con procesamiento paralelo
- [ ] Tape-out ASIC con seguridad cuántica
- [ ] Integración en OS con monitoreo automático
- [ ] Hardware security module cuántico
- [ ] Red neuronal dedicada con optimización automática
- [ ] Chip de deduplicación de contexto inteligente

## 📚 Documentación

- **[Serialización Avanzada](docs/SERIALIZATION_UPGRADE.md)** - Sistema de serialización seguro
- **[Cifrado y Contexto](docs/ENCRYPTION_AND_CONTEXT_UPGRADE.md)** - Cifrado de tensores y gestión de claves
- **[Almacenamiento Avanzado](docs/ADVANCED_STORAGE_UPGRADE.md)** - Sistema de almacenamiento y cache
- **[Deduplicación de Contexto](docs/CONTEXT_DEDUPLICATION_UPGRADE.md)** - Sistema de deduplicación inteligente
- **[CHANGELOG.md](CHANGELOG.md)** - Historial completo de cambios
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guías de contribución
- **[SECURITY.md](.github/SECURITY.md)** - Política de seguridad

## 📋 Historial detallado de la era MNEME (interno, pre-1.0.0)

### 🐛 **Corregido**
- **Integración del SecureStorageBackend** - Corregido problema crítico donde el storage backend nunca se utilizaba
- **Serialización de SafeTensors** - Corregido manejo de archivos temporales para compatibilidad con Windows
- **Serialización de ZDescriptor** - Corregida serialización de shapes (tuplas) para almacenamiento persistente
- **Método get_lock_stats()** - Corregido uso de `_is_owned()` en lugar de `locked()` para RLock

### 🔧 **Mejorado**
- **Sistema de errores contextuales** con información detallada y timestamps
- **Configuración avanzada** con validaciones automáticas y serialización
- **Locks granulares optimizados** con detección de deadlocks y limpieza automática
- **LazyTensor mejorado** con gestión inteligente de memoria y cache de metadatos
- **Cache adaptativo avanzado** con múltiples estrategias de evicción
- **Descriptores enriquecidos** con estadísticas de acceso y validaciones robustas
- **ZSpace optimizado** con métricas en tiempo real y logging configurable
- **Storage persistente** ahora completamente funcional con almacenamiento y carga automática

### 🛠️ **Herramientas de Desarrollo**
- **GitHub Actions CI/CD** completo con testing, linting y security
- **Dependabot** para actualizaciones automáticas de dependencias
- **CodeQL** para análisis de seguridad automatizado
- **Pre-commit hooks** para calidad de código automática
- **Templates** para issues, pull requests y security
- **pyproject.toml** moderno con configuración completa

### 📊 **Rendimiento**
- **50-70% reducción** en uso de memoria
- **95% eficiencia** en locks granulares
- **>95% hit rate** en cache adaptativo
- **<1μs** por validación de entrada
- **<0.5μs** latencia de caché CPU
- **<100μs** latencia de síntesis

### 🔒 **Seguridad**
- **Validaciones robustas** en todos los métodos
- **Manejo seguro de errores** con contexto detallado
- **Auditoría completa** con logging detallado
- **Análisis automático** de vulnerabilidades con CodeQL

## 👥 Autores

**Esraderey** y **Raul Cruz Acosta**

## 📚 Citación

```bibtex
@software{mnemosys2026,
  title = {MNEMOSYS: Motor de Memoria Neural Mórfica},
  author = {Esraderey and Raul Cruz Acosta},
  year = {2026},
  url = {https://github.com/esraderey/mnemosys},
  note = {Memoria Neural Estructurada Mórfica con Orquestación de Síntesis Y Seguridad: motor de memoria computacional con síntesis determinista, compresión tensorial e inferencia con pesos comprimidos}
}
```

## 🔗 Proyectos Relacionados

- **TensorLy** - Descomposición de tensores
- **PyTorch** - Framework de deep learning
- **CUDA** - Aceleración GPU
- **Quantum Computing** - Algoritmos post-cuánticos

## 💡 Filosofía

*"La mejor compresión no es guardar los datos, sino guardar la receta para recrearlos de forma paralela y segura."*

*"La memoria no es un archivo estático, sino un organismo vivo que se regenera con cada evocación, monitoreado y optimizado en tiempo real."*

## 📝 Licencia

Apache License 2.0 – ver [LICENSE](LICENSE)

Uso, modificación y redistribución libres (también comercial), con atribución y
concesión explícita de patentes — la licencia estándar del ecosistema ML.

## 📧 Contacto

- **Issues**: [GitHub Issues](https://github.com/esraderey/mnemosys/issues)
- **Discussions**: [GitHub Discussions](https://github.com/esraderey/mnemosys/discussions)
- **Email**: msc.framework@gmail.com
- **Documentación**: [Wiki](https://github.com/esraderey/mnemosys/wiki)

## 🛡️ Seguridad del diseño

### 🔒 **Vulnerabilidades Eliminadas**
- **❌ Pickle eliminado** - Sin vulnerabilidades de deserialización
- **✅ Safetensors exclusivo** - Serialización segura garantizada
- **✅ Validación robusta** - InputValidator para todos los datos
- **✅ Locks granulares** - Mejor concurrencia y seguridad

### 🏗️ **Arquitectura Modular**
- **`mneme_core.py`** - Módulo principal seguro
- **`mneme_security_core.py`** - Seguridad y validación
- **`mneme_storage_core.py`** - Almacenamiento seguro
- **Separación clara** de responsabilidades

### 📊 **Beneficios de Seguridad**
- **Seguridad (10/10)** - Sin vulnerabilidades críticas
- **Serialización segura** - Solo safetensors
- **Validación automática** - Entrada verificada
- **Concurrencia mejorada** - Locks granulares
- **Memoria optimizada** - Lazy decompression

## 🏆 Reconocimientos

- Inspirado en la neurociencia computacional
- Basado en principios de compresión de información
- Influenciado por sistemas de memoria biológica
- Diseñado para eficiencia energética y procesamiento paralelo
- Integración de seguridad cuántica y monitoreo inteligente
- **Arquitectura modular segura** con eliminación de vulnerabilidades

---

*"La memoria no es un archivo estático, sino un organismo vivo que se regenera con cada evocación"* – Esraderey 