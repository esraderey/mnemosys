# Actualización de Sistema de Almacenamiento Avanzado - MNEME

## Resumen de Cambios

Se ha implementado un sistema de almacenamiento avanzado y cache inteligente que proporciona múltiples backends, políticas de cache sofisticadas, deduplicación automática y monitoreo de salud en tiempo real.

## Nuevas Características Implementadas

### 1. Sistema de Almacenamiento Multi-Backend

#### Backends Disponibles
- **MemoryStorage**: Almacenamiento en memoria con límites configurables
- **DiskStorage**: Almacenamiento persistente en disco con metadatos
- **RedisStorage**: Cache distribuido usando Redis
- **HybridStorage**: Combinación inteligente de múltiples backends

#### Clase StorageBackendInterface
```python
class StorageBackendInterface:
    def store(self, key: bytes, data: bytes) -> bool
    def retrieve(self, key: bytes) -> Optional[bytes]
    def delete(self, key: bytes) -> bool
    def exists(self, key: bytes) -> bool
    def list_keys(self) -> List[bytes]
    def get_stats(self) -> Dict[str, Any]
```

### 2. Sistema de Cache Inteligente

#### Políticas de Cache Disponibles
- **LRU**: Least Recently Used
- **LFU**: Least Frequently Used
- **FIFO**: First In, First Out
- **LIFO**: Last In, First Out
- **TTL**: Time To Live
- **ADAPTIVE**: Política adaptativa que combina LRU y LFU

#### Clase AdvancedCache
```python
class AdvancedCache:
    def get(self, key: bytes) -> Optional[bytes]
    def put(self, key: bytes, value: bytes) -> bool
    def delete(self, key: bytes) -> bool
    def exists(self, key: bytes) -> bool
    def clear(self)
    def get_stats(self) -> Dict[str, Any]
```

### 3. Motor de Deduplicación

#### Características
- **Detección automática**: Identifica contenido duplicado usando SHA256
- **Referencias múltiples**: Soporte para múltiples referencias al mismo contenido
- **Limpieza automática**: Elimina referencias huérfanas
- **Métricas detalladas**: Estadísticas de eficiencia de deduplicación

#### Clase DeduplicationEngine
```python
class DeduplicationEngine:
    def should_deduplicate(self, data: bytes) -> Optional[bytes]
    def register_content(self, key: bytes, data: bytes) -> bytes
    def unregister_content(self, key: bytes) -> bool
    def get_stats(self) -> Dict[str, Any]
```

### 4. Estrategias de Compresión

#### Algoritmos Disponibles
- **NONE**: Sin compresión
- **LZ4**: Compresión rápida
- **ZSTD**: Compresión balanceada
- **GZIP**: Compresión estándar
- **ADAPTIVE**: Selección automática basada en contenido

### 5. Monitoreo de Salud

#### Métricas de Salud
- **Puntuación de salud**: 0-100 basada en múltiples factores
- **Uso de cache**: Porcentaje de utilización
- **Tasa de aciertos**: Eficiencia del cache
- **Ratio de deduplicación**: Eficiencia de deduplicación
- **Advertencias**: Alertas automáticas

## Configuración Actualizada

### Nuevas Opciones en MnemeConfig

```python
@dataclass
class MnemeConfig:
    # ... opciones existentes ...
    
    # Sistema de almacenamiento avanzado
    storage_backend: StorageBackend = StorageBackend.HYBRID
    storage_path: Optional[str] = None
    enable_persistent_storage: bool = True
    enable_deduplication: bool = True
    
    # Cache avanzado
    cache_policy: CachePolicy = CachePolicy.ADAPTIVE
    cache_size_mb: int = 1024
    cache_ttl_seconds: int = 3600
    enable_distributed_cache: bool = False
    redis_url: Optional[str] = None
    
    # Compresión adaptativa
    compression_strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE
    enable_adaptive_compression: bool = True
    compression_threshold_mb: float = 1.0
    
    # Indexación y búsqueda
    enable_indexing: bool = True
    index_type: str = "btree"
    enable_full_text_search: bool = False
    
    # Métricas y monitoreo
    enable_storage_metrics: bool = True
    metrics_interval_seconds: int = 60
```

## Ejemplos de Uso

### 1. Almacenamiento Básico

```python
config = MnemeConfig(
    storage_backend=StorageBackend.DISK,
    storage_path="./my_storage",
    cache_policy=CachePolicy.LRU,
    enable_deduplication=True
)

with mneme_context(config) as mneme:
    tensor = torch.randn(100, 100)
    desc = mneme.register("my_tensor", tensor)
    loaded_tensor = mneme.load("my_tensor")
```

### 2. Cache Distribuido con Redis

```python
config = MnemeConfig(
    storage_backend=StorageBackend.HYBRID,
    enable_distributed_cache=True,
    redis_url="redis://localhost:6379",
    cache_policy=CachePolicy.ADAPTIVE
)
```

### 3. Optimización de Almacenamiento

```python
with mneme_context(config) as mneme:
    # Obtener salud del almacenamiento
    health = mneme.get_storage_health()
    print(f"Salud: {health['health_score']}/100")
    
    # Optimizar almacenamiento
    results = mneme.optimize_storage()
    print(f"Espacio ahorrado: {results['space_saved_bytes']} bytes")
```

### 4. Operaciones Asíncronas

```python
async def async_storage():
    config = MnemeConfig(
        enable_async_context=True,
        max_concurrent_operations=10
    )
    
    async with async_mneme_context(config) as mneme:
        # Operaciones concurrentes
        tasks = []
        for i in range(20):
            tensor = torch.randn(100, 100)
            task = mneme.register_async(f"tensor_{i}", tensor)
            tasks.append(task)
        
        descriptors = await asyncio.gather(*tasks)
```

## Beneficios del Sistema

### 1. Rendimiento
- **Cache inteligente**: Múltiples políticas para diferentes patrones de uso
- **Deduplicación**: Ahorro significativo de espacio
- **Compresión adaptativa**: Optimización automática según el contenido
- **Operaciones concurrentes**: Hasta 10 operaciones simultáneas

### 2. Escalabilidad
- **Backends múltiples**: Memoria, disco, Redis, híbrido
- **Distribución**: Soporte para cache distribuido
- **Persistencia**: Almacenamiento duradero
- **Replicación**: Múltiples copias para redundancia

### 3. Monitoreo
- **Métricas en tiempo real**: Uso, eficiencia, salud
- **Alertas automáticas**: Detección de problemas
- **Optimización**: Limpieza y compresión automática
- **Estadísticas detalladas**: Análisis de rendimiento

### 4. Flexibilidad
- **Configuración granular**: Cada componente se puede ajustar
- **Políticas adaptativas**: Ajuste automático según patrones
- **Backends intercambiables**: Fácil cambio entre sistemas
- **Compatibilidad**: Funciona con configuraciones existentes

## Arquitectura del Sistema

### Flujo de Almacenamiento
1. **Verificación de deduplicación**: Buscar contenido existente
2. **Cache**: Almacenar en cache para acceso rápido
3. **Backend persistente**: Almacenar en disco/Redis
4. **Métricas**: Actualizar estadísticas

### Flujo de Recuperación
1. **Cache**: Buscar en cache primero
2. **Backend**: Recuperar desde almacenamiento persistente
3. **Cache update**: Cargar en cache para futuras consultas
4. **Métricas**: Actualizar estadísticas de acceso

## Nuevas Clases y Enums

### Enums Agregados
```python
class StorageBackend(Enum):
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    S3 = "s3"
    HDFS = "hdfs"
    HYBRID = "hybrid"

class CachePolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    LIFO = "lifo"
    ADAPTIVE = "adaptive"
    TTL = "ttl"

class CompressionStrategy(Enum):
    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    GZIP = "gzip"
    ADAPTIVE = "adaptive"
```

### Clases Principales
- **StorageBackendInterface**: Interfaz base para backends
- **MemoryStorage**: Almacenamiento en memoria
- **DiskStorage**: Almacenamiento en disco
- **RedisStorage**: Almacenamiento en Redis
- **HybridStorage**: Almacenamiento híbrido
- **AdvancedCache**: Sistema de cache inteligente
- **DeduplicationEngine**: Motor de deduplicación

## Métricas y Monitoreo

### Métricas de Cache
- **Hit rate**: Porcentaje de aciertos
- **Usage**: Porcentaje de utilización
- **Evictions**: Número de evicciones
- **Entries**: Número de entradas

### Métricas de Almacenamiento
- **Total size**: Tamaño total almacenado
- **Entries**: Número de entradas
- **Deduplication ratio**: Eficiencia de deduplicación
- **Health score**: Puntuación de salud (0-100)

### Métricas de Deduplicación
- **Unique contents**: Contenidos únicos
- **Total references**: Referencias totales
- **Deduplication ratio**: Ratio de deduplicación

## Archivos Creados/Modificados

1. **mneme_core.py**: Implementación completa del sistema de almacenamiento
2. **example_advanced_storage.py**: Ejemplos de uso completos
3. **ADVANCED_STORAGE_UPGRADE.md**: Esta documentación

## Próximos Pasos

1. **Configurar backends**: Elegir backend de almacenamiento apropiado
2. **Ajustar cache**: Configurar política y tamaño de cache
3. **Habilitar deduplicación**: Activar para ahorrar espacio
4. **Configurar monitoreo**: Habilitar métricas y alertas
5. **Probar ejemplos**: Ejecutar `example_advanced_storage.py`
6. **Optimizar rendimiento**: Ajustar configuración según necesidades

## Notas de Rendimiento

### Cache
- **LRU**: Mejor para patrones temporales
- **LFU**: Mejor para patrones de frecuencia
- **ADAPTIVE**: Mejor para patrones mixtos
- **TTL**: Mejor para datos con expiración

### Almacenamiento
- **MEMORY**: Más rápido, menos persistente
- **DISK**: Balance velocidad/persistencia
- **REDIS**: Cache distribuido, alta disponibilidad
- **HYBRID**: Mejor de todos los mundos

### Deduplicación
- **Beneficios**: Ahorro significativo de espacio
- **Overhead**: Pequeño costo computacional
- **Umbral**: Activar solo si hay contenido duplicado

Esta actualización proporciona un sistema de almacenamiento robusto, escalable y eficiente que mejora significativamente las capacidades de MNEME para manejar grandes volúmenes de datos con optimizaciones automáticas y monitoreo en tiempo real.
