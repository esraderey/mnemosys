# DELETED_CODE.md - Justificacion de Eliminaciones (Sesiones 1-4)

Este documento registra cada pieza de codigo eliminada durante el refactoring,
con su proposito original y la evidencia concreta de por que se elimino.

---

## Sesion 1: Bug Fixes + Deduplicacion

### FASE 2A: Clases Deduplicadas

| Nombre | Tipo | Archivo origen | Accion | Razon |
|--------|------|---------------|--------|-------|
| CircuitBreaker | clase | mneme_optimization.py | Borrado, importado de core | Duplicado exacto (~150 lineas) |
| HealthStatus | enum | mneme_optimization.py | Borrado, importado de core | Duplicado (se anadieron RECOVERING/MAINTENANCE a core) |
| CircuitState | enum | mneme_optimization.py | Borrado, importado de core | Duplicado exacto |

### FASE 2B: Enums Deduplicados

| Nombre | Tipo | Archivo origen | Accion | Razon |
|--------|------|---------------|--------|-------|
| SecurityLevel | enum | mneme_core.py | Borrado, importado de security_core | Pertenece al modulo de seguridad |
| TensorEncryptionMode | enum | mneme_core.py | Borrado completamente | Fachada vacia: 0 implementacion real, campos de MnemeConfig convertidos a str |
| KeyRotationPolicy | enum | mneme_core.py | Borrado completamente | Fachada vacia: 0 implementacion real, campos de MnemeConfig convertidos a str |

---

## Sesion 2: Infraestructura Muerta (mneme_core.py)

### FASE 2C: Event System

| Nombre | Tipo | Lineas aprox | Proposito original | Evidencia de eliminacion |
|--------|------|-------------|-------------------|------------------------|
| EventType | enum | ~12 | Tipos de eventos (TENSOR_STORED, CACHE_HIT, etc.) | 0 handlers suscritos en todo el proyecto |
| MnemeEvent | dataclass | ~16 | Evento con timestamp, trace_id, data | Solo se instanciaba para emitir al vacio |
| EventBus | clase | ~100 | Bus pub/sub con prioridades, sync/async | 0 suscriptores registrados jamas |
| EventHandler | protocol | ~7 | Protocolo para handlers de eventos | 0 implementaciones |
| MnemeContext.emit_event() | metodo | ~4 | Emitir evento a traves del bus | Todas las llamadas emitian al vacio |
| MnemeContext.event_bus | campo | ~1 | Referencia al EventBus | Sin consumidores |
| ~15 llamadas emit_event | llamadas | ~30 | Notificar store/load/delete/cache hit/miss | Emitian sin efecto (0 handlers) |

**Proposito original:** Observabilidad distribuida estilo OpenTelemetry. Componentes externos suscribirian handlers para monitorear operaciones.

**Por que se borro:** MetricsRegistry ya cubre la observabilidad funcional. El EventBus era andamiaje para un sistema de monitoreo externo que nunca se construyo.

### FASE 2D: Plugin System

| Nombre | Tipo | Lineas aprox | Proposito original | Evidencia de eliminacion |
|--------|------|-------------|-------------------|------------------------|
| Plugin | protocol | ~20 | Protocolo base: name, version, initialize, cleanup | 0 implementaciones en todo el proyecto |
| TensorTransformer | protocol | ~10 | Protocolo para transform/inverse_transform | 0 implementaciones |
| MetricsCollector | protocol | ~14 | Protocolo para record_metric/histogram | 0 implementaciones |
| PluginRegistry | clase | ~106 | Registro con categorias, lifecycle, descubrimiento | 0 plugins registrados jamas |
| MnemeConfig.enable_plugins | campo | ~1 | Flag para activar plugins | Activaba un sistema vacio |
| MnemeConfig.plugin_directories | campo | ~1 | Directorios de busqueda de plugins | Nunca usado |
| MnemeConfig.auto_load_plugins | campo | ~1 | Auto-carga de plugins | Nunca usado |
| MnemeContext.plugins | campo | ~3 | Referencia al PluginRegistry | Solo se usaba para list_plugins() en stats |

**Proposito original:** Extensibilidad via plugins. Terceros anadian transformadores de tensores, colectores de metricas, etc.

**Por que se borro:** API pura sin edificio. 0 plugins implementados, 0 usos de los protocols.

### FASE 2E: Pipeline System

| Nombre | Tipo | Lineas aprox | Proposito original | Evidencia de eliminacion |
|--------|------|-------------|-------------------|------------------------|
| PipelineStage | enum | ~10 | Etapas: PRE_VALIDATE, TRANSFORM, COMPRESS, etc. | pipeline.process() nunca se llama |
| PipelineContext | dataclass | ~16 | Contexto que fluye por el pipeline | Nunca instanciado fuera de ProcessingPipeline |
| PipelineStageHandler | ABC | ~18 | Handler abstracto por etapa | 0 subclases |
| PipelineError | exception | ~8 | Error de pipeline con stage info | Nunca lanzado (pipeline nunca ejecuta) |
| ProcessingPipeline | clase | ~102 | Pipeline multi-etapa con hooks pre/post | 0 stage handlers, 0 hooks, process() nunca llamado |
| LockType.PIPELINE | valor enum | ~1 | Tipo de lock para pipeline | Sin uso tras eliminar pipeline |
| ZSpace.pipeline | campo | ~1 | Referencia al ProcessingPipeline | store() y register() bypaseaban el pipeline completamente |

**Proposito original:** Pipeline multi-etapa con hooks pre/post para interceptar operaciones de store/register/load. Validacion, logging, transformacion en cadena.

**Por que se borro:** pipeline.process() nunca se llama. store() y register() van directo a _create_secure_descriptor/_create_smart_descriptor. 0 stage handlers registrados.

---

## Sesion 3: Enums Huerfanos, Config y Clases Sueltas

### FASE 2F: Enums y Clases Huerfanas

**De mneme_core.py:**

| Nombre | Tipo | Lineas aprox | Proposito original | Evidencia de eliminacion |
|--------|------|-------------|-------------------|------------------------|
| StorageBackend | enum | ~7 | Backends: MEMORY, DISK, REDIS, HYBRID, S3 | Nunca consultado; storage siempre usa SQLite via SecureStorageBackend |
| CompressionStrategy | enum | ~7 | Estrategias: LZ4, ZLIB, LZMA, ZSTD, ADAPTIVE | Nunca consultado; compresion siempre usa LZ4 |
| ContextSimilarityMethod | enum | ~6 | Metodos: COSINE, EUCLIDEAN, DOT_PRODUCT, JACCARD | Vestigio de feature no implementada (similarity search) |
| ContextClusteringMethod | enum | ~6 | Metodos: KMEANS, DBSCAN, HIERARCHICAL, SPECTRAL | Vestigio de feature no implementada (clustering) |
| SerializationFormat | enum | ~8 | Formatos: SAFETENSORS, TORCH, MSGPACK, JSON, etc. | Nunca consultado; siempre se usa safetensors |
| MetricPoint | dataclass | ~7 | Punto de metrica con name, value, timestamp, tags | Reemplazado por dict inline en MetricsRegistry._record_history() |
| MnemeFacade | clase | ~85 | Singleton facade sobre ZSpace con API simplificada | No exportada, no usada, no testeada. ZSpace ya es la API publica |

**De mneme_optimization.py:**

| Nombre | Tipo | Lineas aprox | Proposito original | Evidencia de eliminacion |
|--------|------|-------------|-------------------|------------------------|
| OptimizationStrategy | enum | ~8 | Estrategias: MEMORY_FIRST, SPEED_FIRST, etc. | 0 referencias en todo el codigo |
| Optimizable | protocol | ~3 | Protocolo: optimize(), get_memory_footprint() | 0 implementaciones |
| Compressible | protocol | ~3 | Protocolo: compress(), decompress() | 0 implementaciones |
| MetricsExporter | ABC | ~6 | Interfaz: export(), flush() | 0 subclases |
| WorkStealingQueue | clase | ~40 | Cola work-stealing entre workers | Instanciada por ParallelExecutor pero nunca usada para dispatch |
| Batch variables (ParallelTensorProcessor) | campos | ~5 | _batch_queue, _batch_lock, _batch_condition, etc. | Inicializados pero nunca usados |

### FASE 2G: Limpieza de MnemeConfig

| Campo eliminado | Razon |
|----------------|-------|
| serialization_format | Siempre safetensors; enum SerializationFormat eliminado |
| storage_backend | Siempre SQLite; enum StorageBackend eliminado |
| compression_strategy | Siempre LZ4; enum CompressionStrategy eliminado |
| enable_distributed_cache | No hay implementacion Redis |
| redis_url | No hay implementacion Redis |
| enable_plugins | Plugin system eliminado (Sesion 2) |
| plugin_directories | Plugin system eliminado (Sesion 2) |
| auto_load_plugins | Plugin system eliminado (Sesion 2) |

### FASE 2G: Actualizacion de __init__.py

- Eliminados exports de enums/clases borrados
- Anadidos exports faltantes: MNEMEOptimizer, OptimizationLevel
- __version__ sincronizado a "3.0.0"

---

## Sesion 4: Configuracion Muerta de CompressionConfig (2026-08-15)

### Campos retirados de CompressionConfig (mneme_torch.py)

Config que se podia fijar pero que ningun codigo de src/ leia: fijarla era un
no-op silencioso. Se decidio campo por campo entre cablear (si existia un
consumidor natural) o retirar; ninguno lo tenia, todos se retiran.

| Campo | Default | Proposito original | Evidencia de eliminacion |
|-------|---------|-------------------|--------------------------|
| compression_level | BALANCED | Nivel de compresion por capa | register() lo traga por **kwargs sin efecto (ningun parametro del core lo lee); los descriptores sellan el MnemeConfig.compression_level del ZSpace global (otro objeto). optimize_model_memory() lo escribia (MAXIMUM) pero nadie lo leia despues |
| memory_limit | None | Presupuesto de memoria por capa | 0 consumidores; el presupuesto real es optimize_model_memory(target_memory_mb), que aterriza via target_ratio |
| enable_quantization | True | Gatear el fallback INT8 del smart chain | El smart chain actual no tiene fallback INT8 que gatear: la cuantizacion es opt-in via quantization_type; 0 apariciones en el core |
| quantization_bits | 8 | Bits de cuantizacion | Superseded por quantization_type ("int8", "int4_group", "gptq_int4", ...); para la KV cache el campo vivo es kv_cache_bits |
| use_parallel_processing | True | Compresion paralela por capa | 0 consumidores; el paralelismo vive en MnemeConfig.enable_parallel_processing / max_workers |
| enable_security | False | Seguridad por capa | 0 consumidores; la seguridad vive en MnemeConfig.security_level / secret_key / enable_encryption |

Tambien se elimino la escritura huerfana `config.compression_level = CompressionLevel.MAXIMUM`
en optimize_model_memory(): la intencion de "compresion maxima" nunca aterrizaba; la palanca
viva de esa funcion es config.target_ratio (se sigue apretando a <= 0.05).

Campos que SI viven y no se tocaron: target_ratio, decomp_type (reenviado a las 5
register()), group_size, quantization_type, calibration_samples, calibration_data,
mixed_precision_policy, enable_kv_cache_compression, kv_cache_bits y
enable_structured_sparsity (leido por compress_model_calibrated; OJO: el core actual lo
recibe en **kwargs y no lo consume — solo aparece en docstrings; pendiente en chip aparte).
[Cerrado 16-ago-2026: el chip cableo el flag en el core — pre-pass 2:4 antes de cuantizar
en _create_quantized_descriptor (import perezoso de StructuredSparsifier), mascara
np.packbits en el payload y reaplicada en _dequantize_group_payload; sin quantization_type
se rechaza con ValidationError antes del circuit breaker (paridad con decomp_type).
Anclas: test_sparsity_* y test_compress_model_calibrated_entrega_sparsity_2_4.]

Anclas: test_compression_config_* y test_optimize_model_memory_comprime_sin_nivel_por_capa
en tests/test_regresiones_auditoria.py. Usos actualizados en tests/test_mneme.py,
examples/example_mneme.py, README.md y docs/README.md.

---

## Resumen Cuantitativo

| Sesion | Lineas eliminadas | Que se logro |
|--------|------------------|-------------|
| 1 | ~170 | Bugs arreglados, clases/enums deduplicados |
| 2 | ~465 | EventBus, PluginRegistry, Pipeline eliminados |
| 3 | ~330 | Enums huerfanos, MnemeConfig limpio, MnemeFacade eliminada |
| 4 | ~25 | Campos muertos de CompressionConfig retirados (6 campos + escritura huerfana) |
| **Total** | **~990** | Codigo muerto eliminado sin perdida de funcionalidad |

Todas las eliminaciones fueron verificadas con roundtrip tests (register+load, error=0.0) e import checks.
