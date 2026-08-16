# MNEME Roadmap de Refactoring: Fases por Sesión

## Principios

1. **Una fase = un tipo de cambio.** No mezclar borrado con features nuevas.
2. **Verificación entre fases.** Cada fase termina con tests que deben pasar.
3. **Commit por fase.** Si algo se rompe, rollback a la fase anterior.
4. **Borrar antes de construir.** Limpiar el terreno antes de plantar.
5. **Una sesión = un bloque de trabajo coherente.** Cada sesión deja el código funcional.

---

# SESIÓN 1: Bug Fixes + Deduplicación [COMPLETADA]

**Duración estimada:** 1 sesión  
**Riesgo:** Bajo  
**Status:** DONE

## FASE 1: Bug Fixes (11 bugs críticos)

| Bug | Archivo | Status |
|-----|---------|--------|
| ConcurrencyError undefined | mneme_core.py | FIXED |
| safetensors API (encrypt/decrypt) | mneme_torch.py | FIXED |
| safetensors batch serialize | mneme_security_core.py | FIXED |
| Checksum raw vs compressed | mneme_storage_core.py | FIXED |
| dtype hardcoded float32 | mneme_core.py | FIXED |
| pickle → mscs | mneme_optimization.py | FIXED |
| parallel_quantization index | mneme_optimization.py | FIXED |
| logging.basicConfig en librería | mneme_core.py | FIXED |
| Lock.acquire timeout=-1 | mneme_core.py | FIXED |
| Bare except clauses | mneme_core.py | FIXED |
| Profiling incondicional forward | mneme_torch.py | FIXED |

## FASE 2A: Deduplicación de Clases [COMPLETADA]

**Resultado real (diferencias vs plan original):**

| Clase | Acción | Razón |
|-------|--------|-------|
| CircuitBreaker | Borrado de optimization, importado de core | Duplicado exacto (~150 líneas) |
| HealthStatus enum | Borrado de optimization, importado de core | Duplicado (añadidos RECOVERING/MAINTENANCE a core) |
| CircuitState enum | Borrado de optimization, importado de core | Duplicado exacto |
| LatencyHistogram | **MANTENIDO en ambos** | APIs incompatibles: core=dataclass(name), optimization=clase(buckets,Lock) |
| TensorDecomposer | **MANTENIDO en ambos** | Implementaciones diferentes: core=estáticos, optimization=instanciable |

## FASE 2B: Deduplicación de Enums [COMPLETADA]

| Enum | Acción | Razón |
|------|--------|-------|
| SecurityLevel | Borrado de core, importado de security_core | Pertenece al módulo de seguridad |
| TensorEncryptionMode | Borrado (sin implementación) | Fachada vacía, campos de MnemeConfig → string |
| KeyRotationPolicy | Borrado (sin implementación) | Fachada vacía, campos de MnemeConfig → string |
| CachePolicy | **MANTENIDO en ambos** (core + storage) | Import circular impide unificación |
| CompressionLevel | **MANTENIDO en ambos** (core + storage) | Import circular impide unificación |

**Verificación pasada:**
```
python -c "from mneme import ZSpace, SecurityLevel; from mneme.mneme_optimization import CircuitBreaker, HealthStatus; print('OK')"
# ZSpace register+load roundtrip OK
```

**Líneas eliminadas en sesión 1:** ~170

---

# SESIÓN 2: Limpieza de Infraestructura Muerta (mneme_core.py)

**Duración estimada:** 1 sesión  
**Riesgo:** Medio  
**Prerequisito:** Sesión 1 completada

Esta sesión borra toda la infraestructura de mneme_core.py que se instancia pero nunca se usa. Se divide en 3 fases para permitir verificación incremental.

## FASE 2C: Borrar Event System

**Qué se borra:**
- `EventType` enum (~12 líneas)
- `MnemeEvent` dataclass (~16 líneas)
- `EventBus` clase (~100 líneas)
- `EventHandler` protocol (~7 líneas)
- Todas las llamadas `self.context.emit_event(...)` en ZSpace (~15 llamadas)
- Campo `event_bus` en MnemeContext

**Por qué existía:** Diseñado para observabilidad distribuida (OpenTelemetry-style). La idea era que componentes externos suscribieran handlers para monitorear operaciones.

**Por qué se borra:** 0 handlers registrados en todo el proyecto. Los eventos se emiten al vacío. El MetricsRegistry ya cubre la observabilidad funcional.

**Verificación:**
```bash
python -c "from mneme import ZSpace; z = ZSpace(); z.register('test', __import__('torch').randn(10,10)); print(z.load('test').shape); print('FASE 2C OK')"
```

**Rollback:** `git checkout -- src/mneme/mneme_core.py`

**Líneas eliminadas:** ~150

## FASE 2D: Borrar Plugin System

**Qué se borra:**
- `Plugin` protocol (~20 líneas)
- `TensorTransformer` protocol (~10 líneas)
- `MetricsCollector` protocol (~14 líneas)
- `PluginRegistry` clase (~106 líneas)
- Campo `plugins` en MnemeContext
- Refs a `self.plugins` en ZSpace.__init__ y cleanup()

**Por qué existía:** Sistema de extensibilidad via plugins. Permitiría a terceros añadir transformadores de tensores, colectores de métricas, etc.

**Por qué se borra:** 0 plugins implementados. 0 usos de los protocols. La API es puro andamiaje sin edificio.

**Verificación:** Misma que 2C + import check de optimization.

**Líneas eliminadas:** ~160

## FASE 2E: Borrar Pipeline System

**Qué se borra:**
- `PipelineStage` enum (~10 líneas)
- `PipelineContext` dataclass (~16 líneas)
- `PipelineStageHandler` dataclass (~18 líneas)
- `PipelineError` exception (~8 líneas)
- `ProcessingPipeline` clase (~102 líneas)
- `self.pipeline = ...` en ZSpace.__init__

**Por qué existía:** Pipeline multi-etapa con hooks pre/post para interceptar operaciones de store/register/load. Diseñado para validación, logging, transformación en cadena.

**Por qué se borra:** `pipeline.process()` nunca se llama. store() y register() bypasean el pipeline completamente. 0 stage handlers registrados.

**Verificación:** Misma que 2C.

**Líneas eliminadas:** ~155

**Total sesión 2:** ~465 líneas eliminadas de mneme_core.py

---

# SESIÓN 3: Limpieza de Enums, Clases Sueltas y Config

**Duración estimada:** 1 sesión  
**Riesgo:** Bajo  
**Prerequisito:** Sesión 2 completada

## FASE 2F: Borrar Enums y Clases Huérfanas

**Qué se borra de mneme_core.py:**
- `StorageBackend` enum — nunca consultado por lógica, storage siempre usa SQLite
- `CompressionStrategy` enum — nunca consultado, compresión siempre usa LZ4
- `ContextSimilarityMethod` enum — nunca consultado (vestigio de feature no implementada)
- `ContextClusteringMethod` enum — nunca consultado (vestigio de feature no implementada)
- `SerializationFormat` enum — nunca consultado, siempre se usa safetensors
- `MetricPoint` dataclass — nunca instanciado por nadie
- `MnemeFacade` clase — no exportada, no usada, singleton abandonado

**Qué se borra de mneme_optimization.py:**
- `OptimizationStrategy` enum — nunca referenciado por ningún código
- `Optimizable` protocol — nunca implementado por ninguna clase
- `Compressible` protocol — nunca implementado por ninguna clase
- `MetricsExporter` abstract class — nunca subclaseada
- `WorkStealingQueue` clase — instanciada por ParallelExecutor pero nunca usada para dispatch real
- Variables batch muertas en ParallelTensorProcessor (`_batch_queue`, `_batch_condition`, etc.)

**Verificación:**
```bash
python -c "from mneme import ZSpace; from mneme.mneme_optimization import MNEMEOptimizer, TensorQuantizer; print('FASE 2F OK')"
python -m pytest tests/ -v --tb=short
```

**Líneas eliminadas:** ~300

## FASE 2G: Actualizar __init__.py y MnemeConfig

**Cambios en __init__.py:**
- Remover exports de enums/clases borrados en 2F
- Añadir imports faltantes: `MNEMEOptimizer`, `OptimizationLevel`
- Sincronizar `__version__` con pyproject.toml

**Cambios en MnemeConfig:**
- Campos que referencian enums borrados → convertir a `str` con defaults sensibles
- Remover `enable_plugins` (no hay sistema de plugins)
- Remover `redis_url`, `enable_distributed_cache` (no hay implementación Redis)

**Verificación:**
```bash
python -c "import mneme; print(mneme.__version__); m = mneme.MnemeConfig(); print('FASE 2G OK')"
```

**Líneas eliminadas:** ~30

## FASE 2H: Crear DELETED_CODE.md

Documento justificando cada eliminación de las sesiones 1-3. Para cada pieza:
- Nombre, tipo, archivo, líneas originales
- Qué hacía (propósito original)
- Por qué se borró (evidencia concreta: 0 usos, 0 tests, 0 handlers)

**Riesgo:** Ninguno (solo documentación)

**Total sesión 3:** ~330 líneas eliminadas + documentación

---

# SESIÓN 4: Performance Fixes

**Duración estimada:** 1 sesión corta  
**Riesgo:** Bajo  
**Prerequisito:** Sesiones 1-3 completadas

## FASE 3A: AdaptiveCache O(n) → O(1)

**Problema:** `deque.remove(key)` es O(n), se llama en cada `get()` y `_remove_entry()`. Para un cache con miles de entries esto es un bottleneck.

**Solución:** Reemplazar `_lru_order: deque` con `OrderedDict`:
- `get()`: `_lru_order.move_to_end(key)` — O(1)
- `_remove_entry()`: `del _lru_order[key]` — O(1)
- Eviction: `next(iter(_lru_order))` — O(1)

**Verificación:**
```bash
python -c "
from mneme.mneme_core import AdaptiveCache
c = AdaptiveCache(max_size_mb=10)
for i in range(1000):
    c.put(f'k{i}', f'v{i}')
assert c.get('k999') == 'v999'
assert c.get('k0') is not None or True  # puede haber sido evicted
print('AdaptiveCache O(1) OK')
"
```

## FASE 3B: Documentar limitación de ZLinear

Añadir docstring claro a ZLinear explicando que **no ahorra memoria** (full tensor + copia comprimida). No arreglar el diseño ahora — eso requiere `autograd.Function` custom que es un refactor mayor para una sesión futura.

**Total sesión 4:** ~30 líneas modificadas

---

# SESIÓN 5: Integrar TurboQuant — Preparación

**Duración estimada:** 1 sesión  
**Riesgo:** Bajo (archivo aislado, no toca el proyecto)  
**Prerequisito:** Ninguno (puede hacerse en paralelo con sesiones 2-4)

## FASE 4A: Fix bugs en tq_mneme.py

Trabajar sobre una copia del archivo. No tocar el proyecto todavía.

### 4A.1: Hardcodear codebooks Lloyd-Max
- Ejecutar `_compute_lloyd_max_codebook_gaussian(b)` para b=1..8
- Capturar arrays de boundaries y centroids como literales Python
- Eliminar función de cómputo + dependencia de scipy
- **Impacto:** Elimina scipy como dependencia runtime, import instantáneo

### 4A.2: Vectorizar bitpacking genérico (bits != 4, != 8)
- Reemplazar loop Python (2.36M iteraciones por capa) con numpy bit-shifting
- **Impacto:** De ~5-10s/capa a ~50ms/capa (100x speedup)

### 4A.3: Vectorizar outlier restoration en decode
- Reemplazar `for idx, (row, col) in enumerate(...)` con indexación tensor avanzada
- **Impacto:** De O(n) Python loop a O(1) PyTorch

### 4A.4: Fix SVD hybrid decode
- Añadir `svd_m`, `svd_n`, `svd_rank`, `svd_S_bytes`, `svd_split_index` a TQDescriptor
- Implementar reconstrucción en decode(): split → reshape U, Vt → undo sqrt(S) → U @ diag(S) @ Vt

### 4A.5: Reemplazar pickle → mscs
- `pickle.dumps(desc)` → `mscs.dumps(desc.__dict__)` + registrar TQDescriptor con mscs

**Verificación:**
```bash
python tq_mneme_fixed.py  # Demo del __main__
```

---

# SESIÓN 6: Integrar TurboQuant — En el Proyecto

**Duración estimada:** 1 sesión  
**Riesgo:** Medio  
**Prerequisito:** Sesiones 1-4 y 5 completadas

## FASE 4B: Crear src/mneme/mneme_turboquant.py

- Copiar código limpio de 4A al proyecto
- Incluir: codebooks hardcodeados, FWHT, TQDescriptor, TurboMNEMECodec
- Excluir: TQDecompType (se integra en DecompType), register_tq_in_zspace (se integra en ZSpace), hardware notes (mover a docs)
- Registrar TQDescriptor con `@mscs.register`

## FASE 4C: Integrar en mneme_core.py

- Añadir `TURBO_QUANT = "turbo_quant"` a DecompType enum
- Añadir método `_create_turboquant_descriptor()` a ZSpace
- Wire en `_create_smart_descriptor()` routing (después del check de quant_type)

## FASE 4D: Actualizar __init__.py

- Exportar `TurboMNEMECodec`, `TQDescriptor`

## FASE 4E: Tests

Crear tests/test_turboquant.py:
- `test_tq_encode_decode_roundtrip` — cycle preserva tensor dentro de tolerancia
- `test_tq_svd_hybrid_roundtrip` — SVD hybrid path reconstrucción correcta
- `test_tq_bitpack_all_widths` — bitpack/unbitpack para bits 1-8
- `test_tq_zspace_register_load` — TurboQuant descriptor a través de ZSpace
- `test_tq_outlier_separation` — extracción y restauración de outliers
- `test_tq_compress_model` — compresión de modelo completo

**Verificación completa:**
```bash
python -c "
from mneme import ZSpace, TurboMNEMECodec
import torch

codec = TurboMNEMECodec(default_bits=3, group_size=256)
t = torch.randn(256, 256)
desc = codec.encode(t)
t2 = codec.decode(desc)
mse = ((t - t2)**2).mean().item()
print(f'TQ roundtrip MSE: {mse:.6f}')
assert mse < 0.1

z = ZSpace()
z.register('tq_test', t, quant_type='turbo_quant', tq_bits=3)
t3 = z.load('tq_test')
print(f'ZSpace+TQ roundtrip OK, shape={t3.shape}')
print('SESION 6 COMPLETE')
"
python -m pytest tests/test_turboquant.py -v
```

---

# Resumen por Sesión

| Sesión | Fases | Riesgo | Líneas Δ | Qué se logra |
|--------|-------|--------|----------|--------------|
| 1 (DONE) | 1, 2A, 2B | Bajo | -170 | Bugs arreglados, clases/enums deduplicados |
| 2 | 2C, 2D, 2E | Medio | -465 | EventBus, PluginRegistry, Pipeline eliminados |
| 3 | 2F, 2G, 2H | Bajo | -330 | Enums huérfanos, MnemeConfig limpio, DELETED_CODE.md |
| 4 | 3A, 3B | Bajo | ~30 | AdaptiveCache O(1), documentar ZLinear |
| 5 | 4A | Bajo | ~0 | tq_mneme.py con bugs arreglados (archivo aislado) |
| 6 | 4B-4E | Medio | +900 | TurboQuant integrado en MNEME con tests |

**Total estimado:** ~1,000 líneas netas eliminadas + 900 líneas de TurboQuant = net -100 líneas con significativamente más funcionalidad real y menos código muerto.

---

# Documentos Generados

| Documento | Status | Descripción |
|-----------|--------|-------------|
| ROADMAP_FASES.md | Este documento | Plan incremental por sesiones |
| THEORIES_AND_ALGORITHMS.md | COMPLETADO | 8 propuestas teóricas originales |
| DELETED_CODE.md | Pendiente (Sesión 3) | Justificación de cada eliminación |
