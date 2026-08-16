# MNEME: Roadmap de Implementación de Teorías y Algoritmos

**Documento de planificación — Implementación de las 8 propuestas de THEORIES_AND_ALGORITHMS.md**
**Versión:** 1.0
**Fecha:** Abril 2026
**Prerequisito:** Sesiones 1-6 del ROADMAP_FASES.md completadas (refactoring + TurboQuant integrado)

---

## Principios de Implementación

1. **Implementar en orden de ratio impacto/dificultad.** Las teorías con mayor retorno por esfuerzo van primero.
2. **Respetar dependencias.** SAMP antes de Cascade. TurboQuant integrado antes de R3/HLQ.
3. **Cada sesión deja el código funcional y testeado.** No hay sesiones que rompen algo para arreglarlo después.
4. **Integrar con infraestructura existente.** Usar ZSpace, ZDescriptor, TensorDecomposer, TensorQuantizer — no reinventar.
5. **Benchmarks cuantitativos.** Cada teoría se valida con métricas medibles (MSE, ratio de compresión, tiempo).

---

## Mapa de Dependencias

```
TurboQuant (Sesión 6)
    │
    ├──> SAMP (Sesión 7) ──────────────> Cascade (Sesión 9)
    │                                        │
    │                                        └──> Cross-Layer Sharing (Sesión 14)
    │
    ├──> AKVC (Sesión 8)
    │
    ├──> Deduplicación Sub-Tensorial (Sesión 10)
    │
    ├──> R3 (Sesión 11) ──> Cascade puede usar R3 como etapa opcional
    │
    ├──> HLQ (Sesiones 12-13)
    │
    └──> Z-MMU (Sesión 15+) — requiere todas las anteriores como spec
```

---

# FASE 1: Fundaciones Algorítmicas (Inmediata)

---

## SESIÓN 7: SAMP — Spectral-Aware Mixed Precision

**Prioridad:** 1 (ratio I/D = 2.25)
**Riesgo:** Bajo
**Líneas estimadas:** ~200
**Archivo principal:** `src/mneme/mneme_core.py`

### Objetivo

Reemplazar los umbrales fijos de `_create_smart_descriptor()` con el criterio ICE (Indicador de Compresibilidad Espectral) para decisiones adaptativas por capa.

### FASE 5A: Implementar `SpectralAnalyzer`

**Qué se crea en `mneme_core.py`:**
- Clase `SpectralAnalyzer` con método `compute_ice(W) -> float`
  - Calcula SVD parcial (solo valores singulares vía `torch.linalg.svdvals`)
  - Calcula distribución de energía `p_i = σ_i² / Σσ_j²`
  - Calcula entropía normalizada `H_norm = H(σ) / log₂(r)`
  - Retorna `ICE = 1 - H_norm` ∈ [0, 1]
- Método `decide_strategy(W, ice) -> str` que retorna `"decompose"` | `"quantize"` | `"hybrid"`
  - ICE > 0.7 → DECOMPOSE (SVD truncado, alta precisión)
  - ICE < 0.3 → QUANTIZE_DIRECT (cuantización directa)
  - 0.3 ≤ ICE ≤ 0.7 → HYBRID (SVD parcial + residual cuantizado)

**Líneas:** ~80

### FASE 5B: Integrar ICE en `_create_smart_descriptor()`

**Qué se modifica:**
- `_create_smart_descriptor()`: antes del routing actual (SVD/TT/RAW), calcular ICE si el tensor es 2D y ≥ 10k elementos
- Los umbrales fijos (`>= 10k para SVD`) se reemplazan con decisión ICE
- Agregar campo `ice_score` a `ZDescriptor.meta` para diagnóstico
- El routing existente para tensores 3D+ y pequeños se mantiene intacto

**Líneas:** ~40

### FASE 5C: Waterfilling de bits por componente SVD

**Qué se agrega:**
- Método `SpectralAnalyzer.allocate_bits(sigma, target_bits) -> List[int]`
  - Implementa reverse waterfilling: `b_i = b_avg + 0.5 * log₂(σ_i² / geomean(σ²))`
  - Clamp a [2, 16] bits por componente
- Integrar en `_create_svd_descriptor()` para cuantización per-componente diferenciada

**Líneas:** ~40

### FASE 5D: Tests y Benchmarks SAMP

**Tests (`tests/test_samp.py`):**
- `test_ice_concentrated_spectrum` — tensor low-rank tiene ICE > 0.7
- `test_ice_flat_spectrum` — tensor aleatorio tiene ICE < 0.3
- `test_ice_routing_decompose` — ICE alto → selecciona SVD
- `test_ice_routing_quantize` — ICE bajo → selecciona cuantización directa
- `test_ice_routing_hybrid` — ICE medio → estrategia híbrida
- `test_waterfilling_bits` — más bits a componentes de mayor energía
- `test_samp_roundtrip_quality` — SAMP mejora MSE vs umbrales fijos

**Benchmark (`benchmarks/bench_samp.py`):**
- Comparar MSE de `_create_smart_descriptor()` con/sin SAMP en capas reales de Pythia-70M

**Líneas:** ~100

**Verificación:**
```bash
python -m pytest tests/test_samp.py -v
python benchmarks/bench_samp.py
```

---

## SESIÓN 8: AKVC — Adaptive KV Cache Compression

**Prioridad:** 3 (ratio I/D = 1.60)
**Riesgo:** Bajo-Medio
**Líneas estimadas:** ~350
**Archivo principal:** `src/mneme/mneme_torch.py`

### Objetivo

Extender `QuantizedKVCache` con decorrelación PCA per-head, asignación de bits por entropía de atención, y evicción por importancia de token.

### FASE 6A: Decorrelación PCA per-head

**Qué se agrega a `QuantizedKVCache`:**
- Campo `pca_bases: List[Optional[Tensor]]` — base PCA por head
- Método `_update_pca_basis(head_idx, K_data)` — eigendecomp de covarianza `K^T K / T`
  - Se recalcula cada N tokens (N configurable, default 512)
  - Almacena top-d eigenvectors como base de proyección
- En `update()`: `K_pca = K @ pca_basis[h]` antes de cuantizar
- En `get()`: `K_orig = K_pca @ pca_basis[h].T` al reconstruir

**Líneas:** ~80

### FASE 6B: Asignación de bits por entropía

**Qué se agrega:**
- Método `_compute_attention_entropy(scores) -> float`
  - `H(A_h) = -Σ softmax(A_h)_t * log₂(softmax(A_h)_t)`
- Método `_allocate_bits(entropy, H_max) -> int`
  - `b_h = b_min + (b_max - b_min) * H(A_h) / H_max`
  - b_min=2, b_max=8 por default
- Parámetro `attention_scores` opcional en `update()` para activar bit allocation
- Si no se pasan scores, usar precision uniforme (backward-compatible)

**Líneas:** ~60

### FASE 6C: Evicción por importancia de token

**Qué se agrega:**
- Método `_compute_token_importance(attention_scores) -> Tensor`
  - Suma de atención recibida sobre todas las heads y queries recientes
- Método `_evict_tokens(attention_scores)` con política:
  - Siempre mantener primeros S tokens (attention sink, default S=4)
  - Siempre mantener últimos W tokens (ventana local, default W=128)
  - Del resto, mantener top-M por importancia acumulada
  - Tokens eviccionados → descartar (no comprimir a menor precision, por simplicidad)
- Campo `max_seq_len` para trigger de evicción

**Líneas:** ~100

### FASE 6D: Clase `AdaptiveKVCache` unificada

**Qué se crea:**
- Nueva clase `AdaptiveKVCache(QuantizedKVCache)` que hereda de la existente
- Constructor acepta `enable_pca=True`, `enable_entropy_bits=True`, `enable_eviction=True`
- Override de `update()` y `get()` que orquestan PCA + entropía + evicción
- Export en `__init__.py`

**Líneas:** ~60

### FASE 6E: Tests AKVC

**Tests (`tests/test_akvc.py`):**
- `test_pca_decorrelation` — PCA reduce varianza de componentes
- `test_entropy_bit_allocation` — head con atención difusa → más bits
- `test_eviction_keeps_sinks` — primeros S tokens siempre se mantienen
- `test_eviction_keeps_recent` — ventana local se mantiene
- `test_akvc_roundtrip` — update → get preserva shapes correctos
- `test_akvc_backward_compat` — sin scores funciona como QuantizedKVCache

**Benchmark:**
- Medir memoria de cache con/sin AKVC para secuencia de 4096 tokens

**Líneas:** ~100

**Verificación:**
```bash
python -m pytest tests/test_akvc.py -v
```

---

# FASE 2: Pipeline Unificado (Corto Plazo)

---

## SESIÓN 9: MNEME-Cascade — Pipeline Adaptativo Multi-Etapa

**Prioridad:** 2 (ratio I/D = 1.43)
**Riesgo:** Medio
**Líneas estimadas:** ~600
**Archivo nuevo:** `src/mneme/mneme_cascade.py`
**Prerequisito:** Sesión 7 (SAMP) completada

### Objetivo

Crear el pipeline de 4 etapas que unifica TensorDecomposer + TensorQuantizer + TensorSparsifier con análisis de sensibilidad entropica como orquestador.

### FASE 7A: Análisis de Sensibilidad Entropica

**Qué se crea en `mneme_cascade.py`:**
- Clase `SensitivityAnalyzer`
  - `analyze_model(model, calibration_data) -> Dict[str, float]`
    - Para cada capa 2D: `S(l) = H(W_l) * ||∂L/∂W_l||_F`
    - Entropía H vía histograma de 256 bins
    - Norma del gradiente via forward-backward con datos de calibración
  - `compute_thresholds(sensitivities) -> (tau_low, tau_high)`
    - Percentiles 25 y 75 del diccionario de sensibilidades
- Dataclass `CascadeConfig(target_bits, outlier_fraction, energy_threshold, ...)`

**Consideración:** `compute_gradient_norm` requiere datos de calibración + forward-backward pass. Hacer esto **opcional** — si no se pasan datos de calibración, usar solo entropía H(W) como proxy de sensibilidad.

**Líneas:** ~120

### FASE 7B: Selector de Descomposición Adaptativo

**Qué se crea:**
- Función `auto_decompose(W, sensitivity, tau_low, tau_high) -> (type, factors)`
  - Usa ICE de SAMP (Sesión 7) + sensibilidad para decidir
  - Delega a `TensorDecomposer` existente para SVD/TT/Tucker
  - Retorna tipo de descomposición + lista de factores
- Integración con `SpectralAnalyzer.compute_ice()` de Sesión 7

**Líneas:** ~60

### FASE 7C: Cuantización Rotation-Aware de Factores

**Qué se crea:**
- Función `quantize_factors(factors, bit_budget, decomp_type) -> List[QuantizedFactor]`
  - Para cada factor: FWHT → Lloyd-Max codebook → cuantizar
  - Usa TurboMNEMECodec (Sesión 6) si está disponible
  - Fallback a TensorQuantizer para cuantización escalar
- Función `allocate_bit_budget(S, sensitivities, target_bits) -> int`
  - Asigna bits proporcional a sensibilidad relativa de la capa

**Líneas:** ~80

### FASE 7D: Separación Outlier-Sparse del Residual

**Qué se crea:**
- Función `extract_sparse_residual(W_original, W_approx, fraction=0.005) -> SparseTensor`
  - Calcula `R = W - W_approx`
  - Extrae top-p% por magnitud en formato COO
  - Almacena como sparse tensor comprimido
- Dataclass `CascadeDescriptor(decomp_type, factors, sparse_residual, sensitivity, bit_budget, ice_score)`

**Líneas:** ~60

### FASE 7E: Pipeline Orquestador

**Qué se crea:**
- Clase `CascadePipeline`
  - `compress(model, calibration_data=None, config=CascadeConfig()) -> Dict[str, CascadeDescriptor]`
    - Etapa 1: Análisis de sensibilidad (7A)
    - Etapa 2: Selección de descomposición (7B)
    - Etapa 3: Cuantización de factores (7C)
    - Etapa 4: Residual sparse (7D)
  - `decompress(descriptors) -> Dict[str, Tensor]`
    - Reconstrucción: factores dequantizados + FWHT inversa + sparse scatter
  - `compress_single(name, W, sensitivity, thresholds, config) -> CascadeDescriptor`
    - Versión para una sola capa (útil para ZSpace)
- Integración con ZSpace: nuevo tipo `DecompType.CASCADE = "cascade"`
  - Método `_create_cascade_descriptor()` en ZSpace
  - Wire en `_create_smart_descriptor()` como opción cuando se pasa `cascade=True`

**Líneas:** ~180

### FASE 7F: Tests y Benchmarks Cascade

**Tests (`tests/test_cascade.py`):**
- `test_sensitivity_analysis` — capas con outliers tienen mayor sensibilidad
- `test_auto_decompose_svd` — capa low-rank → SVD
- `test_auto_decompose_skip` — capa insensible → no descomponer
- `test_quantize_factors_roundtrip` — factores cuantizados reconstruyen
- `test_sparse_residual_extraction` — top-0.5% se extrae correctamente
- `test_cascade_full_pipeline` — pipeline completo en modelo toy
- `test_cascade_vs_uniform` — Cascade tiene menor MSE que cuantización uniforme al mismo bitrate
- `test_cascade_zspace_integration` — register/load con descriptor Cascade

**Benchmark (`benchmarks/bench_cascade.py`):**
- Cascade vs cuantización uniforme INT4 en capas de Pythia-70M
- Métricas: MSE, ratio de compresión, tiempo de compresión

**Líneas:** ~150

**Verificación:**
```bash
python -m pytest tests/test_cascade.py -v
python benchmarks/bench_cascade.py
```

---

## SESIÓN 10: Deduplicación Content-Addressable Sub-Tensorial

**Prioridad:** 4 (ratio I/D = 1.40)
**Riesgo:** Medio
**Líneas estimadas:** ~400
**Archivo principal:** `src/mneme/mneme_core.py`

### Objetivo

Extender ZSpace con deduplicación a nivel de factores SVD/TT usando hashing exacto + LSH para similitud aproximada.

### FASE 8A: Índice de Hashes Exactos para Factores

**Qué se agrega a ZSpace:**
- Diccionario `_factor_hash_index: Dict[str, ZAddr]` — hash xxhash → dirección del factor
- Método `_store_factor_dedup(factor, name) -> DedupRef`
  - Calcula hash exacto del factor
  - Si match → retorna referencia al factor existente
  - Si no → almacena y registra en índice
- Dataclass `DedupRef(addr, type: "exact"|"approximate"|"new", delta_addr=None)`

**Líneas:** ~80

### FASE 8B: LSH para Similitud Aproximada

**Qué se agrega:**
- Clase `LSHIndex`
  - `__init__(num_planes=64, dim)` — genera hiperplanos aleatorios deterministas
  - `compute_fingerprint(tensor) -> int` — fingerprint de 64 bits
  - `find_candidates(fingerprint, max_hamming=8) -> List[ZAddr]`
  - `add(fingerprint, addr)`
- En `_store_factor_dedup()`: si no hay match exacto, buscar por LSH
  - Si coseno > 0.95 con candidato → almacenar solo delta (cuantizado a 2 bits)
  - El delta es un ZDescriptor propio, referenciado desde el DedupRef

**Líneas:** ~120

### FASE 8C: Integración con Descriptores SVD/TT

**Qué se modifica:**
- `_create_svd_descriptor()`: almacenar factores U, V con `_store_factor_dedup()`
- `_create_tt_descriptor()`: almacenar cores con `_store_factor_dedup()`
- Nuevo campo en descriptor meta: `factor_refs: List[DedupRef]`
- `_do_synthesize()`: al reconstruir, cargar factores via DedupRef + sumar delta si aplica
- Método `dedup_stats() -> Dict` para inspeccionar ratio de deduplicación

**Líneas:** ~100

### FASE 8D: Tests Deduplicación

**Tests (`tests/test_dedup.py`):**
- `test_exact_dedup` — factor idéntico almacenado una sola vez
- `test_lsh_similar_factors` — factores similares (coseno>0.95) comparten base
- `test_delta_reconstruction` — base + delta reconstruye correctamente
- `test_dedup_different_factors` — factores distintos no se deduplicam
- `test_dedup_stats` — estadísticas de deduplicación correctas
- `test_dedup_across_layers` — simular capas adyacentes de transformer, verificar dedup

**Líneas:** ~100

**Verificación:**
```bash
python -m pytest tests/test_dedup.py -v
```

---

# FASE 3: Técnicas de Cuantización Avanzada (Mediano Plazo)

---

## SESIÓN 11: R3 — Residual Rotation Refinement

**Prioridad:** 5 (ratio I/D = 1.17)
**Riesgo:** Medio
**Líneas estimadas:** ~350
**Archivo principal:** `src/mneme/mneme_optimization.py`

### Objetivo

Implementar cuantización iterativa multi-ronda con rotaciones ortogonales diferentes por ronda, capturando progresivamente el error residual.

### FASE 9A: Rotaciones Hadamard Aleatorizadas

**Qué se crea:**
- Clase `RandomizedHadamard(signs, perm, size)`
- Función `build_randomized_hadamard(shape, seed) -> RandomizedHadamard`
  - `H_r = D_r * H * P_r` con signos y permutación deterministas por seed
- Funciones `apply_rotation(x, H_r)` y `apply_inverse_rotation(x, H_r)`
  - Usa FWHT existente de TurboQuant si disponible, sino implementación propia
  - Maneja padding a potencia de 2

**Nota:** Si TurboQuant (Sesión 6) ya tiene FWHT implementado, reusar. No duplicar.

**Líneas:** ~80

### FASE 9B: Compresor/Descompresor R3

**Qué se crea:**
- Dataclass `R3Round(quantized, scales, bits, seed, residual_norm)`
- Dataclass `R3Compressed(rounds: List[R3Round], shape, original_norm)`
- Clase `R3Quantizer`
  - `compress(W, base_bits=4, max_rounds=2, bits_decay=2, seed=42) -> R3Compressed`
    - Ronda 0: rotar con H_0, cuantizar a base_bits
    - Ronda r: rotar residual con H_r, cuantizar a max(2, base_bits - r*decay)
    - Parada temprana si `||R|| / ||W|| < 1e-4`
  - `decompress(compressed) -> Tensor`
    - Suma de contribuciones: `Σ H_r⁻¹ * Q_r`

**Líneas:** ~120

### FASE 9C: Integración con ZSpace

**Qué se agrega:**
- `DecompType.R3 = "r3"` en el enum
- Método `_create_r3_descriptor()` en ZSpace
  - Cada ronda es un sub-descriptor almacenado con `_serialize_components()`
  - Meta incluye `rounds`, `seeds`, `bits_per_round`
- Wire en `_create_smart_descriptor()` cuando se pasa `quant_type='r3'`
- `register()` acepta kwargs `r3_rounds`, `r3_base_bits`

**Líneas:** ~60

### FASE 9D: Tests R3

**Tests (`tests/test_r3.py`):**
- `test_r3_single_round` — equivalente a cuantización simple
- `test_r3_two_rounds_better` — 2 rondas tiene menor MSE que 1 ronda
- `test_r3_diminishing_returns` — ronda 3 aporta poco vs ronda 2
- `test_r3_roundtrip` — compress → decompress preserva shape y calidad
- `test_r3_deterministic` — misma seed → mismo resultado
- `test_r3_zspace_integration` — register/load con R3 descriptor

**Benchmark:**
- MSE vs bits/param para 1, 2, 3 rondas en tensores aleatorios y reales

**Líneas:** ~100

**Verificación:**
```bash
python -m pytest tests/test_r3.py -v
```

---

## SESIONES 12-13: HLQ — Hadamard-Lattice Quantization

**Prioridad:** 6 (ratio I/D = 1.13)
**Riesgo:** Alto
**Líneas estimadas:** ~600
**Archivo nuevo:** `src/mneme/mneme_hlq.py`

### Objetivo

Implementar cuantización vectorial en lattice E8 con rotación Hadamard previa, reemplazando cuantización escalar por vectorial para ~30% menos MSE al mismo bitrate.

### SESIÓN 12: Fundaciones HLQ

#### FASE 10A: Generador de Codebook E8

**Qué se crea en `mneme_hlq.py`:**
- Función `generate_e8_codebook(total_bits) -> np.ndarray`
  - Genera puntos de E8: `{x ∈ Z⁸ ∪ (Z+½)⁸ : Σxᵢ par}`
  - Ordena por norma, trunca a 2^total_bits puntos
  - Pre-computa y almacena como constante para bits comunes (16, 24, 32)
- Codebooks pre-computados como arrays literales para bits=16 (2 bits/dim × 8 dim)

**Decisión:** Pre-computar codebooks offline y hardcodear (como los codebooks Lloyd-Max de TurboQuant). Evitar scipy/scipy.spatial en runtime.

**Líneas:** ~100

#### FASE 10B: KD-Tree para Nearest Neighbor en E8

**Qué se crea:**
- Clase `E8NearestNeighbor`
  - `__init__(codebook)` — construye árbol KD (usa numpy/scipy offline, no en decode)
  - `query(vectors) -> indices` — batch nearest neighbor
  - Alternativa sin scipy: búsqueda exhaustiva vectorizada con PyTorch
    - Para codebook de 65536 puntos (16 bits): `cdist` sobre bloques de 8
    - Viable porque la búsqueda es offline (encoding), no en decode

**Líneas:** ~60

#### FASE 10C: Encoder/Decoder HLQ

**Qué se crea:**
- Clase `HLQQuantizer(bits_per_dim=2, block_size=8)`
  - `encode(weights, group_size=128) -> HLQEncoded`
    - Reshape en bloques de 8 → Hadamard H₈ → escala per-bloque → nearest E8
  - `decode(encoded) -> Tensor`
    - Lookup codebook → des-escalar → Hadamard inversa
- Dataclass `HLQEncoded(indices, scales, shape, group_size)`
- Serialización compatible con safetensors (indices como tensor uint16, scales como float16)

**Líneas:** ~100

### SESIÓN 13: Integración y Optimización HLQ

#### FASE 10D: Integración con ZSpace

**Qué se agrega:**
- `DecompType.HLQ = "hlq"` en el enum
- Método `_create_hlq_descriptor()` en ZSpace
- Wire en `_create_smart_descriptor()` cuando `quant_type='hlq'`
- Serialización vía `_serialize_components()` existente

**Líneas:** ~50

#### FASE 10E: Optimización de Decode

**Qué se crea:**
- LUT (Look-Up Table) pre-computada para decode rápido
  - `codebook_lut[index] → 8 valores float16` — lectura directa sin cómputo
  - Hadamard H₈ es constante → pre-multiplicar en la LUT: `lut[i] = H₈⁻¹ · codebook[i]`
  - Decode se reduce a: `values = lut[indices] * scales` — sin FWHT en decode
- Vectorización con PyTorch: indexación avanzada sobre la LUT

**Líneas:** ~60

#### FASE 10F: Tests y Benchmarks HLQ

**Tests (`tests/test_hlq.py`):**
- `test_e8_codebook_validity` — todos los puntos cumplen propiedad E8
- `test_e8_codebook_size` — exactamente 2^total_bits puntos
- `test_hlq_encode_decode_roundtrip` — shapes preservados
- `test_hlq_mse_vs_scalar` — MSE de HLQ < MSE de cuantización escalar INT4
- `test_hlq_decode_with_lut` — decode con LUT produce mismo resultado
- `test_hlq_zspace_integration` — register/load con HLQ descriptor

**Benchmark:**
- HLQ vs INT4 escalar: MSE, velocidad encode/decode
- Validar ganancia teórica de ~1.5 dB (~29% menos MSE)

**Líneas:** ~120

**Verificación:**
```bash
python -m pytest tests/test_hlq.py -v
python benchmarks/bench_hlq.py
```

---

# FASE 4: Optimización de Sistema (Largo Plazo)

---

## SESIÓN 14: Cross-Layer Weight Sharing

**Prioridad:** 7 (ratio I/D = 1.00)
**Riesgo:** Bajo
**Líneas estimadas:** ~250
**Archivo:** `src/mneme/mneme_optimization.py`
**Prerequisito:** Cascade (Sesión 9) completada

### Objetivo

Detectar capas con distribuciones rotation-compatible y compartir codebooks de cuantización.

### FASE 11A: Detector de Compatibilidad Rotacional

**Qué se crea:**
- Clase `RotationCompatibilityDetector`
  - `analyze(named_weights, rotation_seed) -> compat_matrix`
    - Para cada par: rotar con misma Hadamard → histograma → KL simétrica
  - `find_clusters(compat_matrix, threshold=0.1) -> List[List[str]]`
    - Clustering greedy de capas compatibles

**Líneas:** ~100

### FASE 11B: Codebook Compartido

**Qué se crea:**
- Clase `SharedCodebookCompressor`
  - `compress(named_weights, clusters, bits=4) -> Dict[str, SharedQuantized]`
    - Cluster de 1: codebook individual
    - Cluster de N: codebook compartido (Lloyd-Max sobre concatenación)
  - `decompress(compressed) -> Dict[str, Tensor]`
- Función `estimate_savings(clusters, named_weights, bits) -> Dict`

**Líneas:** ~100

### FASE 11C: Tests

**Tests (`tests/test_cross_layer.py`):**
- `test_similar_layers_clustered` — capas con distribución similar van al mismo cluster
- `test_dissimilar_layers_separate` — capas distintas → clusters separados
- `test_shared_codebook_quality` — calidad comparable a codebooks individuales
- `test_savings_estimation` — cálculo correcto de ahorro

**Líneas:** ~60

---

## SESIÓN 15+: Z-MMU — Co-Diseño Hardware-Algoritmo

**Prioridad:** 8 (ratio I/D = 1.00)
**Riesgo:** Muy Alto (proyecto multi-año)
**Status:** Diseño teórico / Paper

### Objetivo Software (preparación)

No se implementa hardware. Se prepara el software de MNEME para ser "Z-MMU ready":

### FASE 12A: Restricciones Power-of-2 para Escalas (estilo Tender)

**Qué se modifica en `TensorQuantizer`:**
- Opción `power_of_2_scales=True`
  - `log2_scale = round(log₂(max(|W_group|) / (2^(b-1) - 1)))`
  - `scale = 2^log2_scale`
  - Almacena `log2_scale` como int5 en vez de float16
- Beneficio software: ~10% menos almacenamiento de metadata de escalas
- Beneficio hardware futuro: shift en vez de multiplicación

**Líneas:** ~40

### FASE 12B: Spec de Pipeline Z-MMU

**Documento `docs/Z_MMU_SPEC.md`:**
- Especificación del pipeline de descompresión esperado
- Formato de datos comprimidos compatible con hardware
- Restricciones de alineamiento (bloques de 8, potencias de 2)
- Estimaciones de latencia y throughput
- Requisitos para FPGA proof-of-concept

**Líneas:** Documentación, no código

---

# Resumen por Sesión

| Sesión | Teoría | Prioridad | Riesgo | Líneas Δ | Qué se logra |
|--------|--------|:---------:|:------:|:--------:|--------------|
| 7 | **SAMP** | 1 | Bajo | +200 | ICE reemplaza umbrales fijos, waterfilling de bits |
| 8 | **AKVC** | 3 | Bajo-Medio | +350 | KV cache con PCA + entropía + evicción |
| 9 | **Cascade** | 2 | Medio | +600 | Pipeline unificado de 4 etapas |
| 10 | **Dedup Sub-Tensorial** | 4 | Medio | +400 | Deduplicación de factores SVD/TT con LSH |
| 11 | **R3** | 5 | Medio | +350 | Cuantización multi-ronda con rotaciones |
| 12-13 | **HLQ** | 6 | Alto | +600 | Cuantización vectorial lattice E8 |
| 14 | **Cross-Layer** | 7 | Bajo | +250 | Codebooks compartidos entre capas |
| 15+ | **Z-MMU** | 8 | Muy Alto | ~40+doc | Preparación software + spec hardware |

**Total estimado:** ~2,790 líneas de código nuevo + tests + benchmarks

---

# Criterios de Éxito por Teoría

| Teoría | Métrica Principal | Objetivo Mínimo | Objetivo Ideal |
|--------|-------------------|:---------------:|:--------------:|
| SAMP | Reducción MSE vs umbrales fijos | -20% MSE | -40% MSE |
| AKVC | Reducción memoria KV cache | -30% | -50% |
| Cascade | MSE a mismo bitrate vs INT4 uniforme | -30% MSE | -50% MSE |
| Dedup | Ahorro almacenamiento en modelo multi-capa | -10% | -18% |
| R3 | MSE con 2 rondas (6 bits) vs 1 ronda (4 bits) | -90% MSE | -99% MSE |
| HLQ | MSE vs cuantización escalar al mismo bitrate | -20% MSE | -29% MSE |
| Cross-Layer | Reducción de codebooks | -50% codebooks | -73% codebooks |
| Z-MMU | N/A (spec only) | Documento completo | + prototipo FPGA |

---

# Notas de Integración

### Archivos Nuevos
- `src/mneme/mneme_cascade.py` — Pipeline Cascade (Sesión 9)
- `src/mneme/mneme_hlq.py` — HLQ quantizer (Sesiones 12-13)
- `tests/test_samp.py`, `test_akvc.py`, `test_cascade.py`, `test_dedup.py`, `test_r3.py`, `test_hlq.py`, `test_cross_layer.py`
- `benchmarks/bench_samp.py`, `bench_cascade.py`, `bench_hlq.py`

### Archivos Modificados
- `src/mneme/mneme_core.py` — SAMP (SpectralAnalyzer), Dedup (LSHIndex), ZSpace (nuevos DecompTypes)
- `src/mneme/mneme_torch.py` — AKVC (AdaptiveKVCache)
- `src/mneme/mneme_optimization.py` — R3 (R3Quantizer), Cross-Layer (RotationCompatibilityDetector)
- `src/mneme/__init__.py` — Exports de nuevas clases

### Compatibilidad
- Todas las adiciones son **opt-in**. El comportamiento por defecto no cambia.
- `_create_smart_descriptor()` con SAMP es el único cambio que altera comportamiento existente, y es una mejora directa (mejores decisiones de routing).
- APIs existentes (`register()`, `load()`, `update()`) no cambian de firma.
