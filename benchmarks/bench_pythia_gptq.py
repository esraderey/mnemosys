"""
Benchmark: Pythia-410M -- GPTQ INT4 calibrado (Hessian-weighted) v2
Compara PPL baseline (FP16) vs PPL GPTQ-INT4 sobre WikiText-2 real.

Fixes v2:
  - Fix 1: Excluir layers.0 y layers.23 (primera/ultima) de INT4
  - Fix 2: group_size=64 (mas fino que 128)
  - Fix 3: Log Cholesky fallback + damp_percent=0.05
  - Fix 4: 512 muestras de calibracion

Uso:
    python benchmarks/bench_pythia_gptq.py
"""
import sys, os, time, gc, json, math
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Asegurar que MNEME esta en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from torch import nn

# == Configuracion ================================================
MODEL_NAME      = "EleutherAI/pythia-410m"
DATASET_NAME    = "wikitext"
DATASET_CONFIG  = "wikitext-2-raw-v1"
SEQ_LEN         = 1024        # contexto de Pythia
STRIDE          = 512         # stride para sliding-window PPL
GROUP_SIZE      = 64          # Fix 2: 64 en vez de 128
BITS            = 4
DAMP_PERCENT    = 0.05        # Fix 3: 0.05 en vez de 0.01
NUM_CAL_SAMPLES = 512         # Fix 4: 512 en vez de 128
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE           = torch.float16 if DEVICE == "cuda" else torch.float32

# Fix 1: Excluir capas sensibles (primera/ultima + embeddings + norms)
EXCLUDE_PATTERNS = [
    "embed_in", "embed_out", "lm_head",
    "layernorm", "layer_norm", "ln_",
    "layers.0.",    # Fix 1: primera capa del transformer
    "layers.23.",   # Fix 1: ultima capa del transformer
]
N_TRANSFORMER_LAYERS = 24  # Pythia-410M tiene 24 capas
# =================================================================


def should_quantize(name: str) -> bool:
    """Determinar si una capa debe cuantizarse."""
    name_lower = name.lower()
    for pat in EXCLUDE_PATTERNS:
        if pat in name_lower:
            return False
    return True


# ================================================================
# PASO 1: Cargar modelo y tokenizer
# ================================================================
def load_model_and_tokenizer():
    """Cargar Pythia-410M y tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/6] Cargando {MODEL_NAME} en {DEVICE} ({DTYPE})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"       Modelo cargado: {n_params:.1f}M params")
    return model, tokenizer


# ================================================================
# PASO 2: Preparar datos de calibracion
# ================================================================
def load_wikitext(tokenizer):
    """Cargar WikiText-2 test split y tokenizar."""
    from datasets import load_dataset

    print(f"[2/6] Cargando {DATASET_NAME}/{DATASET_CONFIG}...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    n_tokens = input_ids.shape[1]
    print(f"       Tokens en test set: {n_tokens:,}")
    return input_ids


def prepare_calibration_data(tokenizer, num_samples=NUM_CAL_SAMPLES):
    """
    Preparar datos de calibracion del train split de WikiText-2.
    Retorna lista de tensores (batch_size=1, seq_len) en DEVICE.
    """
    from datasets import load_dataset

    print(f"       Preparando {num_samples} muestras de calibracion (train split)...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

    # Concatenar todo el texto y tokenizar
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]  # (total_tokens,)

    # Dividir en chunks de SEQ_LEN
    cal_data = []
    for i in range(0, len(input_ids) - SEQ_LEN, SEQ_LEN):
        chunk = input_ids[i : i + SEQ_LEN].unsqueeze(0).to(DEVICE)
        cal_data.append(chunk)
        if len(cal_data) >= num_samples:
            break

    print(f"       Muestras de calibracion: {len(cal_data)} x seq_len={SEQ_LEN}")
    return cal_data


# ================================================================
# PPL evaluation (sliding window)
# ================================================================
@torch.no_grad()
def evaluate_ppl(model, input_ids, seq_len=SEQ_LEN, stride=STRIDE):
    """
    Evaluar perplexity con sliding window.
    Retorna (ppl, total_tokens_evaluados, tiempo_segundos).
    """
    n_tokens = input_ids.shape[1]
    nlls = []
    n_evaluated = 0
    t0 = time.perf_counter()

    for begin in range(0, n_tokens - 1, stride):
        end = min(begin + seq_len, n_tokens)

        ids = input_ids[:, begin:end].to(DEVICE)
        outputs = model(ids)
        logits = outputs.logits

        # Solo calcular loss sobre porcion de stride (evitar doble-conteo)
        shift_start = max(0, seq_len - stride - 1) if begin > 0 else 0
        shift_logits = logits[:, shift_start:-1, :].contiguous()
        shift_labels = ids[:, shift_start + 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        n_toks = shift_labels.numel()
        nlls.append(loss.item())
        n_evaluated += n_toks

        if end >= n_tokens:
            break

    elapsed = time.perf_counter() - t0
    avg_nll = sum(nlls) / n_evaluated
    ppl = math.exp(avg_nll)
    return ppl, n_evaluated, elapsed


# ================================================================
# PASO 3-5: GPTQ Calibracion y Cuantizacion
# ================================================================
def apply_gptq_quantization(model, cal_data):
    """
    Aplicar GPTQ calibrado con fixes v2:
      - group_size=64, damp_percent=0.05, 512 cal samples
      - Excluir layers.0 y layers.23
      - Log Cholesky fallbacks
    """
    from mneme import GPTQCalibrator, TensorQuantizer

    # -- Paso 3: Crear calibrador --
    print(f"\n[3/6] GPTQCalibrator(bits={BITS}, group_size={GROUP_SIZE}, damp={DAMP_PERCENT})...")
    calibrator = GPTQCalibrator(
        bits=BITS,
        group_size=GROUP_SIZE,
        damp_percent=DAMP_PERCENT,   # Fix 3: 0.05
    )

    # -- Paso 4: Recoger Hessians --
    print(f"[4/6] Recogiendo Hessians con {len(cal_data)} muestras...")
    t_hess_start = time.perf_counter()
    hessians = calibrator.collect_hessian(model, cal_data, num_samples=NUM_CAL_SAMPLES)
    t_hess = time.perf_counter() - t_hess_start
    print(f"       Hessians recogidos: {len(hessians)} capas en {t_hess:.1f}s")

    # Listar capas con Hessian
    hessian_layers = set(hessians.keys())
    print(f"       Capas con Hessian disponible: {len(hessian_layers)}")

    # -- Paso 5: Cuantizar cada capa --
    print(f"\n[5/6] Cuantizando capas (excl. layers.0, layers.23, embeds, norms)...")
    t_quant_start = time.perf_counter()

    quantizer = TensorQuantizer()
    stats = {
        "layers_quantized": 0,
        "layers_skipped_exclude": 0,
        "layers_skipped_no_hessian": 0,
        "layers_skipped_small": 0,
        "total_linear_layers": 0,
        "total_params_quantized": 0,
        "cholesky_ok": 0,
        "cholesky_fallback": 0,
        "excluded_names": [],
        "layer_details": [],
    }

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        stats["total_linear_layers"] += 1
        n_params = module.weight.numel()

        # Excluir capas especificadas
        if not should_quantize(name):
            stats["layers_skipped_exclude"] += 1
            stats["excluded_names"].append(name)
            print(f"       SKIP (excluded): {name} [{module.weight.shape}]")
            continue

        # Necesitamos Hessian
        if name not in hessian_layers:
            stats["layers_skipped_no_hessian"] += 1
            print(f"       SKIP (no hessian): {name}")
            continue

        # Capas muy pequenas
        if n_params < 1000:
            stats["layers_skipped_small"] += 1
            print(f"       SKIP (small): {name} [{n_params} params]")
            continue

        # Cuantizar con GPTQ
        try:
            weight_cpu = module.weight.data.float().cpu()
            hessian_cpu = hessians[name].float().cpu()

            packed, meta = calibrator.quantize_layer(weight_cpu, hessian_cpu)

            # Fix 3: Track Cholesky vs pinv fallback
            if hasattr(calibrator, '_cholesky_ok'):
                if calibrator._cholesky_ok:
                    stats["cholesky_ok"] += 1
                else:
                    stats["cholesky_fallback"] += 1
                    print(f"       WARNING: Cholesky fallback (pinv) en {name}")

            # De-cuantizar para reemplazar peso (inference en FP16)
            dequant = quantizer.dequantize(packed, meta)
            module.weight.data.copy_(dequant.to(module.weight.dtype).to(DEVICE))

            stats["layers_quantized"] += 1
            stats["total_params_quantized"] += n_params
            stats["layer_details"].append({
                "name": name,
                "shape": list(module.weight.shape),
                "params": n_params,
            })

            if stats["layers_quantized"] % 10 == 0:
                print(f"       Cuantizadas: {stats['layers_quantized']} capas...")

        except Exception as e:
            print(f"       ERROR en {name}: {e}")
            import traceback
            traceback.print_exc()
            stats["layers_skipped_no_hessian"] += 1

    t_quant = time.perf_counter() - t_quant_start

    print(f"\n       -- Estadisticas de cuantizacion (v2) --")
    print(f"       Capas totales Linear : {stats['total_linear_layers']}")
    print(f"       Capas cuantizadas    : {stats['layers_quantized']}")
    print(f"       Excluidas (pattern)  : {stats['layers_skipped_exclude']}")
    print(f"       Sin Hessian          : {stats['layers_skipped_no_hessian']}")
    print(f"       Pequenas             : {stats['layers_skipped_small']}")
    print(f"       Params cuantizados   : {stats['total_params_quantized']:,}")
    print(f"       Cholesky OK          : {stats['cholesky_ok']}")
    print(f"       Cholesky fallback    : {stats['cholesky_fallback']}")
    print(f"       Tiempo Hessian       : {t_hess:.1f}s")
    print(f"       Tiempo cuantizacion  : {t_quant:.1f}s")
    print(f"       Tiempo total GPTQ    : {t_hess + t_quant:.1f}s")
    print(f"       Excluidas: {stats['excluded_names']}")

    stats["hessian_time_s"] = round(t_hess, 1)
    stats["quantization_time_s"] = round(t_quant, 1)
    stats["total_gptq_time_s"] = round(t_hess + t_quant, 1)

    return model, stats


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print(" Benchmark: Pythia-410M -- GPTQ INT4 Calibrado v2")
    print(" Fixes: g=64, damp=0.05, cal=512, excl layers.0/23")
    print("=" * 70)
    print()

    # -- Paso 1: Cargar modelo --
    model, tokenizer = load_model_and_tokenizer()
    input_ids = load_wikitext(tokenizer)

    # -- Paso 2: Datos de calibracion --
    cal_data = prepare_calibration_data(tokenizer, num_samples=NUM_CAL_SAMPLES)

    # -- Baseline FP16 --
    print(f"\n[Baseline] Evaluando PPL ({DTYPE})...")
    ppl_base, n_base, t_base = evaluate_ppl(model, input_ids)
    print(f"       PPL baseline  = {ppl_base:.2f}")
    print(f"       Tokens eval   = {n_base:,}")
    print(f"       Tiempo        = {t_base:.1f}s")

    torch.cuda.empty_cache()
    gc.collect()

    # -- Pasos 3-5: GPTQ --
    model, q_stats = apply_gptq_quantization(model, cal_data)

    torch.cuda.empty_cache()
    gc.collect()

    # -- Paso 6: Evaluar PPL con pesos GPTQ --
    print(f"\n[6/6] Evaluando PPL GPTQ-INT4...")
    ppl_gptq, n_gptq, t_gptq = evaluate_ppl(model, input_ids)
    print(f"       PPL GPTQ-INT4 = {ppl_gptq:.2f}")
    print(f"       Tiempo        = {t_gptq:.1f}s")

    # -- Resumen --
    delta_ppl = ppl_gptq - ppl_base
    pct = (delta_ppl / ppl_base) * 100

    print("\n" + "=" * 70)
    print(" RESUMEN v2")
    print("=" * 70)

    results = {
        "model": MODEL_NAME,
        "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "damp_percent": DAMP_PERCENT,
        "num_cal_samples": NUM_CAL_SAMPLES,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "baseline_ppl": round(ppl_base, 2),
        "gptq_int4_ppl": round(ppl_gptq, 2),
        "delta_ppl": round(delta_ppl, 2),
        "ppl_increase_pct": round(pct, 2),
        "layers_quantized": q_stats["layers_quantized"],
        "layers_excluded": q_stats["layers_skipped_exclude"],
        "total_params_quantized": q_stats["total_params_quantized"],
        "cholesky_ok": q_stats["cholesky_ok"],
        "cholesky_fallback": q_stats["cholesky_fallback"],
        "hessian_time_s": q_stats["hessian_time_s"],
        "quantization_time_s": q_stats["quantization_time_s"],
        "total_gptq_time_s": q_stats["total_gptq_time_s"],
        "baseline_eval_time_s": round(t_base, 1),
        "gptq_eval_time_s": round(t_gptq, 1),
        "fixes_applied": [
            "Fix1: exclude layers.0 + layers.23",
            "Fix2: group_size=64",
            "Fix3: damp_percent=0.05 + Cholesky log",
            "Fix4: 512 cal samples",
        ],
    }

    # Comparacion con benchmarks anteriores
    naive_path = Path(__file__).parent / "results_pythia_int4.json"
    prev_gptq_path = Path(__file__).parent / "results_pythia_gptq.json"

    if naive_path.exists():
        with open(naive_path) as f:
            naive_results = json.load(f)
        results["naive_int4_ppl"] = naive_results.get("int4_ppl", "N/A")

    if prev_gptq_path.exists():
        with open(prev_gptq_path) as f:
            prev_gptq = json.load(f)
        results["prev_gptq_ppl"] = prev_gptq.get("gptq_int4_ppl", "N/A")

    for k, v in results.items():
        if k == "fixes_applied":
            print(f"  {k:32s}:")
            for fix in v:
                print(f"    - {fix}")
        else:
            print(f"  {k:32s}: {v}")

    # Guardar JSON
    out_path = Path(__file__).parent / "results_pythia_gptq_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados en: {out_path}")

    print("\n" + "=" * 70)
    print("  Comparacion completa:")
    print(f"    Baseline FP16      : {ppl_base:.2f}")
    if results.get("naive_int4_ppl") and results["naive_int4_ppl"] != "N/A":
        naive_ppl = results["naive_int4_ppl"]
        print(f"    Naive INT4 (g=128) : {naive_ppl:.2f} (+{((naive_ppl - ppl_base) / ppl_base * 100):.1f}%)")
    if results.get("prev_gptq_ppl") and results["prev_gptq_ppl"] != "N/A":
        prev_ppl = results["prev_gptq_ppl"]
        print(f"    GPTQ v1 (g=128)    : {prev_ppl:.2f} (+{((prev_ppl - ppl_base) / ppl_base * 100):.1f}%)")
    print(f"    GPTQ v2 (g=64)     : {ppl_gptq:.2f} (+{pct:.1f}%)")
    if results.get("prev_gptq_ppl") and results["prev_gptq_ppl"] != "N/A":
        improvement = results["prev_gptq_ppl"] - ppl_gptq
        print(f"    Mejora v2 vs v1    : {improvement:.2f} puntos PPL")
    print("=" * 70)


if __name__ == "__main__":
    main()
