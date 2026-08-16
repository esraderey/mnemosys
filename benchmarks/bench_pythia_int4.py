"""
Benchmark: Pythia-410M — INT4 group-wise quantization (sin calibración)
Mide PPL baseline (FP16) vs PPL INT4 sobre datos reales de WikiText-2.

Uso:
    python benchmarks/bench_pythia_int4.py
"""
import sys, os, time, gc, json, math
from pathlib import Path

# Asegurar que MNEME está en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from torch import nn

# ── Configuración ────────────────────────────────────────────────
MODEL_NAME      = "EleutherAI/pythia-410m"
DATASET_NAME    = "wikitext"
DATASET_CONFIG  = "wikitext-2-raw-v1"
SEQ_LEN         = 1024        # longitud de contexto de Pythia
STRIDE          = 512         # stride para evaluación sliding-window
GROUP_SIZE      = 128
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE           = torch.float16 if DEVICE == "cuda" else torch.float32
# ─────────────────────────────────────────────────────────────────


def load_model_and_tokenizer():
    """Cargar Pythia-410M y tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/5] Cargando {MODEL_NAME} en {DEVICE} ({DTYPE})...")
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


def load_wikitext(tokenizer):
    """Cargar WikiText-2 test split y tokenizar."""
    from datasets import load_dataset

    print(f"[2/5] Cargando {DATASET_NAME}/{DATASET_CONFIG} (test split)...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")

    # Concatenar todo el texto test
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids  # (1, total_tokens)
    n_tokens = input_ids.shape[1]
    print(f"       Tokens en test set: {n_tokens:,}")
    return input_ids


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
        target_len = end - begin - 1

        ids = input_ids[:, begin:end].to(DEVICE)

        outputs = model(ids)
        logits = outputs.logits  # (1, seq, vocab)

        # Solo calcular loss sobre la porción de stride (evitar doble-conteo)
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


def quantize_int4_groupwise(weight: torch.Tensor, group_size: int = 128):
    """
    Cuantización INT4 group-wise simétrica (sin calibración).
    Retorna (quantized_int4, scales, zeros, meta) para poder
    de-cuantizar y reemplazar el peso.
    """
    orig_shape = weight.shape
    orig_dtype = weight.dtype
    w = weight.float().reshape(-1)

    # Pad para que sea múltiplo de group_size
    n = w.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        w = F.pad(w, (0, pad))

    groups = w.reshape(-1, group_size)
    n_groups = groups.shape[0]

    # Cuantización asimétrica INT4 (0..15)
    qmax = 15
    g_min = groups.min(dim=1, keepdim=True).values
    g_max = groups.max(dim=1, keepdim=True).values
    scale = (g_max - g_min) / qmax
    scale = scale.clamp(min=1e-10)
    zero_point = (-g_min / scale).round().clamp(0, qmax)

    quantized = ((groups - g_min) / scale).round().clamp(0, qmax).to(torch.uint8)

    # De-cuantizar
    dequant = (quantized.float() - zero_point) * scale + g_min
    dequant = dequant.reshape(-1)[:n].reshape(orig_shape).to(orig_dtype)

    return dequant


def quantize_model_int4(model, group_size=GROUP_SIZE, min_params=1000):
    """
    Reemplazar in-place pesos Linear con versiones INT4 group-wise
    de-cuantizadas. Retorna estadísticas.
    """
    stats = {"layers_quantized": 0, "layers_skipped": 0, "total_params": 0}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n = module.weight.numel()
            stats["total_params"] += n

            if n < min_params:
                stats["layers_skipped"] += 1
                continue

            dequant = quantize_int4_groupwise(
                module.weight.data, group_size=group_size
            )
            module.weight.data.copy_(dequant)
            stats["layers_quantized"] += 1

    return stats


def quantize_model_mneme_int4(model, group_size=GROUP_SIZE, min_params=10000):
    """
    Cuantizar usando el pipeline de MNEME: ZSpace.register() con
    quantization_type='int4_group'.
    Retorna (model_modificado, stats).
    """
    import logging
    logging.disable(logging.INFO)

    from mneme import ZSpace, MnemeConfig

    config = MnemeConfig()
    config.enable_plugins = False
    zs = ZSpace(config)

    stats = {
        "layers_quantized": 0, "layers_skipped": 0,
        "total_params": 0, "compression_ratios": [],
    }

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n = module.weight.numel()
            stats["total_params"] += n

            if n < min_params:
                stats["layers_skipped"] += 1
                continue

            # Registrar peso en ZSpace con INT4
            desc = zs.register(
                f"pythia_{name}_weight",
                module.weight.data,
                quantization_type="int4_group",
                group_size=group_size,
            )

            # Cargar versión de-cuantizada
            dequant = zs.load(f"pythia_{name}_weight")
            module.weight.data.copy_(dequant.to(module.weight.dtype))

            stats["layers_quantized"] += 1
            cr = desc.meta.get("compression_ratio", 0)
            stats["compression_ratios"].append(cr)

    stats["avg_compression_ratio"] = (
        sum(stats["compression_ratios"]) / max(len(stats["compression_ratios"]), 1)
    )
    return model, stats


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print(" Benchmark: Pythia-410M — INT4 Group-Wise (sin calibración)")
    print("=" * 65)
    print()

    model, tokenizer = load_model_and_tokenizer()
    input_ids = load_wikitext(tokenizer)

    # ── Baseline FP16 ────────────────────────────────────────────
    print(f"\n[3/5] Evaluando PPL baseline ({DTYPE})...")
    ppl_base, n_base, t_base = evaluate_ppl(model, input_ids)
    print(f"       PPL baseline  = {ppl_base:.2f}")
    print(f"       Tokens eval   = {n_base:,}")
    print(f"       Tiempo        = {t_base:.1f}s")

    # Liberar cache CUDA
    torch.cuda.empty_cache()
    gc.collect()

    # ── INT4 via MNEME ───────────────────────────────────────────
    print(f"\n[4/5] Aplicando INT4 group-wise (g={GROUP_SIZE}) via MNEME...")
    t_quant_start = time.perf_counter()
    model, q_stats = quantize_model_mneme_int4(model, group_size=GROUP_SIZE)
    t_quant = time.perf_counter() - t_quant_start
    print(f"       Capas cuantizadas : {q_stats['layers_quantized']}")
    print(f"       Capas omitidas    : {q_stats['layers_skipped']}")
    print(f"       Ratio compresión  : {q_stats['avg_compression_ratio']:.4f}")
    print(f"       Tiempo quant      : {t_quant:.1f}s")

    # ── PPL INT4 ─────────────────────────────────────────────────
    print(f"\n[5/5] Evaluando PPL INT4...")
    ppl_int4, n_int4, t_int4 = evaluate_ppl(model, input_ids)
    print(f"       PPL INT4      = {ppl_int4:.2f}")
    print(f"       Tiempo        = {t_int4:.1f}s")

    # ── Resumen ──────────────────────────────────────────────────
    delta_ppl = ppl_int4 - ppl_base
    pct = (delta_ppl / ppl_base) * 100

    print("\n" + "=" * 65)
    print(" RESUMEN")
    print("=" * 65)
    results = {
        "model": MODEL_NAME,
        "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "group_size": GROUP_SIZE,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "baseline_ppl": round(ppl_base, 2),
        "int4_ppl": round(ppl_int4, 2),
        "delta_ppl": round(delta_ppl, 2),
        "ppl_increase_pct": round(pct, 2),
        "layers_quantized": q_stats["layers_quantized"],
        "layers_skipped": q_stats["layers_skipped"],
        "avg_compression_ratio": round(q_stats["avg_compression_ratio"], 4),
        "baseline_eval_time_s": round(t_base, 1),
        "int4_eval_time_s": round(t_int4, 1),
        "quantization_time_s": round(t_quant, 1),
    }

    for k, v in results.items():
        print(f"  {k:28s}: {v}")

    # Guardar JSON
    out_path = Path(__file__).parent / "results_pythia_int4.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados en: {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
