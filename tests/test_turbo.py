"""Batería del banco para ZLinearTurbo / compress_model_turbo.

Criterios preregistrados:
  C1  svd: forward == F.linear con el peso materializado (atol/rtol 1e-4).
  C2  quantized: el peso sintetizado es BIT-idéntico al zspace.load del mismo
      descriptor, y el forward equivale al denso con ese peso.
  C3  raw: sin pérdida — forward equivale al nn.Linear original (1e-6).
  C4  gradientes de entrada y bias == referencia densa (1e-4) en svd y quantized.
  C5  memoria residente: svd(0.1) < 35% del peso denso; int8 < 35%.
  C6  cuda: el pico de memoria del forward svd es menor que el del denso.
  C7  cuda: 30 forwards quantized no acumulan memoria (sin fuga).
  C8  compress_model_turbo reemplaza solo capas grandes y sus logits DIFIEREN
      del original (anti-E6c: la compresión llega a la inferencia) con error
      relativo acotado (< 0.5 con target_ratio=0.25).
  C9  no_grad + eval funcionan en las tres rutas.
"""

import logging

import pytest
import torch
import torch.nn.functional as F
from torch import nn

logging.disable(logging.INFO)

from mneme import CompressionConfig  # noqa: E402
from mneme.mneme_lazy import ZLinearTurbo, compress_model_turbo  # noqa: E402
from mneme.mneme_torch import _zspace  # noqa: E402

CUDA = torch.cuda.is_available()


def hacer_turbo(out_f, in_f, seed, **kw):
    torch.manual_seed(seed)
    linear = nn.Linear(in_f, out_f)
    turbo = ZLinearTurbo.from_linear(linear, name=f"bench_{seed}_{out_f}x{in_f}",
                                     **kw)
    return linear, turbo


# ------------------------------------------------------------ C0: contrato
def test_c0_config_por_defecto_no_arrastra_cuantizacion():
    """CompressionConfig.quantization_type default es "int8" (flujo calibrado);
    el turbo no debe heredarlo implícito: sin decomp_type manda el routing."""
    _, turbo = hacer_turbo(256, 192, seed=99,
                           config=CompressionConfig(target_ratio=0.25))
    assert turbo.decomp_type == "svd"  # routing auto para 2-D de 49k elementos


# ---------------------------------------------------------------- C1: svd
def test_c1_svd_forward_equivale_al_peso_materializado():
    _, turbo = hacer_turbo(256, 192, seed=1,
                           config=CompressionConfig(target_ratio=0.25,
                                                    decomp_type="svd"))
    assert turbo.decomp_type == "svd"
    x = torch.randn(17, 192)
    with torch.no_grad():
        y_turbo = turbo(x)
        y_ref = F.linear(x, turbo.materialize_weight(), turbo.bias)
    assert torch.allclose(y_turbo, y_ref, atol=1e-4, rtol=1e-4)


# ----------------------------------------------------------- C2: quantized
def test_c2_quantized_bit_identico_al_load_y_forward_correcto():
    torch.manual_seed(2)
    linear = nn.Linear(128, 256)
    nombre = "bench_c2_int8"
    turbo = ZLinearTurbo.from_linear(
        linear, name=nombre,
        register_kwargs={"quantization_type": "int8", "group_size": 128})
    assert turbo.decomp_type == "quantized"

    peso_turbo = turbo.materialize_weight()
    peso_canonico = _zspace.load(nombre).cpu()
    assert torch.equal(peso_turbo, peso_canonico)

    x = torch.randn(9, 128)
    with torch.no_grad():
        y_turbo = turbo(x)
        y_ref = F.linear(x, peso_canonico, turbo.bias)
    assert torch.allclose(y_turbo, y_ref, atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------- C3: raw
def test_c3_raw_sin_perdida_equivale_al_linear_original():
    torch.manual_seed(3)
    linear = nn.Linear(64, 48)  # 3072 params < 10k -> routing RAW
    turbo = ZLinearTurbo.from_linear(linear, name="bench_c3_raw")
    assert turbo.decomp_type == "raw"

    x = torch.randn(5, 64)
    with torch.no_grad():
        y_turbo = turbo(x)
        y_ref = linear(x)
    assert torch.allclose(y_turbo, y_ref, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------- C4: gradientes
@pytest.mark.parametrize("ruta", ["svd", "quantized"])
def test_c4_gradientes_de_entrada_y_bias_coinciden_con_referencia(ruta):
    if ruta == "svd":
        _, turbo = hacer_turbo(96, 80, seed=4,
                               config=CompressionConfig(target_ratio=0.5,
                                                        decomp_type="svd"))
    else:
        _, turbo = hacer_turbo(96, 80, seed=5,
                               register_kwargs={"quantization_type": "int8",
                                                "group_size": 128})

    x = torch.randn(11, 80, requires_grad=True)
    y = turbo(x)
    y.square().sum().backward()

    x_ref = x.detach().clone().requires_grad_(True)
    bias_ref = turbo.bias.detach().clone().requires_grad_(True)
    peso = turbo.materialize_weight()
    F.linear(x_ref, peso, bias_ref).square().sum().backward()

    assert torch.allclose(x.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(turbo.bias.grad, bias_ref.grad, atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------- C5: memoria
def test_c5_memoria_residente_bajo_el_35_por_ciento():
    denso = 1024 * 1024 * 4  # bytes del peso fp32

    _, turbo_svd = hacer_turbo(1024, 1024, seed=6,
                               config=CompressionConfig(target_ratio=0.1,
                                                        decomp_type="svd"))
    assert turbo_svd.memoria_residente_bytes() < 0.35 * denso

    _, turbo_q = hacer_turbo(1024, 1024, seed=7,
                             register_kwargs={"quantization_type": "int8",
                                              "group_size": 128})
    assert turbo_q.memoria_residente_bytes() < 0.35 * denso


# ------------------------------------------------------- C6: pico en CUDA
@pytest.mark.skipif(not CUDA, reason="requiere CUDA para medir pico")
def test_c6_pico_de_forward_svd_menor_que_denso():
    """Compara DELTAS de pico sobre la base inmediata de cada fase: el estado
    acumulado del proceso (cache del ZSpace, tensores de otros tests) se
    cancela y queda solo el coste de subir la capa y hacer forward."""
    torch.manual_seed(8)
    linear = nn.Linear(2048, 2048)
    turbo = ZLinearTurbo.from_linear(
        linear, name="bench_c6",
        config=CompressionConfig(target_ratio=0.1, decomp_type="svd"))

    x_gpu = torch.randn(64, 2048, device="cuda")

    def coste_de_fase(modulo):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        modulo_gpu = modulo.cuda()
        with torch.no_grad():
            modulo_gpu(x_gpu)
        torch.cuda.synchronize()
        pico = torch.cuda.max_memory_allocated() - base
        modulo_gpu.cpu()
        return pico

    coste_denso = coste_de_fase(linear)
    coste_turbo = coste_de_fase(turbo)

    assert coste_turbo < coste_denso, (
        f"coste turbo {coste_turbo} >= coste denso {coste_denso}")


# ------------------------------------------------------------ C7: sin fuga
@pytest.mark.skipif(not CUDA, reason="requiere CUDA para medir memoria")
def test_c7_treinta_forwards_quantized_no_acumulan_memoria():
    _, turbo = hacer_turbo(512, 512, seed=9,
                           register_kwargs={"quantization_type": "int8",
                                            "group_size": 128})
    turbo = turbo.cuda()
    x = torch.randn(32, 512, device="cuda")

    with torch.no_grad():
        turbo(x)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()

    with torch.no_grad():
        for _ in range(30):
            turbo(x)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= base + 1024 * 1024  # <= +1MB


# ------------------------------------------- C8: anti-E6c en modelo entero
def test_c8_compress_model_turbo_lleva_la_compresion_a_la_inferencia():
    torch.manual_seed(10)
    modelo = nn.Sequential(
        nn.Linear(32, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 8),
    )
    # Peso central con estructura (rango 16 + 2% de ruido), como un peso
    # entrenado: SVD a rank 64 debe capturarlo casi entero y la banda puede
    # ser estrecha. Sobre ruido puro la banda honesta sería ~0.6-0.9
    # (lección de E1: la estructura compra fidelidad) y no discriminaría.
    with torch.no_grad():
        base = torch.randn(256, 16) @ torch.randn(16, 256)
        base += 0.02 * base.std() * torch.randn(256, 256)
        modelo[2].weight.copy_(base)

    turbo = compress_model_turbo(
        modelo, config=CompressionConfig(target_ratio=0.25, decomp_type="svd"),
        min_params=10000)

    tipos = [type(m).__name__ for m in turbo]
    assert tipos[2] == "ZLinearTurbo"          # 256x256 = 65k: reemplazada
    assert tipos[0] == "Linear"                # 32x256 = 8192 < 10k: intacta
    assert tipos[4] == "Linear"                # 256x8 = 2048 < 10k: intacta

    x = torch.randn(40, 32)
    with torch.no_grad():
        y_orig = modelo(x)
        y_turbo = turbo(x)

    diff = torch.norm(y_orig - y_turbo) / torch.norm(y_orig)
    assert diff > 1e-3, "los logits no difieren: la compresión no llegó al forward"
    assert diff < 0.15, f"degradación excesiva para peso estructurado: {diff:.3f}"


# ------------------------------------------------------------- C9: no_grad
@pytest.mark.parametrize("kw", [
    {"config": CompressionConfig(target_ratio=0.25, decomp_type="svd")},
    {"register_kwargs": {"quantization_type": "int8", "group_size": 128}},
    {},  # raw con capa pequeña se cubre abajo
])
def test_c9_eval_y_no_grad_funcionan(kw):
    if kw:
        _, turbo = hacer_turbo(128, 96, seed=11, **kw)
    else:
        torch.manual_seed(11)
        turbo = ZLinearTurbo.from_linear(nn.Linear(48, 32), name="bench_c9_raw")
    turbo.eval()
    with torch.no_grad():
        salida = turbo(torch.randn(3, turbo.in_features))
    assert salida.shape == (3, turbo.out_features)
    assert not salida.requires_grad


# --------------------------------------------- C10: backward 3-D (M3 del G4)
@pytest.mark.parametrize("ruta", ["svd", "quantized"])
def test_c10_backward_con_entrada_3d_coincide_con_referencia(ruta):
    if ruta == "svd":
        _, turbo = hacer_turbo(64, 48, seed=20,
                               config=CompressionConfig(target_ratio=0.5,
                                                        decomp_type="svd"))
    else:
        _, turbo = hacer_turbo(64, 48, seed=21,
                               register_kwargs={"quantization_type": "int8",
                                                "group_size": 128})

    x = torch.randn(4, 7, 48, requires_grad=True)  # (batch, seq, features)
    turbo(x).square().sum().backward()

    x_ref = x.detach().clone().requires_grad_(True)
    bias_ref = turbo.bias.detach().clone().requires_grad_(True)
    F.linear(x_ref, turbo.materialize_weight(), bias_ref).square().sum().backward()

    assert torch.allclose(x.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(turbo.bias.grad, bias_ref.grad, atol=1e-4, rtol=1e-4)


# ------------------------------------- C11: dtype y device heredados (B2 G4)
def test_c11_modelo_half_se_comprime_y_ejecuta_en_half():
    torch.manual_seed(22)
    modelo = nn.Sequential(nn.Linear(64, 256), nn.ReLU(),
                           nn.Linear(256, 256)).half()
    turbo = compress_model_turbo(
        modelo, config=CompressionConfig(target_ratio=0.25, decomp_type="svd"),
        min_params=10000)
    salida = turbo(torch.randn(3, 64, dtype=torch.float16))
    assert salida.dtype == torch.float16
    assert turbo[2].materialize_weight().dtype == torch.float16


def test_c11b_dtype_float64_se_conserva():
    torch.manual_seed(23)
    linear = nn.Linear(128, 96).double()
    turbo = ZLinearTurbo.from_linear(
        linear, name="bench_c11b",
        config=CompressionConfig(target_ratio=0.25, decomp_type="svd"))
    assert turbo.materialize_weight().dtype == torch.float64
    assert turbo(torch.randn(3, 128, dtype=torch.float64)).dtype == torch.float64


@pytest.mark.skipif(not CUDA, reason="requiere CUDA")
def test_c11c_modelo_ya_en_cuda_se_comprime_y_ejecuta_en_cuda():
    torch.manual_seed(24)
    modelo = nn.Sequential(nn.Linear(64, 256), nn.ReLU(),
                           nn.Linear(256, 256)).cuda()
    turbo = compress_model_turbo(
        modelo, config=CompressionConfig(target_ratio=0.25, decomp_type="svd"),
        min_params=10000)
    salida = turbo(torch.randn(3, 64, device="cuda"))
    assert salida.device.type == "cuda"


# --------------------------------------- C12: state_dict roundtrip (M1 G4)
@pytest.mark.parametrize("modo", ["svd", "quantized", "raw"])
def test_c12_state_dict_roundtrip_reconstruye_capa_funcional(modo):
    torch.manual_seed(25)
    if modo == "svd":
        linear = nn.Linear(96, 128)
        turbo1 = ZLinearTurbo.from_linear(
            linear, name=f"bench_c12_{modo}",
            config=CompressionConfig(target_ratio=0.25, decomp_type="svd"))
    elif modo == "quantized":
        linear = nn.Linear(96, 128)
        turbo1 = ZLinearTurbo.from_linear(
            linear, name=f"bench_c12_{modo}",
            register_kwargs={"quantization_type": "int8", "group_size": 128})
    else:
        linear = nn.Linear(48, 32)  # < 10k -> raw
        turbo1 = ZLinearTurbo.from_linear(linear, name=f"bench_c12_{modo}")

    turbo2 = ZLinearTurbo(linear.in_features, linear.out_features)
    turbo2.load_state_dict(turbo1.state_dict())  # strict=True

    x = torch.randn(5, linear.in_features)
    with torch.no_grad():
        assert torch.equal(turbo1(x), turbo2(x))
    assert turbo2.decomp_type == turbo1.decomp_type


# ------------- C12b/C12c: state_dict con dtype/device no-default (G4 ronda 2)
def test_c12b_state_dict_de_modelo_double_reconstruye_capa_double():
    torch.manual_seed(29)
    linear = nn.Linear(96, 64).double()
    turbo1 = ZLinearTurbo.from_linear(
        linear, name="bench_c12b",
        config=CompressionConfig(target_ratio=0.25, decomp_type="svd"))
    turbo2 = ZLinearTurbo(96, 64)
    turbo2.load_state_dict(turbo1.state_dict())

    assert turbo2.bias.dtype == torch.float64
    x = torch.randn(3, 96, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(turbo1(x), turbo2(x))


def test_c12b2_state_dict_de_modelo_half_reconstruye_capa_half():
    torch.manual_seed(30)
    linear = nn.Linear(128, 96).half()
    turbo1 = ZLinearTurbo.from_linear(
        linear, name="bench_c12b2",
        register_kwargs={"quantization_type": "int8", "group_size": 128})
    turbo2 = ZLinearTurbo(128, 96)
    turbo2.load_state_dict(turbo1.state_dict())

    assert turbo2.bias.dtype == torch.float16
    x = torch.randn(3, 128, dtype=torch.float16)
    with torch.no_grad():
        assert torch.equal(turbo1(x), turbo2(x))


@pytest.mark.skipif(not CUDA, reason="requiere CUDA")
def test_c12c_state_dict_de_modelo_cuda_reconstruye_capa_cuda():
    torch.manual_seed(31)
    linear = nn.Linear(64, 96).cuda()
    turbo1 = ZLinearTurbo.from_linear(
        linear, name="bench_c12c",
        register_kwargs={"quantization_type": "int8", "group_size": 128})
    turbo2 = ZLinearTurbo(64, 96)
    turbo2.load_state_dict(turbo1.state_dict())

    assert turbo2.bias.device.type == "cuda"
    x = torch.randn(3, 64, device="cuda")
    with torch.no_grad():
        assert torch.equal(turbo1(x), turbo2(x))


# ------------------------- C13: rutas de descomposición no-svd (B1 del G4)
def test_c13_decomp_forzado_tt_funciona():
    _, turbo = hacer_turbo(128, 96, seed=26,
                           config=CompressionConfig(target_ratio=0.25,
                                                    decomp_type="tt"))
    assert turbo.decomp_type == "tt"
    x = torch.randn(4, 96)
    with torch.no_grad():
        y_turbo = turbo(x)
        y_ref = F.linear(x, turbo.materialize_weight(), turbo.bias)
    assert torch.allclose(y_turbo, y_ref, atol=1e-4, rtol=1e-4)


def test_c13b_peso_podado_rutea_a_sparse_y_funciona():
    """El caso natural de B1: >90% ceros -> auto_select elige SPARSE sin que
    el usuario toque nada; el forward debe funcionar, no reventar."""
    torch.manual_seed(27)
    linear = nn.Linear(128, 128)
    with torch.no_grad():
        mascara = torch.rand(128, 128) < 0.95
        linear.weight[mascara] = 0.0
    turbo = ZLinearTurbo.from_linear(linear, name="bench_c13b")
    assert turbo.decomp_type == "sparse"
    x = torch.randn(4, 128)
    with torch.no_grad():
        y_turbo = turbo(x)
        y_ref = linear(x)
    assert torch.allclose(y_turbo, y_ref, atol=1e-4, rtol=1e-4)


# ----------------------------------------------- C14: exclude_layers (m2 G4)
def test_c14_exclude_layers_respeta_la_ruta():
    torch.manual_seed(28)
    modelo = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                           nn.Linear(128, 128))
    turbo = compress_model_turbo(
        modelo, config=CompressionConfig(target_ratio=0.25, decomp_type="svd"),
        min_params=1000, exclude_layers=["0"])
    assert type(turbo[0]).__name__ == "Linear"
    assert type(turbo[2]).__name__ == "ZLinearTurbo"
