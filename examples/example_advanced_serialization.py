#!/usr/bin/env python3
"""
Ejemplo de serialización avanzada en MNEME.

La serialización es única y segura (safetensors + LZ4, sin pickle); lo que se
elige por tensor es la ruta del descriptor (RAW, SVD, INT8 por grupos) y las
protecciones (firma HMAC del marco MNEM, cifrado en reposo con secret_key).
Este ejemplo recorre esas opciones con la API vigente.
"""

import time

import torch

from mneme import (
    DecompType,
    MnemeConfig,
    SecureSerializer,
    SecurityConfig,
    SecurityLevel,
    ValidationError,
    ZSpace,
)

SECRET_KEY = b"clave_de_ejemplo_de_32_bytes____"


def demo_descriptor_formats():
    """Comparar las rutas de serialización que elige el routing inteligente.

    RAW y QUANTIZED no se fuerzan con decomp_type: RAW lo decide el routing
    por forma/tamaño y la cuantización se pide con quantization_type.
    """
    print("=== Formatos de descriptor (routing RAW / SVD / TT / INT8) ===")

    variants = [
        ("RAW (2-D pequeño -> safetensors + LZ4, sin pérdida)", "fmt_raw",
         torch.randn(96, 96), {}),
        ("SVD (2-D grande, target_ratio=0.1)", "fmt_svd",
         torch.randn(512, 512), {"decomp_type": DecompType.SVD, "target_ratio": 0.1}),
        ("TT (3-D -> Tensor-Train)", "fmt_tt",
         torch.randn(32, 32, 32), {"target_ratio": 0.1}),
        ("INT8 por grupos (group_size=128)", "fmt_int8",
         torch.randn(512, 512), {"quantization_type": "int8", "group_size": 128}),
    ]

    with ZSpace() as mneme:
        for label, name, tensor, kwargs in variants:
            original_bytes = tensor.numel() * tensor.element_size()

            start = time.time()
            desc = mneme.register(name, tensor, **kwargs)
            register_time = time.time() - start

            start = time.time()
            loaded = mneme.load(name).cpu()
            load_time = time.time() - start

            error = torch.norm(tensor - loaded) / torch.norm(tensor)
            print(f"\n--- {label} ---")
            print(f"Forma: {tuple(tensor.shape)}, "
                  f"original: {original_bytes / 1024:.1f}KB")
            print(f"Tipo de descriptor: {desc.decomp_type.value}")
            print(f"Tamaño serializado: {len(desc.core_data)} bytes "
                  f"({len(desc.core_data) / original_bytes:.1%} del original)")
            print(f"Error de reconstrucción: {error:.6f}")
            print(f"Registro: {register_time * 1000:.1f}ms, "
                  f"carga: {load_time * 1000:.1f}ms")


def demo_security_levels():
    """Demostrar la firma HMAC del marco MNEM y el cifrado en reposo."""
    print("\n=== Niveles de seguridad ===")

    # 2-D pequeño: el routing lo serializa RAW y el roundtrip es exacto
    sensitive = torch.randn(64, 64)

    # 1) Serialización sin firma vs firmada (SecureSerializer, formato MNEM)
    configurations = [
        ("sin firma", SecurityConfig(
            security_level=SecurityLevel.NONE,
            require_signatures=False,
        )),
        ("firmado HMAC", SecurityConfig(
            security_level=SecurityLevel.HMAC,
            require_signatures=True,
            signing_key=SECRET_KEY,
        )),
    ]

    for label, sec_config in configurations:
        serializer = SecureSerializer(sec_config)

        start = time.time()
        blob = serializer.serialize_tensor(sensitive)
        serialize_time = time.time() - start

        restored, _metadata = serializer.deserialize_tensor(blob)
        roundtrip_ok = torch.allclose(sensitive, restored)

        print(f"\n--- {label} ---")
        print(f"Tamaño: {len(blob)} bytes, tiempo: {serialize_time * 1000:.1f}ms")
        print(f"Roundtrip íntegro: {'✓' if roundtrip_ok else '✗'}")

    # 2) Cifrado en reposo dentro de ZSpace (secret_key lo habilita)
    with ZSpace(MnemeConfig(secret_key=SECRET_KEY)) as mneme:
        mneme.register("tensor_sensible", sensitive)
        restored = mneme.load("tensor_sensible").cpu()
        security_stats = mneme.get_stats()["security"]

        print("\n--- cifrado en reposo (ZSpace + secret_key) ---")
        print(f"Roundtrip cifrado: {'✓' if torch.allclose(sensitive, restored) else '✗'}")
        print(f"Nivel de seguridad activo: {security_stats['config']['security_level']}")
        print(f"Eventos de auditoría: {security_stats['audit_events']}")


def demo_compression_behaviour():
    """La compresión LZ4 del descriptor RAW depende de la compresibilidad."""
    print("\n=== Compresión LZ4 según el contenido ===")

    # Tensores 1-D: el routing los manda a RAW (safetensors + LZ4) y la
    # diferencia de tamaño refleja solo la compresibilidad del contenido.
    random_data = torch.randn(250_000)    # Ruido: apenas comprime
    sparse_data = torch.zeros(250_000)    # Disperso: comprime muchísimo
    sparse_data[::100] = 1.0

    with ZSpace() as mneme:
        for label, data in (("aleatorios", random_data), ("dispersos", sparse_data)):
            desc = mneme.register(f"datos_{label}", data)
            original_bytes = data.numel() * data.element_size()
            print(f"Datos {label} ({desc.decomp_type.value}): "
                  f"{original_bytes} -> {len(desc.core_data)} bytes "
                  f"({len(desc.core_data) / original_bytes:.1%})")


def demo_validation_features():
    """Validación de entradas e integridad de descriptores."""
    print("\n=== Validación ===")

    with ZSpace() as mneme:
        desc = mneme.register("tensor_valido", torch.randn(50, 50))
        integrity_ok = desc.verify_integrity()
        print(f"Registro válido: {desc.decomp_type.value}, "
              f"integridad {'✓' if integrity_ok else '✗'}")

        rejected_cases = [
            ("dato no tensorial", "no_tensor", {"clave": "valor"}),
            ("nombre vacío", "", torch.randn(4, 4)),
        ]

        for label, name, data in rejected_cases:
            try:
                mneme.register(name, data)
                print(f"{label}: aceptado (inesperado)")
            except ValidationError as exc:
                print(f"{label}: rechazado ({type(exc).__name__})")


def demo_performance_comparison():
    """Comparar coste de las rutas de descriptor y del cifrado en reposo."""
    print("\n=== Comparación de rendimiento ===")

    # Todos los escenarios usan 1M de elementos (4MB): los 1-D rutean a RAW
    # y los 2-D permiten SVD/INT8, así los tamaños son comparables.
    scenarios = [
        ("RAW 1-D", MnemeConfig(),
         torch.randn(1_000_000), {}),
        ("RAW 1-D + cifrado en reposo", MnemeConfig(secret_key=SECRET_KEY),
         torch.randn(1_000_000), {}),
        ("SVD 2-D (target_ratio=0.1)", MnemeConfig(),
         torch.randn(1000, 1000), {"decomp_type": DecompType.SVD, "target_ratio": 0.1}),
        ("INT8 por grupos 2-D", MnemeConfig(),
         torch.randn(1000, 1000), {"quantization_type": "int8", "group_size": 128}),
    ]

    for label, config, tensor, kwargs in scenarios:
        with ZSpace(config) as mneme:
            start = time.time()
            desc = mneme.register("perf_test", tensor, **kwargs)
            register_time = time.time() - start

            start = time.time()
            loaded = mneme.load("perf_test").cpu()
            load_time = time.time() - start

            error = torch.norm(tensor - loaded) / torch.norm(tensor)
            print(f"\n--- {label} ({desc.decomp_type.value}) ---")
            print(f"Registro: {register_time:.4f}s, carga: {load_time:.4f}s")
            print(f"Tamaño: {len(desc.core_data)} bytes, error: {error:.6f}")


if __name__ == "__main__":
    print("MNEME - Demostración de Serialización Avanzada")
    print("=" * 50)

    try:
        demo_descriptor_formats()
        demo_security_levels()
        demo_compression_behaviour()
        demo_validation_features()
        demo_performance_comparison()

        print("\n" + "=" * 50)
        print("✓ Demostración completada exitosamente")
        print("\nCaracterísticas demostradas:")
        print("- Rutas de descriptor: RAW (sin pérdida), SVD, TT e INT8 por grupos")
        print("- Firma HMAC del marco de serialización MNEM (sin pickle)")
        print("- Cifrado en reposo activado por secret_key")
        print("- Compresión LZ4 sensible a la compresibilidad del dato")
        print("- Validación de entradas e integridad de descriptores")

    except Exception as e:
        print(f"\n✗ Error durante la demostración: {e}")
        import traceback
        traceback.print_exc()
