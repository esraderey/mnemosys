"""Tests ancla de la auditoría (criba + peritaje) de 2026-08.

Cada test de este archivo reproduce un defecto confirmado con PoC durante la
auditoría. Si alguno vuelve a fallar, el defecto ha vuelto.

Referencias: PER-COR-001, PER-COR-011, PER-SEG-002, PER-SEG-005, PER-SEG-015,
PER-CON-006, PER-ROB-017.
"""

import dataclasses
import inspect
import threading

import numpy as np
import pytest
import torch
import torch.nn as nn

from mneme import mneme_torch
from mneme.mneme_core import (
    QUANT_FORMAT_VERSION,
    CompressionError,
    DecompType,
    TensorDecomposer,
    ValidationError,
    ZSpace,
    _dequantize_group_payload,
)
from mneme.mneme_optimization import CheckpointData, StructuredSparsifier, TensorQuantizer
from mneme.mneme_security_core import (
    IntegrityError,
    SecureSerializer,
    SecurityConfig,
    SecurityManager,
)
from mneme.mneme_torch import CompressionConfig, ZConv2d, ZLinear, ZParameter

CLAVE = b"clave-de-prueba-de-32-bytes-just"


def _paso_de_cuantizacion(t: torch.Tensor, n_bits: int = 8) -> float:
    return float((t.max() - t.min()) / ((1 << n_bits) - 1))


# --------------------------------------------------------------------------
# PER-COR-001 — corrupción silenciosa de la cuantización agrupada
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "nombre,tensor",
    [
        # El defecto original: el offset se codificaba como zero-point entero
        # recortado a [0, qmax], que no puede representar un g_min positivo.
        ("todo_positivo_lejos_de_cero", torch.linspace(9.98, 10.03, 1000)),
        ("ganancias_layernorm", torch.linspace(0.94, 1.05, 4096)),
        ("todo_negativo", torch.linspace(-10.03, -9.98, 777)),
        # Control: los grupos que cruzan el cero nunca estuvieron afectados.
        ("cruza_cero", torch.linspace(-0.05, 0.06, 1000)),
    ],
)
def test_cuantizacion_conserva_el_offset_del_grupo(nombre, tensor):
    """El error de reconstrucción debe ser del orden del paso de cuantización.

    Antes del arreglo, un tensor de valores en torno a 10 se reconstruía en torno
    a 0: error de 10^5 veces el paso, sin lanzar ninguna excepción.
    """
    espacio = ZSpace()
    espacio.register(nombre, tensor, quantization_type="int8", group_size=128)
    reconstruido = espacio.load(nombre).cpu()

    assert reconstruido.shape == tensor.shape
    error = float((reconstruido - tensor).abs().max())
    assert error <= 2 * _paso_de_cuantizacion(tensor), (
        f"{nombre}: error {error:.6g} frente a un paso de "
        f"{_paso_de_cuantizacion(tensor):.6g}; el offset del grupo se perdió"
    )


def test_cuantizacion_int4_agrupada_conserva_el_offset():
    """Mismo defecto en TensorQuantizer, ruta por defecto de GPTQ con bits=4."""
    quantizer = TensorQuantizer()
    tensor = torch.linspace(0.94, 1.05, 512).reshape(4, 128)

    empaquetado, meta = quantizer._quantize_int4_group(tensor, 128)
    reconstruido = quantizer._dequantize_int4_group(empaquetado, meta)

    error = float((reconstruido - tensor).abs().max())
    assert error <= 2 * _paso_de_cuantizacion(tensor, n_bits=4)


def test_metadata_int4_sin_g_min_es_rechazada():
    """La metadata del codificador anterior no se interpreta en silencio."""
    quantizer = TensorQuantizer()
    metadata_antigua = {
        "scales": torch.ones(4, 1),
        "zero_points": torch.zeros(4, 1),  # clave del formato viejo
        "original_shape": [4, 128],
        "group_size": 128,
        "pad_cols": 0,
        "rows": 4,
        "num_groups": 1,
    }
    with pytest.raises(ValueError, match="g_min"):
        quantizer._dequantize_int4_group(
            torch.zeros(4, 64, dtype=torch.int8), metadata_antigua
        )


def test_payload_de_cuantizacion_v1_es_rechazado():
    """Un payload sin marca de versión no se lee: sus valores son incorrectos."""
    payload_v1 = {
        "quantized": np.zeros((2, 128), dtype=np.uint8).tobytes(),
        "scale": np.ones(2, dtype=np.float32).tobytes(),
        "zero_point": np.zeros(2, dtype=np.float32).tobytes(),
        "n_bits": 8,
        "group_size": 128,
        "n_groups": 2,
        "original_numel": 256,
        "shape": [256],
        "dtype": "torch.float32",
    }
    with pytest.raises(CompressionError):
        _dequantize_group_payload(payload_v1)


def test_formato_de_cuantizacion_declara_su_version():
    assert QUANT_FORMAT_VERSION >= 2


# --------------------------------------------------------------------------
# PER-COR-011 — división entera en el agrupamiento
# --------------------------------------------------------------------------

@pytest.mark.parametrize("numel", [300, 1000, 15000, 129, 257])
def test_cuantizacion_admite_tamanos_no_multiplos_del_grupo(numel):
    """Antes, todo numel > group_size y no múltiplo hacía fallar el reshape."""
    espacio = ZSpace()
    tensor = torch.linspace(0.5, 2.5, numel)
    espacio.register(f"nomult_{numel}", tensor, quantization_type="int8", group_size=128)
    reconstruido = espacio.load(f"nomult_{numel}").cpu()
    assert reconstruido.shape == tensor.shape


# --------------------------------------------------------------------------
# PER-SEG-002 — integridad que no cubría el payload
# --------------------------------------------------------------------------

def test_descriptor_con_payload_forjado_no_supera_la_integridad():
    """Recalcular los hashes sin la clave ya no basta para pasar por bueno."""
    espacio = ZSpace()
    espacio.register("w", torch.randn(40, 40))
    original = espacio.name_to_desc["w"]
    assert original.verify_integrity()

    forjado = dataclasses.replace(original, core_data=b"X" * len(original.core_data))
    forjado.merkle_root = forjado._compute_merkle_root()

    assert not forjado.verify_integrity()


def test_load_rechaza_un_payload_alterado():
    """La comprobación tiene que estar en el camino REAL de lectura.

    Llamar a `verify_integrity()` a mano no prueba nada: `load()` reconstruye por
    `lazy_tensor` y durante un tiempo no verificaba nada, así que un payload
    alterado se devolvía sin una sola excepción.
    """
    import lz4.frame
    import msgpack

    from mneme.mneme_core import SecurityError

    espacio = ZSpace()
    espacio.register("q", torch.linspace(1, 2, 1000), quantization_type="int8")
    assert espacio.load("q") is not None

    descriptor = espacio.name_to_desc["q"]
    info = msgpack.unpackb(lz4.frame.decompress(descriptor.core_data), raw=False)
    cuantizados = bytearray(info["quantized"])
    cuantizados[0] ^= 0xFF
    info["quantized"] = bytes(cuantizados)
    alterado = lz4.frame.compress(msgpack.packb(info, use_bin_type=True))

    descriptor.core_data = alterado
    descriptor.lazy_tensor.compressed_data = alterado
    descriptor.lazy_tensor._decompressed_tensor = None
    espacio.adaptive_cache.put("desc_q", descriptor)

    with pytest.raises(SecurityError):
        espacio.load("q")


def test_almacenamiento_alterado_no_se_confunde_con_inexistente():
    """Un checksum que no cuadra es alteración, no 'esa clave no existe'."""
    import pathlib
    import secrets
    import tempfile

    from mneme.mneme_storage_core import (
        SecureStorageBackend,
        StorageAuthenticationError,
        StorageConfig,
    )

    directorio = tempfile.mkdtemp()
    backend = SecureStorageBackend(StorageConfig(
        storage_path=directorio,
        enable_encryption=True,
        secret_key=secrets.token_bytes(32),
    ))
    backend.store("k", b"contenido importante" * 50)
    backend.cache.clear()

    archivo = next(p for p in pathlib.Path(directorio).rglob("*.dat"))
    crudo = bytearray(archivo.read_bytes())
    crudo[-5] ^= 0xFF
    archivo.write_bytes(bytes(crudo))

    with pytest.raises(StorageAuthenticationError):
        backend.retrieve("k")

    # Control: una clave que de verdad no existe sí devuelve None.
    assert backend.retrieve("no_existe") is None


def test_clave_de_entorno_corta_es_rechazada(monkeypatch):
    """La validación de longitud también cubre la clave que viene del entorno."""
    from mneme.mneme_core import MnemeConfig, ValidationError

    monkeypatch.setenv("MNEME_SECRET_KEY", "corta")
    with pytest.raises(ValidationError):
        MnemeConfig()


def test_artefacto_alterado_falla_la_verificacion_hmac():
    serializador = SecureSerializer(SecurityConfig(signing_key=CLAVE))
    datos = serializador.serialize_tensor(torch.randn(32, 32))

    alterado = bytearray(datos)
    alterado[-40] ^= 0xFF
    with pytest.raises(IntegrityError):
        serializador.deserialize_tensor(bytes(alterado))


def test_otra_clave_no_verifica_el_artefacto():
    a = SecureSerializer(SecurityConfig(signing_key=CLAVE))
    b = SecureSerializer(SecurityConfig(signing_key=b"otra-clave-distinta-de-32-bytes!"))
    with pytest.raises(IntegrityError):
        b.deserialize_tensor(a.serialize_tensor(torch.randn(8, 8)))


def test_round_trip_seguro_es_exacto():
    serializador = SecureSerializer(SecurityConfig(signing_key=CLAVE))
    tensor = torch.randn(64, 64)
    recuperado, metadatos = serializador.deserialize_tensor(
        serializador.serialize_tensor(tensor, {"nombre": "w"})
    )
    assert torch.equal(recuperado, tensor)
    assert metadatos == {"nombre": "w"}


# --------------------------------------------------------------------------
# PER-SEG-015 — el formato se declara, no se adivina
# --------------------------------------------------------------------------

def test_un_lote_no_se_acepta_donde_se_espera_un_tensor():
    serializador = SecureSerializer(SecurityConfig(signing_key=CLAVE))
    lote = serializador.serialize_batch([torch.randn(4, 4), torch.randn(4, 4)])
    with pytest.raises(IntegrityError):
        serializador.deserialize_tensor(lote)


def test_secure_deserialize_despacha_por_cabecera():
    gestor = SecurityManager(SecurityConfig(signing_key=CLAVE))
    tensores = [torch.randn(4, 4), torch.randn(4, 4)]
    lote, _ = gestor.secure_deserialize(gestor.serializer.serialize_batch(tensores))
    assert isinstance(lote, list) and len(lote) == 2

    uno, _ = gestor.secure_deserialize(
        gestor.serializer.serialize_tensor(torch.randn(4, 4))
    )
    assert isinstance(uno, torch.Tensor)


def test_artefacto_sin_cabecera_mneme_es_rechazado():
    gestor = SecurityManager(SecurityConfig(signing_key=CLAVE))
    with pytest.raises(IntegrityError):
        gestor.secure_deserialize(b"esto no es un artefacto MNEME")


# --------------------------------------------------------------------------
# PER-SEG-005 — sin archivo temporal no hay ventana TOCTOU
# --------------------------------------------------------------------------

def test_el_serializador_no_usa_archivos_temporales():
    fuente = inspect.getsource(SecureSerializer)
    assert "NamedTemporaryFile" not in fuente
    assert "except:" not in fuente


# --------------------------------------------------------------------------
# PER-CON-006 — hook reentrante no debe bloquear
# --------------------------------------------------------------------------

def test_hook_post_store_puede_releer_el_espacio_sin_bloquearse():
    espacio = ZSpace()
    visto = {}

    def hook(descriptor, name=None, **kwargs):
        visto[name] = tuple(espacio.load(name).shape)
        return descriptor

    espacio.register_optimizer_hook("post_store", hook)
    hilo = threading.Thread(
        target=lambda: espacio.register("reentra", torch.randn(16, 16)), daemon=True
    )
    hilo.start()
    hilo.join(timeout=30)

    assert not hilo.is_alive(), "register() se autobloqueó con un hook reentrante"
    assert visto == {"reentra": (16, 16)}


# --------------------------------------------------------------------------
# PER-ROB-017 — los checkpoints usan mscs, nunca pickle
# --------------------------------------------------------------------------

def test_los_checkpoints_no_usan_pickle():
    """Ni llamadas a pickle en el checkpoint, ni el import en todo el módulo.

    Se comprueba el código, no los comentarios: la mención de la palabra en una
    nota explicativa es legítima; una llamada o un import, no.
    """
    import ast

    import mneme.mneme_optimization as modulo

    fuente_checkpoint = inspect.getsource(CheckpointData)
    assert "pickle.dumps" not in fuente_checkpoint
    assert "pickle.loads" not in fuente_checkpoint

    arbol = ast.parse(inspect.getsource(modulo))
    importados = {
        alias.name.split(".")[0]
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Import)
        for alias in nodo.names
    } | {
        nodo.module.split(".")[0]
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.ImportFrom) and nodo.module
    }
    assert "pickle" not in importados, (
        "reintroducir pickle aquí abre ejecución arbitraria al cargar un checkpoint "
        "escrito por otro usuario en el directorio temporal compartido"
    )


# --------------------------------------------------------------------------
# Persistencia entre procesos (hueco detectado en la revisión adversarial)
# --------------------------------------------------------------------------

_GUION_PERSISTENCIA = """
import sys, torch
from mneme import ZSpace
from mneme.mneme_core import MnemeConfig

accion, ruta = sys.argv[1], sys.argv[2]
espacio = ZSpace(MnemeConfig(storage_path=ruta))
esperado = torch.arange(64, dtype=torch.float32)

if accion == "guardar":
    espacio.register("p", esperado)
    espacio.sync_to_storage()
    print("GUARDADO")
else:
    recuperado = espacio.load("p").cpu()
    assert torch.equal(recuperado, esperado), "los valores no sobrevivieron al reinicio"
    print("RECUPERADO")
"""


@pytest.mark.parametrize("con_clave", [False, True])
def test_lo_persistido_sobrevive_a_reiniciar_el_proceso(tmp_path, con_clave):
    """Guardar, terminar el proceso, volver a arrancar y releer.

    Es el ciclo que ni el cifrado en reposo ni la firma HMAC pueden romper: una
    clave generada al vuelo por proceso dejaría los datos ilegibles en el
    siguiente arranque, y de forma silenciosa.
    """
    import os
    import subprocess
    import sys

    guion = tmp_path / "ciclo.py"
    guion.write_text(_GUION_PERSISTENCIA, encoding="utf-8")
    almacen = tmp_path / "almacen"

    entorno = dict(os.environ)
    entorno["PYTHONIOENCODING"] = "utf-8"
    if con_clave:
        entorno["MNEME_SECRET_KEY"] = "clave-estable-de-treinta-y-dos-b"
    else:
        entorno.pop("MNEME_SECRET_KEY", None)
        entorno.pop("MNEME_SIGNING_KEY", None)

    for accion, esperado in (("guardar", "GUARDADO"), ("leer", "RECUPERADO")):
        proceso = subprocess.run(
            [sys.executable, "-W", "ignore", str(guion), accion, str(almacen)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=entorno, timeout=300,
        )
        assert esperado in (proceso.stdout or ""), (
            f"fase '{accion}' falló (con_clave={con_clave}):\n"
            f"{proceso.stdout}\n{proceso.stderr[-1500:]}"
        )


def test_round_trip_de_checkpoint():
    from mneme.mneme_optimization import PerformanceMetrics

    checkpoint = CheckpointData(
        checkpoint_id="abc",
        created_at=0.0,
        optimizer_state={"paso": 1},
        metrics_snapshot=PerformanceMetrics(),
        resource_state={},
        tensor_pool_state={},
        metadata={"nota": "ancla"},
    )
    recuperado = CheckpointData.from_bytes(checkpoint.to_bytes())
    assert recuperado.checkpoint_id == "abc"
    assert recuperado.metadata == {"nota": "ancla"}


# --------------------------------------------------------------------------
# G4-SEC-001 — la clave pasada por código alimenta la firma y el HMAC
# --------------------------------------------------------------------------

OTRA_CLAVE = b"otra-clave-distinta-de-32-bytes!"


def _sin_claves_de_entorno(monkeypatch):
    monkeypatch.delenv("MNEME_SECRET_KEY", raising=False)
    monkeypatch.delenv("MNEME_SIGNING_KEY", raising=False)


def test_la_clave_por_config_firma_el_marco_serializado(monkeypatch, tmp_path):
    """`MnemeConfig(secret_key=...)` debe firmar, no solo cifrar.

    Antes, la clave pasada por código cifraba el almacenamiento pero la firma
    HMAC del marco solo miraba el entorno: los artefactos salían SIN firmar y
    nadie avisaba.
    """
    import lz4.frame

    from mneme.mneme_core import MnemeConfig
    from mneme.mneme_security_core import FLAG_SIGNED

    _sin_claves_de_entorno(monkeypatch)
    espacio = ZSpace(MnemeConfig(secret_key=CLAVE, storage_path=str(tmp_path / "a")))
    espacio.register("w", torch.randn(8, 8))

    marco = lz4.frame.decompress(espacio.name_to_desc["w"].core_data)
    assert marco[:4] == b"MNEM"
    assert marco[6] & FLAG_SIGNED, "el marco no va firmado con la clave de config"

    # Verifica con la misma clave; con otra, no.
    recuperado, _ = SecureSerializer(
        SecurityConfig(signing_key=CLAVE)
    ).deserialize_tensor(marco)
    assert tuple(recuperado.shape) == (8, 8)
    with pytest.raises(IntegrityError):
        SecureSerializer(SecurityConfig(signing_key=OTRA_CLAVE)).deserialize_tensor(marco)


def test_la_clave_por_config_llega_al_hash_del_descriptor(monkeypatch, tmp_path):
    """El hash de integridad del descriptor debe ser HMAC con la clave de config.

    Antes degradaba en silencio a SHA-256 sin clave: cifrado real, integridad
    falsificable, y ningún aviso.
    """
    from mneme.mneme_core import MnemeConfig

    _sin_claves_de_entorno(monkeypatch)
    espacio = ZSpace(MnemeConfig(secret_key=CLAVE, storage_path=str(tmp_path / "b")))
    espacio.register("w", torch.randn(8, 8))
    con_clave = espacio.name_to_desc["w"]
    assert con_clave._clave_integridad == CLAVE

    sin_clave = dataclasses.replace(con_clave, security_hash=None, clave_integridad=None)
    assert con_clave.security_hash != sin_clave.security_hash, (
        "el hash no depende de la clave: sigue degradado a SHA-256 sin clave"
    )
    assert con_clave.verify_integrity()
    assert sin_clave.verify_integrity()


def test_la_clave_no_viaja_en_el_descriptor_persistido(monkeypatch, tmp_path):
    """La clave alimenta el hash pero jamás sale por ninguna serialización.

    Cubre también el hallazgo del revisor G4: con la clave como field del
    dataclass, `repr=False` la escondía del repr pero `dataclasses.asdict()`
    la devolvía cruda. Como InitVar no-field, ninguna de las vías del
    dataclass la conoce.
    """
    from mneme.mneme_core import MnemeConfig, ZDescriptor

    _sin_claves_de_entorno(monkeypatch)
    espacio = ZSpace(MnemeConfig(secret_key=CLAVE, storage_path=str(tmp_path / "c")))
    espacio.register("w", torch.randn(8, 8))
    desc = espacio.name_to_desc["w"]

    assert "clave_integridad" not in desc.to_dict()
    assert CLAVE not in repr(desc).encode()
    assert "clave_integridad" not in {f.name for f in dataclasses.fields(desc)}

    # Y al recargar desde el dict persistido, el espacio la reinyecta.
    recargado = ZDescriptor.from_dict(desc.to_dict())
    assert recargado._clave_integridad is None
    # Sin lazy_tensor adjunto, asdict() sí es ejecutable: era exactamente la
    # ventana en la que la clave se filtraba.
    volcado = dataclasses.asdict(recargado)
    assert "clave_integridad" not in volcado
    recargado._clave_integridad = CLAVE
    volcado = dataclasses.asdict(recargado)
    assert all(v != CLAVE for v in volcado.values())
    assert recargado.verify_integrity()


def test_pickle_de_un_descriptor_no_transporta_la_clave(monkeypatch, tmp_path):
    """Ni pickle ni deepcopy arrastran la clave de integridad.

    pickle está vetado por convención, pero si un tercero picklea un
    descriptor, la clave no debe viajar: el blob no la contiene y el clon
    despickleado queda sin ella (su hash con clave deja de verificar cuando
    tampoco hay clave de entorno — prueba de que no viajó).
    """
    import copy
    import pickle

    from mneme.mneme_core import MnemeConfig

    _sin_claves_de_entorno(monkeypatch)
    espacio = ZSpace(MnemeConfig(secret_key=CLAVE, storage_path=str(tmp_path / "f")))
    espacio.register("w", torch.randn(8, 8))
    desc = espacio.name_to_desc["w"]

    # Sin lazy_tensor (contiene un Lock, no picklable) y con la clave puesta.
    plano = dataclasses.replace(desc, lazy_tensor=None, clave_integridad=CLAVE)
    assert plano._clave_integridad == CLAVE

    blob = pickle.dumps(plano)
    assert CLAVE not in blob, "la clave viaja en crudo dentro del pickle"

    clon = pickle.loads(blob)
    assert clon._clave_integridad is None
    assert not clon.verify_integrity(), (
        "el clon verifica sin clave: entonces el hash no dependía de ella"
    )

    duplicado = copy.deepcopy(plano)
    assert duplicado._clave_integridad is None


def test_repr_de_las_configs_no_muestra_la_clave(monkeypatch):
    """repr(config) acaba en logs: la clave no puede viajar en él.

    Cierre del egreso residual anotado por el G4 del chip de to_dict():
    secret_key era campo plano del dataclass y el repr generado la volcaba
    cruda, tanto en MnemeConfig como en StorageConfig. asdict()/pickle de una
    config SÍ llevan la clave a propósito (la config es su portadora legítima,
    p. ej. hacia workers de otro proceso); el volcado sancionado sin clave es
    to_dict(), que ya la redacta.
    """
    from mneme.mneme_core import MnemeConfig
    from mneme.mneme_storage_core import StorageConfig

    _sin_claves_de_entorno(monkeypatch)
    for config in (
        MnemeConfig(secret_key=CLAVE),
        StorageConfig(secret_key=CLAVE),
    ):
        representacion = repr(config)
        assert CLAVE.decode() not in representacion
        assert "secret_key" not in representacion, (
            "el campo entero debe quedar fuera del repr, no solo su valor"
        )


_GUION_CLAVE_POR_CONFIG = """
import sys, torch
from mneme import ZSpace
from mneme.mneme_core import MnemeConfig

accion, ruta, clave_hex = sys.argv[1], sys.argv[2], sys.argv[3]
espacio = ZSpace(MnemeConfig(secret_key=bytes.fromhex(clave_hex), storage_path=ruta))
esperado = torch.arange(64, dtype=torch.float32)

if accion == "guardar":
    espacio.register("p", esperado)
    espacio.sync_to_storage()
    print("GUARDADO")
else:
    try:
        recuperado = espacio.load("p").cpu()
    except Exception as excepcion:
        print(f"RECHAZADO:{type(excepcion).__name__}")
    else:
        assert torch.equal(recuperado, esperado), "los valores no sobrevivieron"
        print("RECUPERADO")
"""


def test_lo_persistido_con_clave_por_config_sobrevive_al_reinicio(tmp_path):
    """Guardar y releer entre procesos SOLO con la clave por parámetro.

    Es el ciclo completo del hallazgo: sin variable de entorno, la clave de
    `MnemeConfig` tiene que bastar para cifrar, firmar y verificar. Y una clave
    distinta no debe recuperar nada: el rechazo llega como
    StorageAuthenticationError, no como "ese tensor no existe".
    """
    import os
    import subprocess
    import sys

    guion = tmp_path / "ciclo_config.py"
    guion.write_text(_GUION_CLAVE_POR_CONFIG, encoding="utf-8")
    almacen = tmp_path / "almacen"

    entorno = dict(os.environ)
    entorno["PYTHONIOENCODING"] = "utf-8"
    entorno.pop("MNEME_SECRET_KEY", None)
    entorno.pop("MNEME_SIGNING_KEY", None)

    fases = (
        ("guardar", CLAVE, "GUARDADO"),
        ("leer", CLAVE, "RECUPERADO"),
        ("leer", OTRA_CLAVE, "RECHAZADO:StorageAuthenticationError"),
    )
    for accion, clave, esperado in fases:
        proceso = subprocess.run(
            [sys.executable, "-W", "ignore", str(guion), accion, str(almacen), clave.hex()],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=entorno, timeout=300,
        )
        assert esperado in (proceso.stdout or ""), (
            f"fase '{accion}' con clave {clave[:4]!r}… falló:\n"
            f"{proceso.stdout}\n{proceso.stderr[-1500:]}"
        )


# --------------------------------------------------------------------------
# G4-CON-002 — hooks post-store en orden de registro bajo contención
# --------------------------------------------------------------------------

def test_hooks_post_store_se_entregan_en_orden_bajo_contencion():
    """Dos hilos que escriben el mismo nombre no pueden invertir las entregas.

    Mover los hooks fuera del write_lock quitó el autobloqueo pero abría una
    ventana: entre soltar el lock y ejecutar el hook, otro hilo podía escribir
    el mismo nombre y el observador recibía como "último" un descriptor
    obsoleto. El pre_store corre bajo el lock, así que su orden ES el orden de
    registro; las entregas post-store deben llegar en ese mismo orden.
    """
    espacio = ZSpace()
    orden_registro, entregados = [], []

    def pre(tensor, name=None, **kwargs):
        if name == "contendido":
            orden_registro.append(float(tensor[0]))
        return tensor

    def post(desc, name=None, **kwargs):
        if name == "contendido":
            entregados.append(float(desc.lazy_tensor.decompress()[0]))
        return desc

    espacio.register_optimizer_hook("pre_store", pre)
    espacio.register_optimizer_hook("post_store", post)

    barrera = threading.Barrier(6)

    def trabajador(base):
        barrera.wait()
        for i in range(5):
            espacio.register("contendido", torch.full((16,), float(base * 100 + i)))

    hilos = [threading.Thread(target=trabajador, args=(k,)) for k in range(6)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join(timeout=120)

    assert not any(h.is_alive() for h in hilos), "deadlock entregando hooks"
    assert entregados == orden_registro, (
        "las entregas post-store no siguen el orden de registro:\n"
        f"registro:  {orden_registro}\nentregado: {entregados}"
    )
    final = float(espacio.name_to_desc["contendido"].lazy_tensor.decompress()[0])
    assert entregados[-1] == final, (
        "la última entrega no corresponde al estado final: un observador se "
        "quedaría con un descriptor obsoleto como 'último'"
    )


def test_hook_que_realmacena_el_mismo_nombre_no_se_bloquea():
    """Un hook que re-almacena el nombre en curso se encola, no se anida.

    El drenador único garantiza que la entrega del segundo store llega después
    de la del primero y que nada se bloquea ni recursa.
    """
    espacio = ZSpace()
    vistos = []

    def re_almacena(desc, name=None, **kwargs):
        vistos.append(float(desc.lazy_tensor.decompress()[0]))
        if name == "eco" and len(vistos) == 1:
            espacio.register("eco", torch.full((16,), 2.0))
        return desc

    espacio.register_optimizer_hook("post_store", re_almacena)
    hilo = threading.Thread(
        target=lambda: espacio.register("eco", torch.full((16,), 1.0)), daemon=True
    )
    hilo.start()
    hilo.join(timeout=30)

    assert not hilo.is_alive(), "register() se bloqueó con un hook que re-almacena"
    assert vistos == [1.0, 2.0], f"entregas fuera de orden: {vistos}"


# --------------------------------------------------------------------------
# G4-ROB-003 — la manipulación no se disfraza de corrupción LZ4
# --------------------------------------------------------------------------

def test_manipulacion_del_marco_no_se_disfraza_de_error_de_compresion(monkeypatch, tmp_path):
    """Un HMAC que no cuadra es manipulación y debe aflorar como IntegrityError.

    Antes, LazyTensor.decompress envolvía TODO en un CompressionError "lz4":
    quien operaba el sistema no podía distinguir un bit podrido en disco de un
    payload sustituido a propósito.
    """
    import lz4.frame

    from mneme.mneme_core import MnemeConfig

    _sin_claves_de_entorno(monkeypatch)
    espacio = ZSpace(MnemeConfig(secret_key=CLAVE, storage_path=str(tmp_path / "d")))
    espacio.register("w", torch.randn(8, 8))
    desc = espacio.name_to_desc["w"]

    marco = bytearray(lz4.frame.decompress(desc.core_data))
    marco[-1] ^= 0xFF  # LZ4 válido, firma inválida: manipulación, no corrupción
    desc.lazy_tensor.compressed_data = lz4.frame.compress(bytes(marco))
    desc.lazy_tensor._decompressed_tensor = None

    with pytest.raises(IntegrityError):
        desc.lazy_tensor.decompress()


def test_la_corrupcion_lz4_sigue_siendo_error_de_compresion_con_causa():
    """El control: bytes que no son LZ4 siguen siendo CompressionError.

    Y la excepción original viaja en __cause__ en lugar de perderse.
    """
    espacio = ZSpace()
    espacio.register("w", torch.randn(8, 8))
    desc = espacio.name_to_desc["w"]

    desc.lazy_tensor.compressed_data = b"esto no es un marco lz4"
    desc.lazy_tensor._decompressed_tensor = None

    with pytest.raises(CompressionError) as registro:
        desc.lazy_tensor.decompress()
    assert registro.value.__cause__ is not None, "la causa original se perdió"


# --------------------------------------------------------------------------
# Hallazgo G4 2026-08-15 — secret_key salía en MnemeConfig.to_dict()
# --------------------------------------------------------------------------

def test_la_clave_no_sale_en_el_volcado_de_config_ni_en_el_contexto(monkeypatch, tmp_path):
    """`to_dict()` redacta la clave; `get_optimization_context()` queda limpio.

    Antes, to_dict() incluía secret_key en base64 y el contexto de
    optimización (API pública) lo arrastraba: cualquier consumidor que
    loguee o persista ese contexto exfiltraba la clave de cifrado/firma.
    """
    import base64

    from mneme.mneme_core import MnemeConfig

    _sin_claves_de_entorno(monkeypatch)
    config = MnemeConfig(secret_key=CLAVE, storage_path=str(tmp_path / "e"))

    volcado = config.to_dict()
    assert volcado["secret_key"] == "<redactada>"
    assert base64.b64encode(CLAVE).decode() not in repr(volcado)
    assert CLAVE.decode() not in repr(volcado)

    # Sin clave configurada, el volcado lo dice con None, no con el marcador.
    sin_clave = MnemeConfig(enable_encryption=False)
    assert sin_clave.to_dict()["secret_key"] is None

    contexto = repr(ZSpace(config).get_optimization_context())
    assert base64.b64encode(CLAVE).decode() not in contexto
    assert CLAVE.decode() not in contexto
    assert "<redactada>" in contexto


def test_from_dict_jamas_reconstruye_la_clave_desde_un_volcado(monkeypatch):
    """`from_dict()` ignora secret_key: marcador, base64 legado o bytes crudos.

    Semántica del round-trip sin clave: la instancia restaurada conserva todos
    los demás campos, queda sin secret_key y `__post_init__` aplica la regla
    estándar (clave del entorno o cifrado desactivado con aviso). La clave se
    aprovisiona por el constructor o por MNEME_SECRET_KEY, nunca por volcado.
    """
    import base64

    from mneme.mneme_core import MnemeConfig

    _sin_claves_de_entorno(monkeypatch)
    original = MnemeConfig(secret_key=CLAVE)
    volcado = original.to_dict()

    variantes = [
        volcado,                                                      # marcador
        dict(volcado, secret_key=base64.b64encode(CLAVE).decode()),   # volcado pre-arreglo
        dict(volcado, secret_key=CLAVE),                              # bytes crudos
    ]
    for datos in variantes:
        with pytest.warns(RuntimeWarning):
            restaurada = MnemeConfig.from_dict(datos)
        assert restaurada.secret_key is None
        assert restaurada.enable_encryption is False

        campos_intactos = [
            campo for campo in original.__dataclass_fields__
            if campo not in ("secret_key", "enable_encryption")
        ]
        for campo in campos_intactos:
            assert getattr(restaurada, campo) == getattr(original, campo)


# --------------------------------------------------------------------------
# Hallazgo G4 2026-08-15 — decomp_type=ADAPTIVE escribía un descriptor
# irrecuperable
# --------------------------------------------------------------------------

def test_register_con_adaptive_equivale_al_routing_automatico():
    """`register(..., decomp_type=DecompType.ADAPTIVE)` significa "elige por mí".

    Antes del arreglo, ADAPTIVE entraba al camino de descomposición forzada:
    decompose() no tiene rama para él y caía a su fallback {"type": "raw"},
    un payload SIN los datos del tensor que reconstruct() tampoco sabe leer.
    register() terminaba "bien" (descriptor con decomp_type=ADAPTIVE) y
    load() lanzaba CompressionError('Unknown component type: raw'): pérdida
    silenciosa del tensor en el momento de escribir, para cualquier tamaño.
    """
    espacio = ZSpace()
    casos = {
        "grande_2d": torch.randn(100, 100),  # ≥10k elementos → descomposición
        "vector_1d": torch.randn(1500),      # 1-D: el routing lo manda a RAW
        #   seguro; resolver ADAPTIVE con auto_select()+decompose() a ciegas
        #   no serviría (auto_select da RAW para 1-D y decompose() rechaza
        #   RAW con ValidationError: no hay descomposición que ejecutar).
        "peque_2d": torch.randn(10, 10),     # <1k elementos → RAW seguro
    }
    for etiqueta, tensor in casos.items():
        adaptivo = espacio.register(
            f"g4_adaptive_{etiqueta}", tensor, decomp_type=DecompType.ADAPTIVE
        )
        control = espacio.register(f"g4_control_{etiqueta}", tensor)

        # El descriptor guarda el tipo concreto resuelto, nunca ADAPTIVE, y
        # coincide con lo que elige el routing cuando no se fuerza nada.
        assert adaptivo.decomp_type != DecompType.ADAPTIVE, etiqueta
        assert adaptivo.decomp_type == control.decomp_type, etiqueta

        recuperado = espacio.load(f"g4_adaptive_{etiqueta}").cpu()
        assert recuperado.shape == tensor.shape, etiqueta

    # El camino RAW es sin pérdida: el tensor pequeño vuelve exacto.
    assert torch.equal(
        espacio.load("g4_adaptive_peque_2d").cpu(), casos["peque_2d"]
    )


# --------------------------------------------------------------------------
# Hallazgo G4 2026-08-15 — decompose() fabricaba un payload perdedor para
# los tipos sin rama implementada
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tipo_sin_rama",
    [DecompType.RAW, DecompType.QUANTIZED, DecompType.ADAPTIVE],
    ids=lambda tipo: tipo.name.lower(),
)
def test_decompose_rechaza_los_tipos_sin_rama_implementada(tipo_sin_rama):
    """decompose() implementa TT/CP/Tucker/SVD/Sparse; el resto debe fallar claro.

    Antes del arreglo, cualquier tipo sin rama caía a un fallback silencioso
    `{"type": "raw"}`: un payload SIN los datos del tensor que reconstruct()
    tampoco sabe leer. TensorDecomposer es API pública, así que un caller
    directo con RAW o ADAPTIVE (o un miembro futuro del enum) perdía el
    tensor al escribir y solo lo descubría al intentar leer, lejos de la
    causa. Ahora la escritura falla ruidosa, inmediata y nombrando el tipo.
    """
    with pytest.raises(ValidationError) as excinfo:
        TensorDecomposer.decompose(torch.randn(6, 6), tipo_sin_rama)
    assert repr(tipo_sin_rama) in str(excinfo.value)


# --------------------------------------------------------------------------
# Hallazgo G4 2026-08-15 — decomp_type como string entraba al camino forzado
# y reventaba con un AttributeError que enmascaraba la causa
# --------------------------------------------------------------------------

def test_register_acepta_decomp_type_como_valor_string():
    """`decomp_type="svd"` debe equivaler a `decomp_type=DecompType.SVD`.

    Antes del arreglo, un string truthy pasaba el filtro del camino forzado
    (no iguala a ningún miembro de DecompType), decompose() caía a su
    fallback y `decomp_type.value` reventaba con AttributeError en la
    serialización — y otra vez al formatear el log del manejador de
    fallback, enmascarando la causa real.
    """
    espacio = ZSpace()
    tensor = torch.randn(100, 100)

    forzado = espacio.register("g4_str_svd", tensor, decomp_type="svd")
    control = espacio.register(
        "g4_str_svd_control", tensor, decomp_type=DecompType.SVD
    )

    assert forzado.decomp_type == DecompType.SVD
    assert forzado.decomp_type == control.decomp_type
    assert espacio.load("g4_str_svd").cpu().shape == tensor.shape


def test_register_con_string_adaptive_equivale_al_routing_automatico():
    """`decomp_type="adaptive"` hereda la semántica anclada de ADAPTIVE.

    La coacción a enum ocurre antes de la normalización ADAPTIVE→None, así
    que el string también significa "elige por mí": tipo concreto resuelto
    por el routing, nunca ADAPTIVE ni un AttributeError.
    """
    espacio = ZSpace()
    tensor = torch.randn(100, 100)

    adaptivo = espacio.register("g4_str_adaptive", tensor, decomp_type="adaptive")
    control = espacio.register("g4_str_adaptive_control", tensor)

    assert adaptivo.decomp_type != DecompType.ADAPTIVE
    assert adaptivo.decomp_type == control.decomp_type
    assert espacio.load("g4_str_adaptive").cpu().shape == tensor.shape


@pytest.mark.parametrize("valor", ["no_existe", 42], ids=["string", "entero"])
def test_register_rechaza_decomp_type_incoercible(valor):
    """Un decomp_type no coaccionable a DecompType falla claro y temprano.

    Antes del arreglo acababa en el mismo AttributeError enmascarado; ahora
    register() lanza ValidationError nombrando el valor rechazado antes de
    tocar ningún routing ni descomposición.
    """
    espacio = ZSpace()
    with pytest.raises(ValidationError) as excinfo:
        espacio.register(
            "g4_incoercible", torch.randn(100, 100), decomp_type=valor
        )
    assert repr(valor) in str(excinfo.value)


def test_decomp_type_invalido_no_envenena_el_circuit_breaker():
    """El input inválido se rechaza ANTES de la puerta del circuit breaker.

    Cuando la coacción vivía dentro del try de register(), cada
    ValidationError se contaba como fallo de storage y, como los éxitos no
    resetean el contador en estado CLOSED, 5 typos acumulados en la vida de
    la instancia abrían el breaker: todo register() posterior, incluso
    válido, devolvía CircuitBreakerError durante el reset_timeout. Un
    decomp_type mal escrito es un error del caller, no un síntoma de salud
    del storage: se valida junto a name/tensor, fuera del alcance del
    breaker.
    """
    espacio = ZSpace()
    for _ in range(6):
        with pytest.raises(ValidationError):
            espacio.register(
                "g4_typo", torch.randn(100, 100), decomp_type="no_existe"
            )
    valido = espacio.register("g4_tras_typos", torch.randn(10, 10))
    assert valido is not None


# --------------------------------------------------------------------------
# Hallazgo G4 2026-08-15 — CompressionConfig.decomp_type era configuración
# muerta: ninguna capa lo reenviaba a register()
# --------------------------------------------------------------------------

def test_config_decomp_type_se_reenvia_a_register():
    """`CompressionConfig(decomp_type=...)` debe forzar el tipo en la capa.

    Antes del arreglo, el campo existía pero ninguna de las cinco llamadas a
    register() de mneme_torch lo reenviaba: fijarlo no tenía ningún efecto y
    el routing automático decidía siempre (SVD para este peso 2-D grande).
    """
    torch.manual_seed(4)
    lineal = nn.Linear(120, 100)
    with torch.no_grad():
        lineal.weight.copy_(torch.outer(torch.randn(100), torch.randn(120)))

    z = ZLinear.from_existing(
        lineal, config=CompressionConfig(decomp_type=DecompType.TT)
    )

    assert z.weight._descriptor.decomp_type == DecompType.TT

    # El tipo forzado no puede costar el tensor: el peso de rango 1 se
    # reconstruye casi exacto con los rangos TT derivados de target_ratio.
    recuperado = mneme_torch._zspace.load(z.weight._zspace_name).cpu()
    original = lineal.weight.detach()
    error = float(torch.norm(recuperado - original) / torch.norm(original))
    assert error < 1e-3


def test_config_decomp_type_none_y_adaptive_dejan_el_routing_automatico():
    """El default (None) mantiene el routing intacto y ADAPTIVE equivale a él.

    Pasar ``decomp_type=None`` a register() es idéntico a omitirlo, y la
    normalización ADAPTIVE→None del core también aplica cuando el valor
    llega desde la config de la capa.
    """
    torch.manual_seed(5)
    lineal = nn.Linear(120, 100)

    por_defecto = ZLinear.from_existing(lineal, config=CompressionConfig())
    adaptativo = ZLinear.from_existing(
        lineal, config=CompressionConfig(decomp_type=DecompType.ADAPTIVE)
    )

    assert por_defecto.weight._descriptor.decomp_type == DecompType.SVD
    assert adaptativo.weight._descriptor.decomp_type == DecompType.SVD


def test_config_decomp_type_invalido_falla_al_construir_la_capa():
    """Un valor incoercible en la config revienta claro al crear la capa.

    El cableado delega la validación en el core (coerción string→enum,
    ValidationError para lo incoercible) en vez de ignorar el campo — la
    conducta anterior — o degradar en silencio.
    """
    with pytest.raises(ValidationError):
        ZParameter.from_tensor(
            torch.randn(64, 64), "g4_config_invalida",
            config=CompressionConfig(decomp_type="no_existe"),
        )


def test_config_decomp_type_tambien_viaja_en_zconv2d():
    """El peso 4-D de ZConv2d también respeta el tipo forzado (TUCKER).

    TUCKER discrimina de verdad: para un 4-D denso el routing automático
    elige TT — y con este tamaño, además, con los mismos rangos que el
    camino forzado — así que forzar TT aquí no probaría el cableado.
    """
    torch.manual_seed(6)
    capa = ZConv2d(4, 8, 3, config=CompressionConfig(decomp_type=DecompType.TUCKER))

    assert capa.weight._descriptor.decomp_type == DecompType.TUCKER

    recuperado = mneme_torch._zspace.load(capa.weight._zspace_name).cpu()
    original = capa.weight.detach().cpu()
    error = float(torch.norm(recuperado - original) / torch.norm(original))
    assert error < 1e-3


def test_config_decomp_type_tambien_viaja_en_el_constructor_de_zlinear():
    """ZLinear(...) directo — el camino de ZAttention — también fuerza el tipo.

    Sin cableado, el peso aleatorio 2-D de 12000 elementos caería al SVD
    del routing automático.
    """
    torch.manual_seed(8)
    capa = ZLinear(120, 100, config=CompressionConfig(decomp_type=DecompType.TT))
    assert capa.weight._descriptor.decomp_type == DecompType.TT


@pytest.mark.parametrize(
    "no_forzable", [DecompType.RAW, DecompType.QUANTIZED],
    ids=lambda tipo: tipo.name.lower(),
)
def test_config_decomp_type_raw_y_quantized_caen_al_routing_automatico(no_forzable):
    """RAW y QUANTIZED no son forzables: el core los devuelve al routing.

    Fija la semántica que documenta el comentario de CompressionConfig:
    quien los fija no obtiene RAW ni cuantización (eso pide
    quantization_type), sino lo mismo que el routing automático elija.
    """
    torch.manual_seed(9)
    lineal = nn.Linear(120, 100)

    forzado = ZLinear.from_existing(
        lineal, config=CompressionConfig(decomp_type=no_forzable)
    )
    control = ZLinear.from_existing(lineal, config=CompressionConfig())

    assert forzado.weight._descriptor.decomp_type != no_forzable
    assert (
        forzado.weight._descriptor.decomp_type
        == control.weight._descriptor.decomp_type
    )


def test_config_decomp_type_no_pisa_la_cuantizacion_calibrada():
    """En from_existing_calibrated, quantization_type sigue mandando.

    La precedencia vive en el core (cuantización explícita primero); fijar
    también decomp_type en la config no debe alterarla.
    """
    torch.manual_seed(7)
    z = ZLinear.from_existing_calibrated(
        nn.Linear(64, 32),
        config=CompressionConfig(decomp_type=DecompType.TT),
        quant_kwargs={"quantization_type": "int8_group", "group_size": 128},
    )
    assert z.weight._descriptor.decomp_type == DecompType.QUANTIZED


# --------------------------------------------------------------------------
# Hallazgo residual 2026-08-15 — configuración muerta de CompressionConfig
# --------------------------------------------------------------------------

CAMPOS_RETIRADOS = {
    "compression_level": "maximum",
    "memory_limit": 50 * 1024 * 1024,
    "enable_quantization": False,
    "quantization_bits": 4,
    "use_parallel_processing": True,
    "enable_security": True,
}


@pytest.mark.parametrize("campo", sorted(CAMPOS_RETIRADOS))
def test_compression_config_rechaza_cada_campo_retirado(campo):
    """Los seis campos de configuración muerta ya no se pueden fijar.

    Antes del retiro se podían fijar pero ningún código de src/ los leía:
    configurarlos era un no-op silencioso. Retirados, el constructor los
    rechaza en vez de aceptar y callar.
    """
    with pytest.raises(TypeError):
        CompressionConfig(**{campo: CAMPOS_RETIRADOS[campo]})


def test_compression_config_expone_solo_campos_vivos():
    """El conjunto de campos de CompressionConfig es exactamente el vivo.

    Cada campo de esta lista tiene un consumidor real en src/. Si este test
    falla por un campo nuevo, hay que cablearlo a un consumidor (y ampliar
    la lista) o no añadirlo: la configuración que se acepta y se ignora no
    vuelve.
    """
    assert {f.name for f in dataclasses.fields(CompressionConfig)} == {
        "target_ratio",
        "decomp_type",
        "group_size",
        "quantization_type",
        "calibration_samples",
        "calibration_data",
        "mixed_precision_policy",
        "enable_kv_cache_compression",
        "kv_cache_bits",
        "enable_structured_sparsity",
    }


def test_optimize_model_memory_comprime_sin_nivel_por_capa():
    """optimize_model_memory sigue comprimiendo tras retirar compression_level.

    La función escribía config.compression_level = MAXIMUM para pedir
    "compresión máxima", pero esa intención nunca aterrizaba: el core sella
    el nivel del MnemeConfig del ZSpace global, no el de la config por capa.
    Su palanca viva es target_ratio, que sí viaja a register().
    """
    torch.manual_seed(11)
    modelo = nn.Sequential(nn.Linear(120, 100), nn.ReLU(), nn.Linear(100, 10))
    config = CompressionConfig(target_ratio=0.5)

    optimizado = mneme_torch.optimize_model_memory(
        modelo, target_memory_mb=0.01, config=config
    )

    assert config.target_ratio <= 0.05
    assert any(isinstance(m, ZLinear) for m in optimizado.modules())


# --------------------------------------------------------------------------
# Hallazgo residual 2026-08-15 — enable_structured_sparsity se aceptaba sin efecto
# --------------------------------------------------------------------------

def test_sparsity_2_4_aplica_el_patron_al_cuantizar():
    """register() con enable_structured_sparsity aplica el pre-pass 2:4 real.

    Antes del arreglo, el core recibía el flag en **kwargs y no lo consumía:
    compress_model_calibrated prometía sparsity 2:4 que nunca ocurría (0% de
    ceros). Ahora cada grupo de 4 conserva a lo sumo 2 valores no nulos.
    """
    torch.manual_seed(7)
    tensor = torch.randn(64, 64)
    espacio = ZSpace()
    espacio.register("sparsity_2_4", tensor, quantization_type="int8",
                     group_size=128, enable_structured_sparsity=True)
    cargado = espacio.load("sparsity_2_4").cpu()

    assert cargado.shape == tensor.shape
    grupos = cargado.reshape(-1, 4)
    assert bool(((grupos != 0).sum(dim=1) <= 2).all())
    assert float((cargado == 0).float().mean()) >= 0.5


def test_sparsity_2_4_roundtrip_reconstruye_con_la_mascara():
    """La máscara viaja en el payload y restaura el patrón exacto al cargar."""
    torch.manual_seed(8)
    tensor = torch.randn(32, 128)
    esperado, mascara = StructuredSparsifier.apply_2_4_sparsity(tensor)

    espacio = ZSpace()
    espacio.register("sparsity_mascara", tensor, quantization_type="int8",
                     group_size=128, enable_structured_sparsity=True)
    cargado = espacio.load("sparsity_mascara").cpu()

    assert torch.all(cargado[~mascara] == 0), "posiciones podadas no son cero exacto"
    error = float((cargado - esperado).abs().max())
    assert error <= 2 * _paso_de_cuantizacion(esperado)


def test_sparsity_sin_cuantizacion_se_rechaza():
    """enable_structured_sparsity sin quantization_type es ValidationError.

    El pre-pass está definido como sparsificar-luego-cuantizar: la máscara
    viaja en el payload cuantizado. Aceptarlo en las otras rutas (SVD/TT/RAW)
    sería volver a la promesa silenciosamente incumplida; se rechaza antes de
    la puerta del circuit breaker, como los typos de decomp_type.
    """
    espacio = ZSpace()
    with pytest.raises(ValidationError):
        espacio.register("sparsity_sin_quant", torch.randn(64, 64),
                         enable_structured_sparsity=True)


def test_compress_model_calibrated_entrega_sparsity_2_4():
    """La promesa original: config.enable_structured_sparsity aterriza en el peso.

    El peso registrado en ZSpace (lo que se sintetiza al cargar) debe tener el
    patrón 2:4; el ZParameter en memoria conserva el dato denso original.
    """
    torch.manual_seed(9)
    modelo = nn.Sequential(nn.Linear(128, 96))
    config = CompressionConfig(quantization_type="int8_group",
                               enable_structured_sparsity=True)

    comprimido = mneme_torch.compress_model_calibrated(modelo, config,
                                                       min_params=1000)

    z = next(m for m in comprimido.modules() if isinstance(m, ZLinear))
    assert z.weight._descriptor.decomp_type == DecompType.QUANTIZED
    peso = mneme_torch._zspace.load(z.weight._zspace_name).cpu()
    grupos = peso.reshape(-1, 4)
    assert bool(((grupos != 0).sum(dim=1) <= 2).all())
    assert float((peso == 0).float().mean()) >= 0.5


# --------------------------------------------------------------------------
# Hallazgo 2026-08-16 — rehidratación desde storage rota para todo tipo no-RAW
# --------------------------------------------------------------------------

def test_rehidratacion_de_un_tensor_cuantizado_en_otra_instancia(tmp_path):
    """register() en una instancia, load() en otra sobre el mismo almacén.

    Antes del arreglo, _load_from_storage cableaba el closure RAW
    (lz4 + safetensors) para todo decomp_type; el payload cuantizado es
    lz4 + msgpack, así que cualquier tensor QUANTIZED registrado en un
    proceso moría con IntegrityError en el siguiente.
    """
    from mneme.mneme_core import MnemeConfig

    torch.manual_seed(16)
    tensor = torch.randn(64, 64)
    ruta = str(tmp_path / "almacen")

    origen = ZSpace(MnemeConfig(storage_path=ruta))
    desc = origen.register("rehidratado_q", tensor,
                           quantization_type="int8", group_size=128)
    assert desc.decomp_type == DecompType.QUANTIZED
    sintetizado = origen.load("rehidratado_q").cpu()

    destino = ZSpace(MnemeConfig(storage_path=ruta))
    rehidratado = destino.load("rehidratado_q").cpu()

    assert rehidratado.shape == tensor.shape
    assert torch.equal(rehidratado, sintetizado), (
        "la rehidratación no reproduce la síntesis intra-proceso"
    )
    error = float((rehidratado - tensor).abs().max())
    assert error <= 2 * _paso_de_cuantizacion(tensor)


def test_rehidratacion_conserva_el_patron_2_4(tmp_path):
    """La máscara 2:4 del payload sobrevive al ciclo por storage.

    _dequantize_group_payload ya reaplica sparsity_mask; el defecto estaba
    antes: el closure RAW ni siquiera llegaba a decodificar el payload.
    """
    from mneme.mneme_core import MnemeConfig

    torch.manual_seed(17)
    tensor = torch.randn(32, 128)
    esperado, mascara = StructuredSparsifier.apply_2_4_sparsity(tensor)
    ruta = str(tmp_path / "almacen")

    origen = ZSpace(MnemeConfig(storage_path=ruta))
    origen.register("rehidratado_2_4", tensor, quantization_type="int8",
                    group_size=128, enable_structured_sparsity=True)

    destino = ZSpace(MnemeConfig(storage_path=ruta))
    rehidratado = destino.load("rehidratado_2_4").cpu()

    grupos = rehidratado.reshape(-1, 4)
    assert bool(((grupos != 0).sum(dim=1) <= 2).all())
    assert torch.all(rehidratado[~mascara] == 0), "posiciones podadas no son cero exacto"
    error = float((rehidratado - esperado).abs().max())
    assert error <= 2 * _paso_de_cuantizacion(esperado)


def test_rehidratacion_de_una_descomposicion_svd_en_otra_instancia(tmp_path):
    """El payload de componentes (SVD/TT/CP/Tucker) también debe rehidratarse.

    La instancia nueva reconstruye desde los mismos componentes que la que
    registró, así que ambas síntesis deben coincidir.
    """
    from mneme.mneme_core import MnemeConfig

    torch.manual_seed(18)
    tensor = torch.randn(120, 130)
    ruta = str(tmp_path / "almacen")

    origen = ZSpace(MnemeConfig(storage_path=ruta))
    desc = origen.register("rehidratado_svd", tensor, target_ratio=0.3)
    assert desc.decomp_type == DecompType.SVD
    sintetizado = origen.load("rehidratado_svd").cpu()

    destino = ZSpace(MnemeConfig(storage_path=ruta))
    rehidratado = destino.load("rehidratado_svd").cpu()

    assert rehidratado.shape == tensor.shape
    assert torch.allclose(rehidratado, sintetizado, atol=1e-5), (
        "la rehidratación no reproduce la reconstrucción intra-proceso"
    )
    error_rel = float(torch.norm(rehidratado - tensor) / torch.norm(tensor))
    assert error_rel < 0.9


# --------------------------------------------------------------------------
# Hallazgo 2026-08-16 — deuda residual de la rehidratación no-RAW
# --------------------------------------------------------------------------

def test_fallback_de_sintesis_conserva_el_dtype_original(tmp_path):
    """Sin el blob data_{name}, _do_synthesize reconstruye desde core_data.

    Esa rama de descomposición no casteaba al dtype de meta: una SVD
    registrada en float64 volvía float32 tras el reinicio, mientras que con
    el blob presente el closure de _load_from_storage sí lo restauraba.
    """
    from mneme.mneme_core import MnemeConfig

    torch.manual_seed(19)
    tensor = torch.randn(120, 130, dtype=torch.float64)
    ruta = str(tmp_path / "almacen")

    origen = ZSpace(MnemeConfig(storage_path=ruta))
    desc = origen.register("fallback_f64", tensor, target_ratio=0.3)
    assert desc.decomp_type == DecompType.SVD
    sintetizado = origen.load("fallback_f64").cpu()
    assert sintetizado.dtype == torch.float64

    assert origen.storage_backend.delete("data_fallback_f64")

    destino = ZSpace(MnemeConfig(storage_path=ruta))
    rehidratado = destino.load("fallback_f64").cpu()

    # Sin blob no hay lazy_tensor: la síntesis pasó por el fallback, no por
    # el closure de rehidratación.
    assert destino.name_to_desc["fallback_f64"].lazy_tensor is None

    assert rehidratado.dtype == torch.float64, (
        "el fallback de _do_synthesize no restaura el dtype de meta"
    )
    assert rehidratado.shape == tensor.shape
    assert torch.allclose(rehidratado, sintetizado, atol=1e-5)


def test_zaddr_estable_entre_registro_y_rehidratacion(tmp_path):
    """El mismo descriptor debe computar la misma ZAddr tras un reinicio.

    from_dict devolvía ranks como lista JSON y ZAddr.compute usa str(ranks):
    "(30,)" fresco vs "[30]" rehidratado computaban direcciones distintas
    (deriva contable en addr_to_desc y version_graph).
    """
    from mneme.mneme_core import MnemeConfig, ZAddr

    torch.manual_seed(20)
    tensor = torch.randn(120, 130)
    ruta = str(tmp_path / "almacen")

    origen = ZSpace(MnemeConfig(storage_path=ruta))
    desc_fresco = origen.register("addr_estable", tensor, target_ratio=0.3)
    assert desc_fresco.decomp_type == DecompType.SVD
    assert desc_fresco.ranks, "el test necesita un descriptor con ranks"

    destino = ZSpace(MnemeConfig(storage_path=ruta))
    destino.load("addr_estable")
    desc_rehidratado = destino.name_to_desc["addr_estable"]

    assert isinstance(desc_rehidratado.ranks, tuple)
    assert desc_rehidratado.ranks == desc_fresco.ranks
    assert ZAddr.compute(desc_fresco).addr == ZAddr.compute(desc_rehidratado).addr, (
        "la ZAddr del descriptor rehidratado no coincide con la del fresco"
    )
