"""Capas Z con síntesis perezosa: la compresión llega por fin a la inferencia.

Este módulo cierra la brecha medida el 16-ago-2026 (batería E6c): las capas Z
clásicas registran el peso comprimido en ZSpace pero su forward usa el tensor
original intacto. ``ZLinearTurbo`` invierte el contrato: el peso denso NUNCA
queda residente en el módulo — lo residente es la forma comprimida, y el
forward la consume de verdad.

Rutas por tipo de descriptor (pesos 2-D de ``nn.Linear``):

- ``svd``: ni siquiera se materializa el peso. Con ``W = U·diag(S)·Vᵀ`` el
  forward es ``F.linear(F.linear(x, B), A, bias)`` donde ``A = U·diag(S)`` y
  ``B = Vᵀ`` quedan como buffers. Memoria residente ``r·(m+n)`` en lugar de
  ``m·n`` y dos matmuls chicos en lugar de uno grande.
- ``quantized``: el payload INT4/INT8 (con su máscara 2:4 si la hay) queda
  residente como buffer uint8; el peso se decuantiza DENTRO del forward y se
  libera al salir. El backward lo re-sintetiza (recompute) en vez de
  retenerlo — es el intercambio memoria↔cómputo del gradient checkpointing.
- ``raw`` (y cualquier otro tipo): mismo esquema de recompute sobre los bytes
  comprimidos del descriptor.

Los pesos van congelados (no son ``nn.Parameter``); el sesgo sigue siendo un
parámetro entrenable y el gradiente fluye hacia la entrada, así que el modelo
turbo sirve para inferencia y para seguir entrenando capas vecinas.
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Callable

import lz4.frame
import msgpack
import torch
import torch.nn.functional as F
from torch import nn

from .mneme_core import TensorDecomposer, _dequantize_group_payload
from .mneme_torch import CompressionConfig, _zspace

__all__ = ["ZLinearTurbo", "compress_model_turbo"]


def _payload_a_buffer(core_data: bytes) -> torch.Tensor:
    """Bytes del descriptor como buffer uint8: viaja en state_dict y .to()."""
    return torch.frombuffer(bytearray(core_data), dtype=torch.uint8).clone()


class _RecomputeLinear(torch.autograd.Function):
    """F.linear cuyo peso se sintetiza en forward y se RE-sintetiza en backward.

    El peso jamás queda guardado en ``ctx``: solo la entrada. El coste extra es
    una síntesis adicional por backward; la ganancia es que entre forward y
    backward no hay ningún peso denso vivo.
    """

    @staticmethod
    def forward(ctx, entrada: torch.Tensor, bias: torch.Tensor | None,
                synth: Callable[[], torch.Tensor]):
        peso = synth()
        salida = F.linear(entrada, peso, bias)
        del peso
        ctx.save_for_backward(entrada)
        ctx.synth = synth
        ctx.con_bias = bias is not None
        return salida

    @staticmethod
    def backward(ctx, grad_salida: torch.Tensor):
        (entrada,) = ctx.saved_tensors
        grad_entrada = grad_bias = None
        if ctx.needs_input_grad[0]:
            peso = ctx.synth()
            grad_entrada = grad_salida @ peso
            del peso
        if ctx.con_bias and ctx.needs_input_grad[1]:
            grad_bias = grad_salida.reshape(-1, grad_salida.shape[-1]).sum(0)
        return grad_entrada, grad_bias, None


class ZLinearTurbo(nn.Module):
    """Capa lineal cuyo peso residente es la forma comprimida del descriptor."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.decomp_type = "sin_inicializar"
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Marcador de device/dtype: .to() lo mueve y el synth lo consulta.
        self.register_buffer("_marca", torch.empty(0))
        self._decodificar: Callable[[bytes], torch.Tensor] | None = None

    # ------------------------------------------------------------------
    @classmethod
    def from_linear(cls, linear: nn.Linear,
                    config: CompressionConfig | None = None,
                    zspace=None,
                    name: str | None = None,
                    register_kwargs: dict | None = None) -> ZLinearTurbo:
        """Construir desde un ``nn.Linear``: registra el peso y extrae lo residente.

        De ``config`` se derivan SOLO ``target_ratio`` y ``decomp_type``: el
        default ``quantization_type="int8"`` de CompressionConfig pertenece al
        flujo calibrado y aquí no se hereda implícitamente. La ruta cuantizada
        se pide explícita vía ``register_kwargs`` (p. ej.
        ``{"quantization_type": "int8", "group_size": 128}``), que se fusiona
        al final y llega tal cual a ``zspace.register``.
        """
        config = config or CompressionConfig()
        zspace = zspace if zspace is not None else _zspace
        name = name or f"turbo_{uuid.uuid4().hex}"

        kwargs = {"target_ratio": config.target_ratio}
        if config.decomp_type is not None:
            kwargs["decomp_type"] = config.decomp_type
        if register_kwargs:
            kwargs.update(register_kwargs)

        # El descriptor se calcula en float32 cuando el peso llega en media
        # precisión (SVD/cuantización no operan en half); el dtype de USO de
        # la capa se restaura al final con .to().
        dtype_origen = linear.weight.dtype
        device_origen = linear.weight.device
        peso = linear.weight.detach()
        if dtype_origen in (torch.float16, torch.bfloat16):
            peso = peso.float()

        desc = zspace.register(name, peso.cpu(), **kwargs)

        capa = cls(linear.in_features, linear.out_features,
                   bias=linear.bias is not None)
        if linear.bias is not None:
            with torch.no_grad():
                capa.bias.copy_(linear.bias.detach().float())
        capa.decomp_type = desc.decomp_type.value
        capa._instalar_descriptor(desc, zspace)
        # Heredar device y dtype del Linear de origen (B2 del G4): los buffers
        # flotantes y el bias se castean; los payloads uint8 quedan intactos
        # (Module.to solo castea dtype en tensores de punto flotante).
        return capa.to(device=device_origen, dtype=dtype_origen)

    # ------------------------------------------------------------------
    def _instalar_descriptor(self, desc, zspace) -> None:
        tipo = desc.decomp_type.value
        if tipo == "svd":
            comps = zspace._deserialize_components(
                lz4.frame.decompress(desc.core_data))
            u, s, v = comps["U"], comps["S"], comps["V"]
            # W = U·diag(S)·Vᵀ = A·B, con A = U·diag(S) y B = Vᵀ.
            self.register_buffer("_factor_a", (u * s.unsqueeze(0)).contiguous())
            self.register_buffer("_factor_b", v.T.contiguous())
        else:
            self.register_buffer("_payload_q", _payload_a_buffer(desc.core_data))
            if tipo != "quantized":
                self._instalar_decodificador(zspace)

    def _instalar_decodificador(self, zspace=None) -> None:
        """Closure de decodificación para las rutas que no son svd/quantized.

        RAW viaja en el marco MNEM del serializer; TT/CP/TUCKER/SPARSE viajan
        como componentes serializados y se reconstruyen con TensorDecomposer
        (B1 del G4: antes caían todos en el camino RAW y reventaban con
        IntegrityError). Tras un load_state_dict el closure se reconstruye
        contra el ZSpace global.
        """
        zspace = zspace if zspace is not None else _zspace
        if self.decomp_type == "raw":
            deserializar = zspace.security_manager.serializer.deserialize_tensor

            def _decodificar(data: bytes) -> torch.Tensor:
                tensor, _meta = deserializar(lz4.frame.decompress(data))
                return tensor
        else:
            deserializar_comps = zspace._deserialize_components

            def _decodificar(data: bytes) -> torch.Tensor:
                comps = deserializar_comps(lz4.frame.decompress(data))
                return TensorDecomposer.reconstruct(comps)

        self._decodificar = _decodificar

    # ------------------------------------------------------------------
    def _synth(self) -> torch.Tensor:
        """Sintetizar el peso denso en el device/dtype del módulo (temporal)."""
        # El buffer sigue al módulo (.cuda() lo mueve); la decodificación es
        # siempre sobre una vista CPU de los bytes.
        data = self._payload_q.detach().cpu().numpy().tobytes()
        if self.decomp_type == "quantized":
            info = msgpack.unpackb(lz4.frame.decompress(data), raw=False)
            peso = _dequantize_group_payload(info)
        else:
            peso = self._decodificar(data)
        return peso.to(device=self._marca.device, dtype=self._marca.dtype)

    def forward(self, entrada: torch.Tensor) -> torch.Tensor:
        if self.decomp_type == "svd":
            oculto = F.linear(entrada, self._factor_b)
            return F.linear(oculto, self._factor_a, self.bias)
        return _RecomputeLinear.apply(entrada, self.bias, self._synth)

    # ------------------------------------------------------------------
    def materialize_weight(self) -> torch.Tensor:
        """Peso denso reconstruido (CARO: solo para inspección y tests)."""
        if self.decomp_type == "svd":
            return self._factor_a @ self._factor_b
        return self._synth()

    def memoria_residente_bytes(self) -> int:
        """Bytes residentes que sustituyen al peso denso."""
        total = 0
        for nombre, buf in self.named_buffers(recurse=False):
            if nombre != "_marca":
                total += buf.numel() * buf.element_size()
        return total

    # ------------------------------------------------------- state_dict (M1)
    def get_extra_state(self) -> dict:
        return {"decomp_type": self.decomp_type,
                "dtype": str(self._marca.dtype),
                "device": str(self._marca.device)}

    def set_extra_state(self, state: dict) -> None:
        self.decomp_type = state["decomp_type"]
        if self.decomp_type not in ("svd", "quantized", "sin_inicializar"):
            # El closure de decodificación no viaja en el state_dict; se
            # reconstruye contra el ZSpace global de la sesión actual.
            self._instalar_decodificador()
        # _marca y bias existen desde __init__ (float32/CPU) y el copy_ del
        # load estándar conserva el dtype/device del DESTINO, no el del
        # checkpoint: normalizar el módulo entero al dtype/device de origen
        # (hallazgo del G4 sobre el cruce state_dict x modelos half/cuda).
        dtype = getattr(torch, state["dtype"].removeprefix("torch."))
        device = torch.device(state["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        self.to(device=device, dtype=dtype)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Los buffers de la ruta se crean en from_linear, así que una
        # instancia recién construida no los tiene: registrarlos con la forma
        # del state_dict entrante para que super() pueda copiarlos.
        for nombre in ("_factor_a", "_factor_b", "_payload_q"):
            clave = prefix + nombre
            if clave in state_dict and not hasattr(self, nombre):
                self.register_buffer(nombre,
                                     torch.empty_like(state_dict[clave]))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, decomp={self.decomp_type}")


def compress_model_turbo(model: nn.Module,
                         config: CompressionConfig | None = None,
                         min_params: int = 10000,
                         exclude_layers: list[str] | None = None) -> nn.Module:
    """Copia del modelo con cada ``nn.Linear`` grande sustituido por ZLinearTurbo.

    A diferencia de ``compress_model``, aquí la compresión SÍ gobierna la
    inferencia: las salidas del modelo devuelto reflejan la pérdida real de la
    ruta elegida y la memoria residente de los pesos sustituidos es la forma
    comprimida.

    Limitación conocida: los pesos COMPARTIDOS (weight tying, p. ej.
    embedding/lm_head atados) se comprimen por separado en cada ocurrencia y
    dejan de estar atados; excluirlos vía ``exclude_layers`` si debe
    preservarse el aliasing.
    """
    config = config or CompressionConfig()
    exclude_layers = exclude_layers or []
    modelo = copy.deepcopy(model)

    def _reemplazar(contenedor: nn.Module, prefijo: str = "") -> None:
        for nombre, hijo in list(contenedor.named_children()):
            ruta = f"{prefijo}.{nombre}" if prefijo else nombre
            if isinstance(hijo, nn.Linear):
                if ruta in exclude_layers or hijo.weight.numel() < min_params:
                    continue
                setattr(contenedor, nombre,
                        ZLinearTurbo.from_linear(
                            hijo, config=config,
                            name=f"turbo_{ruta}_{uuid.uuid4().hex[:8]}"))
            else:
                _reemplazar(hijo, ruta)

    _reemplazar(modelo)
    return modelo
