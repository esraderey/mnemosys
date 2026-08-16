"""
MNEME PyTorch Integration - Actualizado
Integración avanzada con PyTorch que utiliza las funcionalidades del core
"""

import copy
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mneme_core import DecompType, MnemeConfig, ZDescriptor, ZSpace

logger = logging.getLogger(__name__)

# Instancia global de MNEME con configuración optimizada
_config = MnemeConfig()
_config.enable_async_context = True
_config.max_concurrent_operations = 8
# No se reactiva enable_encryption aquí: MnemeConfig ya decidió en __post_init__ si
# hay una clave estable con la que cifrar (parámetro o MNEME_SECRET_KEY). Forzarlo a
# True después pedía cifrado sin clave, y el backend falla cerrado ante eso.
_zspace = ZSpace(_config)

@dataclass
class LayerPrecisionPolicy:
    """Assigns quantization precision per layer type for mixed-precision compression.

    Maps common transformer layer patterns to quantization types.
    Attention layers stay in higher precision (FP16), FFN layers use INT4_GROUP,
    embeddings use INT8_GROUP, LayerNorm stays in FP32.
    """
    attention_qkv: str = "fp16"
    attention_output: str = "fp16"
    ffn_layers: str = "int4_group"
    embedding: str = "int8_group"
    layer_norm: str = "none"
    default: str = "int8_group"
    custom_overrides: dict[str, str] = field(default_factory=dict)

    def get_precision_for_layer(self, layer_name: str, module: nn.Module) -> str:
        """Determine quantization type for a layer by name and type."""
        if layer_name in self.custom_overrides:
            return self.custom_overrides[layer_name]
        if isinstance(module, nn.LayerNorm):
            return self.layer_norm
        if isinstance(module, nn.Embedding):
            return self.embedding
        name_lower = layer_name.lower()
        if any(k in name_lower for k in ('q_proj', 'k_proj', 'v_proj', 'qkv', 'self_attn.q', 'self_attn.k', 'self_attn.v', 'attn_q', 'attn_k', 'attn_v', 'wq', 'wk', 'wv')):
            return self.attention_qkv
        if any(k in name_lower for k in ('o_proj', 'out_proj', 'attn.out', 'attn.c_proj', 'attn_out', 'wo')):
            return self.attention_output
        if any(k in name_lower for k in ('mlp', 'ffn', 'fc1', 'fc2', 'gate', 'up_proj', 'down_proj', 'c_fc', 'c_proj')):
            # Distinguish MLP c_proj from attention c_proj
            if 'attn' not in name_lower:
                return self.ffn_layers
        return self.default


@dataclass
class CompressionConfig:
    """Configuración de compresión para capas"""
    target_ratio: float = 0.1
    # El core valida decomp_type (coacciona strings, rechaza incoercibles con
    # ValidationError); None, ADAPTIVE, RAW y QUANTIZED no fuerzan nada: caen
    # al routing automático.
    decomp_type: DecompType | None = None
    # Advanced quantization fields
    group_size: int = 128
    quantization_type: str = "int8"
    calibration_samples: int = 128
    calibration_data: Any | None = None
    mixed_precision_policy: LayerPrecisionPolicy | None = None
    enable_kv_cache_compression: bool = False
    kv_cache_bits: int = 8
    enable_structured_sparsity: bool = False

class ZParameter(nn.Parameter):
    """Parámetro respaldado por síntesis MNEME con funcionalidades avanzadas"""

    def __new__(cls, data: torch.Tensor = None,
                descriptor: ZDescriptor = None,
                requires_grad: bool = True,
                config: CompressionConfig = None,
                zspace_name: str = None):
        if descriptor is not None and data is None:
            # Cargar desde descriptor usando nombre explícito o meta
            _name = zspace_name or descriptor.meta.get('_zspace_name', None)
            if _name:
                data = _zspace.load(_name)
            else:
                # Fallback: sintetizar desde lazy_tensor del descriptor
                if hasattr(descriptor, 'lazy_tensor') and descriptor.lazy_tensor:
                    data = descriptor.lazy_tensor.decompress()
                else:
                    data = _zspace._synthesize_tensor(descriptor)

        if data is None:
            raise ValueError("ZParameter requires either data or a valid descriptor")

        instance = super().__new__(cls, data, requires_grad)
        instance._descriptor = descriptor
        instance._zspace_name = zspace_name
        instance._config = config or CompressionConfig()
        instance._last_access = time.time()
        instance._access_count = 0
        return instance

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor,
                   name: str,
                   config: CompressionConfig = None,
                   requires_grad: bool = True):
        """Crear ZParameter desde tensor con compresion MNEME."""
        config = config or CompressionConfig()
        desc = _zspace.register(name, tensor,
                                target_ratio=config.target_ratio,
                                decomp_type=config.decomp_type)
        param = cls(data=tensor, descriptor=desc,
                    requires_grad=requires_grad, config=config,
                    zspace_name=name)
        return param

    def update_delta(self, delta_op: dict):
        """Aplicar actualización delta y obtener nueva versión"""
        if self._zspace_name:
            self._descriptor = _zspace.update(self._zspace_name, delta_op)
            # Recargar datos
            self.data = _zspace.load(self._zspace_name)
            self._last_access = time.time()
            self._access_count += 1

    def get_compression_stats(self) -> dict[str, Any]:
        """Obtener estadísticas de compresión"""
        if self._descriptor:
            return {
                "compression_ratio": self._descriptor.meta.get('compression_ratio', 1.0),
                "decomp_type": self._descriptor.decomp_type.value,
                "version": self._descriptor.version,
                "shape": self._descriptor.shape,
                "size_bytes": len(self._descriptor.core_data),
                "access_count": self._access_count,
                "last_access": self._last_access
            }
        return {}

    def encrypt_parameter(self, key_id: str = None) -> tuple[bytes, dict]:
        """Cifrar parámetro usando seguridad avanzada"""
        if _zspace.security_manager:
            # Usar safetensors para serialización segura
            from safetensors.torch import save
            serialized = save({"data": self.data})
            return _zspace.security_manager.encrypt_data(serialized, key_id)
        return b'', {}

    def decrypt_parameter(self, encrypted_data: bytes, metadata: dict) -> torch.Tensor:
        """Descifrar parámetro"""
        if _zspace.security_manager:
            decrypted_data = _zspace.security_manager.decrypt_data(encrypted_data, metadata)
            from safetensors.torch import load
            loaded_data = load(decrypted_data)
            return loaded_data["data"].to(self.data.device)
        return self.data

class ZLinear(nn.Module):
    """Capa lineal con pesos comprimidos por MNEME.

    LIMITACIÓN DE MEMORIA: ZLinear actualmente mantiene el tensor completo en
    ``self.weight.data`` (requerido por autograd) *además* de la copia comprimida
    almacenada en ZSpace.  Esto significa que el uso de memoria es **mayor** que
    un ``nn.Linear`` equivalente, no menor.

    Para lograr ahorro real de memoria sería necesario implementar un
    ``torch.autograd.Function`` personalizado que descomprima on-the-fly durante
    forward/backward y libere la copia descomprimida entre pasos.  Ese refactor
    queda pendiente para una sesión futura.

    Por ahora, ZLinear es útil como demostración del pipeline de compresión y
    para medir la calidad de reconstrucción, pero no como herramienta de ahorro
    de memoria en producción.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True,
                 config: CompressionConfig = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Configuración de compresión
        self.config = config or CompressionConfig()

        # Inicializar peso
        weight = torch.randn(out_features, in_features) / math.sqrt(in_features)

        # Registrar con MNEME
        name = f"linear_{id(self)}_weight"
        desc = _zspace.register(name, weight, target_ratio=self.config.target_ratio,
                                decomp_type=self.config.decomp_type)
        self.weight = ZParameter(
            data=weight, descriptor=desc, config=self.config, zspace_name=name
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Estadísticas
        self._forward_count = 0
        self._total_time = 0.0
        self._memory_usage = 0.0

    @classmethod
    def from_existing(cls, linear: nn.Linear, config: CompressionConfig = None):
        """Crear ZLinear desde nn.Linear existente, registrando pesos reales."""
        z = object.__new__(cls)
        nn.Module.__init__(z)
        z.in_features = linear.in_features
        z.out_features = linear.out_features
        z.config = config or CompressionConfig()

        # Registrar peso REAL en ZSpace (activa compresion inteligente)
        name = f"zlinear_{id(z)}_weight"
        desc = _zspace.register(name, linear.weight.data,
                                target_ratio=z.config.target_ratio,
                                decomp_type=z.config.decomp_type)
        z.weight = ZParameter(
            data=linear.weight.data, descriptor=desc,
            config=z.config, zspace_name=name,
        )

        # Copiar bias
        if linear.bias is not None:
            z.bias = nn.Parameter(linear.bias.data.clone())
        else:
            z.register_parameter("bias", None)

        z._forward_count = 0
        z._total_time = 0.0
        z._memory_usage = 0.0
        return z

    @classmethod
    def from_existing_calibrated(
        cls, linear: nn.Linear,
        config: CompressionConfig = None,
        quant_kwargs: dict[str, Any] = None,
    ):
        """Create ZLinear from nn.Linear with explicit quantization parameters.

        Passes ``quant_kwargs`` (quantization_type, group_size, gptq_metadata,
        enable_structured_sparsity) through to ``_zspace.register()``.
        """
        z = object.__new__(cls)
        nn.Module.__init__(z)
        z.in_features = linear.in_features
        z.out_features = linear.out_features
        z.config = config or CompressionConfig()

        name = f"zlinear_{id(z)}_weight"
        register_kwargs = {"target_ratio": z.config.target_ratio,
                           "decomp_type": z.config.decomp_type}
        if quant_kwargs:
            register_kwargs.update(quant_kwargs)

        desc = _zspace.register(name, linear.weight.data, **register_kwargs)
        z.weight = ZParameter(
            data=linear.weight.data, descriptor=desc,
            config=z.config, zspace_name=name,
        )

        if linear.bias is not None:
            z.bias = nn.Parameter(linear.bias.data.clone())
        else:
            z.register_parameter("bias", None)

        z._forward_count = 0
        z._total_time = 0.0
        z._memory_usage = 0.0
        return z

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        result = F.linear(input, self.weight, self.bias)
        return result

    def get_performance_stats(self) -> dict[str, Any]:
        """Obtener estadísticas de rendimiento"""
        avg_time = self._total_time / max(1, self._forward_count)
        compression_stats = self.weight.get_compression_stats()

        return {
            "forward_count": self._forward_count,
            "avg_forward_time": avg_time,
            "memory_usage_mb": self._memory_usage,
            "compression": compression_stats
        }

    def extra_repr(self) -> str:
        """Representación adicional con información de compresión"""
        compression_stats = self.weight.get_compression_stats()
        ratio = compression_stats.get('compression_ratio', 1.0)
        decomp_type = compression_stats.get('decomp_type', 'unknown')

        return (f'in_features={self.in_features}, '
               f'out_features={self.out_features}, '
               f'bias={self.bias is not None}, '
               f'compression={ratio:.3f}x, '
               f'type={decomp_type}')

class ZConv2d(nn.Module):
    """Convolución 2D con pesos comprimidos por MNEME"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 config: CompressionConfig = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.config = config or CompressionConfig()

        # Inicializar peso
        weight = torch.randn(out_channels, in_channels // groups,
                           kernel_size, kernel_size)

        # Registrar con MNEME
        name = f"conv2d_{id(self)}_weight"
        desc = _zspace.register(name, weight, target_ratio=self.config.target_ratio,
                                decomp_type=self.config.decomp_type)
        self.weight = ZParameter(
            data=weight, descriptor=desc, config=self.config, zspace_name=name
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        self.weight._last_access = time.time()
        self.weight._access_count += 1
        return F.conv2d(input, self.weight, self.bias,
                       self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        compression_stats = self.weight.get_compression_stats()
        ratio = compression_stats.get('compression_ratio', 1.0)

        return (f'in_channels={self.in_channels}, '
               f'out_channels={self.out_channels}, '
               f'kernel_size={self.kernel_size}, '
               f'compression={ratio:.3f}x')

class QuantizedKVCache:
    """Quantized Key-Value cache for transformer inference.

    Stores K/V tensors in INT8 with per-head per-token scaling to reduce
    memory from FP16/FP32 to INT8 (~50% reduction vs FP16, ~75% vs FP32).

    Usage::

        cache = QuantizedKVCache(num_heads=12, head_dim=64)
        # During inference:
        full_k, full_v = cache.update(new_K, new_V)
        # Use full_k, full_v for attention computation
    """

    def __init__(self, num_heads: int, head_dim: int,
                 max_seq_len: int = 4096, bits: int = 8,
                 device: torch.device = None):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.bits = bits
        self.device = device or torch.device('cpu')
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        self.k_scales: torch.Tensor | None = None
        self.v_scales: torch.Tensor | None = None
        self.seq_len = 0

    def update(
        self, key: torch.Tensor, value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add new K/V entries, quantize, return dequantized full cache.

        Args:
            key: (batch, heads, new_seq, dim) float
            value: (batch, heads, new_seq, dim) float
        Returns:
            (full_key, full_value) dequantized for attention computation.
        """
        # Per-head per-token symmetric INT8 quantization
        k_scale = key.abs().amax(dim=-1, keepdim=True) / 127.0
        k_scale = torch.where(k_scale == 0, torch.ones_like(k_scale), k_scale)
        k_q = (key / k_scale).round().clamp(-128, 127).to(torch.int8)

        v_scale = value.abs().amax(dim=-1, keepdim=True) / 127.0
        v_scale = torch.where(v_scale == 0, torch.ones_like(v_scale), v_scale)
        v_q = (value / v_scale).round().clamp(-128, 127).to(torch.int8)

        if self.k_cache is None:
            self.k_cache = k_q
            self.v_cache = v_q
            self.k_scales = k_scale.half()
            self.v_scales = v_scale.half()
        else:
            self.k_cache = torch.cat([self.k_cache, k_q], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_q], dim=2)
            self.k_scales = torch.cat([self.k_scales, k_scale.half()], dim=2)
            self.v_scales = torch.cat([self.v_scales, v_scale.half()], dim=2)

        self.seq_len += key.shape[2]

        # Dequantize for attention computation
        full_k = self.k_cache.float() * self.k_scales.float()
        full_v = self.v_cache.float() * self.v_scales.float()
        return full_k, full_v

    def reset(self):
        """Reset cache for new sequence."""
        self.k_cache = None
        self.v_cache = None
        self.k_scales = None
        self.v_scales = None
        self.seq_len = 0

    def get_memory_usage(self) -> dict[str, float]:
        """Compare memory usage vs FP16 cache."""
        if self.k_cache is None:
            return {"quantized_mb": 0, "fp16_equivalent_mb": 0, "savings_ratio": 0}
        q_bytes = self.k_cache.numel() + self.v_cache.numel()  # int8
        s_bytes = (self.k_scales.numel() + self.v_scales.numel()) * 2  # fp16
        total_q = q_bytes + s_bytes
        total_fp16 = (self.k_cache.numel() + self.v_cache.numel()) * 2
        return {
            "quantized_mb": total_q / (1024 * 1024),
            "fp16_equivalent_mb": total_fp16 / (1024 * 1024),
            "savings_ratio": total_fp16 / max(total_q, 1),
        }


class ZAttention(nn.Module):
    """Atención multi-cabeza con compresión MNEME y KV-cache cuantizado opcional"""

    def __init__(self, embed_dim: int, num_heads: int,
                 config: CompressionConfig = None,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "embed_dim debe ser divisible por num_heads"

        self.config = config or CompressionConfig()

        # Proyecciones Q, K, V con compresión
        self.q_proj = ZLinear(embed_dim, embed_dim, config=config)
        self.k_proj = ZLinear(embed_dim, embed_dim, config=config)
        self.v_proj = ZLinear(embed_dim, embed_dim, config=config)
        self.out_proj = ZLinear(embed_dim, embed_dim, config=config)

        self.dropout_layer = nn.Dropout(dropout)

        # Optional quantized KV-cache for inference
        self.kv_cache: QuantizedKVCache | None = None
        if self.config.enable_kv_cache_compression:
            self.kv_cache = QuantizedKVCache(
                num_heads=num_heads, head_dim=self.head_dim,
                bits=self.config.kv_cache_bits,
            )

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None,
                return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass de atención"""
        B, L, D = x.shape

        # Proyectar y redimensionar
        Q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # KV-cache quantization during inference
        if self.kv_cache is not None and not self.training:
            K, V = self.kv_cache.update(K, V)

        # Atención
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout_layer(attn)

        out = attn @ V

        # Redimensionar y proyectar
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)

        if return_attention:
            return out, attn
        return out

class ZTransformerBlock(nn.Module):
    """Bloque Transformer con compresión MNEME"""

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0,
                 config: CompressionConfig = None,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()
        self.embed_dim = embed_dim

        # Normalización y atención
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ZAttention(embed_dim, num_heads, config, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP con compresión
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            ZLinear(embed_dim, mlp_dim, config=config),
            self._get_activation(activation),
            nn.Dropout(dropout),
            ZLinear(mlp_dim, embed_dim, config=config)
        )

        self.dropout = nn.Dropout(dropout)

    def _get_activation(self, activation: str):
        """Obtener función de activación"""
        if activation == 'gelu':
            return nn.GELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'swish':
            return nn.SiLU()
        else:
            return nn.GELU()

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass del bloque Transformer"""
        # Self-attention
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_out)

        return x

def compress_model(model: nn.Module,
                  config: CompressionConfig = None,
                  min_params: int = 10000,
                  exclude_layers: list[str] = None) -> nn.Module:
    """Comprimir modelo existente reemplazando capas con versiones MNEME.

    Crea una copia profunda del modelo y reemplaza nn.Linear y nn.Conv2d
    con ZLinear/ZConv2d que registran los pesos reales en ZSpace con
    compresion inteligente (SVD truncado / INT8 / LZ4).
    """
    import copy

    config = config or CompressionConfig()
    exclude_layers = exclude_layers or []
    model_copy = copy.deepcopy(model)

    def replace_linear(module: nn.Module, prefix: str = ""):
        """Reemplazar capas Lineales con ZLinear usando pesos reales."""
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and not isinstance(child, ZLinear):
                if full_name not in exclude_layers and child.weight.numel() >= min_params:
                    z_linear = ZLinear.from_existing(child, config=config)
                    setattr(module, name, z_linear)
                    logger.info(f"Replaced '{full_name}' with ZLinear "
                                f"({child.in_features}->{child.out_features})")
                else:
                    replace_linear(child, full_name)
            else:
                replace_linear(child, full_name)

    replace_linear(model_copy)
    return model_copy


def compress_model_calibrated(
    model: nn.Module,
    config: CompressionConfig = None,
    min_params: int = 10000,
    exclude_layers: list[str] = None,
) -> nn.Module:
    """Compress model with calibrated quantization and mixed-precision policy.

    Supports:
    - GPTQ-calibrated quantization (requires ``config.calibration_data``)
    - Group-wise INT4/INT8 quantization
    - Mixed precision per-layer via ``config.mixed_precision_policy``
    - 2:4 structured sparsity via ``config.enable_structured_sparsity``

    Falls back to standard ``compress_model()`` if no advanced features are
    configured.

    Example::

        config = CompressionConfig(
            quantization_type='gptq_int4',
            group_size=128,
            calibration_data=my_dataloader,
            calibration_samples=128,
            mixed_precision_policy=LayerPrecisionPolicy(),
            enable_structured_sparsity=True,
        )
        compressed = compress_model_calibrated(model, config)
    """
    config = config or CompressionConfig()
    exclude_layers = exclude_layers or []

    # If no advanced features, fall back
    is_gptq = config.quantization_type in ('gptq_int4', 'gptq_int8')
    is_group = config.quantization_type in ('int4_group', 'int8_group')
    has_policy = config.mixed_precision_policy is not None
    if not (is_gptq or is_group or has_policy or config.enable_structured_sparsity):
        return compress_model(model, config, min_params, exclude_layers)

    model_copy = copy.deepcopy(model)

    # Step 1: GPTQ calibration if requested
    gptq_data: dict[str, tuple] = {}
    if is_gptq:
        if config.calibration_data is None:
            raise ValueError(
                "GPTQ quantization requires calibration_data in CompressionConfig"
            )
        from .mneme_optimization import GPTQCalibrator
        bits = 4 if config.quantization_type == 'gptq_int4' else 8
        calibrator = GPTQCalibrator(
            bits=bits, group_size=config.group_size,
        )
        hessians = calibrator.collect_hessian(
            model_copy, config.calibration_data, config.calibration_samples,
        )
        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Linear) and name in hessians:
                if name in exclude_layers or module.weight.numel() < min_params:
                    continue
                _, meta = calibrator.quantize_layer(module.weight.data, hessians[name])
                gptq_data[name] = meta
        logger.info(f"GPTQ calibration complete: {len(gptq_data)} layers quantized")

    # Step 2: Mixed precision policy
    policy = config.mixed_precision_policy or LayerPrecisionPolicy(
        default=config.quantization_type,
    )

    def replace_with_policy(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and not isinstance(child, ZLinear):
                if full_name in exclude_layers or child.weight.numel() < min_params:
                    replace_with_policy(child, full_name)
                    continue

                precision = policy.get_precision_for_layer(full_name, child)

                if precision == 'none':
                    replace_with_policy(child, full_name)
                    continue

                if precision == 'fp16':
                    # Keep as FP16 nn.Linear (no ZLinear wrapping needed)
                    child.weight.data = child.weight.data.half()
                    if child.bias is not None:
                        child.bias.data = child.bias.data.half()
                    replace_with_policy(child, full_name)
                    continue

                quant_kwargs: dict[str, Any] = {
                    'quantization_type': precision,
                    'group_size': config.group_size,
                    'enable_structured_sparsity': config.enable_structured_sparsity,
                }

                if full_name in gptq_data:
                    quant_kwargs['gptq_metadata'] = gptq_data[full_name]
                    quant_kwargs['quantization_type'] = config.quantization_type

                z_linear = ZLinear.from_existing_calibrated(
                    child, config=config, quant_kwargs=quant_kwargs,
                )
                setattr(module, name, z_linear)
                logger.info(
                    f"Replaced '{full_name}' with ZLinear "
                    f"({child.in_features}->{child.out_features}, {precision})"
                )
            else:
                replace_with_policy(child, full_name)

    replace_with_policy(model_copy)
    return model_copy


def get_compression_stats(model: nn.Module) -> dict[str, Any]:
    """Obtener estadísticas de compresión del modelo"""
    stats = {
        "original_params": 0,
        "compressed_params": 0,
        "layers": [],
        "compression_ratios": [],
        "total_layers": 0,
        "compressed_layers": 0
    }

    for name, module in model.named_modules():
        if isinstance(module, (ZLinear, ZConv2d, ZAttention)):
            if hasattr(module, 'weight') and hasattr(module.weight, 'get_compression_stats'):
                compression_stats = module.weight.get_compression_stats()
                original = compression_stats.get('size_bytes', 0) / compression_stats.get('compression_ratio', 1.0)
                compressed = compression_stats.get('size_bytes', 0)
                ratio = compression_stats.get('compression_ratio', 1.0)

                stats["original_params"] += original
                stats["compressed_params"] += compressed
                stats["compression_ratios"].append(ratio)
                stats["compressed_layers"] += 1

                stats["layers"].append({
                    "name": name,
                    "type": type(module).__name__,
                    "original_bytes": original,
                    "compressed_bytes": compressed,
                    "compression_ratio": ratio,
                    "decomp_type": compression_stats.get('decomp_type', 'unknown')
                })

        stats["total_layers"] += 1

    if stats["original_params"] > 0:
        stats["overall_ratio"] = stats["compressed_params"] / stats["original_params"]
        stats["avg_compression_ratio"] = sum(stats["compression_ratios"]) / len(stats["compression_ratios"]) if stats["compression_ratios"] else 1.0

    return stats

def get_model_performance_stats(model: nn.Module) -> dict[str, Any]:
    """Obtener estadísticas de rendimiento del modelo"""
    stats = {
        "total_forward_time": 0.0,
        "total_forward_count": 0,
        "layers": []
    }

    for name, module in model.named_modules():
        if isinstance(module, (ZLinear, ZConv2d, ZAttention)):
            if hasattr(module, 'get_performance_stats'):
                layer_stats = module.get_performance_stats()
                stats["total_forward_time"] += layer_stats.get("avg_forward_time", 0.0) * layer_stats.get("forward_count", 0)
                stats["total_forward_count"] += layer_stats.get("forward_count", 0)

                stats["layers"].append({
                    "name": name,
                    "type": type(module).__name__,
                    **layer_stats
                })

    if stats["total_forward_count"] > 0:
        stats["avg_forward_time"] = stats["total_forward_time"] / stats["total_forward_count"]

    return stats

def optimize_model_memory(model: nn.Module,
                         target_memory_mb: int = 100,
                         config: CompressionConfig = None) -> nn.Module:
    """Optimizar modelo para uso de memoria específico"""
    config = config or CompressionConfig()

    # Calcular compresión necesaria
    current_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    target_ratio = target_memory_mb / current_memory

    if target_ratio < 1.0:
        # Ajustar configuración para mayor compresión
        config.target_ratio = min(target_ratio * 0.8, 0.05)  # 5% mínimo

        # Comprimir modelo
        compressed_model = compress_model(model, config)

        # Verificar memoria resultante
        new_memory = sum(p.numel() * p.element_size() for p in compressed_model.parameters()) / (1024 * 1024)
        logger.info(f"Memory optimization: {current_memory:.1f}MB -> {new_memory:.1f}MB")

        return compressed_model

    return model

def get_system_metrics() -> dict[str, Any]:
    """Obtener métricas del sistema MNEME."""
    return _zspace.get_performance_metrics()

def get_health_status() -> str:
    """Obtener estado de salud del sistema MNEME."""
    return _zspace.get_health_status().value

def optimize_system() -> dict[str, Any]:
    """Optimizar sistema MNEME completo."""
    return _zspace.optimize_system()

# Alias para compatibilidad
MLinear = ZLinear
MConv2d = ZConv2d
MAttention = ZAttention
MTransformerBlock = ZTransformerBlock
