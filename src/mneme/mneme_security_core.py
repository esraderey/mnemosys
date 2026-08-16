"""
MNEME Security Core: Módulo de seguridad y validación
Sistema de seguridad avanzado con validación de entrada y protección contra vulnerabilidades
"""

import hashlib
import hmac
import json
import logging
import os
import struct
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from safetensors.torch import (
    load as safetensors_load,
)
from safetensors.torch import (
    save as safetensors_save,
)

logger = logging.getLogger(__name__)

# --- Enums de Seguridad ---

class SecurityLevel(Enum):
    """Niveles de seguridad para serialización"""
    NONE = "none"
    HMAC = "hmac"
    ENCRYPTED = "encrypted"
    SIGNED = "signed"
    SAFETENSORS = "safetensors"

class TensorEncryptionMode(Enum):
    """Modos de cifrado para tensores"""
    AES_GCM = "aes_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    QUANTUM_SAFE = "quantum_safe"

class KeyRotationPolicy(Enum):
    """Políticas de rotación de claves"""
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    MANUAL = "manual"
    ADAPTIVE = "adaptive"

# --- Clases de Seguridad ---

@dataclass
class SecurityConfig:
    """Configuración de seguridad"""
    security_level: SecurityLevel = SecurityLevel.SAFETENSORS
    encryption_mode: TensorEncryptionMode = TensorEncryptionMode.AES_GCM
    key_rotation_policy: KeyRotationPolicy = KeyRotationPolicy.ADAPTIVE
    max_key_age_hours: int = 24
    max_key_usage: int = 1000
    enable_audit_logging: bool = True
    require_signatures: bool = True
    validate_inputs: bool = True
    max_tensor_size_mb: int = 1024
    max_batch_size: int = 1000
    # Clave HMAC con la que se firman y verifican los artefactos serializados.
    # Si es None se busca en MNEME_SIGNING_KEY y luego en MNEME_SECRET_KEY. Sin
    # ninguna de las tres, la firma se desactiva con un aviso: no se genera una
    # clave efímera, porque haría ilegible al reiniciar todo lo ya persistido.
    signing_key: bytes | None = None


# --- Marco de serialización ---
#
# Formato explícito con magia, versión y tipo. Antes el tipo se adivinaba con una
# cascada de try/except sin marcador, de modo que un artefacto de lote podía
# devolverse silenciosamente donde se esperaba un tensor.
#
#   magia   4B  b"MNEM"
#   version 1B
#   kind    1B  1=tensor, 2=lote
#   flags   1B  bit0 = lleva metadatos, bit1 = firmado
#   reserva 1B
#   len_meta   4B  big-endian
#   len_payload 8B big-endian
#   metadatos (json utf-8)
#   payload (safetensors)
#   [hmac-sha256 32B sobre todo lo anterior, si flags.bit1]
MNEME_MAGIC: bytes = b"MNEM"
MNEME_FRAME_VERSION: int = 1
KIND_TENSOR: int = 1
KIND_BATCH: int = 2
FLAG_HAS_META: int = 1
FLAG_SIGNED: int = 2
_HEADER_STRUCT = ">4sBBBBIQ"
_HEADER_SIZE = struct.calcsize(_HEADER_STRUCT)
_HMAC_SIZE = 32


class IntegrityError(ValueError):
    """El artefacto no supera la verificación de integridad."""


def _resolver_clave_firma(config: "SecurityConfig") -> bytes | None:
    """Clave de firma, o None si no hay ninguna ESTABLE entre procesos.

    Nunca se inventa una clave efímera: los artefactos firmados se persisten, así
    que firmar con una clave que muere con el proceso haría ilegible todo lo
    guardado en el siguiente arranque — y de forma silenciosa, que es peor.
    Sin clave estable no se firma, y se avisa.
    """
    if config.signing_key:
        clave = config.signing_key
        return clave if isinstance(clave, bytes) else str(clave).encode("utf-8")
    for variable in ("MNEME_SIGNING_KEY", "MNEME_SECRET_KEY"):
        valor = os.environ.get(variable)
        if valor:
            return valor.encode("utf-8")
    return None


class InputValidator:
    """Validador de entrada para prevenir ataques"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.max_tensor_size = config.max_tensor_size_mb * 1024 * 1024
        self.max_batch_size = config.max_batch_size

    def validate_tensor(self, tensor: torch.Tensor, name: str = None) -> bool:
        """Validar tensor de entrada"""
        try:
            # Verificar tipo
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")

            # Verificar tamaño
            tensor_size = tensor.numel() * tensor.element_size()
            if tensor_size > self.max_tensor_size:
                raise ValueError(f"Tensor too large: {tensor_size/1024/1024:.2f}MB > {self.max_tensor_size/1024/1024:.2f}MB")

            # Verificar valores finitos
            if not torch.isfinite(tensor).all():
                raise ValueError("Tensor contains non-finite values (inf or nan)")

            # Verificar rangos razonables
            if tensor.abs().max() > 1e6:
                logger.warning(f"Tensor {name} has very large values: {tensor.abs().max()}")

            return True

        except Exception as e:
            logger.error(f"Tensor validation failed: {e}")
            return False

    def validate_batch(self, tensors: list[torch.Tensor]) -> bool:
        """Validar lote de tensores"""
        try:
            if len(tensors) > self.max_batch_size:
                raise ValueError(f"Batch too large: {len(tensors)} > {self.max_batch_size}")

            for i, tensor in enumerate(tensors):
                if not self.validate_tensor(tensor, f"batch[{i}]"):
                    return False

            return True

        except Exception as e:
            logger.error(f"Batch validation failed: {e}")
            return False

    def validate_metadata(self, metadata: dict[str, Any]) -> bool:
        """Validar metadatos"""
        try:
            # Verificar tipos de claves
            for key, value in metadata.items():
                if not isinstance(key, str):
                    raise ValueError(f"Metadata key must be string, got {type(key)}")

                # Verificar tipos de valores permitidos
                if not isinstance(value, (str, int, float, bool, list, dict)):
                    raise ValueError(f"Metadata value type not allowed: {type(value)}")

            # Verificar tamaño total
            metadata_str = json.dumps(metadata)
            if len(metadata_str) > 1024 * 1024:  # 1MB max
                raise ValueError("Metadata too large")

            return True

        except Exception as e:
            logger.error(f"Metadata validation failed: {e}")
            return False

class SecureSerializer:
    """Serializador seguro usando solo safetensors"""

    _aviso_sin_clave_emitido = False

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.validator = InputValidator(config)
        self._clave_firma = _resolver_clave_firma(config)
        # La firma solo se activa si hay una clave estable con la que verificarla
        # después. Sin ella, firmar deja los artefactos ilegibles al reiniciar.
        self._firma_activa = bool(config.require_signatures and self._clave_firma)
        if (config.require_signatures and not self._clave_firma
                and not SecureSerializer._aviso_sin_clave_emitido):
            SecureSerializer._aviso_sin_clave_emitido = True
            warnings.warn(
                "require_signatures está activo pero no hay clave de firma: ni "
                "SecurityConfig.signing_key ni MNEME_SIGNING_KEY/MNEME_SECRET_KEY. La "
                "firma HMAC queda DESACTIVADA. Firmar con una clave generada al vuelo "
                "haría ilegible en el siguiente arranque todo lo que se persista ahora.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _firmar(self, cuerpo: bytes) -> bytes:
        return hmac.new(self._clave_firma, cuerpo, hashlib.sha256).digest()

    def _empaquetar(self, kind: int, payload: bytes, metadata: dict[str, Any] | None) -> bytes:
        meta_bytes = json.dumps(metadata).encode("utf-8") if metadata else b""
        flags = (FLAG_HAS_META if meta_bytes else 0)
        if self._firma_activa:
            flags |= FLAG_SIGNED
        cabecera = struct.pack(
            _HEADER_STRUCT, MNEME_MAGIC, MNEME_FRAME_VERSION, kind, flags, 0,
            len(meta_bytes), len(payload),
        )
        cuerpo = cabecera + meta_bytes + payload
        if flags & FLAG_SIGNED:
            cuerpo += self._firmar(cuerpo)
        return cuerpo

    def _desempaquetar(self, data: bytes, kind_esperado: int) -> tuple[bytes, dict[str, Any]]:
        """Valida marco, tipo e integridad ANTES de devolver nada al llamador."""
        if len(data) < _HEADER_SIZE:
            raise IntegrityError("artefacto demasiado corto para contener una cabecera MNEME")
        magia, version, kind, flags, _, len_meta, len_payload = struct.unpack(
            _HEADER_STRUCT, data[:_HEADER_SIZE]
        )
        if magia != MNEME_MAGIC:
            raise IntegrityError(
                "el artefacto no lleva la cabecera MNEME: fue escrito con un formato "
                "anterior sin marcador y no se puede validar su tipo ni su integridad"
            )
        if version != MNEME_FRAME_VERSION:
            raise IntegrityError(f"versión de marco {version}, se esperaba {MNEME_FRAME_VERSION}")
        if kind != kind_esperado:
            raise IntegrityError(
                f"el artefacto es de tipo {kind} y se esperaba {kind_esperado}: "
                "no se deduce el formato, se declara"
            )

        firmado = bool(flags & FLAG_SIGNED)
        if self._firma_activa and not firmado:
            raise IntegrityError(
                "se exige firma y el artefacto no viene firmado"
            )
        if firmado and not self._clave_firma:
            raise IntegrityError(
                "el artefacto viene firmado y no hay clave con la que verificarlo: "
                "configure MNEME_SIGNING_KEY o SecurityConfig.signing_key"
            )

        fin_cuerpo = len(data) - (_HMAC_SIZE if firmado else 0)
        if firmado:
            if fin_cuerpo < _HEADER_SIZE:
                raise IntegrityError("artefacto firmado truncado")
            esperado = self._firmar(data[:fin_cuerpo])
            if not hmac.compare_digest(esperado, data[fin_cuerpo:]):
                raise IntegrityError(
                    "la firma HMAC no coincide: el artefacto fue alterado o se firmó "
                    "con otra clave"
                )

        ini_meta = _HEADER_SIZE
        ini_payload = ini_meta + len_meta
        if ini_payload + len_payload != fin_cuerpo:
            raise IntegrityError("longitudes declaradas en la cabecera incoherentes")
        metadata = {}
        if flags & FLAG_HAS_META:
            metadata = json.loads(data[ini_meta:ini_payload].decode("utf-8"))
        return data[ini_payload:fin_cuerpo], metadata

    def serialize_tensor(self, tensor: torch.Tensor, metadata: dict[str, Any] = None) -> bytes:
        """Serializar tensor de forma segura"""
        if not self.validator.validate_tensor(tensor):
            raise ValueError("Tensor validation failed")
        if metadata and not self.validator.validate_metadata(metadata):
            raise ValueError("Metadata validation failed")
        # safetensors sobre bytes: sin archivo temporal no hay ventana TOCTOU en la
        # que otro proceso local pueda sustituir el contenido entre escribir y releer.
        payload = safetensors_save({"tensor": tensor.cpu()})
        return self._empaquetar(KIND_TENSOR, payload, metadata)

    def deserialize_tensor(self, data: bytes, device: torch.device = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """Deserializar tensor de forma segura"""
        device = device or torch.device('cpu')
        payload, metadata = self._desempaquetar(data, KIND_TENSOR)
        loaded_data = safetensors_load(payload)
        if "tensor" not in loaded_data:
            raise ValueError("No tensor found in serialized data")
        tensor = loaded_data["tensor"].to(device)
        if not self.validator.validate_tensor(tensor):
            raise ValueError("Deserialized tensor validation failed")
        return tensor, metadata

    def serialize_batch(self, tensors: list[torch.Tensor], metadata: dict[str, Any] = None) -> bytes:
        """Serializar lote de tensores de forma segura"""
        if not self.validator.validate_batch(tensors):
            raise ValueError("Batch validation failed")
        # Un solo blob safetensors con todos los tensores: la variante anterior
        # serializaba cada uno y los envolvía en JSON como listas de enteros, lo que
        # multiplicaba por varias veces el tamaño y no llevaba marcador de formato.
        payload = safetensors_save({f"tensor_{i}": t.cpu() for i, t in enumerate(tensors)})
        meta = dict(metadata or {})
        meta["_count"] = len(tensors)
        return self._empaquetar(KIND_BATCH, payload, meta)

    def deserialize_batch(self, data: bytes, device: torch.device = None) -> tuple[list[torch.Tensor], dict[str, Any]]:
        """Deserializar lote de tensores de forma segura"""
        device = device or torch.device('cpu')
        payload, metadata = self._desempaquetar(data, KIND_BATCH)
        loaded_data = safetensors_load(payload)
        count = int(metadata.pop("_count", len(loaded_data)))
        tensors = [loaded_data[f"tensor_{i}"].to(device) for i in range(count)]
        if not self.validator.validate_batch(tensors):
            raise ValueError("Deserialized batch validation failed")
        return tensors, metadata

class SecurityManager:
    """Gestor de seguridad centralizado"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.serializer = SecureSerializer(config)
        self.validator = InputValidator(config)
        self.audit_log = []
        self.security_violations = 0

    def secure_serialize(self, data: Any, metadata: dict[str, Any] = None) -> bytes:
        """Serialización segura con validación"""
        try:
            if isinstance(data, torch.Tensor):
                return self.serializer.serialize_tensor(data, metadata)
            elif isinstance(data, list) and all(isinstance(x, torch.Tensor) for x in data):
                return self.serializer.serialize_batch(data, metadata)
            else:
                raise ValueError(f"Unsupported data type for secure serialization: {type(data)}")

        except Exception as e:
            self.security_violations += 1
            self._log_security_event("serialization_failed", str(e))
            raise

    def secure_deserialize(self, data: bytes, device: torch.device = None) -> tuple[Any, dict[str, Any]]:
        """Deserialización segura con validación"""
        try:
            # El tipo se lee de la cabecera, no se adivina probando parsers: la cascada
            # anterior devolvía un lote en silencio donde se esperaba un tensor.
            if len(data) < _HEADER_SIZE or data[:4] != MNEME_MAGIC:
                raise IntegrityError(
                    "el artefacto no lleva cabecera MNEME: no se puede determinar su "
                    "tipo ni verificar su integridad"
                )
            kind = data[5]
            if kind == KIND_TENSOR:
                return self.serializer.deserialize_tensor(data, device)
            if kind == KIND_BATCH:
                return self.serializer.deserialize_batch(data, device)
            raise IntegrityError(f"tipo de artefacto desconocido: {kind}")

        except Exception as e:
            self.security_violations += 1
            self._log_security_event("deserialization_failed", str(e))
            raise

    def _log_security_event(self, event_type: str, details: str):
        """Registrar evento de seguridad"""
        if self.config.enable_audit_logging:
            event = {
                "timestamp": time.time(),
                "event_type": event_type,
                "details": details,
                "violations": self.security_violations
            }
            self.audit_log.append(event)
            logger.warning(f"Security event: {event_type} - {details}")

    def get_security_stats(self) -> dict[str, Any]:
        """Obtener estadísticas de seguridad"""
        return {
            "security_violations": self.security_violations,
            "audit_events": len(self.audit_log),
            "config": {
                "security_level": self.config.security_level.value,
                "validate_inputs": self.config.validate_inputs,
                "max_tensor_size_mb": self.config.max_tensor_size_mb
            }
        }

    def validate_input(self, data: Any, data_type: str = "tensor") -> bool:
        """Validar entrada de datos"""
        try:
            if data_type == "tensor" and isinstance(data, torch.Tensor):
                return self.validator.validate_tensor(data)
            elif data_type == "batch" and isinstance(data, list):
                return self.validator.validate_batch(data)
            elif data_type == "metadata" and isinstance(data, dict):
                return self.validator.validate_metadata(data)
            else:
                return False

        except Exception as e:
            self._log_security_event("validation_failed", str(e))
            return False

# --- Funciones de Utilidad de Seguridad ---

def create_secure_config(security_level: SecurityLevel = SecurityLevel.SAFETENSORS,
                         signing_key: bytes | None = None) -> SecurityConfig:
    """Crear configuración de seguridad segura.

    `signing_key` permite propagar una clave pasada por código (p. ej.
    `MnemeConfig.secret_key`); con None se mantiene la resolución por entorno.
    """
    return SecurityConfig(
        security_level=security_level,
        validate_inputs=True,
        enable_audit_logging=True,
        require_signatures=True,
        signing_key=signing_key,
    )

def validate_tensor_safe(tensor: torch.Tensor, max_size_mb: int = 1024) -> bool:
    """Validar tensor de forma segura"""
    try:
        validator = InputValidator(SecurityConfig(max_tensor_size_mb=max_size_mb))
        return validator.validate_tensor(tensor)
    except Exception:
        return False

def secure_tensor_serialize(tensor: torch.Tensor, metadata: dict[str, Any] = None) -> bytes:
    """Serializar tensor de forma segura"""
    config = create_secure_config()
    serializer = SecureSerializer(config)
    return serializer.serialize_tensor(tensor, metadata)

def secure_tensor_deserialize(data: bytes, device: torch.device = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Deserializar tensor de forma segura"""
    config = create_secure_config()
    serializer = SecureSerializer(config)
    return serializer.deserialize_tensor(data, device)
