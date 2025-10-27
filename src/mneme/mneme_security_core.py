"""
MNEME Security Core: Módulo de seguridad y validación
Sistema de seguridad avanzado con validación de entrada y protección contra vulnerabilidades
"""

import hashlib
import hmac
import time
import logging
import tempfile
import os
from typing import Any, Dict, Optional, Union, List, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file, load_file
import io
import json
import struct
import xxhash
import lz4.frame
import warnings

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
    
    def validate_batch(self, tensors: List[torch.Tensor]) -> bool:
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
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
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
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.validator = InputValidator(config)
    
    def serialize_tensor(self, tensor: torch.Tensor, metadata: Dict[str, Any] = None) -> bytes:
        """Serializar tensor de forma segura"""
        try:
            # Validar entrada
            if not self.validator.validate_tensor(tensor):
                raise ValueError("Tensor validation failed")
            
            if metadata and not self.validator.validate_metadata(metadata):
                raise ValueError("Metadata validation failed")
            
            # Usar safetensors para serialización segura
            # SafeTensors necesita un archivo temporal para serializar
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.close()  # Cerrar el archivo para que SafeTensors pueda usarlo
            
            try:
                save_file({"tensor": tensor}, temp_file.name)
                with open(temp_file.name, 'rb') as f:
                    tensor_data = f.read()
            finally:
                # Asegurar que el archivo se elimine
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            
            # Agregar metadatos si es necesario
            if metadata:
                metadata_data = json.dumps(metadata).encode('utf-8')
                # Combinar datos con metadatos
                combined_data = struct.pack('>I', len(metadata_data)) + metadata_data + tensor_data
                return combined_data
            
            return tensor_data
            
        except Exception as e:
            logger.error(f"Secure serialization failed: {e}")
            raise
    
    def deserialize_tensor(self, data: bytes, device: torch.device = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Deserializar tensor de forma segura"""
        try:
            device = device or torch.device('cpu')
            
            # Verificar si hay metadatos
            if len(data) > 4:
                metadata_size = struct.unpack('>I', data[:4])[0]
                if metadata_size > 0 and metadata_size < len(data) - 4:
                    metadata_data = data[4:4+metadata_size]
                    tensor_data = data[4+metadata_size:]
                    metadata = json.loads(metadata_data.decode('utf-8'))
                else:
                    tensor_data = data
                    metadata = {}
            else:
                tensor_data = data
                metadata = {}
            
            # Deserializar con safetensors
            # SafeTensors necesita un archivo temporal para deserializar
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.close()
            
            try:
                with open(temp_file.name, 'wb') as f:
                    f.write(tensor_data)
                # Convertir device a string si es necesario
                device_str = str(device) if device else "cpu"
                if device_str == "cpu":
                    device_str = None  # SafeTensors usa None para CPU
                loaded_data = load_file(temp_file.name, device=device_str)
            finally:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            
            if "tensor" not in loaded_data:
                raise ValueError("No tensor found in serialized data")
            
            tensor = loaded_data["tensor"].to(device)
            
            # Validar tensor deserializado
            if not self.validator.validate_tensor(tensor):
                raise ValueError("Deserialized tensor validation failed")
            
            return tensor, metadata
            
        except Exception as e:
            logger.error(f"Secure deserialization failed: {e}")
            raise
    
    def serialize_batch(self, tensors: List[torch.Tensor], metadata: Dict[str, Any] = None) -> bytes:
        """Serializar lote de tensores de forma segura"""
        try:
            # Validar lote
            if not self.validator.validate_batch(tensors):
                raise ValueError("Batch validation failed")
            
            # Serializar cada tensor
            serialized_tensors = {}
            for i, tensor in enumerate(tensors):
                buffer = io.BytesIO()
                save_file({f"tensor_{i}": tensor}, buffer)
                serialized_tensors[f"tensor_{i}"] = buffer.getvalue()
            
            # Combinar datos
            combined_data = {
                "tensors": serialized_tensors,
                "count": len(tensors),
                "metadata": metadata or {}
            }
            
            return json.dumps(combined_data).encode('utf-8')
            
        except Exception as e:
            logger.error(f"Batch serialization failed: {e}")
            raise
    
    def deserialize_batch(self, data: bytes, device: torch.device = None) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Deserializar lote de tensores de forma segura"""
        try:
            device = device or torch.device('cpu')
            
            # Deserializar datos combinados
            combined_data = json.loads(data.decode('utf-8'))
            tensors = []
            
            for i in range(combined_data["count"]):
                tensor_data = combined_data["tensors"][f"tensor_{i}"]
                buffer = io.BytesIO(tensor_data)
                loaded_data = load_file(buffer)
                tensor = loaded_data[f"tensor_{i}"].to(device)
                tensors.append(tensor)
            
            # Validar lote deserializado
            if not self.validator.validate_batch(tensors):
                raise ValueError("Deserialized batch validation failed")
            
            return tensors, combined_data.get("metadata", {})
            
        except Exception as e:
            logger.error(f"Batch deserialization failed: {e}")
            raise

class SecurityManager:
    """Gestor de seguridad centralizado"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.serializer = SecureSerializer(config)
        self.validator = InputValidator(config)
        self.audit_log = []
        self.security_violations = 0
    
    def secure_serialize(self, data: Any, metadata: Dict[str, Any] = None) -> bytes:
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
    
    def secure_deserialize(self, data: bytes, device: torch.device = None) -> Tuple[Any, Dict[str, Any]]:
        """Deserialización segura con validación"""
        try:
            # Intentar deserializar como tensor individual
            try:
                return self.serializer.deserialize_tensor(data, device)
            except:
                # Intentar deserializar como lote
                return self.serializer.deserialize_batch(data, device)
                
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
    
    def get_security_stats(self) -> Dict[str, Any]:
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

def create_secure_config(security_level: SecurityLevel = SecurityLevel.SAFETENSORS) -> SecurityConfig:
    """Crear configuración de seguridad segura"""
    return SecurityConfig(
        security_level=security_level,
        validate_inputs=True,
        enable_audit_logging=True,
        require_signatures=True
    )

def validate_tensor_safe(tensor: torch.Tensor, max_size_mb: int = 1024) -> bool:
    """Validar tensor de forma segura"""
    try:
        validator = InputValidator(SecurityConfig(max_tensor_size_mb=max_size_mb))
        return validator.validate_tensor(tensor)
    except:
        return False

def secure_tensor_serialize(tensor: torch.Tensor, metadata: Dict[str, Any] = None) -> bytes:
    """Serializar tensor de forma segura"""
    config = create_secure_config()
    serializer = SecureSerializer(config)
    return serializer.serialize_tensor(tensor, metadata)

def secure_tensor_deserialize(data: bytes, device: torch.device = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Deserializar tensor de forma segura"""
    config = create_secure_config()
    serializer = SecureSerializer(config)
    return serializer.deserialize_tensor(data, device)
