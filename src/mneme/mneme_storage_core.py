"""
MNEME Storage Core: Módulo de almacenamiento seguro
Sistema de almacenamiento con validación y protección contra vulnerabilidades
"""

import hashlib
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

import lz4.frame
import torch
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes as _crypto_hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .mneme_security_core import SecurityConfig, SecurityManager, create_secure_config

logger = logging.getLogger(__name__)


class StorageAuthenticationError(Exception):
    """El blob almacenado no supera la verificación de autenticidad AES-GCM.

    Se lanza en `retrieve()` cuando el ciphertext fue alterado o la clave de
    cifrado no corresponde. Nunca se traga en silencio ni se confunde con una
    clave inexistente (eso devuelve None).
    """


class StorageFormatError(Exception):
    """El blob almacenado no lleva el marcador de formato que el lector espera.

    Se lanza en vez de adivinar: adivinar el formato es exactamente lo que hacía
    que un blob ya comprimido por el llamador se descomprimiera de más y se
    devolviera el contenido equivocado.
    """


# --- Formato del sobre cifrado en disco ---
#
# marcador  4B  b"MNSE"  (MNeme Storage Encrypted) — distingue un blob cifrado de
#                        uno sin cifrar por marcador, no por adivinanza.
# version   1B
# nonce    12B  aleatorio por operación (nunca fijo ni reutilizado)
# ciphertext  resto  (AES-256-GCM: incluye el tag de 16 bytes al final, como
#                     produce `AESGCM.encrypt`)
_ENC_MAGIC: bytes = b"MNSE"
_ENC_VERSION: int = 1
_ENC_NONCE_SIZE: int = 12
_ENC_HEADER_SIZE: int = len(_ENC_MAGIC) + 1 + _ENC_NONCE_SIZE
# `info` fijo del proyecto para HKDF: separa por dominio la clave de cifrado de
# almacenamiento de cualquier otro uso que se le dé a la misma secret_key.
_ENC_HKDF_INFO: bytes = b"mneme-storage-core:aes-256-gcm:v1"

# --- Enums de Almacenamiento ---

class StorageTier(Enum):
    """Niveles de almacenamiento"""
    MEMORY = "memory"
    SSD = "ssd"
    HDD = "hdd"
    ARCHIVE = "archive"

class CachePolicy(Enum):
    """Políticas de cache"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    LIFO = "lifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

class CompressionLevel(Enum):
    """Niveles de compresión"""
    ULTRA_FAST = 1
    FAST = 3
    BALANCED = 6
    HIGH = 9
    MAXIMUM = 12

# --- Clases de Almacenamiento ---

@dataclass
class StorageConfig:
    """Configuración de almacenamiento"""
    cache_size_mb: int = 1024
    cache_policy: CachePolicy = CachePolicy.ADAPTIVE
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    enable_compression: bool = True
    enable_deduplication: bool = True
    max_file_size_mb: int = 1024
    enable_encryption: bool = True
    # Material del que se deriva (vía HKDF) la clave AES-256-GCM de cifrado en
    # reposo. Requerido si enable_encryption es True: sin él, el backend falla
    # explícitamente al construirse en vez de almacenar en claro. repr=False:
    # la clave no debe salir por logs que impriman la config.
    secret_key: bytes | None = field(default=None, repr=False)
    storage_path: str = "./mneme_storage"
    backup_enabled: bool = True
    validate_integrity: bool = True

class SecureStorageBackend:
    """Backend de almacenamiento seguro"""

    def __init__(self, config: StorageConfig, security_config: SecurityConfig = None):
        self.config = config
        self.security_config = security_config or create_secure_config()
        self.security_manager = SecurityManager(self.security_config)
        self.lock = Lock()
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Clave AEAD derivada de config.secret_key. Fallo explícito aquí, en la
        # construcción, si enable_encryption está activo y no hay secret_key
        # utilizable: nunca se cae en claro de forma silenciosa.
        self._aead_key: bytes | None = None
        if self.config.enable_encryption:
            secret_key = self.config.secret_key
            if not isinstance(secret_key, (bytes, bytearray)) or len(secret_key) == 0:
                raise ValueError(
                    "StorageConfig.enable_encryption=True requiere StorageConfig."
                    "secret_key (bytes no vacíos) para derivar la clave de cifrado "
                    "AES-256-GCM; no se almacenará en claro de forma silenciosa."
                )
            self._aead_key = self._derive_encryption_key(bytes(secret_key))

        # Inicializar base de datos para metadatos
        self.db_path = self.storage_path / "metadata.db"
        self._init_database()

        # Cache seguro
        self.cache = {}
        self.cache_lock = Lock()
        self.access_times = {}
        self.access_counts = {}

    def _init_database(self):
        """Inicializar base de datos de metadatos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS storage_metadata (
                        key TEXT PRIMARY KEY,
                        size INTEGER,
                        created_at REAL,
                        accessed_at REAL,
                        access_count INTEGER,
                        checksum TEXT,
                        compressed BOOLEAN,
                        tier TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _validate_key(self, key: str) -> bool:
        """Validar clave de almacenamiento"""
        if not isinstance(key, str):
            return False
        if len(key) > 255:  # Límite de longitud
            return False
        if not key.replace('_', '').replace('-', '').isalnum():
            return False
        return True

    def _validate_data(self, data: bytes) -> bool:
        """Validar datos de almacenamiento"""
        if not isinstance(data, bytes):
            return False
        if len(data) > self.config.max_file_size_mb * 1024 * 1024:
            return False
        return True

    def _compute_checksum(self, data: bytes) -> str:
        """Calcular checksum de datos"""
        return hashlib.sha256(data).hexdigest()

    # Marcadores de la capa de compresión del almacenamiento. El blob dice si está
    # comprimido en vez de que el lector lo adivine: los llamadores entregan datos
    # que YA son un frame LZ4 (ZSpace guarda así todos sus tensores), de modo que
    # "intenta descomprimir y si funciona es que estaba comprimido" tenía éxito
    # espurio y devolvía el contenido interno en lugar del blob almacenado.
    _MARCA_COMPRIMIDO = b"MNZ1"
    _MARCA_SIN_COMPRIMIR = b"MNZ0"

    def _compress_data(self, data: bytes) -> bytes:
        """Comprimir datos de forma segura, marcando el resultado"""
        if not self.config.enable_compression:
            return self._MARCA_SIN_COMPRIMIR + data

        try:
            compressed = lz4.frame.compress(
                data,
                compression_level=self.config.compression_level.value
            )
            # Solo usar compresión si es beneficiosa
            if len(compressed) < len(data) * 0.9:
                return self._MARCA_COMPRIMIDO + compressed
            return self._MARCA_SIN_COMPRIMIR + data
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return self._MARCA_SIN_COMPRIMIR + data

    def _decompress_data(self, data: bytes) -> bytes:
        """Descomprimir según el marcador del propio blob"""
        marca, cuerpo = data[:4], data[4:]
        if marca == self._MARCA_COMPRIMIDO:
            return lz4.frame.decompress(cuerpo)
        if marca == self._MARCA_SIN_COMPRIMIR:
            return cuerpo
        raise StorageFormatError(
            "blob de almacenamiento sin marcador de compresión: fue escrito con un "
            "formato anterior en el que el lector adivinaba si descomprimir y podía "
            "devolver el contenido equivocado. Hay que regenerarlo."
        )

    def _derive_encryption_key(self, secret_key: bytes) -> bytes:
        """Deriva una clave AES-256 (32 bytes) a partir de secret_key vía HKDF-SHA256.

        No se usa secret_key en crudo como clave de cifrado: HKDF con un `info`
        fijo del proyecto separa por dominio esta clave de cualquier otro uso que
        se le dé a la misma secret_key en otras partes del sistema.
        """
        hkdf = HKDF(
            algorithm=_crypto_hashes.SHA256(),
            length=32,
            salt=None,
            info=_ENC_HKDF_INFO,
        )
        return hkdf.derive(secret_key)

    def _encrypt_data(self, data: bytes) -> bytes:
        """Cifra datos con AES-256-GCM (AEAD, autenticado).

        Nonce de 12 bytes aleatorio en cada llamada (nunca fijo ni reutilizado);
        se guarda junto al ciphertext en el sobre resultante.
        """
        nonce = os.urandom(_ENC_NONCE_SIZE)
        ciphertext = AESGCM(self._aead_key).encrypt(nonce, data, None)
        return _ENC_MAGIC + bytes([_ENC_VERSION]) + nonce + ciphertext

    def _decrypt_data(self, blob: bytes) -> bytes:
        """Descifra un sobre producido por `_encrypt_data`.

        Lanza StorageAuthenticationError si falta el marcador MNSE, la versión
        no coincide o falla la verificación del tag GCM (dato alterado o clave
        incorrecta). Nunca devuelve datos sin autenticar.
        """
        if len(blob) < _ENC_HEADER_SIZE or blob[:len(_ENC_MAGIC)] != _ENC_MAGIC:
            raise StorageAuthenticationError(
                "Blob cifrado inválido: falta el marcador MNSE o el blob está truncado"
            )
        version = blob[len(_ENC_MAGIC)]
        if version != _ENC_VERSION:
            raise StorageAuthenticationError(
                f"Versión de cifrado no soportada: {version}"
            )
        nonce = blob[len(_ENC_MAGIC) + 1:_ENC_HEADER_SIZE]
        ciphertext = blob[_ENC_HEADER_SIZE:]
        try:
            return AESGCM(self._aead_key).decrypt(nonce, ciphertext, None)
        except InvalidTag as e:
            raise StorageAuthenticationError(
                "Fallo de autenticación AES-GCM: el dato fue alterado o la clave "
                "de cifrado no corresponde"
            ) from e

    def _encode_for_storage(self, compressed_data: bytes) -> bytes:
        """Cifra (si enable_encryption) el bloque ya comprimido antes de escribirlo a disco."""
        if self.config.enable_encryption:
            return self._encrypt_data(compressed_data)
        return compressed_data

    def _decode_from_storage(self, on_disk_data: bytes) -> bytes:
        """Invierte `_encode_for_storage` seguido de descompresión."""
        data = on_disk_data
        if self.config.enable_encryption:
            data = self._decrypt_data(data)
        return self._decompress_data(data)

    def _resolve_safe_path(self, key: str) -> Path:
        """Resuelve la ruta de archivo de `key` y verifica que sigue bajo storage_path.

        Segundo cinturón de seguridad, independiente de `_validate_key`: no basta
        con que la clave "parezca" segura por su alfabeto, la ruta final resuelta
        debe seguir contenida dentro de storage_path antes de tocar el disco.
        """
        storage_root = self.storage_path.resolve()
        candidate = (self.storage_path / f"{key}.dat").resolve()
        try:
            candidate.relative_to(storage_root)
        except ValueError:
            raise ValueError(f"Resolved path escapes storage_path for key: {key!r}") from None
        return candidate

    def store(self, key: str, data: bytes, metadata: dict[str, Any] = None) -> bool:
        """Almacenar datos de forma segura"""
        try:
            with self.lock:
                # Validar entrada
                if not self._validate_key(key):
                    raise ValueError(f"Invalid storage key: {key}")

                if not self._validate_data(data):
                    raise ValueError("Invalid data for storage")

                # Validar metadatos
                if metadata and not self.security_manager.validate_input(metadata, "metadata"):
                    raise ValueError("Invalid metadata")

                # Comprimir si es necesario
                compressed_data = self._compress_data(data)
                is_compressed = len(compressed_data) < len(data)

                # Cifrar (si enable_encryption) el bloque ya comprimido: esto es lo
                # que efectivamente se escribe a disco y se guarda en cache.
                on_disk_data = self._encode_for_storage(compressed_data)

                # Calcular checksum sobre los bytes exactos que se escriben a disco
                checksum = self._compute_checksum(on_disk_data)

                # Determinar nivel de almacenamiento
                tier = self._determine_tier(len(on_disk_data))

                # Almacenar archivo
                file_path = self._resolve_safe_path(key)
                with open(file_path, 'wb') as f:
                    f.write(on_disk_data)

                # Actualizar metadatos en base de datos
                current_time = time.time()
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO storage_metadata
                        (key, size, created_at, accessed_at, access_count, checksum, compressed, tier)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (key, len(on_disk_data), current_time, current_time, 0, checksum, is_compressed, tier.value))
                    conn.commit()

                # Actualizar cache
                with self.cache_lock:
                    self.cache[key] = on_disk_data
                    self.access_times[key] = current_time
                    self.access_counts[key] = 0

                logger.debug(f"Stored {key} ({len(on_disk_data)} bytes, tier: {tier.value})")
                return True

        except Exception as e:
            logger.error(f"Storage failed for key {key}: {e}")
            return False

    def retrieve(self, key: str) -> bytes | None:
        """Recuperar datos de forma segura"""
        try:
            with self.lock:
                if not self._validate_key(key):
                    raise ValueError(f"Invalid storage key: {key}")

                # Verificar cache primero
                with self.cache_lock:
                    if key in self.cache:
                        self.access_counts[key] = self.access_counts.get(key, 0) + 1
                        self.access_times[key] = time.time()
                        return self._decode_from_storage(self.cache[key])

                # Recuperar de almacenamiento (ruta validada y comprobada bajo storage_path)
                file_path = self._resolve_safe_path(key)
                if not file_path.exists():
                    return None

                with open(file_path, 'rb') as f:
                    data = f.read()

                # Verificar integridad. Un checksum que no cuadra significa que los
                # bytes en disco cambiaron; devolver None aquí lo hacía
                # indistinguible de "esa clave no existe" y, además, cortocircuitaba
                # la autenticación AES-GCM, que nunca llegaba a ejecutarse.
                if self.config.validate_integrity:
                    if not self._verify_integrity(key, data):
                        raise StorageAuthenticationError(
                            f"el checksum almacenado de '{key}' no coincide con los "
                            f"bytes en disco: fueron alterados o truncados"
                        )

                # Actualizar metadatos
                current_time = time.time()
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE storage_metadata
                        SET accessed_at = ?, access_count = access_count + 1
                        WHERE key = ?
                    """, (current_time, key))
                    conn.commit()

                # Actualizar cache
                with self.cache_lock:
                    self.cache[key] = data
                    self.access_times[key] = current_time
                    self.access_counts[key] = self.access_counts.get(key, 0) + 1

                return self._decode_from_storage(data)

        except StorageAuthenticationError:
            # No se atrapa junto al resto de excepciones: nunca se devuelve None
            # (indistinguible de "clave no encontrada") cuando en realidad el
            # blob fue alterado o la clave de cifrado no corresponde.
            raise
        except Exception as e:
            logger.error(f"Retrieval failed for key {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Eliminar datos de forma segura"""
        try:
            with self.lock:
                if not self._validate_key(key):
                    raise ValueError(f"Invalid storage key: {key}")

                # Eliminar archivo (ruta validada y comprobada bajo storage_path)
                file_path = self._resolve_safe_path(key)
                if file_path.exists():
                    file_path.unlink()

                # Eliminar de base de datos
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM storage_metadata WHERE key = ?", (key,))
                    conn.commit()

                # Eliminar de cache
                with self.cache_lock:
                    self.cache.pop(key, None)
                    self.access_times.pop(key, None)
                    self.access_counts.pop(key, None)

                logger.debug(f"Deleted {key}")
                return True

        except Exception as e:
            logger.error(f"Deletion failed for key {key}: {e}")
            return False

    def _determine_tier(self, size: int) -> StorageTier:
        """Determinar nivel de almacenamiento basado en tamaño"""
        if size < 1024 * 1024:  # < 1MB
            return StorageTier.MEMORY
        elif size < 100 * 1024 * 1024:  # < 100MB
            return StorageTier.SSD
        elif size < 1024 * 1024 * 1024:  # < 1GB
            return StorageTier.HDD
        else:
            return StorageTier.ARCHIVE

    def _verify_integrity(self, key: str, data: bytes) -> bool:
        """Verificar integridad de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT checksum FROM storage_metadata WHERE key = ?", (key,)
                )
                result = cursor.fetchone()
                if not result:
                    return False

                stored_checksum = result[0]
                current_checksum = self._compute_checksum(data)
                return stored_checksum == current_checksum

        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Obtener estadísticas de almacenamiento"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT
                        COUNT(*) as total_keys,
                        SUM(size) as total_size,
                        AVG(access_count) as avg_access_count,
                        COUNT(CASE WHEN compressed = 1 THEN 1 END) as compressed_count
                    FROM storage_metadata
                """)
                result = cursor.fetchone()

                return {
                    "total_keys": result[0] or 0,
                    "total_size_bytes": result[1] or 0,
                    "average_access_count": result[2] or 0,
                    "compressed_count": result[3] or 0,
                    "cache_size": len(self.cache),
                    "security_violations": self.security_manager.security_violations
                }

        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {}

    def list_keys(self) -> list[str]:
        """Listar todas las claves almacenadas"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT key FROM storage_metadata")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return []

    def cleanup(self):
        """Limpiar recursos"""
        try:
            with self.cache_lock:
                self.cache.clear()
                self.access_times.clear()
                self.access_counts.clear()

            logger.info("Storage backend cleaned up")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

class SecureCache:
    """Cache seguro con validación"""

    def __init__(self, max_size_bytes: int, policy: CachePolicy = CachePolicy.ADAPTIVE):
        self.max_size_bytes = max_size_bytes
        self.policy = policy
        self.cache = {}
        self.lock = Lock()
        self.current_size = 0
        self.access_times = {}
        self.access_counts = {}
        self.creation_times = {}

    def _estimate_size(self, value: Any) -> int:
        """Estimar tamaño de un valor"""
        if isinstance(value, bytes):
            return len(value)
        elif isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        elif isinstance(value, dict):
            return len(str(value))
        else:
            return len(str(value))

    def _evict_adaptive(self):
        """Evictar elemento usando estrategia adaptativa"""
        if not self.cache:
            return

        current_time = time.time()

        # Calcular scores para cada elemento
        scores = {}
        for key in self.cache.keys():
            access_time = self.access_times.get(key, self.creation_times.get(key, current_time))
            access_count = self.access_counts.get(key, 0)
            age = current_time - access_time

            # Score basado en frecuencia, recencia y tamaño
            freq_score = access_count / max(age, 1)
            recency_score = 1.0 / (age + 1)
            size_penalty = 1.0 / (self._estimate_size(self.cache[key]) + 1)

            scores[key] = freq_score * recency_score * size_penalty

        # Evictar elemento con menor score
        if scores:
            evict_key = min(scores.keys(), key=lambda k: scores[k])
            self._evict_key(evict_key)

    def _evict_key(self, key: str):
        """Evictar clave específica"""
        if key in self.cache:
            self.current_size -= self._estimate_size(self.cache[key])
            del self.cache[key]
            self.access_times.pop(key, None)
            self.access_counts.pop(key, None)
            self.creation_times.pop(key, None)

    def get(self, key: str) -> Any | None:
        """Obtener elemento del cache"""
        with self.lock:
            if key not in self.cache:
                return None

            # Actualizar métricas de acceso
            current_time = time.time()
            self.access_times[key] = current_time
            self.access_counts[key] = self.access_counts.get(key, 0) + 1

            return self.cache[key]

    def put(self, key: str, value: Any) -> bool:
        """Almacenar elemento en cache"""
        with self.lock:
            value_size = self._estimate_size(value)

            # Verificar si cabe
            if value_size > self.max_size_bytes:
                return False

            # Evictar elementos si es necesario
            while self.current_size + value_size > self.max_size_bytes and self.cache:
                self._evict_adaptive()

            # Almacenar
            if key in self.cache:
                self.current_size -= self._estimate_size(self.cache[key])

            self.cache[key] = value
            self.current_size += value_size

            # Actualizar métricas
            current_time = time.time()
            self.access_times[key] = current_time
            self.access_counts[key] = 1
            self.creation_times[key] = current_time

            return True

    def delete(self, key: str) -> bool:
        """Eliminar elemento del cache"""
        with self.lock:
            if key in self.cache:
                self._evict_key(key)
                return True
            return False

    def clear(self):
        """Limpiar cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.creation_times.clear()
            self.current_size = 0

    def get_stats(self) -> dict[str, Any]:
        """Obtener estadísticas del cache"""
        with self.lock:
            total_requests = sum(self.access_counts.values())
            hit_rate = (len(self.cache) / max(total_requests, 1)) * 100

            return {
                "size_bytes": self.current_size,
                "max_size_bytes": self.max_size_bytes,
                "usage_percent": (self.current_size / self.max_size_bytes) * 100,
                "entries": len(self.cache),
                "hit_rate": hit_rate,
                "policy": self.policy.value
            }

# --- Funciones de Utilidad ---

def create_secure_storage(config: StorageConfig = None) -> SecureStorageBackend:
    """Crear backend de almacenamiento seguro"""
    if config is None:
        config = StorageConfig()

    security_config = create_secure_config()
    return SecureStorageBackend(config, security_config)

def create_secure_cache(max_size_mb: int = 1024, policy: CachePolicy = CachePolicy.ADAPTIVE) -> SecureCache:
    """Crear cache seguro"""
    return SecureCache(max_size_mb * 1024 * 1024, policy)
