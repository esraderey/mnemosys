# Actualización de Serialización Avanzada - MNEME v2.0

## Resumen de Cambios

Se ha implementado un sistema de serialización avanzado y seguro que reemplaza completamente el uso de pickle por safetensors, proporcionando mayor seguridad y eficiencia.

## Nuevas Características

### 1. Múltiples Formatos de Serialización
- **Safetensors**: Serialización segura sin pickle (recomendado)
- **Torch**: Usando `torch.save/load` (más seguro que pickle)
- **MessagePack**: Serialización binaria rápida
- **JSON**: Para datos simples y portabilidad
- **Binary**: Serialización binaria personalizada
- **Hybrid**: Selección automática del mejor formato

### 2. Niveles de Seguridad
- **NONE**: Sin seguridad adicional
- **HMAC**: Firma criptográfica con HMAC-SHA256
- **ENCRYPTED**: Cifrado simétrico con Fernet
- **SIGNED**: Firma digital para verificación de integridad
- **SAFETENSORS**: Serialización segura sin vulnerabilidades de pickle

### 3. Compresión Avanzada
- Soporte para LZ4 con niveles configurables
- Compresión automática opcional
- Descompresión transparente
- Lazy decompression para optimizar memoria

### 4. Validación de Integridad
- Verificación de tipos de datos
- Metadatos de validación
- Checksums criptográficos
- Verificación de integridad automática

## Nuevas Clases y Enums

### Enums Agregados
```python
class SerializationFormat(Enum):
    TORCH = "torch"
    MSGPACK = "msgpack"
    JSON = "json"
    BINARY = "binary"
    HYBRID = "hybrid"

class SecurityLevel(Enum):
    NONE = "none"
    HMAC = "hmac"
    ENCRYPTED = "encrypted"
    SIGNED = "signed"
```

### Clase AdvancedSerializer
Reemplaza la clase `Serializer` original con funcionalidades avanzadas:

- **Serialización inteligente**: Selecciona automáticamente el mejor formato
- **Seguridad robusta**: Múltiples niveles de protección
- **Compresión eficiente**: LZ4 con niveles configurables
- **Validación completa**: Verificación de tipos e integridad

## Configuración Actualizada

### MnemeConfig - Nuevas Opciones
```python
@dataclass
class MnemeConfig:
    # ... opciones existentes ...
    
    # Nuevas opciones de serialización avanzada
    serialization_format: SerializationFormat = SerializationFormat.HYBRID
    security_level: SecurityLevel = SecurityLevel.HMAC
    enable_encryption: bool = False
    encryption_password: Optional[str] = None
    enable_compression: bool = True
    enable_validation: bool = True
```

## Ejemplos de Uso

### Configuración Básica
```python
config = MnemeConfig(
    serialization_format=SerializationFormat.TORCH,
    security_level=SecurityLevel.HMAC,
    secret_key=b"your_secret_key_32_bytes",
    enable_compression=True
)

with ZSpace(config) as mneme:
    desc = mneme.register("my_tensor", tensor_data)
    loaded_tensor = mneme.load("my_tensor")
```

### Configuración de Alta Seguridad
```python
config = MnemeConfig(
    serialization_format=SerializationFormat.HYBRID,
    security_level=SecurityLevel.ENCRYPTED,
    enable_encryption=True,
    encryption_password="strong_password",
    enable_compression=True,
    enable_validation=True
)
```

### Configuración de Alto Rendimiento
```python
config = MnemeConfig(
    serialization_format=SerializationFormat.MSGPACK,
    security_level=SecurityLevel.HMAC,
    compression_level=CompressionLevel.ULTRA_FAST,
    enable_compression=True
)
```

## Beneficios de Seguridad

### 1. Eliminación de Pickle
- **Antes**: Uso de pickle (vulnerable a ataques de deserialización)
- **Ahora**: Métodos seguros (torch.save/load, MessagePack, JSON)

### 2. Verificación de Integridad
- Checksums criptográficos SHA256
- Verificación HMAC para autenticidad
- Validación de tipos de datos

### 3. Cifrado Opcional
- Cifrado simétrico con Fernet
- Claves derivadas con PBKDF2
- Protección completa de datos sensibles

## Rendimiento

### Formatos por Velocidad
1. **MessagePack**: Más rápido para datos simples
2. **Torch**: Óptimo para tensores
3. **JSON**: Más lento pero más portable
4. **Hybrid**: Balance automático

### Niveles de Compresión
- **ULTRA_FAST**: Compresión mínima, máxima velocidad
- **BALANCED**: Balance entre tamaño y velocidad
- **MAXIMUM**: Máxima compresión, menor velocidad

## Compatibilidad

- **Retrocompatibilidad**: La clase `Serializer` original sigue funcionando
- **Migración gradual**: Se puede actualizar progresivamente
- **Configuración flexible**: Cada componente se puede habilitar/deshabilitar

## Dependencias Agregadas

```txt
# Advanced serialization
msgpack>=1.0.0
```

## Archivos Modificados

1. **mneme_core.py**: Implementación de `AdvancedSerializer`
2. **requirements.txt**: Nueva dependencia `msgpack`
3. **example_advanced_serialization.py**: Ejemplos de uso
4. **SERIALIZATION_UPGRADE.md**: Esta documentación

## Próximos Pasos

1. **Instalar dependencias**: `pip install msgpack`
2. **Probar ejemplos**: Ejecutar `example_advanced_serialization.py`
3. **Configurar según necesidades**: Ajustar niveles de seguridad y compresión
4. **Migrar gradualmente**: Actualizar configuraciones existentes

## Notas de Seguridad

- **Claves secretas**: Usar claves de al menos 32 bytes
- **Contraseñas de cifrado**: Usar contraseñas fuertes
- **Almacenamiento seguro**: Proteger claves y contraseñas
- **Validación**: Habilitar validación en entornos de producción

Esta actualización proporciona un sistema de serialización robusto, seguro y eficiente que elimina las vulnerabilidades de pickle mientras mantiene la compatibilidad y agrega funcionalidades avanzadas.
