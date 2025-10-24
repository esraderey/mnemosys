# Actualización de Cifrado de Tensores y Contexto Mejorado - MNEME

## Resumen de Cambios

Se han implementado funcionalidades avanzadas de cifrado específico para tensores, rotación automática de claves y gestión de contexto mejorada con soporte asíncrono.

## Nuevas Características Implementadas

### 1. Cifrado Específico para Tensores

#### Algoritmos de Cifrado Disponibles
- **AES-GCM**: Cifrado autenticado, recomendado para máxima seguridad
- **AES-CBC**: Cifrado más rápido, menos seguro
- **ChaCha20**: Alternativa moderna a AES
- **Block-Chain**: Cifrado por bloques para tensores grandes

#### Clase TensorEncryptor
```python
class TensorEncryptor:
    """Cifrado especializado para tensores con optimizaciones específicas."""
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]
    def decrypt_tensor(self, encrypted_data: Dict[str, Any], device: torch.device) -> torch.Tensor
```

### 2. Rotación Automática de Claves

#### Políticas de Rotación
- **NEVER**: Sin rotación automática
- **TIME_BASED**: Rotar cada X tiempo
- **USAGE_BASED**: Rotar después de X usos
- **ADAPTIVE**: Rotar basado en patrones de uso

#### Clase KeyManager
```python
class KeyManager:
    """Gestión avanzada de claves con rotación automática."""
    
    def should_rotate_key(self) -> bool
    def rotate_key(self) -> bytes
    def increment_usage(self)
    def get_key_by_version(self, version: int) -> Optional[bytes]
```

### 3. Contexto Asíncrono Mejorado

#### Contexto Síncrono Mejorado
```python
@contextmanager
def mneme_context(config: Optional[MnemeConfig] = None) -> Generator[MnemeContextManager, None, None]:
    """Context manager síncrono para MNEME."""
```

#### Contexto Asíncrono
```python
@asynccontextmanager
async def async_mneme_context(config: Optional[MnemeConfig] = None) -> AsyncGenerator[AsyncMnemeContext, None]:
    """Context manager asíncrono para MNEME."""
```

### 4. Monitoreo de Recursos

#### Características de Monitoreo
- **Memoria RAM**: Monitoreo en tiempo real
- **GPU**: Uso de memoria de GPU
- **Operaciones concurrentes**: Control de concurrencia
- **Presión de memoria**: Alertas automáticas

## Configuración Actualizada

### Nuevas Opciones en MnemeConfig

```python
@dataclass
class MnemeConfig:
    # ... opciones existentes ...
    
    # Opciones de cifrado de tensores
    tensor_encryption_mode: TensorEncryptionMode = TensorEncryptionMode.AES_GCM
    enable_tensor_encryption: bool = False
    tensor_encryption_key: Optional[bytes] = None
    
    # Rotación de claves
    key_rotation_policy: KeyRotationPolicy = KeyRotationPolicy.NEVER
    key_rotation_interval: timedelta = timedelta(days=30)
    key_rotation_usage_count: int = 1000
    enable_key_versioning: bool = False
    
    # Contexto asíncrono
    enable_async_context: bool = False
    max_concurrent_operations: int = 10
```

## Ejemplos de Uso

### 1. Cifrado de Tensores Básico

```python
config = MnemeConfig(
    enable_tensor_encryption=True,
    tensor_encryption_mode=TensorEncryptionMode.AES_GCM,
    tensor_encryption_key=b"your_32_byte_key_here"
)

with mneme_context(config) as mneme:
    tensor = torch.randn(100, 100)
    desc = mneme.register("encrypted_tensor", tensor)
    loaded_tensor = mneme.load("encrypted_tensor")
```

### 2. Rotación Automática de Claves

```python
config = MnemeConfig(
    key_rotation_policy=KeyRotationPolicy.USAGE_BASED,
    key_rotation_usage_count=100,
    enable_key_versioning=True
)

with mneme_context(config) as mneme:
    # Las claves se rotarán automáticamente después de 100 usos
    for i in range(150):
        tensor = torch.randn(50, 50)
        mneme.register(f"tensor_{i}", tensor)
```

### 3. Contexto Asíncrono

```python
async def async_operations():
    config = MnemeConfig(
        enable_async_context=True,
        max_concurrent_operations=5
    )
    
    async with async_mneme_context(config) as mneme:
        # Operaciones concurrentes
        tasks = []
        for i in range(10):
            tensor = torch.randn(100, 100)
            task = mneme.register_async(f"async_tensor_{i}", tensor)
            tasks.append(task)
        
        descriptors = await asyncio.gather(*tasks)
```

### 4. Monitoreo de Recursos

```python
config = MnemeConfig(
    enable_async_context=True,
    memory_pressure_threshold=0.8
)

with mneme_context(config) as mneme:
    # El sistema monitoreará automáticamente el uso de recursos
    large_tensor = torch.randn(1000, 1000)
    desc = mneme.register("large_tensor", large_tensor)
```

## Beneficios de Seguridad

### 1. Cifrado Específico para Tensores
- **Optimización**: Algoritmos optimizados para datos numéricos
- **Eficiencia**: Cifrado por bloques para tensores grandes
- **Flexibilidad**: Múltiples algoritmos según necesidades

### 2. Rotación de Claves
- **Seguridad**: Reduce el riesgo de compromiso de claves
- **Automatización**: Rotación transparente
- **Versionado**: Soporte para múltiples versiones de claves

### 3. Gestión de Contexto
- **Recursos**: Monitoreo automático de memoria y GPU
- **Concurrencia**: Control de operaciones simultáneas
- **Limpieza**: Gestión automática de recursos

## Rendimiento

### Cifrado por Algoritmo
1. **ChaCha20**: Más rápido, buena seguridad
2. **AES-CBC**: Balance velocidad/seguridad
3. **AES-GCM**: Más seguro, autenticado
4. **Block-Chain**: Optimizado para tensores grandes

### Contexto Asíncrono
- **Concurrencia**: Hasta 10 operaciones simultáneas por defecto
- **Thread Pool**: Ejecución en hilos separados
- **Semáforos**: Control de concurrencia

## Archivos Creados/Modificados

1. **mneme_core.py**: Implementación completa de cifrado y contexto
2. **example_advanced_encryption.py**: Ejemplos de uso
3. **ENCRYPTION_AND_CONTEXT_UPGRADE.md**: Esta documentación

## Nuevas Clases y Enums

### Enums Agregados
```python
class TensorEncryptionMode(Enum):
    AES_GCM = "aes_gcm"
    AES_CBC = "aes_cbc"
    CHACHA20 = "chacha20"
    BLOCK_CHAIN = "block_chain"

class KeyRotationPolicy(Enum):
    NEVER = "never"
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    ADAPTIVE = "adaptive"
```

### Clases Principales
- **TensorEncryptor**: Cifrado especializado para tensores
- **KeyManager**: Gestión de claves con rotación
- **AsyncMnemeContext**: Contexto asíncrono
- **MnemeContextManager**: Gestor de contexto mejorado

## Compatibilidad

- **Retrocompatibilidad**: Todas las funcionalidades existentes siguen funcionando
- **Migración gradual**: Se pueden habilitar características progresivamente
- **Configuración flexible**: Cada característica se puede habilitar/deshabilitar

## Notas de Seguridad

### Cifrado de Tensores
- **Claves**: Usar claves de al menos 32 bytes
- **Algoritmos**: AES-GCM recomendado para máxima seguridad
- **Tensores grandes**: Usar Block-Chain para mejor rendimiento

### Rotación de Claves
- **Políticas**: Configurar según necesidades de seguridad
- **Versionado**: Mantener historial de claves para compatibilidad
- **Limpieza**: Limpiar claves antiguas regularmente

### Contexto Asíncrono
- **Concurrencia**: Limitar operaciones simultáneas según recursos
- **Monitoreo**: Habilitar monitoreo en entornos de producción
- **Limpieza**: El contexto se limpia automáticamente

## Próximos Pasos

1. **Configurar cifrado**: Elegir algoritmo y configurar claves
2. **Configurar rotación**: Establecer política de rotación de claves
3. **Probar ejemplos**: Ejecutar `example_advanced_encryption.py`
4. **Monitorear recursos**: Habilitar monitoreo en producción
5. **Optimizar rendimiento**: Ajustar configuración según necesidades

Esta actualización proporciona un sistema completo de cifrado de tensores, rotación automática de claves y gestión de contexto avanzada que mejora significativamente la seguridad y el rendimiento de MNEME.
