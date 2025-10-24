---
name: 🐛 Bug Report
about: Reportar un bug en MNEME v2.0
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''
---

## 🐛 Descripción del Bug
[Descripción clara y concisa del problema]

## 🔄 Pasos para Reproducir
1. Ir a '...'
2. Hacer clic en '...'
3. Desplazarse hasta '...'
4. Ver error

## ✅ Comportamiento Esperado
[Descripción clara de lo que debería pasar]

## ❌ Comportamiento Actual
[Descripción clara de lo que está pasando]

## 📸 Capturas de Pantalla
[Si aplica, agregar capturas de pantalla para ayudar a explicar el problema]

## 🖥️ Información del Sistema
- **OS**: [ej. Windows 10, macOS 12, Ubuntu 20.04]
- **Python**: [ej. 3.8.10, 3.9.7, 3.10.0]
- **PyTorch**: [ej. 1.12.0, 2.0.0]
- **MNEME**: [ej. 2.0.0]
- **Safetensors**: [ej. 0.3.0+]
- **RAM**: [ej. 8GB, 16GB, 32GB]
- **GPU**: [ej. NVIDIA RTX 3080, AMD RX 6800, CPU only]

## 📋 Configuración
```env
# Configuración relevante (sin claves sensibles)
MNEME_CACHE_SIZE_MB=1024
MNEME_COMPRESSION_LEVEL=6
MNEME_ENABLE_PARALLEL_PROCESSING=true
MNEME_SECURITY_LEVEL=STANDARD
```

## 📝 Logs
```
[Logs relevantes del error]
```

## 🔍 Código de Reproducción
```python
import torch
from mneme import ZSpace, CompressionConfig

# Código que reproduce el bug
mneme = ZSpace()
tensor = torch.randn(100, 100)
desc = mneme.register("test", tensor)
# ... más código
```

## 🏷️ Etiquetas Adicionales
- [ ] Procesamiento Paralelo
- [ ] Seguridad
- [ ] Almacenamiento
- [ ] Monitoreo
- [ ] PyTorch Integration
- [ ] Performance
- [ ] Memory
- [ ] GPU

## 📊 Impacto
- [ ] Crítico (sistema no funciona)
- [ ] Alto (funcionalidad principal afectada)
- [ ] Medio (funcionalidad secundaria afectada)
- [ ] Bajo (problema menor)

## 🔧 Solución Propuesta
[Si tienes una idea de cómo solucionarlo]

## 📚 Contexto Adicional
[Cualquier otra información relevante sobre el problema]