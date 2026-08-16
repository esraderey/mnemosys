---
name: 🔒 Security Report
about: Reportar una vulnerabilidad de seguridad en MNEME v2.0
title: '[SECURITY] '
labels: ['security', 'needs-triage']
assignees: ''
---

## 🔒 Descripción de la Vulnerabilidad
[Descripción clara y concisa de la vulnerabilidad de seguridad]

## 🎯 Impacto
- [ ] Crítico (acceso no autorizado, pérdida de datos)
- [ ] Alto (exposición de información sensible)
- [ ] Medio (degradación de seguridad)
- [ ] Bajo (problema menor de seguridad)

## 🔄 Pasos para Reproducir
1. [Paso 1]
2. [Paso 2]
3. [Paso 3]
4. [Ver vulnerabilidad]

## 🛡️ Configuración de Seguridad
```env
# Configuración de seguridad relevante (sin claves sensibles)
MNEME_SECURITY_LEVEL=HIGH
MNEME_ENABLE_ENCRYPTION=true
MNEME_ENABLE_MERKLE=true
MNEME_MAX_FAILED_ATTEMPTS=5
```

## 🖥️ Entorno Afectado
- **OS**: [ej. Windows 10, macOS 12, Ubuntu 20.04]
- **Python**: [ej. 3.8.10, 3.9.7, 3.10.0]
- **PyTorch**: [ej. 1.12.0, 2.0.0]
- **MNEME**: [ej. 2.0.0]
- **Configuración de Seguridad**: [ej. HIGH, MAXIMUM]

## 🔍 Código de Reproducción
```python
# Código que demuestra la vulnerabilidad (sin datos sensibles)
import torch
from mneme import ZSpace, SecurityLevel

# Configuración que expone la vulnerabilidad
config = MnemeConfig(security_level=SecurityLevel.HIGH)
mneme = ZSpace(config)
# ... código que reproduce la vulnerabilidad
```

## 📊 Categoría de Vulnerabilidad
- [ ] Inyección de código
- [ ] Exposición de datos sensibles
- [ ] Bypass de autenticación
- [ ] Elevación de privilegios
- [ ] Denegación de servicio
- [ ] Manipulación de datos
- [ ] Interceptación de comunicación
- [ ] Otro

## 🎯 Componente Afectado
- [ ] Núcleo MNEME
- [ ] Procesamiento Paralelo
- [ ] Seguridad Cuántica
- [ ] Almacenamiento Inteligente
- [ ] Monitoreo de Rendimiento
- [ ] Integración PyTorch
- [ ] API/Interfaz
- [ ] Otro

## 📝 Logs de Seguridad
```
[Logs relevantes de seguridad - sin datos sensibles]
```

## 🛠️ Solución Propuesta
[Si tienes una idea de cómo solucionar la vulnerabilidad]

## 📚 Referencias
[Enlaces a CVE, documentación de seguridad, o recursos relevantes]

## 🔒 Confidencialidad
- [ ] Confirmo que no he divulgado esta vulnerabilidad públicamente
- [ ] Confirmo que no he creado issues públicos sobre esta vulnerabilidad
- [ ] Entiendo que este reporte será tratado de forma confidencial

## 📞 Contacto de Seguridad
Para vulnerabilidades críticas, también contactar:
- **Email**: security@mneme.dev
- **Tiempo de respuesta**: 24-48 horas

## 🏷️ Etiquetas Adicionales
- [ ] Critical
- [ ] High
- [ ] Medium
- [ ] Low
- [ ] Authentication
- [ ] Encryption
- [ ] Data Protection
- [ ] Access Control

## 📊 Severidad
- [ ] Crítica (9.0-10.0)
- [ ] Alta (7.0-8.9)
- [ ] Media (4.0-6.9)
- [ ] Baja (0.1-3.9)

## 📚 Contexto Adicional
[Cualquier otra información relevante sobre la vulnerabilidad]