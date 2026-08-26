# LatidosIA Voice Multi v1.1

## Qué cambia
- Múltiples perfiles de voz
- Idioma y estilo por perfil
- Varias muestras por voz
- Selección de muestra de referencia
- Estado de entrenamiento por voz
- Preparación de dataset JSONL por perfil
- Estructura preparada para un adaptador LoRA independiente por voz

## Arranque
1. `INICIAR_ORPHEUS_GPU.bat`
2. `INICIAR_WINDOWS.bat`
3. Abre `http://127.0.0.1:8777`

## Entrenamiento
La opción **Preparar dataset** crea un archivo en `/training`.
Todavía no ejecuta Unsloth automáticamente. Esa integración debe añadirse como siguiente módulo porque requiere entorno CUDA, dependencias de entrenamiento y configuración de VRAM.

## Seguridad
Usa únicamente voces propias o con autorización.
