# LatidosIA Voice Multi v1.2

Aplicación local para gestionar múltiples voces, clonar en zero-shot con Orpheus y entrenar un adaptador LoRA independiente por voz con Unsloth.

## Funciones
- Perfiles de voz separados
- Idioma, estilo y notas por perfil
- Varias muestras de audio por voz
- Zero-shot con muestra de referencia
- Dataset JSONL automático por voz
- Entrenamiento LoRA desde la interfaz
- Estado y log del entrenamiento
- Adaptador independiente por perfil
- Generación WAV y descarga

## Arranque TTS
1. Ejecuta `INICIAR_ORPHEUS_GPU.bat`.
2. Ejecuta `INICIAR_WINDOWS.bat`.
3. Abre `http://127.0.0.1:8777`.

## Preparar entrenamiento LoRA
1. Ejecuta una vez `INSTALAR_ENTRENAMIENTO.bat`.
2. Reinicia `INICIAR_WINDOWS.bat`. Detectará automáticamente `.venv-training`.
3. Crea una voz y añade varias muestras con su transcripción exacta.
4. Pulsa `Preparar dataset`.
5. Ajusta pasos y LoRA rank.
6. Pulsa `Entrenar voz` y consulta el log desde la interfaz.

## Requisitos de entrenamiento
- GPU NVIDIA con CUDA.
- Python 3.10 recomendado.
- VRAM suficiente para Orpheus 3B. La cantidad real depende de secuencia, cuantización y configuración. Unsloth permite LoRA/QLoRA para reducir memoria, pero no se garantiza entrenamiento razonable en cualquier GPU.

## Dataset
La aplicación acepta desde 3 muestras para permitir pruebas técnicas. Para calidad real conviene usar bastantes más clips limpios, cortos, con transcripciones exactas y variedad fonética. El proyecto Orpheus recomienda del orden de cientos de ejemplos por hablante para obtener mejores resultados.

## Importante sobre inferencia entrenada
El entrenamiento guarda el adaptador en `training/runs/<voice_id>/adapter`. La API envía `voice_adapter` al servidor Orpheus cuando el perfil está entrenado. El servidor de inferencia utilizado debe soportar carga de adaptadores LoRA. Si tu servidor no implementa ese parámetro, puedes seguir usando el modo zero-shot y habrá que adaptar el servicio de inferencia para cargar el LoRA.

## Privacidad
`data/voices`, `outputs`, datasets, adaptadores y entornos virtuales deben permanecer fuera de Git. No publiques muestras de voz sin autorización.
