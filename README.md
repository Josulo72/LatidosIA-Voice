# LatidosIA Voice Multi v1.3

Aplicación local para gestionar múltiples voces, clonar en zero-shot con Orpheus y entrenar/usar un adaptador LoRA independiente por voz.

## Arquitectura

`Panel LatidosIA Voice -> selector automático -> zero-shot Orpheus :5005 | LoRA propio :5006`

- Voz sin entrenar: usa audio + transcripción de referencia.
- Voz entrenada: usa el adaptador LoRA guardado para ese perfil.
- La app decide el motor automáticamente.

## Funciones
- Perfiles de voz separados
- Idioma, estilo y notas por perfil
- Varias muestras de audio por voz
- Zero-shot con muestra de referencia
- Dataset JSONL automático por voz
- Entrenamiento LoRA desde la interfaz
- Estado y log del entrenamiento
- Adaptador independiente por perfil
- Servidor propio de inferencia LoRA con Transformers + PEFT + SNAC
- Caché del adaptador activo
- Generación WAV y descarga

## Instalación principal
1. Ejecuta `INICIAR_ORPHEUS_GPU.bat` para el modo zero-shot.
2. Ejecuta `INICIAR_WINDOWS.bat` para el panel.
3. Abre `http://127.0.0.1:8777`.

## Entrenamiento LoRA
1. Ejecuta una vez `INSTALAR_ENTRENAMIENTO.bat`.
2. Crea una voz y añade muestras con transcripción exacta.
3. Pulsa `Preparar dataset`.
4. Configura pasos y LoRA rank.
5. Pulsa `Entrenar voz`.
6. El adaptador queda en `training/runs/<voice_id>/adapter`.

## Inferencia de voces entrenadas
1. Ejecuta una vez `INSTALAR_INFERENCIA_LORA.bat`.
2. Arranca `INICIAR_INFERENCIA_LORA.bat`.
3. El servidor escucha en `http://127.0.0.1:5006`.
4. Cuando un perfil tiene `training_status=trained`, LatidosIA Voice envía automáticamente la generación a este servidor.

El servidor propio usa Transformers porque existen reportes públicos de resultados inestables con algunos motores alternativos de serving para Orpheus. El flujo implementa los tokens de control de Orpheus y la decodificación SNAC de 7 bandas.

## Requisitos
- Windows 10/11
- Python 3.10+ recomendado
- GPU NVIDIA con CUDA para entrenamiento e inferencia LoRA
- Docker Desktop para el servidor zero-shot incluido
- VRAM suficiente para Orpheus 3B. La cantidad real depende de cuantización, longitud y GPU.

## Dataset
La aplicación permite preparar entrenamiento desde 3 muestras para pruebas técnicas. Para calidad real conviene usar bastantes más clips limpios, cortos, bien transcritos y con variedad fonética.

## Limitaciones actuales
- El código de inferencia LoRA está implementado, pero debe validarse contra tu GPU, versión de CUDA y adaptador entrenado real.
- Cambiar de una voz LoRA a otra obliga a descargar el modelo/adaptador activo y cargar el siguiente. La voz activa se mantiene en caché.
- CPU está bloqueada deliberadamente para LoRA porque la inferencia de Orpheus 3B sería demasiado lenta para un uso normal.

## Privacidad
`data/voices`, `outputs`, datasets, logs, adaptadores y entornos virtuales permanecen fuera de Git. No publiques muestras o modelos de voz de terceros sin autorización.
