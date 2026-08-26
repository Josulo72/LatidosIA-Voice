@echo off
title LatidosIA Voice - Inferencia LoRA
cd /d "%~dp0"
if not exist .venv-inference\Scripts\python.exe (
  echo Falta instalar el entorno. Ejecuta INSTALAR_INFERENCIA_LORA.bat
  pause
  exit /b 1
)
call .venv-inference\Scripts\activate
python -m uvicorn inference.server:app --host 127.0.0.1 --port 5006
pause
