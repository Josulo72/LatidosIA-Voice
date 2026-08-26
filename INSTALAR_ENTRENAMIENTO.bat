@echo off
title LatidosIA Voice - Instalar entrenamiento LoRA
cd /d "%~dp0"
where python >nul 2>nul || (echo Python no encontrado & pause & exit /b 1)
if not exist .venv-training (
  python -m venv .venv-training
)
call .venv-training\Scripts\activate
python -m pip install --upgrade pip
pip install -r training\requirements-training.txt
echo.
echo Entorno de entrenamiento instalado.
echo Para usarlo con la app, inicia Windows con TRAIN_PYTHON apuntando a .venv-training\Scripts\python.exe
pause
