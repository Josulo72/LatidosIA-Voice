@echo off
title LatidosIA Voice - Instalar inferencia LoRA
cd /d "%~dp0"
where python >nul 2>nul || (echo Python no encontrado & pause & exit /b 1)
if not exist .venv-inference python -m venv .venv-inference
call .venv-inference\Scripts\activate
python -m pip install --upgrade pip
pip install -r inference\requirements-inference.txt
echo.
echo Entorno de inferencia LoRA instalado.
pause
