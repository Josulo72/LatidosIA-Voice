@echo off
title LatidosIA Voice Multi v1.1
cd /d "%~dp0"
where python >nul 2>nul || (echo Python no encontrado & pause & exit /b 1)
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (call .venv\Scripts\activate)
start "" http://127.0.0.1:8777
python -m uvicorn app.main:app --host 127.0.0.1 --port 8777
pause
