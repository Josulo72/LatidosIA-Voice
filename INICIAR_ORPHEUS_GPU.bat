@echo off
title Orpheus GPU - LatidosIA Voice
cd /d "%~dp0"
docker compose up -d
echo Orpheus arrancado en http://127.0.0.1:5005
pause
