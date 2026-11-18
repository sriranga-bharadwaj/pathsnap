@echo off
REM PathSnap API - Laptop 1. Usage:  run_api.bat ["path\to\data dir"]
REM Data dir = first argument, else DATA_DIR env var, else the student_test folder.
cd /d "%~dp0"
if not "%~1"=="" set "DATA_DIR=%~1"
if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts\python.exe") else (set "PY=python")
echo Starting PathSnap API on http://0.0.0.0:8000  (DATA_DIR=%DATA_DIR%)
"%PY%" -m api.main --port 8000
pause
