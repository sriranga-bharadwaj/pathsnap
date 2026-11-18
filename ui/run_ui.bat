@echo off
REM PathSnap UI - Laptop 2. Needs only Python (no pip installs).
REM Then open http://localhost:5500 in a browser.
cd /d "%~dp0"
echo Serving PathSnap UI at http://localhost:5500  (Ctrl+C to stop)
python -m http.server 5500
pause
