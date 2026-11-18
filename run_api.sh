#!/usr/bin/env bash
# PathSnap API - Laptop 1. Usage:  ./run_api.sh ["/path/to/data dir"]
# Data dir = first argument, else DATA_DIR env var, else the student_test folder.
cd "$(dirname "$0")"
if [ -n "$1" ]; then export DATA_DIR="$1"; fi
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
echo "Starting PathSnap API on http://0.0.0.0:8000  (DATA_DIR=${DATA_DIR})"
"$PY" -m api.main --port 8000
