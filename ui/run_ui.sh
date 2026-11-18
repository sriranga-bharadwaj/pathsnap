#!/usr/bin/env bash
# PathSnap UI - Laptop 2. Needs only Python (no pip installs). Then open http://localhost:5500
cd "$(dirname "$0")"
echo "Serving PathSnap UI at http://localhost:5500  (Ctrl+C to stop)"
python3 -m http.server 5500
