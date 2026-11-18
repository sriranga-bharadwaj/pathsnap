PathSnap - Laptop 2 (UI only)
=============================

This folder is the complete Laptop-2 application. It contains NO model, NO data and
NO ground truth. It only talks to the PathSnap API running on Laptop 1.

Requirements: Python 3 (any recent version). No pip installs. Works offline
(Leaflet is in vendor/leaflet); the street basemap appears only with internet.

1. Put both laptops on the same Wi-Fi or phone hotspot.
2. On Laptop 1 start the API (run_api.bat) and note its IPv4 address (ipconfig).
3. On this laptop, in this folder, double-click run_ui.bat
   (or run:  python -m http.server 5500 ; on Linux/macOS: ./run_ui.sh).
4. Open http://localhost:5500 in a browser.
5. Type http://<laptop1-ip>:8000 in the "API base URL" box and press Connect.
   (To change the default permanently, edit API_BASE_URL in config.js.)
6. Click any trip. The API on Laptop 1 runs the prediction and returns the route and metrics.

Map colours: blue = matched GPS point, orange = low confidence, red = fallback
(relaxed / unmatched / retry), green line = predicted route.

Check connectivity from this laptop: open http://<laptop1-ip>:8000/health in the browser.
If it fails: firewall on Laptop 1 (port 8000), network profile set to Private, or the
Wi-Fi blocks device-to-device traffic (use a phone hotspot).
