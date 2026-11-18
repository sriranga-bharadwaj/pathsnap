// PathSnap UI configuration (Laptop 2).
// Default API base URL. Change it here OR in the text box on the page (the box is remembered
// in this browser). Use Laptop 1's IP address, e.g. "http://192.168.1.23:8000".
window.PATHSNAP_CONFIG = {
  API_BASE_URL: "http://127.0.0.1:8000",
  LOW_CONFIDENCE: 0.5,          // points whose confidence is below this are drawn orange
  BASEMAP_URL: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",   // optional; blank map if offline
  PREDICT_TIMEOUT_MS: 180000,
};
