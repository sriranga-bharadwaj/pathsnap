"""Smoke test for a running PathSnap API (standard library only - works on either laptop).

Usage:  python scripts/smoke_test.py [http://<laptop1-ip>:8000] [--save results/phase4/api_smoke_test.json]
Calls /health, /trips and /predict for every trip; prints status, metrics and latency.
"""
import json
import sys
import time
import urllib.request

base = next((a for a in sys.argv[1:] if a.startswith("http")), "http://127.0.0.1:8000").rstrip("/")
save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None


def get(path, timeout=300):
    t0 = time.perf_counter()
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        body = json.loads(r.read().decode())
        return r.status, body, (time.perf_counter() - t0) * 1000


report = {}
s, h, ms = get("/health")
print(f"GET /health -> {s} ({ms:.0f} ms): {h}")
report["health"] = h
s, t, ms = get("/trips")
print(f"GET /trips  -> {s} ({ms:.0f} ms): {t['count']} trips: {t['trips']}")
report["trips"] = t
report["predict"] = []
ok = 0
for tid in t["trips"]:
    try:
        s, p, ms = get(f"/predict/{tid}")
        leak = any(k in json.dumps(p) for k in ('"route"', '"reference_route"', '"label'))
        row = dict(trip_id=tid, http=s, status=p["status"], n_points=len(p["gps_points"]),
                   n_predictions=len(p["predictions"]), route_pieces=len(p["route_geometry"]),
                   metrics=p["metrics"], events=p["events"], client_latency_ms=round(ms, 1),
                   reference_keys_in_response=leak)
        ok += p["status"] == "ok"
    except Exception as e:
        row = dict(trip_id=tid, status="error", error=str(e))
    report["predict"].append(row)
    print(f"GET /predict/{tid} -> {row.get('http')} {row['status']} "
          f"points={row.get('n_points')} metrics={row.get('metrics')} events={row.get('events')} "
          f"client_latency={row.get('client_latency_ms')} ms")
print(f"\n{ok}/{len(t['trips'])} trips returned status=ok")
if save:
    with open(save, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved {save}")
