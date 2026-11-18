"""PathSnap API (Laptop 1).

Run:   python -m api.main --data-dir "<folder with one sub-folder per trip>" [--port 8000]
   or: set DATA_DIR env var and run run_api.bat / run_api.sh

Endpoints
  GET /health              status, model, data dir, number of trips
  GET /trips               trip IDs found in DATA_DIR
  GET /predict/{trip_id}   full prediction pipeline + server-side metrics (no reference data returned)

Every request is logged to this console (trip_id, status, latency) so evaluators can see
the API receiving the UI's requests.
"""
import argparse
import logging
import math
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import list_trips
from src.pipeline import PathSnapPredictor
from src.server_eval import evaluate

DEFAULT_DATA_DIR = ROOT / "Student folder" / "Student folder" / "student_test"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")

STATE = {}
PREDICT_LOCK = threading.Lock()   # one prediction at a time: the per-trip caches are shared


@asynccontextmanager
async def lifespan(app):
    load_model()          # load model, preprocessing contract and decoder config once at startup
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def data_dir():
    return Path(STATE.get("data_dir") or os.environ.get("DATA_DIR") or DEFAULT_DATA_DIR)


def trips():
    d = data_dir()
    if not d.is_dir():
        return []
    # a trip folder must contain at least <id>.track, <id>.nodes and <id>.arcs
    return [t for t in list_trips(d)
            if all((d / t / f"{t}.{e}").exists() for e in ("track", "nodes", "arcs"))]


def clean(x):
    """Make values JSON-safe (NaN/inf -> null)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def load_model():
    t0 = time.time()
    STATE["predictor"] = PathSnapPredictor()
    m = STATE["predictor"].meta
    log.info(f"model loaded: {m['model']} (tuned={m.get('tuned')}) in {time.time() - t0:.1f}s")
    log.info(f"decoder params: {STATE['predictor'].params}")
    log.info(f"DATA_DIR = {data_dir()}  ({len(trips())} trips)")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    client = request.client.host if request.client else "?"
    log.info(f"{client} {request.method} {request.url.path} -> {response.status_code} ({ms:.0f} ms)")
    return response


@app.get("/health")
def health():
    p = STATE.get("predictor")
    return dict(status="ok" if p else "loading", model=p.meta["model"] if p else None,
                tuned=p.meta.get("tuned") if p else None, data_dir=str(data_dir()), n_trips=len(trips()))


@app.get("/trips")
def get_trips():
    t = trips()
    return dict(data_dir=str(data_dir()), count=len(t), trips=t)


@app.get("/predict/{trip_id}")
def predict(trip_id: str):
    t0 = time.perf_counter()
    if trip_id not in trips():      # also blocks path tricks like "../x"
        log.info(f"[PREDICT] trip={trip_id} status=not_found")
        return JSONResponse(status_code=404, content=dict(trip_id=trip_id, status="error",
                                                          message="unknown trip_id"))
    tdir = data_dir() / trip_id
    try:
        with PREDICT_LOCK:
            part, internals = STATE["predictor"].predict(tdir, trip_id)   # prediction (no .route)
    except Exception as e:  # never crash the server
        log.exception(f"[PREDICT] trip={trip_id} status=error")
        return JSONResponse(status_code=500, content=dict(trip_id=trip_id, status="error", message=str(e)))
    try:
        m = evaluate(tdir, trip_id, internals)                             # evaluation (after prediction)
    except Exception:
        log.exception(f"[EVAL] trip={trip_id} evaluation failed; metrics set to null")
        m = dict(top1_accuracy=None, route_continuity=None, candidate_miss_rate=None, reference_available=False)
    total_ms = (time.perf_counter() - t0) * 1000
    body = dict(
        trip_id=trip_id, status="ok",
        model=STATE["predictor"].meta["model"],
        gps_points=part["gps_points"],
        predictions=part["predictions"],
        route_geometry=part["route_geometry"],
        metrics=dict(top1_accuracy=m["top1_accuracy"], route_continuity=m["route_continuity"],
                     candidate_miss_rate=m["candidate_miss_rate"],
                     inference_time_ms=part["inference_time_ms"], total_time_ms=round(total_ms, 1),
                     reference_available=m["reference_available"]),
        events=dict(part["events"], low_confidence_points=part["low_confidence_points"]),
    )
    log.info(f"[PREDICT] trip={trip_id} status=ok points={len(part['gps_points'])} "
             f"top1={m['top1_accuracy']} continuity={m['route_continuity']} "
             f"inference={part['inference_time_ms']:.0f}ms total={total_ms:.0f}ms events={part['events']}")
    return clean(body)


def main():
    ap = argparse.ArgumentParser(description="PathSnap API (Laptop 1)")
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"), help="folder with one sub-folder per trip")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if a.data_dir:
        STATE["data_dir"] = a.data_dir
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")   # our middleware logs each request


if __name__ == "__main__":
    main()
