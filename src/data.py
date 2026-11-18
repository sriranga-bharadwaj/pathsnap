"""Raw-data loading and per-trip projection for PathSnap.

File formats (verified in experiments/00_eda.py, see results/eda/):
  NNNNNNNN.track : lon, lat, t   (whitespace separated, no header, CRLF line endings)
  NNNNNNNN.nodes : lon, lat      (row index = node_id, 0-based)
  NNNNNNNN.arcs  : from_node, to_node  (row index = arc_id, 0-based, directed)
  NNNNNNNN.route : arc_id per line (reference route, chronological)

IMPORTANT (leakage rule): load_route() is only for training labels and for the
separate server-side evaluation module. The prediction pipeline must never call it.
"""
import os
from pathlib import Path

import numpy as np
from pyproj import Transformer

# Default dataset root; override with env var DATA_ROOT.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Student folder" / "Student folder"


def data_root():
    return Path(os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT))


def list_trips(partition_dir):
    """Return sorted trip IDs (folder names) inside a partition folder."""
    p = Path(partition_dir)
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def _read_table(path, ncols, dtype):
    """Read a whitespace-separated numeric table. Handles CRLF and LF.
    Raises ValueError if any row does not have exactly `ncols` values."""
    rows = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f):
            parts = line.split()          # split() also strips '\r'
            if not parts:
                continue                  # ignore blank lines
            if len(parts) != ncols:
                raise ValueError(f"{path}: line {line_no} has {len(parts)} columns, expected {ncols}")
            rows.append(parts)
    return np.array(rows, dtype=dtype).reshape(-1, ncols)


def load_track(trip_dir, trip_id):
    """GPS observations -> (lon, lat, t) float64 arrays in file order."""
    a = _read_table(Path(trip_dir) / f"{trip_id}.track", 3, np.float64)
    return a[:, 0], a[:, 1], a[:, 2]


def load_nodes(trip_dir, trip_id):
    a = _read_table(Path(trip_dir) / f"{trip_id}.nodes", 2, np.float64)
    return a[:, 0], a[:, 1]


def load_arcs(trip_dir, trip_id):
    a = _read_table(Path(trip_dir) / f"{trip_id}.arcs", 2, np.int64)
    return a[:, 0], a[:, 1]


def has_route(trip_dir, trip_id):
    return (Path(trip_dir) / f"{trip_id}.route").exists()


def load_route(trip_dir, trip_id):
    """Reference arc sequence. TRAINING LABELS / EVALUATION ONLY - never in prediction."""
    return _read_table(Path(trip_dir) / f"{trip_id}.route", 1, np.int64)[:, 0]


def make_projection(lon, lat):
    """Local azimuthal-equidistant projection centred on the trip's median GPS position.

    Why: trips are spread worldwide, so no single UTM zone fits. AEQD centred on the
    trip gives metres, exact distances from the centre, and negligible distortion
    within a few tens of km (checked numerically in 00_eda.py).
    Returns (forward, inverse): forward(lon, lat) -> (x, y) metres; inverse(x, y) -> (lon, lat).
    """
    lon0, lat0 = float(np.median(lon)), float(np.median(lat))
    crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return (lambda lo, la: fwd.transform(lo, la)), (lambda x, y: inv.transform(x, y))
