"""Server-side evaluation (Laptop 1), run only AFTER prediction has finished.

Reads the trip's .route (if present), aligns the GPS observations to it with the same
monotonic alignment used for training labels (src/alignment.py), and returns NUMBERS ONLY.
No reference arcs or reference geometry leave this module, so they never reach the
API response or the UI. If there is no .route file, reference-based metrics are null.
"""
import numpy as np

from src.alignment import align_track_to_route
from src.data import has_route, load_route
from src.evaluation import route_continuity


def evaluate(trip_dir, trip_id, internals):
    trk, net, cands, arc = internals["trk"], internals["net"], internals["cands"], internals["arc"]
    metrics = dict(
        top1_accuracy=None,
        route_continuity=route_continuity(arc, trk["segment"], net.a_from, net.a_to, net.n_nodes),
        candidate_miss_rate=None,
        reference_available=False,
    )
    if not has_route(trip_dir, trip_id):
        return metrics
    route = load_route(trip_dir, trip_id)
    valid = (route >= 0) & (route < net.n_arcs)
    route = route[valid]
    if len(route) == 0:
        return metrics
    label = align_track_to_route(trk, net, route)["arc"]
    n = len(label)
    in_cands = np.zeros(n, dtype=bool)
    hit = cands["arc"] == label[cands["obs"]]
    in_cands[cands["obs"][hit]] = True
    metrics.update(top1_accuracy=float(np.mean(arc == label)),
                   candidate_miss_rate=float(1 - in_cands.mean()),
                   reference_available=True)
    return metrics
