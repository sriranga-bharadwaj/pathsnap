"""Align GPS observations to the reference route (.route) -> one label arc per observation.

USED ONLY FOR: (a) training labels, (b) server-side evaluation AFTER prediction.
Never imported by the prediction pipeline (candidates / features / models / hmm).

Method: monotonic dynamic programming. The route is an ordered list of arcs
r_0 .. r_{R-1}. Each observation i is assigned a route position j(i) with
j(0) <= j(1) <= ... (the vehicle never goes back along the route). Positions may be
skipped (GPS is sparse: ~5 fixes per arc, sometimes < 2), and several fixes may share
one arc. Cost of assigning observation i to position j:
    c(i, j) = min(dist(i, arc r_j), D_CLIP) + HEADING_W * (1 - cos(heading diff)) / 2
The heading term separates the two passes when a route uses the same road in both
directions. Total cost is minimised exactly with a prefix-minimum recurrence:
    C(i, j) = c(i, j) + min_{k <= j} C(i-1, k)
"""
import numpy as np

from src.geometry import point_to_segments, angle_diff

D_CLIP = 100.0        # metres; far points stop dominating the total cost
HEADING_W = 30.0      # metres of penalty for travelling against the arc direction
MIN_SPEED_HEADING = 2.0   # m/s; below this the movement heading is unreliable

# Confidence rules (thresholds in metres / degrees)
HIGH_D, MED_D = 15.0, 30.0
HIGH_HEADING_DEG = 45.0
PROGRESS_TOL = 10.0   # aligned cost may exceed the best route position by this much
FAR_M = 50.0
WEIGHTS = {"high": 1.0, "medium": 0.5, "low": 0.2}


def _cost_row(i, trk, net, route):
    d, _ = point_to_segments(trk["x"][i], trk["y"][i], net.x0[route], net.y0[route],
                             net.x1[route], net.y1[route])
    c = np.minimum(d, D_CLIP)
    if trk["heading_ok"][i] and trk["speed"][i] >= MIN_SPEED_HEADING:
        dh = angle_diff(trk["heading"][i], net.heading[route])
        c = c + np.where(net.heading_valid[route], HEADING_W * (1 - np.cos(dh)) / 2, 0.0)
    return c, d


def align_track_to_route(trk, net, route):
    """Return dict of per-observation alignment results (length N arrays)."""
    n, R = len(trk["x"]), len(route)
    ptr_dtype = np.uint16 if R < 65535 else np.int32
    ptr = np.zeros((n, R), dtype=ptr_dtype)       # back-pointers (prefix argmin)
    idx = np.arange(R)
    prev, _ = _cost_row(0, trk, net, route)
    for i in range(1, n):
        c, _ = _cost_row(i, trk, net, route)
        pm = np.minimum.accumulate(prev)                  # best previous cost with k <= j
        # index of that best previous position (last index where the running min was set)
        ptr[i] = np.maximum.accumulate(np.where(prev == pm, idx, 0))
        prev = c + pm
    # backtrack
    pos = np.zeros(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = ptr[i, pos[i]]
    del ptr

    arc = route[pos]
    dist = np.empty(n)
    head_deg = np.full(n, np.nan)
    consistent = np.empty(n, dtype=bool)
    for i in range(n):
        c, d = _cost_row(i, trk, net, route)
        dist[i] = d[pos[i]]
        consistent[i] = c[pos[i]] <= c.min() + PROGRESS_TOL
        if trk["heading_ok"][i] and trk["speed"][i] >= MIN_SPEED_HEADING and net.heading_valid[arc[i]]:
            head_deg[i] = np.degrees(angle_diff(trk["heading"][i], net.heading[arc[i]]))
    return dict(route_pos=pos, arc=arc, dist=dist, heading_diff_deg=head_deg,
                progress_consistent=consistent)


def add_margin(al, trk, net, radius=FAR_M):
    """Margin = distance to the nearest network arc that is neither the aligned arc nor its
    reverse twin, minus the aligned distance. Searched within `radius` m (capped there)."""
    n = len(trk["x"])
    pi, ai = net.arcs_within(trk["x"], trk["y"], radius)
    d, _ = point_to_segments(trk["x"][pi], trk["y"][pi], net.x0[ai], net.y0[ai], net.x1[ai], net.y1[ai])
    other = (ai != al["arc"][pi]) & (ai != net.twin[al["arc"][pi]])
    best_other = np.full(n, radius)
    np.minimum.at(best_other, pi[other], d[other])
    al["margin"] = best_other - al["dist"]
    return al


def confidence_flags(al):
    """high / medium / low per observation, and the matching sample weight."""
    hd = al["heading_diff_deg"]
    heading_bad = np.nan_to_num(hd, nan=0.0) > HIGH_HEADING_DEG   # unknown heading is not penalised
    high = (al["dist"] <= HIGH_D) & ~heading_bad & al["progress_consistent"]
    med = ~high & (al["dist"] <= MED_D)
    flag = np.where(high, "high", np.where(med, "medium", "low"))
    weight = np.vectorize(WEIGHTS.get)(flag).astype(float)
    return flag, weight


# ---------------- trip-level quality gate (training data only) ----------------
GATE_EXCLUDE_FAR_SHARE = 0.10   # > 10% of fixes beyond 50 m of their own route -> exclude trip
GATE_WARN_FAR_SHARE = 0.05      # 5-10% -> keep, but those points already carry low weight


def trip_quality(al):
    far_share = float(np.mean(al["dist"] > FAR_M))
    if far_share > GATE_EXCLUDE_FAR_SHARE:
        decision = "exclude"
    elif far_share > GATE_WARN_FAR_SHARE:
        decision = "keep_warn"
    else:
        decision = "keep"
    return far_share, decision
