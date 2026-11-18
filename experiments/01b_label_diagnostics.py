"""Phase 1b - diagnostics on the alignment labels (run after 01_alignment_candidates.py).

1. Progress-inconsistent and far (> 50 m) labels per student_test trip
   (these labels are the yardstick for every model comparison).
2. Trip 00000066 (train): why the aligned far share (9%) exceeds the nearest-route far share (4%):
   list runs of observations that are near SOME route arc but aligned to a far one.
3. Labelled observations whose label arc has an exact duplicate arc (same from/to node,
   different arc id): indistinguishable candidates -> irreducible Top-1 error.

Output: results/phase1/label_diagnostics.txt (printed to console as well)
Run:  .venv\\Scripts\\python.exe experiments/01b_label_diagnostics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import data_root, list_trips, load_route, load_arcs
from src.preprocess import prepare_track
from src.network import TripNetwork
from src.geometry import point_to_segments

CACHE = ROOT / "cache" / "labels"


def runs(mask):
    """Contiguous runs of True as (start, end_inclusive)."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    br = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def main():
    root = data_root()
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    # 1. student_test label quality per trip
    rows = []
    for tid in list_trips(root / "student_test"):
        z = np.load(CACHE / "student_test" / f"{tid}.npz")
        rows.append(dict(trip_id=tid, n=len(z["arc"]), progress_inconsistent=int((~z["progress_consistent"]).sum()),
                         far50=int((z["dist"] > 50).sum()), low=int((z["flag"] == "low").sum()),
                         medium=int((z["flag"] == "medium").sum())))
    say("=== 1. student_test label quality per trip ===")
    say(pd.DataFrame(rows).to_string(index=False))

    # 2. trip 00000066
    tid = "00000066"
    pdir = root / "train"
    trk, fwd, _ = prepare_track(pdir / tid, tid)
    net = TripNetwork(pdir / tid, tid, fwd)
    route = load_route(pdir / tid, tid)
    z = np.load(CACHE / "train" / f"{tid}.npz")
    d_all, _ = point_to_segments(trk["x"][:, None], trk["y"][:, None], net.x0[route][None], net.y0[route][None],
                                 net.x1[route][None], net.y1[route][None])
    near_pos = d_all.argmin(1)
    near_d = d_all.min(1)
    dragged = (z["dist"] > 50) & (near_d <= 50)
    say("\n=== 2. trip 00000066: aligned far but near another route position ===")
    say(f"points aligned > 50 m: {(z['dist'] > 50).sum()}, of which near (<=50 m) another route arc: {dragged.sum()}")
    for s, e in runs(dragged):
        if e - s + 1 < 5:
            continue
        say(f"  obs {s}-{e} ({e - s + 1} pts, t={trk['t'][s]:.0f}-{trk['t'][e]:.0f}s): aligned route_pos "
            f"{z['route_pos'][s]}-{z['route_pos'][e]}, nearest route_pos {near_pos[s]}-{near_pos[e]}, "
            f"aligned dist median {np.median(z['dist'][s:e + 1]):.0f} m")
    say(f"route length {len(route)} arcs; unique arcs {len(np.unique(route))}")

    # 3. duplicate label arcs
    say("\n=== 3. labelled observations whose label arc has an exact duplicate (same from/to nodes) ===")
    tot = {}
    for partition in ["train", "student_test"]:
        n_obs = n_dup = 0
        for tid in list_trips(root / partition):
            z = np.load(CACHE / partition / f"{tid}.npz")
            if partition == "train" and str(z["decision"]) == "exclude":
                continue
            a_from, a_to = load_arcs(root / partition / tid, tid)
            key = a_from * (max(a_from.max(), a_to.max()) + 1) + a_to
            uniq, cnt = np.unique(key, return_counts=True)
            dup_arc = np.isin(key, uniq[cnt > 1])
            n_obs += len(z["arc"])
            n_dup += int(dup_arc[z["arc"]].sum())
        tot[partition] = (n_dup, n_obs)
        say(f"{partition}{' (gate-kept)' if partition == 'train' else ''}: {n_dup} of {n_obs} observations "
            f"({n_dup / n_obs:.4%}) have a duplicate label arc")
    (ROOT / "results" / "phase1" / "label_diagnostics.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
