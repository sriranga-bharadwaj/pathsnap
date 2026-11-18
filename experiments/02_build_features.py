"""Phase 2a - build the (observation, candidate) feature tables for TRAIN and STUDENT_TEST.

Uses the deployed candidate rule (src/candidates.py: 30 m, K=24, one 100 m retry) and the
alignment labels cached by experiments/01_alignment_candidates.py.

Output: cache/features/<partition>.npz  (git-ignored; rebuilt by this script)
  row-level  : X, y, row_trip, row_obs, row_arc
  obs-level  : obs_trip, obs_idx, obs_label, obs_flag, obs_weight, obs_label_in_cands,
               obs_unmatched, obs_segment
  trip-level : trip_ids, trip_gate, trip_n_obs, trip_feat_ms

Run:  .venv\\Scripts\\python.exe experiments/02_build_features.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import data_root, list_trips
from src.preprocess import prepare_track
from src.network import TripNetwork
from src.candidates import generate_candidates
from src.features import build_features, FEATURES

LABELS = ROOT / "cache" / "labels"
OUT = ROOT / "cache" / "features"


def build_partition(partition):
    pdir = data_root() / partition
    R = {k: [] for k in ["X", "y", "row_trip", "row_obs", "row_arc"]}
    O = {k: [] for k in ["obs_trip", "obs_idx", "obs_label", "obs_flag", "obs_weight",
                         "obs_label_in_cands", "obs_unmatched", "obs_segment"]}
    T = {k: [] for k in ["trip_ids", "trip_gate", "trip_n_obs", "trip_feat_ms"]}
    for ti, tid in enumerate(list_trips(pdir)):
        trk, fwd, _ = prepare_track(pdir / tid, tid)
        net = TripNetwork(pdir / tid, tid, fwd)
        t0 = time.time()
        c = generate_candidates(trk, net)
        X = build_features(trk, net, c)
        ms = (time.time() - t0) * 1000
        lab = np.load(LABELS / partition / f"{tid}.npz")
        label = lab["arc"]
        n = len(label)
        y = (c["arc"] == label[c["obs"]]).astype(np.int8)
        in_c = np.zeros(n, dtype=bool)
        in_c[c["obs"][y == 1]] = True
        R["X"].append(X); R["y"].append(y)
        R["row_trip"].append(np.full(len(y), ti, dtype=np.int32))
        R["row_obs"].append(c["obs"].astype(np.int32)); R["row_arc"].append(c["arc"].astype(np.int64))
        O["obs_trip"].append(np.full(n, ti, dtype=np.int32)); O["obs_idx"].append(np.arange(n, dtype=np.int32))
        O["obs_label"].append(label); O["obs_flag"].append(lab["flag"]); O["obs_weight"].append(lab["weight"])
        O["obs_label_in_cands"].append(in_c); O["obs_unmatched"].append(c["unmatched"])
        O["obs_segment"].append(trk["segment"])
        T["trip_ids"].append(tid)
        T["trip_gate"].append(str(lab["decision"]) if partition == "train" else "n/a")
        T["trip_n_obs"].append(n); T["trip_feat_ms"].append(ms)
        print(f"[{partition}] {tid}: obs={n} rows={len(y)} pos={int(y.sum())} "
              f"label_in_cands={in_c.mean():.4f} cand+feat={ms:.0f} ms", flush=True)
    out = {k: np.concatenate(v) for k, v in {**R, **O}.items()}
    out.update({k: np.array(v) for k, v in T.items()})
    out["feature_names"] = np.array(FEATURES)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{partition}.npz", **out)
    print(f"[{partition}] total obs={len(out['obs_idx'])} rows={len(out['y'])} "
          f"positives={int(out['y'].sum())} ({out['y'].mean():.3f} of rows)")


if __name__ == "__main__":
    for p in ["train", "student_test"]:
        build_partition(p)
