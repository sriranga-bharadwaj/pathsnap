"""PathSnap prediction pipeline (runs on Laptop 1 inside the API).

    .track + .nodes/.arcs -> preprocess -> candidates -> features -> classifier -> HMM/Viterbi

LEAKAGE RULE: this module (and everything it imports) never reads the .route file.
Reference-based metrics are computed afterwards by src/server_eval.py.
"""
import json
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np

from src import hmm
from src.candidates import generate_candidates
from src.features import build_features, FEATURES
from src.network import TripNetwork
from src.preprocess import prepare_track

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
LOW_CONFIDENCE = 0.5      # normalised probability below this is flagged in the UI


class PathSnapPredictor:
    def __init__(self, artifacts_dir=ARTIFACTS, cache_size=4):
        a = Path(artifacts_dir)
        self.model = joblib.load(a / "model.joblib")
        self.meta = json.loads((a / "model_meta.json").read_text())
        self.decoder = json.loads((a / "decoder_config.json").read_text())
        assert self.meta["features"] == FEATURES, "feature list of saved model differs from src/features.py"
        # decoder constants live in src/hmm.py; make sure they match the saved config
        for key, val in [("v_max_mps", hmm.V_MAX), ("cutoff_tol_m", hmm.TOL), ("hard_cap_m", hmm.HARD_CAP)]:
            assert abs(self.decoder[key] - val) < 1e-9, f"decoder config mismatch: {key}"
        self.params = {k: self.decoder[k] for k in ("alpha", "beta", "reversal_penalty")}
        self._cache = OrderedDict()   # (trip_dir, trip_id) -> (trk, net, graph): built once per trip
        self._cache_size = cache_size

    def _load_trip(self, trip_dir, trip_id):
        key = (str(trip_dir), trip_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        trk, fwd, _ = prepare_track(trip_dir, trip_id)
        net = TripNetwork(trip_dir, trip_id, fwd)
        item = (trk, net, hmm.RouteGraph(net))
        self._cache[key] = item
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return item

    def predict(self, trip_dir, trip_id):
        """Run the full pipeline on one trip. Returns (response_part, internals).
        internals (trk, net, cands, arc) are kept server-side for evaluation only."""
        t0 = time.perf_counter()
        trk, net, g = self._load_trip(Path(trip_dir), trip_id)
        cands = generate_candidates(trk, net)
        X = build_features(trk, net, cands)
        prob = self.model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
        out = hmm.decode(trk, net, cands, prob, self.params, g=g)
        infer_ms = (time.perf_counter() - t0) * 1000

        lat, lon = trk["lat"], trk["lon"]
        preds = []
        for i in range(len(lat)):
            a = int(out["arc"][i])
            preds.append(dict(observation_index=i, arc_id=a if a >= 0 else None,
                              confidence=round(float(out["confidence"][i]), 4),
                              margin=round(float(out["margin"][i]), 4),
                              transition_type=str(out["transition_type"][i]),
                              fallback=bool(out["fallback"][i])))
        geometry = []
        for piece in out["path_arcs"]:
            nodes = [int(net.a_from[piece[0]])] + [int(net.a_to[a]) for a in piece]
            geometry.append([[round(float(net.node_lat[n]), 6), round(float(net.node_lon[n]), 6)] for n in nodes])
        part = dict(
            gps_points=[[round(float(a), 6), round(float(b), 6)] for a, b in zip(lat, lon)],
            predictions=preds,
            route_geometry=geometry,          # list of polylines [[lat, lon], ...]; split at relaxed jumps / gaps
            events={k: out["events"][k] for k in ("unmatched_points", "relaxed_transitions",
                                                  "multi_hop_transitions", "direct_transitions",
                                                  "reversal_transitions", "segment_starts")},
            low_confidence_points=int(sum(1 for p in preds if p["arc_id"] is not None
                                          and p["confidence"] < LOW_CONFIDENCE)),
            inference_time_ms=round(infer_ms, 1),
        )
        internals = dict(trk=trk, net=net, cands=cands, arc=out["arc"], path_arcs=out["path_arcs"])
        return part, internals
