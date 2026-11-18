"""Phase 3 - HMM/Viterbi: decoder tuning on OUT-OF-FOLD probabilities (train), then
classifier-only vs classifier+HMM on STUDENT_TEST.

Tuning data: the 77 gate-kept TRAIN trips with the out-of-fold RandomForest probabilities
from experiments/03 (in-sample probabilities would be over-confident). Grid over
alpha (emission exponent), beta (transition scale, m) and the reversal penalty; objective =
pooled observation-level Top-1 after decoding. Distance-dependent quantities (shortest paths)
do not depend on these parameters, so they are computed once per trip and reused.
Fixed (not tuned): candidate radius 30 m (Phase 1 sweep; OOF probabilities exist only for
that candidate set), Dijkstra cutoff = min(50 m/s * dt + 150 m, 3000 m) (physical bound).

Outputs (results/phase3/): decoder_grid.csv, fig_decoder_grid.png,
  per_trip_comparison.csv, comparison_summary.csv, fig_hmm_vs_classifier.png
  artifacts/decoder_config.json
Run:  .venv\\Scripts\\python.exe experiments/04_hmm_tune_compare.py
"""
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import data_root, load_route
from src.preprocess import prepare_track
from src.network import TripNetwork
from src.features import FEATURES
from src.hmm import precompute, viterbi, decode, V_MAX, TOL, HARD_CAP, RELAX_CUTOFF, BACKWARD_TOL, JUMP_PENALTY
from src.evaluation import (top1_predictions, normalise_per_obs, route_continuity, arc_set_prf,
                            norm_edit_distance, compress)

OUT = ROOT / "results" / "phase3"
FEAT = ROOT / "cache" / "features"
# Run 1 (alpha 0.5-2, penalty 2-10) -> optimum on the edge (alpha=0.5, penalty=10, top1 0.9093).
# Run 2 (alpha 0.25-2, penalty 5-40) -> still on the edge (alpha=0.25, penalty=40, top1 0.9156).
# Run 3 (this grid) widens again toward smaller alpha / larger penalty -> alpha=0.1, beta=20, penalty=20.
# Note: Viterbi's argmax is unchanged when the whole score is divided by alpha, so the decision
# depends only on alpha*beta (here 2 m) and penalty/alpha. Points along that ridge
# (e.g. alpha=1, beta=2, penalty=80: 0.9163) score almost the same; the "edge" is a scale redundancy.
GRID = dict(alpha=[0.1, 0.25, 0.5, 1.0], beta=[2.0, 5.0, 10.0, 20.0], reversal_penalty=[10.0, 20.0, 40.0, 80.0])
MODEL_NAME = json.loads((ROOT / "artifacts" / "model_meta.json").read_text())["model"]


def trip_cands(D, t):
    """Candidate rows of trip t from a feature table (rows are contiguous per trip, sorted by obs)."""
    r = np.flatnonzero(D["row_trip"] == t)
    X = D["X"][r]
    c = dict(obs=D["row_obs"][r].astype(np.int64), arc=D["row_arc"][r],
             dist=X[:, FEATURES.index("dist")].astype(float), frac=X[:, FEATURES.index("frac")].astype(float))
    return r, c


def pre_trip(partition, tid, c):
    trk, fwd, _ = prepare_track(data_root() / partition / tid, tid)
    net = TripNetwork(data_root() / partition / tid, tid, fwd)
    t0 = time.time()
    segs, _ = precompute(trk, net, c)
    return segs, time.time() - t0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TR = dict(np.load(FEAT / "train.npz"))
    oof = np.load(ROOT / "cache" / "oof" / f"{MODEL_NAME}.npz")["oof"]
    tr_start = np.r_[0, np.cumsum(TR["trip_n_obs"])[:-1]]
    kept = [t for t in range(len(TR["trip_ids"])) if TR["trip_gate"][t] != "exclude"]

    # ---------- 1. precompute shortest-path tables for all kept train trips (parallel) ----------
    t0 = time.time()
    jobs = []
    for t in kept:
        r, c = trip_cands(TR, t)
        assert np.isfinite(oof[r]).all(), "missing OOF probability"
        jobs.append((t, r, c))
    pre = Parallel(n_jobs=6, verbose=0)(delayed(pre_trip)("train", TR["trip_ids"][t], c) for t, r, c in jobs)
    print(f"precompute: {len(kept)} train trips in {time.time() - t0:.0f}s", flush=True)

    # ---------- 2. grid search on OOF probabilities ----------
    labels = {t: TR["obs_label"][tr_start[t]:tr_start[t] + TR["trip_n_obs"][t]] for t in kept}
    pn = {t: normalise_per_obs(c["obs"], oof[r], TR["trip_n_obs"][t]) for t, r, c in jobs}
    # classifier-only reference on the same trips
    clf_correct = sum(int((top1_predictions(c["obs"], c["arc"], oof[r], TR["trip_n_obs"][t])[0]
                           == labels[t]).sum()) for t, r, c in jobs)
    n_total = sum(TR["trip_n_obs"][t] for t in kept)
    grid_rows = []
    for alpha, beta, rp in itertools.product(*GRID.values()):
        params = dict(alpha=alpha, beta=beta, reversal_penalty=rp)
        t1 = time.time()
        correct, relaxed = 0, 0
        for (t, r, c), (segs, _) in zip(jobs, pre):
            chosen, ttype = viterbi(segs, pn[t], params)
            pred = np.full(TR["trip_n_obs"][t], -1)
            for o, row in chosen.items():
                pred[o] = c["arc"][row]
            correct += int((pred == labels[t]).sum())
            relaxed += sum(1 for v in ttype.values() if v == "relaxed")
        grid_rows.append(dict(**params, train_oof_hmm_top1=correct / n_total,
                              train_oof_clf_top1=clf_correct / n_total, relaxed=relaxed,
                              seconds=time.time() - t1))
        print(f"  alpha={alpha} beta={beta} rp={rp}: top1={correct / n_total:.4f} relaxed={relaxed}", flush=True)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "decoder_grid.csv", index=False)
    best = grid.sort_values("train_oof_hmm_top1", ascending=False).iloc[0]
    params = dict(alpha=float(best.alpha), beta=float(best.beta), reversal_penalty=float(best.reversal_penalty))
    print(f"best decoder params (train OOF): {params} top1={best.train_oof_hmm_top1:.4f} "
          f"(classifier-only OOF top1={clf_correct / n_total:.4f})")
    cfg = dict(**params, v_max_mps=V_MAX, cutoff_tol_m=TOL, hard_cap_m=HARD_CAP, relax_cutoff_m=RELAX_CUTOFF,
               backward_tol_m=BACKWARD_TOL, jump_penalty=JUMP_PENALTY, gap_break_s=60.0,
               tuned_on=f"train OOF {MODEL_NAME} probabilities, 77 gate-kept trips",
               created=datetime.now().isoformat(timespec="seconds"))
    (ROOT / "artifacts" / "decoder_config.json").write_text(json.dumps(cfg, indent=2))

    fig, axes = plt.subplots(1, len(GRID["reversal_penalty"]), figsize=(13, 3.6), sharey=True)
    for ax, rp in zip(axes, GRID["reversal_penalty"]):
        g = grid[grid.reversal_penalty == rp].pivot(index="alpha", columns="beta", values="train_oof_hmm_top1")
        im = ax.imshow(g.values, cmap="viridis", vmin=grid.train_oof_hmm_top1.min(), vmax=grid.train_oof_hmm_top1.max())
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, f"{g.values[i, j]:.4f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_xticks(range(g.shape[1])); ax.set_xticklabels(g.columns); ax.set_xlabel("beta (m)")
        ax.set_yticks(range(g.shape[0])); ax.set_yticklabels(g.index); ax.set_title(f"reversal penalty {rp}")
    axes[0].set_ylabel("alpha")
    fig.suptitle("Decoder grid: train out-of-fold Top-1 after Viterbi")
    fig.tight_layout(); fig.savefig(OUT / "fig_decoder_grid.png", dpi=130); plt.close(fig)

    # ---------- 3. student-test: classifier-only vs classifier + HMM ----------
    TE = dict(np.load(FEAT / "student_test.npz"))
    te_start = np.r_[0, np.cumsum(TE["trip_n_obs"])[:-1]]
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    rows = []
    for t, tid in enumerate(TE["trip_ids"]):
        pdir = data_root() / "student_test"
        r, c = trip_cands(TE, t)
        n = TE["trip_n_obs"][t]
        lab = TE["obs_label"][te_start[t]:te_start[t] + n]
        high = TE["obs_flag"][te_start[t]:te_start[t] + n] == "high"
        trk, fwd, _ = prepare_track(pdir / tid, tid)
        net = TripNetwork(pdir / tid, tid, fwd)
        t1 = time.time()
        prob = model.predict_proba(TE["X"][r])[:, 1]
        clf_ms = (time.time() - t1) * 1000
        pred_c, _ = top1_predictions(c["obs"], c["arc"], prob, n)
        t2 = time.time()
        out = decode(trk, net, c, prob, params)
        hmm_ms = (time.time() - t2) * 1000
        route = load_route(pdir / tid, tid)         # evaluation only, after prediction
        nn = net.n_nodes
        hmm_path = np.concatenate(out["path_arcs"]) if out["path_arcs"] else np.array([], int)
        clf_seq = compress(pred_c[pred_c >= 0])
        ref = compress(route)
        for method, pred, seq in [("classifier_only", pred_c, clf_seq), ("classifier+hmm", out["arc"], hmm_path)]:
            p_, r_, f_ = arc_set_prf(seq, route)
            rows.append(dict(
                trip_id=tid, method=method, n_obs=n, top1=float((pred == lab).mean()),
                top1_high=float((pred == lab)[high].mean()),
                route_continuity=route_continuity(pred, trk["segment"], net.a_from, net.a_to, nn),
                arc_precision=p_, arc_recall=r_, arc_f1=f_,
                norm_edit_distance=norm_edit_distance(compress(seq), ref),
                runtime_ms=clf_ms + (hmm_ms if method == "classifier+hmm" else 0.0),
                **({k: v for k, v in out["events"].items()} if method == "classifier+hmm" else {})))
        print(f"[student_test] {tid}: clf top1={rows[-2]['top1']:.4f} hmm top1={rows[-1]['top1']:.4f} "
              f"arcF1 {rows[-2]['arc_f1']:.3f}->{rows[-1]['arc_f1']:.3f} events={out['events']} "
              f"decode={hmm_ms:.0f}ms", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_trip_comparison.csv", index=False)
    metrics = ["top1", "top1_high", "route_continuity", "arc_precision", "arc_recall", "arc_f1",
               "norm_edit_distance", "runtime_ms"]
    macro = per.groupby("method")[metrics].mean().add_suffix("_trip_macro")
    pooled = per.groupby("method").apply(
        lambda d: pd.Series(dict(top1_pooled=(d.top1 * d.n_obs).sum() / d.n_obs.sum())), include_groups=False)
    ev = per[per.method == "classifier+hmm"][["direct_transitions", "multi_hop_transitions", "reversal_transitions",
                                              "relaxed_transitions", "unmatched_points", "segment_starts"]].sum()
    summary = macro.join(pooled)
    summary.to_csv(OUT / "comparison_summary.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    piv = per.pivot(index="trip_id", columns="method", values="top1")
    piv.plot.bar(ax=ax[0], color=["#4393c3", "#d6604d"]); ax[0].set_ylim(0.5, 1); ax[0].set_ylabel("Top-1")
    ax[0].set_title("Student-test Top-1 per trip")
    piv = per.pivot(index="trip_id", columns="method", values="arc_f1")
    piv.plot.bar(ax=ax[1], color=["#4393c3", "#d6604d"]); ax[1].set_ylim(0, 1); ax[1].set_ylabel("arc-set F1")
    ax[1].set_title("Student-test route-level arc-set F1 per trip")
    fig.tight_layout(); fig.savefig(OUT / "fig_hmm_vs_classifier.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\n=== decoder grid (top 5) ===")
    print(grid.sort_values("train_oof_hmm_top1", ascending=False).head(5).round(4).to_string(index=False))
    print("\n=== student-test summary ===")
    print(summary.round(4).T.to_string())
    print("\n=== HMM transition counts (student-test, all trips) ===")
    print(ev.to_string())


if __name__ == "__main__":
    main()
