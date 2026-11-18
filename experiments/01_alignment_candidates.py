"""Phase 1 - route alignment (labels), trip-quality gate, candidate-radius sweep.

Partitions: TRAIN (labels, gate, radius choice) and STUDENT_TEST (confirmation only).
The quality gate thresholds and the radius are decided on TRAIN only.

Outputs:
  cache/labels/<partition>/<trip>.npz       per-observation alignment labels (git-ignored)
  results/phase1/alignment_per_trip.csv     counts of high/medium/low, far share, gate decision
  results/phase1/alignment_summary.csv      totals per partition
  results/phase1/candidate_sweep_per_trip.csv
  results/phase1/candidate_sweep.csv        recall / miss rate / counts / latency per radius
  results/phase1/fig_candidate_sweep.png, fig_quality_gate.png, fig_alignment_dist.png

Run:  .venv\\Scripts\\python.exe experiments/01_alignment_candidates.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import data_root, list_trips, load_route
from src.preprocess import prepare_track
from src.network import TripNetwork
from src.alignment import (align_track_to_route, add_margin, confidence_flags, trip_quality,
                           GATE_EXCLUDE_FAR_SHARE, GATE_WARN_FAR_SHARE, FAR_M)
from src.candidates import generate_candidates, K_MAX, RETRY_RADIUS

OUT = ROOT / "results" / "phase1"
CACHE = ROOT / "cache" / "labels"
RADII = [15.0, 30.0, 50.0]          # fixed sweep (no adaptive variants)
KNEE_TOL = 0.005                    # pick smallest radius within 0.5 pp of the 50 m recall (TRAIN)


def label_recall(cands, label_arc, mask):
    """Share of observations (in mask) whose aligned label arc is in their candidate set."""
    n_arcs_code = int(max(cands["arc"].max(initial=0), label_arc.max()) + 1)
    pair = cands["obs"].astype(np.int64) * n_arcs_code + cands["arc"]
    obs = np.arange(len(label_arc))
    hit = np.isin(obs.astype(np.int64) * n_arcs_code + label_arc, pair)
    return hit[mask]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = data_root()
    al_rows, sw_rows = [], []
    all_dist = {"train": [], "student_test": []}
    counts_by_cfg = {}     # (set, cap, radius) -> list of per-observation candidate counts (pooled p95)
    for partition in ["train", "student_test"]:
        pdir = root / partition
        (CACHE / partition).mkdir(parents=True, exist_ok=True)
        for tid in list_trips(pdir):
            t0 = time.time()
            trk, fwd, _ = prepare_track(pdir / tid, tid)
            net = TripNetwork(pdir / tid, tid, fwd)
            t_net = time.time() - t0
            route = load_route(pdir / tid, tid)        # labels only (offline)
            t1 = time.time()
            al = align_track_to_route(trk, net, route)
            al = add_margin(al, trk, net)
            flag, weight = confidence_flags(al)
            far_share, decision = trip_quality(al)
            t_align = time.time() - t1
            np.savez_compressed(CACHE / partition / f"{tid}.npz", arc=al["arc"], dist=al["dist"],
                                margin=al["margin"], route_pos=al["route_pos"],
                                heading_diff_deg=al["heading_diff_deg"],
                                progress_consistent=al["progress_consistent"],
                                flag=flag, weight=weight, decision=decision)
            all_dist[partition].append(al["dist"])
            n = len(flag)
            al_rows.append(dict(
                partition=partition, trip_id=tid, n_points=n, n_route_arcs=len(route),
                n_segments=int(trk["segment"].max() + 1),
                high=int((flag == "high").sum()), medium=int((flag == "medium").sum()),
                low=int((flag == "low").sum()),
                dist_p50=float(np.median(al["dist"])), dist_p95=float(np.percentile(al["dist"], 95)),
                far_share=far_share, progress_inconsistent=int((~al["progress_consistent"]).sum()),
                margin_p50=float(np.median(al["margin"])),
                route_arcs_used=int(len(np.unique(al["route_pos"]))),
                gate=decision if partition == "train" else "n/a (not gated)",
                net_build_s=round(t_net, 2), align_s=round(t_align, 2)))
            high = flag == "high"
            for r in RADII:
                for k, tag in [(K_MAX, "capped"), (10**6, "uncapped")]:
                    t2 = time.time()
                    c = generate_candidates(trk, net, radius=r, k_max=k, retry_radius=RETRY_RADIUS)
                    ms = (time.time() - t2) * 1000
                    cnt = np.bincount(c["obs"], minlength=n)
                    set_name = ("student_test" if partition == "student_test" else
                                "train_excluded" if decision == "exclude" else "train_kept")
                    counts_by_cfg.setdefault((set_name, tag, r), []).append(cnt)
                    hit_all = label_recall(c, al["arc"], np.ones(n, bool))
                    hit_high = label_recall(c, al["arc"], high)
                    sw_rows.append(dict(
                        partition=partition, trip_id=tid, gate=al_rows[-1]["gate"], radius=r, cap=tag,
                        n_points=n, n_high=int(high.sum()), hits=int(hit_all.sum()),
                        hits_high=int(hit_high.sum()), retried_obs=int(len(np.unique(c["obs"][c["retry"]]))),
                        unmatched=int(c["unmatched"].sum()), cand_mean=float(cnt.mean()),
                        cand_p95=float(np.percentile(cnt, 95)), cand_max=int(cnt.max()),
                        cand_total=int(len(c["obs"])), latency_ms=ms))
            print(f"[{partition}] {tid}: n={n} H/M/L={al_rows[-1]['high']}/{al_rows[-1]['medium']}/"
                  f"{al_rows[-1]['low']} far={far_share:.3f} gate={al_rows[-1]['gate']} "
                  f"net={t_net:.1f}s align={t_align:.1f}s", flush=True)

    al_df = pd.DataFrame(al_rows)
    al_df.to_csv(OUT / "alignment_per_trip.csv", index=False)
    sw = pd.DataFrame(sw_rows)
    sw.to_csv(OUT / "candidate_sweep_per_trip.csv", index=False)

    # ---------- alignment summary ----------
    al_df["set"] = np.where(al_df.partition == "student_test", "student_test",
                            np.where(al_df.gate == "exclude", "train_excluded", "train_kept"))
    s = al_df.groupby("set")[["n_points", "high", "medium", "low", "progress_inconsistent"]].sum()
    s["trips"] = al_df.groupby("set").size()
    for f_ in ["high", "medium", "low"]:
        s[f"{f_}_share"] = s[f_] / s["n_points"]
    s.to_csv(OUT / "alignment_summary.csv")

    # ---------- candidate sweep summary (pooled over observations) ----------
    sw["set"] = np.where(sw.partition == "student_test", "student_test",
                         np.where(sw.gate == "exclude", "train_excluded", "train_kept"))
    g = sw.groupby(["set", "cap", "radius"])
    summ = g.agg(trips=("trip_id", "count"), n_points=("n_points", "sum"), hits=("hits", "sum"),
                 n_high=("n_high", "sum"), hits_high=("hits_high", "sum"),
                 retried_obs=("retried_obs", "sum"), unmatched=("unmatched", "sum"),
                 cand_total=("cand_total", "sum"), latency_ms_total=("latency_ms", "sum"))
    # recall = label arc in candidate set, for the DEPLOYED configuration
    # (radius + one retry at RETRY_RADIUS for observations with no candidate)
    summ["recall_all"] = summ.hits / summ.n_points
    summ["miss_rate_all"] = 1 - summ.recall_all
    # NOTE: recall_high is 1.0 by construction ("high" requires aligned distance <= 15 m,
    # and every radius >= 15 m). Kept only as a sanity check, not as a finding.
    summ["recall_high"] = summ.hits_high / summ.n_high
    summ["cand_mean"] = summ.cand_total / summ.n_points
    summ = summ.reset_index()
    # pooled 95th percentile of candidates per observation (over all observations of the set)
    summ["cand_p95_pooled"] = [float(np.percentile(np.concatenate(counts_by_cfg[(s_, c_, r_)]), 95))
                               for s_, c_, r_ in zip(summ["set"], summ.cap, summ.radius)]
    summ["cand_max"] = [int(np.concatenate(counts_by_cfg[(s_, c_, r_)]).max())
                        for s_, c_, r_ in zip(summ["set"], summ.cap, summ.radius)]
    summ["latency_ms_per_1000_obs"] = summ.latency_ms_total / summ.n_points * 1000
    summ.to_csv(OUT / "candidate_sweep.csv", index=False)

    # knee: smallest radius whose TRAIN (kept trips, capped) recall is within KNEE_TOL of 50 m recall
    tr = summ[(summ["set"] == "train_kept") & (summ.cap == "capped")].set_index("radius")
    best = tr.loc[max(RADII), "recall_all"]
    knee = min(r for r in RADII if tr.loc[r, "recall_all"] >= best - KNEE_TOL)

    # ---------- figures ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for st, col in [("train_kept", "#2166ac"), ("student_test", "#d6604d")]:
        d = summ[(summ["set"] == st) & (summ.cap == "capped")]
        ax[0].plot(d.radius, d.recall_all, "o-", color=col, label=f"{st} (all obs)")
        ax[0].plot(d.radius, d.recall_high, "s--", color=col, label=f"{st} (high-conf)")
        ax[1].plot(d.cand_mean, d.recall_all, "o-", color=col, label=st)
        for _, r in d.iterrows():
            ax[1].annotate(f"{r.radius:.0f} m", (r.cand_mean, r.recall_all), fontsize=8)
    ax[0].set_xlabel("candidate radius (m)"); ax[0].set_ylabel("candidate recall (label arc in set)")
    ax[0].set_title(f"Recall vs radius (cap K={K_MAX})"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("mean candidates per observation"); ax[1].set_ylabel("recall")
    ax[1].set_title("Recall vs candidate-set size"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_candidate_sweep.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    d = al_df.sort_values("far_share", ascending=False)
    colors = ["#d6604d" if p == "student_test" else ("#000000" if g_ == "exclude" else "#2166ac")
              for p, g_ in zip(d.partition, d.gate)]
    ax.bar(range(len(d)), d.far_share, color=colors)
    ax.axhline(GATE_EXCLUDE_FAR_SHARE, color="k", ls="--", lw=1, label="exclude threshold (train)")
    ax.axhline(GATE_WARN_FAR_SHARE, color="grey", ls=":", lw=1, label="warn threshold (train)")
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d.trip_id.str[-3:], rotation=90, fontsize=6)
    ax.set_ylabel(f"share of fixes aligned > {FAR_M:.0f} m from route")
    ax.set_title("Trip quality gate (black = excluded train, blue = kept train, red = student_test)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_quality_gate.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for p, col in [("train", "#2166ac"), ("student_test", "#d6604d")]:
        dd = np.concatenate(all_dist[p])
        ax.hist(np.clip(dd, 0, 100), bins=100, density=True, alpha=0.6, log=True, color=col, label=p)
    ax.set_xlabel("distance fix -> aligned route arc (m, clipped at 100)"); ax.set_ylabel("density (log)")
    ax.set_title("Alignment distance"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_alignment_dist.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\n=== alignment summary ===")
    print(s.round(4).to_string())
    print("\n=== quality gate (train) ===")
    print(al_df[al_df.gate != "keep"][["partition", "trip_id", "n_points", "far_share", "gate"]]
          .sort_values("far_share", ascending=False).to_string())
    print("\n=== candidate sweep ===")
    cols = ["set", "cap", "radius", "trips", "n_points", "recall_all", "miss_rate_all", "recall_high",
            "cand_mean", "cand_p95_pooled", "cand_max", "retried_obs", "unmatched",
            "latency_ms_per_1000_obs"]
    print(summ[cols].round(4).to_string())
    print(f"\nKnee rule (TRAIN kept, capped): smallest radius with recall >= recall@50m - {KNEE_TOL}"
          f" -> {knee:.0f} m (recall@50m = {best:.4f})")


if __name__ == "__main__":
    main()
