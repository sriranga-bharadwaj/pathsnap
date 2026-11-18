"""Phase 5b - final evaluation of the DEPLOYED system on STUDENT_TEST (exactly the API code path:
src/pipeline.PathSnapPredictor), per-trip table, error analysis and Folium maps.

Offline analysis only: the maps here MAY show the reference route (they are never served by the API).

Error categories for every wrong observation (first matching rule wins):
  unmatched       no candidate at all (prediction is null)
  candidate_miss  the reference arc was not in the candidate set (decoder could not pick it)
  direction_flip  predicted the reverse twin of the reference arc (same road, wrong direction)
  gap_fallback    relaxed transition, segment start, or within 3 fixes after a time step > 10 s
  adjacent_arc    predicted arc touches the reference arc at a node (junction / arc-boundary ambiguity)
  parallel_road   any other arc (a different, nearby road)

Outputs (results/phase5/):
  final_per_trip.csv, final_summary.csv, error_categories.csv,
  fig_final_per_trip.png, fig_error_categories.png, map_worst_<trip>.html (3 worst trips)
Run:  .venv\\Scripts\\python.exe experiments/06_final_analysis.py
"""
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import shapely
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import data_root, list_trips, load_route
from src.pipeline import PathSnapPredictor
from src.alignment import align_track_to_route
from src.evaluation import route_continuity, arc_set_prf, norm_edit_distance, compress

OUT = ROOT / "results" / "phase5"
CATS = ["unmatched", "candidate_miss", "direction_flip", "gap_fallback", "adjacent_arc", "parallel_road"]
CAT_COLORS = {"correct": "#2166ac", "unmatched": "#000000", "candidate_miss": "#7b3294",
              "direction_flip": "#d62728", "gap_fallback": "#ff7f0e", "adjacent_arc": "#e7ba52",
              "parallel_road": "#e377c2"}


def categorise(pred, label, in_cands, ttype, trk, net):
    n = len(pred)
    cat = np.array(["correct"] * n, dtype=object)
    wrong = pred != label
    after_gap = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(trk["dt"] > 10):
        after_gap[i:i + 3] = True
    for i in np.flatnonzero(wrong):
        p, l = pred[i], label[i]
        if p < 0:
            c = "unmatched"
        elif not in_cands[i]:
            c = "candidate_miss"
        elif net.twin[l] == p:
            c = "direction_flip"
        elif ttype[i] in ("relaxed", "start") or after_gap[i]:
            c = "gap_fallback"
        elif {net.a_from[p], net.a_to[p]} & {net.a_from[l], net.a_to[l]}:
            c = "adjacent_arc"
        else:
            c = "parallel_road"
        cat[i] = c
    return cat


def make_map(tid, trk, net, route, geometry, cat, top1, out_html, corridor_m=150.0):
    lat, lon = trk["lat"], trk["lon"]
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=14, control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    _, hit = net.tree.query(shapely.points(trk["x"], trk["y"]), predicate="dwithin", distance=corridor_m)
    fg = folium.FeatureGroup(name="Road network (within 150 m)", show=True)
    for a in np.unique(hit):
        folium.PolyLine([[net.node_lat[net.a_from[a]], net.node_lon[net.a_from[a]]],
                         [net.node_lat[net.a_to[a]], net.node_lon[net.a_to[a]]]],
                        color="#9e9e9e", weight=1, opacity=0.6).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup(name="Reference route (.route, offline only)")
    pts = [[net.node_lat[net.a_from[route[0]]], net.node_lon[net.a_from[route[0]]]]]
    for i, a in enumerate(route):
        pts.append([net.node_lat[net.a_to[a]], net.node_lon[net.a_to[a]]])
        if i + 1 < len(route) and net.a_to[a] != net.a_from[route[i + 1]]:
            folium.PolyLine(pts, color="#1a9850", weight=7, opacity=0.5).add_to(fg)
            pts = [[net.node_lat[net.a_from[route[i + 1]]], net.node_lon[net.a_from[route[i + 1]]]]]
    folium.PolyLine(pts, color="#1a9850", weight=7, opacity=0.5).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup(name="Predicted route (PathSnap)")
    for line in geometry:
        folium.PolyLine(line, color="#6a3d9a", weight=3, opacity=0.9).add_to(fg)
    fg.add_to(m)
    fg_ok = folium.FeatureGroup(name="GPS: correct", show=False)
    fg_bad = folium.FeatureGroup(name="GPS: errors (by category)")
    for i in range(len(lat)):
        c = cat[i]
        folium.CircleMarker([lat[i], lon[i]], radius=2 if c == "correct" else 4, color=CAT_COLORS[c],
                            fill=True, fill_opacity=0.9, weight=0,
                            tooltip=f"obs {i}: {c}").add_to(fg_ok if c == "correct" else fg_bad)
    fg_ok.add_to(m); fg_bad.add_to(m)
    items = "".join(f'<span style="color:{CAT_COLORS[k]}">&#9679;</span> {k}<br>' for k in ["correct"] + CATS)
    legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
      padding:8px 10px;border:1px solid #999;font-size:12px">
      <b>student_test trip {tid}</b> - Top-1 {top1:.4f}<br>
      <span style="color:#1a9850">&#9644;</span> reference route &nbsp;
      <span style="color:#6a3d9a">&#9644;</span> predicted route &nbsp;
      <span style="color:#9e9e9e">&#9644;</span> network<br>{items}</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred_ = PathSnapPredictor()
    pdir = data_root() / "student_test"
    rows, cat_rows, keep = [], [], {}
    for tid in list_trips(pdir):
        part, inn = pred_.predict(pdir / tid, tid)        # exactly the API prediction path
        trk, net, cands, arc = inn["trk"], inn["net"], inn["cands"], inn["arc"]
        route = load_route(pdir / tid, tid)               # evaluation only, after prediction
        label = align_track_to_route(trk, net, route)["arc"]
        # confidence flags from Phase 1 (same alignment code -> same labels)
        flag_high = np.load(ROOT / "cache" / "labels" / "student_test" / f"{tid}.npz")["flag"] == "high"
        in_c = np.zeros(len(label), bool)
        in_c[cands["obs"][cands["arc"] == label[cands["obs"]]]] = True
        ttype = np.array([p["transition_type"] for p in part["predictions"]])
        cat = categorise(arc, label, in_c, ttype, trk, net)
        # predicted route = decoded arc sequence incl. shortest-path fill between fixes
        path = np.concatenate(inn["path_arcs"]) if inn["path_arcs"] else np.array([], dtype=np.int64)
        p_, r_, f_ = arc_set_prf(path, route)
        rows.append(dict(trip_id=tid, n_obs=len(label), top1=float((arc == label).mean()),
                         top1_high_conf=float((arc == label)[flag_high].mean()),
                         route_continuity=route_continuity(arc, trk["segment"], net.a_from, net.a_to, net.n_nodes),
                         candidate_miss_rate=float(1 - in_c.mean()),
                         arc_precision=p_, arc_recall=r_, arc_f1=f_,
                         norm_edit_distance=norm_edit_distance(compress(path), compress(route)),
                         inference_ms=part["inference_time_ms"], **part["events"]))
        cnt = pd.Series(cat).value_counts()
        cat_rows.append(dict(trip_id=tid, n_obs=len(label), **{c: int(cnt.get(c, 0)) for c in CATS}))
        keep[tid] = (trk, net, route, part["route_geometry"], cat)
        print(f"{tid}: top1={rows[-1]['top1']:.4f} errors={ {c: int(cnt.get(c, 0)) for c in CATS} }", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "final_per_trip.csv", index=False)
    cats = pd.DataFrame(cat_rows)
    cats.to_csv(OUT / "error_categories.csv", index=False)
    num = [c for c in per.columns if c not in ("trip_id",)]
    summary = pd.DataFrame({"trip_macro": per[num].mean(),
                            "pooled": pd.Series({"top1": (per.top1 * per.n_obs).sum() / per.n_obs.sum(),
                                                 "n_obs": per.n_obs.sum()})})
    summary.to_csv(OUT / "final_summary.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(per.trip_id, per.top1, color="#2166ac", label="Top-1 (all obs)")
    ax.plot(per.trip_id, per.top1_high_conf, "o", color="#d6604d", label="Top-1 (high-confidence labels)")
    ax.set_ylim(0.5, 1); ax.set_ylabel("observation-level Top-1"); ax.legend()
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"Deployed system ({pred_.meta['model']}{', tuned' if pred_.meta.get('tuned') else ''} + HMM) "
                 f"per student-test trip")
    fig.tight_layout(); fig.savefig(OUT / "fig_final_per_trip.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(cats))
    for c in CATS:
        ax.bar(cats.trip_id, cats[c] / cats.n_obs, bottom=bottom, color=CAT_COLORS[c], label=c)
        bottom += (cats[c] / cats.n_obs).values
    ax.set_ylabel("share of observations"); ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=45)
    ax.set_title("Error categories per student-test trip")
    fig.tight_layout(); fig.savefig(OUT / "fig_error_categories.png", dpi=130); plt.close(fig)

    worst = per.sort_values("top1").trip_id.head(3).tolist()
    for tid in worst:
        trk, net, route, geom, cat = keep[tid]
        make_map(tid, trk, net, route, geom, cat, float(per.set_index("trip_id").top1[tid]),
                 OUT / f"map_worst_{tid}.html")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n=== per trip ===")
    print(per.round(4).to_string(index=False))
    print("\n=== summary ===")
    print(summary.round(4).to_string())
    print("\n=== error categories (counts) ===")
    tot = cats[CATS].sum()
    print(cats.to_string(index=False))
    print("total:", tot.to_dict(), " share of all errors:", (tot / tot.sum()).round(3).to_dict())
    print("maps for worst trips:", worst)


if __name__ == "__main__":
    main()
