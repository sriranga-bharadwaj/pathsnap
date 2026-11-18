"""Phase 0 - Exploratory data analysis for PathSnap.

Partitions used: TRAIN (80 trips) and STUDENT_TEST (10 trips). Every row and figure
is labelled with its partition. The evaluator partition is not available and is
never touched.

Outputs (results/eda/):
  trip_summary.csv        one row per trip: inventory, missing/duplicates, time gaps,
                          extents, network size/connectivity, route validity, GPS-to-route distance
  route_breaks.csv        every place where consecutive reference arcs are not connected
  partition_summary.csv   aggregated statistics per partition
  projection_check.csv    projection distortion check (AEQD vs geodesic)
  fig_*.png               EDA figures
  map_*.html              Folium maps of selected trips

Run:  .venv\\Scripts\\python.exe experiments/00_eda.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Geod
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import (data_root, list_trips, load_track, load_nodes, load_arcs,
                      load_route, has_route, make_projection)

OUT = ROOT / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
GEOD = Geod(ellps="WGS84")
GAP_BREAK_S = 60.0          # time gap treated as a segment break (decoder restart)
FAR_M = 50.0                # "far from route" threshold for GPS points


def network_stats(n_nodes, a_from, a_to):
    """Size and connectivity of one directed road network."""
    n_arcs = len(a_from)
    out = {"n_nodes": n_nodes, "n_arcs": n_arcs}
    out["arc_node_id_out_of_range"] = int(((a_from < 0) | (a_from >= n_nodes) |
                                           (a_to < 0) | (a_to >= n_nodes)).sum())
    out["self_loops"] = int((a_from == a_to).sum())
    key = a_from * n_nodes + a_to                         # unique code per directed pair
    out["duplicate_arcs"] = int(n_arcs - len(np.unique(key)))
    rev = a_to * n_nodes + a_from
    out["reverse_twin_share"] = float(np.isin(rev, key).mean())   # arc has opposite-direction twin
    deg = np.bincount(np.concatenate([a_from, a_to]), minlength=n_nodes)
    out["isolated_nodes"] = int((deg == 0).sum())
    g = coo_matrix((np.ones(n_arcs), (a_from, a_to)), shape=(n_nodes, n_nodes)).tocsr()
    # weak components ignore direction; strong components respect one-way streets.
    # isolated nodes are excluded from the "largest share" so they do not dilute it.
    used = deg > 0
    nw, lw = connected_components(g, directed=True, connection="weak")
    ns, ls = connected_components(g, directed=True, connection="strong")
    out["weak_components"] = int(len(np.unique(lw[used])))
    out["largest_weak_share"] = float(np.bincount(lw[used]).max() / used.sum())
    out["strong_components"] = int(len(np.unique(ls[used])))
    out["largest_strong_share"] = float(np.bincount(ls[used]).max() / used.sum())
    return out, ls


def analyse_trip(partition, pdir, tid):
    r = {"partition": partition, "trip_id": tid}
    tdir = pdir / tid
    for ext in ["track", "nodes", "arcs", "route"]:
        f = tdir / f"{tid}.{ext}"
        r[f"has_{ext}"] = f.exists()
        r[f"size_kb_{ext}"] = round(f.stat().st_size / 1024, 1) if f.exists() else np.nan
    raw = (tdir / f"{tid}.track").read_bytes()
    r["track_crlf"] = b"\r\n" in raw

    # ---------- GPS track ----------
    lon, lat, t = load_track(tdir, tid)
    n = len(lon)
    r["n_points"] = n
    r["nan_values"] = int(np.isnan(lon).sum() + np.isnan(lat).sum() + np.isnan(t).sum())
    r["invalid_coords"] = int(((np.abs(lat) > 90) | (np.abs(lon) > 180) |
                               ((lat == 0) & (lon == 0))).sum())
    r["dup_rows"] = int(pd.DataFrame({"a": lon, "b": lat, "c": t}).duplicated().sum())
    r["dup_timestamps"] = int(n - len(np.unique(t)))
    dt = np.diff(t)
    r["non_increasing_steps"] = int((dt <= 0).sum())
    r["t_integer"] = bool(np.all(t == np.round(t)))
    r["share_dt_1s"] = float(np.mean(np.isclose(dt, 1.0)))
    r["dt_median"] = float(np.median(dt))
    r["dt_p99"] = float(np.percentile(dt, 99))
    r["dt_max"] = float(dt.max())
    r["gaps_gt_10s"] = int((dt > 10).sum())
    r["gaps_gt_60s"] = int((dt > GAP_BREAK_S).sum())
    r["duration_h"] = float((t[-1] - t[0]) / 3600)
    r["stationary_steps"] = int(((np.diff(lon) == 0) & (np.diff(lat) == 0)).sum())

    fwd, _ = make_projection(lon, lat)
    x, y = fwd(lon, lat)
    x, y = np.asarray(x), np.asarray(y)
    r["lon_median"], r["lat_median"] = float(np.median(lon)), float(np.median(lat))
    r["lon_min"], r["lon_max"] = float(lon.min()), float(lon.max())
    r["lat_min"], r["lat_max"] = float(lat.min()), float(lat.max())
    r["utm_zone"] = int((np.median(lon) + 180) // 6) + 1
    r["bbox_w_km"] = float((x.max() - x.min()) / 1000)
    r["bbox_h_km"] = float((y.max() - y.min()) / 1000)
    r["max_dist_from_centre_km"] = float(np.hypot(x, y).max() / 1000)
    step = np.hypot(np.diff(x), np.diff(y))
    r["track_length_km"] = float(step.sum() / 1000)
    with np.errstate(divide="ignore", invalid="ignore"):
        spd = np.where(dt > 0, step / dt, np.nan)
    r["speed_p50_ms"] = float(np.nanmedian(spd))
    r["speed_p99_ms"] = float(np.nanpercentile(spd, 99))
    r["speed_gt_70ms_steps"] = int(np.nansum(spd > 70))   # >250 km/h = implausible jump

    # projection distortion check: projected step length vs WGS84 geodesic
    _, _, geo = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    m = geo > 1
    r["proj_step_rel_err_max"] = float(np.max(np.abs(step[m] - geo[m]) / geo[m])) if m.any() else np.nan

    # ---------- road network ----------
    nlon, nlat = load_nodes(tdir, tid)
    a_from, a_to = load_arcs(tdir, tid)
    ns, scc_labels = network_stats(len(nlon), a_from, a_to)
    r.update(ns)
    nx_, ny_ = fwd(nlon, nlat)
    nx_, ny_ = np.asarray(nx_), np.asarray(ny_)
    arc_len = np.hypot(nx_[a_to] - nx_[a_from], ny_[a_to] - ny_[a_from])
    r["arc_len_median_m"] = float(np.median(arc_len))
    r["arc_len_p95_m"] = float(np.percentile(arc_len, 95))
    r["zero_length_arcs"] = int((arc_len < 0.01).sum())
    r["net_bbox_covers_track"] = bool(nx_.min() <= x.min() and nx_.max() >= x.max()
                                      and ny_.min() <= y.min() and ny_.max() >= y.max())
    # distance of every GPS point to the nearest arc of the WHOLE network
    seg = np.stack([np.c_[nx_[a_from], ny_[a_from]], np.c_[nx_[a_to], ny_[a_to]]], axis=1)
    lines = shapely.linestrings(seg)
    pts = shapely.points(x, y)
    tree = shapely.STRtree(lines)
    _, d_net = tree.query_nearest(pts, return_distance=True, all_matches=False)
    r["gps_to_network_p50_m"] = float(np.median(d_net))
    r["gps_to_network_p99_m"] = float(np.percentile(d_net, 99))
    r["gps_to_network_gt30m_share"] = float(np.mean(d_net > 30))

    # ---------- reference route ----------
    breaks = []
    d_route = None
    if r["has_route"]:
        route = load_route(tdir, tid)
        r["n_route_arcs"] = len(route)
        valid = (route >= 0) & (route < len(a_from))
        r["route_invalid_ids"] = int((~valid).sum())
        route = route[valid]
        r["route_unique_arcs"] = int(len(np.unique(route)))
        r["route_consecutive_repeats"] = int((np.diff(route) == 0).sum())
        conn = a_to[route[:-1]] == a_from[route[1:]]
        r["route_breaks"] = int((~conn).sum())
        for i in np.where(~conn)[0]:
            p = nx_[a_to[route[i]]] - nx_[a_from[route[i + 1]]]
            q = ny_[a_to[route[i]]] - ny_[a_from[route[i + 1]]]
            breaks.append({"partition": partition, "trip_id": tid, "route_pos": int(i),
                           "arc_a": int(route[i]), "arc_b": int(route[i + 1]),
                           "gap_m": float(np.hypot(p, q))})
        r["route_length_km"] = float(arc_len[route].sum() / 1000)
        r["route_in_single_scc"] = bool(len(np.unique(scc_labels[a_from[route]])) == 1)
        r["gps_per_route_arc"] = float(n / len(route))
        # distance of each GPS point to its own reference route
        rtree = shapely.STRtree(lines[route])
        _, d_route = rtree.query_nearest(pts, return_distance=True, all_matches=False)
        r["gps_to_route_p50_m"] = float(np.median(d_route))
        r["gps_to_route_p95_m"] = float(np.percentile(d_route, 95))
        r["gps_to_route_p99_m"] = float(np.percentile(d_route, 99))
        r["gps_to_route_gt50m_share"] = float(np.mean(d_route > FAR_M))
        r["route_len_over_track_len"] = r["route_length_km"] / max(r["track_length_km"], 1e-9)
    return r, breaks, dt, d_route


def folium_map(partition, pdir, tid, title, out_html, corridor_m=150.0):
    """Folium map: GPS points, nearby road network, reference route, layer control, legend.
    Only arcs within `corridor_m` metres of the GPS track are drawn (keeps the HTML small
    while showing every road the matcher could realistically consider)."""
    tdir = pdir / tid
    lon, lat, t = load_track(tdir, tid)
    nlon, nlat = load_nodes(tdir, tid)
    a_from, a_to = load_arcs(tdir, tid)
    fwd, _ = make_projection(lon, lat)
    x, y = map(np.asarray, fwd(lon, lat))
    nx_, ny_ = map(np.asarray, fwd(nlon, nlat))
    seg = np.stack([np.c_[nx_[a_from], ny_[a_from]], np.c_[nx_[a_to], ny_[a_to]]], axis=1)
    tree = shapely.STRtree(shapely.linestrings(seg))
    # every arc within corridor_m of at least one GPS point (vectorised STRtree query)
    _, hit = tree.query(shapely.points(x, y), predicate="dwithin", distance=corridor_m)
    near_arcs = np.unique(hit)
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=14,
                   tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    fg_net = folium.FeatureGroup(name=f"Road network ({len(near_arcs)} arcs within {corridor_m:.0f} m)",
                                 show=True)
    for a in near_arcs:
        folium.PolyLine([[nlat[a_from[a]], nlon[a_from[a]]], [nlat[a_to[a]], nlon[a_to[a]]]],
                        color="#888888", weight=1, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)
    if has_route(tdir, tid):
        route = load_route(tdir, tid)
        fg_r = folium.FeatureGroup(name="Reference route (.route)")
        # draw as one polyline per connected run of arcs (breaks stay visible)
        run = [[nlat[a_from[route[0]]], nlon[a_from[route[0]]]]]
        for i, a in enumerate(route):
            run.append([nlat[a_to[a]], nlon[a_to[a]]])
            if i + 1 < len(route) and a_to[a] != a_from[route[i + 1]]:
                folium.PolyLine(run, color="#1a9850", weight=5, opacity=0.8).add_to(fg_r)
                run = [[nlat[a_from[route[i + 1]]], nlon[a_from[route[i + 1]]]]]
        folium.PolyLine(run, color="#1a9850", weight=5, opacity=0.8).add_to(fg_r)
        fg_r.add_to(m)
    fg_g = folium.FeatureGroup(name=f"GPS observations ({len(lon)})")
    step = max(1, len(lon) // 3000)        # thin very long tracks so the HTML stays light
    for i in range(0, len(lon), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color="#2166ac", fill=True,
                            fill_opacity=0.8, weight=0, tooltip=f"obs {i}, t={t[i]:.0f}").add_to(fg_g)
    fg_g.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
      padding:8px 10px;border:1px solid #999;font-size:13px">
      <b>{title}</b><br>partition: {partition}, trip {tid}<br>
      <span style="color:#2166ac">&#9679;</span> GPS observations<br>
      <span style="color:#1a9850">&#9644;</span> reference route<br>
      <span style="color:#888888">&#9644;</span> road network</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))


def figures(df, all_dt, all_droute):
    col = {"train": "#2166ac", "student_test": "#d6604d"}
    # 1. observations per trip
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, g in df.groupby("partition"):
        ax.hist(g["n_points"], bins=30, alpha=0.6, label=f"{p} (n={len(g)})", color=col[p])
    ax.set_xlabel("GPS observations per trip"); ax.set_ylabel("trips"); ax.legend()
    ax.set_title("Observations per trip (train + student_test)")
    fig.tight_layout(); fig.savefig(OUT / "fig_points_per_trip.png", dpi=130); plt.close(fig)
    # 2. sampling interval distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, d in all_dt.items():
        d = np.clip(d, 0, 30)
        ax.hist(d, bins=np.arange(0, 31, 1), alpha=0.6, label=p, color=col[p], density=True, log=True)
    ax.set_xlabel("time step between consecutive fixes (s, clipped at 30)"); ax.set_ylabel("density (log)")
    ax.set_title("Sampling intervals (train + student_test)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_sampling_intervals.png", dpi=130); plt.close(fig)
    # 3. network size vs track length
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, g in df.groupby("partition"):
        ax.scatter(g["n_points"], g["n_arcs"], label=p, color=col[p], s=20)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("GPS points"); ax.set_ylabel("network arcs"); ax.legend()
    ax.set_title("Network size vs trip length (train + student_test)")
    fig.tight_layout(); fig.savefig(OUT / "fig_network_size.png", dpi=130); plt.close(fig)
    # 4. world map of trip locations (GPS coordinate distribution)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for p, g in df.groupby("partition"):
        ax.scatter(g["lon_median"], g["lat_median"], label=p, color=col[p], s=25)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.grid(alpha=0.3)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend()
    ax.set_title("Trip locations (median GPS position)")
    fig.tight_layout(); fig.savefig(OUT / "fig_trip_locations.png", dpi=130); plt.close(fig)
    # 5. GPS-to-reference-route distance
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, d in all_droute.items():
        ax.hist(np.clip(d, 0, 100), bins=100, alpha=0.6, label=p, color=col[p], density=True, log=True)
    ax.axvline(FAR_M, color="k", ls="--", lw=1)
    ax.set_xlabel("distance GPS point -> own reference route (m, clipped at 100)")
    ax.set_ylabel("density (log)"); ax.legend()
    ax.set_title("GPS noise relative to reference route")
    fig.tight_layout(); fig.savefig(OUT / "fig_gps_to_route.png", dpi=130); plt.close(fig)
    # 6. per-trip share of points far from route (quality gate evidence)
    fig, ax = plt.subplots(figsize=(10, 4))
    d = df.sort_values("gps_to_route_gt50m_share", ascending=False)
    ax.bar(range(len(d)), d["gps_to_route_gt50m_share"], color=[col[p] for p in d["partition"]])
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d["trip_id"].str[-3:], rotation=90, fontsize=6)
    ax.set_ylabel(f"share of GPS points > {FAR_M:.0f} m from route")
    ax.set_title("Reference-route coverage per trip (blue=train, red=student_test)")
    fig.tight_layout(); fig.savefig(OUT / "fig_route_coverage.png", dpi=130); plt.close(fig)


def trajectory_plots(root, picks):
    """GPS trajectory + road network plots in projected metres (small multiples)."""
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 5))
    for ax, (partition, tid) in zip(np.atleast_1d(axes), picks):
        tdir = root / partition / tid
        lon, lat, _ = load_track(tdir, tid)
        nlon, nlat = load_nodes(tdir, tid)
        a_from, a_to = load_arcs(tdir, tid)
        fwd, _ = make_projection(lon, lat)
        x, y = map(np.asarray, fwd(lon, lat))
        nx_, ny_ = map(np.asarray, fwd(nlon, nlat))
        pad = 300
        keep = ((nx_[a_from] > x.min() - pad) & (nx_[a_from] < x.max() + pad) &
                (ny_[a_from] > y.min() - pad) & (ny_[a_from] < y.max() + pad))
        segs = np.stack([np.c_[nx_[a_from[keep]], ny_[a_from[keep]]],
                         np.c_[nx_[a_to[keep]], ny_[a_to[keep]]]], axis=1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs, colors="#bbbbbb", linewidths=0.4))
        route = load_route(tdir, tid)
        rs = np.stack([np.c_[nx_[a_from[route]], ny_[a_from[route]]],
                       np.c_[nx_[a_to[route]], ny_[a_to[route]]]], axis=1)
        ax.add_collection(LineCollection(rs, colors="#1a9850", linewidths=2))
        ax.plot(x, y, ".", ms=1.5, color="#2166ac")
        ax.set_xlim(x.min() - pad, x.max() + pad); ax.set_ylim(y.min() - pad, y.max() + pad)
        ax.set_aspect("equal"); ax.set_title(f"{partition} {tid}", fontsize=9)
        ax.set_xlabel("x (m, local AEQD)"); ax.set_ylabel("y (m)")
    fig.suptitle("GPS (blue), reference route (green), road network (grey)")
    fig.tight_layout(); fig.savefig(OUT / "fig_trajectories.png", dpi=130); plt.close(fig)


def main():
    root = data_root()
    rows, breaks = [], []
    all_dt = {"train": [], "student_test": []}
    all_dr = {"train": [], "student_test": []}
    partitions = {"train": root / "train", "student_test": root / "student_test"}
    seen = {}
    for partition, pdir in partitions.items():
        for tid in list_trips(pdir):
            t0 = time.time()
            r, b, dt, dr = analyse_trip(partition, pdir, tid)
            r["eda_seconds"] = round(time.time() - t0, 2)
            rows.append(r); breaks += b
            all_dt[partition].append(dt)
            if dr is not None:
                all_dr[partition].append(dr)
            seen.setdefault(tid, []).append(partition)
            print(f"[{partition}] {tid}: n={r['n_points']} arcs={r['n_arcs']} "
                  f"breaks={r.get('route_breaks')} far50={r.get('gps_to_route_gt50m_share', np.nan):.3f} "
                  f"({r['eda_seconds']}s)", flush=True)
    # integrity check: no trip ID in two partitions
    overlap = {k: v for k, v in seen.items() if len(v) > 1}
    assert not overlap, f"trip IDs found in more than one partition: {overlap}"
    print("Partition overlap check: OK (no trip ID in two partitions)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trip_summary.csv", index=False)
    pd.DataFrame(breaks, columns=["partition", "trip_id", "route_pos", "arc_a", "arc_b", "gap_m"]
                 ).to_csv(OUT / "route_breaks.csv", index=False)
    all_dt = {k: np.concatenate(v) for k, v in all_dt.items()}
    all_dr = {k: np.concatenate(v) for k, v in all_dr.items()}

    agg = df.groupby("partition").agg(
        trips=("trip_id", "count"),
        points_total=("n_points", "sum"), points_min=("n_points", "min"),
        points_median=("n_points", "median"), points_max=("n_points", "max"),
        arcs_min=("n_arcs", "min"), arcs_median=("n_arcs", "median"), arcs_max=("n_arcs", "max"),
        nan_values=("nan_values", "sum"), invalid_coords=("invalid_coords", "sum"),
        dup_rows=("dup_rows", "sum"), dup_timestamps=("dup_timestamps", "sum"),
        non_increasing_steps=("non_increasing_steps", "sum"),
        share_dt_1s_min=("share_dt_1s", "min"), share_dt_1s_max=("share_dt_1s", "max"),
        dt_max=("dt_max", "max"), gaps_gt_60s=("gaps_gt_60s", "sum"),
        isolated_nodes=("isolated_nodes", "sum"),
        reverse_twin_min=("reverse_twin_share", "min"), reverse_twin_max=("reverse_twin_share", "max"),
        route_invalid_ids=("route_invalid_ids", "sum"), route_breaks=("route_breaks", "sum"),
        trips_with_breaks=("route_breaks", lambda s: int((s > 0).sum())),
        gps_per_route_arc_median=("gps_per_route_arc", "median"),
        gps_per_route_arc_min=("gps_per_route_arc", "min"),
        max_dist_from_centre_km=("max_dist_from_centre_km", "max"),
        proj_step_rel_err_max=("proj_step_rel_err_max", "max"),
    ).T
    pooled = {p: {"pooled_share_dt_1s": float(np.mean(np.isclose(all_dt[p], 1.0))),
                  "pooled_gps_to_route_p50_m": float(np.median(all_dr[p])),
                  "pooled_gps_to_route_p99_m": float(np.percentile(all_dr[p], 99)),
                  "pooled_gps_to_route_gt50m_share": float(np.mean(all_dr[p] > FAR_M))}
              for p in all_dt}
    agg = pd.concat([agg, pd.DataFrame(pooled)])
    agg.to_csv(OUT / "partition_summary.csv")
    df[["partition", "trip_id", "max_dist_from_centre_km", "proj_step_rel_err_max"]].to_csv(
        OUT / "projection_check.csv", index=False)

    figures(df, all_dt, all_dr)
    # trajectory plots: one clean student_test trip, one clean train trip, one low-coverage train trip
    tr = df[df.partition == "train"].sort_values("gps_to_route_gt50m_share")
    st = df[df.partition == "student_test"].sort_values("n_points")
    picks = [("student_test", st.iloc[len(st) // 2]["trip_id"]),
             ("train", tr.iloc[len(tr) // 2]["trip_id"]),
             ("train", tr.iloc[-1]["trip_id"])]
    trajectory_plots(root, picks)
    folium_map("student_test", partitions["student_test"], picks[0][1],
               "EDA: typical student_test trip", OUT / f"map_student_test_{picks[0][1]}.html")
    folium_map("train", partitions["train"], picks[2][1],
               "EDA: train trip with poorest route coverage", OUT / f"map_train_{picks[2][1]}.html")

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("\n=== partition summary ===")
    print(agg.to_string())
    print("\n=== route breaks ===")
    print(pd.read_csv(OUT / "route_breaks.csv", dtype={"trip_id": str}).to_string())
    print("\n=== trips with > 5% of GPS points beyond 50 m of their own route ===")
    print(df[df.gps_to_route_gt50m_share > 0.05][["partition", "trip_id", "n_points",
          "gps_to_route_p50_m", "gps_to_route_gt50m_share", "route_len_over_track_len"]]
          .sort_values("gps_to_route_gt50m_share", ascending=False).to_string())
    print("\n=== trips whose reference route is much shorter/longer than the GPS track "
          "(route_len/track_len outside [0.7, 1.3]) ===")
    print(df[(df.route_len_over_track_len < 0.7) | (df.route_len_over_track_len > 1.3)][
          ["partition", "trip_id", "track_length_km", "route_length_km", "route_len_over_track_len",
           "gps_to_route_gt50m_share", "gps_to_network_p50_m"]]
          .sort_values("route_len_over_track_len").to_string())
    print("\n=== route_len/track_len distribution per partition ===")
    print(df.groupby("partition")["route_len_over_track_len"].describe().round(3).to_string())
    print("\n=== lowest reverse-twin shares ===")
    print(df.sort_values("reverse_twin_share")[["partition", "trip_id", "n_arcs", "reverse_twin_share"]]
          .head(6).to_string())
    print("\n=== trips with non-integer timestamps ===")
    print(df[~df.t_integer][["partition", "trip_id", "n_points"]].to_string())
    print("\nFigures/maps picks:", picks)


if __name__ == "__main__":
    main()
