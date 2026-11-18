"""Build the PathSnap technical report PDF from the REAL result files.

  * every number is read from results/*.csv / *.json at build time (nothing typed by hand)
  * all prose lives in report/report_text.md (blocks "@@ key"), so the team can rewrite the
    text in their own words and rebuild; {placeholders} in the text are filled from the results
  * figures come from results/ and screenshots from report/images/

Run:  .venv\\Scripts\\python.exe report/build_report.py      -> report/PathSnap_Technical_Report.pdf
"""
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, ListFlowable, ListItem)

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
IMG = ROOT / "report" / "images"
OUT_PDF = ROOT / "report" / "PathSnap_Technical_Report.pdf"
MODELS = ["LogisticRegression", "GaussianNB", "RandomForest", "HistGradientBoosting", "MLP", "LightGBM"]
FAMILY = {"LogisticRegression": "Linear", "GaussianNB": "Probabilistic (generative)",
          "RandomForest": "Ensemble (bagging)", "HistGradientBoosting": "Boosting (histogram)",
          "MLP": "Neural network", "LightGBM": "Boosting (leaf-wise)"}
# key configuration = the hyperparameters set in experiments/03_train_compare.py::make_models
KEYCFG = {"LogisticRegression": "StandardScaler + L2, C=1.0, lbfgs, max_iter=1000",
          "GaussianNB": "StandardScaler + GaussianNB, var_smoothing=1e-9",
          "RandomForest": "150 trees, max_depth=20, min_samples_leaf=5, max_features=sqrt",
          "HistGradientBoosting": "200 iters, lr=0.1, 31 leaves, no early stop",
          "MLP": "Scaler + MLP(64,32), relu, Adam lr=1e-3, alpha=1e-4, batch 512, 40 epochs",
          "LightGBM": "300 trees, lr=0.05, 31 leaves, min_child_samples=20"}


# ----------------------------------------------------------------------------- values
def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def f4(x):
    return f"{x:.4f}"


def values():
    V = {}
    ps = pd.read_csv(R / "eda/partition_summary.csv", index_col=0)
    ts = pd.read_csv(R / "eda/trip_summary.csv", dtype={"trip_id": str})
    for p, k in [("train", "tr"), ("student_test", "st")]:
        c = ps[p]
        V[f"{k}_trips"] = int(float(c.trips))
        V[f"{k}_points_total"] = f"{int(float(c.points_total)):,}"
        V[f"{k}_gaps_gt_60s"] = int(float(c.gaps_gt_60s))
        V[f"{k}_dt_max"] = f"{float(c.dt_max):,.0f}"
        V[f"{k}_pooled_share_dt_1s"] = pct(float(c.pooled_share_dt_1s))
        V[f"{k}_share_dt_1s_min"] = pct(float(c.share_dt_1s_min))
        V[f"{k}_gps_per_route_arc_median"] = f"{float(c.gps_per_route_arc_median):.1f}"
        V[f"{k}_route_p50"] = f"{float(c.pooled_gps_to_route_p50_m):.1f}"
        V[f"{k}_route_p99"] = f"{float(c.pooled_gps_to_route_p99_m):.1f}"
        V[f"{k}_route_gt50"] = pct(float(c.pooled_gps_to_route_gt50m_share), 2)
        V[f"{k}_route_breaks"] = int(float(c.route_breaks))
        V[f"{k}_trips_with_breaks"] = int(float(c.trips_with_breaks))
    V["all_points_min"] = int(ts.n_points.min()); V["all_points_max"] = f"{int(ts.n_points.max()):,}"
    V["all_arcs_min"] = f"{int(ts.n_arcs.min()):,}"; V["all_arcs_max"] = f"{int(ts.n_arcs.max()):,}"
    V["missing_total"] = int(ts[["nan_values", "invalid_coords", "dup_rows", "dup_timestamps",
                                 "non_increasing_steps"]].sum().sum())
    V["isolated_total"] = int(ts.isolated_nodes.sum())
    tw = ts.sort_values("reverse_twin_share")
    V["twin_min"] = pct(tw.reverse_twin_share.iloc[0], 0); V["twin_min_trip"] = tw.trip_id.iloc[0]
    V["twin_median"] = pct(ts.reverse_twin_share.median(), 0)
    V["route_breaks_total"] = V["tr_route_breaks"] + V["st_route_breaks"]
    V["trips_with_breaks_total"] = V["tr_trips_with_breaks"] + V["st_trips_with_breaks"]
    V["n_gps_per_arc_lt2"] = int((ts.gps_per_route_arc < 2).sum())
    V["max_dist_centre_km"] = f"{ts.max_dist_from_centre_km.max():.0f}"
    V["proj_err_pct"] = f"{100 * ts.proj_step_rel_err_max.max():.4f}%"
    V["n_frac_t_trips"] = int((~ts.t_integer.astype(bool)).sum())
    V["frac_t_trips"] = ", ".join(ts[~ts.t_integer.astype(bool)].trip_id)
    V["t54_net_p50"] = f"{ts.set_index('trip_id').gps_to_network_p50_m['00000054']:.0f}"
    V["max_dup_arcs"] = int(ts.duplicate_arcs.max())
    V["n_multi_scc_routes"] = int((~ts.route_in_single_scc.astype(bool)).sum())

    al = pd.read_csv(R / "phase1/alignment_summary.csv", index_col=0)
    for s, k in [("train_kept", "tr"), ("student_test", "st")]:
        for f in ["high", "medium", "low"]:
            V[f"al_{k}_{f}"] = pct(al.loc[s, f"{f}_share"])
    V["al_tr_points"] = f"{int(al.loc['train_kept', 'n_points']):,}"
    ap = pd.read_csv(R / "phase1/alignment_per_trip.csv", dtype={"trip_id": str})
    ex = ap[ap.gate == "exclude"].sort_values("far_share", ascending=False)
    V["gate_excluded"] = ", ".join(f"{t} ({pct(s, 0)})" for t, s in zip(ex.trip_id, ex.far_share))
    wn = ap[ap.gate == "keep_warn"].sort_values("far_share", ascending=False)
    V["gate_warn"] = ", ".join(f"{t} ({pct(s, 1)})" for t, s in zip(wn.trip_id, wn.far_share))
    V["gate_kept"] = int((ap[ap.partition == "train"].gate != "exclude").sum())
    cs = pd.read_csv(R / "phase1/candidate_sweep.csv")
    for st, k in [("train_kept", "tr"), ("student_test", "st")]:
        for r in [15, 30, 50]:
            row = cs[(cs["set"] == st) & (cs.cap == "capped") & (cs.radius == r)].iloc[0]
            V[f"rec_{k}_{r}"] = f4(row.recall_all); V[f"cand_{k}_{r}"] = f"{row.cand_mean:.1f}"
            V[f"p95_{k}_{r}"] = f"{row.cand_p95_pooled:.0f}"; V[f"miss_{k}_{r}"] = f4(row.miss_rate_all)
            V[f"lat_{k}_{r}"] = f"{row.latency_ms_per_1000_obs:.1f}"
    unc = cs[(cs["set"] == "train_kept") & (cs.cap == "uncapped") & (cs.radius == 30)].iloc[0]
    V["rec_tr_30_uncapped"] = f4(unc.recall_all); V["cand_max_uncapped_30"] = int(unc.cand_max)

    mc = pd.read_csv(R / "phase2/model_comparison.csv").set_index("model")
    for m in MODELS:
        for c in mc.columns:
            v = mc.loc[m, c]
            if isinstance(v, float):
                V[f"{m}_{c}"] = f"{v:.1f}" if c in ("train_time_s", "infer_ms_per_1000_obs", "infer_ms_total") \
                    else f4(v)
        V[f"{m}_gap"] = f"{100 * (mc.loc[m, 'train_top1'] - mc.loc[m, 'cv_top1_mean']):.1f}"
    V["n_train_obs"] = f"{int(mc.n_train_obs.iloc[0]):,}"; V["n_train_rows"] = f"{int(mc.n_train_rows.iloc[0]):,}"
    V["test_miss_rate"] = f4(mc.test_candidate_miss_rate.iloc[0])
    b = pd.read_csv(R / "phase2/baselines.csv").set_index("baseline")
    V["base_nearest"] = f4(b.loc["nearest", "test_top1"]); V["base_heading"] = f4(b.loc["nearest+heading", "test_top1"])
    V["rf_over_heading"] = f"{100 * (mc.loc['RandomForest', 'test_top1'] - b.loc['nearest+heading', 'test_top1']):.2f}"
    mlp = pd.read_csv(R / "phase2/mlp_loss_curve.csv")
    V["mlp_val_first"] = f4(mlp.val_log_loss.iloc[0]); V["mlp_val_last"] = f4(mlp.val_log_loss.iloc[-1])
    V["mlp_tr_first"] = f4(mlp.train_loss.iloc[0]); V["mlp_tr_last"] = f4(mlp.train_loss.iloc[-1])

    cmp3 = pd.read_csv(R / "phase3/comparison_summary.csv", index_col=0)
    for meth, k in [("classifier+hmm", "hmm"), ("classifier_only", "clf")]:
        for c in cmp3.columns:
            V[f"{k}_{c}"] = f"{cmp3.loc[meth, c]:.0f}" if c.startswith("runtime") else f4(cmp3.loc[meth, c])
    g3 = pd.read_csv(R / "phase3/decoder_grid.csv").sort_values("train_oof_hmm_top1", ascending=False)
    V["dec_alpha"], V["dec_beta"], V["dec_rp"] = g3.alpha.iloc[0], g3.beta.iloc[0], g3.reversal_penalty.iloc[0]
    V["dec_train_hmm"] = f4(g3.train_oof_hmm_top1.iloc[0]); V["dec_train_clf"] = f4(g3.train_oof_clf_top1.iloc[0])
    V["dec_grid_n"] = len(g3); V["dec_relaxed_train"] = int(g3.relaxed.iloc[0])
    V["dec_grid_alpha"] = ", ".join(str(x) for x in sorted(g3.alpha.unique()))
    V["dec_grid_beta"] = ", ".join(str(x) for x in sorted(g3.beta.unique()))
    V["dec_grid_rp"] = ", ".join(str(x) for x in sorted(g3.reversal_penalty.unique()))
    pt = pd.read_csv(R / "phase3/per_trip_comparison.csv", dtype={"trip_id": str})
    h = pt[pt.method == "classifier+hmm"].set_index("trip_id"); c_ = pt[pt.method == "classifier_only"].set_index("trip_id")
    for col, k in [("direct_transitions", "t_direct"), ("multi_hop_transitions", "t_multi"),
                   ("reversal_transitions", "t_rev"), ("relaxed_transitions", "t_relaxed"),
                   ("unmatched_points", "t_unmatched"), ("segment_starts", "t_starts")]:
        V[k] = int(h[col].sum())
    d = h.top1 - c_.top1
    V["n_improved"] = int((d > 0).sum()); V["n_trips_st"] = len(d)
    V["worse_trips"] = ", ".join(f"{t} ({c_.top1[t]:.4f} to {h.top1[t]:.4f})" for t in d[d < 0].index) or "none"
    bt = d.idxmax(); V["best_gain_trip"] = f"{bt} ({c_.top1[bt]:.4f} to {h.top1[bt]:.4f})"

    tv = pd.read_csv(R / "phase5/tuned_vs_untuned.csv").set_index("model")
    for m, k in [("RandomForest_untuned", "unt"), ("RandomForest_tuned", "tun")]:
        for c in ["cv_macro_f1_mean", "cv_top1_mean", "test_top1", "test_top1_high", "test_cand_macro_f1",
                  "test_log_loss", "test_route_continuity_trip_macro", "cv_log_loss_mean"]:
            V[f"{k}_{c}"] = f4(tv.loc[m, c])
        V[f"{k}_infer"] = f"{tv.loc[m, 'infer_ms_per_1000_obs']:.1f}"; V[f"{k}_train_s"] = f"{tv.loc[m, 'train_time_s']:.0f}"
    V["tun_gain_cv"] = f"{100 * (tv.loc['RandomForest_tuned', 'cv_top1_mean'] - tv.loc['RandomForest_untuned', 'cv_top1_mean']):.2f}"
    V["tun_gain_test"] = f"{100 * (tv.loc['RandomForest_tuned', 'test_top1'] - tv.loc['RandomForest_untuned', 'test_top1']):.2f}"
    V["opt_trials"] = int(tv.loc["RandomForest_tuned", "optuna_trials"])
    V["opt_minutes"] = f"{tv.loc['RandomForest_tuned', 'optuna_search_minutes']:.1f}"
    ot = pd.read_csv(R / "phase5/optuna_trials.csv")
    best = ot.loc[ot.value.idxmax()]
    V["opt_best_value"] = f4(best.value); V["opt_trial0_value"] = f4(ot.value.iloc[0])
    V["opt_best_trial"] = int(best.number)
    V["opt_params"] = (f"n_estimators={int(best.params_n_estimators)}, max_depth={int(best.params_max_depth)}, "
                       f"min_samples_leaf={int(best.params_min_samples_leaf)}, max_features={best.params_max_features}, "
                       f"class_weight={best.params_class_weight if isinstance(best.params_class_weight, str) else 'None'}")
    fs = pd.read_csv(R / "phase5/final_model_selection.csv").set_index("model")
    for m, k in [("RandomForest", "sel_rf"), ("LightGBM", "sel_lgbm"), ("RandomForest_tuned", "sel_tun")]:
        V[f"{k}_train"] = f4(fs.loc[m, "train_oof_hmm_top1"]); V[f"{k}_test"] = f4(fs.loc[m, "test_hmm_top1_pooled"])
        V[f"{k}_clf"] = f4(fs.loc[m, "train_oof_clf_top1"]); V[f"{k}_arcf1"] = f4(fs.loc[m, "test_hmm_arc_f1"])
        V[f"{k}_runtime"] = f"{fs.loc[m, 'test_runtime_ms_trip']:.0f}"
        V[f"{k}_params"] = f"alpha={fs.loc[m, 'alpha']}, beta={fs.loc[m, 'beta']}, penalty={fs.loc[m, 'reversal_penalty']}"

    ec = pd.read_csv(R / "phase5/error_categories.csv", dtype={"trip_id": str})
    cats = ["unmatched", "candidate_miss", "direction_flip", "gap_fallback", "adjacent_arc", "parallel_road"]
    tot = ec[cats].sum(); n_err = int(tot.sum())
    V["n_errors"] = f"{n_err:,}"
    for c in cats:
        V[f"err_{c}"] = int(tot[c]); V[f"err_{c}_pct"] = pct(tot[c] / n_err, 0)
    V["flip_77"] = int(ec.set_index("trip_id").direction_flip["00000077"])
    V["flip_07"] = int(ec.set_index("trip_id").direction_flip["00000007"])
    fp = pd.read_csv(R / "phase5/final_per_trip.csv", dtype={"trip_id": str})
    V["worst_trips"] = ", ".join(fp.sort_values("top1").trip_id.head(3))

    smoke = json.loads((R / "phase5/api_smoke_test_final.json").read_text())["predict"]
    V["api_ok"] = sum(r["status"] == "ok" for r in smoke); V["api_n"] = len(smoke)
    lat = [r["client_latency_ms"] for r in smoke]
    V["api_lat_min"], V["api_lat_max"] = f"{min(lat):.0f}", f"{max(lat):,.0f}"
    rob = json.loads((R / "phase4/api_robustness_train_80.json").read_text())["predict"]
    V["rob_ok"] = sum(r["status"] == "ok" for r in rob); V["rob_n"] = len(rob)
    slow = max(rob, key=lambda r: r["client_latency_ms"])
    V["rob_slow_trip"] = slow["trip_id"]; V["rob_slow_s"] = f"{slow['client_latency_ms'] / 1000:.1f}"
    V["rob_slow_relaxed"] = slow["events"]["relaxed_transitions"]; V["rob_slow_unmatched"] = slow["events"]["unmatched_points"]
    return V, mc, cmp3, pt, fp, ec, tv, fs


# ----------------------------------------------------------------------------- text
def load_text():
    blocks, key = {}, None
    for line in (ROOT / "report" / "report_text.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^@@\s*(\S+)\s*$", line)
        if m:
            key = m.group(1); blocks[key] = []
        elif key:
            blocks[key].append(line)
    return {k: "\n".join(v).strip() for k, v in blocks.items()}


def inline(s, V):
    s = re.sub(r"\{(\w+)\}", lambda m: str(V[m.group(1)]), s)      # KeyError = missing value
    s = escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', s)
    return s


def render(text, V, st):
    out = []
    for para in re.split(r"\n\s*\n", text):
        lines = [l for l in para.splitlines() if l.strip()]
        if not lines:
            continue
        if all(l.lstrip().startswith("- ") for l in lines):
            items = [ListItem(Paragraph(inline(l.lstrip()[2:], V), st["body"]), leftIndent=12)
                     for l in lines]
            out.append(ListFlowable(items, bulletType="bullet", start="-", leftIndent=12, bulletFontSize=8))
        elif lines[0].startswith("### "):
            out.append(Paragraph(inline(lines[0][4:], V), st["h3"]))
            if lines[1:]:
                out.append(Paragraph(inline(" ".join(lines[1:]), V), st["body"]))
        else:
            out.append(Paragraph(inline(" ".join(lines), V), st["body"]))
    return out


# ----------------------------------------------------------------------------- layout
def styles():
    ss = getSampleStyleSheet()
    st = dict(
        title=ParagraphStyle("t", parent=ss["Title"], fontSize=20, leading=24),
        h1=ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, leading=18, spaceBefore=8, spaceAfter=6,
                          textColor=colors.HexColor("#1f3b57"), keepWithNext=1),
        h2=ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, leading=15, spaceBefore=8, spaceAfter=4,
                          textColor=colors.HexColor("#1f3b57"), keepWithNext=1),
        h3=ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10, leading=13, spaceBefore=6, spaceAfter=2,
                          keepWithNext=1),
        body=ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.2, leading=12.5, spaceAfter=4),
        cell=ParagraphStyle("c", parent=ss["BodyText"], fontSize=7.4, leading=9),
        cellb=ParagraphStyle("cb", parent=ss["BodyText"], fontSize=7.4, leading=9, fontName="Helvetica-Bold"),
        cap=ParagraphStyle("cap", parent=ss["BodyText"], fontSize=7.8, leading=10, textColor=colors.HexColor("#444444"),
                           alignment=TA_CENTER, spaceAfter=8),
        center=ParagraphStyle("ce", parent=ss["BodyText"], fontSize=10, leading=14, alignment=TA_CENTER),
    )
    return st


def table(rows, widths, st, head=True):
    data = [[Paragraph(escape(str(c)), st["cellb"] if (head and i == 0) else st["cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3e9ef") if head else colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def figure(path, width_cm, caption, st):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    w = width_cm * cm
    return KeepTogether([Image(str(path), width=w, height=w * ih / iw), Paragraph(escape(caption), st["cap"])])


def on_page(c, doc):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#666666"))
    c.drawString(1.8 * cm, 1.1 * cm, "PathSnap - UE24CS352A Machine Learning - Technical Report")
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
    c.restoreState()


def build():
    V, mc, cmp3, pt, fp, ec, tv, fs = values()
    T = load_text()
    st = styles()
    S = []
    txt = lambda k: render(T[k], V, st)
    H1 = lambda s: S.append(Paragraph(escape(s), st["h1"]))
    H2 = lambda s: S.append(Paragraph(escape(s), st["h2"]))
    FIG = lambda p, w, cap: S.append(figure(p, w, cap, st))
    W = A4[0] - 3.6 * cm

    # ---- title page
    S += [Spacer(1, 3 * cm), Paragraph("PathSnap", st["title"]),
          Paragraph("Distributed Map-Matching and Route Reconstruction", st["center"]),
          Spacer(1, 0.4 * cm),
          Paragraph("UE24CS352A Machine Learning - Hackathon Technical Report", st["center"]),
          Paragraph("PES University, Department of CSE, RR Campus", st["center"]), Spacer(1, 1.2 * cm)]
    S += txt("title_team")
    S += [Spacer(1, 0.8 * cm)] + txt("title_summary") + [PageBreak()]

    # ---- 1 overview
    H1("1. System Overview"); S += txt("overview")
    S.append(table([["Stage", "Where", "What it does"],
                    ["Preprocess", "Laptop 1", "load .track/.nodes/.arcs, local AEQD projection, speed/heading, 60 s gap segments"],
                    ["Candidates", "Laptop 1", "arcs within 30 m (both directions, cap 24, one 100 m retry)"],
                    ["Features", "Laptop 1", "9 location-independent features per (observation, candidate)"],
                    ["Classifier", "Laptop 1", "probability that the candidate is the true arc"],
                    ["HMM / Viterbi", "Laptop 1", "most plausible connected arc sequence (bounded Dijkstra transitions)"],
                    ["Evaluation", "Laptop 1", "after prediction: reads .route, returns numbers only"],
                    ["UI", "Laptop 2", "lists trips, sends trip ID, draws returned route + metrics"]],
                   [2.6 * cm, 1.9 * cm, W - 4.5 * cm], st))
    S.append(Spacer(1, 0.5 * cm))

    # ---- 2.2 EDA
    H1("2.2 Dataset Understanding and Exploratory Data Analysis")
    S += txt("eda_intro")
    H2("2.2.1 How the data was preprocessed"); S += txt("eda_preprocess")
    H2("2.2.2 Missing-value and data-quality analysis"); S += txt("eda_missing")
    ps = pd.read_csv(R / "eda/partition_summary.csv", index_col=0)
    rows = [["Statistic", "train", "student_test"]]
    for k, lab in [("trips", "trips"), ("points_total", "GPS observations"), ("points_min", "observations per trip (min)"),
                   ("points_max", "observations per trip (max)"), ("arcs_min", "network arcs (min)"),
                   ("arcs_max", "network arcs (max)"), ("nan_values", "NaN values"), ("invalid_coords", "invalid coordinates"),
                   ("dup_rows", "duplicate rows"), ("dup_timestamps", "duplicate timestamps"),
                   ("non_increasing_steps", "non-increasing timestamps"), ("gaps_gt_60s", "time gaps > 60 s"),
                   ("dt_max", "largest time gap (s)"), ("isolated_nodes", "isolated nodes"),
                   ("route_invalid_ids", "invalid route arc ids"), ("route_breaks", "route connectivity breaks")]:
        rows.append([lab] + [f"{float(ps.loc[k, p]):,.0f}" for p in ["train", "student_test"]])
    rows.append(["share of 1 s steps (pooled)"] + [pct(float(ps.loc["pooled_share_dt_1s", p])) for p in ["train", "student_test"]])
    rows.append(["reverse-twin share (min-max)"] + [f"{pct(float(ps.loc['reverse_twin_min', p]), 0)} - "
                                                    f"{pct(float(ps.loc['reverse_twin_max', p]), 0)}" for p in ["train", "student_test"]])
    S.append(table(rows, [6 * cm, 3.5 * cm, 3.5 * cm], st))
    S.append(Paragraph("Table 1. Dataset inventory and quality checks (source: results/eda/partition_summary.csv).", st["cap"]))
    H2("2.2.3 Patterns observed"); S += txt("eda_patterns")
    FIG(R / "eda/fig_trip_locations.png", 14, "Fig 1. Trip locations (median GPS position); partitions: train + student_test.")
    FIG(R / "eda/fig_points_per_trip.png", 11, "Fig 2. GPS observations per trip; partitions: train + student_test.")
    FIG(R / "eda/fig_sampling_intervals.png", 11, "Fig 3. Sampling intervals; partitions: train + student_test.")
    FIG(R / "eda/fig_trajectories.png", 16.5, "Fig 4. GPS trajectory plots with road network (grey) and reference route "
        "(green) in local metres; trips: student_test 00000076, train 00000058, train 00000054.")
    FIG(R / "eda/fig_network_size.png", 11, "Fig 5. Road-network size vs trip length; partitions: train + student_test.")
    FIG(R / "eda/fig_gps_to_route.png", 11, "Fig 6. Distance of each GPS fix to its own reference route; train + student_test.")
    FIG(R / "eda/fig_route_coverage.png", 16, "Fig 7. Share of fixes > 50 m from their reference route per trip "
        "(blue = train, red = student_test).")
    FIG(IMG / "map_eda_00000076.jpg", 15, "Fig 8. Folium map (EDA), student_test trip 00000076 (zoomed section): GPS "
        "observations, reference route and road network within 150 m, with layer control and legend "
        "(results/eda/map_student_test_00000076.html).")
    H2("2.2.4 Labels, trip-quality gate and candidate generation"); S += txt("eda_labels")
    FIG(R / "phase1/fig_quality_gate.png", 16, "Fig 9. Trip-quality gate: share of fixes aligned > 50 m from the route "
        "(black = excluded train trips, blue = kept train, red = student_test, not gated).")
    rows = [["Set", "Radius", "Recall", "Miss rate", "Mean cands", "p95 cands", "ms / 1000 obs"]]
    for k, lab in [("tr", "train (77 kept trips)"), ("st", "student_test")]:
        for r in [15, 30, 50]:
            rows.append([lab, f"{r} m", V[f"rec_{k}_{r}"], V[f"miss_{k}_{r}"], V[f"cand_{k}_{r}"], V[f"p95_{k}_{r}"],
                         V[f"lat_{k}_{r}"]])
    S.append(table(rows, [3.6 * cm, 1.4 * cm, 1.6 * cm, 1.6 * cm, 1.8 * cm, 1.6 * cm, 2.2 * cm], st))
    S.append(Paragraph("Table 2. Candidate radius sweep (cap 24 candidates, one 100 m retry); recall = reference arc "
                       "in candidate set (results/phase1/candidate_sweep.csv).", st["cap"]))
    FIG(R / "phase1/fig_candidate_sweep.png", 15, "Fig 10. Candidate recall vs radius and vs candidate-set size.")
    H2("2.2.5 Features extracted and why"); S += txt("eda_features")
    S.append(PageBreak())

    # ---- 3 models
    H1("3. Model Experimentation and Comparison"); S += txt("models_intro")
    H2("3.1 Overall Model Comparison")
    rows = [["Model", "Algorithm Family", "Key Configuration", "CV Macro F1-score (mean +- std)",
             "Top-1 Accuracy (student-test, classifier only)", "Route continuity (student-test, classifier only)",
             "Student-Test Macro F1-score", "Log Loss (student-test)", "Inference Time (ms / 1000 obs)"]]
    for m in MODELS:
        rows.append([m, FAMILY[m], KEYCFG[m], f"{V[m + '_cv_macro_f1_mean']} +- {V[m + '_cv_macro_f1_std']}",
                     V[m + "_test_top1"], V[m + "_test_route_continuity_trip_macro"], V[m + "_test_cand_macro_f1"],
                     V[m + "_test_log_loss"], V[m + "_infer_ms_per_1000_obs"]])
    S.append(table(rows, [2.3 * cm, 1.8 * cm, 3.4 * cm, 1.9 * cm, 1.7 * cm, 1.7 * cm, 1.5 * cm, 1.3 * cm, 1.5 * cm], st))
    S.append(Paragraph("Table 3. Consolidated comparison, all values measured (results/phase2/model_comparison.csv). "
                       "CV = GroupKFold(5) by trip on 77 train trips; all other columns on the 10 student-test trips.",
                       st["cap"]))
    rows = [["Model", "CV Top-1", "Train Top-1", "Test Top-1 (high-conf)", "Accuracy", "Precision", "Recall",
             "F1 (class 1)", "Train time (s)"]]
    for m in MODELS:
        rows.append([m, V[m + "_cv_top1_mean"], V[m + "_train_top1"], V[m + "_test_top1_high"], V[m + "_test_cand_accuracy"],
                     V[m + "_test_cand_precision"], V[m + "_test_cand_recall"], V[m + "_test_cand_f1"], V[m + "_train_time_s"]])
    S.append(table(rows, [2.6 * cm] + [1.75 * cm] * 8, st))
    S.append(Paragraph("Table 4. Additional metrics. Accuracy/precision/recall/F1 are candidate level on student-test "
                       "(threshold 0.5).", st["cap"]))
    S += txt("metric_definitions")
    H2("3.2 Compulsory Questions for Every Model")
    for m in MODELS:
        S.append(Paragraph(escape(f"3.2.{MODELS.index(m) + 1} {m}"), st["h2"]))
        S += txt(f"{m}.A")
        S += txt(f"{m}.B")
        S.append(Paragraph("<b>C. Experimental Results</b>", st["body"]))
        S.append(table([["7. Student-test", "Top-1", "Top-1 (high-conf)", "Macro F1", "Log Loss", "Route continuity"],
                        ["value", V[m + "_test_top1"], V[m + "_test_top1_high"], V[m + "_test_cand_macro_f1"],
                         V[m + "_test_log_loss"], V[m + "_test_route_continuity_trip_macro"]]],
                       [2.6 * cm] + [2.4 * cm] * 5, st))
        S.append(Spacer(1, 3))
        S += txt(f"{m}.C")
    H2("3.3 Model Visualisations")
    FIG(R / "phase2/fig_cv_macro_f1.png", 13, "Fig 11. GroupKFold(5) CV macro F1 per model (train).")
    FIG(R / "phase2/fig_test_top1_macro_f1.png", 14, "Fig 12. Student-test observation-level Top-1 vs candidate-level macro F1.")
    FIG(R / "phase2/fig_log_loss.png", 13, "Fig 13. Log loss, CV (train) and student-test.")
    FIG(R / "phase2/fig_confusion_matrices.png", 17, "Fig 14. Student-test candidate-level confusion matrices (optional diagnostic).")
    FIG(R / "phase2/fig_mlp_loss.png", 12, "Fig 15. MLP training loss (loss_curve_) and validation log loss on 15% "
        "held-out train trips, per epoch. Only the MLP has an epoch-wise history recorded.")
    FIG(R / "phase2/fig_cost.png", 13, "Fig 16. Training time and inference time.")
    H2("3.4 Model selection"); S += txt("selection")
    S.append(PageBreak())

    # ---- 4 HPO
    H1("4. Bayesian Hyperparameter Optimisation (Random Forest)"); S += txt("hpo")
    rows = [["Metric (same partitions)", "Untuned RF", "Optuna-tuned RF"],
            ["CV Macro F1 (train, GroupKFold 5)", V["unt_cv_macro_f1_mean"], V["tun_cv_macro_f1_mean"]],
            ["CV Top-1 (train)", V["unt_cv_top1_mean"], V["tun_cv_top1_mean"]],
            ["Student-test Top-1 (classifier only)", V["unt_test_top1"], V["tun_test_top1"]],
            ["Student-test Macro F1", V["unt_test_cand_macro_f1"], V["tun_test_cand_macro_f1"]],
            ["Student-test Log Loss", V["unt_test_log_loss"], V["tun_test_log_loss"]],
            ["Inference time (ms / 1000 obs)", V["unt_infer"], V["tun_infer"]],
            ["Training time (s)", V["unt_train_s"], V["tun_train_s"]],
            ["Train OOF Top-1 after HMM", V["sel_rf_train"], V["sel_tun_train"]],
            ["Student-test Top-1 after HMM", V["sel_rf_test"], V["sel_tun_test"]]]
    S.append(table(rows, [7 * cm, 3.5 * cm, 3.5 * cm], st))
    S.append(Paragraph("Table 5. Untuned vs tuned Random Forest (results/phase5/tuned_vs_untuned.csv, "
                       "final_model_selection.csv).", st["cap"]))
    S += txt("hpo_after")
    FIG(R / "phase5/fig_optuna_history.png", 12.5, "Fig 17. Optuna TPE optimisation history (train, CV(3) Top-1).")
    FIG(R / "phase5/fig_optuna_params.png", 16, "Fig 18. Trial value vs hyperparameter.")
    FIG(R / "phase5/fig_tuned_vs_untuned.png", 17, "Fig 19. Untuned vs tuned Random Forest on the same metrics.")
    S.append(PageBreak())

    # ---- 5 HMM
    H1("5. HMM and Viterbi Route Reconstruction"); S += txt("hmm")
    rows = [["Metric (student-test, 10 trips)", "Classifier only", "Classifier + HMM/Viterbi"]]
    for c, lab in [("top1_pooled", "Top-1 (pooled over observations)"), ("top1_trip_macro", "Top-1 (mean of trips)"),
                   ("top1_high_trip_macro", "Top-1, high-confidence labels"), ("route_continuity_trip_macro", "Route continuity"),
                   ("arc_precision_trip_macro", "Arc-set precision"), ("arc_recall_trip_macro", "Arc-set recall"),
                   ("arc_f1_trip_macro", "Arc-set F1"), ("norm_edit_distance_trip_macro", "Normalised edit distance (lower better)"),
                   ("runtime_ms_trip_macro", "Runtime per trip (ms)")]:
        rows.append([lab, V["clf_" + c], V["hmm_" + c]])
    S.append(table(rows, [7 * cm, 3.5 * cm, 3.5 * cm], st))
    S.append(Paragraph("Table 6. Classifier-only vs classifier + HMM, deployed untuned Random Forest "
                       "(results/phase3/comparison_summary.csv).", st["cap"]))
    rows = [["Trip", "Obs", "Top-1 classifier only", "Top-1 + HMM", "Arc F1 classifier only", "Arc F1 + HMM",
             "Multi-hop", "Relaxed"]]
    for t in sorted(pt.trip_id.unique()):
        a = pt[(pt.trip_id == t) & (pt.method == "classifier_only")].iloc[0]
        b = pt[(pt.trip_id == t) & (pt.method == "classifier+hmm")].iloc[0]
        rows.append([t, int(a.n_obs), f4(a.top1), f4(b.top1), f4(a.arc_f1), f4(b.arc_f1),
                     int(b.multi_hop_transitions), int(b.relaxed_transitions)])
    S.append(table(rows, [1.8 * cm, 1.2 * cm, 2.3 * cm, 2 * cm, 2.4 * cm, 2 * cm, 1.6 * cm, 1.5 * cm], st))
    S.append(Paragraph("Table 7. Per-trip comparison (results/phase3/per_trip_comparison.csv).", st["cap"]))
    S += txt("hmm_after")
    FIG(R / "phase3/fig_decoder_grid.png", 17, "Fig 20. Decoder grid on train out-of-fold probabilities (Top-1 after Viterbi).")
    FIG(R / "phase3/fig_hmm_vs_classifier.png", 17, "Fig 21. Student-test per-trip Top-1 and arc-set F1, classifier only vs + HMM.")
    S.append(table([["Category"] + ["unmatched", "candidate miss", "direction flip", "gap / fallback",
                                    "adjacent arc", "parallel road"],
                    ["errors"] + [str(V[f"err_{c}"]) for c in ["unmatched", "candidate_miss", "direction_flip",
                                                                 "gap_fallback", "adjacent_arc", "parallel_road"]],
                    ["share"] + [V[f"err_{c}_pct"] for c in ["unmatched", "candidate_miss", "direction_flip",
                                                              "gap_fallback", "adjacent_arc", "parallel_road"]]],
                   [2 * cm] + [2.3 * cm] * 6, st))
    S.append(Paragraph("Table 8. Error categories of the deployed system on student-test "
                       "(results/phase5/error_categories.csv).", st["cap"]))
    FIG(R / "phase5/fig_error_categories.png", 14, "Fig 22. Error categories per student-test trip (deployed system).")
    S.append(PageBreak())

    # ---- 6 Folium
    H1("6. Geographical Visualisation Using Folium"); S += txt("folium")
    FIG(IMG / "map_final_00000009_overview.jpg", 15.5, "Fig 23. Folium map, student_test trip 00000009 (deployed system): "
        "road network, reference route (green), predicted route (purple), GPS observations (correct + errors by "
        "category), layer control and legend (results/phase5/map_worst_00000009.html).")
    FIG(IMG / "map_final_00000009_zoom.jpg", 13, "Fig 24. Same map zoomed into the error cluster at the start of the trip.")
    S += txt("folium_after")
    S.append(PageBreak())

    # ---- 7 API UI deployment
    H1("7. API, UI and Deployment"); S += txt("deploy")
    FIG(IMG / "ui_laptop2_00000076.jpg", 16, "Fig 25. Evaluator UI with trip 00000076 loaded (screenshot taken on the "
        "development machine with the API at 127.0.0.1; replace with a Laptop-2 screenshot using Laptop 1's IP).")
    FIG(IMG / "api_cmd_laptop1_cropped.png", 16.5, "Fig 26. API console on Laptop 1 (receiving laptop): requests from "
        "Laptop 2 (10.8.120.63) with trip id, status, metrics and latency.")
    S += txt("deploy_after")
    H1("8. Integrity and AI-use statement"); S += txt("integrity")

    # never leave a heading alone at the bottom of a page: bind it to the next flowable
    bound, i = [], 0
    while i < len(S):
        f = S[i]
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2", "h3") and i + 1 < len(S):
            bound.append(KeepTogether([f, S[i + 1]])); i += 2
        else:
            bound.append(f); i += 1
    S = bound

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm, title="PathSnap Technical Report",
                            author="PathSnap team")
    doc.build(S, onFirstPage=on_page, onLaterPages=on_page)
    print(f"written {OUT_PDF}")


if __name__ == "__main__":
    build()
