"""Phase 2b - train and compare five candidate classifiers.

Data:
  TRAIN: 77 gate-kept trips (3 excluded by the Phase 1 quality gate).
         Training rows = observations whose reference arc is in their candidate set,
         SUBSAMPLED to at most MAX_OBS_PER_TRIP observations per trip (seeded) to keep
         training time practical. Rows are weighted by alignment confidence (1.0/0.5/0.2).
  CV:    GroupKFold(5) by trip on the kept train trips. Each fold fits on the subsampled
         rows of its training trips and is scored on ALL rows/observations of its held-out
         trips (out-of-fold probabilities are saved for decoder tuning in Phase 3).
  STUDENT_TEST: 10 trips, all observations, for model comparison (never used for fitting).
Preprocessing (StandardScaler) lives inside each sklearn Pipeline, so it is fitted only on
the training rows of each fit.

Metric definitions: see src/evaluation.py. PRIMARY = observation-level Top-1 accuracy.

Outputs:
  results/phase2/model_comparison.csv      one row per model, all metrics
  results/phase2/cv_folds.csv              per-fold CV results
  results/phase2/per_trip_top1.csv         student-test Top-1 per trip and model
  results/phase2/fig_*.png                 comparison plots, confusion matrices, MLP loss curves
  results/experiments_log.csv              appended: one row per model run
  cache/oof/<model>.npz, cache/models/<model>.joblib  (git-ignored)
  artifacts/model.joblib + artifacts/model_meta.json  best model by student-test Top-1

Run:  .venv\\Scripts\\python.exe experiments/03_train_compare.py
"""
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

# `--only <model>`: train/evaluate just that model, merge its rows into the existing result CSVs,
# and leave artifacts/ untouched (used to add LightGBM after the first five-model run).
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import data_root, load_arcs
from src.evaluation import candidate_metrics, top1_predictions, route_continuity
from src.candidates import RADIUS, K_MAX, RETRY_RADIUS
from src.features import FEATURES

SEED = 42
MAX_OBS_PER_TRIP = 800
N_FOLDS = 5
FEAT = ROOT / "cache" / "features"
OUT = ROOT / "results" / "phase2"
LOG = ROOT / "results" / "experiments_log.csv"


def make_models():
    """The five candidate scorers with their initial (untuned) hyperparameters."""
    return {
        "LogisticRegression": ("Linear", Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])),
        "GaussianNB": ("Probabilistic (generative)", Pipeline([
            ("scale", StandardScaler()),
            ("clf", GaussianNB(var_smoothing=1e-9))])),
        "RandomForest": ("Ensemble (bagging trees)", RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "HistGradientBoosting": ("Boosting", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False,
            random_state=SEED)),
        "MLP": ("Neural network", Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                                  learning_rate_init=1e-3, max_iter=40, early_stopping=False,
                                  random_state=SEED))])),
        # 6th model, added after the first comparison (run with: --only LightGBM)
        "LightGBM": ("Boosting (leaf-wise, LightGBM)", LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
            subsample=1.0, colsample_bytree=1.0, n_jobs=-1, random_state=SEED, verbose=-1)),
    }


def fit_model(model, X, y, w):
    if isinstance(model, Pipeline):
        return model.fit(X, y, clf__sample_weight=w)
    return model.fit(X, y, sample_weight=w)


def params_of(model):
    est = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    keep = {k: v for k, v in est.get_params().items() if isinstance(v, (int, float, str, tuple, type(None)))}
    return json.dumps(keep, default=str)


def load(partition):
    z = np.load(FEAT / f"{partition}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def obs_level(D, trip_mask_obs, row_mask, prob):
    """Top-1 over all observations selected by trip_mask_obs (rows restricted to row_mask)."""
    # global observation id = position in the obs-level arrays
    trip_start = np.r_[0, np.cumsum(D["trip_n_obs"])[:-1]]
    gid = trip_start[D["row_trip"]] + D["row_obs"]
    n_all = len(D["obs_idx"])
    pred, score = top1_predictions(gid[row_mask], D["row_arc"][row_mask], prob, n_all)
    correct = (pred == D["obs_label"]) & trip_mask_obs
    return pred, score, correct


def sample_training_obs(D, kept_trip_mask_obs, rng):
    """Eligible observations (kept trip, label in candidate set), at most MAX_OBS_PER_TRIP per trip."""
    elig = kept_trip_mask_obs & D["obs_label_in_cands"]
    chosen = np.zeros_like(elig)
    for t in np.unique(D["obs_trip"][elig]):
        idx = np.flatnonzero(elig & (D["obs_trip"] == t))
        if len(idx) > MAX_OBS_PER_TRIP:
            idx = rng.choice(idx, MAX_OBS_PER_TRIP, replace=False)
        chosen[idx] = True
    return chosen


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "cache" / "oof").mkdir(parents=True, exist_ok=True)
    (ROOT / "cache" / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    TR, TE = load("train"), load("student_test")
    # ---- integrity checks ----
    assert not set(TR["trip_ids"]) & set(TE["trip_ids"]), "trip in both train and student_test"
    kept_trips = TR["trip_gate"] != "exclude"
    kept_obs = kept_trips[TR["obs_trip"]]
    trip_start = np.r_[0, np.cumsum(TR["trip_n_obs"])[:-1]]
    row_gid = trip_start[TR["row_trip"]] + TR["row_obs"]
    chosen = sample_training_obs(TR, kept_obs, rng)
    train_rows = chosen[row_gid]
    w_rows = TR["obs_weight"][row_gid]
    print(f"kept train trips: {kept_trips.sum()} / {len(kept_trips)}; training obs (subsampled): "
          f"{chosen.sum()} of {(kept_obs & TR['obs_label_in_cands']).sum()} eligible; rows: {train_rows.sum()}")

    # CV folds over kept trips only (groups = trip)
    kept_trip_idx = np.flatnonzero(kept_trips)
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(kept_trip_idx, groups=kept_trip_idx))
    for tr_i, va_i in folds:
        assert not set(kept_trip_idx[tr_i]) & set(kept_trip_idx[va_i]), "trip leak across CV folds"

    # student-test bookkeeping
    te_start = np.r_[0, np.cumsum(TE["trip_n_obs"])[:-1]]
    te_arcs = {ti: load_arcs(data_root() / "student_test" / tid, tid) for ti, tid in enumerate(TE["trip_ids"])}
    te_high = TE["obs_flag"] == "high"
    cand_miss_te = float(1 - TE["obs_label_in_cands"].mean())

    rows, fold_rows, trip_rows, cms = [], [], [], {}
    for name, (family, _) in make_models().items():
        if ONLY and name != ONLY:
            continue
        print(f"\n===== {name} =====", flush=True)
        # ---------- group CV ----------
        oof = np.full(len(TR["y"]), np.nan)
        f1s, top1s, lls = [], [], []
        for k, (tr_i, va_i) in enumerate(folds):
            tr_trips, va_trips = kept_trip_idx[tr_i], kept_trip_idx[va_i]
            fit_rows = train_rows & np.isin(TR["row_trip"], tr_trips)
            va_rows = np.isin(TR["row_trip"], va_trips)
            m = make_models()[name][1]
            t0 = time.time()
            fit_model(m, TR["X"][fit_rows], TR["y"][fit_rows], w_rows[fit_rows])
            fit_s = time.time() - t0
            p = m.predict_proba(TR["X"][va_rows])[:, 1]
            oof[va_rows] = p
            cm = candidate_metrics(TR["y"][va_rows], p)
            va_obs = np.isin(TR["obs_trip"], va_trips)
            _, _, corr = obs_level(TR, va_obs, va_rows, p)
            t1 = corr.sum() / va_obs.sum()
            f1s.append(cm["cand_macro_f1"]); top1s.append(t1); lls.append(cm["log_loss"])
            fold_rows.append(dict(model=name, fold=k, n_val_trips=len(va_trips), fit_s=fit_s,
                                  cand_macro_f1=cm["cand_macro_f1"], top1=t1, log_loss=cm["log_loss"]))
            print(f"  fold {k}: macroF1={cm['cand_macro_f1']:.4f} top1={t1:.4f} logloss={cm['log_loss']:.4f} "
                  f"fit={fit_s:.1f}s", flush=True)
        np.savez_compressed(ROOT / "cache" / "oof" / f"{name}.npz", oof=oof)

        # ---------- final fit on all kept train trips ----------
        m = make_models()[name][1]
        t0 = time.time()
        fit_model(m, TR["X"][train_rows], TR["y"][train_rows], w_rows[train_rows])
        train_s = time.time() - t0
        joblib.dump(m, ROOT / "cache" / "models" / f"{name}.joblib")
        # in-sample (training rows) for over/underfitting evidence
        p_tr = m.predict_proba(TR["X"][train_rows])[:, 1]
        tr_cm = candidate_metrics(TR["y"][train_rows], p_tr)
        _, _, tr_corr = obs_level(TR, chosen, train_rows, p_tr)
        train_top1 = tr_corr.sum() / chosen.sum()

        # ---------- student-test ----------
        t0 = time.time()
        p_te = m.predict_proba(TE["X"])[:, 1]
        infer_ms = (time.time() - t0) * 1000
        te_cm = candidate_metrics(TE["y"], p_te)
        all_te = np.ones(len(TE["obs_idx"]), bool)
        pred, _, corr = obs_level(TE, all_te, np.ones(len(TE["y"]), bool), p_te)
        cont = []
        for ti, tid in enumerate(TE["trip_ids"]):
            s, n = te_start[ti], TE["trip_n_obs"][ti]
            a_from, a_to = te_arcs[ti]
            c_ = route_continuity(pred[s:s + n], TE["obs_segment"][s:s + n], a_from, a_to,
                                  int(max(a_from.max(), a_to.max()) + 1))
            cont.append(c_)
            trip_rows.append(dict(model=name, trip_id=tid, n_obs=n, top1=corr[s:s + n].mean(),
                                  top1_high=corr[s:s + n][te_high[s:s + n]].mean(),
                                  route_continuity=c_))
        cms[name] = np.array(te_cm["confusion"])
        r = dict(
            model=name, family=family, config=params_of(m), seed=SEED,
            cv_macro_f1_mean=np.mean(f1s), cv_macro_f1_std=np.std(f1s),
            cv_top1_mean=np.mean(top1s), cv_top1_std=np.std(top1s), cv_log_loss_mean=np.mean(lls),
            train_macro_f1=tr_cm["cand_macro_f1"], train_top1=train_top1, train_log_loss=tr_cm["log_loss"],
            test_top1=corr.mean(), test_top1_high=corr[te_high].mean(),
            test_top1_trip_macro=np.mean([t["top1"] for t in trip_rows if t["model"] == name]),
            test_cand_macro_f1=te_cm["cand_macro_f1"], test_cand_accuracy=te_cm["cand_accuracy"],
            test_cand_precision=te_cm["cand_precision"], test_cand_recall=te_cm["cand_recall"],
            test_cand_f1=te_cm["cand_f1"], test_log_loss=te_cm["log_loss"],
            test_candidate_miss_rate=cand_miss_te, test_route_continuity_trip_macro=float(np.nanmean(cont)),
            train_time_s=train_s, infer_ms_total=infer_ms,
            infer_ms_per_1000_obs=infer_ms / len(TE["obs_idx"]) * 1000,
            n_train_rows=int(train_rows.sum()), n_train_obs=int(chosen.sum()))
        rows.append(r)
        print(f"  TEST top1={r['test_top1']:.4f} (high {r['test_top1_high']:.4f}) macroF1={r['test_cand_macro_f1']:.4f} "
              f"logloss={r['test_log_loss']:.4f} continuity={r['test_route_continuity_trip_macro']:.4f} "
              f"train={train_s:.1f}s infer={infer_ms:.0f}ms", flush=True)

    new = pd.DataFrame(rows)
    res, folds_df, trips_df = new, pd.DataFrame(fold_rows), pd.DataFrame(trip_rows)
    if ONLY:   # merge into the existing tables (replace any previous rows of this model)
        def merge(df, f):
            old = pd.read_csv(OUT / f, dtype={"trip_id": str})
            return pd.concat([old[old.model != ONLY], df], ignore_index=True)
        res, folds_df, trips_df = (merge(new, "model_comparison.csv"), merge(folds_df, "cv_folds.csv"),
                                   merge(trips_df, "per_trip_top1.csv"))
        for n_ in res.model:        # confusion matrices of the other models from their saved fits
            if n_ not in cms:
                p_ = joblib.load(ROOT / "cache" / "models" / f"{n_}.joblib").predict_proba(TE["X"])[:, 1]
                cms[n_] = np.array(candidate_metrics(TE["y"], p_)["confusion"])
    res.to_csv(OUT / "model_comparison.csv", index=False)
    folds_df.to_csv(OUT / "cv_folds.csv", index=False)
    trips_df.to_csv(OUT / "per_trip_top1.csv", index=False)

    # ---------- experiment log (append only the models run now) ----------
    log = new.copy()
    log.insert(0, "timestamp", datetime.now().isoformat(timespec="seconds"))
    log.insert(1, "phase", "2_untuned_comparison")
    log["candidate_config"] = json.dumps(dict(radius=RADIUS, k_max=K_MAX, retry_radius=RETRY_RADIUS))
    log["subsample_max_obs_per_trip"] = MAX_OBS_PER_TRIP
    if LOG.exists():   # pd.concat keeps columns aligned even if the log has extra columns
        log = pd.concat([pd.read_csv(LOG), log], ignore_index=True)
    log.to_csv(LOG, index=False)

    # ---------- best model -> artifacts (full run only) ----------
    best = res.sort_values(["test_top1", "cv_top1_mean"], ascending=False).iloc[0]
    if not ONLY:
        m = joblib.load(ROOT / "cache" / "models" / f"{best.model}.joblib")
        joblib.dump(m, ROOT / "artifacts" / "model.joblib", compress=3)   # compressed: GitHub 100 MB file limit
        meta = dict(model=best.model, family=best.family, tuned=False, features=FEATURES, seed=SEED,
                    candidate=dict(radius=RADIUS, k_max=K_MAX, retry_radius=RETRY_RADIUS),
                    selection="highest student-test observation-level Top-1 (tie-break CV Top-1)",
                    test_top1=float(best.test_top1), cv_top1_mean=float(best.cv_top1_mean),
                    sklearn_version=__import__("sklearn").__version__,
                    created=datetime.now().isoformat(timespec="seconds"))
        (ROOT / "artifacts" / "model_meta.json").write_text(json.dumps(meta, indent=2))

    plots(res, cms, TR, kept_trip_idx, train_rows, w_rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    cols = ["model", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "cv_top1_std", "train_top1",
            "test_top1", "test_top1_high", "test_cand_macro_f1", "test_cand_accuracy", "test_cand_precision",
            "test_cand_recall", "test_log_loss", "test_candidate_miss_rate", "test_route_continuity_trip_macro",
            "train_time_s", "infer_ms_per_1000_obs"]
    print("\n=== model comparison ===")
    print(res[cols].round(4).to_string(index=False))
    print(f"\nBest by student-test Top-1: {best.model}"
          + (" (artifacts/ NOT changed: --only run)" if ONLY else " -> artifacts/model.joblib"))


def plots(res, cms, TR, kept_trip_idx, train_rows, w_rows):
    names = list(res.model)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, res.cv_macro_f1_mean, yerr=res.cv_macro_f1_std, capsize=4, color="#4393c3")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20); ax.set_ylim(0.5, 1)
    ax.set_ylabel("candidate-level macro F1"); ax.set_title("GroupKFold(5) CV macro F1 (train, mean +- std)")
    fig.tight_layout(); fig.savefig(OUT / "fig_cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, res.test_top1, 0.4, label="observation-level Top-1", color="#2166ac")
    ax.bar(x + 0.2, res.test_cand_macro_f1, 0.4, label="candidate-level macro F1", color="#f4a582")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20); ax.set_ylim(0.5, 1); ax.legend()
    ax.set_title("Student-test: Top-1 vs candidate-level macro F1")
    fig.tight_layout(); fig.savefig(OUT / "fig_test_top1_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, res.cv_log_loss_mean, 0.4, label="CV (train)", color="#4393c3")
    ax.bar(x + 0.2, res.test_log_loss, 0.4, label="student-test", color="#d6604d")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20); ax.set_ylabel("log loss"); ax.legend()
    ax.set_title("Log loss (candidate level, lower is better)")
    fig.tight_layout(); fig.savefig(OUT / "fig_log_loss.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, res.train_time_s, 0.4, label="training time (s)", color="#762a83")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, res.infer_ms_per_1000_obs, 0.4, label="inference ms / 1000 obs", color="#5aae61")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("training time (s)"); ax2.set_ylabel("inference ms / 1000 obs")
    ax.set_title("Computational cost"); fig.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_cost.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2))
    for ax, n in zip(axes, names):
        c = cms[n]
        ax.imshow(c, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{c[i, j]}", ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(n, fontsize=9)
    fig.suptitle("Student-test candidate-level confusion matrices (threshold 0.5)")
    fig.tight_layout(); fig.savefig(OUT / "fig_confusion_matrices.png", dpi=130); plt.close(fig)

    # MLP: real per-epoch training loss and group-held-out validation loss via partial_fit
    # (skipped on an --only run of another model: the existing MLP curve stays valid)
    if ONLY and ONLY != "MLP":
        return
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_t, va_t = next(gss.split(kept_trip_idx, groups=kept_trip_idx))
    tr_r = train_rows & np.isin(TR["row_trip"], kept_trip_idx[tr_t])
    va_r = np.isin(TR["row_trip"], kept_trip_idx[va_t])
    sc = StandardScaler().fit(TR["X"][tr_r])
    Xtr, Xva = sc.transform(TR["X"][tr_r]), sc.transform(TR["X"][va_r])
    mlp = make_models()["MLP"][1].named_steps["clf"]
    tr_loss, va_loss = [], []
    for ep in range(mlp.max_iter):
        mlp.partial_fit(Xtr, TR["y"][tr_r], classes=[0, 1], sample_weight=w_rows[tr_r])
        tr_loss.append(mlp.loss_curve_[-1])
        va_loss.append(log_loss(TR["y"][va_r], mlp.predict_proba(Xva)[:, 1], labels=[0, 1]))
    pd.DataFrame(dict(epoch=np.arange(1, len(tr_loss) + 1), train_loss=tr_loss, val_log_loss=va_loss)
                 ).to_csv(OUT / "mlp_loss_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(tr_loss) + 1), tr_loss, label="training loss (weighted, incl. L2) - loss_curve_")
    ax.plot(range(1, len(va_loss) + 1), va_loss, label="validation log loss (15% held-out trips)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    ax.set_title("MLP training history (only model with epoch-wise history)")
    fig.tight_layout(); fig.savefig(OUT / "fig_mlp_loss.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
