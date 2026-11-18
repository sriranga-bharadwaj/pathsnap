"""Phase 5a - Bayesian hyperparameter optimisation (Optuna TPE) of the chosen model (RandomForest),
then refit and compare untuned vs tuned with the SAME protocol and metrics as Phase 2.

Search (TRAIN only, 77 gate-kept trips):
  objective  = mean observation-level Top-1 over GroupKFold(3) folds (grouped by trip),
               scored on ALL observations of the held-out trips
  data       = at most TUNE_OBS_PER_TRIP observations per trip for fitting (seeded subsample,
               smaller than Phase 2's 800 to keep 30 trials affordable); scoring uses all rows
  sampler    = TPE (seed 42), N_TRIALS trials
  space      = n_estimators [50, 300], max_depth [6, 30], min_samples_leaf [1, 50] (log),
               max_features {sqrt, 0.5, 1.0}, class_weight {None, balanced}
Evaluation of the best config: identical to experiments/03 (GroupKFold(5), 800 obs/trip,
confidence weights, student-test metrics). The tuned model replaces the untuned one in
artifacts/ only if its CV Top-1 (train) is >= the untuned CV Top-1 (decided on TRAIN, not on test).

Outputs: results/phase5/optuna_trials.csv, fig_optuna_history.png, fig_optuna_params.png,
         tuned_vs_untuned.csv, fig_tuned_vs_untuned.png, fig_confusion_tuned_vs_untuned.png
         results/experiments_log.csv (appended), cache/oof/RandomForest_tuned.npz
Run:  .venv\\Scripts\\python.exe experiments/05_optuna_tune.py
"""
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import data_root, load_arcs
from src.evaluation import candidate_metrics, route_continuity
from src.candidates import RADIUS, K_MAX, RETRY_RADIUS
from src.features import FEATURES

# reuse the helpers of the Phase 2 script (same protocol)
_spec = importlib.util.spec_from_file_location("p2", ROOT / "experiments" / "03_train_compare.py")
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)

SEED = 42
N_TRIALS = 30
TUNE_FOLDS = 3
TUNE_OBS_PER_TRIP = 400
OUT = ROOT / "results" / "phase5"
LOG = ROOT / "results" / "experiments_log.csv"


def rf(params):
    return RandomForestClassifier(**params, n_jobs=-1, random_state=SEED)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TR, TE = p2.load("train"), p2.load("student_test")
    kept_trips = TR["trip_gate"] != "exclude"
    kept_obs = kept_trips[TR["obs_trip"]]
    trip_start = np.r_[0, np.cumsum(TR["trip_n_obs"])[:-1]]
    row_gid = trip_start[TR["row_trip"]] + TR["row_obs"]
    w_rows = TR["obs_weight"][row_gid]
    kept_trip_idx = np.flatnonzero(kept_trips)

    # ---------------- 1. Optuna search ----------------
    p2.MAX_OBS_PER_TRIP = TUNE_OBS_PER_TRIP
    chosen_tune = p2.sample_training_obs(TR, kept_obs, np.random.default_rng(SEED))
    tune_rows = chosen_tune[row_gid]
    folds3 = list(GroupKFold(n_splits=TUNE_FOLDS).split(kept_trip_idx, groups=kept_trip_idx))
    print(f"tuning rows: {tune_rows.sum()} ({chosen_tune.sum()} obs), {TUNE_FOLDS} folds, {N_TRIALS} trials", flush=True)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=25),
            max_depth=trial.suggest_int("max_depth", 6, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            max_features=trial.suggest_categorical("max_features", ["sqrt", 0.5, 1.0]),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        )
        top1s, f1s, lls = [], [], []
        t0 = time.time()
        for tr_i, va_i in folds3:
            fit_rows = tune_rows & np.isin(TR["row_trip"], kept_trip_idx[tr_i])
            va_rows = np.isin(TR["row_trip"], kept_trip_idx[va_i])
            m = rf(params).fit(TR["X"][fit_rows], TR["y"][fit_rows], sample_weight=w_rows[fit_rows])
            p = m.predict_proba(TR["X"][va_rows])[:, 1]
            va_obs = np.isin(TR["obs_trip"], kept_trip_idx[va_i])
            _, _, corr = p2.obs_level(TR, va_obs, va_rows, p)
            cm = candidate_metrics(TR["y"][va_rows], p)
            top1s.append(corr.sum() / va_obs.sum()); f1s.append(cm["cand_macro_f1"]); lls.append(cm["log_loss"])
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(top1s)))
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        print(f"  trial {trial.number}: top1={np.mean(top1s):.4f} f1={np.mean(f1s):.4f} {params} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return float(np.mean(top1s))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="rf_top1")
    # trial 0 = the untuned Phase 2 configuration, so the search starts from the baseline
    study.enqueue_trial(dict(n_estimators=150, max_depth=20, min_samples_leaf=5, max_features="sqrt",
                             class_weight=None))
    t_search = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    t_search = time.time() - t_search
    trials = study.trials_dataframe()
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best = study.best_params
    print(f"\nbest params: {best}  CV(3) top1={study.best_value:.4f}  search time {t_search / 60:.1f} min")

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = trials["value"].values
    ax.plot(trials["number"], vals, "o", alpha=0.6, label="trial")
    ax.plot(trials["number"], np.maximum.accumulate(vals), "-", color="C3", label="best so far")
    ax.axhline(vals[0], color="grey", ls=":", label="trial 0 = untuned config")
    ax.set_xlabel("trial"); ax.set_ylabel("CV(3) observation-level Top-1"); ax.legend()
    ax.set_title("Optuna TPE optimisation history (RandomForest, train only)")
    fig.tight_layout(); fig.savefig(OUT / "fig_optuna_history.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, prm in zip(axes, ["params_n_estimators", "params_max_depth", "params_min_samples_leaf"]):
        ax.scatter(trials[prm], trials["value"], alpha=0.7)
        ax.set_xlabel(prm.replace("params_", "")); ax.set_ylabel("CV Top-1")
    axes[2].set_xscale("log")
    fig.suptitle("Trial value vs hyperparameter"); fig.tight_layout()
    fig.savefig(OUT / "fig_optuna_params.png", dpi=130); plt.close(fig)

    # ---------------- 2. evaluate tuned config with the Phase 2 protocol ----------------
    p2.MAX_OBS_PER_TRIP = 800
    chosen = p2.sample_training_obs(TR, kept_obs, np.random.default_rng(SEED))   # same draw as Phase 2
    train_rows = chosen[row_gid]
    folds5 = list(GroupKFold(n_splits=5).split(kept_trip_idx, groups=kept_trip_idx))
    oof = np.full(len(TR["y"]), np.nan)
    f1s, top1s, lls = [], [], []
    for tr_i, va_i in folds5:
        fit_rows = train_rows & np.isin(TR["row_trip"], kept_trip_idx[tr_i])
        va_rows = np.isin(TR["row_trip"], kept_trip_idx[va_i])
        m = rf(best).fit(TR["X"][fit_rows], TR["y"][fit_rows], sample_weight=w_rows[fit_rows])
        p = m.predict_proba(TR["X"][va_rows])[:, 1]
        oof[va_rows] = p
        cm = candidate_metrics(TR["y"][va_rows], p)
        va_obs = np.isin(TR["obs_trip"], kept_trip_idx[va_i])
        _, _, corr = p2.obs_level(TR, va_obs, va_rows, p)
        f1s.append(cm["cand_macro_f1"]); top1s.append(corr.sum() / va_obs.sum()); lls.append(cm["log_loss"])
    np.savez_compressed(ROOT / "cache" / "oof" / "RandomForest_tuned.npz", oof=oof)
    t0 = time.time()
    m = rf(best).fit(TR["X"][train_rows], TR["y"][train_rows], sample_weight=w_rows[train_rows])
    train_s = time.time() - t0
    p_tr = m.predict_proba(TR["X"][train_rows])[:, 1]
    _, _, tr_corr = p2.obs_level(TR, chosen, train_rows, p_tr)
    t0 = time.time()
    p_te = m.predict_proba(TE["X"])[:, 1]
    infer_ms = (time.time() - t0) * 1000
    te_cm = candidate_metrics(TE["y"], p_te)
    pred, _, corr = p2.obs_level(TE, np.ones(len(TE["obs_idx"]), bool), np.ones(len(TE["y"]), bool), p_te)
    te_start = np.r_[0, np.cumsum(TE["trip_n_obs"])[:-1]]
    high = TE["obs_flag"] == "high"
    cont, trip_top1 = [], []
    for ti, tid in enumerate(TE["trip_ids"]):
        s, n = te_start[ti], TE["trip_n_obs"][ti]
        a_from, a_to = load_arcs(data_root() / "student_test" / tid, tid)
        cont.append(route_continuity(pred[s:s + n], TE["obs_segment"][s:s + n], a_from, a_to,
                                     int(max(a_from.max(), a_to.max()) + 1)))
        trip_top1.append(corr[s:s + n].mean())
    joblib.dump(m, ROOT / "cache" / "models" / "RandomForest_tuned.joblib")
    tuned = dict(
        model="RandomForest_tuned", family="Ensemble (bagging trees)", config=json.dumps(best), seed=SEED,
        cv_macro_f1_mean=np.mean(f1s), cv_macro_f1_std=np.std(f1s), cv_top1_mean=np.mean(top1s),
        cv_top1_std=np.std(top1s), cv_log_loss_mean=np.mean(lls),
        train_macro_f1=candidate_metrics(TR["y"][train_rows], p_tr)["cand_macro_f1"],
        train_top1=tr_corr.sum() / chosen.sum(),
        test_top1=corr.mean(), test_top1_high=corr[high].mean(), test_top1_trip_macro=np.mean(trip_top1),
        test_cand_macro_f1=te_cm["cand_macro_f1"], test_cand_accuracy=te_cm["cand_accuracy"],
        test_cand_precision=te_cm["cand_precision"], test_cand_recall=te_cm["cand_recall"],
        test_cand_f1=te_cm["cand_f1"], test_log_loss=te_cm["log_loss"],
        test_candidate_miss_rate=float(1 - TE["obs_label_in_cands"].mean()),
        test_route_continuity_trip_macro=float(np.nanmean(cont)), train_time_s=train_s,
        infer_ms_total=infer_ms, infer_ms_per_1000_obs=infer_ms / len(TE["obs_idx"]) * 1000,
        n_train_rows=int(train_rows.sum()), n_train_obs=int(chosen.sum()))
    untuned = pd.read_csv(ROOT / "results" / "phase2" / "model_comparison.csv").set_index("model").loc["RandomForest"]
    comp = pd.DataFrame([dict(untuned, model="RandomForest_untuned"), tuned])
    comp["optuna_trials"] = [np.nan, N_TRIALS]
    comp["optuna_search_minutes"] = [np.nan, t_search / 60]
    comp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)

    log = pd.DataFrame([tuned])
    log.insert(0, "timestamp", datetime.now().isoformat(timespec="seconds"))
    log.insert(1, "phase", "5_optuna_tuned")
    log["candidate_config"] = json.dumps(dict(radius=RADIUS, k_max=K_MAX, retry_radius=RETRY_RADIUS))
    log["subsample_max_obs_per_trip"] = 800
    old = pd.read_csv(LOG)
    pd.concat([old, log], ignore_index=True).to_csv(LOG, index=False)

    cols = ["cv_macro_f1_mean", "cv_top1_mean", "test_top1", "test_cand_macro_f1", "test_log_loss",
            "test_route_continuity_trip_macro", "infer_ms_per_1000_obs", "train_time_s"]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.3 * len(cols), 3.4))
    for ax, c in zip(axes, cols):
        ax.bar(["untuned", "tuned"], comp[c], color=["#4393c3", "#d6604d"])
        lo = min(comp[c]); hi = max(comp[c])
        if c not in ("infer_ms_per_1000_obs", "train_time_s", "test_log_loss"):
            ax.set_ylim(lo - 0.02, min(1.0, hi + 0.01))
        ax.set_title(c, fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle("RandomForest untuned vs Optuna-tuned (same partitions and metrics)")
    fig.tight_layout(); fig.savefig(OUT / "fig_tuned_vs_untuned.png", dpi=130); plt.close(fig)

    unt_cm = np.array(json.loads(json.dumps(candidate_metrics(TE["y"], joblib.load(
        ROOT / "cache" / "models" / "RandomForest.joblib").predict_proba(TE["X"])[:, 1])["confusion"])))
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, (nm, c) in zip(axes, [("untuned", unt_cm), ("tuned", np.array(te_cm["confusion"]))]):
        ax.imshow(c, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(c[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(nm)
    fig.suptitle("Student-test candidate-level confusion matrices")
    fig.tight_layout(); fig.savefig(OUT / "fig_confusion_tuned_vs_untuned.png", dpi=130); plt.close(fig)

    deploy = tuned["cv_top1_mean"] >= untuned["cv_top1_mean"]
    if deploy:
        joblib.dump(m, ROOT / "artifacts" / "model.joblib", compress=3)
        meta = dict(model="RandomForest", family="Ensemble (bagging trees)", tuned=True, params=best,
                    features=FEATURES, seed=SEED, oof_file="RandomForest_tuned.npz",
                    candidate=dict(radius=RADIUS, k_max=K_MAX, retry_radius=RETRY_RADIUS),
                    selection="Phase 2: highest student-test Top-1; Phase 5: tuned replaces untuned "
                              "because its train CV Top-1 >= untuned CV Top-1",
                    test_top1=float(tuned["test_top1"]), cv_top1_mean=float(tuned["cv_top1_mean"]),
                    sklearn_version=__import__("sklearn").__version__,
                    created=datetime.now().isoformat(timespec="seconds"))
        (ROOT / "artifacts" / "model_meta.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 250)
    print("\n=== untuned vs tuned ===")
    print(comp[["model"] + cols + ["test_top1_high", "cv_macro_f1_std", "cv_top1_std"]].round(4).to_string(index=False))
    print(f"\nDeploy tuned model: {deploy} (tuned CV top1 {tuned['cv_top1_mean']:.4f} vs untuned "
          f"{untuned['cv_top1_mean']:.4f})")


if __name__ == "__main__":
    main()
