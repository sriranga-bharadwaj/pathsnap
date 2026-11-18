"""Final model selection at SYSTEM level (classifier + HMM), using TRAIN data only.

Rule (fixed before looking at the result): among the candidate scorers, deploy the one whose
decoder-tuned, out-of-fold training Top-1 AFTER Viterbi decoding is highest. Student-test numbers
are reported alongside but not used for the choice.

Inputs : results/phase3/decoder_grid.csv            (untuned RandomForest OOF)
         results/phase5/decoder/decoder_grid.csv    (Optuna-tuned RandomForest OOF)
         results/phase5/decoder_lightgbm/decoder_grid.csv (LightGBM OOF)
         matching comparison_summary.csv files (student-test, reported only)
Output : results/phase5/final_model_selection.csv; writes artifacts/model.joblib + model_meta.json
         for the winner (decoder config is then rewritten by experiments/04_hmm_tune_compare.py).
Run:  .venv\\Scripts\\python.exe scripts/select_final_model.py
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
CANDIDATES = {
    # name: (decoder result folder, model file, oof file, tuned flag, params)
    "RandomForest": (R / "phase3", ROOT / "cache/models/RandomForest.joblib", "RandomForest.npz", False),
    "RandomForest_tuned": (R / "phase5/decoder", ROOT / "cache/models/RandomForest_tuned.joblib",
                           "RandomForest_tuned.npz", True),
    "LightGBM": (R / "phase5/decoder_lightgbm", ROOT / "cache/models/LightGBM.joblib", "LightGBM.npz", False),
}

rows = []
for name, (folder, _, _, _) in CANDIDATES.items():
    g = pd.read_csv(folder / "decoder_grid.csv").sort_values("train_oof_hmm_top1", ascending=False).iloc[0]
    s = pd.read_csv(folder / "comparison_summary.csv", index_col=0).loc["classifier+hmm"]
    rows.append(dict(model=name, train_oof_clf_top1=g.train_oof_clf_top1, train_oof_hmm_top1=g.train_oof_hmm_top1,
                     alpha=g.alpha, beta=g.beta, reversal_penalty=g.reversal_penalty,
                     test_hmm_top1_pooled=s.top1_pooled, test_hmm_arc_f1=s.arc_f1_trip_macro,
                     test_runtime_ms_trip=s.runtime_ms_trip_macro))
df = pd.DataFrame(rows).sort_values("train_oof_hmm_top1", ascending=False)
df["selected"] = [True] + [False] * (len(df) - 1)
df.to_csv(R / "phase5" / "final_model_selection.csv", index=False)
print(df.round(4).to_string(index=False))

win = df.iloc[0].model
folder, model_file, oof_file, tuned = CANDIDATES[win]
joblib.dump(joblib.load(model_file), ROOT / "artifacts" / "model.joblib", compress=3)
old = json.loads((ROOT / "artifacts" / "model_meta.json").read_text())
meta = dict(model=win.replace("_tuned", ""), family={"LightGBM": "Boosting (leaf-wise, LightGBM)"}.get(
                win, "Ensemble (bagging trees)"),
            tuned=tuned, features=old["features"], seed=42, oof_file=oof_file, candidate=old["candidate"],
            selection="system-level: highest TRAIN out-of-fold Top-1 after HMM decoding among "
                      + ", ".join(CANDIDATES) + " (results/phase5/final_model_selection.csv)",
            train_oof_hmm_top1=float(df.iloc[0].train_oof_hmm_top1),
            sklearn_version=sklearn.__version__, created=datetime.now().isoformat(timespec="seconds"))
(ROOT / "artifacts" / "model_meta.json").write_text(json.dumps(meta, indent=2))
print(f"\nselected: {win} -> artifacts/model.joblib (tuned={tuned}); now re-run 04 to write its decoder config")
