"""Phase 2c - non-learned reference baselines on STUDENT_TEST (context for the model Top-1).

  nearest          : pick the nearest candidate arc (rank 0)
  nearest+heading  : among candidates, pick the nearest one whose direction agrees with the
                     movement heading (heading_cos > 0); fall back to nearest if none agrees

Same Top-1 definition as src/evaluation.py (denominator = all observations).
Output: results/phase2/baselines.csv
Run:  .venv\\Scripts\\python.exe experiments/03b_baseline_nearest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation import top1_predictions
from src.features import FEATURES

z = np.load(ROOT / "cache" / "features" / "student_test.npz")
X = z["X"]
start = np.r_[0, np.cumsum(z["trip_n_obs"])[:-1]]
gid = start[z["row_trip"]] + z["row_obs"]
n = len(z["obs_idx"])
dist = X[:, FEATURES.index("dist")]
hcos = X[:, FEATURES.index("heading_cos")]
hval = X[:, FEATURES.index("heading_valid")]
high = z["obs_flag"] == "high"
rows = []
for name, score in [("nearest", -dist),
                    ("nearest+heading", -dist - 1000.0 * ((hval == 1) & (hcos <= 0)))]:
    pred, _ = top1_predictions(gid, z["row_arc"], score, n)
    corr = pred == z["obs_label"]
    rows.append(dict(baseline=name, test_top1=corr.mean(), test_top1_high=corr[high].mean()))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "phase2" / "baselines.csv", index=False)
print(df.round(4).to_string(index=False))
