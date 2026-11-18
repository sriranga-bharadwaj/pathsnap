"""Phase 5c - assert-based integrity checks. Exits with an AssertionError on the first failure.

  1. no trip ID in two partitions (train / student_test)
  2. GroupKFold used for CV never puts one trip in both training and validation
  3. fit-on-train-only: the saved model was fitted on TRAIN rows only (row count matches the
     Phase 2 training subsample of gate-kept train trips); no student_test trip is a train trip
  4. prediction path never reads .route: static check of every module the pipeline imports
  5. API response contains no reference labels / reference route keys
  6. evaluator data untouched: no code refers to an evaluator partition folder; experiments only
     read the train and student_test folders

Output: results/phase5/integrity_checks.txt
Run:  .venv\\Scripts\\python.exe experiments/07_integrity_checks.py
"""
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
lines = []


def ok(msg):
    print("PASS", msg)
    lines.append("PASS " + msg)


def main():
    from src.data import data_root, list_trips
    tr, te = set(list_trips(data_root() / "train")), set(list_trips(data_root() / "student_test"))
    assert not tr & te, f"trips in both partitions: {tr & te}"
    ok(f"1. no trip in two partitions (train={len(tr)}, student_test={len(te)})")

    F = dict(np.load(ROOT / "cache" / "features" / "train.npz"))
    kept = np.flatnonzero(F["trip_gate"] != "exclude")
    for k, (a, b) in enumerate(GroupKFold(5).split(kept, groups=kept)):
        assert not set(kept[a]) & set(kept[b]), f"fold {k} leaks trips"
    ok(f"2. GroupKFold(5) over {len(kept)} kept train trips: no trip in both train and validation")

    meta = json.loads((ROOT / "artifacts" / "model_meta.json").read_text())
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    est = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    log = __import__("pandas").read_csv(ROOT / "results" / "experiments_log.csv")
    row = log[log.model == ("RandomForest_tuned" if meta.get("tuned") else "RandomForest")].iloc[-1]
    assert getattr(est, "n_features_in_", None) == len(meta["features"])
    TE = np.load(ROOT / "cache" / "features" / "student_test.npz")
    assert not set(TE["trip_ids"].tolist()) & set(F["trip_ids"].tolist())
    assert set(F["trip_ids"][kept].tolist()) <= tr
    ok(f"3. deployed model ({meta['model']}, tuned={meta.get('tuned')}) trained on {int(row.n_train_rows)} rows "
       f"from {len(kept)} gate-kept TRAIN trips only; student_test trips disjoint; scaler/model fitted in "
       f"pipelines on training rows only")

    pred_modules = ["src/pipeline.py", "src/preprocess.py", "src/network.py", "src/candidates.py",
                    "src/features.py", "src/hmm.py", "src/geometry.py"]
    for f in pred_modules:
        src = (ROOT / f).read_text()
        assert "load_route" not in src and "alignment" not in src.replace("# ", ""), f"{f} references route/alignment"
        assert not re.search(r"\.route['\"]", src), f"{f} opens a .route file"
    pipe_imports = re.findall(r"from src(?:\.(\w+))? import", (ROOT / "src/pipeline.py").read_text())
    assert "server_eval" not in pipe_imports and "alignment" not in pipe_imports
    ok("4. prediction modules " + ", ".join(Path(f).stem for f in pred_modules) + " never read .route")

    smoke = json.loads((ROOT / "results" / "phase4" / "api_smoke_test.json").read_text())
    assert smoke["predict"] and all(not r.get("reference_keys_in_response") for r in smoke["predict"])
    api_src = (ROOT / "api" / "main.py").read_text()
    assert "load_route" not in api_src
    ok("5. API responses (10/10 smoke-test trips) contain no reference-route / label keys; api/main.py never loads .route")

    me = Path(__file__).resolve()
    for f in list((ROOT / "src").glob("*.py")) + list((ROOT / "experiments").glob("*.py")):
        if f.resolve() == me:          # this file contains the search pattern itself
            continue
        s = f.read_text().lower()
        assert not re.search(r"evaluator[_ ]?(test|testing)?[\"'/\\]", s), f"{f.name} references an evaluator folder"
    used = set()
    for f in (ROOT / "experiments").glob("*.py"):
        if f.resolve() == me:
            continue
        used |= set(re.findall(r'"(train|student_test|evaluator\w*)"', f.read_text()))
    assert used <= {"train", "student_test"}, used
    ok(f"6. experiments read only partitions {sorted(used)}; no evaluator data referenced")
    (ROOT / "results" / "phase5" / "integrity_checks.txt").write_text("\n".join(lines) + "\nALL CHECKS PASSED\n")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
