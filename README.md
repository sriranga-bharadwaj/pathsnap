# PathSnap - Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon (PES University). PathSnap matches noisy GPS
trajectories to a road network: a classifier scores (GPS observation, candidate road arc)
pairs and an HMM/Viterbi decoder picks the most plausible connected route. It runs on two
laptops: **Laptop 1** hosts the model + FastAPI, **Laptop 2** hosts a static web UI.

See `architecture.md` for the system design and response contract, and
`AI_DISCLOSURE.md` for AI-assistance disclosure.

## Repository layout
| Folder / file | Contents |
|---|---|
| `src/` | data loading, preprocessing, network + spatial index, alignment (labels), candidates, features, HMM, evaluation, prediction pipeline, server-side evaluation |
| `experiments/` | `00_eda` -> `01_alignment_candidates` -> `02_build_features` -> `03_train_compare` -> `04_hmm_tune_compare` -> `05_optuna_tune` -> `06_final_analysis` -> `07_integrity_checks` (run in this order) |
| `api/` | FastAPI app (`python -m api.main`) |
| `ui/` | complete Laptop-2 app (HTML + JS + vendored Leaflet) |
| `artifacts/` | saved model (`model.joblib`), `model_meta.json`, `decoder_config.json` |
| `results/` | CSVs, figures, Folium maps, run logs, `experiments_log.csv` |
| `scripts/smoke_test.py` | calls /health, /trips, /predict for every trip (standard library only) |
| `run_api.bat` / `run_api.sh` | start the API (Laptop 1) |

The raw dataset is **not** in the repo. Place it as:
```
<repo>/Student folder/Student folder/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
<repo>/Student folder/Student folder/student_test/<trip_id>/...
```
(or set `DATA_ROOT` for experiments and `DATA_DIR` for the API).

## Install (Laptop 1)
Python 3.12 (Windows tested). From the repo folder:
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Linux/macOS: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
Versions are pinned: the saved scikit-learn model must be loaded with scikit-learn 1.9.1.

## Reproduce the experiments (optional, Laptop 1)
```
.venv\Scripts\python.exe experiments/00_eda.py
.venv\Scripts\python.exe experiments/01_alignment_candidates.py
.venv\Scripts\python.exe experiments/01b_label_diagnostics.py
.venv\Scripts\python.exe experiments/02_build_features.py
.venv\Scripts\python.exe experiments/03_train_compare.py
.venv\Scripts\python.exe experiments/03b_baseline_nearest.py
```
(`03_train_compare.py` trains all six models: LR, GaussianNB, RandomForest, HistGradientBoosting,
MLP, LightGBM. LightGBM was added after the first run with `03_train_compare.py --only LightGBM`,
which merges its rows into the result tables without changing `artifacts/`.)
```
.venv\Scripts\python.exe experiments/04_hmm_tune_compare.py
.venv\Scripts\python.exe experiments/05_optuna_tune.py
.venv\Scripts\python.exe experiments/04_hmm_tune_compare.py --out results/phase5/decoder
.venv\Scripts\python.exe experiments/06_final_analysis.py
.venv\Scripts\python.exe experiments/07_integrity_checks.py
```
`05` runs 30 Optuna trials (about 50 minutes). The final model is then chosen at SYSTEM level
(classifier + HMM) on training data only:
```
.venv\Scripts\python.exe experiments/04_hmm_tune_compare.py --model LightGBM --out results/phase5/decoder_lightgbm
.venv\Scripts\python.exe scripts/select_final_model.py
.venv\Scripts\python.exe experiments/04_hmm_tune_compare.py --out results/phase5/decoder_final
```
`select_final_model.py` deploys the candidate (untuned RF, tuned RF, LightGBM) with the highest
train out-of-fold Top-1 after HMM decoding (result: untuned RandomForest, see
`results/phase5/final_model_selection.csv`); the last `04` run writes its `artifacts/decoder_config.json`.
Each script prints its own timings; 01 and 03 take a few minutes each, the rest under a few
minutes on a 12-core laptop. `cache/` holds intermediate files.

## Run the system

### Laptop 1 - model + API
```
run_api.bat                                   (uses student_test by default)
run_api.bat "D:\path\to\evaluator_folder"     (folder with one sub-folder per trip)
```
or `set DATA_DIR=...` then `run_api.bat`; or `.venv\Scripts\python.exe -m api.main --data-dir "<folder>" --port 8000`.
The API binds `0.0.0.0:8000`, enables CORS, and logs every request (trip_id, status, latency)
to the console - keep this window visible for the evaluators.

Find Laptop 1's IP: run `ipconfig` and read **IPv4 Address** of the Wi-Fi adapter (e.g. `192.168.1.23`).

Allow port 8000 through Windows Firewall (run once, in an **Administrator** command prompt):
```
netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000
```
Also make sure the Wi-Fi network profile is **Private** (Settings > Network > Wi-Fi > network > Private).

### Laptop 2 - UI only (needs Python only, no pip installs)
```
cd ui
run_ui.bat          (or ./run_ui.sh)
```
Open `http://localhost:5500`, type `http://<laptop1-ip>:8000` into the API box, press **Connect**,
then click a trip. (You can also set the default in `ui/config.js`.)

Both laptops must be on the **same Wi-Fi or phone hotspot**. Quick check from Laptop 2:
browser or `curl http://<laptop1-ip>:8000/health`.

### Smoke test (either laptop)
```
python scripts/smoke_test.py http://<laptop1-ip>:8000
```

## Troubleshooting
| Symptom | Fix |
|---|---|
| UI: "API unreachable" | API running on Laptop 1? Same Wi-Fi? Firewall rule added? Network set to Private? Correct IP (re-run `ipconfig`, it can change)? |
| `/health` works on Laptop 1 but not from Laptop 2 | firewall / network profile; some campus Wi-Fi blocks device-to-device traffic - use a phone hotspot |
| `/trips` returns 0 trips | `DATA_DIR` must point at the folder that directly contains the trip folders |
| Model fails to load | wrong scikit-learn version; reinstall with `pip install -r requirements.txt` |
| Map shows no streets | basemap needs internet; points and routes still draw on the blank map |
| First request of a trip is slower | the trip's road network and spatial index are built once, then cached in memory |

## Integrity
- The prediction path (`src/pipeline.py` and what it imports) never reads `.route`;
  `src/server_eval.py` reads it only after prediction and returns numbers only.
- No reference labels or reference geometry are sent to the UI.
- The 10 evaluator trips were never used for training, tuning, thresholds or model choice.
