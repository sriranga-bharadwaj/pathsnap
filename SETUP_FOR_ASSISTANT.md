# PathSnap - setup from the zips (step by step)

You have two zips: `pathsnap_repo.zip` (Laptop 1) and `pathsnap_laptop2_ui.zip` (Laptop 2),
plus the dataset folder (not inside the zips).

## Laptop 1 (model + API)
1. Unzip `pathsnap_repo.zip` to a folder without special characters if possible, e.g. `C:\pathsnap`.
   You should see `src\`, `api\`, `ui\`, `artifacts\`, `README.md`, `requirements.txt`, `run_api.bat`.
2. Install Python **3.12** (64-bit). Check: `python --version`.
3. In a command prompt inside `C:\pathsnap`:
   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
4. Dataset. Either
   - copy the provided data so this path exists:
     `C:\pathsnap\Student folder\Student folder\student_test\00000000\00000000.track` (the API default), or
   - point the API at any folder whose sub-folders are trip IDs (each with `.track`, `.nodes`, `.arcs`,
     and optionally `.route`): this is how the organisers' evaluator folder is used.
5. Start the API:
   ```
   run_api.bat
   run_api.bat "D:\path\to\evaluator_folder"
   ```
   Expected console lines: `model loaded: RandomForest ...`, `DATA_DIR = ... (10 trips)`.
   Keep this window open: it shows one log line per request.
6. Firewall (once, Administrator command prompt):
   ```
   netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000
   ```
   and set the Wi-Fi network profile to Private.
7. Note the IPv4 address from `ipconfig` (Wi-Fi adapter), e.g. `192.168.1.23`.

## Laptop 2 (UI only)
1. Unzip `pathsnap_laptop2_ui.zip` anywhere, e.g. `C:\pathsnap_ui`.
2. Python 3 must be installed (no pip packages needed).
3. Double-click `run_ui.bat` (or `python -m http.server 5500` in that folder).
4. Open `http://localhost:5500`, enter `http://<laptop1-ip>:8000`, press Connect.

## Smoke-test checklist
- [ ] Laptop 1 browser: `http://127.0.0.1:8000/health` returns `"status": "ok"`, `"n_trips": 10`.
- [ ] Laptop 2 browser: `http://<laptop1-ip>:8000/health` returns the same (proves network + firewall).
- [ ] `http://<laptop1-ip>:8000/trips` lists the 10 trip IDs.
- [ ] In the UI, click every trip once: each shows the GPS points, a green route, metrics and events,
      and the Laptop-1 console prints a `[PREDICT] trip=... status=ok` line.
- [ ] Optional, from the repo folder on either laptop:
      `python scripts/smoke_test.py http://<laptop1-ip>:8000` -> `10/10 trips returned status=ok`.

## Notes
- First request per trip builds that trip's road network index (up to ~1 s for large networks).
- If the evaluator folder has no `.route` files, `top1_accuracy` and `candidate_miss_rate` are `null`
  (expected), everything else works.
- Do not change `requirements.txt` versions: the saved model needs scikit-learn 1.9.1.
