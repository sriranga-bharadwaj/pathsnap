"""Split the uncommitted work (per `git status`) into one zip per team member, so each member
commits their part from their own GitHub account.

  member1_ui.zip          ui/ (Laptop-2 app)
  member2_api_deploy.zip  API, prediction pipeline, server-side evaluation, run scripts, docs,
                          deployment test results (results/phase4, API logs)
  member3_ml.zip          experiments (LightGBM, Optuna, final analysis, integrity checks),
                          artifacts, requirements, results/phase2 + results/phase5, experiment log

Every changed file goes into exactly one zip (asserted). Paths inside each zip are repo-relative:
unzip into the repository root. Each zip contains MEMBER_README.txt with the exact git commands.
Run:  .venv\\Scripts\\python.exe scripts/make_member_zips.py
"""
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "members"

GROUPS = {
    "member1_ui": dict(
        title="Laptop-2 UI (static HTML/JS + vendored Leaflet)",
        commit="Add Laptop-2 evaluator UI (Leaflet, configurable API URL, offline-safe)",
        match=lambda p: p.startswith("ui/")),
    "member2_api_deploy": dict(
        title="FastAPI backend, prediction pipeline, server-side evaluation, deployment docs",
        commit="Add FastAPI backend, prediction pipeline, server-side evaluation and deployment docs",
        match=lambda p: (p.startswith(("api/", "scripts/", "results/phase4/", "results/phase5/api_"))
                         or p in {"src/pipeline.py", "src/server_eval.py", "run_api.bat", "run_api.sh",
                                  "README.md", "SETUP_FOR_ASSISTANT.md", "architecture.md", ".gitignore"})),
    "member3_ml": dict(
        title="LightGBM model, Optuna tuning, decoder re-tuning, final analysis, integrity checks",
        commit="Add LightGBM comparison, Optuna tuning, final analysis and integrity checks",
        match=lambda p: (p.startswith(("experiments/", "artifacts/", "results/phase2/", "results/phase5/"))
                         or p in {"requirements.txt", "results/experiments_log.csv", "AI_DISCLOSURE.md"})),
}


def changed_files():
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    files = []
    for line in out.splitlines():
        path = line[3:].strip().strip('"')
        files.append(path)
    return files


def main():
    files = changed_files()
    assigned = {}
    for f in files:
        owners = [g for g, spec in GROUPS.items() if spec["match"](f)]
        # member2 claims the API result files under results/phase5 before member3's broad rule
        if len(owners) > 1 and "member2_api_deploy" in owners:
            owners = ["member2_api_deploy"]
        assert len(owners) == 1, f"{f}: assigned to {owners or 'nobody'}"
        assigned.setdefault(owners[0], []).append(f)
    DIST.mkdir(parents=True, exist_ok=True)
    for g, spec in GROUPS.items():
        fl = sorted(assigned.get(g, []))
        readme = [f"PathSnap - {g}: {spec['title']}", "",
                  "1. In your clone of the PathSnap repo:  git pull",
                  "2. Unzip THIS zip into the repo root (the folder that contains .git),",
                  "   overwriting existing files. Do not unzip into a sub-folder.",
                  "3. Commit and push from your own account:",
                  "     git add " + " ".join(f'"{f}"' for f in fl),
                  f'     git commit -m "{spec["commit"]}"',
                  "     git push",
                  "   (if push is rejected: git pull --rebase, then git push)", "",
                  f"Files ({len(fl)}):"] + [f"  {f}" for f in fl]
        z = DIST / f"{g}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zz:
            for f in fl:
                zz.write(ROOT / f, f)
            zz.writestr("MEMBER_README.txt", "\r\n".join(readme))
        print(f"{z.name}: {len(fl)} files, {z.stat().st_size / 1e6:.2f} MB")
    total = sum(len(v) for v in assigned.values())
    assert total == len(files)
    print(f"all {len(files)} changed files assigned exactly once")


if __name__ == "__main__":
    main()
