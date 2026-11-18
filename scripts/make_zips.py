"""Build the two hand-off zips into dist/.

  dist/pathsnap_repo.zip        complete repo WITHOUT raw dataset, instruction PDFs, caches, venv, .git
  dist/pathsnap_laptop2_ui.zip  ONLY the ui/ folder (+ LAPTOP2_README.txt); verified to contain
                                no data, no model and no .route files

Run:  .venv\\Scripts\\python.exe scripts/make_zips.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
EXCLUDE_DIRS = {"Student folder", "PathSnap_student_materials", ".venv", "cache", ".git", "dist",
                "__pycache__", ".claude", ".pytest_cache"}
EXCLUDE_EXT = {".pyc", ".zip"}
DATA_EXT = {".track", ".nodes", ".arcs", ".route"}


def files_under(base):
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_dir() or any(part in EXCLUDE_DIRS for part in rel.parts) or p.suffix in EXCLUDE_EXT:
            continue
        yield p, rel


def main():
    DIST.mkdir(exist_ok=True)
    repo = DIST / "pathsnap_repo.zip"
    with zipfile.ZipFile(repo, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for p, rel in files_under(ROOT):
            assert p.suffix not in DATA_EXT, f"raw data file would be zipped: {rel}"
            z.write(p, Path("pathsnap") / rel)
            n += 1
    print(f"{repo.name}: {n} files, {repo.stat().st_size / 1e6:.1f} MB")

    ui = DIST / "pathsnap_laptop2_ui.zip"
    with zipfile.ZipFile(ui, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for p, rel in files_under(ROOT / "ui"):
            z.write(p, Path("pathsnap_ui") / rel.relative_to("ui"))
            n += 1
    with zipfile.ZipFile(ui) as z:
        names = z.namelist()
    bad = [x for x in names if Path(x).suffix in DATA_EXT | {".joblib", ".npz", ".pkl", ".csv"}]
    assert not bad, f"UI zip contains data/model files: {bad}"
    assert any(x.endswith("LAPTOP2_README.txt") for x in names)
    print(f"{ui.name}: {n} files, {ui.stat().st_size / 1e6:.2f} MB; no data/model/.route files: OK")
    for x in names:
        print("   ", x)


if __name__ == "__main__":
    main()
