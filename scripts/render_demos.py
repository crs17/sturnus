#!/usr/bin/env python3
"""Render notebooks/*.py to demos/ incrementally, skipping notebooks whose .md is newer than their .py."""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
print('ROOT', ROOT)
NOTEBOOKS_DIR = ROOT / "notebooks"
OUTPUT_DIR = ROOT / "demos" / "notebooks"


def needs_render(py_file: Path, md_file: Path) -> bool:
    if not md_file.exists():
        return True
    return py_file.stat().st_mtime > md_file.stat().st_mtime


def main():
    stale = []
    for py_file in sorted(NOTEBOOKS_DIR.glob("*.py")):
        md_file = OUTPUT_DIR / f"{py_file.stem}.md"
        files_dir = OUTPUT_DIR / f"{py_file.stem}_files"
        if needs_render(py_file, md_file):
            stale.append(py_file)
            md_file.unlink(missing_ok=True)
            if files_dir.exists():
                shutil.rmtree(files_dir)

    if not stale:
        print("All notebooks up to date, nothing to render.")
        return

    for py_file in stale:
        print(f"Rendering {py_file.relative_to(ROOT)}...")
        subprocess.run(
            ["uv", "run", "quarto", "render", str(py_file.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()