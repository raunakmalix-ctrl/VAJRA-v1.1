#!/usr/bin/env python3
"""Build the handover archive.

    python3 tools/make_handover.py [-o DIR] [--with-weights]

What is deliberately EXCLUDED, and why each one matters:

  venvs/, venv_main/   Virtual environments are not portable. They hardcode
                       absolute paths and are built against a specific
                       interpreter and CUDA runtime; copying them to another
                       machine produces something that looks installed and
                       fails at import. They are rebuilt by install.sh in
                       minutes.
  models/              Tens to hundreds of gigabytes, and re-downloadable.
                       Include them only for an air-gapped target
                       (--with-weights), where there is no alternative.
  third_party/         Cloned from upstream by install.sh at fixed versions.
  outputs/, uploads/   The previous operator's media. Shipping it would be a
                       privacy problem as much as a size one.
  .git/                History is not needed to run the software, and roughly
                       doubles the archive.

The archive carries a MANIFEST with a SHA-256 for every file, so the recipient
can verify nothing was corrupted in transit -- a truncated zip that still opens
is a real failure mode when moving files between institutions.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXCLUDE_DIRS = {
    "venvs", "venv_main", "models", "third_party", "outputs", "uploads",
    ".git", "__pycache__", ".ipynb_checkpoints", ".claude", ".pytest_cache",
}
EXCLUDE_SUFFIX = (".pyc", ".pyo", ".wav", ".mp4", ".mov", ".mkv", ".zip")
# Media is excluded by suffix, but the interface's own assets must survive.
KEEP_ANYWAY = ("assets/",)


def wanted(rel):
    parts = rel.replace("\\", "/").split("/")
    if any(p in EXCLUDE_DIRS for p in parts):
        return False
    if any(rel.replace("\\", "/").startswith(k) for k in KEEP_ANYWAY):
        return True
    if rel.endswith(EXCLUDE_SUFFIX):
        return False
    return True


def collect():
    out = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            full = os.path.join(base, f)
            rel = os.path.relpath(full, ROOT)
            if wanted(rel):
                out.append((full, rel.replace("\\", "/")))
    return sorted(out, key=lambda t: t[1])


def git_describe():
    try:
        p = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        b = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        if p.returncode == 0:
            return f"{b.stdout.strip()} @ {p.stdout.strip()}"
    except Exception:
        pass
    return "unknown (not a git checkout)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out-dir", default=os.path.dirname(ROOT))
    ap.add_argument("--with-weights", action="store_true",
                    help="include models/ — only for an air-gapped target; "
                         "adds tens of gigabytes")
    args = ap.parse_args()

    files = collect()
    if args.with_weights:
        model_root = os.environ.get("VAJRA_MODELS",
                                    os.path.join(ROOT, "models"))
        if os.path.isdir(model_root):
            for base, _d, fs in os.walk(model_root):
                for f in fs:
                    full = os.path.join(base, f)
                    rel = "models/" + os.path.relpath(full, model_root).replace(
                        "\\", "/")
                    files.append((full, rel))
            print(f"including weights from {model_root}")
        else:
            print(f"!! --with-weights given but {model_root} does not exist",
                  file=sys.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    name = f"VAJRA_handover_{stamp}.zip"
    dest = os.path.join(args.out_dir, name)

    manifest = [
        "VAJRA handover archive",
        f"built     : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"source    : {git_describe()}",
        f"files     : {len(files)}",
        "",
        "NOT INCLUDED (rebuilt or re-downloaded by install.sh):",
        "  venvs/, venv_main/   not portable between machines",
        "  third_party/         cloned from upstream at fixed versions",
        "  models/              re-downloaded, unless --with-weights was used",
        "  outputs/, uploads/   the previous operator's media",
        "",
        "TO INSTALL:  see docs/VAJRA_Deployment_Guide.pdf, or:",
        "  python3 setup/preflight.py     # check the machine first",
        "  bash install.sh                # core modules",
        "  bash run.sh                    # start it",
        "",
        "TO USE:      docs/VAJRA_User_Manual.pdf -- every module and every",
        "             control, how to read the detectability report, and the",
        "             limits worth knowing before promising a result",
        "",
        "SHA-256:",
    ]

    total = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for full, rel in files:
            h = hashlib.sha256()
            try:
                with open(full, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                total += os.path.getsize(full)
            except OSError as e:
                print(f"!! skipped {rel}: {e}", file=sys.stderr)
                continue
            manifest.append(f"  {h.hexdigest()}  {rel}")
            z.write(full, arcname=f"VAJRA/{rel}")
        z.writestr("VAJRA/MANIFEST.txt", "\n".join(manifest) + "\n")

    size = os.path.getsize(dest)
    print(f"\nwrote {dest}")
    print(f"  {len(files)} files, {total / 1e6:.1f} MB uncompressed, "
          f"{size / 1e6:.1f} MB compressed")
    print(f"  source: {git_describe()}")
    if not args.with_weights:
        print("\n  Weights are NOT included. The target host downloads them "
              "during install.\n  For an air-gapped host, re-run with "
              "--with-weights (much larger).")


if __name__ == "__main__":
    main()
