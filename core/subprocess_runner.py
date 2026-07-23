"""
Run an engine worker inside its own isolated venv via subprocess.

Why: XTTS, LatentSync, LTX, Wan2.2 and Qwen-Image-Edit each pin mutually
incompatible torch/transformers/numpy versions that also clash with the main
env's SDXL/diffusers stack. Each runs in a dedicated venv; we shell out to
that venv's python on a small worker script and read back the output path.

Protocol:
  - args are JSON-serialized to a temp file, passed as a single argv.
  - the worker does its work, then prints exactly one line:
        RESULT::/abs/path/to/output
    on success, or exits non-zero on failure.
"""
import os
import re
import sys
import json
import shutil
import tempfile
import subprocess

RESULT_PREFIX = "RESULT::"


def _free_gb(path):
    """Free space (GB) at the nearest existing ancestor of `path`."""
    p = path or "."
    while p and not os.path.isdir(p):
        p = os.path.dirname(p)
    try:
        return shutil.disk_usage(p or ".").free / 1024**3
    except Exception:
        return None


def warn_low_disk(min_gb, label="this model"):
    """Non-blocking heads-up (printed to the worker console) when the HF cache
    disk is low BEFORE a big download — so a half-written, disk-full failure is
    anticipated rather than mysterious. Never blocks: a fully-cached model needs
    no free space, so we must not abort a working re-run."""
    path = os.environ.get("HF_HUB_CACHE") or os.getcwd()
    free = _free_gb(path)
    if free is not None and free < min_gb:
        print(
            f"[disk] WARNING: only {free:.1f} GB free at {path}; {label} may "
            f"need up to ~{min_gb} GB if not already cached. If this fails with "
            f"'No space left on device', set USE_DRIVE = True (notebook Step 2) "
            f"to cache weights on Google Drive, or clear models/hf_cache.",
            flush=True,
        )


def _diagnose(stdout, stderr):
    """Translate a worker's raw failure into one actionable sentence for the UI.
    Returns None when the failure isn't a known pattern (caller keeps detail)."""
    blob = ((stderr or "") + "\n" + (stdout or "")).lower()
    if "no space left on device" in blob or "errno 28" in blob:
        return ("Disk full on the Colab VM — the model weights no longer fit on "
                "local disk. Fix: set USE_DRIVE = True (notebook Step 2) to cache "
                "weights on Google Drive, and/or clear old downloads under "
                "models/hf_cache; then restart the runtime and retry.")
    if "no module named 'torch'" in blob or "no module named torch" in blob:
        return ("This engine's isolated venv is missing PyTorch — its build did "
                "not finish (often because the disk filled during install). "
                "Re-run the setup cell that builds this venv (e.g. "
                "setup/make_wan_venv.sh for Text→Video / Wan2.2), then retry.")
    m = re.search(r"no module named ['\"]([^'\"]+)['\"]", blob)
    if m:
        return (f"This engine's venv is missing '{m.group(1)}' — its build "
                f"likely did not complete. Re-run the venv's setup cell, then "
                f"retry.")
    if ("out of memory" in blob or "outofmemoryerror" in blob
            or "cuda oom" in blob):
        return ("GPU ran out of memory. Lower the resolution / frame count, make "
                "sure no other heavy tab is mid-run, or restart the runtime to "
                "clear VRAM, then retry.")
    return None


def clean_env(extra=None):
    """
    Env for subprocess engines running in isolated venvs:
      - drop PYTHONPATH so the main env's site-packages don't leak in,
      - force a headless matplotlib backend (Colab sets MPLBACKEND to an
        inline backend that doesn't exist outside the notebook kernel, which
        crashes `import matplotlib` in XTTS / SadTalker).
    """
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["MPLBACKEND"] = "Agg"
    if extra:
        env.update(extra)
    return env


def run_worker(venv_python, worker_script, args: dict,
               cwd=None, extra_env=None, timeout=None):
    """
    Returns the output path printed by the worker.
    Raises RuntimeError with captured stderr on failure.
    """
    if not os.path.exists(venv_python):
        raise RuntimeError(
            f"venv interpreter not found: {venv_python}\n"
            f"Run setup/make_venvs.sh first."
        )
    if not os.path.exists(worker_script):
        raise RuntimeError(f"worker script not found: {worker_script}")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(args, tmp)
    tmp.close()

    env = clean_env(extra_env)

    try:
        proc = subprocess.run(
            [venv_python, worker_script, tmp.name],
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    if proc.returncode != 0:
        # Always echo full detail to the app console for debugging ...
        sys.stderr.write(
            f"[run_worker] {os.path.basename(worker_script)} failed "
            f"(exit {proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout[-1500:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}\n"
        )
        sys.stderr.flush()
        # ... but surface a clean, actionable message to the UI when we can.
        hint = _diagnose(proc.stdout, proc.stderr)
        if hint:
            raise RuntimeError(hint)
        last = ""
        if proc.stderr.strip():
            last = proc.stderr.strip().splitlines()[-1]
        raise RuntimeError(
            f"Worker failed (exit {proc.returncode}). See the Colab console for "
            f"the full traceback. {last}"
        )

    result = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            result = line[len(RESULT_PREFIX):].strip()

    if not result or not os.path.exists(result):
        raise RuntimeError(
            f"Worker produced no valid RESULT path.\n"
            f"--- stdout ---\n{proc.stdout[-1500:]}"
        )
    return result


def read_args():
    """Called inside a worker: load the JSON args file passed as argv[1]."""
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        return json.load(f)


def emit_result(path):
    """Called inside a worker: report the output path back to the runner."""
    print(f"{RESULT_PREFIX}{os.path.abspath(path)}", flush=True)
