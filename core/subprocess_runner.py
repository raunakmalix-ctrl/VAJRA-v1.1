"""
Run an engine worker inside its own isolated venv via subprocess.

Why: XTTS (Coqui), SadTalker and LatentSync pin mutually incompatible
torch/transformers/numpy versions that also clash with the main env's
FLUX/diffusers stack. Each runs in a dedicated venv; we shell out to that
venv's python on a small worker script and read back the output path.

Protocol:
  - args are JSON-serialized to a temp file, passed as a single argv.
  - the worker does its work, then prints exactly one line:
        RESULT::/abs/path/to/output
    on success, or exits non-zero on failure.
"""
import os
import sys
import json
import tempfile
import subprocess

RESULT_PREFIX = "RESULT::"


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

    env = os.environ.copy()
    # Don't let the main env's site-packages leak into the venv subprocess.
    env.pop("PYTHONPATH", None)
    if extra_env:
        env.update(extra_env)

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
        tail_out = proc.stdout[-1500:]
        tail_err = proc.stderr[-2000:]
        raise RuntimeError(
            f"Worker failed (exit {proc.returncode}).\n"
            f"--- stdout ---\n{tail_out}\n--- stderr ---\n{tail_err}"
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
