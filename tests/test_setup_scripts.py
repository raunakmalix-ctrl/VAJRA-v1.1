# -*- coding: utf-8 -*-
"""Static checks on the environment builders.

make_venvs.sh runs the per-module builders CONCURRENTLY, but some of their work
targets resources there is only one of: the system site-packages and dpkg. Two
pip processes installing into the same site-packages simultaneously delete and
rewrite each other's files mid-read, which surfaces as an OSError naming a file
in a package nobody asked for. That failure needs three parallel builds on a
clean machine to reproduce, so it cannot be caught by running anything here --
but it CAN be caught by checking that the shared steps stay behind the lock.
"""
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP = os.path.join(_ROOT, "setup")
COMMON = "_venv_common.sh"

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def read(fname):
    with open(os.path.join(SETUP, fname), encoding="utf-8") as fh:
        return fh.read()


builders = sorted(f for f in os.listdir(SETUP)
                  if f.startswith("make_") and f.endswith("_venv.sh"))

print("=" * 74)
print("setup — shared system-level steps are serialised")
print("=" * 74)
check("builders found", len(builders) >= 6, str(builders))

common = read(COMMON)
check("a lock guards the shared bootstrap", "flock" in common)
check("virtualenv install lives behind the lock helper",
      "_with_bootstrap_lock _install_virtualenv" in common)
check("apt install lives behind the lock helper",
      "_with_bootstrap_lock _install_py310" in common)
check("missing flock degrades instead of breaking the build",
      "command -v flock" in common)

# The fast path must not serialise every builder on the lock.
check("already-installed virtualenv short-circuits before locking",
      re.search(r"ensure_virtualenv\(\)\s*\{\s*\n(?:\s*#.*\n)*\s*python -m virtualenv --version",
                common) is not None)
check("already-installed python3.10 short-circuits before locking",
      "if ! command -v python3.10" in common)

print()
print("=" * 74)
print("setup — no builder bypasses the shared helpers")
print("=" * 74)
for b in builders:
    src = read(b)
    # An inline `pip install virtualenv` is exactly the bug: it writes to the
    # system site-packages outside the lock, from up to six parallel processes.
    inline = [ln.strip() for ln in src.split("\n")
              if re.search(r"pip install .*\bvirtualenv\b", ln)]
    check(f"{b} does not install virtualenv itself", not inline,
          "; ".join(inline))
    check(f"{b} sources {COMMON}", COMMON in src)
    check(f"{b} uses an ensure_py helper",
          re.search(r"\bensure_py31[02]\b", src) is not None)

    # apt outside the helper would contend on the dpkg lock the same way.
    apt = [ln.strip() for ln in src.split("\n")
           if re.search(r"^\s*apt-get\s", ln)]
    check(f"{b} does not call apt-get directly", not apt, "; ".join(apt))

print()
print("=" * 74)
print("setup — dispatcher targets map to real scripts")
print("=" * 74)
disp = read("make_venvs.sh")
targets = set()
for line in disp.split("\n"):
    if line.strip().startswith("ALL_TARGETS="):
        targets = set(line.split('"')[1].split())
        break
check("dispatcher declares targets", bool(targets), str(sorted(targets)))
mapped = dict(re.findall(r"\[(\w+)\]=\"([\w.]+\.sh)\"", disp))
for t in sorted(targets):
    script = mapped.get(t)
    check(f"target '{t}' maps to a script that exists",
          bool(script) and os.path.exists(os.path.join(SETUP, script)),
          script or "unmapped")
check("no-arg dispatcher builds nothing",
      'if [ "$#" -eq 0 ]' in disp and "exit 0" in disp)

# ── the host installation path ─────────────────────────────────────────────
# install.sh / run.sh / preflight.py are what a receiving institution actually
# runs. They are shell and therefore not exercised by any unit test; these are
# the static properties that, if broken, would strand a recipient on a machine
# the author cannot reach.
print("=" * 74)
print("setup - host installation path")
print("=" * 74)


def read_root(fname):
    with open(os.path.join(_ROOT, fname), encoding="utf-8") as fh:
        return fh.read()


for fname in ("install.sh", "run.sh"):
    check(f"{fname} exists", os.path.exists(os.path.join(_ROOT, fname)))
check("preflight.py exists", os.path.exists(os.path.join(SETUP, "preflight.py")))

inst = read_root("install.sh")
# Compare the EXECUTED lines only. install.sh's header comment lists the five
# steps in order, so a naive search finds download_models.py in the comment
# long before preflight.py is actually invoked.
inst_code = "\n".join(l for l in inst.split("\n")
                      if l.strip() and not l.lstrip().startswith("#"))
check("install.sh runs preflight before anything is downloaded",
      inst_code.index("preflight.py") < inst_code.index("download_models.py"))
check("install.sh builds venv_main before installing into it",
      inst_code.index("venv_main") < inst_code.index("install_main.sh"))
check("install.sh points install_main.sh at that venv, not system pip",
      'PIP="$VENV_MAIN/bin/pip"' in inst)
inst_targets = set(re.findall(r"TARGETS=\(([a-z0-9 ]+)\)", inst))
declared = set()
for group in inst_targets:
    declared |= set(group.split())
check("every target install.sh offers is one the dispatcher knows",
      declared <= targets, f"{sorted(declared)} vs {sorted(targets)}")

main_sh = read("install_main.sh")
check("install_main.sh takes a pip from the caller", 'PIP="${PIP:-pip}"' in main_sh)
check("install_main.sh elevates only for apt", "$SUDO apt-get" in main_sh)
check("install_main.sh no longer calls bare pip install",
      not re.search(r"^\s*pip install", main_sh, re.M))

run_sh = read_root("run.sh")
check("run.sh refuses to start when not installed", "install.sh" in run_sh
      and "exit 1" in run_sh)
check("run.sh defaults sharing OFF",
      'VAJRA_SHARE="${VAJRA_SHARE:-0}"' in run_sh)

pre = read("preflight.py")
for needed in ("python3.12", "python3.10", "ffmpeg", "nvidia-smi"):
    check(f"preflight checks for {needed}", needed in pre)
check("preflight exits non-zero on a blocking problem", "sys.exit(1)" in pre)

print("=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL SETUP SCRIPT CHECKS PASSED")
