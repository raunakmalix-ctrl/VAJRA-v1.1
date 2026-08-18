#!/usr/bin/env python3
"""Check a machine can run VAJRA, before anything is installed.

Run this first on a new host. Every check reports what it found, what is
needed, and what to do about it -- a bare pass/fail would leave the operator
guessing, and most of these failures otherwise surface an hour into a
multi-gigabyte install.

    python3 setup/preflight.py

Exit code 0 means the machine can run the core modules. Warnings do not fail
the run: several are only relevant to optional modules.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OK, WARN, BAD = "  OK   ", "  WARN ", "  FAIL "
problems = []
warnings = []


def report(state, name, detail=""):
    print(f"{state} {name}" + (f"  —  {detail}" if detail else ""))
    if state is BAD:
        problems.append(name)
    elif state is WARN:
        warnings.append(name)


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return 1, str(e)


def which_version(exe, args=("--version",)):
    path = shutil.which(exe)
    if not path:
        return None, None
    _rc, out = run([exe, *args])
    return path, out.strip().split("\n")[0]


print("=" * 72)
print("VAJRA preflight")
print("=" * 72)

# ── operating system ───────────────────────────────────────────────────────
if sys.platform != "linux":
    report(WARN, "operating system", f"{sys.platform}; the setup scripts assume Linux")
else:
    report(OK, "operating system", "linux")

# ── GPU ────────────────────────────────────────────────────────────────────
smi = shutil.which("nvidia-smi")
if not smi:
    report(BAD, "NVIDIA driver",
           "nvidia-smi not found. Install the driver; without a GPU only "
           "Media Studio is usable.")
else:
    rc, out = run(["nvidia-smi",
                   "--query-gpu=name,memory.total,driver_version",
                   "--format=csv,noheader"])
    if rc != 0:
        report(BAD, "NVIDIA driver", "nvidia-smi failed to run")
    else:
        line = out.strip().split("\n")[0]
        report(OK, "GPU", line)
        try:
            vram_mb = int(line.split(",")[1].strip().split()[0])
            if vram_mb < 20000:
                report(WARN, "GPU memory",
                       f"{vram_mb // 1024} GB. Video Edit needs roughly 16-24 GB; "
                       f"the video-generation modules need far more.")
        except Exception:
            pass

# ── interpreters ───────────────────────────────────────────────────────────
# Two are needed, and this is the least obvious prerequisite: the voice and
# lip-sync stacks depend on packages that publish no wheels for 3.12, so they
# are built from 3.10 while everything else runs on 3.12.
for exe, why in (("python3.12", "principal runtime and the newer module stacks"),
                 ("python3.10", "voice and lip-sync stacks (no 3.12 wheels exist)")):
    path, ver = which_version(exe)
    if path:
        report(OK, exe, ver or path)
    else:
        report(BAD, exe,
               f"needed for the {why}. "
               f"apt install {exe} {exe}-venv {exe}-dev")

# ── tools ──────────────────────────────────────────────────────────────────
for exe, hint in (("ffmpeg", "apt install ffmpeg"),
                  ("git", "apt install git"),
                  ("bash", "required by the setup scripts")):
    path, ver = which_version(exe)
    if path:
        report(OK, exe, (ver or "").split(" version ")[-1][:40] or path)
    else:
        report(BAD, exe, hint)

for exe, hint in (("flock", "util-linux; without it parallel builds are serialised"),
                  ("git-lfs", "apt install git-lfs; some model repos need it")):
    path, _ = which_version(exe, ("--version",))
    report(OK if path else WARN, exe, "" if path else hint)

# ── privileges ─────────────────────────────────────────────────────────────
if os.geteuid() == 0:
    report(OK, "privileges", "running as root; apt steps will work")
elif shutil.which("sudo"):
    report(OK, "privileges", "sudo available for the apt steps")
else:
    report(WARN, "privileges",
           "not root and no sudo. Install ffmpeg/python3.10 beforehand, then "
           "the rest runs unprivileged.")

# ── disk ───────────────────────────────────────────────────────────────────
try:
    free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
    if free_gb < 60:
        report(BAD, "free disk", f"{free_gb:.0f} GB. The core modules need ~60 GB "
                                 f"(environments plus weights).")
    elif free_gb < 120:
        report(WARN, "free disk",
               f"{free_gb:.0f} GB. Enough for the core modules; the video-generation "
               f"modules alone exceed 100 GB each.")
    else:
        report(OK, "free disk", f"{free_gb:.0f} GB")
except Exception as e:
    report(WARN, "free disk", str(e))

# ── network ────────────────────────────────────────────────────────────────
# Model weights are fetched at install time. On an air-gapped host they must be
# copied in instead, which changes the whole procedure -- so it is worth
# knowing now rather than at the first download.
rc, _ = run(["python3", "-c",
             "import socket;socket.create_connection(('huggingface.co',443),5)"])
if rc == 0:
    report(OK, "reaches huggingface.co", "weights can be downloaded")
else:
    report(WARN, "reaches huggingface.co",
           "no route. This host is offline: copy a pre-populated model "
           "directory in and point VAJRA_MODELS at it. See the deployment "
           "guide, 'Air-gapped installation'.")

print("=" * 72)
if problems:
    print(f"{len(problems)} blocking problem(s): " + "; ".join(problems))
    print("Fix these, then run this check again.")
    sys.exit(1)
if warnings:
    print(f"Ready, with {len(warnings)} caveat(s): " + "; ".join(warnings))
else:
    print("Ready.")
sys.exit(0)
