#!/usr/bin/env bash
# Main-env setup: system ffmpeg, clone model repos, install main requirements.
# Run once per Colab session (fast if MODEL_ROOT is on Drive).
set -e

ROOT="${VAJRA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# Which pip to install with. Defaults to the ambient one, which is right in a
# notebook where the runtime IS the environment. On a plain host install.sh
# points this at venv_main instead, so nothing lands in system python.
PIP="${PIP:-pip}"
# The interpreter that goes with $PIP. The basicsr patch below inspects
# INSTALLED packages, so it has to run in the environment they were installed
# into -- bare `python` is the ambient one on a host install, where it would
# find nothing and silently skip the patch.
if [ -z "${PYBIN:-}" ]; then
  if [ "$PIP" != "pip" ] && [ -x "${PIP%/pip}/python" ]; then
    PYBIN="${PIP%/pip}/python"
  else
    PYBIN="python"
  fi
fi
# apt needs privileges the notebook already has; a normal login usually does not.
SUDO=""
if [ "$(id -u)" != "0" ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
THIRD_PARTY="$ROOT/third_party"
mkdir -p "$THIRD_PARTY"

echo "==> apt: ffmpeg"
$SUDO apt-get -qq update && $SUDO apt-get -qq install -y ffmpeg git-lfs >/dev/null

echo "==> cloning model repos"
clone() {  # clone <url> <dir>
  if [ ! -d "$2/.git" ]; then git clone --depth 1 "$1" "$2"; else echo "  $2 exists"; fi
}
clone https://github.com/Rudrabha/Wav2Lip.git          "$THIRD_PARTY/Wav2Lip"
clone https://github.com/bytedance/LatentSync.git       "$THIRD_PARTY/LatentSync"
clone https://github.com/sczhou/CodeFormer.git          "$THIRD_PARTY/CodeFormer"

echo "==> pip: main requirements"
"$PIP" install -q -r "$ROOT/requirements/main.txt"

# Face restoration, installed separately and tolerantly. These are 2021-2022
# packages with legacy setup.py builds: basicsr fails at `setup.py egg_info`
# on Python 3.13, which hosted notebooks now run. faceswap_engine imports
# gfpgan lazily, so the platform starts and every other tab works without
# them -- what is lost is the face ENHANCER, and the operator is told that
# here rather than discovering it as an ImportError mid-demo.
echo "==> pip: face restoration (optional)"
if "$PIP" install -q gfpgan basicsr facexlib lpips 2>/dev/null; then
  echo "    face enhancer available"
  ENHANCER_OK=1
else
  ENHANCER_OK=""
  echo "!! face restoration did not install on this interpreter."
  echo "   Face Swap still works; its enhancer options do not, so choose"
  echo "   'None'. Everything else is unaffected."
  echo "   Cause is almost always the Python version: these packages have no"
  echo "   wheels above 3.12 and their setup.py cannot build there."
fi

# basicsr (pulled by gfpgan) imports torchvision.transforms.functional_tensor,
# which newer torchvision removed. Patch the import to functional.
echo "==> patching basicsr/gfpgan torchvision import"
"$PYBIN" - <<'PY'
import importlib.util, os, re
for mod in ("basicsr",):
    spec = importlib.util.find_spec(mod)
    if not spec or not spec.submodule_search_locations:
        continue
    base = spec.submodule_search_locations[0]
    for dp, _, fs in os.walk(base):
        for f in fs:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f)
            s = open(p, encoding="utf-8").read()
            if "functional_tensor" in s:
                s = s.replace(
                    "torchvision.transforms.functional_tensor",
                    "torchvision.transforms.functional",
                )
                open(p, "w", encoding="utf-8").write(s)
                print("  patched", p)
PY

# Patch the cloned third-party repos (Wav2Lip librosa API, LatentSync
# missing-face tolerance).
bash "$ROOT/setup/patch_thirdparty.sh"

# rembg pulls CPU onnxruntime, which shadows onnxruntime-gpu and forces
# face swap / InsightFace onto CPU. Re-pin the SAME CUDA-12-compatible GPU
# version as requirements/main.txt (unpinned "latest" wants CUDA 13, which
# Colab doesn't have — "libcudart.so.13: cannot open shared object file").
#
# Version choice is made HERE, at install time, and tolerantly: the exact build
# depends on the host's CUDA, and pinning one version in requirements/main.txt
# meant that the day that version was delisted from PyPI, pip rejected the
# whole file and nothing installed at all. Try the CUDA-12 line newest-first,
# take the first that installs, and if none does, fall back to the CPU provider
# with a warning. Face swap on CPU is slow but correct; no face swap at all,
# or a half-installed environment, is neither.
echo "==> ensuring onnxruntime-gpu (GPU provider for face swap)"
"$PIP" uninstall -y -q onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
ORT_OK=""
for v in 1.22.0 1.21.1 1.20.2 1.20.0; do
  if "$PIP" install -q "onnxruntime-gpu==$v" 2>/dev/null; then
    ORT_OK="$v"
    echo "    onnxruntime-gpu $v"
    break
  fi
done
if [ -z "$ORT_OK" ]; then
  echo "!! no CUDA-12 onnxruntime-gpu build installed — falling back to CPU."
  echo "   Face swap will still work, more slowly. If this host has CUDA 13,"
  echo "   install a matching onnxruntime-gpu by hand."
  "$PIP" install -q onnxruntime || true
fi

# rembg -> pymatting -> cupy; Colab's cupy 14 is built for numpy 2 and crashes
# against our numpy<2. Pin a numpy-1.x-compatible cupy.
echo "==> pinning numpy-compatible cupy (for Media Studio background removal)"
"$PIP" install -q "cupy-cuda12x>=13,<14" || echo "  (cupy pin skipped)"

if [ -z "${ENHANCER_OK:-}" ]; then
  echo "==> main env ready — WITHOUT the face enhancer (see the warning above)."
else
  echo "==> main env ready."
fi
