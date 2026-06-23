#!/usr/bin/env bash
# Main-env setup: system ffmpeg, clone model repos, install main requirements.
# Run once per Colab session (fast if MODEL_ROOT is on Drive).
set -e

ROOT="${IMAGE_TALK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
THIRD_PARTY="$ROOT/third_party"
mkdir -p "$THIRD_PARTY"

echo "==> apt: ffmpeg"
apt-get -qq update && apt-get -qq install -y ffmpeg git-lfs >/dev/null

echo "==> cloning model repos"
clone() {  # clone <url> <dir>
  if [ ! -d "$2/.git" ]; then git clone --depth 1 "$1" "$2"; else echo "  $2 exists"; fi
}
clone https://github.com/OpenTalker/SadTalker.git      "$THIRD_PARTY/SadTalker"
clone https://github.com/Rudrabha/Wav2Lip.git          "$THIRD_PARTY/Wav2Lip"
clone https://github.com/bytedance/LatentSync.git       "$THIRD_PARTY/LatentSync"
clone https://github.com/TMElyralab/MuseTalk.git        "$THIRD_PARTY/MuseTalk"
clone https://github.com/sczhou/CodeFormer.git          "$THIRD_PARTY/CodeFormer"

echo "==> pip: main requirements"
pip install -q -r "$ROOT/requirements/main.txt"

# basicsr (pulled by gfpgan) imports torchvision.transforms.functional_tensor,
# which newer torchvision removed. Patch the import to functional.
echo "==> patching basicsr/gfpgan torchvision import"
python - <<'PY'
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

echo "==> main env ready."
