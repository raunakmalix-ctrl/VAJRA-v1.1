#!/usr/bin/env bash
# Idempotent patches to the cloned third-party repos. Safe to re-run.
set -e
ROOT="${IMAGE_TALK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
TP="$ROOT/third_party"

# 1. Wav2Lip: librosa 0.10 made filters.mel() keyword-only.
W="$TP/Wav2Lip/audio.py"
if [ -f "$W" ]; then
  sed -i 's/librosa\.filters\.mel(hp\.sample_rate, hp\.n_fft/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft/' "$W"
  echo "patched Wav2Lip/audio.py"
fi

# 2. LatentSync: don't crash on a single frame with no detected face — reuse the
#    last good detection instead (raise only if it never detected one).
L="$TP/LatentSync/latentsync/utils/image_processor.py"
if [ -f "$L" ]; then
  python - "$L" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if "_last_affine" not in s:
    s = s.replace(
        '        if bbox is None:\n            raise RuntimeError("Face not detected")',
        '        if bbox is None:\n'
        '            if getattr(self, "_last_affine", None) is not None:\n'
        '                return self._last_affine\n'
        '            raise RuntimeError("Face not detected")',
    )
    s = s.replace(
        '        face = rearrange(torch.from_numpy(face), "h w c -> c h w")\n'
        '        return face, box, affine_matrix',
        '        face = rearrange(torch.from_numpy(face), "h w c -> c h w")\n'
        '        self._last_affine = (face, box, affine_matrix)\n'
        '        return face, box, affine_matrix',
    )
    open(p, "w").write(s)
    print("patched LatentSync image_processor.py")
else:
    print("LatentSync image_processor.py already patched")
PY
fi

# 3. CodeFormer vendors its own basicsr, which also imports the removed
#    torchvision.transforms.functional_tensor — patch it too.
CF="$TP/CodeFormer/basicsr"
if [ -d "$CF" ]; then
  grep -rl "functional_tensor" "$CF" 2>/dev/null | while read -r f; do
    sed -i 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/' "$f"
  done
  echo "patched CodeFormer basicsr"
fi

echo "third-party patches applied."
