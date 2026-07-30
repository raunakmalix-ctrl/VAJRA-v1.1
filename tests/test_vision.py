# -*- coding: utf-8 -*-
"""Tests for windowed lip-sync frame compositing (vision/)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vision
from vision import composite as vc

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


print("=" * 74)
print("frame_range")
print("=" * 74)
check("basic conversion", vc.frame_range(1.0, 2.0, 25) == (25, 50),
      str(vc.frame_range(1.0, 2.0, 25)))
# Must round OUTWARD: a partially-covered frame still shows part of the edit.
check("rounds outward", vc.frame_range(1.01, 1.99, 25) == (25, 50),
      str(vc.frame_range(1.01, 1.99, 25)))
check("clamps to frame count", vc.frame_range(0.0, 100.0, 25, n_frames=30) == (0, 30),
      str(vc.frame_range(0.0, 100.0, 25, n_frames=30)))
check("never negative", vc.frame_range(-5.0, 1.0, 25)[0] == 0)
check("zero fps is safe", vc.frame_range(1.0, 2.0, 0) == (0, 0))
check("empty range stays empty", vc.frame_range(2.0, 2.0, 25) == (50, 50))

print()
print("=" * 74)
print("merge_windows")
print("=" * 74)
w = vc.merge_windows([(1.0, 2.0), (5.0, 6.0)])
check("disjoint kept separate", len(w) == 2, str(w))
w = vc.merge_windows([(1.0, 2.0), (2.05, 3.0)], min_gap_sec=0.2)
check("near windows merged", w == [(1.0, 3.0)], str(w))
w = vc.merge_windows([(1.0, 2.0), (2.5, 3.0)], min_gap_sec=0.2)
check("distant windows not merged", len(w) == 2, str(w))
w = vc.merge_windows([(5.0, 6.0), (1.0, 2.0)])
check("unsorted input sorted", w[0][0] < w[1][0], str(w))
w = vc.merge_windows([(1.0, 2.0)], pad_sec=0.25)
check("padding applied", w == [(0.75, 2.25)], str(w))
w = vc.merge_windows([(0.1, 1.0)], pad_sec=0.5)
check("padding clamped at zero", w[0][0] == 0.0, str(w))
w = vc.merge_windows([(9.5, 10.0)], pad_sec=1.0, duration_sec=10.0)
check("padding clamped at duration", w[0][1] == 10.0, str(w))
w = vc.merge_windows([(1.0, 2.0), (1.5, 3.0)])
check("overlapping merged", w == [(1.0, 3.0)], str(w))
check("empty input", vc.merge_windows([]) == [] and vc.merge_windows(None) == [])
check("degenerate window dropped", vc.merge_windows([(2.0, 2.0)]) == [])
# Padding can make a degenerate window valid, which is correct.
check("padded degenerate becomes valid",
      vc.merge_windows([(2.0, 2.0)], pad_sec=0.1) == [(1.9, 2.1)])

print()
print("=" * 74)
print("coverage")
print("=" * 74)
check("half covered", abs(vc.coverage([(0.0, 5.0)], 10.0) - 0.5) < 1e-9)
check("clamped at 1", vc.coverage([(0.0, 20.0)], 10.0) == 1.0)
check("empty is zero", vc.coverage([], 10.0) == 0.0)
check("zero duration safe", vc.coverage([(0, 1)], 0) == 0.0)

print()
print("=" * 74)
print("masks")
print("=" * 74)
H, W = 120, 160
box = (40.0, 50.0, 110.0, 100.0)

m = vc.feathered_box_mask(H, W, box, feather=8)
check("box mask shape", m.shape == (H, W), str(m.shape))
check("box mask in [0,1]", m.min() >= 0.0 and m.max() <= 1.0,
      f"{m.min():.3f}..{m.max():.3f}")
check("centre is fully inside", m[75, 75] > 0.99, f"{m[75,75]:.3f}")
check("far outside is zero", m[5, 5] == 0.0, f"{m[5,5]:.3f}")
# The boundary itself is 0 and the mask ramps to 1 over `feather` px inward --
# that is the contract, and it is what guarantees no seam at the mask edge.
check("mask is zero exactly at the boundary", m[50, 75] == 0.0, f"{m[50,75]:.3f}")
check("mask is partial inside the feather band", 0.0 < m[54, 75] < 1.0,
      f"{m[54,75]:.3f}")
check("mask is full just past the feather band", m[59, 75] > 0.99,
      f"{m[59,75]:.3f}")
# Feathering must be monotonic inward, or the seam shows as a ring.
col = m[50:76, 75]
check("mask rises monotonically inward", np.all(np.diff(col) >= -1e-6),
      "non-monotonic")

e = vc.feathered_ellipse_mask(H, W, box, feather=8)
check("ellipse mask shape", e.shape == (H, W))
check("ellipse centre inside", e[75, 75] > 0.99, f"{e[75,75]:.3f}")
check("ellipse corner outside", e[52, 42] < 0.5, f"{e[52,42]:.3f}")
check("ellipse covers less than box", e.sum() < m.sum(),
      f"ellipse={e.sum():.0f} box={m.sum():.0f}")
check("zero feather gives hard edge",
      set(np.unique(vc.feathered_box_mask(H, W, box, feather=0))) <= {0.0, 1.0})

print()
print("=" * 74)
print("mouth_box")
print("=" * 74)
face = (50.0, 20.0, 150.0, 140.0)     # 100x120 face box
mb = vc.mouth_box(face, frame_shape=(200, 200))
check("mouth box below face midpoint", mb[1] > face[1] + (face[3] - face[1]) * 0.3,
      f"top={mb[1]:.1f} face_top={face[1]}")
check("mouth box reaches the chin", mb[3] >= face[3] - 1e-6, f"{mb[3]:.1f}")
check("mouth box clamped to frame",
      mb[0] >= 0 and mb[1] >= 0 and mb[2] <= 200 and mb[3] <= 200, str(mb))
tiny_frame = vc.mouth_box(face, frame_shape=(100, 100))
check("clamping respects small frames", tiny_frame[3] <= 100 and tiny_frame[2] <= 100,
      str(tiny_frame))

pts = np.array([[90, 40], [110, 40], [95, 100], [105, 100], [100, 110],
                [92, 95], [108, 95]], dtype=np.float32)
lb = vc.mouth_box_from_landmarks(pts, frame_shape=(200, 200))
check("landmark box returned", lb is not None)
check("landmark box centred on lower points", 80 < (lb[0] + lb[2]) / 2 < 120, str(lb))
check("landmark box in lower half", (lb[1] + lb[3]) / 2 > 60, str(lb))
check("too few landmarks -> None",
      vc.mouth_box_from_landmarks(np.array([[1, 2]])) is None)

print()
print("=" * 74)
print("temporal_weight")
print("=" * 74)
n = 10
ws = [vc.temporal_weight(i, n, ramp=3) for i in range(n)]
check("starts below 1", ws[0] < 1.0, f"{ws[0]:.3f}")
check("ends below 1", ws[-1] < 1.0, f"{ws[-1]:.3f}")
check("full strength in the middle", abs(ws[5] - 1.0) < 1e-9, f"{ws[5]:.3f}")
check("all within [0,1]", all(0.0 <= x <= 1.0 for x in ws))
check("ramp rises then falls",
      ws[0] < ws[1] < ws[2] and ws[-1] < ws[-2] < ws[-3],
      str([round(x, 3) for x in ws]))
check("ramp=0 is always full", all(vc.temporal_weight(i, n, ramp=0) == 1.0
                                  for i in range(n)))
check("single frame handled", 0.0 <= vc.temporal_weight(0, 1, ramp=3) <= 1.0)
check("empty window is zero", vc.temporal_weight(0, 0) == 0.0)
check("ramp larger than window is clamped",
      all(0.0 <= vc.temporal_weight(i, 4, ramp=99) <= 1.0 for i in range(4)))

print()
print("=" * 74)
print("blend")
print("=" * 74)
base = np.full((H, W, 3), 40, dtype=np.uint8)
over = np.full((H, W, 3), 200, dtype=np.uint8)

full = vc.blend(base, over, np.ones((H, W), np.float32))
check("mask=1 gives overlay", np.all(full == 200))
none_ = vc.blend(base, over, np.zeros((H, W), np.float32))
check("mask=0 gives base", np.all(none_ == 40))
check("dtype preserved", full.dtype == np.uint8, str(full.dtype))
half = vc.blend(base, over, np.full((H, W), 0.5, np.float32))
check("mask=0.5 blends midway", abs(int(half[0, 0, 0]) - 120) <= 1,
      str(half[0, 0, 0]))
check("strength scales the mask",
      np.all(vc.blend(base, over, np.ones((H, W), np.float32), strength=0.0) == 40))

# The critical guarantee: outside the mask, pixels are bit-identical.
m2 = vc.feathered_ellipse_mask(H, W, box, feather=6)
res = vc.blend(base, over, m2)
outside = m2 == 0.0
check("pixels outside the mask are untouched",
      np.array_equal(res[outside], base[outside]),
      f"{int(outside.sum())} px checked")

# Float frames pass through as float, no integer rounding.
fb = np.zeros((10, 10, 3), np.float32)
fo = np.ones((10, 10, 3), np.float32)
fr = vc.blend(fb, fo, np.full((10, 10), 0.25, np.float32))
check("float frames stay float", fr.dtype == np.float32 and abs(fr[0, 0, 0] - 0.25) < 1e-6,
      str(fr.dtype))
# Greyscale (2-D) frames.
g = vc.blend(np.zeros((10, 10), np.uint8), np.full((10, 10), 100, np.uint8),
             np.ones((10, 10), np.float32))
check("2-D frames supported", g.shape == (10, 10) and np.all(g == 100))

try:
    vc.blend(base, over[:, :80], np.ones((H, W), np.float32))
    check("shape mismatch raises", False)
except ValueError:
    check("shape mismatch raises ValueError", True)
try:
    vc.blend(base, over, np.ones((10, 10), np.float32))
    check("mask mismatch raises", False)
except ValueError:
    check("mask mismatch raises ValueError", True)

print()
print("=" * 74)
print("composite_window")
print("=" * 74)
nf = 8
bases = [np.full((H, W, 3), 40, np.uint8) for _ in range(nf)]
synced = [np.full((H, W, 3), 200, np.uint8) for _ in range(nf)]
mask = vc.feathered_ellipse_mask(H, W, box, feather=6)
outs = vc.composite_window(bases, synced, mask, ramp=2)
check("returns one frame per input", len(outs) == nf, str(len(outs)))
check("shapes and dtype preserved",
      outs[0].shape == bases[0].shape and outs[0].dtype == np.uint8)

cy, cx = 75, 75
centre_vals = [int(o[cy, cx, 0]) for o in outs]
check("first frame is partially blended", 40 < centre_vals[0] < 200,
      str(centre_vals[0]))
check("middle frames fully synced", centre_vals[nf // 2] == 200,
      str(centre_vals[nf // 2]))
check("last frame is partially blended", 40 < centre_vals[-1] < 200,
      str(centre_vals[-1]))
check("centre ramps up then down",
      centre_vals[0] < centre_vals[1] and centre_vals[-1] < centre_vals[-2],
      str(centre_vals))

# Outside the mask nothing changes, on every frame of the window.
outside = mask == 0.0
check("outside-mask pixels untouched across the whole window",
      all(np.array_equal(o[outside], b[outside]) for o, b in zip(outs, bases)))

check("mismatched lengths use the shorter",
      len(vc.composite_window(bases, synced[:5], mask)) == 5)
check("empty input gives empty output", vc.composite_window([], [], mask) == [])

print()
print("=" * 74)
print("changed_region_box")
print("=" * 74)
bs = [np.full((H, W, 3), 60, np.uint8) for _ in range(4)]
# Simulate a lip-sync model altering only a mouth region.
ss = []
for _ in range(4):
    f = bs[0].copy()
    f[70:95, 60:100] = 190
    ss.append(f)
cb = vc.changed_region_box(bs, ss)
check("box found for a local change", cb is not None, str(cb))
if cb:
    check("box encloses the changed pixels",
          cb[0] <= 60 and cb[1] <= 70 and cb[2] >= 100 and cb[3] >= 95, str(cb))
    check("box stays inside the frame",
          cb[0] >= 0 and cb[1] >= 0 and cb[2] <= W and cb[3] <= H, str(cb))

check("identical frames -> None", vc.changed_region_box(bs, [b.copy() for b in bs])
      is None)
# A global change means the model re-encoded the whole frame; masking part of
# that would leave a visible discontinuity, so masking must be declined.
gl = [np.full((H, W, 3), 120, np.uint8) for _ in range(3)]
check("global change -> None (decline to mask)",
      vc.changed_region_box(bs[:3], gl) is None)
check("empty input -> None", vc.changed_region_box([], []) is None)
check("shape mismatch -> None",
      vc.changed_region_box(bs, [np.zeros((10, 10, 3), np.uint8)]) is None)

# End-to-end: derive the box, build a mask, composite -- and confirm the region
# outside the model's own change is byte-identical to the original.
box2 = vc.changed_region_box(bs, ss)
mask2 = vc.feathered_ellipse_mask(H, W, box2, feather=5)
outs2 = vc.composite_window(bs, ss, mask2, ramp=1)
far = mask2 == 0.0
check("composite leaves untouched area byte-identical",
      all(np.array_equal(o[far], b[far]) for o, b in zip(outs2, bs)),
      f"{int(far.sum())} px")
check("composite did change the mouth area",
      int(outs2[2][82, 80, 0]) > 100, str(outs2[2][82, 80, 0]))

print()
print("=" * 74)
print("end-to-end saving estimate")
print("=" * 74)
# A realistic case: two short edits in a 10-minute clip.
dur = 600.0
edits = [(120.0, 120.6), (455.2, 456.1)]
wins = vc.merge_windows(edits, pad_sec=0.35, min_gap_sec=0.5, duration_sec=dur)
cov = vc.coverage(wins, dur)
check("windows stay separate", len(wins) == 2, str(wins))
check("coverage is a small fraction of the clip", cov < 0.01, f"{cov*100:.3f}%")
print(f"   lip-sync would run on {cov*100:.3f}% of the video "
      f"({sum(b-a for a,b in wins):.2f}s of {dur:.0f}s)")




print()
print("=" * 74)
print("LipSyncEngine.run_windowed — routing decisions")
print("=" * 74)
# Stub torch so importing the engine chain works without CUDA.
import types
_t = types.ModuleType("torch")


class _FT:
    pass


_t.Tensor = _FT
_t.cuda = types.SimpleNamespace(is_available=lambda: False,
                                empty_cache=lambda: None,
                                mem_get_info=lambda: (0, 0),
                                get_device_name=lambda i=0: "stub")
sys.modules.setdefault("torch", _t)

import engines.lipsync_engine as le

CALLS = {"whole": 0, "windows": []}


class Probe:
    """Stubs out ffprobe and the model so only the routing logic is exercised."""
    def __init__(self, fps=25.0, n=250, w=320, h=240, dur=10.0):
        self.vals = (fps, n, w, h, dur)


def install(probe, composite_ok=True):
    CALLS["whole"] = 0
    CALLS["windows"] = []
    le.probe_video = lambda p: probe.vals

    def fake_run(self, video_path, audio_path, method="latentsync",
                 inference_steps=20, guidance_scale=1.5):
        # Distinguish whole-video calls from per-window sub-clip calls.
        if str(video_path).endswith("_v.mp4"):
            CALLS["windows"].append(video_path)
            return video_path + ".synced.mp4"
        CALLS["whole"] += 1
        return "whole.mp4"

    le.LipSyncEngine.run = fake_run
    le._cut_video = lambda src, t0, t1, dst: dst
    le._cut_audio = lambda src, t0, t1, dst: dst
    le.to_wav = lambda p: (p, False)

    def fake_composite(src, frame_windows, out, wav, fps, w, h, mask_mouth=True):
        if not composite_ok:
            raise RuntimeError("composite failed")
        CALLS["frame_windows"] = frame_windows
        return out

    le._composite_video = fake_composite


eng = le.LipSyncEngine()

# The engine checks its inputs exist before doing anything, so give it real
# (empty) files -- ffprobe and the model are stubbed above.
import tempfile as _tf
_TD = _tf.mkdtemp(prefix="vajra_vis_")
VID = os.path.join(_TD, "v.mp4")
AUD = os.path.join(_TD, "a.wav")
for _p in (VID, AUD):
    open(_p, "wb").close()

# No windows -> whole video.
install(Probe())
r = eng.run_windowed(VID, AUD, [])
check("no windows falls back to whole video", CALLS["whole"] == 1 and r == "whole.mp4",
      f"whole={CALLS['whole']}")

# Small edits -> windowed.
install(Probe())
r = eng.run_windowed(VID, AUD, [(2.0, 2.5), (7.0, 7.4)])
check("small edits use the windowed path", CALLS["whole"] == 0,
      f"whole={CALLS['whole']}")
check("one model call per window", len(CALLS["windows"]) == 2,
      f"{len(CALLS['windows'])} window calls")
check("frame windows computed", len(CALLS.get("frame_windows", [])) == 2,
      str(CALLS.get("frame_windows")))
fw = CALLS["frame_windows"]
check("frame indices within the clip",
      all(0 <= i0 < i1 <= 250 for i0, i1, _ in fw), str([(a, b) for a, b, _ in fw]))

# Edits covering most of the clip -> whole video (stitching stops paying off).
install(Probe())
r = eng.run_windowed(VID, AUD, [(0.2, 4.0), (4.5, 9.5)])
check("large coverage falls back to whole video", CALLS["whole"] == 1,
      f"whole={CALLS['whole']}")

# Unprobeable video -> whole video rather than a crash.
install(Probe(fps=0, n=0, w=0, h=0, dur=0))
r = eng.run_windowed(VID, AUD, [(1.0, 2.0)])
check("unprobeable video falls back safely", CALLS["whole"] == 1,
      f"whole={CALLS['whole']}")

# Adjacent edits merge into a single window (one model call, one seam pair).
install(Probe())
eng.run_windowed(VID, AUD, [(3.0, 3.2), (3.4, 3.6)])
check("nearby edits merged into one window", len(CALLS["windows"]) == 1,
      f"{len(CALLS['windows'])} window calls")

# A single failing window must not lose the whole edit.
install(Probe())
_orig_run = le.LipSyncEngine.run


def flaky(self, video_path, audio_path, method="latentsync",
          inference_steps=20, guidance_scale=1.5):
    if str(video_path).endswith("win0_v.mp4"):
        raise RuntimeError("model blew up on window 0")
    if str(video_path).endswith("_v.mp4"):
        CALLS["windows"].append(video_path)
        return video_path + ".synced.mp4"
    CALLS["whole"] += 1
    return "whole.mp4"


le.LipSyncEngine.run = flaky
eng.run_windowed(VID, AUD, [(2.0, 2.4), (7.0, 7.4)])
check("one failed window still produces output", len(CALLS["windows"]) == 1,
      f"{len(CALLS['windows'])} succeeded")
check("failed window recorded as None",
      any(seg is None for _, _, seg in CALLS.get("frame_windows", [])),
      str(CALLS.get("frame_windows")))

# All windows failing must raise rather than silently emit unedited video.
install(Probe())


def all_fail(self, video_path, audio_path, method="latentsync",
             inference_steps=20, guidance_scale=1.5):
    if str(video_path).endswith("_v.mp4"):
        raise RuntimeError("model down")
    CALLS["whole"] += 1
    return "whole.mp4"


le.LipSyncEngine.run = all_fail
try:
    eng.run_windowed(VID, AUD, [(2.0, 2.4)])
    check("all windows failing raises", False)
except RuntimeError as ex:
    check("all windows failing raises RuntimeError", "window" in str(ex).lower(),
          str(ex)[:60])


print()
print("=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL VISION TESTS PASSED")
