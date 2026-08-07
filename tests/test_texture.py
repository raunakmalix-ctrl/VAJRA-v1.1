# -*- coding: utf-8 -*-
"""Grain and sharpness matching: does a generated patch end up like its surround?

Built as a controlled experiment rather than a smoke test. Synthetic "footage"
is given a known grain sigma and a known amount of detail; a patch is then
damaged in exactly the two ways a sync model damages one -- blurred (because it
rendered a smaller crop that got upscaled) and denoised (because diffusion
output is clean). The matcher then has to measurably close both gaps.

That structure matters: it is easy to write a filter that changes pixels and
much harder to write one that moves a measured quantity toward a target without
overshooting, and only the second is worth having.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision import texture as T  # noqa: E402
import vision  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def face(h=240, w=240, grain=4.0, seed=0, detail=1.0):
    """Skin-like footage: mostly smooth, some structure, known grain.

    Deliberately mostly smooth, because a face is: the structure is confined to
    a band rather than covering every pixel, so "the flattest fifth" is genuinely
    flat the way real skin is. An image textured edge to edge would make grain
    unmeasurable everywhere — correctly, but it would not be testing footage.
    """
    r = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 120 + 40 * (yy / h) + 25 * np.sin(xx / 19.0)
    band = ((yy > h * 0.55) & (yy < h * 0.78)).astype(np.float32)
    base += detail * 16 * band * np.sin(xx / 3.4) * np.sin(yy / 3.9)
    img = np.repeat(base[:, :, None], 3, axis=2)
    img = img + r.randn(h, w, 1) * grain
    return np.clip(img, 0, 255).astype(np.float32)


def blur(img, r=2):
    return T.box_blur(img, r)


def upscaled(img, factor=2):
    """What a sync model actually does: render smaller, then scale back up.

    Average-pool down and pixel-replicate up — the real damage being corrected,
    rather than an arbitrary blur.
    """
    a = T._as_float(img)
    h, w = a.shape[:2]
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    a = a[:h2, :w2]
    small = a.reshape(h2 // factor, factor, w2 // factor, factor, -1).mean(
        axis=(1, 3))
    up = np.repeat(np.repeat(small, factor, axis=0), factor, axis=1)
    out = T._as_float(img).copy()
    out[:h2, :w2] = up
    return out


print("=" * 74)
print("box blur — the primitive everything else rests on")
print("=" * 74)
flat = np.full((40, 40), 100.0, dtype=np.float32)
b = T.box_blur(flat, 3)
check("a flat field stays flat, including at the edges",
      float(np.abs(b - 100.0).max()) < 1e-3, f"max dev {np.abs(b-100).max():.5f}")
check("shape and dtype are preserved",
      b.shape == flat.shape and b.dtype == np.float32)
colour = face(60, 60, grain=0.0, seed=1)
check("colour images keep their channels", T.box_blur(colour, 2).shape == colour.shape)
check("radius 0 is a no-op", np.allclose(T.box_blur(colour, 0), colour))
noisy = np.random.RandomState(2).randn(80, 80).astype(np.float32) * 10 + 50
check("blurring reduces variance", float(T.box_blur(noisy, 3).std()) < float(noisy.std()))

print()
print("=" * 74)
print("measurement — do the estimators recover what was put in?")
print("=" * 74)
for truth in (2.0, 5.0, 9.0):
    img = face(220, 220, grain=truth, seed=int(truth))
    est, ok = T.grain_sigma(img)
    check(f"grain sigma {truth} recovered as {est:.2f}",
          ok and abs(est - truth) < max(1.0, truth * 0.35),
          f"estimated {est:.2f}")

clean = face(200, 200, grain=0.0, seed=7)
sharp_full = T.sharpness(clean)
sharp_blur = T.sharpness(blur(clean, 2))
check("blurring lowers measured sharpness", sharp_blur < sharp_full,
      f"{sharp_blur:.2f} < {sharp_full:.2f}")

# The contract from dsp.noisefloor.measure_floor: decline when the number would
# not mean anything. Heavy texture with no noise is the case that must decline —
# its high frequencies are detail. Pure noise, by contrast, genuinely HAS a
# measurable grain level and must not be refused.
yy, xx = np.mgrid[0:260, 0:260]
textured = np.repeat(((((xx // 3 + yy // 3) % 2) * 200 + 30)
                      ).astype(np.float32)[:, :, None], 3, axis=2)
_, tex_ok = T.grain_sigma(textured)
check("heavy texture with no noise refuses to report grain", not tex_ok)

pure_noise = np.clip(128 + np.random.RandomState(3).randn(200, 200) * 5,
                     0, 255).astype(np.float32)
_, noise_ok = T.grain_sigma(pure_noise)
check("pure noise is not refused — it really does have grain", noise_ok)

smooth_clean = face(220, 220, grain=0.0, seed=4)
_, clean_ok = T.grain_sigma(smooth_clean)
check("a noiseless region reports nothing to match", not clean_ok)

tiny = face(6, 6, grain=3.0)
_, tiny_ok = T.grain_sigma(tiny)
check("a region too small to measure refuses as well", not tiny_ok)

print()
print("=" * 74)
print("sharpness matching closes the gap without overshooting")
print("=" * 74)
target = T.sharpness(clean)
soft = blur(clean, 2)
before = T.sharpness(soft)
fixed, info = T.match_sharpness(soft, target)
after = T.sharpness(fixed)
check("it acted", info["applied"], str(info.get("reason", "")))
check("the gap narrowed", abs(after - target) < abs(before - target),
      f"{before:.2f} -> {after:.2f} (target {target:.2f})")
check("it did not overshoot past the target",
      after <= target * 1.35, f"{after:.2f} vs target {target:.2f}")

same, info2 = T.match_sharpness(clean, target)
check("a patch already matching its target is left alone",
      not info2["applied"] and np.allclose(same, clean),
      str(info2.get("reason", "")))
very_soft = blur(clean, 8)
_, info3 = T.match_sharpness(very_soft, target * 40)
check("an impossible target is capped rather than pursued",
      info3.get("capped") is True and info3["gain"] <= T.MAX_SHARPEN_GAIN,
      str(info3))

print()
print("=" * 74)
print("grain is added, never removed")
print("=" * 74)
g = T.apply_grain(np.full((120, 120, 3), 128.0, np.float32), 5.0, seed=1)
est, ok = T.grain_sigma(g)
check("requested grain amplitude is what lands", ok and abs(est - 5.0) < 1.6,
      f"{est:.2f}")
check("grain differs between seeds",
      not np.allclose(T.apply_grain(np.full((60, 60), 128.0, np.float32), 4.0, seed=1),
                      T.apply_grain(np.full((60, 60), 128.0, np.float32), 4.0, seed=2)))
check("zero sigma is a no-op",
      np.allclose(T.apply_grain(np.full((30, 30), 90.0, np.float32), 0.0),
                  np.full((30, 30), 90.0, np.float32)))

print()
print("=" * 74)
print("match_texture — the whole job, on a damaged patch")
print("=" * 74)
H = W = 260
base_frame = face(H, W, grain=5.0, seed=11)
mask = np.zeros((H, W), np.float32)
mask[95:165, 95:165] = 1.0

# A sync model's output: blurred (upscaled from a smaller crop) and denoised.
damaged = base_frame.copy()
patch = blur(base_frame[80:180, 80:180], 2)
patch = T.box_blur(patch, 1)
damaged[80:180, 80:180] = patch
inside = mask > 0.5
d_before = vision.texture_distance(damaged, base_frame, mask)
fixed, rep = vision.match_texture(damaged, base_frame, mask, seed=5)
d_after = vision.texture_distance(fixed, base_frame, mask)

check("the damage was detectable to begin with",
      d_before["measurable"] and d_before["sharpness_ratio"] < 0.85,
      str(d_before))
check("sharpness was corrected", rep["sharpness"]["applied"], str(rep["sharpness"]))
check("grain was added", rep["grain"]["applied"], str(rep["grain"]))
check("the patch ends closer to its surroundings",
      abs(d_after["sharpness_ratio"] - 1.0) < abs(d_before["sharpness_ratio"] - 1.0),
      f"ratio {d_before['sharpness_ratio']} -> {d_after['sharpness_ratio']}")
# Severe damage cannot be fully undone in one bounded pass, and pretending
# otherwise would mean lifting the cap and ringing the edges. What matters is
# that it improved and SAID it hit the limit.
check("hitting the correction limit is reported, not hidden",
      rep["sharpness"].get("capped") is True, str(rep["sharpness"]))
mild = base_frame.copy()
mild[80:180, 80:180] = upscaled(base_frame[80:180, 80:180], 2)
before_mild = vision.texture_distance(mild, base_frame, mask)["sharpness_ratio"]
mild_fixed, _ = vision.match_texture(mild, base_frame, mask, seed=5)
after_mild = vision.texture_distance(mild_fixed, base_frame, mask)
check("realistic damage is brought back to acceptable",
      after_mild["verdict"] in ("good", "acceptable"),
      f"{before_mild} -> {after_mild['sharpness_ratio']} ({after_mild['verdict']})")
check("pixels outside the mask are untouched",
      np.allclose(fixed[~inside], damaged[~inside]),
      "the correction leaked outside the generated region")

# An undamaged patch must come out essentially unchanged: a matcher that always
# "improves" something is a matcher that damages correct input.
clean_frame = face(H, W, grain=5.0, seed=12)
untouched, rep2 = vision.match_texture(clean_frame.copy(), clean_frame, mask,
                                       seed=5)
check("a patch that already matches is barely altered",
      float(np.abs(untouched - clean_frame).mean()) < 1.2,
      f"mean delta {np.abs(untouched-clean_frame).mean():.3f}")

print()
print("=" * 74)
print("declining, and saying so")
print("=" * 74)
_, rep3 = vision.match_texture(textured.copy(), textured, mask)
check("grain is declined when the surround has no flat area",
      not rep3["grain"]["applied"] and "reason" in rep3["grain"],
      str(rep3["grain"]))
tiny_mask = np.zeros((H, W), np.float32)
tiny_mask[10:14, 10:14] = 1.0
_, rep4 = vision.match_texture(base_frame.copy(), base_frame, tiny_mask)
check("a region too small to measure is declined with a reason",
      "reason" in rep4, str(rep4)[:90])
d_bad = vision.texture_distance(base_frame, base_frame, tiny_mask)
check("the measurement declines on the same input",
      d_bad["measurable"] is False, str(d_bad))

print()
print("=" * 74)
print("composite_window integration")
print("=" * 74)
base_seq = [face(160, 160, grain=5.0, seed=20 + i) for i in range(6)]
sync_seq = [T.box_blur(f, 2) for f in base_seq]
m = np.zeros((160, 160), np.float32)
m[60:110, 60:110] = 1.0
rep5 = {}
out = vision.composite_window(base_seq, sync_seq, m, report=rep5)
check("every frame comes back", len(out) == 6)
check("texture matching ran and was reported", "texture" in rep5, str(rep5))
check("it reports how many frames it acted on",
      rep5["texture"]["sharpened"] > 0, str(rep5["texture"]))
check("no frame errored", rep5["texture"]["errors"] == 0, str(rep5["texture"]))

off = vision.composite_window(base_seq, sync_seq, m, texture_match=False)
check("it can be switched off for an A/B",
      not np.allclose(out[3], off[3]))

print("=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL TEXTURE TESTS PASSED")
