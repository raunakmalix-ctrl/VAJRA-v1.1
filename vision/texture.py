"""
Match a generated patch's texture to the real footage around it.

The video counterpart of dsp/. Audio taught the lesson: a regenerated span is
not convincing because it is correct, it is convincing because it carries the
same incidental properties as its surroundings -- level, tone, room, noise
floor. Video has exactly the same problem and, until now, none of the answer.

Two properties give a generated mouth away, and neither is about lip motion:

  SHARPNESS.  The sync model renders a fixed-size face crop that is scaled back
  into the frame. Where the crop is smaller than the face in the source, the
  result is upscaled -- a soft mouth inside a sharp face. Raising the crop from
  256 to 512 px halves this but does not remove it, because the source can be
  larger still.

  GRAIN.  Diffusion output is denoised by construction. Real footage carries
  sensor noise and compression noise. A clean patch inside a grainy face reads
  as fake even when geometry, timing and colour are perfect -- the eye is very
  good at spotting a region that is *too* clean.

Both are measured against the real pixels immediately around the patch in the
SAME frame, so lighting changes, shot changes and grade changes are handled for
free. That mirrors dsp/, where every measurement is relative to the
neighbourhood rather than to an absolute target.

The same discipline as dsp/ applies throughout:

  * measure both sides identically, or the comparison means nothing;
  * bound every correction, so a bad measurement cannot produce a wild result;
  * decline, and say so, when the measurement is not trustworthy -- grain cannot
    be separated from detail in a textured region, so a reference with no flat
    area yields no estimate rather than a confident wrong one;
  * only ever ADD grain. Removing it would mean denoising real footage, which
    destroys exactly the evidence we are trying to preserve.

numpy only, deliberately: the rest of vision/ carries no heavy dependency and
this module is on the per-frame path.
"""
import numpy as np

# Sharpening is bounded. An upscaled patch can legitimately need a real lift,
# but beyond this the correction stops restoring detail and starts ringing --
# bright haloes at every edge, which is a worse artefact than the softness.
MAX_SHARPEN_GAIN = 2.2
# Below this ratio the patch is already as sharp as its surroundings; acting
# anyway would just add noise to a correct result.
SHARPEN_DEADZONE = 1.04
# A region must be at least this fraction flat before its grain can be
# estimated. Faces are mostly smooth, so this is easily met on skin -- but a
# patch dominated by teeth and lip edges will correctly refuse.
MIN_FLAT_FRAC = 0.08
# Grain is never scaled beyond this multiple of what was measured.
MAX_GRAIN_SIGMA = 12.0
# How much local structure the "flat" pixels may still carry, as a multiple of
# the noise estimate, before the estimate stops being believable. Pure noise
# sits near 1.0; a textured region with no grain sits far above it.
FLAT_STRUCTURE_RATIO = 2.5


def _as_float(img):
    a = np.asarray(img)
    return a.astype(np.float32) if a.dtype != np.float32 else a


def box_blur(img, radius):
    """Mean over a (2r+1) square, via summed-area tables.

    Exact and O(pixels) regardless of radius, which matters because this runs
    per frame. Edges use the true count of contributing pixels rather than a
    padded approximation, so the border is not darkened.
    """
    a = _as_float(img)
    if radius < 1:
        return a.copy()
    single = a.ndim == 2
    if single:
        a = a[:, :, None]
    h, w, c = a.shape
    r = int(min(radius, max(1, min(h, w) // 2)))

    pad = np.pad(a, ((1, 0), (1, 0), (0, 0)), mode="constant")
    integral = pad.cumsum(axis=0).cumsum(axis=1)

    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - r, 0, h)
    y1 = np.clip(ys + r + 1, 0, h)
    x0 = np.clip(xs - r, 0, w)
    x1 = np.clip(xs + r + 1, 0, w)

    total = (integral[np.ix_(y1, x1)] - integral[np.ix_(y0, x1)]
             - integral[np.ix_(y1, x0)] + integral[np.ix_(y0, x0)])
    count = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)
    out = total / count[:, :, None]
    return out[:, :, 0] if single else out


def _luma(img):
    a = _as_float(img)
    if a.ndim == 2:
        return a
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def high_pass(img, radius=2):
    """Detail above the blur scale: the part sharpening and grain both live in."""
    a = _as_float(img)
    return a - box_blur(a, radius)


def sharpness(img, mask=None, radius=2):
    """Mean absolute high-frequency energy, optionally inside `mask`.

    A scalar rather than a spectrum: we only need to know whether one region is
    softer than another, and a ratio of two identically-measured scalars is
    robust in a way a fitted curve would not be at this size.
    """
    hp = np.abs(high_pass(_luma(img), radius))
    if mask is None:
        return float(hp.mean())
    m = np.asarray(mask, dtype=np.float32)
    total = float(m.sum())
    if total <= 0:
        return 0.0
    return float((hp * m).sum() / total)


def grain_sigma(img, mask=None, radius=2, flat_percentile=20.0):
    """Estimate noise amplitude, and say whether the estimate is trustworthy.

    Noise can only be measured where there is no detail to confuse it with, so
    this looks at the FLATTEST part of the region -- the pixels whose local
    variance sits in the bottom `flat_percentile`. In those, high-frequency
    content is overwhelmingly noise rather than structure. The percentile is
    kept tight on purpose: a face is mostly smooth skin, so the flattest fifth
    is genuinely flat, whereas a looser cut starts admitting pores and stubble
    and the estimate drifts upward into detail.

    Returns (sigma, reliable). `reliable` is False when the region has too
    little flat area for the number to mean anything, in which case the caller
    should decline rather than act on it -- the same contract as
    dsp.noisefloor.measure_floor, and for the same reason.
    """
    y = _luma(img)
    m = (np.ones_like(y) if mask is None
         else np.asarray(mask, dtype=np.float32))
    valid = m > 0.5
    n_valid = int(valid.sum())
    if n_valid < 64:
        return 0.0, False

    # Local variance identifies detail; the quietest pixels are the flat ones.
    mean = box_blur(y, radius * 2)
    var = np.maximum(box_blur(y * y, radius * 2) - mean * mean, 0.0)
    thresh = float(np.percentile(var[valid], flat_percentile))
    flat = valid & (var <= thresh)
    if int(flat.sum()) < max(64, int(MIN_FLAT_FRAC * n_valid)):
        return 0.0, False

    hp = high_pass(y, radius)
    # Robust scale: MAD is not dragged around by the few edges that survive the
    # flatness filter, where a plain standard deviation would be.
    vals = hp[flat]
    sigma = float(np.median(np.abs(vals - np.median(vals))) * 1.4826)
    if sigma <= 1e-6:
        return 0.0, False

    # The flatness threshold is a PERCENTILE, so some pixels always qualify --
    # which would make this "reliable" for any region large enough, including
    # one that is pure texture with no noise at all. The estimate is only
    # trustworthy if the pixels it came from are genuinely near-noise-only: if
    # even the flattest of them still carry local structure well above the
    # measured noise, what was measured was detail, not grain.
    flat_std = float(np.sqrt(max(thresh, 0.0)))
    if flat_std > FLAT_STRUCTURE_RATIO * sigma:
        return 0.0, False
    return sigma, True


def match_sharpness(patch, target, mask=None, radius=2,
                    max_gain=MAX_SHARPEN_GAIN, strength=1.0):
    """Raise `patch`'s high-frequency energy toward `target`, within bounds.

    An unsharp mask: add back a scaled copy of the patch's own detail. It can
    only restore structure that survived the model's downscale-and-upscale, so
    it is a mismatch correction, not a resolution recovery -- and bounding the
    gain is what keeps it the former.
    """
    a = _as_float(patch)
    have = sharpness(a, mask, radius)
    info = {"patch": round(have, 3), "target": round(float(target), 3),
            "applied": False}
    if have <= 1e-6 or target <= 0:
        info["reason"] = "no measurable detail to work with"
        return a, info
    ratio = float(target) / have
    if ratio <= SHARPEN_DEADZONE:
        info["reason"] = "already as sharp as its surroundings"
        return a, info

    gain = min(ratio, max_gain) - 1.0
    gain *= float(np.clip(strength, 0.0, 1.0))
    detail = a - box_blur(a, radius)
    out = a + gain * detail
    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        if out.ndim == 3 and m.ndim == 2:
            m = m[:, :, None]
        out = a + (out - a) * m
    info.update({"applied": True, "gain": round(gain, 3),
                 "ratio": round(ratio, 3),
                 "capped": ratio > max_gain})
    return np.clip(out, 0.0, 255.0), info


def apply_grain(patch, sigma, mask=None, seed=None, correlation=0.5):
    """Lay noise of amplitude `sigma` over `patch`.

    Real footage noise is not white: demosaicing and compression correlate it
    over a pixel or two, so pure per-pixel gaussian looks like digital fizz
    rather than grain. `correlation` mixes in a slightly blurred copy to give it
    a plausible size, and the result is renormalised so the requested sigma is
    what actually lands.

    Applied to all channels jointly rather than independently: sensor grain is
    largely luminance-driven, and per-channel noise reads as colour speckle.
    """
    a = _as_float(patch)
    if sigma <= 0:
        return a
    rng = np.random.RandomState(seed) if seed is not None else np.random
    h, w = a.shape[:2]
    n = rng.randn(h, w).astype(np.float32)
    if correlation > 0:
        n = (1.0 - correlation) * n + correlation * box_blur(n, 1)
        s = float(n.std())
        if s > 1e-8:
            n /= s
    n *= float(sigma)
    if a.ndim == 3:
        n = n[:, :, None]
    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        if a.ndim == 3 and m.ndim == 2:
            m = m[:, :, None]
        n = n * m
    return np.clip(a + n, 0.0, 255.0)


def match_texture(frame, base_frame, mask, radius=2, strength=1.0, seed=None,
                  ring=6):
    """Make the generated region of `frame` sit in `base_frame`'s texture.

    frame       the synced frame, carrying the generated region
    base_frame  the original frame, i.e. ground truth for everything outside
    mask        where the generated content is (1 inside, 0 outside, feathered)

    The reference is a ring of REAL pixels just outside the mask in the same
    frame -- close enough to share lighting and grade, outside the region being
    corrected. That is the video equivalent of dsp's neighbourhood.

    Returns (frame, report). The report names what was done and, where nothing
    was, why -- a silent no-op and a silent correction look identical in the
    output, and only one of them is a bug.
    """
    a = _as_float(frame)
    base = _as_float(base_frame)
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]

    inside = m > 0.5
    report = {"sharpness": None, "grain": None}
    if int(inside.sum()) < 64:
        report["reason"] = "generated region too small to measure"
        return a, report

    # Ring of real pixels hugging the region, from the ORIGINAL frame.
    grown = box_blur(inside.astype(np.float32), ring) > 1e-4
    ref = grown & (~(m > 1e-3))
    if int(ref.sum()) < 64:
        report["reason"] = "no clean surrounding pixels to compare against"
        return a, report

    ref_mask = ref.astype(np.float32)
    in_mask = inside.astype(np.float32)

    target_sharp = sharpness(base, ref_mask, radius)
    a, s_info = match_sharpness(a, target_sharp, m, radius, strength=strength)
    report["sharpness"] = s_info

    ref_sigma, ok = grain_sigma(base, ref_mask, radius)
    have_sigma, have_ok = grain_sigma(a, in_mask, radius)
    g = {"reference": round(ref_sigma, 3), "measurable": bool(ok),
         "patch": round(have_sigma, 3) if have_ok else None, "applied": False}
    if not ok:
        g["reason"] = "surrounding pixels have no flat area; grain unmeasurable"
    else:
        # Only ever ADD. If the patch is already grainier than its surroundings
        # the honest response is to leave it alone: removing grain means
        # denoising, which would damage the real pixels we are trying to match.
        deficit = ref_sigma - (have_sigma if have_ok else 0.0)
        if deficit <= 0.15:
            g["reason"] = "already as grainy as its surroundings"
        else:
            add = float(min(deficit, MAX_GRAIN_SIGMA)) * float(
                np.clip(strength, 0.0, 1.0))
            a = apply_grain(a, add, m, seed=seed)
            g.update({"applied": True, "added_sigma": round(add, 3),
                      "capped": deficit > MAX_GRAIN_SIGMA})
    report["grain"] = g
    return a, report


def texture_distance(frame, base_frame, mask, radius=2, ring=6):
    """How far the generated region still is from its surroundings.

    Pure measurement, for the scorecard: a sharpness ratio (1.0 is a match) and
    a grain difference in code values. Runs the same estimators the matcher
    uses, so a report and a correction can never disagree about what was seen.
    """
    base = _as_float(base_frame)
    a = _as_float(frame)
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    inside = m > 0.5
    grown = box_blur(inside.astype(np.float32), ring) > 1e-4
    ref = (grown & (~(m > 1e-3))).astype(np.float32)
    if int(inside.sum()) < 64 or int(ref.sum()) < 64:
        return {"measurable": False,
                "reason": "region or surround too small to measure"}

    s_patch = sharpness(a, inside.astype(np.float32), radius)
    s_ref = sharpness(base, ref, radius)
    g_patch, gp_ok = grain_sigma(a, inside.astype(np.float32), radius)
    g_ref, gr_ok = grain_sigma(base, ref, radius)
    out = {
        "measurable": True,
        "sharpness_ratio": round(s_patch / s_ref, 3) if s_ref > 1e-6 else None,
        "grain_delta": (round(g_patch - g_ref, 3)
                        if (gp_ok and gr_ok) else None),
    }
    r = out["sharpness_ratio"]
    out["verdict"] = ("unmeasured" if r is None else
                      "good" if 0.85 <= r <= 1.18 else
                      "acceptable" if 0.7 <= r <= 1.4 else "mismatched")
    return out
