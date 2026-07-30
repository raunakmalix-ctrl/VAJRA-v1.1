"""
Splice construction: where the regenerated span is joined back to real audio.

Two failure modes are avoided here.

1. Waveform discontinuity. Butt-joining two waveforms at an arbitrary sample
   leaves a step, and a step is a broadband impulse -- an audible click. Snapping
   each join to a nearby zero crossing removes the step almost entirely, and a
   short crossfade removes what remains.

2. Level dip or bump through the join. A linear crossfade sums two amplitudes;
   for uncorrelated signals (which real and synthesised speech are) that makes
   total power sag by ~3 dB mid-fade. An equal-power (square-root) crossfade
   keeps summed power constant, so no dip is heard.

Boundary placement matters as much as boundary technique: a join placed inside a
vowel is exposed, whereas one placed in the pause between words is masked by the
silence that is already there. `snap_to_quiet` moves each boundary to the
quietest nearby point before any crossfading happens.
"""
import numpy as np

from dsp.audio import (as_float32, to_mono, frame_energy_db, pad_or_trim, EPS)


def nearest_zero_crossing(x, index, search_samples):
    """Index of the zero crossing nearest `index`, or `index` if none found."""
    a = to_mono(x)
    n = a.shape[0]
    index = int(np.clip(index, 0, max(n - 1, 0)))
    if n < 2:
        return index
    lo = max(1, index - int(search_samples))
    hi = min(n - 1, index + int(search_samples))
    if hi <= lo:
        return index
    seg = a[lo - 1:hi + 1]
    # Sign change between consecutive samples.
    sign = np.signbit(seg)
    changes = np.nonzero(sign[:-1] != sign[1:])[0]
    if changes.size == 0:
        return index
    cand = lo + changes
    return int(cand[np.argmin(np.abs(cand - index))])


def snap_to_quiet(x, index, sr, search_ms=120.0):
    """Move `index` to the quietest frame within +/- search_ms.

    Placing a join in a pause rather than mid-phoneme is the single most
    effective thing that can be done about seam audibility, and it costs nothing.
    """
    a = to_mono(x)
    e, starts = frame_energy_db(a, sr, frame_ms=20.0, hop_ms=5.0)
    if e.size == 0:
        return int(index)
    win = int(sr * search_ms / 1000.0)
    lo, hi = index - win, index + win
    sel = (starts >= lo) & (starts <= hi)
    if not np.any(sel):
        return int(index)
    cand_starts = starts[sel]
    cand_e = e[sel]
    return int(cand_starts[int(np.argmin(cand_e))])


def plan_boundaries(original, start, end, sr, snap_ms=120.0, zc_ms=2.0):
    """Refine a (start, end) span to quiet, zero-crossing-aligned sample points.

    Returns (start, end, info). Kept separate from `splice` so the caller can
    resolve boundaries BEFORE synthesising -- the refined span duration is what
    the generated audio should be asked to fill.
    """
    n = to_mono(original).shape[0]
    s0 = int(np.clip(start, 0, n))
    e0 = int(np.clip(end, 0, n))
    if e0 <= s0:
        return s0, e0, {"snapped": False, "reason": "empty span"}

    zc = max(1, int(sr * zc_ms / 1000.0))
    s1 = snap_to_quiet(original, s0, sr, snap_ms)
    e1 = snap_to_quiet(original, e0, sr, snap_ms)
    if e1 <= s1:                        # snapping collapsed the span
        s1, e1 = s0, e0
    s2 = nearest_zero_crossing(original, s1, zc)
    e2 = nearest_zero_crossing(original, e1, zc)
    if e2 <= s2:
        s2, e2 = s1, e1

    return s2, e2, {
        "snapped": True,
        "start_shift_ms": round((s2 - s0) * 1000.0 / sr, 1),
        "end_shift_ms": round((e2 - e0) * 1000.0 / sr, 1),
    }


def crossfade_splice(original, replacement, start, end, sr,
                     fade_ms=25.0, preserve_length=True):
    """Replace original[start:end] with `replacement`, crossfaded at both joins.

    preserve_length=True (default) keeps the returned track exactly as long as
    the input by padding or trimming `replacement` to the span length. That is
    what keeps audio frame-aligned with video -- the whole reason the timeline
    stage exists. Pass False only when a ripple edit is intended and the video
    will be adjusted to match.

    Returns (audio, info).
    """
    a = as_float32(original)
    r = as_float32(replacement)
    n = a.shape[0]
    start = int(np.clip(start, 0, n))
    end = int(np.clip(end, start, n))
    span = end - start

    if a.ndim > 1 and r.ndim == 1:
        r = np.repeat(r[:, None], a.shape[1], axis=1)
    elif a.ndim == 1 and r.ndim > 1:
        r = r.mean(axis=1).astype(np.float32)

    if preserve_length:
        r = pad_or_trim(r, span)

    # Fade length must fit inside both the span and the material either side.
    fade = int(sr * float(fade_ms) / 1000.0)
    fade = int(max(1, min(fade, span // 2, r.shape[0] // 2,
                          start if start > 0 else fade,
                          (n - end) if end < n else fade)))

    head = a[:start]
    tail = a[end:]

    # Equal-power ramps: sum of squares is constant across the fade.
    t = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    up = np.sqrt(t)
    down = np.sqrt(1.0 - t)
    if a.ndim > 1:
        up = up[:, None]
        down = down[:, None]

    body = r.copy()

    # Leading join: fade replacement in while the original's pre-roll fades out.
    if start > 0 and fade > 0:
        pre = a[start - fade:start]
        body[:fade] = body[:fade] * up + pre * down
    # Trailing join: mirror at the far end.
    if end < n and fade > 0:
        post = a[end:end + fade]
        m = min(fade, body.shape[0])
        body[-m:] = body[-m:] * down[:m] + post[:m] * up[:m]

    out = np.concatenate([head, body, tail], axis=0).astype(np.float32)

    return out, {
        "start": start,
        "end": end,
        "span_sec": round(span / float(sr), 3),
        "replacement_sec": round(r.shape[0] / float(sr), 3),
        "fade_ms": round(fade * 1000.0 / sr, 1),
        "length_preserved": bool(preserve_length),
        "output_sec": round(out.shape[0] / float(sr), 3),
    }


def seam_discontinuity_db(x, index, sr, win_ms=20.0):
    """Level jump across a join, in dB.

    The scorecard compares this against natural word-boundary discontinuities in
    the same recording: an edit whose seam sits inside that natural distribution
    is inaudible, whereas one well outside it is detectable regardless of how
    good the voice clone was.
    """
    a = to_mono(x)
    w = max(1, int(sr * win_ms / 1000.0))
    i = int(np.clip(index, w, max(a.shape[0] - w, w)))
    before = a[i - w:i]
    after = a[i:i + w]
    if before.size == 0 or after.size == 0:
        return 0.0
    rb = np.sqrt(np.mean(before.astype(np.float64) ** 2) + EPS)
    ra = np.sqrt(np.mean(after.astype(np.float64) ** 2) + EPS)
    return float(abs(20.0 * np.log10(ra / rb)))
