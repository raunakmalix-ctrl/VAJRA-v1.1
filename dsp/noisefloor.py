"""
Noise-floor (room-tone) matching.

Real recordings are never silent between words: there is air-conditioning hum,
preamp hiss, distant traffic, room ambience. A text-to-speech model outputs a
mathematically clean floor -- often below -80 dBFS -- because it was trained on
studio material and has no reason to synthesise noise.

The result is a hole. Even when the voice, loudness and tone are all correct,
the noise bed *drops out* for the duration of the edit and returns afterwards.
On headphones that is unmistakable, and it survives compression, so it is one of
the most reliable ways to detect an edit automatically.

Fix: harvest genuine room tone from the non-speech parts of the original
recording and lay it under the generated span at the level the original floor
actually sits at. The noise is real, from the same room and microphone, so it
matches by construction rather than by modelling.
"""
import numpy as np

from dsp.audio import (as_float32, to_mono, n_channels, frame_energy_db,
                       rms_db, apply_gain, EPS)

# Frames quieter than (floor + this) are treated as noise-only.
_NOISE_MARGIN_DB = 6.0
# Percentile of frame energy taken as "the floor".
_FLOOR_PCTL = 10.0
_MIN_TONE_SEC = 0.25


# A span must show at least this much spread between its loud and quiet frames
# for the quiet ones to plausibly BE silence rather than just soft speech.
_MIN_DYNAMIC_RANGE_DB = 20.0


def estimate_noise_floor_db(x, sr, percentile=_FLOOR_PCTL):
    """Level of the quiet parts, in dB.

    A low percentile of per-frame energy rather than the minimum: a single
    anomalously quiet frame (a dropout, a digital-silence splice in the source)
    would otherwise define the floor and make everything else look loud.
    """
    return measure_floor(x, sr, percentile)[0]


def measure_floor(x, sr, percentile=_FLOOR_PCTL):
    """Return (floor_db, reliable).

    Reliability matters and is easy to get wrong. A low percentile of frame
    energy only finds the noise floor if the signal actually CONTAINS silence.
    A short span of continuous speech -- exactly what a replaced word or phrase
    is -- has no silent frames at all, so the 10th percentile measures the
    quietest *speech* and lands far above the true floor.

    Acting on that figure is what makes noise injection silently decline on
    precisely the spans that need it most. So the spread between loud and quiet
    frames is checked, and when there is too little separation the measurement
    is flagged unreliable for the caller to handle explicitly.
    """
    e, _ = frame_energy_db(x, sr)
    if e.size == 0:
        return -90.0, False
    finite = e[np.isfinite(e)]
    if finite.size < 5:
        return (float(np.min(finite)) if finite.size else -90.0), False
    floor = float(np.percentile(finite, percentile))
    loud = float(np.percentile(finite, 95.0))
    return floor, bool((loud - floor) >= _MIN_DYNAMIC_RANGE_DB)


def extract_room_tone(x, sr, exclude_ranges=None, margin_db=_NOISE_MARGIN_DB):
    """Collect noise-only audio from `x`.

    exclude_ranges: iterable of (start_sample, end_sample) to skip -- pass the
    edited spans so a region being replaced can't contribute its own tone.

    Returns (tone_mono, info). tone_mono may be empty when the recording has no
    usable silence, which callers must handle rather than assume.
    """
    a = to_mono(x)
    e, starts = frame_energy_db(a, sr)
    if e.size == 0:
        return np.zeros(0, dtype=np.float32), {"found_sec": 0.0}

    floor = estimate_noise_floor_db(a, sr)
    keep = e <= (floor + margin_db)

    frame_len = max(1, int(sr * 0.020))
    excl = list(exclude_ranges or [])

    segments = []
    for i in np.nonzero(keep)[0]:
        s0 = int(starts[i])
        s1 = s0 + frame_len
        if any(s0 < ex_end and s1 > ex_start for ex_start, ex_end in excl):
            continue
        segments.append(a[s0:s1])

    if not segments:
        return np.zeros(0, dtype=np.float32), {"found_sec": 0.0,
                                              "floor_db": round(floor, 2)}

    tone = np.concatenate(segments).astype(np.float32)
    return tone, {
        "found_sec": round(tone.size / float(sr), 3),
        "floor_db": round(floor, 2),
        "n_frames": int(len(segments)),
    }


def build_bed(tone, n_samples, sr, rng=None):
    """Grow a short room-tone sample into an n_samples bed.

    Naive tiling of the same fragment creates an audible periodic pulse at the
    loop rate. Instead we take randomly-positioned chunks and crossfade between
    them, so the bed is statistically the same noise without a repeating
    pattern.
    """
    t = to_mono(tone)
    n_samples = int(n_samples)
    if t.size == 0 or n_samples <= 0:
        return np.zeros(max(0, n_samples), dtype=np.float32)
    if t.size >= n_samples:
        # Enough material: take a single contiguous slice, no seams at all.
        start = 0 if t.size == n_samples else int(
            (rng or np.random).randint(0, t.size - n_samples + 1))
        return t[start:start + n_samples].copy()

    rng = rng or np.random.RandomState(0)
    chunk = max(int(0.10 * sr), 64)
    xf = max(int(0.02 * sr), 16)
    chunk = min(chunk, t.size)
    xf = min(xf, max(1, chunk // 3))

    out = np.zeros(n_samples + chunk, dtype=np.float32)
    win_in = np.sqrt(np.linspace(0.0, 1.0, xf, dtype=np.float32))
    win_out = win_in[::-1]

    pos = 0
    while pos < n_samples:
        start = int(rng.randint(0, max(1, t.size - chunk + 1)))
        seg = t[start:start + chunk].copy()
        if pos > 0:
            seg[:xf] *= win_in
            out[pos:pos + xf] *= win_out
        out[pos:pos + seg.size] += seg
        pos += max(1, chunk - xf)

    return out[:n_samples]


def inject(span, tone, sr, target_floor_db=None, headroom_db=0.0,
           span_has_own_noise=False):
    """Lay room tone under `span` so its noise floor matches the original.

    target_floor_db: the original's floor from `estimate_noise_floor_db`.
    When None, the tone is added at its own natural level.

    span_has_own_noise: whether `span` already contains real recorded noise.
    Default False, which is the case for anything a synthesiser produced.

    This flag exists because a span's noise floor cannot be reliably measured
    from the span itself. A low percentile of frame energy finds the quietest
    frames, but in a short span those frames are quiet *speech* or a *reverb
    tail* -- real signal, not noise. Treating that as the floor reads far too
    high and makes injection decline on exactly the spans that need it.

    So when the span is synthetic (no additive noise of its own) the bed is laid
    at the target level directly. Only when the caller confirms the span carries
    genuine recorded noise is the existing floor power-subtracted, so the sum
    lands on the target rather than overshooting it.
    """
    a = as_float32(span)
    t = to_mono(tone)
    if t.size == 0:
        return a, {"applied": False, "reason": "no room tone available"}

    bed = build_bed(t, a.shape[0], sr)
    tone_db = rms_db(bed)

    span_floor, reliable = measure_floor(a, sr)
    if target_floor_db is None:
        gain_db = 0.0
        wanted = tone_db
    else:
        wanted = float(target_floor_db) + float(headroom_db)
        if not (span_has_own_noise and reliable):
            # Synthetic span, or an unmeasurable one: lay the bed at the target
            # level. See the docstring -- subtracting an unreliable figure is
            # what makes this decline when it should act.
            gain_db = wanted - tone_db
        else:
            # Power-subtract what the span already has, so span + bed lands on
            # `wanted` rather than overshooting it.
            deficit_pow = 10 ** (wanted / 10.0) - 10 ** (span_floor / 10.0)
            if deficit_pow <= 0:
                return a, {"applied": False,
                           "reason": "span already carries noise at or above "
                                     "the target floor",
                           "span_floor_db": round(span_floor, 2),
                           "target_floor_db": round(wanted, 2)}
            gain_db = 10.0 * np.log10(deficit_pow) - tone_db

    bed = apply_gain(bed, gain_db)
    if a.ndim > 1:
        bed = np.repeat(bed[:, None], a.shape[1], axis=1)

    return (a + bed).astype(np.float32), {
        "applied": True,
        "span_floor_db": round(span_floor, 2),
        "span_floor_reliable": bool(reliable),
        "span_has_own_noise": bool(span_has_own_noise),
        "target_floor_db": round(wanted, 2),
        "bed_gain_db": round(float(gain_db), 2),
        "bed_sec": round(bed.shape[0] / float(sr), 3),
    }
