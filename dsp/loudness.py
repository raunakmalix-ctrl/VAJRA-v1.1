"""
Loudness matching (ITU-R BS.1770 / EBU R128 where possible).

A generated span that is even 2-3 dB louder or quieter than the words either
side of it reads instantly as an edit, even when the voice itself is a perfect
clone. This is the cheapest and largest single realism win, so it runs first.

Why not just match RMS: RMS weights all frequencies equally, whereas perceived
loudness is frequency-weighted. Two signals at identical RMS can differ audibly
in loudness if their spectra differ -- which is exactly the case between real
and synthesised speech. BS.1770 applies the K-weighting filter that models this.

BS.1770 integrated loudness needs enough signal to be meaningful (the standard
gates on 400 ms blocks). A single replaced word is often shorter than that, so
we fall back to K-weighted RMS, and then to plain RMS, degrading gracefully
rather than returning a wrong number confidently.
"""
import numpy as np

from dsp.audio import (as_float32, to_mono, rms_db, apply_gain, limit_peak,
                       EPS)

# BS.1770 needs >= 400 ms for a single gating block.
_MIN_INTEGRATED_SEC = 0.45
# Refuse absurd corrections: if the measurement says we need >18 dB, the
# measurement is far more likely wrong than the audio being that mismatched.
_MAX_ABS_GAIN_DB = 18.0


def _meter(sr):
    try:
        import pyloudnorm
        return pyloudnorm.Meter(int(sr))
    except Exception:
        return None


def measure_lufs(x, sr):
    """Integrated loudness in LUFS, or None when it can't be measured.

    None is a meaningful answer here -- callers fall back rather than trusting
    a number produced from too little signal.
    """
    a = to_mono(x)
    if a.size < int(_MIN_INTEGRATED_SEC * sr):
        return None
    meter = _meter(sr)
    if meter is None:
        return None
    try:
        val = float(meter.integrated_loudness(a.astype(np.float64)))
    except Exception:
        return None
    # pyloudnorm returns -inf for silence.
    if not np.isfinite(val):
        return None
    return val


def measure_loudness_db(x, sr):
    """Best available loudness figure, and which method produced it.

    Returns (value_db, method) where method is 'lufs' or 'rms'. Callers must
    only ever compare two values obtained by the SAME method -- mixing a LUFS
    reading with an RMS reading would apply a meaningless correction.
    """
    lufs = measure_lufs(x, sr)
    if lufs is not None:
        return lufs, "lufs"
    return rms_db(x), "rms"


def match_loudness(target, reference, sr, ceiling_db=-1.0, max_gain_db=None):
    """Gain `target` so its loudness matches `reference`.

    Both are measured with the same method: LUFS when both are long enough,
    otherwise RMS for both. Result is peak-limited so the correction cannot
    introduce clipping.

    Returns (audio, info) -- info records what was measured and applied so the
    caller can log or surface it in a scorecard.
    """
    a = as_float32(target)
    max_gain = _MAX_ABS_GAIN_DB if max_gain_db is None else float(max_gain_db)

    t_lufs = measure_lufs(a, sr)
    r_lufs = measure_lufs(reference, sr)
    if t_lufs is not None and r_lufs is not None:
        t_val, r_val, method = t_lufs, r_lufs, "lufs"
    else:
        t_val, r_val, method = rms_db(a), rms_db(reference), "rms"

    delta = float(r_val - t_val)
    clamped = float(np.clip(delta, -max_gain, max_gain))
    out = limit_peak(apply_gain(a, clamped), ceiling_db=ceiling_db)

    return out, {
        "method": method,
        "target_db": round(t_val, 2),
        "reference_db": round(r_val, 2),
        "gain_db": round(clamped, 2),
        "gain_clamped": abs(delta - clamped) > 1e-6,
        "requested_gain_db": round(delta, 2),
    }


def loudness_delta_db(a, b, sr):
    """Absolute loudness difference between two spans, same-method.

    Used by the scorecard to score a splice seam: a real word boundary in the
    same recording has some natural delta, and the edit should sit inside that
    distribution rather than at zero.
    """
    a_lufs, b_lufs = measure_lufs(a, sr), measure_lufs(b, sr)
    if a_lufs is not None and b_lufs is not None:
        return abs(a_lufs - b_lufs)
    return abs(rms_db(a) - rms_db(b))
