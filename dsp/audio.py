"""
Shared array helpers for the DSP realism layer.

Everything in dsp/ works on float32 numpy arrays, mono, shaped (n,) or
multi-channel shaped (n, channels) -- the same convention soundfile uses. Keeping
one convention across the package means the match/splice stages compose without
each one re-guessing the layout.

Deliberately depends only on numpy + scipy (+ soundfile for I/O). The principal
runtime pins numpy<2, so only long-stable numpy APIs are used here.
"""
import os

import numpy as np


EPS = 1e-12


def load(path, mono=True):
    """Read an audio file to (float32 array, sample_rate).

    Prefers soundfile; falls back to pydub (which shells out to ffmpeg) for
    containers soundfile can't open, so callers don't have to care whether a
    span arrived as .wav or inside an .mp4.
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        return (to_mono(data) if mono else as_float32(data)), int(sr)
    except Exception:
        pass
    from pydub import AudioSegment
    seg = AudioSegment.from_file(path)
    raw = np.array(seg.get_array_of_samples())
    if seg.channels > 1:
        raw = raw.reshape((-1, seg.channels))
    a = as_float32(raw)
    return (to_mono(a) if mono else a), int(seg.frame_rate)


def save(path, x, sr, subtype="FLOAT"):
    """Write a float32 array to `path`. Peak-guards to avoid wrap-on-write.

    Defaults to 32-bit float rather than 16-bit PCM. Intermediates in this
    pipeline are re-read by later stages, and 16-bit quantisation would apply a
    small error to EVERY sample -- including the regions the edit never touched.
    Since the whole point is that unedited audio remains the genuine recording,
    intermediates stay lossless and quantisation happens once, at final encode.

    Pass subtype="PCM_16" explicitly for a deliverable that needs it.
    """
    import soundfile as sf
    a = as_float32(x)
    p = peak(a)
    if p > 1.0:
        # Scaling the whole track to fix a peak would also alter untouched
        # audio, so this is a last-resort guard; upstream stages peak-limit the
        # span precisely to keep it from triggering.
        print(f"[dsp.save] WARNING peak {p:.3f} > 1.0; scaling entire track to "
              f"prevent clipping. Untouched regions are affected.")
        a = a / np.float32(p)
    os_dir = os.path.dirname(os.path.abspath(path))
    if os_dir:
        os.makedirs(os_dir, exist_ok=True)
    sf.write(path, a, int(sr), subtype=subtype)
    return path


def as_float32(x):
    """Coerce to float32 without copying when it's already right."""
    a = np.asarray(x)
    if a.dtype == np.float32:
        return a
    if np.issubdtype(a.dtype, np.integer):
        # Integer PCM -> [-1, 1). Scale by the type's magnitude, not max, so
        # the mapping stays symmetric and no sample can exceed 1.0.
        info = np.iinfo(a.dtype)
        scale = float(max(abs(info.min), info.max))
        return (a.astype(np.float32) / np.float32(scale))
    return a.astype(np.float32)


def to_mono(x):
    """Average channels down to (n,). Mono input passes through."""
    a = as_float32(x)
    if a.ndim == 1:
        return a
    return a.mean(axis=1).astype(np.float32)


def n_channels(x):
    a = np.asarray(x)
    return 1 if a.ndim == 1 else a.shape[1]


def apply_gain(x, gain_db):
    """Scale by a dB gain, preserving shape and dtype."""
    g = np.float32(10.0 ** (float(gain_db) / 20.0))
    return (as_float32(x) * g).astype(np.float32)


def rms(x):
    a = to_mono(x)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(a.astype(np.float64))) + EPS))


def rms_db(x):
    return 20.0 * np.log10(rms(x) + EPS)


def peak(x):
    a = np.asarray(x)
    return float(np.max(np.abs(a))) if a.size else 0.0


def limit_peak(x, ceiling_db=-1.0):
    """Scale down (never up) so the true peak sits at or below `ceiling_db`.

    Applied after any loudness match: raising a quiet generated span to match a
    louder neighbourhood can push samples past full scale, and clipping is far
    more audible than being a fraction of a dB quiet.
    """
    a = as_float32(x)
    ceiling = 10.0 ** (float(ceiling_db) / 20.0)
    p = peak(a)
    if p <= ceiling or p <= 0.0:
        return a
    return (a * np.float32(ceiling / p)).astype(np.float32)


def fade(x, n_in, n_out, shape="hann"):
    """Apply in-place-safe fades of n_in / n_out samples at the edges."""
    a = as_float32(x).copy()
    n = a.shape[0]
    n_in = int(max(0, min(n_in, n)))
    n_out = int(max(0, min(n_out, n - n_in))) if n_in < n else 0
    if n_in > 0:
        a[:n_in] *= _ramp(n_in, shape, rising=True).reshape(
            (-1,) + (1,) * (a.ndim - 1))
    if n_out > 0:
        a[n - n_out:] *= _ramp(n_out, shape, rising=False).reshape(
            (-1,) + (1,) * (a.ndim - 1))
    return a


def _ramp(n, shape, rising):
    t = np.linspace(0.0, 1.0, int(n), endpoint=True, dtype=np.float32)
    if shape == "linear":
        r = t
    elif shape == "sqrt":            # equal-power for uncorrelated signals
        r = np.sqrt(t)
    else:                            # raised-cosine (Hann) — smooth, no click
        r = np.float32(0.5) * (np.float32(1.0) - np.cos(np.pi * t))
    return r.astype(np.float32) if rising else r[::-1].astype(np.float32)


def resample(x, sr_from, sr_to):
    """Polyphase resample; returns input unchanged when rates already match."""
    if int(sr_from) == int(sr_to):
        return as_float32(x)
    from math import gcd
    from scipy.signal import resample_poly
    a = as_float32(x)
    g = gcd(int(sr_from), int(sr_to))
    up, down = int(sr_to) // g, int(sr_from) // g
    out = resample_poly(a, up, down, axis=0)
    return out.astype(np.float32)


def pad_or_trim(x, n):
    """Force length to exactly n samples (zero-pad or truncate)."""
    a = as_float32(x)
    n = int(n)
    if a.shape[0] == n:
        return a
    if a.shape[0] > n:
        return a[:n]
    pad = [(0, n - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
    return np.pad(a, pad, mode="constant").astype(np.float32)


def frame_energy_db(x, sr, frame_ms=20.0, hop_ms=10.0):
    """Per-frame energy in dB plus each frame's start sample.

    The basis for silence detection, noise-floor estimation and reverb-decay
    measurement, so all three agree on framing.
    """
    a = to_mono(x)
    fl = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if a.shape[0] < fl:
        return np.array([rms_db(a)], dtype=np.float32), np.array([0])
    starts = np.arange(0, a.shape[0] - fl + 1, hop)
    # Strided view instead of a Python loop: this runs per span on long tracks.
    idx = starts[:, None] + np.arange(fl)[None, :]
    frames = a[idx]
    e = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1) + EPS)
    return (20.0 * np.log10(e)).astype(np.float32), starts


def trim_silence(x, sr, floor_offset_db=12.0, keep_ms=30.0, min_keep_frac=0.25):
    """Strip leading/trailing near-silence, keeping a short natural margin.

    Synthesisers commonly pad their output with silence. That padding is not
    speech, but it counts toward the clip's duration -- so a span that would
    comfortably fit its slot can appear far too long, and get needlessly widened
    or escalated to a whole-line re-voice. Measure the speech, not the padding.

    Conservative by construction: the threshold is relative to the clip's own
    quiet level, a `keep_ms` margin is left at each end so nothing is clipped
    off an onset, and the function declines rather than return an implausibly
    small result (`min_keep_frac`) when the clip has no clear silence to trim.
    """
    a = as_float32(x)
    n = a.shape[0]
    if n < int(sr * 0.05):
        return a, {"trimmed": False, "reason": "too short to measure"}

    e, starts = frame_energy_db(to_mono(a), sr, frame_ms=20.0, hop_ms=10.0)
    finite = e[np.isfinite(e)]
    if finite.size < 4:
        return a, {"trimmed": False, "reason": "not enough frames"}

    quiet = float(np.percentile(finite, 5.0))
    loud = float(np.percentile(finite, 95.0))
    if loud - quiet < 6.0:
        # No meaningful contrast: either all speech or all silence. Either way
        # trimming would be guesswork.
        return a, {"trimmed": False, "reason": "no clear silence"}

    thresh = quiet + float(floor_offset_db)
    voiced = np.where(e >= thresh)[0]
    if voiced.size == 0:
        return a, {"trimmed": False, "reason": "nothing above the threshold"}

    keep = int(sr * float(keep_ms) / 1000.0)
    lo = max(0, int(starts[voiced[0]]) - keep)
    hi = min(n, int(starts[voiced[-1]]) + int(sr * 0.02) + keep)
    if hi - lo < int(min_keep_frac * n) or hi <= lo:
        return a, {"trimmed": False, "reason": "would remove too much"}

    if lo == 0 and hi == n:
        return a, {"trimmed": False, "reason": "nothing to trim"}

    return a[lo:hi], {
        "trimmed": True,
        "removed_lead_sec": round(lo / float(sr), 3),
        "removed_tail_sec": round((n - hi) / float(sr), 3),
        "result_sec": round((hi - lo) / float(sr), 3),
    }
