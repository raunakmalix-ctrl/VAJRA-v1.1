"""
Room / reverberation matching.

A voice recorded in a room carries that room's decay tail. A cloned span may
carry some of it (the cloning reference came from the same recording) but rarely
the same amount, so the perceived room size can shift across the splice.

Honesty about method: recovering a room impulse response blindly from speech is
an unsolved problem in the general case. Deconvolution needs a known excitation,
and we have neither that nor a silent decay we can isolate cleanly. So this
module does NOT claim to estimate the true IR. Instead it:

  1. measures how fast energy decays after speech stops (a proxy for RT60), and
  2. applies a synthetic exponentially-decaying reverb ONLY to close a measured
     deficit, at low wet level.

That is deliberately conservative, and the reason it is: an over-applied or
mis-estimated reverb is far more audible than a small reverb mismatch. Adding
tail to a span that already has it -- likely, since the clone was conditioned on
reverberant reference audio -- produces obvious smearing. This stage therefore
defaults to a low strength and refuses to act unless the original measures
clearly more reverberant than the span.

The measurement is useful on its own: the scorecard reports decay mismatch even
when no correction is applied.
"""
import numpy as np

from dsp.audio import as_float32, to_mono, n_channels, frame_energy_db, EPS

_MIN_DECAY_SEC = 0.08          # ignore tails shorter than this
_MAX_DECAY_SEC = 1.50          # beyond this it's a hall, not a room; clamp
# Only correct when the original's decay exceeds the span's by this factor.
_ACTION_RATIO = 1.35
_DEFAULT_STRENGTH = 0.35       # deliberately timid


def estimate_decay_sec(x, sr, drop_db=25.0):
    """Estimate the energy-decay time after speech offsets, in seconds.

    Finds points where frame energy falls steeply, fits a line to the dB decay
    that follows, and converts the slope to the time it would take to fall
    `drop_db`. Returns None when the recording offers no usable tail (continuous
    speech, or a noise floor high enough to swallow the decay).

    This is an RT-style proxy, not a standards-compliant RT60: it is measured
    from speech offsets rather than from a gated impulse, and it saturates at the
    noise floor. Comparable between two clips from the same recording, which is
    all we need.
    """
    a = to_mono(x)
    e, starts = frame_energy_db(a, sr, frame_ms=20.0, hop_ms=10.0)
    if e.size < 12:
        return None

    finite = e[np.isfinite(e)]
    if finite.size < 12:
        return None
    floor = float(np.percentile(finite, 10.0))
    peak = float(np.percentile(finite, 95.0))
    if peak - floor < 18.0:
        # Not enough dynamic range between speech and silence to see a tail.
        return None

    hop_sec = 0.010
    speech = e > (floor + 0.6 * (peak - floor))
    decays = []

    for i in range(1, e.size - 3):
        if not (speech[i - 1] and not speech[i]):
            continue                      # want the frame where speech stops
        # Follow the decay until it flattens into the floor.
        j = i
        while j < e.size - 1 and e[j] > floor + 3.0 and (j - i) * hop_sec < 1.0:
            j += 1
        n = j - i
        if n < 3:
            continue
        seg = e[i:j]
        t = np.arange(n) * hop_sec
        # Least-squares slope in dB/s (negative).
        slope = float(np.polyfit(t, seg.astype(np.float64), 1)[0])
        if slope >= -1.0:
            continue                      # not actually decaying
        decays.append(drop_db / (-slope))

    if not decays:
        return None
    val = float(np.median(decays))
    return float(np.clip(val, _MIN_DECAY_SEC, _MAX_DECAY_SEC))


def synth_ir(decay_sec, sr, n_taps=None, seed=0):
    """Exponentially-decaying noise burst as a crude room impulse response.

    Noise rather than a comb of discrete taps: discrete early reflections at a
    guessed spacing impose a specific (wrong) geometry and ring tonally, whereas
    decaying noise reads as generic diffuse ambience -- the safer error.
    """
    decay_sec = float(np.clip(decay_sec, _MIN_DECAY_SEC, _MAX_DECAY_SEC))
    n = int(n_taps or min(int(decay_sec * sr), int(1.5 * sr)))
    n = max(n, 64)
    rng = np.random.RandomState(seed)
    noise = rng.randn(n).astype(np.float64)
    t = np.arange(n) / float(sr)
    # -60 dB over decay_sec.
    env = 10.0 ** (-3.0 * t / decay_sec)
    ir = noise * env
    # Unit energy so the wet/dry mix controls level, not the IR length.
    ir /= (np.sqrt(np.sum(ir ** 2)) + EPS)
    return ir


def match_room(target, reference, sr, strength=_DEFAULT_STRENGTH,
               action_ratio=_ACTION_RATIO):
    """Add a little synthetic tail to `target` when `reference` is clearly wetter.

    Returns (audio, info). info always carries both decay measurements so the
    caller can report the mismatch even when nothing was applied.
    """
    from scipy.signal import fftconvolve

    a = as_float32(target)
    d_t = estimate_decay_sec(a, sr)
    d_r = estimate_decay_sec(reference, sr)

    info = {
        "applied": False,
        "span_decay_sec": None if d_t is None else round(d_t, 3),
        "reference_decay_sec": None if d_r is None else round(d_r, 3),
    }

    if d_r is None:
        info["reason"] = "reference has no measurable decay tail"
        return a, info
    if strength <= 0.0:
        info["reason"] = "strength=0"
        return a, info

    # No span measurement -> assume it is dry, but act only at half strength
    # since we are then correcting against an unknown.
    if d_t is None:
        deficit = d_r
        scale = 0.5
    else:
        if d_r < d_t * action_ratio:
            info["reason"] = ("reference not materially wetter than span "
                              "(within action ratio)")
            return a, info
        deficit = float(np.sqrt(max(d_r ** 2 - d_t ** 2, 0.0)))
        scale = 1.0

    if deficit < _MIN_DECAY_SEC:
        info["reason"] = "decay deficit below audible threshold"
        return a, info

    ir = synth_ir(deficit, sr)
    wet_gain = float(np.clip(strength * scale, 0.0, 1.0))

    def _wet(ch):
        w = fftconvolve(ch.astype(np.float64), ir, mode="full")[:ch.size]
        # Normalise wet to the dry channel's level before mixing, so the mix
        # ratio means the same thing regardless of IR length.
        dry_rms = np.sqrt(np.mean(ch.astype(np.float64) ** 2) + EPS)
        wet_rms = np.sqrt(np.mean(w ** 2) + EPS)
        return w * (dry_rms / wet_rms)

    if n_channels(a) == 1:
        wet = _wet(a)
        out = (1.0 - wet_gain) * a.astype(np.float64) + wet_gain * wet
    else:
        cols = []
        for c in range(a.shape[1]):
            wet = _wet(a[:, c])
            cols.append((1.0 - wet_gain) * a[:, c].astype(np.float64)
                        + wet_gain * wet)
        out = np.stack(cols, axis=1)

    info.update({"applied": True,
                 "added_decay_sec": round(deficit, 3),
                 "wet_gain": round(wet_gain, 3)})
    return out.astype(np.float32), info


def decay_mismatch(a, b, sr):
    """Ratio of decay times (>=1), or None when either can't be measured."""
    da, db = estimate_decay_sec(a, sr), estimate_decay_sec(b, sr)
    if da is None or db is None:
        return None
    lo, hi = sorted((da, db))
    return float(hi / max(lo, 1e-6))
