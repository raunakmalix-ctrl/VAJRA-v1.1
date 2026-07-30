"""
Spectral-envelope ("tone") matching.

After loudness, the next thing that gives an edit away is timbre. The original
was recorded through a specific microphone, preamp and room, all of which impose
a broad frequency tilt -- a lavalier sounds different from a shotgun mic, a
phone capsule rolls off below 200 Hz. A synthesised span carries the TTS
model's own spectral signature instead, so the tone shifts mid-sentence even
when the voice identity is right.

Method: measure the long-term average spectrum (LTAS) of the generated span and
of the surrounding real speech, take the ratio as a correction curve, smooth it
heavily in log-frequency, and apply it as a linear-phase FIR.

Why smooth so heavily: an unsmoothed ratio of two short-sample spectra is mostly
estimation noise, and applying it would imprint the reference's *phonetic*
content onto the span rather than its channel character. Smoothing into
fractional-octave bands keeps the broad channel tilt and discards the detail
that belongs to the specific words. Correction is applied with zero phase shift
so the span's timing -- which the splice depends on -- is unchanged.
"""
import numpy as np

from dsp.audio import as_float32, to_mono, n_channels

# Below this, LTAS estimation is too noisy to be worth trusting.
_MIN_SEC = 0.20
# Broad channel character only. Hard cap so a bad estimate can't wreck a span.
_MAX_CORRECTION_DB = 10.0
# Ignore the extremes: near-DC and the last few kHz carry little speech energy
# and produce wild ratios.
_LO_HZ = 60.0
_HI_FRAC = 0.92          # fraction of Nyquist


def _ltas(x, sr, n_fft=1024):
    """Long-term average power spectrum via Welch."""
    from scipy.signal import welch
    a = to_mono(x).astype(np.float64)
    nper = int(min(n_fft, max(256, a.size)))
    f, p = welch(a, fs=sr, nperseg=nper, noverlap=nper // 2,
                 scaling="density", average="median")
    return f, p


def _log_band_edges(lo, hi, per_octave=3):
    """Fractional-octave band edges between lo and hi Hz."""
    lo = max(lo, 1.0)
    n_oct = np.log2(hi / lo)
    n_bands = max(4, int(np.ceil(n_oct * per_octave)))
    return lo * (2.0 ** (np.linspace(0.0, n_oct, n_bands + 1)))


def correction_curve_db(target, reference, sr, per_octave=3,
                        max_correction_db=_MAX_CORRECTION_DB):
    """Smoothed reference-minus-target spectral difference, in dB per frequency.

    Returns (freqs_hz, gain_db) sampled on band centres. Exposed separately from
    `match_spectrum` so the scorecard can report how much correction a span
    needed without re-deriving it.
    """
    ft, pt = _ltas(target, sr)
    fr, pr = _ltas(reference, sr)
    # Welch grids match when nperseg matches; interpolate defensively.
    if ft.shape != fr.shape or not np.allclose(ft, fr):
        pr = np.interp(ft, fr, pr)
    f = ft

    hi = _HI_FRAC * (sr / 2.0)
    edges = _log_band_edges(_LO_HZ, hi, per_octave)
    centres, gains = [], []
    for i in range(len(edges) - 1):
        sel = (f >= edges[i]) & (f < edges[i + 1])
        if not np.any(sel):
            continue
        # Mean power in the band, converted to dB once (not mean of dB, which
        # would be biased by near-zero bins).
        pt_b = float(np.mean(pt[sel])) + 1e-20
        pr_b = float(np.mean(pr[sel])) + 1e-20
        centres.append(np.sqrt(edges[i] * edges[i + 1]))
        gains.append(10.0 * np.log10(pr_b / pt_b))

    if not centres:
        return np.array([0.0, sr / 2.0]), np.array([0.0, 0.0])

    centres = np.asarray(centres, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)

    # Remove the broadband offset: overall level is loudness_match's job, and
    # leaving it in here would double-correct it.
    gains -= float(np.median(gains))

    # A light 3-point smooth across bands, then clamp.
    if gains.size >= 3:
        k = np.array([0.25, 0.5, 0.25])
        gains = np.convolve(np.pad(gains, 1, mode="edge"), k, mode="valid")
    gains = np.clip(gains, -max_correction_db, max_correction_db)

    # Anchor the ends so the FIR design has a defined response at DC/Nyquist.
    f_out = np.concatenate(([0.0], centres, [sr / 2.0]))
    g_out = np.concatenate(([gains[0]], gains, [gains[-1]]))
    return f_out, g_out


def _design_fir(f_hz, gain_db, sr, n_taps=257):
    """Linear-phase FIR whose magnitude follows gain_db(f_hz)."""
    if n_taps % 2 == 0:
        n_taps += 1
    n_fft = 1 << int(np.ceil(np.log2(max(n_taps * 4, 1024))))
    grid = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    mag = 10.0 ** (np.interp(grid, f_hz, gain_db) / 20.0)

    # Zero-phase impulse response: real, even spectrum -> real, symmetric IR.
    full = np.concatenate([mag, mag[-2:0:-1]])
    ir = np.real(np.fft.ifft(full))
    ir = np.concatenate([ir[-(n_taps // 2):], ir[:n_taps // 2 + 1]])
    ir *= np.hanning(ir.size)
    return ir.astype(np.float64)


def match_spectrum(target, reference, sr, strength=1.0,
                   max_correction_db=_MAX_CORRECTION_DB, n_taps=257):
    """Filter `target` so its broad spectral shape matches `reference`.

    `strength` scales the correction (0 disables, 1 applies it fully); useful for
    short spans where the estimate is less reliable.

    Returns (audio, info).
    """
    from scipy.signal import fftconvolve

    a = as_float32(target)
    dur = a.shape[0] / float(sr)
    if dur < _MIN_SEC or strength <= 0.0:
        return a, {"applied": False,
                   "reason": "span too short for reliable LTAS"
                             if dur < _MIN_SEC else "strength=0"}

    # Taper the correction on short spans instead of switching it off abruptly.
    auto = float(np.clip((dur - _MIN_SEC) / 0.6, 0.35, 1.0))
    s = float(strength) * auto

    f, g = correction_curve_db(a, reference, sr,
                               max_correction_db=max_correction_db)
    g = g * s
    ir = _design_fir(f, g, sr, n_taps=n_taps)

    if n_channels(a) == 1:
        out = fftconvolve(a.astype(np.float64), ir, mode="same")
    else:
        out = np.stack([fftconvolve(a[:, c].astype(np.float64), ir, mode="same")
                        for c in range(a.shape[1])], axis=1)

    return out.astype(np.float32), {
        "applied": True,
        "strength": round(s, 3),
        "max_band_gain_db": round(float(np.max(np.abs(g))), 2),
        "n_taps": int(ir.size),
    }


def spectral_distance_db(a, b, sr, per_octave=3):
    """RMS band-wise spectral difference -- a scalar 'how different is the tone'.

    Used by the scorecard, and to compare an edit seam against natural seams
    elsewhere in the same recording.
    """
    _, g = correction_curve_db(a, b, sr, per_octave=per_octave,
                               max_correction_db=60.0)
    return float(np.sqrt(np.mean(np.square(g))))
