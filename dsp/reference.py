"""
Cloning-reference selection.

Zero-shot voice cloning is only as good as the reference it is conditioned on,
and reference quality dominates output quality more than any inference setting
does. Feeding a model the first N seconds of a recording -- the previous
behaviour, which handed it the entire original track -- is close to the worst
available choice: openings routinely contain music stings, room noise before the
speaker settles, a second voice, or the clipped first syllable.

This module scores every candidate window and picks the best one. The criteria
are what actually degrades a clone:

  speech density  A window that is half silence gives the encoder little to work
                  with. Measured as the fraction of frames above the noise floor.
  signal-to-noise Speech level minus noise floor. Noise in the reference gets
                  modelled as part of the voice and reproduced in every output.
  clipping        Clipped samples are unrecoverable distortion, and the model
                  will faithfully learn the distortion.
  level sanity    Very quiet references quantise poorly and push the encoder
                  into a badly-conditioned range.
  continuity      A window spanning a long gap is really two fragments; prefer
                  continuous speech.

Windows overlapping regions about to be replaced are excluded: conditioning the
synthesiser on the audio it is replacing biases it toward reproducing what was
already there.
"""
import numpy as np

from dsp.audio import (as_float32, to_mono, frame_energy_db, peak, rms_db,
                       fade, EPS)

DEFAULT_TARGET_SEC = 18.0
DEFAULT_MIN_SEC = 6.0
_SPEECH_MARGIN_DB = 10.0        # frames this far above the floor count as speech
_CLIP_THRESHOLD = 0.985


def _floor_and_speech(e):
    finite = e[np.isfinite(e)]
    if finite.size == 0:
        return -90.0, -90.0
    return (float(np.percentile(finite, 10.0)),
            float(np.percentile(finite, 90.0)))


def tonality(x, sr, max_frames=48):
    """How harmonic the loud parts are, 0 (noise-like) to 1 (strongly voiced).

    Energy-based scores alone cannot tell loud speech from loud noise, and that
    distinction is exactly what matters when a candidate window overlaps a music
    sting or a burst of room noise: the contamination RAISES the measured speech
    level and so flatters the SNR estimate, making a bad window look good.

    Spectral flatness separates them cleanly. Broadband noise spreads energy
    evenly across frequency, so its geometric and arithmetic spectral means are
    close (flatness near 1). Voiced speech concentrates energy in harmonics, so
    the geometric mean sits far below the arithmetic one (flatness near 0). This
    returns 1 - flatness, measured only on the high-energy frames -- silence is
    noise-like by definition and would otherwise dominate.
    """
    a = to_mono(x)
    n_fft = 512
    if a.size < n_fft * 2:
        return 0.0
    hop = n_fft // 2
    starts = np.arange(0, a.size - n_fft + 1, hop)
    if starts.size == 0:
        return 0.0
    frames = a[starts[:, None] + np.arange(n_fft)[None, :]]
    energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + EPS)
    # Only the loudest half: quiet frames are room tone and tell us nothing here.
    thresh = np.percentile(energy, 60.0)
    sel = np.nonzero(energy >= thresh)[0]
    if sel.size == 0:
        return 0.0
    if sel.size > max_frames:
        sel = sel[np.linspace(0, sel.size - 1, max_frames).astype(int)]
    win = np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames[sel] * win, axis=1)) ** 2 + 1e-12
    # Ignore the lowest bins: DC and rumble skew the geometric mean.
    spec = spec[:, 3:]
    geo = np.exp(np.mean(np.log(spec), axis=1))
    ari = np.mean(spec, axis=1)
    flat = np.clip(geo / (ari + 1e-20), 0.0, 1.0)
    return float(np.clip(1.0 - np.mean(flat), 0.0, 1.0))


def score_window(x, sr, e=None, starts=None, offset=0, length=None):
    """Score one window of audio for suitability as a cloning reference.

    Returns (score, detail). Score is roughly 0-1; higher is better.
    """
    a = to_mono(x)
    if a.size < int(0.5 * sr):
        return 0.0, {"reason": "too short"}

    if e is None:
        e, starts = frame_energy_db(a, sr)
    floor, speech = _floor_and_speech(e)

    finite = e[np.isfinite(e)]
    if finite.size == 0:
        return 0.0, {"reason": "no measurable energy"}

    density = float(np.mean(finite > (floor + _SPEECH_MARGIN_DB)))
    snr = float(speech - floor)
    pk = peak(a)
    clipped = float(np.mean(np.abs(a) >= _CLIP_THRESHOLD))
    level = rms_db(a)

    # Noise-floor stationarity. Without this, a window straddling the boundary
    # between a noisy passage and clean speech scores WELL: the noisy part raises
    # the 90th-percentile "speech" figure while the clean part supplies a low
    # 10th-percentile "floor", so the SNR estimate is flattered by the very
    # contamination we are trying to avoid. Measuring the floor in sub-blocks
    # exposes that -- a window of consistent material has a consistent floor.
    blocks = np.array_split(finite, 4) if finite.size >= 8 else [finite]
    floors = [float(np.percentile(b, 10.0)) for b in blocks if b.size]
    floor_spread = float(max(floors) - min(floors)) if len(floors) > 1 else 0.0

    # Sub-scores, each 0-1.
    s_density = float(np.clip((density - 0.25) / 0.45, 0.0, 1.0))
    s_snr = float(np.clip((snr - 12.0) / 25.0, 0.0, 1.0))
    s_clip = 1.0 if clipped <= 1e-5 else float(np.clip(1.0 - clipped * 200.0, 0.0, 1.0))
    # Level is a PLATEAU, not a peak: anything in a broad healthy range scores
    # full marks. A sharp optimum would make level the tiebreaker between two
    # otherwise-equal windows -- and since contamination (a music sting, a noise
    # burst) raises a window's RMS, a sharp optimum actively rewards the
    # contaminated window. Only genuinely too-quiet or too-hot audio is penalised.
    if -32.0 <= level <= -11.0:
        s_level = 1.0
    else:
        over = (level + 32.0) if level < -32.0 else (-11.0 - level)
        s_level = float(np.clip(1.0 + over / 14.0, 0.0, 1.0))
    # A spread beyond ~12 dB means the window is not one consistent recording.
    s_stat = float(np.clip(1.0 - (floor_spread - 4.0) / 14.0, 0.0, 1.0))

    # Harmonic content of the loud frames: the one measure that distinguishes
    # loud speech from loud noise, and therefore the one that stops a window
    # overlapping a noise burst from scoring well.
    tone = tonality(a, sr)
    # Mapped gently so it does NOT saturate: tonality is the measure that
    # separates clean speech from speech-plus-noise, and a saturating curve
    # discards exactly the small differences it exists to detect.
    s_tone = float(np.clip((tone - 0.20) / 0.80, 0.0, 1.0))

    score = (0.24 * s_density + 0.20 * s_snr + 0.14 * s_clip
             + 0.06 * s_level + 0.12 * s_stat + 0.24 * s_tone)

    return float(score), {
        "score": round(score, 4),
        "speech_density": round(density, 3),
        "snr_db": round(snr, 2),
        "clipped_fraction": round(clipped, 6),
        "rms_db": round(level, 2),
        "peak": round(pk, 4),
        "floor_spread_db": round(floor_spread, 2),
        "tonality": round(tone, 3),
        "sub_scores": {"density": round(s_density, 3), "snr": round(s_snr, 3),
                       "clip": round(s_clip, 3), "level": round(s_level, 3),
                       "stationarity": round(s_stat, 3),
                       "tonality": round(s_tone, 3)},
    }


def _overlaps(a0, a1, ranges):
    return any(a0 < r1 and a1 > r0 for r0, r1 in ranges)


def pick_reference(track, sr, target_sec=DEFAULT_TARGET_SEC,
                   min_sec=DEFAULT_MIN_SEC, exclude_ranges=None,
                   hop_sec=0.5, speech_ranges=None):
    """Choose the best window of `track` to use as a cloning reference.

    exclude_ranges: [(start_sample, end_sample), ...] to avoid -- pass the spans
    being replaced.
    speech_ranges:  optional [(start_sec, end_sec), ...] of known speech (e.g.
                    from the transcript). When given, candidate windows must
                    overlap speech, which avoids selecting a long musical
                    interlude that happens to score well on SNR.

    Returns (audio, info). Falls back to the whole track when it is shorter than
    min_sec, so a caller always gets usable audio.
    """
    a = to_mono(track)
    n = a.shape[0]
    dur = n / float(sr)
    excl = list(exclude_ranges or [])

    if dur <= min_sec:
        _, detail = score_window(a, sr)
        detail.update({"selected": "whole track (shorter than min_sec)",
                       "start_sec": 0.0, "end_sec": round(dur, 2)})
        return a, detail

    win = int(min(target_sec, dur) * sr)
    hop = max(int(hop_sec * sr), 1)

    # Frame the whole track once; windows index into it rather than recomputing.
    e_all, starts_all = frame_energy_db(a, sr)
    frame_hop = max(int(sr * 0.010), 1)

    best = None
    pos = 0
    while pos + win <= n:
        if excl and _overlaps(pos, pos + win, excl):
            pos += hop
            continue
        if speech_ranges:
            w0, w1 = pos / sr, (pos + win) / sr
            if not any(w0 < r1 and w1 > r0 for r0, r1 in speech_ranges):
                pos += hop
                continue
        i0 = pos // frame_hop
        i1 = max(i0 + 1, (pos + win) // frame_hop)
        sc, detail = score_window(a[pos:pos + win], sr,
                                  e=e_all[i0:i1], starts=starts_all[i0:i1])
        if best is None or sc > best[0]:
            best = (sc, pos, detail)
        pos += hop

    if best is None:
        # No window of the requested length avoids the excluded regions. Shrink
        # progressively rather than jumping straight to the whole track: a short
        # CLEAN reference beats a long one containing the audio being replaced,
        # because conditioning on that audio biases the synthesiser toward
        # reproducing what was already there.
        for shrink_sec in (min_sec, min_sec * 0.66, 3.0, 2.0, 1.5):
            win = int(max(shrink_sec, 1.0) * sr)
            if win > n:
                continue
            pos = 0
            while pos + win <= n:
                if not (excl and _overlaps(pos, pos + win, excl)):
                    sc, detail = score_window(a[pos:pos + win], sr)
                    if best is None or sc > best[0]:
                        best = (sc, pos, detail)
                pos += hop
            if best is not None:
                break
    if best is None:
        _, detail = score_window(a, sr)
        detail.update({"selected": "whole track (no usable window)",
                       "start_sec": 0.0, "end_sec": round(dur, 2)})
        return a, detail

    sc, pos, detail = best
    clip = a[pos:pos + win].copy()
    # Short fades so the reference doesn't start or end on a hard edge, which
    # some encoders read as a transient.
    clip = fade(clip, int(0.010 * sr), int(0.010 * sr))
    detail.update({
        "selected": "best-scoring window",
        "start_sec": round(pos / float(sr), 2),
        "end_sec": round((pos + win) / float(sr), 2),
        "duration_sec": round(win / float(sr), 2),
        "considered_sec": round(dur, 2),
    })
    return clip, detail


def describe(info):
    """One-line summary for logs and the UI."""
    if not info:
        return "no reference info"
    return (f"{info.get('start_sec')}s-{info.get('end_sec')}s "
            f"(score {info.get('score')}, SNR {info.get('snr_db')}dB, "
            f"speech {info.get('speech_density')})")
