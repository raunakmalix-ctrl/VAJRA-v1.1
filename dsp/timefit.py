"""
Duration fitting: making a regenerated span occupy its original time slot.

The new words almost never take exactly as long to say as the old ones. If the
span's duration changes, everything after it shifts, and the audio stops being
frame-aligned with the video -- lips desynchronise for the whole remainder of the
clip, not just the edit.

Preference order (best first):

  1. Ask the synthesiser for a specific duration. Some models expose explicit
     duration control; when available that is strictly better than fixing it
     afterwards, because the model distributes the timing across phonemes
     naturally. Not implemented here -- this module is the fallback for models
     that cannot.
  2. Time-scale modification, hard-capped. Speech tolerates roughly +/-10-15%
     before it starts sounding rushed or drawled. Past that the artefact is worse
     than the problem.
  3. Give up and report it. Beyond the cap the caller must choose: re-generate
     with different wording, extend the span to absorb neighbouring silence, or
     ripple the video.

Stage 3 existing as an explicit outcome is the point. The previous
implementation stretched by whatever factor was required, so a badly-mismatched
edit silently produced chipmunk or drawled speech; now it refuses and says so.

Method: WSOLA (waveform-similarity overlap-add) in numpy. Unlike naive
resampling it changes duration WITHOUT changing pitch, and unlike a phase
vocoder it does not smear transients -- it works by finding, for each output
frame, the input offset whose waveform best continues the previous frame, so
periodicity is preserved across joins.
"""
import numpy as np

from dsp.audio import as_float32, to_mono, n_channels, EPS

DEFAULT_MAX_STRETCH = 0.15          # +/-15%


def required_rate(src_sec, target_sec):
    """Playback rate needed to fit src into target. >1 means speed up."""
    if target_sec <= 0:
        return 1.0
    return float(src_sec) / float(target_sec)


def _wsola_mono(x, rate, sr, frame_ms=40.0, tol_ms=10.0):
    """Time-scale a mono signal by `rate` (>1 shortens) preserving pitch."""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0 or abs(rate - 1.0) < 1e-4:
        return a.astype(np.float32)

    N = max(64, int(sr * frame_ms / 1000.0))
    if N % 2:
        N += 1
    hop_s = N // 2                                  # synthesis hop
    hop_a = max(1, int(round(hop_s * rate)))        # analysis hop
    tol = max(1, int(sr * tol_ms / 1000.0))
    win = np.hanning(N)

    n_out = int(np.ceil(a.size / rate)) + N
    out = np.zeros(n_out, dtype=np.float64)
    wsum = np.zeros(n_out, dtype=np.float64)

    # `nat` is where the previous frame would naturally have continued; we look
    # for the offset near the ideal analysis position that best matches it.
    prev_tail = None
    read = 0
    write = 0
    while read + N <= a.size and write + N <= n_out:
        if prev_tail is None:
            best = read
        else:
            lo = max(0, read - tol)
            hi = min(a.size - N, read + tol)
            if hi <= lo:
                best = int(np.clip(read, 0, max(a.size - N, 0)))
            else:
                # Correlate the candidate frame heads against the previous
                # frame's tail; the argmax is the similarity-preserving offset.
                cand = np.lib.stride_tricks.sliding_window_view(
                    a[lo:hi + N], N)[: hi - lo + 1]
                scores = cand[:, :hop_s] @ prev_tail
                norms = np.sqrt(np.sum(cand[:, :hop_s] ** 2, axis=1) + EPS)
                best = lo + int(np.argmax(scores / norms))

        frame = a[best:best + N] * win
        out[write:write + N] += frame
        wsum[write:write + N] += win
        prev_tail = a[best + hop_s:best + N].copy()
        read = best + hop_a
        write += hop_s

    nz = wsum > 1e-6
    out[nz] /= wsum[nz]
    target_len = int(round(a.size / rate))
    return out[:max(target_len, 1)].astype(np.float32)


def time_scale(x, sr, rate, frame_ms=40.0, tol_ms=10.0):
    """Time-scale audio by `rate`, preserving pitch. Handles multi-channel."""
    a = as_float32(x)
    if n_channels(a) == 1:
        return _wsola_mono(a, rate, sr, frame_ms, tol_ms)
    cols = [_wsola_mono(a[:, c], rate, sr, frame_ms, tol_ms)
            for c in range(a.shape[1])]
    m = min(c.size for c in cols)
    return np.stack([c[:m] for c in cols], axis=1).astype(np.float32)


def fit_duration(x, sr, target_sec, max_stretch=DEFAULT_MAX_STRETCH,
                 tolerance_ms=15.0):
    """Fit `x` to target_sec by pitch-preserving time-scaling, within a cap.

    Returns (audio, info). info['status'] is one of:

      'exact'     already within tolerance; untouched
      'stretched' time-scaled; info['rate'] records by how much
      'exceeded'  required rate outside the cap. Audio is returned time-scaled
                  to the CAP (the closest honest attempt) and info['shortfall_sec']
                  states how far off it still is, so the caller can decide.
    """
    a = as_float32(x)
    src_sec = a.shape[0] / float(sr)
    target_sec = float(target_sec)

    info = {"source_sec": round(src_sec, 3),
            "target_sec": round(target_sec, 3),
            "max_stretch": max_stretch}

    if target_sec <= 0:
        info["status"] = "exact"
        return a, info

    if abs(src_sec - target_sec) * 1000.0 <= tolerance_ms:
        info.update({"status": "exact", "rate": 1.0})
        return a, info

    rate = required_rate(src_sec, target_sec)
    lo, hi = 1.0 - max_stretch, 1.0 + max_stretch

    if lo <= rate <= hi:
        out = time_scale(a, sr, rate)
        info.update({"status": "stretched", "rate": round(rate, 4),
                     "result_sec": round(out.shape[0] / float(sr), 3)})
        return out, info

    capped = float(np.clip(rate, lo, hi))
    out = time_scale(a, sr, capped)
    achieved = out.shape[0] / float(sr)
    info.update({
        "status": "exceeded",
        "required_rate": round(rate, 4),
        "applied_rate": round(capped, 4),
        "result_sec": round(achieved, 3),
        "shortfall_sec": round(achieved - target_sec, 3),
        "advice": ("Reword the edit to a similar syllable count, widen the span "
                   "to absorb neighbouring silence, or allow a ripple edit."),
    })
    return out, info
