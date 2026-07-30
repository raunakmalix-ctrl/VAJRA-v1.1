"""
The realism layer: turn a raw synthesised span into one that is acoustically
indistinguishable from the audio either side of it, then splice it in.

Stage order is not arbitrary -- each stage assumes the previous one has run:

  1. duration fit      Do this first: every later measurement is length-
                       dependent, and time-scaling after filtering would smear
                       whatever was just corrected.
  2. spectral match    Before loudness, because changing the tone curve changes
                       the measured loudness. Corrects channel/mic character.
  3. room match        After tone, because reverb inherits the tone it is fed.
                       Conservative by design; often declines to act.
  4. loudness match    After everything that alters level, so it has the final
                       word and lands the span exactly on its neighbours.
  5. noise floor       Last of the audio stages: it deliberately adds low-level
                       noise, and no later stage should then re-gain or filter
                       it (loudness matching a bed of hiss would fight itself).
  6. splice            Zero-crossing-snapped equal-power crossfade.

Every stage returns a report, and `match_and_splice` collects them. That trail is
what makes a bad result diagnosable instead of mysterious -- and it feeds the
scorecard directly.

The reference for stages 2-5 is the real audio immediately AROUND the span, not
the whole track: the goal is to be indistinguishable from the neighbours, and a
recording's character drifts over minutes (speaker moves off-mic, a door opens).
Matching a local neighbourhood is both easier and more correct.
"""
import numpy as np

from dsp.audio import as_float32, to_mono, limit_peak
from dsp import loudness as _loud
from dsp import spectral as _spec
from dsp import room as _room
from dsp import noisefloor as _nf
from dsp import splice as _splice
from dsp import timefit as _time

DEFAULT_CONTEXT_SEC = 2.0


def neighbourhood(track, start, end, sr, context_sec=DEFAULT_CONTEXT_SEC,
                  exclude_span=True):
    """Real audio surrounding [start, end), for use as the match reference.

    Excludes the span itself by default -- including it would mean matching the
    span to a reference that is partly the span, weakening every correction.
    """
    a = as_float32(track)
    n = a.shape[0]
    ctx = int(context_sec * sr)
    pre = a[max(0, start - ctx):start]
    post = a[end:min(n, end + ctx)]
    if not exclude_span:
        return a[max(0, start - ctx):min(n, end + ctx)]
    if pre.size and post.size:
        return np.concatenate([pre, post], axis=0)
    return pre if pre.size else post


def match_span(span, reference, sr, target_sec=None,
               room_tone=None, target_floor_db=None,
               spectral_strength=1.0, room_strength=0.35,
               max_stretch=_time.DEFAULT_MAX_STRETCH):
    """Run the full match chain on one span. Returns (audio, report)."""
    report = {}
    a = as_float32(span)

    if target_sec is not None:
        a, report["duration"] = _time.fit_duration(
            a, sr, target_sec, max_stretch=max_stretch)

    a, report["spectral"] = _spec.match_spectrum(
        a, reference, sr, strength=spectral_strength)

    a, report["room"] = _room.match_room(
        a, reference, sr, strength=room_strength)

    a, report["loudness"] = _loud.match_loudness(a, reference, sr)

    if room_tone is not None:
        # span_has_own_noise=False: everything reaching this function is
        # synthesiser output, so it carries no recorded noise to subtract.
        a, report["noise_floor"] = _nf.inject(
            a, room_tone, sr, target_floor_db=target_floor_db,
            span_has_own_noise=False)
    else:
        report["noise_floor"] = {"applied": False,
                                 "reason": "no room tone supplied"}

    a = limit_peak(a, ceiling_db=-1.0)
    return a, report


def match_and_splice(track, span_audio, start, end, sr,
                     context_sec=DEFAULT_CONTEXT_SEC,
                     fade_ms=25.0, preserve_length=True,
                     snap_boundaries=True,
                     spectral_strength=1.0, room_strength=0.35,
                     max_stretch=_time.DEFAULT_MAX_STRETCH,
                     room_tone=None):
    """Match `span_audio` to its surroundings in `track` and splice it in.

    track       full original audio (float32, sr)
    span_audio  freshly synthesised replacement for track[start:end]
    start, end  sample indices into `track`

    Returns (new_track, report).
    """
    a = as_float32(track)
    n = a.shape[0]
    start = int(np.clip(start, 0, n))
    end = int(np.clip(end, start, n))
    report = {}

    if snap_boundaries:
        start, end, report["boundaries"] = _splice.plan_boundaries(
            a, start, end, sr)
    else:
        report["boundaries"] = {"snapped": False}

    ref = neighbourhood(a, start, end, sr, context_sec)
    if ref.size == 0:
        # Degenerate: the edit covers the entire track, so there are no
        # neighbours to match against. Match to the track itself and say so.
        ref = a
        report["reference"] = {"source": "whole track (span covers all audio)"}
    else:
        report["reference"] = {"source": "surrounding audio",
                              "sec": round(ref.shape[0] / float(sr), 2)}

    if room_tone is None:
        room_tone, tone_info = _nf.extract_room_tone(
            a, sr, exclude_ranges=[(start, end)])
        report["room_tone"] = tone_info
        if room_tone.size == 0:
            room_tone = None

    target_floor = _nf.estimate_noise_floor_db(ref, sr) if ref.size else None
    span_sec = (end - start) / float(sr)

    matched, span_report = match_span(
        span_audio, ref, sr,
        target_sec=span_sec if preserve_length else None,
        room_tone=room_tone, target_floor_db=target_floor,
        spectral_strength=spectral_strength, room_strength=room_strength,
        max_stretch=max_stretch)
    report.update(span_report)

    out, splice_info = _splice.crossfade_splice(
        a, matched, start, end, sr, fade_ms=fade_ms,
        preserve_length=preserve_length)
    report["splice"] = splice_info

    # Seam quality, measured on the result -- the number that actually matters.
    report["seam"] = {
        "start_discontinuity_db": round(
            _splice.seam_discontinuity_db(out, start, sr), 2),
        "end_discontinuity_db": round(
            _splice.seam_discontinuity_db(out, end, sr), 2),
    }
    return out, report


def summarise(report):
    """One-line human summary of a match report, for logs and the UI."""
    bits = []
    d = report.get("duration") or {}
    if d.get("status") == "stretched":
        bits.append(f"stretched x{d.get('rate')}")
    elif d.get("status") == "exceeded":
        bits.append(f"DURATION CAPPED (needed x{d.get('required_rate')}, "
                    f"off by {d.get('shortfall_sec')}s)")
    if (report.get("spectral") or {}).get("applied"):
        bits.append(f"tone {report['spectral']['max_band_gain_db']:+.1f}dB")
    if (report.get("room") or {}).get("applied"):
        bits.append(f"room +{report['room']['added_decay_sec']}s")
    lo = report.get("loudness") or {}
    if lo:
        bits.append(f"level {lo.get('gain_db', 0):+.1f}dB ({lo.get('method')})")
    if (report.get("noise_floor") or {}).get("applied"):
        bits.append("room tone laid in")
    seam = report.get("seam") or {}
    if seam:
        bits.append(f"seam {seam.get('start_discontinuity_db')}/"
                    f"{seam.get('end_discontinuity_db')}dB")
    return "; ".join(bits) if bits else "no corrections applied"
