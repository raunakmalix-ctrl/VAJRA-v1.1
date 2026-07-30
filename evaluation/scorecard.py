"""
Objective detectability metrics for a finished edit.

The question "does it sound fake?" is unanswerable as posed, and answering it by
ear does not scale or survive disagreement. This turns it into numbers.

The central idea, and the reason this module is worth more than a list of
thresholds: an edit should not be measured against silence or against an absolute
target, but against the natural variation ALREADY PRESENT in the same recording.

Every real word boundary in real speech has some level discontinuity and some
spectral shift across it. Speakers change loudness between words, breathe, move
relative to the microphone. So a splice with a 2 dB discontinuity is not
inherently detectable -- it is only detectable if 2 dB is unusual for *this*
recording. Comparing an edit's seam against the distribution of natural seams
nearby converts an arbitrary threshold into a percentile with a real
interpretation: "this join is less abrupt than 70% of the genuine word
boundaries around it, so there is nothing there to notice."

Metrics needing no model, all implemented here:
  seam discontinuity   vs the natural-seam distribution (the headline number)
  loudness match       span vs its neighbourhood, same-method
  spectral match       broad tone difference from the neighbourhood
  noise-floor match    does the bed continue across the span
  duration accuracy    did the span land in its slot, and was stretching capped

Metrics needing models are deliberately NOT faked here. Speaker similarity
(ECAPA/WavLM), naturalness (UTMOS/DNSMOS) and intelligibility (an independent
ASR) each need a network and a download; `pending_metrics()` names them so a
report is explicit about what it did not measure rather than appearing complete.
"""
import numpy as np

from dsp.audio import to_mono, frame_energy_db
from dsp import loudness as _loud
from dsp import spectral as _spec
from dsp import noisefloor as _nf
from dsp import splice as _splice
from dsp import match as _match


# Below this, a span contains too little non-speech material for a noise-floor
# percentile to mean anything, regardless of its dynamic range.
_MIN_FLOOR_SPAN_SEC = 0.6

# Verdict bands for the seam percentile: how the edit's seam ranks among the
# genuine word boundaries of the same recording.
_SEAM_BANDS = [
    (0.50, "inaudible", "quieter than most genuine word boundaries here"),
    (0.80, "unlikely to be noticed", "within the range of genuine boundaries"),
    (0.95, "possibly audible", "more abrupt than most genuine boundaries"),
    (1.01, "likely audible", "more abrupt than nearly every genuine boundary"),
]


def natural_seam_distribution(track, sr, exclude_ranges=None,
                              min_gap_ms=60.0, max_points=400):
    """Discontinuity magnitudes at genuine word boundaries in `track`.

    Word boundaries are located as local energy minima -- the pauses between
    words -- rather than from the transcript, so this works even when word
    timings are unavailable and measures the same quantity the edit's seam is
    measured with.

    Regions being replaced are excluded: their boundaries are not genuine.
    """
    a = to_mono(track)
    e, starts = frame_energy_db(a, sr, frame_ms=20.0, hop_ms=10.0)
    if e.size < 8:
        return np.zeros(0, dtype=np.float32)

    finite = e[np.isfinite(e)]
    if finite.size == 0:
        return np.zeros(0, dtype=np.float32)
    floor = float(np.percentile(finite, 10.0))
    peak = float(np.percentile(finite, 90.0))
    if peak - floor < 6.0:
        return np.zeros(0, dtype=np.float32)

    # A boundary candidate is a frame quieter than its neighbours and below the
    # midpoint between floor and speech level.
    mid = floor + 0.45 * (peak - floor)
    excl = list(exclude_ranges or [])
    min_gap = int(min_gap_ms / 10.0)          # frames, at a 10 ms hop

    cands = []
    last = -min_gap
    for i in range(1, e.size - 1):
        if i - last < min_gap:
            continue
        if e[i] > mid:
            continue
        if not (e[i] <= e[i - 1] and e[i] <= e[i + 1]):
            continue
        s = int(starts[i])
        if any(s >= r0 and s <= r1 for r0, r1 in excl):
            continue
        cands.append(s)
        last = i

    if not cands:
        return np.zeros(0, dtype=np.float32)
    if len(cands) > max_points:
        idx = np.linspace(0, len(cands) - 1, max_points).astype(int)
        cands = [cands[i] for i in idx]

    return np.asarray(
        [_splice.seam_discontinuity_db(a, c, sr) for c in cands],
        dtype=np.float32)


def seam_percentile(value_db, distribution):
    """Fraction of natural seams at least as smooth as `value_db`.

    0 means smoother than every genuine boundary; 1 means more abrupt than all
    of them. None when there is no distribution to compare against, which must
    be reported rather than silently treated as a pass.
    """
    d = np.asarray(distribution, dtype=np.float32)
    if d.size == 0:
        return None
    return float(np.mean(d <= float(value_db)))


def _band(pct):
    if pct is None:
        return "unknown", "no genuine boundaries available to compare against"
    for limit, verdict, why in _SEAM_BANDS:
        if pct < limit:
            return verdict, why
    return "likely audible", "more abrupt than nearly every genuine boundary"


def score_edit(track, sr, span, neighbourhood=None, report=None,
               natural=None, exclude_ranges=None):
    """Score one edited span against its surroundings.

    track   the FINISHED audio
    span    (start_sample, end_sample) of the edited region
    report  the dsp match report for this span, if available -- supplies the
            duration outcome, which cannot be recovered from the audio alone
    natural the natural-seam distribution; computed if not supplied (pass it in
            when scoring several spans, so it is measured once)

    Returns a metrics dict. Every entry carries its own value and verdict, so a
    failing edit points at the stage responsible instead of yielding one opaque
    number.
    """
    a = to_mono(track)
    s0, s1 = int(span[0]), int(span[1])
    s0 = max(0, min(s0, a.shape[0]))
    s1 = max(s0, min(s1, a.shape[0]))
    span_audio = a[s0:s1]

    if neighbourhood is None:
        neighbourhood = _match.neighbourhood(a, s0, s1, sr)
    if natural is None:
        natural = natural_seam_distribution(
            a, sr, exclude_ranges=exclude_ranges or [(s0, s1)])

    out = {"span_sec": round((s1 - s0) / float(sr), 3)}

    # ── seam: the headline metric ──────────────────────────────────────────
    start_db = _splice.seam_discontinuity_db(a, s0, sr)
    end_db = _splice.seam_discontinuity_db(a, s1, sr)
    worst = max(start_db, end_db)
    pct = seam_percentile(worst, natural)
    verdict, why = _band(pct)
    out["seam"] = {
        "start_db": round(start_db, 2),
        "end_db": round(end_db, 2),
        "worst_db": round(worst, 2),
        "natural_median_db": (round(float(np.median(natural)), 2)
                              if len(natural) else None),
        "natural_p90_db": (round(float(np.percentile(natural, 90)), 2)
                           if len(natural) else None),
        "n_natural_seams": int(len(natural)),
        "percentile": None if pct is None else round(pct, 3),
        "verdict": verdict,
        "interpretation": why,
    }

    # ── loudness ───────────────────────────────────────────────────────────
    if span_audio.size and neighbourhood.size:
        d = _loud.loudness_delta_db(span_audio, neighbourhood, sr)
        out["loudness"] = {
            "delta_db": round(d, 2),
            # A 1 dB step at a splice is around the threshold of noticeability
            # for speech; 3 dB is clearly audible.
            "verdict": ("good" if d < 1.0 else
                        "acceptable" if d < 3.0 else "audible"),
        }

    # ── tone ───────────────────────────────────────────────────────────────
    if span_audio.size > sr // 8 and neighbourhood.size > sr // 8:
        sd = _spec.spectral_distance_db(span_audio, neighbourhood, sr)
        out["tone"] = {
            "distance_db": round(sd, 2),
            "verdict": ("good" if sd < 4.0 else
                        "acceptable" if sd < 8.0 else "mismatched"),
        }

    # ── noise floor ────────────────────────────────────────────────────────
    if neighbourhood.size:
        f_span, reliable = _nf.measure_floor(span_audio, sr)
        f_ref, _ = _nf.measure_floor(neighbourhood, sr)
        # A short span can pass the dynamic-range test on a single syllable plus
        # its release and still contain no actual silence, so its "floor" would
        # be quiet speech. Require enough material for the percentile to mean
        # something before reporting a verdict on it -- an unreliable number
        # presented confidently is worse than an admitted gap.
        long_enough = (s1 - s0) >= int(_MIN_FLOOR_SPAN_SEC * sr)
        reliable = bool(reliable and long_enough)
        entry = {"span_db": round(f_span, 2), "neighbourhood_db": round(f_ref, 2),
                 "measurable": reliable}
        if reliable:
            gap = abs(f_span - f_ref)
            entry.update({"delta_db": round(gap, 2),
                          "verdict": "good" if gap < 6.0 else "mismatched"})
        else:
            # Saying so beats reporting a confident wrong number: on a short
            # span a low energy percentile finds quiet speech, not noise.
            entry["verdict"] = "not measurable on a span this short"
        out["noise_floor"] = entry

    # ── duration ───────────────────────────────────────────────────────────
    dur = (report or {}).get("duration") or {}
    if dur:
        status = dur.get("status")
        out["duration"] = {
            "status": status,
            "verdict": ("good" if status in ("exact", "stretched")
                        else "timing compromised"),
        }
        if status == "exceeded":
            out["duration"]["shortfall_sec"] = dur.get("shortfall_sec")
            out["duration"]["advice"] = dur.get("advice")

    out["verdict"] = _overall(out)
    return out


def _overall(metrics):
    """Worst-case roll-up. An edit is only as good as its most audible defect."""
    bad = []
    seam = metrics.get("seam", {})
    if seam.get("verdict") in ("possibly audible", "likely audible"):
        bad.append("seam")
    for key, fail in (("loudness", {"audible"}),
                      ("tone", {"mismatched"}),
                      ("noise_floor", {"mismatched"}),
                      ("duration", {"timing compromised"})):
        if metrics.get(key, {}).get("verdict") in fail:
            bad.append(key)
    if not bad:
        return "pass"
    return "review: " + ", ".join(bad)


def score_all(track, sr, spans, reports=None):
    """Score every edited span in a finished track.

    The natural-seam distribution is measured once, with all edited regions
    excluded, so no edit's own seam contaminates the baseline it is judged by.
    """
    a = to_mono(track)
    spans = [(int(s), int(e)) for s, e in spans]
    natural = natural_seam_distribution(a, sr, exclude_ranges=spans)
    reports = reports or [None] * len(spans)

    results = []
    for i, (s0, s1) in enumerate(spans):
        rep = reports[i] if i < len(reports) else None
        results.append(score_edit(a, sr, (s0, s1), report=rep, natural=natural,
                                  exclude_ranges=spans))
    verdicts = [r["verdict"] for r in results]
    return {
        "spans": results,
        "n_spans": len(results),
        "n_natural_seams": int(len(natural)),
        "overall": "pass" if all(v == "pass" for v in verdicts) else "review",
        "pending": pending_metrics(),
    }


def pending_metrics():
    """Metrics this scorecard does NOT measure, and what each would need.

    Listed explicitly so a report reads as partial rather than complete. A
    scorecard that quietly omits speaker similarity invites the reader to assume
    identity was verified when it was not.
    """
    return [
        {"metric": "speaker similarity",
         "needs": "an ECAPA-TDNN or WavLM speaker encoder",
         "why": "confirms the clone is the same person, not merely well matched"},
        {"metric": "naturalness (MOS)",
         "needs": "UTMOS or DNSMOS",
         "why": "the gap against unedited spans matters, not the absolute value"},
        {"metric": "intelligibility",
         "needs": "an ASR independent of the one used for alignment",
         "why": "confirms the edit says the intended words"},
        {"metric": "lip-sync accuracy",
         "needs": "SyncNet (LSE-C / LSE-D)",
         "why": "compares edited windows against unedited ones"},
    ]


def summarise(card):
    """Multi-line human summary of a full scorecard."""
    if not card:
        return "no scorecard"
    lines = [f"Scorecard: {card['overall'].upper()} "
             f"({card['n_spans']} span(s), "
             f"{card['n_natural_seams']} genuine boundaries as baseline)"]
    for i, sp in enumerate(card.get("spans", []), 1):
        seam = sp.get("seam", {})
        pct = seam.get("percentile")
        pct_s = "n/a" if pct is None else f"{pct*100:.0f}th pct"
        bits = [f"seam {seam.get('worst_db')}dB ({pct_s}, {seam.get('verdict')})"]
        for k, field in (("loudness", "delta_db"), ("tone", "distance_db")):
            if k in sp:
                bits.append(f"{k} {sp[k][field]}dB [{sp[k]['verdict']}]")
        if "noise_floor" in sp:
            bits.append(f"floor [{sp['noise_floor']['verdict']}]")
        if "duration" in sp:
            bits.append(f"duration {sp['duration']['status']}")
        lines.append(f"  span {i} ({sp['span_sec']}s): " + "; ".join(bits)
                     + f"  -> {sp['verdict']}")
    if card.get("pending"):
        lines.append("  not measured: "
                     + ", ".join(p["metric"] for p in card["pending"]))
    return "\n".join(lines)
