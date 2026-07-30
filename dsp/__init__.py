"""
dsp — the realism layer for transcript-driven audio editing.

Voice cloning is the easy part: a short reference gets a convincing timbre. What
gives an edit away is the SEAM, and the seam is signal processing, not machine
learning:

  - the generated span's mic tone and room differ from the original's
  - it has a dead-silent noise floor where the original has room tone
  - its loudness and spectral tilt jump at the join
  - it doesn't occupy the original's time slot, so video timing drifts
  - butt-joined waveforms click

Each of those has a stage here. The guiding principle is to keep as much real
audio as possible and make the regenerated region indistinguishable from its
immediate neighbours -- so a four-word edit leaves ~97% genuine recording, and
the 3% is matched to what surrounds it.

Typical use:

    from dsp import match_and_splice
    new_track, report = match_and_splice(
        track, generated_span, start_sample, end_sample, sr)
    print(summarise(report))

Every stage is independently callable and every one returns a report, so a poor
result can be traced to the stage responsible.

Depends only on numpy and scipy (plus pyloudnorm for BS.1770 loudness, which
degrades to RMS if unavailable). No GPU, no model weights -- it runs in the
principal runtime alongside everything else.
"""
from dsp.audio import (load, save, as_float32, to_mono, apply_gain,
                       limit_peak, resample, rms_db, peak, fade, pad_or_trim,
                       frame_energy_db, trim_silence)
from dsp.loudness import (measure_lufs, measure_loudness_db, match_loudness,
                          loudness_delta_db)
from dsp.spectral import (match_spectrum, correction_curve_db,
                          spectral_distance_db)
from dsp.room import (estimate_decay_sec, match_room, decay_mismatch, synth_ir)
from dsp.noisefloor import (estimate_noise_floor_db, measure_floor,
                            extract_room_tone, build_bed, inject)
from dsp.splice import (nearest_zero_crossing, snap_to_quiet, plan_boundaries,
                        crossfade_splice, seam_discontinuity_db)
from dsp.timefit import (required_rate, time_scale, fit_duration,
                         DEFAULT_MAX_STRETCH)
from dsp.match import (neighbourhood, match_span, match_and_splice, summarise,
                       DEFAULT_CONTEXT_SEC)

__all__ = [
    # audio
    "load", "save", "as_float32", "to_mono", "apply_gain", "limit_peak", "resample",
    "rms_db", "peak", "fade", "pad_or_trim", "frame_energy_db",
    "trim_silence",
    # loudness
    "measure_lufs", "measure_loudness_db", "match_loudness",
    "loudness_delta_db",
    # spectral
    "match_spectrum", "correction_curve_db", "spectral_distance_db",
    # room
    "estimate_decay_sec", "match_room", "decay_mismatch", "synth_ir",
    # noise floor
    "estimate_noise_floor_db", "measure_floor", "extract_room_tone",
    "build_bed", "inject",
    # splice
    "nearest_zero_crossing", "snap_to_quiet", "plan_boundaries",
    "crossfade_splice", "seam_discontinuity_db",
    # time
    "required_rate", "time_scale", "fit_duration", "DEFAULT_MAX_STRETCH",
    # orchestration
    "neighbourhood", "match_span", "match_and_splice", "summarise",
    "DEFAULT_CONTEXT_SEC",
]
