# -*- coding: utf-8 -*-
"""Numerical tests for the dsp realism layer. Run from the project root."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import dsp
from dsp import audio, loudness, spectral, room, noisefloor, splice, timefit, match

SR = 24000
rng = np.random.RandomState(7)
FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def synth_speech(sec, f0=120.0, sr=SR, seed=0, bright=1.0):
    """Crude voiced-speech surrogate: harmonic stack + formant-ish shaping,
    amplitude-modulated at a syllable rate, with pauses."""
    r = np.random.RandomState(seed)
    n = int(sec * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for k in range(1, 26):
        amp = 1.0 / (k ** (1.35 / bright))
        x += amp * np.sin(2 * np.pi * f0 * k * t + r.uniform(0, 2 * np.pi))
    # syllable envelope ~4 Hz with real pauses
    env = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t - np.pi / 2))
    env = np.clip(env - 0.12, 0, None) / 0.88
    gate = (np.sin(2 * np.pi * 0.7 * t) > -0.55).astype(float)
    # Smooth the gate: an instantaneous offset has no decay tail to measure,
    # whereas real speech offsets fall over tens of milliseconds.
    from scipy.signal import lfilter as _lf
    al = np.exp(-1.0 / (0.035 * sr))
    gate = _lf([1 - al], [1, -al], gate)
    x *= env * gate
    x += 0.002 * r.randn(n)
    x /= (np.max(np.abs(x)) + 1e-9)
    return (0.35 * x).astype(np.float32)


def add_noise_floor(x, db, seed=1):
    r = np.random.RandomState(seed)
    amp = 10 ** (db / 20.0)
    return (x + amp * r.randn(x.size).astype(np.float32)).astype(np.float32)


def reverb(x, decay_sec, sr=SR, wet=0.35, seed=3):
    from scipy.signal import fftconvolve
    ir = room.synth_ir(decay_sec, sr, seed=seed)
    w = fftconvolve(x.astype(np.float64), ir, mode="full")[:x.size]
    w *= (np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12)
          / (np.sqrt(np.mean(w ** 2) + 1e-12)))
    return ((1 - wet) * x + wet * w).astype(np.float32)


print("=" * 74)
print("dsp.audio")
print("=" * 74)
x = synth_speech(2.0, seed=1)
check("as_float32 keeps float32 without copy", audio.as_float32(x) is x)
i16 = (x * 32767).astype(np.int16)
conv = audio.as_float32(i16)
check("int16 -> float in [-1,1]", np.max(np.abs(conv)) <= 1.0,
      f"peak={np.max(np.abs(conv)):.4f}")
check("int16 round-trip close", np.allclose(conv, x, atol=2e-4))
st = np.stack([x, x * 0.5], axis=1)
check("to_mono averages channels", np.allclose(audio.to_mono(st), x * 0.75, atol=1e-6))
check("apply_gain +6dB doubles", abs(audio.rms_db(audio.apply_gain(x, 6.0))
                                    - audio.rms_db(x) - 6.0) < 0.01)
loud = (x * 8).astype(np.float32)
lim = audio.limit_peak(loud, -1.0)
check("limit_peak caps at ceiling", audio.peak(lim) <= 10 ** (-1 / 20.0) + 1e-6,
      f"peak={audio.peak(lim):.4f}")
check("limit_peak never boosts", np.allclose(audio.limit_peak(x * 0.01, -1.0), x * 0.01))
rs = audio.resample(x, SR, 16000)
check("resample length ~ ratio", abs(rs.size - x.size * 16000 / SR) < 10,
      f"{rs.size} vs {int(x.size*16000/SR)}")
check("resample no-op when equal", audio.resample(x, SR, SR).size == x.size)
check("pad_or_trim exact", audio.pad_or_trim(x, 1000).size == 1000
      and audio.pad_or_trim(x, x.size + 500).size == x.size + 500)
e, s0 = audio.frame_energy_db(x, SR)
check("frame_energy_db shapes align", e.size == s0.size and e.size > 50)

print()
print("=" * 74)
print("dsp.loudness")
print("=" * 74)
a = synth_speech(3.0, seed=2)
b = audio.apply_gain(a, -7.0)
lu = loudness.measure_lufs(a, SR)
check("measure_lufs returns a value for 3s", lu is not None, f"{lu}")
check("measure_lufs None for 0.1s", loudness.measure_lufs(a[:2400], SR) is None)
out, info = loudness.match_loudness(b, a, SR)
check("match_loudness closes the gap",
      loudness.loudness_delta_db(out, a, SR) < 0.6,
      f"delta={loudness.loudness_delta_db(out, a, SR):.3f}dB via {info['method']}")
check("match_loudness reports ~+7dB", abs(info["gain_db"] - 7.0) < 0.6, str(info["gain_db"]))
# clamp behaviour
tiny = audio.apply_gain(a, -40.0)
_, ci = loudness.match_loudness(tiny, a, SR)
check("gain is clamped at the cap", ci["gain_clamped"] and ci["gain_db"] <= 18.0,
      f"applied={ci['gain_db']} requested={ci['requested_gain_db']}")
# short span falls back to RMS for BOTH sides (never mixes methods)
sh = a[:4000]
_, si = loudness.match_loudness(sh, a, SR)
check("short span uses rms method consistently", si["method"] == "rms")
peaky = (a * 6).astype(np.float32)
po, _ = loudness.match_loudness(peaky, a, SR)
check("no clipping after match", audio.peak(po) <= 10 ** (-1 / 20.0) + 1e-6)

print()
print("=" * 74)
print("dsp.spectral")
print("=" * 74)
from scipy.signal import butter, lfilter
ref = synth_speech(3.0, seed=4, bright=1.0)
# Make a tonally-different version: heavy low-pass = "wrong mic"
def tilt(y, db_at_top, sr=SR):
    """Gentle first-order spectral tilt -- the scale of a real mic change."""
    bl = butter(1, 1500 / (sr / 2), btype="low")
    lo = lfilter(bl[0], bl[1], y)
    hi = y - lo
    return (lo + (10 ** (db_at_top / 20.0)) * hi).astype(np.float32)

bq = butter(4, 1800 / (SR / 2), btype="low")   # extreme, for the cap test
dull = tilt(ref, -8.0)
d0 = spectral.spectral_distance_db(dull, ref, SR)
fixed, sinfo = spectral.match_spectrum(dull, ref, SR)
d1 = spectral.spectral_distance_db(fixed, ref, SR)
check("match_spectrum applied", sinfo["applied"], str(sinfo))
check("spectral distance substantially reduced", d1 < d0 * 0.6,
      f"{d0:.2f}dB -> {d1:.2f}dB")
# Extreme mismatch: correction is capped by design, so expect improvement but
# NOT full correction -- the cap is what stops a bad estimate wrecking a span.
extreme = lfilter(bq[0], bq[1], ref).astype(np.float32)
e0 = spectral.spectral_distance_db(extreme, ref, SR)
efix, _ = spectral.match_spectrum(extreme, ref, SR)
e1 = spectral.spectral_distance_db(efix, ref, SR)
check("extreme mismatch improved but cap-limited", e1 < e0,
      f"{e0:.2f}dB -> {e1:.2f}dB (cap +/-10dB)")
check("length preserved", fixed.size == dull.size)
short = ref[:int(0.1 * SR)]
_, s2 = spectral.match_spectrum(short, ref, SR)
check("declines on too-short span", not s2["applied"], s2.get("reason", ""))
_, s3 = spectral.match_spectrum(dull, ref, SR, strength=0.0)
check("strength=0 is a no-op", not s3["applied"])
f_, g_ = spectral.correction_curve_db(dull, ref, SR)
check("correction curve clamped", np.max(np.abs(g_)) <= 10.0 + 1e-6,
      f"max|g|={np.max(np.abs(g_)):.2f}dB")
check("curve spans to nyquist", abs(f_[-1] - SR / 2) < 1e-6)

print()
print("=" * 74)
print("dsp.room")
print("=" * 74)
dry = synth_speech(4.0, seed=5)
wet = reverb(dry, 0.60, wet=0.45)
d_dry = room.estimate_decay_sec(dry, SR)
d_wet = room.estimate_decay_sec(wet, SR)
check("decay measurable on both", d_dry is not None and d_wet is not None,
      f"dry={d_dry} wet={d_wet}")
gated = np.zeros(SR, dtype=np.float32)
gated[:1000] = 0.3
check("returns None gracefully with no usable tail",
      room.estimate_decay_sec(gated, SR) is None)
if d_dry and d_wet:
    check("wetter clip measures longer decay", d_wet > d_dry, f"{d_wet:.3f} > {d_dry:.3f}")
out_r, rinfo = room.match_room(dry, wet, SR, strength=0.35)
check("room match acts when reference wetter", rinfo["applied"], str(rinfo))
check("room match preserves length", out_r.size == dry.size)
_, rinfo2 = room.match_room(wet, dry, SR)
check("declines when reference is drier", not rinfo2["applied"],
      rinfo2.get("reason", ""))
_, rinfo3 = room.match_room(dry, wet, SR, strength=0.0)
check("strength=0 declines", not rinfo3["applied"])
ir = room.synth_ir(0.5, SR)
check("synth_ir unit energy", abs(np.sum(ir ** 2) - 1.0) < 1e-6)

print()
print("=" * 74)
print("dsp.noisefloor")
print("=" * 74)
clean = synth_speech(5.0, seed=6)
noisy = add_noise_floor(clean, -52.0)
fl = noisefloor.estimate_noise_floor_db(noisy, SR)
check("floor estimate in sane range", -70 < fl < -30, f"{fl:.1f} dBFS")
tone, tinfo = noisefloor.extract_room_tone(noisy, SR)
check("room tone harvested", tone.size > 0, f"{tinfo['found_sec']}s")
bed = noisefloor.build_bed(tone, int(0.4 * SR), SR)
check("bed exact length", bed.size == int(0.4 * SR))
big = noisefloor.build_bed(tone[:int(0.05 * SR)], int(2.0 * SR), SR)
check("bed tiles beyond source length", big.size == int(2.0 * SR))
# a TTS-like span: dead silent floor
tts = synth_speech(1.0, seed=8)
tts_floor_before = noisefloor.estimate_noise_floor_db(tts, SR)
inj, ninfo = noisefloor.inject(tts, tone, SR, target_floor_db=fl)
check("noise injected", ninfo["applied"], str(ninfo))
after = noisefloor.estimate_noise_floor_db(inj, SR)
check("floor raised toward target", abs(after - fl) < 6.0,
      f"before={tts_floor_before:.1f} after={after:.1f} target={fl:.1f}")
# Real recorded audio: its floor IS measurable, so declining is correct.
_, ninfo2 = noisefloor.inject(noisy[:SR], tone, SR, target_floor_db=-80.0,
                              span_has_own_noise=True)
check("declines when real noise already above target", not ninfo2["applied"],
      ninfo2.get("reason", ""))
_, ninfo3 = noisefloor.inject(tts, np.zeros(0, dtype=np.float32), SR)
check("handles absent room tone", not ninfo3["applied"])
# Truly continuous audio has no silence, so its floor is unmeasurable and must
# be flagged rather than trusted.
steady = (0.2 * np.sin(2 * np.pi * 220 * np.arange(int(0.5 * SR)) / SR)
          ).astype(np.float32)
_, rel = noisefloor.measure_floor(steady, SR)
fl_t, rel_t = noisefloor.measure_floor(noisy, SR)
check("floor unreliable on continuous audio", not rel)
check("floor reliable on a track with real silence", rel_t, f"{fl_t:.1f}dBFS")
# A synthetic span carries no recorded noise, so the bed goes in at the target
# level even though the span's own measured 'floor' sits above it.
wetish = reverb(synth_speech(0.8, seed=41), 0.5, wet=0.4)
inj2, ni2 = noisefloor.inject(wetish, tone, SR, target_floor_db=-50.0,
                              span_has_own_noise=False)
check("injects into a synthetic span regardless of its measured floor",
      ni2["applied"], str(ni2))
# Declining is still correct when the caller confirms real noise above target.
_, ni3 = noisefloor.inject(noisy[:SR], tone, SR, target_floor_db=-80.0,
                           span_has_own_noise=True)
check("declines when real noise already exceeds target", not ni3["applied"],
      ni3.get("reason", ""))

print()
print("=" * 74)
print("dsp.splice")
print("=" * 74)
sig = synth_speech(4.0, seed=9)
zc = splice.nearest_zero_crossing(sig, 10000, int(0.002 * SR))
check("zero crossing found near index", abs(zc - 10000) <= int(0.002 * SR))
check("zc really is a crossing",
      np.signbit(sig[zc - 1]) != np.signbit(sig[zc]) or abs(sig[zc]) < 1e-3)
q = splice.snap_to_quiet(sig, int(2.0 * SR), SR, 150.0)
e_all, st_all = audio.frame_energy_db(sig, SR, 20.0, 5.0)
w = int(SR * 0.15)
sel = (st_all >= int(2.0 * SR) - w) & (st_all <= int(2.0 * SR) + w)
check("snap_to_quiet picks a local minimum",
      abs(e_all[sel].min() - e_all[st_all == q][0]) < 1e-4 if np.any(st_all == q) else True)
s_, e_, binfo = splice.plan_boundaries(sig, int(1.0 * SR), int(1.5 * SR), SR)
check("plan_boundaries returns ordered span", e_ > s_, f"{s_}..{e_} {binfo}")

repl = synth_speech(0.5, f0=150.0, seed=10)
start, end = int(1.0 * SR), int(1.5 * SR)
out_s, sinfo2 = splice.crossfade_splice(sig, repl, start, end, SR)
check("splice preserves total length", out_s.size == sig.size,
      f"{out_s.size} vs {sig.size}")
check("audio outside span untouched (head)", np.allclose(out_s[:start - 2000], sig[:start - 2000]))
check("audio outside span untouched (tail)", np.allclose(out_s[end + 2000:], sig[end + 2000:]))
# a hard butt-join should be measurably worse at the seam than a crossfade
butt = sig.copy()
butt[start:end] = audio.pad_or_trim(repl, end - start)
d_butt = max(splice.seam_discontinuity_db(butt, start, SR),
             splice.seam_discontinuity_db(butt, end, SR))
d_xf = max(splice.seam_discontinuity_db(out_s, start, SR),
           splice.seam_discontinuity_db(out_s, end, SR))
check("crossfade seam <= butt-join seam", d_xf <= d_butt + 0.05,
      f"butt={d_butt:.2f}dB crossfade={d_xf:.2f}dB")
long_repl = synth_speech(1.2, seed=11)
out_rp, ri = splice.crossfade_splice(sig, long_repl, start, end, SR,
                                     preserve_length=False)
check("ripple mode changes length", out_rp.size != sig.size, f"{out_rp.size}")

print()
print("=" * 74)
print("dsp.timefit")
print("=" * 74)
src = synth_speech(1.00, seed=12)
check("required_rate math", abs(timefit.required_rate(1.0, 0.5) - 2.0) < 1e-9)
o1, i1 = timefit.fit_duration(src, SR, 1.00)
check("exact when already right", i1["status"] == "exact", str(i1["status"]))
o2, i2 = timefit.fit_duration(src, SR, 0.92)
check("stretch within cap applied", i2["status"] == "stretched", str(i2))
check("stretched length ~ target", abs(o2.size / SR - 0.92) < 0.03,
      f"{o2.size/SR:.3f}s")
o3, i3 = timefit.fit_duration(src, SR, 0.50)
check("beyond cap reported, not silently done", i3["status"] == "exceeded", str(i3["status"]))
check("capped attempt returned", abs(i3["applied_rate"] - 1.15) < 1e-6, str(i3["applied_rate"]))
check("shortfall reported", "shortfall_sec" in i3 and i3["shortfall_sec"] > 0)
# pitch preservation: dominant frequency should not move
def dom_freq(y):
    Y = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    f = np.fft.rfftfreq(y.size, 1 / SR)
    band = (f > 80) & (f < 400)
    return f[band][np.argmax(Y[band])]
check("time_scale preserves pitch", abs(dom_freq(o2) - dom_freq(src)) < 4.0,
      f"{dom_freq(src):.1f}Hz -> {dom_freq(o2):.1f}Hz")
st2 = np.stack([src, src], axis=1)
o4 = timefit.time_scale(st2, SR, 1.1)
check("time_scale handles stereo", o4.ndim == 2 and o4.shape[1] == 2)

print()
print("=" * 74)
print("dsp.match  (end-to-end)")
print("=" * 74)
# Build a realistic scenario: original is reverberant, noisy, bright.
orig_dry = synth_speech(8.0, seed=20)
orig = add_noise_floor(reverb(orig_dry, 0.5, wet=0.35), -50.0, seed=21)
start, end = int(3.0 * SR), int(3.8 * SR)

# The "TTS output": different pitch, dull tone, dead-silent floor, too loud,
# and the wrong duration.
tts_raw = synth_speech(1.05, f0=150.0, seed=22, bright=0.6)
tts_raw = lfilter(bq[0], bq[1], tts_raw).astype(np.float32)
tts_raw = audio.apply_gain(tts_raw, +5.0)

new_track, rep = match.match_and_splice(orig, tts_raw, start, end, SR)
print("  report:", match.summarise(rep))

check("output length preserved", new_track.size == orig.size,
      f"{new_track.size} vs {orig.size}")
check("boundaries were snapped", rep["boundaries"]["snapped"])
check("duration stage ran", "duration" in rep and rep["duration"]["status"] in
      ("exact", "stretched", "exceeded"))
check("spectral applied", rep["spectral"]["applied"])
check("loudness applied", "gain_db" in rep["loudness"])
check("room tone harvested from original", rep["room_tone"]["found_sec"] > 0.2,
      f"{rep['room_tone']['found_sec']}s")
check("noise floor injected", rep["noise_floor"]["applied"], str(rep["noise_floor"]))
check("no clipping in output", audio.peak(new_track) <= 1.0)

s_, e_ = rep["splice"]["start"], rep["splice"]["end"]
# Compare matched span against its neighbours vs. the raw TTS against them.
ref_nb = match.neighbourhood(orig, s_, e_, SR)
span_out = new_track[s_:e_]
raw_fit = audio.pad_or_trim(tts_raw, e_ - s_)

d_raw = loudness.loudness_delta_db(raw_fit, ref_nb, SR)
d_new = loudness.loudness_delta_db(span_out, ref_nb, SR)
check("loudness match improved vs raw", d_new < d_raw,
      f"raw={d_raw:.2f}dB -> matched={d_new:.2f}dB")

t_raw = spectral.spectral_distance_db(raw_fit, ref_nb, SR)
t_new = spectral.spectral_distance_db(span_out, ref_nb, SR)
check("tone match improved vs raw", t_new < t_raw,
      f"raw={t_raw:.2f}dB -> matched={t_new:.2f}dB")

f_raw = noisefloor.estimate_noise_floor_db(raw_fit, SR)
f_new = noisefloor.estimate_noise_floor_db(span_out, SR)
f_ref = noisefloor.estimate_noise_floor_db(ref_nb, SR)
check("noise floor closer to neighbourhood",
      abs(f_new - f_ref) < abs(f_raw - f_ref),
      f"raw={f_raw:.1f} matched={f_new:.1f} target={f_ref:.1f} dBFS")

# Seam vs. a naive splice of the raw span (what the old pipeline did)
naive = orig.copy()
naive[s_:e_] = audio.fade(raw_fit, int(0.008 * SR), int(0.008 * SR))
seam_naive = max(splice.seam_discontinuity_db(naive, s_, SR),
                 splice.seam_discontinuity_db(naive, e_, SR))
seam_new = max(rep["seam"]["start_discontinuity_db"],
               rep["seam"]["end_discontinuity_db"])
check("seam better than naive splice", seam_new < seam_naive,
      f"naive={seam_naive:.2f}dB -> matched={seam_new:.2f}dB")

# Degenerate inputs must not explode
try:
    z, _ = match.match_and_splice(orig, np.zeros(10, dtype=np.float32),
                                  start, start, SR)
    check("empty span handled", z.size == orig.size)
except Exception as ex:
    check("empty span handled", False, repr(ex))
try:
    tiny_track = synth_speech(0.6, seed=30)
    z2, r2 = match.match_and_splice(tiny_track, synth_speech(0.2, seed=31),
                                    0, tiny_track.size, SR)
    check("span covering whole track handled", z2.size == tiny_track.size,
          r2["reference"]["source"])
except Exception as ex:
    check("span covering whole track handled", False, repr(ex))
try:
    st_track = np.stack([orig, orig], axis=1)
    z3, _ = match.match_and_splice(st_track, tts_raw, start, end, SR)
    check("stereo track handled", z3.shape == st_track.shape, str(z3.shape))
except Exception as ex:
    check("stereo track handled", False, repr(ex))

print()
print("=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL DSP TESTS PASSED")
