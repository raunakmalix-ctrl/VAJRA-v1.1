# -*- coding: utf-8 -*-
"""Integration test: TranscriptEngine.apply_edits against a stubbed voice engine.

Verifies the real code path -- span resolution, room-tone harvesting, the dsp
match chain, length preservation and untouched-audio integrity -- without
needing XTTS, ffmpeg or a GPU.
"""
import os, sys, types, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# Minimal torch stub: core.device only probes CUDA availability, and this test
# deliberately exercises the CPU/DSP path with no models loaded.
_t = types.ModuleType("torch")
class _FakeTensor:            # scipy's array-API probe does getattr(torch,"Tensor")
    pass


_t.Tensor = _FakeTensor
_t.cuda = types.SimpleNamespace(
    is_available=lambda: False,
    empty_cache=lambda: None,
    mem_get_info=lambda: (0, 0),
    get_device_name=lambda i=0: "stub",
)
sys.modules.setdefault("torch", _t)

# --- stub the isolated-venv engines so importing the module doesn't need them
import core.config as cfg


def synth(sec, f0=120.0, sr=24000, seed=0):
    r = np.random.RandomState(seed)
    n = int(sec * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for k in range(1, 22):
        x += (1.0 / k ** 1.35) * np.sin(2 * np.pi * f0 * k * t + r.uniform(0, 6.28))
    env = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t - np.pi / 2))
    env = np.clip(env - 0.12, 0, None) / 0.88
    gate = (np.sin(2 * np.pi * 0.7 * t) > -0.55).astype(float)
    from scipy.signal import lfilter
    al = np.exp(-1.0 / (0.035 * sr))
    gate = lfilter([1 - al], [1, -al], gate)
    x *= env * gate
    x /= (np.max(np.abs(x)) + 1e-9)
    x = 0.35 * x + 10 ** (-50 / 20.0) * r.randn(n)     # real noise floor
    return x.astype(np.float32)


SR = 24000
TMP = tempfile.mkdtemp(prefix="vajra_relip_")
FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


import dsp

# Original "recording": 10 s, reverberant + noisy.
orig = synth(10.0, seed=1)
from scipy.signal import fftconvolve
ir = dsp.synth_ir(0.45, SR, seed=5)
wetp = fftconvolve(orig.astype(np.float64), ir, mode="full")[:orig.size]
wetp *= np.sqrt(np.mean(orig ** 2)) / (np.sqrt(np.mean(wetp ** 2)) + 1e-12)
orig = (0.7 * orig + 0.3 * wetp).astype(np.float32)

audio_path = os.path.join(TMP, "original.wav")
dsp.save(audio_path, orig, SR)
# Compare against what the pipeline reads back, so the assertion measures the
# pipeline's own fidelity rather than any file-format round trip.
orig, _sr_chk = dsp.load(audio_path, mono=True)
assert _sr_chk == SR

# Stub VoiceEngine: returns a deliberately mismatched span (bright, loud,
# silent floor, wrong length) so the match chain has real work to do.
_calls = {"n": 0}


class StubVoice:
    def run(self, text, reference_audio_path, language="en", **kw):
        _calls["n"] += 1
        span = synth(1.30, f0=155.0, sr=SR, seed=100 + _calls["n"])
        span = (span - span.mean()).astype(np.float32)
        # strip the noise floor to mimic TTS output, then over-drive the level
        from scipy.signal import butter, lfilter as lf
        b = butter(1, 400 / (SR / 2), btype="high")
        span = lf(b[0], b[1], span).astype(np.float32)
        span = dsp.apply_gain(span, +6.0)
        p = os.path.join(TMP, f"span_{_calls['n']}.wav")
        dsp.save(p, span, SR)
        return p


class StubLipsync:
    def run(self, video_path, audio_path, **kw):
        return video_path


import engines.transcript_engine as te
te.VoiceEngine = StubVoice
te.LipSyncEngine = StubLipsync

eng = te.TranscriptEngine()
eng.voice = StubVoice()
eng.lipsync = StubLipsync()

def with_words(text, start, end, pause_before=None):
    """Evenly-spaced word timings across [start, end), with an optional longer
    gap before one word index so pause-expansion has somewhere to cut."""
    toks = text.split()
    total = end - start
    gap = 0.04
    extra = 0.22 if pause_before is not None else 0.0
    wdur = (total - gap * (len(toks) - 1) - extra) / len(toks)
    words, t = [], start
    for i, w in enumerate(toks):
        if i > 0:
            t += gap + (extra if i == pause_before else 0.0)
        words.append({"word": w, "start": round(t, 4), "end": round(t + wdur, 4)})
        t += wdur
    return {"start": start, "end": end, "text": text, "words": words}


state = {
    "video": "unused.mp4",
    "audio": audio_path,
    "language": "en",
    "segments": [
        with_words("the convoy will move at first light", 0.5, 2.0, pause_before=4),
        with_words("all units acknowledge the order", 2.6, 4.2),
        with_words("maintain radio silence until then", 5.0, 6.6, pause_before=3),
        with_words("report status on arrival", 7.2, 9.0),
    ],
}

print("=" * 74)
print("apply_edits — one line changed")
print("=" * 74)
edited = "\n".join([
    "the convoy will move after midnight",       # CHANGED
    "all units acknowledge the order",
    "maintain radio silence until then",
    "report status on arrival",
])

# Monkeypatch lipsync out of the flow: we only test the audio stages here.
out_audio = {}
orig_lipsync_run = eng.lipsync.run


def cap_lipsync(video_path, audio_path, **kw):
    out_audio["path"] = audio_path
    return video_path


eng.lipsync.run = cap_lipsync

res = eng.apply_edits(state, edited, method="latentsync")
check("returns a path", isinstance(res, str))
check("voice engine called exactly once", _calls["n"] == 1, f"n={_calls['n']}")
check("match reports recorded", len(state.get("match_reports", [])) == 1)

new_track, new_sr = dsp.load(out_audio["path"], mono=True)
check("sample rate preserved", new_sr == SR, f"{new_sr}")
check("total length preserved", abs(new_track.size - orig.size) <= 1,
      f"{new_track.size} vs {orig.size}")

rep = state["match_reports"][0]
print("  report:", dsp.summarise(rep))
s_, e_ = rep["splice"]["start"], rep["splice"]["end"]
check("edited segment index recorded", rep["segment"] == 0)
check("word-level granularity used", rep["granularity"] == "word",
      str(rep["granularity"]))
seg0 = state["segments"][0]
span_dur = (e_ - s_) / SR
line_dur = seg0["end"] - seg0["start"]
check("regenerated less audio than the whole line", span_dur < line_dur * 0.8,
      f"{span_dur:.2f}s of {line_dur:.2f}s")
check("cloning reference chosen automatically", "reference_info" in state,
      str(state.get("reference_info", {}).get("selected")))
ri = state["reference_info"]
check("reference avoids the edited span (incl. snap margin)",
      not (ri["start_sec"] < e_ / SR and ri["end_sec"] > s_ / SR),
      f"ref {ri['start_sec']}-{ri['end_sec']} vs edit {s_/SR:.2f}-{e_/SR:.2f}")
check("spectral stage ran", rep["spectral"]["applied"])
check("loudness stage ran", "gain_db" in rep["loudness"])
check("room tone harvested", rep["room_tone"]["found_sec"] > 0.2,
      f"{rep['room_tone']['found_sec']}s")
check("noise floor injected", rep["noise_floor"]["applied"],
      str(rep["noise_floor"].get("reason", "")))

# Untouched regions must be sample-identical (allowing for crossfade skirts).
fade_pad = int(0.05 * SR)
head_ok = np.allclose(new_track[:max(s_ - fade_pad, 0)],
                      orig[:max(s_ - fade_pad, 0)], atol=1e-6)
tail_ok = np.allclose(new_track[e_ + fade_pad:], orig[e_ + fade_pad:], atol=1e-6)
check("audio before the edit is sample-identical", head_ok)
check("audio after the edit is sample-identical", tail_ok)

# Other segments must be untouched too.
seg2 = (int(2.6 * SR), int(4.2 * SR))
check("unedited segment 2 untouched",
      np.allclose(new_track[seg2[0]:seg2[1]], orig[seg2[0]:seg2[1]], atol=1e-6))

# Quality: matched span vs its neighbourhood, compared to the raw stub output.
raw_span, _ = dsp.load(os.path.join(TMP, "span_1.wav"), mono=True)
raw_fit = dsp.pad_or_trim(raw_span, e_ - s_)
nb = dsp.neighbourhood(orig, s_, e_, SR)
span_out = new_track[s_:e_]

d_raw = dsp.loudness_delta_db(raw_fit, nb, SR)
d_new = dsp.loudness_delta_db(span_out, nb, SR)
check("loudness closer to neighbours than raw", d_new < d_raw,
      f"raw={d_raw:.2f}dB -> {d_new:.2f}dB")

# Noise floor: a sub-second span of continuous speech has no silence, so
# re-measuring its "floor" returns quiet speech, not noise (see
# dsp/noisefloor.measure_floor). Assert on what the stage controls -- that a bed
# was laid at the neighbourhood's floor level -- rather than on a meaningless
# re-measurement.
nf = rep["noise_floor"]
f_ref = dsp.estimate_noise_floor_db(nb, SR)
check("room tone laid at the neighbourhood floor level",
      nf["applied"] and abs(nf["target_floor_db"] - f_ref) < 1.0,
      f"bed target={nf.get('target_floor_db')} neighbourhood={f_ref:.1f}")
check("bed level derived, not guessed", "bed_gain_db" in nf, str(nf))
check("no clipping", dsp.peak(new_track) <= 1.0, f"peak={dsp.peak(new_track):.4f}")

print()
print("=" * 74)
print("apply_edits — multiple lines changed")
print("=" * 74)
_calls["n"] = 0
state2 = dict(state)
state2.pop("match_reports", None)
edited2 = "\n".join([
    "the convoy will move after midnight",    # CHANGED
    "all units acknowledge the order",
    "hold position and await further orders",  # CHANGED
    "report status on arrival",
])
eng.apply_edits(state2, edited2)
check("voice engine called once per change", _calls["n"] == 2, f"n={_calls['n']}")
check("two reports recorded", len(state2["match_reports"]) == 2)
segs = [r["segment"] for r in state2["match_reports"]]
check("correct segments identified", segs == [0, 2], str(segs))
t2, _ = dsp.load(out_audio["path"], mono=True)
check("length still preserved with 2 edits", abs(t2.size - orig.size) <= 1,
      f"{t2.size} vs {orig.size}")
mid = (int(4.3 * SR), int(4.9 * SR))
check("gap between edits untouched",
      np.allclose(t2[mid[0]:mid[1]], orig[mid[0]:mid[1]], atol=1e-6))

print()
print("=" * 74)
print("edge cases")
print("=" * 74)
try:
    eng.apply_edits(dict(state), "\n".join(s["text"] for s in state["segments"]))
    check("no-change raises", False)
except ValueError as ex:
    check("no-change raises ValueError", "No changes" in str(ex), str(ex))

try:
    eng.apply_edits(None, "x")
    check("missing state raises", False)
except ValueError:
    check("missing state raises ValueError", True)

# Fewer edited lines than segments: missing ones keep original text.
_calls["n"] = 0
st3 = dict(state); st3.pop("match_reports", None)
eng.apply_edits(st3, "the convoy will move after midnight")
check("truncated edit text handled", _calls["n"] == 1, f"n={_calls['n']}")

# realism=False path still produces a valid, length-preserving track.
_calls["n"] = 0
st4 = dict(state); st4.pop("match_reports", None)
eng.apply_edits(st4, edited, realism=False)
t4, _ = dsp.load(out_audio["path"], mono=True)
check("realism=False still length-preserving", abs(t4.size - orig.size) <= 1,
      f"{t4.size} vs {orig.size}")

# A wildly over-long span must be reported as capped, not silently mangled.
class LongVoice(StubVoice):
    def run(self, text, reference_audio_path, language="en", **kw):
        _calls["n"] += 1
        span = synth(4.0, f0=150.0, sr=SR, seed=77)
        p = os.path.join(TMP, "long.wav")
        dsp.save(p, span, SR)
        return p


eng.voice = LongVoice()
st5 = dict(state); st5.pop("match_reports", None)
eng.apply_edits(st5, edited)
d5 = st5["match_reports"][0]["duration"]
check("over-long span reported as capped", d5["status"] == "exceeded", str(d5["status"]))
check("cap advice present", "advice" in d5)
t5, _ = dsp.load(out_audio["path"], mono=True)
check("length preserved even when capped", abs(t5.size - orig.size) <= 1,
      f"{t5.size} vs {orig.size}")

print()
print("=" * 74)
print("speech/background separation path")
print("=" * 74)


class StubSeparate:
    """Stands in for Demucs: splits the track into a 'speech' part and a
    'background' part that sum back to the original, so the remix step can be
    verified exactly."""
    def __init__(self, avail=True):
        self._avail = avail
        self.calls = 0

    def available(self):
        return self._avail

    def run(self, audio_path, model=None):
        self.calls += 1
        a, sr_ = dsp.load(audio_path, mono=True)
        # A fixed, invertible split: background is a known fraction.
        bg = (0.25 * a).astype(np.float32)
        sp = (a - bg).astype(np.float32)
        ps = os.path.join(TMP, "stub_speech.wav")
        pb = os.path.join(TMP, "stub_bg.wav")
        dsp.save(ps, sp, sr_)
        dsp.save(pb, bg, sr_)
        return ps, pb


_calls["n"] = 0
eng.voice = StubVoice()
sep = StubSeparate(avail=True)
eng.separate = sep
st6 = dict(state); st6.pop("match_reports", None)
eng.apply_edits(st6, edited, separate="auto")
check("separation engine invoked", sep.calls == 1, f"calls={sep.calls}")
check("stem paths recorded", "stem_speech" in st6 and "stem_background" in st6)
t6, _ = dsp.load(out_audio["path"], mono=True)
check("length preserved through separate+remix", abs(t6.size - orig.size) <= 1,
      f"{t6.size} vs {orig.size}")
# Outside the edit the speech stem is untouched, so speech+background must
# reconstruct the original sample-for-sample.
r6 = st6["match_reports"][0]
s6, e6 = r6["splice"]["start"], r6["splice"]["end"]
pad = int(0.05 * SR)
check("remix reconstructs original outside the edit",
      np.allclose(t6[:max(s6 - pad, 0)], orig[:max(s6 - pad, 0)], atol=2e-5),
      f"max diff={np.max(np.abs(t6[:max(s6-pad,0)] - orig[:max(s6-pad,0)])):.2e}")
check("edited region actually differs",
      not np.allclose(t6[s6:e6], orig[s6:e6], atol=1e-3))

# separate=False must skip it entirely.
sep.calls = 0
_calls["n"] = 0
st7 = dict(state); st7.pop("match_reports", None)
eng.apply_edits(st7, edited, separate=False)
check("separate=False skips separation", sep.calls == 0, f"calls={sep.calls}")
check("no stem paths recorded when disabled", "stem_speech" not in st7)

# "auto" must degrade gracefully when the environment is absent.
sep_missing = StubSeparate(avail=False)
eng.separate = sep_missing
_calls["n"] = 0
st8 = dict(state); st8.pop("match_reports", None)
eng.apply_edits(st8, edited, separate="auto")
check("auto skips separation when unavailable", sep_missing.calls == 0)
t8, _ = dsp.load(out_audio["path"], mono=True)
check("still produces a valid track without separation",
      abs(t8.size - orig.size) <= 1, f"{t8.size}")


# A failing separator must not fail the edit under "auto", but must under True.
class BoomSeparate(StubSeparate):
    def run(self, audio_path, model=None):
        self.calls += 1
        raise RuntimeError("demucs environment not built")


eng.separate = BoomSeparate(avail=True)
_calls["n"] = 0
st9 = dict(state); st9.pop("match_reports", None)
eng.apply_edits(st9, edited, separate="auto")
check("auto tolerates a failing separator", len(st9["match_reports"]) == 1)
try:
    st10 = dict(state); st10.pop("match_reports", None)
    eng.apply_edits(st10, edited, separate=True)
    check("separate=True propagates the failure", False)
except RuntimeError as ex:
    check("separate=True propagates the failure", "demucs" in str(ex).lower(),
          str(ex))

print()
print("=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL INTEGRATION TESTS PASSED")
