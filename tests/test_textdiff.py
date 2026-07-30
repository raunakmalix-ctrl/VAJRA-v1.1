# -*- coding: utf-8 -*-
"""Tests for word-level edit resolution and cloning-reference selection."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import textdiff as td

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def seg_with_words(text, start=0.0, wdur=0.30, gap=0.05, gaps=None):
    """Build a segment whose words are evenly spaced, with optional custom gaps.

    gaps: dict {index: gap_before_this_word_in_sec}
    """
    toks = text.split()
    words, t = [], start
    for i, w in enumerate(toks):
        if i > 0:
            t += (gaps or {}).get(i, gap)
        words.append({"word": w, "start": round(t, 4), "end": round(t + wdur, 4)})
        t += wdur
    return {"start": start, "end": round(t, 4), "text": text, "words": words}


print("=" * 74)
print("tokenize / normalise")
print("=" * 74)
check("splits on punctuation",
      td.tokenize("the convoy, will move: at first light!") ==
      ["the", "convoy", "will", "move", "at", "first", "light"])
check("keeps apostrophes", td.tokenize("don't stop") == ["don't", "stop"])
check("handles empty", td.tokenize("") == [] and td.tokenize(None) == [])
check("keeps devanagari", td.tokenize("काफिला रात") == ["काफिला", "रात"])
check("normalise casefolds", td.normalise("The") == td.normalise("the"))

print()
print("=" * 74)
print("changed_ranges")
print("=" * 74)
o = ["the", "convoy", "will", "move", "at", "first", "light"]
n1 = ["the", "convoy", "will", "move", "after", "midnight"]
r = td.changed_ranges(o, n1)
check("tail replacement localised", r == [(4, 7, ["after", "midnight"])], str(r))
r2 = td.changed_ranges(o, o)
check("identical -> no ranges", r2 == [], str(r2))
r3 = td.changed_ranges(o, ["the", "convoy", "will", "not", "move", "at", "first", "light"])
check("insertion detected", len(r3) == 1 and r3[0][0] == r3[0][1], str(r3))
r4 = td.changed_ranges(o, ["convoy", "will", "move", "at", "first", "light"])
check("deletion detected", len(r4) == 1 and r4[0][2] == [], str(r4))
r5 = td.changed_ranges(o, ["a", "convoy", "will", "move", "at", "last", "light"])
check("two separate edits found", len(r5) == 2, str(r5))
check("case-only change is not an edit",
      td.changed_ranges(o, ["The", "Convoy", "will", "move", "at", "first", "light"]) == [])

print()
print("=" * 74)
print("words_from_segment")
print("=" * 74)
s = seg_with_words("alpha bravo charlie")
check("dict words parsed", len(td.words_from_segment(s)) == 3)
check("absent words -> empty",
      td.words_from_segment({"start": 0, "end": 1, "text": "x"}) == [])


class W:
    def __init__(self, w, a, b):
        self.word, self.start, self.end = w, a, b


check("attribute-style words parsed",
      len(td.words_from_segment({"words": [W("a", 0, 1), W("b", 1, 2)]})) == 2)
check("drops words missing timings",
      len(td.words_from_segment({"words": [{"word": "a"}, {"word": "b", "start": 0, "end": 1}]})) == 1)

print()
print("=" * 74)
print("resolve_edit_spans — surgical behaviour")
print("=" * 74)
# A long line with a clear pause before the final phrase.
seg = seg_with_words("the convoy will move at first light", start=1.0,
                     wdur=0.30, gap=0.04, gaps={4: 0.40})
spans = td.resolve_edit_spans(seg, "the convoy will move after midnight")
check("one span produced", len(spans) == 1, str(len(spans)))
sp = spans[0]
check("word granularity used", sp["granularity"] == "word", sp["granularity"])
seg_dur = seg["end"] - seg["start"]
span_dur = sp["t_end"] - sp["t_start"]
check("span is much shorter than the line", span_dur < 0.55 * seg_dur,
      f"{span_dur:.2f}s of {seg_dur:.2f}s")
check("span starts at the pause boundary", abs(sp["t_start"] - seg["words"][4]["start"]) < 1e-6,
      f"{sp['t_start']} vs {seg['words'][4]['start']}")
check("replacement text carried", sp["text"] == "after midnight", sp["text"])
check("original text carried", sp["orig_text"] == "at first light", sp["orig_text"])
print("  ", td.summarise_spans(spans, seg))

# No internal pause: expansion should widen to the whole line (natural prosody).
seg_tight = seg_with_words("the convoy will move at first light", gap=0.02)
sp_t = td.resolve_edit_spans(seg_tight, "the convoy will move after midnight")
check("tight line expands outward", sp_t[0]["expanded"] is True, str(sp_t[0]["expanded"]))

# expand_to_pauses=False gives the raw changed words (the infill-model path).
sp_raw = td.resolve_edit_spans(seg, "the convoy will move after midnight",
                               expand_to_pauses=False)
check("unexpanded span is tighter still",
      (sp_raw[0]["t_end"] - sp_raw[0]["t_start"]) <= span_dur + 1e-9,
      f"{sp_raw[0]['t_end'] - sp_raw[0]['t_start']:.2f}s")
check("unexpanded not marked expanded", sp_raw[0]["expanded"] is False)

print()
print("=" * 74)
print("resolve_edit_spans — fallbacks and edge cases")
print("=" * 74)
check("unchanged text -> no spans",
      td.resolve_edit_spans(seg, "the convoy will move at first light") == [])
check("empty edit -> no spans", td.resolve_edit_spans(seg, "") == [])
check("punctuation-only change -> no spans",
      td.resolve_edit_spans(seg, "The convoy will move, at first light.") == [])

nw = {"start": 0.0, "end": 3.0, "text": "alpha bravo charlie"}
sp_nw = td.resolve_edit_spans(nw, "alpha bravo delta")
check("no word timings -> segment granularity",
      len(sp_nw) == 1 and sp_nw[0]["granularity"] == "segment", str(sp_nw))
check("segment fallback covers the whole line",
      sp_nw[0]["t_start"] == 0.0 and sp_nw[0]["t_end"] == 3.0)

# A near-total rewrite should be one segment-wide span, not many fragments.
sp_rw = td.resolve_edit_spans(seg, "hold position and await further orders now")
check("wholesale rewrite -> segment granularity",
      sp_rw[0]["granularity"] == "segment", sp_rw[0]["granularity"])

# Two distant edits in a long line stay separate.
long_seg = seg_with_words(
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima",
    wdur=0.28, gap=0.05, gaps={4: 0.6, 8: 0.6})
sp_two = td.resolve_edit_spans(
    long_seg, "alpha zulu charlie delta echo foxtrot golf hotel india juliet kilo yankee")
check("distant edits kept separate", len(sp_two) == 2, f"{len(sp_two)} spans")
if len(sp_two) == 2:
    check("spans do not overlap", sp_two[0]["t_end"] <= sp_two[1]["t_start"],
          f"{sp_two[0]['t_end']} <= {sp_two[1]['t_start']}")

# Adjacent edits should merge into one span rather than two seam pairs.
adj = seg_with_words("one two three four five six", wdur=0.25, gap=0.03)
sp_adj = td.resolve_edit_spans(adj, "one nine three seven five six")
check("adjacent edits merged", len(sp_adj) == 1, f"{len(sp_adj)} spans")

# Spans must never leave the segment's own bounds.
for name, sgi, txt in [("mid", seg, "the convoy will move after midnight"),
                       ("tight", seg_tight, "the convoy will move after midnight")]:
    for s_ in td.resolve_edit_spans(sgi, txt):
        ok = (s_["t_start"] >= sgi["start"] - 1e-6
              and s_["t_end"] <= sgi["end"] + 1e-6
              and s_["t_end"] > s_["t_start"])
        check(f"span within segment bounds ({name})", ok,
              f"{s_['t_start']:.3f}-{s_['t_end']:.3f} in {sgi['start']}-{sgi['end']}")

# Insertion at the very start.
sp_ins = td.resolve_edit_spans(seg, "urgently the convoy will move at first light")
check("leading insertion handled", len(sp_ins) >= 1 and sp_ins[0]["t_end"] > sp_ins[0]["t_start"],
      str([(round(s['t_start'],2), round(s['t_end'],2)) for s in sp_ins]))
# Deletion of the tail.
sp_del = td.resolve_edit_spans(seg, "the convoy will move")
check("tail deletion handled", len(sp_del) >= 1, str(len(sp_del)))

print()
print("=" * 74)
print("dsp.reference — cloning-reference selection")
print("=" * 74)
from dsp import reference as ref
import dsp

SR = 24000


def speech(sec, sr=SR, seed=0, f0=120.0, level=0.3):
    r = np.random.RandomState(seed)
    n = int(sec * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for k in range(1, 22):
        x += (1.0 / k ** 1.35) * np.sin(2 * np.pi * f0 * k * t + r.uniform(0, 6.28))
    env = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t - np.pi / 2))
    env = np.clip(env - 0.10, 0, None) / 0.90
    x *= env
    x /= (np.max(np.abs(x)) + 1e-9)
    return (level * x).astype(np.float32)


def noise(sec, db, sr=SR, seed=1):
    r = np.random.RandomState(seed)
    return (10 ** (db / 20.0) * r.randn(int(sec * sr))).astype(np.float32)


# Track: 8s of loud noise, then 20s of clean speech, then 8s of clipped speech.
bad_noise = noise(8.0, -14.0, seed=2)
clean = speech(20.0, seed=3) + noise(20.0, -58.0, seed=4)
clipped = np.clip(speech(8.0, seed=5, level=0.9) * 3.0, -1.0, 1.0).astype(np.float32)
track = np.concatenate([bad_noise, clean, clipped]).astype(np.float32)

clip, info = ref.pick_reference(track, SR, target_sec=10.0)
print("  ", ref.describe(info))
start = info["start_sec"]
check("avoids the noisy opening entirely", start >= 8.0, f"start={start}s")
check("floor is stationary in the chosen window",
      info["floor_spread_db"] < 12.0, f"{info['floor_spread_db']}dB spread")
check("avoids the clipped tail", info["end_sec"] <= 28.5, f"end={info['end_sec']}s")
check("returned clip has requested length",
      abs(clip.size / SR - 10.0) < 0.05, f"{clip.size/SR:.2f}s")
check("clip is clean (low clipping)", info["clipped_fraction"] < 1e-4,
      str(info["clipped_fraction"]))
check("good SNR reported", info["snr_db"] > 15.0, f"{info['snr_db']}dB")

# Scoring must rank clean speech above noise and above clipped speech.
s_clean, _ = ref.score_window(clean[:int(10 * SR)], SR)
s_noise, _ = ref.score_window(bad_noise, SR)
s_clip, _ = ref.score_window(clipped, SR)
check("clean scores above noise", s_clean > s_noise, f"{s_clean:.3f} > {s_noise:.3f}")
check("clean scores above clipped", s_clean > s_clip, f"{s_clean:.3f} > {s_clip:.3f}")

# Exclusion: the region being replaced must not become the reference.
excl = [(8 * SR, 20 * SR)]
clip2, info2 = ref.pick_reference(track, SR, target_sec=8.0, exclude_ranges=excl)
overlaps = (info2["start_sec"] < 20.0 and info2["end_sec"] > 8.0)
check("excluded range avoided", not overlaps,
      f"{info2['start_sec']}-{info2['end_sec']} vs excluded 8-20s")

# speech_ranges keeps it out of a long non-speech stretch.
clip3, info3 = ref.pick_reference(track, SR, target_sec=8.0,
                                  speech_ranges=[(8.0, 28.0)])
check("respects speech_ranges", info3["start_sec"] >= 7.5 and info3["end_sec"] <= 28.5,
      f"{info3['start_sec']}-{info3['end_sec']}")

# Short track: return everything rather than failing.
tiny = speech(3.0, seed=9)
clip4, info4 = ref.pick_reference(tiny, SR, target_sec=18.0, min_sec=6.0)
check("short track returns whole track", clip4.size == tiny.size, str(info4["selected"]))

# Everything excluded: still returns usable audio.
clip5, info5 = ref.pick_reference(track, SR, target_sec=8.0,
                                  exclude_ranges=[(0, track.size)])
check("fully-excluded track still returns audio", clip5.size > 0, str(info5["selected"]))

print()
print("=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL TEXTDIFF / REFERENCE TESTS PASSED")
