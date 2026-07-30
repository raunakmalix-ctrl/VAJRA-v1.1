# -*- coding: utf-8 -*-
"""Tests for the engine router and the objective scorecard."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import router as R

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


print("=" * 74)
print("router — language normalisation")
print("=" * 74)
check("plain code passes", R.normalise_lang("en") == "en")
check("region stripped", R.normalise_lang("en-US") == "en")
check("case folded", R.normalise_lang("FR") == "fr")
check("underscore form handled", R.normalise_lang("pt_BR") == "pt")
check("chinese folded to zh-cn", R.normalise_lang("zh") == "zh-cn"
      and R.normalise_lang("zh-CN") == "zh-cn"
      and R.normalise_lang("zh_TW") == "zh-cn")
check("empty defaults to en", R.normalise_lang("") == "en"
      and R.normalise_lang(None) == "en")

print()
print("=" * 74)
print("router — tier resolution")
print("=" * 74)
AVAIL = {"xtts": True}

d = R.resolve("en", has_word_timings=True, available=AVAIL)
check("english resolves", d["engine"] == "xtts", d["engine"])
check("english gets tier C today", d["tier"] == "C", d["tier"])
# The honest part: English *could* do better, and the router must say so rather
# than let the operator assume C is the ceiling.
check("english ceiling reported as A", d["best_possible_tier"] == "A",
      d["best_possible_tier"])
check("gap explained in warnings",
      any("ViiTor" in w for w in d["warnings"]), str(d["warnings"]))
print("  ", R.describe(d))

d = R.resolve("es", has_word_timings=True, available=AVAIL)
check("spanish supported by xtts", d["engine"] == "xtts" and d["tier"] == "C",
      f"{d['engine']}/{d['tier']}")
check("spanish ceiling is B (NC infill)", d["best_possible_tier"] == "B",
      d["best_possible_tier"])
check("spanish gap mentions the NC engine",
      any("VoiceCraft" in w for w in d["warnings"]), str(d["warnings"]))

d = R.resolve("hi", has_word_timings=True, available=AVAIL)
check("hindi supported", d["engine"] == "xtts" and d["tier"] == "C")
check("hindi ceiling is C (no infill engine covers it)",
      d["best_possible_tier"] == "C", d["best_possible_tier"])
check("no misleading upgrade warning for hindi",
      not any("would give a better result" in w for w in d["warnings"]),
      str(d["warnings"]))

# Missing word timings force tier D regardless of engine capability.
d = R.resolve("en", has_word_timings=False, available=AVAIL)
check("no word timings forces tier D", d["tier"] == "D", d["tier"])
check("tier D uses segment granularity", d["granularity"] == "segment")
check("tier D warns about precision",
      any("word-level" in w for w in d["warnings"]), str(d["warnings"]))

# Unsupported language: fall back, but say so.
d = R.resolve("ta", has_word_timings=True, available=AVAIL)
check("unsupported language still resolves", d["engine"] == "xtts")
check("unsupported language is tier D", d["tier"] == "D", d["tier"])
check("unsupported language synthesises in english",
      d["synthesis_language"] == "en", d["synthesis_language"])
check("unsupported language warns it won't sound native",
      any("native" in w for w in d["warnings"]), str(d["warnings"]))

# allow_nc changes what is considered, but not what is installed.
d_off = R.resolve("ko", has_word_timings=True, allow_nc=False, available=AVAIL)
d_on = R.resolve("ko", has_word_timings=True, allow_nc=True, available=AVAIL)
check("NC engine not selected while uninstalled even when allowed",
      d_on["engine"] == "xtts", d_on["engine"])
# An engine that is BOTH uninstalled and non-commercial is blocked twice over.
# Naming only the licence would imply that enabling allow_nc fixes it, and
# naming only the integration gap would send someone to wire up an engine the
# licence setting would then still veto. Both have to be said.
check("uninstalled NC engine names the integration gap",
      any("not integrated" in w for w in d_off["warnings"]),
      str(d_off["warnings"]))
check("uninstalled NC engine also names the licence",
      any("non-commercial" in w for w in d_off["warnings"]),
      str(d_off["warnings"]))
check("both blockers appear in one message, not two contradictory ones",
      sum(1 for w in d_off["warnings"] if "VoiceCraft" in w) == 1,
      str(d_off["warnings"]))
# Once such an engine IS integrated, the licence becomes the real blocker and
# must be reported as such.
_saved = R.ENGINES["voicecraft_x"]["integrated"]
try:
    R.ENGINES["voicecraft_x"]["integrated"] = True
    d_lic = R.resolve("ko", has_word_timings=True, allow_nc=False,
                      available=AVAIL)
    check("integrated NC engine reports the licence as a blocker",
          any("non-commercial" in w for w in d_lic["warnings"]),
          str(d_lic["warnings"]))
    # It is ALSO not built here. Reporting only one would send the operator to
    # do work that the other blocker would then still veto.
    check("every blocker is named, not just the first",
          any("not built" in w and "non-commercial" in w
              for w in d_lic["warnings"]), str(d_lic["warnings"]))
    d_built = R.resolve("ko", has_word_timings=True, allow_nc=False,
                        available={"xtts": True, "voicecraft_x": True})
    check("a resolved blocker stops being reported",
          not any("not built" in w for w in d_built["warnings"]),
          str(d_built["warnings"]))
    d_ok = R.resolve("ko", has_word_timings=True, allow_nc=True,
                     available={"xtts": True, "voicecraft_x": True})
    check("allow_nc plus availability selects the better tier",
          d_ok["tier"] == "B" and d_ok["engine"] == "voicecraft_x",
          f"{d_ok['engine']}/{d_ok['tier']}")
    check("infill granularity reported for tier B",
          d_ok["granularity"] == "infill", d_ok["granularity"])
finally:
    R.ENGINES["voicecraft_x"]["integrated"] = _saved

# An unavailable engine must not be selected.
d = R.resolve("en", has_word_timings=True, available={"xtts": False})
check("unavailable engine falls through to tier D", d["tier"] == "D", d["tier"])
check("resolve never returns None", R.resolve(None) is not None)

print()
print("=" * 74)
print("router — candidates and capability table")
print("=" * 74)
c = R.candidates("en", integrated_only=False, allow_nc=True)
check("candidates sorted best tier first",
      [s["tier"] for _, s in c] == sorted([s["tier"] for _, s in c]), str(c))
check("integrated_only filters uninstalled",
      all(s.get("integrated") for _, s in R.candidates("en")), "found uninstalled")
check("NC excluded unless allowed",
      not any(s["non_commercial"] for _, s in R.candidates("ko", allow_nc=False,
                                                          integrated_only=False)))
check("zh-cn matches an engine registered as plain zh",
      any(k == "voicecraft_x" for k, _ in
          R.candidates("zh-cn", integrated_only=False, allow_nc=True)),
      "zh mapping failed")
check("the mapping is in _engine_supports, not the caller",
      R._engine_supports({"languages": {"zh"}}, "zh-cn")
      and not R._engine_supports({"languages": {"zh"}}, "ja"))

tbl = R.capability_table(available={"xtts": True})
check("capability table covers many languages", len(tbl) >= 17, str(len(tbl)))
check("table reports both current and possible tiers",
      all({"tier", "best_possible_tier"} <= set(v) for v in tbl.values()))
check("english shows an upgrade path when tier A is not built",
      tbl["en"]["tier"] == "C" and tbl["en"]["best_possible_tier"] == "A",
      str(tbl["en"]))
check("and no upgrade path once it is built",
      R.capability_table(
          available={"xtts": True, "viitor_nar": True})["en"]["tier"] == "A")

print()
print("=" * 74)
print("scorecard — natural seam baseline")
print("=" * 74)
import evaluation as EV
import dsp

SR = 24000

# The realism chain injects a randomly assembled room-tone bed on purpose, so
# pin the global RNG to keep these measurements reproducible across runs.
np.random.seed(20260730)


def speech(sec, sr=SR, seed=0, f0=120.0, level=0.3):
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
    return (level * x + 10 ** (-52 / 20.0) * r.randn(n)).astype(np.float32)


track = speech(12.0, seed=3)
nat = EV.natural_seam_distribution(track, SR)
check("baseline found genuine boundaries", nat.size > 5, f"{nat.size} seams")
check("baseline values are non-negative dB", np.all(nat >= 0.0))
check("silence yields no baseline",
      EV.natural_seam_distribution(np.zeros(SR, np.float32), SR).size == 0)
check("too-short input safe",
      EV.natural_seam_distribution(np.zeros(100, np.float32), SR).size == 0)
# Excluded regions must not contribute boundaries.
nat_ex = EV.natural_seam_distribution(track, SR, exclude_ranges=[(0, track.size)])
check("full exclusion empties the baseline", nat_ex.size == 0, f"{nat_ex.size}")

print()
print("=" * 74)
print("scorecard — seam percentile")
print("=" * 74)
dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.float32)
check("smoother than all -> 0", EV.seam_percentile(0.5, dist) == 0.0)
check("more abrupt than all -> 1", EV.seam_percentile(9.0, dist) == 1.0)
check("midpoint -> 0.6", abs(EV.seam_percentile(3.0, dist) - 0.6) < 1e-6,
      str(EV.seam_percentile(3.0, dist)))
check("no baseline -> None", EV.seam_percentile(1.0, np.zeros(0)) is None)

print()
print("=" * 74)
print("scorecard — scoring a good vs a bad edit")
print("=" * 74)
# A GOOD edit: matched via the dsp chain.
good = track.copy()
s0, e0 = int(4.0 * SR), int(4.8 * SR)
raw_span = dsp.apply_gain(speech(0.8, seed=44, f0=150.0), +5.0)
good, rep_good = dsp.match_and_splice(good, raw_span, s0, e0, SR)
gs, ge = rep_good["splice"]["start"], rep_good["splice"]["end"]
card_good = EV.score_all(good, SR, [(gs, ge)], [rep_good])
print(EV.summarise(card_good))
check("good edit produces one span score", card_good["n_spans"] == 1)
check("good edit has a seam verdict",
      card_good["spans"][0]["seam"]["verdict"] in
      ("inaudible", "unlikely to be noticed", "possibly audible", "likely audible"),
      card_good["spans"][0]["seam"]["verdict"])
check("good edit loudness is matched",
      card_good["spans"][0]["loudness"]["verdict"] in ("good", "acceptable"),
      str(card_good["spans"][0]["loudness"]))
check("baseline used for comparison", card_good["n_natural_seams"] > 3,
      str(card_good["n_natural_seams"]))

# A BAD edit: raw span butt-joined with no matching at all.
bad = track.copy()
bad[s0:e0] = dsp.pad_or_trim(raw_span, e0 - s0)
card_bad = EV.score_all(bad, SR, [(s0, e0)], [None])
print(EV.summarise(card_bad))
check("bad edit is flagged for review", card_bad["overall"] == "review",
      card_bad["overall"])
gseam = card_good["spans"][0]["seam"]["worst_db"]
bseam = card_bad["spans"][0]["seam"]["worst_db"]
check("matched edit has a smoother seam than the raw splice", gseam < bseam,
      f"matched={gseam}dB raw={bseam}dB")
# Percentiles are only comparable against the SAME baseline. Each score_all
# call above derives its own from its own edited track, so compare the two
# edits against one baseline taken from the untouched original instead.
nat_shared = EV.natural_seam_distribution(track, SR, exclude_ranges=[(s0, e0)])
gp = EV.score_edit(good, SR, (gs, ge), report=rep_good,
                   natural=nat_shared)["seam"]["percentile"]
bp = EV.score_edit(bad, SR, (s0, e0), natural=nat_shared)["seam"]["percentile"]
# The percentile is a RANK, so it can only separate two seams that fall on
# opposite sides of some genuine boundary. It must never invert, and it must
# separate an obviously worse seam -- but two values inside one gap of the
# distribution tying is correct behaviour, not a miss.
check("percentile never inverts the dB ordering", gp <= bp,
      f"matched={gp}({gseam}dB) raw={bp}({bseam}dB)")
awful = track.copy()
awful[s0:e0] = dsp.pad_or_trim(dsp.apply_gain(raw_span, +14.0), e0 - s0)
ap = EV.score_edit(awful, SR, (s0, e0), natural=nat_shared)
check("a grossly mismatched splice ranks worse",
      ap["seam"]["percentile"] > bp,
      f"awful={ap['seam']['percentile']}({ap['seam']['worst_db']}dB) raw={bp}")
check("a grossly mismatched splice is called out",
      ap["seam"]["verdict"] in ("possibly audible", "likely audible"),
      ap["seam"]["verdict"])

print()
print("=" * 74)
print("scorecard — reporting discipline")
print("=" * 74)
check("pending metrics are named, not faked", len(EV.pending_metrics()) >= 4,
      str([p["metric"] for p in EV.pending_metrics()]))
check("every pending metric says what it needs",
      all({"metric", "needs", "why"} <= set(p) for p in EV.pending_metrics()))
check("scorecard carries the pending list", "pending" in card_good)
check("summary mentions what was not measured",
      "not measured" in EV.summarise(card_good))

# A duration failure must surface as a compromised verdict.
rep_cap = dict(rep_good)
rep_cap["duration"] = {"status": "exceeded", "shortfall_sec": 0.4,
                       "advice": "reword"}
card_cap = EV.score_all(good, SR, [(gs, ge)], [rep_cap])
check("capped duration is flagged",
      card_cap["spans"][0]["duration"]["verdict"] == "timing compromised",
      str(card_cap["spans"][0]["duration"]))
check("capped duration drives the overall verdict",
      card_cap["overall"] == "review", card_cap["overall"])

# Short spans must report the floor as unmeasurable rather than guess.
tiny = EV.score_edit(good, SR, (gs, gs + int(0.2 * SR)))
check("short span floor reported as unmeasurable",
      tiny["noise_floor"]["measurable"] is False
      or "not measurable" in str(tiny["noise_floor"]["verdict"]),
      str(tiny["noise_floor"]))

# Degenerate inputs
check("empty span does not crash",
      EV.score_edit(good, SR, (gs, gs)) is not None)
check("no spans gives a pass with zero spans",
      EV.score_all(good, SR, [], [])["n_spans"] == 0)

print()
print("=" * 74)
print("XTTS language exposure")
print("=" * 74)
from engines.voice_engine import SUPPORTED_LANGUAGES as SL
check("17 languages exposed", len(SL) == 17, str(len(SL)))
for name, code in [("English", "en"), ("Hindi", "hi"), ("Spanish", "es"),
                   ("Chinese", "zh-cn"), ("Japanese", "ja"), ("Arabic", "ar")]:
    check(f"{name} exposed", SL.get(name) == code, str(SL.get(name)))
check("every exposed code is routable",
      all(R.candidates(c) for c in SL.values()),
      str([c for c in SL.values() if not R.candidates(c)]))

print()
print("=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL ROUTER / SCORECARD TESTS PASSED")
