# -*- coding: utf-8 -*-
"""A replacement must never be cut off mid-phrase.

The reported failure: an edit sounded like the new wording followed by the old
words. Cause: a word-level slot is only as long as the words it replaces, so a
replacement with more syllables did not fit. fit_duration capped the stretch,
crossfade_splice trimmed the rest, and the ORIGINAL recording resumed at the end
of the slot -- so the listener heard both.

Truncation is the worst of the available options and must never be the outcome.
The span grows into the neighbouring silence first (bounded by the adjacent
words, which must not be swallowed), and if that is still not enough the whole
line is re-voiced instead. Both are reported.
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The engine imports core.device, which imports torch. Nothing under test here
# touches the GPU, so a stub keeps this suite CPU-only like the rest.
if "torch" not in sys.modules:
    _t = types.ModuleType("torch")
    _t.cuda = types.SimpleNamespace(is_available=lambda: False,
                                    empty_cache=lambda: None,
                                    ipc_collect=lambda: None)
    _t.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = _t

from core import textdiff  # noqa: E402
from engines.transcript_engine import _widen_to_fit, _ESCALATE_SEC  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


SR = 24000


def seg_with_gaps(word_durs, gaps, start=0.0):
    """Build a segment whose words are separated by the given gaps."""
    words, t = [], start
    for i, d in enumerate(word_durs):
        words.append({"word": f"w{i}", "start": round(t, 4),
                      "end": round(t + d, 4)})
        t += d + (gaps[i] if i < len(gaps) else 0.0)
    return {"start": start, "end": round(t, 4),
            "text": " ".join(w["word"] for w in words), "words": words}


print("=" * 74)
print("spans carry the slack either side")
print("=" * 74)
# 5 words, a generous 0.40s pause after word 2 and a small one before it.
seg = seg_with_gaps([0.30, 0.30, 0.30, 0.30, 0.30],
                    [0.05, 0.06, 0.40, 0.05])
spans = textdiff.resolve_edit_spans(seg, "w0 w1 REPLACED w3 w4",
                                    expand_to_pauses=False)
check("one span resolved", len(spans) == 1, str(len(spans)))
sp = spans[0]
check("span carries slack_start", "slack_start" in sp)
check("span carries slack_end", "slack_end" in sp)
check("slack_end reaches the following pause, not into the next word",
      abs(sp["slack_end"] - seg["words"][3]["start"]) < 1e-6,
      f"{sp['slack_end']} vs next word at {seg['words'][3]['start']}")
check("slack_start stops at the previous word's end",
      abs(sp["slack_start"] - seg["words"][1]["end"]) < 1e-6,
      f"{sp['slack_start']} vs prev word ends {seg['words'][1]['end']}")
check("slack is wider than the span itself",
      (sp["slack_end"] - sp["slack_start"]) > (sp["t_end"] - sp["t_start"]))

whole = textdiff.resolve_edit_spans({"start": 0.0, "end": 2.0,
                                     "text": "a b", "words": []}, "c d")
check("segment-level spans carry slack too",
      "slack_start" in whole[0] and "slack_end" in whole[0])

print()
print("=" * 74)
print("the slot grows into silence rather than truncating")
print("=" * 74)
n = int(10 * SR)
start, end = int(sp["t_start"] * SR), int(sp["t_end"] * SR)
slot = end - start
meta = dict(sp)

# Needs a bit more than the slot: the trailing pause should absorb it.
need = int(slot * 1.35)
s2, e2, info = _widen_to_fit(need, start, end, SR, meta, 0.15, n)
check("slot grew", (e2 - s2) > slot, f"{slot} -> {e2 - s2} samples")
check("growth came from the trailing pause first", s2 == start and e2 > end)
check("nothing is left short", info["short_sec"] == 0.0, str(info))
check("growth stayed inside the slack",
      e2 <= int(round(meta["slack_end"] * SR)) + 1,
      f"{e2} vs limit {int(round(meta['slack_end'] * SR))}")
check("a neighbouring word is never swallowed",
      e2 <= int(seg["words"][3]["start"] * SR) + 1)

# Fits already: must not move at all.
s3, e3, info3 = _widen_to_fit(int(slot * 0.5), start, end, SR, meta, 0.15, n)
check("a replacement that already fits is left alone",
      (s3, e3) == (start, end) and info3["widened_sec"] == 0.0)

# Far too long: slack cannot cover it, and the shortfall must be REPORTED,
# not silently absorbed -- that is what triggers the whole-line re-voice.
s4, e4, info4 = _widen_to_fit(int(slot * 4.0), start, end, SR, meta, 0.15, n)
check("an impossible fit reports a shortfall",
      info4["short_sec"] > _ESCALATE_SEC, str(info4))
check("it still grew as far as it legitimately could",
      e4 == int(round(meta["slack_end"] * SR)),
      f"{e4} vs {int(round(meta['slack_end'] * SR))}")
check("and still did not cross into the next word",
      e4 <= int(seg["words"][3]["start"] * SR) + 1)

print()
print("=" * 74)
print("no slack available")
print("=" * 74)
tight = seg_with_gaps([0.30, 0.30, 0.30], [0.0, 0.0])
tspans = textdiff.resolve_edit_spans(tight, "w0 REPLACED w2",
                                     expand_to_pauses=False)
tsp = tspans[0]
ts, te = int(tsp["t_start"] * SR), int(tsp["t_end"] * SR)
s5, e5, info5 = _widen_to_fit(int((te - ts) * 2.0), ts, te, SR, dict(tsp),
                              0.15, n)
check("with no gaps the span cannot grow",
      (s5, e5) == (ts, te), f"{(s5, e5)} vs {(ts, te)}")
check("and the shortfall is reported so the caller escalates",
      info5["short_sec"] > _ESCALATE_SEC, str(info5))

print()
print("=" * 74)
print("the escalation threshold is meaningful")
print("=" * 74)
check("threshold is small enough to catch a cut word",
      _ESCALATE_SEC <= 0.20, str(_ESCALATE_SEC))
check("threshold is large enough not to fire on rounding",
      _ESCALATE_SEC >= 0.05, str(_ESCALATE_SEC))

print("=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL FIT-REGRESSION TESTS PASSED")
