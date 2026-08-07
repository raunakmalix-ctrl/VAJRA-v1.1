# -*- coding: utf-8 -*-
"""Regressions found by running on a GPU, pinned so they cannot come back.

Every check here corresponds to something that passed the offline suite and
still failed in a real session. They are cheap; the sessions that found them
were not.
"""
import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
if "torch" not in sys.modules:
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: False,
                                   empty_cache=lambda: None, ipc_collect=lambda: None)
    t.Tensor = type("Tensor", (), {}); sys.modules["torch"] = t
FAILS = []
def check(n, c, d=""):
    print(("  PASS  " if c else "  FAIL  ") + n + (f"   {d}" if d else ""))
    if not c: FAILS.append(n)

from core.textnorm import expand_digits
print("=" * 66); print("digit expansion (the Hindi crash)"); print("=" * 66)
check("hindi digits become words, not a crash",
      "2024" not in expand_digits("साल 2024 में", "hi"))
check("every digit is spoken", expand_digits("कमरा 302", "hi").count(" ") >= 3)
check("text without digits is returned untouched",
      expand_digits("कोई अंक नहीं", "hi") == "कोई अंक नहीं")
check("empty text is safe", expand_digits("", "hi") == "")
check("an unknown language still yields speakable words",
      any(c.isalpha() for c in expand_digits("room 7", "zz")))
check("no digit survives for any supported language",
      all(not any(ch.isdigit() for ch in expand_digits("a 1 b 23 c 456", L))
          for L in ("hi", "en", "ar", "zz")))

print(); print("=" * 66); print("viitor readiness"); print("=" * 66)
from engines import viitor_engine as V
check("every backing service is checked, not just the gateway",
      set(V.VIITOR_SERVICE_PORTS) >= {"encoder", "llm", "decoder"},
      str(sorted(V.VIITOR_SERVICE_PORTS)))
check("nothing running means every service reports down",
      len(V._services_down(timeout=0.3)) == len(V.VIITOR_SERVICE_PORTS))
check("health is false when the group is down", V._health(timeout=0.4) is False)
src = open(r"D:\VAJRA-v1.1\engines\viitor_engine.py", encoding="utf-8").read()
check("a gateway-only check is no longer treated as ready",
      "return not _services_down()" in src)
check("the timeout message names the service that failed",
      "never started" in src)

print(); print("=" * 66); print("tier-D message honesty"); print("=" * 66)
te = open(r"D:\VAJRA-v1.1\engines\transcript_engine.py", encoding="utf-8").read()
check("a contradictory 'no word timings' claim is replaced",
      "Word timings exist, but every edited line changed too" in te)
check("and it says what to do about it", "Change fewer words per line" in te)
print("=" * 66)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):"); [print("   -", f) for f in FAILS]; sys.exit(1)
print("ALL FIX CHECKS PASSED")
