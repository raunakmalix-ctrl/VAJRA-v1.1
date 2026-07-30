# -*- coding: utf-8 -*-
"""ViiTorVoice-NAR integration: routing, availability, and the HTTP contract.

No service runs here, so what is checked is everything that can be wrong without
one: that the router only promises tier A when all three prerequisites are
actually present, that an operator's tier request is honoured or explained, that
the multipart body sent to /v1/text-local-edit matches the documented field
names, and that a missing install says WHICH part is missing.

The remaining risk -- whether the five-service group starts and returns good
audio on a GPU -- cannot be reached from here and is called out in
tests/README.md rather than papered over.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "torch" not in sys.modules:
    _t = types.ModuleType("torch")
    _t.cuda = types.SimpleNamespace(is_available=lambda: False,
                                    empty_cache=lambda: None,
                                    ipc_collect=lambda: None)
    _t.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = _t

from core import router as R  # noqa: E402
from engines import viitor_engine as VE  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


print("=" * 74)
print("router — tier A is English-only and gated on being built")
print("=" * 74)
BOTH = {"xtts": True, "viitor_nar": True}
XTTS_ONLY = {"xtts": True}

d = R.resolve("en", available=BOTH)
check("english routes to tier A when built",
      d["tier"] == "A" and d["engine"] == "viitor_nar",
      f"{d['tier']}/{d['engine']}")
check("tier A is infill granularity", d["granularity"] == "infill")
check("no warnings when already at the ceiling", not d["warnings"],
      str(d["warnings"]))

d = R.resolve("en", available=XTTS_ONLY)
check("english falls to tier C when not built", d["tier"] == "C", d["tier"])
check("and says it is a BUILD problem, not a language one",
      any("not built" in w for w in d["warnings"]), str(d["warnings"]))

# Chinese was deliberately dropped. It must fall back like any other
# unsupported language rather than silently routing to an unvalidated path.
d = R.resolve("zh", available=BOTH)
check("chinese does NOT reach tier A", d["tier"] != "A", d["tier"])
check("chinese routes to xtts instead", d["engine"] == "xtts", d["engine"])
check("chinese ceiling is not A either", d["best_possible_tier"] != "A",
      d["best_possible_tier"])
check("viitor claims english only", set(VE.LANGUAGES) == {"en"},
      str(VE.LANGUAGES))
check("registry agrees with the engine",
      R.ENGINES["viitor_nar"]["languages"] == {"en"},
      str(R.ENGINES["viitor_nar"]["languages"]))

d = R.resolve("hi", available=BOTH)
check("hindi is unaffected by tier A existing", d["tier"] == "C", d["tier"])

print()
print("=" * 74)
print("router — the operator can choose a tier")
print("=" * 74)
d = R.resolve("en", available=BOTH, prefer_tier="C")
check("a LOWER tier is honoured exactly",
      d["tier"] == "C" and d["engine"] == "xtts", f"{d['tier']}/{d['engine']}")
d = R.resolve("en", available=BOTH, prefer_tier="A")
check("requesting the tier already chosen changes nothing", d["tier"] == "A")
d = R.resolve("en", available=XTTS_ONLY, prefer_tier="A")
check("an unavailable tier falls back rather than failing", d["tier"] == "C")
check("and explains that it fell back",
      any("not available" in w for w in d["warnings"]), str(d["warnings"]))
d = R.resolve("en", available=BOTH, prefer_tier="D")
check("tier D forces segment granularity",
      d["tier"] == "D" and d["granularity"] == "segment")
check("tier D says what it did",
      any("whole" in w for w in d["warnings"]), str(d["warnings"]))
d = R.resolve("en", available=BOTH, prefer_tier="nonsense")
check("an unknown tier is ignored, with a warning, not an exception",
      d["tier"] == "A" and any("Unknown tier" in w for w in d["warnings"]))
check("prefer_tier is case-insensitive",
      R.resolve("en", available=BOTH, prefer_tier="c")["tier"] == "C")

check("selectable tiers reflect what is built",
      R.selectable_tiers("en", available=BOTH) == ["A", "C", "D"],
      str(R.selectable_tiers("en", available=BOTH)))
check("selectable tiers shrink when tier A is absent",
      R.selectable_tiers("en", available=XTTS_ONLY) == ["C", "D"],
      str(R.selectable_tiers("en", available=XTTS_ONLY)))
check("D is always offered", "D" in R.selectable_tiers("ta", available={}))

print()
print("=" * 74)
print("capability table reflects what is BUILT, not merely coded")
print("=" * 74)
t_no = R.capability_table(available=XTTS_ONLY)
t_yes = R.capability_table(available=BOTH)
check("unbuilt tier A is not advertised as current",
      t_no["en"]["tier"] == "C", str(t_no["en"]))
check("but the ceiling still shows it is reachable",
      t_no["en"]["best_possible_tier"] == "A")
check("built tier A becomes current", t_yes["en"]["tier"] == "A")

print()
print("=" * 74)
print("availability requires all three prerequisites")
print("=" * 74)
saved = (VE.VENV_VIITOR_PY, VE.VIITOR_MODELS, VE.VIITOR_DIR)
check("a bare install is not reported as available", not VE.available(),
      "nothing is installed in this checkout")
msg = VE._unavailable_message()
check("the failure names the environment", "Step 6" in msg, msg)
check("the failure names the weights", "Step 7" in msg, msg)
check("the failure is specific, not just 'unavailable'",
      "missing" in msg and len(msg) > 60, msg)

print()
print("=" * 74)
print("HTTP contract matches the documented API")
print("=" * 74)
import tempfile  # noqa: E402

tmp = tempfile.mkdtemp(prefix="vajra_viitor_test_")
wav = os.path.join(tmp, "clip.wav")
with open(wav, "wb") as fh:
    fh.write(b"RIFF____WAVEfmt ")

body, ctype = VE._multipart(
    {"original_text": "the old words", "edited_text": "the new words",
     "language": "en", "align_granularity": "word",
     "expand_mask_ratio": "0.1", "output_format": "wav"},
    {"source_audio": wav})
check("content type declares a boundary", "boundary=" in ctype, ctype)
boundary = ctype.split("boundary=")[1]
check("the boundary actually appears in the body",
      boundary.encode() in body)
for field in ("original_text", "edited_text", "language",
              "align_granularity", "expand_mask_ratio", "output_format"):
    check(f"field '{field}' is sent",
          f'name="{field}"'.encode() in body)
check("the audio is sent as a file part, with a filename",
      b'name="source_audio"' in body and b'filename="clip.wav"' in body)
check("the file's bytes are included", b"RIFFWAVEfmt" in body.replace(b"_", b""))
check("the body is correctly terminated",
      body.rstrip().endswith(f"--{boundary}--".encode()))

try:
    VE._multipart({}, {"source_audio": os.path.join(tmp, "nope.wav")})
    check("a missing upload is rejected before any request", False)
except ValueError as e:
    check("a missing upload is rejected before any request",
          "not found" in str(e), str(e))

print()
print("=" * 74)
print("the aligner language is set explicitly")
print("=" * 74)
env = VE._service_env()
check("aligner language is overridden away from the upstream zh default",
      env.get("VIITORVOICE_ALIGNER_LANGUAGE") == "en",
      env.get("VIITORVOICE_ALIGNER_LANGUAGE"))
check("the launcher is pointed at our venv",
      "venv_viitor" in env.get("VIITORVOICE_V2_VENV_DIR", ""),
      env.get("VIITORVOICE_V2_VENV_DIR"))
check("weights are pointed under MODEL_ROOT, so USE_DRIVE persists them",
      env.get("VIITORVOICE_LOCAL_MODELS", "").endswith("viitor"),
      env.get("VIITORVOICE_LOCAL_MODELS"))
check("health check does not raise when nothing is listening",
      VE._health(timeout=0.4) is False)

print()
print("=" * 74)
print("outputs are identifiable as VAJRA artefacts")
print("=" * 74)
from core.utils import timestamp_file, OUTPUT_PREFIX  # noqa: E402

for stem in ("viitor_clone", "relip", "edited_audio", "extracted"):
    name = os.path.basename(timestamp_file(stem, "wav"))
    check(f"'{stem}' output is prefixed", name.startswith(OUTPUT_PREFIX + "_"),
          name)
    check(f"'{stem}' keeps what it is", stem.split("_")[-1] in name, name)
already = os.path.basename(timestamp_file("vajra_edit", "wav"))
check("an already-prefixed stem is not doubled",
      already.count("vajra") == 1, already)
check("extension preserved", timestamp_file("x", "mp4").endswith(".mp4"))

print()
print("=" * 74)
print("tier A output skips the disguise chain")
print("=" * 74)
# The chain exists to push a sentence synthesiser's output into a recording it
# never heard. Tier A output was produced FROM that recording, so correcting it
# again stacks processing on audio that needs none.
eng_src = open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "engines", "transcript_engine.py"), encoding="utf-8").read()
check("tier A is branched separately at splice time",
      'if realism and decision["tier"] == "A":' in eng_src)
check("spectral correction is disabled for tier A",
      "spectral_strength=0.0" in eng_src)
check("room convolution is disabled for tier A",
      "room_strength=0.0" in eng_src)
check("no room tone is injected under tier A output",
      "room_tone=None, max_stretch=stretch" in eng_src)
check("the skip is recorded in the report",
      'rep["disguise"]' in eng_src)
check("tier C still gets the full chain",
      "room_tone=room_tone, max_stretch=stretch" in eng_src)
check("separation is off by default",
      'separate=False,' in eng_src)

print("=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL VIITOR TESTS PASSED")
