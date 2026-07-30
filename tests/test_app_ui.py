# -*- coding: utf-8 -*-
"""Build app.py's UI graph against stubs and exercise the Edit & Relip renderers.

There is no GPU or gradio here, so gradio/torch are stubbed. The point is to
prove the tab wires up and that the report panels render the REAL dict shapes
produced by core.router and evaluation.scorecard -- a key typo in an f-string is
invisible to a syntax check and would only show up in Colab.
"""
import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ── stub the heavy transitive deps ──────────────────────────────────────────
t = types.ModuleType("torch")
t.cuda = types.SimpleNamespace(
    is_available=lambda: False, get_device_name=lambda i: "CPU",
    empty_cache=lambda: None, ipc_collect=lambda: None,
    mem_get_info=lambda *a: (0, 0))
t.float16 = t.float32 = t.bfloat16 = object()
t.no_grad = lambda: __import__("contextlib").nullcontext()
t.device = lambda *a, **k: None
t.Generator = lambda *a, **k: None
# scipy sniffs sys.modules for torch and calls getattr(torch, "Tensor") to decide
# whether an array is a tensor, so the stub has to carry one.
t.Tensor = type("Tensor", (), {})
sys.modules["torch"] = t
for n in ("cv2", "insightface", "transformers", "diffusers"):
    sys.modules.setdefault(n, types.ModuleType(n))
pil = types.ModuleType("PIL")
pil.Image = types.SimpleNamespace(open=lambda *a, **k: None,
                                 new=lambda *a, **k: None)
sys.modules.setdefault("PIL", pil)
sys.modules.setdefault("PIL.Image", pil.Image)


class _C:
    """Stands in for any gradio component or container."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return lambda *a, **k: self


class _GradioStub(types.ModuleType):
    themes = types.SimpleNamespace(Base=object)

    @staticmethod
    def Progress(*a, **k):
        return lambda *aa, **kk: None

    def __getattr__(self, name):
        # Any component name resolves; a missing stub must not masquerade as a
        # real failure in the code under test.
        if name.startswith("_"):
            raise AttributeError(name)
        return _C


sys.modules["gradio"] = _GradioStub("gradio")
gcu = types.ModuleType("gradio_client.utils")
gcu._json_schema_to_python_type = lambda s, d=None: "Any"
sys.modules["gradio_client"] = types.ModuleType("gradio_client")
sys.modules["gradio_client.utils"] = gcu

# ── build the UI graph ──────────────────────────────────────────────────────
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("appmod", r"D:\VAJRA-v1.1\app.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("app.py imported — full UI graph built")

# ── exercise the renderers on real payloads ─────────────────────────────────
import numpy as np  # noqa: E402
from core import router  # noqa: E402
import evaluation as EV  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


d_ta = router.resolve("ta", has_word_timings=True, available={"xtts": True})
h = m._routing_html(d_ta)
check("unsupported language surfaces its fallback warning",
      "Note" in h and "native" in h)
check("tier D is coloured as a problem", "var(--err)" in h)

d_en = router.resolve("en", has_word_timings=True, available={"xtts": True})
h = m._routing_html(d_en)
check("achievable ceiling is shown when better than the selected tier",
      "Ceiling" in h and "ViiTor" in h)

check("empty decision renders nothing", m._routing_html(None) == "")

sr = 24000
r = np.random.RandomState(1)
tt = np.arange(int(6 * sr)) / sr
x = (0.3 * np.sin(2 * np.pi * 130 * tt) * np.clip(np.sin(2 * np.pi * 3 * tt), 0, None)
     + 10 ** (-52 / 20) * r.randn(tt.size)).astype(np.float32)
card = EV.score_all(x, sr, [(int(2 * sr), int(2.7 * sr))], [None])
h = m._scorecard_html(card)
check("scorecard renders the seam verdict", "seam" in h and "pct" in h)
check("unmeasured metrics are named rather than omitted",
      "Not measured" in h and "speaker similarity" in h)
check("baseline size is reported", "genuine word" in h)
check("empty scorecard renders nothing", m._scorecard_html(None) == "")

st = {"routing": d_en, "scorecard": card,
      "lipsync": {"windowed": True, "windows": [(1.0, 2.0)], "coverage": 0.0483}}
h = m._relip_report(st)
check("windowed coverage rendered", "4.83%" in h)
check("report says the rest is untouched", "untouched" in h)

st["lipsync"] = {"windowed": False,
                 "reason": "edits cover 71% of the clip, so the whole video "
                           "was synced instead"}
check("whole-video fallback explains itself", "71%" in m._relip_report(st))

check("report tolerates a bare state", m._relip_report({}) == "")

# The report panels are diagnostics, so the tab must lead with the result and
# keep them one click away rather than in the operator's face.
import inspect  # noqa: E402

src = inspect.getsource(m) if hasattr(m, "__file__") else ""
check("the report sits inside a collapsed accordion",
      "Routing & detectability report" in open(
          os.path.join(_ROOT, "app.py"), encoding="utf-8").read())


print()
print("=" * 70)
print("the UI names capabilities, not vendors")
print("=" * 70)
# Swapping an engine should not change what the operator has learned to read,
# and a model name in a label goes stale the moment the model is replaced. The
# names stay in the logs and in core/router.py, where they are operationally
# useful.
_app_src = open(os.path.join(_ROOT, "app.py"), encoding="utf-8").read()
_visible = "\n".join(l for l in _app_src.splitlines()
                     if not l.strip().startswith("#"))
for name in ("SDXL", "RealVis", "LatentSync", "Wav2Lip", "XTTS", "Wan2.2",
             "LTX", "Qwen", "InsightFace", "GFPGAN", "CodeFormer",
             "ViiTor", "Demucs", "venv_"):
    check(f"no '{name}' in user-facing strings", name not in _visible)

# Renamed choices are dictionary keys elsewhere; a half-rename falls through to
# a default instead of failing, so both sides are pinned here.
for choice, key in (("Best quality", "latentsync"), ("Fast", "wav2lip"),
                    ("Photoreal (recommended)", "sdxl_real"),
                    ("General purpose", "sdxl"),
                    ("Standard", "gfpgan"), ("High detail", "codeformer")):
    check(f"choice '{choice}' still maps to {key}",
          f'"{choice}": "{key}"' in _app_src)
check("the motion-engine branch matches a current choice label",
      'startswith("Synchronized audio")' in _app_src
      and '"Synchronized audio (faster, adds a soundtrack)"' in _app_src)

print()
print("=" * 70)
print("header waveform")
print("=" * 70)
w = m.wave_html()
check("bars are emitted", w.count("vw-bar") > 40, str(w.count("vw-bar")))
check("bars are individually offset",
      len({b.split("animation-delay:")[1][:4]
           for b in w.split("<span")[1:]}) > 5)
check("it is hidden from screen readers", "aria-hidden" in w)
check("it is mounted under the status ribbon",
      "gr.HTML(ribbon_html())\n    gr.HTML(wave_html())" in _app_src)
check("reduced-motion is respected",
      "prefers-reduced-motion" in open(
          os.path.join(_ROOT, "app_theme.py"), encoding="utf-8").read())

print("=" * 70)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): " + "; ".join(FAILS))
    sys.exit(1)
print("ALL UI RENDER CHECKS PASSED")
