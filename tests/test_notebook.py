# -*- coding: utf-8 -*-
"""Every code cell in the Colab notebook must actually parse as Python.

This exists because a broken string literal in a setup cell is invisible until
someone runs it on a GPU -- the failure surfaces as a SyntaxError in Colab,
minutes into a session, on a cell that looks fine in a diff. It has happened
twice. Parsing every cell here costs milliseconds.

IPython shell escapes (`!cmd`) and magics (`%cmd`) are not Python, so they are
replaced with `pass` at the same indentation before parsing -- that keeps block
structure intact, which is exactly what a mis-split string breaks.
"""
import ast
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(_ROOT, "VAJRA_2.5_Colab.ipynb")

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def as_python(source):
    """Notebook cell source -> parseable Python, magics neutralised."""
    out = []
    for line in source.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            out.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            out.append(line)
    return "\n".join(out)


print("=" * 74)
print("notebook — structure")
print("=" * 74)
check("notebook exists", os.path.exists(NOTEBOOK), NOTEBOOK)
if not os.path.exists(NOTEBOOK):
    sys.exit(1)

with open(NOTEBOOK, encoding="utf-8") as fh:
    nb = json.load(fh)
check("notebook is valid JSON", isinstance(nb.get("cells"), list))

code_cells = [(i, c) for i, c in enumerate(nb["cells"])
              if c.get("cell_type") == "code"]
check("notebook has code cells", len(code_cells) > 0, f"{len(code_cells)} cells")

print()
print("=" * 74)
print("notebook — every code cell parses")
print("=" * 74)
for i, cell in code_cells:
    src = "".join(cell["source"])
    try:
        ast.parse(as_python(src))
        ok, detail = True, ""
    except SyntaxError as e:
        ok = False
        detail = f"line {e.lineno}: {e.msg}"
    check(f"cell {i}", ok, detail)

print()
print("=" * 74)
print("notebook — setup flags match the build dispatcher")
print("=" * 74)
# The Step 6 cell passes target names straight to make_venvs.sh. A rename on
# either side silently builds nothing, or fails deep in a Colab session.
dispatcher = os.path.join(_ROOT, "setup", "make_venvs.sh")
with open(dispatcher, encoding="utf-8") as fh:
    sh = fh.read()
valid = set()
for line in sh.split("\n"):
    if line.strip().startswith("ALL_TARGETS="):
        valid = set(line.split('"')[1].split())
        break
check("dispatcher declares its targets", bool(valid), str(sorted(valid)))

step6 = next((("".join(c["source"])) for _, c in code_cells
              if "make_venvs.sh" in "".join(c["source"])), "")
check("step 6 cell found", bool(step6))
named = {t for t in valid if f"'{t}'" in step6}
missing = valid - named
check("every dispatcher target is selectable in the notebook",
      not missing, f"missing: {sorted(missing)}" if missing else "")

# And nothing is offered that the dispatcher would reject.
import re

offered = set(re.findall(r"\('([a-z0-9]+)',\s*[A-Z]", step6))
unknown = offered - valid
check("notebook offers no target the dispatcher rejects",
      not unknown, f"unknown: {sorted(unknown)}" if unknown else "")

print()
print("=" * 74)
print("notebook — the checkout is pinned to a branch")
print("=" * 74)
# Cloning the default branch fetches main, which lacks the VAJRA 2.0 packages
# and ships an older setup script that ignores build targets. That failure is
# silent -- the wrong code runs and looks fine -- so the branch must be explicit.
clone_cell = next((("".join(c["source"])) for _, c in code_cells
                   if "git clone" in "".join(c["source"])), "")
check("clone cell found", bool(clone_cell))
check("a branch is named explicitly", "BRANCH" in clone_cell)
check("clone pins the branch", "--branch $BRANCH" in clone_cell,
      "a bare `git clone` takes the repo default branch")
check("an existing checkout is updated, not left stale",
      "pull" in clone_cell and "fetch origin" in clone_cell)
for pkg in ("core/router.py", "evaluation/scorecard.py", "dsp/match.py"):
    check(f"{pkg} exists on the pinned branch",
          os.path.exists(os.path.join(_ROOT, pkg)))

print()
print("=" * 74)
print("notebook — cell source is stored as lines, and carries no legacy naming")
print("=" * 74)
# A cell may store its source one CHARACTER per list element. Colab renders it
# fine, so it goes unnoticed -- but no multi-character token then appears
# contiguously in the file, which means grep cannot find it and a search-replace
# silently misses it. That exact fault hid a stale env-var name through a
# project-wide rename, so it is checked rather than trusted.
split = [i for i, c in enumerate(nb["cells"])
         if isinstance(c.get("source"), list) and c["source"]
         and max(len(x) for x in c["source"]) == 1 and len(c["source"]) > 4]
check("no cell is stored one character per element", not split, str(split))

joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for legacy in ("IMAGE_TALK", "image_talk", "Image Talk", "Image-Talk"):
    check(f"no '{legacy}' anywhere in the notebook", legacy not in joined)

# The notebook sets these; core/config.py and the shell builders read them. A
# rename on one side alone leaves the other silently on its default.
for var in ("VAJRA_ROOT", "VAJRA_MODELS"):
    check(f"{var} is set by the notebook", var in joined)
    cfg = open(os.path.join(_ROOT, "core", "config.py"), encoding="utf-8").read()
    check(f"{var} is read by core/config.py", var in cfg)

print()
print("=" * 74)
print("v2.5 upgrades are wired end to end")
print("=" * 74)
cfg = open(os.path.join(_ROOT, "core", "config.py"), encoding="utf-8").read()
dl = open(os.path.join(_ROOT, "setup", "download_models.py"),
          encoding="utf-8").read()
te = open(os.path.join(_ROOT, "engines", "transcript_engine.py"),
          encoding="utf-8").read()
ls = open(os.path.join(_ROOT, "engines", "lipsync_engine.py"),
          encoding="utf-8").read()

# The checkpoint and the UNet config are a matched pair: 1.6 is a 512px model
# and will not load against the 256px stage2 config. Pinning one without the
# other loads a mismatched network rather than failing loudly.
check("lip-sync weights are pinned to 1.6", "LatentSync-1.6" in cfg)
check("and the 512px UNet config goes with them", "stage2_512.yaml" in cfg)
check("no 256px config left behind",
      '"stage2.yaml"' not in cfg)
check("the downloader uses the pinned repo, not a literal",
      "LATENTSYNC_HF_REPO" in dl and "LatentSync-1.5" not in dl)
# Using a name is not the same as importing it: the first version of this
# referenced LATENTSYNC_HF_REPO without adding it to the import list, which
# passed the check above and would have crashed on the first download.
check("and actually imports it",
      "LATENTSYNC_HF_REPO,
)" in dl or "LATENTSYNC_HF_REPO," in dl.split("
)")[0])
# Successive releases ship the same filename with different architectures, so
# a plain existence check would keep stale weights and pair them with the new
# config -- a mismatched model that loads and produces bad output.
check("the weight cache is keyed to the release",
      ".vajra_source" in dl and "want.split" in dl)

# 24 kHz has to survive all the way to the file, not just the working buffer.
check("the master rate is 24 kHz", "MASTER_SR = 24000" in te)
check("extraction uses it rather than a hardcoded rate",
      'str(MASTER_SR)' in te and '"16000"' not in te)
check("the finished video is re-muxed with the master audio",
      "_remux_master" in ls)
check("windowed compositing muxes the master, not the model's copy",
      "_composite_video(video_path, frame_windows, out_path, audio_path," in ls)
check("the models are still fed 16 kHz, which is what they want",
      "to_wav(audio_path)" in ls)

check("the notebook pins the 2.5 branch", "vajra-2.5" in joined)
check("and says what 2.5 changes", "What 2.5 changes over 2.0" in joined)

print("=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL NOTEBOOK CHECKS PASSED")
