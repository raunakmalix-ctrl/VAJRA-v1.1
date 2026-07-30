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
NOTEBOOK = os.path.join(_ROOT, "VAJRA_v1.1_Colab.ipynb")

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

print("=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL NOTEBOOK CHECKS PASSED")
