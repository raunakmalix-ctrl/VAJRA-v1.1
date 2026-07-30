"""
Engine router: decides HOW a given edit gets regenerated, and says so honestly.

The constraint this exists to surface. Surgical local-editing models -- the ones
that regenerate a span in place, conditioned on the surrounding real audio -- do
not cover every language. Sentence-level synthesisers cover many more. So the
achievable quality of an edit depends on the language, and pretending otherwise
would mean silently giving a Tamil edit a visibly worse result than an English
one with no indication why.

Four tiers, best first:

  A  local infill          Regenerates only the changed words, conditioned on the
                           real audio either side. Nothing else is synthetic.
  B  local infill (NC)     Same technique, wider language coverage, but under a
                           non-commercial licence -- so it is opt-in, never a
                           silent default.
  C  sentence regeneration What this project ships today. Regenerates a
                           pause-bounded phrase and splices it in. Materially
                           worse than infill for a two-word change, materially
                           better than re-reading the whole track.
  D  segment only          No word timings for this language, so the edit cannot
                           be localised below the transcript line.

The router reports the tier it selected AND the tier the language would qualify
for if the relevant engine were installed. That distinction matters: "your
language only supports C" and "your language supports A but that engine is not
built" are different problems with different fixes, and collapsing them into one
message would hide a fixable situation.

Registering an engine that is not installed is deliberate. It keeps the
capability matrix in one place, lets the UI state what is achievable, and means
adding the engine later is a wiring change rather than a redesign.
"""

# ── engine registry ────────────────────────────────────────────────────────
# 'available' is resolved at call time, so an engine can appear the moment its
# environment is built without any change here.
ENGINES = {
    "viitor_nar": {
        "label": "ViiTorVoice-NAR",
        "tier": "A",
        "granularity": "infill",
        # English only, deliberately. The model supports Chinese upstream, but
        # this deployment does not enable it, so claiming it here would have the
        # router promise a tier no one has validated.
        "languages": {"en"},
        "licence": "Apache-2.0",
        "non_commercial": False,
        "note": ("True local infill: completes masked audio tokens under the "
                 "surrounding real audio, so only the changed words are "
                 "synthetic."),
        # Ships as five gRPC services behind an HTTP gateway rather than as a
        # library, so engines/viitor_engine.py supervises the process group and
        # speaks HTTP to it. Availability is still probed per call, since the
        # environment and weights are an optional build.
        "integrated": True,
    },
    "voicecraft_x": {
        "label": "VoiceCraft-X",
        "tier": "B",
        "granularity": "infill",
        "languages": {"en", "zh", "zh-cn", "ko", "ja", "es", "fr", "de", "nl",
                      "it", "pt", "pl"},
        "licence": "CC-BY-NC-4.0",
        "non_commercial": True,
        "note": ("Local infill across more languages, under a non-commercial "
                 "licence -- must be enabled explicitly."),
        "integrated": False,
    },
    "xtts": {
        "label": "XTTS-v2",
        "tier": "C",
        "granularity": "sentence",
        # XTTS-v2's 17 documented languages.
        "languages": {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                      "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"},
        "licence": "Coqui Public Model License",
        "non_commercial": False,
        "note": ("Regenerates a pause-bounded phrase and splices it in. The "
                 "phrase is synthetic; the rest of the line is the original "
                 "recording."),
        "integrated": True,
    },
}

TIER_ORDER = ["A", "B", "C", "D"]

TIER_QUALITY = {
    "A": "best — only the changed words are regenerated",
    "B": "best — only the changed words are regenerated (non-commercial engine)",
    "C": "good — a pause-bounded phrase is regenerated, the rest stays original",
    "D": "reduced — the edit cannot be localised below the transcript line",
}


def normalise_lang(code):
    """Fold a language tag to the form the registry uses.

    Accepts 'en-US', 'EN', 'zh_CN' and similar. Chinese is kept distinguishable
    because engines disagree on whether it is 'zh' or 'zh-cn'.
    """
    if not code:
        return "en"
    c = str(code).strip().lower().replace("_", "-")
    if c.startswith("zh"):
        return "zh-cn"
    return c.split("-")[0]


def _engine_supports(spec, lang):
    langs = spec["languages"]
    if lang in langs:
        return True
    # 'zh-cn' should match an engine that registered plain 'zh'.
    if lang == "zh-cn" and "zh" in langs:
        return True
    return False


def candidates(lang, allow_nc=False, integrated_only=True):
    """Engines that could serve `lang`, best tier first."""
    lang = normalise_lang(lang)
    out = []
    for key, spec in ENGINES.items():
        if not _engine_supports(spec, lang):
            continue
        if spec["non_commercial"] and not allow_nc:
            continue
        if integrated_only and not spec.get("integrated"):
            continue
        out.append((key, spec))
    out.sort(key=lambda kv: TIER_ORDER.index(kv[1]["tier"]))
    return out


def resolve(language, has_word_timings=True, allow_nc=False,
            available=None, prefer_tier=None):
    """Choose an engine and tier for an edit.

    language          transcript language code
    has_word_timings  whether word-level timings exist for this segment
    allow_nc          permit non-commercial engines
    available         optional {engine_key: bool}; when given, an engine is only
                      selected if its entry is truthy. Lets the caller reflect
                      what is actually built without this module probing.
    prefer_tier       'A'|'B'|'C'|'D' to request a specific tier instead of the
                      best available. Deliberately a REQUEST, not a command: a
                      tier that this language and installation cannot deliver
                      falls back to what they can, and says so. Asking for a
                      LOWER tier is honoured exactly -- that is how an operator
                      A/Bs infill against sentence regeneration.

    Returns a decision dict. Never raises and never returns None -- an edit must
    always resolve to something, even if that something is tier D with a warning.
    """
    lang = normalise_lang(language)

    usable = candidates(lang, allow_nc=allow_nc, integrated_only=True)
    if available is not None:
        usable = [(k, s) for k, s in usable if available.get(k)]

    warnings = []
    prefer_tier = (prefer_tier or "").strip().upper() or None
    if prefer_tier and prefer_tier not in TIER_ORDER:
        warnings.append(f"Unknown tier '{prefer_tier}' requested; using the "
                        f"best available instead.")
        prefer_tier = None
    if prefer_tier:
        exact = [(k, sp) for k, sp in usable if sp["tier"] == prefer_tier]
        if exact:
            usable = exact
        elif prefer_tier == "D":
            # D is not an engine, it is a granularity: honour it by forcing
            # segment-level below rather than by filtering the engine list.
            pass
        else:
            reachable = [sp["tier"] for _, sp in usable]
            warnings.append(
                f"Tier {prefer_tier} was requested but is not available for "
                f"'{lang}' with what is installed"
                + (f" — using tier {sorted(reachable)[0]}." if reachable
                   else " — falling back.")
            )
            prefer_tier = None

    # What the language could achieve if every engine were installed and NC
    # material were permitted -- the honest ceiling, used to explain the gap.
    ceiling = candidates(lang, allow_nc=True, integrated_only=False)
    best_possible = ceiling[0][1]["tier"] if ceiling else "D"

    if not usable:
        # Nothing serves this language. Fall back to the sentence engine in
        # English rather than refusing: an edit the operator can hear is more
        # useful than an error, provided the compromise is stated.
        spec = ENGINES["xtts"]
        decision = {
            "engine": "xtts",
            "label": spec["label"],
            "tier": "D",
            "granularity": "segment",
            "language": lang,
            "synthesis_language": "en",
        }
        warnings.append(
            f"No engine covers '{lang}'. Falling back to {spec['label']} with "
            f"English pronunciation, which will not sound native."
        )
    else:
        key, spec = usable[0]
        tier = spec["tier"]
        gran = spec["granularity"]
        if prefer_tier == "D":
            # An explicit request to edit whole transcript lines.
            tier = "D"
            gran = "segment"
            warnings.append("Tier D was requested: edits cover whole "
                            "transcript lines rather than individual words.")
        if not has_word_timings:
            # Without word timings the edit cannot be localised, whatever the
            # engine is capable of.
            tier = "D"
            gran = "segment"
            warnings.append(
                "No word-level timings for this segment, so the edit covers the "
                "whole transcript line. Precision is reduced."
            )
        decision = {
            "engine": key,
            "label": spec["label"],
            "tier": tier,
            "granularity": gran,
            "language": lang,
            "synthesis_language": lang,
        }

    decision["quality"] = TIER_QUALITY[decision["tier"]]
    decision["best_possible_tier"] = best_possible

    # Explain a gap between what was used and what the language could support.
    if (TIER_ORDER.index(best_possible) < TIER_ORDER.index(decision["tier"])
            and has_word_timings):
        better = [s for _, s in ceiling
                  if TIER_ORDER.index(s["tier"]) < TIER_ORDER.index(decision["tier"])]
        if better:
            b = better[0]
            b_key = next((k for k, sp in ENGINES.items() if sp is b), None)
            # An engine can be blocked by more than one thing at once, and
            # reporting only the first is actively misleading: telling someone
            # to build an engine that a licence setting will then still exclude
            # sends them to do work that changes nothing. List every blocker.
            blockers = []
            if not b.get("integrated"):
                blockers.append("is not integrated yet")
            elif available is not None and not available.get(b_key):
                # Integrated in code but not built in this session -- a
                # different problem from "no engine covers this language", and a
                # fixable one, so it must not be reported as the same thing.
                blockers.append("is not built in this session")
            if b["non_commercial"] and not allow_nc:
                blockers.append("is non-commercial and not enabled")
            if blockers:
                warnings.append(
                    f"{b['label']} (tier {b['tier']}) would give a better result "
                    f"for '{lang}' but " + " and ".join(blockers) + "."
                )

    decision["warnings"] = warnings
    return decision


def describe(decision):
    """One-line summary for logs and the UI."""
    if not decision:
        return "no routing decision"
    s = (f"tier {decision['tier']} via {decision['label']} "
         f"({decision['granularity']}, {decision['language']}) — "
         f"{decision['quality']}")
    if decision.get("warnings"):
        s += "  [" + "; ".join(decision["warnings"]) + "]"
    return s


def selectable_tiers(language, allow_nc=False, available=None):
    """Tiers an operator may actually pick for `language`, best first.

    The UI offers a choice, so it must offer only real options: listing tier A
    for a language no infill engine covers would invite a request that silently
    downgrades. D is always offered -- editing whole lines is always possible.
    """
    lang = normalise_lang(language)
    usable = candidates(lang, allow_nc=allow_nc, integrated_only=True)
    if available is not None:
        usable = [(k, sp) for k, sp in usable if available.get(k)]
    tiers = sorted({sp["tier"] for _, sp in usable},
                   key=TIER_ORDER.index)
    if "D" not in tiers:
        tiers.append("D")
    return tiers


def capability_table(allow_nc=False, available=None):
    """Per-language achievable tier, for surfacing the matrix in the UI.

    `available` matters: an engine can be integrated in code but not built in
    this session, and a table that showed its tier anyway would tell the
    operator they have a capability they do not. The 'tier' column is what would
    actually run right now; 'best_possible_tier' is the ceiling.
    """
    langs = set()
    for spec in ENGINES.values():
        langs |= {normalise_lang(l) for l in spec["languages"]}
    table = {}
    for lang in sorted(langs):
        now = candidates(lang, allow_nc=allow_nc, integrated_only=True)
        if available is not None:
            now = [(k, sp) for k, sp in now if available.get(k)]
        best = candidates(lang, allow_nc=True, integrated_only=False)
        table[lang] = {
            "tier": now[0][1]["tier"] if now else "D",
            "engine": now[0][1]["label"] if now else "—",
            "best_possible_tier": best[0][1]["tier"] if best else "D",
            "best_possible_engine": best[0][1]["label"] if best else "—",
        }
    return table
