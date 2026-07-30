"""
Word-level edit resolution: turn "the user retyped this line" into "regenerate
exactly these seconds of audio".

Why this matters. The previous behaviour regenerated a whole transcript line
whenever any part of it changed. Changing two words in a nine-second sentence
replaced all nine seconds, so 100% of that sentence became synthetic when 80% of
it could have stayed genuine. Keeping real audio is the single strongest realism
lever there is -- every second retained is a second that needs no disguising.

So: diff at word level, map the changed words back to their timestamps, and
regenerate only that region.

One important constraint shapes the result. XTTS is a sentence-level
synthesiser, not a local-infill model: hand it three words in isolation and it
produces them with no coarticulation into the surrounding speech and with
sentence-final prosody, which sounds worse than the timing error it saves. So
the changed word range is EXPANDED outward to the nearest natural pause. That is
still meaningfully surgical -- a pause-bounded phrase is usually far shorter than
the full line -- while giving the synthesiser a complete utterance to work with.

When a true infill model is wired in (it conditions on the surrounding audio and
can regenerate a span in place), the expansion can shrink to the changed words
themselves. `resolve_edit_spans(..., expand_to_pauses=False)` already exposes
that path.
"""
import re
import difflib

# A gap of at least this long between two words reads as a pause we can cut at.
DEFAULT_MIN_GAP_SEC = 0.12
# Changed regions closer together than this are merged: two synthesised spans
# mean two pairs of seams, and one slightly longer span is usually the better
# trade.
DEFAULT_MERGE_GAP_SEC = 0.45

_PUNCT = re.compile(r"[^\w'ऀ-ॿ]+", re.UNICODE)


def tokenize(text):
    """Split into comparison tokens. Keeps Devanagari (Hindi is supported)."""
    if not text:
        return []
    return [t for t in _PUNCT.sub(" ", str(text)).split() if t]


def normalise(tok):
    """Casefold for comparison, so 'The' and 'the' are not a spurious edit."""
    return tok.casefold()


def words_from_segment(seg):
    """Normalise a segment's word list to [{'word','start','end'}, ...].

    Tolerates the several shapes a recogniser may hand back (objects with
    attributes, dicts with 'word' or 'text'), and returns [] when word timings
    are absent so callers can fall back to segment-level editing.
    """
    raw = seg.get("words") if isinstance(seg, dict) else getattr(seg, "words", None)
    if not raw:
        return []
    out = []
    for w in raw:
        if isinstance(w, dict):
            text = w.get("word", w.get("text", ""))
            start, end = w.get("start"), w.get("end")
        else:
            text = getattr(w, "word", getattr(w, "text", ""))
            start, end = getattr(w, "start", None), getattr(w, "end", None)
        text = (text or "").strip()
        if not text or start is None or end is None:
            continue
        out.append({"word": text, "start": float(start), "end": float(end)})
    return out


def changed_ranges(orig_tokens, new_tokens):
    """Index ranges of `orig_tokens` that differ, with their replacement text.

    Returns [(i0, i1, [new tokens]), ...] where orig_tokens[i0:i1] should become
    the given replacements. i0 == i1 denotes a pure insertion at that position.
    """
    sm = difflib.SequenceMatcher(
        a=[normalise(t) for t in orig_tokens],
        b=[normalise(t) for t in new_tokens],
        autojunk=False,
    )
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.append((i1, i2, list(new_tokens[j1:j2])))
    return out


def _merge(ranges, words, merge_gap_sec):
    """Coalesce changed ranges separated by little unchanged audio."""
    if not ranges:
        return []
    merged = [list(ranges[0])]
    for i0, i1, rep in ranges[1:]:
        p0, p1, prep = merged[-1]
        # Time actually separating the two changed regions.
        if p1 <= len(words) - 1 and i0 < len(words) and p1 <= i0 and p1 > 0:
            gap = words[i0]["start"] - words[p1 - 1]["end"] if p1 >= 1 and i0 < len(words) else 0.0
        else:
            gap = 0.0
        between = list(words[p1:i0]) if p1 <= i0 else []
        span_between = ((between[-1]["end"] - between[0]["start"])
                        if between else max(gap, 0.0))
        if span_between <= merge_gap_sec:
            # Absorb the unchanged words between them into one span.
            kept = [w["word"] for w in between]
            merged[-1] = [p0, i1, prep + kept + rep]
        else:
            merged.append([i0, i1, rep])
    return [tuple(m) for m in merged]


def _expand_to_pauses(i0, i1, words, min_gap_sec):
    """Widen [i0, i1) outward until each edge abuts a pause or the line end."""
    n = len(words)
    lo = max(0, min(i0, n))
    hi = max(lo, min(i1, n))

    # Walk left while the preceding gap is too small to cut at.
    while lo > 0:
        gap = words[lo]["start"] - words[lo - 1]["end"]
        if gap >= min_gap_sec:
            break
        lo -= 1
    # Walk right likewise.
    while hi < n:
        if hi == 0:
            hi += 1
            continue
        gap = (words[hi]["start"] - words[hi - 1]["end"]) if hi < n else min_gap_sec
        if gap >= min_gap_sec:
            break
        hi += 1
    return lo, max(hi, lo + 1 if n else lo)


def resolve_edit_spans(seg, new_text, min_gap_sec=DEFAULT_MIN_GAP_SEC,
                       merge_gap_sec=DEFAULT_MERGE_GAP_SEC,
                       expand_to_pauses=True):
    """Work out which parts of one segment to regenerate, and what to say.

    seg       {'start','end','text', 'words': [...]} -- words optional
    new_text  the user's edited text for this segment

    Returns a list of span dicts:
        {'t_start','t_end'      audio range to replace (seconds, absolute)
         'slack_start','slack_end'
                                how far that range may grow before it would
                                swallow a neighbouring word -- the silence a
                                longer replacement is allowed to absorb
         'text'                 what the synthesiser should say for that range
         'orig_text'            what it currently says
         'word_start','word_end' token indices covered
         'granularity'          'word' | 'segment'
         'expanded'             whether pause expansion widened the range}

    Falls back to a single segment-wide span when word timings are unavailable
    or the edit cannot be localised -- correct behaviour, just less surgical, and
    reported as such via 'granularity' so the UI can say so.
    """
    seg_start = float(seg["start"])
    seg_end = float(seg["end"])
    orig_text = (seg.get("text") or "").strip()
    new_text = (new_text or "").strip()

    if not new_text or normalise(new_text) == normalise(orig_text):
        return []

    def _whole():
        return [{
            "t_start": seg_start, "t_end": seg_end,
            # A whole-segment span already spans its line, so its only slack is
            # the line itself. Present on every span so callers never special-case.
            "slack_start": seg_start, "slack_end": seg_end,
            "text": new_text, "orig_text": orig_text,
            "word_start": None, "word_end": None,
            "granularity": "segment", "expanded": False,
        }]

    words = words_from_segment(seg)
    if not words:
        return _whole()

    orig_tokens = [w["word"] for w in words]
    new_tokens = tokenize(new_text)
    if not new_tokens:
        return _whole()

    ranges = changed_ranges(orig_tokens, new_tokens)
    if not ranges:
        # Text differs only in punctuation/case -- nothing to regenerate.
        return []

    # A rewrite touching most of the line is cheaper and more natural as one
    # span than as several stitched fragments.
    touched = sum(max(i1 - i0, 1) for i0, i1, _ in ranges)
    if touched >= 0.7 * len(orig_tokens):
        return _whole()

    ranges = _merge(ranges, words, merge_gap_sec)

    spans = []
    for i0, i1, rep in ranges:
        if expand_to_pauses:
            lo, hi = _expand_to_pauses(i0, i1, words, min_gap_sec)
            expanded = (lo != i0 or hi != i1)
        else:
            lo, hi = i0, max(i1, i0 + 1)
            expanded = False
        lo = max(0, min(lo, len(words) - 1))
        hi = max(lo + 1, min(hi, len(words)))

        # Text for the span = unchanged words either side of the edit that the
        # expansion pulled in, with the replacement in the middle.
        prefix = orig_tokens[lo:i0]
        suffix = orig_tokens[max(i1, lo):hi] if i1 < hi else []
        span_tokens = list(prefix) + list(rep) + list(suffix)
        if not span_tokens:
            continue

        # How far the span may grow WITHOUT swallowing a neighbouring word.
        # A word-level span is only as long as the words it replaces, so a
        # replacement with more syllables cannot fit and would be truncated
        # mid-phrase -- leaving the listener hearing part of the new wording
        # followed by the old words. The gaps either side are the room a human
        # editor would use, and the word timings already tell us where they are.
        slack_start = (float(words[lo - 1]["end"]) if lo > 0 else seg_start)
        slack_end = (float(words[hi]["start"]) if hi < len(words) else seg_end)

        spans.append({
            "t_start": max(seg_start, float(words[lo]["start"])),
            "t_end": min(seg_end, float(words[hi - 1]["end"])),
            "slack_start": max(seg_start, slack_start),
            "slack_end": min(seg_end, slack_end),
            "text": " ".join(span_tokens),
            "orig_text": " ".join(orig_tokens[lo:hi]),
            "word_start": lo, "word_end": hi,
            "granularity": "word", "expanded": bool(expanded),
        })

    if not spans:
        return _whole()

    # Guard against pause expansion having grown the spans until they overlap.
    spans.sort(key=lambda s: s["t_start"])
    out = [spans[0]]
    for s in spans[1:]:
        if s["t_start"] <= out[-1]["t_end"]:
            prev = out[-1]
            prev["t_end"] = max(prev["t_end"], s["t_end"])
            prev["slack_start"] = min(prev["slack_start"], s["slack_start"])
            prev["slack_end"] = max(prev["slack_end"], s["slack_end"])
            prev["text"] = prev["text"] + " " + s["text"]
            prev["orig_text"] = prev["orig_text"] + " " + s["orig_text"]
            prev["word_end"] = max(prev["word_end"] or 0, s["word_end"] or 0)
            prev["expanded"] = True
        else:
            out.append(s)
    return out


def summarise_spans(spans, seg):
    """Human-readable note on how much of a line is being regenerated."""
    if not spans:
        return "unchanged"
    seg_dur = max(float(seg["end"]) - float(seg["start"]), 1e-6)
    total = sum(s["t_end"] - s["t_start"] for s in spans)
    pct = 100.0 * total / seg_dur
    gran = spans[0]["granularity"]
    return (f"{len(spans)} span(s), {total:.2f}s of {seg_dur:.2f}s "
            f"({pct:.0f}% of the line) at {gran} granularity")
