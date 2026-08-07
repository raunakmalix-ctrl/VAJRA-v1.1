"""
Text normalisation shared between the principal runtime and the isolated
voice environment.

Stdlib only, deliberately: workers import this before their own heavy
dependencies are touched, and it must not drag anything in.
"""
import re

# XTTS expands digits to words itself, via num2words, before tokenising. For
# several of the languages it claims to support -- Hindi among them -- num2words
# has no converter, so ANY text containing a digit raises NotImplementedError
# deep inside the tokeniser and the whole synthesis fails. A transcript with a
# year, a price or a phone number in it is enough.
#
# So digits are expanded HERE, before the model ever sees them, leaving nothing
# for its own expander to trip over. Where a proper converter exists it is used;
# otherwise digits are read out individually in the target language, which is
# how people usually say identifiers anyway and is unambiguously better than a
# crash.
_DIGIT_WORDS = {
    "hi": ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"],
    "ar": ["صفر", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة",
           "ثمانية", "تسعة"],
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven",
           "eight", "nine"],
}


def _num2words_supports(lang):
    try:
        from num2words import CONVERTER_CLASSES
    except Exception:
        return False
    return lang in CONVERTER_CLASSES or lang.split("-")[0] in CONVERTER_CLASSES


def expand_digits(text, language="en"):
    """Replace digit runs with spoken words, so XTTS never expands them itself.

    Returns the text unchanged when it holds no digits, which is the common
    case and must cost nothing.
    """
    import re
    if not text or not any(ch.isdigit() for ch in text):
        return text
    lang = (language or "en").split("-")[0].lower()

    if _num2words_supports(lang):
        from num2words import num2words

        def repl(m):
            try:
                return " " + num2words(int(m.group(0)), lang=lang) + " "
            except Exception:
                return " " + _spell_digits(m.group(0), lang) + " "
    else:
        def repl(m):
            return " " + _spell_digits(m.group(0), lang) + " "

    out = re.sub(r"\d+", repl, text)
    return re.sub(r"\s+", " ", out).strip()


def _spell_digits(run, lang):
    words = _DIGIT_WORDS.get(lang, _DIGIT_WORDS["en"])
    return " ".join(words[int(d)] for d in run if d.isdigit())


