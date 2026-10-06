"""Split raw document text into clause-sized pieces (sentences / numbered items)."""

import re

MIN_CLAUSE_CHARS = 25

# A new clause starts at a line beginning with a list marker: "1.", "2.3", "(a)", "a)", "i.", "-", "•".
_LIST_MARKER = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?|\(?[a-zA-Z]\)|[ivxIVX]+\.|[-•*▪●])\s+")

# Abbreviations whose trailing dot must not end a sentence.
_ABBREVIATIONS = [
    "U.S.", "U.K.", "e.g.", "i.e.", "etc.", "Inc.", "Ltd.", "Pvt.", "Co.", "No.", "Rs.",
    "Mr.", "Mrs.", "Ms.", "Dr.", "St.", "vs.", "approx.", "Sec.", "Art.", "cl.",
]
_DOT = "․"  # placeholder for protected dots

# Sentence end: . ! ? (optionally followed by a quote/bracket) then whitespace and an uppercase/digit/quote.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9\"'(])")


def _protect(text: str) -> str:
    for abbr in _ABBREVIATIONS:
        text = re.sub(re.escape(abbr), abbr.replace(".", _DOT), text, flags=re.IGNORECASE)
    # A leading list number such as "2." or "3.1." is not a sentence end.
    text = re.sub(r"^(\s*\d+(?:\.\d+)*)\.", lambda m: m.group(1) + _DOT, text)
    # Decimals such as 2.5 or 1.2.3 section numbers inside a sentence.
    return re.sub(r"(?<=\d)\.(?=\d)", _DOT, text)


HEADING_MAX_CHARS = 60


def _is_heading(line: str) -> bool:
    """Short line without closing punctuation, e.g. "4. Changes to These Terms" or "Termination"."""
    return len(line) <= HEADING_MAX_CHARS and not re.search(r"[.!?:;,]$", line)


def _blocks(text: str) -> list[str]:
    """Group lines into blocks: a blank line, a list marker or a heading line starts a new block,
    other lines are treated as wrapped continuations of the previous line."""
    blocks: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            blocks.append([])
            continue
        # A wrapped sentence continues in lowercase; a new sentence after a heading starts uppercase.
        previous_is_heading = bool(blocks and blocks[-1]) and _is_heading(blocks[-1][-1]) and line[0].isupper()
        if not blocks or not blocks[-1] or _LIST_MARKER.match(line) or previous_is_heading:
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return [" ".join(b) for b in blocks if b]


def split_clauses(text: str) -> list[str]:
    """Return clauses of at least MIN_CLAUSE_CHARS characters, in document order."""
    clauses: list[str] = []
    for block in _blocks(text.replace("\r", "")):
        if _is_heading(block):  # a standalone heading is a title, not a clause
            continue
        for sentence in _SENTENCE_END.split(_protect(block)):
            sentence = re.sub(r"\s+", " ", sentence).strip().replace(_DOT, ".")
            if len(sentence) >= MIN_CLAUSE_CHARS:
                clauses.append(sentence)
    return clauses
