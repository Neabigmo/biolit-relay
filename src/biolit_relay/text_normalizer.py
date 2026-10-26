"""Deterministic text normalisation for the criterion-evidence rule engine.

The rule engine compares human-curated cue phrases against title/abstract text.
Before any comparison happens the text is normalised *reversibly*: the normalised
copy is what the regexes run on, while a character-index map
:attr:`~biolit_relay.core.NormalizedText.orig_index` maps every normalised
character back to the original string so that evidence spans (start/end offsets
and the evidence sentence) can always be reported verbatim for audit.

Design constraints implemented here (design pack, doc 03, sections 3 and 9):

* lower-case everything, apply Unicode NFKC, unify hyphen variants and collapse
  whitespace;
* never delete or rewrite negation words - negation is handled explicitly by
  :func:`is_negated`, not by text mangling;
* matching is purely lexical and deterministic; no model, no randomness, no
  access to any evaluation metadata.

Additional, documented normalisation choices:

* apostrophe variants (``'``, ``'``, ``'``, ``` `` ```, ``´``) are mapped to the
  ASCII apostrophe so that cues such as ``"crohn's disease"`` survive
  typographic quoting;
* whitespace runs collapse to a single space that points at the *first*
  original whitespace character of the run; leading/trailing whitespace is
  stripped;
* a single original character may expand to several normalised characters
  (e.g. ``"ﬁ"`` -> ``"fi"``, ``"İ"`` -> ``"i̇"``); every expansion character maps
  to the same original index.  NFKC is applied per character so the offset map
  stays exact.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

from .core import NormalizedText, Token

__all__ = [
    "ABBREVIATIONS",
    "HYPHEN_VARIANTS",
    "NEGATION_CUES",
    "NEGATION_EXCEPTIONS",
    "is_negated",
    "normalize",
    "sentence_at",
    "split_sentences",
    "tokenize",
]

# ---------------------------------------------------------------------------
# Character-level normalisation tables
# ---------------------------------------------------------------------------

#: Unicode hyphen-like characters unified to the ASCII hyphen-minus ``-``.
HYPHEN_VARIANTS: dict[str, str] = {
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2015": "-",  # horizontal bar
    "\u2212": "-",  # minus sign
}

_APOSTROPHE_VARIANTS: dict[str, str] = {
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark / typographic apostrophe
    "\u201b": "'",  # single high-reversed-9 quotation mark
    "\u02bc": "'",  # modifier letter apostrophe
    "\u00b4": "'",  # acute accent used as apostrophe
    "\u0060": "'",  # grave accent used as apostrophe
}

_SOFT_HYPHEN = "\u00ad"  # invisible discretionary hyphen: dropped

#: Conservative sentence-boundary abbreviation list.  A period directly after
#: one of these tokens (case-insensitive, without the trailing period) does not
#: end a sentence.  Single-letter initials (``J.``) are handled separately.
ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "e.g",
        "i.e",
        "eg",
        "ie",
        "vs",
        "v",
        "no",
        "nos",
        "fig",
        "figs",
        "al",
        "et al",
        "ca",
        "cf",
        "approx",
        "dr",
        "prof",
        "mr",
        "mrs",
        "ms",
        "st",
        "eq",
        "ref",
        "refs",
        "tab",
        "suppl",
        "vol",
        "pp",
        "p",
        "inc",
        "ltd",
        "co",
        "jr",
        "sr",
        "dept",
        "univ",
        "ext",
        "min",
        "max",
        "avg",
        "etc",
        "resp",
        "u.s",
        "u.k",
        "e.g.",
        "i.e.",
        "vs.",
        "et al.",
    }
)

#: Default negation cues used when a caller does not pass a lexicon entry.
#: The frozen rule lexicon (``configs/rule_lexicon.yaml``) is the source of
#: truth for the experiments; these defaults only make the function usable
#: standalone.
NEGATION_CUES: tuple[str, ...] = (
    "not",
    "no",
    "without",
    "lack of",
    "lacked",
    "failed to",
    "non-",
    "absence of",
    "absent",
    "neither",
    "nor",
    "never",
    "excluded",
    "excluding",
    "except",
    "except for",
    "free of",
    "devoid of",
)

#: Phrases that contain a negation cue but do not negate the following text.
NEGATION_EXCEPTIONS: tuple[str, ...] = (
    "not only",
    "not limited to",
)

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*", re.UNICODE)
_SENTENCE_END_CHARS = ".!?;"
_CONTRAST_PATTERN = re.compile(
    r"\b(?:but|however|whereas|although|though|yet|except)\b"
)
_COORDINATOR_PATTERN = re.compile(r"\b(?:and|or)\b")
_SENTENCE_BREAK_PATTERN = re.compile(r"[.!?;]")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize(text: str) -> NormalizedText:
    """Normalise ``text`` while preserving a map back to the original string.

    Parameters
    ----------
    text:
        Original title/abstract text.  ``None`` is rejected; callers should
        replace missing text with the empty string before calling.

    Returns
    -------
    NormalizedText
        ``text`` is the normalised string (lower-case, NFKC, unified hyphens
        and apostrophes, collapsed whitespace).  ``original`` is the untouched
        input.  ``orig_index[j]`` is the index in ``original`` of the
        character that produced normalised character ``j``; therefore
        ``len(orig_index) == len(normalized text)`` as required by the frozen
        contract in :mod:`biolit_relay.core`.

    Raises
    ------
    TypeError
        If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize() expects str, got {type(text).__name__}")

    chars: list[str] = []
    index: list[int] = []

    for position, character in enumerate(text):
        if character in HYPHEN_VARIANTS:
            replaced = HYPHEN_VARIANTS[character]
        elif character in _APOSTROPHE_VARIANTS:
            replaced = _APOSTROPHE_VARIANTS[character]
        elif character == _SOFT_HYPHEN:
            continue
        else:
            # Per-character NFKC + lower keeps the offset map exact even when a
            # single code point expands to several characters.
            replaced = unicodedata.normalize("NFKC", character).lower()
        for expanded in replaced:
            chars.append(" " if expanded.isspace() else expanded)
            index.append(position)

    normalised_chars, normalised_index = _collapse_whitespace(chars, index)
    return NormalizedText(
        text="".join(normalised_chars),
        original=text,
        orig_index=tuple(normalised_index),
    )


def _collapse_whitespace(
    chars: list[str], index: list[int]
) -> tuple[list[str], list[int]]:
    """Collapse whitespace runs to one space and strip the ends.

    The emitted space keeps the original index of the first whitespace
    character of the run, so ``to_original_span`` still points into the
    original string.
    """
    out_chars: list[str] = []
    out_index: list[int] = []
    previous_was_space = True  # also strips leading whitespace
    for character, position in zip(chars, index):
        if character == " ":
            if previous_was_space:
                continue
            out_chars.append(" ")
            out_index.append(position)
            previous_was_space = True
        else:
            out_chars.append(character)
            out_index.append(position)
            previous_was_space = False
    if out_chars and out_chars[-1] == " ":
        out_chars.pop()
        out_index.pop()
    return out_chars, out_index


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def tokenize(norm: NormalizedText) -> list[Token]:
    """Split normalised text into word tokens.

    Tokens contain letters/digits, may contain internal ASCII apostrophes or
    hyphens (``"randomized-controlled"``, ``"crohn's"``) and are at least two
    characters long: single letters and single digits carry no lexical value
    for the rule engine.  Offsets are positions in ``norm.text``.

    Parameters
    ----------
    norm:
        A :class:`~biolit_relay.core.NormalizedText` produced by
        :func:`normalize`.

    Returns
    -------
    list[Token]
        Tokens in reading order, non-overlapping.
    """
    tokens: list[Token] = []
    for match in _TOKEN_PATTERN.finditer(norm.text):
        start, end = match.span()
        if end - start < 2:
            continue
        tokens.append(Token(text=match.group(0), start=start, end=end))
    return tokens


# ---------------------------------------------------------------------------
# Conservative sentence splitting
# ---------------------------------------------------------------------------


def split_sentences(
    norm: NormalizedText,
    abbreviations: Iterable[str] | None = None,
    forced_breaks: Iterable[int] = (),
) -> list[tuple[int, int]]:
    """Split normalised text into sentence spans.

    Splitting is deliberately conservative: a boundary is accepted only for
    ``.``, ``!`` or ``?`` that is followed by whitespace or the end of the
    text; ``;`` always separates clauses because the design treats it as a
    separator.  Periods inside decimal numbers and after common abbreviations
    or single-letter initials are not boundaries.

    Parameters
    ----------
    norm:
        Normalised text.
    abbreviations:
        Optional override for the abbreviation list; entries may include or
        omit the trailing period.
    forced_breaks:
        Optional normalised offsets where a sentence must end even without
        terminal punctuation.  The evidence engine uses this to keep a title
        without a final period from merging into the first abstract sentence;
        it is opt-in so the plain function keeps its documented behaviour.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, end)`` spans in normalised coordinates, contiguous and
        covering the whole text (``[]`` for empty text).
    """
    text = norm.text
    if not text:
        return []
    abbreviation_set = (
        ABBREVIATIONS
        if abbreviations is None
        else frozenset(_strip_period(a.lower()) for a in abbreviations)
    )
    break_positions = sorted(
        {
            int(position)
            for position in forced_breaks
            if 0 < int(position) < len(text)
            # a boundary directly after existing terminal punctuation is
            # already handled by the conservative punctation rule
            and text[int(position) - 1] not in _SENTENCE_END_CHARS
        }
    )
    break_index = 0

    spans: list[tuple[int, int]] = []
    length = len(text)
    start = 0
    position = 0
    while position < length:
        # Hard boundary that is not represented by punctuation in the text.
        if break_index < len(break_positions) and break_positions[break_index] <= position:
            forced = break_positions[break_index]
            break_index += 1
            if forced > start:
                spans.append((start, forced))
                next_position = forced
                while next_position < length and text[next_position] == " ":
                    next_position += 1
                start = next_position
                position = next_position
                continue
        character = text[position]
        if character not in _SENTENCE_END_CHARS:
            position += 1
            continue
        at_end = position + 1 >= length
        followed_by_space = at_end or text[position + 1] == " "
        if character == ";":
            # Clause separator: always a boundary, then skip following spaces.
            spans.append((start, position + 1))
            next_position = position + 1
            while next_position < length and text[next_position] == " ":
                next_position += 1
            start = next_position
            position = next_position
            continue
        if not followed_by_space:
            position += 1
            continue
        if character == "." and not at_end and _period_is_nonterminal(
            text, position, abbreviation_set
        ):
            position += 1
            continue
        spans.append((start, position + 1))
        next_position = position + 1
        while next_position < length and text[next_position] == " ":
            next_position += 1
        start = next_position
        position = next_position

    if start < length:
        spans.append((start, length))
    return spans


def _strip_period(token: str) -> str:
    return token[:-1] if token.endswith(".") else token


def _period_is_nonterminal(
    text: str, position: int, abbreviations: frozenset[str]
) -> bool:
    """Return True when the period at ``position`` is not a sentence end."""
    # Decimal number, e.g. "1.5" or "p = 0.04".
    if (
        0 < position < len(text) - 1
        and text[position - 1].isdigit()
        and text[position + 1].isdigit()
    ):
        return True
    # Extract the word immediately before the period.
    word_start = position
    while word_start > 0 and text[word_start - 1] not in " ,;:()[]":
        word_start -= 1
    word = text[word_start:position].strip("()[]\"'")
    if not word:
        return True  # leading period, nothing to end
    lowered = word.lower()
    if lowered in abbreviations or _strip_period(lowered) in abbreviations:
        return True
    if len(word) == 1 and word.isalpha():
        return True  # single-letter initial, e.g. "J. Smith"
    return False


def sentence_at(
    sentences: Sequence[tuple[int, int]], offset: int
) -> tuple[int, int] | None:
    """Return the sentence span containing normalised ``offset``.

    ``None`` is returned when the offset is outside every span; callers treat
    that as "no sentence available" and leave the evidence sentence empty.
    """
    for start, end in sentences:
        if start <= offset < end:
            return start, end
    return None


# ---------------------------------------------------------------------------
# Local negation detection
# ---------------------------------------------------------------------------


def is_negated(
    norm: NormalizedText,
    tokens: Sequence[Token],
    match_start: int,
    window_tokens_before: int = 5,
    cues: Sequence[str] = NEGATION_CUES,
    exceptions: Sequence[str] = NEGATION_EXCEPTIONS,
    scope_boundaries: Sequence[int] = (),
) -> tuple[bool, bool]:
    """Check whether a phrase starting at ``match_start`` is locally negated.

    The check is intentionally shallow and auditable: it looks at the
    ``window_tokens_before`` tokens ending at or before ``match_start``, finds
    negation cues there and rejects cues that belong to an exception phrase
    (``"not only"``), that are cut off by a sentence/clause boundary, or that
    are separated from the match by a contrastive connector (``but``,
    ``however``, ...).  A ``"non-"`` prefix attached directly to the matched
    phrase counts as negation.

    Scope policy (design doc 03, section 3): the engine never performs
    aggressive semantic flipping.  When the local scope is genuinely unclear
    (comma or ``and``/``or`` between the cue and the match, or several cues in
    scope) the function reports ``(False, True)`` - "ambiguous, not
    confidently negated" - and the caller downgrades the evidence instead of
    inventing a state.  When the cue is clearly in scope it returns
    ``(True, False)``.

    Parameters
    ----------
    norm:
        Normalised text containing the match.
    tokens:
        Tokens of ``norm`` (see :func:`tokenize`).  Must be in reading order.
    match_start:
        Normalised start offset of the matched phrase.
    window_tokens_before:
        How many preceding tokens are inspected (design default: 5).
    cues:
        Negation cue phrases, matched case-insensitively.  The lexicographic
        defaults are only a fallback; the frozen lexicon is authoritative.
    exceptions:
        Phrases that contain a negation cue but are not negations.
    scope_boundaries:
        Normalised offsets that act as implicit sentence boundaries even when
        the text has no punctuation there (the evidence engine passes the
        title/abstract boundary so a title negation cannot leak into the
        abstract).

    Returns
    -------
    tuple[bool, bool]
        ``(negated, ambiguous)``.  ``negated`` is True only when a cue is
        confidently in scope; ``ambiguous`` is True when negation-like material
        is present but its scope cannot be resolved locally.

    Raises
    ------
    ValueError
        If ``window_tokens_before`` is negative or ``match_start`` is outside
        the normalised text.
    """
    if window_tokens_before < 0:
        raise ValueError("window_tokens_before must be >= 0")
    if not 0 <= match_start <= len(norm.text):
        raise ValueError(
            f"match_start {match_start} outside normalised text "
            f"(len={len(norm.text)})"
        )

    text = norm.text
    if _has_attached_non_prefix(text, match_start):
        return True, False

    previous_tokens = [token for token in tokens if token.end <= match_start]
    if not previous_tokens:
        return False, False
    window = previous_tokens[-window_tokens_before:] if window_tokens_before else []
    if not window:
        return False, False

    cue_pattern = _phrase_pattern(cues)
    if cue_pattern is None:
        return False, False
    exception_pattern = _phrase_pattern(exceptions)

    joined_parts: list[str] = []
    token_spans: list[tuple[int, int, int, int]] = []
    cursor = 0
    for position, token in enumerate(window):
        if position:
            joined_parts.append(" ")
            cursor += 1
        token_spans.append((cursor, cursor + len(token.text), token.start, token.end))
        joined_parts.append(token.text)
        cursor += len(token.text)
    joined = "".join(joined_parts)

    exception_spans: list[tuple[int, int]] = []
    if exception_pattern is not None:
        exception_spans = [match.span() for match in exception_pattern.finditer(joined)]

    in_scope = 0
    ambiguous = False
    for match in cue_pattern.finditer(joined):
        if any(
            exception_start <= match.start() and match.end() <= exception_end
            for exception_start, exception_end in exception_spans
        ):
            continue
        _, cue_end = _map_joined_span(match.start(), match.end(), token_spans)
        gap = text[cue_end:match_start]
        if _SENTENCE_BREAK_PATTERN.search(gap):
            continue  # cue belongs to a previous sentence/clause: out of scope
        if any(cue_end < boundary <= match_start for boundary in scope_boundaries):
            continue  # implicit hard boundary (e.g. title/abstract split)
        if _CONTRAST_PATTERN.search(gap):
            continue  # "not X, but Y": cue does not reach the match
        if _COORDINATOR_PATTERN.search(gap) or "," in gap:
            ambiguous = True
            continue
        in_scope += 1

    if in_scope == 0:
        return False, ambiguous
    if in_scope > 1 or ambiguous:
        return True, True
    return True, False


def _has_attached_non_prefix(text: str, match_start: int) -> bool:
    """True when the match is directly preceded by a ``non-`` prefix."""
    if match_start < 4:
        return False
    if text[match_start - 4 : match_start] != "non-":
        return False
    if match_start == 4:
        return True
    return not text[match_start - 5].isalnum()


def _map_joined_span(
    joined_start: int,
    joined_end: int,
    token_spans: Sequence[tuple[int, int, int, int]],
) -> tuple[int, int]:
    """Map a span in the space-joined window back to normalised coordinates."""
    covered = [
        span
        for span in token_spans
        if span[1] > joined_start and span[0] < joined_end
    ]
    if not covered:
        return 0, 0
    first = covered[0]
    last = covered[-1]
    norm_start = first[2] + (joined_start - first[0])
    norm_end = last[3] - (last[1] - joined_end)
    return max(0, norm_start), max(0, norm_end)


def _phrase_pattern(phrases: Sequence[str] | None) -> re.Pattern[str] | None:
    """Build a case-insensitive whole-phrase regex from cue phrases."""
    if not phrases:
        return None
    cleaned = sorted(
        {phrase.strip().lower() for phrase in phrases if phrase and phrase.strip()},
        key=lambda phrase: (-len(phrase), phrase),
    )
    if not cleaned:
        return None
    body = "|".join(_phrase_to_regex(phrase) for phrase in cleaned)
    return re.compile(body)


def _phrase_to_regex(phrase: str) -> str:
    """Turn one cue phrase into a regex with flexible space/hyphen joining.

    Spaces and hyphens inside a cue are interchangeable so that
    ``"cross-sectional"`` also matches ``"cross sectional"`` and vice versa.
    A word boundary is only added on an end that is alphanumeric: cues such as
    ``"non-"`` must be allowed to attach to the following word.
    """
    escaped = re.escape(phrase)
    escaped = escaped.replace(r"\-", "[- ]").replace(r"\ ", "[- ]")
    prefix = r"(?<![a-z0-9])" if phrase[0].isalnum() else ""
    suffix = r"(?![a-z0-9])" if phrase[-1].isalnum() else ""
    return prefix + escaped + suffix
