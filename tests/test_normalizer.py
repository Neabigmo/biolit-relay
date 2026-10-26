"""Unit tests for the deterministic text normaliser (design doc 03, section 8).

Covers: lower-casing and offset mapping, Unicode/hyphen/apostrophe
normalisation, whitespace collapse, tokenisation, conservative sentence
splitting (abbreviations, decimals, semicolons) and local negation detection
including the ``not only`` exception, the ``non-`` prefix and ambiguous-scope
reporting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biolit_relay.core import Token  # noqa: E402
from biolit_relay.text_normalizer import (  # noqa: E402
    HYPHEN_VARIANTS,
    is_negated,
    normalize,
    sentence_at,
    split_sentences,
    tokenize,
)


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_keeps_original() -> None:
    original = "Knee Osteoarthritis (KOOS) Study"
    norm = normalize(original)
    assert norm.original == original
    assert norm.text == "knee osteoarthritis (koos) study"
    assert len(norm.orig_index) == len(norm.text)


def test_normalize_orig_index_maps_every_character() -> None:
    norm = normalize("Randomized–Controlled Trial of MSC Therapy")
    assert len(norm.orig_index) == len(norm.text)
    assert all(0 <= index < len(norm.original) for index in norm.orig_index)
    # duplicated original indices (expansions) must stay ordered
    assert list(norm.orig_index) == sorted(norm.orig_index)


def test_normalize_maps_spans_back_to_original() -> None:
    norm = normalize("A Randomized Controlled Trial in Knee Osteoarthritis")
    start = norm.text.index("randomized controlled trial")
    end = start + len("randomized controlled trial")
    original_start, original_end = norm.to_original_span(start, end)
    assert norm.original[original_start:original_end] == "Randomized Controlled Trial"


def test_normalize_nfkc_ligature_expansion_maps_back() -> None:
    norm = normalize("hepatic ﬁbrosis study")
    assert "fibrosis" in norm.text
    start = norm.text.index("fibrosis")
    original_start, original_end = norm.to_original_span(start, start + len("fibrosis"))
    assert norm.original[original_start:original_end] == "ﬁbrosis"


@pytest.mark.parametrize("variant", sorted(HYPHEN_VARIANTS))
def test_normalize_unifies_every_hyphen_variant(variant: str) -> None:
    norm = normalize(f"cross{variant}sectional")
    assert norm.text == "cross-sectional"


def test_normalize_collapses_whitespace_and_strips() -> None:
    norm = normalize("  Randomized\t\n  controlled\u00a0trial  ")
    assert norm.text == "randomized controlled trial"
    assert len(norm.orig_index) == len(norm.text)
    # first emitted char points at the first non-whitespace original character
    assert norm.orig_index[0] == 2


def test_normalize_preserves_negation_words() -> None:
    norm = normalize("No MSC therapy was not given without consent")
    for word in ("no", "not", "without"):
        assert word in norm.text.split()


def test_normalize_unifies_apostrophes() -> None:
    norm = normalize("Crohn\u2019s disease and the patient\u0060s history")
    assert norm.text == "crohn's disease and the patient's history"


def test_normalize_empty_text() -> None:
    norm = normalize("")
    assert norm.text == ""
    assert norm.original == ""
    assert norm.orig_index == ()


def test_normalize_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalize(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# tokenize()
# ---------------------------------------------------------------------------


def test_tokenize_word_rules_and_offsets() -> None:
    text = "Randomized-controlled trial (RCT) with MTX, 52 weeks, Crohn's disease, 5 mg."
    norm = normalize(text)
    tokens = tokenize(norm)
    assert all(isinstance(token, Token) for token in tokens)
    assert [token.text for token in tokens] == [
        "randomized-controlled",
        "trial",
        "rct",
        "with",
        "mtx",
        "52",
        "weeks",
        "crohn's",
        "disease",
        "mg",
    ]
    for token in tokens:
        assert norm.text[token.start : token.end] == token.text
        assert token.end - token.start >= 2
    assert "5" not in [token.text for token in tokens]


def test_tokenize_empty_text() -> None:
    assert tokenize(normalize("")) == []


# ---------------------------------------------------------------------------
# split_sentences()
# ---------------------------------------------------------------------------


def test_split_sentences_is_conservative_about_abbreviations_and_decimals() -> None:
    text = (
        "Knee OA, e.g. grade 3, was common vs. control. "
        "No. of patients was 52. The dose was 1.5 mg. "
        "The trial was randomized; outcomes were recorded. Was it effective? Yes."
    )
    norm = normalize(text)
    sentences = [norm.text[start:end] for start, end in split_sentences(norm)]
    assert sentences == [
        "knee oa, e.g. grade 3, was common vs. control.",
        "no. of patients was 52.",
        "the dose was 1.5 mg.",
        "the trial was randomized;",
        "outcomes were recorded.",
        "was it effective?",
        "yes.",
    ]


def test_split_sentences_keeps_final_sentence_without_punctuation() -> None:
    norm = normalize("First sentence. Second without punctuation")
    spans = split_sentences(norm)
    assert [norm.text[start:end] for start, end in spans] == [
        "first sentence.",
        "second without punctuation",
    ]


def test_split_sentences_covers_whole_text_contiguously() -> None:
    norm = normalize("One. Two! Three? Four; Five")
    spans = split_sentences(norm)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(norm.text)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end <= start  # only collapsed whitespace can be skipped
        assert norm.text[end:start].strip() == ""


def test_split_sentences_semicolon_is_a_separator() -> None:
    norm = normalize("randomized; controlled")
    assert [norm.text[start:end] for start, end in split_sentences(norm)] == [
        "randomized;",
        "controlled",
    ]


def test_split_sentences_forced_break_without_punctuation() -> None:
    norm = normalize("A title without final period An abstract sentence. Second one.")
    title_length = len(normalize("A title without final period").text)
    assert norm.text[title_length] == " "
    spans = split_sentences(norm, forced_breaks=(title_length,))
    assert [norm.text[start:end] for start, end in spans] == [
        "a title without final period",
        "an abstract sentence.",
        "second one.",
    ]


def test_split_sentences_forced_break_ignored_after_punctuation() -> None:
    norm = normalize("Knee OA. Abstract sentence.")
    boundary = len(normalize("Knee OA.").text)  # normalised offset of the space
    assert norm.text[boundary] == " "
    spans = split_sentences(norm, forced_breaks=(boundary,))
    assert [norm.text[start:end] for start, end in spans] == [
        "knee oa.",
        "abstract sentence.",
    ]


def test_split_sentences_empty_text() -> None:
    assert split_sentences(normalize("")) == []


def test_sentence_at_lookup() -> None:
    norm = normalize("First sentence. Second sentence.")
    spans = split_sentences(norm)
    assert sentence_at(spans, 0) == spans[0]
    assert sentence_at(spans, len(norm.text) - 1) == spans[-1]
    assert sentence_at([], 0) is None


# ---------------------------------------------------------------------------
# is_negated()
# ---------------------------------------------------------------------------


def _check_negation(text: str, phrase: str, window: int = 5) -> tuple[bool, bool]:
    norm = normalize(text)
    tokens = tokenize(norm)
    match_start = norm.text.index(phrase)
    return is_negated(norm, tokens, match_start, window_tokens_before=window)


def test_is_negated_detects_preceding_no() -> None:
    assert _check_negation("No MSC therapy was used in the control group", "msc therapy") == (
        True,
        False,
    )


def test_is_negated_detects_not_in_window() -> None:
    assert _check_negation("The study did not use MSC therapy", "msc therapy") == (True, False)


def test_is_negated_false_without_cue() -> None:
    assert _check_negation("MSC therapy was used", "msc therapy") == (False, False)


def test_is_negated_respects_window_length() -> None:
    text = "No patients received intra-articular MSC therapy"
    assert _check_negation(text, "msc therapy", window=5) == (True, False)
    # the cue is more than two tokens before the phrase: out of the local window
    assert _check_negation(text, "msc therapy", window=2) == (False, False)


def test_is_negated_not_only_exception() -> None:
    assert _check_negation(
        "The trial assessed not only MSC therapy but also usual care", "msc therapy"
    ) == (False, False)


def test_is_negated_attached_non_prefix() -> None:
    assert _check_negation("non-MSC therapy was evaluated", "msc therapy") == (True, False)
    # "non" separated by a space is not the documented attached-prefix rule, so
    # the negation is not asserted (a spaced "non" is not a lexicon cue either)
    assert _check_negation("non MSC therapy comparison", "msc therapy") == (False, False)


def test_is_negated_ambiguous_scope_with_comma() -> None:
    text = "patients did not receive placebo, MSC therapy was administered"
    assert _check_negation(text, "msc therapy") == (False, True)


def test_is_negated_ambiguous_scope_with_coordinator() -> None:
    text = "no placebo was given and MSC therapy was administered"
    assert _check_negation(text, "msc therapy") == (False, True)


def test_is_negated_contrast_connector_ends_scope() -> None:
    text = "not the placebo but MSC therapy produced results"
    assert _check_negation(text, "msc therapy") == (False, False)


def test_is_negated_sentence_boundary_ends_scope() -> None:
    text = "MSC therapy was not used. Patients with knee osteoarthritis improved."
    assert _check_negation(text, "knee osteoarthritis") == (False, False)


def test_is_negated_implicit_scope_boundary() -> None:
    norm = normalize("No prior treatment with MSC therapy")
    tokens = tokenize(norm)
    match_start = norm.text.index("msc therapy")
    assert is_negated(norm, tokens, match_start) == (True, False)
    boundary = norm.text.index("msc therapy")
    assert is_negated(norm, tokens, match_start, scope_boundaries=(boundary,)) == (
        False,
        False,
    )


def test_is_negated_double_negation_is_ambiguous() -> None:
    assert _check_negation("not without MSC therapy", "msc therapy") == (True, True)


def test_is_negated_validates_arguments() -> None:
    norm = normalize("MSC therapy was used")
    tokens = tokenize(norm)
    with pytest.raises(ValueError):
        is_negated(norm, tokens, 0, window_tokens_before=-1)
    with pytest.raises(ValueError):
        is_negated(norm, tokens, len(norm.text) + 1)
