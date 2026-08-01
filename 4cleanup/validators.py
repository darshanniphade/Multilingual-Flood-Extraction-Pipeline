"""Article validation rules and their machine-readable rejection reasons.

Validation runs *after* cleaning.  This ordering is deliberate: an article
padded to 120 words by advertising chrome is not a 120-word article, so the
word-count floor must be measured against the text that will actually be
written to disk.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Final, NamedTuple

import cleaner
import config


class SkipReason(StrEnum):
    """Enumerated reasons an article may be excluded from the output."""

    MALFORMED_JSON = "malformed_json"
    UNREADABLE_FILE = "unreadable_file"
    NOT_AN_OBJECT = "not_a_json_object"
    MISSING_TEXT = "missing_translated_text"
    EMPTY_TEXT = "empty_translated_text"
    EMPTY_TEXT_AFTER_CLEANING = "empty_translated_text_after_cleaning"
    MISSING_TITLE = "missing_translated_title"
    EMPTY_TITLE = "empty_translated_title"
    TOO_SHORT = "fewer_than_min_words"
    GARBLED_TEXT = "garbled_text"
    MISSING_ARTICLE_ID = "missing_article_id"
    WRITE_FAILED = "write_failed"
    DUPLICATE = "duplicate_text"


#: Description templates surfaced in the processing report.  Placeholders are
#: filled by :func:`describe_reason` so a run over raw ``title``/``text`` does
#: not report failures against ``translated_*`` field names it never read.
REASON_TEMPLATES: Final[dict[str, str]] = {
    SkipReason.MALFORMED_JSON: "File is not valid JSON.",
    SkipReason.UNREADABLE_FILE: "File could not be read from disk.",
    SkipReason.NOT_AN_OBJECT: "Top-level JSON value is not an object.",
    SkipReason.MISSING_TEXT: "Field '{text_field}' is absent or not a string.",
    SkipReason.EMPTY_TEXT: "Field '{text_field}' is empty or whitespace only.",
    SkipReason.EMPTY_TEXT_AFTER_CLEANING: (
        "Field '{text_field}' contained only boilerplate and was empty after cleaning."
    ),
    SkipReason.MISSING_TITLE: "Field '{title_field}' is absent or not a string.",
    SkipReason.EMPTY_TITLE: "Field '{title_field}' is empty after cleaning.",
    SkipReason.TOO_SHORT: "Cleaned '{text_field}' has fewer than {min_words} words.",
    SkipReason.GARBLED_TEXT: "Text is dominated by garbled syndication markup.",
    SkipReason.MISSING_ARTICLE_ID: "Field 'article_id' is absent and no fallback was derivable.",
    SkipReason.WRITE_FAILED: "Output file could not be written.",
    SkipReason.DUPLICATE: "Cleaned text is byte-identical to an earlier article.",
}


def describe_reason(
    reason: str,
    text_field: str = config.SOURCE_TEXT_FIELD,
    title_field: str = config.SOURCE_TITLE_FIELD,
    min_words: int = config.MIN_WORD_COUNT,
) -> str:
    """Render a reason code as prose naming the fields the run actually read.

    Args:
        reason: Reason code, normally a :class:`SkipReason` value.
        text_field: Source field the run read as the article body.
        title_field: Source field the run read as the article title.
        min_words: Post-cleaning word-count floor the run enforced.

    Returns:
        A human-readable description, or ``"Unspecified."`` for unknown codes.
    """
    template = REASON_TEMPLATES.get(reason)
    if template is None:
        return "Unspecified."
    return template.format(
        text_field=text_field, title_field=title_field, min_words=min_words
    )


class ValidationResult(NamedTuple):
    """Outcome of validating and cleaning one article.

    Attributes:
        ok: True when the article should be written to the output tree.
        reason: Rejection reason when ``ok`` is False, otherwise ``None``.
        article_id: Resolved article identifier.
        title: Cleaned title.
        text: Cleaned body text.
        word_count: Word count of ``text``.
    """

    ok: bool
    reason: SkipReason | None
    article_id: str
    title: str
    text: str
    word_count: int


def is_garbled(text: str) -> bool:
    """Report whether text is dominated by ROT47-garbled syndication markup.

    A small number of wire-service articles in the corpus are stored in an
    encoded form (``k^Am``, ``E96``) that renders them useless for NLP. The
    check counts encoding signatures relative to token count.

    Args:
        text: Cleaned text to inspect.

    Returns:
        True when the signature density exceeds the configured threshold.
    """
    words = len(text.split())
    if words == 0:
        return False
    hits = len(config.GARBLED_SIGNATURE.findall(text))
    return (hits / words) > config.GARBLED_TEXT_THRESHOLD


def _resolve_article_id(document: dict[str, Any], fallback: str) -> str:
    """Determine an article identifier, falling back to the filename stem.

    Args:
        document: Parsed source document.
        fallback: Identifier to use when the field is absent or unusable.

    Returns:
        A non-empty identifier string.
    """
    raw = document.get("article_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, int):
        return str(raw)
    return fallback


def validate_and_clean(
    document: dict[str, Any],
    fallback_id: str,
    min_word_count: int = config.MIN_WORD_COUNT,
    title_field: str = config.SOURCE_TITLE_FIELD,
    text_field: str = config.SOURCE_TEXT_FIELD,
) -> ValidationResult:
    """Clean an article and decide whether it qualifies for the output corpus.

    Checks are ordered cheapest-first so that obviously invalid documents are
    rejected before the full cleaning pipeline runs.

    Args:
        document: Parsed source document.
        fallback_id: Identifier used when ``article_id`` is missing, normally
            the source filename stem.
        min_word_count: Minimum tokens required after cleaning.
        title_field: Source field to read as the title. Defaults to the
            stage-3 name; raw scrape trees use ``"title"``.
        text_field: Source field to read as the body. Defaults to the stage-3
            name; raw scrape trees use ``"text"``.

    Returns:
        A :class:`ValidationResult` describing the outcome.
    """
    article_id = _resolve_article_id(document, fallback_id)

    raw_text = document.get(text_field)
    if not isinstance(raw_text, str):
        return ValidationResult(False, SkipReason.MISSING_TEXT, article_id, "", "", 0)
    if not raw_text.strip():
        return ValidationResult(False, SkipReason.EMPTY_TEXT, article_id, "", "", 0)

    raw_title = document.get(title_field)
    if not isinstance(raw_title, str):
        return ValidationResult(False, SkipReason.MISSING_TITLE, article_id, "", "", 0)

    title = cleaner.clean_title(raw_title)
    if not title:
        return ValidationResult(False, SkipReason.EMPTY_TITLE, article_id, "", "", 0)

    text = cleaner.clean_text(raw_text)
    if not text:
        return ValidationResult(
            False, SkipReason.EMPTY_TEXT_AFTER_CLEANING, article_id, title, "", 0
        )

    word_count = cleaner.count_words(text)
    if word_count < min_word_count:
        return ValidationResult(
            False, SkipReason.TOO_SHORT, article_id, title, text, word_count
        )

    if config.ENABLE_GARBLED_TEXT_CHECK and is_garbled(text):
        return ValidationResult(
            False, SkipReason.GARBLED_TEXT, article_id, title, text, word_count
        )

    return ValidationResult(True, None, article_id, title, text, word_count)
