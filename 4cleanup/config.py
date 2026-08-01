"""Central configuration for the flood-news cleaning pipeline.

Every tunable knob lives here so that behaviour can be adjusted without
touching pipeline logic.  The boilerplate rules are expressed as data
(ordered lists of compiled patterns) rather than code, which makes them
reviewable and extensible by non-programmers.

The pattern sets below were derived empirically from a 12,000-article random
sample of ``translated_articles`` -- see ``README.md`` for the measured hit
rates that justify each rule.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Patterns here are compiled by `regex`, not `re`. A regex-module pattern is
# not an instance of `re.Pattern`, so `regex.Pattern` is the accurate
# annotation despite linters suggesting the stdlib alias.
import regex
from regex import Pattern

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

INPUT_ROOT: Final[Path] = Path(r"C:\darsh\pipeline\data\translated_articles")
OUTPUT_ROOT: Final[Path] = Path(r"C:\darsh\pipeline\data\only_english")
REPORT_PATH: Final[Path] = OUTPUT_ROOT.parent / "processing_report.json"
LOG_PATH: Final[Path] = OUTPUT_ROOT.parent / "processing.log"

#: Fields copied to the output document.  Everything else is discarded.
#: The title and body keep whatever names they had in the source -- this stage
#: cleans text, it never renames fields -- so the second and third entries here
#: are the defaults, not a guarantee.  See SOURCE_TITLE_FIELD below.
OUTPUT_FIELDS: Final[tuple[str, ...]] = (
    "article_id",
    "translated_title",
    "translated_text",
)

# --------------------------------------------------------------------------
# Source field names
# --------------------------------------------------------------------------
# Which fields of the *input* document carry the title and body.  Stage 3
# (``translator/``) emits ``translated_*``, so those are the defaults.  Raw
# stage-0 scrape trees (e.g. ``H:\<year>``) instead carry ``title``/``text``
# and are read by passing ``--title-field title --text-field text``.
#
# These names are used for the output document too: cleaned values are written
# back under the key they were read from.

SOURCE_TITLE_FIELD: Final[str] = "translated_title"
SOURCE_TEXT_FIELD: Final[str] = "translated_text"

# --------------------------------------------------------------------------
# Validation thresholds
# --------------------------------------------------------------------------

#: Minimum number of whitespace-delimited tokens required *after* cleaning.
MIN_WORD_COUNT: Final[int] = 100

#: When True, articles whose text is dominated by ROT47-garbled syndication
#: markup (e.g. ``k^Am``, ``E96``) are rejected.  Measured at ~0.25% of the
#: corpus; disabled by default because it is not part of the required spec.
ENABLE_GARBLED_TEXT_CHECK: Final[bool] = False

#: Ratio of garble signatures per word above which text is deemed corrupt.
GARBLED_TEXT_THRESHOLD: Final[float] = 0.05

# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------

#: Files handed to a worker per IPC round-trip.  Batching is essential: at
#: ~233k files, per-file IPC dominates runtime.  200 amortises it to noise.
BATCH_SIZE: Final[int] = 200

#: Worker process count.  ``None`` -> os.cpu_count().
WORKER_COUNT: Final[int | None] = None

#: Number of batches queued ahead of the workers.  Bounds memory: only this
#: many batches of *paths* (never file contents) are resident at once.
QUEUE_DEPTH_BATCHES: Final[int] = 64

#: Indent output JSON for human inspection.  Set False for ~10% smaller files.
PRETTY_OUTPUT: Final[bool] = True

#: Cap on individual error/skip examples retained for the report.
MAX_LOGGED_EXAMPLES: Final[int] = 50


def resolve_worker_count() -> int:
    """Return the number of worker processes to spawn.

    Returns:
        Configured worker count, or all available CPU cores when unset.
    """
    if WORKER_COUNT is not None:
        return max(1, WORKER_COUNT)
    return max(1, os.cpu_count() or 1)


# --------------------------------------------------------------------------
# Unicode / text normalisation
# --------------------------------------------------------------------------

#: Unicode normalisation form.  NFKC folds compatibility characters (full-width
#: latin, ligatures, U+2026 -> "...") which is the usual choice for NLP corpora.
UNICODE_NORMAL_FORM: Final[str] = "NFKC"

#: Ellipsis policy: "keep" normalises runs of 3+ dots to exactly "...";
#: "collapse" reduces them to a single ".".  "keep" is the default because
#: ellipses are semantically meaningful inside quoted speech (7.4% of articles).
ELLIPSIS_POLICY: Final[str] = "keep"

#: Curly/typographic quotes -> ASCII equivalents.
QUOTE_TRANSLATION: Final[dict[int, str]] = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',
    0x2039: "'", 0x203A: "'",
    0x00AB: '"', 0x00BB: '"',
    0x2032: "'", 0x2033: '"',
    0xFF02: '"', 0xFF07: "'",
    0x301D: '"', 0x301E: '"',
}

#: Dash-like characters -> ASCII hyphen-minus.
DASH_TRANSLATION: Final[dict[int, str]] = {
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-",
    0x2014: "-", 0x2015: "-", 0x2212: "-",
    0xFE58: "-", 0xFE63: "-", 0xFF0D: "-",
}

#: Zero-width, bidirectional-control and other invisible formatting codepoints.
INVISIBLE_CHARS: Final[Pattern[str]] = regex.compile(
    "["
    "­"              # SOFT HYPHEN
    "᠎"              # MONGOLIAN VOWEL SEPARATOR
    "​-‏"       # ZERO WIDTH SPACE .. RIGHT-TO-LEFT MARK
    "‪-‮"       # bidirectional embedding/override controls
    "⁠-⁤"       # WORD JOINER .. INVISIBLE PLUS
    "⁦-⁯"       # bidirectional isolates and deprecated formatters
    "﻿"              # ZERO WIDTH NO-BREAK SPACE / BOM
    "￹-￻"       # interlinear annotation controls
    "]"
)

#: C0/C1 control characters, excluding tab and newline which are handled by the
#: whitespace normaliser.
CONTROL_CHARS: Final[Pattern[str]] = regex.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"
)

# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

#: <script>/<style> elements including their contents.
HTML_SCRIPT_STYLE: Final[Pattern[str]] = regex.compile(
    r"<(script|style)\b[^>]*>.*?</\1\s*>", regex.IGNORECASE | regex.DOTALL
)

#: HTML/XML tags, comments and CDATA sections.  Deliberately conservative: the
#: element-name anchor prevents mangling prose such as "levels < 5 m > normal".
HTML_TAG: Final[Pattern[str]] = regex.compile(
    r"<(?:/?[A-Za-z][A-Za-z0-9:._-]*(?:\s[^<>]*)?/?|!--.*?--|!\[CDATA\[.*?\]\]|![A-Za-z][^<>]*)>",
    regex.DOTALL,
)

# --------------------------------------------------------------------------
# URLs
# --------------------------------------------------------------------------

#: Absolute URLs and bare ``www.`` hosts.  Trailing sentence punctuation is
#: intentionally excluded from the match so surrounding prose stays intact.
URL_PATTERN: Final[Pattern[str]] = regex.compile(
    r"""(?xi)
    \b
    (?:
        (?:https?|ftp)://          # scheme-qualified
      | www\.                      # or bare www host
    )
    [^\s<>"'()\[\]{}]+             # body
    (?<![.,;:!?'"])                # do not swallow trailing punctuation
    """
)

# --------------------------------------------------------------------------
# Boilerplate rules
# --------------------------------------------------------------------------
# Rules are applied in the order listed within each group, and the groups are
# applied in the order: standalone lines -> line markers -> CTA lines ->
# copyright -> inline phrases.


def _line_marker(*alternatives: str) -> Pattern[str]:
    """Compile a rule that deletes a marker and the remainder of its line.

    ``READ:``-style markers introduce an interstitial cross-promotion whose
    *whole line* is boilerplate ("READ: Closures and Delays").  Deleting only
    the token would strand the headline fragment in the corpus, so the match
    extends to the end of the line.

    Args:
        *alternatives: Literal marker strings (regex-escaped by the caller's
            choice of raw text; pass pre-escaped fragments).

    Returns:
        Compiled pattern matching the marker through end-of-line.
    """
    body = "|".join(alternatives)
    return regex.compile(rf"(?im)^[ \t]*[\[(]?\s*(?:{body})[^\n]*$")


def _standalone_line(*alternatives: str) -> Pattern[str]:
    """Compile a rule that deletes a line consisting solely of boilerplate.

    Args:
        *alternatives: Literal alternatives forming the line body.

    Returns:
        Compiled pattern matching the entire line, brackets optional.
    """
    body = "|".join(alternatives)
    return regex.compile(rf"(?im)^[ \t]*[\[(]?\s*(?:{body})\s*[\])]?[ \t]*[.:!]?[ \t]*$")


def _cta(*alternatives: str) -> Pattern[str]:
    """Compile a call-to-action rule spanning to the end of the line.

    Anchored at line start or immediately after sentence-ending punctuation so
    that mid-sentence prose usage ("residents had to sign up for alerts") is
    not destroyed.

    Args:
        *alternatives: Literal alternatives introducing the CTA.

    Returns:
        Compiled pattern matching the CTA through end-of-line.
    """
    body = "|".join(alternatives)
    return regex.compile(rf"(?im)(?:^|(?<=[.!?]\s))[ \t]*(?:{body})[^\n]*$")


#: Lines whose entire content is chrome.  Dropped wholesale.
STANDALONE_LINE_RULES: Final[tuple[Pattern[str], ...]] = (
    _standalone_line(
        r"ADVERTISEMENT",
        r"Advertisements?",
        r"Sponsored(?:\s+Content|\s+Links?)?",
        r"Article continues below this ad",
        r"Continue\s+Reading",
        r"Read\s+More",
        r"Loading\.{2,}",
        r"Video\s+Player",
        r"SUMMARY",
        r"Share\s+this\s+story",
        r"Cookie\s+Policy",
        r"Privacy\s+Policy",
        r"Terms\s+of\s+Service",
        r"Follow\s+us",
        r"Follow\s+Rappler",
        r"Photo\s+Credit",
        r"Image\s+Credit",
    ),
)

#: Cross-promotion markers: delete marker plus the rest of the line.
LINE_MARKER_RULES: Final[tuple[Pattern[str], ...]] = (
    # Parenthesised cross-promotions embedded mid-sentence, e.g. Rappler's
    # "(READ: Help people affected by flash floods)". Bounded by the closing
    # bracket, so this cannot run away across a paragraph.
    # Deliberately case-sensitive: the corpus writes cross-promos as "READ:"
    # or "Related:", whereas the lowercase "(read: in other words)" idiom is
    # ordinary prose and must survive.
    regex.compile(
        r"[\(\[]\s*(?:ALSO\s+|Also\s+|MUST\s+|Must\s+|SEE\s+ALSO\s+|See\s+[Aa]lso\s+)?"
        r"(?:READ(?:\s+MORE)?|Read(?:\s+More)?|RELATED|Related)\s*:[^)\]\n]{0,400}[\)\]]"
    ),
    _line_marker(
        r"(?:ALSO\s+|MUST\s+|SEE\s+ALSO\s+)?READ\s*:",
        r"READ\s+MORE\s*:",
        r"RELATED\s*(?:STORIES|ARTICLES|TOPICS|COVERAGE)?\s*:",
    ),
)

#: Promotional calls to action running to the end of their line.
CTA_LINE_RULES: Final[tuple[Pattern[str], ...]] = (
    _cta(
        r"Subscribe\b",
        r"Sign\s+up\b",
        r"Follow\s+us\b",
        r"Follow\s+Rappler\b",
        r"Click\s+here\b",
        r"Share\s+this\s+story\b",
        r"Continue\s+Reading\b",
        r"Read\s+More\b",
    ),
)

#: Legal / rights boilerplate.  The sentence carrying "All rights reserved" and
#: the standard wire-service redistribution notice are removed in full.
COPYRIGHT_RULES: Final[tuple[Pattern[str], ...]] = (
    # "This material may not be published, broadcast, rewritten or redistributed."
    regex.compile(
        r"(?i)This material may not be published,?\s*broadcast,?\s*rewritten"
        r"(?:\s*,?\s*or\s+redistributed)?[^.\n]*\.?"
    ),
    # A sentence ending in "All rights reserved."
    regex.compile(r"(?i)(?:(?<=^)|(?<=[.!?]\s)|(?<=\n))[^.\n]*All rights reserved[^.\n]*\.?"),
    # Residual copyright lines: "Copyright 2025 Associated Press", "© ANTARA 2021".
    # "TM" is listed because NFKC folds U+2122 to the two letters before this
    # rule ever runs, so a literal U+2122 alternative alone would never match.
    regex.compile(r"(?im)^[ \t]*(?:\(c\)|\(r\)|[©®™]|TM(?=[\s&])|Copyright)[^\n]*$"),
    regex.compile(r"(?i)Copyright\s*©?\s*\d{4}[^.\n]*\.?"),
    regex.compile(r"[™©®]\s*&?\s*\d{0,4}"),
)

#: Fixed phrases removed wherever they appear, including mid-sentence.
INLINE_PHRASE_RULES: Final[tuple[Pattern[str], ...]] = (
    regex.compile(
        r"(?i)This is AI[- ]generated summari[sz]ation,?\s*which may have errors\.?"
    ),
    regex.compile(r"(?i)For context,?\s*always refer to the full article\.?"),
    regex.compile(r"(?i)How does this make you feel\s*\?"),
    regex.compile(r"(?i)\[\s*advertisement\s*\]"),
    regex.compile(r"(?i)\badvertisement\b"),
    regex.compile(r"(?i)\bcookie policy\b"),
    regex.compile(r"(?i)\bprivacy policy\b"),
    regex.compile(r"(?i)\bterms of service\b"),
    regex.compile(r"(?i)\ball rights reserved\b\.?"),
    regex.compile(r"(?i)\bphoto credit\s*:?"),
    regex.compile(r"(?i)\bimage credit\s*:?"),
    regex.compile(r"(?i)\bvideo player\b"),
    regex.compile(r"(?i)\bloading\.{2,}"),
    regex.compile(r"(?i)\bsponsored\b"),
)

#: Full ordered rule chain consumed by :func:`cleaner.strip_boilerplate`.
BOILERPLATE_RULE_CHAIN: Final[tuple[Pattern[str], ...]] = (
    STANDALONE_LINE_RULES
    + LINE_MARKER_RULES
    + CTA_LINE_RULES
    + COPYRIGHT_RULES
    + INLINE_PHRASE_RULES
)

# --------------------------------------------------------------------------
# Punctuation / whitespace repair
# --------------------------------------------------------------------------

#: Runs of 3+ dots (post-NFKC this includes U+2026 ellipses).
ELLIPSIS_RUN: Final[Pattern[str]] = regex.compile(r"\.{3,}")

#: Exactly two dots not part of a longer run -> single period.
DOUBLE_DOT: Final[Pattern[str]] = regex.compile(r"(?<!\.)\.\.(?!\.)")

#: Repeated identical punctuation (",,", ";;", "!!", "??", "::", "--").
REPEATED_PUNCT: Final[Pattern[str]] = regex.compile(r"([,;:!?\-])\1+")

#: Mixed comma/period debris left behind by boilerplate excision (", ." -> ".").
ORPHAN_PUNCT_RUN: Final[Pattern[str]] = regex.compile(r"[ \t]*([,;:])[ \t]*(?=[.!?])")

#: Whitespace stranded before sentence punctuation by our own deletions.
SPACE_BEFORE_PUNCT: Final[Pattern[str]] = regex.compile(r"[ \t]+([,.;:!?])")

#: Maximum convergence passes for :func:`cleaner.fix_punctuation`.
#: Closing up "? ?" into "??" can expose a duplicate that the collapse rule
#: has already run past, so the repair stage iterates until stable. Two passes
#: suffice in practice; the cap is a guard against pathological input.
MAX_PUNCT_PASSES: Final[int] = 5

#: A line reduced to nothing but punctuation/brackets after cleaning.
PUNCT_ONLY_LINE: Final[Pattern[str]] = regex.compile(
    r"(?m)^[ \t]*[\p{P}\p{S}]{1,4}[ \t]*$"
)

#: Empty bracket pairs left by inline phrase removal.
EMPTY_BRACKETS: Final[Pattern[str]] = regex.compile(r"[\(\[\{]\s*[\)\]\}]")

#: Horizontal whitespace runs.
HORIZONTAL_WS: Final[Pattern[str]] = regex.compile(r"[^\S\n]+")

#: Three or more newlines -> exactly one blank line.
BLANK_LINE_RUN: Final[Pattern[str]] = regex.compile(r"\n{3,}")

#: ROT47 garbled-syndication signatures (see ENABLE_GARBLED_TEXT_CHECK).
GARBLED_SIGNATURE: Final[Pattern[str]] = regex.compile(
    r"[A-Za-z0-9][@\[\]^{}|~`][A-Za-z0-9]|k\^Am|kAm"
)


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Runtime settings resolved from CLI arguments and module defaults.

    Attributes:
        input_root: Directory tree scanned for ``*.json`` articles.
        output_root: Directory tree that mirrors ``input_root``.
        report_path: Destination of ``processing_report.json``.
        log_path: Destination of the plain-text run log.
        workers: Number of worker processes.
        batch_size: Files per IPC batch.
        min_word_count: Post-cleaning word-count floor.
        pretty: Whether output JSON is indented.
        deduplicate: Whether exact-duplicate removal is active.
        count_first: Whether to pre-scan for an exact progress-bar total.
        limit: Optional cap on files processed (for smoke tests).
        title_field: Source field read as the article title.
        text_field: Source field read as the article body.
    """

    input_root: Path = INPUT_ROOT
    output_root: Path = OUTPUT_ROOT
    report_path: Path = REPORT_PATH
    log_path: Path = LOG_PATH
    workers: int = field(default_factory=resolve_worker_count)
    batch_size: int = BATCH_SIZE
    min_word_count: int = MIN_WORD_COUNT
    pretty: bool = PRETTY_OUTPUT
    deduplicate: bool = True
    count_first: bool = True
    limit: int | None = None
    title_field: str = SOURCE_TITLE_FIELD
    text_field: str = SOURCE_TEXT_FIELD
