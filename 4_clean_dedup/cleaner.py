"""Research-grade text normalisation for translated news articles.

The cleaning pipeline is a fixed, documented sequence of idempotent stages.
Order is load-bearing and is spelled out in :func:`clean_text`.

Two entry points are exposed:

* :func:`clean_text`  -- full pipeline, used for ``translated_text``.
* :func:`clean_title` -- structural normalisation only, used for
  ``translated_title``.  Boilerplate excision is deliberately skipped for
  titles: a headline such as "READ: Dam overflows" is a legitimate title, and
  line-level rules would delete it entirely.
"""

from __future__ import annotations

import html
import unicodedata
from typing import Final

import config

#: Translation table applied in a single pass for quotes and dashes.
_CHAR_TRANSLATION: Final[dict[int, str]] = {
    **config.QUOTE_TRANSLATION,
    **config.DASH_TRANSLATION,
}


def decode_entities(text: str) -> str:
    """Decode HTML character references, including double-encoded ones.

    ``&amp;lt;`` occurs in scraped corpora where an already-escaped document
    was escaped a second time; a single ``unescape`` would leave ``&lt;``.
    A second pass runs only when the first changed something and the result
    still contains a reference marker, so well-formed text costs one pass.

    Args:
        text: Raw text possibly containing HTML entities.

    Returns:
        Text with character references resolved.
    """
    if "&" not in text:
        return text
    decoded = html.unescape(text)
    if "&" in decoded and decoded != text:
        decoded = html.unescape(decoded)
    return decoded


def strip_html(text: str) -> str:
    """Remove HTML/XML markup, discarding ``<script>``/``<style>`` bodies.

    Args:
        text: Text possibly containing markup.

    Returns:
        Text with tags removed. Tag positions become spaces so that
        ``a<br>b`` does not become ``ab``.
    """
    if "<" not in text:
        return text
    text = config.HTML_SCRIPT_STYLE.sub(" ", text)
    return config.HTML_TAG.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """Apply Unicode normalisation as configured.

    NFKC folds compatibility forms -- full-width Latin, ligatures, and
    U+2026 HORIZONTAL ELLIPSIS into three dots -- which the punctuation stage
    then normalises consistently.

    Args:
        text: Text to normalise.

    Returns:
        Normalised text.
    """
    return unicodedata.normalize(config.UNICODE_NORMAL_FORM, text)


def remove_invisible(text: str) -> str:
    """Delete zero-width, bidirectional-control and control characters.

    Applied after normalisation because NFKC neither removes nor introduces
    these codepoints.

    Args:
        text: Text to filter.

    Returns:
        Text without invisible formatting characters.
    """
    text = config.INVISIBLE_CHARS.sub("", text)
    return config.CONTROL_CHARS.sub("", text)


def normalize_symbols(text: str) -> str:
    """Fold typographic quotes and dash variants to ASCII equivalents.

    Args:
        text: Text to fold.

    Returns:
        Text with straight quotes and hyphen-minus dashes.
    """
    return text.translate(_CHAR_TRANSLATION)


def remove_urls(text: str) -> str:
    """Delete absolute URLs and bare ``www.`` hosts.

    Args:
        text: Text possibly containing links.

    Returns:
        Text with links removed.
    """
    if "http" not in text and "www." not in text and "ftp" not in text:
        return text
    return config.URL_PATTERN.sub(" ", text)


def strip_boilerplate(text: str) -> str:
    """Apply the ordered boilerplate rule chain from :mod:`config`.

    Rules run in dependency order: whole-line chrome first, then
    marker-to-end-of-line cross-promotions, calls to action, legal notices,
    and finally fixed inline phrases. Earlier groups remove the structural
    context that later groups would otherwise have to reason about.

    Args:
        text: Text to strip.

    Returns:
        Text with boilerplate removed. Removals leave whitespace that the
        punctuation and whitespace stages subsequently repair.
    """
    for rule in config.BOILERPLATE_RULE_CHAIN:
        text = rule.sub(" ", text)
    return text


def _fix_punctuation_once(text: str) -> str:
    """Apply one pass of the punctuation repair rules.

    Args:
        text: Text to repair.

    Returns:
        Text after a single pass; may still contain artefacts newly exposed
        by the pass itself.
    """
    # Runs of 3+ dots are resolved first. A surviving "..." is then immune to
    # the remaining rules: DOUBLE_DOT's lookarounds exclude adjacent dots, and
    # REPEATED_PUNCT's character class deliberately omits ".".
    ellipsis_replacement = "." if config.ELLIPSIS_POLICY == "collapse" else "..."
    text = config.ELLIPSIS_RUN.sub(ellipsis_replacement, text)
    text = config.DOUBLE_DOT.sub(".", text)
    text = config.SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = config.REPEATED_PUNCT.sub(r"\1", text)
    text = config.EMPTY_BRACKETS.sub(" ", text)
    text = config.ORPHAN_PUNCT_RUN.sub("", text)
    return text


def fix_punctuation(text: str) -> str:
    """Repair duplicated and orphaned punctuation, iterating until stable.

    Handles the ``,,`` / ``;;`` / ``..`` artefacts present in the source data
    as well as debris created by boilerplate excision (``", ."`` -> ``"."``).
    Ellipsis handling follows :data:`config.ELLIPSIS_POLICY`.

    A single pass is not sufficient. Closing the gap in ``"? ?"`` yields
    ``"??"``, but the duplicate-collapse rule has already scanned past that
    position, so the duplicate survives to the next call -- which is precisely
    how idempotence was being broken. The stage therefore runs to a fixed
    point, bounded by :data:`config.MAX_PUNCT_PASSES`.

    Args:
        text: Text to repair.

    Returns:
        Text with punctuation normalised and stable under further passes.
    """
    for _ in range(config.MAX_PUNCT_PASSES):
        repaired = _fix_punctuation_once(text)
        if repaired == text:
            break
        text = repaired
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse spaces, drop punctuation-only lines and trim.

    Args:
        text: Text to normalise.

    Returns:
        Text with single spaces, at most one blank line between paragraphs,
        no leading/trailing whitespace on any line, and no residual lines
        consisting only of stray punctuation left by boilerplate removal.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = config.HORIZONTAL_WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = config.PUNCT_ONLY_LINE.sub("", text)
    text = config.BLANK_LINE_RUN.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Run the full cleaning pipeline over an article body.

    Stage order, and why:

    1. **Decode entities** -- so ``&lt;p&gt;`` becomes a real tag for stage 2.
    2. **Strip HTML** -- before normalisation, so tag internals never reach it.
    3. **Normalise Unicode** (NFKC).
    4. **Remove invisible characters** -- NFKC preserves them.
    5. **Fold quotes and dashes** to ASCII.
    6. **Remove URLs** -- before boilerplate, so link-only chrome collapses to
       an empty line that the line rules then delete.
    7. **Strip boilerplate.**
    8. **Fix punctuation** -- repairs both source artefacts and stage-7 debris.
    9. **Normalise whitespace** -- last, so every prior stage may leave loose
       spacing behind.

    The function is idempotent: ``clean_text(clean_text(x)) == clean_text(x)``.

    Args:
        text: Raw ``translated_text`` value.

    Returns:
        Cleaned text, or the empty string when ``text`` is not a non-empty
        string.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = decode_entities(text)
    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_invisible(text)
    text = normalize_symbols(text)
    text = remove_urls(text)
    text = strip_boilerplate(text)
    text = fix_punctuation(text)
    return normalize_whitespace(text)


def clean_title(title: str) -> str:
    """Normalise a headline without applying boilerplate excision.

    Titles receive stages 1-6 and 8-9 of :func:`clean_text`. The boilerplate
    stage is skipped because its line-level rules match whole lines, and a
    title is a single line -- "READ: Dam overflows" would be erased entirely
    rather than cleaned.

    Args:
        title: Raw ``translated_title`` value.

    Returns:
        Cleaned single-line title, or the empty string when ``title`` is not a
        non-empty string.
    """
    if not isinstance(title, str) or not title:
        return ""
    title = decode_entities(title)
    title = strip_html(title)
    title = normalize_unicode(title)
    title = remove_invisible(title)
    title = normalize_symbols(title)
    title = remove_urls(title)
    title = fix_punctuation(title)
    return normalize_whitespace(title).replace("\n", " ").strip()


def count_words(text: str) -> int:
    """Count whitespace-delimited tokens.

    Args:
        text: Text to measure.

    Returns:
        Number of tokens.
    """
    return len(text.split())
