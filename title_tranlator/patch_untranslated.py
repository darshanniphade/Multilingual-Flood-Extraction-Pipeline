"""Re-translate the titles that the main run copied through untranslated.

The main run trusts each record's declared ``language`` field. A large slice of
the 2023 crawl declares ``"language": "en"`` on titles that are plainly not
English (395 of a 400-title sample), so those never reached language detection:
``classify()`` mapped "en" -> ``eng_Latn``, ``process()`` saw the code equal the
target language, and the title was filed as "already English" and copied. A
second, smaller group has no declared language and a title shorter than
``detect_min_chars``, so detection was skipped for them too.

This program repairs those records without redoing the 2.95M-title run. It
re-reads only ``translated_titles.jsonl``, finds the rows where
``original_title == translated_title`` and the text is not actually English,
re-detects the source language from the text itself, and re-translates just
those on the GPU.

Source language is resolved from the *script* first and ``langdetect`` second,
which is the opposite of what a general-purpose pipeline would do, because the
script is unforgeable evidence here: a title written in Han characters cannot be
English no matter what the record claims, and short CJK headlines (``时政要闻``)
are exactly the ones langdetect gives up on. langdetect still decides between
languages that share a script (ru/uk/bg, ar/fa/ur, hi/mr/ne), but its answer is
discarded when it contradicts the script the title is actually written in.

Two phases, so the repair never races a downstream stage reading the file:

    # 1. find, translate, and save corrections to a separate patch file
    python patch_untranslated.py

    # 2. later, when nothing is reading the dataset, merge them in
    python patch_untranslated.py --apply

Phase 1 only reads. Phase 2 rewrites ``translated_titles.jsonl`` via a temp file
and an atomic swap, keeping the previous copy as ``.bak``.

Run with the project virtual environment, e.g.::

    C:\\darsh\\AI_MODELS\\translator_env\\Scripts\\python.exe patch_untranslated.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

# Import the main program for its Config/Translator. It also sets the HF offline
# environment variables at module scope, which must happen before transformers
# is imported - Translator does that lazily, so importing it here is safe.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_titles import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    Config,
    Stats,
    Translator,
    load_config,
    to_flores,
)

# Default name of the corrections file, written next to the dataset it repairs.
PATCH_FILENAME = "translated_titles_patch.jsonl"

# The scan is by far the slow half of this program: finding the ~44k broken rows
# means running langdetect over every one of the ~851k titles the main run
# copied through, which takes the better part of an hour. Its result is cached
# so that re-running - in particular flipping --include-latin - costs seconds
# instead of repeating the whole sweep. Both buckets are cached regardless of
# which one this run intends to translate, so the flag never forces a rescan.
TARGETS_FILENAME = "translated_titles_targets.jsonl"

# A Latin-script title is only re-translated when langdetect is at least this
# sure it is not English, and the title is at least this long. Latin-script
# headlines are full of proper nouns that make langdetect guess wildly on short
# input, and a false positive here would corrupt a title that is currently
# correct - so this bucket stays off unless --include-latin is passed.
LATIN_MIN_CONFIDENCE = 0.97
LATIN_MIN_CHARS = 20


# --------------------------------------------------------------------------- #
# Script detection
# --------------------------------------------------------------------------- #

# Unicode blocks that identify a writing system, in the order they are tested.
# Only the ranges that actually appear in this dataset are listed; anything
# alphabetic that matches none of them is counted as Latin.
SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("kana",       ((0x3040, 0x309F), (0x30A0, 0x30FF))),
    ("hangul",     ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))),
    ("han",        ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))),
    ("cyrillic",   ((0x0400, 0x04FF), (0x0500, 0x052F))),
    ("arabic",     ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF))),
    ("devanagari", ((0x0900, 0x097F),)),
    ("bengali",    ((0x0980, 0x09FF),)),
    ("tamil",      ((0x0B80, 0x0BFF),)),
    ("telugu",     ((0x0C00, 0x0C7F),)),
    ("kannada",    ((0x0C80, 0x0CFF),)),
    ("malayalam",  ((0x0D00, 0x0D7F),)),
    ("gujarati",   ((0x0A80, 0x0AFF),)),
    ("gurmukhi",   ((0x0A00, 0x0A7F),)),
    ("sinhala",    ((0x0D80, 0x0DFF),)),
    ("greek",      ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("hebrew",     ((0x0590, 0x05FF),)),
    ("thai",       ((0x0E00, 0x0E7F),)),
    ("lao",        ((0x0E80, 0x0EFF),)),
    ("khmer",      ((0x1780, 0x17FF),)),
    ("myanmar",    ((0x1000, 0x109F),)),
    ("georgian",   ((0x10A0, 0x10FF),)),
    ("armenian",   ((0x0530, 0x058F),)),
    ("ethiopic",   ((0x1200, 0x137F),)),
)

# Where a script alone is enough to pick a FLORES-200 code. Scripts shared by
# several languages (Cyrillic, Arabic, Devanagari) resolve to their most common
# member only as a last resort, when langdetect has failed outright.
SCRIPT_TO_FLORES: dict[str, str] = {
    "kana": "jpn_Jpan",
    "hangul": "kor_Hang",
    "han": "zho_Hans",
    "greek": "ell_Grek",
    "hebrew": "heb_Hebr",
    "thai": "tha_Thai",
    "lao": "lao_Laoo",
    "khmer": "khm_Khmr",
    "myanmar": "mya_Mymr",
    "georgian": "kat_Geor",
    "armenian": "hye_Armn",
    "ethiopic": "amh_Ethi",
    "bengali": "ben_Beng",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "kannada": "kan_Knda",
    "malayalam": "mal_Mlym",
    "gujarati": "guj_Gujr",
    "gurmukhi": "pan_Guru",
    "sinhala": "sin_Sinh",
    "cyrillic": "rus_Cyrl",
    "arabic": "arb_Arab",
    "devanagari": "hin_Deva",
}

# The script half of a FLORES-200 code that each observed script may legitimately
# produce. Used to throw out a langdetect answer that contradicts the script the
# title is actually written in - the very mistake this program exists to repair.
SCRIPT_TO_FLORES_SUFFIX: dict[str, frozenset[str]] = {
    "kana": frozenset({"Jpan"}),
    "hangul": frozenset({"Hang"}),
    "han": frozenset({"Hans", "Hant", "Jpan"}),
    "cyrillic": frozenset({"Cyrl"}),
    "arabic": frozenset({"Arab"}),
    "devanagari": frozenset({"Deva"}),
    "bengali": frozenset({"Beng"}),
    "tamil": frozenset({"Taml"}),
    "telugu": frozenset({"Telu"}),
    "kannada": frozenset({"Knda"}),
    "malayalam": frozenset({"Mlym"}),
    "gujarati": frozenset({"Gujr"}),
    "gurmukhi": frozenset({"Guru"}),
    "sinhala": frozenset({"Sinh"}),
    "greek": frozenset({"Grek"}),
    "hebrew": frozenset({"Hebr"}),
    "thai": frozenset({"Thai"}),
    "lao": frozenset({"Laoo"}),
    "khmer": frozenset({"Khmr"}),
    "myanmar": frozenset({"Mymr"}),
    "georgian": frozenset({"Geor"}),
    "armenian": frozenset({"Armn"}),
    "ethiopic": frozenset({"Ethi"}),
    "latin": frozenset({"Latn"}),
}

_SCRIPT_CACHE: dict[int, str] = {}


def _script_of_char(char: str) -> Optional[str]:
    """Name the writing system of one character, or None if it is not a letter."""
    if not char.isalpha():
        return None
    code = ord(char)
    cached = _SCRIPT_CACHE.get(code)
    if cached is not None:
        return cached
    found = "latin"
    for name, ranges in SCRIPT_RANGES:
        if any(low <= code <= high for low, high in ranges):
            found = name
            break
    _SCRIPT_CACHE[code] = found
    return found


def dominant_script(text: str) -> tuple[str, float]:
    """Return the most common writing system in ``text`` and its share of letters.

    Titles are routinely mixed - a Chinese headline quoting an English brand, a
    Greek one with a Latin acronym - so the decision is made on the majority
    script rather than on the presence of any single character.
    """
    counts: Counter = Counter()
    for char in text:
        script = _script_of_char(char)
        if script is not None:
            counts[script] += 1
    total = sum(counts.values())
    if not total:
        return "none", 0.0
    # Kana settles Japanese vs Chinese outright: a title holding any kana is
    # Japanese even when Han characters outnumber it, which they usually do.
    if counts.get("kana"):
        return "kana", counts["kana"] / total
    script, count = counts.most_common(1)[0]
    return script, count / total


# --------------------------------------------------------------------------- #
# Language resolution for the affected rows
# --------------------------------------------------------------------------- #


class Detector:
    """Resolves a source FLORES-200 code from the title text alone.

    This runs in-process rather than on the main program's pool of detection
    processes. It is not cheap - every title the main run copied through has to
    be checked to find the broken ones, which is ~851k langdetect calls - but it
    is paid once and cached (see ``TARGETS_FILENAME``), so a second run reuses
    the result instead of repeating the sweep.
    """

    def __init__(self) -> None:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0
        self._detect_langs = detect_langs
        self.by_source: Counter = Counter()

    def _langdetect(self, text: str) -> tuple[Optional[str], float]:
        try:
            ranked = self._detect_langs(text)
        except Exception:  # noqa: BLE001 - undetectable titles are not errors
            return None, 0.0
        if not ranked:
            return None, 0.0
        best = ranked[0]
        return best.lang, float(best.prob)

    def resolve(self, title: str) -> Optional[str]:
        """FLORES code to translate this title from, or None to leave it alone."""
        text = title.strip()
        if not text:
            return None
        script, share = dominant_script(text)
        if script in ("none", "latin"):
            return None  # Latin-script rows are handled by resolve_latin().

        guess, _prob = self._langdetect(text)
        flores = to_flores(guess)
        allowed = SCRIPT_TO_FLORES_SUFFIX.get(script, frozenset())
        if flores is not None and flores.split("_")[-1] in allowed:
            self.by_source[f"{script}:langdetect"] += 1
            return flores

        # langdetect either failed or returned a language written in a different
        # script than the title - fall back to what the script itself proves.
        fallback = SCRIPT_TO_FLORES.get(script)
        if fallback is not None:
            self.by_source[f"{script}:script"] += 1
        return fallback

    def resolve_latin(self, title: str) -> Optional[str]:
        """Same, for Latin-script titles - deliberately far more conservative."""
        text = title.strip()
        if len(text) < LATIN_MIN_CHARS:
            return None
        guess, prob = self._langdetect(text)
        if guess is None or guess == "en" or prob < LATIN_MIN_CONFIDENCE:
            return None
        flores = to_flores(guess)
        if flores is None or not flores.endswith("_Latn"):
            return None
        self.by_source[f"latin:{guess}"] += 1
        return flores


# --------------------------------------------------------------------------- #
# Phase 1 - find and translate
# --------------------------------------------------------------------------- #


def iter_records(path: Path) -> Iterator[dict]:
    """Stream the output file, skipping anything unparseable."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield record


def find_targets(path: Path, detector: Detector) -> tuple[list[dict], int]:
    """Collect every row that needs re-translating, in both buckets.

    Rows are tagged ``bucket`` = "script" (non-Latin, safe to fix outright) or
    "latin" (Latin-script, only translated when the caller opts in). Both are
    returned so the caller can cache them together and switch between them later
    without paying for another sweep.
    """
    targets: list[dict] = []
    scanned = 0

    bar = tqdm(desc="[scan]", unit="rec", unit_scale=True, dynamic_ncols=True)
    for record in iter_records(path):
        scanned += 1
        bar.update(1)
        original = record.get("original_title") or ""
        translated = record.get("translated_title") or ""
        # Only rows the main run left untouched can be mistakes; anything the
        # model already rewrote was translated from a language it was given.
        if not original.strip() or original != translated:
            continue

        flores = detector.resolve(original)
        bucket = "script"
        if flores is None:
            flores = detector.resolve_latin(original)
            bucket = "latin"
        if flores is None:
            continue
        targets.append(
            {
                "article_id": str(record["article_id"]),
                "title": original,
                "flores": flores,
                "bucket": bucket,
            }
        )
    bar.close()
    return targets, scanned


def save_targets(path: Path, targets: list[dict], scanned: int) -> None:
    """Cache the scan so a re-run does not repeat the ~851k langdetect calls."""
    with path.open("w", encoding="utf-8", buffering=1 << 20) as handle:
        handle.write(json.dumps({"_scanned": scanned}, ensure_ascii=False) + "\n")
        for target in targets:
            handle.write(json.dumps(target, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_targets(path: Path) -> tuple[list[dict], int]:
    """Read a cached scan back. Returns ``([], 0)`` if it is unusable."""
    targets: list[dict] = []
    scanned = 0
    for record in iter_records(path):
        if "_scanned" in record:
            scanned = int(record["_scanned"])
            continue
        if {"article_id", "title", "flores", "bucket"} <= set(record):
            targets.append(record)
    return targets, scanned


def write_patch(patch_path: Path, translator: Translator, targets: list[dict]) -> int:
    """Translate every target on the GPU and write the corrections file."""
    written = 0
    unchanged = 0
    # Translator.translate_buffer works on (article_id, title, flores) tuples.
    buffer = [(t["article_id"], t["title"], t["flores"]) for t in targets]
    bar = tqdm(
        total=len(buffer), desc="[gpu]", unit="title", dynamic_ncols=True, smoothing=0.1
    )
    with patch_path.open("w", encoding="utf-8", buffering=1 << 20) as handle:
        for article_id, original, translated in translator.translate_buffer(buffer):
            text = (translated or "").strip()
            if not text or text == original:
                # The model gave back nothing usable; leaving the original in
                # place is strictly better than blanking a searchable title.
                unchanged += 1
                bar.update(1)
                continue
            handle.write(
                json.dumps(
                    {
                        "article_id": article_id,
                        "original_title": original,
                        "translated_title": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            bar.update(1)
        handle.flush()
        os.fsync(handle.fileno())
    bar.close()
    if unchanged:
        print(f"[gpu] {unchanged} titles came back empty or unchanged; left as-is")
    return written


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #

# NLLB is reliable on a real headline and unreliable on a fragment. The titles
# recovered here skew very short - a two-character Chinese section label carries
# almost no context - and with nothing to work from the model invents: 海豹
# ("seal") came back as "The beach", Λεμεσός (Limassol) as "Other, including
# fruit". That is a worse outcome than leaving the title untranslated, because a
# hallucinated English title can raise a FALSE match in the downstream keyword
# search, whereas an untranslated one merely fails to match. These three checks
# drop the output that shows the known signatures of that failure.

def _is_mojibake(text: str) -> bool:
    """Source was already corrupt, so anything derived from it is noise."""
    return "�" in text


def _is_repetition_loop(text: str) -> bool:
    """Degenerate decode: 'I love you, I love you, I love you, ...'."""
    words = text.split()
    return len(words) >= 8 and len(set(words)) <= max(2, len(words) // 4)


def _is_bloated(original: str, translated: str) -> bool:
    """Long output from a very short input is invented, not translated."""
    src = len(original.strip())
    return src <= 6 and len(translated.strip()) > src * 4 + 8


def rejection_reason(original: str, translated: str) -> Optional[str]:
    """Why this correction should not be trusted, or None if it looks sound."""
    if _is_mojibake(original):
        return "mojibake-source"
    if _is_repetition_loop(translated):
        return "repetition-loop"
    if _is_bloated(original, translated):
        return "bloated-from-short"
    return None


def split_patch(patch_path: Path) -> tuple[list[dict], list[dict]]:
    """Partition a patch file into (trustworthy, rejected) corrections."""
    good: list[dict] = []
    bad: list[dict] = []
    for record in iter_records(patch_path):
        reason = rejection_reason(
            record.get("original_title", ""), record.get("translated_title", "")
        )
        if reason is None:
            good.append(record)
        else:
            bad.append({**record, "reject_reason": reason})
    return good, bad


# --------------------------------------------------------------------------- #
# Phase 2 - merge the corrections back in
# --------------------------------------------------------------------------- #


def rewrite_dataset(output_path: Path, decide, label: str) -> int:
    """Stream the dataset through ``decide`` and atomically swap the result in.

    ``decide(record)`` returns the replacement ``translated_title`` for a row, or
    None to pass it through untouched. Writing to a temp file and swapping means
    an interruption can never leave a half-rewritten dataset.
    """
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    backup_path = output_path.with_suffix(output_path.suffix + ".bak")
    changed = 0

    bar = tqdm(desc=f"[{label}]", unit="rec", unit_scale=True, dynamic_ncols=True)
    with output_path.open("r", encoding="utf-8", errors="replace") as src, tmp_path.open(
        "w", encoding="utf-8", buffering=1 << 20
    ) as dst:
        for line in src:
            stripped = line.strip()
            bar.update(1)
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                dst.write(line if line.endswith("\n") else line + "\n")
                continue
            replacement = decide(record)
            if replacement is None:
                dst.write(line if line.endswith("\n") else line + "\n")
            else:
                record["translated_title"] = replacement
                changed += 1
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
        dst.flush()
        os.fsync(dst.fileno())
    bar.close()

    # Keep the FIRST backup, never a later one. A second run would otherwise
    # rotate the pristine pre-patch dataset out and replace it with an already-
    # patched copy, which is no backup at all.
    if backup_path.exists():
        print(f"[{label}] keeping the existing {backup_path.name} (pre-patch copy)")
        os.replace(tmp_path, output_path)
    else:
        os.replace(output_path, backup_path)
        os.replace(tmp_path, output_path)
        print(f"[{label}] previous dataset kept at {backup_path.name}")
    return changed


def apply_patch(output_path: Path, patch_path: Path, use_all: bool) -> int:
    """Merge the trustworthy corrections into the dataset."""
    if not patch_path.exists():
        raise SystemExit(f"[error] no patch file to apply: {patch_path}")

    good, bad = split_patch(patch_path)
    print(f"[apply] {len(good) + len(bad):,} corrections in {patch_path.name}")
    if bad:
        reasons = Counter(record["reject_reason"] for record in bad)
        state = "APPLIED ANYWAY (--no-quality-gate)" if use_all else "held back"
        print(f"[apply] {len(bad):,} failed the quality gate, {state}:")
        for reason, count in reasons.most_common():
            print(f"          {reason:<22} {count:>7,}")

    fixes = {
        str(r["article_id"]): r["translated_title"] for r in (good + bad if use_all else good)
    }
    if not fixes:
        return 0

    def decide(record: dict) -> Optional[str]:
        fixed = fixes.get(str(record.get("article_id")))
        # Only touch a row the main run left untranslated. This makes --apply
        # idempotent: a row already corrected no longer satisfies it.
        if fixed is None or record.get("original_title") != record.get(
            "translated_title"
        ):
            return None
        return fixed

    return rewrite_dataset(output_path, decide, "apply")


def revert_bad(output_path: Path, patch_path: Path) -> int:
    """Undo corrections that failed the quality gate but were already applied.

    Restores ``translated_title`` to the original text, putting those rows back
    to the state the main run left them in - untranslated, but honest.
    """
    if not patch_path.exists():
        raise SystemExit(f"[error] no patch file to check: {patch_path}")

    _good, bad = split_patch(patch_path)
    if not bad:
        print("[revert] nothing in the patch fails the quality gate.")
        return 0
    reasons = Counter(record["reject_reason"] for record in bad)
    print(f"[revert] {len(bad):,} defective corrections to undo:")
    for reason, count in reasons.most_common():
        print(f"           {reason:<22} {count:>7,}")

    # article_id -> (bad text that was written, original text to restore)
    undo = {
        str(r["article_id"]): (r["translated_title"], r["original_title"]) for r in bad
    }

    def decide(record: dict) -> Optional[str]:
        entry = undo.get(str(record.get("article_id")))
        if entry is None:
            return None
        bad_text, original = entry
        # Only revert if the bad text is actually what is stored, so this is
        # safe to run whether or not the patch was ever applied.
        if record.get("translated_title") != bad_text:
            return None
        return original

    return rewrite_dataset(output_path, decide, "revert")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-translate the titles that the main run copied through "
            "untranslated because the source record declared the wrong language."
        )
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the JSON config."
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Dataset to repair (defaults to the config's output path).",
    )
    parser.add_argument(
        "--patch-file",
        default=None,
        help=f"Corrections file (defaults to {PATCH_FILENAME} beside the dataset).",
    )
    parser.add_argument(
        "--targets-file",
        default=None,
        help=f"Cached scan (defaults to {TARGETS_FILENAME} beside the dataset).",
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Ignore the cached scan and sweep the dataset again (~47 min).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Merge an existing patch file into the dataset instead of translating.",
    )
    parser.add_argument(
        "--revert-bad",
        action="store_true",
        help=(
            "Undo already-applied corrections that fail the quality gate, "
            "restoring those titles to their untranslated original."
        ),
    )
    parser.add_argument(
        "--no-quality-gate",
        action="store_true",
        help="With --apply, merge every correction including the suspect ones.",
    )
    parser.add_argument(
        "--include-latin",
        action="store_true",
        help=(
            "Also re-translate Latin-script titles that langdetect is >=97%% sure "
            "are not English. Off by default: a false positive would corrupt a "
            "title that is currently correct."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be re-translated, then stop without loading the model.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg: Config = load_config(Path(args.config).expanduser())

    output_path = (
        Path(args.output_file).expanduser() if args.output_file else cfg.output_path
    )
    patch_path = (
        Path(args.patch_file).expanduser()
        if args.patch_file
        else output_path.with_name(PATCH_FILENAME)
    )
    targets_path = (
        Path(args.targets_file).expanduser()
        if args.targets_file
        else output_path.with_name(TARGETS_FILENAME)
    )

    if not output_path.exists():
        print(f"[error] dataset not found: {output_path}", file=sys.stderr)
        return 1
    print(f"[config] dataset : {output_path}")
    print(f"[config] patch   : {patch_path}")

    if args.revert_bad:
        reverted = revert_bad(output_path, patch_path)
        print(f"\n[done] {reverted:,} defective titles restored in {output_path.name}")
        return 0

    if args.apply:
        applied = apply_patch(output_path, patch_path, args.no_quality_gate)
        print(f"\n[done] {applied:,} titles replaced in {output_path.name}")
        return 0

    all_targets: list[dict] = []
    scanned = 0
    if targets_path.exists() and not args.rescan:
        all_targets, scanned = load_targets(targets_path)
        if all_targets:
            print(f"[scan] reusing cached scan: {targets_path.name} (--rescan to redo)")
    if not all_targets:
        detector = Detector()
        all_targets, scanned = find_targets(output_path, detector)
        save_targets(targets_path, all_targets, scanned)
        print("\n[scan] source language resolved by:")
        for source, count in detector.by_source.most_common(15):
            print(f"         {source:<24} {count:>8,}")

    script_targets = [t for t in all_targets if t["bucket"] == "script"]
    latin_targets = [t for t in all_targets if t["bucket"] == "latin"]
    targets = script_targets + (latin_targets if args.include_latin else [])

    print(f"\n[scan] {scanned:,} records read")
    print(f"[scan] {len(script_targets):,} non-Latin titles to re-translate")
    if latin_targets:
        state = "included" if args.include_latin else "NOT included (--include-latin)"
        print(f"[scan] {len(latin_targets):,} Latin-script candidates {state}")
    print(f"[scan] {len(targets):,} titles queued for the GPU")

    if not targets:
        print("\n[done] nothing to repair.")
        return 0
    if args.dry_run:
        print("\n[dry-run] stopping before the model loads.")
        return 0

    stats = Stats()
    translator = Translator(cfg, stats)
    written = write_patch(patch_path, translator, targets)

    print(
        "\n[summary]\n"
        f"  records scanned    : {scanned:,}\n"
        f"  titles targeted    : {len(targets):,}\n"
        f"  corrections written: {written:,}\n"
        f"  batch fallbacks    : {stats.batch_errors}\n"
        f"  oom batch splits   : {stats.oom_splits}\n"
        f"  patch file         : {patch_path.resolve()}\n"
        f"\nReview it, then merge with:\n"
        f"  python {Path(__file__).name} --apply"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
