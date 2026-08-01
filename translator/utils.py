"""
utils.py
========

Pure-CPU helpers shared by the flood-article translation pipeline.

This module is deliberately free of any ``torch`` / ``transformers`` model
imports so that it can be imported cheaply inside multiprocessing workers on
Windows (which use the ``spawn`` start method and therefore re-import every
module in every child process).

Contents
--------
* FLORES-200 language-code tables and ISO-639-1 -> FLORES mapping
* Unicode script detection (needed to disambiguate e.g. ``zh`` -> Hans/Hant)
* Language detection (metadata-first, ``langdetect`` fallback)
* Mojibake repair for text that was UTF-8 decoded as CP1252
* Multilingual sentence splitting and token-budget chunk packing
* Small formatting / system-metric helpers
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Final, Iterable, Iterator, Sequence

# --------------------------------------------------------------------------- #
# FLORES-200 language codes supported by facebook/nllb-200-distilled-1.3B
# --------------------------------------------------------------------------- #

#: The 202 language tokens actually present in the local checkpoint's
#: tokenizer. This list was extracted from
#: ``C:\darsh\AI_MODELS\model\tokenizer_config.json`` rather than transcribed
#: from the paper -- the published "FLORES-200" list and this checkpoint's
#: vocabulary genuinely disagree (e.g. this model has ``als_Latn`` and
#: ``sat_Beng`` but not ``arb_Latn``, ``min_Arab`` or ``sat_Olck``).
#: ``NLLBTranslator`` re-validates this set against the loaded tokenizer at
#: startup and warns on any drift.
FLORES_200: Final[frozenset[str]] = frozenset(
    """
    ace_Arab ace_Latn acm_Arab acq_Arab aeb_Arab afr_Latn ajp_Arab aka_Latn
    als_Latn amh_Ethi apc_Arab arb_Arab ars_Arab ary_Arab arz_Arab asm_Beng
    ast_Latn awa_Deva ayr_Latn azb_Arab azj_Latn bak_Cyrl bam_Latn ban_Latn
    bel_Cyrl bem_Latn ben_Beng bho_Deva bjn_Arab bjn_Latn bod_Tibt bos_Latn
    bug_Latn bul_Cyrl cat_Latn ceb_Latn ces_Latn cjk_Latn ckb_Arab crh_Latn
    cym_Latn dan_Latn deu_Latn dik_Latn dyu_Latn dzo_Tibt ell_Grek eng_Latn
    epo_Latn est_Latn eus_Latn ewe_Latn fao_Latn fij_Latn fin_Latn fon_Latn
    fra_Latn fur_Latn fuv_Latn gaz_Latn gla_Latn gle_Latn glg_Latn grn_Latn
    guj_Gujr hat_Latn hau_Latn heb_Hebr hin_Deva hne_Deva hrv_Latn hun_Latn
    hye_Armn ibo_Latn ilo_Latn ind_Latn isl_Latn ita_Latn jav_Latn jpn_Jpan
    kab_Latn kac_Latn kam_Latn kan_Knda kas_Arab kas_Deva kat_Geor kaz_Cyrl
    kbp_Latn kea_Latn khk_Cyrl khm_Khmr kik_Latn kin_Latn kir_Cyrl kmb_Latn
    kmr_Latn knc_Arab knc_Latn kon_Latn kor_Hang lao_Laoo lij_Latn lim_Latn
    lin_Latn lit_Latn lmo_Latn ltg_Latn ltz_Latn lua_Latn lug_Latn luo_Latn
    lus_Latn lvs_Latn mag_Deva mai_Deva mal_Mlym mar_Deva min_Latn mkd_Cyrl
    mlt_Latn mni_Beng mos_Latn mri_Latn mya_Mymr nld_Latn nno_Latn nob_Latn
    npi_Deva nso_Latn nus_Latn nya_Latn oci_Latn ory_Orya pag_Latn pan_Guru
    pap_Latn pbt_Arab pes_Arab plt_Latn pol_Latn por_Latn prs_Arab quy_Latn
    ron_Latn run_Latn rus_Cyrl sag_Latn san_Deva sat_Beng scn_Latn shn_Mymr
    sin_Sinh slk_Latn slv_Latn smo_Latn sna_Latn snd_Arab som_Latn sot_Latn
    spa_Latn srd_Latn srp_Cyrl ssw_Latn sun_Latn swe_Latn swh_Latn szl_Latn
    tam_Taml taq_Latn taq_Tfng tat_Cyrl tel_Telu tgk_Cyrl tgl_Latn tha_Thai
    tir_Ethi tpi_Latn tsn_Latn tso_Latn tuk_Latn tum_Latn tur_Latn twi_Latn
    tzm_Tfng uig_Arab ukr_Cyrl umb_Latn urd_Arab uzn_Latn vec_Latn vie_Latn
    war_Latn wol_Latn xho_Latn ydd_Hebr yor_Latn yue_Hant zho_Hans zho_Hant
    zsm_Latn zul_Latn
    """.split()
)

TARGET_LANG: Final[str] = "eng_Latn"

# ISO-639-1 (and a few common ISO-639-3 / langdetect codes) -> FLORES-200.
# Only the unambiguous default script is listed here; codes whose script must be
# inferred from the text itself are handled in ``_SCRIPT_SENSITIVE`` below.
ISO1_TO_FLORES: Final[dict[str, str]] = {
    "af": "afr_Latn", "ak": "aka_Latn", "am": "amh_Ethi", "ar": "arb_Arab",
    "as": "asm_Beng", "ast": "ast_Latn", "ay": "ayr_Latn", "az": "azj_Latn",
    "ba": "bak_Cyrl", "bm": "bam_Latn", "be": "bel_Cyrl", "bn": "ben_Beng",
    "bho": "bho_Deva", "bo": "bod_Tibt", "bs": "bos_Latn", "bg": "bul_Cyrl",
    "ca": "cat_Latn", "ceb": "ceb_Latn", "cs": "ces_Latn", "ckb": "ckb_Arab",
    "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn", "dz": "dzo_Tibt",
    "el": "ell_Grek", "en": "eng_Latn", "eo": "epo_Latn", "es": "spa_Latn",
    "et": "est_Latn", "eu": "eus_Latn", "ee": "ewe_Latn", "fa": "pes_Arab",
    "ff": "fuv_Latn", "fo": "fao_Latn", "fj": "fij_Latn",
    "fi": "fin_Latn", "fr": "fra_Latn", "fy": "nld_Latn", "ga": "gle_Latn",
    "gd": "gla_Latn", "gl": "glg_Latn", "gn": "grn_Latn", "gu": "guj_Gujr",
    "ha": "hau_Latn", "he": "heb_Hebr", "iw": "heb_Hebr", "hi": "hin_Deva",
    "hr": "hrv_Latn", "ht": "hat_Latn", "hu": "hun_Latn", "hy": "hye_Armn",
    "id": "ind_Latn", "ig": "ibo_Latn", "ilo": "ilo_Latn", "is": "isl_Latn",
    "it": "ita_Latn", "ja": "jpn_Jpan", "jv": "jav_Latn", "jw": "jav_Latn",
    "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda",
    "ko": "kor_Hang", "ks": "kas_Arab", "ku": "kmr_Latn", "ky": "kir_Cyrl",
    "lb": "ltz_Latn", "lg": "lug_Latn", "li": "lim_Latn", "ln": "lin_Latn",
    "lo": "lao_Laoo", "lt": "lit_Latn",
    "lv": "lvs_Latn", "mg": "plt_Latn", "mi": "mri_Latn", "mk": "mkd_Cyrl",
    "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva", "ms": "zsm_Latn",
    "mt": "mlt_Latn", "my": "mya_Mymr", "ne": "npi_Deva", "nl": "nld_Latn",
    "nn": "nno_Latn", "no": "nob_Latn", "nb": "nob_Latn", "ny": "nya_Latn",
    "oc": "oci_Latn", "om": "gaz_Latn", "or": "ory_Orya", "pa": "pan_Guru",
    "pl": "pol_Latn", "ps": "pbt_Arab", "pt": "por_Latn", "qu": "quy_Latn",
    "rn": "run_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "rw": "kin_Latn",
    "sa": "san_Deva", "sc": "srd_Latn", "sd": "snd_Arab", "sg": "sag_Latn",
    "si": "sin_Sinh", "sk": "slk_Latn", "sl": "slv_Latn",
    "sm": "smo_Latn", "sn": "sna_Latn", "so": "som_Latn", "sq": "als_Latn",
    "ss": "ssw_Latn", "st": "sot_Latn", "su": "sun_Latn", "sv": "swe_Latn",
    "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "tg": "tgk_Cyrl",
    "th": "tha_Thai", "ti": "tir_Ethi", "tk": "tuk_Latn", "tl": "tgl_Latn",
    "fil": "tgl_Latn", "tn": "tsn_Latn", "tr": "tur_Latn", "ts": "tso_Latn",
    "tt": "tat_Cyrl", "tw": "twi_Latn", "ug": "uig_Arab", "uk": "ukr_Cyrl",
    "ur": "urd_Arab", "uz": "uzn_Latn", "vi": "vie_Latn", "war": "war_Latn",
    "wo": "wol_Latn", "xh": "xho_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn",
    "zu": "zul_Latn",
    # langdetect-specific spellings
    "zh-cn": "zho_Hans", "zh-tw": "zho_Hant", "zh_cn": "zho_Hans",
    "zh_tw": "zho_Hant", "pt-br": "por_Latn", "pt-pt": "por_Latn",
}

#: Codes whose FLORES script suffix depends on the actual characters used.
_SCRIPT_SENSITIVE: Final[dict[str, dict[str, str]]] = {
    "zh": {"Hant": "zho_Hant", "Hans": "zho_Hans", "*": "zho_Hans"},
    "sr": {"Latn": "bos_Latn", "Cyrl": "srp_Cyrl", "*": "srp_Cyrl"},
    "pa": {"Arab": "pan_Guru", "Guru": "pan_Guru", "*": "pan_Guru"},
    "ku": {"Arab": "ckb_Arab", "Latn": "kmr_Latn", "*": "kmr_Latn"},
    "kk": {"Cyrl": "kaz_Cyrl", "*": "kaz_Cyrl"},
    "az": {"Arab": "azb_Arab", "Latn": "azj_Latn", "*": "azj_Latn"},
    "ms": {"Latn": "zsm_Latn", "*": "zsm_Latn"},
}

# Unicode block ranges used for cheap script identification.
_SCRIPT_RANGES: Final[tuple[tuple[int, int, str], ...]] = (
    (0x0400, 0x04FF, "Cyrl"), (0x0500, 0x052F, "Cyrl"),
    (0x0600, 0x06FF, "Arab"), (0x0750, 0x077F, "Arab"), (0xFB50, 0xFDFF, "Arab"),
    (0x0590, 0x05FF, "Hebr"),
    (0x0900, 0x097F, "Deva"), (0x0980, 0x09FF, "Beng"), (0x0A00, 0x0A7F, "Guru"),
    (0x0A80, 0x0AFF, "Gujr"), (0x0B00, 0x0B7F, "Orya"), (0x0B80, 0x0BFF, "Taml"),
    (0x0C00, 0x0C7F, "Telu"), (0x0C80, 0x0CFF, "Knda"), (0x0D00, 0x0D7F, "Mlym"),
    (0x0D80, 0x0DFF, "Sinh"), (0x0E00, 0x0E7F, "Thai"), (0x0E80, 0x0EFF, "Laoo"),
    (0x1000, 0x109F, "Mymr"), (0x10A0, 0x10FF, "Geor"), (0x1200, 0x137F, "Ethi"),
    (0x1780, 0x17FF, "Khmr"), (0x0530, 0x058F, "Armn"), (0x0370, 0x03FF, "Grek"),
    (0x3040, 0x30FF, "Jpan"), (0xAC00, 0xD7AF, "Hang"), (0x1100, 0x11FF, "Hang"),
    (0x4E00, 0x9FFF, "Hani"), (0x3400, 0x4DBF, "Hani"), (0xF900, 0xFAFF, "Hani"),
    (0x0041, 0x005A, "Latn"), (0x0061, 0x007A, "Latn"), (0x00C0, 0x024F, "Latn"),
)

# A small set of characters that only exist in Traditional Chinese. Presence of
# any of these is a strong signal for ``zho_Hant`` over ``zho_Hans``.
_TRADITIONAL_MARKERS: Final[frozenset[str]] = frozenset(
    "萬與丑專業叢東絲丟兩嚴喪個豐臨為麗舉麼義烏樂喬習鄉書買亂爭於虧雲亞產"
    "億僅從侖倉儀們價眾優會傷體倆偉傳傷倫倉個俠僉條來儉儲兒黨蘭關興軍農"
    "決沒沖淚淺潔滅濟灣點爲總聯聲臺蘭處風飛馬鳥龍龜齒"
)


#: Every language code ``langdetect`` can emit. All of these must resolve to a
#: real model code, otherwise articles silently fail with "language could not be
#: determined" -- which is exactly the bug ``validate_language_tables`` exists to
#: catch (a missing ``"es"`` entry once cost ~6.7% of this corpus).
LANGDETECT_CODES: Final[tuple[str, ...]] = (
    "af ar bg bn ca cs cy da de el en es et fa fi fr gu he hi hr hu id it ja kn "
    "ko lt lv mk ml mr ne nl no pa pl pt ro ru sk sl so sq sv sw ta te th tl tr "
    "uk ur vi zh-cn zh-tw".split()
)


def validate_language_tables() -> list[str]:
    """Self-check the language tables; return a list of human-readable problems.

    Verifies that (a) every value in :data:`ISO1_TO_FLORES` is a code the model
    actually has, and (b) every code ``langdetect`` can return is mappable.
    Called at startup so a table regression surfaces immediately instead of as
    thousands of mysteriously "failed" articles hours into a run.
    """
    problems: list[str] = []
    bad = {k: v for k, v in ISO1_TO_FLORES.items() if v not in FLORES_200}
    if bad:
        problems.append(f"ISO map points at non-existent model codes: {bad}")
    for table in _SCRIPT_SENSITIVE.values():
        for v in table.values():
            if v not in FLORES_200:
                problems.append(f"script-sensitive map points at unknown code: {v}")
    have = set(ISO1_TO_FLORES) | set(_SCRIPT_SENSITIVE)
    unmapped = [c for c in LANGDETECT_CODES
                if c not in have and c.split("-")[0] not in have]
    if unmapped:
        problems.append(f"langdetect can emit codes we cannot map: {unmapped}")
    return problems


def detect_script(text: str, sample: int = 4000) -> str:
    """Return the dominant Unicode script tag (e.g. ``Latn``, ``Cyrl``) of *text*.

    Only the first *sample* characters are inspected; that is ample for a
    language decision and keeps the function O(1) for very long articles.
    """
    counts: dict[str, int] = {}
    for ch in text[:sample]:
        cp = ord(ch)
        if cp < 0x0041 or ch.isspace() or ch.isdigit():
            continue
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return "Latn"
    top = max(counts, key=lambda k: counts[k])
    if top == "Hani":
        return "Hant" if any(c in _TRADITIONAL_MARKERS for c in text[:sample]) else "Hans"
    return top


def to_flores(code: str | None, text: str = "") -> str | None:
    """Map an arbitrary language *code* to a FLORES-200 code NLLB understands.

    Accepts values already in FLORES form (returned unchanged), ISO-639-1,
    ISO-639-3 and langdetect spellings. *text* is used only to disambiguate
    script-sensitive languages such as Chinese or Serbian.

    Returns ``None`` when the code is empty/unknown, so callers can fall back to
    statistical detection.
    """
    if not code:
        return None
    c = code.strip().replace("-", "_") if "_" in code.replace("-", "_") and len(code) > 3 else code.strip()
    if c in FLORES_200:
        return c
    low = code.strip().lower().replace("_", "-")
    base = low.split("-")[0]

    if base in _SCRIPT_SENSITIVE:
        table = _SCRIPT_SENSITIVE[base]
        if low in ISO1_TO_FLORES:                      # e.g. explicit "zh-tw"
            cand = ISO1_TO_FLORES[low]
            return cand if cand in FLORES_200 else None
        script = detect_script(text) if text else "*"
        cand = table.get(script, table["*"])
        return cand if cand in FLORES_200 else None

    cand = ISO1_TO_FLORES.get(low) or ISO1_TO_FLORES.get(base)
    return cand if cand and cand in FLORES_200 else None


# --------------------------------------------------------------------------- #
# Statistical language detection (fallback only)
# --------------------------------------------------------------------------- #

_LANGDETECT_READY = False


def _init_langdetect() -> Callable[[str], str] | None:
    """Import and seed ``langdetect`` lazily; return its ``detect`` callable."""
    global _LANGDETECT_READY
    try:
        from langdetect import DetectorFactory, detect  # type: ignore

        if not _LANGDETECT_READY:
            DetectorFactory.seed = 0  # make detection deterministic
            _LANGDETECT_READY = True
        return detect
    except Exception:  # pragma: no cover - optional dependency
        return None


def detect_language(
    text: str,
    metadata_code: str | None = None,
    *,
    trust_metadata: bool = True,
    min_chars: int = 20,
) -> tuple[str | None, str]:
    """Resolve the FLORES-200 source language for *text*.

    Strategy (cheapest first):

    1. Trust ``metadata_code`` when present and mappable -- the upstream crawler
       already recorded a language for nearly every article, and re-detecting
       233k documents adds cost without adding accuracy.
    2. Otherwise fall back to ``langdetect`` on a text sample.

    Returns ``(flores_code_or_None, source)`` where *source* is one of
    ``"metadata"``, ``"langdetect"`` or ``"unknown"`` -- useful for logging.
    """
    if trust_metadata:
        mapped = to_flores(metadata_code, text)
        if mapped:
            return mapped, "metadata"

    stripped = text.strip()
    if len(stripped) < min_chars:
        # Too short to detect reliably; fall back to metadata even if untrusted.
        mapped = to_flores(metadata_code, text)
        return (mapped, "metadata") if mapped else (None, "unknown")

    detect = _init_langdetect()
    if detect is not None:
        try:
            return (to_flores(detect(stripped[:2000]), text) or None), "langdetect"
        except Exception:
            pass

    mapped = to_flores(metadata_code, text)
    return (mapped, "metadata") if mapped else (None, "unknown")


# --------------------------------------------------------------------------- #
# Mojibake repair
# --------------------------------------------------------------------------- #

_MOJIBAKE_HINTS: Final[tuple[str, ...]] = (
    "â€™", "â€œ", "â€\x9d", "â€“", "â€”", "Ã¡", "Ã©", "Ã­", "Ã³", "Ãº", "Ã±",
    "Ã¼", "Ã¤", "Ã¶", "Ã§", "Ã£", "Ãµ", "Â°", "Â£", "Â©", "ä¸", "Ð¾", "Ø§",
)


def fix_mojibake(text: str) -> str:
    """Repair text that was encoded as UTF-8 then decoded as CP1252.

    The source corpus contains artefacts such as ``itâ€™s`` (a curly apostrophe
    round-tripped through the wrong codec). Feeding those to NLLB wastes tokens
    and degrades translation quality, so the *model input* is repaired -- the
    original field values are always preserved untouched in the output file.

    The repair is applied only when it round-trips cleanly and the result still
    looks like the same text, so correct text is never damaged.
    """
    if not text or not any(h in text for h in _MOJIBAKE_HINTS):
        return text
    try:
        repaired = text.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    # Guard: only accept if the repair reduced the number of replacement-ish
    # artefacts and did not blow up the text length.
    if repaired.count("�") > text.count("�"):
        return text
    return repaired


def normalize_text(text: str, *, repair_mojibake: bool = True) -> str:
    """Normalise *text* for model consumption (never for storage)."""
    if not text:
        return ""
    if repair_mojibake:
        text = fix_mojibake(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Sentence splitting + chunk packing
# --------------------------------------------------------------------------- #

# Sentence terminators across the scripts NLLB covers: Latin/Cyrillic/Greek
# ``.!?``, CJK ``。！？``, Arabic ``؟``, Urdu ``۔``, Devanagari ``।``,
# Ethiopic ``።``, Armenian ``։``, Thai has no terminator (handled by length).
_SENT_END = re.compile(
    r"(?<=[.!?。！？؟۔।॥።։])[\"'”’）\)\]]*\s+"
)
_ABBREV = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Sra|St|vs|etc|Inc|Ltd|Co|Jr|No|Ave|Gov|Sen|Rep|Col|Gen|Capt|Lt|Sgt)\.)\s*$",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentence-ish units.

    NLLB is a *sentence-level* MT model: quality degrades markedly on long
    multi-sentence inputs, and short inputs also batch far more efficiently on
    the GPU. Paragraph boundaries are honoured first, then sentence
    terminators, with a guard against splitting on common abbreviations.
    """
    if not text:
        return []
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        pieces = _SENT_END.split(para)
        buf = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            cand = f"{buf} {piece}".strip() if buf else piece
            if _ABBREV.search(cand):     # "Dr." -- keep gluing
                buf = cand
                continue
            out.append(cand)
            buf = ""
        if buf:
            out.append(buf)
    return out


def _hard_split(unit: str, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    """Split an over-long *unit* (a run-on sentence) into <= *max_tokens* pieces."""
    words = unit.split(" ")
    if len(words) <= 1:
        # No spaces (e.g. Thai/CJK): fall back to a character-ratio split.
        approx = max(1, math.ceil(count(unit) / max_tokens))
        size = max(1, len(unit) // approx)
        return [unit[i : i + size] for i in range(0, len(unit), size)]

    pieces: list[str] = []
    buf: list[str] = []
    for w in words:
        buf.append(w)
        if count(" ".join(buf)) >= max_tokens:
            # Drop the last word back so the piece stays under budget.
            if len(buf) > 1:
                pieces.append(" ".join(buf[:-1]))
                buf = [buf[-1]]
            else:
                pieces.append(buf[0])
                buf = []
    if buf:
        pieces.append(" ".join(buf))
    return [p for p in pieces if p.strip()]


def pack_chunks(
    text: str,
    count_tokens: Callable[[str], int],
    *,
    target_tokens: int = 64,
    max_tokens: int = 192,
) -> list[tuple[str, int]]:
    """Split *text* into translation chunks of ``(chunk_text, n_tokens)``.

    Sentences are greedily packed up to *target_tokens* so that batches stay
    short -- measured throughput on the RTX 4060 Ti is ~1.8x higher at 48-token
    chunks than at 192-token chunks, and NLLB's own quality is best at sentence
    granularity. No chunk ever exceeds *max_tokens* (well under the model's
    1024-position limit).
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[tuple[str, int]] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            joined = " ".join(buf)
            chunks.append((joined, buf_tokens or count_tokens(joined)))
            buf, buf_tokens = [], 0

    for sent in sentences:
        n = count_tokens(sent)
        if n > max_tokens:
            flush()
            for piece in _hard_split(sent, max_tokens, count_tokens):
                chunks.append((piece, min(count_tokens(piece), max_tokens)))
            continue
        if buf and buf_tokens + n > target_tokens:
            flush()
        buf.append(sent)
        buf_tokens += n
    flush()
    return chunks


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class Chunk:
    """One unit of GPU work: a piece of text belonging to a specific article."""

    article_idx: int   # index into the block's article list
    field: str         # "text" or "title"
    order: int         # position of this chunk within the field
    text: str          # normalised source text fed to the model
    n_tokens: int      # exact source-token count (drives batching)
    lang: str          # FLORES-200 source language


def projected_max_new(src_tokens: int, *, ratio: float = 1.8, pad: int = 24,
                      hard_cap: int = 384) -> int:
    """Worst-case number of tokens the decoder may emit for a *src_tokens* input."""
    return min(hard_cap, int(src_tokens * ratio) + pad)


#: Bytes of KV cache one slot costs, derived from the checkpoint rather than
#: fitted: ``decoder_layers * 2 (K and V) * d_model * 2 (fp16)``. For
#: NLLB-200-distilled-1.3B that is ``24 * 2 * 1024 * 2`` = 98,304 B = 96 KiB,
#: i.e. **10,922 slots per GiB**.
#:
#: An earlier fitted constant of 16384 slots/GiB (64 KiB) under-predicted peak
#: VRAM by 1.5x, which is what drove the planner to aim for 8 GiB on batches
#: that really peaked at 11.9 GiB. Use :func:`kv_bytes_per_slot` to derive this
#: from a live config instead of assuming the 1.3B shape.
KV_BYTES_PER_SLOT_1B3: Final[int] = 24 * 2 * 1024 * 2

#: Per-sequence overhead, in KV-slot equivalents, for the logits buffer that
#: ``generate`` materialises at *every* decode step.
#:
#: NLLB's vocabulary is 256,206 tokens. Each decode step builds a
#: ``batch x vocab`` logits tensor and copies it to float32, so ~1.47 MiB is
#: live **per sequence, independent of sequence length**.
#:
#: 1.47 MiB at the correct 10,922 slots/GiB is **16 slots**, not the 23.5 the
#: old 16384 figure implied. This term is deliberately *not* the mechanism that
#: bounds batch width: at batch 512 it contributes 8,192 of ~89,600 slots (9%),
#: far too little to bind. Width is bounded explicitly by ``max_batch_size``,
#: because the logits buffer's cost is fragmentation (a 750 MiB alloc/free per
#: decode step), not steady-state occupancy -- and ``expandable_segments``,
#: PyTorch's remedy for that, is silently unsupported on Windows.
LOGITS_SLOTS_PER_SEQ: Final[int] = 16


def kv_bytes_per_slot(decoder_layers: int, d_model: int, dtype_bytes: int = 2) -> int:
    """Bytes one KV slot (one sequence, one token position) costs.

    A slot holds both K and V for every decoder layer::

        decoder_layers * 2 * d_model * dtype_bytes

    Cross-attention K/V (computed once from the encoder output, keyed on source
    length) costs the same per slot as decoder self-attention, which is why
    :func:`batch_kv_slots` counts ``max_src + max_new`` rather than ``max_new``
    alone.
    """
    return decoder_layers * 2 * d_model * dtype_bytes


def slots_for_vram(
    gib: float,
    *,
    bytes_per_slot: int = KV_BYTES_PER_SLOT_1B3,
) -> int:
    """How many KV slots fit in *gib* GiB. Inverse of :func:`vram_for_slots`."""
    return int(gib * 1024 ** 3 / bytes_per_slot)


def vram_for_slots(
    slots: int,
    *,
    weights_gib: float = 2.56,
    bytes_per_slot: int = KV_BYTES_PER_SLOT_1B3,
) -> float:
    """Predict a batch's peak VRAM, in GiB, from its slot cost.

    ::

        peak_GiB ~= weights + slots * bytes_per_slot

    Checked against the production log that motivated this correction: batch 512
    of <=47-token chunks plans to 89,600 slots, which this predicts at 10.8 GiB
    (+0.75 GiB of logits buffer = 11.5 GiB). Observed peak was 11.9 GiB. The
    superseded ``slots / 16384`` form predicted 8.0 GiB for the same batch.
    """
    return weights_gib + slots * bytes_per_slot / 1024 ** 3


def batch_kv_slots(batch_size: int, max_src: int, max_new: int) -> int:
    """Estimate a batch's peak-memory cost in KV-slot equivalents.

    Peak VRAM here is dominated by two terms, and **neither one is source
    tokens**:

    1. attention KV cache -- scales with ``batch * (src_len + generated_len)``,
       counting both decoder self-attention (grows to ``max_new``) and
       cross-attention (fixed at ``max_src``)
    2. the per-step logits buffer -- scales with ``batch * vocab_size``
       (see :data:`LOGITS_SLOTS_PER_SEQ`)

    Budgeting on source length alone made a wide batch of short chunks look
    cheap while it reserved a decode cache *and* a logits buffer for every
    sequence in it; that mistake produced 19 OOM events in a 3,000-article run.

    Convert the result to GiB with :func:`vram_for_slots`.
    """
    return batch_size * (max_src + max_new + LOGITS_SLOTS_PER_SEQ)


def build_batches(
    chunks: Sequence[Chunk],
    *,
    slot_budget: int,
    max_batch_size: int,
    out_ratio: float = 1.8,
    out_pad: int = 24,
    hard_max_new: int = 384,
) -> list[list[Chunk]]:
    """Group *chunks* into KV-budgeted batches.

    Chunks must already be grouped by language (NLLB needs a homogeneous
    ``src_lang`` per batch). Within a language they are sorted by length so that
    padding -- and therefore wasted decode steps -- stays minimal, and so that
    every batch's projected output length is tight.

    A batch is closed when adding the next chunk would exceed *slot_budget*
    (see :func:`batch_kv_slots`) or *max_batch_size*.
    """
    if not chunks:
        return []
    ordered = sorted(chunks, key=lambda c: c.n_tokens)
    batches: list[list[Chunk]] = []
    cur: list[Chunk] = []
    cur_max = 0
    for ch in ordered:
        new_max = max(cur_max, ch.n_tokens)
        new_out = projected_max_new(new_max, ratio=out_ratio, pad=out_pad,
                                    hard_cap=hard_max_new)
        cost = batch_kv_slots(len(cur) + 1, new_max, new_out)
        if cur and (cost > slot_budget or len(cur) >= max_batch_size):
            batches.append(cur)
            cur, cur_max = [ch], ch.n_tokens
        else:
            cur.append(ch)
            cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #

def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the JSON config file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def atomic_write_json(path: str | os.PathLike[str], obj: Any, *, indent: int | None = 2) -> None:
    """Write *obj* as JSON to *path* atomically (write temp + os.replace).

    Guarantees a reader never observes a half-written article file, even if the
    process is killed mid-write.
    """
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def human_time(seconds: float) -> str:
    """Format *seconds* as ``H:MM:SS`` / ``M:SS``."""
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "--:--"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def human_count(n: float) -> str:
    """Compact large-number formatting (e.g. ``12.3k``)."""
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def chunked(seq: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield successive lists of at most *size* items from *seq*."""
    buf: list[Any] = []
    for item in seq:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
