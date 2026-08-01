"""Exact-duplicate detection over cleaned article text.

Strategy
--------
Workers hash their own cleaned text (parallel, CPU-bound) and return only the
32-byte digest.  The parent process owns the single authoritative index and
makes every keep/drop decision.  Cleaned text never crosses the process
boundary, which keeps IPC volume proportional to the *number* of articles
rather than their size.

Because the driver consumes worker results in submission order and the
filesystem walk is sorted, "first occurrence wins" is deterministic: the same
corpus always yields the same survivor.

Only exact matching is performed -- SHA-256 over the cleaned text.  No fuzzy
or near-duplicate logic is involved.

Memory
------
The index holds one 32-byte digest plus one relative path per *unique*
article.  At 1M unique articles that is roughly 150-200 MB, which is the
intended trade-off for being able to name the original file in the duplicate
log.  Setting ``track_originals=False`` drops the paths and roughly halves it.
"""

from __future__ import annotations

import hashlib
from typing import Final, NamedTuple

#: Encoding used to serialise text prior to hashing.
HASH_ENCODING: Final[str] = "utf-8"

#: Name of the digest algorithm, surfaced in the processing report.
HASH_ALGORITHM: Final[str] = "sha256"


def hash_text(text: str) -> bytes:
    """Return the SHA-256 digest of ``text``.

    The raw 32-byte digest is returned rather than its hex form: it is half
    the size in memory and in IPC payloads, and hex is only needed at the
    reporting boundary.

    Args:
        text: Cleaned article text.

    Returns:
        The 32-byte SHA-256 digest.
    """
    return hashlib.sha256(text.encode(HASH_ENCODING)).digest()


class DuplicateRecord(NamedTuple):
    """A rejected duplicate and the article it duplicates.

    Attributes:
        duplicate_path: Root-relative path of the rejected article.
        original_path: Root-relative path of the retained article.
        digest: Hex SHA-256 digest of the shared cleaned text.
    """

    duplicate_path: str
    original_path: str
    digest: str


class Deduplicator:
    """In-memory index of cleaned-text digests seen so far.

    Not thread-safe or process-safe by design: exactly one instance exists, in
    the parent process, and it is consulted serially.
    """

    __slots__ = ("_index", "_track_originals", "_unique", "_duplicates")

    def __init__(self, track_originals: bool = True) -> None:
        """Initialise an empty index.

        Args:
            track_originals: When True, remember the path of each first
                occurrence so duplicates can be logged against it. When False,
                only digests are stored, reducing memory at the cost of
                reporting ``""`` as the original path.
        """
        self._index: dict[bytes, str] = {}
        self._track_originals = track_originals
        self._unique = 0
        self._duplicates = 0

    def check(self, digest: bytes, path: str) -> DuplicateRecord | None:
        """Register ``digest`` and report whether it was already present.

        Args:
            digest: 32-byte SHA-256 digest of the cleaned text.
            path: Root-relative path of the article being registered.

        Returns:
            ``None`` when this is the first occurrence (the article should be
            kept), otherwise a :class:`DuplicateRecord` naming the original.
        """
        existing = self._index.get(digest)
        if existing is not None:
            self._duplicates += 1
            return DuplicateRecord(path, existing, digest.hex())
        self._index[digest] = path if self._track_originals else ""
        self._unique += 1
        return None

    @property
    def unique_count(self) -> int:
        """Number of distinct cleaned texts registered.

        Returns:
            Count of unique articles.
        """
        return self._unique

    @property
    def duplicate_count(self) -> int:
        """Number of duplicates rejected.

        Returns:
            Count of duplicate articles.
        """
        return self._duplicates

    def clear(self) -> None:
        """Release the index, freeing its memory."""
        self._index.clear()
        self._unique = 0
        self._duplicates = 0

    def __len__(self) -> int:
        """Return the number of unique digests held.

        Returns:
            Size of the index.
        """
        return len(self._index)
