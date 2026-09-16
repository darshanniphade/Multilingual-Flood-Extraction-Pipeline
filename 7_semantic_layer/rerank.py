"""Stage 7h - cross-encoder reranking (optional, offline-only).

Bi-encoder retrieval embeds the query and the document independently, so it can
only ever compare two summaries of meaning. A cross-encoder reads the pair
together and scores it directly, which is markedly more precise over a short
candidate list - and far too slow to run over a whole corpus. Hence the
standard arrangement this file completes: hybrid retrieval proposes ~100
candidates, the cross-encoder reorders the top `rerank.top_n` of them.

Offline policy, enforced here rather than hoped for: nothing is downloaded. A
reranker is used only when `rerank.model` points at a local directory or at a
model already in the Hugging Face cache. Otherwise `Reranker.available` is
False, `search.py` keeps the RRF order, and both say so in their output. No
silent fallback, no silent download.

To switch it on, put a cross-encoder in the cache (bge-reranker-base and
ms-marco-MiniLM-L-6-v2 are the usual choices) and set:

    "rerank": {"enabled": true, "model": "BAAI/bge-reranker-base"}

or a filesystem path. Then:

    python rerank.py --check                       # is a local reranker usable?
    python rerank.py --query "Assam floods" --docs a.txt b.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common import DEFAULT_CONFIG, apply_offline_env, load_config, setup_logging

log = logging.getLogger("events.rerank")


def _is_locally_available(model: str) -> tuple[bool, str]:
    """True when `model` can be loaded without touching the network."""
    if not model:
        return False, "rerank.model is not set"
    path = Path(model)
    if path.exists() and path.is_dir():
        return True, f"local directory {path}"
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=model, local_files_only=True)
        return True, f"HF cache {local}"
    except Exception as exc:
        return False, f"{model} is not in the local HF cache ({type(exc).__name__})"


class Reranker:
    """Cross-encoder scorer with a hard offline guarantee.

    `available` is False whenever no local model was found; callers must check
    it and fall back to the fusion order rather than assuming a score exists.
    """

    def __init__(self, cfg: dict, force: bool = False):
        rc = cfg.get("rerank") or {}
        self.cfg = rc
        self.model = None
        self.available = False
        self.reason = ""
        if not rc.get("enabled") and not force:
            self.reason = "rerank.enabled is false"
            return
        ok, message = _is_locally_available(rc.get("model") or "")
        if not ok:
            self.reason = message
            log.warning("reranking disabled: %s", message)
            return
        apply_offline_env(cfg)
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                rc["model"],
                device=rc.get("device", "cuda"),
                max_length=int(rc.get("max_length", 512)),
            )
            self.available = True
            self.reason = f"loaded from {message}"
            log.info("cross-encoder %s ready (%s)", rc["model"], message)
        except Exception as exc:  # a cached-but-broken model must not kill a search
            self.reason = f"failed to load {rc.get('model')}: {exc}"
            log.error(self.reason)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not self.available:
            raise RuntimeError(f"no reranker available: {self.reason}")
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, batch_size=int(self.cfg.get("batch_size", 32)))
        return [float(s) for s in scores]

    def rerank(self, query: str, candidates: list[dict], text_key: str = "text") -> list[dict]:
        """Reorder `candidates` in place-safe fashion, attaching `rerank_score`.
        Only the first `top_n` are rescored; the rest keep their fusion order
        behind them, which is what makes this affordable."""
        if not self.available or not candidates:
            return candidates
        top_n = int(self.cfg.get("top_n", 50))
        head, tail = candidates[:top_n], candidates[top_n:]
        scores = self.score(query, [c.get(text_key, "") for c in head])
        for candidate, score in zip(head, scores):
            candidate["rerank_score"] = round(score, 5)
        head.sort(key=lambda c: -c["rerank_score"])
        return head + tail


def main() -> int:
    ap = argparse.ArgumentParser(description="Check or exercise the local cross-encoder reranker")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--check", action="store_true", help="report whether a local reranker is usable")
    ap.add_argument("--query")
    ap.add_argument("--docs", nargs="*", default=[], help="text files or literal strings to score")
    ap.add_argument("--force", action="store_true", help="load even if rerank.enabled is false")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    reranker = Reranker(cfg, force=args.force or bool(args.query))

    if args.check or not args.query:
        print(f"reranker available: {reranker.available}")
        print(f"reason: {reranker.reason}")
        print(f"configured model: {(cfg.get('rerank') or {}).get('model')}")
        if not reranker.available:
            print(
                "\nsearch.py will keep the reciprocal-rank-fusion order and report "
                "'reranker: unavailable'. Nothing will be downloaded."
            )
        return 0

    documents = []
    for item in args.docs:
        path = Path(item)
        documents.append(path.read_text(encoding="utf-8") if path.exists() else item)
    if not documents:
        raise SystemExit("--query needs --docs")
    for score, doc in sorted(zip(reranker.score(args.query, documents), documents), key=lambda x: -x[0]):
        print(f"{score:8.4f}  {doc[:100]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
