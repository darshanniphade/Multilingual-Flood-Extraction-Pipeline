"""
dashboard.py
============

Live web dashboard for the flood-article translation pipeline.

Reads the same config the translator runs with and serves a single-page
dashboard at ``http://127.0.0.1:8765``. It is a *reader*: it never writes to the
run's state, so it is safe to start, stop and restart while a translation is in
flight.

Where the numbers come from
---------------------------
=========================  ==================================================
``checkpoint.json``        headline counters -- completed / skipped / failed /
                           corrupt, src+out tokens, gpu seconds. Rewritten by
                           the run every ``checkpoint.every_n_articles``.
``translation_records.jsonl``  per-article detail -- language, language source,
                           status, chunks, src tokens, batch size, peak VRAM,
                           error text. Tailed incrementally (only appended
                           bytes are ever re-read), so a 60 MB journal costs
                           one read of the new bytes per poll, not a reparse.
``translation_<year>.log`` block history, OOM/split events, model + GPU facts,
                           the exact event-id total, and the final summary.
                           Also tailed incrementally.
``article_index.json``     per-month denominators, scanned once at startup in a
                           background thread (byte-level regex, no JSON parse).
=========================  ==================================================

Rates are not taken from any file: a sampler thread snapshots the counters every
``--sample-interval`` seconds into a ring buffer (persisted to
``dashboard_samples.jsonl`` so history survives a dashboard restart), and the
page differentiates consecutive samples. That is what makes "articles/s right
now" meaningful rather than an average since the run began.

Usage
-----
::

    python dashboard.py -c config_2023.json
    python dashboard.py -c config_2022.json --port 8766
    python dashboard.py -c config_2023.json --no-browser
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
import webbrowser
from collections import Counter, defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

try:  # optional: used only to tell "mid-block" apart from "not running"
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

HERE = Path(__file__).resolve().parent

log = logging.getLogger("dashboard")


# --------------------------------------------------------------------------- #
# Incremental file tailing
# --------------------------------------------------------------------------- #


class Tail:
    """Yields newly-appended complete lines from a growing file.

    Keeps a byte offset plus any partial trailing line, so a poll costs one read
    of the delta rather than a full reparse -- the difference between ~1 ms and
    ~2 s once ``translation_records.jsonl`` reaches 60 MB. A file that shrinks
    (log rotation, ``--no-resume``) is detected and re-read from zero.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        self.pos = 0
        self._buf = b""

    def reset(self) -> None:
        self.pos = 0
        self._buf = b""

    def lines(self) -> Iterator[str]:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size < self.pos:  # truncated or rotated -> replay from the start
            self.reset()
        if size == self.pos:
            return
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.pos)
                data = fh.read(size - self.pos)
                self.pos = fh.tell()
        except OSError:
            return
        parts = (self._buf + data).split(b"\n")
        self._buf = parts.pop()  # possibly-partial final line, kept for next time
        for raw in parts:
            raw = raw.rstrip(b"\r")
            if raw:
                yield raw.decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #

_TS = r"(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"
_NUM = r"[\d.]+[kMB]?"

RE_BLOCK_DONE = re.compile(
    rf"{_TS} \| INFO\s+\| \S+\s+\| block done: (?P<n>\d+) article\(s\) in "
    rf"(?P<secs>[\d.]+)s \| (?P<src>{_NUM}) src-tok, (?P<out>{_NUM}) out-tok \| "
    rf"(?P<done>\d+)/(?P<total>\d+) \| (?P<rate>[\d.]+) art/s \| "
    rf"(?P<tps>{_NUM}) out-tok/s \| ETA (?P<eta>[\d:]+) \| oom=(?P<oom>\d+)"
)
RE_BLOCK_PLAN = re.compile(
    rf"{_TS} \| INFO\s+\| \S+\s+\| block plan: (?P<chunks>{_NUM}) chunk\(s\) \| "
    rf"(?P<langs>\d+) langs \| (?P<batches>\d+) batches \| "
    rf"size min/med/mean/max (?P<mn>\d+)/(?P<med>\d+)/(?P<mean>[\d.]+)/(?P<mx>\d+)"
)
RE_OOM = re.compile(
    rf"{_TS} \| WARNING \| \S+\s+\| oom: split batch (?P<bs>\d+) -> \S+ \| "
    rf"kv_slot_budget now (?P<kv>\d+)"
)
RE_EVENT_IDS = re.compile(r"flood events: (?P<files>\d+) file\(s\) -> (?P<ids>\d+) unique")
RE_INDEX = re.compile(r"index: (?:loaded from cache \(.*?\)|cached \d+ path\(s\)) -> (?P<n>\d+) path")
RE_MODEL = re.compile(
    r"model: loaded in (?P<secs>[\d.]+)s \| (?P<dtype>\S+) \| (?P<device>.+?) \| "
    r"attn=(?P<attn>\S+) \| weights (?P<w>[\d.]+) GiB"
)
RE_BUDGET = re.compile(
    r"gpu: autotuned budget (?P<slots>\d+) slots \((?P<per>\d+) KiB/slot -> KV (?P<kv>[\d.]+) GiB\)"
)
RE_HEADER = re.compile(rf"{_TS} \| INFO\s+\| translator\s+\| flood article translator")
RE_COMPLETE = re.compile(rf"{_TS} \| INFO\s+\| translator\s+\| run complete in (?P<t>[\d:]+)")
RE_LEVEL = re.compile(rf"{_TS} \| (?P<lvl>\w+)\s+\| (?P<mod>\S+)\s+\| (?P<msg>.*)$")


def unhuman(s: str) -> float:
    """Inverse of ``utils.human_count`` -- ``"12.3k"`` -> ``12300.0``."""
    s = s.strip()
    mult = {"k": 1e3, "M": 1e6, "B": 1e9}.get(s[-1:], 1.0)
    if mult != 1.0:
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return 0.0


def _epoch(ts: str) -> float:
    try:
        return time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return 0.0


class LogState:
    """Rolling view of the human log: blocks, OOM events, model + GPU facts."""

    MAX_BLOCKS = 400
    MAX_LINES = 300

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._tail = Tail(path)
        self.blocks: deque[dict[str, Any]] = deque(maxlen=self.MAX_BLOCKS)
        self.plans: deque[dict[str, Any]] = deque(maxlen=self.MAX_BLOCKS)
        # Counted separately from the deques, which forget their oldest entries:
        # "is a block in flight" is plans-seen vs blocks-seen, not len() vs len().
        self.plan_seq = 0
        self.block_seq = 0
        # (chunks, seconds, articles) per finished block, kept ACROSS invocations
        # on purpose: it is what lets a freshly-resumed run estimate when its
        # current block will land, before it has finished a block of its own.
        self.block_rates: deque[tuple[float, float, int]] = deque(maxlen=60)
        self.ooms: deque[dict[str, Any]] = deque(maxlen=200)
        self.lines: deque[str] = deque(maxlen=self.MAX_LINES)
        self.problems: deque[dict[str, str]] = deque(maxlen=80)
        self.levels: Counter[str] = Counter()
        self.event_ids = 0
        self.event_files = 0
        self.indexed = 0
        self.model: dict[str, Any] = {}
        self.gpu: dict[str, Any] = {}
        self.kv_budget = 0
        self.oom_count = 0
        self.run_started = 0.0
        self.last_ts = 0.0
        self.finished_at = 0.0
        self.finished_in = ""

    def poll(self) -> None:
        for line in self._tail.lines():
            self.lines.append(line)
            m = RE_LEVEL.match(line)
            if m:
                self.levels[m.group("lvl")] += 1
                self.last_ts = _epoch(m.group("ts"))
                if m.group("lvl") in ("WARNING", "ERROR", "CRITICAL"):
                    self.problems.append({"ts": m.group("ts"), "level": m.group("lvl"),
                                          "msg": m.group("msg")})

            if RE_HEADER.match(line):
                # A new header means a fresh invocation: block history from the
                # previous one is no longer part of "this run".
                self.run_started = _epoch(RE_HEADER.match(line).group("ts"))
                self.blocks.clear()
                self.plans.clear()
                self.ooms.clear()
                self.plan_seq = self.block_seq = self.oom_count = 0
                self.finished_at = 0.0
                self.finished_in = ""
                continue

            m = RE_BLOCK_DONE.match(line)
            if m:
                self.blocks.append({
                    "ts": m.group("ts"), "t": _epoch(m.group("ts")),
                    "articles": int(m.group("n")), "seconds": float(m.group("secs")),
                    "src_tokens": unhuman(m.group("src")), "out_tokens": unhuman(m.group("out")),
                    "done": int(m.group("done")), "total": int(m.group("total")),
                    "art_per_s": float(m.group("rate")), "out_tok_per_s": unhuman(m.group("tps")),
                    "eta": m.group("eta"), "oom": int(m.group("oom")),
                })
                self.block_seq += 1
                self.oom_count = int(m.group("oom"))
                # The plan line for this block is the one most recently seen.
                if self.plans:
                    self.block_rates.append((float(self.plans[-1].get("chunks") or 0.0),
                                             float(m.group("secs")), int(m.group("n"))))
                continue

            m = RE_BLOCK_PLAN.match(line)
            if m:
                self.plan_seq += 1
                self.plans.append({
                    "ts": m.group("ts"), "t": _epoch(m.group("ts")),
                    "chunks": unhuman(m.group("chunks")),
                    "langs": int(m.group("langs")), "batches": int(m.group("batches")),
                    "min": int(m.group("mn")), "med": int(m.group("med")),
                    "mean": float(m.group("mean")), "max": int(m.group("mx")),
                })
                continue

            m = RE_OOM.match(line)
            if m:
                self.ooms.append({"ts": m.group("ts"), "batch": int(m.group("bs")),
                                  "kv": int(m.group("kv"))})
                self.kv_budget = int(m.group("kv"))
                continue

            m = RE_EVENT_IDS.search(line)
            if m:
                self.event_files = int(m.group("files"))
                self.event_ids = int(m.group("ids"))
                continue

            m = RE_INDEX.search(line)
            if m:
                self.indexed = int(m.group("n"))
                continue

            m = RE_MODEL.search(line)
            if m:
                self.model = {"load_seconds": float(m.group("secs")), "dtype": m.group("dtype"),
                              "device": m.group("device"), "attn": m.group("attn"),
                              "weights_gib": float(m.group("w"))}
                continue

            m = RE_BUDGET.search(line)
            if m:
                self.gpu = {"slots": int(m.group("slots")), "kib_per_slot": int(m.group("per")),
                            "kv_gib": float(m.group("kv"))}
                self.kv_budget = int(m.group("slots"))
                continue

            m = RE_COMPLETE.match(line)
            if m:
                self.finished_at = _epoch(m.group("ts"))
                self.finished_in = m.group("t")


# --------------------------------------------------------------------------- #
# Per-article record aggregation
# --------------------------------------------------------------------------- #

_CHUNK_BUCKETS = [(1, "1"), (2, "2"), (3, "3"), (4, "4"), (8, "5-8"), (16, "9-16"),
                  (32, "17-32"), (64, "33-64"), (128, "65-128"), (10 ** 9, "129+")]


def _chunk_bucket(n: int) -> str:
    for hi, label in _CHUNK_BUCKETS:
        if n <= hi:
            return label
    return "129+"


class RecordState:
    """Rolling aggregation of ``translation_records.jsonl``."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._tail = Tail(path)
        self.articles = 0
        self.statuses: Counter[str] = Counter()
        self.languages: Counter[str] = Counter()
        self.lang_sources: Counter[str] = Counter()
        self.months: Counter[str] = Counter()
        self.month_status: dict[str, Counter[str]] = defaultdict(Counter)
        self.errors: Counter[str] = Counter()
        self.error_samples: dict[str, str] = {}
        self.batch_sizes: Counter[int] = Counter()
        self.chunk_hist: Counter[str] = Counter()
        self.lang_tokens: Counter[str] = Counter()
        self.src_tokens = 0
        self.chunks = 0
        self.gpu_mem_peak = 0.0
        self.malformed = 0

    def poll(self) -> None:
        for line in self._tail.lines():
            try:
                rec = json.loads(line)
            except Exception:
                self.malformed += 1
                continue
            self.articles += 1

            status = rec.get("status") or "ok"
            self.statuses[status] += 1

            aid = rec.get("article_id") or ""
            month = aid.split("/", 1)[0] if "/" in aid else "?"
            self.months[month] += 1
            self.month_status[month][status] += 1

            lang = rec.get("language") or "und"
            self.languages[lang] += 1
            self.lang_sources[rec.get("lang_source") or "unknown"] += 1

            src = int(rec.get("src_tokens") or 0)
            self.src_tokens += src
            self.lang_tokens[lang] += src

            chunks = int(rec.get("chunks") or 0)
            self.chunks += chunks
            self.chunk_hist[_chunk_bucket(chunks)] += 1

            bs = int(rec.get("batch_size") or 0)
            if bs:
                # Bucket to 16 so the histogram is readable at 384 max batch.
                self.batch_sizes[(bs // 16) * 16] += 1

            mem = float(rec.get("gpu_mem_gib") or 0.0)
            if mem > self.gpu_mem_peak:
                self.gpu_mem_peak = mem

            err = rec.get("error")
            if err:
                key = _norm_error(str(err))
                self.errors[key] += 1
                self.error_samples.setdefault(key, str(err)[:400])


_ERR_DIGITS = re.compile(r"\d+")
_ERR_PATH = re.compile(r"[A-Za-z]:\\[^\s'\"]+|/[^\s'\"]{4,}")


def _norm_error(msg: str) -> str:
    """Collapse an error message to a class key so counts group sensibly."""
    msg = _ERR_PATH.sub("<path>", msg)
    msg = _ERR_DIGITS.sub("N", msg)
    return msg[:180]


# --------------------------------------------------------------------------- #
# Per-month denominators (scanned once, off the hot path)
# --------------------------------------------------------------------------- #


def scan_month_totals(index_path: Path) -> Counter[str]:
    """Count article ids per ``YYYY_MM`` in the cached index, without parsing it.

    ``article_index.json`` is tens of megabytes; a byte-level scan for the key
    pattern ``"YYYY_MM/`` costs a sequential read and no allocation per entry.
    The leading quote is what makes it key-specific -- inside a *path value* the
    same ``2023_01/`` substring is preceded by a separator, never by a quote.
    """
    pat = re.compile(rb'"(\d{4}_\d{2})/')
    totals: Counter[str] = Counter()
    try:
        with open(index_path, "rb") as fh:
            carry = b""
            while True:
                buf = fh.read(1 << 23)  # 8 MiB
                if not buf:
                    break
                data = carry + buf
                for m in pat.finditer(data):
                    totals[m.group(1).decode()] += 1
                carry = data[-32:]  # overlap so a key split across reads still matches
    except OSError:
        pass
    return totals


# --------------------------------------------------------------------------- #
# Metrics engine
# --------------------------------------------------------------------------- #


class Metrics:
    """Owns every derived number the page displays."""

    MAX_SAMPLES = 4000

    def __init__(self, cfg: dict[str, Any], cfg_path: Path, sample_interval: float) -> None:
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.paths = cfg["paths"]
        self.state_dir = Path(self.paths["state_dir"])
        self.sample_interval = sample_interval

        self.records = RecordState(self.state_dir / "translation_records.jsonl")
        self.logstate = LogState(self.paths.get("log_file", "translation.log"))
        self.samples_path = self.state_dir / "dashboard_samples.jsonl"
        self.samples: deque[dict[str, Any]] = deque(maxlen=self.MAX_SAMPLES)

        self.month_totals: Counter[str] = Counter()
        self.month_totals_ready = False
        self.checkpoint: dict[str, Any] = {}
        self.checkpoint_mtime = 0.0
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._proc: Any = None
        self._proc_scan_t = 0.0
        # Serialises the incremental readers. Tail.lines() reads (size - pos)
        # bytes and only then advances pos, so two threads entering it together
        # both consume the same region -- every appended line counted twice.
        # The sampler thread and each HTTP request both poll, so this is not a
        # rare race: it fires whenever a page load lands on a sampler tick.
        self._poll_lock = threading.Lock()

        self._load_samples()
        threading.Thread(target=self._scan_months, daemon=True).start()
        threading.Thread(target=self._sampler, daemon=True).start()

    # -- background work ---------------------------------------------------- #

    def _scan_months(self) -> None:
        totals = scan_month_totals(self.state_dir / "article_index.json")
        with self._lock:
            self.month_totals = totals
            self.month_totals_ready = True
        log.info("month totals: %d month(s), %d id(s)", len(totals), sum(totals.values()))

    def _load_samples(self) -> None:
        """Restore the sample ring from disk so restarting the dashboard keeps history."""
        try:
            with open(self.samples_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        self.samples.append(json.loads(line))
                    except Exception:
                        continue
        except OSError:
            pass

    def _sampler(self) -> None:
        while True:
            try:
                self.poll()
                self._take_sample()
            except Exception:  # a dashboard must never die on a bad line
                log.exception("sampler")
            time.sleep(self.sample_interval)

    def _take_sample(self) -> None:
        cp = self.checkpoint
        sample = {
            "t": round(time.time(), 1),
            "articles": self.records.articles,
            "completed": int(cp.get("completed") or 0),
            "src_tokens": int(cp.get("src_tokens") or 0),
            "out_tokens": int(cp.get("out_tokens") or 0),
            "gpu_seconds": float(cp.get("gpu_seconds") or 0.0),
        }
        with self._lock:
            prev = self.samples[-1] if self.samples else None
            # Don't record dead time: if nothing moved and nothing is running,
            # the ring would fill with flat points and squash the live window.
            if prev and sample["completed"] == prev["completed"] and \
                    sample["articles"] == prev["articles"] and \
                    sample["out_tokens"] == prev["out_tokens"]:
                return
            self.samples.append(sample)
        try:
            with open(self.samples_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(sample) + "\n")
        except OSError:
            pass

    # -- polling ------------------------------------------------------------ #

    def poll(self) -> None:
        with self._poll_lock:
            self._read_checkpoint()
            self.records.poll()
            self.logstate.poll()

    def _read_checkpoint(self) -> None:
        path = self.state_dir / "checkpoint.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime == self.checkpoint_mtime:
            return
        try:
            self.checkpoint = json.loads(path.read_text(encoding="utf-8"))
            self.checkpoint_mtime = mtime
        except (OSError, ValueError):
            pass  # mid-rewrite; the next poll picks it up

    # -- derived ------------------------------------------------------------ #

    def _live_rates(self) -> dict[str, float]:
        """Rates over the trailing window, from the sample ring.

        Summed pair-by-pair rather than as (last - first) / (last.t - first.t):
        the ring is persisted across dashboard restarts and the run itself gets
        stopped and resumed, so it contains gaps. Spanning one divides a real
        delta by hours of downtime -- or, worse, attributes a whole resumed
        checkpoint to the seconds after a restart.
        """
        with self._lock:
            pts = list(self.samples)[-60:]
        out = {"art_per_s": 0.0, "out_tok_per_s": 0.0, "src_tok_per_s": 0.0,
               "gpu_busy": 0.0, "window_seconds": 0.0}
        if len(pts) < 2:
            return out

        max_gap = max(60.0, self.sample_interval * 4)
        dt_sum = d_art = d_out = d_src = d_gpu = 0.0
        for a, b in zip(pts, pts[1:]):
            dt = b["t"] - a["t"]
            if dt <= 0 or dt > max_gap:
                continue  # a stop/restart sits between these two samples
            # Prefer the record journal for article counts: it advances with
            # every flushed block, while `completed` only moves on a checkpoint
            # write. A negative delta means the counters were reset.
            art = max(b["articles"] - a["articles"], b["completed"] - a["completed"])
            if art < 0:
                continue
            dt_sum += dt
            d_art += art
            d_out += max(0, b["out_tokens"] - a["out_tokens"])
            d_src += max(0, b["src_tokens"] - a["src_tokens"])
            d_gpu += max(0.0, b["gpu_seconds"] - a["gpu_seconds"])

        if dt_sum <= 0:
            return out
        out["window_seconds"] = dt_sum
        out["art_per_s"] = d_art / dt_sum
        out["out_tok_per_s"] = d_out / dt_sum
        out["src_tok_per_s"] = d_src / dt_sum
        out["gpu_busy"] = 100.0 * d_gpu / dt_sum
        return out

    def _process_info(self) -> dict[str, Any]:
        """Find the translator process for *this* config, if it is running.

        Worth the trouble: a block is 24k articles, so between the "block plan"
        and "block done" lines the run writes nothing at all for tens of
        minutes. Judging liveness by file activity alone would report a busy GPU
        as "Idle" for most of the run.
        """
        if psutil is None:
            return {"available": False, "alive": False}

        proc = self._proc
        if proc is not None:
            try:
                if not proc.is_running():
                    proc = self._proc = None
            except Exception:
                proc = self._proc = None

        if proc is None and time.time() - self._proc_scan_t > 10:
            self._proc_scan_t = time.time()
            needle, cfgname = "translate_flood_articles", self.cfg_path.name.lower()
            for cand in psutil.process_iter(["cmdline"]):
                try:
                    cl = " ".join(cand.info.get("cmdline") or []).lower()
                except Exception:
                    continue
                if needle in cl and cfgname in cl:
                    proc = self._proc = cand
                    break

        if proc is None:
            return {"available": True, "alive": False}
        try:
            info = {
                "available": True, "alive": True, "pid": proc.pid,
                "rss_gib": proc.memory_info().rss / 1024 ** 3,
                # First call seeds the baseline; later calls are the average
                # since the previous snapshot, which is what we want to show.
                "cpu_percent": proc.cpu_percent(None),
                "started": proc.create_time(),
                "workers": len(proc.children()),
            }
        except Exception:
            return {"available": True, "alive": False}
        return info

    def _is_running(self, proc: dict[str, Any]) -> bool:
        """Whether the translator looks alive."""
        if self.logstate.finished_at:
            return False
        if proc.get("alive"):
            return True
        with self._lock:
            last = self.samples[-1]["t"] if self.samples else 0.0
        return time.time() - max(last, self.logstate.last_ts) < 180

    def snapshot(self) -> dict[str, Any]:
        self.poll()
        cp = self.checkpoint
        rec = self.records
        ls = self.logstate

        total = ls.event_ids or ls.indexed or sum(self.month_totals.values()) or 0
        completed = max(int(cp.get("completed") or 0), rec.articles)
        remaining = max(0, total - completed)
        rates = self._live_rates()
        proc = self._process_info()

        # A block is in flight when its plan line has appeared but its "block
        # done" line has not. Nothing else is written during that window, so
        # this is the only progress signal there is mid-block.
        # Chunks/second measured over every block ever seen. Chunks, not
        # articles: block duration tracks decoder work, and a 2023 article is
        # roughly twice the length of a 2021 one, so articles/second is not
        # portable between blocks while chunks/second very nearly is.
        chunks_per_s = art_per_s_block = None
        if ls.block_rates:
            tot_s = sum(s for _, s, _ in ls.block_rates)
            if tot_s > 0:
                chunks_per_s = sum(c for c, _, _ in ls.block_rates) / tot_s
                art_per_s_block = sum(a for _, _, a in ls.block_rates) / tot_s

        in_flight = None
        if ls.plan_seq > ls.block_seq and ls.plans:
            pl = dict(ls.plans[-1])
            pl["elapsed"] = max(0.0, time.time() - pl.get("t", 0.0))
            # Number it by blocks finished overall, not by this invocation's
            # plan counter -- after a resume the latter restarts at 1 and calls
            # the fifth block of the corpus "block 1".
            pl["index"] = len(ls.block_rates) + 1
            if chunks_per_s:
                est = pl.get("chunks", 0.0) / chunks_per_s
                pl["estimated_seconds"] = est
                pl["remaining_seconds"] = max(0.0, est - pl["elapsed"])
                pl["finish_at"] = time.time() + max(0.0, est - pl["elapsed"])
            in_flight = pl

        # Elapsed runs against the wall clock while the process is alive: the log
        # is silent for the whole of a 24k-article block, so anchoring the end to
        # the last log line would freeze the timer (and inflate the average rate)
        # for tens of minutes at a stretch. Once the run stops, it freezes at the
        # last thing that actually happened.
        running = self._is_running(proc)
        end = time.time() if running else (ls.finished_at or ls.last_ts or time.time())
        elapsed = max(1e-6, end - (ls.run_started or self.started_at))
        # The average is measured over finished blocks, NOT as completed/elapsed:
        # `completed` includes every article carried over from previous
        # invocations while `elapsed` covers only this one, so after any resume
        # that division reports a rate the machine never achieved.
        avg_rate = art_per_s_block or 0.0
        # ETA prefers the block average -- it is measured over hours of real
        # work, where the live rate is a few minutes of sampling that reads zero
        # for the whole of a block and then spikes when one lands.
        rate = avg_rate or rates["art_per_s"]
        eta = remaining / rate if rate > 0 else None

        out_tokens = float(cp.get("out_tokens") or 0)
        src_tokens = float(cp.get("src_tokens") or 0) or float(rec.src_tokens)
        gpu_seconds = float(cp.get("gpu_seconds") or 0.0)
        # gpu_seconds and session_wall_seconds are written together and cover the
        # same window, so their ratio stays honest even when the checkpoint is a
        # block behind. Dividing by this dashboard's `elapsed` would not: after a
        # resume it charges the previous invocation's GPU time to this one's
        # clock, which is how "GPU busy 746%" happens.
        session_wall = float(cp.get("session_wall_seconds") or 0.0)
        gpu_busy = min(100.0, 100.0 * gpu_seconds / session_wall) if session_wall > 0 else None

        with self._lock:
            samples = list(self.samples)[-600:]
            month_totals = dict(self.month_totals)
            months_ready = self.month_totals_ready

        months = []
        keys = sorted(set(month_totals) | set(rec.months))
        for k in keys:
            st = rec.month_status.get(k, Counter())
            months.append({
                "month": k,
                "total": month_totals.get(k, 0),
                "done": rec.months.get(k, 0),
                "ok": st.get("ok", 0),
                "skipped_english": st.get("skipped_english", 0),
                "failed": st.get("failed", 0),
                "corrupt": st.get("corrupt", 0),
            })

        return {
            "generated_at": time.time(),
            "config": {
                "path": str(self.cfg_path),
                "events_dir": self.paths.get("events_dir"),
                "data_dir": self.paths.get("data_dir"),
                "output_dir": self.paths.get("output_dir"),
                "state_dir": self.paths.get("state_dir"),
                "log_file": self.paths.get("log_file"),
                "model": self.cfg.get("model", {}).get("name"),
                "workers": self.cfg.get("cpu", {}).get("workers"),
                "block_size": self.cfg.get("cpu", {}).get("block_size"),
                "target_peak_vram_gib": self.cfg.get("gpu", {}).get("target_peak_vram_gib"),
                "max_batch_size": self.cfg.get("gpu", {}).get("max_batch_size"),
                "chunk_target_tokens": self.cfg.get("chunking", {}).get("target_tokens"),
                "chunk_max_tokens": self.cfg.get("chunking", {}).get("max_tokens"),
                "num_beams": self.cfg.get("generation", {}).get("num_beams"),
            },
            "progress": {
                "total": total,
                "completed": completed,
                "remaining": remaining,
                "percent": (100.0 * completed / total) if total else 0.0,
                "event_files": ls.event_files,
                "indexed": ls.indexed,
                "months_ready": months_ready,
            },
            "status": {
                "ok": rec.statuses.get("ok", 0),
                "skipped_english": rec.statuses.get("skipped_english",
                                                    int(cp.get("skipped_english") or 0)),
                "failed": rec.statuses.get("failed", int(cp.get("failed") or 0)),
                "corrupt": rec.statuses.get("corrupt", int(cp.get("corrupt") or 0)),
                "written": rec.articles - rec.statuses.get("corrupt", 0),
            },
            "tokens": {
                "src": src_tokens,
                "out": out_tokens,
                "ratio": (out_tokens / src_tokens) if src_tokens else 0.0,
                "src_per_article": (rec.src_tokens / rec.articles) if rec.articles else 0.0,
                "chunks": rec.chunks,
                "chunks_per_article": (rec.chunks / rec.articles) if rec.articles else 0.0,
            },
            "rates": {
                **rates,
                "avg_art_per_s": avg_rate,
                "avg_out_tok_per_s": (out_tokens / gpu_seconds) if gpu_seconds else 0.0,
                "eta_seconds": eta,
                "elapsed_seconds": elapsed,
                "running": running,
                "finished_in": ls.finished_in,
            },
            "process": proc,
            "in_flight": in_flight,
            "block_counts": {"planned": ls.plan_seq, "done": ls.block_seq,
                             "measured": len(ls.block_rates),
                             "chunks_per_s": chunks_per_s,
                             "block_art_per_s": art_per_s_block},
            "gpu": {
                "seconds": gpu_seconds,
                "busy_percent": gpu_busy,
                "session_wall_seconds": session_wall,
                "peak_gib": rec.gpu_mem_peak,
                "target_gib": self.cfg.get("gpu", {}).get("target_peak_vram_gib"),
                "kv_budget": ls.kv_budget,
                "oom_count": ls.oom_count or len(ls.ooms),
                "model": ls.model,
                "autotune": ls.gpu,
                "ooms": list(ls.ooms)[-40:],
            },
            "languages": [{"lang": k, "articles": v, "src_tokens": rec.lang_tokens.get(k, 0)}
                          for k, v in rec.languages.most_common(24)],
            "lang_sources": dict(rec.lang_sources),
            "months": months,
            "batch_sizes": [{"bucket": k, "count": v} for k, v in sorted(rec.batch_sizes.items())],
            "chunk_hist": [{"bucket": lbl, "count": rec.chunk_hist.get(lbl, 0)}
                           for _, lbl in _CHUNK_BUCKETS if rec.chunk_hist.get(lbl, 0)],
            "errors": [{"key": k, "count": v, "sample": rec.error_samples.get(k, "")}
                       for k, v in rec.errors.most_common(20)],
            "blocks": list(ls.blocks)[-60:],
            "plans": list(ls.plans)[-60:],
            "problems": list(ls.problems)[-40:],
            "log_levels": dict(ls.levels),
            "samples": samples,
            "checkpoint": cp,
        }


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #


class Handler(BaseHTTPRequestHandler):
    metrics: Metrics = None  # type: ignore[assignment]
    server_version = "TranslatorDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                body = (HERE / "dashboard.html").read_bytes()
            except OSError:
                self._send(500, b"dashboard.html not found next to dashboard.py",
                           "text/plain; charset=utf-8")
                return
            self._send(200, body, "text/html; charset=utf-8")
        elif path == "/api/metrics":
            try:
                data = self.metrics.snapshot()
            except Exception as exc:  # surface the failure in the page, not a blank screen
                log.exception("snapshot")
                data = {"error": f"{type(exc).__name__}: {exc}"}
            self._send(200, json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif path == "/api/log":
            lines = list(self.metrics.logstate.lines)[-200:]
            self._send(200, "\n".join(lines).encode("utf-8"), "text/plain; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")


class DashboardServer(ThreadingHTTPServer):
    """Threaded server that refuses to share its port.

    ``allow_reuse_address`` defaults to true, and on Windows that flag behaves
    like ``SO_REUSEPORT``: a second process binds a port that is already in use
    and the two silently split the incoming connections, so half the requests
    are answered by somebody else's server. Turning it off turns a port clash
    into a startup error naming the port, which is the failure we can act on.
    """

    allow_reuse_address = False
    daemon_threads = True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="dashboard",
        description="Live web dashboard for the flood-article translation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config", default=str(HERE / "config_2023.json"),
                   help="the config the translation run uses")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--sample-interval", type=float, default=10.0,
                   help="seconds between throughput samples")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    Handler.metrics = Metrics(cfg, cfg_path, args.sample_interval)

    try:
        httpd = DashboardServer((args.host, args.port), Handler)
    except OSError as exc:
        log.error("cannot bind %s:%d (%s) -- something else is already listening. "
                  "Pick another port with --port.", args.host, args.port, exc)
        return 1
    url = f"http://{args.host}:{args.port}/"
    log.info("dashboard: %s", url)
    log.info("  config : %s", cfg_path)
    log.info("  state  : %s", cfg["paths"]["state_dir"])
    log.info("  log    : %s", cfg["paths"].get("log_file"))
    log.info("Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
