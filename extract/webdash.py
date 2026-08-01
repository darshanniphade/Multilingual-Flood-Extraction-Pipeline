"""Local web dashboard for the flood-extraction pipeline.

Serves a live 2D dashboard fed by the pipeline's own output files. It only tails
<output_root>/*.jsonl and *.failed.jsonl, so it works no matter how the extraction
was started (dashboard.py, extract.py, any number of restarts) and never touches
the run itself. Safe to start, stop and restart mid-run.

    python webdash.py                                  # uses config_reliefweb.json
    python webdash.py --config config_2022.json --port 8790
    python webdash.py --recount                        # ignore the cached input totals

Port 8790 by default: 8765 belongs to another project's downloader.
"""

import argparse
import json
import os
import re
import threading
import time
from collections import Counter, deque
from fnmatch import fnmatch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLL_SECONDS = 1.5
ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-\d{2}")

# Event keys that are structure rather than measured impact - excluded from the
# "impact fields captured" tally so it reflects what was actually extracted.
NON_IMPACT_FIELDS = {"summary", "country", "state", "province", "district", "city",
                     "village", "location", "event_date", "start_date", "end_date"}


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


class Tailer:
    """Incrementally consumes the pipeline's JSONL outputs and keeps running
    tallies. First poll catches up on everything already on disk; after that
    only appended bytes are read, so polling stays cheap for a million-line run."""

    def __init__(self, cfg: dict, recount: bool = False):
        self.cfg = cfg
        self.field = cfg["output"]["extraction_field"]
        self.title_key = cfg["output"].get("title_key", "translated_title")
        self.label = cfg["output"].get("partition_column", "month")
        self.out_root = Path(cfg["paths"]["output_root"])
        self.started = time.time()

        self.parts = {
            name: {
                "total": total, "ok": 0, "fail": 0, "skipped": 0,
                "floods": 0, "dated": 0, "verifiable": 0,
                "last_change": 0.0,
                # Resuming re-attempts anything in .failed.jsonl, so one article can
                # be logged repeatedly across runs. Count article_ids, not lines,
                # or a part reads past 100% (2022_02 once showed 8649/8648).
                "ok_ids": set(), "fail_ids": set(),
            }
            for name, total in self._input_totals(recount).items()
        }

        self.offsets: dict[Path, int] = {}
        self.buffers: dict[Path, bytes] = {}
        self.feed = deque(maxlen=40)
        self.locations = Counter()
        self.countries = Counter()
        self.event_fields = Counter()
        self.errors = Counter()
        self.flood_months = [0] * 12          # seasonality of extracted flood dates
        self.events_total = 0
        self.samples = deque(maxlen=1200)     # (t, done) - rate window source
        self.rate_history = deque(maxlen=180)  # smoothed rate/s for the trend chart
        self.last_change = 0.0
        # Progress cannot move while the run re-attempts articles already logged
        # in .failed.jsonl - resume only reads <part>.jsonl, so every restart
        # retries them, and a retry that succeeds is ok+1/fail-1, i.e. done
        # unchanged. Count records ingested separately or such a stretch reads as
        # IDLE for hours while the model is working flat out.
        self.ingested = 0
        self.last_ingest = 0.0
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- totals

    def _input_totals(self, recount: bool) -> dict[str, int]:
        """How many source files each partition holds - the denominator for every
        percentage on the page. Counting means walking the whole input tree
        (~1M entries for the ReliefWeb archive, tens of seconds), and the corpus
        is static, so the result is cached beside the output. --recount to redo it."""
        input_root = Path(self.cfg["paths"]["input_root"])
        sub = self.cfg["input"]["articles_subdir"]
        pattern = self.cfg["input"].get("file_glob", "article_*.json")
        cache_path = self.out_root / ".input_totals.json"
        key = f"{input_root}|{sub}|{pattern}"

        if not recount:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("key") == key:
                    return {k: int(v) for k, v in cached["totals"].items()}
            except (OSError, ValueError, KeyError):
                pass

        print(f"counting input files under {input_root} ...", flush=True)
        totals = {}
        for d in sorted(p for p in input_root.iterdir() if p.is_dir()):
            # scandir rather than glob: glob's per-path stat() made the server
            # take ~30s to bind on a 150k-file tree, and this one is 7x bigger
            with os.scandir(d / sub) as it:
                totals[d.name] = sum(1 for e in it if fnmatch(e.name, pattern))
        print(f"  {sum(totals.values()):,} files across {len(totals)} partitions", flush=True)

        try:
            self.out_root.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"key": key, "totals": totals}), encoding="utf-8"
            )
        except OSError:
            pass  # a read-only output dir is not a reason to refuse to serve
        return totals

    # ------------------------------------------------------------- tail loop

    def run(self):
        prev_done = -1
        while True:
            for name in self.parts:
                self._tail(self.out_root / f"{name}.jsonl", name, failed=False)
                self._tail(self.out_root / f"{name}.failed.jsonl", name, failed=True)
            now = time.time()
            with self.lock:
                done = sum(p["ok"] + p["fail"] for p in self.parts.values())
                self.samples.append((now, done))
                if prev_done < 0:
                    # the first pass books everything already on disk - catch-up,
                    # not live work, and it would read as activity for 90s
                    self.ingested = 0
                    self.last_ingest = 0.0
                    for p in self.parts.values():
                        p["last_change"] = 0.0
                if done != prev_done and prev_done >= 0:
                    self.last_change = now
                prev_done = done
                self.rate_history.append(round(self._rate(now), 3))
            time.sleep(POLL_SECONDS)

    def _tail(self, path: Path, part: str, failed: bool):
        try:
            size = path.stat().st_size
        except OSError:
            return
        off = self.offsets.get(path, 0)
        if size < off:  # file replaced/truncated - re-read it from the top
            off = 0
            self.buffers.pop(path, None)
        if size <= off:
            return
        with path.open("rb") as fh:
            fh.seek(off)
            chunk = fh.read(size - off)
        self.offsets[path] = size
        data = self.buffers.get(path, b"") + chunk
        *lines, tail = data.split(b"\n")
        self.buffers[path] = tail
        with self.lock:
            for raw in lines:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                self._ingest(rec, part, failed)

    def _touch(self, p: dict) -> None:
        """One accepted record: partition freshness plus the global ingest counter
        (lock held). Monotonic - unlike ok/fail it is never walked back."""
        now = time.time()
        p["last_change"] = now
        self.ingested += 1
        self.last_ingest = now

    def _ingest(self, rec: dict, part: str, failed: bool):
        p = self.parts[part]
        aid = str(rec.get("article_id", "?"))
        if failed:
            # already counted under either outcome - a repeat attempt, not new work
            if aid in p["fail_ids"] or aid in p["ok_ids"]:
                return
            p["fail_ids"].add(aid)
            p["fail"] += 1
            self._touch(p)
            error = str(rec.get("error") or "failed")
            self.errors[error.split(":")[0].strip() or "unknown"] += 1
            self.feed.appendleft({
                "ok": False, "flood": False, "id": aid, "part": part,
                "title": error[:160], "dates": [], "locations": [], "country": "",
            })
            return
        if aid in p["ok_ids"]:
            return
        p["ok_ids"].add(aid)
        # a later success supersedes an earlier logged failure for the same article
        if aid in p["fail_ids"]:
            p["fail_ids"].discard(aid)
            p["fail"] -= 1
        self._touch(p)
        p["ok"] += 1
        # skipped by the config's text prefilter - written, but never model-scored
        skipped = bool(rec.get("prefiltered"))
        if skipped:
            p["skipped"] += 1

        ext = rec.get(self.field) or {}
        flood = bool(ext.get("contains_flood_event"))
        dates = ext.get("flood_dates") if isinstance(ext.get("flood_dates"), list) else []
        locs = ext.get("flooded_locations") if isinstance(ext.get("flooded_locations"), list) else []
        country = str(rec.get("country") or "").strip()

        if flood:
            p["floods"] += 1
            self.locations.update(str(v).strip() for v in locs if str(v).strip())
            if country:
                self.countries.update(
                    c.strip() for c in country.split(",") if c.strip()
                )
            for value in dates:
                match = ISO_DATE.match(str(value).strip())
                if match:
                    month = int(match.group(2))
                    if 1 <= month <= 12:
                        self.flood_months[month - 1] += 1
            events = ext.get("events")
            if isinstance(events, list):
                self.events_total += len(events)
                for event in events:
                    if isinstance(event, dict):
                        self.event_fields.update(
                            k for k in event if k not in NON_IMPACT_FIELDS
                        )
        if dates:
            p["dated"] += 1
        if ext.get("is_verifiable_flood"):
            p["verifiable"] += 1

        # Prefiltered records are counted everywhere but kept out of the feed:
        # at a ~78% skip rate they would crowd out every actual extraction.
        if not skipped:
            self.feed.appendleft({
                "ok": True, "flood": flood, "id": aid, "part": part,
                "title": str(rec.get(self.title_key) or "").strip()[:200],
                "dates": [str(v) for v in dates[:3]],
                "locations": [str(v) for v in locs[:3]],
                "country": country[:60],
            })

    def _rate(self, now: float) -> float:
        """Articles/s over roughly the last 60s of samples (lock held)."""
        base = None
        for t, done in reversed(self.samples):
            base = (t, done)
            if now - t >= 60:
                break
        if base is None or now - base[0] < 5:
            return 0.0
        return (self.samples[-1][1] - base[1]) / (self.samples[-1][0] - base[0])

    # ----------------------------------------------------------------- stats

    def stats(self) -> dict:
        now = time.time()
        with self.lock:
            parts = []
            grand = {"total": 0, "done": 0, "ok": 0, "fail": 0, "skipped": 0,
                     "floods": 0, "dated": 0, "verifiable": 0}
            for name, p in self.parts.items():
                done = p["ok"] + p["fail"]
                fresh = bool(p["last_change"]) and now - p["last_change"] < 120
                if p["total"] and done >= p["total"]:
                    # every article accounted for, but still taking writes: the
                    # run is working through this partition's failure backlog
                    state = "retry" if fresh else "done"
                elif fresh:
                    state = "running"
                elif done:
                    state = "partial"
                else:
                    state = "queued"
                parts.append({
                    "name": name, "total": p["total"], "done": done, "ok": p["ok"],
                    "floods": p["floods"], "fails": p["fail"], "skipped": p["skipped"],
                    "verifiable": p["verifiable"], "state": state,
                    "pct": round(100 * done / p["total"], 1) if p["total"] else 0,
                })
                grand["total"] += p["total"]
                grand["done"] += done
                grand["ok"] += p["ok"]
                grand["fail"] += p["fail"]
                grand["skipped"] += p["skipped"]
                grand["floods"] += p["floods"]
                grand["dated"] += p["dated"]
                grand["verifiable"] += p["verifiable"]

            rate = self._rate(now)
            first = self.samples[0] if self.samples else (now, grand["done"])
            span = now - first[0]
            avg = (grand["done"] - first[1]) / span if span >= 5 else 0.0
            remaining = grand["total"] - grand["done"]
            # Over a multi-day run the 60s rate is far too jittery to project
            # from - a lull between partitions swings the finish date by months -
            # so once the session has enough history the average carries the ETA.
            # Withheld entirely below a minute: the first poll books everything
            # already on disk, and a rate measured across that warm-up is noise.
            basis = avg if span >= 300 and avg > 0 else rate
            eta = int(remaining / basis) if basis > 0 and span >= 60 else None
            return {
                "model": self.cfg["model"]["name"],
                "workers": self.cfg["run"]["workers"],
                "ctx": self.cfg["generation"]["num_ctx"],
                "label": self.label,
                "input_root": str(self.cfg["paths"]["input_root"]),
                "output_root": str(self.out_root),
                "live": bool(self.last_change and now - self.last_change < 90),
                # working, but on articles already counted - see _touch
                "active": bool(self.last_ingest and now - self.last_ingest < 90),
                "since_change": int(now - self.last_change) if self.last_change else None,
                "since_ingest": int(now - self.last_ingest) if self.last_ingest else None,
                "ingested": self.ingested,
                "uptime": int(now - self.started),
                "grand": {**grand, "pct": round(100 * grand["done"] / grand["total"], 3)
                          if grand["total"] else 0},
                "rate_now": round(rate, 3),
                "rate_avg": round(avg, 3),
                "eta_seconds": eta,
                "parts": parts,
                "locations": self.locations.most_common(12),
                "countries": self.countries.most_common(12),
                "event_fields": self.event_fields.most_common(12),
                "events_total": self.events_total,
                "flood_months": list(self.flood_months),
                "errors": self.errors.most_common(8),
                "feed": list(self.feed),
                "rate_history": list(self.rate_history),
            }


def make_handler(tailer: Tailer):
    html_path = HERE / "webdash.html"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                # re-read per request so the page can be edited without restarts
                body = html_path.read_bytes()
                ctype = "text/html; charset=utf-8"
            elif self.path == "/stats":
                body = json.dumps(tailer.stats()).encode("utf-8")
                ctype = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the console quiet
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Web dashboard for the extraction run")
    parser.add_argument("--config", type=Path, default=HERE / "config_reliefweb.json")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--recount", action="store_true",
                        help="re-walk the input tree instead of using the cached totals")
    args = parser.parse_args()

    tailer = Tailer(load_config(args.config), recount=args.recount)
    threading.Thread(target=tailer.run, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(tailer))
    print(f"dashboard: http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
