"""Flood event extraction over English articles using a local Ollama model.

Input : <input_root>/<part>/<articles_subdir>/<file_glob>
Output: <output_root>/<part>.jsonl   (one line per article)
        <output_root>/<part>.failed.jsonl

A "part" is whatever the input tree is partitioned by - YYYY_MM for the news
corpora, YYYY for the ReliefWeb archive. Input layout and field names are all
config-driven, so a new corpus needs a new config, not a new script.

Settings live in config.json and the prompt in system_prompt.txt, both next to
this file - edit those rather than this script. CLI flags override config.json.
Runs are resumable: article_ids already present in the output JSONL are skipped.

    python extract.py                               # everything, using config.json
    python extract.py --months 2021_07 2021_08      # selected parts
    python extract.py --config config_reliefweb.json --months 2015
    python extract.py --limit 20 --workers 1        # smoke test
"""

import argparse
import json
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import ollama

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_system_prompt(cfg: dict) -> str:
    return Path(cfg["paths"]["system_prompt_file"]).read_text(encoding="utf-8").strip()


def month_dirs(cfg: dict, selected: list[str] | None) -> list[Path]:
    root = Path(cfg["paths"]["input_root"])
    months = sorted(d for d in root.iterdir() if d.is_dir())
    if selected:
        wanted = set(selected)
        months = [m for m in months if m.name in wanted]
    return months


def article_files(cfg: dict, month: Path) -> list[Path]:
    pattern = cfg["input"].get("file_glob", "article_*.json")
    return sorted((month / cfg["input"]["articles_subdir"]).glob(pattern))


def already_done(out_file: Path) -> set[str]:
    if not out_file.exists():
        return set()
    done = set()
    with out_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["article_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def normalize_date(value) -> str | None:
    """'2015-05-20T00:00:00+00:00' -> '2015-05-20'. The prompt asks for bare
    YYYY-MM-DD, and a trailing timestamp is noise the model has to step over.
    Anything not starting with an ISO date passes through untouched."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = ISO_DATE.match(text)
    return match.group(1) if match else text


def resolve_publish_date(cfg: dict, month_name: str, article_id: str, article: dict) -> str | None:
    """Give the model a stated publication year, so 'Saturday, January 9' resolves
    instead of being dropped or guessed at. Two sources, in order of preference:

    publish_date_inline - the article file carries the date itself (ReliefWeb).
    publish_date_root   - a parallel tree with the same layout holds it (news corpora).

    Neither configured, or the field missing -> None, and no PUBLISHED: line is sent."""
    inp = cfg["input"]
    field = inp["publish_date_field"]
    if inp.get("publish_date_inline"):
        return normalize_date(article.get(field))
    root = inp.get("publish_date_root")
    if not root:
        return None
    path = Path(root) / month_name / inp["articles_subdir"] / f"{article_id}.json"
    try:
        with path.open(encoding="utf-8") as fh:
            return normalize_date(json.load(fh).get(field))
    except (OSError, json.JSONDecodeError):
        return None


@lru_cache(maxsize=8)
def _compile(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def prefilter(cfg: dict) -> re.Pattern | None:
    """The cheap text gate that decides whether an article is worth a model call.

    PHASE 1 of the prompt only sets contains_flood_event when the article names a
    location explicitly flooded / submerged / inundated, so an article containing
    no flood vocabulary at all cannot qualify - and on the ReliefWeb archive that
    is ~78% of the corpus. Skipping those turns a 10-day run into a ~2.5-day one.

    The pattern lives in config so it can be widened without touching code, and it
    is deliberately multilingual: the archive is 85% English but 8% French and 5%
    Spanish, and an English-only gate silently dropped every 'Inondations' report.
    It also matches rainfall/storm causes the prompt does not strictly require -
    insurance that costs ~5 points of skip rate. Measured against 117
    model-confirmed flood reports, the shipped pattern misses none.

    Returns None when disabled, in which case every article goes to the model."""
    pf = cfg["input"].get("prefilter") or {}
    if not pf.get("enabled") or not pf.get("pattern"):
        return None
    return _compile(pf["pattern"])


def no_flood_result() -> dict:
    """The prompt's own EARLY EXIT object, for articles the gate skipped. A fresh
    dict per call - these go into records that are written and must not alias."""
    return {
        "contains_flood_event": False,
        "flood_dates": [],
        "is_verifiable_flood": False,
        "flooded_locations": [],
        "events": [],
    }


def build_user_message(cfg: dict, article: dict, publish_date: str | None = None) -> str:
    inp = cfg["input"]
    title = article.get(inp["title_field"], "")
    text = article.get(inp["text_field"], "")
    limit = inp["max_article_chars"]
    if limit and len(text) > limit:
        text = text[:limit]
    header = f"PUBLISHED: {publish_date}\n\n" if publish_date else ""
    return f"{header}TITLE: {title}\n\nARTICLE:\n{text}"


def parse_response(raw: str) -> dict:
    """Ollama's format='json' normally returns clean JSON, but be forgiving."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # strip a stray ```json fence or leading prose, then take the outermost object
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(raw[start : end + 1])


def extract_one(
    client: ollama.Client,
    cfg: dict,
    system_prompt: str,
    article: dict,
    publish_date: str | None = None,
) -> dict:
    gen, mdl = cfg["generation"], cfg["model"]
    options = {
        "temperature": gen["temperature"],
        "top_p": gen["top_p"],
        "num_ctx": gen["num_ctx"],
    }
    # A ceiling on generated tokens, and deliberately a loose one - this is a
    # safety rail, not a throughput knob. Measured over 23,685 calls: the median
    # answer is 42 tokens (the EARLY EXIT object) and the longest legitimate one
    # was 982, so 1024 truncates nothing real. Capping tighter is actively worse
    # than leaving it off: a cut-off answer is unparseable JSON, and at
    # temperature 0 the retries reproduce it verbatim, so one long-but-valid
    # generation becomes three truncated failures. Omit to leave unbounded.
    if gen.get("num_predict"):
        options["num_predict"] = gen["num_predict"]
    response = client.chat(
        model=mdl["name"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(cfg, article, publish_date)},
        ],
        format=gen["format"],
        think=mdl["think"],
        keep_alive=mdl["keep_alive"],
        options=options,
    )
    return parse_response(response["message"]["content"])


def process_one(
    client: ollama.Client, cfg: dict, system_prompt: str, path: Path, month_name: str
) -> tuple[bool, dict]:
    """Returns (ok, record). Runs on a worker thread; does no file writing."""
    inp = cfg["input"]
    article = json.loads(path.read_text(encoding="utf-8"))
    article_id = str(article.get(inp["id_field"], path.stem))
    publish_date = resolve_publish_date(cfg, month_name, article_id, article)
    run, out = cfg["run"], cfg["output"]

    def build(result: dict, skipped: bool = False) -> dict:
        record = {
            "article_id": article_id,
            "month": month_name,
            out["extraction_field"]: result,
        }
        if skipped:
            # Marked, not omitted: every source document still gets a line, so the
            # output stays complete and these are re-runnable if the gate widens.
            record["prefiltered"] = True
        if publish_date:
            record["publish_date"] = publish_date
        if out["include_title"]:
            record[out.get("title_key", "translated_title")] = article.get(
                inp["title_field"], ""
            )
        # Corpus metadata the model never sees, carried through so downstream
        # work can group/verify without reopening a million source files.
        for key in inp.get("passthrough_fields", []):
            value = article.get(key)
            if value not in (None, "", [], {}):
                record[key] = value
        return record

    gate = prefilter(cfg)
    if gate is not None:
        haystack = f"{article.get(inp['title_field'], '')}\n{article.get(inp['text_field'], '')}"
        if not gate.search(haystack):
            return True, build(no_flood_result(), skipped=True)

    error = None
    for attempt in range(run["retries"] + 1):
        try:
            result = extract_one(client, cfg, system_prompt, article, publish_date)
            return True, build(result)
        except Exception as exc:  # network, timeout, or unparseable JSON
            error = f"{type(exc).__name__}: {exc}"
            if attempt < run["retries"]:
                time.sleep(run["retry_backoff_seconds"] * (attempt + 1))

    return False, {"article_id": article_id, "error": error}


def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class Stats:
    """Counters shared between the worker pool and the progress ticker."""

    RECENT_WINDOW = 60.0  # seconds of history behind the "recent" rate

    def __init__(self, total: int, label: str, workers: int):
        self.total, self.label, self.workers = total, label, workers
        self.lock = threading.Lock()
        self.started = time.time()
        self.done = self.ok = self.failed = self.skipped = 0
        self.flood = self.dated = self.verifiable = 0
        self.recent: deque[float] = deque()

    def record(self, ok: bool, record: dict, extraction_field: str) -> None:
        now = time.time()
        with self.lock:
            self.done += 1
            self.recent.append(now)
            while self.recent and now - self.recent[0] > self.RECENT_WINDOW:
                self.recent.popleft()
            if not ok:
                self.failed += 1
                return
            self.ok += 1
            if record.get("prefiltered"):
                self.skipped += 1
            result = record.get(extraction_field) or {}
            if result.get("contains_flood_event"):
                self.flood += 1
            if result.get("flood_dates"):
                self.dated += 1
            if result.get("is_verifiable_flood"):
                self.verifiable += 1

    def snapshot(self) -> str:
        with self.lock:
            elapsed = time.time() - self.started
            done, total = self.done, self.total
            rate = done / elapsed if elapsed else 0.0
            # ignore the recent window until it spans enough time to be meaningful,
            # otherwise the first couple of completions read as a huge rate
            span = (self.recent[-1] - self.recent[0]) if len(self.recent) > 1 else 0.0
            recent_rate = (len(self.recent) - 1) / span if span >= 5.0 else rate
            eta = (total - done) / recent_rate if recent_rate > 0 else 0.0
            pct = 100 * done / total if total else 0.0
            gated = f" skip {self.skipped} ({100 * self.skipped / done:.0f}%)" if self.skipped else ""
            return (
                f"[{self.label}] {done}/{total} {pct:5.1f}% | "
                f"{rate:.2f}/s (60s {recent_rate:.2f}/s, {self.workers}w) | "
                f"ok {self.ok} fail {self.failed}{gated} | "
                f"flood {self.flood} dated {self.dated} verif {self.verifiable} | "
                f"{fmt_hms(elapsed)} eta {fmt_hms(eta)}"
            )


def start_ticker(stats: Stats, stop: threading.Event, every: float):
    """Repaints one status line every `every` seconds until `stop` is set."""

    def tick():
        width = 0
        while not stop.wait(every):
            line = stats.snapshot()
            width = max(width, len(line))
            print("\r" + line.ljust(width), end="", flush=True)
        print("\r" + stats.snapshot().ljust(width), flush=True)

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    return thread


def run_month(
    client: ollama.Client, cfg: dict, system_prompt: str, month: Path, limit: int | None
) -> None:
    out_root = Path(cfg["paths"]["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / f"{month.name}.jsonl"
    fail_file = out_root / f"{month.name}.failed.jsonl"

    done = already_done(out_file)
    files = [f for f in article_files(cfg, month) if f.stem not in done]
    if limit:
        files = files[:limit]

    total = len(files)
    print(f"[{month.name}] {total} to process ({len(done)} already done)", flush=True)
    if not total:
        return

    workers = cfg["run"]["workers"]
    field = cfg["output"]["extraction_field"]
    stats = Stats(total, month.name, workers)
    stop = threading.Event()
    ticker = start_ticker(stats, stop, cfg["run"]["progress_seconds"])

    # Workers only call the model; all writes happen here on the main thread.
    try:
        with out_file.open("a", encoding="utf-8") as out, fail_file.open("a", encoding="utf-8") as fail:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(process_one, client, cfg, system_prompt, path, month.name): path
                    for path in files
                }
                for future in as_completed(futures):
                    # process_one retries model errors internally, but e.g. a
                    # transiently locked article file raises straight through -
                    # log it as a failure instead of killing a multi-day run
                    try:
                        ok, record = future.result()
                    except Exception as exc:
                        ok = False
                        record = {
                            "article_id": futures[future].stem,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    stats.record(ok, record, field)

                    if ok:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out.flush()
                    else:
                        fail.write(json.dumps(record) + "\n")
                        fail.flush()
    finally:
        stop.set()
        ticker.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract flood events from English articles")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--months", nargs="*", help="e.g. 2021_01 2021_02 (default: all)")
    parser.add_argument("--limit", type=int, help="max articles per month (smoke tests)")
    parser.add_argument("--model", help="override model.name")
    parser.add_argument("--num-ctx", type=int, help="override generation.num_ctx")
    parser.add_argument("--workers", type=int, help="override run.workers")
    parser.add_argument(
        "--progress-seconds", type=float, help="override run.progress_seconds (0.5 = twice a second)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.num_ctx:
        cfg["generation"]["num_ctx"] = args.num_ctx
    if args.workers:
        cfg["run"]["workers"] = args.workers
    if args.progress_seconds:
        cfg["run"]["progress_seconds"] = args.progress_seconds

    system_prompt = load_system_prompt(cfg)
    # ollama.Client() defaults to timeout=None: a single wedged generation would
    # pin its worker for the rest of the run. Cap it, and let retries handle it.
    timeout = cfg["run"].get("request_timeout_seconds")
    client = ollama.Client(timeout=timeout) if timeout else ollama.Client()

    months = month_dirs(cfg, args.months)
    if not months:
        print("no matching month folders under", cfg["paths"]["input_root"], file=sys.stderr)
        return 1

    print(
        f"model={cfg['model']['name']} num_ctx={cfg['generation']['num_ctx']} "
        f"workers={cfg['run']['workers']} months={len(months)}",
        flush=True,
    )
    for month in months:
        run_month(client, cfg, system_prompt, month, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
