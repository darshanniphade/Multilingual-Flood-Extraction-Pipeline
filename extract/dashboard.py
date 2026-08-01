"""Live colour dashboard for the flood-extraction pipeline.

Same engine, same resumable outputs as extract.py (which this imports) - the
model calls, retries, and JSONL writes are identical. This only replaces the
one-line ticker with a full-screen terminal dashboard: overall + current month
progress, live throughput with a sparkline, flood/date/verifiable tallies,
per-month status, and a feed of the latest articles.

    python dashboard.py --config config_2023.json
    python dashboard.py --config config_2023.json --months 2023_07 --limit 50

When the last month finishes it shells out to to_csv.py with the same config,
so a completed run ends with <output_root>/extractions.csv. Ctrl+C is safe:
finished articles are already on disk and the next run resumes after them.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path

import ollama
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

import extract

HERE = Path(__file__).resolve().parent
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def run_label(names) -> str:
    """The year(s) a run covers, read off the part names ('2023_01' -> '2023').
    This used to be a hardcoded '2022', so every later run wore the wrong title."""
    years = sorted({name.split("_")[0] for name in names})
    if not years:
        return "run"
    return years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"


class Board:
    """All run state plus the renderer. record() runs on the writer loop,
    render() on Live's refresh thread - shared state sits behind one lock.
    Colour roles are deliberate and few: green/red are reserved for ok/fail
    status, cyan carries progress/throughput, bright_blue the flood signal,
    magenta the verifiable subset; labels stay dim."""

    RECENT_WINDOW = 60.0  # seconds of history behind the "now" rate
    SPARK_EVERY = 5.0     # seconds between sparkline samples

    def __init__(self, plan, cfg):
        self.lock = threading.Lock()
        self.started = time.time()
        self.cfg = cfg
        self.month_rows = {}
        grand_total = pre_done = 0
        for month, _todo, done_before, total in plan:
            self.month_rows[month.name] = {
                "total": total,
                "completed": done_before,
                "floods": 0,
                "fails": 0,
                "state": "queued",
            }
            grand_total += total
            pre_done += done_before
        self.current = None
        self.finished = False
        self.ok = self.fail = self.flood = self.dated = self.verifiable = 0
        self.recent = deque()
        self.spark = deque(maxlen=56)
        self.last_spark = self.started
        self.feed = deque(maxlen=8)

        self.progress = Progress(
            TextColumn("[bold]{task.description:>7}[/]"),
            BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        self.label = run_label(self.month_rows)
        self.overall_task = self.progress.add_task(
            self.label, total=grand_total, completed=pre_done
        )
        self.month_task = None

    # -- state updates ----------------------------------------------------

    def _focus(self, name):
        """Point the second progress bar at `name` (lock held). Only ever moves
        forward: work from two months overlaps at a boundary, and following every
        straggler backwards would make the bar flap."""
        if self.current is not None and name <= self.current:
            return
        self.current = name
        row = self.month_rows[name]
        if self.month_task is not None:
            self.progress.remove_task(self.month_task)
        self.month_task = self.progress.add_task(
            name, total=row["total"], completed=row["completed"]
        )

    def record(self, ok, record, field, month_name):
        now = time.time()
        with self.lock:
            self._focus(month_name)
            row = self.month_rows[month_name]
            row["completed"] += 1
            row["state"] = "done" if row["completed"] >= row["total"] else "running"
            self.recent.append(now)
            while self.recent and now - self.recent[0] > self.RECENT_WINDOW:
                self.recent.popleft()
            self.progress.advance(self.overall_task)
            # a straggler from the previous month must not nudge the focused bar
            if self.month_task is not None and month_name == self.current:
                self.progress.advance(self.month_task)
            if ok:
                self.ok += 1
                result = record.get(field) or {}
                is_flood = bool(result.get("contains_flood_event"))
                if is_flood:
                    self.flood += 1
                    row["floods"] += 1
                if result.get("flood_dates"):
                    self.dated += 1
                if result.get("is_verifiable_flood"):
                    self.verifiable += 1
                title_key = self.cfg["output"].get("title_key", "translated_title")
                title = (record.get(title_key) or "").replace("\n", " ").strip()
                self.feed.appendleft((True, is_flood, record.get("article_id", "?"), title))
            else:
                self.fail += 1
                row["fails"] += 1
                self.feed.appendleft(
                    (False, False, record.get("article_id", "?"), record.get("error") or "")
                )

    # -- rendering --------------------------------------------------------

    def _rates(self):
        elapsed = max(time.time() - self.started, 1e-9)
        avg = (self.ok + self.fail) / elapsed
        # ignore the recent window until it spans enough time to be meaningful
        span = (self.recent[-1] - self.recent[0]) if len(self.recent) > 1 else 0.0
        now_rate = (len(self.recent) - 1) / span if span >= 5.0 else avg
        return avg, now_rate, elapsed

    def render(self):
        with self.lock:
            avg, now_rate, elapsed = self._rates()
            if time.time() - self.last_spark >= self.SPARK_EVERY:
                self.spark.append(now_rate)
                self.last_spark = time.time()
            remaining = sum(r["total"] - r["completed"] for r in self.month_rows.values())
            eta = remaining / now_rate if now_rate > 0 else 0.0
            return Group(
                self._header(elapsed),
                Panel(self.progress, box=box.ROUNDED, border_style="grey50", padding=(0, 1)),
                self._tiles(avg, now_rate, eta),
                self._body(),
            )

    def _header(self, elapsed):
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="center")
        grid.add_column(justify="right")
        left = Text()
        left.append(" FLOOD EXTRACTION ", style="bold black on cyan")
        left.append(f" {self.label}", style="bold cyan")
        mid = Text(
            f"{self.cfg['model']['name']} · {self.cfg['run']['workers']} workers"
            f" · ctx {self.cfg['generation']['num_ctx']}",
            style="dim",
        )
        right = Text()
        if self.finished:
            right.append("● DONE ", style="bold green")
        else:
            right.append("● RUNNING ", style="bold green")
        right.append(extract.fmt_hms(elapsed), style="bold")
        grid.add_row(left, mid, right)
        return Panel(grid, box=box.HEAVY, border_style="cyan", padding=(0, 1))

    def _tiles(self, avg, now_rate, eta):
        def tile(label, value, style):
            body = Table.grid(expand=True)
            body.add_column(justify="center")
            body.add_row(Text(value, style=f"bold {style}"))
            body.add_row(Text(label, style="dim"))
            return Panel(body, box=box.ROUNDED, border_style="grey50")

        hit = f" {100 * self.flood / self.ok:.0f}%" if self.ok else ""
        cells = [
            tile("art/s now", f"{now_rate:.2f}", "cyan"),
            tile("art/s avg", f"{avg:.2f}", "cyan"),
            tile("eta", extract.fmt_hms(eta), "white"),
            tile("ok", f"{self.ok:,}", "green"),
            tile("failed", f"{self.fail:,}", "red" if self.fail else "grey50"),
            tile(f"floods{hit}", f"{self.flood:,}", "bright_blue"),
            tile("dated", f"{self.dated:,}", "blue"),
            tile("verifiable", f"{self.verifiable:,}", "magenta"),
        ]
        grid = Table.grid(expand=True)
        for _ in cells:
            grid.add_column(ratio=1)
        grid.add_row(*cells)
        return grid

    def _body(self):
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=5)
        grid.add_column(ratio=7)
        grid.add_row(self._months_table(), Group(self._sparkline(), self._feed()))
        return grid

    def _months_table(self):
        t = Table(box=box.SIMPLE_HEAD, expand=False, padding=(0, 1), show_edge=False)
        t.add_column("month", style="bold", no_wrap=True, min_width=7)
        t.add_column("progress", no_wrap=True, width=12)
        t.add_column("done", justify="right", no_wrap=True, min_width=11)
        t.add_column("floods", justify="right", style="bright_blue", no_wrap=True, min_width=6)
        t.add_column("", no_wrap=True, width=1)
        for name, row in self.month_rows.items():
            bar = ProgressBar(
                total=max(row["total"], 1),
                completed=row["completed"],
                width=12,
                complete_style="cyan",
                finished_style="green",
            )
            if row["state"] == "running":
                status = Text("►", style="bold yellow")
            elif row["state"] == "done":
                status = Text("●", style="green")
            else:
                status = Text("·", style="dim")
            t.add_row(
                name,
                bar,
                f"{row['completed']}/{row['total']}",
                f"{row['floods']:,}" if row["floods"] else "·",
                status,
            )
        return Panel(t, title="months", title_align="left", border_style="grey50", box=box.ROUNDED)

    def _sparkline(self):
        vals = list(self.spark)
        if len(vals) < 2:
            body = Text("collecting samples...", style="dim")
        else:
            hi = max(vals) or 1.0
            body = Text()
            for v in vals:
                idx = min(len(SPARK_BLOCKS) - 1, int(v / hi * len(SPARK_BLOCKS)))
                body.append(SPARK_BLOCKS[idx], style="cyan")
            body.append(f"  peak {hi:.2f}/s", style="dim")
        return Panel(
            body,
            title=f"throughput · {self.SPARK_EVERY:.0f}s samples",
            title_align="left",
            border_style="grey50",
            box=box.ROUNDED,
        )

    def _feed(self):
        lines = []
        for ok, is_flood, art_id, tail in self.feed:
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append("● " if ok else "× ", style="green" if ok else "bold red")
            line.append(f"{art_id}  ", style="dim")
            if is_flood:
                line.append(" FLOOD ", style="bold black on bright_blue")
                line.append(" ")
            line.append(tail, style="default" if ok else "red")
            lines.append(line)
        if not lines:
            lines.append(Text("waiting for first completions...", style="dim"))
        return Panel(
            Group(*lines),
            title="latest articles",
            title_align="left",
            border_style="grey50",
            box=box.ROUNDED,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Flood extraction with a live dashboard")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--months", nargs="*", help="e.g. 2023_01 2023_02 (default: all)")
    parser.add_argument("--limit", type=int, help="max articles per month (smoke tests)")
    parser.add_argument("--workers", type=int, help="override run.workers")
    parser.add_argument("--no-csv", action="store_true", help="skip the to_csv.py step at the end")
    args = parser.parse_args()

    cfg = extract.load_config(args.config)
    if args.workers:
        cfg["run"]["workers"] = args.workers
    system_prompt = extract.load_system_prompt(cfg)
    # ollama's client defaults to timeout=None - a request that never comes back
    # pins a worker forever, and at the tail of a month that stalls everything
    # behind it. Cap it: a stuck article now fails, retries, and moves on.
    client = ollama.Client(timeout=cfg["run"].get("request_timeout_seconds", 180))
    field = cfg["output"]["extraction_field"]

    months = extract.month_dirs(cfg, args.months)
    if not months:
        print("no matching month folders under", cfg["paths"]["input_root"], file=sys.stderr)
        return 1

    out_root = Path(cfg["paths"]["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    plan = []
    for month in months:
        files = extract.article_files(cfg, month)
        done = extract.already_done(out_root / f"{month.name}.jsonl")
        todo = [f for f in files if f.stem not in done]
        if args.limit:
            todo = todo[: args.limit]
        plan.append((month, todo, len(done), len(files)))

    board = Board(plan, cfg)
    console = Console()
    workers = cfg["run"]["workers"]
    interrupted = False

    # One continuous queue across every month rather than a pool per month. A
    # month boundary used to be a barrier: the last few articles ran while the
    # other workers sat idle, and one slow article held up the whole next month.
    # Here a worker that finishes the tail of January picks up February at once.
    work = [(path, month.name) for month, todo, _d, _t in plan for path in todo]
    for month, todo, done_before, total in plan:
        if not todo:
            board.month_rows[month.name]["state"] = "done" if done_before >= total else "partial"

    # Writes stay on this thread, so each month needs its handles open for as long
    # as its work is in flight - which now overlaps the next month.
    def open_handles(stack):
        handles = {}
        for month, todo, _d, _t in plan:
            if todo:
                handles[month.name] = (
                    stack.enter_context((out_root / f"{month.name}.jsonl").open("a", encoding="utf-8")),
                    stack.enter_context(
                        (out_root / f"{month.name}.failed.jsonl").open("a", encoding="utf-8")
                    ),
                )
        return handles

    try:
        with Live(get_renderable=board.render, console=console, screen=True, refresh_per_second=6), \
                ExitStack() as stack:
            handles = open_handles(stack)
            pool = ThreadPoolExecutor(max_workers=workers)
            queue = iter(work)
            pending = {}

            def submit_next() -> bool:
                item = next(queue, None)
                if item is None:
                    return False
                path, month_name = item
                future = pool.submit(
                    extract.process_one, client, cfg, system_prompt, path, month_name
                )
                pending[future] = (path, month_name)
                return True

            try:
                # keep a modest backlog queued rather than submitting all 150k at
                # once, so memory stays flat and Ctrl+C has little to cancel
                for _ in range(workers * 4):
                    if not submit_next():
                        break
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        path, month_name = pending.pop(future)
                        # process_one retries model errors internally, but e.g. a
                        # transiently locked article file raises straight through -
                        # log it as a failure instead of killing a multi-day run
                        try:
                            ok, record = future.result()
                        except Exception as exc:
                            ok = False
                            record = {
                                "article_id": path.stem,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        board.record(ok, record, field, month_name)
                        out, fail = handles[month_name]
                        if ok:
                            out.write(json.dumps(record, ensure_ascii=False) + "\n")
                            out.flush()
                        else:
                            fail.write(json.dumps(record) + "\n")
                            fail.flush()
                        submit_next()
            finally:
                # normal exit: everything is done, nothing to cancel.
                # Ctrl+C: drop queued work so only in-flight requests drain.
                pool.shutdown(wait=False, cancel_futures=True)
            board.finished = True
            time.sleep(1.5)  # let the final DONE frame paint before leaving alt-screen
    except KeyboardInterrupt:
        interrupted = True

    done_this_run = board.ok + board.fail
    elapsed = extract.fmt_hms(time.time() - board.started)
    console.print(
        f"[bold cyan]run summary[/] · {done_this_run:,} processed in {elapsed} · "
        f"[green]ok {board.ok:,}[/] [red]fail {board.fail:,}[/] · "
        f"[bright_blue]floods {board.flood:,}[/] dated {board.dated:,} "
        f"[magenta]verifiable {board.verifiable:,}[/]"
    )
    if interrupted:
        console.print(
            "[yellow]interrupted[/] - finished articles are on disk; "
            "rerun the same command to resume (in-flight requests may take a moment to drain)"
        )
        return 130

    if not args.no_csv:
        console.print("\n[bold cyan]flattening to CSV...[/]")
        subprocess.run(
            [sys.executable, str(HERE / "to_csv.py"), "--config", str(args.config)], check=False
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
