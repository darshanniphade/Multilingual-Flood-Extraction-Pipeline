"""Count the source language of every article in the raw crawl.

Reads only the head of each file and pulls `language` with a byte regex - the
field sits before `text`, so a few KB is enough and the article body is never
decoded. 2021 is not rescanned: H:\\2021clouddata\\language_counts.csv already
holds the full 1,383,600-article tally.

    python lang_scan.py 2022 2023
"""
import json
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool

LANG = re.compile(rb'"language"\s*:\s*"([^"]*)"')
HEAD = 8192

ROOTS = {
    "2022": r"H:\2022",
    "2023": r"C:\darsh\pipeline\data\ssd\2023",
    "2024": r"C:\darsh\pipeline\data\ssd\2024\downloaded_articles",
}
OUT_DIR = r"C:\darsh\pipeline\metrics"


def month_dirs(root):
    """Every <month>/articles folder under the year root, in name order."""
    out = []
    for m in sorted(os.listdir(root)):
        p = os.path.join(root, m)
        if not os.path.isdir(p):
            continue
        a = os.path.join(p, "articles")
        out.append((m, a if os.path.isdir(a) else p))
    return out


def scan_dir(job):
    month, path = job
    c = Counter()
    n = 0
    with os.scandir(path) as it:
        for e in it:
            if not e.name.endswith(".json"):
                continue
            n += 1
            try:
                with open(e.path, "rb") as fh:
                    head = fh.read(HEAD)
            except OSError:
                c["<unreadable>"] += 1
                continue
            m = LANG.search(head)
            if not m:
                c["<absent>"] += 1
                continue
            v = m.group(1).decode("utf-8", "replace").strip()
            c[v if v else "<empty>"] += 1
    return month, n, c


def main(years):
    for year in years:
        root = ROOTS[year]
        jobs = month_dirs(root)
        print(f"\n=== {year} ===  {len(jobs)} month folders under {root}", flush=True)
        t0 = time.time()
        total = Counter()
        per_month = {}
        files = 0
        with Pool(processes=min(24, (os.cpu_count() or 8))) as pool:
            for month, n, c in pool.imap_unordered(scan_dir, jobs):
                total.update(c)
                per_month[month] = dict(c.most_common())
                files += n
                print(f"  {month}: {n:,} files, {len(c)} languages "
                      f"({time.time()-t0:.0f}s)", flush=True)
        dt = time.time() - t0
        print(f"{year}: {files:,} files, {len(total)} distinct languages, "
              f"{dt:.0f}s ({files/max(dt,1):,.0f} files/s)")

        out = os.path.join(OUT_DIR, f"languages_{year}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"year": year, "root": root, "files": files,
                       "distinct": len(total),
                       "totals": dict(total.most_common()),
                       "by_month": per_month}, fh, indent=1, ensure_ascii=False)

        csv_path = os.path.join(OUT_DIR, f"languages_{year}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            fh.write("language,count,percentage\n")
            for k, v in total.most_common():
                fh.write(f"{k},{v},{100*v/max(1,files):.4f}\n")
        print("  wrote", out, "and", csv_path)


if __name__ == "__main__":
    main(sys.argv[1:] or ["2023"])
