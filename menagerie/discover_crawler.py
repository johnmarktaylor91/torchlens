"""Starter crawler for discovering candidate model architectures absent from the menagerie.

This is intentionally minimal scaffolding for the discovery sweep described in
``menagerie/DISCOVER_MODELS.md``. It harvests recent arXiv listings across the key ML
categories, extracts candidate architecture names from titles, and flags those that do not
already appear in the catalog. It is a STARTING POINT — extend it with OpenReview, Papers
With Code, GitHub-trending, venue proceedings, non-English sources, and citation-graph walks.

Usage
-----
    python -m menagerie.discover_crawler --days 120 --out /tmp/candidates.tsv
    python -m menagerie.discover_crawler --categories cs.CV cs.CL --max-per-cat 600

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
DATA_DIR = Path(__file__).resolve().parent / "data"

# Categories where new neural-net architectures most often appear. NOT exhaustive — add more.
DEFAULT_CATEGORIES = (
    "cs.LG",
    "cs.CV",
    "cs.CL",
    "cs.NE",
    "cs.AI",
    "stat.ML",
    "eess.AS",
    "eess.IV",
    "eess.SP",
    "cs.SD",
    "cs.IR",
    "cs.RO",
    "cs.CR",
)

# Heuristic suffixes/markers that often signal a named architecture in a title.
NAME_MARKERS = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:Net|Former|GPT|BERT|Mixer|Mamba|NeRF|GAN|VAE|RNN|CNN|GNN|LM|"
    r"Net\+\+|ViT|Diffusion|Flow|Operator|Transformer))\b"
)
# "ArchName: a/the ... model" — many papers name the model before the colon.
COLON_NAME = re.compile(r"^\s*([A-Z][\w\-/]{1,30}(?:\s[A-Z][\w\-/]{1,30}){0,2})\s*:")
# Bare acronyms (2-9 uppercase/digit chars), a weak but useful signal.
ACRONYM = re.compile(r"\b([A-Z][A-Z0-9]{1,8})\b")


def _norm(text: str) -> str:
    """Lowercase and strip non-alphanumerics for loose membership comparison."""

    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_catalog_names(catalog_tsv: Path) -> set[str]:
    """Load normalized catalog names from the canonical catalog TSV (column 2 = name).

    Falls back to the raw master catalog (column 1 = name) when the canonical file is absent.
    """

    if catalog_tsv.exists():
        name_col = 1  # canonical_catalog: model_id, name, ...
    else:
        catalog_tsv = DATA_DIR / "master_catalog.tsv"
        name_col = 0  # master_catalog: name, zoo, ...
    names: set[str] = set()
    if not catalog_tsv.exists():
        print(
            f"warning: no catalog at {catalog_tsv}; every candidate will look new", file=sys.stderr
        )
        return names
    with catalog_tsv.open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) > name_col and row[name_col] not in ("name", "model_id", ""):
                names.add(_norm(row[name_col]))
    return names


def last_crawl_date() -> str | None:
    """Read ``last_exhaustive_crawl`` (YYYY-MM-DD) from the crawl history, if present."""

    import json

    path = DATA_DIR / "crawl_history.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("last_exhaustive_crawl")
    except Exception:  # noqa: BLE001 -- best-effort
        return None


def candidate_names(title: str) -> set[str]:
    """Extract candidate architecture names from a paper title (heuristic, noisy)."""

    found: set[str] = set()
    colon = COLON_NAME.match(title)
    if colon:
        found.add(colon.group(1).strip())
    found.update(NAME_MARKERS.findall(title))
    found.update(
        a for a in ACRONYM.findall(title) if a not in {"A", "AN", "THE", "WE", "II", "III"}
    )
    return {name for name in (n.strip() for n in found) if len(name) >= 2}


def fetch_arxiv(
    category: str, max_results: int, page_size: int = 100, pause: float = 3.0
) -> list[dict]:
    """Fetch the most recent arXiv entries for a category (newest first).

    Respects arXiv API etiquette with a pause between paged requests.
    """

    entries: list[dict] = []
    for start in range(0, max_results, page_size):
        params = {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": min(page_size, max_results - start),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                feed = ET.parse(response).getroot()
        except Exception as error:  # noqa: BLE001 -- network is best-effort
            print(f"arxiv fetch failed ({category}, start={start}): {error}", file=sys.stderr)
            break
        page = feed.findall(f"{ATOM}entry")
        if not page:
            break
        for entry in page:
            title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
            arxiv_id = (entry.findtext(f"{ATOM}id") or "").rsplit("/", 1)[-1]
            published = (entry.findtext(f"{ATOM}published") or "")[:10]
            entries.append(
                {"title": title, "id": arxiv_id, "published": published, "category": category}
            )
        time.sleep(pause)
    return entries


def run(args: argparse.Namespace) -> int:
    """Harvest candidates and write absent ones to a TSV."""

    catalog = load_catalog_names(DATA_DIR / "catalog_canonical.tsv")
    print(f"catalog names loaded: {len(catalog)}", file=sys.stderr)
    # Date window: explicit --since wins, else the last exhaustive crawl date, else --days back.
    cutoff = args.since or last_crawl_date()
    if cutoff is None and args.days:
        import datetime

        cutoff = (datetime.date.today() - datetime.timedelta(days=args.days)).isoformat()
    if cutoff:
        print(f"date cutoff: keeping arXiv papers on/after {cutoff}", file=sys.stderr)

    seen_ids: set[str] = set()
    rows: list[tuple] = []
    for category in args.categories:
        print(f"fetching {category} ...", file=sys.stderr)
        for entry in fetch_arxiv(category, args.max_per_cat):
            if cutoff and entry["published"] < cutoff:
                break  # feed is newest-first
            if entry["id"] in seen_ids:
                continue
            seen_ids.add(entry["id"])
            for name in candidate_names(entry["title"]):
                in_catalog = _norm(name) in catalog or any(
                    _norm(name) in c for c in catalog if len(name) > 4
                )
                if not in_catalog:
                    rows.append(
                        (name, entry["id"], entry["published"], entry["category"], entry["title"])
                    )

    # collapse duplicate candidate names, keep the earliest-seen evidence
    best: dict[str, tuple] = {}
    for row in rows:
        key = _norm(row[0])
        if key not in best:
            best[key] = row
    out_rows = sorted(best.values(), key=lambda r: (r[3], r[0].lower()))

    out_path = Path(args.out)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("candidate_name", "arxiv_id", "published", "category", "title"))
        writer.writerows(out_rows)
    print(
        f"\nwrote {len(out_rows)} candidate names absent from catalog -> {out_path}",
        file=sys.stderr,
    )
    print(
        "NOTE: candidates are noisy heuristics. Investigate each, dedup against the catalog by",
        "alias, and apply the family-not-variant bar before adding. See DISCOVER_MODELS.md.",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the crawler CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument(
        "--since",
        help="only keep papers on/after this YYYY-MM-DD "
        "(default: last_exhaustive_crawl from data/crawl_history.json)",
    )
    parser.add_argument(
        "--days", type=int, default=180, help="fallback window in days when no --since / crawl date"
    )
    parser.add_argument(
        "--max-per-cat", type=int, default=400, help="max arXiv entries to scan per category"
    )
    parser.add_argument("--out", default="/tmp/menagerie_candidates.tsv")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the crawler CLI."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
