#!/usr/bin/env python
"""photomind-mcp terminal demo.

Runs real queries against the indexed library and prints formatted output.
Designed to be recorded with asciinema and converted to a GIF for the README.

Usage
-----
# Run once to preview:
  uv run python scripts/demo.py

# Record + convert to GIF:
  asciinema rec demo.cast --cols 72 --rows 28 -c "uv run python scripts/demo.py"
  agg --cols 72 --rows 28 demo.cast docs/demo.gif

Install recording tools (macOS):
  brew install asciinema
  brew install agg
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

# Resolve project root so the script works from any cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from photomind.database import DatabaseManager  # noqa: E402
from photomind.embeddings import CLIPEmbedder  # noqa: E402

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE   = "\033[34m"
WHITE  = "\033[37m"


def _print(text: str = "") -> None:
    print(text, flush=True)


def _typewrite(text: str, delay: float = 0.045) -> None:
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()


def _pause(s: float) -> None:
    time.sleep(s)


def _header() -> None:
    line = "─" * 64
    _print(f"\n{BOLD}{CYAN}{line}{RESET}")
    _print(f"{BOLD}{CYAN}  photomind-mcp  ·  Intelligent Photo Library{RESET}")
    _print(f"{BOLD}{CYAN}  Local · Private · CLIP-powered semantic search{RESET}")
    _print(f"{BOLD}{CYAN}{line}{RESET}\n")
    _pause(0.6)


def _prompt(query: str) -> None:
    _print()
    print(f"{BOLD}{YELLOW}❯ {RESET}", end="", flush=True)
    _typewrite(query, delay=0.05)
    _pause(0.25)


def _ok(msg: str) -> None:
    _print(f"\n{BOLD}{GREEN}✓{RESET}  {msg}")


def _row(icon: str, key: str, value: str) -> None:
    _print(f"  {icon}  {BOLD}{key:<18}{RESET}{value}")


# ---------------------------------------------------------------------------
# Demo steps
# ---------------------------------------------------------------------------

def step_stats(db: DatabaseManager) -> None:
    _prompt("How many photos are indexed?")
    count = db.photo_count()
    emb_count = db.conn.execute(
        "SELECT COUNT(*) FROM photo_embeddings"
    ).fetchone()[0]
    _ok("Library status")
    _row("📷", "Photos indexed", str(count))
    _row("🧠", "Embeddings ready", str(emb_count))
    _pause(0.7)


def step_semantic_search(db: DatabaseManager, embedder: CLIPEmbedder) -> None:
    _prompt("Search for photos of outdoor scenery")
    _print(f"  {DIM}encoding query with CLIP ViT-B-32…{RESET}")
    query_emb = embedder.encode_text("outdoor scenery")
    results = db.search_by_embedding(query_emb, limit=3)
    _ok(f"Top {len(results)} semantic matches")
    _print()
    for i, r in enumerate(results, 1):
        date   = (r.get("date_taken") or "")[:10]
        model  = r.get("camera_model") or "unknown"
        score  = r.get("similarity_score", 0.0)
        lat    = r.get("latitude")
        loc    = f"{lat:.4f}°N" if lat else "no GPS"
        _print(f"  {DIM}{i}.{RESET}  {BOLD}{r['filename'][:32]}{RESET}")
        _print(f"       {DIM}{date}  ·  {model}  ·  {loc}  ·  score {score:.4f}{RESET}")
    _pause(0.8)


def step_duplicates(db: DatabaseManager) -> None:
    _prompt("Find duplicate photos in the library")
    groups = db.find_duplicate_groups(threshold=0.95)
    if groups:
        _ok(f"Found {len(groups)} duplicate group(s)")
        _print()
        for i, group in enumerate(groups, 1):
            date = (group[0].get("date_taken") or "")[:10]
            _print(f"  {YELLOW}Group {i}{RESET}  —  {len(group)} photos  ·  {date}")
            for photo in group:
                sim = photo.get("similarity_score", 0.0)
                _print(f"    {DIM}·{RESET}  {photo['filename'][:40]}  "
                       f"{DIM}(sim {sim:.4f}){RESET}")
    else:
        _ok("No duplicates found at threshold 0.95")
    _pause(0.8)


def step_quality(db: DatabaseManager) -> None:
    from photomind.quality import compute_sharpness

    _prompt("Flag blurry or poor-quality photos")
    photos = db.photos_with_paths()

    # Try live scoring first (requires Full Disk Access).
    scores: list[tuple[float, str]] = []
    for p in photos:
        s = compute_sharpness(p.get("filepath"))
        if s is not None:
            scores.append((s, p["filename"]))

    # Fall back to quality_score values already stored in the DB
    # (written by a previous flag_poor_quality tool call with FDA).
    if not scores:
        for p in photos:
            s = p.get("quality_score")
            if s is not None:
                scores.append((float(s), p["filename"]))

    if not scores:
        _ok("Quality scores unavailable (Full Disk Access required to read images)")
        _pause(0.8)
        return

    scores.sort()
    bottom = scores[:3]
    _ok(f"Scored {len(scores)} photos  ·  sharpest: {scores[-1][0]:,.0f}  ·  "
        f"blurriest: {scores[0][0]:,.0f}")
    _print()
    _print(f"  {DIM}Bottom 3 by sharpness:{RESET}")
    for score, name in bottom:
        bar = "░" * int(min(score / 100, 20))
        _print(f"    {MAGENTA}{score:>8,.0f}{RESET}  {bar}  {DIM}{name[:36]}{RESET}")
    _pause(0.8)


def step_organise(db: DatabaseManager) -> None:
    _prompt("Organise library by year / month")
    rows = db.conn.execute(
        "SELECT date_taken FROM photos ORDER BY date_taken"
    ).fetchall()
    structure: dict[str, int] = defaultdict(int)
    for (dt,) in rows:
        if dt and len(dt) >= 7:
            structure[f"{dt[:4]}/{dt[5:7]}"] += 1
    _ok(f"Suggested structure  ·  {len(structure)} folders  ·  "
        f"{sum(structure.values())} photos")
    _print()
    max_count = max(structure.values(), default=1)
    for folder, cnt in sorted(structure.items()):
        bar_len = max(1, round(cnt / max_count * 18))
        bar = f"{CYAN}{'█' * bar_len}{DIM}{'░' * (18 - bar_len)}{RESET}"
        _print(f"  {BOLD}{MAGENTA}{folder}{RESET}  {bar}  {DIM}{cnt:>2} photo{'s' if cnt != 1 else ''}{RESET}")
    _pause(0.6)


def _footer() -> None:
    _print()
    _print(f"{DIM}{'─' * 64}{RESET}")
    _print(f"{BOLD}photomind-mcp{RESET}  —  fully local · no cloud · no API keys")
    _print(f"{DIM}github.com/davidcjw/photomind-mcp{RESET}")
    _print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _header()

    db = DatabaseManager()
    db.connect()

    # Warm up CLIP silently before the recorded session starts
    _print(f"{DIM}loading CLIP model…{RESET}", )
    embedder = CLIPEmbedder()
    embedder.encode_text("warmup")   # trigger lazy load
    # Move cursor up 1 line and clear it so the warmup line disappears
    print("\033[1A\033[2K", end="", flush=True)

    step_stats(db)
    step_semantic_search(db, embedder)
    step_duplicates(db)
    step_quality(db)
    step_organise(db)

    _footer()
    db.close()


if __name__ == "__main__":
    main()
