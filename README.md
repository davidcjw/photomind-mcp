# photomind-mcp

A local-first MCP server that gives AI agents intelligent access to your macOS Photos library — semantic search via CLIP, plus metadata search by date, location, camera, and people. Fully on-device, no cloud dependency.

[![AgentReady Score](https://agentready-gules.vercel.app/api/badge/davidcjw/photomind-mcp)](https://agentready-gules.vercel.app/results/davidcjw/photomind-mcp)

## Prerequisites

- macOS with Photos.app (at least one photo library opened)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Full Disk Access granted to your terminal in **System Settings → Privacy & Security → Full Disk Access**

## Demo

![photomind-mcp demo](docs/demo.gif)

## Quick Start

```bash
git clone https://github.com/davidcjw/photomind-mcp
cd photomind-mcp
uv sync
```

**Index your library** (run from a terminal with Full Disk Access):

```bash
.venv/bin/python -c "
from photomind.database import DatabaseManager
from photomind.embeddings import CLIPEmbedder
from photomind.indexer import PhotoIndexer

db = DatabaseManager()
db.connect()
result = PhotoIndexer(db, embedder=CLIPEmbedder()).sync()
db.close()
print(result)
"
```

The first run downloads the CLIP model (~350 MB). Subsequent syncs are incremental — only new photos are embedded.

## Claude Desktop Integration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "photomind": {
      "command": "/path/to/photomind-mcp/.venv/bin/photomind-mcp"
    }
  }
}
```

Replace `/path/to/photomind-mcp` with the absolute path to this repo, then restart Claude Desktop.

## Tools

| Tool | Description |
|---|---|
| `sync_library()` | Index Photos.app library into SQLite + generate CLIP embeddings. Safe to re-run. |
| `search_photos(query)` | **Semantic search** — natural language → CLIP → nearest neighbours. |
| `get_photo_metadata(photo_id)` | Full metadata for a photo by UUID. |
| `search_by_date(start_date, end_date)` | Photos taken in a date range (ISO-8601). |
| `search_by_location(latitude, longitude, radius_km)` | Photos near GPS coordinates, sorted by distance. |
| `search_by_metadata(keywords, camera_model, persons)` | Filter by keywords, camera model, or tagged people. |
| `find_duplicates(threshold)` | Find near-duplicate photo groups via CLIP cosine similarity (union-find clustering). |
| `flag_poor_quality(blur_threshold)` | Score every photo's sharpness; return and optionally persist the blurriest. |
| `organise_photos(group_by)` | Suggest a folder hierarchy (by year/month or year) — read-only, no files moved. |
| `get_delete_candidates(duplicate_threshold, blur_threshold)` | Identify duplicates and blurry photos to clean up, with date hints for finding them in Photos.app. |
| `sync_from_directory(directory)` | Index photos from a plain folder (no Photos.app needed) — JPEG, PNG, HEIC, TIFF, WebP. |
| `delete_photos(photo_ids, dry_run, permanent)` | Delete filesystem photos (from `sync_from_directory`) — moves to Trash by default. Refuses Photos.app library paths. |

> **Note on deletion:** `delete_photos` only works for photos indexed via `sync_from_directory`. macOS blocks programmatic deletion from the Photos.app library; use `get_delete_candidates` + manual deletion in Photos.app for those.

## Architecture

```
photomind/
├── config.py       # DB path + constants
├── database.py     # SQLite layer — metadata queries, embedding storage, numpy KNN
├── embeddings.py   # CLIP ViT-B-32 pipeline (lazy-loaded, MPS-accelerated, HEIC support)
├── indexer.py      # osxphotos ingestion + EXIF extraction + embedding pass
└── server.py       # FastMCP server + tool definitions
```

**Storage**: `~/Library/Application Support/photomind-mcp/photomind.db`  
**Embeddings**: packed float32 BLOBs in SQLite, cosine similarity via numpy  
**Privacy**: fully local — no API calls, no cloud, model weights cached on-device

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| MCP framework | FastMCP | Fastest Python MCP server builder |
| Photo access | osxphotos | Programmatic access to Photos.app library |
| Vision model | CLIP ViT-B-32 via OpenCLIP | Local semantic search, no API cost |
| HEIC support | pillow-heif | Decode iPhone HEIC photos for CLIP |
| Vector search | numpy cosine similarity | Zero dependencies, fast for <10k photos |
| Storage | SQLite | Local, no external services |

## Running Tests

```bash
uv run pytest tests/ -v
```

## Recording the demo

```bash
# Install tools (macOS)
brew install asciinema agg

# Record (72×28 fits neatly in a GitHub README)
asciinema rec demo.cast --cols 72 --rows 28 -c "uv run python scripts/demo.py"

# Convert to GIF
agg --cols 72 --rows 28 demo.cast docs/demo.gif
```

Then replace the placeholder in the Demo section above with:
```markdown
![photomind-mcp demo](docs/demo.gif)
```
