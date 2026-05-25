"""FastMCP server — photomind MCP tools."""
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastmcp import Context, FastMCP

from photomind.config import DEFAULT_RADIUS_KM, DEFAULT_SEARCH_LIMIT
from photomind.database import DatabaseManager
from photomind.embeddings import CLIPEmbedder
from photomind.indexer import PhotoIndexer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncGenerator[dict[str, Any], None]:
    db = DatabaseManager()
    db.connect()
    embedder = CLIPEmbedder()  # lazy — model loads on first encode call
    logger.info("photomind-mcp started. DB: %s (vec=%s)", db._db_path, db.vec_available)
    try:
        yield {"db": db, "embedder": embedder}
    finally:
        db.close()
        logger.info("photomind-mcp stopped.")


mcp = FastMCP(
    "photomind",
    instructions=(
        "Intelligent photo library management. "
        "Start with sync_library() to index your Photos.app library, "
        "then use search tools to explore photos by date, location, or metadata."
    ),
    lifespan=app_lifespan,
)


def _db(ctx: Context) -> DatabaseManager:
    return ctx.lifespan_context["db"]


def _embedder(ctx: Context) -> CLIPEmbedder:
    return ctx.lifespan_context["embedder"]


@mcp.tool()
def sync_library(ctx: Context) -> dict[str, Any]:
    """Index all photos from Photos.app into the local SQLite database.

    Must be called at least once before search tools return results.
    Safe to re-run; existing records are updated in place.

    Returns a summary with total, indexed, skipped, errors, duration_seconds.
    """
    indexer = PhotoIndexer(_db(ctx), embedder=_embedder(ctx))
    try:
        summary = indexer.sync()
    except (ImportError, RuntimeError) as exc:
        return {"success": False, "error": str(exc)}
    return {"success": True, **summary}


@mcp.tool()
def get_photo_metadata(photo_id: str, ctx: Context) -> dict[str, Any]:
    """Return the full metadata record for a single photo by its UUID.

    Args:
        photo_id: The Photos.app UUID (e.g. from search results).
    """
    record = _db(ctx).get_photo(photo_id)
    if record is None:
        return {
            "found": False,
            "error": f"No photo with id '{photo_id}'. Run sync_library() first.",
        }
    return {"found": True, "photo": record}


@mcp.tool()
def search_by_date(
    start_date: str,
    end_date: str,
    ctx: Context,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict[str, Any]:
    """Find photos taken within a date range.

    Args:
        start_date: ISO-8601 date string, e.g. '2023-01-01'.
        end_date:   ISO-8601 date string, e.g. '2023-12-31'.
        limit:      Maximum results (default 50, max 500).
    """
    limit = min(limit, 500)
    try:
        photos = _db(ctx).search_by_date(start_date, end_date, limit)
    except Exception as exc:
        return {"error": str(exc), "photos": [], "count": 0}
    return {"photos": photos, "count": len(photos)}


@mcp.tool()
def search_by_location(
    latitude: float,
    longitude: float,
    ctx: Context,
    radius_km: float = DEFAULT_RADIUS_KM,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict[str, Any]:
    """Find photos taken near a GPS coordinate.

    Args:
        latitude:  Decimal degrees, e.g. 1.3521 (Singapore).
        longitude: Decimal degrees, e.g. 103.8198.
        radius_km: Search radius in kilometres (default 5.0).
        limit:     Maximum results (default 50).
    """
    if not (-90 <= latitude <= 90):
        return {"error": "latitude must be between -90 and 90.", "photos": [], "count": 0}
    if not (-180 <= longitude <= 180):
        return {"error": "longitude must be between -180 and 180.", "photos": [], "count": 0}
    if radius_km <= 0:
        return {"error": "radius_km must be positive.", "photos": [], "count": 0}
    limit = min(limit, 500)
    try:
        photos = _db(ctx).search_by_location(latitude, longitude, radius_km, limit)
    except Exception as exc:
        return {"error": str(exc), "photos": [], "count": 0}
    return {
        "center": {"latitude": latitude, "longitude": longitude},
        "radius_km": radius_km,
        "photos": photos,
        "count": len(photos),
    }


@mcp.tool()
def search_by_metadata(
    ctx: Context,
    keywords: list[str] | None = None,
    camera_model: str | None = None,
    persons: list[str] | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict[str, Any]:
    """Find photos by metadata: keywords, camera model, or tagged people.

    At least one filter must be provided. All filters combine with AND logic.

    Args:
        keywords:     Keywords to filter by (AND-combined).
        camera_model: Partial camera model name, e.g. 'iPhone 15' (case-insensitive).
        persons:      Person names from Photos.app face tags (AND-combined).
        limit:        Maximum results (default 50).
    """
    if not keywords and not camera_model and not persons:
        return {
            "error": "Provide at least one of: keywords, camera_model, persons.",
            "photos": [],
            "count": 0,
        }
    limit = min(limit, 500)
    try:
        photos = _db(ctx).search_by_metadata(keywords, camera_model, persons, limit)
    except ValueError as exc:
        return {"error": str(exc), "photos": [], "count": 0}
    except Exception as exc:
        return {"error": f"Search failed: {exc}", "photos": [], "count": 0}
    return {"photos": photos, "count": len(photos)}


@mcp.tool()
def search_photos(
    query: str,
    ctx: Context,
    limit: int = 10,
) -> dict[str, Any]:
    """Search photos using natural language via CLIP semantic similarity.

    Converts the query to a vision-language embedding and finds the most
    visually/semantically similar photos in the library.

    Requires sync_library() to have been run with embedding generation enabled,
    and sqlite-vec to be available.

    Args:
        query: Natural language description, e.g. 'sunset at the beach',
               'food photo', 'people laughing outdoors'.
        limit: Maximum results to return (default 10).
    """
    limit = min(limit, 100)
    try:
        query_embedding = _embedder(ctx).encode_text(query)
        photos = _db(ctx).search_by_embedding(query_embedding, limit)
    except Exception as exc:
        return {"error": f"Search failed: {exc}", "photos": [], "count": 0}

    return {"query": query, "photos": photos, "count": len(photos)}


@mcp.tool()
def find_duplicates(
    ctx: Context,
    threshold: float = 0.98,
) -> dict[str, Any]:
    """Find groups of near-duplicate photos using CLIP semantic similarity.

    Compares all embedded photos pairwise and returns groups where any pair
    has cosine similarity ≥ threshold. Uses union-find so transitive duplicates
    are in the same group.

    Args:
        threshold: Cosine similarity cutoff (0–1). Default 0.98 is strict;
                   lower to 0.90 to surface more near-duplicates.
    """
    if not (0.0 < threshold <= 1.0):
        return {"error": "threshold must be between 0 (exclusive) and 1 (inclusive)."}

    groups = _db(ctx).find_duplicate_groups(threshold)
    total_duplicates = sum(len(g) - 1 for g in groups)

    return {
        "threshold": threshold,
        "groups": [
            {"group_id": i + 1, "count": len(g), "photos": g}
            for i, g in enumerate(groups)
        ],
        "total_groups": len(groups),
        "total_duplicates": total_duplicates,
    }


@mcp.tool()
def flag_poor_quality(
    ctx: Context,
    blur_threshold: float = 100.0,
    update_db: bool = True,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict[str, Any]:
    """Score photos by sharpness and return those that appear blurry or low-quality.

    Loads each photo's image file and computes a sharpness score using edge-pixel
    variance. Photos below blur_threshold are considered poor quality.

    Args:
        blur_threshold: Sharpness cutoff (default 100.0). Typical range: ~50 (very
                        blurry) to ~500+ (sharp). Lower to catch only the blurriest.
        update_db:      Persist quality_score to the database (default True).
        limit:          Max poor-quality photos to return (default 50).
    """
    from photomind.quality import compute_sharpness

    db = _db(ctx)
    photos = db.photos_with_paths()
    limit = min(limit, 500)

    scored = 0
    failed = 0
    poor_quality: list[dict[str, Any]] = []

    for photo in photos:
        score = compute_sharpness(photo.get("filepath"))
        if score is None:
            failed += 1
            continue
        scored += 1
        if update_db:
            db.update_quality_score(photo["id"], score)
        if score < blur_threshold:
            photo["quality_score"] = round(score, 2)
            poor_quality.append(photo)

    poor_quality.sort(key=lambda p: p.get("quality_score") or 0.0)

    return {
        "blur_threshold": blur_threshold,
        "poor_quality_photos": poor_quality[:limit],
        "count": len(poor_quality),
        "scored": scored,
        "failed": failed,
    }


@mcp.tool()
def organise_photos(
    ctx: Context,
    group_by: str = "year_month",
) -> dict[str, Any]:
    """Analyse library structure and suggest a folder organisation.

    Read-only: does NOT move or modify any files. Returns a suggested folder
    hierarchy with photo counts and up to 3 sample UUIDs per folder.

    Args:
        group_by: Grouping strategy — 'year_month' (YYYY/MM, default) or 'year'.
    """
    if group_by not in ("year_month", "year"):
        return {"error": "group_by must be 'year_month' or 'year'."}

    rows = _db(ctx).conn.execute(
        "SELECT id, filename, date_taken FROM photos ORDER BY date_taken"
    ).fetchall()

    from collections import defaultdict

    structure: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "sample_ids": []}
    )
    unorganised = 0

    for row in rows:
        d = dict(row)
        date_str = d.get("date_taken") or ""
        if len(date_str) < 7:
            unorganised += 1
            continue
        try:
            year, month = date_str[:4], date_str[5:7]
            folder = f"{year}/{month}" if group_by == "year_month" else year
            structure[folder]["count"] += 1
            if len(structure[folder]["sample_ids"]) < 3:
                structure[folder]["sample_ids"].append(d["id"])
        except (IndexError, TypeError):
            unorganised += 1

    return {
        "group_by": group_by,
        "structure": dict(sorted(structure.items())),
        "total_folders": len(structure),
        "total_photos": len(rows),
        "unorganised": unorganised,
    }


@mcp.tool()
def get_delete_candidates(
    ctx: Context,
    duplicate_threshold: float = 0.98,
    blur_threshold: float = 500.0,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict[str, Any]:
    """Identify photos that are candidates for deletion: duplicates and blurry shots.

    Does NOT delete anything. Returns a prioritised list with the information
    needed to find and delete candidates manually in Photos.app.

    macOS prevents programmatic deletion from Photos.app (the Photos AppleScript
    delete verb is blocked for non-bundled processes). Use this tool to identify
    what to remove, then act in Photos.app:
      1. Navigate to the date shown for each candidate
      2. Select the photo → press Delete (⌫)  or  Image → Delete Photo

    Args:
        duplicate_threshold: Cosine similarity cutoff for duplicates (default 0.98).
        blur_threshold:      Sharpness score cutoff — photos below this are flagged
                             (default 500; scores typically range 200–5000+).
        limit:               Max candidates to return (default 50).
    """
    db = _db(ctx)
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Duplicates — keep the sharpest in each group, flag the rest
    for group_id, group in enumerate(db.find_duplicate_groups(duplicate_threshold), 1):
        group_sorted = sorted(
            group, key=lambda p: p.get("quality_score") or 0.0, reverse=True
        )
        keeper = group_sorted[0]
        for photo in group_sorted[1:]:
            if photo["id"] in seen_ids:
                continue
            seen_ids.add(photo["id"])
            candidates.append({
                "reason": "duplicate",
                "group_id": group_id,
                "id": photo["id"],
                "filename": photo["filename"],
                "date_taken": (photo.get("date_taken") or "")[:10],
                "quality_score": round(photo.get("quality_score") or 0.0, 1),
                "similarity_score": photo.get("similarity_score", 0.0),
                "keep_instead": keeper["filename"],
                "find_in_photos": _photos_date_hint(photo),
            })

    # Poor quality — below blur_threshold
    for photo in db.photos_with_paths():
        score = photo.get("quality_score")
        if score is None or float(score) >= blur_threshold or photo["id"] in seen_ids:
            continue
        seen_ids.add(photo["id"])
        candidates.append({
            "reason": "poor_quality",
            "id": photo["id"],
            "filename": photo["filename"],
            "date_taken": (photo.get("date_taken") or "")[:10],
            "quality_score": round(float(score), 1),
            "find_in_photos": _photos_date_hint(photo),
        })

    # Duplicates first, then blurriest-first within each group
    candidates.sort(key=lambda c: (c["reason"] != "duplicate", c.get("quality_score", 0)))

    return {
        "candidates": candidates[:limit],
        "total_candidates": len(candidates),
        "how_to_delete": (
            "Photos.app → navigate to the date → select photo → "
            "Delete (⌫)  or  Image → Delete Photo"
        ),
        "note": (
            "macOS blocks programmatic deletion from Photos.app. "
            "This tool identifies candidates; deletion is manual."
        ),
    }


def _photos_date_hint(photo: dict[str, Any]) -> str:
    """Return a human-readable hint for locating a photo in Photos.app."""
    date_str = photo.get("date_taken") or ""
    if len(date_str) >= 10:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(date_str)
            return f"Navigate to {dt.strftime('%-d %b %Y')} in Photos.app"
        except Exception:
            pass
    return "Search Photos.app by filename or date"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
