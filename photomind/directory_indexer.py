"""Directory-based photo indexer — no Photos.app required.

Walks a plain filesystem folder, extracts EXIF via Pillow, and stores
records in the same SQLite schema used by the Photos.app indexer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif", ".webp"}
)

# Refuse to delete files that live inside a Photos.app library bundle
PHOTOS_LIBRARY_MARKER = "Photos Library.photoslibrary"


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def photo_id_from_path(path: Path) -> str:
    """Return a stable UUID-shaped ID derived from the absolute file path."""
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    h = digest[:32]
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ---------------------------------------------------------------------------
# EXIF extraction
# ---------------------------------------------------------------------------

def _parse_gps_coord(
    coord: tuple | None,
    ref: str | None,
) -> float | None:
    """Convert a GPS DMS tuple + hemisphere ref to signed decimal degrees."""
    if not coord or not ref:
        return None
    try:
        degrees = float(coord[0])
        minutes = float(coord[1])
        seconds = float(coord[2])
        decimal = degrees + minutes / 60.0 + seconds / 3600.0
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal
    except (TypeError, IndexError, ValueError):
        return None


def extract_exif(path: Path) -> dict[str, Any]:
    """Return EXIF-derived metadata for an image file.

    Returns a dict with keys: date_taken, latitude, longitude,
    camera_make, camera_model.  All values default to None on failure.
    """
    result: dict[str, Any] = {
        "date_taken": None,
        "latitude": None,
        "longitude": None,
        "camera_make": None,
        "camera_model": None,
    }
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS

        img = Image.open(path)
        raw = img._getexif()  # type: ignore[attr-defined]
        if not raw:
            return result

        exif = {TAGS.get(k, k): v for k, v in raw.items()}

        # Date
        for field in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
            raw_dt = exif.get(field)
            if raw_dt:
                try:
                    dt = datetime.strptime(str(raw_dt), "%Y:%m:%d %H:%M:%S")
                    result["date_taken"] = dt.replace(tzinfo=timezone.utc).isoformat()
                    break
                except (ValueError, TypeError):
                    pass

        # Camera
        result["camera_make"] = exif.get("Make") or None
        result["camera_model"] = exif.get("Model") or None

        # GPS
        gps_raw = exif.get("GPSInfo")
        if gps_raw:
            gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
            result["latitude"] = _parse_gps_coord(
                gps.get("GPSLatitude"), gps.get("GPSLatitudeRef")
            )
            result["longitude"] = _parse_gps_coord(
                gps.get("GPSLongitude"), gps.get("GPSLongitudeRef")
            )
    except Exception as exc:
        logger.debug("EXIF extraction failed for %s: %s", path, exc)

    return result


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class DirectoryIndexer:
    """Index photos from a plain filesystem directory into the shared DB."""

    def __init__(self, db: Any, embedder: Any = None) -> None:
        self._db = db
        self._embedder = embedder

    def sync(self, directory: str | Path) -> dict[str, Any]:
        """Walk *directory* recursively and index all supported images.

        Returns a summary dict with total, indexed, errors, embedded, etc.
        """
        from photomind.config import SYNC_BATCH_SIZE

        root = Path(directory).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        files = [
            p for p in root.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
        ]

        total = len(files)
        indexed = 0
        errors = 0
        start = time.time()

        # Metadata pass
        batch: list[dict[str, Any]] = []
        for path in files:
            try:
                batch.append(self._build_row(path))
                indexed += 1
            except Exception as exc:
                logger.warning("Failed to index %s: %s", path, exc)
                errors += 1
            if len(batch) >= SYNC_BATCH_SIZE:
                self._db.upsert_photos_batch(batch)
                self._db.conn.commit()
                batch = []

        if batch:
            self._db.upsert_photos_batch(batch)
            self._db.conn.commit()

        # Embedding pass — skips photos already embedded
        embedded = 0
        embed_errors = 0
        if self._embedder:
            already_embedded = self._db.embedded_photo_ids()
            for photo in self._db.photos_with_paths():
                if photo["id"] in already_embedded:
                    continue
                emb = self._embedder.encode_image(photo["filepath"])
                if emb:
                    self._db.upsert_embedding(photo["id"], emb)
                    self._db.conn.commit()
                    embedded += 1
                else:
                    embed_errors += 1

        return {
            "total": total,
            "indexed": indexed,
            "errors": errors,
            "embedded": embedded,
            "embed_errors": embed_errors,
            "duration_seconds": round(time.time() - start, 1),
            "directory": str(root),
        }

    def _build_row(self, path: Path) -> dict[str, Any]:
        exif = extract_exif(path)
        return {
            "id": photo_id_from_path(path),
            "filename": path.name,
            "filepath": str(path),
            "date_taken": exif["date_taken"],
            "latitude": exif["latitude"],
            "longitude": exif["longitude"],
            "location_name": None,
            "camera_make": exif["camera_make"],
            "camera_model": exif["camera_model"],
            "keywords": json.dumps([]),
            "albums": json.dumps([]),
            "persons": json.dumps([]),
            "is_duplicate": 0,
            "quality_score": None,
        }


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def is_photos_library_path(filepath: str) -> bool:
    """Return True if *filepath* lives inside a Photos.app library bundle."""
    return PHOTOS_LIBRARY_MARKER in filepath
