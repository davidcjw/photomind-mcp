"""Ingestion of Photos.app library via osxphotos."""
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from photomind.config import MAX_SYNC_ERRORS, SYNC_BATCH_SIZE
from photomind.database import DatabaseManager

if TYPE_CHECKING:
    from photomind.embeddings import CLIPEmbedder

logger = logging.getLogger(__name__)


class PhotoIndexer:
    def __init__(self, db: DatabaseManager, embedder: "CLIPEmbedder | None" = None) -> None:
        self._db = db
        self._embedder = embedder

    def sync(self) -> dict[str, Any]:
        """Index all photos from Photos.app into SQLite.

        Returns {total, indexed, skipped, errors, duration_seconds}.
        """
        try:
            from osxphotos import PhotosDB
        except ImportError as exc:
            raise ImportError("osxphotos is not installed. Run: uv sync") from exc

        try:
            photos_db = PhotosDB()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot open Photos.app library: {exc}. "
                "Ensure Photos.app has been opened at least once and "
                "the app has Full Disk Access in System Settings > Privacy & Security."
            ) from exc

        started_at = datetime.now(timezone.utc)
        photos = photos_db.photos()
        total = len(photos)
        indexed = skipped = errors = 0
        batch: list[dict[str, Any]] = []
        error_count = 0

        for photo in photos:
            if error_count >= MAX_SYNC_ERRORS:
                logger.error(
                    "Aborting sync: %d errors exceeded MAX_SYNC_ERRORS=%d after %d photos.",
                    error_count, MAX_SYNC_ERRORS, indexed,
                )
                break

            try:
                row = self._extract(photo)
            except Exception as exc:
                logger.warning("Skipping photo %s: %s", getattr(photo, "uuid", "?"), exc)
                error_count += 1
                errors += 1
                continue

            if row is None:
                skipped += 1
                continue

            batch.append(row)
            if len(batch) >= SYNC_BATCH_SIZE:
                self._db.upsert_photos_batch(batch)
                indexed += len(batch)
                logger.debug("Committed batch of %d photos.", len(batch))
                batch = []

        if batch:
            self._db.upsert_photos_batch(batch)
            indexed += len(batch)

        duration = (datetime.now(timezone.utc) - started_at).total_seconds()
        logger.info(
            "Metadata sync: %d/%d indexed, %d skipped, %d errors in %.1fs",
            indexed, total, skipped, errors, duration,
        )

        # --- Embedding pass ---
        embedded = 0
        embed_errors = 0
        if self._embedder is not None:
            embedded, embed_errors = self._embed_pass()

        total_duration = (datetime.now(timezone.utc) - started_at).total_seconds()
        return {
            "total": total,
            "indexed": indexed,
            "skipped": skipped,
            "errors": errors,
            "embedded": embedded,
            "embed_errors": embed_errors,
            "duration_seconds": round(total_duration, 1),
        }

    def _embed_pass(self) -> tuple[int, int]:
        """Generate CLIP embeddings for all photos not yet embedded."""
        already_done = self._db.embedded_photo_ids()
        rows = self._db.conn.execute(
            "SELECT id, filepath FROM photos WHERE filepath IS NOT NULL"
        ).fetchall()

        to_embed = [(r["id"], r["filepath"]) for r in rows if r["id"] not in already_done]
        logger.info("Embedding %d new photos (skipping %d already done)…", len(to_embed), len(already_done))

        embedded = errors = 0
        for photo_id, filepath in to_embed:
            embedding = self._embedder.encode_image(filepath)
            if embedding is None:
                errors += 1
                continue
            with self._db.conn:
                self._db.upsert_embedding(photo_id, embedding)
            embedded += 1
            if embedded % 10 == 0:
                logger.info("Embedded %d/%d photos…", embedded, len(to_embed))

        logger.info("Embedding pass complete: %d embedded, %d errors.", embedded, errors)
        return embedded, errors

    def _extract(self, photo: Any) -> dict[str, Any] | None:
        uuid: str = photo.uuid
        if not uuid:
            return None

        filepath: str | None = None
        try:
            filepath = photo.path
        except Exception:
            pass

        date_taken: str | None = None
        try:
            dt = photo.date
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                date_taken = dt.isoformat()
        except Exception:
            pass

        latitude: float | None = None
        longitude: float | None = None
        try:
            loc = photo.location
            if loc is not None and len(loc) == 2:
                lat, lon = loc
                if lat is not None and lon is not None:
                    latitude = float(lat)
                    longitude = float(lon)
        except Exception:
            pass

        camera_make: str | None = None
        camera_model: str | None = None
        try:
            exif = photo.exif_info
            if exif is not None:
                camera_make = getattr(exif, "camera_make", None) or None
                camera_model = getattr(exif, "camera_model", None) or None
        except Exception:
            pass

        keywords: list[str] = _safe_list(photo, "keywords")
        albums: list[str] = _safe_list(photo, "albums")
        persons: list[str] = _safe_list(photo, "persons")

        return {
            "id": uuid,
            "filename": getattr(photo, "filename", None),
            "filepath": filepath,
            "date_taken": date_taken,
            "latitude": latitude,
            "longitude": longitude,
            "location_name": None,
            "camera_make": camera_make,
            "camera_model": camera_model,
            "keywords": json.dumps(keywords),
            "albums": json.dumps(albums),
            "persons": json.dumps(persons),
            "is_duplicate": int(getattr(photo, "duplicate", False) or False),
            "quality_score": None,
        }


def _safe_list(photo: Any, attr: str) -> list[str]:
    try:
        val = getattr(photo, attr, None)
        if isinstance(val, list):
            return [str(v) for v in val if v is not None]
    except Exception:
        pass
    return []
