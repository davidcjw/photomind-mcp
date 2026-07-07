"""SQLite persistence layer."""
import json
import logging
import sqlite3
import struct
from collections import defaultdict
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from photomind.config import DB_PATH

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS photos (
    id             TEXT PRIMARY KEY,
    filename       TEXT,
    filepath       TEXT,
    date_taken     DATETIME,
    latitude       REAL,
    longitude      REAL,
    location_name  TEXT,
    camera_make    TEXT,
    camera_model   TEXT,
    keywords       TEXT,
    albums         TEXT,
    persons        TEXT,
    is_duplicate   BOOLEAN DEFAULT 0,
    quality_score  REAL,
    indexed_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_date_taken ON photos(date_taken);
CREATE INDEX IF NOT EXISTS idx_camera     ON photos(camera_model);

CREATE TABLE IF NOT EXISTS photo_embeddings (
    photo_id  TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);
"""


class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_DDL)
        self._conn.commit()
        logger.info("Database opened: %s", self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DatabaseManager":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def vec_available(self) -> bool:
        """Always False — sqlite-vec replaced by numpy cosine similarity."""
        return False

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("DatabaseManager not connected. Call connect() first.")
        return self._conn

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_photo(self, row: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO photos
                (id, filename, filepath, date_taken,
                 latitude, longitude, location_name,
                 camera_make, camera_model,
                 keywords, albums, persons,
                 is_duplicate, quality_score)
            VALUES
                (:id, :filename, :filepath, :date_taken,
                 :latitude, :longitude, :location_name,
                 :camera_make, :camera_model,
                 :keywords, :albums, :persons,
                 :is_duplicate, :quality_score)
            """,
            row,
        )

    def upsert_photos_batch(self, rows: list[dict[str, Any]]) -> None:
        with self.conn:
            for row in rows:
                self.upsert_photo(row)

    def photo_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM photos").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_photo(self, photo_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
        return _deserialize(dict(row)) if row else None

    def search_by_date(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        # Bare date strings (YYYY-MM-DD) won't match ISO datetime values stored in
        # the DB (e.g. "2025-12-06T22:11:09+08:00") because the datetime is
        # lexicographically greater than the bare date.  Extend to end-of-day.
        if "T" not in start_date:
            start_date = start_date + "T00:00:00"
        if "T" not in end_date:
            end_date = end_date + "T23:59:59"
        rows = self.conn.execute(
            """
            SELECT * FROM photos
            WHERE date_taken BETWEEN ? AND ?
            ORDER BY date_taken DESC
            LIMIT ?
            """,
            (start_date, end_date, limit),
        ).fetchall()
        return [_deserialize(dict(r)) for r in rows]

    def search_by_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * cos(radians(latitude)) + 1e-9)

        candidates = self.conn.execute(
            """
            SELECT * FROM photos
            WHERE latitude  BETWEEN ? AND ?
              AND longitude BETWEEN ? AND ?
              AND latitude IS NOT NULL
            """,
            (
                latitude - lat_delta, latitude + lat_delta,
                longitude - lon_delta, longitude + lon_delta,
            ),
        ).fetchall()

        results = []
        for r in candidates:
            d = dict(r)
            dist = _haversine(latitude, longitude, d["latitude"], d["longitude"])
            if dist <= radius_km:
                d["distance_km"] = round(dist, 3)
                results.append(_deserialize(d))

        results.sort(key=lambda x: x["distance_km"])
        return results[:limit]

    def search_by_metadata(
        self,
        keywords: list[str] | None = None,
        camera_model: str | None = None,
        persons: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if camera_model:
            clauses.append("LOWER(camera_model) LIKE LOWER(?)")
            params.append(f"%{camera_model}%")

        for kw in (keywords or []):
            clauses.append("LOWER(keywords) LIKE LOWER(?)")
            params.append(f'%"{kw.lower()}%')

        for person in (persons or []):
            clauses.append("LOWER(persons) LIKE LOWER(?)")
            params.append(f'%"{person.lower()}%')

        if not clauses:
            raise ValueError("At least one filter must be provided.")

        where = " AND ".join(clauses)
        rows = self.conn.execute(
            f"SELECT * FROM photos WHERE {where} ORDER BY date_taken DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [_deserialize(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embedded_photo_ids(self) -> set[str]:
        """Return the set of photo_ids that already have embeddings."""
        rows = self.conn.execute("SELECT photo_id FROM photo_embeddings").fetchall()
        return {r[0] for r in rows}

    def upsert_embedding(self, photo_id: str, embedding: list[float]) -> None:
        """Insert or replace a CLIP embedding (stored as packed float32 blob)."""
        blob = _pack(embedding)
        self.conn.execute(
            "INSERT OR REPLACE INTO photo_embeddings(photo_id, embedding) VALUES (?, ?)",
            (photo_id, blob),
        )

    def search_by_embedding(
        self, query_embedding: list[float], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return photos ranked by cosine similarity to query_embedding.

        Loads all embeddings from the DB and scores them with numpy.
        Fast enough for libraries up to ~10k photos on CPU.
        """
        rows = self.conn.execute(
            "SELECT photo_id, embedding FROM photo_embeddings"
        ).fetchall()
        if not rows:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9

        ids = [r[0] for r in rows]
        matrix = np.stack([_unpack(r[1]) for r in rows])  # (N, 512)
        scores = matrix @ q                                 # cosine similarity

        top_k = min(limit, len(ids))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for i in top_idx:
            photo = self.get_photo(ids[i])
            if photo:
                photo["similarity_score"] = round(float(scores[i]), 4)
                results.append(photo)
        return results


    def find_duplicate_groups(self, threshold: float = 0.98) -> list[list[dict[str, Any]]]:
        """Return groups of near-duplicate photos by CLIP cosine similarity.

        Uses union-find on pairwise similarities so transitive duplicates are
        placed in the same group.  Returns only groups with ≥ 2 members.
        """
        rows = self.conn.execute(
            "SELECT photo_id, embedding FROM photo_embeddings"
        ).fetchall()
        if len(rows) < 2:
            return []

        ids = [r[0] for r in rows]
        n = len(ids)
        matrix = np.stack([_unpack(r[1]) for r in rows])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        matrix = matrix / norms
        sim = matrix @ matrix.T  # (N, N) pairwise cosine similarities

        # Union-Find
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            px, py = _find(x), _find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if float(sim[i, j]) >= threshold:
                    _union(i, j)

        component_map: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            component_map[_find(i)].append(i)

        groups = []
        for component_indices in component_map.values():
            if len(component_indices) < 2:
                continue
            ref = component_indices[0]
            group_photos = []
            for idx in component_indices:
                photo = self.get_photo(ids[idx])
                if photo:
                    score = 1.0 if idx == ref else float(sim[ref, idx])
                    photo["similarity_score"] = round(score, 4)
                    group_photos.append(photo)
            if len(group_photos) >= 2:
                groups.append(group_photos)
        return groups

    def update_duplicate_flags(self, photo_ids: list[str], is_duplicate: bool) -> int:
        """Set is_duplicate on a list of photos. Returns number of rows updated."""
        val = 1 if is_duplicate else 0
        with self.conn:
            self.conn.executemany(
                "UPDATE photos SET is_duplicate = ? WHERE id = ?",
                [(val, pid) for pid in photo_ids],
            )
        return len(photo_ids)

    def update_quality_score(self, photo_id: str, score: float) -> None:
        """Persist a quality_score for a single photo."""
        with self.conn:
            self.conn.execute(
                "UPDATE photos SET quality_score = ? WHERE id = ?",
                (score, photo_id),
            )

    def photos_with_paths(self) -> list[dict[str, Any]]:
        """Return all photos that have a non-null filepath."""
        rows = self.conn.execute(
            "SELECT * FROM photos WHERE filepath IS NOT NULL ORDER BY date_taken DESC"
        ).fetchall()
        return [_deserialize(dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def _deserialize(row: dict[str, Any]) -> dict[str, Any]:
    for col in ("keywords", "albums", "persons"):
        if isinstance(row.get(col), str):
            try:
                row[col] = json.loads(row[col])
            except (json.JSONDecodeError, TypeError):
                row[col] = []
    return row


def _pack(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)
