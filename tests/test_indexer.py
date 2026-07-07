"""Unit tests for database and indexer — no Photos.app required."""
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from photomind.database import DatabaseManager, _haversine
from photomind.indexer import PhotoIndexer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / "test.db")
    manager.connect()
    yield manager
    manager.close()


@pytest.fixture
def indexer(db: DatabaseManager) -> PhotoIndexer:
    return PhotoIndexer(db)


def _row(uuid: str = "abc-123", **overrides) -> dict:
    base = {
        "id": uuid,
        "filename": "IMG_0001.JPG",
        "filepath": "/Photos/2023/IMG_0001.JPG",
        "date_taken": "2023-06-15T10:30:00+00:00",
        "latitude": 1.3521,
        "longitude": 103.8198,
        "location_name": None,
        "camera_make": "Apple",
        "camera_model": "iPhone 15 Pro",
        "keywords": json.dumps(["travel", "Singapore"]),
        "albums": json.dumps(["Asia Trip 2023"]),
        "persons": json.dumps(["Alice"]),
        "is_duplicate": 0,
        "quality_score": None,
    }
    base.update(overrides)
    return base


def _mock_photo(
    uuid="uuid-1",
    filename="IMG_001.JPG",
    path="/path/IMG_001.JPG",
    date=None,
    location=(1.3521, 103.8198),
    keywords=None,
    albums=None,
    persons=None,
    camera_make="Apple",
    camera_model="iPhone 15 Pro",
) -> MagicMock:
    photo = MagicMock()
    photo.uuid = uuid
    photo.filename = filename
    photo.path = path
    photo.date = date or datetime(2023, 6, 15, 10, 30, tzinfo=timezone.utc)
    photo.location = location
    photo.keywords = keywords or ["travel"]
    photo.albums = albums or ["Trip"]
    photo.persons = persons or ["Bob"]
    exif = MagicMock()
    exif.camera_make = camera_make
    exif.camera_model = camera_model
    photo.exif_info = exif
    return photo


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------

class TestDatabaseManager:
    def test_schema_created(self, db: DatabaseManager):
        count = db.conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='photos'"
        ).fetchone()[0]
        assert count == 1

    def test_upsert_and_get(self, db: DatabaseManager):
        db.upsert_photos_batch([_row()])
        db.conn.commit()
        result = db.get_photo("abc-123")
        assert result is not None
        assert result["filename"] == "IMG_0001.JPG"
        assert isinstance(result["keywords"], list)
        assert "travel" in result["keywords"]

    def test_upsert_is_idempotent(self, db: DatabaseManager):
        db.upsert_photos_batch([_row()])
        db.upsert_photos_batch([_row(filename="IMG_updated.JPG")])
        db.conn.commit()
        assert db.photo_count() == 1
        assert db.get_photo("abc-123")["filename"] == "IMG_updated.JPG"

    def test_get_photo_not_found(self, db: DatabaseManager):
        assert db.get_photo("no-such-id") is None

    def test_search_by_date(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("p1", date_taken="2023-01-15T00:00:00+00:00"),
            _row("p2", date_taken="2023-06-15T00:00:00+00:00"),
            _row("p3", date_taken="2024-01-01T00:00:00+00:00"),
        ])
        db.conn.commit()
        results = db.search_by_date("2023-01-01", "2023-12-31", limit=10)
        ids = [r["id"] for r in results]
        assert "p1" in ids
        assert "p2" in ids
        assert "p3" not in ids

    def test_search_by_date_bare_date_includes_same_day(self, db: DatabaseManager):
        """Bare YYYY-MM-DD end_date must match datetime values on that same day."""
        db.upsert_photos_batch([
            _row("evening", date_taken="2023-06-15T22:11:09+08:00"),
        ])
        db.conn.commit()
        results = db.search_by_date("2023-06-15", "2023-06-15", limit=10)
        assert any(r["id"] == "evening" for r in results)

    def test_search_by_location_near(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("close", latitude=1.3521, longitude=103.8198),
            _row("far", latitude=35.6762, longitude=139.6503),
        ])
        db.conn.commit()
        results = db.search_by_location(1.3521, 103.8198, radius_km=10.0)
        ids = [r["id"] for r in results]
        assert "close" in ids
        assert "far" not in ids

    def test_search_by_location_null_excluded(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("no-gps", latitude=None, longitude=None)])
        db.conn.commit()
        results = db.search_by_location(1.3521, 103.8198, radius_km=9999.0)
        assert all(r["id"] != "no-gps" for r in results)

    def test_search_by_location_distance_km_field(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("p", latitude=1.3521, longitude=103.8198)])
        db.conn.commit()
        results = db.search_by_location(1.3521, 103.8198, radius_km=1.0)
        assert results[0]["distance_km"] == pytest.approx(0.0, abs=0.001)

    def test_search_by_metadata_camera(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("iphone", camera_model="iPhone 15 Pro"),
            _row("canon", camera_model="Canon EOS R5"),
        ])
        db.conn.commit()
        results = db.search_by_metadata(camera_model="iPhone")
        assert any(r["id"] == "iphone" for r in results)
        assert all(r["id"] != "canon" for r in results)

    def test_search_by_metadata_no_filters_raises(self, db: DatabaseManager):
        with pytest.raises(ValueError):
            db.search_by_metadata()


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point(self):
        assert _haversine(1.0, 103.0, 1.0, 103.0) == pytest.approx(0.0, abs=0.001)

    def test_singapore_to_kl(self):
        dist = _haversine(1.3521, 103.8198, 3.1390, 101.6869)
        assert 305 < dist < 320


# ---------------------------------------------------------------------------
# PhotoIndexer._extract
# ---------------------------------------------------------------------------

class TestExtract:
    def test_normal(self, indexer: PhotoIndexer):
        photo = _mock_photo()
        row = indexer._extract(photo)
        assert row is not None
        assert row["id"] == "uuid-1"
        assert row["latitude"] == pytest.approx(1.3521)
        assert row["camera_model"] == "iPhone 15 Pro"
        assert json.loads(row["keywords"]) == ["travel"]

    def test_none_path_allowed(self, indexer: PhotoIndexer):
        photo = _mock_photo(path=None)
        row = indexer._extract(photo)
        assert row["filepath"] is None

    def test_none_location_allowed(self, indexer: PhotoIndexer):
        photo = _mock_photo(location=None)
        row = indexer._extract(photo)
        assert row["latitude"] is None
        assert row["longitude"] is None

    def test_partial_none_location(self, indexer: PhotoIndexer):
        photo = _mock_photo(location=(None, 103.8))
        row = indexer._extract(photo)
        assert row["latitude"] is None

    def test_none_uuid_returns_none(self, indexer: PhotoIndexer):
        photo = _mock_photo(uuid=None)
        assert indexer._extract(photo) is None

    def test_none_exif_graceful(self, indexer: PhotoIndexer):
        photo = _mock_photo()
        photo.exif_info = None
        row = indexer._extract(photo)
        assert row["camera_make"] is None
        assert row["camera_model"] is None

    def test_naive_datetime_utc_attached(self, indexer: PhotoIndexer):
        photo = _mock_photo(date=datetime(2023, 6, 15, 10, 30))  # no tzinfo
        row = indexer._extract(photo)
        assert row["date_taken"] is not None
        assert "+00:00" in row["date_taken"]


# ---------------------------------------------------------------------------
# PhotoIndexer.sync
# ---------------------------------------------------------------------------

class TestSync:
    @pytest.mark.macos  # patches osxphotos.PhotosDB, which requires a real (macOS-only) osxphotos import
    def test_error_counted_but_good_photo_indexed(self, db: DatabaseManager):
        good = _mock_photo("good-uuid")
        indexer = PhotoIndexer(db)

        with patch("osxphotos.PhotosDB") as mock_pdb_cls:
            mock_pdb_cls.return_value.photos.return_value = [good, MagicMock()]
            original_extract = indexer._extract

            call_count = [0]

            def extract_side_effect(photo):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise OSError("permission denied")
                return original_extract(photo)

            with patch.object(indexer, "_extract", side_effect=extract_side_effect):
                result = indexer.sync()

        assert result["errors"] == 1
        assert result["indexed"] == 1
        assert db.get_photo("good-uuid") is not None

    def test_missing_osxphotos_raises(self, db: DatabaseManager):
        indexer = PhotoIndexer(db)
        with patch.dict("sys.modules", {"osxphotos": None}):
            with pytest.raises(ImportError):
                indexer.sync()


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _pack_vec(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


class TestDuplicates:
    def _insert_embedding(self, db: DatabaseManager, photo_id: str, vec: list[float]) -> None:
        db.upsert_photos_batch([_row(photo_id)])
        db.conn.commit()
        db.conn.execute(
            "INSERT OR REPLACE INTO photo_embeddings(photo_id, embedding) VALUES (?,?)",
            (photo_id, _pack_vec(vec)),
        )
        db.conn.commit()

    def test_no_embeddings_returns_empty(self, db: DatabaseManager):
        assert db.find_duplicate_groups() == []

    def test_single_embedding_returns_empty(self, db: DatabaseManager):
        self._insert_embedding(db, "solo", [1.0, 0.0, 0.0])
        assert db.find_duplicate_groups() == []

    def test_identical_embeddings_grouped(self, db: DatabaseManager):
        vec = [1.0, 0.0, 0.0]
        self._insert_embedding(db, "p1", vec)
        self._insert_embedding(db, "p2", vec)
        groups = db.find_duplicate_groups(threshold=0.98)
        assert len(groups) == 1
        ids_in_group = {p["id"] for p in groups[0]}
        assert ids_in_group == {"p1", "p2"}

    def test_dissimilar_embeddings_no_groups(self, db: DatabaseManager):
        self._insert_embedding(db, "a", [1.0, 0.0, 0.0])
        self._insert_embedding(db, "b", [0.0, 1.0, 0.0])
        assert db.find_duplicate_groups(threshold=0.98) == []

    def test_update_duplicate_flags(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("dup1"), _row("dup2")])
        db.conn.commit()
        count = db.update_duplicate_flags(["dup1", "dup2"], is_duplicate=True)
        assert count == 2
        assert db.get_photo("dup1")["is_duplicate"] == 1
        db.update_duplicate_flags(["dup1"], is_duplicate=False)
        assert db.get_photo("dup1")["is_duplicate"] == 0

    def test_vec_available_is_false(self, db: DatabaseManager):
        assert db.vec_available is False


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

class TestQuality:
    def test_update_quality_score(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("q1")])
        db.conn.commit()
        db.update_quality_score("q1", 123.45)
        assert db.get_photo("q1")["quality_score"] == pytest.approx(123.45)

    def test_photos_with_paths_excludes_null(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("has-path", filepath="/some/path.jpg"),
            _row("no-path", filepath=None),
        ])
        db.conn.commit()
        results = db.photos_with_paths()
        ids = [r["id"] for r in results]
        assert "has-path" in ids
        assert "no-path" not in ids


# ---------------------------------------------------------------------------
# compute_sharpness
# ---------------------------------------------------------------------------


class TestComputeSharpness:
    def test_none_filepath_returns_none(self):
        from photomind.quality import compute_sharpness
        assert compute_sharpness(None) is None

    def test_missing_file_returns_none(self, tmp_path: Path):
        from photomind.quality import compute_sharpness
        assert compute_sharpness(tmp_path / "ghost.jpg") is None

    def test_blank_image_low_score(self, tmp_path: Path):
        from PIL import Image
        from photomind.quality import compute_sharpness

        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        path = tmp_path / "blank.png"
        img.save(path)
        score = compute_sharpness(path)
        assert score is not None
        assert score < 500.0  # uniform image → low sharpness (resampling adds minor artifacts)

    def test_edge_image_high_score(self, tmp_path: Path):
        from PIL import Image
        from photomind.quality import compute_sharpness

        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:50, :] = 255  # sharp horizontal edge at midpoint
        img = Image.fromarray(arr)
        path = tmp_path / "edge.png"
        img.save(path)
        score = compute_sharpness(path)
        assert score is not None
        assert score > 100.0
