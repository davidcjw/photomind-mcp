"""Tests for directory_indexer — no Photos.app required."""
import json
from pathlib import Path

import pytest
from PIL import Image

from photomind.database import DatabaseManager
from photomind.directory_indexer import (
    DirectoryIndexer,
    _parse_gps_coord,
    extract_exif,
    is_photos_library_path,
    photo_id_from_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / "test.db")
    manager.connect()
    yield manager
    manager.close()


def _make_jpeg(path: Path, color: tuple = (128, 128, 128)) -> Path:
    """Save a minimal JPEG with no EXIF to *path*."""
    img = Image.new("RGB", (64, 64), color=color)
    img.save(path, format="JPEG")
    return path


# ---------------------------------------------------------------------------
# photo_id_from_path
# ---------------------------------------------------------------------------

class TestPhotoIdFromPath:
    def test_deterministic(self, tmp_path: Path):
        p = tmp_path / "a.jpg"
        assert photo_id_from_path(p) == photo_id_from_path(p)

    def test_different_paths_differ(self, tmp_path: Path):
        assert photo_id_from_path(tmp_path / "a.jpg") != photo_id_from_path(tmp_path / "b.jpg")

    def test_uuid_format(self, tmp_path: Path):
        pid = photo_id_from_path(tmp_path / "a.jpg")
        parts = pid.split("-")
        assert len(parts) == 5
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]


# ---------------------------------------------------------------------------
# _parse_gps_coord
# ---------------------------------------------------------------------------

class TestParseGpsCoord:
    def test_north(self):
        assert _parse_gps_coord((1, 21, 7.56), "N") == pytest.approx(1.35210, rel=1e-4)

    def test_south_negative(self):
        result = _parse_gps_coord((33, 52, 0), "S")
        assert result is not None
        assert result < 0

    def test_west_negative(self):
        result = _parse_gps_coord((103, 49, 0), "W")
        assert result is not None
        assert result < 0

    def test_none_coord_returns_none(self):
        assert _parse_gps_coord(None, "N") is None

    def test_none_ref_returns_none(self):
        assert _parse_gps_coord((1, 0, 0), None) is None


# ---------------------------------------------------------------------------
# extract_exif
# ---------------------------------------------------------------------------

class TestExtractExif:
    def test_plain_jpeg_returns_none_fields(self, tmp_path: Path):
        path = _make_jpeg(tmp_path / "plain.jpg")
        exif = extract_exif(path)
        assert exif["date_taken"] is None
        assert exif["latitude"] is None
        assert exif["camera_make"] is None

    def test_missing_file_returns_none_fields(self, tmp_path: Path):
        exif = extract_exif(tmp_path / "ghost.jpg")
        assert exif["date_taken"] is None


# ---------------------------------------------------------------------------
# is_photos_library_path
# ---------------------------------------------------------------------------

class TestIsPhotosLibraryPath:
    def test_photos_library_detected(self):
        assert is_photos_library_path(
            "/Users/me/Pictures/Photos Library.photoslibrary/originals/A/abc.heic"
        )

    def test_plain_path_not_detected(self):
        assert not is_photos_library_path("/Users/me/Downloads/photos/IMG_001.jpg")

    def test_empty_string(self):
        assert not is_photos_library_path("")


# ---------------------------------------------------------------------------
# DirectoryIndexer.sync
# ---------------------------------------------------------------------------

class TestDirectoryIndexer:
    def test_indexes_supported_files(self, db: DatabaseManager, tmp_path: Path):
        _make_jpeg(tmp_path / "a.jpg")
        _make_jpeg(tmp_path / "b.jpg")
        (tmp_path / "readme.txt").write_text("not a photo")

        result = DirectoryIndexer(db).sync(tmp_path)
        assert result["total"] == 2
        assert result["indexed"] == 2
        assert result["errors"] == 0
        assert db.photo_count() == 2

    def test_nested_directory_walked(self, db: DatabaseManager, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _make_jpeg(sub / "deep.jpg")
        result = DirectoryIndexer(db).sync(tmp_path)
        assert result["total"] == 1

    def test_nonexistent_directory_raises(self, db: DatabaseManager, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            DirectoryIndexer(db).sync(tmp_path / "ghost")

    def test_file_path_raises(self, db: DatabaseManager, tmp_path: Path):
        f = _make_jpeg(tmp_path / "a.jpg")
        with pytest.raises(NotADirectoryError):
            DirectoryIndexer(db).sync(f)

    def test_resync_is_idempotent(self, db: DatabaseManager, tmp_path: Path):
        _make_jpeg(tmp_path / "a.jpg")
        DirectoryIndexer(db).sync(tmp_path)
        DirectoryIndexer(db).sync(tmp_path)
        assert db.photo_count() == 1

    def test_row_has_expected_fields(self, db: DatabaseManager, tmp_path: Path):
        _make_jpeg(tmp_path / "test.jpg")
        DirectoryIndexer(db).sync(tmp_path)
        photo = db.conn.execute("SELECT * FROM photos").fetchone()
        assert photo is not None
        d = dict(photo)
        assert d["filename"] == "test.jpg"
        assert d["filepath"] == str(tmp_path / "test.jpg")
        assert isinstance(json.loads(d["keywords"]), list)
