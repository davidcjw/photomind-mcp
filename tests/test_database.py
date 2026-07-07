"""Unit tests for photomind.database.DatabaseManager.

Pure SQLite + numpy layer — no macOS/Photos.app dependency, runs on Linux CI.
All tests use a tmp_path-based sqlite file, never the real ~/Library path.
"""
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from photomind.database import DatabaseManager, _haversine


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / "photomind_test.db")
    manager.connect()
    yield manager
    manager.close()


def _row(uuid: str = "abc-123", **overrides) -> dict:
    base = {
        "id": uuid,
        "filename": "IMG_0001.JPG",
        "filepath": "/Photos/2023/IMG_0001.JPG",
        "date_taken": "2023-06-15T10:30:00+00:00",
        "latitude": 1.3521,
        "longitude": 103.8198,
        "location_name": "Singapore",
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


def _pack_vec(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _insert_embedding(db: DatabaseManager, photo_id: str, vec: list[float]) -> None:
    """Insert a photo row + its embedding blob."""
    db.upsert_photos_batch([_row(photo_id)])
    db.conn.execute(
        "INSERT OR REPLACE INTO photo_embeddings(photo_id, embedding) VALUES (?, ?)",
        (photo_id, _pack_vec(vec)),
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# connect() / close() and schema
# ---------------------------------------------------------------------------

class TestConnectionAndSchema:
    def test_connect_creates_photos_table(self, db: DatabaseManager):
        row = db.conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='photos'"
        ).fetchone()
        assert row[0] == 1

    def test_connect_creates_embeddings_table(self, db: DatabaseManager):
        row = db.conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='photo_embeddings'"
        ).fetchone()
        assert row[0] == 1

    def test_connect_creates_indexes(self, db: DatabaseManager):
        names = {
            r[0]
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_date_taken" in names
        assert "idx_camera" in names

    def test_photos_table_columns(self, db: DatabaseManager):
        cols = {r[1] for r in db.conn.execute("PRAGMA table_info(photos)").fetchall()}
        expected = {
            "id", "filename", "filepath", "date_taken", "latitude", "longitude",
            "location_name", "camera_make", "camera_model", "keywords", "albums",
            "persons", "is_duplicate", "quality_score", "indexed_at",
        }
        assert expected <= cols

    def test_close_releases_connection(self, tmp_path: Path):
        manager = DatabaseManager(db_path=tmp_path / "close_test.db")
        manager.connect()
        manager.close()
        with pytest.raises(RuntimeError):
            _ = manager.conn

    def test_conn_before_connect_raises(self, tmp_path: Path):
        manager = DatabaseManager(db_path=tmp_path / "never.db")
        with pytest.raises(RuntimeError):
            _ = manager.conn

    def test_context_manager(self, tmp_path: Path):
        with DatabaseManager(db_path=tmp_path / "ctx.db") as manager:
            assert manager.photo_count() == 0


# ---------------------------------------------------------------------------
# Insert / read round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_photo_round_trips(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("rt-1")])
        result = db.get_photo("rt-1")
        assert result is not None
        assert result["id"] == "rt-1"
        assert result["filename"] == "IMG_0001.JPG"
        assert result["filepath"] == "/Photos/2023/IMG_0001.JPG"
        assert result["camera_make"] == "Apple"
        assert result["camera_model"] == "iPhone 15 Pro"
        assert result["latitude"] == pytest.approx(1.3521)
        assert result["longitude"] == pytest.approx(103.8198)

    def test_json_columns_deserialized_to_lists(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("rt-2")])
        result = db.get_photo("rt-2")
        assert result["keywords"] == ["travel", "Singapore"]
        assert result["albums"] == ["Asia Trip 2023"]
        assert result["persons"] == ["Alice"]

    def test_get_missing_photo_returns_none(self, db: DatabaseManager):
        assert db.get_photo("does-not-exist") is None

    def test_photo_count(self, db: DatabaseManager):
        assert db.photo_count() == 0
        db.upsert_photos_batch([_row("a"), _row("b"), _row("c")])
        assert db.photo_count() == 3

    def test_upsert_replaces_existing(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("dup")])
        db.upsert_photos_batch([_row("dup", filename="IMG_NEW.JPG")])
        assert db.photo_count() == 1
        assert db.get_photo("dup")["filename"] == "IMG_NEW.JPG"


# ---------------------------------------------------------------------------
# search_by_date
# ---------------------------------------------------------------------------

class TestSearchByDate:
    def test_filters_to_range(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("before", date_taken="2022-12-31T23:59:59+00:00"),
            _row("inside", date_taken="2023-06-15T00:00:00+00:00"),
            _row("after", date_taken="2024-01-01T00:00:00+00:00"),
        ])
        ids = [r["id"] for r in db.search_by_date("2023-01-01", "2023-12-31")]
        assert ids == ["inside"]

    def test_boundaries_are_inclusive(self, db: DatabaseManager):
        """BETWEEN is inclusive of both endpoints when full ISO datetimes match."""
        db.upsert_photos_batch([
            _row("start", date_taken="2023-06-01T00:00:00+00:00"),
            _row("end", date_taken="2023-06-30T00:00:00+00:00"),
        ])
        ids = {
            r["id"]
            for r in db.search_by_date(
                "2023-06-01T00:00:00+00:00", "2023-06-30T00:00:00+00:00"
            )
        }
        assert ids == {"start", "end"}

    def test_just_outside_boundary_excluded(self, db: DatabaseManager):
        """A row one second past the (exclusive-of-anything-later) end is dropped."""
        db.upsert_photos_batch([
            _row("on_end", date_taken="2023-06-30T12:00:00+00:00"),
            _row("past_end", date_taken="2023-06-30T12:00:01+00:00"),
        ])
        ids = {
            r["id"]
            for r in db.search_by_date(
                "2023-06-01T00:00:00+00:00", "2023-06-30T12:00:00+00:00"
            )
        }
        assert ids == {"on_end"}
        assert "past_end" not in ids

    def test_bare_end_date_includes_same_day_datetime(self, db: DatabaseManager):
        """A bare YYYY-MM-DD end is extended to end-of-day so same-day rows match."""
        db.upsert_photos_batch([
            _row("evening", date_taken="2023-06-15T22:11:09+00:00"),
        ])
        ids = [r["id"] for r in db.search_by_date("2023-06-15", "2023-06-15")]
        assert "evening" in ids

    def test_results_ordered_desc(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("old", date_taken="2023-01-01T00:00:00+00:00"),
            _row("mid", date_taken="2023-06-01T00:00:00+00:00"),
            _row("new", date_taken="2023-12-01T00:00:00+00:00"),
        ])
        ids = [r["id"] for r in db.search_by_date("2023-01-01", "2023-12-31")]
        assert ids == ["new", "mid", "old"]

    def test_limit_respected(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row(f"p{i}", date_taken=f"2023-06-{i + 1:02d}T00:00:00+00:00")
            for i in range(5)
        ])
        assert len(db.search_by_date("2023-01-01", "2023-12-31", limit=2)) == 2


# ---------------------------------------------------------------------------
# search_by_location (haversine)
# ---------------------------------------------------------------------------

class TestSearchByLocation:
    # Reference point: Singapore, Marina Bay
    LAT, LON = 1.3521, 103.8198

    def test_within_radius_included_out_excluded(self, db: DatabaseManager):
        # ~1.11 km north (0.01 deg latitude ≈ 1.11 km)
        near_lat = self.LAT + 0.01
        # ~55 km north (0.5 deg latitude ≈ 55.6 km)
        far_lat = self.LAT + 0.5
        db.upsert_photos_batch([
            _row("near", latitude=near_lat, longitude=self.LON),
            _row("far", latitude=far_lat, longitude=self.LON),
        ])
        ids = [r["id"] for r in db.search_by_location(self.LAT, self.LON, radius_km=5.0)]
        assert "near" in ids
        assert "far" not in ids

    def test_distance_matches_hand_computed_haversine(self, db: DatabaseManager):
        near_lat = self.LAT + 0.01
        db.upsert_photos_batch([_row("near", latitude=near_lat, longitude=self.LON)])
        results = db.search_by_location(self.LAT, self.LON, radius_km=5.0)
        expected = _haversine(self.LAT, self.LON, near_lat, self.LON)
        assert expected == pytest.approx(1.112, abs=0.01)  # hand-computed ~1.112 km
        assert results[0]["distance_km"] == pytest.approx(round(expected, 3))

    def test_results_sorted_by_distance(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("d3", latitude=self.LAT + 0.03, longitude=self.LON),
            _row("d1", latitude=self.LAT + 0.01, longitude=self.LON),
            _row("d2", latitude=self.LAT + 0.02, longitude=self.LON),
        ])
        results = db.search_by_location(self.LAT, self.LON, radius_km=10.0)
        ids = [r["id"] for r in results]
        assert ids == ["d1", "d2", "d3"]
        dists = [r["distance_km"] for r in results]
        assert dists == sorted(dists)

    def test_exactly_on_reference_is_zero(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("here", latitude=self.LAT, longitude=self.LON)])
        results = db.search_by_location(self.LAT, self.LON, radius_km=1.0)
        assert results[0]["distance_km"] == pytest.approx(0.0, abs=0.001)

    def test_null_location_excluded(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("nogps", latitude=None, longitude=None)])
        results = db.search_by_location(self.LAT, self.LON, radius_km=9999.0)
        assert all(r["id"] != "nogps" for r in results)

    def test_limit_respected(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row(f"p{i}", latitude=self.LAT + i * 0.005, longitude=self.LON)
            for i in range(5)
        ])
        results = db.search_by_location(self.LAT, self.LON, radius_km=100.0, limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# search_by_metadata
# ---------------------------------------------------------------------------

class TestSearchByMetadata:
    def test_camera_model_substring(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("iphone", camera_model="iPhone 15 Pro"),
            _row("canon", camera_model="Canon EOS R5"),
        ])
        ids = [r["id"] for r in db.search_by_metadata(camera_model="iPhone")]
        assert ids == ["iphone"]

    def test_camera_model_case_insensitive(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("iphone", camera_model="iPhone 15 Pro")])
        assert [r["id"] for r in db.search_by_metadata(camera_model="iphone")] == ["iphone"]
        assert [r["id"] for r in db.search_by_metadata(camera_model="IPHONE")] == ["iphone"]

    def test_keyword_filter(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("beach", keywords=json.dumps(["beach", "sunset"])),
            _row("city", keywords=json.dumps(["city", "night"])),
        ])
        ids = [r["id"] for r in db.search_by_metadata(keywords=["beach"])]
        assert ids == ["beach"]

    def test_keyword_case_insensitive(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("beach", keywords=json.dumps(["Beach"]))])
        assert [r["id"] for r in db.search_by_metadata(keywords=["beach"])] == ["beach"]

    def test_persons_filter(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("with_alice", persons=json.dumps(["Alice", "Bob"])),
            _row("with_carol", persons=json.dumps(["Carol"])),
        ])
        ids = [r["id"] for r in db.search_by_metadata(persons=["Alice"])]
        assert ids == ["with_alice"]

    def test_filters_are_and_combined(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row(
                "match",
                camera_model="iPhone 15 Pro",
                keywords=json.dumps(["beach"]),
                persons=json.dumps(["Alice"]),
            ),
            _row(
                "wrong_camera",
                camera_model="Canon EOS R5",
                keywords=json.dumps(["beach"]),
                persons=json.dumps(["Alice"]),
            ),
            _row(
                "wrong_keyword",
                camera_model="iPhone 15 Pro",
                keywords=json.dumps(["city"]),
                persons=json.dumps(["Alice"]),
            ),
            _row(
                "wrong_person",
                camera_model="iPhone 15 Pro",
                keywords=json.dumps(["beach"]),
                persons=json.dumps(["Carol"]),
            ),
        ])
        ids = [
            r["id"]
            for r in db.search_by_metadata(
                keywords=["beach"], camera_model="iPhone", persons=["Alice"]
            )
        ]
        assert ids == ["match"]

    def test_multiple_keywords_all_required(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("both", keywords=json.dumps(["beach", "sunset"])),
            _row("one", keywords=json.dumps(["beach"])),
        ])
        ids = [r["id"] for r in db.search_by_metadata(keywords=["beach", "sunset"])]
        assert ids == ["both"]

    def test_no_filters_raises_value_error(self, db: DatabaseManager):
        with pytest.raises(ValueError):
            db.search_by_metadata()

    def test_empty_lists_and_none_raise_value_error(self, db: DatabaseManager):
        """Empty keyword/person lists produce no clauses → still a ValueError."""
        with pytest.raises(ValueError):
            db.search_by_metadata(keywords=[], persons=[], camera_model=None)


# ---------------------------------------------------------------------------
# search_by_embedding (cosine KNN)
# ---------------------------------------------------------------------------

class TestSearchByEmbedding:
    def test_empty_returns_empty(self, db: DatabaseManager):
        assert db.search_by_embedding([1.0, 0.0, 0.0]) == []

    def test_closest_first(self, db: DatabaseManager):
        _insert_embedding(db, "exact", [1.0, 0.0, 0.0])
        _insert_embedding(db, "close", [0.9, 0.1, 0.0])
        _insert_embedding(db, "far", [0.0, 1.0, 0.0])
        results = db.search_by_embedding([1.0, 0.0, 0.0], limit=3)
        assert [r["id"] for r in results] == ["exact", "close", "far"]

    def test_similarity_scores_monotonic(self, db: DatabaseManager):
        _insert_embedding(db, "exact", [1.0, 0.0, 0.0])
        _insert_embedding(db, "close", [0.9, 0.1, 0.0])
        _insert_embedding(db, "far", [0.0, 1.0, 0.0])
        results = db.search_by_embedding([1.0, 0.0, 0.0], limit=3)
        scores = [r["similarity_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0]["similarity_score"] == pytest.approx(1.0, abs=1e-4)
        assert results[-1]["similarity_score"] == pytest.approx(0.0, abs=1e-4)

    def test_limit_caps_results(self, db: DatabaseManager):
        # Unit vectors at increasing angles from the query axis → strictly
        # decreasing cosine similarity, so the top-2 are unambiguous.
        for i in range(5):
            angle = math.radians(i * 10)
            _insert_embedding(db, f"p{i}", [math.cos(angle), math.sin(angle), 0.0])
        results = db.search_by_embedding([1.0, 0.0, 0.0], limit=2)
        assert len(results) == 2
        # The two returned must be the two most similar (smallest angle).
        assert {r["id"] for r in results} == {"p0", "p1"}


# ---------------------------------------------------------------------------
# find_duplicate_groups (union-find)
# ---------------------------------------------------------------------------

def _unit(angle_deg: float) -> list[float]:
    """2D unit vector at the given angle (z padded to 0)."""
    r = math.radians(angle_deg)
    return [math.cos(r), math.sin(r), 0.0]


class TestFindDuplicateGroups:
    def test_no_embeddings(self, db: DatabaseManager):
        assert db.find_duplicate_groups() == []

    def test_single_embedding(self, db: DatabaseManager):
        _insert_embedding(db, "solo", [1.0, 0.0, 0.0])
        assert db.find_duplicate_groups() == []

    def test_identical_grouped(self, db: DatabaseManager):
        _insert_embedding(db, "a", [1.0, 0.0, 0.0])
        _insert_embedding(db, "b", [1.0, 0.0, 0.0])
        groups = db.find_duplicate_groups(threshold=0.98)
        assert len(groups) == 1
        assert {p["id"] for p in groups[0]} == {"a", "b"}

    def test_dissimilar_not_grouped(self, db: DatabaseManager):
        _insert_embedding(db, "a", [1.0, 0.0, 0.0])
        _insert_embedding(db, "b", [0.0, 1.0, 0.0])
        assert db.find_duplicate_groups(threshold=0.98) == []

    def test_transitive_duplicates_one_group(self, db: DatabaseManager):
        """A~B and B~C but A NOT ~C directly — union-find still merges all three.

        Angles: A=0°, B=18.19°, C=36.38°. With threshold 0.9:
          cos(A,B)=cos(18.19°)=0.95  >= 0.9  (linked)
          cos(B,C)=cos(18.19°)=0.95  >= 0.9  (linked)
          cos(A,C)=cos(36.38°)=0.805 <  0.9  (NOT directly linked)
        """
        phi = math.degrees(math.acos(0.95))  # ~18.19°
        a, b, c = _unit(0.0), _unit(phi), _unit(2 * phi)

        # Sanity-check the geometry the test relies on.
        av, bv, cv = np.array(a), np.array(b), np.array(c)
        assert float(av @ bv) == pytest.approx(0.95, abs=1e-6)
        assert float(bv @ cv) == pytest.approx(0.95, abs=1e-6)
        assert float(av @ cv) < 0.9

        _insert_embedding(db, "A", a)
        _insert_embedding(db, "B", b)
        _insert_embedding(db, "C", c)

        groups = db.find_duplicate_groups(threshold=0.9)
        assert len(groups) == 1
        assert {p["id"] for p in groups[0]} == {"A", "B", "C"}

    def test_two_separate_groups(self, db: DatabaseManager):
        _insert_embedding(db, "a1", [1.0, 0.0, 0.0])
        _insert_embedding(db, "a2", [1.0, 0.0, 0.0])
        _insert_embedding(db, "b1", [0.0, 1.0, 0.0])
        _insert_embedding(db, "b2", [0.0, 1.0, 0.0])
        groups = db.find_duplicate_groups(threshold=0.98)
        assert len(groups) == 2
        clustered = [frozenset(p["id"] for p in g) for g in groups]
        assert frozenset({"a1", "a2"}) in clustered
        assert frozenset({"b1", "b2"}) in clustered

    def test_group_members_have_similarity_scores(self, db: DatabaseManager):
        _insert_embedding(db, "a", [1.0, 0.0, 0.0])
        _insert_embedding(db, "b", [1.0, 0.0, 0.0])
        groups = db.find_duplicate_groups(threshold=0.98)
        for photo in groups[0]:
            assert "similarity_score" in photo


# ---------------------------------------------------------------------------
# quality_score / is_duplicate persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_update_quality_score(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("q")])
        db.update_quality_score("q", 88.5)
        assert db.get_photo("q")["quality_score"] == pytest.approx(88.5)

    def test_update_duplicate_flags(self, db: DatabaseManager):
        db.upsert_photos_batch([_row("d1"), _row("d2")])
        count = db.update_duplicate_flags(["d1", "d2"], is_duplicate=True)
        assert count == 2
        assert db.get_photo("d1")["is_duplicate"] == 1
        assert db.get_photo("d2")["is_duplicate"] == 1
        db.update_duplicate_flags(["d1"], is_duplicate=False)
        assert db.get_photo("d1")["is_duplicate"] == 0

    def test_embedded_photo_ids(self, db: DatabaseManager):
        _insert_embedding(db, "e1", [1.0, 0.0, 0.0])
        _insert_embedding(db, "e2", [0.0, 1.0, 0.0])
        db.upsert_photos_batch([_row("no_embed")])
        assert db.embedded_photo_ids() == {"e1", "e2"}

    def test_photos_with_paths_excludes_null(self, db: DatabaseManager):
        db.upsert_photos_batch([
            _row("has", filepath="/x.jpg"),
            _row("none", filepath=None),
        ])
        ids = [r["id"] for r in db.photos_with_paths()]
        assert "has" in ids
        assert "none" not in ids
