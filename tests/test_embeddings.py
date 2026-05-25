"""Tests for embeddings pipeline and vector search — no real CLIP model needed."""
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from photomind.embeddings import EMBEDDING_DIM, CLIPEmbedder, serialize
from photomind.database import DatabaseManager


# ---------------------------------------------------------------------------
# serialize()
# ---------------------------------------------------------------------------

def test_serialize_length():
    v = [0.1] * EMBEDDING_DIM
    blob = serialize(v)
    assert len(blob) == EMBEDDING_DIM * 4  # 4 bytes per float32


def test_serialize_roundtrip():
    v = [float(i) / EMBEDDING_DIM for i in range(EMBEDDING_DIM)]
    blob = serialize(v)
    unpacked = list(struct.unpack(f"{EMBEDDING_DIM}f", blob))
    assert len(unpacked) == EMBEDDING_DIM
    assert unpacked[0] == pytest.approx(v[0], abs=1e-6)


# ---------------------------------------------------------------------------
# CLIPEmbedder (mocked — no model download)
# ---------------------------------------------------------------------------

def _fake_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    import math
    v = [float(i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


class TestCLIPEmbedderMocked:
    @pytest.fixture
    def embedder(self) -> CLIPEmbedder:
        e = CLIPEmbedder()
        # Inject a fake loaded state so _load() is skipped
        e._model = MagicMock()
        e._preprocess = MagicMock()
        e._tokenizer = MagicMock()
        e._device = "cpu"
        return e

    def test_encode_text_returns_list(self, embedder: CLIPEmbedder):
        import torch
        fake_feat = torch.zeros(1, EMBEDDING_DIM)
        fake_feat[0, 0] = 1.0
        embedder._model.encode_text.return_value = fake_feat
        embedder._tokenizer.return_value = torch.zeros(1, 77, dtype=torch.long)

        result = embedder.encode_text("sunset at the beach")
        assert isinstance(result, list)
        assert len(result) == EMBEDDING_DIM

    def test_encode_image_returns_none_on_missing_file(self, embedder: CLIPEmbedder):
        result = embedder.encode_image("/nonexistent/path/image.jpg")
        assert result is None

    def test_encode_image_returns_list_on_success(self, embedder: CLIPEmbedder):
        import torch
        from unittest.mock import patch as _patch

        fake_feat = torch.zeros(1, EMBEDDING_DIM)
        fake_feat[0, 0] = 1.0
        embedder._model.encode_image.return_value = fake_feat

        fake_img = MagicMock()
        fake_tensor = torch.zeros(3, 224, 224)
        embedder._preprocess.return_value = fake_tensor

        with _patch("PIL.Image") as mock_image_mod:
            mock_image_mod.open.return_value.convert.return_value = fake_img
            result = embedder.encode_image("/fake/image.jpg")

        assert isinstance(result, list)
        assert len(result) == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Database embedding methods (requires sqlite-vec)
# ---------------------------------------------------------------------------

@pytest.fixture
def db_with_vec(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / "test_vec.db")
    manager.connect()
    yield manager
    manager.close()


def _photo_row(uuid: str = "abc-123") -> dict:
    import json
    return {
        "id": uuid,
        "filename": "test.heic",
        "filepath": "/fake/test.heic",
        "date_taken": "2024-01-01T00:00:00+00:00",
        "latitude": None,
        "longitude": None,
        "location_name": None,
        "camera_make": "Apple",
        "camera_model": "iPhone 14 Pro",
        "keywords": json.dumps([]),
        "albums": json.dumps([]),
        "persons": json.dumps([]),
        "is_duplicate": 0,
        "quality_score": None,
    }


class TestDatabaseEmbeddings:
    def test_upsert_and_search(self, db_with_vec: DatabaseManager):
        db_with_vec.upsert_photos_batch([_photo_row("p1"), _photo_row("p2")])
        db_with_vec.conn.commit()

        emb1 = _fake_embedding()
        emb2 = [x * -1 for x in emb1]  # opposite direction

        with db_with_vec.conn:
            db_with_vec.upsert_embedding("p1", emb1)
            db_with_vec.upsert_embedding("p2", emb2)

        # Searching with emb1 should return p1 first (distance ~0)
        results = db_with_vec.search_by_embedding(emb1, limit=2)
        assert len(results) >= 1
        assert results[0]["id"] == "p1"
        assert "similarity_score" in results[0]

    def test_upsert_is_idempotent(self, db_with_vec: DatabaseManager):
        db_with_vec.upsert_photos_batch([_photo_row("p1")])
        db_with_vec.conn.commit()

        emb = _fake_embedding()
        with db_with_vec.conn:
            db_with_vec.upsert_embedding("p1", emb)
            db_with_vec.upsert_embedding("p1", emb)  # second upsert — should not error

        results = db_with_vec.search_by_embedding(emb, limit=1)
        assert results[0]["id"] == "p1"

    def test_embedded_photo_ids(self, db_with_vec: DatabaseManager):
        db_with_vec.upsert_photos_batch([_photo_row("p1"), _photo_row("p2")])
        db_with_vec.conn.commit()

        assert db_with_vec.embedded_photo_ids() == set()

        with db_with_vec.conn:
            db_with_vec.upsert_embedding("p1", _fake_embedding())

        assert db_with_vec.embedded_photo_ids() == {"p1"}

    def test_search_returns_empty_when_no_embeddings(self, db_with_vec: DatabaseManager):
        results = db_with_vec.search_by_embedding(_fake_embedding(), limit=5)
        assert results == []
