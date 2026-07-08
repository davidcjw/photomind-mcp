"""Tests for photomind.quality.compute_sharpness — pure PIL/numpy, no macOS deps."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from photomind.quality import compute_sharpness


# ---------------------------------------------------------------------------
# Fixtures — synthetic images generated with PIL (no real photos needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def sharp_image(tmp_path: Path) -> Path:
    """A high-contrast random-noise image — lots of edges, high sharpness."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    path = tmp_path / "sharp.png"
    Image.fromarray(arr, mode="L").save(path)
    return path


@pytest.fixture
def blurred_image(tmp_path: Path, sharp_image: Path) -> Path:
    """A heavily blurred version of the sharp image — fewer edges."""
    img = Image.open(sharp_image).filter(ImageFilter.GaussianBlur(radius=8))
    path = tmp_path / "blurred.png"
    img.save(path)
    return path


@pytest.fixture
def flat_image(tmp_path: Path) -> Path:
    """A solid-color image — no edges at all."""
    path = tmp_path / "flat.png"
    Image.new("L", (256, 256), color=128).save(path)
    return path


# ---------------------------------------------------------------------------
# Sharpness ordering
# ---------------------------------------------------------------------------

def test_sharp_scores_higher_than_blurred(sharp_image: Path, blurred_image: Path):
    sharp_score = compute_sharpness(sharp_image)
    blurred_score = compute_sharpness(blurred_image)
    assert sharp_score is not None
    assert blurred_score is not None
    assert sharp_score > blurred_score


def test_flat_image_sharpness_very_low(flat_image: Path, sharp_image: Path):
    """A solid-color image has essentially no edges, so its score is a tiny
    fraction of a textured image's. (It is not exactly zero: FIND_EDGES leaves
    a small, constant border-artifact floor regardless of image content.)"""
    flat_score = compute_sharpness(flat_image)
    sharp_score = compute_sharpness(sharp_image)
    assert flat_score is not None
    assert sharp_score is not None
    assert flat_score < sharp_score / 50


# ---------------------------------------------------------------------------
# None / invalid inputs return None (never raise)
# ---------------------------------------------------------------------------

def test_none_input_returns_none():
    assert compute_sharpness(None) is None


def test_empty_string_returns_none():
    assert compute_sharpness("") is None


def test_nonexistent_file_returns_none(tmp_path: Path):
    assert compute_sharpness(tmp_path / "does_not_exist.jpg") is None


def test_corrupt_image_returns_none(tmp_path: Path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(np.random.default_rng(0).integers(0, 256, size=1024, dtype=np.uint8).tobytes())
    assert compute_sharpness(corrupt) is None
