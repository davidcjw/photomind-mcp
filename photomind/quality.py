"""Image quality assessment utilities."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pillow_heif  # noqa: F401
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def compute_sharpness(filepath: str | Path | None) -> float | None:
    """Return a sharpness score for the image at filepath.

    Uses variance of edge-filtered pixels (PIL FIND_EDGES). Higher = sharper.
    Returns None if filepath is falsy or the image cannot be opened.
    """
    if not filepath:
        return None
    try:
        from PIL import Image, ImageFilter

        img = Image.open(filepath).convert("L").resize((512, 512))
        edges = img.filter(ImageFilter.FIND_EDGES)
        arr = np.array(edges, dtype=np.float32)
        return float(arr.var())
    except Exception as exc:
        logger.debug("Sharpness check failed for %s: %s", filepath, exc)
        return None
