"""CLIP vision-language embeddings pipeline."""
import logging
import struct

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC support optional; JPEG/PNG still work

logger = logging.getLogger(__name__)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBEDDING_DIM = 512


class CLIPEmbedder:
    """Lazy-loaded CLIP model. Weights download on first use (~350 MB).

    Uses MPS on Apple Silicon, CUDA if available, otherwise CPU.
    """

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch

        self._device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        logger.info("Loading CLIP model %s on %s…", MODEL_NAME, self._device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self._model = model.to(self._device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        logger.info("CLIP model ready.")

    def encode_image(self, image_path: str) -> list[float] | None:
        """Return 512-dim L2-normalised embedding for an image, or None on failure."""
        self._load()
        import torch
        from PIL import Image

        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self._preprocess(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.squeeze().cpu().tolist()
        except Exception as exc:
            logger.warning("encode_image failed for %s: %s", image_path, exc)
            return None

    def encode_text(self, query: str) -> list[float]:
        """Return 512-dim L2-normalised embedding for a text query."""
        self._load()
        import torch

        tokens = self._tokenizer([query]).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze().cpu().tolist()


def serialize(embedding: list[float]) -> bytes:
    """Pack a float list into a binary blob for sqlite-vec MATCH queries."""
    return struct.pack(f"{len(embedding)}f", *embedding)
