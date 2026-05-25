"""Centralised configuration and constants."""
from pathlib import Path

APP_SUPPORT = Path.home() / "Library" / "Application Support" / "photomind-mcp"
DB_PATH: Path = APP_SUPPORT / "photomind.db"

SYNC_BATCH_SIZE: int = 500
MAX_SYNC_ERRORS: int = 50

DEFAULT_SEARCH_LIMIT: int = 50
DEFAULT_RADIUS_KM: float = 5.0
