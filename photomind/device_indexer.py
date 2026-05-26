"""iPhone USB device indexer — mounts via ifuse, copies DCIM, delegates to DirectoryIndexer."""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif", ".webp"}
)


def _check_tool(name: str) -> bool:
    """Return True if *name* is available on PATH."""
    return shutil.which(name) is not None


def _list_devices() -> list[str]:
    """Return list of connected device UDIDs via idevice_id."""
    result = subprocess.run(
        ["idevice_id", "-l"], capture_output=True, text=True
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _get_device_name(udid: str) -> str:
    """Return the device's display name via ideviceinfo."""
    result = subprocess.run(
        ["ideviceinfo", "-u", udid, "-k", "DeviceName"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() or "Unknown Device"
