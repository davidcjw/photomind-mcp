"""iPhone USB device indexer — mounts via ifuse, copies DCIM, delegates to DirectoryIndexer."""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from photomind.directory_indexer import DirectoryIndexer

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


class DeviceIndexer:
    """Copy photos from a USB-connected iPhone and index them via DirectoryIndexer."""

    def __init__(self, db: Any, embedder: Any = None) -> None:
        self._db = db
        self._embedder = embedder

    def sync(self, destination: str, device_id: str | None = None) -> dict[str, Any]:
        """Mount iPhone via ifuse, copy DCIM photos to *destination*, then index.

        Returns a summary dict with device info, copy counts, and index stats.
        Raises RuntimeError for missing tools, no device, or mount failure.
        """
        for tool in ("idevice_id", "ideviceinfo", "ifuse"):
            if not _check_tool(tool):
                raise RuntimeError(
                    f"'{tool}' not found. Install with: "
                    "brew install libimobiledevice ifuse"
                )

        devices = _list_devices()
        if not devices:
            raise RuntimeError(
                "No iPhone detected. Connect via USB and accept "
                "'Trust This Computer?' on the device."
            )

        if device_id is not None and device_id not in devices:
            raise RuntimeError(
                f"Device '{device_id}' not found. Connected: {devices}"
            )
        udid = device_id or devices[0]
        device_name = _get_device_name(udid)
        logger.info("Syncing from device: %s (%s)", device_name, udid)

        dest = Path(destination).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)

        mountpoint = tempfile.mkdtemp(prefix="photomind-device-")
        try:
            mount_result = subprocess.run(
                ["ifuse", "-u", udid, mountpoint],
                capture_output=True, text=True,
            )
            if mount_result.returncode != 0:
                raise RuntimeError(f"ifuse mount failed: {mount_result.stderr.strip()}")

            copied, skipped, copy_errors = self._copy_dcim(
                Path(mountpoint) / "DCIM", dest
            )
        finally:
            subprocess.run(["umount", mountpoint], capture_output=True)
            try:
                Path(mountpoint).rmdir()
            except OSError:
                pass

        index_result = DirectoryIndexer(self._db, embedder=self._embedder).sync(str(dest))

        return {
            "device": device_name,
            "udid": udid,
            "destination": str(dest),
            "copied": copied,
            "skipped_existing": skipped,
            "copy_errors": copy_errors,
            **{f"index_{k}": v for k, v in index_result.items() if k != "directory"},
        }

    def _copy_dcim(self, dcim: Path, dest: Path) -> tuple[int, int, int]:
        """Copy supported photo files from *dcim* to *dest*, preserving subfolder structure.

        Returns (copied, skipped_existing, copy_errors).
        """
        copied = skipped = copy_errors = 0

        for src_file in dcim.rglob("*"):
            if not src_file.is_file():
                continue
            if src_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            rel = src_file.relative_to(dcim)
            dest_file = dest / rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            if dest_file.exists():
                skipped += 1
                continue

            try:
                shutil.copy2(str(src_file), str(dest_file))
                copied += 1
            except Exception as exc:
                logger.warning("Failed to copy %s: %s", src_file.name, exc)
                copy_errors += 1

        return copied, skipped, copy_errors
