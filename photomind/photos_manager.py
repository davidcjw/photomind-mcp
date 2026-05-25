"""Photos.app management operations via AppleScript."""
from __future__ import annotations

import subprocess
from typing import Any


def delete_from_photos(photo_ids: list[str], timeout: int = 30) -> dict[str, Any]:
    """Move photos to Recently Deleted in Photos.app using AppleScript.

    Photos are NOT permanently deleted — they sit in Recently Deleted for
    30 days and can be recovered from there.

    Args:
        photo_ids: Photos.app UUIDs to delete.
        timeout:   osascript timeout in seconds.

    Returns:
        {"deleted": int, "errors": list[str]}

    Raises:
        RuntimeError: if AppleScript execution fails.
    """
    if not photo_ids:
        return {"deleted": 0, "errors": []}

    # Build a comma-separated quoted UUID list for AppleScript
    uuid_list = ", ".join(f'"{uid}"' for uid in photo_ids)

    script = f"""\
tell application "Photos"
    set uuids to {{{uuid_list}}}
    set itemsToDelete to {{}}
    repeat with aUUID in uuids
        try
            set matched to (every media item whose id contains aUUID)
            if (count of matched) > 0 then
                set itemsToDelete to itemsToDelete & matched
            end if
        end try
    end repeat
    if (count of itemsToDelete) > 0 then
        delete itemsToDelete
    end if
    return count of itemsToDelete
end tell
"""

    result = subprocess.run(
        ["osascript"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"AppleScript error: {result.stderr.strip() or 'unknown error'}"
        )

    stdout = result.stdout.strip()
    deleted_count = int(stdout) if stdout.isdigit() else 0
    return {"deleted": deleted_count, "errors": []}
