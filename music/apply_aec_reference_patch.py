"""Install a reviewed native-AirPlay unified diff by unique content matching.

Unified-diff line numbers are informational. Each hunk's old-side block must
occur exactly once in the current source before it is replaced.
"""

import hashlib
from pathlib import Path

SESSION_PATH = Path(
    "/app/venv/lib/python3.14/site-packages/"
    "music_assistant/providers/airplay/stream_session.py"
)
HELPER_SOURCE = Path("/tmp/aec-reference/aec_reference.py")
PATCH_SOURCE = Path("/tmp/aec-reference/aec_reference.patch")
EXPECTED_SHA256 = "09eb3a9662309a5868c3dc550311ba9bac0fd74aef9f293896682c455e798ecc"


def apply_patch(source: str, patch: str) -> str:
    """Apply a single-file unified patch by replacing unique old-side blocks.

    Args:
        source:
            Complete verified upstream AirPlay session source.

        patch:
            Unified diff containing only stream_session.py changes.

    Returns:
        Patched AirPlay session source.

    Raises:
        RuntimeError: If the patch format is invalid or a hunk is not unique.

    """
    lines = patch.splitlines(keepends=True)
    if len(lines) < 3 or lines[:2] != ["--- stream_session.py\n", "+++ stream_session.py\n"]:
        raise RuntimeError("AEC patch must target only stream_session.py")
    result = source
    index = 2
    hunk_number = 0
    while index < len(lines):
        if not lines[index].startswith("@@ "):
            raise RuntimeError("Invalid AEC patch hunk header")
        index += 1
        hunk_number += 1
        hunk: list[str] = []
        while index < len(lines) and not lines[index].startswith("@@ "):
            hunk.append(lines[index])
            index += 1
        if any(not line.startswith((" ", "+", "-")) for line in hunk):
            raise RuntimeError(f"Invalid AEC patch hunk contents: {hunk_number}")
        old = "".join(line[1:] for line in hunk if not line.startswith("+"))
        new = "".join(line[1:] for line in hunk if not line.startswith("-"))
        if result.count(old) != 1:
            raise RuntimeError(f"AEC patch hunk is not a unique source match: {hunk_number}")
        result = result.replace(old, new, 1)
    return result


def main() -> None:
    """Verify the upstream revision and install the native AirPlay hooks."""
    source = SESSION_PATH.read_text(encoding="utf-8")
    actual_sha256 = hashlib.sha256(source.encode()).hexdigest()
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            "Unsupported Music Assistant AirPlay stream_session.py revision: "
            f"expected {EXPECTED_SHA256}, got {actual_sha256}"
        )
    source = apply_patch(source, PATCH_SOURCE.read_text(encoding="utf-8"))
    SESSION_PATH.with_name("aec_reference.py").write_text(
        HELPER_SOURCE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    SESSION_PATH.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
