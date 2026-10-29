"""Guard and patch Spotify's raw-response cache for generic daylist refreshes."""

import hashlib
from pathlib import Path

PROVIDER_PATH = Path(
    "/app/venv/lib/python3.14/site-packages/"
    "music_assistant/providers/spotify/provider.py"
)
EXPECTED_SHA256 = "ab5fa09a7ea949ffd29de4246f6a5f7f744a2e38be2f11b40db5c3ddee2490dd"
OLD = """        if cached := await self.mass.cache.get(
            cache_key, provider=self.instance_id, checksum=cache_checksum, allow_bypass=False
        ):
"""
NEW = """        if cached := await self.mass.cache.get(
            cache_key,
            provider=self.instance_id,
            checksum=cache_checksum,
            allow_bypass=endpoint == "playlists/37i9dQZF1EP6YuccBxUcC1/items",
        ):
"""


def patch_source(source: str) -> str:
    """Verify the reviewed provider revision and enable daylist cache bypass.

    Args:
        source:
            Complete upstream Spotify provider source.

    Returns:
        Provider source with exactly one targeted cache call changed.

    Raises:
        RuntimeError: If the upstream revision or expected cache call differs.
    """
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            "Unsupported Music Assistant Spotify provider.py revision: "
            f"expected {EXPECTED_SHA256}, got {actual}"
        )
    if source.count(OLD) != 1:
        raise RuntimeError("Spotify cache call is not a unique source match")
    return source.replace(OLD, NEW, 1)


def main() -> None:
    """Install the guarded source patch in the image."""
    source = PROVIDER_PATH.read_text(encoding="utf-8")
    PROVIDER_PATH.write_text(patch_source(source), encoding="utf-8")


if __name__ == "__main__":
    main()
