"""Test the image patch and scoped raw-response cache bypass."""

import ast
import asyncio
import contextvars
import hashlib
from typing import Any, cast

import pytest

from music import apply_spotify_daylist_patch as patch

DAYLIST = "playlists/37i9dQZF1EP6YuccBxUcC1/items"
BYPASS = contextvars.ContextVar("bypass", default=False)
SOURCE = '''class Provider:
    async def _get_data_with_caching(
        self, endpoint: str, cache_checksum: str | None, **kwargs: Any
    ) -> dict[str, Any]:
        """Get data from api with caching."""
        cache_key_parts = [endpoint]
        for key in sorted(kwargs.keys()):
            cache_key_parts.append(f"{key}{kwargs[key]}")
        cache_key = ".".join(map(str, cache_key_parts))
''' + patch.OLD + '''            return cast("dict[str, Any]", cached)
        result = await self._get_data(endpoint, **kwargs)
        await self.mass.cache.set(
            cache_key, result, provider=self.instance_id, checksum=cache_checksum
        )
        return result
'''


class Cache:
    """Small cache implementing MA's explicit allow_bypass contract."""

    values: dict[tuple[str, str, str | None], dict[str, Any]]
    allow_bypass: list[bool]

    def __init__(self) -> None:
        """Initialize an empty cache and call history."""
        self.values = {}
        self.allow_bypass = []

    async def get(
        self, key: str, *, provider: str, checksum: str | None, allow_bypass: bool
    ) -> dict[str, Any] | None:
        """Return a matching entry unless the refresh context bypasses it."""
        self.allow_bypass.append(allow_bypass)
        if allow_bypass and BYPASS.get():
            return None
        return self.values.get((key, provider, checksum))

    async def set(
        self, key: str, value: dict[str, Any], *, provider: str, checksum: str | None
    ) -> None:
        """Store a response by key, provider, and checksum."""
        self.values[key, provider, checksum] = value


def test_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject changed revisions, missing matches, duplicates, and reapplication."""
    with pytest.raises(RuntimeError, match="revision"):
        patch.patch_source(SOURCE)
    for source in (SOURCE.replace(patch.OLD, ""), SOURCE + patch.OLD):
        monkeypatch.setattr(patch, "EXPECTED_SHA256", hashlib.sha256(source.encode()).hexdigest())
        with pytest.raises(RuntimeError, match="unique"):
            patch.patch_source(source)
    monkeypatch.setattr(patch, "EXPECTED_SHA256", hashlib.sha256(SOURCE.encode()).hexdigest())
    patched = patch.patch_source(SOURCE)
    assert patched == SOURCE.replace(patch.OLD, patch.NEW, 1)
    with pytest.raises(RuntimeError, match="revision"):
        patch.patch_source(patched)


def test_cache_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the patched upstream method with normal and refresh contexts."""
    monkeypatch.setattr(patch, "EXPECTED_SHA256", hashlib.sha256(SOURCE.encode()).hexdigest())
    module = ast.parse(patch.patch_source(SOURCE))
    namespace: dict[str, Any] = {"Any": Any, "cast": cast}
    exec(compile(module, "fixture", "exec"), namespace)  # noqa: S102 - trusted fixture
    provider = namespace["Provider"]()
    cache = Cache()
    provider.mass = type("Mass", (), {"cache": cache})()
    provider.instance_id = "spotify-instance"
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fetch(endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Return a distinct response for each simulated API request."""
        calls.append((endpoint, kwargs))
        return {"generation": len(calls)}

    provider._get_data = fetch

    async def exercise() -> None:
        """Verify hits, misses, refreshes, isolation, and cache writes."""
        first = await provider._get_data_with_caching(DAYLIST, "etag", limit=50, offset=0)
        assert await provider._get_data_with_caching(DAYLIST, "etag", limit=50, offset=0) == first
        token = BYPASS.set(True)
        try:
            refreshed = await provider._get_data_with_caching(DAYLIST, "etag", limit=50, offset=0)
            assert refreshed != first
            for endpoint in (
                DAYLIST + "?x=1", DAYLIST + "/", "playlists/other/items",
                "playlists/37i9dQZF1EP6YuccBxUcC1", "albums/other/tracks",
            ):
                value = await provider._get_data_with_caching(endpoint, "etag", limit=50, offset=0)
                assert await provider._get_data_with_caching(endpoint, "etag", limit=50, offset=0) == value
            assert cache.allow_bypass == [True, True, True] + [False] * 10
            page = await provider._get_data_with_caching(DAYLIST, "etag", limit=50, offset=50)
            refreshed_page = await provider._get_data_with_caching(
                DAYLIST, "etag", limit=50, offset=50
            )
            assert refreshed_page != page
        finally:
            BYPASS.reset(token)
        assert await provider._get_data_with_caching(DAYLIST, "etag", offset=0, limit=50) == refreshed
        assert await provider._get_data_with_caching(DAYLIST, "other-etag", limit=50, offset=0) != refreshed
        assert await provider._get_data_with_caching(DAYLIST, "etag", limit=50, offset=50) == refreshed_page

    asyncio.run(exercise())
