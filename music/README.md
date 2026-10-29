# Custom Music Assistant image

The image layers the AirPlay AEC bridge and a Spotify daylist cache patch onto
Music Assistant. The Dockerfile uses the upstream `latest` tag. The patch
installers verify the exact upstream source SHA-256 and fail the build if the
source has changed. To upgrade, review the new upstream code and cache semantics,
update the source guards, adapt patches as needed, and rerun their tests and the
image build. Do not simply update a hash without reviewing the changes. Pin the
base image by digest if reproducible builds are required.

The Spotify patch sets `allow_bypass=True` only for raw responses whose endpoint
is exactly `playlists/37i9dQZF1EP6YuccBxUcC1/items`. In an MA refresh context
where `BYPASS_CACHE` is true, cached pages are missed, fetched again, and
written back to the usual cache keys. Outside that context, they retain normal
cache behavior. All other endpoints retain `allow_bypass=False`. The provider
fetches playlist pagination metadata on page zero and reuses an in-memory
snapshot for later pages; this patch does not change that behavior or bypass
other caching layers.

Build without starting or restarting a service:

```sh
docker build -t home-assist/music-assistant-aec:reviewed music
.venv/bin/pytest -q tests/test_spotify_daylist_patch.py
```

Building does not deploy the image. Review and explicitly deploy separately.
