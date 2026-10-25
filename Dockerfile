FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS base

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        git \
        intel-opencl-icd \
        libigdgmm12 \
        libasound2 \
        libasound2-plugins \
        libmpv2 \
        libportaudio2 \
        libsndfile1 \
        libze1 \
        pipewire-alsa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/hoast
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

FROM base

ENV PATH=/opt/hoast/.venv/bin:$PATH
ENV PYTHONPATH=/workspace/src
ENV HOME=/workspace/src
ENV UV_CACHE_DIR=/workspace/src/.cache/uv
WORKDIR /workspace/src
USER 1000:1000
ENTRYPOINT ["/opt/hoast/.venv/bin/python"]
CMD ["-m", "hoast", "--config", "/workspace/src/config.toml"]
