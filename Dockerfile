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

FROM base AS kernel-build

RUN apt-get update \
    && apt-get install --yes --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY hoast /opt/hoast/build-src/hoast
COPY tools/build_speech_kernels.py /opt/hoast/build-src/tools/build_speech_kernels.py
WORKDIR /opt/hoast/build-src
RUN PYTHONPATH=/opt/hoast/build-src /opt/hoast/.venv/bin/python \
    -m tools.build_speech_kernels \
    --output /opt/hoast/kernels/libhoast_speech.so

FROM base

COPY --from=kernel-build /opt/hoast/kernels /opt/hoast/kernels
ENV PATH=/opt/hoast/.venv/bin:$PATH
ENV PYTHONPATH=/workspace/src
ENV HOME=/workspace/src
ENV UV_CACHE_DIR=/workspace/src/.cache/uv
ENV HOAST_SPEECH_KERNEL=/opt/hoast/kernels/libhoast_speech.so
WORKDIR /workspace/src
USER 1000:1000
ENTRYPOINT ["/opt/hoast/.venv/bin/python"]
CMD ["-m", "hoast", "--config", "/workspace/src/config.toml"]
