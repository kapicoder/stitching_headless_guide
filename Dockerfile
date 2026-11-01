# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm AS wheel-builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.runtime.txt ./

# GDAL's Python extension must be compiled against the exact Debian runtime ABI.
RUN python -m pip wheel --wheel-dir /wheels -r requirements.runtime.txt \
    && python -m pip install --no-index /wheels/numpy-*.whl \
    && GDAL_VERSION="$(gdal-config --version)" \
    && python -m pip wheel --wheel-dir /wheels "GDAL==${GDAL_VERSION}"


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=0 \
    PYTHONHASHSEED=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/opt/numba-cache \
    NUMBA_CPU_NAME=generic \
    OPENCV_LOG_LEVEL=ERROR \
    GDAL_CACHEMAX=512 \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgdal32 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid app --create-home --shell /usr/sbin/nologin app

COPY --from=wheel-builder /wheels /wheels
COPY requirements.runtime.txt /tmp/requirements.runtime.txt
RUN python -m pip install --no-index /wheels/gdal-*.whl \
    && python -m pip install --no-index --find-links=/wheels -r /tmp/requirements.runtime.txt \
    && rm -rf /wheels /tmp/requirements.runtime.txt

WORKDIR /workspace
COPY docker/precompile.py /tmp/precompile.py

# Application source is not copied; Compose mounts it at /workspace.
RUN mkdir -p /opt/numba-cache \
    && python /tmp/precompile.py \
    && python -m compileall -q -j 2 \
        /usr/local/lib/python3.11/site-packages/stitching \
        /usr/local/lib/python3.11/site-packages/largestinteriorrectangle \
        /usr/local/lib/python3.11/site-packages/flask \
    && rm -f /tmp/precompile.py \
    && chown -R app:app /opt/numba-cache

USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read()"

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8000", "--workers=1", "--threads=4", "--timeout=0", "--access-logfile=-", "--error-logfile=-", "stitching_headless_api:app"]
