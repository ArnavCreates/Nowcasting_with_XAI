# =============================================================================
# Indra — single-container production image
#
# Two stages: Node builds the vendored frontend, Python runs the service and
# serves the built assets from the same origin. One container, no CORS, no
# separate web server.
#
#   docker build -t indra:latest .
#   docker run -p 8000:8000 \
#     -v $(pwd)/checkpoints:/app/checkpoints:ro \
#     -v $(pwd)/models:/app/models:ro \
#     -v $(pwd)/vectorstore:/app/vectorstore \
#     indra:latest
#
# The image starts without any of those mounts. Every external artefact is
# reported as unavailable on /healthz rather than aborting the boot, so the
# container is inspectable before the weights arrive.
#
# For development with hot reload, use docker-compose.yml instead: it runs the
# Vite dev server separately and bind-mounts the source.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — build the frontend
# -----------------------------------------------------------------------------
FROM node:20-bookworm-slim AS frontend-build

WORKDIR /build

# package.json alone, so a source-only change does not re-resolve the whole
# dependency tree on every rebuild.
#
# npm install, not npm ci: the vendored frontend carries no package-lock.json.
# That means this stage is not reproducible in the way requirements.txt is --
# a caret range can resolve differently between builds. Committing a lockfile
# upstream is the fix; pinning here would only hide the drift.
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2 — the service
# -----------------------------------------------------------------------------
# Both stages pin the Debian release, not just the language version. The
# system libraries below are versioned into their SONAME -- libgdal32 is GDAL
# 3.6, libhdf5-103-1 is HDF5 1.10 -- so the package names are only valid for
# one Debian release. An unqualified python:3.11-slim follows Debian stable,
# and the day trixie was promoted the same Dockerfile stopped resolving
# libgdal32 and libhdf5-103-1 with no change on our side.
FROM python:3.11-slim-bookworm AS runtime

# CPU wheels by default: roughly 2 GB smaller than the CUDA build, and what a
# laptop demo wants. Override for a GPU deployment:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    # Keep the sentence-transformer cache inside a mountable directory rather
    # than in a home directory that vanishes with the container.
    HF_HOME=/app/models/.huggingface

# System libraries the geospatial and meteorological stack links against.
# The list is the one documented at the top of requirements.txt; eccodes in
# particular has no wheel and cfgrib will not import without it.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libeccodes0 \
        libgdal32 \
        libproj25 \
        libgeos-c1v5 \
        libhdf5-103-1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so editing a module does not reinstall torch.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt --extra-index-url "${TORCH_INDEX_URL}"

COPY src/ ./src/
COPY configs/ ./configs/

# The built frontend lands where indra.api.state.FRONTEND_DIST looks for it:
# REPO_ROOT/frontend/dist, and REPO_ROOT resolves to /app from
# /app/src/indra/config.py. Present, so main.py mounts it at / and the
# single-container deployment serves the UI and the API from one origin.
COPY --from=frontend-build /build/dist ./frontend/dist

# Artefact mount points, created so a run without volumes finds directories
# rather than failing on a missing path. Empty is a state the service reports;
# absent is one it would have to guess about.
RUN mkdir -p /app/checkpoints /app/models /app/vectorstore /app/data \
             /app/replay_buffer \
    && useradd --create-home --uid 10001 indra \
    && chown -R indra:indra /app

USER indra

EXPOSE 8000

# /healthz answers even when nothing is loaded, which is exactly what makes it
# usable as a liveness probe here: it reports degraded rather than refusing to
# respond.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "indra.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
