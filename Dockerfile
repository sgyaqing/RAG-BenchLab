# No `# syntax=docker/dockerfile:1` on purpose. That directive makes BuildKit
# fetch the dockerfile frontend image before it can read a single instruction,
# and behind a registry mirror that answers 401 for it the build never starts —
# which is exactly how the first amd64 attempt died, on line 1 of a file whose
# contents were fine. Nothing here needs a frontend newer than the built-in one.

# --------------------------------------------------------------------------
# Stage 1 — build the SPA.
#
# Node and npm stay in this stage; the runtime image receives the compiled
# bundle and nothing else. package.json and the lock file are copied on their
# own first, so the dependency layer survives every edit to the source.
# --------------------------------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /build
ARG NPM_REGISTRY=https://registry.npmjs.org
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --registry="$NPM_REGISTRY"
COPY frontend/ ./
RUN npm run build

# --------------------------------------------------------------------------
# Stage 2 — runtime.
# --------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# A slower network can point at a mirror without editing this file.
ARG PIP_INDEX=https://pypi.org/simple

WORKDIR /app

# Dependency layer first, so it only rebuilds when requirements.txt changes.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --index-url "$PIP_INDEX" -r /app/backend/requirements.txt

# The application and nothing else. `COPY backend/` would also take the test
# suite, pytest.ini, and whatever the developer's checkout happens to carry —
# .DS_Store and .pytest_cache both made it into a release that way. What the
# suite needs is added by Dockerfile.test, which builds on this image.
# ragas counts document tokens while it builds the transform pipeline, and
# tiktoken fetches its BPE vocabulary the first time that happens — a
# synchronous request to a CDN. On a slow path to it the download held the
# event loop for a measured 30 s, which was long enough that every LLM call in
# flight read-timeouted and the whole run died. Both encodings ragas names are
# baked in, so at runtime the lookup is a file read and never a download.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
# Belt and braces for the opt-out in app/core/telemetry.py. That one only takes
# effect in a process that imports our app; any other process in this image
# would report usage, and on a network that blackholes the lookup that costs
# 10 s of stall per call — the exact freeze this project spent a day chasing.
ENV RAGAS_DO_NOT_TRACK=true
RUN python -c "import tiktoken; [tiktoken.get_encoding(n) for n in ('cl100k_base', 'o200k_base')]"

COPY backend/app /app/backend/app
COPY --from=frontend /build/dist /app/frontend/dist

# The licence and the attribution notice travel inside the image, because for
# most customers the image *is* the distribution: they pull it and never see the
# repository. The prompts under backend/app/assets/testset_prompts are derived
# from ragas, so whoever receives them receives ragas' material too, and the
# Apache License asks for a copy of the licence to come with it (section 4).
# NOTICE says what was taken and what was changed.
COPY LICENSE NOTICE /app/

# core/config.py resolves BASE_DIR as parents[3] of its own file, which lands
# on /app — so this layout is what makes data/, logs/ and frontend/dist come
# out right without a single environment variable.
#
# data/ and logs/ have to exist here and be owned by the runtime user: a named
# volume is initialised from the image directory and keeps its ownership, so
# without this the app would start unable to write its own database. (A bind
# mount is not given that treatment and keeps whatever the host has.)
RUN useradd --create-home --uid 10001 rbl \
 && mkdir -p /app/data /app/logs \
 && chown -R rbl:rbl /app

# What "localhost" means to a request that leaves the app. This container runs
# RAG-BenchLab and nothing else, so a customer whose RAG system or embedding
# server sits on the same machine needs the host, not the container — and
# host.docker.internal is not a name to make them learn. The app translates
# every loopback address it is about to call; set this to empty to stop that.
# See app/core/host.py.
#
# Declared here rather than with the other ENVs on purpose: anything above the
# pip install invalidates that layer when it changes, and on the emulated amd64
# half that layer is ten minutes. This is a one-line knob and should not cost
# that to turn.
ENV RBL_HOST_GATEWAY=host.docker.internal

USER rbl
WORKDIR /app/backend

EXPOSE 6742

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:6742/api/health', timeout=4).status == 200 else 1)"]

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6742"]
