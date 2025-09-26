# --- builder -------------------------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /w

COPY pyproject.toml README.md ./
COPY docker/service/requirements-app.txt ./requirements.txt

# Build wheels for third-party deps
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-cache-dir -r requirements.txt -w /wheels

# Add your package source
COPY lion_linker ./lion_linker

# Build wheel for *your* package only (no deps to avoid duplicates)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-cache-dir --no-deps . -w /wheels

# --- runtime -------------------------------------------------------
FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /wheels /wheels
COPY docker/service/requirements-app.txt /requirements.txt
COPY app ./app

# Let pip RESOLVE from the wheelhouse instead of installing "*" directly
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/wheels -r /requirements.txt /wheels/lion_linker-*.whl

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","9000","--reload","--reload-dir","/app"]
