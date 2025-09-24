FROM python:3.11-slim

WORKDIR /app

# Copy project metadata first to preserve pip cache
COPY pyproject.toml README.md ./
COPY lion_linker ./lion_linker
COPY docker/service/requirements-app.txt ./requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e ".[app]"

# Copy application source
COPY app ./app

# Ensure runtime directories exist (mounted volumes will overlay if provided)
RUN mkdir -p data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
