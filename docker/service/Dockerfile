FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy the FastAPI application code
COPY app ./app

# Copy project configuration and requirements from the repository root
COPY pyproject.toml .
COPY ./lion_linker ./lion_linker

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[app]"

# Expose the FastAPI port and run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
