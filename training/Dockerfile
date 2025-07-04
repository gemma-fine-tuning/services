FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y && rm -rf /var/lib/apt/lists/*

# Install UV for faster package installation
RUN pip install --no-cache-dir uv

# Copy requirements and install Python dependencies using UV
COPY pyproject.toml uv.lock ./

# Install UV deps
RUN uv venv && uv sync --frozen --no-cache --verbose

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8080

# Run the application
CMD ["uv", "run", "app.py"]
