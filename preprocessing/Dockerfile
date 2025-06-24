FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and UV
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy requirements
COPY pyproject.toml uv.lock ./

# Install Python dependencies using UV
RUN uv sync --frozen --no-cache

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run the application with gunicorn
CMD ["uv", "run", "app.py"]