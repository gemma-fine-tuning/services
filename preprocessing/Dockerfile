FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and UV
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy requirements first for better caching
# COPY requirements.txt .
COPY pyproject.toml uv.lock ./

# Install Python dependencies using UV (much faster than pip)
# RUN uv pip install --system --no-cache -r requirements.txt
RUN uv sync --frozen --no-cache

# Download NLTK data
# RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('punkt')"

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run the application with gunicorn
CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]
