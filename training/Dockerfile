FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including C compiler for Triton and CUDA support
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for C compiler
ENV CC=gcc
ENV CXX=g++

# Install UV for faster package installation
RUN pip install --no-cache-dir uv

# Copy requirements and install Python dependencies using UV
COPY pyproject.toml uv.lock ./

# Install PyTorch with CUDA support first, then remaining dependencies, then force CUDA again
RUN uv venv && \
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    uv sync --frozen --no-cache --verbose && \
    uv pip install --upgrade unsloth unsloth_zoo

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8081
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8081

# Run the application
CMD ["python", "app.py"]
