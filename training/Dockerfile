FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu124.py310

# Set working directory
WORKDIR /app

# Install UV for faster package installation
RUN pip install --no-cache-dir uv

# Copy requirements and install Python dependencies using UV
# Note: The base image already has CUDA, PyTorch, and many ML libraries pre-installed
# COPY requirements.txt .
# RUN uv pip install --system --no-cache -r requirements.txt
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8081

# Expose port
EXPOSE 8081

# Run the application
CMD ["uv", "run", "app.py"]
