# Unstable Singularity Detector - Production Docker Image
# Based on DeepMind's breakthrough methodology for fluid dynamics singularities
#
# Build: docker build -t unstable-singularity-detector:1.0.0 .
# Run:   docker run -it --rm unstable-singularity-detector:1.0.0

FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

LABEL maintainer="Flamehaven Research <research@flamehaven.ai>"
LABEL description="Revolutionary implementation of DeepMind's breakthrough in fluid dynamics singularities"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Install package in editable mode
RUN pip install -e .

# Create output directories
RUN mkdir -p /app/outputs /app/logs /app/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch
ENV MPLCONFIGDIR=/app/.matplotlib

# Expose port for Gradio interface (added in Phase 3)
EXPOSE 7860

# Health check - Enhanced version
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.unstable_singularity_detector import UnstableSingularityDetector; \
                   detector = UnstableSingularityDetector(equation_type='ipm'); \
                   print('Health check passed')" || exit 1

# Default command: show info
CMD ["singularity-detect", "info"]