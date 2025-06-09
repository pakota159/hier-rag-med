# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/vectordb logs

# Set environment variables for MPS (Metal Performance Shaders)
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Expose port for Streamlit
EXPOSE 8501

# Set the entrypoint
ENTRYPOINT ["streamlit", "run", "src/web/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 