# Base image with CUDA for GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
COPY . /app

# Runtime: run the pipeline with GPU support
# Use --gpus all to enable GPU access
# Use --skip-debug to skip if running without GPU, or remove for full debug
ENTRYPOINT ["python3", "-m", "src.main"]
