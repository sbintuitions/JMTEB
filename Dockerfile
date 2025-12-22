# Use CUDA 12.1 for better compatibility with PyTorch 2.6+
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python 3.11 and essential tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build stage: export dependencies
FROM base AS builder

WORKDIR /build

# Install poetry
RUN pip install --no-cache-dir poetry==1.8.2

# Copy only dependency files first for better layer caching
COPY pyproject.toml poetry.lock* ./

# Export dependencies to requirements.txt
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --only main

# Runtime stage: install application
FROM base AS runtime

WORKDIR /app

# Copy application code and dependency files
COPY src ./src
COPY README.md pyproject.toml poetry.lock* ./

# Install poetry temporarily to handle the installation
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir poetry==1.8.2 \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi \
    && pip uninstall -y poetry

# Set default command to run JMTEB
CMD ["python", "-m", "jmteb.v2"]
