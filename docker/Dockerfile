FROM python:3.11-slim

# Install system dependencies
# - curl: for installing uv
# - build-essential: for compiling C extensions (needed for some python packages)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv using pip instead (more reliable in docker than installer script for some base images)
RUN pip install uv

WORKDIR /app

# Copy package definition AND README (required for build backend)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies using uv
# -e .[observability] installs the project in editable mode with observability extras
# We also explicitly install scalene
RUN uv pip install --system ".[observability]" scalene

# Copy the rest of the code
# Note: In docker-compose, we mount the volume so this is just a fallback
COPY . .

# Default command (keep container alive)
CMD ["tail", "-f", "/dev/null"]
