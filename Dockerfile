# syntax=docker/dockerfile:1.4

# Use a recent, stable version of PyTorch with CUDA 12.1.
# The original Dockerfile referenced a non-existent PyTorch version (2.7).
# We use a -devel image to ensure build tools are available for PyG's compiled extensions.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV CUDA_HOME=/usr/local/cuda-12.1

# Use an ARG to specify the PyG version for reproducibility and easy updates.
ARG PYG_VERSION=2.5.3

# Install git to clone the repository.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone a specific version of the repository for reproducible builds,
# and use --depth 1 to reduce image size.
RUN git clone --depth 1 --branch ${PYG_VERSION} https://github.com/pyg-team/pytorch_geometric .

# Install all Python dependencies in a single layer to reduce image size.
# The wheel source URL is updated to match the PyTorch/CUDA version.
RUN pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html && \
# Install torch_geometric itself from the cloned source, which also installs
# other Python dependencies listed in pyproject.toml.
    pip install --no-cache-dir .

# Create an entrypoint to run the gcn.py example.
COPY --chmod=755 <<'ENTRYPOINT' /app/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

# The gcn.py example script trains a GCN model on the Cora dataset.
# The Planetoid dataset class will download the data on the first run.
python examples/gcn.py
ENTRYPOINT

ENTRYPOINT ["/app/entrypoint.sh"]
