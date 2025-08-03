# Tiny-VLM Development Container
FROM mcr.microsoft.com/devcontainers/rust:1.75

# Install additional dependencies for WASM and ML development
RUN apt-get update && apt-get install -y \
    # WASM tools
    binaryen \
    wabt \
    # Build tools
    cmake \
    ninja-build \
    # Python for data processing and benchmarking
    python3 \
    python3-pip \
    python3-venv \
    # Image processing libraries
    libimage-dev \
    # Performance profiling tools
    valgrind \
    perf \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Rust components for WASM development
RUN rustup target add wasm32-unknown-unknown wasm32-wasi
RUN rustup component add rustfmt clippy llvm-tools-preview

# Install wasm-pack for building WASM packages
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Install cargo tools for development
RUN cargo install \
    wasm-bindgen-cli \
    cargo-audit \
    cargo-outdated \
    cargo-watch \
    cargo-expand \
    flamegraph \
    cargo-llvm-cov \
    cargo-fuzz

# Install Node.js tools for WASM examples
RUN npm install -g \
    @wasm-tool/wasm-pack-plugin \
    webpack \
    webpack-cli \
    webpack-dev-server \
    typescript \
    ts-node

# Create workspace for Python tools
RUN python3 -m venv /opt/python-tools && \
    /opt/python-tools/bin/pip install \
    numpy \
    matplotlib \
    jupyter \
    pytest \
    black

# Set up development environment
WORKDIR /workspace
ENV PATH="/opt/python-tools/bin:$PATH"
ENV CARGO_TARGET_DIR="/workspace/target"

# Configure git for container development
RUN git config --global --add safe.directory /workspace