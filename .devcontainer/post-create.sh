#!/bin/bash

# Post-create script for Tiny-VLM development container
set -e

echo "🚀 Setting up Tiny-VLM development environment..."

# Make sure we're in the right directory
cd /workspace

# Install Rust dependencies
echo "📦 Installing Rust dependencies..."
if [ -f "Cargo.toml" ]; then
    cargo fetch
fi

# Install pre-commit hooks
echo "🔧 Setting up git hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
fi

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p examples/web
mkdir -p examples/native
mkdir -p benchmarks
mkdir -p models
mkdir -p data/samples

# Install Node.js dependencies if package.json exists
if [ -f "package.json" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Set up git configuration for the container
echo "⚙️ Configuring git..."
git config --global core.autocrlf input
git config --global init.defaultBranch main

# Create sample data directories
echo "📊 Setting up sample data structure..."
mkdir -p data/{images,models,benchmarks}

# Download sample images for testing (create placeholders)
echo "🖼️ Creating sample test data..."
cat > data/images/README.md << 'EOF'
# Sample Images

This directory contains sample images for testing the VLM model.

## Structure
- `test_images/` - Small test images for unit tests
- `benchmark_images/` - Standardized images for performance benchmarking
- `examples/` - Example images for demos

## Usage
Place your test images here or use the provided sample generation scripts.
EOF

# Create example model structure
echo "🤖 Setting up model directory structure..."
cat > models/README.md << 'EOF'
# Model Directory

This directory contains pre-trained model weights and configurations.

## Structure
- `tiny-vlm-25m/` - 25M parameter model variant
- `tiny-vlm-50m/` - 50M parameter model variant  
- `tiny-vlm-100m/` - 100M parameter model variant
- `quantized/` - Quantized model variants (INT8, INT4)

## Format
Models are stored in a custom binary format optimized for fast loading.
See `docs/model-format.md` for details.
EOF

# Set up benchmarking environment
echo "📈 Setting up benchmarking environment..."
cat > benchmarks/README.md << 'EOF'
# Benchmarks

Performance benchmarking suite for Tiny-VLM.

## Structure
- `inference/` - Inference speed benchmarks
- `memory/` - Memory usage profiling
- `accuracy/` - Model accuracy evaluation
- `cross_platform/` - Cross-platform performance comparison

## Running Benchmarks
```bash
cargo bench
npm run bench:wasm
python3 scripts/benchmark_accuracy.py
```
EOF

# Create development scripts
echo "🔨 Creating development scripts..."
mkdir -p scripts

cat > scripts/build-wasm.sh << 'EOF'
#!/bin/bash
# Build WASM package with optimizations
set -e

echo "Building WASM package..."
wasm-pack build --target web --features simd --release

echo "Optimizing WASM binary..."
wasm-opt -Oz -o pkg/tiny_vlm_bg.wasm pkg/tiny_vlm_bg.wasm

echo "WASM build complete!"
EOF
chmod +x scripts/build-wasm.sh

cat > scripts/run-benchmarks.sh << 'EOF'
#!/bin/bash
# Run comprehensive benchmarks
set -e

echo "Running Rust benchmarks..."
cargo bench --features native

echo "Building and testing WASM..."
./scripts/build-wasm.sh

echo "Running cross-platform tests..."
cargo test --all-features

echo "Benchmarks complete!"
EOF
chmod +x scripts/run-benchmarks.sh

cat > scripts/dev-setup.sh << 'EOF'
#!/bin/bash
# Development environment setup
set -e

echo "Installing development dependencies..."
cargo install cargo-watch cargo-audit

echo "Setting up pre-commit hooks..."
cat > .pre-commit-config.yaml << 'EOL'
repos:
  - repo: local
    hooks:
      - id: cargo-fmt
        name: cargo fmt
        entry: cargo fmt --all --
        language: system
        types: [rust]
      - id: cargo-clippy
        name: cargo clippy
        entry: cargo clippy --all-targets --all-features -- -D warnings
        language: system
        types: [rust]
        pass_filenames: false
      - id: cargo-test
        name: cargo test
        entry: cargo test --all-features
        language: system
        types: [rust]
        pass_filenames: false
EOL

if command -v pre-commit &> /dev/null; then
    pre-commit install
fi

echo "Development setup complete!"
EOF
chmod +x scripts/dev-setup.sh

# Print completion message
echo "✅ Tiny-VLM development environment setup complete!"
echo ""
echo "🔥 Quick start:"
echo "  cargo test           # Run tests"
echo "  cargo bench          # Run benchmarks"  
echo "  ./scripts/build-wasm.sh  # Build WASM package"
echo "  cargo watch -x test  # Watch mode for tests"
echo ""
echo "📖 See CONTRIBUTING.md for detailed development guidelines"