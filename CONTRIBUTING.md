# Contributing to Tiny-VLM-Rust-WASM

Thank you for your interest in contributing to Tiny-VLM-Rust-WASM! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Rust 1.75+ with `wasm32-unknown-unknown` target
- `wasm-pack` 0.12+
- Node.js 20+ (for tooling and examples)
- Git

### Local Development
```bash
# Clone and setup
git clone https://github.com/yourusername/Tiny-VLM-Rust-WASM
cd Tiny-VLM-Rust-WASM

# Install Rust WASM target
rustup target add wasm32-unknown-unknown

# Install development dependencies
cargo install wasm-pack
npm install

# Run tests
cargo test
cargo test --target wasm32-unknown-unknown

# Build WASM module
wasm-pack build --target web --features simd

# Run benchmarks
cargo bench --features native
```

## Contribution Guidelines

### Code Style
- Follow standard Rust conventions (`rustfmt` and `clippy`)
- Use meaningful variable and function names
- Add documentation for public APIs
- Include unit tests for new functionality

### Performance Guidelines
- All performance-critical code must include benchmarks
- SIMD optimizations should have both scalar and vector versions
- Memory allocations in hot paths must be justified
- Profile before and after optimization changes

### SIMD Development
```rust
// Always provide fallback implementations
#[cfg(target_feature = "neon")]
fn process_neon(data: &[f32]) -> Vec<f32> {
    // ARM NEON implementation
}

#[cfg(not(target_feature = "neon"))]
fn process_scalar(data: &[f32]) -> Vec<f32> {
    // Scalar fallback
}
```

### WebAssembly Considerations
- Minimize memory allocations
- Use `wasm-bindgen` properly for JS interop
- Test in multiple browsers
- Verify SIMD support detection

## Pull Request Process

1. **Create Feature Branch**: Branch from `main` with descriptive name
2. **Implement Changes**: Include tests and documentation
3. **Performance Testing**: Run benchmarks for performance changes
4. **Cross-Platform Testing**: Test on WASM and native targets
5. **Update Documentation**: Update README and code comments
6. **Create PR**: Use the pull request template

### PR Requirements
- [ ] All tests pass (`cargo test`)
- [ ] WASM compilation succeeds (`wasm-pack build`)
- [ ] Benchmarks show no performance regression
- [ ] Code follows style guidelines (`cargo fmt && cargo clippy`)
- [ ] Documentation updated if needed
- [ ] No new compiler warnings

## Issue Reporting

### Bug Reports
Include:
- Rust version and target platform
- Browser version (for WASM issues)
- Minimal reproduction case
- Expected vs actual behavior
- Performance measurements if relevant

### Feature Requests
Include:
- Clear use case description
- Performance requirements
- Platform compatibility needs
- Implementation suggestions if any

## Code Review

All submissions require review by a maintainer. We look for:
- Correctness and safety
- Performance implications
- Cross-platform compatibility
- Code clarity and maintainability
- Test coverage

## Performance Standards

### Benchmarking
```bash
# Run comprehensive benchmarks
cargo bench --features native

# WASM-specific benchmarks
wasm-pack build --target web --features simd
npm run bench:wasm
```

### Performance Targets
- Inference latency: < 200ms on iPhone 17
- Memory usage: < 100MB peak
- Binary size: < 5MB WASM
- Startup time: < 100ms

## Security

Report security vulnerabilities privately to the maintainers. Do not create public issues for security problems.

## Community

- Be respectful and inclusive
- Follow the Code of Conduct
- Help newcomers get started
- Share knowledge and best practices

## License

By contributing, you agree that your contributions will be licensed under the MIT License.