# Tiny-VLM-Rust-WASM

Pure Rust FastVLM-style vision encoder. Implements PatchEmbedder, MultiHeadAttention, TransformerBlock, TinyVLMEncoder with 4 blocks. Compiles to WASM.

## Architecture

```
Image [H×W×3]
    ↓
PatchEmbedder (16×16 patches → linear projection)
    ↓
TransformerBlock × N (LayerNorm → MHA → residual → LayerNorm → FFN → residual)
    ↓
Embeddings [n_patches, embed_dim]
```

### Components

- **`PatchEmbedder`** — Splits image into 16×16 patches, projects each to `embed_dim` via learned linear layer
- **`MultiHeadAttention`** — Scaled dot-product attention with Q/K/V/O projections, split across `num_heads` heads
- **`TransformerBlock`** — Pre-norm transformer block with GELU FFN (4× hidden expansion)
- **`TinyVLMEncoder`** — Full encoder: patch embedder + N transformer blocks (default 4)

## Usage

```rust
use tiny_vlm::TinyVLMEncoder;

// Create encoder: embed_dim=256, 4 heads, 4 blocks, 224×224 images
let encoder = TinyVLMEncoder::new(256, 4, 4);

// Encode a 224×224 RGB image (flat HWC f32 array)
let image = vec![0.5f32; 224 * 224 * 3];
let embeddings = encoder.forward(&image);
// embeddings: [196, 256] — one 256-dim vector per patch

println!("Patches: {}", embeddings.len());       // 196
println!("Embed dim: {}", embeddings[0].len());   // 256
println!("Params: {}", encoder.param_count());
```

## Tests

```bash
cargo test
```

13 tests covering shape correctness, construction, and parameter counting.

## WASM Build

```bash
wasm-pack build --target web
```

No external dependencies — pure Rust, zero-dep, compiles cleanly to WASM.

## License

MIT
