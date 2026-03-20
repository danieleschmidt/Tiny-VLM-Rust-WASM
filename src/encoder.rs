use crate::patch_embedder::PatchEmbedder;
use crate::transformer::TransformerBlock;

/// Full Tiny VLM Encoder: PatchEmbedder + N TransformerBlocks.
pub struct TinyVLMEncoder {
    patch_embedder: PatchEmbedder,
    blocks: Vec<TransformerBlock>,
    embed_dim: usize,
}

impl TinyVLMEncoder {
    /// Create encoder with given embed_dim, num_heads, and num_blocks (default: 4).
    /// Uses patch_size=16, image_size=224 by default.
    pub fn new(embed_dim: usize, num_heads: usize, num_blocks: usize) -> Self {
        Self::with_image_size(embed_dim, num_heads, num_blocks, 224)
    }

    /// Create encoder with explicit image_size (must be divisible by 16).
    pub fn with_image_size(embed_dim: usize, num_heads: usize, num_blocks: usize, image_size: usize) -> Self {
        let patch_embedder = PatchEmbedder::new(16, embed_dim, image_size);
        let blocks = (0..num_blocks)
            .map(|_| TransformerBlock::new(embed_dim, num_heads))
            .collect();
        TinyVLMEncoder {
            patch_embedder,
            blocks,
            embed_dim,
        }
    }

    /// Forward: image flat [H*W*C] -> [n_patches, embed_dim]
    pub fn forward(&self, image: &[f32]) -> Vec<Vec<f32>> {
        let mut x = self.patch_embedder.forward(image);
        for block in &self.blocks {
            x = block.forward(&x);
        }
        x
    }

    /// Count total parameters across all components.
    pub fn param_count(&self) -> usize {
        let patch_dim = self.patch_embedder.patch_dim();
        // PatchEmbedder projection: [patch_dim, embed_dim]
        let pe_params = patch_dim * self.embed_dim;

        // Each transformer block:
        // - 4 attention matrices: 4 * embed_dim^2
        // - ffn_w1: embed_dim * 4*embed_dim
        // - ffn_w2: 4*embed_dim * embed_dim
        // - 4 layer norm vectors: 4 * embed_dim
        let ffn_dim = 4 * self.embed_dim;
        let block_params = 4 * self.embed_dim * self.embed_dim
            + self.embed_dim * ffn_dim
            + ffn_dim * self.embed_dim
            + 4 * self.embed_dim;

        pe_params + self.blocks.len() * block_params
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}
