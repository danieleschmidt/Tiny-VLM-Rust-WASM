pub mod patch_embedder;
pub mod attention;
pub mod transformer;
pub mod encoder;

pub use patch_embedder::PatchEmbedder;
pub use attention::MultiHeadAttention;
pub use transformer::TransformerBlock;
pub use encoder::TinyVLMEncoder;

#[cfg(test)]
mod tests {
    use super::*;

    // --- PatchEmbedder tests ---

    #[test]
    fn test_patch_embedder_new() {
        let pe = PatchEmbedder::new(16, 256, 224);
        assert_eq!(pe.patch_dim(), 16 * 16 * 3);
    }

    #[test]
    fn test_patch_embedder_num_patches() {
        let pe = PatchEmbedder::new(16, 256, 224);
        // 224 / 16 = 14, 14 * 14 = 196
        assert_eq!(pe.num_patches(), 196);
    }

    #[test]
    fn test_patch_embedder_patch_dim() {
        let pe = PatchEmbedder::new(16, 256, 224);
        // 16 * 16 * 3 = 768
        assert_eq!(pe.patch_dim(), 768);
    }

    #[test]
    fn test_patch_embedder_forward_shape() {
        // Use small image for speed: 32x32, patch_size=16 => 2x2=4 patches
        let embed_dim = 64;
        let pe = PatchEmbedder::new(16, embed_dim, 32);
        let image = vec![0.5f32; 32 * 32 * 3];
        let out = pe.forward(&image);
        assert_eq!(out.len(), 4);
        for row in &out {
            assert_eq!(row.len(), embed_dim);
        }
    }

    // --- MultiHeadAttention tests ---

    #[test]
    fn test_attention_new() {
        let mha = MultiHeadAttention::new(64, 4);
        // Just checks it constructs without panic
        let _ = mha;
    }

    #[test]
    fn test_attention_forward_shape() {
        let embed_dim = 64;
        let num_heads = 4;
        let seq_len = 4;
        let mha = MultiHeadAttention::new(embed_dim, num_heads);
        let x: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| vec![0.1f32; embed_dim])
            .collect();
        let out = mha.forward(&x);
        assert_eq!(out.len(), seq_len);
        for row in &out {
            assert_eq!(row.len(), embed_dim);
        }
    }

    // --- TransformerBlock tests ---

    #[test]
    fn test_transformer_block_new() {
        let tb = TransformerBlock::new(64, 4);
        let _ = tb;
    }

    #[test]
    fn test_transformer_block_forward_shape() {
        let embed_dim = 64;
        let num_heads = 4;
        let seq_len = 4;
        let tb = TransformerBlock::new(embed_dim, num_heads);
        let x: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| vec![0.1f32; embed_dim])
            .collect();
        let out = tb.forward(&x);
        assert_eq!(out.len(), seq_len);
        for row in &out {
            assert_eq!(row.len(), embed_dim);
        }
    }

    // --- TinyVLMEncoder tests ---

    #[test]
    fn test_encoder_new() {
        let enc = TinyVLMEncoder::new(64, 4, 4);
        let _ = enc;
    }

    #[test]
    fn test_encoder_forward_shape() {
        let embed_dim = 64;
        // Use image_size=32, patch_size=16 => 2x2=4 patches
        let enc = TinyVLMEncoder::with_image_size(embed_dim, 4, 4, 32);
        let image = vec![0.1f32; 32 * 32 * 3];
        let out = enc.forward(&image);
        assert_eq!(out.len(), 4);
        for row in &out {
            assert_eq!(row.len(), embed_dim);
        }
    }

    #[test]
    fn test_encoder_param_count_positive() {
        let enc = TinyVLMEncoder::new(64, 4, 4);
        assert!(enc.param_count() > 0);
    }

    #[test]
    fn test_encoder_num_blocks() {
        let enc = TinyVLMEncoder::new(64, 4, 4);
        assert_eq!(enc.num_blocks(), 4);
    }

    #[test]
    fn test_encoder_embed_dim() {
        let enc = TinyVLMEncoder::new(64, 4, 4);
        assert_eq!(enc.embed_dim(), 64);
    }
}
