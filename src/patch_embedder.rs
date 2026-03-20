/// 16x16 patch projection.
/// Input: flat image [H*W*C] in HWC format (C=3 RGB).
/// Patches: (H/patch_size)*(W/patch_size) patches each of size patch_size*patch_size*C.
/// Output: projected embeddings [n_patches, embed_dim].
pub struct PatchEmbedder {
    pub patch_size: usize,
    pub embed_dim: usize,
    pub image_size: usize,
    /// Weight matrix [patch_dim, embed_dim]
    projection: Vec<Vec<f32>>,
}

impl PatchEmbedder {
    /// Initialize with small random weights (deterministic via simple formula).
    pub fn new(patch_size: usize, embed_dim: usize, image_size: usize) -> Self {
        let patch_dim = patch_size * patch_size * 3;
        // Simple deterministic initialization: w[i][j] = sin(i + j) / sqrt(patch_dim)
        let scale = 1.0 / (patch_dim as f32).sqrt();
        let projection: Vec<Vec<f32>> = (0..patch_dim)
            .map(|i| {
                (0..embed_dim)
                    .map(|j| ((i + j) as f32).sin() * scale)
                    .collect()
            })
            .collect();
        PatchEmbedder {
            patch_size,
            embed_dim,
            image_size,
            projection,
        }
    }

    /// Number of patches: (image_size / patch_size)^2
    pub fn num_patches(&self) -> usize {
        let n = self.image_size / self.patch_size;
        n * n
    }

    /// Dimension of each flattened patch: patch_size * patch_size * 3
    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * 3
    }

    /// Forward pass: image -> [n_patches, embed_dim]
    pub fn forward(&self, image: &[f32]) -> Vec<Vec<f32>> {
        let patches = self.extract_patches(image);
        patches.iter().map(|p| self.project(p)).collect()
    }

    /// Extract flat patches from HWC image.
    fn extract_patches(&self, image: &[f32]) -> Vec<Vec<f32>> {
        let n_grid = self.image_size / self.patch_size;
        let channels = 3usize;
        let patch_dim = self.patch_dim();
        let mut patches = Vec::with_capacity(n_grid * n_grid);

        for row in 0..n_grid {
            for col in 0..n_grid {
                let mut patch = Vec::with_capacity(patch_dim);
                for pr in 0..self.patch_size {
                    for pc in 0..self.patch_size {
                        let img_row = row * self.patch_size + pr;
                        let img_col = col * self.patch_size + pc;
                        for c in 0..channels {
                            let idx = (img_row * self.image_size + img_col) * channels + c;
                            patch.push(if idx < image.len() { image[idx] } else { 0.0 });
                        }
                    }
                }
                patches.push(patch);
            }
        }
        patches
    }

    /// Project a single patch via matrix multiply: patch [patch_dim] x projection [patch_dim, embed_dim] -> [embed_dim]
    fn project(&self, patch: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.embed_dim];
        for (i, &p) in patch.iter().enumerate() {
            if i < self.projection.len() {
                for (j, o) in out.iter_mut().enumerate() {
                    *o += p * self.projection[i][j];
                }
            }
        }
        out
    }
}
