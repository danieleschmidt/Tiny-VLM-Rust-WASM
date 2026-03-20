use crate::attention::MultiHeadAttention;

/// Transformer block: LayerNorm -> MHA -> residual -> LayerNorm -> FFN -> residual
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    ffn_w1: Vec<Vec<f32>>, // [embed_dim, 4*embed_dim]
    ffn_w2: Vec<Vec<f32>>, // [4*embed_dim, embed_dim]
    ln1_scale: Vec<f32>,
    ln1_bias: Vec<f32>,
    ln2_scale: Vec<f32>,
    ln2_bias: Vec<f32>,
}

impl TransformerBlock {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let ffn_dim = 4 * embed_dim;
        let scale = 1.0 / (embed_dim as f32).sqrt();

        let ffn_w1: Vec<Vec<f32>> = (0..embed_dim)
            .map(|i| {
                (0..ffn_dim)
                    .map(|j| ((i * ffn_dim + j) as f32).sin() * scale)
                    .collect()
            })
            .collect();

        let ffn_w2: Vec<Vec<f32>> = (0..ffn_dim)
            .map(|i| {
                (0..embed_dim)
                    .map(|j| ((i * embed_dim + j) as f32).cos() * scale)
                    .collect()
            })
            .collect();

        TransformerBlock {
            attention: MultiHeadAttention::new(embed_dim, num_heads),
            ffn_w1,
            ffn_w2,
            ln1_scale: vec![1.0f32; embed_dim],
            ln1_bias: vec![0.0f32; embed_dim],
            ln2_scale: vec![1.0f32; embed_dim],
            ln2_bias: vec![0.0f32; embed_dim],
        }
    }

    /// Forward: x [seq_len, embed_dim] -> [seq_len, embed_dim]
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Pre-norm + attention + residual
        let normed1: Vec<Vec<f32>> = x
            .iter()
            .map(|row| Self::layer_norm(row, &self.ln1_scale, &self.ln1_bias))
            .collect();
        let attn_out = self.attention.forward(&normed1);
        let x2: Vec<Vec<f32>> = x
            .iter()
            .zip(attn_out.iter())
            .map(|(xi, ai)| xi.iter().zip(ai.iter()).map(|(a, b)| a + b).collect())
            .collect();

        // Pre-norm + FFN + residual
        let normed2: Vec<Vec<f32>> = x2
            .iter()
            .map(|row| Self::layer_norm(row, &self.ln2_scale, &self.ln2_bias))
            .collect();
        let ffn_out: Vec<Vec<f32>> = normed2.iter().map(|row| self.ffn(row)).collect();
        x2.iter()
            .zip(ffn_out.iter())
            .map(|(xi, fi)| xi.iter().zip(fi.iter()).map(|(a, b)| a + b).collect())
            .collect()
    }

    /// Layer normalization
    fn layer_norm(x: &[f32], scale: &[f32], bias: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
        let std = (var + 1e-5).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, &v)| (v - mean) / std * scale[i] + bias[i])
            .collect()
    }

    /// Feed-forward network: x [embed_dim] -> hidden [4*embed_dim] -> out [embed_dim]
    fn ffn(&self, x: &[f32]) -> Vec<f32> {
        let ffn_dim = self.ffn_w1[0].len();
        // First linear: [embed_dim] x [embed_dim, ffn_dim] -> [ffn_dim]
        let mut hidden = vec![0.0f32; ffn_dim];
        for (i, &xi) in x.iter().enumerate() {
            for j in 0..ffn_dim {
                hidden[j] += xi * self.ffn_w1[i][j];
            }
        }
        // GELU activation
        let hidden_act: Vec<f32> = hidden.iter().map(|&v| Self::gelu(v)).collect();
        // Second linear: [ffn_dim] x [ffn_dim, embed_dim] -> [embed_dim]
        let embed_dim = x.len();
        let mut out = vec![0.0f32; embed_dim];
        for (i, &hi) in hidden_act.iter().enumerate() {
            for j in 0..embed_dim {
                out[j] += hi * self.ffn_w2[i][j];
            }
        }
        out
    }

    /// GELU activation approximation
    fn gelu(x: f32) -> f32 {
        // Approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
    }
}
