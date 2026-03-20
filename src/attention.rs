/// Multi-Head Self-Attention module.
pub struct MultiHeadAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    w_q: Vec<Vec<f32>>, // [embed_dim, embed_dim]
    w_k: Vec<Vec<f32>>,
    w_v: Vec<Vec<f32>>,
    w_o: Vec<Vec<f32>>,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert_eq!(embed_dim % num_heads, 0, "embed_dim must be divisible by num_heads");
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (embed_dim as f32).sqrt();

        let init_matrix = |offset: usize| -> Vec<Vec<f32>> {
            (0..embed_dim)
                .map(|i| {
                    (0..embed_dim)
                        .map(|j| ((i * embed_dim + j + offset) as f32).cos() * scale)
                        .collect()
                })
                .collect()
        };

        MultiHeadAttention {
            embed_dim,
            num_heads,
            head_dim,
            w_q: init_matrix(0),
            w_k: init_matrix(1),
            w_v: init_matrix(2),
            w_o: init_matrix(3),
        }
    }

    /// Forward: x [seq_len, embed_dim] -> [seq_len, embed_dim]
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = x.len();

        // Project Q, K, V
        let q = Self::matmul(x, &self.w_q);
        let k = Self::matmul(x, &self.w_k);
        let v = Self::matmul(x, &self.w_v);

        // Split into heads and compute attention per head
        let mut head_outputs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;

            let q_h: Vec<Vec<f32>> = q.iter().map(|row| row[start..end].to_vec()).collect();
            let k_h: Vec<Vec<f32>> = k.iter().map(|row| row[start..end].to_vec()).collect();
            let v_h: Vec<Vec<f32>> = v.iter().map(|row| row[start..end].to_vec()).collect();

            let attn_h = Self::scaled_dot_product_attention(&q_h, &k_h, &v_h);
            head_outputs.push(attn_h);
        }

        // Concatenate heads: [seq_len, embed_dim]
        let concat: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| {
                let mut row = Vec::with_capacity(self.embed_dim);
                for h in 0..self.num_heads {
                    row.extend_from_slice(&head_outputs[h][i]);
                }
                row
            })
            .collect();

        // Output projection
        Self::matmul(&concat, &self.w_o)
    }

    fn scaled_dot_product_attention(
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let head_dim = q[0].len();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // scores = Q * K^T / sqrt(head_dim)  [seq_len, seq_len]
        let k_t = Self::transpose(k);
        let mut scores = Self::matmul(q, &k_t);
        for row in scores.iter_mut() {
            for s in row.iter_mut() {
                *s *= scale;
            }
        }

        // Softmax each row
        let attn_weights: Vec<Vec<f32>> = scores.iter().map(|row| Self::softmax(row)).collect();

        // Output = attn_weights * V
        Self::matmul(&attn_weights, v)
    }

    fn softmax(x: &[f32]) -> Vec<f32> {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Matrix multiply: a [m, k] * b [k, n] -> [m, n]
    fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if a.is_empty() || b.is_empty() {
            return vec![];
        }
        let m = a.len();
        let k = a[0].len();
        let n = b[0].len();
        let mut out = vec![vec![0.0f32; n]; m];
        for i in 0..m {
            for p in 0..k {
                let a_ip = a[i][p];
                if a_ip == 0.0 {
                    continue;
                }
                for j in 0..n {
                    out[i][j] += a_ip * b[p][j];
                }
            }
        }
        out
    }

    fn transpose(m: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if m.is_empty() {
            return vec![];
        }
        let rows = m.len();
        let cols = m[0].len();
        let mut t = vec![vec![0.0f32; rows]; cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j][i] = m[i][j];
            }
        }
        t
    }
}
