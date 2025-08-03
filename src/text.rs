//! Text processing and tokenization for language model

use crate::{
    memory::{MemoryPool, Tensor, TensorShape},
    Result, TinyVlmError,
};
use serde::{Deserialize, Serialize};
use core::collections::{BTreeMap, VecDeque};

/// Configuration for the tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Padding token ID
    pub pad_token: u32,
    /// Beginning of sequence token ID
    pub bos_token: u32,
    /// End of sequence token ID
    pub eos_token: u32,
    /// Unknown token ID
    pub unk_token: u32,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            max_length: 512,
            pad_token: 0,
            bos_token: 1,
            eos_token: 2,
            unk_token: 3,
        }
    }
}

/// A Byte-Pair Encoding (BPE) tokenizer
pub struct Tokenizer {
    /// Configuration
    config: TokenizerConfig,
    /// Vocabulary mapping from tokens to IDs
    vocab: BTreeMap<String, u32>,
    /// Inverse vocabulary mapping from IDs to tokens
    id_to_token: BTreeMap<u32, String>,
    /// BPE merge rules
    merges: Vec<(String, String)>,
    /// Embedding matrix for token embeddings
    embeddings: Tensor<f32>,
}

impl Tokenizer {
    /// Create a new tokenizer with given configuration
    pub fn new(config: TokenizerConfig, embedding_dim: usize) -> Result<Self> {
        let mut vocab = BTreeMap::new();
        let mut id_to_token = BTreeMap::new();
        let mut merges = Vec::new();

        // Initialize basic vocabulary
        Self::init_basic_vocab(&mut vocab, &mut id_to_token, &config)?;
        
        // Initialize BPE merges (simplified for demo)
        Self::init_merges(&mut merges)?;

        // Initialize embedding matrix
        let embedding_shape = TensorShape::new(&[config.vocab_size, embedding_dim])?;
        let mut embeddings = Tensor::zeros(embedding_shape)?;
        Self::init_embeddings(&mut embeddings)?;

        Ok(Self {
            config,
            vocab,
            id_to_token,
            merges,
            embeddings,
        })
    }

    /// Tokenize input text into token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(vec![self.config.bos_token, self.config.eos_token]);
        }

        // Preprocessing: lowercase and basic normalization
        let normalized = self.normalize_text(text);
        
        // Split into words
        let words = self.split_words(&normalized);
        
        // Apply BPE to each word
        let mut tokens = Vec::new();
        tokens.push(self.config.bos_token);
        
        for word in words {
            let word_tokens = self.apply_bpe(&word)?;
            tokens.extend(word_tokens);
        }
        
        tokens.push(self.config.eos_token);
        
        // Truncate if too long
        if tokens.len() > self.config.max_length {
            tokens.truncate(self.config.max_length - 1);
            tokens.push(self.config.eos_token);
        }
        
        Ok(tokens)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut result = String::new();
        
        for &token_id in token_ids {
            if token_id == self.config.bos_token || token_id == self.config.eos_token {
                continue;
            }
            
            if let Some(token) = self.id_to_token.get(&token_id) {
                if !token.starts_with("##") && !result.is_empty() {
                    result.push(' ');
                }
                let clean_token = token.strip_prefix("##").unwrap_or(token);
                result.push_str(clean_token);
            }
        }
        
        Ok(result)
    }

    /// Get embedding for a token ID
    pub fn get_embedding(&self, token_id: u32) -> Result<&[f32]> {
        if token_id as usize >= self.config.vocab_size {
            return Err(TinyVlmError::text_processing("Token ID out of vocabulary range"));
        }
        
        let embedding_dim = self.embeddings.shape().dims[1];
        let start_idx = token_id as usize * embedding_dim;
        let end_idx = start_idx + embedding_dim;
        
        Ok(&self.embeddings.data()[start_idx..end_idx])
    }

    /// Get embeddings for a sequence of token IDs
    pub fn embed_sequence(&self, token_ids: &[u32], memory_pool: &mut MemoryPool<f32>) -> Result<Tensor<f32>> {
        let seq_len = token_ids.len();
        let embedding_dim = self.embeddings.shape().dims[1];
        
        let output_shape = TensorShape::new(&[1, seq_len, embedding_dim])?;
        let mut output = memory_pool.allocate(output_shape)?;
        let output_data = output.data_mut();
        
        for (i, &token_id) in token_ids.iter().enumerate() {
            let embedding = self.get_embedding(token_id)?;
            let start_idx = i * embedding_dim;
            let end_idx = start_idx + embedding_dim;
            output_data[start_idx..end_idx].copy_from_slice(embedding);
        }
        
        Ok(output)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get maximum sequence length
    pub fn max_length(&self) -> usize {
        self.config.max_length
    }

    /// Get special token IDs
    pub fn special_tokens(&self) -> (u32, u32, u32, u32) {
        (self.config.pad_token, self.config.bos_token, self.config.eos_token, self.config.unk_token)
    }

    // Private helper methods

    fn init_basic_vocab(
        vocab: &mut BTreeMap<String, u32>,
        id_to_token: &mut BTreeMap<u32, String>,
        config: &TokenizerConfig,
    ) -> Result<()> {
        // Special tokens
        let special_tokens = vec![
            ("<pad>", config.pad_token),
            ("<bos>", config.bos_token),
            ("<eos>", config.eos_token),
            ("<unk>", config.unk_token),
        ];

        for (token, id) in special_tokens {
            vocab.insert(token.to_string(), id);
            id_to_token.insert(id, token.to_string());
        }

        // Basic ASCII characters
        let mut current_id = 4;
        for ch in 32u8..127u8 {
            let token = (ch as char).to_string();
            vocab.insert(token.clone(), current_id);
            id_to_token.insert(current_id, token);
            current_id += 1;
        }

        // Common subwords (simplified set)
        let common_subwords = vec![
            "##ing", "##ed", "##er", "##est", "##ly", "##s", "##ion", "##tion",
            "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they",
            "i", "at", "be", "this", "have", "from", "or", "one", "had",
            "by", "word", "but", "not", "what", "all", "were", "we", "when",
        ];

        for subword in common_subwords {
            if current_id < config.vocab_size as u32 {
                vocab.insert(subword.to_string(), current_id);
                id_to_token.insert(current_id, subword.to_string());
                current_id += 1;
            }
        }

        // Fill remaining vocabulary with placeholder tokens
        while current_id < config.vocab_size as u32 {
            let token = format!("<unused{}>", current_id);
            vocab.insert(token.clone(), current_id);
            id_to_token.insert(current_id, token);
            current_id += 1;
        }

        Ok(())
    }

    fn init_merges(merges: &mut Vec<(String, String)>) -> Result<()> {
        // Simplified BPE merge rules
        let merge_rules = vec![
            ("t", "h"),
            ("e", "r"),
            ("i", "n"),
            ("o", "n"),
            ("a", "n"),
            ("th", "e"),
            ("er", "e"),
            ("in", "g"),
            ("on", "e"),
            ("an", "d"),
        ];

        for (first, second) in merge_rules {
            merges.push((first.to_string(), second.to_string()));
        }

        Ok(())
    }

    fn init_embeddings(embeddings: &mut Tensor<f32>) -> Result<()> {
        let data = embeddings.data_mut();
        let embedding_dim = embeddings.shape().dims[1];
        
        // Initialize with small random values
        for (i, val) in data.iter_mut().enumerate() {
            let pseudo_random = ((i * 1234567 + 89) % 1000000) as f32 / 1000000.0;
            *val = (pseudo_random * 2.0 - 1.0) * 0.02; // Small random values
        }

        Ok(())
    }

    fn normalize_text(&self, text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?".contains(*c))
            .collect()
    }

    fn split_words(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| word.to_string())
            .collect()
    }

    fn apply_bpe(&self, word: &str) -> Result<Vec<u32>> {
        if word.is_empty() {
            return Ok(Vec::new());
        }

        // Start with character-level tokens
        let mut tokens: VecDeque<String> = word.chars().map(|c| c.to_string()).collect();

        // Apply BPE merges
        for (first, second) in &self.merges {
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if tokens[i] == *first && tokens[i + 1] == *second {
                    let merged = format!("{}{}", first, second);
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        // Convert tokens to IDs
        let mut token_ids = Vec::new();
        for token in tokens {
            if let Some(&id) = self.vocab.get(&token) {
                token_ids.push(id);
            } else {
                // Try to find the token with ## prefix for subwords
                let subword_token = format!("##{}", token);
                if let Some(&id) = self.vocab.get(&subword_token) {
                    token_ids.push(id);
                } else {
                    token_ids.push(self.config.unk_token);
                }
            }
        }

        Ok(token_ids)
    }
}

/// Text embedding layer for converting token IDs to dense representations
pub struct TextEmbedding {
    /// Embedding matrix
    embeddings: Tensor<f32>,
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    embedding_dim: usize,
}

impl TextEmbedding {
    /// Create a new text embedding layer
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Result<Self> {
        let embedding_shape = TensorShape::new(&[vocab_size, embedding_dim])?;
        let mut embeddings = Tensor::zeros(embedding_shape)?;
        
        // Initialize embeddings
        Self::init_embeddings(&mut embeddings)?;

        Ok(Self {
            embeddings,
            vocab_size,
            embedding_dim,
        })
    }

    /// Forward pass: convert token IDs to embeddings
    pub fn forward(&self, token_ids: &[u32], memory_pool: &mut MemoryPool<f32>) -> Result<Tensor<f32>> {
        let seq_len = token_ids.len();
        let output_shape = TensorShape::new(&[1, seq_len, self.embedding_dim])?;
        let mut output = memory_pool.allocate(output_shape)?;
        
        let embeddings_data = self.embeddings.data();
        let output_data = output.data_mut();

        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id as usize >= self.vocab_size {
                return Err(TinyVlmError::text_processing("Token ID out of range"));
            }

            let embedding_start = token_id as usize * self.embedding_dim;
            let embedding_end = embedding_start + self.embedding_dim;
            
            let output_start = i * self.embedding_dim;
            let output_end = output_start + self.embedding_dim;

            output_data[output_start..output_end]
                .copy_from_slice(&embeddings_data[embedding_start..embedding_end]);
        }

        Ok(output)
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn init_embeddings(embeddings: &mut Tensor<f32>) -> Result<()> {
        let data = embeddings.data_mut();
        
        // Xavier uniform initialization
        let fan_in = embeddings.shape().dims[1];
        let bound = (6.0 / fan_in as f32).sqrt();
        
        for (i, val) in data.iter_mut().enumerate() {
            let pseudo_random = ((i * 7919 + 127) % 1000000) as f32 / 1000000.0;
            *val = (pseudo_random * 2.0 - 1.0) * bound;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_config() {
        let config = TokenizerConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.max_length, 512);
    }

    #[test]
    fn test_tokenizer_creation() {
        let config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(config, 768);
        assert!(tokenizer.is_ok());
    }

    #[test]
    fn test_basic_encoding() {
        let config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(config, 768).unwrap();
        
        let text = "hello world";
        let tokens = tokenizer.encode(text).unwrap();
        
        // Should contain BOS, some tokens, and EOS
        assert!(tokens.len() >= 3);
        assert_eq!(tokens[0], tokenizer.config.bos_token);
        assert_eq!(tokens[tokens.len() - 1], tokenizer.config.eos_token);
    }

    #[test]
    fn test_decode() {
        let config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(config, 768).unwrap();
        
        let text = "test";
        let tokens = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();
        
        // Decoded text should contain the original word
        assert!(decoded.to_lowercase().contains("test"));
    }

    #[test]
    fn test_embedding_lookup() {
        let config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(config, 768).unwrap();
        
        let embedding = tokenizer.get_embedding(1);
        assert!(embedding.is_ok());
        assert_eq!(embedding.unwrap().len(), 768);
    }

    #[test]
    fn test_text_embedding() {
        let embedding = TextEmbedding::new(1000, 256);
        assert!(embedding.is_ok());
        
        let embed = embedding.unwrap();
        assert_eq!(embed.vocab_size(), 1000);
        assert_eq!(embed.embedding_dim(), 256);
    }

    #[test]
    fn test_sequence_embedding() {
        let config = TokenizerConfig::default();
        let tokenizer = Tokenizer::new(config, 768).unwrap();
        let mut memory_pool = MemoryPool::new(1000000);
        
        let tokens = vec![1, 100, 200, 2]; // BOS, some tokens, EOS
        let result = tokenizer.embed_sequence(&tokens, &mut memory_pool);
        
        assert!(result.is_ok());
        let tensor = result.unwrap();
        assert_eq!(tensor.shape().dims[1], 4); // sequence length
        assert_eq!(tensor.shape().dims[2], 768); // embedding dim
    }
}