//! Tokenizer implementation.
//!
//! Provides a unified interface to HuggingFace tokenizers with
//! optimizations for LLM inference.

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokenizers::{
    Tokenizer as HfTokenizer,
    Encoding,
    PaddingParams,
    TruncationParams,
};
use tracing::{debug, info, warn};

use crate::error::TokenizerError;

/// Token ID type.
pub type TokenId = u32;

/// Rusteeze tokenizer wrapper.
pub struct Tokenizer {
    /// Inner HuggingFace tokenizer.
    inner: Arc<HfTokenizer>,

    /// Tokenizer configuration.
    config: TokenizerConfig,

    /// Special tokens.
    special_tokens: SpecialTokens,
}

impl Tokenizer {
    /// Create tokenizer from file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        info!("Loading tokenizer from: {}", path.display());

        let inner = HfTokenizer::from_file(path)
            .map_err(|e| TokenizerError::LoadError(e.to_string()))?;

        let config = TokenizerConfig::default();
        let special_tokens = SpecialTokens::from_tokenizer(&inner)?;

        debug!(
            "Loaded tokenizer with vocab size: {}",
            inner.get_vocab_size(true)
        );

        Ok(Self {
            inner: Arc::new(inner),
            config,
            special_tokens,
        })
    }

    /// Create tokenizer from pretrained model directory.
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self, TokenizerError> {
        let model_dir = model_dir.as_ref();

        // Try tokenizer.json first
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            return Self::from_file(&tokenizer_path);
        }

        // Fall back to loading from config
        Err(TokenizerError::LoadError(format!(
            "No tokenizer.json found in {}",
            model_dir.display()
        )))
    }

    /// Create tokenizer from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let inner = HfTokenizer::from_bytes(bytes)
            .map_err(|e| TokenizerError::LoadError(e.to_string()))?;

        let config = TokenizerConfig::default();
        let special_tokens = SpecialTokens::from_tokenizer(&inner)?;

        Ok(Self {
            inner: Arc::new(inner),
            config,
            special_tokens,
        })
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encode text to tokens.
    pub fn encode<S: AsRef<str>>(
        &self,
        text: S,
        add_special_tokens: bool,
    ) -> Result<EncodingResult, TokenizerError> {
        let encoding = self.inner
            .encode(text.as_ref(), add_special_tokens)
            .map_err(|e| TokenizerError::EncodeError(e.to_string()))?;

        Ok(EncodingResult::from_encoding(encoding))
    }

    /// Encode batch of texts.
    pub fn encode_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        add_special_tokens: bool,
    ) -> Result<Vec<EncodingResult>, TokenizerError> {
        let texts: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();

        let encodings = self.inner
            .encode_batch(texts, add_special_tokens)
            .map_err(|e| TokenizerError::EncodeError(e.to_string()))?;

        Ok(encodings.into_iter().map(EncodingResult::from_encoding).collect())
    }

    /// Decode tokens to text.
    pub fn decode(&self, ids: &[TokenId], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::DecodeError(e.to_string()))
    }

    /// Decode batch of token sequences.
    pub fn decode_batch(
        &self,
        batch: &[Vec<TokenId>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>, TokenizerError> {
        let batch_refs: Vec<&[TokenId]> = batch.iter().map(|v| v.as_slice()).collect();

        self.inner
            .decode_batch(&batch_refs, skip_special_tokens)
            .map_err(|e| TokenizerError::DecodeError(e.to_string()))
    }

    /// Get token ID for a piece of text.
    pub fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.inner.token_to_id(token)
    }

    /// Get text for a token ID.
    pub fn id_to_token(&self, id: TokenId) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Get special tokens.
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Get BOS token ID.
    pub fn bos_token_id(&self) -> Option<TokenId> {
        self.special_tokens.bos_token_id
    }

    /// Get EOS token ID.
    pub fn eos_token_id(&self) -> Option<TokenId> {
        self.special_tokens.eos_token_id
    }

    /// Get PAD token ID.
    pub fn pad_token_id(&self) -> Option<TokenId> {
        self.special_tokens.pad_token_id
    }

    /// Get UNK token ID.
    pub fn unk_token_id(&self) -> Option<TokenId> {
        self.special_tokens.unk_token_id
    }

    /// Check if token ID is a special token.
    pub fn is_special_token(&self, id: TokenId) -> bool {
        self.special_tokens.is_special(id)
    }

    /// Get configuration.
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Clone the inner tokenizer.
    pub fn clone_inner(&self) -> Arc<HfTokenizer> {
        Arc::clone(&self.inner)
    }
}

impl Clone for Tokenizer {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            config: self.config.clone(),
            special_tokens: self.special_tokens.clone(),
        }
    }
}

/// Tokenizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Maximum sequence length.
    pub max_length: Option<usize>,

    /// Truncation strategy.
    pub truncation: Option<TruncationStrategy>,

    /// Padding strategy.
    pub padding: Option<PaddingStrategy>,

    /// Add BOS token.
    pub add_bos_token: bool,

    /// Add EOS token.
    pub add_eos_token: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: None,
            truncation: None,
            padding: None,
            add_bos_token: true,
            add_eos_token: false,
        }
    }
}

/// Truncation strategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Truncate from the end.
    TruncateEnd,
    /// Truncate from the start.
    TruncateStart,
}

/// Padding strategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// Pad to max length.
    MaxLength,
    /// Pad to longest in batch.
    Longest,
    /// No padding.
    None,
}

/// Special tokens configuration.
#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    /// Beginning of sequence token ID.
    pub bos_token_id: Option<TokenId>,

    /// End of sequence token ID.
    pub eos_token_id: Option<TokenId>,

    /// Padding token ID.
    pub pad_token_id: Option<TokenId>,

    /// Unknown token ID.
    pub unk_token_id: Option<TokenId>,

    /// Separator token ID.
    pub sep_token_id: Option<TokenId>,

    /// Classification token ID.
    pub cls_token_id: Option<TokenId>,

    /// Mask token ID.
    pub mask_token_id: Option<TokenId>,

    /// All special token IDs.
    all_special_ids: Vec<TokenId>,
}

impl SpecialTokens {
    /// Create from tokenizer.
    pub fn from_tokenizer(tokenizer: &HfTokenizer) -> Result<Self, TokenizerError> {
        let mut tokens = Self::default();

        // Try to get special tokens from added vocabulary
        let added = tokenizer.get_added_vocabulary();
        for (token, id) in added.get_vocab() {
            let token_lower: String = token.to_lowercase();
            if token_lower.contains("bos") || token_lower.contains("<s>") {
                tokens.bos_token_id = Some(*id);
            } else if token_lower.contains("eos") || token_lower.contains("</s>") {
                tokens.eos_token_id = Some(*id);
            } else if token_lower.contains("pad") {
                tokens.pad_token_id = Some(*id);
            } else if token_lower.contains("unk") {
                tokens.unk_token_id = Some(*id);
            }
        }

        // Common fallbacks
        if tokens.bos_token_id.is_none() {
            tokens.bos_token_id = tokenizer.token_to_id("<s>");
        }
        if tokens.eos_token_id.is_none() {
            tokens.eos_token_id = tokenizer.token_to_id("</s>");
        }
        if tokens.pad_token_id.is_none() {
            tokens.pad_token_id = tokenizer.token_to_id("<pad>");
        }
        if tokens.unk_token_id.is_none() {
            tokens.unk_token_id = tokenizer.token_to_id("<unk>");
        }

        // Collect all special IDs
        for id in [
            tokens.bos_token_id,
            tokens.eos_token_id,
            tokens.pad_token_id,
            tokens.unk_token_id,
            tokens.sep_token_id,
            tokens.cls_token_id,
            tokens.mask_token_id,
        ] {
            if let Some(id) = id {
                tokens.all_special_ids.push(id);
            }
        }

        Ok(tokens)
    }

    /// Check if token ID is special.
    pub fn is_special(&self, id: TokenId) -> bool {
        self.all_special_ids.contains(&id)
    }
}

/// Encoding result from tokenization.
#[derive(Debug, Clone)]
pub struct EncodingResult {
    /// Token IDs.
    ids: Vec<TokenId>,

    /// Attention mask.
    attention_mask: Vec<u32>,

    /// Type IDs.
    type_ids: Vec<u32>,

    /// Offsets (character spans).
    offsets: Vec<(usize, usize)>,

    /// Original tokens (optional).
    tokens: Option<Vec<String>>,

    /// Word IDs (optional).
    word_ids: Option<Vec<Option<u32>>>,
}

impl EncodingResult {
    /// Create from HuggingFace encoding.
    pub fn from_encoding(encoding: Encoding) -> Self {
        Self {
            ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
            type_ids: encoding.get_type_ids().to_vec(),
            offsets: encoding.get_offsets().to_vec(),
            tokens: Some(encoding.get_tokens().to_vec()),
            word_ids: Some(encoding.get_word_ids().to_vec()),
        }
    }

    /// Get token IDs.
    pub fn ids(&self) -> &[TokenId] {
        &self.ids
    }

    /// Get attention mask.
    pub fn attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    /// Get type IDs.
    pub fn type_ids(&self) -> &[u32] {
        &self.type_ids
    }

    /// Get offsets.
    pub fn offsets(&self) -> &[(usize, usize)] {
        &self.offsets
    }

    /// Get tokens.
    pub fn tokens(&self) -> Option<&[String]> {
        self.tokens.as_deref()
    }

    /// Get word IDs.
    pub fn word_ids(&self) -> Option<&[Option<u32>]> {
        self.word_ids.as_deref()
    }

    /// Get sequence length.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Take ownership of IDs.
    pub fn into_ids(self) -> Vec<TokenId> {
        self.ids
    }
}

/// Streaming decoder for incremental text generation.
pub struct StreamingDecoder {
    /// Tokenizer reference.
    tokenizer: Arc<HfTokenizer>,

    /// Accumulated tokens.
    tokens: Vec<TokenId>,

    /// Last decoded position.
    last_pos: usize,

    /// Skip special tokens.
    skip_special: bool,
}

impl StreamingDecoder {
    /// Create new streaming decoder.
    pub fn new(tokenizer: Arc<HfTokenizer>, skip_special: bool) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            last_pos: 0,
            skip_special,
        }
    }

    /// Add token and get incremental text.
    pub fn add(&mut self, token_id: TokenId) -> Result<String, TokenizerError> {
        self.tokens.push(token_id);

        // Decode all tokens
        let full_text = self.tokenizer
            .decode(&self.tokens, self.skip_special)
            .map_err(|e| TokenizerError::DecodeError(e.to_string()))?;

        // Decode up to last position
        let prev_text = if self.last_pos > 0 {
            self.tokenizer
                .decode(&self.tokens[..self.last_pos], self.skip_special)
                .map_err(|e| TokenizerError::DecodeError(e.to_string()))?
        } else {
            String::new()
        };

        self.last_pos = self.tokens.len();

        // Return the difference
        Ok(full_text[prev_text.len()..].to_string())
    }

    /// Reset decoder.
    pub fn reset(&mut self) {
        self.tokens.clear();
        self.last_pos = 0;
    }

    /// Get all accumulated tokens.
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_config_default() {
        let config = TokenizerConfig::default();
        assert!(config.add_bos_token);
        assert!(!config.add_eos_token);
    }

    #[test]
    fn test_encoding_result() {
        let result = EncodingResult {
            ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            type_ids: vec![0, 0, 0],
            offsets: vec![(0, 1), (1, 2), (2, 3)],
            tokens: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            word_ids: None,
        };

        assert_eq!(result.len(), 3);
        assert!(!result.is_empty());
        assert_eq!(result.ids(), &[1, 2, 3]);
    }
}
