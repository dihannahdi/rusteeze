//! Chat template handling.
//!
//! This module provides chat template rendering for different model formats
//! including Llama, Mistral, ChatML, and custom templates.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::TokenizerError;

/// Role in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message.
    System,
    /// User message.
    User,
    /// Assistant message.
    Assistant,
    /// Tool/Function call.
    Tool,
    /// Function (deprecated, use Tool).
    Function,
}

impl Role {
    /// Get role name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Function => "function",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message role.
    pub role: Role,

    /// Message content.
    pub content: String,

    /// Optional name for the message author.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Create a new message.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
        }
    }

    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// Set name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Chat template format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplateFormat {
    /// Llama format.
    Llama,
    /// Llama 3 format.
    Llama3,
    /// Mistral format.
    Mistral,
    /// ChatML format.
    ChatML,
    /// Zephyr format.
    Zephyr,
    /// Vicuna format.
    Vicuna,
    /// Alpaca format.
    Alpaca,
    /// Phi-3 format.
    Phi3,
    /// Raw (no template).
    Raw,
}

impl ChatTemplateFormat {
    /// Auto-detect format from model name.
    pub fn from_model_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();

        if name_lower.contains("llama-3") || name_lower.contains("llama3") {
            Self::Llama3
        } else if name_lower.contains("llama") {
            Self::Llama
        } else if name_lower.contains("mistral") || name_lower.contains("mixtral") {
            Self::Mistral
        } else if name_lower.contains("zephyr") {
            Self::Zephyr
        } else if name_lower.contains("vicuna") {
            Self::Vicuna
        } else if name_lower.contains("alpaca") {
            Self::Alpaca
        } else if name_lower.contains("phi-3") || name_lower.contains("phi3") {
            Self::Phi3
        } else if name_lower.contains("chatml") {
            Self::ChatML
        } else {
            Self::ChatML // Default fallback
        }
    }
}

/// Chat template renderer.
pub struct ChatTemplate {
    /// Template format.
    format: ChatTemplateFormat,

    /// BOS token.
    bos_token: String,

    /// EOS token.
    eos_token: String,

    /// Custom Jinja template (optional).
    custom_template: Option<String>,
}

impl ChatTemplate {
    /// Create new chat template.
    pub fn new(format: ChatTemplateFormat) -> Self {
        let (bos, eos) = match format {
            ChatTemplateFormat::Llama | ChatTemplateFormat::Llama3 => ("<s>", "</s>"),
            ChatTemplateFormat::Mistral => ("<s>", "</s>"),
            ChatTemplateFormat::ChatML => ("", ""),
            ChatTemplateFormat::Zephyr => ("", "</s>"),
            ChatTemplateFormat::Vicuna => ("", "</s>"),
            ChatTemplateFormat::Alpaca => ("", ""),
            ChatTemplateFormat::Phi3 => ("", ""),
            ChatTemplateFormat::Raw => ("", ""),
        };

        Self {
            format,
            bos_token: bos.to_string(),
            eos_token: eos.to_string(),
            custom_template: None,
        }
    }

    /// Create with custom BOS/EOS tokens.
    pub fn with_tokens(format: ChatTemplateFormat, bos: &str, eos: &str) -> Self {
        Self {
            format,
            bos_token: bos.to_string(),
            eos_token: eos.to_string(),
            custom_template: None,
        }
    }

    /// Set custom template.
    pub fn with_custom_template(mut self, template: impl Into<String>) -> Self {
        self.custom_template = Some(template.into());
        self
    }

    /// Apply chat template to messages.
    pub fn apply(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        if let Some(ref template) = self.custom_template {
            return self.apply_custom_template(template, messages, add_generation_prompt);
        }

        match self.format {
            ChatTemplateFormat::Llama => self.apply_llama(messages, add_generation_prompt),
            ChatTemplateFormat::Llama3 => self.apply_llama3(messages, add_generation_prompt),
            ChatTemplateFormat::Mistral => self.apply_mistral(messages, add_generation_prompt),
            ChatTemplateFormat::ChatML => self.apply_chatml(messages, add_generation_prompt),
            ChatTemplateFormat::Zephyr => self.apply_zephyr(messages, add_generation_prompt),
            ChatTemplateFormat::Vicuna => self.apply_vicuna(messages, add_generation_prompt),
            ChatTemplateFormat::Alpaca => self.apply_alpaca(messages, add_generation_prompt),
            ChatTemplateFormat::Phi3 => self.apply_phi3(messages, add_generation_prompt),
            ChatTemplateFormat::Raw => self.apply_raw(messages),
        }
    }

    /// Apply Llama 2 template.
    fn apply_llama(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let mut result = String::new();
        let mut system_message = String::new();

        // Extract system message
        for msg in messages {
            if msg.role == Role::System {
                system_message = msg.content.clone();
                break;
            }
        }

        for (i, msg) in messages.iter().enumerate() {
            match msg.role {
                Role::System => continue, // Handled separately
                Role::User => {
                    result.push_str(&self.bos_token);
                    result.push_str("[INST] ");
                    if i == 0 || (i == 1 && !system_message.is_empty()) {
                        if !system_message.is_empty() {
                            result.push_str("<<SYS>>\n");
                            result.push_str(&system_message);
                            result.push_str("\n<</SYS>>\n\n");
                        }
                    }
                    result.push_str(&msg.content);
                    result.push_str(" [/INST]");
                }
                Role::Assistant => {
                    result.push(' ');
                    result.push_str(&msg.content);
                    result.push_str(&self.eos_token);
                }
                _ => {}
            }
        }

        if add_generation_prompt {
            // Already ends with [/INST] if last message is user
        }

        Ok(result)
    }

    /// Apply Llama 3 template.
    fn apply_llama3(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let mut result = String::new();
        result.push_str("<|begin_of_text|>");

        for msg in messages {
            result.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n", msg.role));
            result.push_str(&msg.content);
            result.push_str("<|eot_id|>");
        }

        if add_generation_prompt {
            result.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }

        Ok(result)
    }

    /// Apply Mistral template.
    fn apply_mistral(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let mut result = String::new();
        result.push_str(&self.bos_token);

        for msg in messages {
            match msg.role {
                Role::User => {
                    result.push_str("[INST] ");
                    result.push_str(&msg.content);
                    result.push_str(" [/INST]");
                }
                Role::Assistant => {
                    result.push_str(&msg.content);
                    result.push_str(&self.eos_token);
                }
                Role::System => {
                    // Mistral doesn't have explicit system role, prepend to first user message
                    result.push_str("[INST] ");
                    result.push_str(&msg.content);
                    result.push_str("\n\n");
                }
                _ => {}
            }
        }

        Ok(result)
    }

    /// Apply ChatML template.
    fn apply_chatml(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let mut result = String::new();

        for msg in messages {
            result.push_str(&format!("<|im_start|>{}\n", msg.role));
            result.push_str(&msg.content);
            result.push_str("<|im_end|>\n");
        }

        if add_generation_prompt {
            result.push_str("<|im_start|>assistant\n");
        }

        Ok(result)
    }

    /// Apply Zephyr template.
    fn apply_zephyr(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let mut result = String::new();

        for msg in messages {
            result.push_str(&format!("<|{}|>\n", msg.role));
            result.push_str(&msg.content);
            result.push_str(&self.eos_token);
            result.push('\n');
        }

        if add_generation_prompt {
            result.push_str("<|assistant|>\n");
        }

        Ok(result)
    }

    /// Apply Vicuna template.
    fn apply_vicuna(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let mut result = String::new();
        let mut has_system = false;

        for msg in messages {
            match msg.role {
                Role::System => {
                    result.push_str(&msg.content);
                    result.push_str("\n\n");
                    has_system = true;
                }
                Role::User => {
                    result.push_str("USER: ");
                    result.push_str(&msg.content);
                    result.push('\n');
                }
                Role::Assistant => {
                    result.push_str("ASSISTANT: ");
                    result.push_str(&msg.content);
                    result.push_str(&self.eos_token);
                    result.push('\n');
                }
                _ => {}
            }
        }

        if add_generation_prompt {
            result.push_str("ASSISTANT:");
        }

        Ok(result)
    }

    /// Apply Alpaca template.
    fn apply_alpaca(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let mut result = String::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    result.push_str(&msg.content);
                    result.push_str("\n\n");
                }
                Role::User => {
                    result.push_str("### Instruction:\n");
                    result.push_str(&msg.content);
                    result.push_str("\n\n");
                }
                Role::Assistant => {
                    result.push_str("### Response:\n");
                    result.push_str(&msg.content);
                    result.push_str("\n\n");
                }
                _ => {}
            }
        }

        if add_generation_prompt {
            result.push_str("### Response:\n");
        }

        Ok(result)
    }

    /// Apply Phi-3 template.
    fn apply_phi3(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let mut result = String::new();

        for msg in messages {
            result.push_str(&format!("<|{}|>\n", msg.role));
            result.push_str(&msg.content);
            result.push_str("<|end|>\n");
        }

        if add_generation_prompt {
            result.push_str("<|assistant|>\n");
        }

        Ok(result)
    }

    /// Apply raw template (just concatenate).
    fn apply_raw(&self, messages: &[Message]) -> Result<String, TokenizerError> {
        let mut result = String::new();
        for msg in messages {
            result.push_str(&msg.content);
            result.push('\n');
        }
        Ok(result)
    }

    /// Apply custom Jinja-like template.
    fn apply_custom_template(
        &self,
        _template: &str,
        messages: &[Message],
        _add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        // Simplified template rendering
        // For full Jinja support, would need minijinja crate
        // For now, fall back to ChatML format
        self.apply_chatml(messages, _add_generation_prompt)
    }
}

impl Default for ChatTemplate {
    fn default() -> Self {
        Self::new(ChatTemplateFormat::ChatML)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama3_template() {
        let template = ChatTemplate::new(ChatTemplateFormat::Llama3);
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hello!"),
        ];

        let result = template.apply(&messages, true).unwrap();
        assert!(result.contains("<|begin_of_text|>"));
        assert!(result.contains("system"));
        assert!(result.contains("user"));
        assert!(result.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn test_chatml_template() {
        let template = ChatTemplate::new(ChatTemplateFormat::ChatML);
        let messages = vec![
            Message::user("What is 2+2?"),
        ];

        let result = template.apply(&messages, true).unwrap();
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            ChatTemplateFormat::from_model_name("meta-llama/Llama-3.1-8B"),
            ChatTemplateFormat::Llama3
        );
        assert_eq!(
            ChatTemplateFormat::from_model_name("mistralai/Mistral-7B"),
            ChatTemplateFormat::Mistral
        );
    }
}
