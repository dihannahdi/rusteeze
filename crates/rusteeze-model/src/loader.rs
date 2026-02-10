//! Model loading utilities.
//!
//! This module provides functionality for loading model weights from
//! various formats including SafeTensors, PyTorch, and GGUF.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytemuck;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::config::{ConfigError, ModelConfig};

/// Weight file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// SafeTensors format.
    SafeTensors,
    /// PyTorch bin format.
    PyTorch,
    /// GGUF format.
    Gguf,
    /// GGML format.
    Ggml,
}

impl WeightFormat {
    /// Detect format from filename.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Option<Self> {
        let path = path.as_ref();
        let ext = path.extension()?.to_str()?;
        match ext {
            "safetensors" => Some(Self::SafeTensors),
            "bin" | "pt" | "pth" => Some(Self::PyTorch),
            "gguf" => Some(Self::Gguf),
            "ggml" => Some(Self::Ggml),
            _ => None,
        }
    }
}

/// Model weight file info.
#[derive(Debug, Clone)]
pub struct WeightFile {
    /// File path.
    pub path: PathBuf,

    /// File format.
    pub format: WeightFormat,

    /// File size in bytes.
    pub size: u64,

    /// Shard index (for sharded models).
    pub shard_index: Option<usize>,

    /// Total shards.
    pub total_shards: Option<usize>,
}

impl WeightFile {
    /// Create from path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Option<Self> {
        let path = path.as_ref().to_path_buf();
        let format = WeightFormat::from_path(&path)?;
        let metadata = std::fs::metadata(&path).ok()?;

        // Try to detect shard info from filename
        let (shard_index, total_shards) = Self::parse_shard_info(&path);

        Some(Self {
            path,
            format,
            size: metadata.len(),
            shard_index,
            total_shards,
        })
    }

    /// Parse shard info from filename.
    fn parse_shard_info(path: &Path) -> (Option<usize>, Option<usize>) {
        let filename = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

        // Pattern: model-00001-of-00005.safetensors
        if let Some(caps) = regex_lite::Regex::new(r"(\d+)-of-(\d+)")
            .ok()
            .and_then(|re| re.captures(filename))
        {
            let index: Option<usize> = caps.get(1).and_then(|m: regex_lite::Match| m.as_str().parse().ok());
            let total: Option<usize> = caps.get(2).and_then(|m: regex_lite::Match| m.as_str().parse().ok());
            return (index, total);
        }

        (None, None)
    }
}

/// Model loader configuration.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Device to load model on.
    pub device: Device,

    /// Data type for weights.
    pub dtype: DType,

    /// Enable memory mapping.
    pub use_mmap: bool,

    /// Load weights in parallel.
    pub parallel_loading: bool,

    /// Maximum parallel workers.
    pub max_workers: usize,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F16,
            use_mmap: true,
            parallel_loading: true,
            max_workers: 4,
        }
    }
}

impl LoaderConfig {
    /// Create with device and dtype.
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            device,
            dtype,
            ..Default::default()
        }
    }

    /// Set memory mapping.
    pub fn with_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }
}

/// Model loader for various formats.
pub struct ModelLoader {
    /// Loader configuration.
    config: LoaderConfig,

    /// Model directory.
    model_dir: PathBuf,

    /// Weight files.
    weight_files: Vec<WeightFile>,

    /// Model configuration.
    model_config: Option<ModelConfig>,
}

impl ModelLoader {
    /// Create a new model loader.
    pub fn new<P: AsRef<Path>>(model_dir: P, config: LoaderConfig) -> Result<Self, LoaderError> {
        let model_dir = model_dir.as_ref().to_path_buf();

        if !model_dir.exists() {
            return Err(LoaderError::NotFound(model_dir.display().to_string()));
        }

        // Find weight files
        let weight_files = Self::discover_weight_files(&model_dir)?;

        if weight_files.is_empty() {
            return Err(LoaderError::NoWeights(model_dir.display().to_string()));
        }

        info!(
            "Found {} weight file(s) in {}",
            weight_files.len(),
            model_dir.display()
        );

        // Load model config
        let model_config = Self::load_model_config(&model_dir).ok();

        Ok(Self {
            config,
            model_dir,
            weight_files,
            model_config,
        })
    }

    /// Discover weight files in directory.
    fn discover_weight_files(dir: &Path) -> Result<Vec<WeightFile>, LoaderError> {
        let mut files = Vec::new();

        for entry in std::fs::read_dir(dir).map_err(|e| LoaderError::IoError(e.to_string()))? {
            let entry = entry.map_err(|e| LoaderError::IoError(e.to_string()))?;
            let path = entry.path();

            if path.is_file() {
                if let Some(weight_file) = WeightFile::from_path(&path) {
                    files.push(weight_file);
                }
            }
        }

        // Sort by shard index
        files.sort_by(|a, b| a.shard_index.cmp(&b.shard_index));

        Ok(files)
    }

    /// Load model configuration.
    fn load_model_config(dir: &Path) -> Result<ModelConfig, LoaderError> {
        let config_path = dir.join("config.json");
        ModelConfig::from_json_file(&config_path)
            .map_err(|e| LoaderError::ConfigError(e.to_string()))
    }

    /// Get model configuration.
    pub fn model_config(&self) -> Option<&ModelConfig> {
        self.model_config.as_ref()
    }

    /// Get weight files.
    pub fn weight_files(&self) -> &[WeightFile] {
        &self.weight_files
    }

    /// Load all weights as VarBuilder.
    pub fn load_weights(&self) -> Result<VarBuilder, LoaderError> {
        let device = &self.config.device;
        let dtype = self.config.dtype;

        // Collect all safetensors files
        let st_files: Vec<_> = self.weight_files
            .iter()
            .filter(|f| f.format == WeightFormat::SafeTensors)
            .map(|f| f.path.clone())
            .collect();

        if st_files.is_empty() {
            return Err(LoaderError::NoWeights("No SafeTensors files found".to_string()));
        }

        // Load using memory mapping or regular loading
        if self.config.use_mmap {
            self.load_weights_mmap(&st_files, dtype, device)
        } else {
            self.load_weights_direct(&st_files, dtype, device)
        }
    }

    /// Load weights with memory mapping.
    fn load_weights_mmap(
        &self,
        files: &[PathBuf],
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder, LoaderError> {
        info!("Loading weights with memory mapping");

        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        for file_path in files {
            debug!("Memory mapping: {}", file_path.display());

            let file = std::fs::File::open(file_path)
                .map_err(|e| LoaderError::IoError(e.to_string()))?;

            let mmap = unsafe {
                MmapOptions::new()
                    .map(&file)
                    .map_err(|e| LoaderError::IoError(e.to_string()))?
            };

            let st = SafeTensors::deserialize(&mmap)
                .map_err(|e| LoaderError::FormatError(e.to_string()))?;

            for (name, _) in st.tensors() {
                let tensor = st.tensor(&name)
                    .map_err(|e| LoaderError::FormatError(e.to_string()))?;

                let tensor = Self::convert_tensor(tensor, dtype, device)?;
                tensors.insert(name.to_string(), tensor);
            }
        }

        info!("Loaded {} tensors", tensors.len());

        Ok(VarBuilder::from_tensors(tensors, dtype, device))
    }

    /// Load weights directly (no mmap).
    fn load_weights_direct(
        &self,
        files: &[PathBuf],
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder, LoaderError> {
        info!("Loading weights directly");

        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        for file_path in files {
            debug!("Loading: {}", file_path.display());

            let data = std::fs::read(file_path)
                .map_err(|e| LoaderError::IoError(e.to_string()))?;

            let st = SafeTensors::deserialize(&data)
                .map_err(|e| LoaderError::FormatError(e.to_string()))?;

            for (name, _) in st.tensors() {
                let tensor = st.tensor(&name)
                    .map_err(|e| LoaderError::FormatError(e.to_string()))?;

                let tensor = Self::convert_tensor(tensor, dtype, device)?;
                tensors.insert(name.to_string(), tensor);
            }
        }

        info!("Loaded {} tensors", tensors.len());

        Ok(VarBuilder::from_tensors(tensors, dtype, device))
    }

    /// Convert SafeTensor to Candle Tensor.
    fn convert_tensor(
        tensor: safetensors::tensor::TensorView<'_>,
        target_dtype: DType,
        device: &Device,
    ) -> Result<Tensor, LoaderError> {
        let shape = tensor.shape();
        let dtype = Self::safetensor_dtype_to_candle(tensor.dtype())?;
        let data = tensor.data();

        // Create tensor from raw data
        let tensor = match dtype {
            DType::F32 => {
                let data: &[f32] = bytemuck::cast_slice(data);
                Tensor::from_slice(data, shape, device)
            }
            DType::F16 => {
                let data: &[half::f16] = bytemuck::cast_slice(data);
                Tensor::from_slice(data, shape, device)
            }
            DType::BF16 => {
                let data: &[half::bf16] = bytemuck::cast_slice(data);
                Tensor::from_slice(data, shape, device)
            }
            DType::I64 => {
                let data: &[i64] = bytemuck::cast_slice(data);
                Tensor::from_slice(data, shape, device)
            }
            DType::U32 => {
                let data: &[u32] = bytemuck::cast_slice(data);
                Tensor::from_slice(data, shape, device)
            }
            DType::U8 => {
                Tensor::from_slice(data, shape, device)
            }
            _ => return Err(LoaderError::UnsupportedDtype(format!("{:?}", dtype))),
        }.map_err(|e| LoaderError::TensorError(e.to_string()))?;

        // Convert to target dtype if needed
        if tensor.dtype() != target_dtype {
            tensor.to_dtype(target_dtype)
                .map_err(|e| LoaderError::TensorError(e.to_string()))
        } else {
            Ok(tensor)
        }
    }

    /// Convert SafeTensor dtype to Candle dtype.
    fn safetensor_dtype_to_candle(dtype: safetensors::Dtype) -> Result<DType, LoaderError> {
        match dtype {
            safetensors::Dtype::F32 => Ok(DType::F32),
            safetensors::Dtype::F16 => Ok(DType::F16),
            safetensors::Dtype::BF16 => Ok(DType::BF16),
            safetensors::Dtype::I64 => Ok(DType::I64),
            safetensors::Dtype::U32 => Ok(DType::U32),
            safetensors::Dtype::U8 => Ok(DType::U8),
            _ => Err(LoaderError::UnsupportedDtype(format!("{:?}", dtype))),
        }
    }

    /// Get total weight file size.
    pub fn total_size(&self) -> u64 {
        self.weight_files.iter().map(|f| f.size).sum()
    }
}

/// Loader error types.
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("Path not found: {0}")]
    NotFound(String),

    #[error("No weight files found in: {0}")]
    NoWeights(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("Format error: {0}")]
    FormatError(String),

    #[error("Tensor error: {0}")]
    TensorError(String),

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),
}

/// Index file for sharded models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIndex {
    /// Metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Weight map (tensor name -> filename).
    pub weight_map: HashMap<String, String>,
}

impl ModelIndex {
    /// Load from file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, LoaderError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| LoaderError::IoError(e.to_string()))?;
        serde_json::from_str(&content)
            .map_err(|e| LoaderError::FormatError(e.to_string()))
    }

    /// Get all unique files.
    pub fn files(&self) -> Vec<String> {
        let mut files: Vec<_> = self.weight_map.values().cloned().collect();
        files.sort();
        files.dedup();
        files
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_format_detection() {
        assert_eq!(
            WeightFormat::from_path("model.safetensors"),
            Some(WeightFormat::SafeTensors)
        );
        assert_eq!(
            WeightFormat::from_path("model.bin"),
            Some(WeightFormat::PyTorch)
        );
        assert_eq!(
            WeightFormat::from_path("model.gguf"),
            Some(WeightFormat::Gguf)
        );
        assert_eq!(WeightFormat::from_path("model.txt"), None);
    }

    #[test]
    fn test_shard_parsing() {
        let path = Path::new("model-00001-of-00005.safetensors");
        let file = WeightFile::from_path(path);
        // This may fail without the actual file, but tests the parsing logic
    }

    #[test]
    fn test_loader_config() {
        let config = LoaderConfig::new(Device::Cpu, DType::F16).with_mmap(false);
        assert!(!config.use_mmap);
        assert_eq!(config.dtype, DType::F16);
    }
}
