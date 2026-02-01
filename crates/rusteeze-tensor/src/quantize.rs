//! Quantization support for model weights.
//!
//! This module provides quantization utilities for reducing model memory
//! footprint and potentially improving inference speed.

use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

/// Quantization method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMethod {
    /// No quantization (full precision).
    None,
    /// 8-bit integer quantization.
    Int8,
    /// 4-bit integer quantization.
    Int4,
    /// FP8 E4M3 quantization.
    Fp8E4m3,
    /// FP8 E5M2 quantization.
    Fp8E5m2,
    /// GPTQ quantization.
    Gptq,
    /// AWQ quantization.
    Awq,
    /// GGML/GGUF quantization.
    Ggml,
}

/// Quantization configuration.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Quantization method.
    pub method: QuantMethod,

    /// Number of bits.
    pub bits: u8,

    /// Group size for grouped quantization.
    pub group_size: usize,

    /// Whether to quantize per-channel.
    pub per_channel: bool,

    /// Symmetric vs asymmetric quantization.
    pub symmetric: bool,

    /// Layers to exclude from quantization.
    pub exclude_layers: Vec<String>,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            method: QuantMethod::None,
            bits: 16,
            group_size: 128,
            per_channel: true,
            symmetric: true,
            exclude_layers: vec![],
        }
    }
}

impl QuantConfig {
    /// Create INT8 quantization config.
    pub fn int8() -> Self {
        Self {
            method: QuantMethod::Int8,
            bits: 8,
            ..Default::default()
        }
    }

    /// Create INT4 quantization config.
    pub fn int4() -> Self {
        Self {
            method: QuantMethod::Int4,
            bits: 4,
            group_size: 128,
            ..Default::default()
        }
    }

    /// Create FP8 quantization config.
    pub fn fp8() -> Self {
        Self {
            method: QuantMethod::Fp8E4m3,
            bits: 8,
            ..Default::default()
        }
    }

    /// Create GPTQ config.
    pub fn gptq(bits: u8, group_size: usize) -> Self {
        Self {
            method: QuantMethod::Gptq,
            bits,
            group_size,
            ..Default::default()
        }
    }

    /// Create AWQ config.
    pub fn awq(bits: u8, group_size: usize) -> Self {
        Self {
            method: QuantMethod::Awq,
            bits,
            group_size,
            ..Default::default()
        }
    }

    /// Exclude layers from quantization.
    pub fn exclude<S: Into<String>>(mut self, layers: Vec<S>) -> Self {
        self.exclude_layers = layers.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Set group size.
    pub fn with_group_size(mut self, group_size: usize) -> Self {
        self.group_size = group_size;
        self
    }
}

/// Quantization state for a tensor.
#[derive(Debug, Clone)]
pub struct QuantState {
    /// Scale factors.
    pub scales: Tensor,

    /// Zero points (for asymmetric quantization).
    pub zero_points: Option<Tensor>,

    /// Group size used.
    pub group_size: usize,

    /// Original shape.
    pub original_shape: Vec<usize>,

    /// Original dtype.
    pub original_dtype: DType,
}

/// Quantize a tensor to INT8.
pub fn quantize_int8(tensor: &Tensor, per_channel: bool) -> Result<(Tensor, QuantState)> {
    let device = tensor.device();
    let dtype = tensor.dtype();
    let shape = tensor.dims().to_vec();

    if per_channel {
        // Per-channel quantization along first dimension
        let num_channels = shape[0];
        let channel_size: usize = shape[1..].iter().product();

        // Compute per-channel min/max
        let flat = tensor.reshape((num_channels, channel_size))?;
        let max_vals = flat.max(1)?;
        let min_vals = flat.min(1)?;

        // Compute scale: (max - min) / 255
        let range = (&max_vals - &min_vals)?;
        let scales = (&range / 255.0)?;

        // Compute zero point
        let zero_points = (&min_vals.neg()? / &scales)?;

        // Quantize
        let scales_bc = scales.reshape((num_channels, 1))?;
        let zp_bc = zero_points.reshape((num_channels, 1))?;
        let quantized = ((flat / &scales_bc)? + &zp_bc)?;
        let quantized = quantized.clamp(0.0, 255.0)?;
        let quantized = quantized.to_dtype(DType::U8)?;
        let quantized = quantized.reshape(&shape)?;

        Ok((
            quantized,
            QuantState {
                scales,
                zero_points: Some(zero_points.to_dtype(DType::U8)?),
                group_size: 0,
                original_shape: shape,
                original_dtype: dtype,
            },
        ))
    } else {
        // Per-tensor quantization
        let max_val = tensor.max_all()?;
        let min_val = tensor.min_all()?;

        let max_f: f32 = max_val.to_scalar()?;
        let min_f: f32 = min_val.to_scalar()?;

        let scale = (max_f - min_f) / 255.0;
        let zero_point = (-min_f / scale).round() as u8;

        let scales = Tensor::new(&[scale], device)?;
        let zero_points = Tensor::new(&[zero_point], device)?;

        let quantized = ((tensor - min_val)? / scale)?;
        let quantized = quantized.clamp(0.0, 255.0)?;
        let quantized = quantized.to_dtype(DType::U8)?;

        Ok((
            quantized,
            QuantState {
                scales,
                zero_points: Some(zero_points),
                group_size: 0,
                original_shape: shape,
                original_dtype: dtype,
            },
        ))
    }
}

/// Dequantize INT8 tensor.
pub fn dequantize_int8(quantized: &Tensor, state: &QuantState) -> Result<Tensor> {
    let float = quantized.to_dtype(DType::F32)?;

    if let Some(ref zp) = state.zero_points {
        let zp_float = zp.to_dtype(DType::F32)?;
        let scales_float = state.scales.to_dtype(DType::F32)?;

        // Reshape for broadcasting if per-channel
        if scales_float.dims().len() == 1 && scales_float.dim(0)? > 1 {
            let num_channels = scales_float.dim(0)?;
            let scales_bc = scales_float.reshape((num_channels, 1))?;
            let zp_bc = zp_float.reshape((num_channels, 1))?;
            let flat_shape = vec![num_channels, state.original_shape[1..].iter().product()];
            let float_flat = float.reshape(&flat_shape)?;
            let dequant = ((float_flat - zp_bc)? * scales_bc)?;
            dequant.reshape(&state.original_shape)?.to_dtype(state.original_dtype)
        } else {
            let dequant = ((float - zp_float)? * scales_float)?;
            dequant.to_dtype(state.original_dtype)
        }
    } else {
        let dequant = (float * &state.scales)?;
        dequant.to_dtype(state.original_dtype)
    }
}

/// Quantized linear layer.
#[derive(Debug)]
pub struct QuantizedLinear {
    /// Quantized weights.
    pub weight: Tensor,

    /// Quantization state.
    pub state: QuantState,

    /// Optional bias (not quantized).
    pub bias: Option<Tensor>,

    /// Quantization method.
    pub method: QuantMethod,
}

impl QuantizedLinear {
    /// Create from a regular weight tensor.
    pub fn from_float(
        weight: &Tensor,
        bias: Option<&Tensor>,
        config: &QuantConfig,
    ) -> Result<Self> {
        let (quantized, state) = match config.method {
            QuantMethod::Int8 => quantize_int8(weight, config.per_channel)?,
            QuantMethod::Int4 => {
                // INT4 uses grouped quantization
                quantize_grouped(weight, 4, config.group_size)?
            }
            _ => {
                // Fallback to INT8
                quantize_int8(weight, config.per_channel)?
            }
        };

        Ok(Self {
            weight: quantized,
            state,
            bias: bias.cloned(),
            method: config.method,
        })
    }

    /// Forward pass with dequantization.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Dequantize weights on the fly
        let weight = match self.method {
            QuantMethod::Int8 => dequantize_int8(&self.weight, &self.state)?,
            QuantMethod::Int4 => dequantize_grouped(&self.weight, &self.state)?,
            _ => dequantize_int8(&self.weight, &self.state)?,
        };

        // Linear operation
        let result = x.matmul(&weight.t()?)?;

        if let Some(ref bias) = self.bias {
            result.broadcast_add(bias)
        } else {
            Ok(result)
        }
    }

    /// Get memory savings compared to FP16.
    pub fn memory_savings(&self) -> f64 {
        let original_bytes = self.state.original_shape.iter().product::<usize>() * 2; // FP16
        let quantized_bytes = match self.method {
            QuantMethod::Int8 => self.weight.elem_count(),
            QuantMethod::Int4 => self.weight.elem_count() / 2,
            _ => self.weight.elem_count(),
        };
        1.0 - (quantized_bytes as f64 / original_bytes as f64)
    }
}

/// Grouped quantization for INT4/AWQ.
pub fn quantize_grouped(
    tensor: &Tensor,
    bits: u8,
    group_size: usize,
) -> Result<(Tensor, QuantState)> {
    let device = tensor.device();
    let dtype = tensor.dtype();
    let shape = tensor.dims().to_vec();

    // Flatten to 2D for grouped quantization
    let (rows, cols) = if shape.len() == 2 {
        (shape[0], shape[1])
    } else {
        (shape[0], shape[1..].iter().product())
    };

    let flat = tensor.reshape((rows, cols))?;

    // Pad columns to be divisible by group_size
    let padded_cols = ((cols + group_size - 1) / group_size) * group_size;
    let flat = if padded_cols > cols {
        let padding = Tensor::zeros((rows, padded_cols - cols), dtype, device)?;
        Tensor::cat(&[&flat, &padding], 1)?
    } else {
        flat
    };

    let num_groups = padded_cols / group_size;
    let flat_grouped = flat.reshape((rows, num_groups, group_size))?;

    // Compute per-group scales
    let max_vals = flat_grouped.max(2)?;
    let min_vals = flat_grouped.min(2)?;
    let max_range = 2.0f32.powi(bits as i32) - 1.0;

    let range = (&max_vals - &min_vals)?;
    let scales = (&range / max_range)?;

    // Quantize
    let scales_bc = scales.unsqueeze(2)?;
    let min_bc = min_vals.unsqueeze(2)?;
    let quantized = ((flat_grouped - min_bc)? / &scales_bc)?;
    let quantized = quantized.clamp(0.0, max_range as f64)?;

    // Pack INT4 into bytes if needed
    let quantized = if bits == 4 {
        let quantized_u8 = quantized.to_dtype(DType::U8)?;
        let quantized_flat = quantized_u8.reshape((rows, padded_cols))?;
        // Pack two 4-bit values per byte
        pack_int4(&quantized_flat)?
    } else {
        quantized.to_dtype(DType::U8)?.reshape((rows, padded_cols))?
    };

    Ok((
        quantized,
        QuantState {
            scales: scales.reshape((rows, num_groups))?,
            zero_points: Some(min_vals.reshape((rows, num_groups))?),
            group_size,
            original_shape: shape,
            original_dtype: dtype,
        },
    ))
}

/// Dequantize grouped tensor.
pub fn dequantize_grouped(quantized: &Tensor, state: &QuantState) -> Result<Tensor> {
    let (rows, packed_cols) = quantized.dims2()?;
    let group_size = state.group_size;

    // Unpack if INT4
    let unpacked = if packed_cols * 2 > rows * state.original_shape.get(1).copied().unwrap_or(1) {
        unpack_int4(quantized)?
    } else {
        quantized.to_dtype(DType::F32)?
    };

    let (_, cols) = unpacked.dims2()?;
    let num_groups = cols / group_size;

    // Reshape for grouped dequantization
    let unpacked = unpacked.reshape((rows, num_groups, group_size))?;
    let scales = state.scales.unsqueeze(2)?;
    let zeros = state.zero_points.as_ref().unwrap().unsqueeze(2)?;

    // Dequantize: x = scale * quantized + zero
    let dequant = ((unpacked * &scales)? + zeros)?;
    
    // Reshape back to original
    let dequant = dequant.reshape((rows, cols))?;
    
    // Trim padding if needed
    let original_cols = state.original_shape.get(1).copied().unwrap_or(cols);
    let dequant = if cols > original_cols {
        dequant.narrow(1, 0, original_cols)?
    } else {
        dequant
    };

    dequant.reshape(&state.original_shape)?.to_dtype(state.original_dtype)
}

/// Pack two 4-bit values into one byte.
fn pack_int4(tensor: &Tensor) -> Result<Tensor> {
    let (rows, cols) = tensor.dims2()?;
    let packed_cols = cols / 2;
    let device = tensor.device();

    // Get tensor data
    let data: Vec<u8> = tensor.flatten_all()?.to_vec1()?;

    // Pack pairs of values
    let mut packed = Vec::with_capacity(rows * packed_cols);
    for r in 0..rows {
        for c in 0..packed_cols {
            let idx = r * cols + c * 2;
            let low = data[idx] & 0x0F;
            let high = (data[idx + 1] & 0x0F) << 4;
            packed.push(low | high);
        }
    }

    Tensor::from_vec(packed, (rows, packed_cols), device)
}

/// Unpack byte into two 4-bit values.
fn unpack_int4(tensor: &Tensor) -> Result<Tensor> {
    let (rows, packed_cols) = tensor.dims2()?;
    let cols = packed_cols * 2;
    let device = tensor.device();

    let data: Vec<u8> = tensor.flatten_all()?.to_vec1()?;

    // Unpack to floats
    let mut unpacked = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..packed_cols {
            let idx = r * packed_cols + c;
            let byte = data[idx];
            let low = (byte & 0x0F) as f32;
            let high = ((byte >> 4) & 0x0F) as f32;
            unpacked.push(low);
            unpacked.push(high);
        }
    }

    Tensor::from_vec(unpacked, (rows, cols), device)
}

/// GPTQ state for a layer.
#[derive(Debug)]
pub struct GptqState {
    /// Quantized weights (packed).
    pub qweight: Tensor,

    /// Zero points (packed).
    pub qzeros: Tensor,

    /// Scales.
    pub scales: Tensor,

    /// Optional bias.
    pub bias: Option<Tensor>,

    /// Group size.
    pub group_size: usize,

    /// Number of bits.
    pub bits: u8,
}

/// AWQ state for a layer.
#[derive(Debug)]
pub struct AwqState {
    /// Quantized weights.
    pub qweight: Tensor,

    /// Scales.
    pub scales: Tensor,

    /// Zero points.
    pub zeros: Tensor,

    /// Optional bias.
    pub bias: Option<Tensor>,

    /// Group size.
    pub group_size: usize,
}

/// Quantization statistics.
#[derive(Debug, Clone, Default)]
pub struct QuantStats {
    /// Number of layers quantized.
    pub layers_quantized: usize,

    /// Original size in bytes.
    pub original_size: usize,

    /// Quantized size in bytes.
    pub quantized_size: usize,

    /// Mean quantization error.
    pub mean_error: f64,

    /// Max quantization error.
    pub max_error: f64,
}

impl QuantStats {
    /// Get compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.quantized_size == 0 {
            0.0
        } else {
            self.original_size as f64 / self.quantized_size as f64
        }
    }

    /// Get memory savings percentage.
    pub fn memory_savings(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            1.0 - (self.quantized_size as f64 / self.original_size as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_quantization() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0.0f32, 1.0, (64, 128), &device)?;

        let (quantized, state) = quantize_int8(&tensor, true)?;
        assert_eq!(quantized.dtype(), DType::U8);

        let dequant = dequantize_int8(&quantized, &state)?;
        assert_eq!(dequant.dims(), tensor.dims());

        // Check error is reasonable
        let diff = (&tensor - &dequant)?.abs()?;
        let max_diff: f32 = diff.max_all()?.to_scalar()?;
        assert!(max_diff < 0.1, "Quantization error too large: {}", max_diff);

        Ok(())
    }

    #[test]
    fn test_grouped_quantization() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0.0f32, 1.0, (64, 128), &device)?;

        let (quantized, state) = quantize_grouped(&tensor, 4, 32)?;

        let dequant = dequantize_grouped(&quantized, &state)?;
        assert_eq!(dequant.dims(), tensor.dims());

        Ok(())
    }

    #[test]
    fn test_quant_config() {
        let config = QuantConfig::int4().with_group_size(64);
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 64);

        let config = QuantConfig::gptq(4, 128);
        assert_eq!(config.method, QuantMethod::Gptq);
    }

    #[test]
    fn test_pack_unpack_int4() -> Result<()> {
        let device = Device::Cpu;
        // Create test data with values 0-15
        let data: Vec<u8> = (0..8).collect();
        let tensor = Tensor::from_vec(data.clone(), (1, 8), &device)?;

        let packed = pack_int4(&tensor)?;
        assert_eq!(packed.dims(), &[1, 4]);

        let unpacked = unpack_int4(&packed)?;
        let unpacked_data: Vec<f32> = unpacked.flatten_all()?.to_vec1()?;
        
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(unpacked_data[i] as u8, v);
        }

        Ok(())
    }
}
