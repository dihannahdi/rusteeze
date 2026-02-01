//! Device abstraction for CPU and GPU operations.
//!
//! This module provides a unified interface for managing compute devices,
//! including CPUs and GPUs (CUDA, Metal, etc.).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a compute device for tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    /// CPU device
    Cpu,
    /// CUDA GPU device
    Cuda,
    /// Metal GPU device (macOS)
    Metal,
    /// Vulkan GPU device
    Vulkan,
}

impl Default for DeviceType {
    fn default() -> Self {
        Self::Cpu
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Cuda => write!(f, "cuda"),
            DeviceType::Metal => write!(f, "metal"),
            DeviceType::Vulkan => write!(f, "vulkan"),
        }
    }
}

impl std::str::FromStr for DeviceType {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(DeviceType::Cpu),
            "cuda" | "gpu" => Ok(DeviceType::Cuda),
            "metal" | "mps" => Ok(DeviceType::Metal),
            "vulkan" => Ok(DeviceType::Vulkan),
            _ => Err(crate::error::Error::config(format!(
                "Unknown device type: {s}. Valid options: cpu, cuda, metal, vulkan"
            ))),
        }
    }
}

/// A specific device instance with its type and ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Device {
    /// The type of device
    pub device_type: DeviceType,
    /// The ordinal/index of the device (0 for first GPU, etc.)
    pub ordinal: usize,
}

impl Device {
    /// Create a new CPU device.
    pub const fn cpu() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            ordinal: 0,
        }
    }

    /// Create a new CUDA device with the specified ordinal.
    pub const fn cuda(ordinal: usize) -> Self {
        Self {
            device_type: DeviceType::Cuda,
            ordinal,
        }
    }

    /// Create a new Metal device with the specified ordinal.
    pub const fn metal(ordinal: usize) -> Self {
        Self {
            device_type: DeviceType::Metal,
            ordinal,
        }
    }

    /// Check if this is a CPU device.
    pub const fn is_cpu(&self) -> bool {
        matches!(self.device_type, DeviceType::Cpu)
    }

    /// Check if this is a GPU device.
    pub const fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }

    /// Check if this is a CUDA device.
    pub const fn is_cuda(&self) -> bool {
        matches!(self.device_type, DeviceType::Cuda)
    }

    /// Check if this is a Metal device.
    pub const fn is_metal(&self) -> bool {
        matches!(self.device_type, DeviceType::Metal)
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.device_type {
            DeviceType::Cpu => write!(f, "cpu"),
            _ => write!(f, "{}:{}", self.device_type, self.ordinal),
        }
    }
}

impl std::str::FromStr for Device {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();

        if s == "cpu" {
            return Ok(Self::cpu());
        }

        // Parse "cuda:0" or "metal:0" format
        if let Some((device_type, ordinal)) = s.split_once(':') {
            let device_type: DeviceType = device_type.parse()?;
            let ordinal: usize = ordinal.parse().map_err(|_| {
                crate::error::Error::config(format!("Invalid device ordinal: {ordinal}"))
            })?;
            return Ok(Self {
                device_type,
                ordinal,
            });
        }

        // Default ordinal to 0
        let device_type: DeviceType = s.parse()?;
        Ok(Self {
            device_type,
            ordinal: 0,
        })
    }
}

/// Information about a compute device's capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// The device identifier
    pub device: Device,
    /// Human-readable name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: u64,
    /// Free memory in bytes
    pub free_memory: u64,
    /// Compute capability (for CUDA devices)
    pub compute_capability: Option<(u32, u32)>,
    /// Number of streaming multiprocessors (for CUDA)
    pub num_sms: Option<u32>,
    /// Maximum threads per block
    pub max_threads_per_block: Option<u32>,
    /// Whether the device supports FP16
    pub supports_fp16: bool,
    /// Whether the device supports BF16
    pub supports_bf16: bool,
    /// Whether the device supports Flash Attention
    pub supports_flash_attention: bool,
}

impl DeviceInfo {
    /// Create a CPU device info
    pub fn cpu() -> Self {
        let total_memory = sys_info::mem_info()
            .map(|m| m.total * 1024)
            .unwrap_or(0);
        let free_memory = sys_info::mem_info()
            .map(|m| m.avail * 1024)
            .unwrap_or(0);

        Self {
            device: Device::cpu(),
            name: "CPU".to_string(),
            total_memory,
            free_memory,
            compute_capability: None,
            num_sms: None,
            max_threads_per_block: None,
            supports_fp16: true,
            supports_bf16: true,
            supports_flash_attention: false,
        }
    }

    /// Get memory utilization as a percentage
    pub fn memory_utilization(&self) -> f64 {
        if self.total_memory == 0 {
            return 0.0;
        }
        let used = self.total_memory.saturating_sub(self.free_memory);
        (used as f64 / self.total_memory as f64) * 100.0
    }
}

/// System-level information for memory
mod sys_info {
    /// Memory information
    pub struct MemInfo {
        pub total: u64,
        pub avail: u64,
    }

    /// Get system memory information
    #[cfg(target_os = "linux")]
    pub fn mem_info() -> Option<MemInfo> {
        use std::fs;
        let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
        let mut total = 0u64;
        let mut avail = 0u64;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total = line.split_whitespace().nth(1)?.parse().ok()?;
            } else if line.starts_with("MemAvailable:") {
                avail = line.split_whitespace().nth(1)?.parse().ok()?;
            }
        }

        Some(MemInfo { total, avail })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn mem_info() -> Option<MemInfo> {
        // Fallback for non-Linux systems
        Some(MemInfo {
            total: 16 * 1024 * 1024, // 16GB default
            avail: 8 * 1024 * 1024,  // 8GB default
        })
    }
}

/// Data type for tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    /// 32-bit floating point
    #[default]
    F32,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 64-bit floating point
    F64,
    /// 8-bit unsigned integer
    U8,
    /// 8-bit signed integer
    I8,
    /// 32-bit unsigned integer
    U32,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
}

impl DType {
    /// Get the size of this data type in bytes
    pub const fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::U32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F64 | DType::I64 => 8,
            DType::U8 | DType::I8 => 1,
        }
    }

    /// Check if this is a floating point type
    pub const fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16 | DType::F64)
    }

    /// Check if this is an integer type
    pub const fn is_integer(&self) -> bool {
        !self.is_float()
    }

    /// Check if this is a half-precision type
    pub const fn is_half(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F64 => write!(f, "f64"),
            DType::U8 => write!(f, "u8"),
            DType::I8 => write!(f, "i8"),
            DType::U32 => write!(f, "u32"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
        }
    }
}

impl std::str::FromStr for DType {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "f32" | "float32" | "float" => Ok(DType::F32),
            "f16" | "float16" | "half" => Ok(DType::F16),
            "bf16" | "bfloat16" => Ok(DType::BF16),
            "f64" | "float64" | "double" => Ok(DType::F64),
            "u8" | "uint8" => Ok(DType::U8),
            "i8" | "int8" => Ok(DType::I8),
            "u32" | "uint32" => Ok(DType::U32),
            "i32" | "int32" | "int" => Ok(DType::I32),
            "i64" | "int64" | "long" => Ok(DType::I64),
            _ => Err(crate::error::Error::config(format!(
                "Unknown data type: {s}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_parsing() {
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::cpu());
        assert_eq!("cuda:0".parse::<Device>().unwrap(), Device::cuda(0));
        assert_eq!("cuda:1".parse::<Device>().unwrap(), Device::cuda(1));
        assert_eq!("metal:0".parse::<Device>().unwrap(), Device::metal(0));
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::cpu().to_string(), "cpu");
        assert_eq!(Device::cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::metal(1).to_string(), "metal:1");
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::BF16.size_in_bytes(), 2);
        assert_eq!(DType::I64.size_in_bytes(), 8);
    }

    #[test]
    fn test_dtype_parsing() {
        assert_eq!("f32".parse::<DType>().unwrap(), DType::F32);
        assert_eq!("bf16".parse::<DType>().unwrap(), DType::BF16);
        assert_eq!("float16".parse::<DType>().unwrap(), DType::F16);
    }
}
