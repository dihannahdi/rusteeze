//! Progress bar utilities.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::time::Duration;

/// Progress bar style presets.
pub struct ProgressStyles;

impl ProgressStyles {
    /// Standard progress bar.
    pub fn standard() -> ProgressStyle {
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-")
    }

    /// Download progress bar.
    pub fn download() -> ProgressStyle {
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .unwrap()
            .progress_chars("#>-")
    }

    /// Spinner.
    pub fn spinner() -> ProgressStyle {
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap()
    }

    /// Inference progress.
    pub fn inference() -> ProgressStyle {
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tokens ({per_sec})")
            .unwrap()
            .progress_chars("=>-")
    }
}

/// Create a standard progress bar.
pub fn create_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(ProgressStyles::standard());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create a download progress bar.
pub fn create_download_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(ProgressStyles::download());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create a spinner.
pub fn create_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyles::spinner());
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Progress tracker for multiple operations.
pub struct ProgressTracker {
    multi: MultiProgress,
    bars: Vec<ProgressBar>,
}

impl ProgressTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self {
            multi: MultiProgress::new(),
            bars: Vec::new(),
        }
    }

    /// Add a progress bar.
    pub fn add_bar(&mut self, total: u64, style: ProgressStyle) -> &ProgressBar {
        let pb = self.multi.add(ProgressBar::new(total));
        pb.set_style(style);
        pb.enable_steady_tick(Duration::from_millis(100));
        self.bars.push(pb);
        self.bars.last().unwrap()
    }

    /// Add a standard bar.
    pub fn add_standard(&mut self, total: u64) -> &ProgressBar {
        self.add_bar(total, ProgressStyles::standard())
    }

    /// Add a download bar.
    pub fn add_download(&mut self, total: u64) -> &ProgressBar {
        self.add_bar(total, ProgressStyles::download())
    }

    /// Finish all bars.
    pub fn finish_all(&self) {
        for bar in &self.bars {
            bar.finish();
        }
    }

    /// Get the multi-progress instance.
    pub fn multi(&self) -> &MultiProgress {
        &self.multi
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII progress bar that finishes on drop.
pub struct ScopedProgress {
    bar: ProgressBar,
    finish_message: Option<String>,
}

impl ScopedProgress {
    /// Create a new scoped progress bar.
    pub fn new(total: u64) -> Self {
        Self {
            bar: create_progress_bar(total),
            finish_message: None,
        }
    }

    /// Create with spinner.
    pub fn spinner(message: &str) -> Self {
        Self {
            bar: create_spinner(message),
            finish_message: None,
        }
    }

    /// Set finish message.
    pub fn with_finish_message(mut self, msg: impl Into<String>) -> Self {
        self.finish_message = Some(msg.into());
        self
    }

    /// Get the progress bar.
    pub fn bar(&self) -> &ProgressBar {
        &self.bar
    }

    /// Increment progress.
    pub fn inc(&self, delta: u64) {
        self.bar.inc(delta);
    }

    /// Set position.
    pub fn set_position(&self, pos: u64) {
        self.bar.set_position(pos);
    }

    /// Set message.
    pub fn set_message(&self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        self.bar.set_message(msg);
    }
}

impl Drop for ScopedProgress {
    fn drop(&mut self) {
        if let Some(ref msg) = self.finish_message {
            self.bar.finish_with_message(msg.clone());
        } else {
            self.bar.finish();
        }
    }
}
