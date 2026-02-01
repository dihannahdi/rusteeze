//! Output formatting utilities.

use colored::Colorize;

/// Format a table.
pub fn format_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    if rows.is_empty() {
        return String::new();
    }

    // Calculate column widths
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();

    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.len());
            }
        }
    }

    let mut output = String::new();

    // Header
    let header_line: String = headers
        .iter()
        .enumerate()
        .map(|(i, h)| format!("{:width$}", h, width = widths[i]))
        .collect::<Vec<_>>()
        .join(" │ ");
    output.push_str(&format!("{}\n", header_line.bright_cyan().bold()));

    // Separator
    let sep: String = widths
        .iter()
        .map(|w| "─".repeat(*w))
        .collect::<Vec<_>>()
        .join("─┼─");
    output.push_str(&format!("{}\n", sep));

    // Rows
    for row in rows {
        let row_line: String = row
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let width = widths.get(i).copied().unwrap_or(0);
                format!("{:width$}", cell, width = width)
            })
            .collect::<Vec<_>>()
            .join(" │ ");
        output.push_str(&format!("{}\n", row_line));
    }

    output
}

/// Format a key-value list.
pub fn format_kv_list(items: &[(&str, String)]) -> String {
    let max_key_len = items.iter().map(|(k, _)| k.len()).max().unwrap_or(0);

    items
        .iter()
        .map(|(k, v)| {
            format!(
                "  {}: {}",
                format!("{:width$}", k, width = max_key_len).bright_cyan(),
                v
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format bytes as human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration.
pub fn format_duration(seconds: f64) -> String {
    if seconds < 0.001 {
        format!("{:.2} µs", seconds * 1_000_000.0)
    } else if seconds < 1.0 {
        format!("{:.2} ms", seconds * 1000.0)
    } else if seconds < 60.0 {
        format!("{:.2} s", seconds)
    } else if seconds < 3600.0 {
        format!("{:.1} min", seconds / 60.0)
    } else {
        format!("{:.1} h", seconds / 3600.0)
    }
}

/// Format number with commas.
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();

    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }

    result
}

/// Format percentage.
pub fn format_percentage(value: f64) -> String {
    format!("{:.1}%", value * 100.0)
}

/// Print a section header.
pub fn print_section(title: &str) {
    println!("\n{}", title.bright_green().bold());
    println!("{}", "─".repeat(title.len()).bright_green());
}

/// Print a success message.
pub fn print_success(message: &str) {
    println!("{} {}", "✓".bright_green(), message);
}

/// Print an error message.
pub fn print_error(message: &str) {
    println!("{} {}", "✗".bright_red(), message);
}

/// Print a warning message.
pub fn print_warning(message: &str) {
    println!("{} {}", "⚠".bright_yellow(), message);
}

/// Print an info message.
pub fn print_info(message: &str) {
    println!("{} {}", "ℹ".bright_blue(), message);
}
