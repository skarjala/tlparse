use clap::Parser;

use anyhow::{bail, Context};
use std::fs;
use std::path::PathBuf;

use tlparse::{parse_path, ParseConfig};

// Main output filename used by both single rank and multi-rank processing
const MAIN_OUTPUT_FILENAME: &str = "index.html";

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    path: PathBuf,
    /// Parse most recent log
    #[arg(long)]
    latest: bool,
    /// Output directory, defaults to `tl_out`
    #[arg(short, default_value = "tl_out")]
    out: PathBuf,
    /// Delete out directory if it already exists
    #[arg(long)]
    overwrite: bool,
    /// Return non-zero exit code if unrecognized log lines are found.  Mostly useful for unit
    /// testing.
    #[arg(long)]
    strict: bool,
    /// Return non-zero exit code if some log lines do not have associated compile id.  Used for
    /// unit testing
    #[arg(long)]
    strict_compile_id: bool,
    /// Don't open browser at the end
    #[arg(long)]
    no_browser: bool,
    /// Some custom HTML to append to the top of report
    #[arg(long, default_value = "")]
    custom_header_html: String,
    /// Be more chatty
    #[arg(short, long)]
    verbose: bool,
    /// Some parsers will write output as rendered html for prettier viewing.
    /// Enabiling this option will enforce output as plain text for easier diffing
    #[arg(short, long)]
    plain_text: bool,
    /// For export specific logs
    #[arg(short, long)]
    export: bool,
    /// For inductor provenance tracking highlighter
    #[arg(short, long)]
    inductor_provenance: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let path = if cli.latest {
        let input_path = &cli.path;
        // Path should be a directory
        if !input_path.is_dir() {
            bail!(
                "Input path {} is not a directory (required when using --latest)",
                input_path.display()
            );
        }

        let last_modified_file = std::fs::read_dir(&input_path)
            .with_context(|| format!("Couldn't access directory {}", input_path.display()))?
            .flatten()
            .filter(|f| f.metadata().unwrap().is_file())
            .max_by_key(|x| x.metadata().unwrap().modified().unwrap());

        let Some(last_modified_file) = last_modified_file else {
            bail!("No files found in directory {}", input_path.display());
        };
        last_modified_file.path()
    } else {
        cli.path.clone()
    };

    let out_path = cli.out.clone();
    if out_path.exists() {
        if !cli.overwrite {
            bail!(
                "Directory {} already exists, use -o OUTDIR to write to another location or pass --overwrite to overwrite the old contents",
                out_path.display()
            );
        }
        fs::remove_dir_all(&out_path)?;
    }

    // Use handle_one_rank for single rank processing
    handle_one_rank(&path, &out_path, &cli)?;

    if !cli.no_browser {
        opener::open(out_path.join(MAIN_OUTPUT_FILENAME))?;
    }
    Ok(())
}

// Helper function to handle parsing and writing output for a single rank
// Returns the relative path to the main output file within the rank directory
fn handle_one_rank(
    rank_path: &PathBuf,
    rank_out_dir: &PathBuf,
    cli: &Cli,
) -> anyhow::Result<PathBuf> {
    let config = ParseConfig {
        strict: cli.strict,
        strict_compile_id: cli.strict_compile_id,
        custom_parsers: Vec::new(),
        custom_header_html: cli.custom_header_html.clone(),
        verbose: cli.verbose,
        plain_text: cli.plain_text,
        export: cli.export,
        inductor_provenance: cli.inductor_provenance,
    };

    let output = parse_path(rank_path, config)?;

    let mut main_output_path = None;

    // Write output files to output directory
    for (filename, content) in output {
        let out_file = rank_out_dir.join(&filename);
        if let Some(dir) = out_file.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(out_file, content)?;

        // Track the main output file (typically index.html)
        if filename.file_name().and_then(|name| name.to_str()) == Some(MAIN_OUTPUT_FILENAME) {
            main_output_path = Some(filename);
        }
    }

    Ok(main_output_path.unwrap_or_else(|| PathBuf::from(MAIN_OUTPUT_FILENAME)))
}
