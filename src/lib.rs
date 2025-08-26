use anyhow::{anyhow, bail};
use chrono::Datelike;
use fxhash::{FxHashMap, FxHashSet};
use md5::{Digest, Md5};
use std::ffi::{OsStr, OsString};

use html_escape::encode_text;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use regex::Regex;
use serde_json::Value;
use std::cell::RefCell;
use std::fs::{self, File};
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::time::Instant;
use tinytemplate::TinyTemplate;

use crate::parsers::default_parsers;
use crate::parsers::ParserOutput;
use crate::parsers::StructuredLogParser;
use crate::templates::*;
use crate::types::*;
pub mod parsers;
mod templates;
mod types;
pub mod execution_order {
    use fxhash::FxHashMap;

    /// Issue types detected at a given execution index across ranks
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ExecOrderIssue {
        ScheduleMismatch,
        CacheSkew,
    }

    /// One row in the execution-order report
    #[derive(Debug, Clone)]
    pub struct ExecOrderIndexRow {
        /// Zero-based index into per-rank execution orders
        pub index: usize,
        /// Mapping: rank -> compile_id (compile directory name)
        pub by_rank: FxHashMap<u32, String>,
        /// Issues found for this index
        pub issues: Vec<ExecOrderIssue>,
    }

    /// Final report for execution-order diagnostics
    #[derive(Debug, Clone, Default)]
    pub struct ExecOrderReport {
        pub by_index: Vec<ExecOrderIndexRow>,
    }

    /// Analyze per-rank execution orders, aligning entries by index and flagging issues
    /// using provided per-graph properties.
    ///
    /// - `exec_orders`: rank -> ordered list of compile_id directory names
    /// - `collective_schedule_by_graph`: (rank, compile_id) -> schedule ops sequence
    /// - `cache_status_by_graph`: (rank, compile_id) -> cache status marker (e.g., "✅", "❌", "❓")
    ///
    /// Notes:
    /// - O(K·N) pass over data; uses internal memoization for property lookups
    /// - Evaluates comparisons only among present ranks at each index
    pub fn analyze_execution_order(
        exec_orders: &FxHashMap<u32, Vec<String>>,
        collective_schedule_by_graph: &FxHashMap<(u32, String), Vec<String>>,
        cache_status_by_graph: &FxHashMap<(u32, String), String>,
    ) -> ExecOrderReport {
        // Determine max length across ranks (N)
        let max_len = exec_orders.values().map(|v| v.len()).max().unwrap_or(0);

        // Fast no-op
        if max_len == 0 || exec_orders.is_empty() {
            return ExecOrderReport::default();
        }

        // Memoize property lookups per (rank, compile_id)
        let mut sched_memo: FxHashMap<(u32, String), Option<Vec<String>>> = FxHashMap::default();
        let mut cache_memo: FxHashMap<(u32, String), Option<String>> = FxHashMap::default();

        let mut rows: Vec<ExecOrderIndexRow> = Vec::with_capacity(max_len);

        for idx in 0..max_len {
            // Gather present ranks and their compile_ids at this index
            let mut by_rank: FxHashMap<u32, String> = FxHashMap::default();
            for (&rank, order) in exec_orders.iter() {
                if let Some(cid) = order.get(idx) {
                    by_rank.insert(rank, cid.clone());
                }
            }

            if by_rank.is_empty() {
                continue;
            }

            // Evaluate issues among present ranks
            let mut issues: Vec<ExecOrderIssue> = Vec::new();

            // Schedule mismatch: compare collective op sequences
            {
                // Collect schedules for ranks that have them (clone to avoid borrowing issues)
                let mut schedules: Vec<Vec<String>> = Vec::new();
                for (&rank, cid) in by_rank.iter() {
                    let key = (rank, cid.clone());
                    let entry = sched_memo.entry((rank, cid.clone())).or_insert_with(|| {
                        collective_schedule_by_graph
                            .get(&key)
                            .cloned()
                            .map(Some)
                            .unwrap_or(None)
                    });
                    if let Some(ref ops) = entry {
                        schedules.push(ops.clone());
                    }
                }
                if schedules.len() >= 2 {
                    let first = &schedules[0];
                    if schedules.iter().any(|s| s != first) {
                        issues.push(ExecOrderIssue::ScheduleMismatch);
                    }
                }
            }

            // Cache skew: compare cache status markers
            {
                let mut statuses: Vec<String> = Vec::new();
                for (&rank, cid) in by_rank.iter() {
                    let key = (rank, cid.clone());
                    let entry = cache_memo.entry((rank, cid.clone())).or_insert_with(|| {
                        cache_status_by_graph
                            .get(&key)
                            .cloned()
                            .map(Some)
                            .unwrap_or(None)
                    });
                    if let Some(ref s) = entry {
                        if !s.is_empty() {
                            statuses.push(s.clone());
                        }
                    }
                }
                if statuses.len() >= 2 {
                    let first = &statuses[0];
                    if statuses.iter().any(|s| s != first) {
                        issues.push(ExecOrderIssue::CacheSkew);
                    }
                }
            }

            rows.push(ExecOrderIndexRow { index: idx, by_rank, issues });
        }

        ExecOrderReport { by_index: rows }
    }

    /// Parse the JSON payload for `artifact.name = "graph_execution"` and extract
    /// the `graph_execution_order` sequence. The payload shape is expected to be:
    /// { "graph_execution_order": [ "<compile_id_dir>", ... ] }
    /// This function is tolerant to variations where entries might be objects with a
    /// `graph`/`compile_id`/`id` string field; in such cases it prefers `graph`, then
    /// `compile_id`, then `id`.
    pub fn parse_graph_execution_order(payload: &str) -> anyhow::Result<Vec<String>> {
        let value: serde_json::Value = serde_json::from_str(payload)?;
        let arr = value
            .get("graph_execution_order")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("missing graph_execution_order array"))?;

        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            match item {
                serde_json::Value::String(s) => out.push(s.clone()),
                serde_json::Value::Object(map) => {
                    if let Some(s) = map
                        .get("graph")
                        .and_then(|v| v.as_str())
                        .or_else(|| map.get("compile_id").and_then(|v| v.as_str()))
                        .or_else(|| map.get("id").and_then(|v| v.as_str()))
                    {
                        out.push(s.to_string());
                    }
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

pub use types::{
    ArtifactFlags, Diagnostics, DivergenceFlags, DivergenceGroup, GraphAnalysis, GraphRuntime,
    RankMetaData, RuntimeAnalysis, RuntimeRankDetail,
};
pub use execution_order::{analyze_execution_order, parse_graph_execution_order, ExecOrderIssue, ExecOrderIndexRow, ExecOrderReport};

#[derive(Debug)]
enum ParserResult {
    NoPayload,
    PayloadFilename(String),
}

pub struct ParseConfig {
    pub strict: bool,
    pub strict_compile_id: bool,
    pub custom_parsers: Vec<Box<dyn crate::parsers::StructuredLogParser>>,
    pub custom_header_html: String,
    pub verbose: bool,
    pub plain_text: bool,
    pub export: bool,
    pub inductor_provenance: bool,
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            strict: false,
            strict_compile_id: false,
            custom_parsers: Vec::default(),
            custom_header_html: String::default(),
            verbose: false,
            plain_text: false,
            export: false,
            inductor_provenance: false,
        }
    }
}

fn maybe_remove_convert_frame_suffixes(frames: &mut Vec<FrameSummary>) {
    let all_target_frames = [
        [
            ("torch/_dynamo/convert_frame.py", "catch_errors"),
            ("torch/_dynamo/convert_frame.py", "_convert_frame"),
            ("torch/_dynamo/convert_frame.py", "_convert_frame_assert"),
        ],
        [
            ("torch/_dynamo/convert_frame.py", "__call__"),
            ("torch/_dynamo/convert_frame.py", "__call__"),
            ("torch/_dynamo/convert_frame.py", "__call__"),
        ],
    ];

    let len = frames.len();
    for target_frames in all_target_frames {
        if len >= target_frames.len() {
            let suffix = &frames[len - target_frames.len()..];
            if suffix
                .iter()
                .zip(target_frames.iter())
                .all(|(frame, target)| {
                    simplify_filename(unintern_str(frame.filename).as_ref()) == target.0
                        && frame.name == target.1
                })
            {
                frames.truncate(len - target_frames.len());
            }
        }
    }
}

fn add_unique_suffix(raw_filename: PathBuf, output_count: i32) -> PathBuf {
    if let Some(stem) = raw_filename.file_stem() {
        let mut r = OsString::new();
        r.push(stem);
        r.push(OsStr::new("_"));
        r.push(output_count.to_string());
        if let Some(e) = raw_filename.extension() {
            r.push(OsStr::new("."));
            r.push(e);
        };
        raw_filename.with_file_name(r)
    } else {
        raw_filename
    }
}

fn add_file_output(
    filename: PathBuf,
    content: String,
    output: &mut ParseOutput,
    compile_directory: &mut Vec<OutputFile>,
    output_count: &mut i32,
) {
    let is_stack_traces = is_stack_traces_file(&filename);
    let maybe_content = if is_stack_traces {
        Some(content.clone())
    } else {
        None
    };
    output.push((filename.clone(), content));
    let filename_str = filename.to_string_lossy().to_string();
    let suffix = if filename_str.contains("cache_miss") {
        "❌".to_string()
    } else if filename_str.contains("cache_hit") {
        "✅".to_string()
    } else if filename_str.contains("cache_bypass") {
        "❓".to_string()
    } else {
        "".to_string()
    };
    let readable_url = if let Some(c) = maybe_content {
        Some(add_stack_traces_html(&filename, &c, output, output_count))
    } else {
        None
    };
    compile_directory.push(OutputFile {
        url: filename_str.clone(),
        name: filename_str,
        number: *output_count,
        suffix: suffix,
        readable_url,
    });
    *output_count += 1;
}

fn is_stack_traces_file(path: &PathBuf) -> bool {
    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
        name.starts_with("inductor_provenance_tracking_kernel_stack_traces")
            && name.ends_with(".json")
    } else {
        false
    }
}

fn add_stack_traces_html(
    json_path: &PathBuf,
    json_content: &str,
    output: &mut ParseOutput,
    output_count: &mut i32,
) -> String {
    let parsed: Value = match serde_json::from_str(json_content) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    let mut html = String::from("<html><body>\n");
    if let Some(map) = parsed.as_object() {
        for (kernel, traces) in map {
            html.push_str(&format!("<h3>{}</h3>\n", encode_text(kernel)));
            if let Some(arr) = traces.as_array() {
                for t in arr {
                    if let Some(s) = t.as_str() {
                        // The JSON strings encode newlines as "\\n" sequences, so translate
                        // those into real line breaks for the HTML view.
                        let decoded = s.replace("\\n", "\n");
                        html.push_str("<pre>");
                        html.push_str(&encode_text(decoded.trim_end_matches('\n')));
                        html.push_str("</pre>\n");
                    }
                }
            }
        }
    }
    html.push_str("</body></html>\n");
    let mut html_path = json_path.clone();
    if let Some(stem) = json_path.file_stem().and_then(|s| s.to_str()) {
        html_path.set_file_name(format!("{stem}_readable.html"));
    } else {
        html_path.set_extension("html");
    }
    let html_path_str = html_path.to_string_lossy().to_string();
    output.push((html_path.clone(), html));
    *output_count += 1;
    html_path_str
}

fn run_parser<'t>(
    lineno: usize,
    parser: &Box<dyn StructuredLogParser + 't>,
    e: &Envelope,
    payload: &str,
    output_count: &mut i32,
    output: &mut ParseOutput,
    compile_directory: &mut Vec<OutputFile>,
    multi: &MultiProgress,
    stats: &mut Stats,
) -> ParserResult {
    let mut payload_filename = ParserResult::NoPayload;
    if let Some(md) = parser.get_metadata(&e) {
        let results = parser.parse(lineno, md, e.rank, &e.compile_id, &payload);
        match results {
            Ok(results) => {
                for parser_result in results {
                    match parser_result {
                        ParserOutput::File(raw_filename, out) => {
                            let filename = add_unique_suffix(raw_filename, *output_count);
                            add_file_output(filename, out, output, compile_directory, output_count);
                        }
                        ParserOutput::GlobalFile(filename, out) => {
                            add_file_output(filename, out, output, compile_directory, output_count);
                        }
                        ParserOutput::PayloadFile(raw_filename) => {
                            let filename = add_unique_suffix(raw_filename, *output_count);
                            payload_filename = ParserResult::PayloadFilename(
                                filename.to_string_lossy().to_string(),
                            );
                            add_file_output(
                                filename,
                                payload.to_string(),
                                output,
                                compile_directory,
                                output_count,
                            );
                        }
                        ParserOutput::PayloadReformatFile(raw_filename, formatter) => {
                            let filename = add_unique_suffix(raw_filename, *output_count);
                            match formatter(payload) {
                                Ok(formatted_content) => {
                                    payload_filename = ParserResult::PayloadFilename(
                                        filename.to_string_lossy().to_string(),
                                    );
                                    add_file_output(
                                        filename,
                                        formatted_content,
                                        output,
                                        compile_directory,
                                        output_count,
                                    );
                                }
                                Err(err) => {
                                    multi.suspend(|| {
                                        eprintln!(
                                            "Failed to format payload for {}: {}",
                                            filename.to_string_lossy(),
                                            err
                                        )
                                    });
                                    stats.fail_parser += 1;
                                }
                            }
                        }
                        ParserOutput::Link(name, url) => {
                            compile_directory.push(OutputFile {
                                url: url,
                                name: name,
                                number: *output_count,
                                suffix: "".to_string(),
                                readable_url: None,
                            });
                            *output_count += 1;
                        }
                    }
                }
            }
            Err(err) => match parser.name() {
                "dynamo_guards" => {
                    multi.suspend(|| eprintln!("Failed to parse guards json: {}", err));
                    stats.fail_dynamo_guards_json += 1;
                }
                name => {
                    multi.suspend(|| eprintln!("Parser {name} failed: {err}"));
                    stats.fail_parser += 1;
                }
            },
        }
    }
    payload_filename
}

fn directory_to_json(
    directory: &FxIndexMap<Option<CompileId>, Vec<OutputFile>>,
) -> serde_json::Value {
    let mut json_map = serde_json::Map::new();

    for (compile_id, output_files) in directory {
        let key = compile_id
            .as_ref()
            .map_or_else(|| "unknown".to_string(), |cid| cid.to_string());

        let artifacts: Vec<serde_json::Value> = output_files
            .iter()
            .map(|file| {
                serde_json::json!({
                    "url": file.url,
                    // Strip away any leading directory names, that will just be in the url path anyway
                    "name": file.name.split('/').last().unwrap_or(&file.name),
                    "number": file.number,
                    "suffix": file.suffix,
                    "readable_url": file.readable_url,
                })
            })
            .collect();

        json_map.insert(key, serde_json::json!({"artifacts": artifacts}));
    }
    serde_json::Value::Object(json_map)
}

fn handle_guard(
    failure_type: &str,
    reason: &str,
    lineno: usize,
    e: &Envelope,
    payload: &str,
    output_count: &mut i32,
    output: &mut Vec<(PathBuf, String)>,
    compile_directory: &mut Vec<OutputFile>,
    multi: &MultiProgress,
    stats: &mut Stats,
    tt: &TinyTemplate,
    sym_expr_info_index: &RefCell<SymExprInfoIndex>,
    export_failures: &mut Vec<ExportFailure>,
) {
    let sym_expr_info_index_borrowed = sym_expr_info_index.borrow();
    let parser: Box<dyn StructuredLogParser> =
        Box::new(crate::parsers::PropagateRealTensorsParser {
            tt,
            sym_expr_info_index: &sym_expr_info_index_borrowed,
        });
    let _ = run_parser(
        lineno,
        &parser,
        e,
        payload,
        output_count,
        output,
        compile_directory,
        multi,
        stats,
    );

    let filename = format!(
        "symbolic_guard_information_{}.html",
        (*output_count - 1).to_string()
    );
    let compile_id_dir: PathBuf = e
        .compile_id
        .as_ref()
        .map_or(format!("unknown_{lineno}"), |cid| cid.as_directory_name())
        .into();
    let additional_info = format!(
        "Please click <a href='{}/{}'>here</a> for more information.",
        compile_id_dir.display(),
        filename,
    );

    export_failures.push(ExportFailure {
        failure_type: failure_type.to_string(),
        reason: reason.to_string(),
        additional_info,
    });
}

pub fn parse_path(path: &PathBuf, config: &ParseConfig) -> anyhow::Result<ParseOutput> {
    let strict = config.strict;
    if !path.is_file() {
        bail!("{} is not a file", path.display())
    }
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();

    // TODO: abstract out this spinner to not be part of the library
    // Instead, add a callback trait for CLIs to implement
    let multi = MultiProgress::new();
    let pb = multi.add(ProgressBar::new(file_size));
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} [{bytes_per_sec}] ({eta})")?
        .progress_chars("#>-"));
    let spinner = multi.add(ProgressBar::new_spinner());

    let reader = io::BufReader::new(file);

    let re_glog = Regex::new(concat!(
        r"(?<level>[VIWEC])(?<month>\d{2})(?<day>\d{2}) ",
        r"(?<hour>\d{2}):(?<minute>\d{2}):(?<second>\d{2}).(?<millisecond>\d{6}) ",
        r"(?<thread>\d+)",
        r"(?<pathname>[^:]+):(?<line>\d+)\] ",
        r"(?<payload>.)"
    ))?;

    // Helper functions to reduce repetitive serde_json::Value creation
    let make_string_value = |caps: &regex::Captures, name: &str| -> serde_json::Value {
        serde_json::Value::String(caps.name(name).unwrap().as_str().to_string())
    };

    let make_number_value = |caps: &regex::Captures, name: &str| -> serde_json::Value {
        let parsed: u64 = caps.name(name).unwrap().as_str().parse().unwrap();
        serde_json::Value::Number(serde_json::Number::from(parsed))
    };

    // Helper function to format timestamp as ISO-8601
    let format_timestamp = |caps: &regex::Captures| -> String {
        let month: u32 = caps.name("month").unwrap().as_str().parse().unwrap();
        let day: u32 = caps.name("day").unwrap().as_str().parse().unwrap();
        let hour: u32 = caps.name("hour").unwrap().as_str().parse().unwrap();
        let minute: u32 = caps.name("minute").unwrap().as_str().parse().unwrap();
        let second: u32 = caps.name("second").unwrap().as_str().parse().unwrap();
        let microsecond: u32 = caps.name("millisecond").unwrap().as_str().parse().unwrap();

        // Assume current year since glog doesn't include year
        let year = chrono::Utc::now().year();

        // Format as ISO-8601 with microsecond precision
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
            year, month, day, hour, minute, second, microsecond
        )
    };

    let mut stack_trie = StackTrieNode::default();
    let mut unknown_stack_trie = StackTrieNode::default();

    let mut stats = Stats::default();
    let _mod_count: FxHashMap<String, i32> = FxHashMap::default();

    let mut bytes_read: u64 = 0;

    // Some stuff for profiling
    let mut fastest_time = std::time::Duration::MAX;
    let mut slowest_time = std::time::Duration::ZERO;

    let mut expected_rank: Option<Option<u32>> = None;

    // Each entry is a compile id => (link, rendered name, output number)
    // For files, link and rendered name are the same
    // For links, you can specify a custom name for the link
    let mut directory: FxIndexMap<Option<CompileId>, Vec<OutputFile>> = FxIndexMap::default();

    let mut metrics_index: CompilationMetricsIndex = FxIndexMap::default();
    let stack_index: RefCell<StackIndex> = RefCell::new(FxHashMap::default());

    let symbolic_shape_specialization_index: RefCell<SymbolicShapeSpecializationIndex> =
        RefCell::new(FxHashMap::default());
    let guard_added_fast_index: RefCell<GuardAddedFastIndex> = RefCell::new(FxHashMap::default());
    let sym_expr_info_index: RefCell<SymExprInfoIndex> = RefCell::new(FxHashMap::default());

    // Store results in an output ParseOutput
    let mut output: ParseOutput = Vec::new();

    // Store raw.jsonl content (without payloads)
    let mut shortraw_content = String::new();

    let mut tt: TinyTemplate = TinyTemplate::new();
    tt.add_formatter("format_unescaped", tinytemplate::format_unescaped);
    if config.export {
        tt.add_template("index.html", TEMPLATE_EXPORT_INDEX)?;
        tt.add_template(
            "symbolic_guard_information.html",
            TEMPLATE_SYMBOLIC_GUARD_INFO,
        )?;
    } else {
        tt.add_template("index.html", TEMPLATE_INDEX)?;
        tt.add_template("failures_and_restarts.html", TEMPLATE_FAILURES_AND_RESTARTS)?;
        tt.add_template("dynamo_guards.html", TEMPLATE_DYNAMO_GUARDS)?;
        tt.add_template("compilation_metrics.html", TEMPLATE_COMPILATION_METRICS)?;
        tt.add_template(
            "bwd_compilation_metrics.html",
            TEMPLATE_BWD_COMPILATION_METRICS,
        )?;
        tt.add_template(
            "aot_autograd_backward_compilation_metrics.html",
            TEMPLATE_AOT_AUTOGRAD_BACKWARD_COMPILATION_METRICS,
        )?;
    }
    tt.add_template("provenance_tracking.html", TEMPLATE_PROVENANCE_TRACKING)?;

    let mut unknown_fields: FxHashSet<String> = FxHashSet::default();

    let mut output_count = 0;

    let mut breaks = RestartsAndFailuresContext {
        css: TEMPLATE_FAILURES_CSS,
        failures: Vec::new(),
        qps: TEMPLATE_QUERY_PARAM_SCRIPT,
    };

    let mut export_failures: Vec<ExportFailure> = Vec::new();

    // NB: Sometimes, the log output we get from Logarithm stutters with a blank line.
    // Filter them out, they're never valid (a blank line in payload will still be \t)
    let mut iter = reader
        .lines()
        .enumerate()
        .filter_map(|(i, l)| match l {
            // 1-indexed line numbers please
            Ok(l) if !l.is_empty() => Some((i + 1, l)),
            _ => None,
        })
        .peekable();

    let default_parsers = default_parsers(&tt, config);
    let mut all_parsers: Vec<&Box<dyn StructuredLogParser>> = default_parsers.iter().collect();
    let mut chromium_events: Vec<serde_json::Value> = Vec::new();
    all_parsers.extend(config.custom_parsers.iter());

    while let Some((lineno, line)) = iter.next() {
        bytes_read += line.len() as u64;
        pb.set_position(bytes_read);
        spinner.set_message(format!("{}", stats));
        //spinner.set_message(format!("{:?} {:?}", slowest_time, fastest_time));
        let start = Instant::now();

        let Some(caps) = re_glog.captures(&line) else {
            multi.suspend(|| eprintln!("Failed to parse glog prefix on line {}", lineno));
            stats.fail_glog += 1;
            continue;
        };

        let end = start.elapsed();
        if end < fastest_time {
            fastest_time = end;
        }
        if end > slowest_time {
            slowest_time = end;
        }
        let payload = &line[caps.name("payload").unwrap().start()..];
        let original_json_envelope = payload; // Store the original JSON envelope

        // Helper function to safely insert keys and detect conflicts
        let try_insert = |obj: &mut serde_json::Map<String, serde_json::Value>,
                          key: &str,
                          value: serde_json::Value,
                          multi: &MultiProgress,
                          stats: &mut Stats|
         -> bool {
            if obj.contains_key(key) {
                multi.suspend(|| {
                    eprintln!("Key conflict: '{}' already exists in JSON payload, skipping raw.jsonl JSONL conversion", key);
                });
                stats.fail_key_conflict += 1;
                false
            } else {
                obj.insert(key.to_string(), value);
                true
            }
        };

        // Create cleanup lambda to handle raw.jsonl writing as JSONL
        let write_to_shortraw = |shortraw_content: &mut String,
                                 payload_filename: Option<String>,
                                 multi: &MultiProgress,
                                 stats: &mut Stats| {
            match serde_json::from_str::<serde_json::Value>(original_json_envelope) {
                Ok(mut json_value) => {
                    if let Some(obj) = json_value.as_object_mut() {
                        // Try to add all log fields, abort on any conflict
                        let success = try_insert(
                            obj,
                            "timestamp",
                            serde_json::Value::String(format_timestamp(&caps)),
                            multi,
                            stats,
                        ) && try_insert(
                            obj,
                            "thread",
                            make_number_value(&caps, "thread"),
                            multi,
                            stats,
                        ) && try_insert(
                            obj,
                            "pathname",
                            make_string_value(&caps, "pathname"),
                            multi,
                            stats,
                        ) && try_insert(
                            obj,
                            "lineno",
                            make_number_value(&caps, "line"),
                            multi,
                            stats,
                        );

                        // Try to add payload filename if provided
                        let success = if let Some(payload_file) = payload_filename {
                            success
                                && try_insert(
                                    obj,
                                    "payload_filename",
                                    serde_json::Value::String(payload_file),
                                    multi,
                                    stats,
                                )
                        } else {
                            success
                        };

                        if !success {
                            // Drop line due to key conflict - don't write anything to maintain JSONL format
                            return;
                        }

                        // Output as JSONL
                        match serde_json::to_string(&json_value) {
                            Ok(jsonl_line) => {
                                shortraw_content.push_str(&jsonl_line);
                                shortraw_content.push('\n');
                            }
                            Err(e) => {
                                multi.suspend(|| {
                                    eprintln!("Failed to serialize JSON for raw.jsonl: {}", e);
                                });
                                stats.fail_json_serialization += 1;
                                // Drop line to maintain JSONL format - don't write anything
                            }
                        }
                    } else {
                        // Not a JSON object, drop line to maintain JSONL format
                        multi.suspend(|| {
                            eprintln!(
                                "JSON payload is not an object, dropping line from raw.jsonl"
                            );
                        });
                        stats.fail_json += 1;
                    }
                }
                Err(e) => {
                    // JSON parsing failed, drop line to maintain JSONL format
                    multi.suspend(|| {
                        eprintln!("Failed to parse JSON envelope for raw.jsonl: {}", e);
                    });
                    stats.fail_json += 1;
                }
            }
        };

        let e = match serde_json::from_str::<Envelope>(payload) {
            Ok(r) => r,
            Err(err) => {
                multi.suspend(|| {
                    eprintln!("Failed to parse metadata JSON: {}\n{:?}", payload, err);
                });
                stats.fail_json += 1;
                write_to_shortraw(&mut shortraw_content, None, &multi, &mut stats);
                continue;
            }
        };

        stats.unknown += e._other.len() as u64;

        for k in e._other.keys() {
            unknown_fields.insert(k.clone());
            if config.verbose {
                multi.suspend(|| eprintln!("Unknown field {}", k))
            }
        }

        if let Some((s, i)) = e.str {
            let mut intern_table = INTERN_TABLE.lock().unwrap();
            intern_table.insert(i, s);
            continue;
        };

        let mut payload = String::new();
        if let Some(ref expect) = e.has_payload {
            let mut first = true;
            while let Some((_payload_lineno, payload_line)) =
                iter.next_if(|(_, l)| l.starts_with('\t'))
            {
                // Careful! Distinguish between missing EOL and not
                if !first {
                    payload.push('\n');
                }
                first = false;
                payload.push_str(&payload_line[1..]);
            }
            let mut hasher = Md5::new();
            hasher.update(&payload);
            let hash = hasher.finalize();
            let mut expect_buf = [0u8; 16];
            if base16ct::lower::decode(expect, &mut expect_buf).is_ok() {
                if expect_buf != hash[..] {
                    // TODO: error log
                    stats.fail_payload_md5 += 1;
                }
            } else {
                stats.fail_payload_md5 += 1;
            }
        }

        match expected_rank {
            Some(rank) => {
                if rank != e.rank {
                    stats.other_rank += 1;
                    write_to_shortraw(&mut shortraw_content, None, &multi, &mut stats);
                    continue;
                }
            }
            None => {
                // Allow logs with no rank and then some rank to be processed
                // Logs with no rank may be initialized before distributed rank is set
                if e.rank.is_some() {
                    multi.suspend(|| {
                        eprintln!("Detected rank: {:?}", e.rank);
                    });
                    expected_rank = Some(e.rank);
                }
            }
        };

        stats.ok += 1;

        // Some runtime compile ids don't have attempts. Collapse these entries into
        // attempt 0 for now.
        let mut compile_id_entry = e.compile_id.clone();
        if let Some(ref mut entry) = compile_id_entry {
            if entry.frame_compile_id.is_some() && entry.attempt.is_none() {
                entry.attempt = Some(0);
            }
        }

        // TODO: output should be able to generate this without explicitly creating
        let compile_directory = directory.entry(compile_id_entry).or_default();

        let mut parser_payload_filename = ParserResult::NoPayload;
        for parser in &all_parsers {
            let result = run_parser(
                lineno,
                parser,
                &e,
                &payload,
                &mut output_count,
                &mut output,
                compile_directory,
                &multi,
                &mut stats,
            );
            // Take the last PayloadFilename entry as per the requirement
            if matches!(result, ParserResult::PayloadFilename(_)) {
                parser_payload_filename = result;
            }
        }

        if let Some(ref m) = e.compilation_metrics {
            let copied_directory = compile_directory.clone();
            let compile_id_dir: PathBuf = e
                .compile_id
                .as_ref()
                .map_or(format!("unknown_{lineno}"), |cid| cid.as_directory_name())
                .into();
            let parser: Box<dyn StructuredLogParser> =
                Box::new(crate::parsers::CompilationMetricsParser {
                    tt: &tt,
                    stack_index: &stack_index,
                    symbolic_shape_specialization_index: &symbolic_shape_specialization_index,
                    guard_added_fast_index: &guard_added_fast_index,
                    output_files: &copied_directory,
                    compile_id_dir: &compile_id_dir,
                });
            let result = run_parser(
                lineno,
                &parser,
                &e,
                &payload,
                &mut output_count,
                &mut output,
                compile_directory,
                &multi,
                &mut stats,
            );
            // Take the last PayloadFilename entry as per the requirement
            if matches!(result, ParserResult::PayloadFilename(_)) {
                parser_payload_filename = result;
            }

            // compilation metrics is always the last output, since it just ran
            let metrics_filename = format!(
                "compilation_metrics_{}.html",
                (output_count - 1).to_string(),
            );
            let id = e.compile_id.clone().map_or("(unknown) ".to_string(), |c| {
                format!(
                    "<a href='{}/{}'>{cid}</a> ",
                    compile_id_dir.display(),
                    metrics_filename,
                    cid = c,
                )
            });
            if let Some(rr) = m.restart_reasons.as_ref() {
                for restart in rr {
                    breaks.failures.push((
                        id.clone(),
                        format!("{}", FailureReason::Restart(restart.clone())),
                    ));
                }
            }
            if let Some(f) = m.fail_type.as_ref() {
                let reason = m
                    .fail_reason
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("Fail reason not found"))?;
                let user_frame_filename = m
                    .fail_user_frame_filename
                    .clone()
                    .unwrap_or(String::from("N/A"));
                let user_frame_lineno = m.fail_user_frame_lineno.unwrap_or(0);
                let failure_reason = FailureReason::Failure((
                    f.clone(),
                    reason.clone(),
                    user_frame_filename.clone(),
                    user_frame_lineno.clone(),
                ));
                breaks
                    .failures
                    .push((id.clone(), format!("{failure_reason}")));
            }
            let mut cid = e.compile_id.clone();
            if let Some(c) = cid.as_mut() {
                if let Some(_frame_id) = c.frame_compile_id {
                    // data migration for old logs that don't have attempt
                    c.attempt = Some(0);
                }
            }
            metrics_index.entry(cid).or_default().push(m.clone());
        }

        if config.export {
            if let Some(ref guard) = e.guard_added {
                if guard.prefix.as_deref() != Some("eval") {
                    write_to_shortraw(&mut shortraw_content, None, &multi, &mut stats);
                    continue;
                }
                let failure_type = "Guard Evaluated";

                let reason = format!(
                    "When exporting, the following guard was evaluated <code>{}</code>. This
                    might've resulted in a constraint violation error.",
                    guard.expr.clone().unwrap(),
                );

                handle_guard(
                    failure_type,
                    &reason,
                    lineno,
                    &e,
                    &payload,
                    &mut output_count,
                    &mut output,
                    compile_directory,
                    &multi,
                    &mut stats,
                    &tt,
                    &sym_expr_info_index,
                    &mut export_failures,
                );
            }

            if let Some(ref guard) = e.propagate_real_tensors_provenance {
                let failure_type = "Data Dependent Error";

                let reason = format!(
                    "When exporting, we were unable to figure out if the
                    expression <code>{}</code> always holds.<br> As a result, it
                    was specialized to evaluate to <code>{}</code>, and asserts
                    were inserted into the graph.",
                    guard.expr.clone().unwrap(),
                    guard.result.clone().unwrap()
                );

                handle_guard(
                    failure_type,
                    &reason,
                    lineno,
                    &e,
                    &payload,
                    &mut output_count,
                    &mut output,
                    compile_directory,
                    &multi,
                    &mut stats,
                    &tt,
                    &sym_expr_info_index,
                    &mut export_failures,
                );
            }

            if let Some(fake_kernel) = e.missing_fake_kernel {
                let failure_type = "Missing Fake Kernel";

                let reason = format!(
                    "<code>torch.ops.{}</code> is missing a fake kernel implementation",
                    fake_kernel.op.unwrap()
                );

                let additional_info = "Please refer to <a href='https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz'>this doc</a> for more detailed instructions on how to write a fake kernel.";

                export_failures.push(ExportFailure {
                    failure_type: failure_type.to_string(),
                    reason: reason,
                    additional_info: additional_info.to_string(),
                });
            }

            if let Some(fake_kernel) = e.mismatched_fake_kernel {
                let failure_type = "Mismatched Fake Kernel";

                let reason = format!(
                    "<code>torch.ops.{}</code> has a fake kernel implementation,
                    but it has incorrect behavior, based on the real kernel.<br>
                    The reason for the mismatch is: {}",
                    fake_kernel.op.unwrap(),
                    fake_kernel.reason.unwrap(),
                );

                let additional_info = "Please refer to <a href='https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz'>this doc</a> for more detailed instructions on how to write a fake kernel.";

                export_failures.push(ExportFailure {
                    failure_type: failure_type.to_string(),
                    reason: reason,
                    additional_info: additional_info.to_string(),
                });
            }

            if let Some(sym_expr_info) = e.expression_created {
                sym_expr_info_index
                    .borrow_mut()
                    .insert(sym_expr_info.result_id.unwrap(), sym_expr_info);
            }

            if let Some(unbacked_symbol) = e.create_unbacked_symbol {
                sym_expr_info_index.borrow_mut().insert(
                    unbacked_symbol.node_id.unwrap(),
                    SymExprInfoMetadata {
                        result: unbacked_symbol.symbol.clone(),
                        result_id: unbacked_symbol.node_id.clone(),
                        user_stack: unbacked_symbol.user_stack.clone(),
                        stack: unbacked_symbol.stack.clone(),
                        ..Default::default()
                    },
                );
            }
        }

        if let Some(stack) = e.stack {
            unknown_stack_trie.insert(stack.clone(), None);
        }

        if let Some(_) = e.chromium_event {
            chromium_events.push(serde_json::from_str(&payload)?);
        }

        if let Some(specialization) = e.symbolic_shape_specialization {
            symbolic_shape_specialization_index
                .borrow_mut()
                .entry(e.compile_id.clone())
                .or_default()
                .push(specialization);
        }
        if let Some(guard_added_fast) = e.guard_added_fast {
            guard_added_fast_index
                .borrow_mut()
                .entry(e.compile_id.clone())
                .or_default()
                .push(guard_added_fast)
        }

        if let Some(m) = e.dynamo_start {
            if let Some(mut stack) = m.stack {
                maybe_remove_convert_frame_suffixes(&mut stack);
                stack_index
                    .borrow_mut()
                    .insert(e.compile_id.clone(), stack.clone());
                stack_trie.insert(stack, e.compile_id.clone());
            };
        };

        // Handle payload file writing and determine final payload filename, but skip chromium events
        let final_payload_filename = match parser_payload_filename {
            ParserResult::PayloadFilename(filename) => Some(filename),
            ParserResult::NoPayload => {
                if let Some(ref expect) = e.has_payload {
                    // Only write payload file if no parser generated PayloadFile/PayloadReformatFile output and not a chromium event
                    if !payload.is_empty() && e.chromium_event.is_none() {
                        let hash_str = expect;
                        let payload_path = PathBuf::from(format!("payloads/{}.txt", hash_str));
                        output.push((payload_path, payload.clone()));
                        Some(format!("payloads/{}.txt", hash_str))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        };

        // Write to raw.jsonl with optional payload filename, but skip chromium events
        if e.chromium_event.is_none() {
            write_to_shortraw(
                &mut shortraw_content,
                final_payload_filename,
                &multi,
                &mut stats,
            );
        }
    }

    if config.export {
        let num_failures = export_failures.len();

        let exported_program_url = directory
            .values()
            .flatten()
            .find(|output_file| output_file.url.contains("exported_program"))
            .map(|output_file| output_file.url.clone());

        let index_context = ExportIndexContext {
            css: EXPORT_CSS,
            javascript: JAVASCRIPT,
            custom_header_html: config.custom_header_html.clone(),
            directory: directory
                .drain(..)
                .map(|(x, y)| (x.map_or("(unknown)".to_string(), |e| e.to_string()), y))
                .collect(),
            failures: export_failures,
            num_failures: num_failures,
            success: num_failures == 0,
            exported_program_url: exported_program_url.unwrap_or("".to_string()),
            qps: TEMPLATE_QUERY_PARAM_SCRIPT,
        };

        output.push((
            PathBuf::from("index.html"),
            tt.render("index.html", &index_context)?,
        ));

        return Ok(output);
    }

    output.push((
        PathBuf::from("failures_and_restarts.html"),
        tt.render("failures_and_restarts.html", &breaks)?,
    ));
    pb.finish_with_message("done");
    spinner.finish();

    output.push((
        PathBuf::from("chromium_events.json"),
        serde_json::to_string_pretty(&chromium_events).unwrap(),
    ));

    eprintln!("{}", stats);
    if unknown_fields.len() > 0 {
        eprintln!(
            "Unknown fields: {:?} (consider updating tlparse to render these)",
            unknown_fields
        );
    }

    let has_unknown_compile_id = directory.contains_key(&None);

    let directory_names: Vec<String> = directory
        .iter()
        .map(|(x, _)| {
            x.as_ref()
                .map_or("(unknown)".to_string(), |e| e.as_directory_name())
        })
        .collect();
    output.push((
        PathBuf::from("compile_directory.json"),
        serde_json::to_string_pretty(&directory_to_json(&directory))?,
    ));
    let index_context = IndexContext {
        css: CSS,
        javascript: JAVASCRIPT,
        custom_header_html: config.custom_header_html.clone(),
        directory: directory
            .drain(..)
            .map(|(x, y)| (x.map_or("(unknown)".to_string(), |e| e.to_string()), y))
            .collect(),
        stack_trie_html: stack_trie
            .fmt(Some(&metrics_index), "Stack", false)
            .unwrap(),
        unknown_stack_trie_html: unknown_stack_trie
            .fmt(Some(&metrics_index), "Stack", false)
            .unwrap(),
        has_unknown_stack_trie: !unknown_stack_trie.is_empty(),
        num_breaks: breaks.failures.len(),
        has_chromium_events: !chromium_events.is_empty(),
        qps: TEMPLATE_QUERY_PARAM_SCRIPT,
        has_inductor_provenance: config.inductor_provenance,
        directory_names: directory_names.clone(),
    };
    output.push((
        PathBuf::from("index.html"),
        tt.render("index.html", &index_context)?,
    ));

    output.push((PathBuf::from("raw.log"), fs::read_to_string(path)?));

    // Create string table from INTERN_TABLE as an array with nulls for missing indices
    let intern_table = INTERN_TABLE.lock().unwrap();
    let max_index = intern_table.keys().max().copied().unwrap_or(0) as usize;
    let mut string_table: Vec<Option<String>> = vec![None; max_index + 1];
    for (&index, value) in intern_table.iter() {
        string_table[index as usize] = Some(value.clone());
    }
    drop(intern_table); // Release the lock early

    // Serialize string table as JSON object
    let string_table_json = serde_json::json!({
        "string_table": string_table
    });
    let string_table_line = serde_json::to_string(&string_table_json)?;

    // Prepend string table to raw.jsonl content
    let mut final_shortraw_content =
        String::with_capacity(string_table_line.len() + 1 + shortraw_content.len());
    final_shortraw_content.push_str(&string_table_line);
    final_shortraw_content.push('\n');
    final_shortraw_content.push_str(&shortraw_content);

    output.push((PathBuf::from("raw.jsonl"), final_shortraw_content));

    // other_rank is included here because you should only have logs from one rank when
    // configured properly
    if strict
        && (stats.fail_glog
            + stats.fail_json
            + stats.fail_payload_md5
            + stats.other_rank
            + stats.fail_dynamo_guards_json
            + stats.fail_parser
            > 0)
    {
        // Report something went wrong
        return Err(anyhow!("Something went wrong"));
    }

    if config.strict_compile_id && has_unknown_compile_id {
        return Err(anyhow!("Some log entries did not have compile id"));
    }

    if config.inductor_provenance {
        // Helper function to get file content for a specific directory name
        fn get_file_content(
            output: &[(PathBuf, String)],
            filename_patterns: &[&str],
            directory_name: &str,
        ) -> String {
            // Try each pattern in order and return the first match found
            for pattern in filename_patterns {
                if let Some((_, content)) = output.iter().rev().find(|(path, _)| {
                    path.to_string_lossy()
                        .contains(&format!("{}/{}", directory_name, pattern))
                }) {
                    return content.clone();
                }
            }
            String::default()
        }

        // Generate HTML for each directory name
        for directory_name in &directory_names {
            let pre_grad_graph_content = get_file_content(
                &output,
                &["before_pre_grad_graph", "inductor_pre_grad_graph"],
                directory_name,
            );
            let post_grad_graph_content = get_file_content(
                &output,
                &["after_post_grad_graph", "inductor_post_grad_graph"],
                directory_name,
            );
            let output_code_content =
                get_file_content(&output, &["inductor_output_code"], directory_name);
            let aot_code_content =
                get_file_content(&output, &["inductor_aot_wrapper_code"], directory_name);
            let node_mappings_content = get_file_content(
                &output,
                &["inductor_provenance_tracking_node_mappings"],
                directory_name,
            );

            // Convert node mappings to line number mappings
            let line_mappings_content = convert_node_mappings_to_line_numbers(
                &node_mappings_content,
                &pre_grad_graph_content,
                &post_grad_graph_content,
                &output_code_content,
                &aot_code_content,
            );
            let line_mappings_content_str = serde_json::to_string_pretty(&line_mappings_content)
                .unwrap_or_else(|_| "{}".to_string());

            output.push((
                PathBuf::from(format!("provenance_tracking_{}.html", directory_name)),
                tt.render(
                    "provenance_tracking.html",
                    &ProvenanceContext {
                        css: PROVENANCE_CSS,
                        js: PROVENANCE_JS,
                        pre_grad_graph_content,
                        post_grad_graph_content,
                        output_code_content,
                        aot_code_content,
                        line_mappings_content: line_mappings_content_str,
                    },
                )?,
            ));
        }
    }

    Ok(output)
}

pub fn read_chromium_events_with_pid(
    path: &std::path::Path,
    rank_num: u32,
) -> anyhow::Result<Vec<serde_json::Value>> {
    use std::fs;

    if !path.exists() {
        return Ok(Vec::new());
    }

    let file_content = fs::read_to_string(path)?;

    match serde_json::from_str::<Vec<serde_json::Value>>(&file_content) {
        Ok(mut events) => {
            for event in &mut events {
                if let Some(obj) = event.as_object_mut() {
                    obj.insert("pid".to_string(), serde_json::json!(rank_num));
                }
            }
            Ok(events)
        }
        Err(_) => Ok(Vec::new()),
    }
}

pub fn generate_multi_rank_html(
    out_path: &PathBuf,
    sorted_ranks: Vec<String>,
    cfg: &ParseConfig,
    has_chromium_events: bool,
    show_desync_warning: bool,
    compile_id_divergence: bool,
    diagnostics: Diagnostics,
) -> anyhow::Result<(PathBuf, String)> {
    // Create the TinyTemplate instance for rendering the landing page.
    let mut tt = TinyTemplate::new();
    tt.add_formatter("format_unescaped", tinytemplate::format_unescaped);
    tt.add_template("multi_rank_index.html", TEMPLATE_MULTI_RANK_INDEX)?;

    let ctx = MultiRankContext {
        css: CSS,
        custom_header_html: &cfg.custom_header_html,
        num_ranks: sorted_ranks.len(),
        ranks: sorted_ranks,
        qps: TEMPLATE_QUERY_PARAM_SCRIPT,
        has_chromium_events,
        show_desync_warning,
        compile_id_divergence,
        diagnostics,
    };
    let html = tt.render("multi_rank_index.html", &ctx)?;
    let landing_page_path = out_path.join("index.html");

    Ok((landing_page_path, html))
}

fn prepare_and_validate_graphs(
    runtime_estimations: &[GraphRuntime],
) -> Option<(
    std::collections::HashMap<u32, Vec<(&str, f64)>>,
    Vec<u32>,
    usize,
)> {
    use std::collections::HashMap;

    let rank_graphs: HashMap<u32, Vec<(&str, f64)>> = runtime_estimations
        .iter()
        .map(|gr| {
            (
                gr.rank,
                &gr.graph,
                gr.ops.iter().map(|op| op.estimated_runtime_ns).sum::<f64>(),
            )
        })
        .fold(HashMap::new(), |mut acc, (rank, graph, runtime)| {
            acc.entry(rank).or_default().push((graph, runtime));
            acc
        });

    let max_graphs = rank_graphs.values().map(|graphs| graphs.len()).max()?;
    let min_graphs = rank_graphs.values().map(|graphs| graphs.len()).min()?;

    if max_graphs != min_graphs {
        return None; // Different number of graphs across ranks
    }

    let mut ranks: Vec<_> = rank_graphs.keys().copied().collect();
    ranks.sort_unstable();

    Some((rank_graphs, ranks, max_graphs))
}

fn compare_graph_runtimes(
    rank_graphs: std::collections::HashMap<u32, Vec<(&str, f64)>>,
    ranks: Vec<u32>,
    max_graphs: usize,
) -> Vec<GraphAnalysis> {
    (0..max_graphs)
        .filter_map(|index| {
            let runtimes: Vec<_> = ranks
                .iter()
                .map(|&rank| {
                    rank_graphs
                        .get(&rank)
                        .and_then(|g| g.get(index))
                        .map(|(graph_id, runtime)| (rank, *graph_id, *runtime))
                })
                .collect::<Option<Vec<_>>>()?;

            let (min_runtime, max_runtime, fastest_rank, slowest_rank) = runtimes.iter().fold(
                (f64::INFINITY, f64::NEG_INFINITY, 0_u32, 0_u32),
                |(min_rt, max_rt, fast_rank, slow_rank), &(rank, _, rt)| {
                    let (new_min_rt, new_fast) = if rt <= min_rt {
                        (rt, rank)
                    } else {
                        (min_rt, fast_rank)
                    };
                    let (new_max_rt, new_slow) = if rt >= max_rt {
                        (rt, rank)
                    } else {
                        (max_rt, slow_rank)
                    };
                    (new_min_rt, new_max_rt, new_fast, new_slow)
                },
            );

            let delta_ns = max_runtime - min_runtime;

            Some(GraphAnalysis {
                graph_index: index,
                graph_id: runtimes[0].1.to_string(),
                delta_ms: (delta_ns / 1e6 * 1000.0).round() / 1000.0,
                rank_details: vec![
                    RuntimeRankDetail {
                        rank: fastest_rank,
                        runtime_ms: (min_runtime / 1e6 * 1000.0).round() / 1000.0,
                    },
                    RuntimeRankDetail {
                        rank: slowest_rank,
                        runtime_ms: (max_runtime / 1e6 * 1000.0).round() / 1000.0,
                    },
                ],
            })
        })
        .collect()
}

pub fn analyze_graph_runtime_deltas(
    runtime_estimations: &[GraphRuntime],
) -> Option<RuntimeAnalysis> {
    let Some((rank_graphs, ranks, max_graphs)) = prepare_and_validate_graphs(runtime_estimations)
    else {
        return Some(RuntimeAnalysis {
            graphs: vec![],
            has_mismatched_graph_counts: true,
        });
    };

    let mut graphs = compare_graph_runtimes(rank_graphs, ranks, max_graphs);
    graphs.sort_by(|a, b| a.graph_id.cmp(&b.graph_id));

    Some(RuntimeAnalysis {
        graphs,
        has_mismatched_graph_counts: false,
    })
}

/// Converts node-based mappings to line number-based mappings for visualization.
///
/// This function processes node mappings and converts them to line number mappings
/// that can be used to highlight corresponding lines across different views.
/// It handles pre-grad graph, post-grad graph, and generated code files.
fn convert_node_mappings_to_line_numbers(
    node_mappings_content: &str,
    pre_grad_graph_content: &str,
    post_grad_graph_content: &str,
    output_code_content: &str,
    aot_code_content: &str,
) -> serde_json::Value {
    // Parse the node mappings JSON
    let node_mappings: serde_json::Value = match serde_json::from_str(node_mappings_content) {
        Ok(mappings) => mappings,
        Err(_) => return serde_json::json!({}),
    };

    let version = node_mappings
        .get("version")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as i64;

    // Helper function to check if a line is valid (not empty and doesn't start with comment)
    fn valid_line(line: &str, symbol: &str) -> bool {
        let stripped = line.trim();
        !stripped.is_empty() && !stripped.starts_with(symbol)
    }

    // Helper function to extract node name from a line
    fn extract_node_name(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if valid_line(trimmed, "#") {
            // Split on '=' and take everything before it
            let before_equals = trimmed.split('=').next()?;
            // Split on ':' and take everything before it
            let node_name = before_equals.split(':').next()?.trim();
            if !node_name.is_empty() {
                return Some(node_name.to_string());
            }
        }
        None
    }

    // Helper function to build node-to-line lookup map from graph content
    fn build_node_to_lines_map(content: &str) -> std::collections::HashMap<String, usize> {
        let mut node_to_lines = std::collections::HashMap::new();
        for (i, line) in content.lines().enumerate() {
            if let Some(node_name) = extract_node_name(line) {
                node_to_lines.insert(node_name, i + 1); // 1-based line numbers
            }
        }
        node_to_lines
    }

    // Helper function to build Python kernel-to-lines lookup map
    fn build_python_kernel_to_lines_map(
        content: &str,
        kernel_names: &[&str],
        _version: i64,
    ) -> std::collections::HashMap<String, Vec<usize>> {
        let content = content
            .lines()
            .skip_while(|line| line.is_empty())
            .collect::<Vec<&str>>()
            .join("\n");
        let mut kernel_to_lines = std::collections::HashMap::new();

        // Find the line number of "def call(args)" - allowing for whitespace between tokens
        let run_impl_line = content
            .lines()
            .position(|line| {
                line.contains("def") && line.contains("call") && line.contains("(args)")
            })
            .unwrap_or(0);
        let first_line_number = content
            .lines()
            .position(|line| line.contains("# AOT ID:"))
            .unwrap_or(0);

        // For each kernel name (e.g. triton_poi_fused_mul_1:2):
        // - Extract pure_kernel_name (triton_poi_fused_mul_1) before the ':'
        // - If kernel name found: map to next line containing pure_kernel_name
        // - If kernel_name not found: map to all lines with pure_kernel_name
        for kernel_name in kernel_names {
            // Get pure kernel name before ':' if it exists
            let pure_kernel_name = if let Some(idx) = kernel_name.find(':') {
                &kernel_name[..idx]
            } else {
                kernel_name
            };

            let mut found = false;
            // If kernel_name contains a debug handle and we found it, we can stop after first match
            if kernel_name.contains(':') {
                for (i, line) in content.lines().enumerate().skip(run_impl_line) {
                    if line.contains(kernel_name) {
                        // Found kernel name, look for next line with pure_kernel_name
                        for (j, next_line) in content.lines().enumerate().skip(i + 1) {
                            if next_line.contains(pure_kernel_name) {
                                kernel_to_lines
                                    .entry(kernel_name.to_string())
                                    .or_insert_with(Vec::new)
                                    .push(j + 1 - first_line_number);
                                found = true;
                                break;
                            }
                        }
                        break;
                    }
                }
            }

            // If exact kernel name not found, map all lines with pure kernel name
            if !found {
                for (i, line) in content.lines().enumerate().skip(run_impl_line) {
                    if line.contains(pure_kernel_name) {
                        kernel_to_lines
                            .entry(kernel_name.to_string())
                            .or_insert_with(Vec::new)
                            .push(i + 1 - first_line_number);
                    }
                }
            }
        }
        kernel_to_lines
    }

    // Helper function to build C++ kernel-to-lines lookup map
    // We only consider lines after "::run_impl(" and skip the empty lines at the beginning when computing line numbers
    fn build_cpp_kernel_to_lines_map(
        content: &str,
        kernel_names: &[&str],
        _version: i64,
    ) -> std::collections::HashMap<String, Vec<usize>> {
        // remove empty lines at the beginning and end of the content
        // We need to do this because empty lines are ignored in html <pre> tags
        let content = content
            .lines()
            .skip_while(|line| line.is_empty())
            .collect::<Vec<&str>>()
            .join("\n");
        let mut kernel_to_lines = std::collections::HashMap::new();

        // Find the line number of "::run_impl("
        let run_impl_line = content
            .lines()
            .position(|line| line.contains("::run_impl("))
            .unwrap_or(0);

        // For each kernel name (e.g. triton_poi_fused_mul_1:2):
        // - Extract pure_kernel_name (triton_poi_fused_mul_1) before the ':'
        // - If kernel name found: map to next line containing pure_kernel_name
        // - If kernel_name not found: map to all lines with pure_kernel_name
        for kernel_name in kernel_names {
            // Get pure kernel name before ':' if it exists
            let pure_kernel_name = if let Some(idx) = kernel_name.find(':') {
                &kernel_name[..idx]
            } else {
                kernel_name
            };

            let mut found = false;
            if kernel_name.contains(':') {
                for (i, line) in content.lines().enumerate().skip(run_impl_line) {
                    if valid_line(line, "def")
                        && valid_line(line, "static inline void")
                        && line.contains(kernel_name)
                    {
                        // Found exact kernel name - map to next matching line
                        let next_line = content
                            .lines()
                            .skip(i + 1)
                            .position(|l| l.contains(pure_kernel_name))
                            .map(|pos| i + pos + 2);

                        if let Some(line_num) = next_line {
                            kernel_to_lines
                                .entry(kernel_name.to_string())
                                .or_insert_with(Vec::new)
                                .push(line_num);
                            found = true;
                            break;
                        }
                    }
                }
            }
            if !found {
                for (i, line) in content.lines().enumerate().skip(run_impl_line) {
                    if line.contains(pure_kernel_name) {
                        kernel_to_lines
                            .entry(kernel_name.to_string())
                            .or_insert_with(Vec::new)
                            .push(i + 1);
                    }
                }
            }
        }
        kernel_to_lines
    }

    // Helper function to process mappings from source to target
    fn process_mappings<F>(
        source_mappings: &serde_json::Map<String, serde_json::Value>,
        source_lookup: &std::collections::HashMap<String, usize>,
        _target_lookup: &std::collections::HashMap<String, usize>,
        target_line_processor: F,
    ) -> std::collections::HashMap<usize, Vec<usize>>
    where
        F: Fn(&str) -> Option<usize>,
    {
        let mut result = std::collections::HashMap::new();

        for (source_node, target_nodes) in source_mappings {
            if let Some(source_line) = source_lookup.get(source_node) {
                let mut target_lines = Vec::new();
                if let Some(target_nodes_array) = target_nodes.as_array() {
                    for target_node in target_nodes_array {
                        if let Some(target_node_str) = target_node.as_str() {
                            if let Some(target_line) = target_line_processor(target_node_str) {
                                target_lines.push(target_line);
                            }
                        }
                    }
                }
                if !target_lines.is_empty() {
                    result.insert(*source_line, target_lines);
                }
            }
        }
        result
    }

    // Helper function to process kernel-to-post mappings
    fn process_kernel_to_post_mappings(
        kernel_mappings: &serde_json::Map<String, serde_json::Value>,
        kernel_lookup: &std::collections::HashMap<String, Vec<usize>>,
        post_lookup: &std::collections::HashMap<String, usize>,
    ) -> std::collections::HashMap<usize, Vec<usize>> {
        let mut result = std::collections::HashMap::new();

        for (kernel_name, post_nodes) in kernel_mappings {
            if let Some(kernel_lines) = kernel_lookup.get(kernel_name) {
                for kernel_line in kernel_lines {
                    let mut target_lines = Vec::new();
                    if let Some(post_nodes_array) = post_nodes.as_array() {
                        for post_node in post_nodes_array {
                            if let Some(post_node_str) = post_node.as_str() {
                                if let Some(post_line) = post_lookup.get(post_node_str) {
                                    target_lines.push(*post_line);
                                }
                            }
                        }
                    }
                    if !target_lines.is_empty() {
                        result.insert(*kernel_line, target_lines);
                    }
                }
            }
        }
        result
    }

    // Helper function to process post-to-kernel mappings
    fn process_post_to_kernel_mappings(
        post_mappings: &serde_json::Map<String, serde_json::Value>,
        post_lookup: &std::collections::HashMap<String, usize>,
        kernel_lookup: &std::collections::HashMap<String, Vec<usize>>,
    ) -> std::collections::HashMap<usize, Vec<usize>> {
        let mut result = std::collections::HashMap::new();

        for (post_node, kernel_names) in post_mappings {
            if let Some(post_line) = post_lookup.get(post_node) {
                let mut target_lines = Vec::new();
                if let Some(kernel_names_array) = kernel_names.as_array() {
                    for kernel_name in kernel_names_array {
                        if let Some(kernel_name_str) = kernel_name.as_str() {
                            if let Some(kernel_lines) = kernel_lookup.get(kernel_name_str) {
                                target_lines.extend(kernel_lines);
                            }
                        }
                    }
                }
                if !target_lines.is_empty() {
                    result.insert(*post_line, target_lines);
                }
            }
        }
        result
    }

    // Helper function to convert HashMap to JSON Map
    fn hashmap_to_json_map(
        map: std::collections::HashMap<usize, Vec<usize>>,
    ) -> serde_json::Map<String, serde_json::Value> {
        map.into_iter()
            .map(|(k, v)| (k.to_string(), serde_json::json!(v)))
            .collect()
    }

    let kernel_names: Vec<&str> = node_mappings
        .get("cppCodeToPost")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().map(|k| k.as_str()).collect())
        .unwrap_or_default();

    // Build lookup maps
    let pre_grad_node_to_lines = build_node_to_lines_map(pre_grad_graph_content);
    let post_grad_node_to_lines = build_node_to_lines_map(post_grad_graph_content);
    let py_kernel_to_lines =
        build_python_kernel_to_lines_map(output_code_content, &kernel_names, version);
    let cpp_code_to_lines = build_cpp_kernel_to_lines_map(aot_code_content, &kernel_names, version);

    // Process all mappings using helper functions
    let line_pre_to_post =
        if let Some(pre_to_post) = node_mappings.get("preToPost").and_then(|v| v.as_object()) {
            process_mappings(
                pre_to_post,
                &pre_grad_node_to_lines,
                &post_grad_node_to_lines,
                |node_name| post_grad_node_to_lines.get(node_name).copied(),
            )
        } else {
            std::collections::HashMap::new()
        };

    let line_post_to_pre =
        if let Some(post_to_pre) = node_mappings.get("postToPre").and_then(|v| v.as_object()) {
            process_mappings(
                post_to_pre,
                &post_grad_node_to_lines,
                &pre_grad_node_to_lines,
                |node_name| pre_grad_node_to_lines.get(node_name).copied(),
            )
        } else {
            std::collections::HashMap::new()
        };

    let line_cpp_code_to_post = if let Some(cpp_code_to_post) = node_mappings
        .get("cppCodeToPost")
        .and_then(|v| v.as_object())
    {
        process_kernel_to_post_mappings(
            cpp_code_to_post,
            &cpp_code_to_lines,
            &post_grad_node_to_lines,
        )
    } else {
        std::collections::HashMap::new()
    };

    let line_post_to_cpp_code = if let Some(post_to_cpp_code) = node_mappings
        .get("postToCppCode")
        .and_then(|v| v.as_object())
    {
        process_post_to_kernel_mappings(
            post_to_cpp_code,
            &post_grad_node_to_lines,
            &cpp_code_to_lines,
        )
    } else {
        std::collections::HashMap::new()
    };

    let line_py_code_to_post = if let Some(cpp_code_to_post) = node_mappings
        .get("cppCodeToPost")
        .and_then(|v| v.as_object())
    {
        process_kernel_to_post_mappings(
            cpp_code_to_post,
            &py_kernel_to_lines,
            &post_grad_node_to_lines,
        )
    } else {
        std::collections::HashMap::new()
    };

    let line_post_to_py_code = if let Some(post_to_cpp_code) = node_mappings
        .get("postToCppCode")
        .and_then(|v| v.as_object())
    {
        process_post_to_kernel_mappings(
            post_to_cpp_code,
            &post_grad_node_to_lines,
            &py_kernel_to_lines,
        )
    } else {
        std::collections::HashMap::new()
    };

    // Convert all HashMaps to JSON objects
    serde_json::json!({
        "preToPost": hashmap_to_json_map(line_pre_to_post),
        "postToPre": hashmap_to_json_map(line_post_to_pre),
        "pyCodeToPost": hashmap_to_json_map(line_py_code_to_post),
        "postToPyCode": hashmap_to_json_map(line_post_to_py_code),
        "cppCodeToPost": hashmap_to_json_map(line_cpp_code_to_post),
        "postToCppCode": hashmap_to_json_map(line_post_to_cpp_code)
    })
}
