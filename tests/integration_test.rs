use assert_cmd::Command;
use predicates::boolean::PredicateBooleanExt;
use predicates::str;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use tempfile::tempdir;
use tlparse;

fn prefix_exists(map: &HashMap<PathBuf, String>, prefix: &str) -> bool {
    map.keys()
        .any(|key| key.to_str().map_or(false, |s| s.starts_with(prefix)))
}

#[test]
fn test_parse_simple() {
    let expected_files = [
        "-_0_0_0/aot_forward_graph",
        "-_0_0_0/dynamo_output_graph",
        "index.html",
        "compile_directory.json",
        "failures_and_restarts.html",
        "-_0_0_0/inductor_post_grad_graph",
        "-_0_0_0/inductor_output_code",
    ];
    // Read the test file
    // simple.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/test python test/inductor/test_torchinductor.py  -k test_custom_op_fixed_layout_channels_last_cpu
    let path = Path::new("tests/inputs/simple.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }

    // Check that raw.jsonl exists and has exactly 26 lines (non-payload lines from original, excluding chromium_event entries)
    assert!(
        map.contains_key(&PathBuf::from("raw.jsonl")),
        "raw.jsonl not found in output"
    );
    let shortraw_content = &map[&PathBuf::from("raw.jsonl")];
    let shortraw_lines = shortraw_content.lines().count();
    assert_eq!(
        shortraw_lines, 15,
        "raw.jsonl should have exactly 15 lines (1 string table + 14 log entries, excluding 50 chromium_event entries and 12 str entries)"
    );

    // Verify that the first line contains the string table
    let first_line = shortraw_content
        .lines()
        .next()
        .expect("raw.jsonl should have at least one line");
    assert!(
        first_line.starts_with("{\"string_table\":"),
        "First line of raw.jsonl should be the string table object"
    );
}

#[test]
fn test_parse_compilation_metrics() {
    let expected_files = [
        "-_0_0_1/dynamo_output_graph",
        "-_0_0_1/compilation_metrics",
        "-_1_0_1/dynamo_output_graph",
        "-_1_0_1/compilation_metrics",
        "-_2_0_0/dynamo_output_graph",
        "-_2_0_0/compilation_metrics",
        "index.html",
        "compile_directory.json",
        "failures_and_restarts.html",
    ];
    // Read the test file
    // comp_metrics.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/comp_metrics python test/dynamo/test_misc.py -k test_graph_break_compilation_metrics
    let path = Path::new("tests/inputs/comp_metrics.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }

    // Check that raw.jsonl exists and has exactly 26 lines (non-payload lines from original)
    assert!(
        map.contains_key(&PathBuf::from("raw.jsonl")),
        "raw.jsonl not found in output"
    );
    let shortraw_content = &map[&PathBuf::from("raw.jsonl")];
    let shortraw_lines = shortraw_content.lines().count();
    assert_eq!(shortraw_lines, 13, "raw.jsonl should have exactly 13 lines (1 string table + 12 log entries, excluding 14 str entries)");

    // Verify that the first line contains the string table
    let first_line = shortraw_content
        .lines()
        .next()
        .expect("raw.jsonl should have at least one line");
    assert!(
        first_line.starts_with("{\"string_table\":"),
        "First line of raw.jsonl should be the string table object"
    );

    // Check that exactly the expected payload files exist (no more, no less)
    // With conditional payload writing, only payloads not handled by parsers are written
    let expected_payload_hashes: std::collections::HashSet<&str> = [
        "11726d08889974e57b12edee2812504e",
        "29e35548d59d0e446f0c8a3f3010cc72",
        "e18e1bcb67140c0a67427a6119556f7a",
    ]
    .iter()
    .cloned()
    .collect();

    // Extract actual payload hashes from the output
    let actual_payload_hashes: std::collections::HashSet<String> = map
        .keys()
        .filter_map(|path| {
            path.to_str().and_then(|s| {
                if s.starts_with("payloads/") && s.ends_with(".txt") {
                    Some(
                        s.strip_prefix("payloads/")?
                            .strip_suffix(".txt")?
                            .to_string(),
                    )
                } else {
                    None
                }
            })
        })
        .collect();

    // Convert expected hashes to String for comparison
    let expected_payload_hashes: std::collections::HashSet<String> = expected_payload_hashes
        .iter()
        .map(|s| s.to_string())
        .collect();

    assert_eq!(
        actual_payload_hashes, expected_payload_hashes,
        "Payload file hashes don't match. Expected: {:?}, Actual: {:?}",
        expected_payload_hashes, actual_payload_hashes
    );

    // Verify that raw.jsonl is in JSONL format and contains payload_filename for written payload files
    let shortraw_lines: Vec<&str> = shortraw_content.lines().collect();
    let mut payload_filename_count = 0;
    let mut jsonl_lines_with_log_fields = 0;

    for line in shortraw_lines {
        // Verify each line is valid JSON
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(json_value) => {
                if let Some(obj) = json_value.as_object() {
                    // Check that log fields are present
                    if obj.contains_key("timestamp")
                        && obj.contains_key("thread")
                        && obj.contains_key("pathname")
                    {
                        jsonl_lines_with_log_fields += 1;

                        // Verify log fields have correct types
                        assert!(
                            obj.get("timestamp").unwrap().is_string(),
                            "timestamp should be string"
                        );
                        assert!(
                            obj.get("thread").unwrap().is_number(),
                            "thread should be number"
                        );
                        assert!(
                            obj.get("pathname").unwrap().is_string(),
                            "pathname should be string"
                        );
                        assert!(
                            obj.get("lineno").unwrap().is_number(),
                            "lineno should be number"
                        );
                    }

                    // Check for payload_filename
                    if obj.contains_key("payload_filename") {
                        payload_filename_count += 1;
                        let payload_file = obj.get("payload_filename").unwrap().as_str().unwrap();
                        // Verify the payload_filename points to an actual file in the output
                        let file_exists_in_output = map
                            .keys()
                            .any(|path| path.to_str().map_or(false, |s| s == payload_file));
                        assert!(file_exists_in_output, "payload_filename in raw.jsonl should point to an existing output file: {}", payload_file);
                    }
                }
            }
            Err(e) => {
                panic!("raw.jsonl line is not valid JSON: {} - Error: {}", line, e);
            }
        }
    }

    // Verify we have the right number of lines with log fields (should be all non-empty lines)
    assert!(
        jsonl_lines_with_log_fields > 0,
        "Should have JSONL lines with log fields"
    );

    // We should have at least as many payload_filename entries as payload files written
    // (since parsers can also generate payload files)
    assert!(
        payload_filename_count >= expected_payload_hashes.len(),
        "Number of payload_filename entries ({}) should be at least the number of expected payload files ({})",
        payload_filename_count,
        expected_payload_hashes.len()
    );
}

#[test]
fn test_parse_compilation_failures() {
    let expected_files = [
        "-_0_0_0/dynamo_output_graph",
        "-_0_0_0/compilation_metrics",
        "index.html",
        "compile_directory.json",
        "failures_and_restarts.html",
    ];
    // Read the test file
    // comp_failure.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/comp_metrics python test/dynamo/test_misc.py -k test_graph_break_compilation_metrics_on_failure
    let path = Path::new("tests/inputs/comp_failure.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}

#[test]
fn test_parse_artifact() {
    let expected_files = ["-_0_0_0/fx_graph_cache_hash", "index.html"];
    // Read the test file
    // artifacts.log was generated from the following:
    // NOTE: this test command looks wrong, and is not producing anything close to artifacts.log
    // TORCH_TRACE=~/trace_logs/test python test/inductor/test_torchinductor.py  -k TORCH_TRACE=~/trace_logs/comp_metrics python test/dynamo/test_misc.py -k test_graph_break_compilation_metrics_on_failure
    let path = Path::new("tests/inputs/artifacts.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}

#[test]
fn test_parse_chromium_event() {
    let expected_files = ["chromium_events.json", "index.html"];
    // Read the test file
    // chromium_events.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/comp_metrics python test/dynamo/test_misc.py -k test_graph_break_compilation_metrics_on_failure
    let path = Path::new("tests/inputs/chromium_events.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}

#[test]
fn test_cache_hit_miss() {
    let expected_files = [
        "-_1_0_0/fx_graph_cache_miss_33.json",
        "-_1_0_0/fx_graph_cache_miss_9.json",
        "-_1_0_0/fx_graph_cache_hit_20.json",
        "compile_directory.json",
        "index.html",
    ];
    // Generated via TORCH_TRACE=~/trace_logs/test python test/inductor/test_codecache.py -k test_flex_attention_caching
    let path = Path::new("tests/inputs/cache_hit_miss.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}

#[test]
fn test_export_report() {
    let expected_files = [
        "-_-_-_-/exported_program",
        "index.html",
        "-_-_-_-/symbolic_guard_information",
    ];
    // Read the test file
    // chromium_events.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/test python test/export/test_draft_export.py -k test_complex_data_dependent
    let path = Path::new("tests/inputs/export.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        export: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    println!("{:?}", map.keys());
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}

#[test]
fn test_export_guard_report() {
    let expected_files = [
        "-_-_-_-/exported_program",
        "index.html",
        "-_-_-_-/symbolic_guard_information",
    ];
    // Read the test file
    // chromium_events.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/test python test/export/test_draft_export.py -k test_shape_failure
    let path = Path::new("tests/inputs/export_guard_added.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        export: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    println!("{:?}", map.keys());
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}

#[test]
fn test_execution_order_end_to_end() {
    // Build exec orders from JSON payloads (parser + analyzer together)
    let p0 = r#"{"graph_execution_order": ["G1", {"graph":"G2"}, {"compile_id":"G3"}, {"id":"G4"}]}"#;
    let p1 = r#"{"graph_execution_order": ["G1", "G2", "G3"]}"#;
    let seq0 = tlparse::execution_order::parse_graph_execution_order(p0).unwrap();
    let seq1 = tlparse::execution_order::parse_graph_execution_order(p1).unwrap();
    let exec_orders: fxhash::FxHashMap<u32, Vec<String>> =
        vec![(0_u32, seq0), (1_u32, seq1)].into_iter().collect();

    // Collective schedules: equal at idx0, mismatch at idx1, equal at idx2, only rank0 at idx3
    let collective: fxhash::FxHashMap<(u32, String), Vec<String>> = vec![
        ((0_u32, "G1".into()), vec!["op".into()]),
        ((1_u32, "G1".into()), vec!["op".into()]),
        ((0_u32, "G2".into()), vec!["all_reduce".into(), "barrier".into()]),
        ((1_u32, "G2".into()), vec!["barrier".into(), "all_reduce".into()]),
        ((0_u32, "G3".into()), vec!["noop".into()]),
        ((1_u32, "G3".into()), vec!["noop".into()]),
        ((0_u32, "G4".into()), vec!["noop".into()]),
    ]
    .into_iter()
    .collect();

    // Cache statuses: equal at idx0, blank on rank1 at idx1 (ignored), skew at idx2, rank0-only at idx3
    let cache: fxhash::FxHashMap<(u32, String), String> = vec![
        ((0_u32, "G1".into()), "✅".into()),
        ((1_u32, "G1".into()), "✅".into()),
        ((0_u32, "G2".into()), "✅".into()),
        ((1_u32, "G2".into()), "".into()),
        ((0_u32, "G3".into()), "✅".into()),
        ((1_u32, "G3".into()), "❌".into()),
        ((0_u32, "G4".into()), "❌".into()),
    ]
    .into_iter()
    .collect();

    let report = tlparse::execution_order::analyze_execution_order(&exec_orders, &collective, &cache);
    assert_eq!(report.by_index.len(), 4);

    // idx0: both ranks present, equal schedules and cache => no issues
    assert_eq!(report.by_index[0].by_rank.len(), 2);
    assert!(report.by_index[0].issues.is_empty());

    // idx1: both ranks present, schedule mismatch; blank cache ignored => only ScheduleMismatch
    assert!(report.by_index[1]
        .issues
        .iter()
        .any(|&i| i == tlparse::execution_order::ExecOrderIssue::ScheduleMismatch));
    assert!(!report.by_index[1]
        .issues
        .iter()
        .any(|&i| i == tlparse::execution_order::ExecOrderIssue::CacheSkew));

    // idx2: cache skew across ranks
    assert!(report.by_index[2]
        .issues
        .iter()
        .any(|&i| i == tlparse::execution_order::ExecOrderIssue::CacheSkew));

    // idx3: only rank0 present => nothing to compare
    assert_eq!(report.by_index[3].by_rank.len(), 1);
    assert!(report.by_index[3].issues.is_empty());
}

#[test]
fn test_provenance_tracking_aot_cuda() {
    let expected_files = [
        "-_-_-_-/before_pre_grad_graph_0.txt",
        "-_-_-_-/after_post_grad_graph_6.txt",
        "provenance_tracking_-_-_-_-.html",
        "-_-_-_-/inductor_provenance_tracking_node_mappings_12.json",
    ];
    // Read the test file
    let path = Path::new("tests/inputs/inductor_provenance_aot_cuda_log.txt").to_path_buf();
    let config = tlparse::ParseConfig {
        inductor_provenance: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    println!("{:?}", map.keys());
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }

    // Read the HTML file and verify the line mappings
    let html_path = map
        .keys()
        .find(|p| {
            p.to_str()
                .unwrap()
                .contains("provenance_tracking_-_-_-_-.html")
        })
        .unwrap();
    let html_content = map.get(html_path).unwrap();

    // Extract the line mappings JSON from the script tag
    let script_start = html_content
        .find(r#"<script id="lineMappings" type="application/json">"#)
        .unwrap();
    let json_start = html_content[script_start..].find(">").unwrap() + script_start + 1;
    let json_end = html_content[json_start..].find("</script>").unwrap() + json_start;
    let line_mappings_str = &html_content[json_start..json_end];
    let line_mappings: serde_json::Value = serde_json::from_str(line_mappings_str).unwrap();

    // Verify the line mappings match the expected values
    let expected_mappings = serde_json::json!({
        "cppCodeToPost": {
            "1060": [14, 27, 24],
            "1064": [21, 18, 15],
            "1071": [24],
            "1079": [14, 27, 24],
            "1084": [35, 31, 28, 34, 33, 32]
        },
        "postToCppCode": {
            "14": [1060, 1079],
            "15": [1064],
            "18": [1064],
            "21": [1064],
            "24": [1071, 1060, 1079],
            "27": [1060, 1079],
            "28": [1084],
            "31": [1084],
            "32": [1084],
            "33": [1084],
            "34": [1084],
            "35": [1084]
        },
        "postToPre": {
            "11": [8],
            "14": [8],
            "15": [8],
            "18": [11],
            "21": [14],
            "24": [17],
            "27": [20],
            "28": [20],
            "31": [23],
            "32": [23],
            "33": [23],
            "34": [23],
            "35": [23]
        },
        "postToPyCode": {},
        "preToPost": {
            "11": [18],
            "14": [21],
            "17": [24],
            "20": [27, 28],
            "23": [31, 32, 33, 34, 35],
            "8": [11, 14, 15]
        },
        "pyCodeToPost": {}
    });

    assert_eq!(line_mappings, expected_mappings);
}

#[test]
fn test_provenance_tracking_aot_debug_handle() {
    let expected_files = [
        "-_-_-_-/before_pre_grad_graph_0.txt",
        "-_-_-_-/after_post_grad_graph_6.txt",
        "provenance_tracking_-_-_-_-.html",
        "-_-_-_-/inductor_provenance_tracking_node_mappings_10.json",
    ];

    let path = Path::new("tests/inputs/inductor_provenance_aot_debug_handle_log.txt").to_path_buf();
    let config = tlparse::ParseConfig {
        inductor_provenance: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();

    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }

    // Read the HTML file and verify the line mappings
    let html_path = map
        .keys()
        .find(|p| {
            p.to_str()
                .unwrap()
                .contains("provenance_tracking_-_-_-_-.html")
        })
        .unwrap();
    let html_content = map.get(html_path).unwrap();

    // Extract the line mappings JSON from the script tag
    let script_start = html_content
        .find(r#"<script id="lineMappings" type="application/json">"#)
        .unwrap();
    let json_start = html_content[script_start..].find(">").unwrap() + script_start + 1;
    let json_end = html_content[json_start..].find("</script>").unwrap() + json_start;
    let line_mappings_str = &html_content[json_start..json_end];
    let line_mappings: serde_json::Value = serde_json::from_str(line_mappings_str).unwrap();

    // Verify the line mappings match the expected values for debug handle version
    let expected_mappings = serde_json::json!(        {
      "cppCodeToPost": {
        "1074": [
          12
        ],
        "1079": [
          19,
          16,
          13
        ],
        "1087": [
          22
        ],
        "1096": [
          25
        ],
        "1102": [
          33,
          29,
          26,
          32,
          31,
          30
        ]
      },
      "postToCppCode": {
        "12": [
          1074
        ],
        "13": [
          1079
        ],
        "16": [
          1079
        ],
        "19": [
          1079
        ],
        "22": [
          1087
        ],
        "25": [
          1096
        ],
        "26": [
          1102
        ],
        "29": [
          1102
        ],
        "30": [
          1102
        ],
        "31": [
          1102
        ],
        "32": [
          1102
        ],
        "33": [
          1102
        ]
      },
      "postToPre": {
        "11": [
          8
        ],
        "12": [
          8
        ],
        "13": [
          8
        ],
        "16": [
          11
        ],
        "19": [
          14
        ],
        "22": [
          17
        ],
        "25": [
          20
        ],
        "26": [
          20
        ],
        "29": [
          23
        ],
        "30": [
          23
        ],
        "31": [
          23
        ],
        "32": [
          23
        ],
        "33": [
          23
        ]
      },
      "postToPyCode": {},
      "preToPost": {
        "11": [
          16
        ],
        "14": [
          19
        ],
        "17": [
          22
        ],
        "20": [
          25,
          26
        ],
        "23": [
          29,
          30,
          31,
          32,
          33
        ],
        "8": [
          11,
          12,
          13
        ]
      },
      "pyCodeToPost": {}
    });

    assert_eq!(line_mappings, expected_mappings);
}

#[test]
fn test_provenance_tracking_aot_log() {
    let expected_files = [
        "-_-_-_-/before_pre_grad_graph_0.txt",
        "-_-_-_-/after_post_grad_graph_6.txt",
        "provenance_tracking_-_-_-_-.html",
        "-_-_-_-/inductor_provenance_tracking_node_mappings_11.json",
    ];

    let path = Path::new("tests/inputs/inductor_provenance_aot_log.txt").to_path_buf();
    let config = tlparse::ParseConfig {
        inductor_provenance: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();

    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }

    // Read the HTML file and verify the line mappings
    let html_path = map
        .keys()
        .find(|p| {
            p.to_str()
                .unwrap()
                .contains("provenance_tracking_-_-_-_-.html")
        })
        .unwrap();
    let html_content = map.get(html_path).unwrap();

    // Extract the line mappings JSON from the script tag
    let script_start = html_content
        .find(r#"<script id="lineMappings" type="application/json">"#)
        .unwrap();
    let json_start = html_content[script_start..].find(">").unwrap() + script_start + 1;
    let json_end = html_content[json_start..].find("</script>").unwrap() + json_start;
    let line_mappings_str = &html_content[json_start..json_end];
    let line_mappings: serde_json::Value = serde_json::from_str(line_mappings_str).unwrap();

    // For jit log, we expect similar structure to jit cuda but with different kernel names
    let expected_mappings = serde_json::json!(  {
      "cppCodeToPost": {
        "813": [
          12,
          24,
          21
        ],
        "821": [
          18,
          15,
          21
        ],
        "829": [
          12,
          24,
          21
        ],
        "834": [
          31,
          27,
          30,
          29,
          28
        ]
      },
      "postToCppCode": {
        "12": [
          813,
          829
        ],
        "15": [
          821
        ],
        "18": [
          821
        ],
        "21": [
          821,
          813,
          829
        ],
        "24": [
          813,
          829
        ],
        "27": [
          834
        ],
        "28": [
          834
        ],
        "29": [
          834
        ],
        "30": [
          834
        ],
        "31": [
          834
        ]
      },
      "postToPre": {
        "11": [
          8
        ],
        "12": [
          8
        ],
        "15": [
          11
        ],
        "18": [
          14
        ],
        "21": [
          17
        ],
        "24": [
          20
        ],
        "27": [
          23
        ],
        "28": [
          23
        ],
        "29": [
          23
        ],
        "30": [
          23
        ],
        "31": [
          23
        ]
      },
      "postToPyCode": {},
      "preToPost": {
        "11": [
          15
        ],
        "14": [
          18
        ],
        "17": [
          21
        ],
        "20": [
          24
        ],
        "23": [
          27,
          28,
          29,
          30,
          31
        ],
        "8": [
          11,
          12
        ]
      },
      "pyCodeToPost": {}
    });

    assert_eq!(line_mappings, expected_mappings);
}

#[test]
fn test_provenance_tracking_aot_log_old() {
    let expected_files = [
        "-_-_-_-/inductor_pre_grad_graph_0.txt",
        "-_-_-_-/inductor_post_grad_graph_8.txt",
        "provenance_tracking_-_-_-_-.html",
        "-_-_-_-/inductor_provenance_tracking_node_mappings_11.json",
    ];

    let path = Path::new("tests/inputs/inductor_provenance_aot_log_old.txt").to_path_buf();
    let config = tlparse::ParseConfig {
        inductor_provenance: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();

    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }

    // Read the HTML file and verify the line mappings
    let html_path = map
        .keys()
        .find(|p| {
            p.to_str()
                .unwrap()
                .contains("provenance_tracking_-_-_-_-.html")
        })
        .unwrap();
    let html_content = map.get(html_path).unwrap();

    // Extract the line mappings JSON from the script tag
    let script_start = html_content
        .find(r#"<script id="lineMappings" type="application/json">"#)
        .unwrap();
    let json_start = html_content[script_start..].find(">").unwrap() + script_start + 1;
    let json_end = html_content[json_start..].find("</script>").unwrap() + json_start;
    let line_mappings_str = &html_content[json_start..json_end];
    let line_mappings: serde_json::Value = serde_json::from_str(line_mappings_str).unwrap();

    // For old log, we expect the same structure as the regular aot log
    let expected_mappings = serde_json::json!(       {
      "cppCodeToPost": {
        "704": [
          21
        ],
        "717": [
          31,
          27,
          30,
          29,
          28
        ]
      },
      "postToCppCode": {
        "21": [
          704
        ],
        "27": [
          717
        ],
        "28": [
          717
        ],
        "29": [
          717
        ],
        "30": [
          717
        ],
        "31": [
          717
        ]
      },
      "postToPre": {
        "11": [
          8
        ],
        "12": [
          8
        ],
        "15": [
          11
        ],
        "18": [
          14
        ],
        "21": [
          17
        ],
        "24": [
          20
        ],
        "27": [
          23
        ],
        "28": [
          23
        ],
        "29": [
          23
        ],
        "30": [
          23
        ],
        "31": [
          23
        ]
      },
      "postToPyCode": {},
      "preToPost": {
        "11": [
          15
        ],
        "14": [
          18
        ],
        "17": [
          21
        ],
        "20": [
          24
        ],
        "23": [
          27,
          28,
          29,
          30,
          31
        ],
        "8": [
          11,
          12
        ]
      },
      "pyCodeToPost": {}
    });

    assert_eq!(line_mappings, expected_mappings);
}

#[test]
fn test_provenance_tracking_jit_cuda() {
    let expected_files = [
        "-_0_0_0/before_pre_grad_graph_1.txt",
        "-_0_0_0/after_post_grad_graph_8.txt",
        "provenance_tracking_-_0_0_0.html",
        "-_0_0_0/inductor_provenance_tracking_node_mappings_14.json",
    ];

    let path = Path::new("tests/inputs/inductor_provenance_jit_cuda_log.txt").to_path_buf();
    let config = tlparse::ParseConfig {
        inductor_provenance: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();

    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }

    // Read the HTML file and verify the line mappings
    let html_path = map
        .keys()
        .find(|p| {
            p.to_str()
                .unwrap()
                .contains("provenance_tracking_-_0_0_0.html")
        })
        .unwrap();
    let html_content = map.get(html_path).unwrap();

    // Extract the line mappings JSON from the script tag
    let script_start = html_content
        .find(r#"<script id="lineMappings" type="application/json">"#)
        .unwrap();
    let json_start = html_content[script_start..].find(">").unwrap() + script_start + 1;
    let json_end = html_content[json_start..].find("</script>").unwrap() + json_start;
    let line_mappings_str = &html_content[json_start..json_end];
    let line_mappings: serde_json::Value = serde_json::from_str(line_mappings_str).unwrap();

    // Verify the line mappings match the expected values for jit cuda
    let expected_mappings = serde_json::json!(        {
      "cppCodeToPost": {},
      "postToCppCode": {},
      "postToPre": {
        "11": [
          14
        ],
        "14": [
          17
        ],
        "17": [
          20
        ],
        "20": [
          23
        ],
        "21": [
          23
        ],
        "24": [
          26
        ],
        "25": [
          26
        ],
        "26": [
          26
        ],
        "27": [
          26
        ],
        "28": [
          26
        ],
        "4": [
          11
        ],
        "7": [
          11
        ],
        "8": [
          11
        ]
      },
      "postToPyCode": {
        "11": [
          192
        ],
        "14": [
          192
        ],
        "17": [
          197,
          186,
          201
        ],
        "20": [
          186,
          201
        ],
        "21": [
          207
        ],
        "24": [
          207
        ],
        "25": [
          207
        ],
        "26": [
          207
        ],
        "27": [
          207
        ],
        "28": [
          207
        ],
        "7": [
          186,
          201
        ],
        "8": [
          192
        ]
      },
      "preToPost": {
        "11": [
          4,
          7,
          8
        ],
        "14": [
          11
        ],
        "17": [
          14
        ],
        "20": [
          17
        ],
        "23": [
          20,
          21
        ],
        "26": [
          24,
          25,
          26,
          27,
          28
        ]
      },
      "pyCodeToPost": {
        "186": [
          7,
          20,
          17
        ],
        "192": [
          14,
          11,
          8
        ],
        "197": [
          17
        ],
        "201": [
          7,
          20,
          17
        ],
        "207": [
          28,
          24,
          21,
          27,
          26,
          25
        ]
      }
    });

    assert_eq!(line_mappings, expected_mappings);
}

#[test]
fn test_provenance_tracking_jit_log() {
    let expected_files = [
        "-_0_0_0/before_pre_grad_graph_1.txt",
        "-_0_0_0/after_post_grad_graph_8.txt",
        "provenance_tracking_-_0_0_0.html",
        "-_0_0_0/inductor_provenance_tracking_node_mappings_13.json",
    ];

    let path = Path::new("tests/inputs/inductor_provenance_jit_log.txt").to_path_buf();
    let config = tlparse::ParseConfig {
        inductor_provenance: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();

    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }

    // Read the HTML file and verify the line mappings
    let html_path = map
        .keys()
        .find(|p| {
            p.to_str()
                .unwrap()
                .contains("provenance_tracking_-_0_0_0.html")
        })
        .unwrap();
    let html_content = map.get(html_path).unwrap();

    // Extract the line mappings JSON from the script tag
    let script_start = html_content
        .find(r#"<script id="lineMappings" type="application/json">"#)
        .unwrap();
    let json_start = html_content[script_start..].find(">").unwrap() + script_start + 1;
    let json_end = html_content[json_start..].find("</script>").unwrap() + json_start;
    let line_mappings_str = &html_content[json_start..json_end];
    let line_mappings: serde_json::Value = serde_json::from_str(line_mappings_str).unwrap();

    // For jit log, we expect similar structure to jit cuda but with different kernel names
    let expected_mappings = serde_json::json!(       {
      "cppCodeToPost": {},
      "postToCppCode": {},
      "postToPre": {
        "11": [
          17
        ],
        "14": [
          20
        ],
        "17": [
          23
        ],
        "20": [
          26
        ],
        "21": [
          26
        ],
        "22": [
          26
        ],
        "23": [
          26
        ],
        "24": [
          26
        ],
        "4": [
          11
        ],
        "5": [
          11
        ],
        "8": [
          14
        ]
      },
      "postToPyCode": {
        "11": [
          138
        ],
        "14": [
          138,
          132,
          142
        ],
        "17": [
          132,
          142
        ],
        "20": [
          147
        ],
        "21": [
          147
        ],
        "22": [
          147
        ],
        "23": [
          147
        ],
        "24": [
          147
        ],
        "5": [
          132,
          142
        ],
        "8": [
          138
        ]
      },
      "preToPost": {
        "11": [
          4,
          5
        ],
        "14": [
          8
        ],
        "17": [
          11
        ],
        "20": [
          14
        ],
        "23": [
          17
        ],
        "26": [
          20,
          21,
          22,
          23,
          24
        ]
      },
      "pyCodeToPost": {
        "132": [
          5,
          17,
          14
        ],
        "138": [
          11,
          8,
          14
        ],
        "142": [
          5,
          17,
          14
        ],
        "147": [
          24,
          20,
          23,
          22,
          21
        ]
      }
    });

    assert_eq!(line_mappings, expected_mappings);
}

#[test]
fn test_provenance_tracking_jit_debug_handle() {
    let expected_files = [
        "-_0_0_0/before_pre_grad_graph_1.txt",
        "-_0_0_0/after_post_grad_graph_11.txt",
        "provenance_tracking_-_0_0_0.html",
        "-_0_0_0/inductor_provenance_tracking_node_mappings_14.json",
    ];

    let path = Path::new("tests/inputs/inductor_provenance_jit_debug_handle_log.txt").to_path_buf();
    let config = tlparse::ParseConfig {
        inductor_provenance: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, &config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();

    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }

    // Read the HTML file and verify the line mappings
    let html_path = map
        .keys()
        .find(|p| {
            p.to_str()
                .unwrap()
                .contains("provenance_tracking_-_0_0_0.html")
        })
        .unwrap();
    let html_content = map.get(html_path).unwrap();

    // Extract the line mappings JSON from the script tag
    let script_start = html_content
        .find(r#"<script id="lineMappings" type="application/json">"#)
        .unwrap();
    let json_start = html_content[script_start..].find(">").unwrap() + script_start + 1;
    let json_end = html_content[json_start..].find("</script>").unwrap() + json_start;
    let line_mappings_str = &html_content[json_start..json_end];
    let line_mappings: serde_json::Value = serde_json::from_str(line_mappings_str).unwrap();

    // For jit log, we expect similar structure to jit cuda but with different kernel names
    let expected_mappings = serde_json::json!(         {
      "cppCodeToPost": {},
      "postToCppCode": {},
      "postToPre": {
        "12": [
          17
        ],
        "15": [
          20
        ],
        "18": [
          23
        ],
        "19": [
          23
        ],
        "22": [
          26
        ],
        "23": [
          26
        ],
        "24": [
          26
        ],
        "25": [
          26
        ],
        "26": [
          26
        ],
        "4": [
          11
        ],
        "5": [
          11
        ],
        "6": [
          11
        ],
        "9": [
          14
        ]
      },
      "postToPyCode": {
        "12": [
          204
        ],
        "15": [
          211
        ],
        "18": [
          216
        ],
        "19": [
          223
        ],
        "22": [
          223
        ],
        "23": [
          223
        ],
        "24": [
          223
        ],
        "25": [
          223
        ],
        "26": [
          223
        ],
        "29": [
          204
        ],
        "5": [
          197
        ],
        "6": [
          204
        ],
        "9": [
          204
        ]
      },
      "preToPost": {
        "11": [
          4,
          5,
          6
        ],
        "14": [
          9
        ],
        "17": [
          12
        ],
        "20": [
          15
        ],
        "23": [
          18,
          19
        ],
        "26": [
          22,
          23,
          24,
          25,
          26
        ]
      },
      "pyCodeToPost": {
        "197": [
          5
        ],
        "204": [
          12,
          9,
          6,
          29
        ],
        "211": [
          15
        ],
        "216": [
          18
        ],
        "223": [
          26,
          22,
          19,
          25,
          24,
          23
        ]
      }
    });

    assert_eq!(line_mappings, expected_mappings);
}

#[test]
fn test_provenance_stack_trace_readable() {
    let path = Path::new("tests/inputs/inductor_provenance_extended_log.txt").to_path_buf();
    let config = tlparse::ParseConfig {
        inductor_provenance: true,
        ..Default::default()
    };
    let map: HashMap<PathBuf, String> = tlparse::parse_path(&path, &config)
        .unwrap()
        .into_iter()
        .collect();

    assert!(map.keys().any(|k| {
        let s = k.to_str().unwrap_or("");
        s.contains("inductor_provenance_tracking_kernel_stack_traces") && s.ends_with(".json")
    }));
    assert!(map.keys().any(|k| {
        let s = k.to_str().unwrap_or("");
        s.contains("inductor_provenance_tracking_kernel_stack_traces")
            && s.ends_with("_readable.html")
    }));

    let html = map
        .iter()
        .find(|(k, _)| {
            let s = k.to_str().unwrap_or("");
            s.contains("inductor_provenance_tracking_kernel_stack_traces")
                && s.ends_with("_readable.html")
        })
        .unwrap()
        .1;
    assert!(html.contains("line 1\n  foo()"));
    assert!(!html.contains("\\n"));

    let index_html = map.get(Path::new("index.html")).unwrap();
    assert!(index_html.contains("_readable.html\">readable_html</a>"));
}

#[test]
fn test_all_ranks_basic() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_logs");
    let temp_dir = tempdir().unwrap();
    let out_dir = temp_dir.path().join("out");

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(&input_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(&out_dir)
        .arg("--no-browser");
    cmd.assert().success();

    let rank0_index = out_dir.join("rank_0/index.html");
    let rank1_index = out_dir.join("rank_1/index.html");
    let landing_page = out_dir.join("index.html");

    assert!(rank0_index.exists());
    assert!(rank1_index.exists());
    assert!(landing_page.exists());

    let landing_content = fs::read_to_string(landing_page).unwrap();
    assert!(landing_content.contains(r#"<a href="rank_0/index.html">"#));
    assert!(landing_content.contains(r#"<a href="rank_1/index.html">"#));
    Ok(())
}

#[test]
fn test_all_ranks_messy_input() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_messy_input");
    let temp_dir = tempdir().unwrap();
    let out_dir = temp_dir.path().join("out");

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(&input_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(&out_dir)
        .arg("--no-browser");

    cmd.assert().success();

    // Check for landing page and rank-specific index files
    let landing_page = out_dir.join("index.html");
    let rank0_index = out_dir.join("rank_0/index.html");
    let rank1_index = out_dir.join("rank_1/index.html");

    assert!(
        rank0_index.exists(),
        "rank 0 index.html should exist in messy input test"
    );
    assert!(
        rank1_index.exists(),
        "rank 1 index.html should exist in messy input test"
    );
    assert!(
        landing_page.exists(),
        "toplevel index.html should exist in messy input test"
    );

    let landing_content = fs::read_to_string(landing_page).unwrap();
    assert!(landing_content.contains(r#"<a href="rank_0/index.html">"#));
    assert!(landing_content.contains(r#"<a href="rank_1/index.html">"#));
    Ok(())
}

#[test]
fn test_all_ranks_no_browser() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_logs");
    let temp_dir = tempdir().unwrap();
    let out_dir = temp_dir.path().join("out");

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(&input_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(&out_dir)
        .arg("--no-browser");

    cmd.assert().success().stdout(
        str::contains("Multi-rank report generated").and(str::contains(out_dir.to_str().unwrap())),
    );

    // Check that files were created but don't try to open them
    let rank0_index = out_dir.join("rank_0/index.html");
    let rank1_index = out_dir.join("rank_1/index.html");
    let landing_page = out_dir.join("index.html");

    assert!(rank0_index.exists());
    assert!(rank1_index.exists());
    assert!(landing_page.exists());
    Ok(())
}

#[test]
fn test_all_ranks_with_latest_fails() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_logs");
    let temp_dir = tempdir().unwrap();
    let out_dir = temp_dir.path().join("out");

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(&input_dir)
        .arg("--all-ranks-html")
        .arg("--latest")
        .arg("-o")
        .arg(&out_dir)
        .arg("--no-browser");

    cmd.assert().failure().stderr(str::contains(
        "--latest cannot be used with --all-ranks-html",
    ));

    Ok(())
}

#[test]
fn test_all_ranks_no_logs() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    let empty_dir = temp_dir.path();

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(empty_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("--no-browser");

    cmd.assert()
        .failure()
        .stderr(str::contains("No rank log files found"));

    Ok(())
}

#[test]
fn test_all_ranks_chromium_events_combined() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_logs");
    let temp_out_dir = tempdir()?;
    let out_dir = temp_out_dir.path();

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(&input_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(out_dir)
        .arg("--no-browser");
    cmd.assert().success();

    // check that chromium_events.json is created and contains events from all ranks
    let combined_events_path = out_dir.join("chromium_events.json");
    assert!(combined_events_path.exists());

    let events_content = fs::read_to_string(combined_events_path)?;
    let events: Vec<serde_json::Value> = serde_json::from_str(&events_content)?;
    assert!(!events.is_empty());

    // collect all unique process IDs (ranks) from the events
    let pids: std::collections::HashSet<u64> = events
        .iter()
        .filter_map(|event| event.get("pid").and_then(|v| v.as_u64()))
        .collect();

    let expected_pids: std::collections::HashSet<u64> = [0, 2, 3].iter().cloned().collect();
    assert_eq!(pids, expected_pids);

    // verify each rank-specific chromium_events.json file
    for rank in 0u64..=3 {
        let rank_events_path = out_dir.join(format!("rank_{}/chromium_events.json", rank));
        assert!(rank_events_path.exists());
        let rank_events_content = fs::read_to_string(&rank_events_path)?;
        let rank_events: Vec<serde_json::Value> = serde_json::from_str(&rank_events_content)?;

        if expected_pids.contains(&(rank as u64)) {
            assert!(!rank_events.is_empty());
            let combined_for_rank: Vec<&serde_json::Value> = events
                .iter()
                .filter(|ev| ev.get("pid").and_then(|v| v.as_u64()) == Some(rank as u64))
                .collect();
            assert_eq!(rank_events.len(), combined_for_rank.len());
        } else {
            assert!(rank_events.is_empty());
        }
    }

    let landing_page_path = out_dir.join("index.html");
    assert!(landing_page_path.exists());
    let landing_content = fs::read_to_string(landing_page_path)?;
    for i in 0..4 {
        assert!(landing_content.contains(&format!("rank_{}", i)));
        assert!(out_dir.join(format!("rank_{}/index.html", i)).exists());
    }

    Ok(())
}

#[test]
fn test_all_ranks_chromium_events_sparse() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_logs");
    let temp_out_dir = tempdir()?;
    let out_dir = temp_out_dir.path();

    let chromium_log_source = Path::new("tests/inputs/chromium_events.log");

    // Rank 0 and 2 will have traces rank 1 will have an empty log (no trace events).
    fs::copy(
        &chromium_log_source,
        input_dir.join("dedicated_log_torch_trace_rank_0.log"),
    )?;

    {
        let rank1_path = input_dir.join("dedicated_log_torch_trace_rank_1.log");
        fs::File::create(rank1_path)?;
    }

    fs::copy(
        &chromium_log_source,
        input_dir.join("dedicated_log_torch_trace_rank_2.log"),
    )?;

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(input_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(out_dir)
        .arg("--no-browser");
    cmd.assert().success();

    let combined_events_path = out_dir.join("chromium_events.json");
    assert!(combined_events_path.exists());

    let events_content = fs::read_to_string(combined_events_path)?;
    let events: Vec<serde_json::Value> = serde_json::from_str(&events_content)?;
    assert!(!events.is_empty());

    // collect all unique process IDs (ranks) from the events
    let pids: std::collections::HashSet<u64> = events
        .iter()
        .filter_map(|event| event.get("pid").and_then(|v| v.as_u64()))
        .collect();

    let expected_pids: std::collections::HashSet<u64> = [0, 2, 3].iter().cloned().collect();
    assert_eq!(pids, expected_pids);

    // verify each rank-specific chromium_events.json file
    for rank in 0u64..=3 {
        let rank_events_path = out_dir.join(format!("rank_{}/chromium_events.json", rank));
        assert!(rank_events_path.exists());
        let rank_events_content = fs::read_to_string(&rank_events_path)?;
        let rank_events: Vec<serde_json::Value> = serde_json::from_str(&rank_events_content)?;

        if expected_pids.contains(&(rank as u64)) {
            assert!(!rank_events.is_empty());
            let combined_for_rank: Vec<&serde_json::Value> = events
                .iter()
                .filter(|ev| ev.get("pid").and_then(|v| v.as_u64()) == Some(rank as u64))
                .collect();
            assert_eq!(rank_events.len(), combined_for_rank.len());
        } else {
            assert!(rank_events.is_empty());
        }
    }

    let landing_page_path = out_dir.join("index.html");
    assert!(landing_page_path.exists());
    let landing_content = fs::read_to_string(landing_page_path)?;

    for i in 0..4 {
        assert!(landing_content.contains(&format!("rank_{}", i)));
    }

    assert!(landing_content.contains("chromium_events.json"));

    Ok(())
}

// Detect diverging compile-ID sets: should raise warning.
#[test]
fn test_diverging_compile_ids_warning() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_logs");
    let temp_dir = tempdir().unwrap();
    let out_dir = temp_dir.path();

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(&input_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(out_dir)
        .arg("--no-browser");
    cmd.assert().success();

    let landing_page = out_dir.join("index.html");
    assert!(
        landing_page.exists(),
        "Expected {} to exist",
        landing_page.display()
    );
    let landing_content = fs::read_to_string(&landing_page)?;
    assert!(
        landing_content.contains("Diverging Compilation IDs detected"),
        "Expected divergence warning to be present"
    );

    Ok(())
}

// Two ranks with identical logs, no divergence warning
#[test]
fn test_no_compile_id_divergence() -> Result<(), Box<dyn std::error::Error>> {
    // Create temp input dir with identical logs for rank 0 and 1
    let temp_in = tempdir()?;
    let src_log = PathBuf::from("tests/inputs/simple.log");

    for rank in 0..=1 {
        let dest = temp_in
            .path()
            .join(format!("dedicated_log_torch_trace_rank_{}.log", rank));
        fs::copy(&src_log, dest)?;
    }

    let temp_out = tempdir()?;

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(temp_in.path())
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(temp_out.path())
        .arg("--no-browser");
    cmd.assert().success();

    let landing_page = temp_out.path().join("index.html");
    assert!(
        landing_page.exists(),
        "Expected {} to exist",
        landing_page.display()
    );
    let landing_content = fs::read_to_string(&landing_page)?;
    assert!(
        !landing_content.contains("Diverging Compilation IDs detected"),
        "Did not expect divergence warning for identical logs"
    );

    Ok(())
}

// Detect diverging cache hit/miss patterns: should raise warning
#[test]
fn test_diverging_cache_events_warning() -> Result<(), Box<dyn std::error::Error>> {
    // Create temp input dir with different logs for rank 0 and 1
    let temp_in = tempdir()?;
    let src_log_hits = PathBuf::from("tests/inputs/cache_hit_miss.log");
    let src_log_no_hits = PathBuf::from("tests/inputs/simple.log");

    fs::copy(
        &src_log_hits,
        temp_in.path().join("dedicated_log_torch_trace_rank_0.log"),
    )?;
    fs::copy(
        &src_log_no_hits,
        temp_in.path().join("dedicated_log_torch_trace_rank_1.log"),
    )?;

    let temp_out = tempdir()?;

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(temp_in.path())
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(temp_out.path())
        .arg("--no-browser");
    cmd.assert().success();

    let landing_page = temp_out.path().join("index.html");
    let landing_content = fs::read_to_string(&landing_page)?;
    assert!(landing_content.contains("Diverging Cache hit/miss patterns detected"));

    Ok(())
}

// Two ranks with identical cache logs, no divergence warning
#[test]
fn test_no_cache_event_divergence() -> Result<(), Box<dyn std::error::Error>> {
    // Create temp input dir with identical logs for rank 0 and 1
    let temp_in = tempdir()?;
    let src_log = PathBuf::from("tests/inputs/cache_hit_miss.log");

    for rank in 0..=1 {
        let dest = temp_in
            .path()
            .join(format!("dedicated_log_torch_trace_rank_{}.log", rank));
        fs::copy(&src_log, dest)?;
    }

    let temp_out = tempdir()?;

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(temp_in.path())
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(temp_out.path())
        .arg("--no-browser");
    cmd.assert().success();

    let landing_page = temp_out.path().join("index.html");
    let landing_content = fs::read_to_string(&landing_page)?;
    assert!(!landing_content.contains("Diverging Cache hit/miss patterns detected"));

    Ok(())
}

// Test diverging cache hit/miss patterns using the existing multi_rank_logs directory should create > 2 groups
#[test]
fn test_diverging_cache_events_multiple_groups() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_logs");
    let temp_out = tempdir()?;

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(&input_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(temp_out.path())
        .arg("--no-browser");
    cmd.assert().success();

    let landing_page = temp_out.path().join("index.html");
    let landing_content = fs::read_to_string(&landing_page)?;
    assert!(landing_content.contains("Diverging Cache hit/miss patterns detected"));

    Ok(())
}

#[test]
fn test_collective_schedule_parsing() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_schedule");
    let temp_dir = tempdir().unwrap();
    let out_dir = temp_dir.path().join("out");

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(&input_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(&out_dir)
        .arg("--no-browser");
    cmd.assert().success();

    // Check that collective schedule files are created for each rank
    for rank in 0..=2 {
        let rank_dir = out_dir.join(format!("rank_{}", rank));
        assert!(rank_dir.exists(), "rank_{} directory should exist", rank);

        let index_file = rank_dir.join("index.html");
        assert!(index_file.exists(), "rank_{} index.html should exist", rank);
    }

    // Check that landing page exists
    let landing_page = out_dir.join("index.html");
    assert!(landing_page.exists(), "Landing page should exist");

    // Check collective_schedules.json exists and has correct structure
    let collective_schedules_file = out_dir.join("collective_schedules.json");
    assert!(collective_schedules_file.exists());

    let schedules: Vec<serde_json::Value> =
        serde_json::from_str(&fs::read_to_string(&collective_schedules_file)?)?;
    assert!(!schedules.is_empty());

    // Verify ranks 0 and 2 have same ops, rank 1 is different
    let rank0_ops = schedules
        .iter()
        .find(|s| s["rank"] == 0 && s["graph"] == "-_0_0_0")
        .map(|s| &s["ops"])
        .unwrap();
    let rank1_ops = schedules
        .iter()
        .find(|s| s["rank"] == 1 && s["graph"] == "-_0_0_0")
        .map(|s| &s["ops"])
        .unwrap();
    let rank2_ops = schedules
        .iter()
        .find(|s| s["rank"] == 2 && s["graph"] == "-_0_0_0")
        .map(|s| &s["ops"])
        .unwrap();

    assert_eq!(rank0_ops, rank2_ops);
    assert_ne!(rank0_ops, rank1_ops);
    assert_eq!(rank0_ops.as_array().unwrap().len(), 6);
    assert_eq!(rank1_ops.as_array().unwrap().len(), 4);

    Ok(())
}

#[test]
fn test_collective_schedule_no_divergence() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path();

    // Copy identical logs (rank 0 and 2 have same collective schedule)
    fs::copy(
        "tests/inputs/multi_rank_schedule/dedicated_log_torch_trace_rank_0_6u3fubwl.log",
        input_dir.join("dedicated_log_torch_trace_rank_0.log"),
    )?;
    fs::copy(
        "tests/inputs/multi_rank_schedule/dedicated_log_torch_trace_rank_2.log",
        input_dir.join("dedicated_log_torch_trace_rank_2.log"),
    )?;

    let temp_out_dir = tempdir().unwrap();
    let out_dir = temp_out_dir.path();

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(input_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(out_dir)
        .arg("--no-browser");
    cmd.assert().success();

    let landing_page = out_dir.join("index.html");
    assert!(landing_page.exists(), "Landing page should exist");
    let html_content = fs::read_to_string(&landing_page)?;

    // Should NOT have desync warning since ranks 0 and 2 have identical collective schedules
    assert!(!html_content.contains("Warning:</strong> Diverging collective operation sequences"));

    Ok(())
}

#[test]
fn test_collective_schedule_with_divergence() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_schedule");
    let temp_dir = tempdir().unwrap();
    let out_dir = temp_dir.path();

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(&input_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(out_dir)
        .arg("--no-browser");
    cmd.assert().success();

    let landing_page = out_dir.join("index.html");
    assert!(landing_page.exists(), "Landing page should exist");
    let html_content = fs::read_to_string(&landing_page)?;

    // Should have desync warning since rank 1 has different collective schedule
    assert!(html_content.contains("Warning:</strong> Diverging collective operation sequences"));

    // Check that ranks 0 and 2 are grouped (same sequence)
    assert!(html_content.contains("Ranks: 0, 2"));

    // Check that rank 1 separate (different sequence)
    assert!(html_content.contains("Ranks: 1"));

    Ok(())
}

#[test]
fn test_runtime_estimation_parsing() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_runtime");
    let out_dir = input_dir.join("out");

    Command::cargo_bin("tlparse")?
        .arg(&input_dir)
        .args(&["--all-ranks-html", "--overwrite", "-o"])
        .arg(&out_dir)
        .arg("--no-browser")
        .assert()
        .success();

    let estimations: Vec<serde_json::Value> = serde_json::from_str(&fs::read_to_string(
        out_dir.join("runtime_estimations.json"),
    )?)?;

    assert!(!estimations.is_empty());
    assert!(estimations.iter().any(|e| e["rank"] == 0));
    assert!(estimations.iter().any(|e| e["rank"] == 1));

    // Verify structure
    for estimation in &estimations {
        for op in estimation["ops"].as_array().unwrap() {
            assert!(op["name"].is_string() && op["estimated_runtime_ns"].is_number());
            assert!(!op.as_object().unwrap().contains_key("type"));
        }
    }

    Ok(())
}

fn setup_runtime_test_with_ranks(
    ranks: &[u32],
) -> Result<(tempfile::TempDir, tempfile::TempDir), Box<dyn std::error::Error>> {
    let temp_in = tempdir()?;
    let temp_out = tempdir()?;
    let src_dir = PathBuf::from("tests/inputs/multi_rank_runtime");

    for &rank in ranks {
        let src_file = src_dir.join(format!("dedicated_log_torch_trace_rank_{}.log", rank));
        let dest_file = temp_in
            .path()
            .join(format!("dedicated_log_torch_trace_rank_{}.log", rank));
        fs::copy(&src_file, &dest_file)?;
    }

    Ok((temp_in, temp_out))
}

#[test]
fn test_runtime_analysis_working() -> Result<(), Box<dyn std::error::Error>> {
    let (input_dir, output_dir) = setup_runtime_test_with_ranks(&[0, 1, 2, 3])?;

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(input_dir.path())
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(output_dir.path())
        .arg("--no-browser");
    cmd.assert().success();

    let landing_page = output_dir.path().join("index.html");
    assert!(landing_page.exists(), "Landing page should exist");

    let html_content = fs::read_to_string(&landing_page)?;

    assert!(html_content.contains("Graph Runtime Analysis"));
    assert!(!html_content.contains("Runtime analysis not available"));
    assert!(html_content.contains("ms delta"));

    Ok(())
}

#[test]
fn test_runtime_analysis_mismatched_graphs() -> Result<(), Box<dyn std::error::Error>> {
    // Use entire directory - rank 4 is missing a graph compared to ranks 0,1,2,3
    let input_dir = PathBuf::from("tests/inputs/multi_rank_runtime");
    let temp_out = tempdir()?;
    let output_dir = temp_out.path();

    let mut cmd = Command::cargo_bin("tlparse")?;
    cmd.arg(&input_dir)
        .arg("--all-ranks-html")
        .arg("--overwrite")
        .arg("-o")
        .arg(&output_dir)
        .arg("--no-browser");
    cmd.assert().success();

    let landing_page = output_dir.join("index.html");
    assert!(landing_page.exists(), "Landing page should exist");

    let html_content = fs::read_to_string(&landing_page)?;

    assert!(html_content.contains("Graph Runtime Analysis"));
    assert!(html_content.contains("Runtime analysis not available"));
    assert!(!html_content.contains("ms delta"));

    Ok(())
}

#[test]
fn test_chromium_trace_with_runtime() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_runtime");
    let temp_out = tempdir()?;
    let out_dir = temp_out.path();

    Command::cargo_bin("tlparse")?
        .arg(&input_dir)
        .args(&["--all-ranks-html", "--overwrite", "-o"])
        .arg(&out_dir)
        .arg("--no-browser")
        .assert()
        .success();

    let runtime_trace_path = out_dir.join("chromium_trace_with_runtime.json");
    assert!(runtime_trace_path.exists());

    let trace_events: Vec<serde_json::Value> =
        serde_json::from_str(&fs::read_to_string(&runtime_trace_path)?)?;
    assert!(!trace_events.is_empty());

    let runtime_events: Vec<&serde_json::Value> = trace_events
        .iter()
        .filter(|e| e["ph"] == "X" && e["cat"] == "runtime")
        .collect();
    assert!(!runtime_events.is_empty());

    for e in &runtime_events {
        assert!(e["name"].is_string());
        let dur = e["dur"].as_u64().expect("dur should be u64");
        assert!(dur > 0);
        assert!(e["pid"].as_u64().is_some());
        assert!(e["tid"].as_u64().is_some());
        assert!(e["args"]["runtime_ns"].is_number());
        assert!(e["args"]["graph"].is_string());
        if let (Some(pid), Some(rank)) = (e["pid"].as_u64(), e["args"]["rank"].as_u64()) {
            assert_eq!(pid, rank);
        }
    }

    // Verify exact rank set matches input logs
    let expected_ranks: std::collections::HashSet<u64> = std::fs::read_dir(&input_dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter_map(|name| {
            name.strip_prefix("dedicated_log_torch_trace_rank_")
                .and_then(|s| s.strip_suffix(".log"))
                .and_then(|n| n.parse::<u64>().ok())
        })
        .collect();

    let pids: std::collections::HashSet<u64> = runtime_events
        .iter()
        .filter_map(|e| e["pid"].as_u64())
        .collect();
    assert_eq!(pids, expected_ranks, "pid set != expected rank set");

    Ok(())
}

#[test]
fn test_tensor_meta_divergence_groups() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = PathBuf::from("tests/inputs/multi_rank_runtime");
    let temp_out = tempdir()?;
    let out_dir = temp_out.path();

    Command::cargo_bin("tlparse")?
        .arg(&input_dir)
        .args(&["--all-ranks-html", "--overwrite", "-o"])
        .arg(&out_dir)
        .arg("--no-browser")
        .assert()
        .success();

    let landing_page = out_dir.join("index.html");
    let html_content = fs::read_to_string(&landing_page)?;

    // Should always show tensor meta analysis section
    assert!(html_content.contains("Tensor Metadata Analysis"));

    // Should show divergence since ranks have different tensor meta
    assert!(html_content.contains("Ranks exhibit divergent inductor tensor meta"));

    // Ranks 5 and 6 should be grouped together (same tensor meta)
    assert!(html_content.contains("Ranks: 5, 6"));

    Ok(())
}
