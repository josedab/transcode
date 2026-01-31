---
sidebar_position: 19
title: EDL Import & Export
description: Parse and generate Edit Decision Lists in CMX 3600, CSV, and FCP XML formats
---

# EDL Import & Export

The `transcode-edit` crate supports importing and exporting Edit Decision Lists (EDLs) in industry-standard formats: CMX 3600, CSV, and Final Cut Pro XML.

## Overview

EDLs describe a sequence of edits as a list of source clips with in/out points. Transcode can:

- **Parse** CMX 3600 EDL files from professional editing systems
- **Import** CSV and FCP XML formats
- **Export** to CMX 3600, CSV, JSON, and FCP XML
- **Round-trip** between formats without data loss

## Quick Start

```toml
[dependencies]
transcode-edit = "1.0"
```

### Parsing a CMX 3600 EDL

```rust
use transcode_edit::import::{parse_cmx3600, ParsedEdl};

let edl_text = r#"TITLE: My Project

001  CLIP001  V  C  01:00:00:00 01:00:10:00 00:00:00:00 00:00:10:00
002  CLIP002  V  D  01:00:05:00 01:00:20:00 00:00:10:00 00:00:25:00
003  CLIP003  V  C  01:00:00:00 01:00:08:00 00:00:25:00 00:00:33:00
"#;

let parsed = parse_cmx3600(edl_text)?;
println!("Title: {:?}", parsed.title);
println!("Entries: {}", parsed.entries.len());

for entry in &parsed.entries {
    println!("  {} → {:.2}s - {:.2}s ({})",
        entry.source,
        entry.record_in,
        entry.record_out,
        entry.transition.as_ref().map(|t| format!("{:?}", t)).unwrap_or("Cut".into())
    );
}
```

### Importing CSV

```rust
use transcode_edit::import::parse_csv;

let csv_text = "source,source_in,source_out,record_in,record_out
CLIP001,0.0,10.0,0.0,10.0
CLIP002,5.0,20.0,10.0,25.0
";

let parsed = parse_csv(csv_text)?;
println!("Imported {} entries", parsed.entries.len());
```

### FCP XML Export

Generate Final Cut Pro XML from an EDL:

```rust
use transcode_edit::import::export_fcp_xml;
use transcode_edit::edl::EditDecisionList;

let edl = EditDecisionList::new("My Project");
// ... add entries ...

let xml = export_fcp_xml(&edl, 30.0); // 30fps timeline
std::fs::write("project.xml", &xml)?;
```

### FCP XML Import

```rust
use transcode_edit::import::parse_fcp_xml;

let xml = std::fs::read_to_string("project.xml")?;
let parsed = parse_fcp_xml(&xml)?;

println!("Title: {:?}", parsed.title);
for entry in &parsed.entries {
    println!("  {}: {:.2}s → {:.2}s", entry.source, entry.record_in, entry.record_out);
}
```

### Round-Trip: CMX 3600 → FCP XML

```rust
use transcode_edit::import::{parse_cmx3600, export_fcp_xml};
use transcode_edit::edl::EditDecisionList;

// Parse CMX 3600
let parsed = parse_cmx3600(&cmx_text)?;

// Build EDL
let mut edl = EditDecisionList::new(parsed.title.unwrap_or_default());
for entry in parsed.entries {
    edl.add_entry(entry);
}

// Export as FCP XML
let xml = export_fcp_xml(&edl, 30.0);
std::fs::write("output.xml", &xml)?;
```

## Timecode Handling

Convert between timecodes and seconds:

```rust
use transcode_edit::import::parse_timecode;

let secs = parse_timecode("01:30:45:15", 30.0)?;
println!("{:.4} seconds", secs); // 5445.5000 seconds
```

## Supported Formats

| Format | Import | Export | Notes |
|--------|--------|--------|-------|
| CMX 3600 | ✅ | ✅ | Standard EDL with C/D/W transitions |
| CSV | ✅ | ✅ | Simple comma-separated format |
| JSON | ✅ | ✅ | Via serde serialization |
| FCP XML | ✅ | ✅ | XMEML-compatible |

## API Reference

| Function | Description |
|----------|-------------|
| `parse_cmx3600()` | Parse CMX 3600 EDL text |
| `parse_csv()` | Parse CSV EDL |
| `parse_fcp_xml()` | Parse Final Cut Pro XML |
| `export_fcp_xml()` | Generate FCP XML from EDL |
| `parse_timecode()` | Convert HH:MM:SS:FF to seconds |
| `ParsedEdl` | Parsed result with title and entries |

## Next Steps

- [Basic Transcoding](/docs/guides/basic-transcoding) — Apply edits during transcoding
- [Filter Chains](/docs/guides/filter-chains) — Add transitions and effects
