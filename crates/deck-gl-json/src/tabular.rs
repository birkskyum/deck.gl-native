//! CSV, TSV and newline delimited JSON as `data`: every record becomes a JSON object so the
//! usual `@@=` accessors work on it.

use serde_json::{Map, Value};

use crate::{JsonError, Result};

/// The table formats told apart by file extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TabularFormat {
    Csv,
    Tsv,
    /// One JSON value per line (`.ndjson`, `.jsonl`)
    NdJson,
}

impl TabularFormat {
    /// The format of a file or URL by its extension, ignoring a query string.
    pub fn from_source(source: &str) -> Option<Self> {
        let path = source.split(['?', '#']).next().unwrap_or(source);
        let extension = path.rsplit('.').next()?.to_ascii_lowercase();
        match extension.as_str() {
            "csv" => Some(Self::Csv),
            "tsv" | "tab" => Some(Self::Tsv),
            "ndjson" | "jsonl" | "jsonlines" => Some(Self::NdJson),
            _ => None,
        }
    }
}

/// A cell as JSON: numbers when they parse, booleans, `null` for empty cells, else text.
fn cell_value(text: &str) -> Value {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Value::Null;
    }
    match trimmed {
        "true" | "TRUE" | "True" => return Value::Bool(true),
        "false" | "FALSE" | "False" => return Value::Bool(false),
        _ => {}
    }
    if let Ok(int) = trimmed.parse::<i64>() {
        return Value::from(int);
    }
    if let Ok(float) = trimmed.parse::<f64>() {
        if float.is_finite() {
            return Value::from(float);
        }
    }
    Value::String(text.to_string())
}

/// Parse delimited text with a header row into rows of objects.
pub fn parse_delimited(text: &str, delimiter: u8, source: &str) -> Result<Vec<Value>> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true)
        .trim(csv::Trim::Headers)
        .from_reader(text.as_bytes());
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| JsonError::Load {
            url: source.to_string(),
            message: format!("bad header row: {e}"),
        })?
        .iter()
        .map(str::to_string)
        .collect();
    let mut rows = Vec::new();
    for (line, record) in reader.records().enumerate() {
        let record = record.map_err(|e| JsonError::Load {
            url: source.to_string(),
            message: format!("row {}: {e}", line + 2),
        })?;
        let mut object = Map::with_capacity(headers.len());
        for (column, cell) in record.iter().enumerate() {
            let key = headers
                .get(column)
                .cloned()
                .unwrap_or_else(|| format!("column{column}"));
            object.insert(key, cell_value(cell));
        }
        rows.push(Value::Object(object));
    }
    Ok(rows)
}

/// Parse newline delimited JSON; blank lines are skipped.
pub fn parse_ndjson(text: &str, source: &str) -> Result<Vec<Value>> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|e| JsonError::Load {
                url: source.to_string(),
                message: format!("line {}: invalid JSON: {e}", index + 1),
            })
        })
        .collect()
}

/// Parse text in the given format into rows.
pub fn parse(text: &str, format: TabularFormat, source: &str) -> Result<Vec<Value>> {
    match format {
        TabularFormat::Csv => parse_delimited(text, b',', source),
        TabularFormat::Tsv => parse_delimited(text, b'\t', source),
        TabularFormat::NdJson => parse_ndjson(text, source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_cells_become_typed_values() {
        let rows = parse(
            "lng,lat,name,ok,size\n1.5,2,Oslo,true,\n-3,4.25,\"Rome, IT\",false,7\n",
            TabularFormat::Csv,
            "x.csv",
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["lng"], 1.5);
        assert_eq!(rows[0]["lat"], 2);
        assert_eq!(rows[0]["name"], "Oslo");
        assert_eq!(rows[0]["ok"], true);
        assert_eq!(rows[0]["size"], Value::Null);
        assert_eq!(rows[1]["name"], "Rome, IT");
        assert_eq!(rows[1]["size"], 7);
        let tsv = parse("a\tb\n1\t2\n", TabularFormat::Tsv, "x.tsv").unwrap();
        assert_eq!(tsv[0]["b"], 2);
        let nd = parse("{\"a\": 1}\n\n{\"a\": [2]}\n", TabularFormat::NdJson, "x.ndjson").unwrap();
        assert_eq!(nd.len(), 2);
        assert_eq!(nd[1]["a"][0], 2);
        assert!(parse("{\"a\": 1}\nnot json\n", TabularFormat::NdJson, "x.ndjson").is_err());
        assert_eq!(
            TabularFormat::from_source("https://h/data.CSV?x=1"),
            Some(TabularFormat::Csv)
        );
        assert_eq!(
            TabularFormat::from_source("rows.jsonl"),
            Some(TabularFormat::NdJson)
        );
        assert_eq!(TabularFormat::from_source("rows.json"), None);
    }
}
