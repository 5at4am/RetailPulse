"""Quickstart: load all three Retail Pulse upload files and sanity-check them.

    python load_quickstart.py

Reads, in order:
    retail_pulse_sales.csv           250,000 line-item transactions  (27 columns)
    retail_pulse_demand_panel.csv    780,000 store x product x week (15 columns)
    retail_pulse_data_dictionary.md  column-by-column documentation

For each CSV it prints shape, dtypes, the first 5 rows, null counts and the date
range. The markdown dictionary is reported by size and its section headings. Finally
every row/column count is checked against facts.json, so a silent truncation or a botched
export shows up as FAIL instead of quietly shipping.

Standalone on purpose -- it imports nothing from retail_pulse.py, so the folder works on
its own wherever the three files are uploaded. Needs pandas.
"""
import json
import os
import sys

import pandas as pd

# This script sits at the project root. The three uploads live in data/raw/; fall back to
# the root itself so a bare copy -- just this file plus the three CSVs -- still runs.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE

_RAW = os.path.join(ROOT, "data", "raw")
RAW_DIR = _RAW if os.path.isdir(_RAW) else HERE

SALES_CSV = os.path.join(RAW_DIR, "retail_pulse_sales.csv")
PANEL_CSV = os.path.join(RAW_DIR, "retail_pulse_demand_panel.csv")
DICTIONARY_MD = os.path.join(RAW_DIR, "retail_pulse_data_dictionary.md")
FACTS_JSON = os.path.join(ROOT, "facts.json")

# facts.json keys: (section, expected rows, expected columns)
EXPECTED = {
    "retail_pulse_sales.csv": ("sales", 250_000, 27),
    "retail_pulse_demand_panel.csv": ("panel", 780_000, 15),
}

RULE = "=" * 72


def use_utf8_console():
    """Make stdout/stderr UTF-8 before printing anything.

    A Windows console defaults to cp1252, which has no glyph for characters the data
    dictionary uses (the two-way arrow, box drawing). Printing one raises
    UnicodeEncodeError and kills the report halfway through, so switch first and keep
    errors="replace" as a backstop -- a mangled character is better than a traceback.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass            # redirected to a file or a stream that cannot change


def head(title):
    print(f"\n{RULE}\n{title}\n{RULE}")


def missing(path):
    print(f"MISSING: {path}\n  expected it in {RAW_DIR}.")
    return True


def describe_csv(df, label, date_column):
    """Print the EDA block for one CSV and return (rows, cols) for the facts check."""
    head(f"{label}  --  {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\ndtypes:")
    print(df.dtypes.to_string())

    # Transposed so all 27 columns stay readable instead of being cut off sideways.
    print(f"\nfirst 5 rows (one line per column):")
    print(df.head(5).T.to_string())

    nulls = df.isna().sum()
    empty = nulls[nulls == 0]
    print(f"\nnulls: {len(empty)} of {len(df.columns)} columns are complete")
    if len(empty) < len(df.columns):
        print(nulls[nulls > 0].to_string())
    else:
        print("  (no missing values anywhere)")

    if date_column in df.columns:
        col = df[date_column]
        print(f"\n{date_column}: {col.min().date()} .. {col.max().date()}"
              f"  ({col.nunique():,} distinct)")
        print(f"  null dates: {int(col.isna().sum()):,}")

    return df.shape[0], df.shape[1]


def markdown_headings(text):
    """Real markdown headings only.

    A plain startswith("#") also catches Python comments inside fenced code blocks --
    the dictionary's Quick start section has three -- and those are not sections.
    """
    headings, fenced = [], False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("#"):
            headings.append(line)
    return headings


def describe_dictionary(path):
    head("retail_pulse_data_dictionary.md  --  documentation")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines()
    headings = markdown_headings(text)
    print(f"size    : {os.path.getsize(path):,} bytes")
    print(f"lines   : {len(lines):,}")
    print(f"sections: {len(headings):,}")
    for h in headings:
        print(f"  {h}")


def check_facts(stats):
    """Compare what we loaded against facts.json. Returns True if everything matched."""
    head("integrity check against facts.json")
    if not os.path.exists(FACTS_JSON):
        print("facts.json not found -- skipping the cross-check")
        return True

    with open(FACTS_JSON, encoding="utf-8") as fh:
        facts = json.load(fh)

    ok = True
    for name, (rows, cols) in stats.items():
        section, want_rows, want_cols = EXPECTED[name]
        got = facts.get(section, {})
        exp_rows = got.get("rows", want_rows)
        exp_cols = got.get("columns", want_cols)
        row_ok, col_ok = rows == exp_rows, cols == exp_cols
        ok &= row_ok and col_ok
        print(f"  {name}")
        print(f"    rows    {rows:>9,}  expected {exp_rows:>9,}  "
              f"{'PASS' if row_ok else 'FAIL'}")
        print(f"    columns {cols:>9,}  expected {exp_cols:>9,}  "
              f"{'PASS' if col_ok else 'FAIL'}")
    return ok


def main():
    use_utf8_console()
    stats, failed = {}, False

    if not os.path.exists(SALES_CSV):
        failed |= missing(SALES_CSV)
    else:
        # parse_dates so the date column is real datetimes, not strings.
        df = pd.read_csv(SALES_CSV, parse_dates=["date"])
        stats["retail_pulse_sales.csv"] = describe_csv(df, "retail_pulse_sales.csv", "date")

    if not os.path.exists(PANEL_CSV):
        failed |= missing(PANEL_CSV)
    else:
        df = pd.read_csv(PANEL_CSV, parse_dates=["week_start_date"])
        stats["retail_pulse_demand_panel.csv"] = describe_csv(
            df, "retail_pulse_demand_panel.csv", "week_start_date")

    if not os.path.exists(DICTIONARY_MD):
        failed |= missing(DICTIONARY_MD)
    else:
        describe_dictionary(DICTIONARY_MD)

    if not failed:
        failed = not check_facts(stats)

    head("result")
    if failed:
        print("FAIL -- do not upload yet, fix the file listed above")
        return 1
    print("PASS -- all 3 files loaded and match facts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
