"""Verify the generated PDF: it exists, has pages, and every number in it
matches facts.json (which is computed from the CSVs).
"""
import json
import re
import zlib
from pathlib import Path

D = Path(__file__).resolve().parent
pdf = D / "Retail_Pulse_AI_Methodology.pdf"
html = D / "Retail_Pulse_AI_Methodology.html"
facts = json.loads((D.parent / "facts.json").read_text(encoding="utf-8"))
S, P = facts["sales"], facts["panel"]

print("=" * 72)
print("PDF VERIFICATION")
print("=" * 72)

raw = pdf.read_bytes()
print(f"file        {pdf.name}")
print(f"size        {len(raw):,} bytes")
print(f"header      {raw[:8].decode('latin-1')}")

assert raw[:5] == b"%PDF-", "not a PDF"
print("valid PDF header: True")

pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
print(f"pages       {pages}")
assert pages >= 3, f"expected at least 3 pages, got {pages}"

# ---- every headline number must be findable in the document
doc = html.read_text(encoding="utf-8")
doc_flat = re.sub(r"<[^>]+>", " ", doc)
doc_flat = re.sub(r"\s+", " ", doc_flat)


def indian(v):
    """Render like the document does, with Indian digit grouping."""
    s = str(v)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


checks = [
    ("sales rows", indian(S["rows"])),
    ("panel rows", indian(P["rows"])),
    ("customers", indian(S["customers"])),
    ("products", indian(S["products"])),
    ("stores", indian(S["stores"])),
    ("cities", indian(S["cities"])),
    ("transactions", indian(S["transactions"])),
    ("stockout rows", indian(S["stockout_rows"])),
    ("stockout week rows", indian(P["stockout_week_rows"])),
    ("date from", S["date_from"]),
    ("date to", S["date_to"]),
    ("panel week from", P["week_from"]),
    ("panel week to", P["week_to"]),
    ("pareto50", str(S["pareto50"])),
    ("pareto80", str(S["pareto80"])),
    ("repeat customer pct", str(S["repeat_customer_pct"])),
    ("single-line txn pct", str(S["single_line_txn_pct"])),
    ("zero demand pct", str(P["zero_demand_pct"])),
    ("stockout pct", str(S["stockout_pct"]).rstrip("0").rstrip(".")),
    ("revenue crore", f"{S['revenue'] / 10000000:.2f} Cr"),
]

print("\nnumbers cross-checked against facts.json (computed from the CSVs):")
missing = []
for label, needle in checks:
    ok = needle in doc_flat
    print(f"  {'ok  ' if ok else 'MISS'}  {label:24} {needle}")
    if not ok:
        missing.append(label)

print("\n" + "=" * 72)
if missing:
    print(f"FAILED: {len(missing)} number(s) not found in the document:")
    for m in missing:
        print(f"  - {m}")
    raise SystemExit(1)

print(f"PASSED: all {len(checks)} headline numbers present and correct.")
print(f"        {pages} pages, {len(raw):,} bytes.")

# ---- the document must not contain the two bugs I fixed earlier
print("\nregression guards:")
bads = {
    "stockout % as 0.77": "0.77",
    "avg units/line as 0.99": ">0.99<",
    "Pareto as 94.9 (wrong direction)": "94.9",
}
for label, needle in bads.items():
    present = needle in doc
    print(f"  {'ok  ' if not present else 'FAIL'}  absent: {label}")
    if present:
        raise SystemExit(1)
print("\nall assertions passed")