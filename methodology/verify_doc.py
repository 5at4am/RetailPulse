"""Final text-level checks on the methodology document.

Cannot verify visual layout in this environment (no PDF/image input), so this
covers what is machine-checkable: template leftovers, section completeness,
unbalanced markup, and over-long cells that could overflow the printed page.
"""
import re
from pathlib import Path

D = Path(__file__).resolve().parent
html = (D / "Retail_Pulse_AI_Methodology.html").read_text(encoding="utf-8")
text = re.sub(r"<[^>]+>", " ", html)
text = re.sub(r"\s+", " ", text).strip()

print("=" * 72)
print("DOCUMENT SANITY")
print("=" * 72)

print(f"html size      {len(html):,} bytes")
print(f"visible text   {len(text):,} chars")

# 1. no unrendered template placeholders
leftovers = re.findall(r"\{\{[^}]*\}\}|\$\{[^}]*\}|undefined|NaN|\[object", html)
print(f"\nunrendered placeholders / undefined: {len(leftovers)}")
if leftovers:
    print(f"  -> {sorted(set(leftovers))[:6]}")
    raise SystemExit("document contains unrendered template output")

# 2. every section heading present
sections = [
    "What this is", "How it was built", "Why the numbers look realistic",
    "Evidence of quality", "What the dataset can be used for",
    "Known limitations", "Likely questions", "What is delivered",
]
print("\nsections:")
missing = []
for s in sections:
    ok = s in html
    print(f"  {'ok  ' if ok else 'MISS'}  {s}")
    if not ok:
        missing.append(s)

# 3. markup balance
print("\nmarkup balance:")
pairs = {"div": 0, "table": 0, "tr": 0, "td": 0, "th": 0, "ul": 0, "ol": 0, "li": 0,
         "p": 0, "pre": 0, "strong": 0}
for tag in pairs:
    o = len(re.findall(rf"<{tag}[\s>]", html))
    c = len(re.findall(rf"</{tag}>", html))
    # <li> is used both plainly and with class
    ok = (o == c)
    pairs[tag] = ok
    print(f"  {'ok  ' if ok else 'FAIL'}  <{tag}> {o} open / {c} close")
if not all(pairs.values()):
    raise SystemExit("unbalanced markup")

# 4. the honesty section must actually be present and non-trivial
print("\ncontent guarantees:")
must_have = {
    "states data is synthetic": "synthetic" in html.lower(),
    "names the real reference": "UCI Online Retail II" in html,
    "no-returns limitation": "No returns" in html,
    "no-margin limitation": "No cost or margin" in html,
    "panel subset limitation": "150 of 1,200" in html,
    "not-for-commercial-advice warning": "not</strong> be the sole basis" in html
        or "not the sole basis" in html.lower(),
    "partial-week caveat": "2025-12-29" in html,
    "Q&A for the conversation": "how to answer" in html.lower(),
}
for label, ok in must_have.items():
    print(f"  {'ok  ' if ok else 'MISS'}  {label}")
    if not ok:
        missing.append(label)

# 5. long unbroken tokens can overflow a printed page
tokens = re.findall(r"[A-Za-z0-9_./\\-]{60,}", text)
print(f"\nunbreakable tokens over 60 chars: {len(tokens)}")
for t in tokens[:5]:
    print(f"  {t}")

if missing:
    print(f"\nFAILED: {len(missing)} issue(s)")
    raise SystemExit(1)
print("\nall document checks passed")
print("NOTE: visual layout not verified - no PDF/image rendering available here.")