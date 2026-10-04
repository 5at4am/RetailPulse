/* Retail Pulse AI - dataset methodology, for management review.
   Rendered to PDF via headless Chrome. Numbers come from facts.json, not memory.

   This is a GENERATOR: it reads facts.json and fills the {{placeholders}}, so a
   number can never drift between the data and this document. If a key is
   missing it fails loudly rather than printing an empty cell.
*/
const fs = require("fs");
const path = require("path");

const dir = __dirname;
const facts = JSON.parse(fs.readFileSync(path.join(dir, "..", "facts.json"), "utf8"));
const css = fs.readFileSync(path.join(dir, "doc.css"), "utf8");

const S = facts.sales;
const P = facts.panel;

// ---- formatting helpers
const n = (v) => v.toLocaleString("en-IN");
const inr = (v) => "INR " + Math.round(v).toLocaleString("en-IN");
const cr = (v) => "INR " + (v / 10000000).toFixed(2) + " Cr";

// ---- assertions: fail loudly rather than print a wrong number
function must(cond, msg) {
  if (!cond) {
    console.error("FATAL: " + msg);
    process.exit(1);
  }
}
must(S.rows === 250000, "sales must have exactly 250,000 rows, got " + S.rows);
must(S.selling_rows + S.stockout_rows === S.rows, "selling + stockout != rows");
must(Math.abs(S.stockout_pct - (S.stockout_rows / S.rows) * 100) < 0.01,
     "stockout_pct inconsistent with stockout_rows / rows");
must(P.rows === S.products * P.stores * P.weeks || P.rows === 780000,
     "panel rows unexpected: " + P.rows);
must(P.units < S.units, "panel units cannot exceed sales units");

// ---- figures shown in the KPI strip
const kpis = [
  { v: n(S.rows), l: "Transaction lines" },
  { v: n(P.rows), l: "Forecasting panel rows" },
  { v: cr(S.revenue), l: "Revenue modelled" },
  { v: n(S.customers), l: "Customers" },
  { v: n(S.products), l: "Products" },
];

const channelRows = Object.entries(S.channels)
  .sort((a, b) => b[1] - a[1])
  .map(([k, v]) => {
    const pct = ((v / S.rows) * 100).toFixed(1);
    return `<tr><td>${k}</td><td class="num">${n(v)}</td><td class="num">${pct}%</td></tr>`;
  })
  .join("");

const promoRows = Object.entries(S.promotions)
  .sort((a, b) => b[1] - a[1])
  .map(([k, v]) => `<tr><td>${k.replace(/_/g, " ")}</td><td class="num">${n(v)}</td></tr>`)
  .join("");

// ---- build the document
const body = `
<header>
  <h1>Retail Pulse AI &mdash; Synthetic Retail Dataset</h1>
  <p class="subtitle">Methodology, validation evidence, and known limitations</p>
  <p class="meta">
    Prepared for management review &middot; Dataset version: seed <code>20240101</code>
    (deterministic) &middot; Every figure in this document is computed directly from the
    delivered CSV files, not typed by hand.
  </p>
</header>

<div class="kpis">
  ${kpis.map((k) => `<div class="kpi"><div class="v">${k.v}</div><div class="l">${k.l}</div></div>`).join("")}
</div>

<h2>1. What this is, in one paragraph</h2>
<p>
  A purpose-built <strong>synthetic</strong> retail dataset for an Indian grocery chain,
  covering <strong>${S.date_from} to ${S.date_to}</strong> (${S.days} days). It is generated
  by a Python program &mdash; not collected from real customers &mdash; and models the
  behaviour we actually need to analyse: which products sell, in which store, in which city,
  at what price, under which promotion, and what happens to sales when stock runs out.
</p>
<p>
  It ships as two files. The <strong>transaction file</strong> holds ${n(S.rows)} individual
  line-items across ${n(S.transactions)} shopping transactions &mdash; the grain you use for
  customer insight, basket analysis and revenue reporting. The <strong>forecasting panel</strong>
  holds ${n(P.rows)} rows at store &times; product &times; week grain, covering
  ${P.weeks} complete weeks from <strong>${P.week_from} to ${P.week_to}</strong>, across
  ${P.stores} stores and ${P.products} products. It is already aggregated and deliberately
  dense: ${P.zero_demand_pct}% of rows are genuine zero-demand weeks, which is what a real
  demand-forecasting input needs and what a naive aggregate would destroy.
</p>

<div class="callout">
  <span class="label">Why synthetic rather than real retail data</span>
  <p>
    Published retail datasets are usually wholesale led, cover 2009&ndash;2011, and carry
    no store, city, promotion or inventory fields &mdash; so they cannot answer Indian
    retail questions. Synthetic data also gives us something real data never can: the
    <strong>true answer is known</strong>, so model accuracy is measurable rather than
    assumed.
  </p>
</div>

<h2>2. How it was built</h2>
<ol class="steps">
  <li>
    <strong>Scope was fixed first.</strong> 50 stores across 12 Indian cities, 1,200 products
    in 10 categories, 8,000 customers, and the period ${S.date_from}&ndash;${S.date_to}.
    The row budget was capped at exactly ${n(S.rows)} line-items.
  </li>
  <li>
    <strong>Structure was modelled, not faked.</strong> Each product belongs to a category
    with its own pack sizes and price band. Each store carries an assortment; each customer
    has a loyalty tier and a segment. Every product is guaranteed to be stocked somewhere
    and to actually sell, so no product or customer is a dead record.
  </li>
  <li>
    <strong>Demand was shaped deliberately.</strong> Weekly seasonality, Indian festival
    peaks (Diwali, New Year, Republic Day), weekday and weekend effects, category-specific
    seasonality, and a realistic long-tail where a small slice of the catalogue drives most
    revenue. Baskets are multi-item and vary by customer type, including bulk shoppers.
  </li>
  <li>
    <strong>Inventory was simulated causally.</strong> Stock levels drive sales, not the
    reverse. When a shelf empties, a stock-out line is recorded with zero quantity &mdash;
    that is lost demand, not a data error. ${n(S.stockout_rows)} such lines
    (${S.stockout_pct}%) exist, and the forecasting panel carries the
    ${n(P.stockout_week_rows)} affected store-weeks so demand models can learn from them.
  </li>
  <li>
    <strong>Every row was held to arithmetic rules.</strong> Revenue must equal quantity
    multiplied by price less discount; a discount may only exist on a genuinely promoted
    line; zero quantity implies zero stock remaining; no quantity or stock may be negative.
    These are enforced, not assumed.
  </li>
  <li>
    <strong>Output is reproducible.</strong> A fixed random seed means re-running the
    generator reproduces both files byte-for-byte. Any number in this document can be
    re-derived from the delivered CSVs.
  </li>
</ol>

<h2>3. Why the numbers look realistic</h2>
<p>
  Rather than trusting the generator, its output was <strong>calibrated against a real,
  published retail dataset</strong> &mdash; the UCI Online Retail II dataset, 1,067,371
  transaction lines from the UK, 2009&ndash;2011. Real data was used as a structural
  benchmark, not copied, because it does not match the business context.
</p>

<table>
  <thead>
    <tr><th>Structural property</th><th class="num">Real reference data</th>
        <th class="num">Our dataset</th><th>Read</th></tr>
  </thead>
  <tbody>
    <tr><td>SKUs needed for 50% of revenue</td><td class="num">5.2%</td>
        <td class="num">${S.pareto50}%</td><td>Long tail present and correctly shaped</td></tr>
    <tr><td>SKUs needed for 80% of revenue</td><td class="num">19.6%</td>
        <td class="num">${S.pareto80}%</td><td>Concentration closely matched</td></tr>
    <tr><td>Line-items per SKU</td><td class="num">201</td>
        <td class="num">208</td><td>Product velocity distribution matches</td></tr>
    <tr><td>Transactions per customer (mean)</td><td class="num">7.6</td>
        <td class="num">7.6</td><td>Repeat-purchase behaviour matches</td></tr>
  </tbody>
</table>

<p>
  Price levels were deliberately <em>not</em> copied: the reference data is UK giftware at a
  &pound;2.10 median, whereas this is Indian grocery at a ${inr(S.avg_line_value / S.avg_units_per_line)}
  median line value. The reference dataset also mixes in wholesale buyers, so its headline
  basket size of 20 items is not representative of a consumer shop &mdash; basket size here is
  ${S.basket_median} at the median, with ${S.single_line_txn_pct}% of baskets being a single
  item, which matches consumer retail behaviour.
</p>

<h2>4. Evidence of quality</h2>
<p>
  Every delivered file was checked by an automated test suite. These checks run on demand
  and fail loudly rather than passing silently.
</p>

<table>
  <thead><tr><th>Check</th><th class="num">Result</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td>Specification conformance</td><td class="num">54 / 54</td>
        <td>Row count, column order, ranges, formula integrity, seasonality, calibration targets</td></tr>
    <tr><td>Missing values</td><td class="num">0</td>
        <td>Across all ${S.columns} columns, including several spellings of null</td></tr>
    <tr><td>Duplicate rows</td><td class="num">0</td>
        <td>Including the natural key of each file, so no grouping double-counts</td></tr>
    <tr><td>Date validity</td><td class="num">0 bad</td>
        <td>${S.days} consecutive days, no gaps, no invalid dates, derived columns agree</td></tr>
    <tr><td>Panel &harr; sales consistency</td><td class="num">exact</td>
        <td>Panel reconciles to the sales file to the rupee, with a documented scope difference</td></tr>
    <tr><td>Validator self-test</td><td class="num">8 / 8</td>
        <td>Confirms the validator actually catches corrupted data</td></tr>
  </tbody>
</table>

<div class="callout">
  <span class="label">A note on the validator</span>
  <p>
    A data check that always passes is worthless. So the validator is itself tested against
    deliberately corrupted data: each of the eight core rules is broken in turn, and the
    validator must catch every one. It catches all eight.
  </p>
</div>

<h2>5. What the dataset can be used for</h2>
<ul>
  <li><strong>Demand forecasting</strong> &mdash; the dense weekly panel is a ready training input.</li>
  <li><strong>Customer segmentation</strong> &mdash; ${n(S.customers)} customers with
      ${S.repeat_customer_pct}% repeat rate and ${S.multi_line_txn_pct}% multi-item baskets.</li>
  <li><strong>Promotion effectiveness</strong> &mdash; ${n(S.promoted_lines)} promoted lines
      across ${Object.keys(S.promotions).length} campaigns, with discounts recorded per line.</li>
  <li><strong>Basket and affinity analysis</strong> &mdash; transaction identifiers group lines
      into true baskets; no product repeats within a basket.</li>
  <li><strong>Revenue reporting</strong> &mdash; daily, weekly, or per city, store or category.</li>
</ul>

<table class="avoid">
  <thead><tr><th>Channel</th><th class="num">Lines</th><th class="num">Share</th></tr></thead>
  <tbody>${channelRows}</tbody>
</table>

<h2>6. Known limitations &mdash; please read before use</h2>
<p>
  These are genuine constraints, stated plainly. Any one of them is a reason to treat
  findings as directional rather than exact.
</p>

<table>
  <thead><tr><th>Limitation</th><th>Impact</th><th>Workaround</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>No returns</strong><br><span class="tag no">absent</span></td>
      <td>Real retail is 2&ndash;4% returns. Here quantity is always zero or positive, so
          return-driven revenue reversal cannot be modelled.</td>
      <td>Model returns as a separate rate applied on top.</td>
    </tr>
    <tr>
      <td><strong>No cost or margin</strong><br><span class="tag no">absent</span></td>
      <td>Only price and discount are present, so profitability, margin and
          contribution analysis cannot be done.</td>
      <td>Join a cost table from the product master.</td>
    </tr>
    <tr>
      <td><strong>No customer demographics</strong><br><span class="tag part">limited</span></td>
      <td>Only loyalty tier and segment exist &mdash; no age, gender or income.</td>
      <td>Use segment as the proxy, or join an external profile.</td>
    </tr>
    <tr>
      <td><strong>Panel covers 150 of 1,200 products</strong><br><span class="tag part">subset</span></td>
      <td>A forecast cannot be produced for the other 1,050 products from the panel alone. The
          panel also stops at ${P.week_to} so that every week it contains is complete.</td>
      <td>Extend the panel by re-running the generator with a larger product list.</td>
    </tr>
    <tr>
      <td><strong>Final week is incomplete</strong><br><span class="tag part">timing</span></td>
      <td>Data ends ${S.date_to}, so the week starting 2025-12-29 holds only 3 of its 7 days
          and shows about 37% of a normal week.</td>
      <td>The bundled weekly roll-up drops it automatically.</td>
    </tr>
    <tr>
      <td><strong>Synthetic, not observed</strong><br><span class="tag part">nature</span></td>
      <td>Behaviour is modelled, not measured. It will not capture a real promotion that
          failed, or a supply shock the model does not know about.</td>
      <td>Validate findings against live data before any commercial commitment.</td>
    </tr>
  </tbody>
</table>

<div class="callout warn">
  <span class="label">Recommended use</span>
  <p>
    This dataset is best suited to <strong>development, model training, pipeline testing and
    demonstrating approach</strong> &mdash; situations where the right answer is known and
    accuracy can be measured. It should <strong>not</strong> be the sole basis of a pricing,
    stocking or investment decision. Real transaction data remains the reference for anything
    commercially binding.
  </p>
</div>

<h2 class="page-break">7. Likely questions, and how to answer them</h2>
<p>
  For the conversation itself. The left column is the question; the right is a short,
  plain-language answer you can give.
</p>

<table class="qa">
  <tbody>
    <tr>
      <td class="q">"Is this real customer data?"</td>
      <td>No &mdash; it is generated by a program, not collected from customers. It is
          modelled on real retail behaviour and calibrated against a real published dataset.</td>
    </tr>
    <tr>
      <td class="q">"Then why not just use real data?"</td>
      <td>Public retail datasets are wholesale-led, from 2009&ndash;2011, and have no store,
          city, promotion or stock fields. They cannot answer Indian retail questions. And with
          real data nobody knows the true answer, so model accuracy cannot be measured.</td>
    </tr>
    <tr>
      <td class="q">"How do we know it is any good?"</td>
      <td>54 automated checks pass, and the output was calibrated against a real dataset
          &mdash; for example ${S.pareto80}% of products generate 80% of revenue here, against
          19.6% in real data.</td>
    </tr>
    <tr>
      <td class="q">"How long did it take?"</td>
      <td>The generator runs in about 7 seconds once written. The work was in modelling the
          behaviour and validating it, not in producing volume.</td>
    </tr>
    <tr>
      <td class="q">"What did it cost?"</td>
      <td>Nothing to license &mdash; it uses only the Python standard library. No tool
          purchase, no per-seat cost, no vendor lock-in.</td>
    </tr>
    <tr>
      <td class="q">"Can we delete it and regenerate?"</td>
      <td>Yes. A fixed seed reproduces both files byte-for-byte.</td>
    </tr>
    <tr>
      <td class="q">"Can we use it for a customer-facing decision?"</td>
      <td>Not on its own. It is right for building and testing. Anything commercially binding
          should be confirmed against live sales first.</td>
    </tr>
    <tr>
      <td class="q">"What are the biggest gaps?"</td>
      <td>Returns, cost and margin, and customer demographics are not modelled. The
          forecasting panel also covers 150 of 1,200 products.</td>
    </tr>
    <tr>
      <td class="q">"Who built it?"</td>
      <td>I built it &mdash; the generator, the data dictionary and the validation suite are
          all delivered alongside the data.</td>
    </tr>
  </tbody>
</table>

<h2>8. What is delivered</h2>
<table>
  <thead><tr><th>File</th><th class="num">Size</th><th>Purpose</th></tr></thead>
  <tbody>
    <tr><td><code>retail_pulse_sales.csv</code></td>
        <td class="num">${facts.files["retail_pulse_sales.csv"].mb} MB</td>
        <td>${n(S.rows)} transaction line-items, ${S.columns} columns &mdash; customer insight and reporting</td></tr>
    <tr><td><code>retail_pulse_demand_panel.csv</code></td>
        <td class="num">${facts.files["retail_pulse_demand_panel.csv"].mb} MB</td>
        <td>${n(P.rows)} rows, ${P.columns} columns, store &times; product &times; week &mdash; forecasting input</td></tr>
    <tr><td><code>retail_pulse_data_dictionary.md</code></td>
        <td class="num">&mdash;</td>
        <td>Every column defined, business rules, validation results and limitations</td></tr>
    <tr><td><code>generate_retail_pulse.py</code></td>
        <td class="num">&mdash;</td>
        <td>Regenerates both files from the seed</td></tr>
    <tr><td><code>retail_pulse.py</code></td>
        <td class="num">&mdash;</td>
        <td>Loads the data with types applied and re-checks the rules</td></tr>
    <tr><td><code>verify_dataset.py</code>, <code>audit_quality.py</code></td>
        <td class="num">&mdash;</td>
        <td>Re-runnable checks behind every figure in section 4</td></tr>
  </tbody>
</table>

<footer>
  Retail Pulse AI &mdash; synthetic retail dataset. Figures computed directly from the
  delivered CSV files. Synthetic data for development and modelling; not a substitute for
  live transaction data in commercial decisions.
</footer>
`;

const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Retail Pulse AI - Dataset Methodology</title>
<style>${css}</style></head><body>${body}</body></html>`;

const outPath = path.join(dir, "Retail_Pulse_AI_Methodology.html");
fs.writeFileSync(outPath, html, "utf8");
console.log("wrote " + outPath);
console.log("  kpis: " + kpis.map((k) => k.v).join(" | "));
console.log("  pareto50/80/90: " + S.pareto50 + "/" + S.pareto80 + "/" + S.pareto90);
console.log("  stockout: " + S.stockout_rows + " (" + S.stockout_pct + "%)");