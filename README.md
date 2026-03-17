# 🛒 BasketInsight

<div align="center">

**Market Basket Analysis — powered by Apriori & Flask**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com)
[![pandas](https://img.shields.io/badge/pandas-Data%20Processing-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)


Upload a CSV of transaction data → discover hidden product associations → export actionable rules.

</div>

---

## 📖 What is Market Basket Analysis?

Market Basket Analysis (MBA) is a data mining technique that finds items that **frequently appear together** in transactions. It uses the **Apriori algorithm** to generate *association rules* — patterns like:

> 🛒 `eggs & milk` → `bread` *(customers who buy eggs and milk also tend to buy bread)*

Used in retail, e-commerce, and supply chain to drive:

- **Cross-selling** — recommend related products
- **Shelf placement** — put associated items near each other
- **Promotions** — bundle frequently co-purchased items
- **Inventory planning** — stock items that sell together

---

## ✨ Features

### 🔍 Analysis

| Feature | Description |
| --- | --- |
| Apriori algorithm | Industry-standard frequent itemset mining via `mlxtend` |
| Association rules | Support, Confidence, Lift, Leverage, Conviction, Zhang's Metric |
| Configurable thresholds | Set minimum support % and confidence % before running |
| Re-run without re-uploading | Adjust thresholds directly on the results page |
| Demo dataset | Try the app instantly — no file upload needed |

### 📊 Visualizations

| Chart | Description |
| --- | --- |
| Bar chart | Top 20 antecedents by support, colored by confidence |
| Scatter plot | Support vs. Confidence for all rules, colored by lift |
| Network graph | Interactive node-link diagram of item associations (top 30 rules by lift) |

### 📋 Results Table

- **Sortable** columns — click any header to sort
- **Live search** — 500 ms debounce, searches all columns
- **Metric filters** — filter by min/max Lift, Confidence, Support
- **Confidence mini-bars** — inline visual progress bar per rule
- **Lift color chips** — 🟢 green (≥ 2.0), 🔵 blue (≥ 1.5), ⚪ gray (< 1.5)
- **Column tooltips** — hover any header for a one-line definition
- Pagination (50 rows / page), filters preserved across pages

### 📤 Export

| Format | Route |
| --- | --- |
| Excel (.xlsx) | `/download` |
| CSV | `/download/csv` |
| JSON | `/download/json` |

### 📈 Dataset Statistics

Shown automatically after each analysis:

- Total transactions
- Unique items
- Average basket size
- Top 5 most frequent items

### 🕓 Analysis History

The home page shows your last **5 analyses** — filename, thresholds used, rule count, and timestamp — stored in your browser session.

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/BasketInsight.git
cd BasketInsight
pip install -r requirements.txt
```

### 2. (Optional) Set a secret key

```bash
cp .env.example .env
# Edit .env and set SECRET_KEY=your-random-string
```

### 3. Run

```bash
cd app
python app.py
```

Open **<http://localhost:5000>** in your browser.

### 4. No file? Use the demo

Click **"▶ Try with Demo Dataset"** on the home page — no upload needed.

---

## 🖥️ Desktop GUI (optional)

A standalone Tkinter desktop app is included with the same Apriori + K-Means analysis:

```bash
python main.py
```

Features: load CSV, run Apriori, search rules, run K-Means clustering, generate text report, export to Excel.

---

## 📁 Project Structure

```text
BasketInsight/
├── main.py                  # Tkinter desktop GUI
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
├── README.md
└── app/
    ├── app.py               # Flask routes, analysis logic, chart generation
    ├── uploads/             # Uploaded CSVs (auto-cleaned after 24 h)
    ├── static/
    │   ├── css/style.css    # Full design system (CSS variables, responsive)
    │   └── results.csv      # Last analysis output (auto-generated)
    └── templates/
        ├── index.html       # Upload page + demo + history + guides
        └── results.html     # KPI cards, charts, network, filters, table
```

---

## 📄 CSV Input Format

**No header row.** Each line = one transaction. Items are comma-separated.

```text
milk, bread, eggs
bread, butter, jam
milk, eggs, sugar, flour
coffee, milk
bread, eggs, butter
```

| Rule | Detail |
| --- | --- |
| Headers | None — first row is treated as a transaction |
| Encoding | UTF-8 or Latin-1 (auto-detected) |
| Empty values | Ignored automatically |
| Case | Case-sensitive (`Milk` ≠ `milk`) |
| Max file size | 16 MB |

---

## ⚙️ Parameters

### Minimum Support (%)

> *"How common must an itemset be across all transactions?"*

The fraction of transactions that contain a given set of items.

| Value | Effect |
| --- | --- |
| Lower (1–5%) | More rules, slower run, catches rare patterns |
| Higher (10–20%) | Fewer rules, faster run, only common patterns |

**Typical start:** `2–5%`

### Minimum Confidence (%)

> *"How reliable must a rule be?"*

Given the antecedent is bought, how often is the consequent also bought?

| Value | Effect |
| --- | --- |
| Lower (30–50%) | More rules, includes weak associations |
| Higher (70–90%) | Fewer, highly reliable rules only |

**Typical start:** `50%`

---

## 📐 Output Metrics Reference

Each rule: **Antecedent → Consequent** *(e.g., `eggs, milk → bread`)*

| Metric | Range | What it means |
| --- | --- | --- |
| **Support** | 0 – 1 | Fraction of all transactions containing both sides |
| **Confidence** | 0 – 1 | P(consequent given antecedent). Shown as a mini-bar in the table. |
| **Lift** | 0 – ∞ | How much more likely the consequent is bought with the antecedent vs. random. **> 1** = positive association. |
| **Leverage** | −0.25 – 0.25 | Co-occurrence above chance. Positive = appear together more than random. |
| **Conviction** | 0 – ∞ | Directional dependency strength. > 1 = not coincidental. |
| **Zhang's Metric** | −1 – 1 | Balanced association score. Positive = co-occur more than expected. |

### Lift Guide

| Lift | Color | Interpretation |
| --- | --- | --- |
| ≥ 2.0 | 🟢 Green | Strong positive association — highly actionable |
| 1.5 – 2.0 | 🔵 Blue | Moderate positive association — worth investigating |
| ~1.0 | ⚪ Gray | No meaningful association |
| < 1.0 | — | Negative association — items avoid each other |

---

## 💡 Tips for Good Results

1. **Start low** — Support 2–5%, Confidence 50%. See if rules exist before tightening.
2. **Zero rules?** — Lower support first, then confidence.
3. **Too many rules?** — Raise support to filter for common patterns, or raise confidence for reliability.
4. **Focus on Lift ≥ 1.5** — Below that, the association may be coincidental.
5. **Use the filter bar** — On the results page, filter by metric ranges without re-running.
6. **Re-run freely** — Change thresholds on the results page without re-uploading the file.
7. **Large datasets** — Start at 5%+ support for files with > 50 k transactions.

---

## 🔒 Security Notes

- CSRF protection via Flask-WTF on all forms
- Uploaded files handled with `werkzeug.utils.secure_filename`
- Files older than 24 hours are auto-deleted from the uploads folder
- Set `SECRET_KEY` in `.env` for production — never use the default

---

## 🗺️ Routes

| Route | Method | Description |
| --- | --- | --- |
| `/` | GET, POST | Home page — upload form |
| `/demo` | GET | Run analysis on the built-in demo dataset |
| `/rerun` | POST | Re-run analysis on last file with new thresholds |
| `/results` | GET | Results page with charts, filters, and table |
| `/download` | GET | Export results as Excel (.xlsx) |
| `/download/csv` | GET | Export results as CSV |
| `/download/json` | GET | Export results as JSON |

---

## 📦 Dependencies

| Package | Purpose |
| --- | --- |
| `flask` ≥ 2.0 | Web framework |
| `flask-wtf` | CSRF-protected forms |
| `wtforms` | Form field validation |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `mlxtend` | Apriori algorithm + TransactionEncoder |
| `plotly` | Interactive bar, scatter, and network charts |
| `openpyxl` | Excel (.xlsx) export |
| `scikit-learn` | K-Means clustering (desktop GUI only) |
| `werkzeug` | Secure filename handling |

---

<div align="center">

Made with Python, Flask, and mlxtend · Apriori-powered association rule mining

</div>
