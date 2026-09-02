# Login Forms Detection

## Project Overview

Detects malicious phishing login forms by analysing the structural signature of a page's HTML DOM — no raw HTML, no text content, no screenshots. The input is a compact tree string (`html_signature`) that encodes element hierarchy only.

Binary classification: `LOGIN_FORM_MALICIOUS` vs `NO_FORM`.

---

## Directory Structure

```
data/
  train/     — training CSVs (login_form + no_form)
  retest/    — held-out validation CSVs (signed retest)
notebooks/
  01_data_exploration.ipynb          — EDA shared by both model tracks
  sequence_model/
    02_xgboost_detector.ipynb         — XGBoost on 6-gram sequence features       → sequence_xgboost.pkl
    03_random_forest_detector.ipynb   — Random Forest on 6-gram sequence features  → sequence_random_forest.pkl
  structural_model/
    04_xgboost_structural.ipynb       — XGBoost on 14 structural features          → structural_xgboost.pkl
    05_random_forest_structural.ipynb — Random Forest on 14 structural features    → structural_random_forest.pkl
    06_xgboost_analysis.ipynb         — Deep analysis for structural XGBoost model
    07_random_forest_analysis.ipynb   — Deep analysis for structural Random Forest model
  09_accuracy_comparison.ipynb        — Accuracy comparison across all four models
src/
  features.py   — shared helpers (tokenize, mine_sequences, structural_features, …)
models/         — saved model artifacts (populated after training)
  sequence_qualifying_seqs.json       — 6-gram sequences mined from train split (shared by both sequence models)
```

---

## Datasets

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `url` | string | Crawled page URL |
| `label` | string | `LOGIN_FORM_MALICIOUS` or `NO_FORM` (retest uses `TEST_LOGIN_FORM` / `TEST_NO_FORM`) |
| `reason` | string | Optional analyst note |
| `host_id` | float | Internal host identifier |
| `html_signature` | string | Compact DOM tree, e.g. `(body(div(form(input)(button))))` |

### Sizes

| Split | Malicious | No-Form | Total |
|-------|-----------|---------|-------|
| Training (deduplicated) | 878 | 1,194 | 2,072 |
| Retest (held-out) | 109 | 99 | 208 |

Training is mildly imbalanced (42.4% / 57.6%); retest is near-balanced (52.4% / 47.6%).

**Data loading** — `train_*.csv` files: login_form is semicolon-delimited; no_form is comma-delimited with an index column.

---

## HTML Signature Format

```
(body(div(header(nav(ul(li(a))(li(a)))))(main(div(form(label(input))(button))))))
```

- Each `(tag(...children...))` encodes an element and its children
- Content, attributes, and class names are stripped — structure only
- Duplicate signatures are dropped before training (18 duplicates found in no_form)

---

## ML Pipeline

### Preprocessing

- Drop duplicate `html_signature` rows (keep first occurrence per class)
- 80/20 stratified train/test split on the training data
- Retest set is never touched during training or feature selection

### Feature Tracks

**Sequence model** (`src/features.py: mine_sequences, build_features`)

- Tokenise each signature into a flat tag sequence, then extract 6-grams
- Mine qualifying sequences on the **train split only**: ≥10% malicious presence, ≤1% no-form presence, must contain at least one credential tag (`form`, `input`, `label`, `iframe`, `style`)
- Results in 17 binary flags + a match-count feature (18 total)
- Weakness: features are kit-specific; generalises poorly to unseen phishing kits

**Structural model** (`src/features.py: structural_features`)

14 kit-agnostic features:

| Feature | Description |
|---------|-------------|
| `has_{form,input,label,iframe,style}` | Credential-tag presence flags |
| `input_count`, `label_count`, `form_count`, `button_count` | Raw tag counts |
| `signature_length` | Character length of the signature string |
| `total_tag_count` | Total number of tags in the DOM |
| `form_density` | `(form + input + label) / total_tags` |
| `input_per_form` | `input_count / form_count` |
| `has_credential_pattern` | 1 if all 5 credential tags are present |

### Models

Both tracks train both **XGBoost** (`n_estimators=200, max_depth=4, learning_rate=0.1`) and **Random Forest** (`n_estimators=200, class_weight="balanced"`), enabling a direct algorithm comparison within each feature track.

---

## Results

### Retest (held-out) performance

| Model | Features | Accuracy | ROC-AUC | MAL Recall | MAL Precision |
|-------|----------|----------|---------|------------|---------------|
| XGBoost | Sequence (6-gram) | 72.1% | 73.1% | 45.7% | 98.0% |
| Random Forest | Sequence (6-gram) | 72.6% | 73.0% | 47.6% | 96.2% |
| **XGBoost** | **Structural (14 features)** | **82.2%** | **91.1%** | **94.3%** | **76.2%** |
| Random Forest | Structural (14 features) | 79.8% | 92.0% | 93.3% | 73.7% |

**Key finding:** structural features generalise far better to unseen kits than sequence-based features. Malicious pages have short, dense signatures (mean 1,173 chars vs 8,649 for no-form); `form_density` and `signature_length` are the strongest discriminators.

### Label-noise observation

10 `NO_FORM` retest rows contained "login" in their URL. 4 had real login structure (`has_form=1`, `has_input=1`) and have been relabeled to `TEST_LOGIN_FORM`. The remaining 6 are parked/taken-down URLs with no DOM structure and are correctly labeled `NO_FORM`.

### Threshold

Default 0.5 is the accuracy-optimal operating point (confirmed via GroupKFold threshold sweep). To trade precision for recall, lower the threshold — a missed malicious login form is costlier than a false positive.
