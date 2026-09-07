"""
Extract false positives and false negatives for all four trained models on the retest set.
Writes one Excel sheet per model, with all original fields plus error metadata.

Usage (from project root, with venv active):
    python extract_errors.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from features import STRUCTURAL_COLS, structural_features, build_features

NGRAM_N   = 6
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data" / "retest"
OUT_FILE   = ROOT / "error_analysis.xlsx"

# ── Load retest data ──────────────────────────────────────────────────────────
lf_rt = pd.read_csv(DATA_DIR / "signed_retest_login_form.csv", index_col=0)
nf_rt = pd.read_csv(DATA_DIR / "signed_retest_no_form.csv",    index_col=0)

df_rt = pd.concat([lf_rt, nf_rt], ignore_index=True)
df_rt.drop(columns=["schreenshot_path"], errors="ignore", inplace=True)
df_rt.dropna(subset=["html_signature"], inplace=True)
df_rt.reset_index(drop=True, inplace=True)
df_rt["binary_label"] = df_rt["label"].str.contains("LOGIN_FORM", case=False, na=False).astype(int)

print(f"Retest: {len(df_rt)} rows | Malicious: {df_rt['binary_label'].sum()} | No-Form: {(df_rt['binary_label']==0).sum()}")

# ── Feature matrices ──────────────────────────────────────────────────────────
df_struct = structural_features(df_rt.copy())
X_struct  = df_struct[STRUCTURAL_COLS].values

with open(MODELS_DIR / "sequence_qualifying_seqs.json") as f:
    QUALIFYING_SEQS = json.load(f)
X_seq = build_features(df_rt["html_signature"], QUALIFYING_SEQS, NGRAM_N)

y_true = df_rt["binary_label"].values

MODELS = {
    "structural_xgboost": (joblib.load(MODELS_DIR / "structural_xgboost.pkl"),    X_struct),
    "structural_rf":      (joblib.load(MODELS_DIR / "structural_random_forest.pkl"), X_struct),
    "sequence_xgboost":   (joblib.load(MODELS_DIR / "sequence_xgboost.pkl"),      X_seq),
    "sequence_rf":        (joblib.load(MODELS_DIR / "sequence_random_forest.pkl"), X_seq),
}

# ── Extract errors and write Excel ───────────────────────────────────────────
BASE_COLS = [c for c in ["url", "sha256", "label", "html_signature"] if c in df_rt.columns]

with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
    for model_name, (model, X) in MODELS.items():
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        fp_idx = (y_pred == 1) & (y_true == 0)
        fn_idx = (y_pred == 0) & (y_true == 1)

        parts = []
        for mask, error_type, pred_label in [
            (fp_idx, "FALSE_POSITIVE", "LOGIN_FORM_MALICIOUS"),
            (fn_idx, "FALSE_NEGATIVE", "NO_FORM"),
        ]:
            chunk = df_rt.loc[mask, BASE_COLS].copy()
            chunk.insert(0, "error_type",       error_type)
            chunk.insert(1, "predicted_label",   pred_label)
            chunk.insert(2, "confidence_score",  y_prob[mask].round(4))
            parts.append(chunk)

        sheet = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        sheet.to_excel(writer, sheet_name=model_name, index=False)
        print(f"  {model_name:25s}  FP={fp_idx.sum():3d}  FN={fn_idx.sum():3d}  → sheet '{model_name}'")

print(f"\nSaved → {OUT_FILE}")
