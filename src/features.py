"""Shared feature helpers for login-form detection notebooks."""
import re
from collections import Counter

import pandas as pd
import scipy.sparse as sp

CREDENTIAL_GROUP = ["form", "input", "label", "iframe", "style"]

STRUCTURAL_COLS = (
    [f"has_{t}" for t in CREDENTIAL_GROUP]
    + ["input_count", "label_count", "form_count", "button_count"]
    + ["signature_length", "total_tag_count", "form_density", "input_per_form"]
    + ["has_credential_pattern"]
)


# sequence helpers

def tokenize(sig):
    return re.findall(r"\((\w+)", sig)


def get_ngrams(tokens, n):
    return [" > ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def count_ngrams(sigs, n):
    counts = Counter()
    for sig in sigs:
        for ng in set(get_ngrams(tokenize(sig), n)):
            counts[ng] += 1
    return counts


def has_cred_tag(ng):
    return any(t in ng.split(" > ") for t in CREDENTIAL_GROUP)


def mine_sequences(mal_sigs, nf_sigs, n, min_mal_pct=10.0, max_nf_pct=1.0):
    """Return qualifying exclusive n-gram sequences. Call on the train split only."""
    total_mal, total_nf = len(mal_sigs), len(nf_sigs)
    mal_counts = count_ngrams(mal_sigs, n)
    nf_counts = count_ngrams(nf_sigs, n)
    qualifying = []
    for ng, cnt in mal_counts.items():
        if not has_cred_tag(ng):
            continue
        m_pct = cnt / total_mal * 100
        n_pct = nf_counts.get(ng, 0) / total_nf * 100 if total_nf else 0.0
        if m_pct >= min_mal_pct and n_pct <= max_nf_pct:
            qualifying.append(ng)
    return qualifying


def build_features(html_signatures, qualifying_seqs, n):
    rows = []
    for sig in html_signatures:
        page_ngrams = set(get_ngrams(tokenize(sig), n))
        flags = [int(ng in page_ngrams) for ng in qualifying_seqs]
        rows.append(flags + [sum(flags)])
    return sp.csr_matrix(rows, dtype=float)


# structural helpers

def has_tag(sig, tag):
    return int(bool(re.search(rf"\({tag}\b", sig)))


def count_tag(sig, tag):
    return len(re.findall(rf"\({tag}\b", sig))


def count_all_tags(sig):
    return len(re.findall(r"\(\w+", sig))


def structural_features(frame):
    """Add STRUCTURAL_COLS to a copy of *frame* and return it."""
    out = frame.copy()
    for tag in CREDENTIAL_GROUP:
        out[f"has_{tag}"] = out["html_signature"].apply(lambda s, t=tag: has_tag(s, t))
    for tag in ["input", "label", "form", "button"]:
        out[f"{tag}_count"] = out["html_signature"].apply(lambda s, t=tag: count_tag(s, t))
    out["signature_length"] = out["html_signature"].str.len()
    out["total_tag_count"] = out["html_signature"].apply(count_all_tags)
    out["form_density"] = (
        (out["form_count"] + out["input_count"] + out["label_count"])
        / out["total_tag_count"].replace(0, 1)
    )
    out["input_per_form"] = out["input_count"] / out["form_count"].replace(0, 1)
    cred_cols = [f"has_{t}" for t in CREDENTIAL_GROUP]
    out["has_credential_pattern"] = out[cred_cols].all(axis=1).astype(int)
    return out
