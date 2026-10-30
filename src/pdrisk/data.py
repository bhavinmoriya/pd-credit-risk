"""Data acquisition, documentation, decoding and target definition.

Source
------
Statlog (German Credit Data), Prof. Dr. Hans Hofmann, Universität Hamburg, donated 1994
to the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/144).
1000 consumer loans from one German bank; 20 application-time attributes + outcome.

The raw file is the original coded version (A11, A34, ...). We decode it ourselves from the
official attribute description instead of using a pre-decoded mirror, because we found two
pre-decoded copies to be wrong (see DATA_CARD.md, "Provenance checks").
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from .config import RAW_DIR, SEED, TEST_SIZE, ensure_dirs

RAW_FILE = RAW_DIR / "german.data"
# SHA-256 of the canonical CSV serialisation of the 1000x21 raw code table.
CANONICAL_SHA256 = "ec12a88b9fc14d74ba646ea0410cf7ff4533bec2eb61652f8ad76796bbfec017"
SOURCES = [
    ("UCI zip", "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"),
    ("GitHub mirror", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"),
]

RAW_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "amount", "savings",
    "employment_duration", "installment_rate", "personal_status_sex", "other_debtors",
    "present_residence", "property", "age", "other_installment_plans", "housing",
    "number_credits", "job", "people_liable", "telephone", "foreign_worker", "class",
]

# Official UCI attribute codes -> labels. (A47 'vacation' and A95 'female: single' have 0 rows.)
CODEBOOK: dict[str, dict[str, str]] = {
    "status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM / salary assignment",
               "A14": "no checking account"},
    "credit_history": {"A30": "no credits taken / all paid duly", "A31": "all credits at this bank paid duly",
                       "A32": "existing credits paid duly till now", "A33": "delay in paying off in the past",
                       "A34": "critical account / other credits elsewhere"},
    "purpose": {"A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
                "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs",
                "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business",
                "A410": "others"},
    "savings": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM",
                "A65": "unknown / no savings account"},
    "employment_duration": {"A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years",
                            "A74": "4-7 years", "A75": ">= 7 years"},
    "personal_status_sex": {"A91": "male: divorced/separated", "A92": "female: divorced/separated/married",
                            "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "building society savings / life insurance",
                 "A123": "car or other", "A124": "unknown / no property"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemployed / unskilled non-resident", "A172": "unskilled resident",
            "A173": "skilled employee / official", "A174": "management / self-employed / highly qualified"},
    "telephone": {"A191": "no", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

NUMERIC = ["duration", "amount", "age", "installment_rate", "present_residence",
           "number_credits", "people_liable"]
CATEGORICAL = list(CODEBOOK)
FEATURES = NUMERIC + CATEGORICAL
# Attributes that are (or proxy) legally protected characteristics; used in the fairness audit.
PROTECTED = ["age", "personal_status_sex", "foreign_worker"]

DATA_DICTIONARY = {
    "status": "Status of existing checking account (balance band)",
    "duration": "Loan duration in months",
    "credit_history": "Credit history with this and other lenders",
    "purpose": "Purpose of the loan",
    "amount": "Credit amount in Deutsche Mark (DM)",
    "savings": "Savings account / bonds balance band",
    "employment_duration": "Present employment since",
    "installment_rate": "Instalment rate as % of disposable income, ordinal code 1-4 (as coded by UCI)",
    "personal_status_sex": "Personal status and sex (combined attribute; sex is NOT recorded separately)",
    "other_debtors": "Other debtors / guarantors",
    "present_residence": "Present residence since, ordinal code 1-4 (as coded by UCI)",
    "property": "Most valuable property owned",
    "age": "Age in years",
    "other_installment_plans": "Other instalment plans (bank/stores/none)",
    "housing": "Housing situation",
    "number_credits": "Number of existing credits at this bank (1-4)",
    "job": "Job category",
    "people_liable": "Number of people liable to provide maintenance (1 or 2)",
    "telephone": "Telephone registered in customer's name",
    "foreign_worker": "Foreign worker",
    "default": "TARGET. 1 = bad credit / default, 0 = good credit (UCI class 2 -> 1, class 1 -> 0)",
}


def _canonical_hash(raw: pd.DataFrame) -> str:
    return hashlib.sha256(raw.to_csv(index=False, header=False).encode()).hexdigest()


def _parse(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), header=None, sep=r"[\s,]+", engine="python")
    raw.columns = RAW_COLUMNS
    return raw


def _download() -> pd.DataFrame:
    errors = []
    for name, url in SOURCES:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                payload = r.read()
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(payload)) as z:
                    payload = z.read("german.data")
            raw = _parse(payload.decode())
            if _canonical_hash(raw) == CANONICAL_SHA256:
                return raw
            errors.append(f"{name}: checksum mismatch")
        except Exception as e:  # noqa: BLE001 - we want to try the next source
            errors.append(f"{name}: {e}")
    raise RuntimeError("Could not obtain a checksum-verified copy of the data:\n  " + "\n  ".join(errors))


def load_raw() -> pd.DataFrame:
    """Return the raw coded table, downloading and verifying it if not cached."""
    ensure_dirs()
    if RAW_FILE.exists():
        raw = _parse(RAW_FILE.read_text())
        if _canonical_hash(raw) == CANONICAL_SHA256:
            return raw
    raw = _download()
    raw.to_csv(RAW_FILE, index=False, header=False)
    return raw


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Decoded features X (object columns for categoricals) and binary default target y."""
    raw = load_raw()
    X = raw.drop(columns="class").copy()
    for col, mapping in CODEBOOK.items():
        decoded = X[col].map(mapping)
        if decoded.isna().any():
            raise ValueError(f"Undecodable codes in {col}: {sorted(X.loc[decoded.isna(), col].unique())}")
        X[col] = decoded
    # UCI: 1 = good, 2 = bad. Default (event of interest) = bad.
    if set(raw["class"].unique()) != {1, 2}:
        raise ValueError("Unexpected class labels")
    y = (raw["class"] == 2).astype(int).rename("default")
    return X[FEATURES], y


def sex_from_status(X: pd.DataFrame) -> pd.Series:
    """Derive sex from the combined personal-status attribute (fairness analysis only)."""
    return X["personal_status_sex"].str.split(":").str[0].rename("sex")


def split(X: pd.DataFrame, y: pd.Series):
    """Single stratified hold-out split. The test set is touched exactly once, at the end."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
