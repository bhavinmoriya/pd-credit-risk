import hashlib

import numpy as np

from pdrisk.data import CANONICAL_SHA256, CODEBOOK, FEATURES, load_raw, sex_from_status, split


def test_checksum_pins_the_raw_file():
    raw = load_raw()
    assert hashlib.sha256(raw.to_csv(index=False, header=False).encode()).hexdigest() == CANONICAL_SHA256


def test_shape_and_no_missing(data):
    X, y = data
    assert X.shape == (1000, 20) and list(X.columns) == FEATURES
    assert not X.isna().any().any() and set(y.unique()) == {0, 1}


def test_target_polarity_is_default_equals_one(data):
    """UCI codes 1=good, 2=bad. Row 1 is a good loan, row 2 a bad one; 300 of 1000 are bad."""
    _, y = data
    assert y.sum() == 300
    assert y.iloc[0] == 0 and y.iloc[1] == 1


def test_decoded_marginals_match_uci_documentation(data):
    """Guards against the mislabelled mirrors we found (status A11, purpose A42..A410)."""
    X, _ = data
    expect = {("status", "< 0 DM"): 274, ("status", "no checking account"): 394,
              ("purpose", "radio/television"): 280, ("purpose", "furniture/equipment"): 181,
              ("purpose", "car (new)"): 234, ("purpose", "business"): 97, ("purpose", "others"): 12,
              ("purpose", "retraining"): 9, ("purpose", "education"): 50,
              ("personal_status_sex", "male: single"): 548, ("personal_status_sex", "female: divorced/separated/married"): 310,
              ("credit_history", "critical account / other credits elsewhere"): 293, ("foreign_worker", "yes"): 963}
    for (col, lev), n in expect.items():
        assert (X[col] == lev).sum() == n, (col, lev)


def test_codebook_covers_every_code():
    assert all(len(m) >= 2 for m in CODEBOOK.values())


def test_sex_derivation(data):
    X, _ = data
    s = sex_from_status(X)
    assert s.value_counts().to_dict() == {"male": 690, "female": 310}


def test_split_is_disjoint_stratified_and_reproducible(data):
    X, y = data
    a = split(X, y); b = split(X, y)
    Xtr, Xte, ytr, yte = a
    assert not set(Xtr.index) & set(Xte.index) and len(Xtr) == 750 and len(Xte) == 250
    assert abs(ytr.mean() - 0.3) < 0.005 and abs(yte.mean() - 0.3) < 0.005
    assert list(Xte.index) == list(b[1].index)
