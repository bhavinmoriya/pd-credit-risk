"""Central configuration. Every random choice in the project derives from SEED."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
MODEL_DIR = ROOT / "models"

SEED = 42
TEST_SIZE = 0.25

# Cost matrix published with the UCI Statlog (German Credit Data) documentation:
# granting a loan to a customer who defaults costs 5x as much as refusing a good one.
COST_FALSE_ACCEPT = 5.0   # approve an applicant who then defaults (loss given default, net of margin)
COST_FALSE_REJECT = 1.0   # reject an applicant who would have repaid (lost margin)


@dataclass(frozen=True)
class CVConfig:
    outer_splits: int = 5
    outer_repeats: int = 2
    inner_splits: int = 3
    rf_trees: int = 200
    bootstrap: int = 1000

    @classmethod
    def fast(cls) -> "CVConfig":
        return cls(outer_splits=3, outer_repeats=1, inner_splits=3, rf_trees=100, bootstrap=200)


def ensure_dirs() -> None:
    for d in (RAW_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
