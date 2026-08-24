import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIT_MAP = ROOT / "CFMS-34" / "trait_map.csv"

ALL_TRAIT_COLUMNS = []
TRAIT_COLUMNS = []
CFMS_TRAIT_ID = {}
TRAIT_DEFINITIONS = {}

ESSAY_COLUMN = "作文内容"
TOTAL_COLUMN = "总分"
COT_COLUMN = "Generated_CoT"
ID_COLUMN = "编号"


def _is_core(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_trait_map(path=None):
    """Load all 34 CFMS-34 traits (C/S/E/Cv) ↔ original B-codes.

    HiFTS trains and evaluates on the Pearson top-20 subset marked hifts_core=1.
    TRAIT_COLUMNS is that subset; ALL_TRAIT_COLUMNS is the full rubric.
    """
    global ALL_TRAIT_COLUMNS, TRAIT_COLUMNS, CFMS_TRAIT_ID, TRAIT_DEFINITIONS
    map_path = Path(path) if path else Path(os.environ.get("TRAIT_MAP_PATH", DEFAULT_TRAIT_MAP))
    with map_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty trait map: {map_path}")
    ALL_TRAIT_COLUMNS = [row["hifts_id"] for row in rows]
    CFMS_TRAIT_ID = {row["hifts_id"]: row["cfms_id"] for row in rows}
    TRAIT_DEFINITIONS = {row["hifts_id"]: row["definition"] for row in rows}
    if "hifts_core" in rows[0]:
        TRAIT_COLUMNS = [row["hifts_id"] for row in rows if _is_core(row.get("hifts_core"))]
    else:
        TRAIT_COLUMNS = list(ALL_TRAIT_COLUMNS)
    if not TRAIT_COLUMNS:
        raise ValueError(f"No HiFTS core traits in: {map_path}")
    return CFMS_TRAIT_ID


def cfms_id(hifts_id: str) -> str:
    if not CFMS_TRAIT_ID:
        load_trait_map()
    return CFMS_TRAIT_ID[hifts_id]


load_trait_map()
