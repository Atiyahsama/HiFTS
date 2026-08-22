import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIT_MAP = ROOT / "CFMS-34" / "trait_map.csv"

TRAIT_COLUMNS = []
CFMS_TRAIT_ID = {}
TRAIT_DEFINITIONS = {}

ESSAY_COLUMN = "作文内容"
TOTAL_COLUMN = "总分"
COT_COLUMN = "Generated_CoT"
ID_COLUMN = "编号"


def load_trait_map(path=None):
    """Load the 20 core HiFTS traits (C/S/E) ↔ CFMS-34 B-codes.

    Appendix A: these 20 traits are the training-set Pearson top-20 with the holistic score.
    """
    global TRAIT_COLUMNS, CFMS_TRAIT_ID, TRAIT_DEFINITIONS
    map_path = Path(path) if path else Path(os.environ.get("TRAIT_MAP_PATH", DEFAULT_TRAIT_MAP))
    with map_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty trait map: {map_path}")
    TRAIT_COLUMNS = [row["hifts_id"] for row in rows]
    CFMS_TRAIT_ID = {row["hifts_id"]: row["cfms_id"] for row in rows}
    TRAIT_DEFINITIONS = {row["hifts_id"]: row["definition"] for row in rows}
    return CFMS_TRAIT_ID


def cfms_id(hifts_id: str) -> str:
    if not CFMS_TRAIT_ID:
        load_trait_map()
    return CFMS_TRAIT_ID[hifts_id]


load_trait_map()
