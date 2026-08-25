import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .traits import TRAIT_COLUMNS, load_trait_map

OVERALL_ALIASES = ("总分", "overall_score", "total_score", "score")


@dataclass
class EvalConfig:
    pred_path: Path
    gold_prefix: str
    pred_prefix: str
    min_score: int
    max_score: int
    output_json: Optional[Path]


def _clip_round(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.round(np.clip(x, lo, hi)).astype(int)


def qwk(y_true: np.ndarray, y_pred: np.ndarray, lo: int = 0, hi: int = 10) -> float:
    t = _clip_round(y_true, lo, hi)
    p = _clip_round(y_pred, lo, hi)
    if t.size == 0:
        return float("nan")
    if np.array_equal(t, p):
        return 1.0
    if len(np.unique(t)) == 1 and len(np.unique(p)) == 1:
        return 0.0
    return float(cohen_kappa_score(t, p, labels=list(range(lo, hi + 1)), weights="quadratic"))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true - y_pred) ** 2))


def _candidate_trait_columns(trait: str, prefix: str) -> List[str]:
    return [f"{prefix}{trait}", trait]


def _pick_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def resolve_columns(df: pd.DataFrame, gold_prefix: str, pred_prefix: str) -> Dict:
    gold_overall = _pick_existing(
        df, [f"{gold_prefix}{n}" for n in OVERALL_ALIASES] + list(OVERALL_ALIASES)
    )
    pred_overall = _pick_existing(
        df,
        [f"{pred_prefix}{n}" for n in OVERALL_ALIASES]
        + [f"{n}_pred" for n in OVERALL_ALIASES]
        + ["prediction", "pred"],
    )
    if gold_overall is None or pred_overall is None:
        raise ValueError("Cannot resolve overall score columns.")

    trait_cols: Dict[str, Tuple[str, str]] = {}
    for trait in TRAIT_COLUMNS:
        g = _pick_existing(df, _candidate_trait_columns(trait, gold_prefix))
        p = _pick_existing(
            df,
            _candidate_trait_columns(trait, pred_prefix) + [f"{trait}_pred"],
        )
        if g is not None and p is not None:
            trait_cols[trait] = (g, p)
    return {"overall": (gold_overall, pred_overall), "traits": trait_cols}


def evaluate(df: pd.DataFrame, cfg: EvalConfig) -> Dict:
    load_trait_map()
    resolved = resolve_columns(df, cfg.gold_prefix, cfg.pred_prefix)
    g_col, p_col = resolved["overall"]

    o = df[[g_col, p_col]].dropna()
    overall_qwk = qwk(o[g_col].to_numpy(float), o[p_col].to_numpy(float), cfg.min_score, cfg.max_score)
    overall_mse = mse(o[g_col].to_numpy(float), o[p_col].to_numpy(float))

    trait_scores: Dict[str, float] = {}
    for trait, (g, p) in resolved["traits"].items():
        sub = df[[g, p]].dropna()
        trait_scores[trait] = qwk(
            sub[g].to_numpy(float),
            sub[p].to_numpy(float),
            cfg.min_score,
            cfg.max_score,
        )

    trait_qwk_mean = float(np.nanmean(list(trait_scores.values()))) if trait_scores else float("nan")
    return {
        "overall_qwk": overall_qwk,
        "trait_qwk": trait_qwk_mean,
        "mse": overall_mse,
        "num_samples": int(o.shape[0]),
        "num_trait_dims": int(len(trait_scores)),
        "trait_qwk_detail": trait_scores,
    }


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Overall QWK / Trait QWK / MSE")
    p.add_argument("--pred_path", required=True)
    p.add_argument("--gold_prefix", default="gold_")
    p.add_argument("--pred_prefix", default="pred_")
    p.add_argument("--min_score", type=int, default=0)
    p.add_argument("--max_score", type=int, default=10)
    p.add_argument("--output_json", default="")
    a = p.parse_args()
    out = Path(a.output_json) if a.output_json else None
    return EvalConfig(
        pred_path=Path(a.pred_path),
        gold_prefix=a.gold_prefix,
        pred_prefix=a.pred_prefix,
        min_score=a.min_score,
        max_score=a.max_score,
        output_json=out,
    )


def main():
    cfg = parse_args()
    df = pd.read_csv(cfg.pred_path)
    result = evaluate(df, cfg)
    print(f"Overall QWK : {result['overall_qwk']:.6f}")
    print(f"Trait QWK   : {result['trait_qwk']:.6f}")
    print(f"MSE         : {result['mse']:.6f}")
    print(f"Samples     : {result['num_samples']}")
    print(f"Trait Dims  : {result['num_trait_dims']}")
    if cfg.output_json is not None:
        cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
