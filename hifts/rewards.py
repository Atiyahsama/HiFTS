import re
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

from .prompts import SYSTEM_PROMPT
from .traits import (
    COT_COLUMN,
    ESSAY_COLUMN,
    TOTAL_COLUMN,
    TRAIT_COLUMNS,
    load_trait_map,
)


def reward_lambdas_from_alpha(alpha: float) -> Dict[str, float]:
    """Paper GRPO/PPO reward weights. Default alpha=0.8."""
    return {
        "QWK_Total": float(alpha),
        "QWK_Traits": 0.5 * float(alpha),
        "Cosine": 1.0 - float(alpha),
        "Structure": 0.1,  # paper δ
        "MSE_Traits": 0.5 * float(alpha),
    }


def _get_cell(row, *names):
    for name in names:
        if name in row and pd.notnull(row[name]):
            return row[name]
    return None


def _paired_score(row, name: str):
    """CFMS-34: two raters score 0–5; HiFTS sums them to the 0–10 scale used in the paper."""
    a = _get_cell(row, f"{name}_1")
    b = _get_cell(row, f"{name}_2")
    if a is not None and b is not None:
        return float(a) + float(b)
    single = _get_cell(row, name)
    if single is not None:
        return float(single)
    if a is not None:
        return float(a)
    if b is not None:
        return float(b)
    return None


def load_and_process_data(df_or_path, tokenizer) -> Dataset:
    load_trait_map()
    df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path
    records = []
    for _, row in df.iterrows():
        essay = str(
            _get_cell(row, ESSAY_COLUMN, "text", "input_text", "essay_text") or ""
        )
        total = float(
            _paired_score(row, TOTAL_COLUMN)
            or _get_cell(row, "total_score", "score", "overall_score")
            or 0.0
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": essay},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        traits = {}
        for trait in TRAIT_COLUMNS:
            val = _paired_score(row, trait)
            if val is not None:
                traits[trait] = float(val)
        review = str(_get_cell(row, COT_COLUMN, "reference_text", "reference") or "")
        records.append(
            {
                "prompt": prompt,
                "ground_truth_response": {
                    "review": review,
                    "traits": traits,
                    "total": total,
                },
            }
        )
    return Dataset.from_list(records)


load_rl_dataset = load_and_process_data


def parse_model_output(text: str) -> Dict:
    out = {"review": "", "traits": {}, "total": -1.0, "format_valid": False}
    review = re.search(
        r"\[分析\]\s*(.*?)\s*\[细分(?:维度)?(?:得分|评分)\]",
        text,
        re.DOTALL,
    )
    if review is None:
        review = re.search(r"\[Analysis\]\s*(.*?)\s*\[Trait Scores\]", text, re.DOTALL)
    if review:
        out["review"] = review.group(1).strip()

    for key, score in re.findall(r"([A-Z]\d{2,3}):(\d+\.?\d*)", text):
        if key in TRAIT_COLUMNS:
            out["traits"][key] = float(score)

    total = re.search(r"\[总分\]\s*(\d+\.?\d*)", text)
    if total is None:
        total = re.search(r"\[Overall Score\]\s*(\d+\.?\d*)", text)
    if total:
        out["total"] = float(total.group(1))

    out["format_valid"] = bool(out["review"]) and bool(out["traits"]) and out["total"] >= 0.0
    return out


class CustomRewardFunction:
    def __init__(
        self,
        alpha: float = 0.8,
        embedding_model: str = "./models/bge-small-zh",
        device: Optional[str] = None,
        history_len: int = 64,
        lambdas: Optional[Dict[str, float]] = None,
    ):
        self.alpha = float(alpha)
        self.lambdas = lambdas if lambdas is not None else reward_lambdas_from_alpha(self.alpha)
        self.encoder = SentenceTransformer(embedding_model, device=device)
        self.keys = TRAIT_COLUMNS
        self.hist_pred = deque(maxlen=history_len)
        self.hist_true = deque(maxlen=history_len)

    @staticmethod
    def _align(pred: Dict[str, float], true: Dict[str, float], keys: List[str]):
        return np.array([pred.get(k, 0.0) for k in keys]), np.array(
            [true.get(k, 0.0) for k in keys]
        )

    @staticmethod
    def _qwk(y_pred: np.ndarray, y_true: np.ndarray, max_val: int = 10) -> float:
        p = np.round(np.clip(y_pred, 0, max_val)).astype(int)
        t = np.round(np.clip(y_true, 0, max_val)).astype(int)
        if np.array_equal(p, t):
            return 1.0
        if len(np.unique(p)) == 1 and len(np.unique(t)) == 1:
            return 0.0
        try:
            return float(
                cohen_kappa_score(t, p, labels=list(range(max_val + 1)), weights="quadratic")
            )
        except Exception:
            return 0.0

    def compute_reward(self, generated_text: str, gt_data: Dict) -> float:
        pred = parse_model_output(generated_text)
        gt_total = float(gt_data["total"])
        gt_traits = gt_data["traits"]
        gt_review = gt_data["review"]

        r_structure = 1.0 if pred["format_valid"] else -2.0
        pred_total = pred["total"] if pred["total"] >= 0.0 else 0.0

        r_cosine = 0.0
        if pred["review"] and gt_review:
            emb = self.encoder.encode([pred["review"], gt_review])
            r_cosine = float(cosine_similarity([emb[0]], [emb[1]])[0][0])

        pred_traits, true_traits = self._align(pred["traits"], gt_traits, self.keys)
        mixed_pred = np.append(pred_traits, pred_total)
        mixed_true = np.append(true_traits, gt_total)
        r_qwk_traits = self._qwk(mixed_pred, mixed_true)

        diff = min(abs(pred_total - gt_total), 10.0)
        r_mse = 1.0 - (diff / 10.0)

        self.hist_pred.append(pred_total)
        self.hist_true.append(gt_total)
        r_qwk_total = 0.0
        if len(self.hist_pred) >= 50:
            r_qwk_total = self._qwk(np.array(self.hist_pred), np.array(self.hist_true))

        return (
            self.lambdas["QWK_Total"] * r_qwk_total
            + self.lambdas["QWK_Traits"] * r_qwk_traits
            + self.lambdas["Cosine"] * r_cosine
            + self.lambdas["Structure"] * r_structure
            + self.lambdas["MSE_Traits"] * r_mse
        )


def group_normalize_rewards(rewards, eps: float = 1e-8):
    mean = rewards.mean()
    std = rewards.std()
    if float(std) < eps:
        return rewards - mean
    return (rewards - mean) / (std + eps)
