# HiFTS & CFMS-34

Code and data accompanying the paper:

**A Unified Framework to Elicit Structured Feedback for Interpretable Multi-Trait Essay Scoring**

Shihang Yang, Sanwoo Lee, Ningning Zhao, Yunfang Wu. Findings of EMNLP 2026. [OpenReview](https://openreview.net/forum?id=IAvnGjEEUd)

This release has two parts:

1. **CFMS-34** — Chinese Fine-grained Multi-trait Scoring. 951 primary-school narrative essays with dual expert ratings on a **34-trait** rubric plus a holistic score.
2. **HiFTS**, Hierarchical Feedback-to-Score Reasoning — a unified autoregressive framework that first generates global-to-local CoT feedback, then predicts **20 core trait scores** and a holistic score.

Student models are warmed up with SFT on teacher CoT, then aligned with GRPO. PPO is an ablation. At inference, a BERT holistic prior is inserted into the prompt as a soft anchor, not the final score.

ASAP++ is used in the paper but is not bundled here.

## Layout

```text
CFMS-34/
  README.md
  train.csv  dev.csv  test.csv     # 751 / 100 / 100
  trait_map.csv                    # 20 core HiFTS traits ↔ original B-codes
hifts/                             # prompts, trait map, rewards, metrics
scripts/                           # GRPO, PPO, inference, eval
```

Baselines follow the original authors’ code and are not included.

## CFMS-34

HiFTS does not train on all 34 traits. We select the **20** traits with the highest Pearson correlation to the holistic score on the training set; `trait_map.csv` records that subset.

See [`CFMS-34/README.md`](CFMS-34/README.md) for scores, splits, and anonymization.

## Method

A teacher LLM first writes hierarchical feedback together with trait scores and an overall score. The student is warmed up with SFT on this feedback-to-score sequence, then aligned with GRPO; PPO is an ablation. At inference, a BERT holistic prior is prepended to the system prompt to stabilize long-form reasoning. The student prompt is in `hifts/prompts.py`.

## Setup

```bash
pip install -r requirements.txt
```

SFT uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). After an SFT checkpoint is ready, `python scripts/train_grpo.py` runs GRPO on `CFMS-34`. `python scripts/infer.py --essay_file essay.txt` generates hierarchical feedback and scores for a single essay. Point `MODEL_PATH` and `LLM_PATH` to local checkpoints.

This artifact is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

```bibtex
@inproceedings{yang-etal-2026-unified,
  title     = {A Unified Framework to Elicit Structured Feedback for Interpretable Multi-Trait Essay Scoring},
  author    = {Shihang Yang and Sanwoo Lee and Ningning Zhao and Yunfang Wu},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  month     = oct,
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  url       = {https://openreview.net/forum?id=IAvnGjEEUd}
}
```
