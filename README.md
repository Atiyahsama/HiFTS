# A Unified Framework to Elicit Structured Feedback for Interpretable Multi-Trait Essay Scoring

[![EMNLP 2026 Findings](https://img.shields.io/badge/EMNLP_2026_Findings-OpenReview-red.svg)](https://openreview.net/forum?id=IAvnGjEEUd)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

Source code and data of our EMNLP 2026 Findings paper **A Unified Framework to Elicit Structured Feedback for Interpretable Multi-Trait Essay Scoring**.

This release includes **HiFTS** (Hierarchical Feedback-to-Score Reasoning) and **CFMS-34**. ASAP++ is used in the paper but is not bundled here. Baselines follow the original authors’ code and are not included.

## Prerequisites

```bash
pip install -r requirements.txt
```

SFT uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Point `MODEL_PATH` and `LLM_PATH` to local checkpoints.

## CFMS-34

CFMS-34 contains 951 dual-rated Chinese primary-school essays with a 34-trait rubric. `CFMS-34/trait_map.csv` lists all 34 C/S/E/Cv codes. HiFTS trains on the 20 traits with the highest Pearson correlation to the holistic score on the training set. See [`CFMS-34/README.md`](CFMS-34/README.md) for splits, score scale, and anonymization.

## HiFTS

A teacher LLM writes hierarchical feedback together with trait scores and an overall score. The student is warmed up with SFT on this sequence, then aligned with GRPO; PPO is an ablation. At inference, a BERT holistic prior is prepended to the system prompt. The student prompt is in `hifts/prompts.py`.

```bash
python scripts/train_grpo.py
python scripts/infer.py --essay_file essay.txt
```

## Citation

If you find our work helpful, feel free to cite us:

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
