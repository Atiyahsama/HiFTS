# CFMS-34

Chinese Fine-grained Multi-trait Scoring: 951 in-class primary-school essays, each scored independently by two Chinese-language education specialists.

| | |
| --- | --- |
| Essays | 951 |
| Language | Simplified Chinese |
| Setting | Unit writing prompts, 50-minute classroom exams |
| Raters | 2 experts per essay |
| Labels | Holistic score + 34 rubric traits |
| Per-rater range | 0.0 – 5.0 |
| Split | train 751 / dev 100 / test 100 |

The official split is 8:1:1. The test set only contains essays whose two raters agree on the overall score.

## Files

```
CFMS-34/
  trait_map.csv          all 34 traits: id, layer, core flag, definition
  full/                  complete 34-trait labels
    train.csv
    dev.csv
    test.csv
  core/                  20 traits with hifts_core=1 (used by HiFTS)
    train.csv
    dev.csv
    test.csv
```

Each CSV has essay id `编号`, text `作文内容`, title `作文题目`, dual holistic scores `总分_1/2`, and dual trait scores `{id}_1/{id}_2`. Trait ids follow `trait_map.csv` (`C01`–`C06`, `S01`–`S12`, `E01`–`E12`, `Cv01`–`Cv04`).

`full/` keeps all 34 traits. `core/` keeps only the 20 traits marked `hifts_core=1` in `trait_map.csv`. Those 20 are the traits with the highest Pearson correlation to the holistic score on the training set: C02, C04–C06, S02–S03, S06–S12, E01–E02, E05–E07, E11–E12.

## Score scale in HiFTS

Annotators use 0–5. For modeling, the two ratings are summed to 0–10 for the 20 core traits and the holistic score. Do not average `_1` and `_2` if you are reproducing HiFTS. The `hifts` code does this automatically from the dual-rater columns.

## Intended use

Released files are for research use, not high-stakes decisions without human oversight. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Raw exam records and handwriting images are not released.
