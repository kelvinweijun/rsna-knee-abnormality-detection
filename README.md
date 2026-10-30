# RSNA 2026 Knee Abnormality Detection

Working repository for the RSNA 2026 knee-MRI competition: a 12-finding multi-label
classifier over knee MRI studies (ACL, MCL, medial/lateral meniscus, medial/lateral/PF OA,
effusion, synovitis, Baker's cyst, contusion, fracture).

**Final submission:** 0.958 public / 0.942 private (ref 57459811).

## Layout

| Path | Contents |
|---|---|
| `knee_meta/` | Core library: data loading, imaging/volume preprocessing, labels, model, metrics, inference |
| `kernels/` | Kaggle kernel sources (`train`, `infer`) |
| `meta_notebooks/` | Post-competition research kernels (CoAtNet/ConvNeXt fine-tunes, SSL, gold evaluation, soups) |
| `notebooks/` | End-to-end training / inference notebooks |
| `tools/` | Stacking, blending, fold-publishing and notebook-build utilities |
| `tests/` | Unit tests for the library |
| `comp_pages/`, `comp_discussion/` | Competition description pages and scraped discussion threads |
| `folds-data*/` | Fold metadata (`cv_summary.json`, `dataset-metadata.json`); the `fold*.pt` weights are excluded (see below) |
| `infer-out/` | Inference log and final `submission.csv` |
| `train.csv`, `train_series.csv` | Competition metadata |

## Post-competition research findings

All arms were re-scored on the only labelled holdout available (58 studies, macro AUC),
using one harness (`meta_notebooks/gold_eval_arms*/`):

| Configuration | gold58 |
|---|---|
| Distilled CoAtNet-384 + scale TTA (1.00/0.85) | **0.9389** |
| Distilled CoAtNet-384 (best single reader) | 0.9371 |
| Soup distilled + label-fine-tune (50/50) | 0.9366 |
| Label fine-tune from distilled, 12 epochs (`ft_dist61b`) | 0.9309 |
| Nartaa CoAtNet-384 SWA | 0.9142 |
| ConvNeXt-S 384, 16 epochs | 0.8816 |
| ConvNeXt-S 384 from 6k-step domain SSL | 0.8856 |

Negative results worth recording:

- **Blending does not generalize.** Per-finding weights fitted on half of the 58 studies and
  applied to the other half beat the best single reader in only 22% of 200 splits (mean
  −0.0063). Weight soups, mirror TTA and window-count changes (k = 12/44) also failed to
  beat the plain 384 reader at k = 24.
- **Domain SSL gave no transfer.** SimCLR on 24k unlabelled series (hard positives from
  different slices, 6k steps) initialised ConvNeXt-S no better than ImageNet weights.
- **ConvNeXt-S is the wrong family for this corpus** (~0.88 vs ~0.91 for CoAtNet at
  comparable cost).
- **Labels are not the bottleneck.** Fused labels score 0.9547 on gold58 vs 0.9273 for the
  frontier-LLM hybrid labels; mixing the two hurts.
- **Scale TTA is the one positive lever** (+0.0017, 95% CI −0.0035..+0.0074).

## Excluded files

The 15 `folds-data*` checkpoints (`fold*.pt`, 85 MB each, ~1.28 GB total) are excluded via
`.gitignore`. Everything else is committed.
