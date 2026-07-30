# Phase 6 summary — quantization robustness

64 quantized evaluations + 2 unquantized reference passes over the full 79,829-sample test split, inference only, evaluated in **fp32**. Total wall clock 0.58 h. ΔMSE is referenced to the **measured fp32 baseline**.

## Baseline verification

| Checkpoint | Table 2 MSE (bf16) | Unquantized MSE (fp32) | Offset |
|---|---|---|---|
| s4_concat | 10.232754 | 10.210532 | -0.022222 |
| s4_modulate_full | 10.567923 | 10.524977 | -0.042946 |

## How to read ΔMSE

The unweighted aggregate MSE is **98.7% `x_entry`** — the five active targets differ in scale by four orders of magnitude, and a plain mean is dominated by the largest. ΔMSE therefore tracks `x_entry` almost exclusively and is close to blind to the three direction components. A configuration can post ΔMSE ≤ 0 while `n_y` and `n_z` measurably degrade. Read the per-target section below (and `quantization_per_target.csv`) before concluding that a bit-width is free.

## Headline — joint A+B+C quantization

> **2-bit destroys both models.** MSE lands 23x to 1284x above baseline there, so those configurations are not usable at any quality bar and their relative ordering carries no signal — differences between two unusable models are noise, not robustness. The interpretable comparison is at the bit-widths above.

The realistic deployment case: all three state-space matrices quantized simultaneously at the same bit-width. **ΔMSE** measures robustness (how much each checkpoint loses); **MSE** measures what you actually deploy. They can disagree, and where they do, that is the finding.

### per-tensor

| Bits | s4_concat ΔMSE | s4_modulate_full ΔMSE | s4_concat MSE | s4_modulate_full MSE | More robust (ΔMSE) | Better absolute MSE |
|---|---|---|---|---|---|---|
| 8 | +0.219 | +0.1859 | 10.4295 | 10.7109 | s4_modulate_full | s4_concat |
| 6 | +8.741 | +2.111 | 18.9517 | 12.6359 | s4_modulate_full | s4_modulate_full |
| 4 | +48.87 | +30.29 | 59.0773 | 40.8117 | s4_modulate_full | s4_modulate_full |
| 2 | +5342 | +2858 | 5351.8020 | 2868.5833 | s4_modulate_full | s4_modulate_full |

### per-channel

| Bits | s4_concat ΔMSE | s4_modulate_full ΔMSE | s4_concat MSE | s4_modulate_full MSE | More robust (ΔMSE) | Better absolute MSE |
|---|---|---|---|---|---|---|
| 8 | -0.05186 | +0.2514 | 10.1587 | 10.7764 | s4_concat | s4_concat |
| 6 | +9.005 | +1.941 | 19.2153 | 12.4659 | s4_modulate_full | s4_modulate_full |
| 4 | +36 | +13.53 | 46.2148 | 24.0594 | s4_modulate_full | s4_modulate_full |
| 2 | +9728 | +1285 | 9737.8947 | 1295.4908 | s4_modulate_full | s4_modulate_full |

## Which matrix is most sensitive

ΔMSE per subset, averaged over the checkpoints, resolved by bit-width. Deliberately not averaged ACROSS bit-widths: the lowest one is orders of magnitude larger and would swamp the rest.

### per-tensor (mean of 2 checkpoints)

| Quantized matrices | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|
| A | +0.1941 | +5.055 | +29.56 | +7314 |
| B | +0.00584 | +0.02731 | +0.2293 | +4.079 |
| C | +0.03005 | +0.08668 | +4.63 | +270.5 |
| A+B+C | +0.2024 | +5.426 | +39.58 | +4100 |

### per-channel (mean of 2 checkpoints)

| Quantized matrices | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|
| A | +0.09919 | +5.457 | +23.34 | +6607 |
| B | +0.001894 | -0.004058 | +0.01323 | +0.5496 |
| C | +0.007873 | -0.006508 | +0.1786 | +19.41 |
| A+B+C | +0.09979 | +5.473 | +24.77 | +5506 |

## Per-target degradation (A+B+C)

Percent change in per-target MAE versus each checkpoint's own unquantized baseline. Positive = worse.

### 8-bit

| Checkpoint | Gran. | x_entry MAE | y_entry MAE | n_x MAE | n_y MAE | n_z MAE | agg MSE |
|---|---|---|---|---|---|---|---|
| s4_concat | PT | +1.92% | +2.72% | -0.45% | +0.33% | +4.78% | +2.14% |
| s4_concat | PC | +0.02% | +2.87% | +1.79% | +1.99% | +3.02% | -0.51% |
| s4_modulate_full | PT | +1.22% | +2.86% | +3.52% | +5.70% | +1.15% | +1.77% |
| s4_modulate_full | PC | +1.66% | +3.29% | +3.55% | +6.36% | +1.65% | +2.39% |

### 6-bit

| Checkpoint | Gran. | x_entry MAE | y_entry MAE | n_x MAE | n_y MAE | n_z MAE | agg MSE |
|---|---|---|---|---|---|---|---|
| s4_concat | PT | +56.48% | +25.36% | +61.05% | +18.69% | +44.49% | +85.61% |
| s4_concat | PC | +57.92% | +21.84% | +69.02% | +22.39% | +43.65% | +88.19% |
| s4_modulate_full | PT | +14.29% | +21.87% | +30.70% | +17.53% | +9.57% | +20.06% |
| s4_modulate_full | PC | +13.45% | +15.33% | +32.41% | +16.87% | +8.29% | +18.44% |

## Reproduce

```
python -m Models.quantization.sweep \
    --checkpoint-root /workspace/table2_run_1 \
    --data-dir /workspace/repo/preprocessed_data
```

Full per-target degradation: `quantization_per_target.csv`. All aggregates and both delta conventions: `Table4_all.csv`. Raw records including per-tensor weight-space quantization error: `results.jsonl`.
