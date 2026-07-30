# Phase 6 summary — quantization robustness

64 quantized evaluations + 2 unquantized reference passes over the full 79,829-sample test split, inference only, evaluated in **fp32**. Total wall clock 0.58 h. ΔMSE is referenced to the **measured fp32 baseline**.

## Baseline verification

| Checkpoint | Table 2 MSE (bf16) | Unquantized MSE (fp32) | Offset |
|---|---|---|---|
| s4_concat | 10.232754 | 10.210532 | -0.022222 |
| s4_modulate_full | 10.567923 | 10.524977 | -0.042946 |

## Headline — joint A+B+C quantization

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

Mean ΔMSE across bit-widths, per subset (higher = more sensitive to quantization).

| Quantized matrices | s4_concat (PT) | s4_concat (PC) | s4_modulate_full (PT) | s4_modulate_full (PC) |
|---|---|---|---|---|
| A | +3290 | +2959 | +384.7 | +359.3 |
| B | +1.185 | +0.1521 | +0.986 | +0.1282 |
| C | +79.42 | +8.439 | +58.22 | +1.359 |
| A+B+C | +1350 | +2443 | +722.7 | +325.2 |

## Reproduce

```
python -m Models.quantization.sweep \
    --checkpoint-root /workspace/table2_run_1 \
    --data-dir /workspace/repo/preprocessed_data
```

Full per-target degradation: `quantization_per_target.csv`. All aggregates and both delta conventions: `Table4_all.csv`. Raw records including per-tensor weight-space quantization error: `results.jsonl`.
