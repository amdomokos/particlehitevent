# Table 4 — Quantization sensitivity (ΔMSE)

Each cell is **ΔMSE = MSE_quantized − MSE_reference**, where the reference is the **measured fp32 baseline** for that checkpoint. Lower is better; 0 means quantization was free. Metrics are in physical space, over the full test split, computed by the same `compute_table_row` that produced Table 2.

Fake quantization is asymmetric uniform (min/max range, integer zero-point). `A` = `log_A_real` + `A_imag`, `B` = `B`, `C` = `C_real` + `C_imag`; real and imaginary components are quantized independently. `log_dt` is never quantized.

## Baseline verification

| Checkpoint | Table 2 MSE (bf16) | Unquantized MSE (fp32) | Offset |
|---|---|---|---|
| s4_concat | 10.232754 | 10.210532 | -0.022222 |
| s4_modulate_full | 10.567923 | 10.524977 | -0.042946 |

## ΔMSE by checkpoint and granularity

### s4_concat — per-tensor

| Quantized matrices | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|
| A | +0.1941 | +8.102 | +47.47 | +1.31e+04 |
| B | +0.01463 | +0.001059 | +0.03198 | +4.691 |
| C | +0.06174 | +0.1899 | +7.468 | +309.9 |
| A+B+C | +0.219 | +8.741 | +48.87 | +5342 |

### s4_concat — per-channel

| Quantized matrices | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|
| A | -0.04224 | +9.068 | +34.53 | +1.179e+04 |
| B | +0.002844 | -0.007295 | -0.01814 | +0.6311 |
| C | +0.00887 | -0.01324 | +0.195 | +33.56 |
| A+B+C | -0.05186 | +9.005 | +36 | +9728 |

### s4_modulate_full — per-tensor

| Quantized matrices | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|
| A | +0.194 | +2.007 | +11.65 | +1525 |
| B | -0.002953 | +0.05357 | +0.4267 | +3.467 |
| C | -0.001648 | -0.01651 | +1.792 | +231.1 |
| A+B+C | +0.1859 | +2.111 | +30.29 | +2858 |

### s4_modulate_full — per-channel

| Quantized matrices | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|
| A | +0.2406 | +1.846 | +12.15 | +1423 |
| B | +0.000944 | -0.0008219 | +0.04459 | +0.468 |
| C | +0.006875 | +0.0002239 | +0.1622 | +5.265 |
| A+B+C | +0.2514 | +1.941 | +13.53 | +1285 |

## Combined view — granularity side by side

ΔMSE; **PT** = per-tensor, **PC** = per-channel.

### s4_concat

| Quantized matrices | Gran. | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|---|
| A | PT | +0.1941 | +8.102 | +47.47 | +1.31e+04 |
| A | PC | -0.04224 | +9.068 | +34.53 | +1.179e+04 |
| B | PT | +0.01463 | +0.001059 | +0.03198 | +4.691 |
| B | PC | +0.002844 | -0.007295 | -0.01814 | +0.6311 |
| C | PT | +0.06174 | +0.1899 | +7.468 | +309.9 |
| C | PC | +0.00887 | -0.01324 | +0.195 | +33.56 |
| A+B+C | PT | +0.219 | +8.741 | +48.87 | +5342 |
| A+B+C | PC | -0.05186 | +9.005 | +36 | +9728 |

### s4_modulate_full

| Quantized matrices | Gran. | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|---|
| A | PT | +0.194 | +2.007 | +11.65 | +1525 |
| A | PC | +0.2406 | +1.846 | +12.15 | +1423 |
| B | PT | -0.002953 | +0.05357 | +0.4267 | +3.467 |
| B | PC | +0.000944 | -0.0008219 | +0.04459 | +0.468 |
| C | PT | -0.001648 | -0.01651 | +1.792 | +231.1 |
| C | PC | +0.006875 | +0.0002239 | +0.1622 | +5.265 |
| A+B+C | PT | +0.1859 | +2.111 | +30.29 | +2858 |
| A+B+C | PC | +0.2514 | +1.941 | +13.53 | +1285 |

## Δn_z MAE — the target §5.1 predicts degrades most

Δn_z MAE against the same reference. n_z is bounded away from zero (|n_z| ≥ 0.099) and determined jointly with n_x/n_y, so it is the most sensitive direction component.

### s4_concat — per-tensor

| Quantized matrices | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|
| A | +0.0004283 | +0.003411 | +0.01876 | +0.1848 |
| B | +8.511e-06 | +9.563e-05 | +0.000171 | +0.003093 |
| C | -2.741e-05 | -3.351e-05 | +0.00185 | +0.03224 |
| A+B+C | +0.0003761 | +0.003499 | +0.02486 | +0.1487 |

### s4_concat — per-channel

| Quantized matrices | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|
| A | +0.0002425 | +0.003519 | +0.01473 | +0.1848 |
| B | -5.098e-06 | +9.546e-06 | +0.0002264 | +0.001217 |
| C | +1.727e-06 | -3.647e-05 | -4.38e-06 | +0.01119 |
| A+B+C | +0.0002378 | +0.003433 | +0.01481 | +0.2536 |

### s4_modulate_full — per-tensor

| Quantized matrices | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|
| A | +0.0001045 | +0.0006594 | +0.006762 | +0.1174 |
| B | -1.015e-06 | +9.543e-05 | +0.0002647 | +0.003776 |
| C | +7.887e-06 | +5.875e-05 | +0.001162 | +0.01915 |
| A+B+C | +0.0001017 | +0.0008453 | +0.008853 | +0.1248 |

### s4_modulate_full — per-channel

| Quantized matrices | 8-bit | 6-bit | 4-bit | 2-bit |
|---|---|---|---|---|
| A | +0.0001457 | +0.0006573 | +0.00654 | +0.1115 |
| B | +2.707e-07 | -5.679e-06 | +8.531e-05 | -0.0002977 |
| C | +5.253e-06 | +5.899e-05 | +1.436e-06 | +0.001012 |
| A+B+C | +0.0001458 | +0.0007326 | +0.006874 | +0.1103 |

