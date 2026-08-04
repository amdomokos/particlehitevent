"""Phase 7 — weight-perturbation robustness of the S4 state-space matrices.

Read-only with respect to training. This package loads trained checkpoints,
adds Gaussian noise to A/B/C in memory, evaluates on the test split, and
restores the trained weights before the next configuration. It never trains,
never touches an optimizer, and never writes a checkpoint.

Answers the proposal's Section 7.4 question: how does test MSE degrade as
``theta' = theta + eps``, ``eps ~ N(0, sigma^2)``, and is the slope of that
degradation smaller for target-aware modulated models than for the plain SSM
baseline?

Distinct from ``Models.quantization``, which measures discrete low-bit
rounding. Same "perturb, evaluate, restore" machinery — reused by import, not
by copy — but a different, continuous and stochastic, perturbation mechanism.
"""
