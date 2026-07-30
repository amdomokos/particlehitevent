"""Post-training quantization analysis of the S4 state-space matrices (Phase 6).

Read-only with respect to the training pipeline: this package loads trained
checkpoints, perturbs their A/B/C parameters in place on an in-memory copy,
and evaluates. It never trains, never writes a checkpoint, and never imports
anything from the training path except the frozen evaluation code
(``Models.common.metrics``) and the frozen model builders.
"""
