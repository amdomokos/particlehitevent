"""Phase 7 perturbation-robustness sweep — inference only, never trains.

python -m Models.robustness.sweep --checkpoint-root /workspace/table2_run_1 \
    --data-dir /workspace/repo/preprocessed_data \
    [--output-dir <root>/robustness_results] \
    [--models s4 s4_concat s4_modulate_full] [--subsets A B C A+B+C] \
    [--sigmas 0.001 ... 1.0] [--repeats 3] [--headline-repeats 5] \
    [--batch-size 512] [--num-workers 4] [--precision fp32]
    [--limit-samples N] [--baseline-only] [--dry-run] [--force] [--no-tables]
    [--checkpoint-name best.pt]

Default plan: 3 checkpoints x 7 sigmas, with the joint A+B+C subset at 5 seeds
(the headline curve, which needs tight error bars) and the individual A / B / C
subsets at 3 seeds (a qualitative ordering question), plus one unperturbed
sigma=0 pass per checkpoint. 105 + 189 + 3 = 297 evaluations. Every one is a
single forward pass over the test split: no backward pass, no optimizer, and
nothing is ever written back to a checkpoint.

Why repeats. Quantization is deterministic given a bit-width; Gaussian
perturbation is not. A single draw per (model, sigma) would produce a curve
whose wiggles are indistinguishable from perturbation-instance variance, and
the proposal asks specifically whether degradation is SMOOTH. Reporting
mean +/- std over independent seeds is what makes that question answerable.

FP32 reference. Same reasoning as Phase 6: Table 2 ran under bf16 autocast,
but bf16 compute noise is itself a weight perturbation of roughly 0.4%
relative, which sits inside this study's sigma range and would contaminate the
small-sigma end. This sweep evaluates in fp32 and measures its own sigma=0
baseline per checkpoint, so every delta is referenced to arithmetic identical
to the perturbed runs. ``delta_mse_vs_table2`` is also recorded so the
bf16-vs-fp32 offset stays visible.

Weight restore is inherited from the quantization harness — the pristine
state_dict is deep-copied to CPU once per checkpoint, reloaded before every
configuration, and verified bit-exact. It matters more here than there: noise
accumulates additively, so an unrestored model would make sigma effectively
grow down the sweep order while every table still looked plausible.

Resumability: each configuration appends one JSON line to results.jsonl on
completion and a rerun skips keys already present (``--force`` re-runs them).
"""
import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone

import torch

import Models.models_import_all  # noqa: F401 — populates registry
from Models.common.device import get_device, gpu_summary
# The training engine's own evaluation loop. Reusing it — private name and all
# — is what keeps these MSEs comparable to Table 2 and to Phase 6's Table 4.
from Models.common.engine import _collect_predictions, load_checkpoint
from Models.common.metrics import compute_table_row
from Models.common.registry import build_model, list_models
from Models.robustness import tables
from Models.robustness.perturb import SIGMAS, SUBSETS, apply_perturbation, \
    config_seed, validate_sigma
# Proven Phase 6 machinery, imported rather than copied. Phase 6 is not
# modified by this package.
from Models.quantization.sweep import (
    _resolve_amp,
    append_result,
    build_test_loader,
    completed_keys,
    load_reference,
    verify_restore,
)

DEFAULT_MODELS = ('s4', 's4_concat', 's4_modulate_full')
DEFAULT_REPEATS = 3
DEFAULT_HEADLINE_REPEATS = 5
RESULTS_FILENAME = 'results.jsonl'
RUN_CONFIG_FILENAME = 'sweep_config.json'
BASELINE_SUBSET = 'none'


def _log(msg):
    print(f"[perturb] {msg}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(
        prog='python -m Models.robustness.sweep',
        description='Gaussian weight-perturbation robustness sweep over the S4 '
                    'state-space matrices (inference only, never trains).')
    p.add_argument('--checkpoint-root', required=True,
                   help='dir holding <model>/best.pt and <model>/'
                        'final_report.json, e.g. /workspace/table2_run_1')
    p.add_argument('--data-dir', required=True,
                   help='preprocessed_data dir holding chunk_*.pt')
    p.add_argument('--output-dir', default=None,
                   help='default: <checkpoint-root>/robustness_results')
    p.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    p.add_argument('--subsets', nargs='+', default=list(SUBSETS),
                   choices=list(SUBSETS))
    p.add_argument('--sigmas', nargs='+', type=float, default=list(SIGMAS),
                   help='relative noise scales: eps ~ N(0, (sigma*std(W))^2)')
    p.add_argument('--repeats', type=int, default=DEFAULT_REPEATS,
                   help='independent seeds per (model, subset, sigma) for the '
                        'single-matrix subsets')
    p.add_argument('--headline-repeats', type=int,
                   default=DEFAULT_HEADLINE_REPEATS,
                   help='seeds for the joint A+B+C subset, which carries the '
                        'headline curve and its error bars')
    p.add_argument('--checkpoint-name', default='best.pt',
                   help='checkpoint filename inside <checkpoint-root>/<model>/')
    p.add_argument('--batch-size', type=int, default=512,
                   help='eval-only, so much larger than training: amortizes '
                        'the T=80 Python-loop recurrence over more samples')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--precision', choices=['fp32', 'bf16', 'fp16', 'auto'],
                   default='fp32',
                   help="fp32 (default) keeps bf16 mantissa noise — itself a "
                        "~0.4% relative weight perturbation — out of the "
                        "small-sigma end of this study")
    p.add_argument('--limit-samples', type=int, default=None,
                   help='evaluate only the first N test samples — for a fast '
                        'end-to-end smoke of the harness, NOT for real numbers')
    p.add_argument('--baseline-only', action='store_true',
                   help='run just the sigma=0 reference pass per checkpoint '
                        'and stop (the baseline-anchor sanity check)')
    p.add_argument('--dry-run', action='store_true',
                   help='preflight and print the configuration list, then exit')
    p.add_argument('--force', action='store_true',
                   help='re-run configurations already present in results.jsonl')
    p.add_argument('--no-tables', action='store_true',
                   help='skip table/plot generation (rebuild later with '
                        'python -m Models.robustness.tables)')
    return p


def config_key(model, subset, sigma, repeat):
    """Stable identity of one configuration, used for resume/skip.

    ``{sigma:.10g}`` is the same formatting ``perturb.config_seed`` uses, so a
    key and its noise seed can never disagree about which configuration they
    describe.
    """
    if sigma == 0:
        return f"{model}|{BASELINE_SUBSET}|0|0"
    return f"{model}|{subset}|{sigma:.10g}|{repeat}"


def repeats_for(subset, args):
    """-> number of seeds for this subset. The joint subset carries the
    headline curve and gets more; the per-matrix breakdown answers an ordering
    question and needs fewer."""
    from Models.robustness.perturb import JOINT_SUBSET
    return args.headline_repeats if subset == JOINT_SUBSET else args.repeats


def enumerate_configs(args):
    """-> list of (model, subset, sigma, repeat); sigma == 0 is the baseline.

    The unperturbed baseline comes first for each checkpoint so a sweep that
    dies early still leaves a usable reference behind — every delta in the
    tables is measured against it. Sigmas are ordered ascending within a
    subset so a truncated sweep degrades into a shorter curve rather than a
    gapped one.
    """
    configs = []
    for model in args.models:
        configs.append((model, BASELINE_SUBSET, 0.0, 0))
        if args.baseline_only:
            continue
        for subset in args.subsets:
            for sigma in sorted(s for s in args.sigmas if s > 0):
                for repeat in range(repeats_for(subset, args)):
                    configs.append((model, subset, float(sigma), repeat))
    return configs


def prepare_model(checkpoint_root, model_name, device, checkpoint_name='best.pt'):
    """Build the registered model, load the checkpoint, validate, and return
    (model, pristine CPU state_dict, Table 2 reference MSE, ckpt).

    Mirrors ``Models.quantization.sweep.prepare_model``; it exists separately
    only because this study needs the checkpoint filename to be a parameter
    (checkpoints are named ``<model>_best.pt`` in the committed local results
    tree and ``best.pt`` on the pod), and Phase 6's module is not modified.

    Every mismatch is fatal by design: the task is to evaluate the trained
    checkpoints as they are, so an architecture that does not line up with the
    checkpoint means something is wrong upstream, not something this script
    should paper over.
    """
    ckpt_path = os.path.join(checkpoint_root, model_name, checkpoint_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved_name = ckpt.get('model_name')
    if saved_name != model_name:
        raise RuntimeError(
            f"{ckpt_path} was saved as model_name='{saved_name}' but is being "
            f"loaded as '{model_name}'")
    model_config = ckpt.get('model_config') or {}

    model = build_model(model_name, model_config).to(device)
    load_checkpoint(ckpt_path, model)      # strict=True inside; raises on drift

    ref_mse, ref_params, ref_precision = load_reference(checkpoint_root,
                                                        model_name)
    live_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if live_params != ref_params:
        raise RuntimeError(
            f"'{model_name}' built from model_config={model_config} has "
            f"{live_params:,} trainable parameters, but its final_report.json "
            f"records {ref_params:,}. The registered config no longer matches "
            f"the trained checkpoint — stopping rather than producing "
            f"uninterpretable numbers.")

    _log(f"loaded {ckpt_path} — epoch {ckpt.get('epoch')}, "
         f"{live_params:,} params, trained in "
         f"{ckpt.get('metadata', {}).get('precision')}")
    _log(f"  Table 2 reference: mse={ref_mse:.6f} (evaluated in "
         f"{ref_precision})")

    pristine = copy.deepcopy({k: v.detach().cpu()
                              for k, v in model.state_dict().items()})
    return model, pristine, ref_mse, ckpt


def run_config(model, pristine, model_name, ref_mse, subset, sigma, repeat,
               loader, device, amp_dtype, context):
    """Restore the trained weights, perturb, evaluate once.

    Returns the JSON-serializable record for results.jsonl.
    """
    t0 = time.time()
    model.load_state_dict(pristine)
    verify_restore(model, pristine)

    perturbation, seed = None, None
    if sigma > 0:
        seed = config_seed(model_name, subset, sigma, repeat)
        perturbation = apply_perturbation(model, subset, sigma, seed)

    preds, targets = _collect_predictions(model, loader, device, amp_dtype)
    row = compute_table_row(preds, targets, model_name, model)

    return {
        'key': config_key(model_name, subset, sigma, repeat),
        'model': model_name,
        'subset': subset if sigma > 0 else BASELINE_SUBSET,
        'sigma': float(sigma),
        'repeat': int(repeat),
        'seed': seed,
        'perturbed': sigma > 0,
        'mse_fp32_table2': ref_mse,
        'delta_mse_vs_table2': row['mse'] - ref_mse,
        'wall_clock_sec': time.time() - t0,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'perturbation': perturbation,
        **{k: row[k] for k in (
            'mse', 'rmse', 'mae', 'r2', 'nz_mae', 'ny_mae',
            'per_target_mse', 'per_target_rmse', 'per_target_mae',
            'per_target_r2', 'direction_norm_mean_dev', 'params', 'space',
            'n_test_samples')},
        **context,
    }


def preflight(args, output_dir):
    """Everything that can fail cheaply, before any inference runs."""
    unknown = [m for m in args.models if m not in list_models()]
    if unknown:
        _log(f"FAILED — unknown --models entries {unknown}; "
             f"registered: {list_models()}")
        return None
    try:
        for sigma in args.sigmas:
            validate_sigma(sigma)
    except ValueError as e:
        _log(f"FAILED — bad --sigmas: {e}")
        return None
    if args.repeats < 1 or args.headline_repeats < 1:
        _log("FAILED — --repeats and --headline-repeats must be >= 1")
        return None

    for model in args.models:
        for name in (args.checkpoint_name, 'final_report.json'):
            path = os.path.join(args.checkpoint_root, model, name)
            if not os.path.isfile(path):
                _log(f"FAILED — missing {path}")
                return None

    if not os.path.isdir(args.data_dir):
        _log(f"FAILED — --data-dir {args.data_dir} is not a directory")
        return None

    device = get_device()
    amp_dtype, precision = _resolve_amp(args.precision, device)
    _log(f"precision: {precision} (requested '{args.precision}')")
    _log(f"checkpoint root: {os.path.abspath(args.checkpoint_root)}")
    _log(f"output dir:      {os.path.abspath(output_dir)}")

    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)
        probe = os.path.join(output_dir, '.write_probe')
        try:
            with open(probe, 'w') as f:
                f.write('ok')
            os.remove(probe)
        except OSError as e:
            _log(f"FAILED — output dir not writable: {e}")
            return None

    return {'device': device, 'amp_dtype': amp_dtype, 'precision': precision,
            'gpu': gpu_summary(device)}


def _fmt_hms(secs):
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or os.path.join(args.checkpoint_root,
                                                 'robustness_results')
    ctx = preflight(args, output_dir)
    if ctx is None:
        return 1

    configs = enumerate_configs(args)
    results_path = os.path.join(output_dir, RESULTS_FILENAME)
    done = set() if args.force else completed_keys(results_path)
    todo = [c for c in configs if config_key(*c) not in done]

    _log(f"{len(configs)} configuration(s) planned, {len(configs) - len(todo)} "
         f"already in {RESULTS_FILENAME}, {len(todo)} to run")
    if args.dry_run:
        for cfg in configs:
            key = config_key(*cfg)
            _log(f"  [{'SKIP' if key in done else 'RUN '}] {key}")
        return 0
    if not todo:
        _log("nothing to run")
        if not args.no_tables:
            tables.build(results_path, output_dir)
        return 0

    device, amp_dtype = ctx['device'], ctx['amp_dtype']
    loader, n_eval, n_full = build_test_loader(
        args.data_dir, args.batch_size, args.num_workers, args.limit_samples)
    if args.limit_samples is not None:
        _log(f"WARNING: --limit-samples {args.limit_samples} -> evaluating "
             f"{n_eval} of {n_full} test samples. These numbers are a harness "
             f"smoke test and are NOT comparable to Table 2.")
    else:
        _log(f"test split: {n_eval} samples, batch_size={args.batch_size}, "
             f"{len(loader)} batches")

    context = {
        'precision': ctx['precision'],
        'device': device.type,
        'gpu': ctx['gpu']['name'],
        'batch_size': args.batch_size,
        'n_eval_samples': n_eval,
        'n_test_samples_full': n_full,
        'limited': args.limit_samples is not None,
    }
    with open(os.path.join(output_dir, RUN_CONFIG_FILENAME), 'w') as f:
        json.dump({'cli_args': vars(args), 'resolved': dict(context),
                   'gpu': ctx['gpu'],
                   'timestamp': datetime.now(timezone.utc).isoformat()},
                  f, indent=2)

    # Group by checkpoint so the checkpoint is read once per model, not once
    # per configuration — 297 reads off a network volume would dominate a
    # sweep whose actual work is a 27-second forward pass.
    by_model = {}
    for cfg in todo:
        by_model.setdefault(cfg[0], []).append(cfg)

    t_sweep = time.time()
    n_ok, failures = 0, []
    for model_name, model_configs in by_model.items():
        model, pristine, ref_mse, _ = prepare_model(
            args.checkpoint_root, model_name, device, args.checkpoint_name)
        for i, (_, subset, sigma, repeat) in enumerate(model_configs, 1):
            key = config_key(model_name, subset, sigma, repeat)
            label = f"{key}  ({i}/{len(model_configs)})"
            try:
                record = run_config(
                    model, pristine, model_name, ref_mse, subset, sigma,
                    repeat, loader, device, amp_dtype, context)
            except Exception as e:      # one bad config must not lose the rest
                _log(f"[FAILED] {label} — {type(e).__name__}: {e}")
                failures.append({'key': key,
                                 'error': f"{type(e).__name__}: {e}"})
                continue
            append_result(results_path, record)
            n_ok += 1
            extra = ('' if record['perturbation'] is None else
                     f" relL2={record['perturbation']['mean_rel_l2_err']:.4g}"
                     f" ratio={record['perturbation']['mean_realized_sigma_ratio']:.4g}")
            # ASCII only in log lines: a Windows console defaults to cp1252 and
            # would raise UnicodeEncodeError on a delta sign.
            _log(f"[OK] {label} mse={record['mse']:.6f} "
                 f"d_vs_table2={record['delta_mse_vs_table2']:+.6f} "
                 f"nz_mae={record['nz_mae']:.6g} ny_mae={record['ny_mae']:.6g}"
                 f"{extra} t={record['wall_clock_sec']:.1f}s")
        del model, pristine

    _log(f"sweep finished: {n_ok} succeeded, {len(failures)} failed, "
         f"wall clock {_fmt_hms(time.time() - t_sweep)}")
    if failures:
        with open(os.path.join(output_dir, 'failures.json'), 'w') as f:
            json.dump(failures, f, indent=2)
        for fail in failures:
            _log(f"  FAILED {fail['key']}: {fail['error']}")

    if not args.no_tables:
        tables.build(results_path, output_dir)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
