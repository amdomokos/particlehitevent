"""Phase 6 quantization sweep — 64 inference-only evaluations, no training.

python -m Models.quantization.sweep --checkpoint-root /workspace/table2_run_1 \
    --data-dir /workspace/repo/preprocessed_data \
    [--output-dir <root>/quantization_results] \
    [--models s4_concat s4_modulate_full] \
    [--granularities per_tensor per_channel] [--subsets A B C A+B+C] \
    [--bits 8 6 4 2] [--batch-size 512] [--num-workers 4]
    [--precision fp32] [--limit-samples N] [--baseline-only] [--dry-run]
    [--force] [--no-tables]

Crosses 2 checkpoints x 2 granularities x 4 matrix subsets x 4 bit-widths =
64 configurations, plus one unquantized pass per checkpoint (the FP32
reference, see below). Every configuration is a single forward pass over the
test split: no backward pass, no optimizer, no gradients, and nothing is ever
written back to a checkpoint.

FP32 reference. Table 2 was produced under bf16 autocast, but this sweep
evaluates in fp32 by default: bf16 has an 8-bit mantissa, so bf16 compute
noise would partly mask 8-bit *weight* quantization and add variance to
exactly the signal being measured. That leaves an unknown bf16-vs-fp32 offset
between Table 2's MSE and this harness's arithmetic, so each checkpoint also
gets one unquantized fp32 pass. Both deltas are reported —
``delta_mse_vs_table2`` and ``delta_mse_vs_measured`` — so the offset is
visible rather than silently folded into every cell. The unquantized pass
doubles as the correctness check that this harness reproduces the trained
model's known result.

Weight restore. The pristine state_dict is deep-copied to CPU once per
checkpoint and reloaded before each configuration, rather than re-reading
best.pt 32 times from the network volume. Because the whole study rests on
each configuration starting from the trained weights and not from the previous
configuration's quantized ones, the restore is verified for bit-exactness
against the pristine copy on every configuration — 320k tensor comparisons is
nothing against a multi-minute forward pass, and a silent restore failure
would make all 64 numbers compound garbage in a way the tables could not
reveal.

Resumability. Each configuration appends one JSON line to results.jsonl the
moment it finishes, and a rerun skips keys already present (``--force``
re-runs them). Per-config cost is a couple of minutes, so this is cheap
insurance against losing a partial sweep, not a real checkpointing scheme.
"""
import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader, Subset

import Models.models_import_all  # noqa: F401 — populates registry
from Models.common.device import get_amp_dtype, get_device, gpu_summary
# _collect_predictions is the training engine's own evaluation loop (eval mode,
# no_grad, 6->5 active-target selection, full-set accumulation rather than a
# running mean). Reusing it — private name and all — is deliberate: it is the
# single source of truth for "how the test set is evaluated", and
# reimplementing it here is precisely how a quantization table stops being
# comparable to Table 2.
from Models.common.engine import _collect_predictions, load_checkpoint
from Models.common.metrics import compute_table_row
from Models.common.registry import build_model, list_models
from Models.common.seeding import worker_init_fn
from Models.quantization import tables
from Models.quantization.fake_quant import (
    BITS,
    GRANULARITIES,
    SUBSETS,
    apply_fake_quant,
)

DEFAULT_MODELS = ('s4_concat', 's4_modulate_full')
RESULTS_FILENAME = 'results.jsonl'
RUN_CONFIG_FILENAME = 'sweep_config.json'


def _log(msg):
    print(f"[quant] {msg}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(
        prog='python -m Models.quantization.sweep',
        description='Post-training fake-quantization sweep over the S4 '
                    'state-space matrices (inference only, never trains).')
    p.add_argument('--checkpoint-root', required=True,
                   help='dir holding <model>/best.pt and <model>/'
                        'final_report.json, e.g. /workspace/table2_run_1')
    p.add_argument('--data-dir', required=True,
                   help='preprocessed_data dir holding chunk_*.pt')
    p.add_argument('--output-dir', default=None,
                   help='default: <checkpoint-root>/quantization_results')
    p.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    p.add_argument('--granularities', nargs='+', default=list(GRANULARITIES),
                   choices=list(GRANULARITIES))
    p.add_argument('--subsets', nargs='+', default=list(SUBSETS),
                   choices=list(SUBSETS))
    p.add_argument('--bits', nargs='+', type=int, default=list(BITS))
    p.add_argument('--batch-size', type=int, default=512,
                   help='eval-only, so much larger than training: amortizes '
                        'the T=80 Python-loop recurrence over more samples')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--precision', choices=['fp32', 'bf16', 'fp16', 'auto'],
                   default='fp32',
                   help="fp32 (default) keeps bf16 mantissa noise out of a "
                        "weight-precision study; 'bf16' reproduces Table 2's "
                        "own arithmetic")
    p.add_argument('--limit-samples', type=int, default=None,
                   help='evaluate only the first N test samples — for a fast '
                        'end-to-end smoke of the harness, NOT for real numbers')
    p.add_argument('--baseline-only', action='store_true',
                   help='run just the unquantized reference pass per '
                        'checkpoint and stop (the step-3 sanity check)')
    p.add_argument('--dry-run', action='store_true',
                   help='preflight and print the configuration list, then exit')
    p.add_argument('--force', action='store_true',
                   help='re-run configurations already present in results.jsonl')
    p.add_argument('--no-tables', action='store_true',
                   help='skip table generation (rebuild later with '
                        'python -m Models.quantization.tables)')
    return p


def config_key(model, granularity, subset, bits):
    """Stable identity of one configuration, used for resume/skip."""
    if bits is None:
        return f"{model}|baseline|none|fp32"
    return f"{model}|{granularity}|{subset}|{bits}"


def enumerate_configs(args):
    """-> list of (model, granularity, subset, bits); bits None = baseline.

    The unquantized baseline comes first for each checkpoint so a sweep that
    dies early still leaves a usable reference behind.
    """
    configs = []
    for model in args.models:
        configs.append((model, None, None, None))
        if args.baseline_only:
            continue
        for granularity in args.granularities:
            for subset in args.subsets:
                for bits in args.bits:
                    configs.append((model, granularity, subset, bits))
    return configs


def _resolve_amp(precision, device):
    """-> (autocast dtype or None, resolved name). Never returns 'auto'."""
    if precision == 'fp32':
        return None, 'fp32'
    if precision == 'bf16':
        return torch.bfloat16, 'bf16'
    if precision == 'fp16':
        if device.type == 'cpu':
            raise RuntimeError('fp16 autocast is not supported on CPU')
        return torch.float16, 'fp16'
    dtype = get_amp_dtype(device)
    return dtype, {torch.bfloat16: 'bf16', torch.float16: 'fp16',
                   None: 'fp32'}[dtype]


def load_reference(checkpoint_root, model):
    """-> (fp32 reference MSE from Table 2, params) out of final_report.json.

    This is the number the proposal's Table 4 measures ΔMSE against. Read from
    the report, never recomputed, so it is byte-identical to the published
    Table 2 cell.
    """
    path = os.path.join(checkpoint_root, model, 'final_report.json')
    with open(path) as f:
        report = json.load(f)
    row = report['table_row']
    if row['model'] != model:
        raise RuntimeError(
            f"{path} reports model '{row['model']}' but lives in the '{model}' "
            f"directory — refusing to guess which checkpoint it describes")
    return row['mse'], row['params'], report['metadata'].get('precision')


def prepare_model(checkpoint_root, model_name, device):
    """Build the registered model, load best.pt into it, validate, and return
    (model, pristine CPU state_dict).

    Every mismatch here is fatal by design: the task is to evaluate the
    trained checkpoints as they are, so an architecture that does not line up
    with the checkpoint means something is wrong upstream, not something this
    script should paper over.
    """
    ckpt_path = os.path.join(checkpoint_root, model_name, 'best.pt')
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

    ref_mse, ref_params, ref_precision = load_reference(checkpoint_root, model_name)
    live_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if live_params != ref_params:
        raise RuntimeError(
            f"'{model_name}' built from model_config={model_config} has "
            f"{live_params:,} trainable parameters, but its final_report.json "
            f"records {ref_params:,}. The registered config no longer matches "
            f"the trained checkpoint — stopping rather than producing "
            f"uninterpretable numbers.")

    _log(f"loaded {ckpt_path} — epoch {ckpt.get('epoch')}, "
         f"best_val_mse {ckpt.get('best_val_mse'):.6f}, "
         f"{live_params:,} params, trained in "
         f"{ckpt.get('metadata', {}).get('precision')}")
    _log(f"  Table 2 reference: mse={ref_mse:.6f} (evaluated in "
         f"{ref_precision})")

    pristine = copy.deepcopy({k: v.detach().cpu()
                              for k, v in model.state_dict().items()})
    return model, pristine, ref_mse, ckpt


def build_test_loader(data_dir, batch_size, num_workers, limit_samples=None):
    """The same test split Table 2 used: PixelClusterDataset(split='test'),
    canonical chunk split, shuffle=False."""
    from Data.dataset_v2 import PixelClusterDataset
    dataset = PixelClusterDataset(data_dir, 'test')
    n_full = len(dataset)
    if n_full == 0:
        raise ValueError(f"test split is empty for data_dir={data_dir}")
    if limit_samples is not None:
        dataset = Subset(dataset, range(min(limit_samples, n_full)))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    return loader, len(dataset), n_full


def completed_keys(results_path):
    """-> set of config keys already recorded. Malformed trailing lines (a pod
    killed mid-write) are ignored rather than fatal."""
    keys = set()
    if not os.path.isfile(results_path):
        return keys
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)['key'])
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def append_result(results_path, record):
    with open(results_path, 'a') as f:
        f.write(json.dumps(record) + '\n')
        f.flush()
        os.fsync(f.fileno())


def verify_restore(model, pristine):
    """Assert the live model is bit-identical to the pristine trained weights.

    Guards the one assumption every configuration depends on: that quantization
    is applied to the TRAINED weights, not to the previous configuration's
    already-quantized ones. Compounding would inflate every cell of Table 4
    monotonically down the sweep order — a failure mode the tables themselves
    would look plausible under, which is exactly why it is checked rather than
    assumed.
    """
    live = model.state_dict()
    for key, want in pristine.items():
        got = live[key].detach().cpu()
        if not torch.equal(got, want):
            raise RuntimeError(
                f"weight restore failed for '{key}': the model did not return "
                f"to its trained values (max abs diff "
                f"{(got.float() - want.float()).abs().max().item():.3e}). "
                f"Refusing to evaluate — results would compound across "
                f"configurations.")


def run_config(model, pristine, model_name, ref_mse, granularity, subset, bits,
               loader, device, amp_dtype, context):
    """Restore FP32 weights, apply the fake quantization, evaluate once.

    Returns the JSON-serializable record for results.jsonl.
    """
    t0 = time.time()
    model.load_state_dict(pristine)
    verify_restore(model, pristine)

    quant = None
    if bits is not None:
        quant = apply_fake_quant(model, subset, bits, granularity)

    preds, targets = _collect_predictions(model, loader, device, amp_dtype)
    row = compute_table_row(preds, targets, model_name, model)

    record = {
        'key': config_key(model_name, granularity, subset, bits),
        'model': model_name,
        'granularity': granularity or 'baseline',
        'subset': subset or 'none',
        'bits': bits,
        'quantized': bits is not None,
        'mse_fp32_table2': ref_mse,
        'delta_mse_vs_table2': row['mse'] - ref_mse,
        'wall_clock_sec': time.time() - t0,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'quant': quant,
        **{k: row[k] for k in (
            'mse', 'rmse', 'mae', 'r2', 'nz_mae', 'ny_mae',
            'per_target_mse', 'per_target_rmse', 'per_target_mae',
            'per_target_r2', 'direction_norm_mean_dev', 'params', 'space',
            'n_test_samples')},
        **context,
    }
    return record


def preflight(args, output_dir):
    """Everything that can fail cheaply, before any inference runs."""
    unknown = [m for m in args.models if m not in list_models()]
    if unknown:
        _log(f"FAILED — unknown --models entries {unknown}; "
             f"registered: {list_models()}")
        return None
    bad_bits = [b for b in args.bits if not 2 <= b <= 16]
    if bad_bits:
        _log(f"FAILED — --bits entries outside [2, 16]: {bad_bits}")
        return None

    for model in args.models:
        for name in ('best.pt', 'final_report.json'):
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
                                                 'quantization_results')
    ctx = preflight(args, output_dir)
    if ctx is None:
        return 1

    configs = enumerate_configs(args)
    results_path = os.path.join(output_dir, RESULTS_FILENAME)
    done = set() if args.force else completed_keys(results_path)
    todo = [c for c in configs
            if config_key(c[0], c[1], c[2], c[3]) not in done]

    _log(f"{len(configs)} configuration(s) planned, {len(configs) - len(todo)} "
         f"already in {RESULTS_FILENAME}, {len(todo)} to run")
    if args.dry_run:
        for model, gran, subset, bits in configs:
            key = config_key(model, gran, subset, bits)
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
        json.dump({'cli_args': vars(args), 'resolved': {
            k: v for k, v in context.items()}, 'gpu': ctx['gpu'],
            'timestamp': datetime.now(timezone.utc).isoformat()}, f, indent=2)

    # Group by checkpoint so best.pt is read once per model, not once per config.
    by_model = {}
    for cfg in todo:
        by_model.setdefault(cfg[0], []).append(cfg)

    t_sweep = time.time()
    n_ok, failures = 0, []
    for model_name, model_configs in by_model.items():
        model, pristine, ref_mse, _ = prepare_model(
            args.checkpoint_root, model_name, device)
        for i, (_, gran, subset, bits) in enumerate(model_configs, 1):
            key = config_key(model_name, gran, subset, bits)
            label = f"{key}  ({i}/{len(model_configs)})"
            try:
                record = run_config(
                    model, pristine, model_name, ref_mse, gran, subset, bits,
                    loader, device, amp_dtype, context)
            except Exception as e:      # one bad config must not lose the rest
                _log(f"[FAILED] {label} — {type(e).__name__}: {e}")
                failures.append({'key': key, 'error': f"{type(e).__name__}: {e}"})
                continue
            append_result(results_path, record)
            n_ok += 1
            extra = ('' if record['quant'] is None else
                     f" relL2={record['quant']['mean_rel_l2_err']:.4g}")
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
