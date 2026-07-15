"""Single-machine sweep orchestrator (Phase 5) — what a RunPod session runs.

python -m scripts.sweep --output-root <persistent-dir> [--models m1 m2 ...]
    [--batch-size 64] [--max-epochs 50] [--lr 1e-3] [--lambda-dir 0.01]
    [--precision auto] [--num-workers 4] [--resume]
    [--data-dir <path>] [--seed <int>] [--dry-run]

Runs the seven-model comparison sequentially with identical flags, one
``python -m Models.train`` subprocess per model. Subprocess, not import:
GPU memory frees fully between models, one model's crash (or segfault)
can't kill the sweep, and each failure is recorded to failure.json while
the remaining models still run. No parallelism, no DDP — one GPU, one
model at a time.

Resume is model-scoped: a model dir holding latest.pt but no
final_report.json is resumed (the engine restores optimizer/scheduler/RNG
state itself); a dir holding final_report.json is skipped. There is no
sweep-level resume checkpoint. The --resume flag is accepted for
explicitness but this per-model behavior is automatic either way — never
restart-from-scratch over a dir that holds trained state.

On RunPod, --output-root must point at the persistent network volume; the
sweep cannot verify that, so it prints the resolved path first thing —
a misconfiguration should be visible within the first 5 seconds of pod
time. GPU type is recorded as metadata, never auto-selected on; resuming
on a different GPU than a checkpoint was created on logs a warning and
continues.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from Data import compute_stats_hashes
from Data.config import NORM_STATS_PATH, SEED, TARGET_STATS_PATH
from Models.evaluate import CANONICAL_MODEL_ORDER

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_MODELS = CANONICAL_MODEL_ORDER


def _log(msg):
    print(f"[sweep] {msg}", flush=True)


def _git_info():
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=_REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=_REPO_ROOT, text=True).strip())
        return commit, dirty
    except Exception:
        return 'unknown', None


def build_parser():
    p = argparse.ArgumentParser(
        prog='python -m scripts.sweep',
        description='Sequential, resumable training sweep over all models.')
    p.add_argument('--output-root', required=True,
                   help='one subdir per model is created here — point at '
                        'persistent storage on RunPod')
    p.add_argument('--models', nargs='+', default=list(EXPECTED_MODELS),
                   help=f'subset/order override (default: {" ".join(EXPECTED_MODELS)})')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lambda-dir', type=float, default=0.01)
    p.add_argument('--precision', choices=['auto', 'bf16', 'fp16', 'fp32'],
                   default='auto')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--resume', action='store_true',
                   help='accepted for explicitness; per-model resume from '
                        'latest.pt is automatic regardless')
    p.add_argument('--data-dir',
                   default=os.path.join(_REPO_ROOT, 'preprocessed_data'))
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--dry-run', action='store_true',
                   help='run preflight and print the planned commands '
                        'without invoking Models.train or writing anything')
    return p


def _resolved_precision(requested, device):
    if requested != 'auto':
        return requested
    import torch
    from Models.common.device import get_amp_dtype
    return {torch.bfloat16: 'bf16', torch.float16: 'fp16',
            None: 'fp32'}[get_amp_dtype(device)]


def preflight(args):
    """Fast, cheap, always runs (dry-run included). Returns context dict or
    raises SystemExit-style by returning None after printing the error."""
    commit, dirty = _git_info()
    _log(f"git commit {commit}" + (' (DIRTY working tree)' if dirty else ''))
    _log(f"output root: {os.path.abspath(args.output_root)} "
         f"(must be persistent storage on RunPod)")
    _log(f"data dir:    {os.path.abspath(args.data_dir)}")

    from Models.common.device import get_device, gpu_summary
    device = get_device()
    gpu = gpu_summary(device)

    ok, msg = compute_stats_hashes.verify()
    if not ok:
        _log(f"PREFLIGHT FAILED — {msg}")
        return None
    _log(msg)

    for path in (TARGET_STATS_PATH, NORM_STATS_PATH):
        try:
            with open(path) as f:
                json.load(f)
        except Exception as e:
            _log(f"PREFLIGHT FAILED — cannot read {path}: {e}")
            return None

    import Models.models_import_all  # noqa: F401 — populates registry
    from Models.common.registry import list_models
    missing = [m for m in EXPECTED_MODELS if m not in list_models()]
    if missing:
        _log(f"PREFLIGHT FAILED — models missing from registry: {missing}. "
             f"Phase 4 registers them in Models/*/; check that "
             f"Models/models_import_all.py imports every architecture module.")
        return None
    unknown = [m for m in args.models if m not in list_models()]
    if unknown:
        _log(f"PREFLIGHT FAILED — unknown --models entries: {unknown}. "
             f"Registered: {list_models()}")
        return None

    precision = _resolved_precision(args.precision, device)
    _log(f"precision: {precision} (requested '{args.precision}')")

    ctx = {
        'git_commit': commit,
        'git_dirty': dirty,
        'gpu': gpu,
        'precision_resolved': precision,
        'seed': args.seed,
        'cli_args': vars(args),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    if not args.dry_run:
        os.makedirs(args.output_root, exist_ok=True)
        probe = os.path.join(args.output_root, '.write_probe')
        try:
            with open(probe, 'w') as f:
                f.write('ok')
            os.remove(probe)
        except OSError as e:
            _log(f"PREFLIGHT FAILED — output root not writable: {e}")
            return None
        with open(os.path.join(args.output_root, 'sweep_config.json'), 'w') as f:
            json.dump(ctx, f, indent=2)
        _log(f"wrote {os.path.join(args.output_root, 'sweep_config.json')}")
    return ctx


def _model_mode(ckpt_dir):
    """-> 'skip' | 'resume' | 'train' from what the model dir already holds."""
    if os.path.isfile(os.path.join(ckpt_dir, 'final_report.json')):
        return 'skip'
    if os.path.isfile(os.path.join(ckpt_dir, 'latest.pt')):
        return 'resume'
    return 'train'


def _train_command(model, ckpt_dir, args, resume):
    cmd = [
        sys.executable, '-m', 'Models.train',
        '--model', model,
        '--checkpoint-dir', ckpt_dir,
        '--batch-size', str(args.batch_size),
        '--max-epochs', str(args.max_epochs),
        '--lr', str(args.lr),
        '--lambda-dir', str(args.lambda_dir),
        '--precision', args.precision,
        '--num-workers', str(args.num_workers),
        '--seed', str(args.seed),
        '--data-dir', args.data_dir,
    ]
    if resume:
        cmd.append('--resume')
    return cmd


def _warn_if_gpu_changed(latest_path, current_gpu, model):
    """Resuming on a different GPU is allowed; make it visible, not fatal."""
    try:
        import torch
        ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)
        started_on = ckpt['metadata']['gpu']['name']
        if started_on != current_gpu['name']:
            _log(f"WARNING: '{model}' started on GPU '{started_on}' but is "
                 f"resuming on '{current_gpu['name']}' — continuing; the "
                 f"table will report the GPU of the final run.")
    except Exception as e:
        _log(f"WARNING: could not inspect {latest_path} for GPU metadata "
             f"({e}); the train subprocess will validate the checkpoint.")


def _run_model(model, ckpt_dir, cmd):
    """Stream the subprocess through with a [model] prefix. -> (exit, secs)."""
    env = dict(os.environ,
               PYTHONUNBUFFERED='1',
               PYTHONPATH=_REPO_ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''))
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=_REPO_ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            errors='replace')
    for line in proc.stdout:
        print(f"[{model}] {line}", end='', flush=True)
    proc.wait()
    return proc.returncode, time.time() - t0


def _fmt_hms(secs):
    if secs is None:
        return '-'
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _print_summary_table(entries, output_root):
    header = (f"{'model':<14}{'status':<10}{'wall':>10}{'val MSE':>12}"
              f"{'test MSE':>12}{'params':>12}  gpu")
    print('\n' + header)
    print('-' * len(header))
    for model, e in entries.items():
        val = test = params = gpu = '-'
        report_path = os.path.join(output_root, model, 'final_report.json')
        if os.path.isfile(report_path):
            with open(report_path) as f:
                report = json.load(f)
            val = f"{report['best_val_mse']:.4f}"
            test = f"{report['table_row']['mse']:.4f}"
            params = f"{report['table_row']['params']:,}"
            gpu = report['metadata']['gpu']['name']
        print(f"{model:<14}{e['status']:<10}{_fmt_hms(e['wall_clock_sec']):>10}"
              f"{val:>12}{test:>12}{params:>12}  {gpu}")
    print()


def main(argv=None):
    args = build_parser().parse_args(argv)
    ctx = preflight(args)
    if ctx is None:
        return 1

    if args.dry_run:
        _log("DRY RUN — commands that would run, in order:")
        for model in args.models:
            ckpt_dir = os.path.join(args.output_root, model)
            mode = _model_mode(ckpt_dir) if os.path.isdir(ckpt_dir) else 'train'
            if mode == 'skip':
                _log(f"  [SKIP]  {model}: final_report.json already present")
                continue
            cmd = _train_command(model, ckpt_dir, args, resume=(mode == 'resume'))
            _log(f"  [{mode.upper():<6}] {' '.join(cmd)}")
        return 0

    # Wall-clocks of models finished by an earlier session survive into this
    # session's summary, so an interrupted-then-resumed sweep still reports
    # every row.
    prior = {}
    summary_path = os.path.join(args.output_root, 'sweep_summary.json')
    if os.path.isfile(summary_path):
        try:
            with open(summary_path) as f:
                prior = json.load(f).get('models', {})
        except Exception:
            prior = {}

    entries = {}
    for model in args.models:
        ckpt_dir = os.path.join(args.output_root, model)
        mode = _model_mode(ckpt_dir)
        report_path = os.path.join(ckpt_dir, 'final_report.json')

        if mode == 'skip':
            _log(f"[SKIP] {model} — {report_path} already present")
            entries[model] = {
                'status': 'skipped',
                'wall_clock_sec': prior.get(model, {}).get('wall_clock_sec'),
                'final_report': report_path,
            }
            continue

        if mode == 'resume':
            _warn_if_gpu_changed(os.path.join(ckpt_dir, 'latest.pt'),
                                 ctx['gpu'], model)
        _log(f"[{'RESUME' if mode == 'resume' else 'TRAIN'}] {model} "
             f"-> {ckpt_dir}")
        cmd = _train_command(model, ckpt_dir, args, resume=(mode == 'resume'))
        exit_code, secs = _run_model(model, ckpt_dir, cmd)

        if exit_code != 0 or not os.path.isfile(report_path):
            reason = (f"exit code {exit_code}" if exit_code != 0 else
                      "exit 0 but no final_report.json produced")
            _log(f"[FAILED] {model} after {_fmt_hms(secs)} — {reason}; "
                 f"continuing with remaining models")
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(os.path.join(ckpt_dir, 'failure.json'), 'w') as f:
                json.dump({
                    'exit_code': exit_code,
                    'reason': reason,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'wall_clock_sec': secs,
                    'command': cmd,
                }, f, indent=2)
            entries[model] = {'status': 'failed', 'wall_clock_sec': secs,
                              'final_report': None}
            continue

        with open(report_path) as f:
            report = json.load(f)
        pt = report['table_row']['per_target_mse']
        _log(f"[OK] {model} in {_fmt_hms(secs)} — test mse="
             f"{report['table_row']['mse']:.6f} per-target="
             f"[{', '.join(f'{v:.4g}' for v in pt)}]")
        entries[model] = {'status': 'success', 'wall_clock_sec': secs,
                          'final_report': report_path}

    summary = {
        'git_commit': ctx['git_commit'],
        'gpu': ctx['gpu'],
        'precision': ctx['precision_resolved'],
        'seed': args.seed,
        'finished_at': datetime.now(timezone.utc).isoformat(),
        'models': entries,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    _log(f"wrote {summary_path}")
    _print_summary_table(entries, args.output_root)

    failed = [m for m, e in entries.items() if e['status'] == 'failed']
    if failed:
        _log(f"sweep finished with failures: {failed}")
        return 1
    _log("sweep finished — every model succeeded or was already complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())
