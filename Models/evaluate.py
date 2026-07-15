"""Aggregate a sweep's final_report.json files into Table 2 (Phase 5).

python -m Models.evaluate --input-root <sweep-output-dir>
    [--output-csv <path>] [--output-md <path>] [--strict]

Reads reports, never checkpoints: every number in the emitted table came
from the training run's own final evaluation (the frozen compute_table_row
code in Models/common/metrics.py). No inference, no plotting, no new
metrics — deterministic and idempotent by construction.

Cross-row comparability is enforced, not assumed: all rows must share the
same seed, split ratios, data dir, and frozen-stats hashes. Mismatches are
a hard error under --strict, a WARNING plus a note in the emitted table
otherwise. The split ratios are Data.config constants (not RunConfig
fields), so they are compared via run_config.get() — absent in every report
is consistent by definition, and the target_stats_hash check pins them
transitively (target_stats.json records the ratios it was computed with).

Wall-clock per row comes from the sweep's sweep_summary.json when present
in --input-root; a report aggregated without one gets an empty cell rather
than an error, since the report schema itself (frozen in Phase 3) does not
record wall-clock.
"""
import argparse
import csv
import json
import os
import sys

# The paper's row order tells a story: baselines -> recurrence -> SSM ->
# target-aware. scripts/sweep.py imports this as its default model list.
CANONICAL_MODEL_ORDER = (
    'mlp', 'cnn', 'cnn_concat', 'gru', 's4', 's4_concat', 's4_modulate',
)

# (label, how to read it from the report) for the consistency check.
_CONSISTENCY_KEYS = (
    ('seed', lambda r: r['run_config']['seed']),
    ('train_ratio', lambda r: r['run_config'].get('train_ratio')),
    ('val_ratio', lambda r: r['run_config'].get('val_ratio')),
    ('data_dir', lambda r: r['run_config']['data_dir']),
    ('target_stats_hash', lambda r: r['metadata']['target_stats_hash']),
    ('norm_stats_hash', lambda r: r['metadata']['norm_stats_hash']),
)

CSV_COLUMNS = (
    'model', 'test_mse', 'test_mse_weighted', 'test_rmse', 'test_mae',
    'test_r2', 'mse_x_entry', 'mse_y_entry', 'mse_n_x', 'mse_n_y', 'mse_n_z',
    'nz_mae', 'ny_mae', 'params', 'wall_clock_sec', 'gpu', 'precision',
    'best_val_epoch',
)


def canonical_sort(names):
    """Canonical sweep order first, then unknown names alphabetically."""
    known = {n: i for i, n in enumerate(CANONICAL_MODEL_ORDER)}
    return sorted(names,
                  key=lambda n: (known.get(n, len(known)), n))


def discover_reports(input_root):
    """-> {subdir_name: report dict} for every subdir with a final_report.json.

    Subdirs without one are skipped silently — that's the partial-sweep case
    (only some models finished), not an error.
    """
    reports = {}
    for name in sorted(os.listdir(input_root)):
        path = os.path.join(input_root, name, 'final_report.json')
        if os.path.isfile(path):
            with open(path) as f:
                reports[name] = json.load(f)
    return reports


def check_consistency(reports):
    """-> list of human-readable mismatch strings (empty when consistent)."""
    names = canonical_sort(reports)
    ref_name = names[0]
    mismatches = []
    for label, getter in _CONSISTENCY_KEYS:
        ref = getter(reports[ref_name])
        for name in names[1:]:
            val = getter(reports[name])
            if val != ref:
                mismatches.append(
                    f"{label}: '{name}' has {val!r}, "
                    f"but '{ref_name}' has {ref!r}"
                )
    return mismatches


def _best_val_epoch(report):
    val_mse = report.get('history', {}).get('val_mse') or []
    if not val_mse:
        return None
    return 1 + min(range(len(val_mse)), key=val_mse.__getitem__)


def load_wall_clocks(input_root):
    """model -> wall_clock_sec from sweep_summary.json, {} if absent."""
    path = os.path.join(input_root, 'sweep_summary.json')
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        summary = json.load(f)
    return {name: entry.get('wall_clock_sec')
            for name, entry in summary.get('models', {}).items()}


def extract_row(report, wall_clock_sec=None):
    """One Table 2 row (dict keyed by CSV_COLUMNS) from a final report."""
    tr = report['table_row']
    pt_mse = tr['per_target_mse']
    return {
        'model': tr['model'],
        'test_mse': tr['mse'],
        'test_mse_weighted': report['test_aggregate_mse_weighted'],
        'test_rmse': tr['rmse'],
        'test_mae': tr['mae'],
        'test_r2': tr['r2'],
        'mse_x_entry': pt_mse[0],
        'mse_y_entry': pt_mse[1],
        'mse_n_x': pt_mse[2],
        'mse_n_y': pt_mse[3],
        'mse_n_z': pt_mse[4],
        'nz_mae': tr['nz_mae'],
        'ny_mae': tr['ny_mae'],
        'params': tr['params'],
        'wall_clock_sec': wall_clock_sec,
        'gpu': report['metadata']['gpu']['name'],
        'precision': report['metadata']['precision'],
        'best_val_epoch': _best_val_epoch(report),
    }


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _sig4(x):
    return f"{x:.4g}"


def write_markdown(rows, path, space, seed, warnings=()):
    """Paper-ready subset. 4 sig figs for error metrics, 2 decimals for R^2."""
    lines = [
        '# Table 2 — Main results',
        '',
        f"Metrics in **{space}** space; split seed **{seed}**. "
        f"MSE/RMSE/MAE are unweighted means over the 5 active targets.",
        '',
    ]
    for w in warnings:
        lines.append(f"> **WARNING (consistency):** {w}")
    if warnings:
        lines.append('')
    lines += [
        '| Model | MSE | RMSE | MAE | R² | n_z MAE | n_y MAE | Params |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {_sig4(r['test_mse'])} | {_sig4(r['test_rmse'])} "
            f"| {_sig4(r['test_mae'])} | {r['test_r2']:.2f} "
            f"| {_sig4(r['nz_mae'])} | {_sig4(r['ny_mae'])} "
            f"| {r['params']:,} |"
        )
    with open(path, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m Models.evaluate',
        description='Aggregate final_report.json files into Table 2 (CSV + MD).')
    parser.add_argument('--input-root', required=True,
                        help='sweep output dir; each subdir with a '
                             'final_report.json becomes one table row')
    parser.add_argument('--output-csv', default=None,
                        help='default: <input-root>/Table2.csv')
    parser.add_argument('--output-md', default=None,
                        help='default: <input-root>/Table2.md')
    parser.add_argument('--strict', action='store_true',
                        help='exit 1 on any cross-row consistency mismatch')
    args = parser.parse_args(argv)

    if not os.path.isdir(args.input_root):
        print(f"[ERROR] --input-root {args.input_root} is not a directory")
        return 1
    reports = discover_reports(args.input_root)
    if not reports:
        print(f"[ERROR] no final_report.json found under any subdirectory of "
              f"{args.input_root} — nothing to aggregate. Run the sweep first "
              f"(python -m scripts.sweep).")
        return 1

    mismatches = check_consistency(reports)
    if mismatches and args.strict:
        print("[ERROR] cross-row consistency check failed (--strict):")
        for m in mismatches:
            print(f"  {m}")
        return 1
    for m in mismatches:
        print(f"[WARNING] {m}")

    wall_clocks = load_wall_clocks(args.input_root)
    ordered = canonical_sort(reports)
    rows = [extract_row(reports[n], wall_clocks.get(n)) for n in ordered]

    first = reports[ordered[0]]
    space = first['table_row'].get('space', 'unknown')
    seed = first['run_config']['seed']

    out_csv = args.output_csv or os.path.join(args.input_root, 'Table2.csv')
    out_md = args.output_md or os.path.join(args.input_root, 'Table2.md')
    write_csv(rows, out_csv)
    write_markdown(rows, out_md, space, seed, warnings=mismatches)

    print(f"[OK] aggregated {len(rows)} model(s): {', '.join(ordered)}")
    print(f"[OK] consistency issues: {len(mismatches)}")
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_md}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
