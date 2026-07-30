"""Build the proposal's Table 4 (§7.3) and its companions from results.jsonl.

python -m Models.quantization.tables --results <dir>/results.jsonl
    [--output-dir <dir>] [--delta-ref measured|table2]

Reads records, never checkpoints — every number here was computed by the
frozen ``compute_table_row`` during the sweep, so this step is deterministic
and idempotent and can be re-run to reformat without re-evaluating anything.

Outputs, all under --output-dir:

  Table4_<model>_<granularity>.csv   4 files. Rows = matrix subset, columns =
                                     bit-width, cells = ΔMSE. The literal
                                     Table 4 template.
  Table4.md                          All four grids plus a combined view with a
                                     granularity column, and a Δn_z MAE grid
                                     (§5.1 flags n_z as the target most likely
                                     to degrade, so it is not left to the CSV).
  Table4_all.csv                     Long format: one row per configuration
                                     with every aggregate metric and both
                                     delta columns.
  quantization_per_target.csv        Supplementary — the full per-target
                                     MSE/RMSE/MAE/R² vector for every
                                     configuration, plus the weight-space
                                     quantization error that produced it.
  SUMMARY.md                         The headline: does s4_concat or
                                     s4_modulate_full degrade less under joint
                                     A+B+C quantization, per bit-width.

Two ΔMSE references exist because Table 2 was evaluated under bf16 autocast
while this sweep evaluates in fp32 (see Models/quantization/sweep.py):

  'measured' (default) — ΔMSE against this harness's own unquantized fp32 pass
      on the same checkpoint. Same arithmetic on both sides, so the delta is
      pure quantization effect.
  'table2' — ΔMSE against the published Table 2 cell. Traceable to the paper,
      but folds the bf16-vs-fp32 offset into every cell.

Both are always written to the CSVs; this flag only picks which one the
markdown grids headline. The offset between them is printed in every output so
it can never be mistaken for a quantization effect.
"""
import argparse
import csv
import json
import os
import sys

from Data.config import ACTIVE_TARGET_NAMES
from Models.quantization.fake_quant import BITS, GRANULARITIES, JOINT_SUBSET, SUBSETS

DELTA_REFS = ('measured', 'table2')
_GRAN_LABEL = {'per_tensor': 'per-tensor', 'per_channel': 'per-channel'}
_GRAN_TAG = {'per_tensor': 'PT', 'per_channel': 'PC'}


def load_records(results_path):
    """-> (list of quantized records, {model: baseline record}).

    Keyed by config key with last-write-wins, so a ``--force`` rerun that
    appended fresh lines supersedes the originals rather than double-counting.
    """
    by_key = {}
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_key[rec['key']] = rec
    records = list(by_key.values())
    baselines = {r['model']: r for r in records if not r['quantized']}
    quantized = [r for r in records if r['quantized']]
    return quantized, baselines


def reference_mse(record, baselines, delta_ref):
    """-> (reference MSE, label) for this record under the chosen convention."""
    if delta_ref == 'measured':
        base = baselines.get(record['model'])
        if base is not None:
            return base['mse'], 'measured fp32 baseline'
    return record['mse_fp32_table2'], 'Table 2 (bf16)'


def delta(record, baselines, delta_ref, metric='mse'):
    """ΔMetric for one configuration against the chosen reference."""
    if delta_ref == 'measured':
        base = baselines.get(record['model'])
        if base is not None:
            return record[metric] - base[metric]
    if metric == 'mse':
        return record['delta_mse_vs_table2']
    return None       # Table 2's report carries no per-metric reference here


def _index(records):
    """-> {(model, granularity, subset, bits): record}"""
    return {(r['model'], r['granularity'], r['subset'], r['bits']): r
            for r in records}


def _present(records, field, ordered):
    """Values of ``field`` actually present, in the canonical ``ordered`` order,
    so a partial sweep produces a smaller table rather than a table of holes."""
    seen = {r[field] for r in records}
    return [v for v in ordered if v in seen]


def _fmt(v, spec='+.4g'):
    return '—' if v is None else format(v, spec)


# --------------------------------------------------------------------------
# Table 4 grids
# --------------------------------------------------------------------------

def grid_rows(records, baselines, model, granularity, delta_ref,
              metric='mse', subsets=None, bits=None):
    """-> (subsets, bits, {(subset, bits): value}) for one Table 4 grid."""
    idx = _index(records)
    subsets = subsets or _present(
        [r for r in records if r['model'] == model
         and r['granularity'] == granularity], 'subset', SUBSETS)
    bits = bits or _present(
        [r for r in records if r['model'] == model
         and r['granularity'] == granularity], 'bits', BITS)
    cells = {}
    for subset in subsets:
        for b in bits:
            rec = idx.get((model, granularity, subset, b))
            cells[(subset, b)] = (None if rec is None
                                  else delta(rec, baselines, delta_ref, metric))
    return subsets, bits, cells


def write_grid_csv(path, subsets, bits, cells, metric_label):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([f'subset / {metric_label}'] + [f'{b}-bit' for b in bits])
        for subset in subsets:
            w.writerow([subset] + [cells[(subset, b)] for b in bits])


def _md_grid(subsets, bits, cells, first_col='Quantized matrices'):
    lines = [f"| {first_col} | " + ' | '.join(f'{b}-bit' for b in bits) + ' |',
             '|---' * (len(bits) + 1) + '|']
    for subset in subsets:
        lines.append(f"| {subset} | "
                     + ' | '.join(_fmt(cells[(subset, b)]) for b in bits) + ' |')
    return lines


def _limited_warning(records):
    """A --limit-samples run is a harness smoke test. Its tables must announce
    that on their face, or a subsampled grid gets mistaken for a real result
    the moment it leaves this directory."""
    limited = [r for r in records if r.get('limited')]
    if not limited:
        return []
    rec = limited[0]
    return ['> ## :warning: SMOKE RUN — NOT A RESULT',
            '>',
            f"> These numbers come from `--limit-samples`: "
            f"{rec['n_test_samples']:,} of {rec.get('n_test_samples_full')} "
            f"test samples ({len(limited)} of {len(records)} configurations "
            f"affected). They exercise the harness and are NOT comparable to "
            f"Table 2. Re-run without `--limit-samples` for real numbers.",
            '']


def _offset_lines(baselines):
    """The bf16-vs-fp32 gap, stated everywhere so nobody reads it as a
    quantization effect."""
    if not baselines:
        return ["> No unquantized baseline pass in results.jsonl — ΔMSE is "
                "referenced to Table 2, which folds in a bf16-vs-fp32 offset "
                "of unknown size."]

    lines = ['| Checkpoint | Table 2 MSE (bf16) | Unquantized MSE (fp32) '
             '| Offset |', '|---|---|---|---|']
    for model in sorted(baselines):
        b = baselines[model]
        lines.append(f"| {model} | {b['mse_fp32_table2']:.6f} | {b['mse']:.6f} "
                     f"| {b['mse'] - b['mse_fp32_table2']:+.6f} |")
    return lines


def write_table4_markdown(path, records, baselines, delta_ref):
    models = _present(records, 'model', sorted({r['model'] for r in records}))
    grans = _present(records, 'granularity', GRANULARITIES)
    _, ref_label = (reference_mse(records[0], baselines, delta_ref)
                    if records else (None, 'n/a'))

    lines = ['# Table 4 — Quantization sensitivity (ΔMSE)', '']
    lines += _limited_warning(records)
    lines += [
        f"Each cell is **ΔMSE = MSE_quantized − MSE_reference**, where the "
        f"reference is the **{ref_label}** for that checkpoint. Lower is "
        f"better; 0 means quantization was free. Metrics are in physical "
        f"space, over the full test split, computed by the same "
        f"`compute_table_row` that produced Table 2.",
        '',
        'Fake quantization is asymmetric uniform (min/max range, integer '
        'zero-point). `A` = `log_A_real` + `A_imag`, `B` = `B`, '
        '`C` = `C_real` + `C_imag`; real and imaginary components are '
        'quantized independently. `log_dt` is never quantized.',
        '',
        '## Baseline verification',
        '',
    ]
    lines += _offset_lines(baselines)
    lines += ['', '## ΔMSE by checkpoint and granularity', '']

    for model in models:
        for gran in grans:
            subsets, bits, cells = grid_rows(
                records, baselines, model, gran, delta_ref)
            if not subsets or not bits:
                continue
            lines += [f"### {model} — {_GRAN_LABEL.get(gran, gran)}", '']
            lines += _md_grid(subsets, bits, cells)
            lines.append('')

    # Combined view: granularity as a column so the comparison the proposal's
    # §5.1 note asks for is one glance rather than four scrolls.
    lines += ['## Combined view — granularity side by side', '',
              'ΔMSE; **PT** = per-tensor, **PC** = per-channel.', '']
    for model in models:
        bits = _present([r for r in records if r['model'] == model],
                        'bits', BITS)
        subsets = _present([r for r in records if r['model'] == model],
                           'subset', SUBSETS)
        if not bits or not subsets:
            continue
        header = ['Quantized matrices', 'Gran.'] + [f'{b}-bit' for b in bits]
        lines += [f"### {model}", '',
                  '| ' + ' | '.join(header) + ' |',
                  '|---' * len(header) + '|']
        for subset in subsets:
            for gran in grans:
                _, _, cells = grid_rows(records, baselines, model, gran,
                                        delta_ref, subsets=[subset], bits=bits)
                lines.append(f"| {subset} | {_GRAN_TAG.get(gran, gran)} | "
                             + ' | '.join(_fmt(cells[(subset, b)])
                                          for b in bits) + ' |')
        lines.append('')

    # n_z is the target §5.1 predicts suffers most — surfaced here rather than
    # buried, while the full per-target vectors stay in the supplementary CSV.
    lines += ['## Δn_z MAE — the target §5.1 predicts degrades most', '',
              'Δn_z MAE against the same reference. n_z is bounded away from '
              'zero (|n_z| ≥ 0.099) and determined jointly with n_x/n_y, so it '
              'is the most sensitive direction component.', '']
    for model in models:
        for gran in grans:
            subsets, bits, cells = grid_rows(
                records, baselines, model, gran, delta_ref, metric='nz_mae')
            if not subsets or not bits:
                continue
            lines += [f"### {model} — {_GRAN_LABEL.get(gran, gran)}", '']
            lines += _md_grid(subsets, bits, cells)
            lines.append('')

    with open(path, 'w', newline='\n', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------
# long-format CSVs
# --------------------------------------------------------------------------

_LONG_COLUMNS = (
    'model', 'granularity', 'subset', 'bits', 'mse', 'rmse', 'mae', 'r2',
    'nz_mae', 'ny_mae', 'mse_fp32_table2', 'mse_baseline_measured',
    'delta_mse_vs_table2', 'delta_mse_vs_measured', 'delta_mse_pct_vs_measured',
    'delta_nz_mae_vs_measured', 'delta_ny_mae_vs_measured',
    'weight_mean_rel_l2_err', 'weight_max_rel_l2_err', 'weight_max_abs_err',
    'n_params_quantized', 'n_tensors_quantized',
    'direction_norm_mean_dev', 'params', 'space', 'n_test_samples',
    'wall_clock_sec', 'precision', 'gpu',
)


def _long_row(rec, baselines):
    base = baselines.get(rec['model'])
    base_mse = base['mse'] if base else None
    q = rec.get('quant') or {}
    d_meas = delta(rec, baselines, 'measured')
    return {
        'model': rec['model'],
        'granularity': rec['granularity'],
        'subset': rec['subset'],
        'bits': rec['bits'],
        'mse': rec['mse'], 'rmse': rec['rmse'], 'mae': rec['mae'],
        'r2': rec['r2'], 'nz_mae': rec['nz_mae'], 'ny_mae': rec['ny_mae'],
        'mse_fp32_table2': rec['mse_fp32_table2'],
        'mse_baseline_measured': base_mse,
        'delta_mse_vs_table2': rec['delta_mse_vs_table2'],
        'delta_mse_vs_measured': d_meas,
        'delta_mse_pct_vs_measured': (
            100.0 * d_meas / base_mse
            if base_mse not in (None, 0) and d_meas is not None else None),
        'delta_nz_mae_vs_measured': delta(rec, baselines, 'measured', 'nz_mae'),
        'delta_ny_mae_vs_measured': delta(rec, baselines, 'measured', 'ny_mae'),
        'weight_mean_rel_l2_err': q.get('mean_rel_l2_err'),
        'weight_max_rel_l2_err': q.get('max_rel_l2_err'),
        'weight_max_abs_err': q.get('max_abs_err'),
        'n_params_quantized': q.get('n_params_quantized'),
        'n_tensors_quantized': q.get('n_tensors'),
        'direction_norm_mean_dev': rec['direction_norm_mean_dev'],
        'params': rec['params'], 'space': rec['space'],
        'n_test_samples': rec['n_test_samples'],
        'wall_clock_sec': rec['wall_clock_sec'],
        'precision': rec.get('precision'), 'gpu': rec.get('gpu'),
    }


def _sort_key(rec):
    def rank(value, ordered):
        return ordered.index(value) if value in ordered else len(ordered)
    return (rec['model'], rank(rec['granularity'], list(GRANULARITIES)),
            rank(rec['subset'], list(SUBSETS)),
            -(rec['bits'] or 0))


def write_long_csv(path, records, baselines):
    rows = [_long_row(r, baselines) for r in sorted(records, key=_sort_key)]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=_LONG_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_per_target_csv(path, records, baselines):
    """Full per-target breakdown — the §5.1 requirement that per-target
    degradation be inspectable, not just aggregate MSE. Baseline rows are
    included (bits empty) so every delta can be recomputed from this file
    alone."""
    metrics = ('mse', 'rmse', 'mae', 'r2')
    columns = (['model', 'granularity', 'subset', 'bits', 'quantized',
                'agg_mse', 'agg_mae', 'agg_r2']
               + [f'{m}_{t}' for m in metrics for t in ACTIVE_TARGET_NAMES]
               + [f'delta_mae_{t}' for t in ACTIVE_TARGET_NAMES]
               + [f'delta_mse_{t}' for t in ACTIVE_TARGET_NAMES])
    ordered = (sorted(baselines.values(), key=lambda r: r['model'])
               + sorted(records, key=_sort_key))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for rec in ordered:
            base = baselines.get(rec['model'])
            row = {
                'model': rec['model'], 'granularity': rec['granularity'],
                'subset': rec['subset'], 'bits': rec['bits'],
                'quantized': rec['quantized'],
                'agg_mse': rec['mse'], 'agg_mae': rec['mae'],
                'agg_r2': rec['r2'],
            }
            for m in metrics:
                for i, t in enumerate(ACTIVE_TARGET_NAMES):
                    row[f'{m}_{t}'] = rec[f'per_target_{m}'][i]
            for i, t in enumerate(ACTIVE_TARGET_NAMES):
                for m in ('mae', 'mse'):
                    row[f'delta_{m}_{t}'] = (
                        None if base is None
                        else rec[f'per_target_{m}'][i]
                        - base[f'per_target_{m}'][i])
            w.writerow(row)


# --------------------------------------------------------------------------
# headline summary
# --------------------------------------------------------------------------

def _joint_comparison(records, baselines, delta_ref, granularity):
    """-> list of per-bit-width comparison dicts at the A+B+C subset."""
    idx = _index(records)
    models = sorted({r['model'] for r in records})
    out = []
    for b in _present(records, 'bits', BITS):
        entry = {'bits': b, 'per_model': {}}
        for model in models:
            rec = idx.get((model, granularity, JOINT_SUBSET, b))
            if rec is None:
                continue
            entry['per_model'][model] = {
                'mse': rec['mse'],
                'delta': delta(rec, baselines, delta_ref),
                'nz_mae': rec['nz_mae'],
                'delta_nz': delta(rec, baselines, delta_ref, 'nz_mae'),
            }
        if entry['per_model']:
            out.append(entry)
    return out


def _winner(entry, field):
    """-> (model with the smallest value of ``field``, or None if tied/absent)."""
    scored = [(v[field], m) for m, v in entry['per_model'].items()
              if v.get(field) is not None]
    if len(scored) < 2:
        return None
    scored.sort()
    return None if scored[0][0] == scored[1][0] else scored[0][1]


def write_summary_markdown(path, records, baselines, delta_ref):
    models = sorted({r['model'] for r in records})
    _, ref_label = (reference_mse(records[0], baselines, delta_ref)
                    if records else (None, 'n/a'))
    n_samples = records[0]['n_test_samples'] if records else 0
    precision = records[0].get('precision') if records else '?'
    total_wall = sum(r['wall_clock_sec'] for r in records) + sum(
        b['wall_clock_sec'] for b in baselines.values())

    lines = ['# Phase 6 summary — quantization robustness', '']
    lines += _limited_warning(records)
    lines += [
        f"{len(records)} quantized evaluations + {len(baselines)} unquantized "
        f"reference passes over the full {n_samples:,}-sample test split, "
        f"inference only, evaluated in **{precision}**. Total wall clock "
        f"{total_wall / 3600:.2f} h. ΔMSE is referenced to the "
        f"**{ref_label}**.",
        '',
        '## Baseline verification',
        '',
    ]
    lines += _offset_lines(baselines)
    lines += [
        '',
        '## Headline — joint A+B+C quantization',
        '',
        'The realistic deployment case: all three state-space matrices '
        'quantized simultaneously at the same bit-width. **ΔMSE** measures '
        'robustness (how much each checkpoint loses); **MSE** measures what '
        'you actually deploy. They can disagree, and where they do, that is '
        'the finding.',
        '',
    ]

    for gran in _present(records, 'granularity', GRANULARITIES):
        comparison = _joint_comparison(records, baselines, delta_ref, gran)
        if not comparison:
            continue
        header = (['Bits'] + [f'{m} ΔMSE' for m in models]
                  + [f'{m} MSE' for m in models]
                  + ['More robust (ΔMSE)', 'Better absolute MSE'])
        lines += [f"### {_GRAN_LABEL.get(gran, gran)}", '',
                  '| ' + ' | '.join(header) + ' |',
                  '|---' * len(header) + '|']
        for entry in comparison:
            per = entry['per_model']
            cells = (
                [str(entry['bits'])]
                + [_fmt(per.get(m, {}).get('delta')) for m in models]
                + [_fmt(per.get(m, {}).get('mse'), '.4f') for m in models]
                + [_winner(entry, 'delta') or 'tie / n/a',
                   _winner(entry, 'mse') or 'tie / n/a']
            )
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')

    # Per-subset sensitivity ranking, averaged over bit-widths — answers
    # "which matrix is the fragile one" without reading the whole grid.
    lines += ['## Which matrix is most sensitive', '',
              'Mean ΔMSE across bit-widths, per subset (higher = more '
              'sensitive to quantization).', '']
    header = ['Quantized matrices'] + [f'{m} ({_GRAN_TAG.get(g, g)})'
                                       for m in models
                                       for g in _present(records,
                                                         'granularity',
                                                         GRANULARITIES)]
    lines += ['| ' + ' | '.join(header) + ' |', '|---' * len(header) + '|']
    for subset in _present(records, 'subset', SUBSETS):
        cells = [subset]
        for model in models:
            for gran in _present(records, 'granularity', GRANULARITIES):
                vals = [delta(r, baselines, delta_ref) for r in records
                        if r['model'] == model and r['granularity'] == gran
                        and r['subset'] == subset]
                vals = [v for v in vals if v is not None]
                cells.append(_fmt(sum(vals) / len(vals) if vals else None))
        lines += ['| ' + ' | '.join(cells) + ' |']
    lines.append('')

    lines += ['## Reproduce', '',
              '```', 'python -m Models.quantization.sweep \\',
              '    --checkpoint-root /workspace/table2_run_1 \\',
              '    --data-dir /workspace/repo/preprocessed_data', '```', '',
              'Full per-target degradation: `quantization_per_target.csv`. '
              'All aggregates and both delta conventions: `Table4_all.csv`. '
              'Raw records including per-tensor weight-space quantization '
              'error: `results.jsonl`.']

    with open(path, 'w', newline='\n', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------

def build(results_path, output_dir, delta_ref='measured'):
    """Write every table from results.jsonl. -> list of written paths."""
    if delta_ref not in DELTA_REFS:
        raise ValueError(f"delta_ref must be one of {DELTA_REFS}")
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"no results file at {results_path}")
    records, baselines = load_records(results_path)
    if not records:
        print(f"[tables] no quantized records in {results_path}; "
              f"nothing to tabulate")
        return []
    if delta_ref == 'measured' and len(baselines) < len(
            {r['model'] for r in records}):
        print(f"[tables] WARNING: unquantized baseline missing for some "
              f"checkpoints; those rows fall back to the Table 2 reference")

    os.makedirs(output_dir, exist_ok=True)
    written = []

    for model in sorted({r['model'] for r in records}):
        for gran in _present([r for r in records if r['model'] == model],
                             'granularity', GRANULARITIES):
            subsets, bits, cells = grid_rows(
                records, baselines, model, gran, delta_ref)
            path = os.path.join(output_dir, f'Table4_{model}_{gran}.csv')
            write_grid_csv(path, subsets, bits, cells, 'delta_mse')
            written.append(path)

    for name, fn in (
        ('Table4.md', write_table4_markdown),
        ('Table4_all.csv', write_long_csv),
        ('quantization_per_target.csv', write_per_target_csv),
        ('SUMMARY.md', write_summary_markdown),
    ):
        path = os.path.join(output_dir, name)
        if fn in (write_table4_markdown, write_summary_markdown):
            fn(path, records, baselines, delta_ref)
        else:
            fn(path, records, baselines)
        written.append(path)

    for path in written:
        print(f"[tables] wrote {path}")
    return written


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='python -m Models.quantization.tables',
        description='Build Table 4 and companions from a sweep results.jsonl.')
    p.add_argument('--results', required=True, help='path to results.jsonl')
    p.add_argument('--output-dir', default=None,
                   help='default: the directory holding --results')
    p.add_argument('--delta-ref', choices=list(DELTA_REFS), default='measured',
                   help="which reference the markdown grids headline "
                        "(default: measured — this harness's own unquantized "
                        "fp32 pass, so the delta is pure quantization effect)")
    args = p.parse_args(argv)
    output_dir = args.output_dir or os.path.dirname(
        os.path.abspath(args.results))
    build(args.results, output_dir, args.delta_ref)
    return 0


if __name__ == '__main__':
    sys.exit(main())
