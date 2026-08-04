"""Turn Phase 7's results.jsonl into tables, curves, and a verdict.

python -m Models.robustness.tables --results <dir>/results.jsonl \
    [--output-dir <dir>] [--no-plots]

Pure post-processing: reads only the sweep's own output, runs no model, and is
idempotent — rebuilding over an existing output dir overwrites cleanly. Kept
separate from the sweep so the analysis can be reworked without re-running
297 forward passes.

The proposal asks one question — "is the slope of MSE degradation smaller for
the modulated model than for the plain SSM baseline?" — and this module
answers it with two statistics, because either one alone is misleading.

1. LOG-LOG SLOPE of dMSE against sigma. For a small perturbation, a second-
   order expansion of the loss around a trained minimum (where the gradient
   vanishes) gives

       dMSE ~ 0.5 * sigma_eff^2 * tr(H)

   so log10(dMSE) should be linear in log10(sigma) with slope near 2. The
   slope is therefore mostly a check that the perturbation is in the
   quadratic regime, and the INTERCEPT — log10 of the curvature scale — is
   what actually differs between models. Both are reported, with the fit's
   R^2, over the pre-saturation region only (sigma values with dMSE > 0 and
   MSE below SATURATION_FACTOR x baseline). Past saturation the model is
   producing garbage and the curve flattens for reasons that have nothing to
   do with robustness.

2. SIGMA@2x — the sigma at which mean MSE reaches twice the model's own
   baseline, log-interpolated between the bracketing sweep points. This is
   the scale-free version of the same question and is the headline ranking
   statistic: larger sigma@2x = more perturbation tolerated before a fixed
   relative quality loss. It needs no fit region and no regime assumption,
   and because it is referenced to each model's OWN baseline it does not
   confuse "starts out better" with "degrades more slowly".

Reading MSE here carries the same caveat as Phase 6: the unweighted aggregate
is ~98.7% x_entry, so it tracks that one target and is nearly blind to the
three direction components. The per-target CSV is where n_y and n_z
degradation is visible.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

from Data.config import ACTIVE_TARGET_NAMES

RAW_CSV = 'robustness_raw.csv'
SUMMARY_CSV = 'robustness_summary.csv'
PER_TARGET_CSV = 'robustness_per_target.csv'
SLOPES_CSV = 'robustness_slopes.csv'
SUMMARY_MD = 'SUMMARY.md'

# Above this multiple of baseline MSE the model is not merely degraded, it is
# broken, and its curve flattens out for numerical reasons rather than
# robustness ones. Fitting through that region would drag every slope toward
# zero and reward whichever model saturates soonest.
SATURATION_FACTOR = 10.0
MIN_FIT_POINTS = 3
DOUBLING_FACTOR = 2.0

JOINT_SUBSET = 'A+B+C'
MATRIX_SUBSETS = ('A', 'B', 'C')


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_records(results_path):
    """-> (perturbed records, {model: baseline record}).

    Later lines win on duplicate keys, so a ``--force`` re-run supersedes the
    original rather than double-counting it. A malformed trailing line (a pod
    killed mid-write) is skipped rather than fatal.
    """
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"no results file at {results_path}")
    by_key = {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                by_key[rec['key']] = rec
            except (json.JSONDecodeError, KeyError):
                continue

    perturbed, baselines = [], {}
    for rec in by_key.values():
        if rec.get('perturbed'):
            perturbed.append(rec)
        else:
            baselines[rec['model']] = rec
    perturbed.sort(key=lambda r: (r['model'], r['subset'], r['sigma'],
                                  r['repeat']))
    return perturbed, baselines


def baseline_mse(baselines, model):
    """-> the model's own measured sigma=0 MSE, or None if it was not run.

    Never falls back to the Table 2 (bf16) number: mixing arithmetic between
    the reference and the perturbed runs would put a fixed offset into every
    delta, and at the small-sigma end that offset is the same size as the
    effect being measured.
    """
    rec = baselines.get(model)
    return rec['mse'] if rec else None


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    """Sample std (n-1). Zero for a single repeat, which is honest: one draw
    carries no information about spread."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def aggregate(perturbed, baselines):
    """-> list of per-(model, subset, sigma) summary dicts, sorted.

    Raw per-repeat records are never discarded — they are written verbatim to
    robustness_raw.csv — but every curve, fit and plot works off these means.
    """
    groups = defaultdict(list)
    for rec in perturbed:
        groups[(rec['model'], rec['subset'], rec['sigma'])].append(rec)

    rows = []
    for (model, subset, sigma), recs in groups.items():
        mses = [r['mse'] for r in recs]
        base = baseline_mse(baselines, model)
        row = {
            'model': model, 'subset': subset, 'sigma': sigma,
            'n_repeats': len(recs),
            'mse_mean': _mean(mses), 'mse_std': _std(mses),
            'mse_min': min(mses), 'mse_max': max(mses),
            'mse_baseline': base,
            'delta_mse_mean': (_mean(mses) - base) if base is not None else None,
            'mse_ratio': (_mean(mses) / base) if base else None,
            'mae_mean': _mean([r['mae'] for r in recs]),
            'r2_mean': _mean([r['r2'] for r in recs]),
            'nz_mae_mean': _mean([r['nz_mae'] for r in recs]),
            'ny_mae_mean': _mean([r['ny_mae'] for r in recs]),
            'rel_l2_err_mean': _mean(
                [r['perturbation']['mean_rel_l2_err'] for r in recs]),
            'realized_sigma_ratio_mean': _mean(
                [r['perturbation']['mean_realized_sigma_ratio'] for r in recs]),
            'seeds': ','.join(str(r['repeat']) for r in sorted(
                recs, key=lambda x: x['repeat'])),
        }
        for i, name in enumerate(ACTIVE_TARGET_NAMES):
            row[f'mse_{name}'] = _mean([r['per_target_mse'][i] for r in recs])
            row[f'mae_{name}'] = _mean([r['per_target_mae'][i] for r in recs])
        rows.append(row)

    rows.sort(key=lambda r: (r['model'], r['subset'], r['sigma']))
    return rows


# --------------------------------------------------------------------------
# the two headline statistics
# --------------------------------------------------------------------------

def fit_slope(sigmas, deltas, base_mse, mses,
              saturation_factor=SATURATION_FACTOR):
    """Least-squares fit of log10(dMSE) on log10(sigma) over the usable region.

    Returns {slope, intercept, r2, n_points, sigma_lo, sigma_hi, excluded} —
    or the same dict with slope None if fewer than MIN_FIT_POINTS survive
    filtering, which is a real outcome (a matrix so insensitive that no sigma
    in the grid moved it) and is reported rather than papered over.

    Excluded: non-positive dMSE (noise happened to help, or the effect is
    below evaluation jitter — undefined in log space) and saturated points
    (MSE >= saturation_factor x baseline, where the curve flattens for
    numerical rather than robustness reasons).
    """
    xs, ys, excluded = [], [], []
    limit = saturation_factor * base_mse if base_mse else float('inf')
    for sigma, delta, mse in zip(sigmas, deltas, mses):
        if sigma <= 0:
            continue
        if delta is None or delta <= 0:
            excluded.append((sigma, 'delta<=0'))
            continue
        if mse >= limit:
            excluded.append((sigma, 'saturated'))
            continue
        xs.append(math.log10(sigma))
        ys.append(math.log10(delta))

    out = {'slope': None, 'intercept': None, 'r2': None, 'n_points': len(xs),
           'sigma_lo': None, 'sigma_hi': None,
           'excluded': ';'.join(f"{s:g}:{why}" for s, why in excluded)}
    if len(xs) < MIN_FIT_POINTS:
        return out

    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return out
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    out.update({
        'slope': slope, 'intercept': intercept,
        'r2': (1 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        'n_points': n,
        'sigma_lo': 10 ** min(xs), 'sigma_hi': 10 ** max(xs),
    })
    return out


def sigma_at_ratio(sigmas, mses, base_mse, ratio=DOUBLING_FACTOR):
    """-> the sigma at which mean MSE first reaches ``ratio`` x baseline.

    Log-log linear interpolation between the two bracketing sweep points,
    which is the right geometry for a curve that is a power law in this
    region. Returns None if the sweep never got there (a lower bound — the
    caller should say ">= max sigma" rather than pretend a number).
    """
    if not base_mse:
        return None
    target = ratio * base_mse
    pts = sorted(zip(sigmas, mses))
    prev = None
    for sigma, mse in pts:
        if sigma <= 0 or mse <= 0:
            continue
        if mse >= target:
            if prev is None:
                # Already over threshold at the smallest sigma tested: the
                # crossing is below the grid, so report that bound.
                return sigma
            s0, m0 = prev
            if m0 <= 0 or m0 == mse:
                return sigma
            # solve for sigma on the log-log segment through (s0,m0)-(sigma,mse)
            t = ((math.log10(target) - math.log10(m0))
                 / (math.log10(mse) - math.log10(m0)))
            return 10 ** (math.log10(s0)
                          + t * (math.log10(sigma) - math.log10(s0)))
        prev = (sigma, mse)
    return None


def slope_rows(summary, baselines):
    """-> one row per (model, subset) holding both headline statistics."""
    groups = defaultdict(list)
    for row in summary:
        groups[(row['model'], row['subset'])].append(row)

    rows = []
    for (model, subset), rs in groups.items():
        rs = sorted(rs, key=lambda r: r['sigma'])
        base = baseline_mse(baselines, model)
        sigmas = [r['sigma'] for r in rs]
        mses = [r['mse_mean'] for r in rs]
        deltas = [r['delta_mse_mean'] for r in rs]
        fit = fit_slope(sigmas, deltas, base, mses)
        s2x = sigma_at_ratio(sigmas, mses, base)
        rows.append({
            'model': model, 'subset': subset,
            'mse_baseline': base,
            'slope': fit['slope'], 'intercept': fit['intercept'],
            'fit_r2': fit['r2'], 'fit_n_points': fit['n_points'],
            'fit_sigma_lo': fit['sigma_lo'], 'fit_sigma_hi': fit['sigma_hi'],
            'fit_excluded': fit['excluded'],
            'sigma_at_2x': s2x,
            'sigma_at_2x_censored': s2x is None,
            'sigma_max_tested': max(sigmas) if sigmas else None,
            'mse_at_max_sigma': mses[-1] if mses else None,
            'mse_ratio_at_max_sigma': (mses[-1] / base) if base and mses
                                      else None,
        })
    rows.sort(key=lambda r: (r['subset'] != JOINT_SUBSET, r['subset'],
                             r['model']))
    return rows


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------

def _write_csv(path, rows, columns):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


_RAW_COLUMNS = (
    ['key', 'model', 'subset', 'sigma', 'repeat', 'seed', 'perturbed',
     'mse', 'rmse', 'mae', 'r2', 'nz_mae', 'ny_mae', 'mse_baseline',
     'delta_mse_vs_baseline', 'mse_fp32_table2', 'delta_mse_vs_table2']
    + [f'mse_{n}' for n in ACTIVE_TARGET_NAMES]
    + [f'mae_{n}' for n in ACTIVE_TARGET_NAMES]
    + ['weight_mean_rel_l2_err', 'weight_max_rel_l2_err',
       'realized_sigma_ratio', 'n_params_perturbed', 'n_tensors_perturbed',
       'direction_norm_mean_dev', 'params', 'n_test_samples',
       'wall_clock_sec', 'precision', 'gpu'])


def write_raw_csv(records, baselines, output_dir):
    """Every (model, subset, sigma, repeat) evaluation, including baselines.

    This is the preservation-of-raw-data output: the summary aggregates
    repeats away, and someone re-analyzing this study should not have to parse
    JSONL to get them back.
    """
    rows = []
    ordered = sorted(list(baselines.values()) + list(records),
                     key=lambda r: (r['model'], r['subset'], r['sigma'],
                                    r['repeat']))
    for rec in ordered:
        base = baseline_mse(baselines, rec['model'])
        p = rec.get('perturbation') or {}
        row = {k: rec.get(k) for k in (
            'key', 'model', 'subset', 'sigma', 'repeat', 'seed', 'perturbed',
            'mse', 'rmse', 'mae', 'r2', 'nz_mae', 'ny_mae', 'mse_fp32_table2',
            'delta_mse_vs_table2', 'direction_norm_mean_dev', 'params',
            'n_test_samples', 'wall_clock_sec', 'precision', 'gpu')}
        row['mse_baseline'] = base
        row['delta_mse_vs_baseline'] = (rec['mse'] - base) if base is not None \
            else None
        row['weight_mean_rel_l2_err'] = p.get('mean_rel_l2_err')
        row['weight_max_rel_l2_err'] = p.get('max_rel_l2_err')
        row['realized_sigma_ratio'] = p.get('mean_realized_sigma_ratio')
        row['n_params_perturbed'] = p.get('n_params_perturbed')
        row['n_tensors_perturbed'] = p.get('n_tensors')
        for i, name in enumerate(ACTIVE_TARGET_NAMES):
            row[f'mse_{name}'] = rec['per_target_mse'][i]
            row[f'mae_{name}'] = rec['per_target_mae'][i]
        rows.append(row)
    return _write_csv(os.path.join(output_dir, RAW_CSV), rows,
                      list(_RAW_COLUMNS))


_SUMMARY_COLUMNS = (
    ['model', 'subset', 'sigma', 'n_repeats', 'mse_mean', 'mse_std',
     'mse_min', 'mse_max', 'mse_baseline', 'delta_mse_mean', 'mse_ratio',
     'mae_mean', 'r2_mean', 'nz_mae_mean', 'ny_mae_mean',
     'rel_l2_err_mean', 'realized_sigma_ratio_mean', 'seeds']
    + [f'mse_{n}' for n in ACTIVE_TARGET_NAMES]
    + [f'mae_{n}' for n in ACTIVE_TARGET_NAMES])


def write_summary_csv(summary, output_dir):
    return _write_csv(os.path.join(output_dir, SUMMARY_CSV), summary,
                      list(_SUMMARY_COLUMNS))


def write_per_target_csv(summary, baselines, output_dir):
    """Per-target MAE degradation as a percentage of each model's own
    baseline, mirroring Phase 6's quantization_per_target.csv.

    Necessary because aggregate MSE is ~98.7% x_entry: a configuration can
    look nearly free in MSE while n_y and n_z have measurably degraded, and
    the proposal's targets are not equally weighted in anyone's judgment of
    whether the model still works.
    """
    rows = []
    for row in summary:
        base_rec = baselines.get(row['model'])
        out = {'model': row['model'], 'subset': row['subset'],
               'sigma': row['sigma'], 'n_repeats': row['n_repeats'],
               'agg_mse': row['mse_mean'], 'agg_r2': row['r2_mean']}
        for i, name in enumerate(ACTIVE_TARGET_NAMES):
            mae = row[f'mae_{name}']
            out[f'mae_{name}'] = mae
            out[f'mse_{name}'] = row[f'mse_{name}']
            if base_rec:
                b = base_rec['per_target_mae'][i]
                out[f'delta_mae_pct_{name}'] = (100.0 * (mae - b) / b) \
                    if b else None
            else:
                out[f'delta_mae_pct_{name}'] = None
        rows.append(out)
    columns = (['model', 'subset', 'sigma', 'n_repeats', 'agg_mse', 'agg_r2']
               + [f'{p}_{n}' for n in ACTIVE_TARGET_NAMES
                  for p in ('mse', 'mae', 'delta_mae_pct')])
    return _write_csv(os.path.join(output_dir, PER_TARGET_CSV), rows, columns)


_SLOPE_COLUMNS = ['model', 'subset', 'mse_baseline', 'slope', 'intercept',
                  'fit_r2', 'fit_n_points', 'fit_sigma_lo', 'fit_sigma_hi',
                  'sigma_at_2x', 'sigma_at_2x_censored', 'sigma_max_tested',
                  'mse_at_max_sigma', 'mse_ratio_at_max_sigma',
                  'fit_excluded']


def write_slopes_csv(slopes, output_dir):
    return _write_csv(os.path.join(output_dir, SLOPES_CSV), slopes,
                      list(_SLOPE_COLUMNS))


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------

def _fmt(v, spec='.4g'):
    if v is None:
        return 'n/a'
    return format(v, spec)


def _verdict_lines(slopes, baselines, models):
    """Prose answer to the proposal's actual question, from the joint-subset
    fits. Written to say what the numbers show, including when that is 'no
    difference' or 'the opposite of the hypothesis'."""
    joint = {r['model']: r for r in slopes if r['subset'] == JOINT_SUBSET}
    ordered = [m for m in models if m in joint]
    if len(ordered) < 2:
        return ["Not enough models with a joint A+B+C curve to compare "
                "degradation slopes."]

    lines = []
    ranked = sorted(
        (m for m in ordered if joint[m]['sigma_at_2x'] is not None),
        key=lambda m: joint[m]['sigma_at_2x'], reverse=True)
    censored = [m for m in ordered if joint[m]['sigma_at_2x'] is None]

    if ranked:
        ordering = ' > '.join(
            '`{}` ({})'.format(m, _fmt(joint[m]['sigma_at_2x']))
            for m in ranked)
        tail = '.'
        if censored:
            tail = ('; {} never reached 2x within the tested sigma range '
                    '(more robust than any ranked model on this statistic).'
                    ).format(', '.join('`{}`'.format(m) for m in censored))
        lines.append('**By sigma@2x (larger = more robust), the ordering is '
                     + ordering + tail + '**')
        lines.append('')
        lines.append('On this statistic the most robust ranked checkpoint is '
                     '`{}`.'.format(ranked[0]))
    elif censored:
        lines.append("No model reached 2x its baseline MSE within the tested "
                     "sigma range, so sigma@2x is censored for all of them and "
                     "the slope fit below carries the comparison.")

    lines.append('')
    fitted = [m for m in ordered if joint[m]['slope'] is not None]
    if len(fitted) >= 2:
        by_slope = sorted(fitted, key=lambda m: joint[m]['slope'])
        by_inter = sorted(fitted, key=lambda m: joint[m]['intercept'])
        lines.append(
            "Log-log slopes: "
            + ', '.join(f"`{m}` {_fmt(joint[m]['slope'], '.3f')} "
                        f"(R2 {_fmt(joint[m]['fit_r2'], '.3f')})"
                        for m in by_slope) + ".")
        lines.append('')
        lines.append(
            "Intercepts (log10 dMSE at sigma=1, extrapolated — the curvature "
            "scale, and the number that actually separates the models when "
            "slopes agree): "
            + ', '.join(f"`{m}` {_fmt(joint[m]['intercept'], '.3f')}"
                        for m in by_inter) + ". Lower is more robust.")
        spread = (max(joint[m]['slope'] for m in fitted)
                  - min(joint[m]['slope'] for m in fitted))
        lines.append('')
        if spread < 0.25:
            lines.append(
                f"The slopes agree to within {spread:.3f}, consistent with the "
                f"quadratic dMSE ~ sigma^2 prediction holding for every "
                f"checkpoint. The models therefore differ in the CONSTANT of "
                f"degradation, not in its exponent: none of them degrades "
                f"with a fundamentally different shape, and 'smaller slope' is "
                f"not the axis on which they separate.")
        else:
            lines.append(
                f"The slopes differ by {spread:.3f}, so the checkpoints do not "
                f"share a single degradation exponent; read the per-model fits "
                f"rather than a single summary number.")
    return lines


def write_summary_markdown(summary, slopes, baselines, records, output_dir,
                           plot_files=()):
    models = []
    for row in summary:
        if row['model'] not in models:
            models.append(row['model'])

    n_evals = len(records) + len(baselines)
    total_sec = sum(r.get('wall_clock_sec') or 0
                    for r in list(records) + list(baselines.values()))
    limited = any(r.get('limited') for r in records)
    n_samples = records[0]['n_eval_samples'] if records else 0

    L = ['# Phase 7 summary — weight-perturbation robustness', '']
    L.append(f"{len(records)} perturbed evaluations + {len(baselines)} "
             f"unperturbed sigma=0 reference passes over "
             f"{n_samples:,} test samples, inference only, evaluated in "
             f"**fp32**. Total wall clock "
             f"{total_sec / 3600:.2f} h across {n_evals} evaluations.")
    if limited:
        L.append('')
        L.append('> **WARNING — this run used `--limit-samples`.** The numbers '
                 'below are a harness smoke test on a truncated test split and '
                 'are NOT comparable to Table 2 or to Phase 6.')
    L += ['', 'Perturbation is **relative**: `eps ~ N(0, (sigma*std(W))^2)` per '
          'tensor, so `sigma` is a dimensionless fraction of each tensor\'s own '
          'spread rather than the absolute `N(0, sigma^2)` the proposal writes. '
          'The S4 matrices differ in scale by three orders of magnitude '
          '(`A_imag` spans [0, 97.4], `C` has std ~0.125), so an absolute sigma '
          'would rank the matrices by how small their weights happen to be. '
          'See `perturb.py` for the full argument.', '']

    # Baselines
    L += ['## Baseline verification', '',
          '| Checkpoint | Table 2 MSE (bf16) | sigma=0 MSE (fp32) | Offset |',
          '|---|---|---|---|']
    for model in models:
        rec = baselines.get(model)
        if not rec:
            L.append(f"| {model} | n/a | **not run** | n/a |")
            continue
        L.append(f"| {model} | {rec['mse_fp32_table2']:.6f} | "
                 f"{rec['mse']:.6f} | {rec['delta_mse_vs_table2']:+.6f} |")

    # Headline curve
    L += ['', '## Headline — joint A+B+C perturbation', '',
          'Mean MSE over independent seeds, +/- sample std. `x` is the ratio to '
          'each checkpoint\'s own sigma=0 baseline.', '']
    joint_rows = [r for r in summary if r['subset'] == JOINT_SUBSET]
    if joint_rows:
        sigmas = sorted({r['sigma'] for r in joint_rows})
        L.append('| sigma | ' + ' | '.join(models) + ' |')
        L.append('|---' * (len(models) + 1) + '|')
        for sigma in sigmas:
            cells = []
            for model in models:
                hit = [r for r in joint_rows
                       if r['model'] == model and r['sigma'] == sigma]
                if not hit:
                    cells.append('n/a')
                    continue
                r = hit[0]
                ratio = f" ({r['mse_ratio']:.2f}x)" if r['mse_ratio'] else ''
                cells.append(f"{r['mse_mean']:.4g} +/- {r['mse_std']:.3g}"
                             f"{ratio}")
            L.append(f"| {sigma:g} | " + ' | '.join(cells) + ' |')
    else:
        L.append('_No A+B+C configurations in this run._')

    # The verdict
    L += ['', "## Does modulation degrade more slowly?", '',
          "The proposal's hypothesis (Section 7.4): if target-aware modulation "
          "improves robustness, the slope of MSE degradation should be smaller "
          "than the plain SSM baseline's.", '']
    L += _verdict_lines(slopes, baselines, models)

    L += ['', '### Fitted degradation statistics', '',
          '| Checkpoint | Matrices | slope | intercept | R2 | pts | sigma@2x |',
          '|---|---|---|---|---|---|---|']
    for r in slopes:
        s2x = ('> ' + _fmt(r['sigma_max_tested'], 'g')) \
            if r['sigma_at_2x'] is None else _fmt(r['sigma_at_2x'], '.4g')
        L.append(f"| {r['model']} | {r['subset']} | "
                 f"{_fmt(r['slope'], '.3f')} | {_fmt(r['intercept'], '.3f')} | "
                 f"{_fmt(r['fit_r2'], '.3f')} | {r['fit_n_points']} | "
                 f"{s2x} |")

    # Per-matrix comparison against Phase 6
    L += ['', '## Which matrix is most sensitive', '',
          'Mean MSE ratio to baseline at each sigma, averaged over checkpoints. '
          "Phase 6 found A dramatically more sensitive than C, and C more than "
          'B, under low-bit quantization (2-bit per-tensor dMSE: A +7314, '
          'C +270, B +4.1). Whether that ordering is intrinsic to the '
          'architecture or specific to grid rounding is exactly what this '
          'section tests.', '']
    matrix_rows = [r for r in summary if r['subset'] in MATRIX_SUBSETS]
    if matrix_rows:
        sigmas = sorted({r['sigma'] for r in matrix_rows})
        L.append('| Perturbed matrix | ' +
                 ' | '.join(f"sigma={s:g}" for s in sigmas) + ' |')
        L.append('|---' * (len(sigmas) + 1) + '|')
        for subset in MATRIX_SUBSETS:
            cells = []
            for sigma in sigmas:
                vals = [r['mse_ratio'] for r in matrix_rows
                        if r['subset'] == subset and r['sigma'] == sigma
                        and r['mse_ratio'] is not None]
                cells.append(f"{_mean(vals):.4g}x" if vals else 'n/a')
            L.append(f"| {subset} | " + ' | '.join(cells) + ' |')
        L.append('')
        L += _matrix_ordering_lines(matrix_rows)
    else:
        L.append('_No single-matrix configurations in this run._')

    L += ['', '## Reproduce', '', '```',
          'python -m Models.robustness.sweep \\',
          '    --checkpoint-root /workspace/table2_run_1 \\',
          '    --data-dir /workspace/repo/preprocessed_data', '```', '']
    L.append(f'Raw per-repeat data (nothing aggregated away): `{RAW_CSV}`. '
             f'Aggregated curves: `{SUMMARY_CSV}`. Per-target degradation: '
             f'`{PER_TARGET_CSV}`. Fits: `{SLOPES_CSV}`. Full records '
             f'including per-tensor realized noise: `results.jsonl`.')
    if plot_files:
        L.append('')
        L.append('Plots: ' + ', '.join(f'`{os.path.basename(p)}`'
                                       for p in plot_files) + '.')

    path = os.path.join(output_dir, SUMMARY_MD)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return path


def _matrix_ordering_lines(matrix_rows):
    """State whether the per-matrix sensitivity ordering matches Phase 6's."""
    sigmas = sorted({r['sigma'] for r in matrix_rows})
    if not sigmas:
        return []
    largest = sigmas[-1]
    means = {}
    for subset in MATRIX_SUBSETS:
        vals = [r['mse_ratio'] for r in matrix_rows
                if r['subset'] == subset and r['sigma'] == largest
                and r['mse_ratio'] is not None]
        if vals:
            means[subset] = _mean(vals)
    if len(means) < 2:
        return []
    order = sorted(means, key=means.get, reverse=True)
    line = (f"At the largest sigma tested ({largest:g}) the sensitivity "
            f"ordering is **{' > '.join(order)}** "
            f"({', '.join(f'{k} {means[k]:.3g}x' for k in order)}).")
    if order == ['A', 'C', 'B']:
        line += (" This reproduces Phase 6's quantization ordering exactly, so "
                 "A's dominance is a property of the model's dependence on the "
                 "state-transition matrix, not an artifact of grid rounding.")
    elif order[0] == 'A':
        line += (" A is the most sensitive matrix here as it was under "
                 "quantization, but the ordering of B and C differs from "
                 "Phase 6's A > C > B.")
    else:
        line += (" This does NOT reproduce Phase 6's A > C > B ordering — A's "
                 "outsized quantization sensitivity does not carry over to "
                 "continuous Gaussian noise, which points to a rounding-"
                 "specific mechanism rather than an intrinsic one.")
    return [line]


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build(results_path, output_dir=None, make_plots=True):
    """-> list of written file paths. Idempotent."""
    output_dir = output_dir or os.path.dirname(os.path.abspath(results_path))
    os.makedirs(output_dir, exist_ok=True)

    records, baselines = load_records(results_path)
    if not records and not baselines:
        raise ValueError(f"{results_path} contains no usable records")
    summary = aggregate(records, baselines)
    slopes = slope_rows(summary, baselines)

    written = [
        write_raw_csv(records, baselines, output_dir),
        write_summary_csv(summary, output_dir),
        write_per_target_csv(summary, baselines, output_dir),
        write_slopes_csv(slopes, output_dir),
    ]

    plot_files = []
    if make_plots and summary:
        # Plot failure must not lose the tables — matplotlib is the one
        # dependency here that can be absent or misconfigured on a pod.
        try:
            from Models.robustness import plots
            plot_files = plots.build_all(summary, slopes, baselines, output_dir)
            written += plot_files
        except Exception as e:                             # noqa: BLE001
            print(f"[perturb] WARNING: plotting failed "
                  f"({type(e).__name__}: {e}); tables were still written")

    written.append(write_summary_markdown(summary, slopes, baselines, records,
                                          output_dir, plot_files))
    return written


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m Models.robustness.tables')
    p.add_argument('--results', required=True, help='path to results.jsonl')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args(argv)
    for path in build(args.results, args.output_dir,
                      make_plots=not args.no_plots):
        print(f"wrote {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
