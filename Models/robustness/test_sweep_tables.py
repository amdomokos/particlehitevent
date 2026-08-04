"""Tests for the Phase 7 sweep driver and its table/plot builders.

Run from the repo root:  pytest Models/robustness/test_sweep_tables.py

No checkpoints and no dataset: config enumeration is pure, and the table tests
run against synthetic results.jsonl fixtures written into tmp_path. The
analysis is where a robustness study can most easily fool itself — a slope fit
that silently includes saturated points, or a sigma@2x that interpolates on
the wrong geometry, produces a number that looks fine and means nothing — so
the estimators are pinned against curves whose answers are known in closed
form.
"""
import json
import math

import pytest

from Models.robustness import tables
from Models.robustness.perturb import SIGMAS, SUBSETS
from Models.robustness.sweep import (
    BASELINE_SUBSET,
    DEFAULT_MODELS,
    build_parser,
    config_key,
    enumerate_configs,
    repeats_for,
)

TARGETS = ('x_entry', 'y_entry', 'n_x', 'n_y', 'n_z')


def parse(argv):
    return build_parser().parse_args(
        ['--checkpoint-root', 'root', '--data-dir', 'data'] + argv)


# ------------------------------------------------------- config enumeration ---

def test_default_plan_is_the_designed_297_evaluations():
    args = parse([])
    configs = enumerate_configs(args)
    n_models, n_sigmas = len(DEFAULT_MODELS), len(SIGMAS)
    joint = n_models * n_sigmas * args.headline_repeats          # 105
    single = n_models * 3 * n_sigmas * args.repeats              # 189
    assert len(configs) == joint + single + n_models             # + baselines
    assert len(configs) == 297


def test_config_keys_are_unique():
    configs = enumerate_configs(parse([]))
    keys = [config_key(*c) for c in configs]
    assert len(set(keys)) == len(keys)


def test_baseline_comes_first_for_each_checkpoint():
    """A sweep that dies early must still leave the reference every delta is
    measured against."""
    configs = enumerate_configs(parse([]))
    seen = set()
    for model, subset, sigma, _ in configs:
        if model not in seen:
            assert sigma == 0 and subset == BASELINE_SUBSET, \
                f'{model} ran a perturbed config before its baseline'
            seen.add(model)
    assert seen == set(DEFAULT_MODELS)


def test_baseline_key_is_subset_and_repeat_independent():
    """One baseline per checkpoint, not one per subset: it is the same
    unperturbed evaluation however you reach it."""
    assert config_key('s4', 'A', 0.0, 0) == config_key('s4', 'C', 0.0, 3)
    assert config_key('s4', 'A', 0.0, 0) == 's4|none|0|0'


def test_baseline_only_runs_just_the_references():
    configs = enumerate_configs(parse(['--baseline-only']))
    assert len(configs) == len(DEFAULT_MODELS)
    assert all(c[2] == 0 for c in configs)


def test_joint_subset_gets_more_seeds_than_the_single_matrices():
    args = parse([])
    assert repeats_for('A+B+C', args) == args.headline_repeats
    assert repeats_for('A', args) == args.repeats
    assert args.headline_repeats > args.repeats


def test_filters_narrow_the_plan():
    args = parse(['--models', 's4_concat', '--subsets', 'A',
                  '--sigmas', '0.1', '1.0', '--repeats', '2'])
    configs = enumerate_configs(args)
    assert len(configs) == 1 + 2 * 2            # baseline + 2 sigmas x 2 seeds
    assert {c[1] for c in configs} == {BASELINE_SUBSET, 'A'}


def test_sigmas_are_enumerated_ascending():
    """A truncated sweep should degrade into a shorter curve, not a gapped
    one."""
    args = parse(['--sigmas', '1.0', '0.001', '0.1', '--subsets', 'A',
                  '--models', 's4', '--repeats', '1'])
    sigmas = [c[2] for c in enumerate_configs(args) if c[2] > 0]
    assert sigmas == sorted(sigmas)


def test_zero_sigma_in_the_grid_does_not_duplicate_the_baseline():
    args = parse(['--sigmas', '0', '0.1', '--subsets', 'A', '--models', 's4',
                  '--repeats', '1'])
    configs = enumerate_configs(args)
    assert sum(1 for c in configs if c[2] == 0) == 1


def test_config_key_formatting_matches_the_seed_derivation():
    """config_key and perturb.config_seed must format sigma identically, or a
    resumed sweep would draw different noise for the 'same' configuration."""
    from Models.robustness.perturb import config_seed
    sigma = 0.00316
    key = config_key('s4', 'A', sigma, 2)
    assert key == f"s4|A|{sigma:.10g}|2"
    assert config_seed('s4', 'A', sigma, 2) == config_seed('s4', 'A', sigma, 2)


# ------------------------------------------------------------ fixtures ---

def _record(model, subset, sigma, repeat, mse, base_mse=10.0):
    return {
        'key': config_key(model, subset, sigma, repeat),
        'model': model, 'subset': subset if sigma > 0 else BASELINE_SUBSET,
        'sigma': sigma, 'repeat': repeat, 'seed': 1 if sigma > 0 else None,
        'perturbed': sigma > 0,
        'mse': mse, 'rmse': math.sqrt(mse), 'mae': mse / 3, 'r2': 0.9,
        'nz_mae': 0.01, 'ny_mae': 0.02,
        'mse_fp32_table2': base_mse + 0.02,
        'delta_mse_vs_table2': mse - (base_mse + 0.02),
        'per_target_mse': [mse * 0.987, 0.1, 0.01, 0.02, 0.03],
        'per_target_rmse': [1.0] * 5,
        'per_target_mae': [mse / 5, 0.2, 0.02, 0.03, 0.04],
        'per_target_r2': [0.9] * 5,
        'direction_norm_mean_dev': 0.01, 'params': 1000, 'space': 'physical',
        'n_test_samples': 100, 'wall_clock_sec': 1.0,
        'timestamp': '2026-08-03T00:00:00+00:00',
        'precision': 'fp32', 'device': 'cpu', 'gpu': 'test',
        'batch_size': 8, 'n_eval_samples': 100, 'n_test_samples_full': 100,
        'limited': False,
        'perturbation': None if sigma == 0 else {
            'n_tensors': 10, 'n_params_perturbed': 5120, 'sigma': sigma,
            'seed': 1, 'mean_rel_l2_err': sigma, 'max_rel_l2_err': sigma * 1.1,
            'max_abs_err': sigma, 'mean_realized_sigma_ratio': sigma,
            'per_tensor': []},
    }


def write_jsonl(path, records):
    with open(path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    return str(path)


def quadratic_fixture(tmp_path, models=('s4', 's4_concat'), k=(100.0, 10.0),
                      base=10.0, sigmas=(0.001, 0.01, 0.1), repeats=2):
    """results.jsonl for curves that are exactly dMSE = k * sigma^2.

    Known closed-form answers: log-log slope 2, intercept log10(k), and
    sigma@2x = sqrt(base / k).
    """
    records = []
    for model, kk in zip(models, k):
        records.append(_record(model, BASELINE_SUBSET, 0.0, 0, base, base))
        for sigma in sigmas:
            for r in range(repeats):
                records.append(_record(model, 'A+B+C', sigma, r,
                                       base + kk * sigma ** 2, base))
    return write_jsonl(tmp_path / 'results.jsonl', records)


# --------------------------------------------------------------- loading ---

def test_load_records_splits_baselines_from_perturbed(tmp_path):
    path = quadratic_fixture(tmp_path)
    perturbed, baselines = tables.load_records(path)
    assert set(baselines) == {'s4', 's4_concat'}
    assert len(perturbed) == 2 * 3 * 2
    assert all(r['perturbed'] for r in perturbed)


def test_duplicate_keys_are_last_write_wins(tmp_path):
    """A --force re-run supersedes the original rather than double-counting."""
    recs = [_record('s4', 'A+B+C', 0.1, 0, 11.0),
            _record('s4', 'A+B+C', 0.1, 0, 12.0)]
    path = write_jsonl(tmp_path / 'r.jsonl', recs)
    perturbed, _ = tables.load_records(path)
    assert len(perturbed) == 1 and perturbed[0]['mse'] == 12.0


def test_truncated_final_line_is_tolerated(tmp_path):
    path = tmp_path / 'r.jsonl'
    write_jsonl(path, [_record('s4', BASELINE_SUBSET, 0.0, 0, 10.0),
                       _record('s4', 'A+B+C', 0.1, 0, 11.0)])
    with open(path, 'a') as f:
        f.write('{"key": "s4|A|0.1|1", "mse": 1')
    perturbed, baselines = tables.load_records(str(path))
    assert len(perturbed) == 1 and len(baselines) == 1


def test_missing_results_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tables.load_records(str(tmp_path / 'nope.jsonl'))


# ----------------------------------------------------------- aggregation ---

def test_aggregate_reports_mean_and_std_over_repeats(tmp_path):
    recs = [_record('s4', BASELINE_SUBSET, 0.0, 0, 10.0)]
    for r, mse in enumerate([11.0, 13.0, 12.0]):
        recs.append(_record('s4', 'A+B+C', 0.1, r, mse))
    perturbed, baselines = tables.load_records(
        write_jsonl(tmp_path / 'r.jsonl', recs))
    row = tables.aggregate(perturbed, baselines)[0]
    assert row['n_repeats'] == 3
    assert row['mse_mean'] == pytest.approx(12.0)
    assert row['mse_std'] == pytest.approx(1.0)      # sample std of 11,12,13
    assert row['mse_min'] == 11.0 and row['mse_max'] == 13.0
    assert row['delta_mse_mean'] == pytest.approx(2.0)
    assert row['mse_ratio'] == pytest.approx(1.2)


def test_single_repeat_reports_zero_std(tmp_path):
    recs = [_record('s4', BASELINE_SUBSET, 0.0, 0, 10.0),
            _record('s4', 'A+B+C', 0.1, 0, 11.0)]
    perturbed, baselines = tables.load_records(
        write_jsonl(tmp_path / 'r.jsonl', recs))
    assert tables.aggregate(perturbed, baselines)[0]['mse_std'] == 0.0


def test_delta_is_referenced_to_the_measured_baseline_not_table2(tmp_path):
    """The bf16 Table 2 number and the fp32 baseline differ by an offset the
    size of the small-sigma effect, so deltas must use the measured one."""
    recs = [_record('s4', BASELINE_SUBSET, 0.0, 0, 10.0, base_mse=10.0),
            _record('s4', 'A+B+C', 0.1, 0, 11.0, base_mse=10.0)]
    perturbed, baselines = tables.load_records(
        write_jsonl(tmp_path / 'r.jsonl', recs))
    row = tables.aggregate(perturbed, baselines)[0]
    assert row['mse_baseline'] == 10.0            # not 10.02, the table2 value
    assert row['delta_mse_mean'] == pytest.approx(1.0)


# ------------------------------------------------------------- estimators ---

def test_slope_fit_recovers_a_known_power_law():
    """dMSE = 100 * sigma^2 -> slope 2, intercept log10(100) = 2."""
    sigmas = [0.001, 0.01, 0.1]
    deltas = [100 * s ** 2 for s in sigmas]
    mses = [10.0 + d for d in deltas]
    fit = tables.fit_slope(sigmas, deltas, 10.0, mses)
    assert fit['slope'] == pytest.approx(2.0, abs=1e-9)
    assert fit['intercept'] == pytest.approx(2.0, abs=1e-9)
    assert fit['r2'] == pytest.approx(1.0)
    assert fit['n_points'] == 3


def test_slope_fit_recovers_a_linear_power_law():
    sigmas = [0.001, 0.01, 0.1, 1.0]
    deltas = [0.5 * s for s in sigmas]
    fit = tables.fit_slope(sigmas, deltas, 1e9, [1.0] * 4)
    assert fit['slope'] == pytest.approx(1.0, abs=1e-9)


def test_slope_fit_excludes_saturated_points():
    """Past 10x baseline the curve flattens for numerical reasons; fitting
    through it would reward whichever model breaks soonest."""
    sigmas = [0.001, 0.01, 0.1, 1.0]
    deltas = [100 * s ** 2 for s in sigmas[:3]] + [500.0]   # saturated & flat
    mses = [10.0 + d for d in deltas]
    fit = tables.fit_slope(sigmas, deltas, 10.0, mses)
    assert fit['n_points'] == 3
    assert fit['slope'] == pytest.approx(2.0, abs=1e-9)
    assert 'saturated' in fit['excluded']
    assert fit['sigma_hi'] == pytest.approx(0.1)


def test_slope_fit_excludes_nonpositive_deltas():
    sigmas = [0.001, 0.01, 0.1, 0.316]
    deltas = [-0.001] + [100 * s ** 2 for s in sigmas[1:]]
    fit = tables.fit_slope(sigmas, deltas, 1e9, [1.0] * 4)
    assert fit['n_points'] == 3
    assert 'delta<=0' in fit['excluded']


def test_slope_fit_returns_none_when_too_few_points_survive():
    """A matrix nothing moved is a real outcome and must be reported as such,
    not as a fabricated slope."""
    fit = tables.fit_slope([0.1, 1.0], [-0.1, -0.2], 10.0, [9.9, 9.8])
    assert fit['slope'] is None and fit['n_points'] == 0


def test_sigma_at_2x_recovers_the_closed_form():
    """MSE = 10 + 1000*sigma^2 hits 20 at sigma = sqrt(10/1000) = 0.1."""
    sigmas = [0.01, 0.0316, 0.1, 0.316]
    mses = [10.0 + 1000 * s ** 2 for s in sigmas]
    assert tables.sigma_at_ratio(sigmas, mses, 10.0) == pytest.approx(0.1,
                                                                     rel=1e-9)


def test_sigma_at_2x_interpolates_between_grid_points():
    """The crossing lands between two tested sigmas and must be interpolated,
    not snapped to the next grid point."""
    sigmas = [0.01, 0.1, 1.0]
    mses = [10.5, 15.0, 60.0]
    s2x = tables.sigma_at_ratio(sigmas, mses, 10.0)
    assert 0.1 < s2x < 1.0


def test_sigma_at_2x_is_none_when_never_reached():
    """Censored, not zero and not the largest sigma — the caller reports a
    bound."""
    assert tables.sigma_at_ratio([0.01, 0.1, 1.0], [10.1, 10.5, 12.0],
                                 10.0) is None


def test_slope_rows_carry_both_statistics(tmp_path):
    path = quadratic_fixture(tmp_path, k=(1000.0, 1000.0), base=10.0,
                             sigmas=(0.001, 0.01, 0.1))
    perturbed, baselines = tables.load_records(path)
    summary = tables.aggregate(perturbed, baselines)
    rows = tables.slope_rows(summary, baselines)
    assert len(rows) == 2
    for r in rows:
        assert r['slope'] == pytest.approx(2.0, abs=1e-6)
        assert r['intercept'] == pytest.approx(3.0, abs=1e-6)
        assert r['sigma_at_2x'] == pytest.approx(0.1, rel=1e-6)


def test_more_robust_model_gets_a_larger_sigma_at_2x(tmp_path):
    """The end-to-end ranking claim: a model with 10x less curvature tolerates
    sqrt(10)x more noise before doubling, and the estimator must say so."""
    # The grid must extend past sturdy's crossing (sigma = sqrt(10/100) =
    # 0.316) or its sigma@2x is censored and there is nothing to compare.
    path = quadratic_fixture(tmp_path, models=('fragile', 'sturdy'),
                             k=(1000.0, 100.0), base=10.0,
                             sigmas=(0.001, 0.01, 0.1, 0.316, 1.0))
    perturbed, baselines = tables.load_records(path)
    rows = {r['model']: r for r in tables.slope_rows(
        tables.aggregate(perturbed, baselines), baselines)}
    assert rows['sturdy']['sigma_at_2x'] > rows['fragile']['sigma_at_2x']
    assert (rows['sturdy']['sigma_at_2x'] / rows['fragile']['sigma_at_2x']
            == pytest.approx(math.sqrt(10), rel=1e-3))
    # Same exponent, different constant — the intercepts separate them.
    assert rows['sturdy']['slope'] == pytest.approx(rows['fragile']['slope'],
                                                    abs=1e-6)
    assert rows['sturdy']['intercept'] < rows['fragile']['intercept']


# ---------------------------------------------------------------- outputs ---

EXPECTED_FILES = {tables.RAW_CSV, tables.SUMMARY_CSV, tables.PER_TARGET_CSV,
                  tables.SLOPES_CSV, tables.SUMMARY_MD}


def test_build_writes_the_expected_files(tmp_path):
    path = quadratic_fixture(tmp_path)
    written = tables.build(path, str(tmp_path), make_plots=False)
    assert EXPECTED_FILES <= {p.split('\\')[-1].split('/')[-1]
                              for p in written}


def test_build_is_idempotent(tmp_path):
    path = quadratic_fixture(tmp_path)
    first = tables.build(path, str(tmp_path), make_plots=False)
    contents = {p: open(p, encoding='utf-8').read() for p in first}
    second = tables.build(path, str(tmp_path), make_plots=False)
    assert set(first) == set(second)
    for p, text in contents.items():
        assert open(p, encoding='utf-8').read() == text


def test_raw_csv_preserves_every_repeat(tmp_path):
    """The output requirement: aggregation must not be the only surviving
    record of the per-repeat data."""
    path = quadratic_fixture(tmp_path, sigmas=(0.001, 0.01, 0.1), repeats=3)
    tables.build(path, str(tmp_path), make_plots=False)
    import csv
    with open(tmp_path / tables.RAW_CSV, newline='') as f:
        rows = list(csv.DictReader(f))
    # 2 models x (1 baseline + 3 sigmas x 3 repeats)
    assert len(rows) == 2 * (1 + 9)
    assert {r['repeat'] for r in rows if r['perturbed'] == 'True'} == \
        {'0', '1', '2'}


def test_summary_csv_has_one_row_per_model_subset_sigma(tmp_path):
    path = quadratic_fixture(tmp_path, sigmas=(0.001, 0.01, 0.1), repeats=3)
    tables.build(path, str(tmp_path), make_plots=False)
    import csv
    with open(tmp_path / tables.SUMMARY_CSV, newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 3
    assert all(r['n_repeats'] == '3' for r in rows)


def test_per_target_csv_reports_percent_degradation(tmp_path):
    path = quadratic_fixture(tmp_path)
    tables.build(path, str(tmp_path), make_plots=False)
    import csv
    with open(tmp_path / tables.PER_TARGET_CSV, newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows
    for name in TARGETS:
        assert f'delta_mae_pct_{name}' in rows[0]
    # x_entry MAE is mse/5 in the fixture, so it degrades with mse.
    worst = max(rows, key=lambda r: float(r['sigma']))
    assert float(worst['delta_mae_pct_x_entry']) > 0


def test_summary_markdown_answers_the_proposal_question(tmp_path):
    path = quadratic_fixture(tmp_path, models=('s4', 's4_modulate_full'),
                             k=(1000.0, 100.0), base=10.0,
                             sigmas=(0.001, 0.01, 0.1, 0.316, 1.0))
    tables.build(path, str(tmp_path), make_plots=False)
    text = open(tmp_path / tables.SUMMARY_MD, encoding='utf-8').read()
    assert 'Does modulation degrade more slowly?' in text
    assert 'sigma@2x' in text
    assert 's4_modulate_full' in text
    # The more robust model must be named first in the ordering.
    ordering = text.split('the ordering is')[1].split('\n')[0]
    assert ordering.index('s4_modulate_full') < ordering.index('`s4`')
    # Equal slopes -> the summary must say they differ in constant, not shape.
    assert 'CONSTANT of degradation' in text


def test_summary_markdown_reports_a_censored_sigma_at_2x(tmp_path):
    """A model that never doubles within the tested range is MORE robust than
    every ranked one, and the summary must say that rather than drop it."""
    path = quadratic_fixture(tmp_path, models=('fragile', 'sturdy'),
                             k=(1000.0, 1.0), base=10.0,
                             sigmas=(0.001, 0.01, 0.1))
    tables.build(path, str(tmp_path), make_plots=False)
    text = open(tmp_path / tables.SUMMARY_MD, encoding='utf-8').read()
    assert 'never reached 2x' in text and 'sturdy' in text
    # And the slopes table shows a lower bound, not a fabricated number.
    assert '> 0.1' in text


def test_summary_markdown_flags_a_limited_run(tmp_path):
    recs = [_record('s4', BASELINE_SUBSET, 0.0, 0, 10.0),
            _record('s4', 'A+B+C', 0.1, 0, 11.0)]
    recs[1]['limited'] = True
    path = write_jsonl(tmp_path / 'r.jsonl', recs)
    tables.build(path, str(tmp_path), make_plots=False)
    text = open(tmp_path / tables.SUMMARY_MD, encoding='utf-8').read()
    assert '--limit-samples' in text and 'NOT comparable' in text


def test_matrix_ordering_verdict_detects_the_phase6_pattern(tmp_path):
    """A > C > B at the largest sigma must be recognized as reproducing the
    quantization finding."""
    recs = [_record('s4', BASELINE_SUBSET, 0.0, 0, 10.0)]
    for subset, mse in (('A', 100.0), ('C', 30.0), ('B', 11.0)):
        recs.append(_record('s4', subset, 1.0, 0, mse))
    path = write_jsonl(tmp_path / 'r.jsonl', recs)
    tables.build(path, str(tmp_path), make_plots=False)
    text = open(tmp_path / tables.SUMMARY_MD, encoding='utf-8').read()
    assert 'A > C > B' in text
    assert 'reproduces' in text


def test_matrix_ordering_verdict_detects_a_departure(tmp_path):
    recs = [_record('s4', BASELINE_SUBSET, 0.0, 0, 10.0)]
    for subset, mse in (('A', 11.0), ('C', 30.0), ('B', 100.0)):
        recs.append(_record('s4', subset, 1.0, 0, mse))
    path = write_jsonl(tmp_path / 'r.jsonl', recs)
    tables.build(path, str(tmp_path), make_plots=False)
    text = open(tmp_path / tables.SUMMARY_MD, encoding='utf-8').read()
    assert 'does NOT reproduce' in text.replace('DOES NOT', 'does NOT')


def test_build_rejects_an_empty_results_file(tmp_path):
    path = write_jsonl(tmp_path / 'r.jsonl', [])
    with pytest.raises(ValueError, match='no usable records'):
        tables.build(path, str(tmp_path), make_plots=False)


# ------------------------------------------------------------------ plots ---

def test_plots_are_written(tmp_path):
    from Models.robustness import plots
    path = quadratic_fixture(tmp_path, sigmas=(0.001, 0.01, 0.1))
    written = tables.build(path, str(tmp_path), make_plots=True)
    names = {p.replace('\\', '/').split('/')[-1] for p in written}
    assert plots.CURVES_PNG in names
    assert plots.DELTA_PNG in names
    for name in (plots.CURVES_PNG, plots.DELTA_PNG):
        assert (tmp_path / name).stat().st_size > 1000


def test_by_matrix_plot_is_written_when_single_matrices_are_present(tmp_path):
    from Models.robustness import plots
    recs = [_record('s4', BASELINE_SUBSET, 0.0, 0, 10.0)]
    for subset in ('A', 'B', 'C', 'A+B+C'):
        for sigma in (0.01, 0.1, 1.0):
            recs.append(_record('s4', subset, sigma, 0, 10.0 + 100 * sigma ** 2))
    path = write_jsonl(tmp_path / 'r.jsonl', recs)
    written = tables.build(path, str(tmp_path), make_plots=True)
    names = {p.replace('\\', '/').split('/')[-1] for p in written}
    assert plots.BY_MATRIX_PNG in names


def test_error_band_stays_positive_when_spread_exceeds_the_mean(tmp_path):
    """Regression: the band used to be mean +/- std clamped at 1e-12. Seed
    spread at moderate sigma routinely approaches the mean, so the lower edge
    went non-positive, hit the clamp, and dragged the shared log y-axis down a
    dozen decades — flattening every panel of the by-matrix figure into a line.
    The band is the observed min-max now, which cannot go non-positive.
    """
    from Models.robustness import plots
    recs = [_record('s4', BASELINE_SUBSET, 0.0, 0, 10.0)]
    # One draw lands in a high-curvature direction and the others don't, so
    # the sample std (~1725) exceeds the mean (~1007) outright. This is the
    # real shape of the data: the local smoke run saw a 1.5x spread between
    # two seeds at sigma=0.01 alone.
    for r, mse in enumerate([11.0, 11.0, 3000.0]):
        recs.append(_record('s4', 'A+B+C', 0.1, r, mse))
    perturbed, baselines = tables.load_records(
        write_jsonl(tmp_path / 'r.jsonl', recs))
    summary = tables.aggregate(perturbed, baselines)
    row = summary[0]
    assert row['mse_mean'] - row['mse_std'] < 0, 'fixture no longer exercises it'

    _, means, lo, hi = plots._series(summary, 's4', 'A+B+C')
    assert all(v > 0 for v in lo)
    assert lo == [11.0] and hi == [3000.0]
    assert all(l <= m <= h for l, m, h in zip(lo, means, hi))

    plots.plot_curves(summary, baselines, str(tmp_path))
    assert (tmp_path / plots.CURVES_PNG).stat().st_size > 1000


def test_color_is_bound_to_the_model_not_to_plot_order():
    """A run that omits a model must not repaint the survivors."""
    from Models.robustness import plots
    three = plots._style(['a', 'b', 'c'])
    assert three['a'][0] != three['b'][0] != three['c'][0]
    # Same models in the same fixed order -> same assignment, always.
    assert plots._style(['a', 'b', 'c']) == three


def test_too_many_series_fails_loudly_rather_than_cycling_hues():
    from Models.robustness import plots
    with pytest.raises(ValueError, match='validated categorical slots'):
        plots._style(['a', 'b', 'c', 'd'])


def test_table_build_survives_a_plotting_failure(tmp_path, monkeypatch):
    """Plots are a nice-to-have; losing 2.2 h of tables to a matplotlib
    misconfiguration on a pod is not acceptable."""
    from Models.robustness import plots
    monkeypatch.setattr(plots, 'build_all',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('x')))
    path = quadratic_fixture(tmp_path)
    written = tables.build(path, str(tmp_path), make_plots=True)
    assert any(p.endswith(tables.SUMMARY_CSV) for p in written)
    assert (tmp_path / tables.SUMMARY_MD).exists()
