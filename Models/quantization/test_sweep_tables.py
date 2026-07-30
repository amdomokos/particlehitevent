"""Tests for the Phase 6 sweep orchestration and table generation.

No checkpoints, no dataset, no inference — these cover the parts that decide
WHAT gets run and HOW numbers reach the tables:
  - the sweep really enumerates 64 quantized configurations + 2 baselines;
  - restoring pristine weights between configurations is bit-exact, and a
    failure to restore is detected rather than compounded;
  - ΔMSE is computed against the reference the caller asked for;
  - a --limit-samples smoke run is labelled as such in its own output.

Run from the repo root:
    pytest Models/quantization/test_sweep_tables.py
"""
import argparse
import json
import os

import pytest
import torch

import Models.models_import_all  # noqa: F401 — populates registry
from Models.common.registry import build_model
from Models.quantization import tables
from Models.quantization.fake_quant import BITS, GRANULARITIES, SUBSETS, apply_fake_quant
from Models.quantization.sweep import (
    build_parser,
    config_key,
    enumerate_configs,
    verify_restore,
)

TINY_S4 = {'d_model': 16, 'd_state': 8, 'n_layers': 2, 'dropout': 0.0,
           'y_module_hidden': 8}


def _args(**overrides):
    args = build_parser().parse_args(
        ['--checkpoint-root', 'x', '--data-dir', 'y'])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


# --------------------------------------------------------------------------
# configuration enumeration — the "64 runs" contract
# --------------------------------------------------------------------------

def test_default_sweep_is_64_quantized_plus_2_baselines():
    configs = enumerate_configs(_args())
    quantized = [c for c in configs if c[3] is not None]
    baselines = [c for c in configs if c[3] is None]
    assert len(quantized) == 64, (
        f"expected 2 checkpoints x 2 granularities x 4 subsets x 4 bit-widths "
        f"= 64, got {len(quantized)}")
    assert len(baselines) == 2
    assert len(configs) == 66


def test_enumeration_covers_the_full_cross_product():
    configs = enumerate_configs(_args())
    quantized = {c for c in configs if c[3] is not None}
    expected = {(m, g, s, b)
                for m in ('s4_concat', 's4_modulate_full')
                for g in GRANULARITIES
                for s in SUBSETS
                for b in BITS}
    assert quantized == expected


def test_config_keys_are_unique():
    configs = enumerate_configs(_args())
    keys = [config_key(*c) for c in configs]
    assert len(keys) == len(set(keys))


def test_baseline_runs_first_per_checkpoint():
    """A sweep killed early must still leave a usable reference behind."""
    configs = enumerate_configs(_args())
    for model in ('s4_concat', 's4_modulate_full'):
        idx = [i for i, c in enumerate(configs) if c[0] == model]
        assert configs[idx[0]][3] is None, f"{model} does not start with baseline"


def test_baseline_only_skips_quantized_configs():
    configs = enumerate_configs(_args(baseline_only=True))
    assert len(configs) == 2
    assert all(c[3] is None for c in configs)


def test_subset_filters_narrow_the_sweep():
    configs = enumerate_configs(_args(
        models=['s4_concat'], granularities=['per_tensor'],
        subsets=['A+B+C'], bits=[8]))
    assert len(configs) == 2                    # baseline + one config
    assert config_key(*configs[1]) == 's4_concat|per_tensor|A+B+C|8'


def test_baseline_key_is_distinct_from_any_quantized_key():
    baseline = config_key('s4_concat', None, None, None)
    assert baseline == 's4_concat|baseline|none|fp32'
    assert baseline not in {config_key(*c) for c in enumerate_configs(_args())
                            if c[3] is not None}


# --------------------------------------------------------------------------
# weight restore — the assumption all 64 numbers rest on
# --------------------------------------------------------------------------

def _pristine(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _detrain(model, seed=7):
    """Perturb the SSM parameters away from their initialization.

    A freshly built S4Layer is a degenerate quantization target: log_A_real is
    a constant (zero range) and A_imag is pi*{0..N-1}, which lands exactly on
    the uniform grid at several bit-widths — so quantizing 'A' on an untrained
    model can legitimately be a no-op. Trained weights are not like that, and
    tests about whether quantization CHANGED anything need weights that are
    not accidentally already quantized.
    """
    from Models.quantization.fake_quant import iter_ssm_params
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, _, _, param in iter_ssm_params(model):
            param.add_(torch.randn(param.shape, generator=g) * 0.1)
    return model


def test_restore_undoes_quantization_bit_exactly():
    model = _detrain(build_model('s4_concat', TINY_S4))
    pristine = _pristine(model)
    apply_fake_quant(model, 'A+B+C', 2, 'per_tensor')
    assert any(not torch.equal(v.detach().cpu(), pristine[k])
               for k, v in model.state_dict().items()), "nothing was quantized"

    model.load_state_dict(pristine)
    verify_restore(model, pristine)             # must not raise
    for k, v in model.state_dict().items():
        assert torch.equal(v.detach().cpu(), pristine[k]), f"{k} not restored"


def test_verify_restore_detects_unrestored_weights():
    """The failure this guard exists for: quantizing on top of quantized
    weights, which would inflate every later cell of Table 4."""
    model = _detrain(build_model('s4_concat', TINY_S4))
    pristine = _pristine(model)
    apply_fake_quant(model, 'A', 4, 'per_tensor')
    with pytest.raises(RuntimeError, match='restore failed'):
        verify_restore(model, pristine)


def test_repeated_quantization_after_restore_is_reproducible():
    """Two runs of the same configuration, with a different configuration in
    between, must produce identical weights — the sweep's determinism."""
    model = _detrain(build_model('s4_modulate_full', TINY_S4))
    pristine = _pristine(model)

    apply_fake_quant(model, 'C', 6, 'per_channel')
    first = _pristine(model)

    model.load_state_dict(pristine)
    apply_fake_quant(model, 'A+B+C', 2, 'per_tensor')     # interference
    model.load_state_dict(pristine)
    apply_fake_quant(model, 'C', 6, 'per_channel')

    for k, v in model.state_dict().items():
        assert torch.equal(v.detach().cpu(), first[k]), f"{k} not reproducible"


# --------------------------------------------------------------------------
# table generation
# --------------------------------------------------------------------------

def _record(model, gran, subset, bits, mse, table2_mse=10.0, **extra):
    per_target = [mse, 0.5, 0.01, 0.001, 0.002]
    rec = {
        'key': config_key(model, gran if bits else None,
                          subset if bits else None, bits),
        'model': model, 'granularity': gran or 'baseline',
        'subset': subset or 'none', 'bits': bits,
        'quantized': bits is not None,
        'mse': mse, 'rmse': mse ** 0.5, 'mae': mse / 10, 'r2': 0.99,
        'nz_mae': 0.008 + (0.001 if bits else 0),
        'ny_mae': 0.003 + (0.0005 if bits else 0),
        'per_target_mse': per_target,
        'per_target_rmse': [v ** 0.5 for v in per_target],
        'per_target_mae': [v / 10 for v in per_target],
        'per_target_r2': [0.99] * 5,
        'mse_fp32_table2': table2_mse,
        'delta_mse_vs_table2': mse - table2_mse,
        'direction_norm_mean_dev': 0.005, 'params': 320645,
        'space': 'physical', 'n_test_samples': 79829,
        'wall_clock_sec': 90.0, 'precision': 'fp32', 'gpu': 'A4000',
        'quant': (None if bits is None else {
            'n_tensors': 20, 'n_params_quantized': 81920,
            'mean_rel_l2_err': 0.01, 'max_rel_l2_err': 0.02,
            'max_abs_err': 0.1, 'per_tensor': []}),
    }
    rec.update(extra)
    return rec


def _write_jsonl(path, records):
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')


@pytest.fixture
def sweep_dir(tmp_path):
    """A minimal but complete two-checkpoint result set."""
    records = [
        _record('s4_concat', None, None, None, 10.05, 10.0),
        _record('s4_modulate_full', None, None, None, 10.60, 10.5),
    ]
    for model, base in (('s4_concat', 10.0), ('s4_modulate_full', 10.5)):
        for gran in GRANULARITIES:
            for subset in SUBSETS:
                for bits in BITS:
                    bump = {8: 0.01, 6: 0.05, 4: 0.4, 2: 5.0}[bits]
                    if gran == 'per_channel':
                        bump *= 0.5
                    if subset == 'A+B+C':
                        bump *= 2
                    records.append(_record(model, gran, subset, bits,
                                           base + 0.05 + bump, base))
    path = tmp_path / 'results.jsonl'
    _write_jsonl(path, records)
    return tmp_path, path


def test_load_records_splits_baselines_from_quantized(sweep_dir):
    _, path = sweep_dir
    quantized, baselines = tables.load_records(str(path))
    assert len(quantized) == 64
    assert set(baselines) == {'s4_concat', 's4_modulate_full'}


def test_load_records_last_write_wins(tmp_path):
    """A --force rerun appends; the rerun must supersede, not double-count."""
    path = tmp_path / 'results.jsonl'
    _write_jsonl(path, [
        _record('s4_concat', 'per_tensor', 'A', 8, 11.0),
        _record('s4_concat', 'per_tensor', 'A', 8, 12.0),
    ])
    quantized, _ = tables.load_records(str(path))
    assert len(quantized) == 1
    assert quantized[0]['mse'] == 12.0


def test_load_records_tolerates_a_truncated_final_line(tmp_path):
    """A pod killed mid-append must not make the whole sweep unreadable."""
    path = tmp_path / 'results.jsonl'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(_record('s4_concat', 'per_tensor', 'A', 8, 11.0)) + '\n')
        f.write('{"key": "truncated", "mse": 1')
    quantized, _ = tables.load_records(str(path))
    assert len(quantized) == 1


def test_delta_measured_uses_the_baseline_not_table2(sweep_dir):
    _, path = sweep_dir
    quantized, baselines = tables.load_records(str(path))
    rec = next(r for r in quantized
               if r['key'] == 's4_concat|per_tensor|A|8')
    # baseline 10.05, table2 10.0, quantized 10.0 + 0.05 + 0.01 = 10.06
    assert tables.delta(rec, baselines, 'measured') == pytest.approx(0.01)
    assert tables.delta(rec, baselines, 'table2') == pytest.approx(0.06)


def test_delta_measured_falls_back_to_table2_without_a_baseline(sweep_dir):
    _, path = sweep_dir
    quantized, _ = tables.load_records(str(path))
    rec = next(r for r in quantized if r['key'] == 's4_concat|per_tensor|A|8')
    assert tables.delta(rec, {}, 'measured') == pytest.approx(0.06)


def test_grid_is_subsets_by_bits(sweep_dir):
    _, path = sweep_dir
    quantized, baselines = tables.load_records(str(path))
    subsets, bits, cells = tables.grid_rows(
        quantized, baselines, 's4_concat', 'per_tensor', 'measured')
    assert subsets == list(SUBSETS)
    assert bits == list(BITS)
    assert len(cells) == 16
    # degradation must grow as bits fall, within a subset
    col = [cells[('A+B+C', b)] for b in (8, 6, 4, 2)]
    assert col == sorted(col)


def test_build_writes_every_expected_output(sweep_dir):
    tmp_path, path = sweep_dir
    out = tmp_path / 'out'
    written = tables.build(str(path), str(out), 'measured')
    names = {os.path.basename(p) for p in written}
    assert names == {
        'Table4_s4_concat_per_tensor.csv',
        'Table4_s4_concat_per_channel.csv',
        'Table4_s4_modulate_full_per_tensor.csv',
        'Table4_s4_modulate_full_per_channel.csv',
        'Table4.md', 'Table4_all.csv',
        'quantization_per_target.csv', 'SUMMARY.md',
    }
    for p in written:
        assert os.path.getsize(p) > 0


def test_build_is_idempotent(sweep_dir):
    tmp_path, path = sweep_dir
    out = tmp_path / 'out'
    tables.build(str(path), str(out), 'measured')
    first = {p: open(p, encoding='utf-8').read()
             for p in (str(out / 'Table4.md'), str(out / 'Table4_all.csv'))}
    tables.build(str(path), str(out), 'measured')
    for p, text in first.items():
        assert open(p, encoding='utf-8').read() == text


def test_long_csv_has_one_row_per_configuration(sweep_dir):
    import csv
    tmp_path, path = sweep_dir
    out = tmp_path / 'out'
    tables.build(str(path), str(out), 'measured')
    with open(out / 'Table4_all.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 64
    assert all(r['delta_mse_vs_measured'] for r in rows)
    assert all(r['delta_nz_mae_vs_measured'] for r in rows)


def test_per_target_csv_includes_baselines_and_every_target(sweep_dir):
    import csv
    from Data.config import ACTIVE_TARGET_NAMES
    tmp_path, path = sweep_dir
    out = tmp_path / 'out'
    tables.build(str(path), str(out), 'measured')
    with open(out / 'quantization_per_target.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 66                       # 64 quantized + 2 baselines
    for target in ACTIVE_TARGET_NAMES:
        assert f'mae_{target}' in rows[0]
        assert f'delta_mae_{target}' in rows[0]
    assert sum(1 for r in rows if r['quantized'] == 'False') == 2


def test_summary_names_a_robustness_winner(sweep_dir):
    tmp_path, path = sweep_dir
    out = tmp_path / 'out'
    tables.build(str(path), str(out), 'measured')
    text = open(out / 'SUMMARY.md', encoding='utf-8').read()
    assert 'Headline' in text
    assert 'A+B+C' in text or 'joint' in text
    # the synthetic fixture makes s4_concat and s4_modulate_full degrade
    # identically, so a winner column must exist even when it is a tie
    assert 'More robust' in text
    assert 'PT' in text and 'PC' in text
    assert 'PE' not in text, "granularity tag regression (per_tensor[:2])"


def test_smoke_runs_are_labelled_in_their_own_output(tmp_path):
    """A --limit-samples table must announce itself; the alternative is a
    subsampled grid escaping this directory as if it were a result."""
    path = tmp_path / 'results.jsonl'
    _write_jsonl(path, [
        _record('s4_concat', None, None, None, 500.0, 10.0,
                limited=True, n_test_samples=128, n_test_samples_full=79829),
        _record('s4_concat', 'per_tensor', 'A', 8, 501.0, 10.0,
                limited=True, n_test_samples=128, n_test_samples_full=79829),
    ])
    out = tmp_path / 'out'
    tables.build(str(path), str(out), 'measured')
    for name in ('Table4.md', 'SUMMARY.md'):
        text = open(out / name, encoding='utf-8').read()
        assert 'SMOKE RUN' in text, f"{name} does not flag the limited run"
        assert '79829' in text


def test_full_runs_carry_no_smoke_banner(sweep_dir):
    tmp_path, path = sweep_dir
    out = tmp_path / 'out'
    tables.build(str(path), str(out), 'measured')
    assert 'SMOKE RUN' not in open(out / 'SUMMARY.md', encoding='utf-8').read()


def test_partial_sweep_produces_a_smaller_grid_not_holes(tmp_path):
    path = tmp_path / 'results.jsonl'
    _write_jsonl(path, [
        _record('s4_concat', None, None, None, 10.05, 10.0),
        _record('s4_concat', 'per_tensor', 'A', 8, 10.06, 10.0),
        _record('s4_concat', 'per_tensor', 'B', 8, 10.07, 10.0),
    ])
    quantized, baselines = tables.load_records(str(path))
    subsets, bits, _ = tables.grid_rows(
        quantized, baselines, 's4_concat', 'per_tensor', 'measured')
    assert subsets == ['A', 'B']
    assert bits == [8]
    tables.build(str(path), str(tmp_path / 'out'), 'measured')   # must not raise


def test_build_rejects_an_unknown_delta_ref(sweep_dir):
    tmp_path, path = sweep_dir
    with pytest.raises(ValueError, match='delta_ref'):
        tables.build(str(path), str(tmp_path / 'out'), 'nonsense')


def test_build_raises_on_a_missing_results_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        tables.build(str(tmp_path / 'nope.jsonl'), str(tmp_path))
