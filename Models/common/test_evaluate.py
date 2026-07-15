"""Phase 5 tests — Models/evaluate.py aggregator + scripts/sweep.py CLI.

Aggregator tests run entirely on synthetic final_report.json fixtures (the
frozen Phase 3 schema); no training, no chunk files. Sweep tests exercise
only the cheap paths (--dry-run, preflight failure) — real subprocess
orchestration is covered by the smoke sweep, not unit tests.

Run from the repo root:
    pytest Models/common/test_evaluate.py
"""
import csv
import json
import os

import pytest

from Data import compute_stats_hashes
from Models import evaluate
from scripts import sweep

GPU_NAME = 'Test GPU 9000'


def make_report(model, seed=42, data_dir='/vol/preprocessed_data',
                target_hash='t' * 64, norm_hash='n' * 64,
                mse=0.1, val_mse_history=(0.5, 0.3, 0.4)):
    """Synthetic final_report.json matching the frozen Phase 3 schema."""
    return {
        'table_row': {
            'model': model,
            'mse': mse,
            'rmse': mse ** 0.5,
            'mae': mse / 2,
            'r2': 0.9,
            'nz_mae': 0.01,
            'ny_mae': 0.02,
            'per_target_mse': [0.1, 0.2, 0.3, 0.4, 0.5],
            'per_target_rmse': [0.1] * 5,
            'per_target_mae': [0.1] * 5,
            'per_target_r2': [0.9] * 5,
            'direction_norm_mean_dev': 0.05,
            'params': 12345,
            'space': 'physical',
            'n_test_samples': 1000,
        },
        'test_aggregate_mse_weighted': mse * 1.1,
        'best_val_mse': min(val_mse_history),
        'epochs_trained': len(val_mse_history),
        'run_config': {
            'model_name': model,
            'seed': seed,
            'data_dir': data_dir,
            'batch_size': 64,
            'lr': 1e-3,
        },
        'metadata': {
            'seed': seed,
            'gpu': {'name': GPU_NAME, 'backend': 'cuda'},
            'precision': 'bf16',
            'target_stats_hash': target_hash,
            'norm_stats_hash': norm_hash,
        },
        'history': {
            'train_loss': [1.0] * len(val_mse_history),
            'val_mse': list(val_mse_history),
        },
    }


def write_report(root, model, **kw):
    d = root / model
    d.mkdir(parents=True, exist_ok=True)
    with open(d / 'final_report.json', 'w') as f:
        json.dump(make_report(model, **kw), f)


def read_csv_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


# --- aggregator -----------------------------------------------------------

def test_empty_input_dir_errors(tmp_path, capsys):
    assert evaluate.main(['--input-root', str(tmp_path)]) == 1
    assert 'no final_report.json' in capsys.readouterr().out


def test_single_model_row_fields(tmp_path):
    write_report(tmp_path, 'mlp', mse=0.25)
    with open(tmp_path / 'sweep_summary.json', 'w') as f:
        json.dump({'models': {'mlp': {'status': 'success',
                                      'wall_clock_sec': 12.5}}}, f)
    assert evaluate.main(['--input-root', str(tmp_path)]) == 0

    rows = read_csv_rows(tmp_path / 'Table2.csv')
    assert len(rows) == 1
    r = rows[0]
    assert r['model'] == 'mlp'
    assert float(r['test_mse']) == 0.25
    assert float(r['test_mse_weighted']) == pytest.approx(0.25 * 1.1)
    assert float(r['mse_n_z']) == 0.5
    assert r['params'] == '12345'
    assert r['gpu'] == GPU_NAME
    assert r['precision'] == 'bf16'
    assert r['wall_clock_sec'] == '12.5'
    assert r['best_val_epoch'] == '2'  # argmin of (0.5, 0.3, 0.4)

    md = (tmp_path / 'Table2.md').read_text()
    assert '| mlp |' in md
    assert 'physical' in md and '42' in md  # space + split seed in header


def test_matching_hashes_no_warnings(tmp_path, capsys):
    for m in ('mlp', 'cnn', 'gru'):
        write_report(tmp_path, m)
    assert evaluate.main(['--input-root', str(tmp_path)]) == 0
    assert '[WARNING]' not in capsys.readouterr().out
    assert 'WARNING (consistency)' not in (tmp_path / 'Table2.md').read_text()


def test_hash_mismatch_warns_then_strict_fails(tmp_path, capsys):
    write_report(tmp_path, 'mlp')
    write_report(tmp_path, 'cnn', target_hash='x' * 64)

    assert evaluate.main(['--input-root', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert '[WARNING]' in out and 'target_stats_hash' in out
    assert 'WARNING (consistency)' in (tmp_path / 'Table2.md').read_text()

    assert evaluate.main(['--input-root', str(tmp_path), '--strict']) == 1
    assert 'consistency check failed' in capsys.readouterr().out


def test_subdir_without_report_is_skipped(tmp_path):
    write_report(tmp_path, 'mlp')
    (tmp_path / 's4').mkdir()  # partial sweep: s4 never finished
    (tmp_path / 's4' / 'latest.pt').write_bytes(b'not a report')
    assert evaluate.main(['--input-root', str(tmp_path)]) == 0
    assert [r['model'] for r in read_csv_rows(tmp_path / 'Table2.csv')] == ['mlp']


def test_output_order_is_canonical_not_alphabetical(tmp_path):
    for m in ('s4_modulate', 'cnn', 'mlp'):  # written out of order
        write_report(tmp_path, m)
    assert evaluate.main(['--input-root', str(tmp_path)]) == 0
    rows = read_csv_rows(tmp_path / 'Table2.csv')
    assert [r['model'] for r in rows] == ['mlp', 'cnn', 's4_modulate']


# --- sweep CLI ------------------------------------------------------------

def test_sweep_dry_run_prints_commands_without_training(tmp_path, capsys):
    out_root = tmp_path / 'out'
    assert sweep.main(['--output-root', str(out_root), '--dry-run']) == 0
    out = capsys.readouterr().out
    assert 'DRY RUN' in out
    assert out.count('-m Models.train') == len(sweep.EXPECTED_MODELS)
    assert not out_root.exists()  # dry run writes nothing


def test_sweep_preflight_fails_on_missing_hashes_file(tmp_path, capsys,
                                                      monkeypatch):
    monkeypatch.setattr(compute_stats_hashes, 'HASHES_PATH',
                        str(tmp_path / 'absent_hashes.txt'))
    assert sweep.main(['--output-root', str(tmp_path / 'out'),
                       '--dry-run']) == 1
    out = capsys.readouterr().out
    assert 'PREFLIGHT FAILED' in out and 'absent_hashes.txt' in out


def test_sweep_models_flag_restricts_selection(tmp_path, capsys):
    assert sweep.main(['--output-root', str(tmp_path / 'out'), '--dry-run',
                       '--models', 'mlp', 'cnn']) == 0
    out = capsys.readouterr().out
    assert out.count('-m Models.train') == 2
    assert '--model mlp' in out and '--model cnn' in out
    assert '--model gru' not in out
