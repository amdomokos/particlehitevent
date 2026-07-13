"""Single training entrypoint for every model in the comparison sweep.
python -m Models.train --model NAME --checkpoint-dir PATH [...]

Phase 4 model files each call ``register_model``; import them below (see the
marked block) to make them selectable. With an empty registry (Phase 3
state) this script prints a pointer at Phase 4 and exits — ``--help`` still
works, which is the first thing a fresh RunPod pod runs.
"""
import argparse
import json
import os
import sys

from Data.config import SEED
from Models.common.engine import RunConfig, fit, smoke_test
from Models.common.registry import list_models

# --- Phase 4: import model modules here so they self-register ------------
# e.g.  import Models.mlp.model  # noqa: F401
# --------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_parser(models):
    p = argparse.ArgumentParser(
        prog='python -m Models.train',
        description='Shared training engine — one trainer for all models.')
    p.add_argument('--model', required=True, choices=models or None,
                   help='registered model name'
                        + ('' if models else ' (registry is currently EMPTY)'))
    p.add_argument('--config-json', default=None,
                   help='path to a JSON file with the model_config dict')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--max-epochs', type=int, default=50)
    p.add_argument('--lambda-dir', type=float, default=0.01)
    p.add_argument('--precision', choices=['auto', 'bf16', 'fp16', 'fp32'],
                   default='auto')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--strict-deterministic', action='store_true')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--early-stop-patience', type=int, default=10)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--data-dir',
                   default=os.path.join(_REPO_ROOT, 'preprocessed_data'),
                   help='chunk_*.pt directory (default: repo-root anchored '
                        'preprocessed_data so it resolves from any cwd)')
    p.add_argument('--checkpoint-dir', required=True,
                   help='where best.pt/latest.pt/final_report.json go — '
                        'point at persistent storage on RunPod')
    p.add_argument('--smoke', action='store_true',
                   help='run smoke_test() and exit (first thing a pod does)')
    p.add_argument('--resume', action='store_true',
                   help='resume from latest.pt in --checkpoint-dir')
    return p


def main(argv=None, train_dataset=None, val_dataset=None, test_dataset=None):
    """Dataset kwargs exist for tests to inject synthetic data; CLI use
    always builds PixelClusterDataset from --data-dir."""
    models = list_models()
    args = build_parser(models).parse_args(argv)

    if not models:
        print(f"Unknown model '{args.model}'. Registered models: [] — the "
              f"registry is empty. Phase 4 adds one file per architecture, "
              f"each calling Models.common.registry.register_model(); "
              f"import them at the top of Models/train.py. Nothing to "
              f"train yet.")
        return 2

    model_config = {}
    if args.config_json:
        with open(args.config_json) as f:
            model_config = json.load(f)

    cfg = RunConfig(
        model_name=args.model,
        model_config=model_config,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        lambda_dir=args.lambda_dir,
        precision=args.precision,
        seed=args.seed,
        strict_deterministic=args.strict_deterministic,
        num_workers=args.num_workers,
        early_stop_patience=args.early_stop_patience,
        grad_clip=args.grad_clip,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        smoke=False,
    )

    if args.smoke:
        smoke_test(cfg, train_dataset=train_dataset, val_dataset=val_dataset)
        return 0

    try:
        summary = fit(cfg, train_dataset=train_dataset,
                      val_dataset=val_dataset, test_dataset=test_dataset,
                      resume=args.resume)
    except KeyboardInterrupt:
        # fit() already saved interrupted.pt before re-raising
        print('[interrupted] exiting; resume with --resume after moving '
              'interrupted.pt to latest.pt if desired')
        return 130
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
