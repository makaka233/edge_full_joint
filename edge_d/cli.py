"""Current experiment entrypoint. Legacy experiments have a separate archival CLI."""
import argparse
import json
import os
from pathlib import Path


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('placement-train', help='Train the current direct matrix policy and/or original Agent-S')
    train.add_argument('--seed', type=int, default=20261113)
    train.add_argument('--gs', type=int, default=64)
    train.add_argument('--eval-every', type=int, default=8)
    train.add_argument('--fixed-scenes', type=int, default=4)
    train.add_argument('--holdout-scenes', type=int, default=8)
    train.add_argument('--mode', choices=('d-only', 's-only', 'joint', 'periodic'), default='d-only')
    train.add_argument('--periods', type=int, default=1)
    train.add_argument('--inventory-routing', action='store_true', help='Experimental inventory-aware CPU transport repair')
    train.add_argument('--queue-residual', action='store_true', help='Experimental workload prior for original DDQN')
    train.add_argument('--out', type=Path, required=True)
    plot = sub.add_parser('plot', help='Regenerate figures from experiment data')
    plot.add_argument('--run-dir', type=Path)
    plot.add_argument('--out', type=Path)
    args = parser.parse_args()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(variable, '1')
    from scripts.research_artifacts import render_catalog, render_run

    if args.command == 'plot':
        if args.run_dir:
            render_run(args.run_dir, args.out or args.run_dir/'figures')
        else:
            render_catalog(output=args.out)
        return
    from edge_joint.placement_training import PlacementConfig, train as train_placement

    settings = vars(args).copy()
    out = settings.pop('out')
    settings.pop('command')
    try:
        result = train_placement(PlacementConfig(**settings), out)
    except (Exception, KeyboardInterrupt):
        if (out/'submission_errors.jsonl').is_file():
            render_run(out, out/'figures')
        raise
    render_run(out, out/'figures')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    cli()
