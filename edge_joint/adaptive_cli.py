"""Expose continuous deployment through the project's main CLI."""
from pathlib import Path
import sys
import traceback


def add_arguments(parser):
    parser.add_argument('action', choices=('train', 'evaluate', 'verify'))
    parser.add_argument('--seed', type=int, choices=(20261113, 20261114), required=True)
    parser.add_argument('--prefix', type=Path, help='output filename prefix, default: runs/results/EXP019.seed<seed>')
    parser.add_argument('--episodes', type=int, default=128, help='absolute training target')
    parser.add_argument('--periods', type=int, default=8, help='periods per continuous episode')
    parser.add_argument('--period-seconds', type=float, default=24.)
    parser.add_argument('--image-fraction', type=float, default=.001, help='fraction of the SSD footprint transferred')
    parser.add_argument('--eval-every', type=int, default=16, help='fixed validation interval in episodes')
    parser.add_argument('--s-batch', type=int, default=128, help='original DDQN replay minibatch size')
    parser.add_argument('--freeze-ds', action='store_true', help='learn the macro policy with fixed D and greedy S')
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument('--init-checkpoint', type=Path, help='same-seed EXP018 D/S initialization')
    initial.add_argument('--resume', action='store_true', help='restore the complete episode checkpoint')


def run(parser, args):
    if (args.episodes<1 or args.periods<2 or not 0<args.period_seconds<float('inf') or
            not 0<args.image_fraction<=1 or args.eval_every<1 or not 1<=args.s_batch<=25000):
        parser.error('Invalid adaptive training parameters')
    if (args.action!='train' and args.resume) or (args.action=='evaluate' and args.init_checkpoint):
        parser.error('Invalid adaptive checkpoint flags for this action')
    import torch
    from .adaptive_runner import evaluate, train, verify
    from .adaptive_storage import RunFiles, run_lock
    from .adaptive_training import Config

    checkout = Path(__file__).resolve().parents[1]
    workspace = checkout.parent if checkout.name=='source' and (checkout/'.git').is_file() else checkout
    out = RunFiles((args.prefix or workspace/'runs/results'/f'EXP019.seed{args.seed}').resolve())
    cfg = Config(seed=args.seed, periods=args.periods, period_seconds=args.period_seconds,
                 image_fraction=args.image_fraction, eval_every=args.eval_every, s_batch=args.s_batch,
                 freeze_ds=args.freeze_ds)
    torch.set_num_threads(1)
    try:
        with run_lock(out):
            if args.action=='train':
                train(cfg, out, args.episodes, resume=args.resume, init=args.init_checkpoint)
            elif args.action=='verify':
                verify(cfg, out, args.init_checkpoint)
            else:
                evaluate(out, args.seed)
    except (Exception, KeyboardInterrupt) as error:
        message = traceback.format_exc()
        sys.stderr.write(message)
        try:
            with (out/'errors.log').open('a', encoding='utf8') as file:
                file.write(message+'\n')
        except OSError:
            pass
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    return 0
