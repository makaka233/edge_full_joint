"""Expose continuous deployment through the project's main CLI."""
from pathlib import Path
import sys
import traceback


def add_arguments(parser):
    parser.add_argument('action', choices=('train', 'evaluate', 'verify'))
    parser.add_argument('--seed', type=int, choices=(20261113, 20261114), required=True)
    parser.add_argument('--prefix', type=Path,
                        help='filename prefix in runs/results/EXP019, e.g. instant.learned.seed20261113')
    parser.add_argument('--episodes', type=int, default=128, help='absolute training target')
    parser.add_argument('--periods', type=int, default=8, help='periods per continuous episode')
    parser.add_argument('--period-seconds', type=float, default=24.)
    parser.add_argument('--eval-every', type=int, default=16, help='fixed validation interval in episodes')
    parser.add_argument('--s-batch', type=int, default=128, help='original DDQN replay minibatch size')
    parser.add_argument('--macro-mode', choices=('learned', 'keep', 'redeploy'), default='learned',
                        help='train the gate or fix the action at eligible periods')
    parser.add_argument('--freeze-ds', action='store_true', help='learn the macro policy with fixed D and greedy S')
    parser.add_argument('--replay-from', type=Path, help='verify an archived run by filename prefix')
    parser.add_argument('--replay-gs', type=int, nargs='+', help='blocked GS to replay from episode one')
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument('--init-checkpoint', type=Path, help='same-seed EXP018 D/S initialization')
    initial.add_argument('--resume', action='store_true', help='restore the complete episode checkpoint')


def run(parser, args):
    if (args.episodes<1 or args.periods<2 or not 0<args.period_seconds<float('inf') or
            args.eval_every<1 or not 1<=args.s_batch<=25000):
        parser.error('Invalid adaptive training parameters')
    if (args.action!='train' and args.resume) or (args.action=='evaluate' and args.init_checkpoint):
        parser.error('Invalid adaptive checkpoint flags for this action')
    if args.action!='train' and args.macro_mode!='learned':
        parser.error('--macro-mode is a training setting. Evaluation reads it from the run.')
    if args.replay_from is not None or args.replay_gs is not None:
        if (args.action!='verify' or args.init_checkpoint is not None
                or args.replay_from is None or args.replay_gs is None):
            parser.error('Use --replay-from and --replay-gs together with verify and no --init-checkpoint.')
    import torch
    from .adaptive_runner import evaluate, replay_blocked, train, verify
    from .adaptive_storage import RunFiles, run_lock
    from .adaptive_training import Config

    checkout = Path(__file__).resolve().parents[1]
    results = (checkout/'runs/results/EXP019').resolve()
    default_mode = 'verify' if args.action=='verify' else args.macro_mode
    prefix = (args.prefix or results/f'instant.{default_mode}.seed{args.seed}').resolve()
    if prefix.parent != results:
        parser.error(f'EXP019 results must use a filename prefix inside {results}')
    original = None
    if args.replay_from is not None:
        archived = args.replay_from.resolve()
        if archived.parent != results or archived == prefix:
            parser.error('Historical replay needs a distinct source prefix inside runs/results/EXP019.')
        original = RunFiles(archived)
    out = RunFiles(prefix)
    checkpoint = out/'resume_latest.pt'
    if (args.resume or args.action=='evaluate') and not checkpoint.is_file():
        parser.error(
            f'No EXP019 resume checkpoint at {checkpoint}. '
            'Start a fresh run with train and no --resume. '
            'For evaluation or resume pass the same --prefix as the training run.')
    cfg = Config(seed=args.seed, periods=args.periods, period_seconds=args.period_seconds,
                 eval_every=args.eval_every, s_batch=args.s_batch,
                 freeze_ds=args.freeze_ds, macro_mode=args.macro_mode)
    torch.set_num_threads(1)
    try:
        with run_lock(out):
            if args.action=='train':
                train(cfg, out, args.episodes, resume=args.resume, init=args.init_checkpoint)
            elif original is not None:
                replay_blocked(args.seed, original, out, args.replay_gs)
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
