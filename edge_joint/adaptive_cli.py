"""Expose continuous deployment through the project's main CLI."""
from pathlib import Path
import re
import sys
import traceback


def add_arguments(parser):
    parser.add_argument('action', choices=('train', 'evaluate', 'verify', 'baseline'))
    seed = parser.add_mutually_exclusive_group(required=True)
    seed.add_argument('--seed', type=int, help='single run seed')
    seed.add_argument('--seeds', type=int, nargs='+', help='run multiple distinct seeds in order')
    parser.add_argument('--experiment', type=str.upper, default='EXP019',
                        help='results directory ID under runs/results (default: EXP019)')
    parser.add_argument('--prefix', type=Path,
                        help='filename prefix inside the selected experiment directory; --seeds appends .seed<seed>')
    parser.add_argument('--episodes', type=int,
                        help='absolute training target (default: 128), or required checkpoint length for evaluation')
    parser.add_argument('--periods', type=int, default=8, help='periods per continuous episode')
    parser.add_argument('--period-seconds', type=float, default=24.)
    parser.add_argument('--eval-every', type=int, default=16, help='fixed validation interval in episodes')
    parser.add_argument('--evaluate-after', action='store_true',
                        help='run independent evaluation after all requested training runs reach the target')
    parser.add_argument('--s-batch', type=int, default=128, help='original DDQN replay minibatch size')
    parser.add_argument('--s-update-timing', choices=('episode', 'period'), default='episode',
                        help='DDQN update clock with the same optimizer budget per episode')
    parser.add_argument('--d-observation', choices=('original', 'blind', 'resource'), default='original',
                        help='Gaussian Actor observation; blind masks the equally-sized resource branch')
    parser.add_argument('--d-diagnostics', action='store_true', help='record D credit and one same-state decode pair per update')
    parser.add_argument('--macro-mode', choices=('learned', 'keep', 'redeploy'), default='learned',
                        help='train the gate or fix the action at eligible periods')
    parser.add_argument('--freeze-ds', action='store_true', help='learn the macro policy with fixed D and greedy S')
    parser.add_argument('--replay-from', type=Path, help='verify an archived run by filename prefix')
    parser.add_argument('--replay-gs', type=int, nargs='+', help='blocked GS to replay from episode one')
    parser.add_argument('--fixed-only', action='store_true', help='limit the baseline to the fixed scenarios')
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument('--init-checkpoint', type=Path, help='same-seed EXP018 D/S initialization')
    initial.add_argument('--resume', action='store_true', help='restore the complete episode checkpoint')


def run(parser, args):
    target = args.episodes if args.episodes is not None else 128
    if (target<1 or args.periods<2 or not 0<args.period_seconds<float('inf') or
            args.eval_every<1 or not 1<=args.s_batch<=25000):
        parser.error('Invalid adaptive training parameters')
    batch = args.seeds is not None
    seeds = args.seeds if batch else [args.seed]
    if len(seeds) != len(set(seeds)) or any(not 0 <= seed < 2**32 for seed in seeds):
        parser.error('Seeds must be distinct integers from 0 to 4294967295.')
    if batch and (args.action not in ('train', 'evaluate') or args.prefix is None
                  or args.init_checkpoint is not None):
        parser.error('--seeds requires train or evaluate, an explicit --prefix, and no --init-checkpoint.')
    if (args.action!='train' and args.resume) or (
            args.action in ('evaluate','baseline') and args.init_checkpoint):
        parser.error('Invalid adaptive checkpoint flags for this action')
    if args.evaluate_after and args.action != 'train':
        parser.error('--evaluate-after is only used with train.')
    if args.action!='train' and args.macro_mode!='learned':
        parser.error('--macro-mode is a training setting. Evaluation reads it from the run.')
    if args.s_update_timing == 'period' and (args.action != 'train' or args.macro_mode != 'redeploy' or args.freeze_ds):
        parser.error('Period DDQN updates require joint fixed-REDEPLOY training.')
    if (args.d_observation != 'original' or args.d_diagnostics) and args.action != 'train':
        parser.error('The D observation and diagnostic flags are training settings.')
    if args.fixed_only and args.action!='baseline':
        parser.error('--fixed-only is a baseline setting.')
    if args.replay_from is not None or args.replay_gs is not None:
        if (args.action!='verify' or args.init_checkpoint is not None
                or args.replay_from is None or args.replay_gs is None):
            parser.error('Use --replay-from and --replay-gs together with verify and no --init-checkpoint.')
    if re.fullmatch(r'EXP[0-9]{3}', args.experiment) is None:
        parser.error('--experiment must be an ID such as EXP023.')
    checkout = Path(__file__).resolve().parents[1]
    results_root = checkout/'runs/results'
    results = (results_root/args.experiment).resolve()
    default_mode = 'verify' if args.action=='verify' else args.macro_mode
    base = (args.prefix or results/f'instant.{default_mode}.seed{args.seed}').resolve()
    if base.parent != results:
        parser.error(f'{args.experiment} results must use a filename prefix inside {results}')
    prefixes = [base.with_name(f'{base.name}.seed{seed}') if batch else base for seed in seeds]
    original = None
    if args.replay_from is not None:
        archived = args.replay_from.resolve()
        if archived.parent != (results_root/'EXP019').resolve() or archived == base:
            parser.error('Historical replay needs a distinct source prefix inside runs/results/EXP019.')
        original = archived

    import torch
    from .adaptive_storage import RunFiles

    runs = [RunFiles(prefix) for prefix in prefixes]
    if args.action == 'evaluate' or (args.resume and not batch):
        for out in runs:
            checkpoint = out/'resume_latest.pt'
            if not checkpoint.is_file():
                parser.error(
                    f'No {args.experiment} resume checkpoint at {checkpoint}. '
                    'Start a fresh run with train and no --resume. '
                    'For evaluation or resume pass the same --experiment and --prefix as the training run.')
    torch.set_num_threads(1)
    source = RunFiles(original) if original is not None else None
    actions = (args.action, 'evaluate') if args.evaluate_after else (args.action,)
    for action in actions:
        for seed, out in zip(seeds, runs, strict=True):
            if batch:
                print(f'{action}: seed={seed} prefix={out.prefix}', flush=True)
            code = run_one(args, action, seed, out, target, source)
            if code:
                return code
    return 0


def run_one(args, action, seed, out, target, original):
    from .adaptive_runner import baseline, evaluate, replay_blocked, train, verify
    from .adaptive_storage import RunFiles, run_lock
    from .adaptive_training import Config

    cfg = Config(seed=seed, periods=args.periods, period_seconds=args.period_seconds,
                 eval_every=args.eval_every, s_batch=args.s_batch,
                 freeze_ds=args.freeze_ds, macro_mode=args.macro_mode, s_update_timing=args.s_update_timing,
                 d_observation=args.d_observation, d_diagnostics=args.d_diagnostics)
    prefix = out.prefix
    lock = RunFiles(prefix.with_name(prefix.name+'.baseline')) if action=='baseline' else out
    try:
        with run_lock(lock):
            if action=='train':
                resume = args.resume and (args.seeds is None or (out/'resume_latest.pt').is_file())
                train(cfg, out, target, resume=resume, init=args.init_checkpoint)
            elif action=='baseline':
                baseline(out, seed, fixed_only=args.fixed_only)
            elif original is not None:
                replay_blocked(seed, original, out, args.replay_gs)
            elif action=='verify':
                verify(cfg, out, args.init_checkpoint)
            else:
                expected = target if args.evaluate_after else args.episodes
                evaluate(out, seed, expected_episodes=expected)
    except (Exception, KeyboardInterrupt) as error:
        message = traceback.format_exc()
        sys.stderr.write(message)
        try:
            with (lock/'errors.log').open('a', encoding='utf8') as file:
                file.write(message+'\n')
        except OSError:
            pass
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    return 0
