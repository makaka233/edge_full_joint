"""Single command-line entry for smoke, training, testing and plots."""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import replace
from pathlib import Path
from .world import make_world
from .training import TrainConfig, new_actor, train, evaluate, load_checkpoint
from .rollout import sample_layout

def cli():
    p=argparse.ArgumentParser(description="Edge Agent-D clean BUILD-only research baseline")
    sub=p.add_subparsers(dest="command",required=True)

    placement = sub.add_parser('placement-train', help='Bounded latent deployment and original-S training')
    placement.add_argument('--seed', type=int, default=20261113)
    placement.add_argument('--gs', type=int, default=64)
    placement.add_argument('--eval-every', type=int, default=8)
    placement.add_argument('--fixed-scenes', type=int, default=4)
    placement.add_argument('--holdout-scenes', type=int, default=8)
    placement.add_argument('--mode', choices=('d-only', 's-only', 'joint', 'periodic'), default='d-only')
    placement.add_argument('--periods', type=int, default=1, help='Use >1 only for periodic KEEP/BUILD')
    placement.add_argument('--out', type=Path, required=True)

    smoke=sub.add_parser("smoke",help="One actual policy build + physical evaluation")
    smoke.add_argument("--seed",type=int,default=20261113)
    smoke.add_argument("--scene",type=int,default=0)
    smoke.add_argument("--unrestricted",action="store_true")

    run=sub.add_parser("train",help="REINFORCE baseline; --updates is not macro GS")
    run.add_argument("--seed",type=int,default=20261113)
    run.add_argument("--updates",type=int,default=64)
    run.add_argument("--rollouts",type=int,default=4)
    run.add_argument("--eval-every",type=int,default=8)
    run.add_argument("--fixed-scenes",type=int,default=2)
    run.add_argument("--fixed-draws",type=int,default=2)
    run.add_argument("--holdout-scenes",type=int,default=4)
    run.add_argument("--holdout-draws",type=int,default=2)
    run.add_argument("--max-actions",type=int,default=1100)
    run.add_argument("--unrestricted",action="store_true")
    run.add_argument("--out",type=Path,default=Path("runs/seed20261113"))

    ev=sub.add_parser("evaluate",help="Load local trusted checkpoint and evaluate")
    ev.add_argument("--checkpoint",type=Path,required=True)
    ev.add_argument("--split",choices=("fixed","holdout"),default="holdout")
    ev.add_argument("--out",type=Path,default=Path("runs/manual_evaluation"))

    pl=sub.add_parser("plot",help="Plot raw unfiltered Reward/E2E and fixed evaluations")
    pl.add_argument("--run-dir",type=Path,required=True)

    # Separate, fully executable scheduling and coupled D/S training entrypoints.
    for command,mode in (("s-train","s-only"),("joint-train","joint")):
        sp=sub.add_parser(command,help=f"Real-physics {mode} training with masked Dueling DDQN")
        sp.add_argument("--seed",type=int,default=20261113)
        sp.add_argument("--updates",type=int,default=64,
                        help="D/S macro update iterations; NOT 64 simulated full days")
        sp.add_argument("--rollouts",type=int,default=2)
        sp.add_argument("--s-updates-per-step",type=int,default=2)
        sp.add_argument("--s-batch",type=int,default=64)
        sp.add_argument("--eval-every",type=int,default=8)
        sp.add_argument("--fixed-scenes",type=int,default=2)
        sp.add_argument("--fixed-draws",type=int,default=2)
        sp.add_argument("--holdout-scenes",type=int,default=4)
        sp.add_argument("--holdout-draws",type=int,default=2)
        sp.add_argument("--max-actions",type=int,default=1100)
        sp.add_argument("--trace-s-transitions",action="store_true")
        sp.add_argument("--out",type=Path,default=Path("runs")/command)
    je=sub.add_parser("joint-evaluate",help="Evaluate locally generated joint/s-only checkpoint")
    je.add_argument("--checkpoint",type=Path,required=True)
    je.add_argument("--split",choices=("fixed","holdout"),default="holdout")
    je.add_argument("--out",type=Path,default=Path("runs/joint_manual_eval"))
    js=sub.add_parser("joint-smoke",help="One joint real-physics rollout with real S transitions")
    js.add_argument("--seed",type=int,default=20261113)
    js.add_argument("--scene",type=int,default=0)
    per=sub.add_parser("periodic-train",help="Experimental persistent KEEP/BUILD + Agent-S joint learning")
    per.add_argument("--seed",type=int,default=20261113)
    per.add_argument("--updates",type=int,default=64)
    per.add_argument("--periods",type=int,default=3)
    per.add_argument("--episodes-per-update",type=int,default=2)
    per.add_argument("--eval-every",type=int,default=8)
    per.add_argument("--fixed-episodes",type=int,default=1)
    per.add_argument("--holdout-episodes",type=int,default=2)
    per.add_argument("--s-updates-per-update",type=int,default=2)
    per.add_argument("--max-actions",type=int,default=1100)
    per.add_argument("--out",type=Path,default=Path("runs/periodic"))
    pe=sub.add_parser("periodic-evaluate",help="Evaluate a locally generated periodic checkpoint")
    pe.add_argument("--checkpoint",type=Path,required=True)
    pe.add_argument("--split",choices=("fixed","holdout"),default="holdout")
    pe.add_argument("--out",type=Path,default=Path("runs/periodic_manual_eval"))
    adaptive=sub.add_parser("adaptive",help="Continuous KEEP/REDEPLOY and original DDQN")
    from edge_joint.adaptive_cli import add_arguments
    add_arguments(adaptive)
    args=p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    if args.command=='adaptive':
        from edge_joint.adaptive_cli import run as run_adaptive
        raise SystemExit(run_adaptive(adaptive,args))
    elif args.command=='placement-train':
        from edge_joint.placement_training import PlacementConfig, train as train_placement
        settings = vars(args).copy()
        out = settings.pop('out')
        settings.pop('command')
        print(json.dumps(train_placement(PlacementConfig(**settings), out), indent=2))
    elif args.command=="smoke":
        seed=args.seed
        actor=new_actor(seed)
        record,_=sample_layout(seed=seed,scene=args.scene,split="train",actor=actor,
                 policy_seed=seed+1001,physical_seed=seed+2002,
                 cpu_finish_guard=not args.unrestricted)
        print(json.dumps({k:v for k,v in record.items() if k not in ("actions","matrix_packed_hex")},
                         ensure_ascii=False,indent=2))
    elif args.command=="train":
        c=TrainConfig(seed=args.seed,updates=args.updates,rollouts=args.rollouts,
                      eval_every=args.eval_every,fixed_scenes=args.fixed_scenes,
                      fixed_draws=args.fixed_draws,holdout_scenes=args.holdout_scenes,
                      holdout_draws=args.holdout_draws,max_actions=args.max_actions,
                      cpu_finish_guard=not args.unrestricted)
        print(json.dumps(train(c,args.out),indent=2))
    elif args.command=="evaluate":
        c,actor,state=load_checkpoint(args.checkpoint)
        args.out.mkdir(parents=True,exist_ok=True)
        print(json.dumps(evaluate(c,actor,state["update"],args.split,args.out,"manual"),indent=2))
    elif args.command in ("s-train","joint-train"):
        from edge_joint.training import JointConfig,train as train_both
        c=JointConfig(mode="s-only" if args.command=="s-train" else "joint",
            seed=args.seed,updates=args.updates,rollouts=args.rollouts,
            s_updates_per_step=args.s_updates_per_step,s_batch=args.s_batch,
            eval_every=args.eval_every,fixed_scenes=args.fixed_scenes,
            fixed_draws=args.fixed_draws,holdout_scenes=args.holdout_scenes,
            holdout_draws=args.holdout_draws,max_actions=args.max_actions,
            trace_s_transitions=args.trace_s_transitions)
        print(json.dumps(train_both(c,args.out),indent=2))
    elif args.command=="joint-evaluate":
        from edge_joint.training import load_checkpoint,evaluate as eval_both
        c,d,s,state=load_checkpoint(args.checkpoint)
        args.out.mkdir(parents=True,exist_ok=True)
        print(json.dumps(eval_both(c,d,s,state['step'],args.split,args.out,'manual'),indent=2))
    elif args.command=="joint-smoke":
        from edge_joint.training import JointConfig,create_agents,execute
        c=JointConfig(seed=args.seed)
        d,s=create_agents(c)
        row,logps,transitions=execute(c,d,s,args.scene,'train',0,1,training=True,epsilon=.2)
        print(json.dumps({'reward':row['reward'],'e2e':row['penalized_e2e'],
             'layout':row['matrix_sha256'],'s_transitions':len(transitions),
             's_obs_shape':list(transitions[0].obs.shape) if transitions else None,
             's_legal_actions':all(t.mask[t.action] for t in transitions),
             'd_log_probs':len(logps)},indent=2))
    elif args.command=="periodic-train":
        from edge_joint.periodic import PeriodicConfig,train as train_periodic
        c=PeriodicConfig(seed=args.seed,updates=args.updates,periods=args.periods,
            episodes_per_update=args.episodes_per_update,eval_every=args.eval_every,
            fixed_episodes=args.fixed_episodes,holdout_episodes=args.holdout_episodes,
            s_updates_per_update=args.s_updates_per_update,max_actions=args.max_actions)
        print(json.dumps(train_periodic(c,args.out),indent=2))
    elif args.command=="periodic-evaluate":
        from edge_joint.periodic import load_checkpoint,evaluate as eval_periodic
        c,d,m,s,state=load_checkpoint(args.checkpoint)
        args.out.mkdir(parents=True,exist_ok=True)
        print(json.dumps(eval_periodic(c,d,m,s,state['step'],args.split,args.out,'manual'),indent=2))
    elif args.command=="plot":
        from .plotting import plots
        print(*plots(args.run_dir),sep="\n")

if __name__=="__main__":
    cli()
