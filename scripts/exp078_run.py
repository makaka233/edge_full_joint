"""Random-initialized joint D/S training under fixed deployment cadences."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from itertools import islice, product
from pathlib import Path
from statistics import mean
from time import time

from scripts import exp065_run as phys
from scripts import exp043_run as history
from scripts import exp075_run as past
from scripts.exp055_joint import inputs, projection_matches
from scripts.exp060_run import StructuredConfig, initial_proof, settings, weights_digest
from scripts.exp047_report import plt
from comparison.report import records, count_actions
from comparison.run import MeasuredMultistageSimulator
from edge_s.agent import resource_advantage
from edge_joint.adaptive_physics import capacity_ok
from edge_joint.adaptive_runner import read_checkpoint, train
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from edge_joint.adaptive_training import RolloutControl, components, decode_placement, episode, load_model
from edge_joint.inventory_joint import load_scale
from scripts.exp019_report import unpack
from scripts.exp027_run import verify_periods

import numpy as np
import torch

ROOT = Path('runs/results/EXP078')
REVIEW = Path('research/EXP078/AI_RESEARCH_REVIEW.md')
BASE = '0e50c7da2b4e10bea13d9292645e75d427e34a03'
ARMS = {'A':1,'B':4}
SLOTS = 8
EPISODES = 63
FIXED = (0,8,16,32,48,63)
CELLS = ('random','joint','learned_d_prior','reference_learned_s','reference_prior','random_d_learned_s')
TREATMENTS = (('natural','stable',None),('regional','regional',None),
              ('service','service',None),('pressure','compound',.70))


@dataclass(frozen=True)
class CadenceConfig(StructuredConfig):
    redeploy_every: int = 1


def config(base,seed,arm):
    cfg=settings(base,seed,'B')
    data=asdict(cfg)
    if cfg.freeze_ds or cfg.macro_mode!='redeploy' or cfg.periods!=SLOTS or cfg.period_seconds!=24.:
        raise ValueError('Historical native joint training recipe changed')
    if not cfg.s_resource_q_prior or cfg.s_update_timing!='period' or cfg.s_two_step_watkins:
        raise ValueError('Unexpected Agent-S architecture or update rule')
    return CadenceConfig(**data,redeploy_every=ARMS[arm])


def out_for(root,seed,arm):
    return RunFiles(root/f'{arm}.seed{seed}',nested=True)


def case_for(seed,split,tag,first,pressure=None):
    key=('EXP078-v1',seed,split,tag)
    return dict(seed=seed,tag=tag,first=first,pressure=pressure,
                scene=phys.stable_seed(*key,'scene'),population=phys.stable_seed(*key,'population'))


def register(root,base,smoke):
    seeds=[20261115] if smoke else list(phys.SEEDS)
    episodes=2 if smoke else EPISODES
    blocks={}
    vals={}
    held=[]
    flavor='smoke' if smoke else 'formal'
    for seed in seeds:
        native=[case_for(seed,flavor+'-train-path',i,kind)
                for i,kind in enumerate(('stable','regional','service','compound'))]
        blocks[str(seed)]=[dict(path=native[i//16],offset=(i%16)*SLOTS,
                               scene=phys.stable_seed('EXP078',flavor,seed,i,'episode'))
                           for i in range(episodes)]
        vals[str(seed)]=[dict(path=case_for(seed,flavor+'-fixed',j,kind),offset=offset)
                         for j,(kind,offset) in enumerate((('stable',8),('compound',20)))]
        for h in vals[str(seed)]:
            h['scene']=h['path']['scene']
        held.extend(dict(case_for(seed,flavor+'-held',tag,first,pressure),slots=8 if smoke else 32)
                    for tag,first,pressure in (TREATMENTS[-1:] if smoke else TREATMENTS))
    defaults=[dict(case) for case in held]
    for s in seeds:
        unique={r['path']['scene']:r['path'] for r in blocks[str(s)]}
        defaults.extend(unique.values())
        defaults.extend(v['path'] for v in vals[str(s)])
    keys=list(past.external_keys(defaults))+[r['scene'] for block in blocks.values() for r in block]
    old={key for n in range(65,78) for f in [Path(f'runs/results/EXP{n:03d}/protocol.json')]
         if f.exists() for key in past.external_keys(phys.read(f)['plan'])}
    if len(set(keys))!=len(keys) or old.intersection(keys):
        raise ValueError('Traffic/population key collision')
    plan=dict(parent=BASE,smoke=smoke,seeds=seeds,episodes=episodes,blocks=blocks,fixed=vals,held=held,
              cadence=ARMS,demand=phys.read(phys.CONFIG),original=phys.fingerprint(base),
              fixed_milestones=[0,1,2] if smoke else list(FIXED),decoder='native-GS-varying',
              protocol=phys.digest(REVIEW.with_name('PROTOCOL.md')))
    p=root/'protocol.json'
    if p.exists():
        if phys.read(p)['plan']!=plan:
            raise ValueError('EXP078 scenario registration changed')
    else:
        phys.write(p,dict(plan=plan,source=phys.source()))
    return plan


def make_factory(base,plan,seed):
    unique={r['scene']:r for r in [*plan['blocks'][str(seed)],*plan['fixed'][str(seed)]]}
    def factory(model_seed,scene,split):
        if model_seed!=seed:
            raise ValueError('Training factory called with wrong model seed')
        case=unique[scene]
        e=phys.environment(base,plan,case['path'],split)
        for _ in range(case['offset']):
            e.w.advance()
        return e
    return factory


def audit_rows(e,rows,cadence,*,reference=False):
    verify_periods(e,rows)
    error=history.verify_work(rows)
    for r in rows:
        m=unpack(e,r['active_matrix'])
        expected=('BOOTSTRAP' if r['period']==1 else 'REDEPLOY'
                  if not reference and (r['period']-1)%cadence==0 else 'KEEP')
        if (r['action']!=expected or m.shape!=(28,32) or not m.any(1).all()
                or not capacity_ok(m,e.spec,e.capacity) or r['s_decisions']<=0):
            raise ValueError('A cadence, deployment or node decision ledger is invalid')
        if expected=='KEEP' and (r['active_matrix']!=r['old_matrix'] or r['switch_cost']):
            raise ValueError('KEEP mutated the matrix or paid a switch penalty')
        past.ledger(e,r)
    return error


def train_one(root,base,plan,seed,arm,target=None,*,label=None,cfg_override=None,s_pair_credit=None):
    target=plan['episodes'] if target is None else target
    out=out_for(root,seed,label or arm)
    cfg=config(base,seed,arm) if cfg_override is None else cfg_override
    fixed_s=getattr(cfg,'s_fixed_resource_policy',False)
    cadence=cfg.redeploy_every
    origin=initial_proof(base,seed,'B')
    factory=make_factory(base,plan,seed)
    stamp=phys.fingerprint(plan)
    resume=out/'resume_latest.pt'
    def start(state):
        if state['origin']!={'kind':'random'} or state['episode'] or state['d_optimizer']['state'] or (
                state['s']['updates'] or state['s']['optimizer']['state']):
            raise ValueError('Fresh D/S run did not start from random weights and empty optimizer state')
        if any(weights_digest(state[key])!=origin[field] for key,field in (('d','actor'),('s_online','s_online'))):
            raise ValueError('Initial weights did not match registered independent constructors')
        phys.write(out/'initial.json',origin)
    with run_lock(out):
        train(cfg,out,target,resume=resume.is_file(),environment_id=stamp,world_factory=factory,
              simulator_factory=MeasuredMultistageSimulator,initial_snapshot=start,
              train_scenes=[r['scene'] for r in plan['blocks'][str(seed)]],
              fixed_scenes=tuple(r['scene'] for r in plan['fixed'][str(seed)]),
              evaluation_episodes=plan['fixed_milestones'],s_pair_credit=s_pair_credit)
    state=read_checkpoint(resume)
    if (state['episode'],state['d_updates'],state['s']['updates'])!=(
            target,target,0 if fixed_s else target*SLOTS*cfg.s_updates_per_gs):
        raise ValueError('Actual D/S optimizer budget differs from registration')
    if state['origin']!={'kind':'random'} or not projection_matches(read_checkpoint(out/'last.pt'),state):
        raise ValueError('Final parameters do not match the resumable random-origin run')
    s_changed=weights_digest(state['s_online'])!=origin['s_online']
    if (weights_digest(state['d'])==origin['actor'] or not state['d_optimizer']['state']
            or s_changed==fixed_s or bool(state['s']['optimizer']['state'])==fixed_s):
        raise ValueError('Actual D/S optimization is missing')
    rows=records(out/'train.jsonl')
    d_actions=s_actions=0
    verify=0.
    ups=records(out/'updates.jsonl')
    for i in range(target):
        trial=list(islice(rows,SLOTS))
        u=next(ups,None)
        if [r['gs'] for r in trial]!=list(range(i*SLOTS+1,(i+1)*SLOTS+1)):
            raise ValueError('Missing or duplicate GS in training ledger')
        if u is None or u['d_samples']!=SLOTS//cadence or u['macro_policy_samples'] or u['d_change']<=0. or any(
                (u[k]!=0 if fixed_s else u[k]<=0) for k in ('s_change','s_target_change','replay_size')):
            raise ValueError('Invalid D/S gradients, Replay or cadence sample count')
        d_actions+=u['d_samples']
        s_actions+=sum(map(count_actions,trial))
        env=factory(seed,trial[0]['scene'],'train')
        verify=max(verify,audit_rows(env,trial,cadence))
        for r in trial:
            if r['action']=='KEEP' and (r['d_logp'] is not None or r['d_latent'] is not None):
                raise ValueError('KEEP generated a fictitious D credit')
    if next(rows,None) is not None or next(ups,None) is not None or state['s_actions']!=s_actions:
        raise ValueError('Action or update count differs from physical ledger')
    proof=dict(seed=seed,arm=arm,gs=target*SLOTS,origin=origin,d_updates=state['d_updates'],
               s_updates=state['s']['updates'],d_actions=d_actions,
               s_actions=state['s_actions'],max_work_error=verify,wall_seconds=state['wall_seconds'],
               checkpoints=sorted({r['episode']*SLOTS for r in records(out/'fixed.jsonl')}),resume=phys.digest(resume),
               source=phys.source())
    phys.write(out/'proof.json',proof)
    return proof


def same_tree(a,b):
    if type(a)!=type(b):
        return False
    if isinstance(a,torch.Tensor):
        return torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(same_tree(u,v) for u,v in zip(a,b,strict=True))
    return a==b


class PriorScheduler:
    """The fixed resource-prior component of EXP043 masked scheduling."""
    version=0
    updates=0

    def __init__(self,agent):
        self.config=agent.config

    def scheduler(self,rng,epsilon):
        @torch.no_grad()
        def choose(obs,mask):
            q=resource_advantage(torch.as_tensor(obs[None]),self.config.latency_ref_seconds)[0].numpy()
            legal=np.flatnonzero(mask)
            return int(legal[q[legal].argmax()])
        return choose


def summary(rows):
    m=phys.summarize(rows)
    m.update(severe=sum(r['reward_raw']<=-20. for r in rows),
             hold=mean(r['holding_cost'] for r in rows),
             switch=sum(r['switch_cost'] for r in rows),
             delay=mean(-r['reward_raw']-r['holding_cost']-r['switch_cost'] for r in rows),
             d_actions=sum(r['action'] in ('BOOTSTRAP','REDEPLOY') for r in rows),
             s_actions=sum(map(count_actions,rows)),
             replicas=mean(sum(r['active_replicas']) for r in rows))
    return m


@torch.inference_mode()
def evaluate_case(root,base,plan,seed,arm,case):
    output=root/f"held.{seed}.{arm}.{case['tag']}.json"
    out=out_for(root,seed,arm)
    checkpoint=out/'last.pt'
    stamp=phys.fingerprint(dict(plan=phys.fingerprint(plan),sha=phys.digest(checkpoint),case=case))
    if output.exists():
        doc=phys.read(output)
        if doc['stamp']!=stamp:
            raise ValueError('Saved independent output differs from registered endpoint')
        return doc
    trained=read_checkpoint(checkpoint)
    if trained['origin']!={'kind':'random'} or trained['environment_id']!=phys.fingerprint(plan):
        raise ValueError('Independent endpoint did not originate from random training')
    cfg=replace(config(base,seed,arm),freeze_ds=True,periods=case['slots'])
    loads=(case['pressure'],)*cfg.periods
    control=RolloutControl(environment_seed=case['population'],loads=loads,sample_d=False,epsilon=0.)
    neutral,s0,*_=components(cfg)
    layout_env=phys.environment(base,plan,case,'holdout')
    if case['pressure'] is not None:
        layout_env.w.scale*=case['pressure']/load_scale(layout_env)
    key=control.decode_key(case['population'],'holdout',case['scene'],0)
    reference=decode_placement(cfg,layout_env,neutral(layout_env).numpy(),key).matrix
    if s0.updates or not reference.any(1).all() or not capacity_ok(reference,layout_env.spec,layout_env.capacity):
        raise ValueError('Invalid preregistered physical reference')
    arms={}
    for name in CELLS:
        tick=time()
        d,s,_,_,gate,_=components(cfg)
        static=name in ('reference_learned_s','reference_prior')
        if name in ('joint','learned_d_prior'):
            load_model(trained,d,s,gate)
        elif name in ('random_d_learned_s','reference_learned_s'):
            s.online.load_state_dict(trained['s_online'])
        if name.endswith('prior'):
            s=PriorScheduler(s)
        env=phys.environment(base,plan,case,'holdout')
        layout=[reference]+[None]*(cfg.periods-1) if static else None
        rows=episode(cfg,env,d,s,gate,case['scene'],'holdout',1,control=control,
                     recorded_matrices=layout,simulator_factory=MeasuredMultistageSimulator)[0]
        audit_rows(env,rows,ARMS[arm],reference=static)
        if arms:
            phys.same_flow(arms['random']['rows'],rows)
        arms[name]=dict(rows=rows,summary=summary(rows),seconds=time()-tick)
        print(f"EXP078 holdout {seed}/{arm}/{case['tag']}/{name}: {time()-tick:.1f}s",flush=True)
    for a,b in (('joint','learned_d_prior'),('reference_learned_s','reference_prior'),
                ('random','random_d_learned_s')):
        if any(x['active_matrix']!=y['active_matrix'] for x,y in zip(arms[a]['rows'],arms[b]['rows'],strict=True)):
            raise ValueError('S intervention altered the paired D placement')
    result=dict(stamp=stamp,seed=seed,arm=arm,case=case,arms=arms,source=phys.source())
    phys.write(output,result)
    return result


def seed_run(root,base,plan,seed):
    torch.set_num_threads(1)
    a,b=(train_one(root,base,plan,seed,k) for k in ARMS)
    if a['origin']!=b['origin']:
        raise ValueError('A/B do not share the model seed')
    rr=[list(records(out_for(root,seed,k)/'train.jsonl')) for k in ARMS]
    phys.same_flow(*rr)
    for case in plan['held']:
        if case['seed']==seed:
            a,b=(evaluate_case(root,base,plan,seed,arm,case) for arm in ARMS)
            phys.same_flow(a['arms']['joint']['rows'],b['arms']['joint']['rows'])


def smoke(root,base,plan):
    seed=plan['seeds'][0]
    start=time()
    a=train_one(root,base,plan,seed,'A',1)
    train_one(root,base,plan,seed,'B',1)
    b=train_one(root,base,plan,seed,'B')
    train_one(root,base,plan,seed,'B',label='continuous')
    left,right=(read_checkpoint(out_for(root,seed,k)/'resume_latest.pt') for k in ('B','continuous'))
    for k in ('d','s','d_optimizer','gate','gate_optimizer','replay','rng','d_credit_offset'):
        if not same_tree(left[k],right[k]):
            raise ValueError(f'Resumed D/S state diverged in {k}')
    if a['origin']!=b['origin']:
        raise ValueError('A/B did not share their true random initialization')
    evidence=evaluate_case(root,base,plan,seed,'B',plan['held'][0])
    phys.write(root/'checks.json',dict(a=a,b=b,restore=True,arms=len(evidence['arms']),seconds=time()-start,
                                      source=phys.source()))


def fmt(x,n=4):
    return '—' if x is None else f'{x:.{n}f}'


def metrics_str(x,keys,n=4):
    return '/'.join(fmt(x[k],n) for k in keys.split())


def committed_training(root=None):
    root=ROOT if root is None else root
    result={}
    for file in sorted(root.glob('*.seed*/proof.json')):
        proof=phys.read(file)
        out=RunFiles(file.parent,nested=True)
        online=list(records(out/'train.jsonl'))
        updates=list(records(out/'updates.jsonl'))
        fixed={}
        for row in records(out/'fixed.jsonl'):
            fixed.setdefault(int(row['episode'])*SLOTS,[]).append(row)
        result[proof['seed'],proof['arm']]=dict(proof=proof,online=online,updates=updates,
                                               fixed={k:summary(v) for k,v in fixed.items()})
    return result


def delta(rows,a,b,k):
    v=[(r['arms'][a]['summary'][k],r['arms'][b]['summary'][k]) for r in rows]
    return mean(x-y for x,y in v) if all(x is not None and y is not None for x,y in v) else None


def training_outcome(train,docs,arm,seeds):
    group=[r for r in docs if r['arm']==arm]
    if len(group)!=len(seeds)*len(TREATMENTS) or any((s,arm) not in train for s in seeds):
        return []
    fixed=[]
    independent=[]
    deteriorated=[]
    for seed in seeds:
        f=train[seed,arm]['fixed']
        a,b=f[0],f[504]
        e=b['e2e']-a['e2e'] if a['e2e'] is not None and b['e2e'] is not None else None
        fixed.append((b['reward']-a['reward'],e))
        cells=[r for r in group if r['seed']==seed]
        independent.append(tuple(delta(cells,'joint','random',k) for k in ('reward','e2e')))
        ref=f[256]
        broken=any(all(f[i][k]<ref[k]-.30 if k=='reward' else f[i][k] is None or (
                       ref[k] is not None and f[i][k]>ref[k]*1.05)
                       for i in (384,504)) for k in ('reward','e2e'))
        if broken:
            deteriorated.append(seed)
    ok=lambda x:sum(r>0 and e is not None and e<0 for r,e in x)
    improvement=lambda x:all(e is not None for _,e in x) and mean(r for r,_ in x)>0 and mean(e for _,e in x)<0
    bad=[]
    for r in group:
        if r['case']['tag']!='pressure':
            continue
        a,b=(r['arms'][k]['summary'] for k in ('random','joint'))
        failures=[k for k in ('severe','censored','terminal') if b[k]>a[k]]
        if failures:
            bad.append((r['seed'],failures))
    success=ok(fixed)>=2 and ok(independent)>=2 and improvement(fixed) and improvement(independent) and not bad
    return [f"{arm} 固定验证/独立双改善为 {ok(fixed)}/{len(seeds)} 与 {ok(independent)}/{len(seeds)}。"
            f"逐种子固定 ΔR/ΔE2E 为 `{fixed}`，独立为 `{independent}`。",
            f"{arm} 压力风险上升记录 `{bad}`，后期持续回退种子 `{deteriorated}`。"
            +('已通过登记的性能门槛。' if success else '未通过登记的性能门槛。')
            +('已确认存在后期稳定性限制。' if deteriorated else '未命中登记的持续退化判据。')]


def verified_archives():
    a=phys.read(history.ROOT/'summary.json')
    base=phys.read(history.ROOT/'protocol.json')['plan']
    snaps={}
    for s in phys.SEEDS:
        init,last=(read_checkpoint(history.files(s)/f'{k}.pt') for k in ('initial','last'))
        if init['episode']!=0 or last['episode']!=63 or init['origin']!={'kind':'random'} or (
                init['config']!=base['configs'][str(s)] or last['config']!=init['config']):
            raise ValueError('Historical EXP043 model evidence differs')
        snaps[str(s)]={'initial':phys.digest(history.files(s)/'initial.pt'),
                       'last':phys.digest(history.files(s)/'last.pt')}
    return dict(verified=snaps,acceptance=a['acceptance'],overall=a['independent']['overall'],
                physics=base['physics'])


def figures(train,docs):
    paths=[]
    def save(fig,name):
        f=ROOT/name
        fig.savefig(f,dpi=135,bbox_inches='tight')
        plt.close(fig)
        paths.append(f)
    for seed in sorted({s for s,_ in train}):
        fig,axes=plt.subplots(4,2,figsize=(12,10),sharex=True)
        fig2,level=plt.subplots(2,2,figsize=(10,6),sharex=True)
        for (s,a),r in train.items():
            if s!=seed:
                continue
            online=r['online']
            series=[([q['gs'] for q in online],y,label) for y,label in (
                ([x['reward_raw'] for x in online],'Raw Reward'),
                ([x['e2e_completed'] for x in online],'Completed E2E'),
                ([x['p99'] for x in online],'P99'),
                ([x['censored'] for x in online],'Censored'),
                ([max(k['work_pending'] for k in x['cpu']) for x in online],'Max CPU backlog (CU)'),
                ([sum(x['active_replicas']) for x in online],'Active replicas'),
                ([x['s_boundary']['grad_norm'] for x in online],'S gradient'))]
            series.append(([x['gs'] for x in r['updates']],[x['d_gradient'] for x in r['updates']],'D gradient'))
            for ax,(xs,ys,label) in zip(axes.flat,series,strict=True):
                ax.plot(xs,ys,linewidth=.8,label=a)
                ax.set(ylabel=label,xlabel='Training GS')
            for ax,k in zip(level.flat,('reward','e2e','p99','censored'),strict=True):
                ax.plot(list(r['fixed']),[x[k] for x in r['fixed'].values()],'.-',label=a)
                ax.set(xlabel='Training GS at fixed evaluation',ylabel=k)
        axes[0,0].legend()
        level[0,0].legend()
        save(fig,f'fig.train.{seed}.png')
        save(fig2,f'fig.fixed.{seed}.png')
    for seed,arm in sorted({(d['seed'],d['arm']) for d in docs}):
        cs=[x for x in docs if x['seed']==seed and x['arm']==arm]
        fig,axes=plt.subplots(len(cs),3,figsize=(13,2.7*len(cs)),squeeze=False)
        for d,level in zip(cs,axes,strict=True):
            for name,r in d['arms'].items():
                xs=[x['period'] for x in r['rows']]
                for ax,k in zip(level,('reward_raw','e2e_completed','p99'),strict=True):
                    ax.plot(xs,[x[k] for x in r['rows']],label=name,linewidth=.8)
                    ax.set(ylabel=d['case']['tag']+' '+k,xlabel='Held-out GS')
        axes[0,0].legend(fontsize=6)
        save(fig,f'fig.held.{seed}.{arm}.png')
    return paths


def render():
    train=committed_training()
    docs=[phys.read(p) for p in sorted(ROOT.glob('held.*.json'))]
    smoke=ROOT/'smoke'/'checks.json'
    archival=ROOT/'historical.json'
    if not archival.exists():
        phys.write(archival,verified_archives())
    h=phys.read(archival)
    lines=['# EXP078 联合 D/S 固定周期训练','',
           f'指定基线 `{BASE}`。协议见 [PROTOCOL.md](PROTOCOL.md)。'
           'EXP077 的固定动作、风险抽样与经济性负结果归档于 6cb9172。','',
           f'正式训练 {len(train)}/6 臂，独立评价 {len(docs)}/24 组。','',
           '## 模型、物理与随机起点','',
           'Agent-D 在随机结构化 Actor 上采样 92 维 Gaussian 动作，解码为 28×32 合法矩阵。'
           'D 的策略梯度使用原有折扣回报、滞后校准和价值基线。固定时机时仅更新该价值基线，'
           '不执行 Gate 策略更新。Agent-S 为资源先验掩码 dueling Double DQN，逐请求逐阶段选点。'
           'S 使用一步 Replay、目标网络 Polyak、每 GS 八次优化，保留 CPU finish_pf、有线 KKT 和无线。','',
           '核实 EXP043 随机起点与 504 GS 端点模型、world/experiment 配置和历史验收。'
           'EXP043 五种子固定 R 从 −8.747 升至 −7.097，完成 E2E 从 0.8681 s 降至 0.6908 s。'
           f"历史独立联合端点的平均 ΔR={h['overall']['joint']['reward']:+.3f}，"
           f"ΔE2E={h['overall']['joint']['e2e']:+.4f} s。"
           'EXP061 验证的结构化 D 分支及原信用语义用于本轮。'
           '历史模型和配置的核验指纹见 `runs/results/EXP078/historical.json`。'
           f"历史验收 A/B 为 {h['acceptance']['A']}/{h['acceptance']['B']}。",'',
           '两个组都使用原逐 GS 变键；A 每槽选择部署，B 每四槽选择部署。'
           '从同种子的同一随机 D/S 起点开始，真实 D 决策和每次 Adam 的训练样本量不同。'
           '训练宏观需求按本地连续分段复原，物理队列只在 8 GS 回合内连续；32 GS 独立评价全程连续。'
           '每宏观槽只推进 24 秒原物理事件。','']
    if (ROOT/'protocol.json').exists():
        lines+=['正式登记提交 `'+phys.read(ROOT/'protocol.json')['source']['commit']+'`。','']
    if smoke.exists():
        r=phys.read(smoke)
        lines+=[f"真实 D/S 联合烟测完成，B 断点与连续训练的优化、Replay、目标网络、模型及 RNG 一致。"
                f"独立物理消融 {r['arms']} 臂。总耗时 {r['seconds']:.1f} 秒，"
                f"A/B 在线与验证写盘累计秒 {r['a']['wall_seconds']:.1f}/{r['b']['wall_seconds']:.1f}。",'']
    lines+=['## 正式训练、损失与固定验证','',
            '| Seed/臂 | GS | D 实际决策/更新 | S 实际动作/更新 | D/S 损失均值 | D/S 梯度均值 | 在线 R/E2E/P99 | 严重/截尾/末端 | CPU/链路峰值 | 原始最坏 R | 训练秒 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (seed,a),d in train.items():
        p,u,rows=d['proof'],d['updates'],d['online']
        m=summary(rows)
        grads=[v['s_boundary']['grad_norm'] for v in rows if v['s_boundary']['grad_norm'] is not None]
        lines.append(f"| {seed}/{a} | {p['gs']} | {p['d_actions']}/{p['d_updates']} | {p['s_actions']}/{p['s_updates']} | "
                     f"{mean(x['d_loss'] for x in u):.4f}/{mean(x['s_loss'] for x in u):.4f} | "
                     f"{mean(x['d_gradient'] for x in u):.4f}/{mean(grads):.4f} | "
                     f"{metrics_str(m,'reward e2e p99')} | {m['severe']}/{m['censored']}/{m['terminal']} | "
                     f"{metrics_str(m,'cpu wire',2)} | {m['worst']:.3f} | {p['wall_seconds']:.1f} |")
    if train:
        lines+=['','| Seed/臂 | 原始请求 / 窗口内事件 | 平均副本 | 实际矩阵切换 | D/S 参数累计移动量 |',
                '|---|---:|---:|---:|---:|']
        for (seed,a),d in train.items():
            rows=d['online']
            lines.append(f"| {seed}/{a} | {sum(r['requests'] for r in rows):,}/"
                         f"{sum(r['events'] for r in rows):,} | {mean(sum(r['active_replicas']) for r in rows):.2f} | "
                         f"{sum(r['active_matrix']!=r['old_matrix'] for r in rows)} | "
                         f"{sum(u['d_change'] for u in d['updates']):.4f}/"
                         f"{sum(u['s_change'] for u in d['updates']):.4f} |")
    lines+=['','| Seed/臂 | GS | 固定 R/E2E/P99 | CPU/链路峰值 | 严重/截尾/末端 |',
            '|---|---:|---:|---:|---:|']
    for (seed,a),d in train.items():
        for gs,m in d['fixed'].items():
            lines.append(f"| {seed}/{a} | {gs} | {metrics_str(m,'reward e2e p99')} | "
                         f"{metrics_str(m,'cpu wire',2)} | {m['severe']}/{m['censored']}/{m['terminal']} |")
    lines+=['','## 独立评价与 D/S 消融','',
            '静态参考固定为该场景 GS0 Actor 均值的首槽合法矩阵。固定 S 为同一资源先验启发式。'
            '随周期重解码的 GS0 D + 最终 S 进一步控制部署次数。'
            '所有消融共用同流请求，同一 D 因子所形成的矩阵逐槽核对一致。','',
            '| Seed/周期/场景/臂 | R/E2E/P99 | 严重/截尾/末端 | CPU/链路峰值 | 最坏 R | 平均副本 | 切换/持有/延迟成本 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for d in docs:
        for cell,r in d['arms'].items():
            m=r['summary']
            lines.append(f"| {d['seed']}/{d['arm']}/{d['case']['tag']}/{cell} | {metrics_str(m,'reward e2e p99')} | "
                         f"{m['severe']}/{m['censored']}/{m['terminal']} | {metrics_str(m,'cpu wire',2)} | "
                         f"{m['worst']:.3f} | {m['replicas']:.2f} | {metrics_str(m,'switch hold delay')} |")
    tests=(('整体端点','joint','random'),('D 同周期','joint','random_d_learned_s'),
           ('D 相对静态参考','joint','reference_learned_s'),('S 在学习 D','joint','learned_d_prior'),
           ('S 在静态 D','reference_learned_s','reference_prior'),
           ('联合优于静态双参考','joint','reference_prior'))
    if docs:
        lines+=['','| 周期 | 对照 | ΔR/ΔE2E | 双改善场景数 |',
                '|---|---|---:|---:|']
        for a in ARMS:
            f=[d for d in docs if d['arm']==a]
            if not f:
                continue
            for name,left,right in tests:
                rr,ee=(delta(f,left,right,k) for k in ('reward','e2e'))
                wins=sum(delta([d],left,right,'reward')>0 and e is not None and e<0
                         for d in f for e in (delta([d],left,right,'e2e'),))
                lines.append(f"| {a} | {name} | {fmt(rr)}/{fmt(ee)} | {wins}/{len(f)} |")
            inter={k:[delta([d],'joint','learned_d_prior',k)-
                      delta([d],'reference_learned_s','reference_prior',k) for d in f
                      if all(d['arms'][x]['summary'][k] is not None for x in
                             ('joint','learned_d_prior','reference_learned_s','reference_prior'))]
                   for k in ('reward','e2e')}
            lines+=['',f"{a} 的四因子交互差（R/E2E）为 "
                        +'/'.join(fmt(mean(x)) if x else '—' for x in inter.values())+'。']
        by_case={(d['seed'],d['case']['tag'],d['arm']):d for d in docs}
        cases=sorted({(d['seed'],d['case']['tag']) for d in docs})
        lines+=['','| Seed/场景 | 联合端点 B−A 的 R/E2E/P99 | B−A 切换/持有/时延成本 |',
                 '|---|---:|---:|']
        for key in cases:
            a,b=(by_case.get((*key,k)) for k in ARMS)
            if a and b:
                x,y=(d['arms']['joint']['summary'] for d in (a,b))
                paired={k:y[k]-x[k] if x[k] is not None and y[k] is not None else None
                        for k in ('reward','e2e','p99','switch','hold','delay')}
                lines.append(f"| {key[0]}/{key[1]} | {metrics_str(paired,'reward e2e p99')} | "
                             f"{metrics_str(paired,'switch hold delay')} |")
    lines+=['','## 收敛判定','']
    for a in ARMS:
        lines+=training_outcome(train,docs,a,phys.SEEDS)
    if len(docs)<24:
        lines+=['最终判定等待完整独立端点与实际后期固定验证。']
    lines+=['','原始记录与恢复检查点位于 `runs/results/EXP078/`。下列曲线逐点保留训练和留出失败。','']
    for f in figures(train,docs):
        lines.append(f'![{f.name}](../../{f.as_posix()})')
    lines+=['','```powershell','python -m scripts.exp078_run smoke','python -m scripts.exp078_run all',
            'python -m scripts.exp078_run report','```','']
    atomic(REVIEW,lambda fd:fd.write('\n'.join(lines).encode('utf8')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('smoke','all','report'))
    mode=parser.parse_args().mode
    torch.set_num_threads(1)
    root=ROOT/'smoke' if mode=='smoke' else ROOT
    with run_lock(RunFiles(root/'experiment')):
        if mode=='report':
            render()
            return
        base,_=inputs()
        plan=register(root,base,mode=='smoke')
        if mode=='smoke':
            smoke(root,base,plan)
            return
        if not (ROOT/'smoke'/'checks.json').exists():
            raise ValueError('The joint D/S engineering smoke must finish first')
        try:
            with ProcessPoolExecutor(max_workers=3) as pool:
                jobs=[pool.submit(seed_run,root,base,plan,s) for s in plan['seeds']]
                for future in as_completed(jobs):
                    future.result()
                    render()
        finally:
            render()


if __name__=='__main__':
    main()
