"""Train EXP079 C with finite, paired system-delay credit for S only."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from shutil import copy2
from statistics import mean, median
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from comparison.report import records
from edge_joint.adaptive_runner import read_checkpoint
from edge_joint.adaptive_storage import RunFiles, atomic, publish, run_lock
from edge_joint.adaptive_training import RolloutControl, components, episode, load_model
from scripts import exp039_run as probe
from scripts import exp065_run as phys
from scripts import exp075_run as keys
from scripts import exp078_run as joint
from scripts import exp079_run as prior
from scripts.exp027_cases import boundary_record
from scripts.exp047_report import plt
from scripts.exp055_joint import inputs


ROOT = Path('runs/results/EXP080')
REVIEW = Path('research/EXP080/AI_RESEARCH_REVIEW.md')
BASE = 'e45d783998dce4911d2123f76fb4b1f555b8bfc2'
ARMS = ('A', 'B', 'C', 'D')
HISTORY = ('initial.json', 'train.jsonl', 'updates.jsonl', 'fixed.jsonl',
           'best.pt', 'last.pt', 'resume_latest.pt')
MILESTONES = (*joint.FIXED, 126, 252, 378, 504, 630)
STRIDE = 32
MAX_USES = 96
RATIO = .15


@dataclass(frozen=True)
class Config(prior.Config):
    s_pair_credit_digest: str | None = None
    record_d_gradient_direction: bool = True
    record_s_policy_sample: bool = True


def settings(base, seed, arm, credit=None):
    old = prior.settings(base, seed, 'C' if arm == 'D' else arm)
    return Config(**asdict(old), s_pair_credit_digest=credit.digest if credit else None)


def case_for(seed, role, tag, first, pressure=None):
    key = ('EXP080-v1', seed, role, tag)
    return dict(seed=seed, tag=tag, first=first, pressure=pressure,
                scene=phys.stable_seed(*key, 'scene'), population=phys.stable_seed(*key, 'population'))


def register(root, base, smoke):
    old = phys.read(prior.ROOT/'protocol.json')['plan']
    if old['original'] != phys.fingerprint(base) or old['demand'] != phys.read(phys.CONFIG):
        raise ValueError('EXP079 physical fingerprints changed')
    seeds = [20261114] if smoke else old['seeds']
    episodes = 2 if smoke else 630
    blocks, fixed, bank, held, fresh = {}, {}, {}, [], []
    for seed in seeds:
        token = str(seed)
        prefix = 'smoke' if smoke else 'formal'
        blocks[token] = old['blocks'][token][:episodes] if smoke else list(old['blocks'][token])
        first = len(blocks[token])
        for j in range(first, episodes):
            block = (j-first)//16
            kind = joint.TREATMENTS[block % len(joint.TREATMENTS)][1]
            path = case_for(seed, 'train-path', block, kind)
            if (j-first)%16 == 0:
                fresh.append(path)
            blocks[token].append(dict(path=path, offset=(j-first)%16*joint.SLOTS,
                                      scene=phys.stable_seed('EXP080-v1', seed, 'train', j)))
        fixed[token] = old['fixed'][token]
        tags = joint.TREATMENTS[-1:] if smoke else joint.TREATMENTS
        held.extend(dict(case_for(seed, prefix+'-holdout', tag, kind, pressure), slots=8 if smoke else 32)
                    for tag, kind, pressure in tags)
        bank[token] = [case_for(seed, prefix+'-credit', tag, kind, pressure)
                       for tag, kind, pressure in (joint.TREATMENTS[::3] if smoke else joint.TREATMENTS)]
    fresh.extend(held)
    fresh.extend(c for cases in bank.values() for c in cases)
    fresh_keys = list(keys.external_keys(fresh))
    fresh_keys.extend(b['scene'] for seq in blocks.values() for b in seq[63:])
    old_keys = {key for n in range(65, 80) for p in [Path(f'runs/results/EXP{n:03d}/protocol.json')]
                if p.is_file() for key in keys.external_keys(phys.read(p)['plan'])}
    if len(fresh_keys) != len(set(fresh_keys)) or set(fresh_keys)&old_keys:
        raise ValueError('Registered streams overlap')
    plan = dict(experiment='EXP080', parent=BASE, original=old['original'], demand=old['demand'],
                archive=phys.fingerprint(old), episodes=episodes, seeds=seeds, blocks=blocks,
                fixed=fixed, held=held, credit=bank, fixed_milestones=[0, 1, 2] if smoke else list(MILESTONES),
                protocol=phys.digest(REVIEW.with_name('PROTOCOL.md')), smoke=smoke)
    f = root/'protocol.json'
    if f.exists():
        if phys.read(f)['plan'] != plan:
            raise ValueError('EXP080 experimental registration changed')
    else:
        phys.write(f, dict(plan=plan, source=phys.source()))
    return plan


def verified(seed, arm, names):
    directory = joint.out_for(prior.ROOT, seed, arm)
    manifest = phys.read(prior.ROOT/'artifact_index.json')
    paths = {name: directory/name for name in names}
    for path in paths.values():
        if phys.digest(path) != manifest[path.as_posix()]['sha256']:
            raise ValueError(f'EXP079 artifact digest mismatch at {path}')
    return paths


class CalibratedDecisions(probe.Decisions):
    def score(self, x, prior, residual):
        legal = x[0, :, 7].numpy() > .5
        center = residual[legal].mean()
        return prior + center + self.agent.residual_scale*(residual-center)


def elapsed(request, decision, deadline, penalty=0.):
    if request.finish is not None and request.finish <= deadline:
        return request.finish - decision
    return deadline - decision + penalty


def physical_pair(capture, opening, factual, decision, alternative):
    force = dict(decision, node=alternative)
    reference, native, native_work = factual
    branch, changed, changed_work = probe.frozen(capture, opening, 2, 'full', force,
                                                decision_factory=CalibratedDecisions)
    phys.same_flow(native, changed)
    if any(r['active_matrix'] != a['active_matrix'] for r, a in zip(native, changed, strict=True)):
        raise ValueError('The action intervention changed the placement')
    request_id, stage_index = decision['key']
    identity = tuple(decision['key'])
    if branch.audit.events.get(identity, {}).get('selected') != alternative:
        raise ValueError('Paired S intervention was not applied')
    boundary = decision['time']
    deadlines = {r['period']:r['score_deadline'] for r in native}
    horizon = native[-1]['score_deadline']
    before = {r.id:r for r in reference.requests}
    after = {r.id:r for r in branch.requests}
    arrivals = lambda by_id: {i:(r.arrival.time, r.arrival.user, r.arrival.service, r.arrival.access)
                              for i,r in by_id.items()}
    if arrivals(before) != arrivals(after):
        raise ValueError('Exact paired request identities differ')
    cohort = {row['period']:[] for row in native}
    for i, r in before.items():
        for row in native:
            if row['start'] <= r.arrival.time < row['end']:
                cohort[row['period']].append(i)
                break
    penalty = capture['state'][0].failure_penalty
    weight = capture['state'][0].cost['delay_weight']
    changes = []
    for period, ids in cohort.items():
        if len(ids) != native[period-opening]['requests']:
            raise ValueError('Pair cohort partition does not match the original ledger')
        delta = {i:elapsed(before[i], before[i].arrival.time, deadlines[period], penalty)
                 - elapsed(after[i], after[i].arrival.time, deadlines[period], penalty) for i in ids}
        changes.append(dict(period=period, n=len(ids), delay_sum=sum(delta.values()),
                            focal_delay=delta.get(request_id, 0.),
                            reward_gain=weight*sum(delta.values())/max(len(ids), 1)))
    recorded_gain = sum(r['reward_raw']-n['reward_raw'] for n,r in zip(native, changed, strict=True))
    if not np.isclose(sum(x['reward_gain'] for x in changes), recorded_gain, atol=1e-9, rtol=0.):
        raise ValueError('Paired request credit failed to reconcile to original Reward')
    def stage_duration(r):
        marks = {kind:t for j,kind,t,_ in r.trace if j==stage_index and kind in ('ready', 'finish')}
        return marks['finish']-marks['ready'] if len(marks)==2 else None
    incumbent = [i for i,r in before.items() if i!=request_id and r.arrival.time<=boundary
                 and (r.finish is None or r.finish>boundary)]
    common_delay = lambda i: elapsed(before[i], boundary, horizon)-elapsed(after[i], boundary, horizon)
    stage = [stage_duration(before[request_id]),stage_duration(after[request_id])]
    return dict(decision=decision, alternative=alternative, cohorts=changes, native_stage=stage[0],
                alternative_stage=stage[1], focal_e2e_gain=common_delay(request_id),
                focal_e2e=[r.finish-r.arrival.time if r.finish is not None else None
                           for r in (before[request_id],after[request_id])],
                concurrent=len(incumbent), concurrent_delay_gain=sum(map(common_delay,incumbent)),
                system_gain=recorded_gain/len(native), native=joint.summary(native), changed=joint.summary(changed),
                native_work=native_work, changed_work=changed_work)


@torch.inference_mode()
def collect_case(root, base, plan, seed, case):
    torch.set_num_threads(1)
    file = root/f"pairs.{seed}.{case['tag']}.json"
    checkpoint = verified(seed, 'C', ('last.pt',))['last.pt']
    stamp = phys.fingerprint([plan,case,phys.digest(checkpoint)])
    if file.exists():
        document = phys.read(file)
        if document['stamp'] != stamp:
            raise ValueError(f'Mismatched physical data at {file}')
        return document
    tick = perf_counter()
    cfg = replace(prior.settings(base, seed, 'C'), periods=8, freeze_ds=True)
    d,s,_,_,gate,_ = components(cfg)
    load_model(read_checkpoint(checkpoint),d,s,gate)
    # This dedicated split supplies naturally settled training credit and is disjoint from the holdout.
    split = 'train'
    env = phys.environment(base,plan,case,split)
    population,scene = case['population'],case['scene']
    loads = (case['pressure'],)*cfg.periods
    openings = {}
    anchors = (3,6)

    def capture(e,sim,p,phase):
        if phase == 'start' and p in anchors:
            openings[p] = dict(state=phys.fork(e,sim),agent=s,s16=None,cfg=cfg,d=d,gate=gate,
                               scene=scene,env_seed=population,split=split,loads=loads,index=1,
                               snapshot=boundary_record(e,sim,population,split,scene,p))

    control = RolloutControl(environment_seed=population,loads=loads,epsilon=0.,sample_d=False,
                             trace_requests=True)
    rows = episode(cfg,env,d,s,gate,scene,split,1,control=control,
                   simulator_factory=probe.ProbeSimulator,boundary=capture)[0]
    joint.audit_rows(env,rows,1)
    points = []
    for opening, captured in openings.items():
        captured['reference'] = rows
        factual = probe.frozen(captured,opening,2,'full',legal_sizes=range(2,33),
                              decision_factory=CalibratedDecisions)
        # Avoid a mismatch from terminal drain when checking the overlapping opening cohort.
        probe.check_native(rows[opening-1:opening],factual[1][:1])
        candidates = [v for v in factual[0].audit.events.values() if v['time'] < rows[opening-1]['end']
                      and len(v['legal']) >= 2]
        if not candidates:
            continue
        disagreement = [v for v in candidates if v['selected'] != v['prior_node']]
        early = min(disagreement or candidates,key=lambda v:(v['time'],v['key']))
        high = max(candidates,key=lambda v:v['cpu_before'][v['legal'].index(v['selected'])])
        for name, point in [('first_disagreement',early),('peak_cpu',high)]:
            if name=='peak_cpu' and point['key']==early['key']:
                continue
            alt = point['prior_node'] if point['prior_node']!=point['selected'] else point['full_rank'][1]
            points.append(dict(opening=opening, selection=name,
                               **physical_pair(captured,opening,factual,point,alt)))
    doc = dict(seed=seed,case=case,points=points,seconds=perf_counter()-tick,stamp=stamp,
               source=phys.source(),checkpoint=phys.digest(checkpoint))
    phys.write(file,doc)
    print(f"EXP080 paired {seed}/{case['tag']}: {len(points)} interventions in {doc['seconds']:.1f}s",flush=True)
    return doc


class PairedCredit:
    """Restore the same exposures using only the bank digest and the true S update counter."""

    def __init__(self, documents):
        self.digest = phys.fingerprint([d['points'] for d in documents])
        self.samples = []
        for doc in documents:
            for point in doc['points']:
                gain = point['system_gain']
                if abs(gain) <= 1e-8:
                    continue
                action, alternative = point['decision']['selected'], point['alternative']
                nodes = (alternative,action) if gain>0 else (action,alternative)
                self.samples.append((torch.tensor(point['decision']['obs'],dtype=torch.float32),nodes,gain))
        if not self.samples:
            raise ValueError('No measured system-delay direction in the registered credit bank')

    def __call__(self, agent, bellman):
        position = agent.updates//STRIDE
        metrics = dict(s_pair_steps=0.,s_pair_id=0.,s_pair_signed_gain=0.,s_pair_loss=0.,
                       s_pair_coefficient=0.,s_pair_grad_ratio=0.,s_pair_grad_cosine=0.,s_pair_q_gap=0.)
        if agent.updates%STRIDE or position >= MAX_USES*len(self.samples):
            return bellman, metrics
        index = position%len(self.samples)
        obs,(preferred,other),gain = self.samples[index]
        scores = agent.team_q(obs[None])[0]
        gap = scores[preferred]-scores[other]
        advantage = abs(gain)/(.02+abs(gain))*F.softplus(-gap)
        params = tuple(agent.online.parameters())
        def gradient(loss):
            return torch.cat([p.detach().flatten() for p in torch.autograd.grad(loss,params,retain_graph=True)])
        base,aux = gradient(bellman),gradient(advantage)
        base_norm,aux_norm = float(base.norm()),float(aux.norm())
        coefficient = min(1.,RATIO*base_norm/max(aux_norm,1e-12))
        ratio = coefficient*aux_norm/max(base_norm,1e-12)
        cosine = float(base.dot(aux))/max(base_norm*aux_norm,1e-12)
        metrics.update(s_pair_steps=1.,s_pair_id=float(index+1),s_pair_signed_gain=gain,
                       s_pair_loss=float(advantage.detach()),s_pair_coefficient=coefficient,
                       s_pair_grad_ratio=ratio,s_pair_grad_cosine=cosine,s_pair_q_gap=float(gap.detach()))
        return bellman+coefficient*advantage, metrics


def import_control(root,base,plan,seed,arm):
    new = joint.out_for(root,seed,arm)
    with run_lock(new):
        if (new/'resume_latest.pt').exists():
            return
        old = phys.read(prior.ROOT/'protocol.json')['plan']
        token = str(seed)
        if (phys.fingerprint(old)!=plan['archive'] or old['fixed'][token]!=plan['fixed'][token]
                or old['blocks'][token]!=plan['blocks'][token][:63]):
            raise ValueError('Continuation does not match the archived 504 GS protocol')
        paths = verified(seed,arm,HISTORY)
        state = read_checkpoint(paths['resume_latest.pt'])
        if (state['config']!=asdict(prior.settings(base,seed,arm)) or state['episode']!=63
                or state['environment_id']!=plan['archive'] or state['origin']!={'kind':'random'}
                or not joint.projection_matches(read_checkpoint(paths['last.pt']),state)):
            raise ValueError('Unverified EXP079 random-origin checkpoint')
        for name in ('initial.json','train.jsonl','updates.jsonl','fixed.jsonl'):
            copy2(paths[name],new/name)
        state['environment_id'] = phys.fingerprint(plan)
        config = asdict(settings(base,seed,arm))
        state.update(config=config)
        state['best'].update(environment_id=state['environment_id'],config=config)
        publish(new/'resume_latest.pt',state)


def credit_for(root,base,plan,seed):
    docs = [collect_case(root,base,plan,seed,c) for c in plan['credit'][str(seed)]]
    return PairedCredit(docs)


def training(root,base,plan,seed,arm):
    torch.set_num_threads(1)
    credit = credit_for(root,base,plan,seed) if arm=='D' else None
    if arm!='D' and not plan['smoke']:
        import_control(root,base,plan,seed,arm)
    return joint.train_one(root,base,plan,seed,arm,cfg_override=settings(base,seed,arm,credit),
                           s_pair_credit=credit)


def evaluation(root,base,plan,case):
    torch.set_num_threads(1)
    seed = case['seed']
    shared = None
    for arm in ARMS:
        result = prior.evaluation(root,base,plan,seed,arm,case,shared,config=settings(base,seed,arm))
        shared = result['arms']['initial_d_prior']


def parallel(function,jobs,workers):
    if workers==1:
        return [function(*args) for args in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(function,*args) for args in jobs]
        return [f.result() for f in as_completed(futures)]


def audit(root,base,plan):
    seed = 20261114
    cases = [c for c in plan['credit'][str(seed)] if c['tag'] in ('natural','pressure')]
    result = parallel(collect_case,[(root,base,plan,seed,c) for c in cases],2)
    history = [phys.read(prior.ROOT/f'diagnostic.{tag}.json') for tag in ('pressure','regional')]
    # Historical stress failures are diagnostic-only and never enter D's credit bank.
    support = sum(p['native_stage'] is not None and p['alternative_stage'] is not None
                  and p['alternative_stage']>p['native_stage']+1e-8 and p['system_gain']>1e-8
                  for d in result for p in d['points'])
    summary = dict(stage_vs_system_conflicts=support,historical=[dict(tag=d['tag'],cases=[dict(gs=p['gs'],native=p['native'],
                    prior=p['prior'],cases=[dict(stage=x['decision']['stage'],q_gap=x['decision']['q_gain'],
                    focal_gain=x['effect']['own_delay_delta'],incumbent_gain=x['effect']['other_delay_sum_delta'],
                    raw_gain=-x['effect']['cohort_reward_delta']) for x in p['cases']]) for p in d['probes']])
                    for d in history],new=[dict(case=d['case'],pairs=len(d['points']),wall=d['seconds']) for d in result])
    phys.write(root/'diagnosis.json',summary)
    return summary


def smoke(root,base,plan):
    seed = plan['seeds'][0]
    credit = credit_for(root,base,plan,seed)
    cfg = settings(base,seed,'D',credit)
    tick = perf_counter()
    first = joint.train_one(root,base,plan,seed,'D',1,cfg_override=cfg,s_pair_credit=credit)
    restored = training(root,base,plan,seed,'D')
    joint.train_one(root,base,plan,seed,'D',label='continuous',cfg_override=cfg,s_pair_credit=credit)
    a,b = (read_checkpoint(joint.out_for(root,seed,name)/'resume_latest.pt') for name in ('D','continuous'))
    for name in ('d','s','d_optimizer','gate','gate_optimizer','replay','rng','d_credit_offset'):
        if not joint.same_tree(a[name],b[name]):
            raise ValueError(f'System-credit resume diverged at {name}')
    trace = list(records(joint.out_for(root,seed,'D')/'train.jsonl'))
    boundaries = [r['s_boundary'] for r in trace]
    exposure = sum(m['s_pair_steps']*m['optimizer_steps'] for m in boundaries)
    if exposure!=4 or max(m['s_pair_grad_ratio']*m['optimizer_steps'] for m in boundaries)>RATIO+1e-7:
        raise ValueError('S auxiliary exposure or gradient cap failed')
    x = prior.evaluation(root,base,plan,seed,'D',plan['held'][0],config=cfg)
    evidence = dict(first=first,restored=restored,exact_resume=True,credit_exposures=exposure,
                    credit_pairs=len(credit.samples),held={k:v['summary'] for k,v in x['arms'].items()},
                    seconds=perf_counter()-tick,source=phys.source())
    phys.write(root/'checks.json',evidence)
    return evidence


def opposite(x,y):
    return abs(x)>1e-8 and abs(y)>1e-8 and x*y<0


def condensed_training(out):
    fixed = {}
    for row in records(out/'fixed.jsonl'):
        fixed.setdefault(row['episode']*joint.SLOTS,[]).append(row)
    fixed = {gs:joint.summary(rows) for gs,rows in fixed.items()}
    labels = ('reward_raw','e2e_completed','p99','censored')
    curve = {k:[] for k in (*labels,'cpu','wire','replicas')}
    max_failed_run = failed_run = terminal = 0
    aux = dict.fromkeys(('s_pair_steps','s_pair_grad_ratio','s_pair_grad_cosine','s_pair_loss'),0.)
    s_total = s_overrides = 0
    direction_risk = {}
    stage_replicas = np.zeros(28)
    node_use = np.zeros(32)
    changes = []
    for row in records(out/'train.jsonl'):
        summary = {k:row[k] for k in labels}
        summary.update(cpu=max(v['work_pending'] for v in row['cpu']),
                       wire=max(v['work_pending'] for v in row['wired']),
                       replicas=sum(row['active_replicas']))
        for k,v in summary.items():
            curve[k].append(v)
        failed_run = failed_run+1 if row['reward_raw']<=-20 else 0
        max_failed_run = max(max_failed_run,failed_run)
        terminal += row.get('residual_requests',0)
        n = row['episode']
        direction_risk[n] = direction_risk.get(n,False) or row['reward_raw']<=-20
        stage_replicas += np.asarray(row['active_replicas'])
        node_use += [v['utilization'] for v in row['cpu']]
        changes.append((int(row['old_matrix'],16)^int(row['active_matrix'],16)).bit_count())
        s_total += len(row.get('s_policy_samples',()))
        s_overrides += sum(x['selected']!=x['prior_node'] for x in row.get('s_policy_samples',()))
        step = row.get('s_boundary',{})
        for k in aux:
            aux[k] += step.get(k,0.)*step.get('optimizer_steps',0)
    total = len(changes)
    grads,fit,adv = [],[],[]
    directions,reference,labels = [],[],[]
    for update in records(out/'updates.jsonl'):
        grads.append(update['d_gradient'])
        fit.append(update['d_value_mae'])
        adv.append(update.get('d_actor_advantage_mae'))
        if 'd_gradient_unit' in update:
            direction = np.asarray(update['d_gradient_unit'],dtype=np.float32)
            labels.append(direction_risk[update['gs']//joint.SLOTS])
            directions.append(direction)
            if not labels[-1]:
                reference.append(direction)
    angle = [None,None]
    if reference:
        center = np.mean(reference,axis=0)
        center /= max(np.linalg.norm(center),1e-12)
        for group in (0,1):
            cosines = [float(d@center) for d,risk in zip(directions,labels,strict=True) if risk==bool(group)]
            angle[group] = median(cosines) if cosines else None
    return dict(curve=curve, fixed=fixed, gs=total, terminal=terminal, peak_run=max_failed_run,
                stage=(stage_replicas/total).tolist(), node=(node_use/total).tolist(),bits=mean(changes),
                draws=s_total,overrides=s_overrides,aux=aux,grad=np.quantile(grads,[0,.5,.9,1]),
                clip=sum(x>5. for x in grads),fit=median(fit),angle=angle,
                advantage=median([a for a in adv if a is not None]) if any(a is not None for a in adv) else None)


def match_training_flows(plan):
    for seed in plan['seeds']:
        paths = [joint.out_for(ROOT,seed,arm)/'train.jsonl' for arm in ARMS]
        for group in zip(*(records(path) for path in paths),strict=True):
            for row in group[1:]:
                phys.same_flow((group[0],),(row,))


def figures(training,held):
    def save(fig,path):
        fig.savefig(ROOT/path,dpi=135,bbox_inches='tight')
        plt.close(fig)
        return f'../../../runs/results/EXP080/{path}'
    paths = []
    for seed in sorted({s for s,_ in training}):
        fig,axes = plt.subplots(3,2,figsize=(13,9))
        f,fx = plt.subplots(2,2,figsize=(10,6))
        for (s,arm),t in training.items():
            if seed!=s:
                continue
            for ax,(field,title) in zip(axes.flat,zip(('reward_raw','e2e_completed','p99','cpu','wire','censored'),
                    ('Raw Reward','Completed E2E (s)','Penalized P99 (s)','CPU (CU)','Wire (Mbit)','Censored'))):
                ax.plot(range(1,t['gs']+1),t['curve'][field],linewidth=.65,label=arm)
                ax.set(xlabel='Training GS',ylabel=title)
            for ax,k in zip(fx.flat,('reward','e2e','p99','censored')):
                gs = sorted(t['fixed'])
                ax.plot(gs,[t['fixed'][g][k] for g in gs],'.-',label=arm)
                ax.set(xlabel='Fixed evaluation GS',ylabel=k)
        axes[0,0].legend()
        fx[0,0].legend()
        paths.extend([save(fig,f'fig.train.{seed}.png'),save(f,f'fig.fixed.{seed}.png')])
    if held:
        documents = [d for d in held if d['arm']=='D']
        for seed in sorted({d['seed'] for d in documents}):
            rows = [d for d in documents if d['seed']==seed]
            f,axes = plt.subplots(len(rows),3,figsize=(13,2.5*len(rows)),squeeze=False)
            for d,group in zip(rows,axes,strict=True):
                for arm in prior.CELLS:
                    for ax,field in zip(group,('reward_raw','e2e_completed','p99')):
                        ax.plot([r['period'] for r in d['arms'][arm]['rows']],
                                [r[field] for r in d['arms'][arm]['rows']],label=arm,linewidth=.8)
                        ax.set(xlabel='Independent GS',ylabel=f"{d['case']['tag']} {field}")
            axes[0,0].legend(fontsize=6)
            paths.append(save(f,f'fig.held.D.{seed}.png'))
    return paths


def render():
    ROOT.mkdir(parents=True,exist_ok=True)
    plan = phys.read(ROOT/'protocol.json')['plan']
    training = {}
    completed = []
    for seed,arm in product(plan['seeds'],ARMS):
        out = joint.out_for(ROOT,seed,arm)
        if (out/'proof.json').exists():
            proof = phys.read(out/'proof.json')
            if proof['gs']==plan['episodes']*joint.SLOTS:
                training[seed,arm] = condensed_training(out)
                completed.append(proof)
    held = []
    for f in sorted(ROOT.glob('held.*.json')):
        d = phys.read(f)
        columns = ('period','reward_raw','e2e_completed','p99')
        for entry in d['arms'].values():
            entry['rows'] = [{k:r[k] for k in columns} for r in entry['rows']]
        held.append(d)
    done_train = len(completed)==len(plan['seeds'])*len(ARMS)
    done_eval = done_train and len(held)==len(plan['held'])*len(ARMS)
    lines = ['# EXP080 Agent-S 系统时延信用与联合可靠性','',
             f'EXP079 源码基线 `{BASE}`。当前生成版本 `{phys.source().get("commit","未取得")}`。',
             '实验设置见 [PROTOCOL.md](PROTOCOL.md)。D 组在 0.25 残差校准上增加有限期系统时延排序辅助。原 DDQN 事实目标、D、布局解码、KKT、物理窗口和仿真 Reward 沿用 EXP079。',
             '',f'正式 5040 GS 训练臂 **{len(completed)}/{len(plan["seeds"])*len(ARMS)}**，独立评价 **{len(held)}/{len(plan["held"])*len(ARMS)}**。',
             '', '## 物理信用审计','',
             'S 的原奖励为 `−(阶段完成时刻−阶段就绪时刻)/latency_ref_seconds`，后继为同一请求的下一阶段。现有局部信用包含该阶段的有线传输、等待和计算，环境队列特征可间接反映竞争。其他请求受该动作影响的时延未被直接计入。',
             '历史全局信用把连续全局决策之间的未完成请求积分作为奖励，改变了转移、观察和归属；当前自适应轨迹的派发跟踪与其互斥。历史资源归属按资源服务比例分摊驻留成本。EXP080 的信用由单次合法节点替换的真实两槽物理账本结算。','']
    diag = ROOT/'diagnosis.json'
    if diag.exists():
        for doc in phys.read(diag)['historical']:
            for point in doc['cases']:
                a,b = point['native'],point['prior']
                lines.append(f"- EXP079 {doc['tag']} GS{point['gs']} 两槽同布局学习/先验 CPU 峰值 {a['cpu']:.2f}/{b['cpu']:.2f} CU，截尾 {a['censored']}/{b['censored']}。")
    docs = [phys.read(f) for f in sorted(ROOT.glob('pairs.*.json'))]
    pairs = [p for doc in docs for p in doc['points']]
    if pairs:
        conflicted = [p for p in pairs if p['native_stage'] is not None and p['alternative_stage'] is not None
                      and p['native_stage']<p['alternative_stage']-1e-8 and p['system_gain']>1e-8]
        both = [p for p in pairs if p['native_stage'] is not None and p['alternative_stage'] is not None
                and p['native_stage']>p['alternative_stage']+1e-8 and p['system_gain']>1e-8]
        lines += ['',f'在 {len(docs)} 条专用物理流中执行 {len(pairs)} 次单动作替换。分叉的请求指纹、同布局、回报分解均已核对。',
                  f'本人/同时在途请求的时延作用反向 {sum(opposite(p["focal_e2e_gain"],p["concurrent_delay_gain"]) for p in pairs)}/{len(pairs)}，计分 2 槽替换收益为正 {sum(p["system_gain"]>1e-8 for p in pairs)}/{len(pairs)}。',
                  f'替代提高两槽 R、却增加本人当前阶段用时的样本 {len(conflicted)} 个。另有 {len(both)} 个同时改善阶段用时和两槽 R，冻结 S 仍选择了原节点。两类结果分别支持局部目标冲突及原 Q 排序/长程信用偏差，后一类不必归因于并发外部性。',
                  '', '| 种子/场景/GS/阶段 | 学习→替代 | Q 学习−替代 | 阶段用时 s | 本人/其他时延节省 s | 窗口 ΔR | CPU CU/有线 Mbit 学习→替代 |',
                  '|---|---|---:|---:|---:|---:|---|']
        for doc in docs:
            for p in doc['points']:
                d = p['decision']
                x = d['legal'].index(d['selected'])
                y = d['legal'].index(p['alternative'])
                a,b = p['native'],p['changed']
                lines.append(f"| {doc['seed']}/{doc['case']['tag']}/{p['opening']}/{d['stage']} | {d['selected']}→{p['alternative']} | {d['q'][x]-d['q'][y]:+.4f} | "
                             f"{joint.fmt(p['native_stage'])}/{joint.fmt(p['alternative_stage'])} | {p['focal_e2e_gain']:+.4f}/{p['concurrent_delay_gain']:+.4f} | {p['system_gain']:+.5f} | {a['cpu']:.2f}/{a['wire']:.2f}→{b['cpu']:.2f}/{b['wire']:.2f} |")
        lines += ['', '分叉可以识别选定开口的外部性和原 Q 排序错序。状态、替代点和冻结续策有限，无法识别信用遗漏是否是 S 整体学习失准的主要原因，也不构成资源先验的最优性证明。']
    checkpoints = ROOT/'smoke'/'checks.json'
    if checkpoints.exists():
        z = phys.read(checkpoints)
        lines += ['',f"烟测到 GS{z['restored']['gs']}，D/S 更新 {z['restored']['d_updates']}/{z['restored']['s_updates']}。断点续跑的权重、优化器、Replay 与连续运行一致，辅助更新 {z['credit_exposures']:.0f} 次，独立完成 {len(z['held'])} 个调度消融。"]
    lines += ['', '## 联合训练','',
              '| 种子/臂 | GS/D/S 更新 | 原始 R/E2E/P99 均值 | 最低 R | 严重/截尾/未完成 | 训练秒 |',
              '|---|---|---|---:|---|---:|']
    for proof in completed:
        t = training[proof['seed'],proof['arm']]
        c = t['curve']
        m = [mean(x for x in c[k] if x is not None) for k in ('reward_raw','e2e_completed','p99')]
        lines.append(f"| {proof['seed']}/{proof['arm']} | {proof['gs']}/{proof['d_updates']}/{proof['s_updates']} | "
                     f"{m[0]:.5f}/{m[1]:.5f}/{m[2]:.5f} | {min(c['reward_raw']):.3f} | "
                     f"{sum(r<=-20 for r in c['reward_raw'])}/{sum(c['censored'])}/{t['terminal']} | {proof['wall_seconds']:.1f} |")
    for (seed,arm),t in training.items():
        n = t['aux']['s_pair_steps']
        ratio = t['aux']['s_pair_grad_ratio']/n if n else 0.
        cosine = t['aux']['s_pair_grad_cosine']/n if n else 0.
        lines += ['', f"**{seed}/{arm}** D 裁剪 {t['clip']} 次，裁剪前梯度最小/中位/P90/最大 {list(np.round(t['grad'],3))}。价值绝对误差中位 {t['fit']:.3f}，校准后绝对优势中位 {joint.fmt(t['advantage'])}。非严重/严重回合梯度与非严重平均方向的中位余弦 {list(t['angle'])}。最长严重槽连续段 {t['peak_run']}。",
                  f"S 抽样动作背离先验 {t['overrides']}/{t['draws']}（含探索），辅助优化 {int(n)} 次，平均辅助/Bellman 梯度比例 {ratio:.4f}，平均梯度余弦 {cosine:.4f}。",
                  f"平均副本 {mean(t['curve']['replicas']):.2f}，平均变化 {t['bits']:.2f} 位，阶段平均副本 {list(np.round(t['stage'],2))}。节点平均 CPU 利用率留存在逐槽原始账本，峰值平均利用率 {max(t['node']):.4f}。"]
    lines += ['', '## 独立评价','', '| 种子/臂/场景 | 联合 R/E2E/P99 | 学习 D 时 S 的 ΔR/ΔE2E | 初始 D 时 S 的 ΔR/ΔE2E | 联合严重/截尾/未完成 | 联合/同布局先验 CPU CU | 联合有线 Mbit |',
              '|---|---|---|---|---|---|---:|']
    effects = {}
    for d in held:
        cell = {k:v['summary'] for k,v in d['arms'].items()}
        a,b,c,h,start = (cell[k] for k in prior.CELLS)
        delta = lambda x,y,k: x[k]-y[k]
        ds = (delta(a,b,'reward'),delta(a,b,'e2e'))
        ds0 = (delta(c,h,'reward'),delta(c,h,'e2e'))
        effects[d['seed'],d['case']['tag'],d['arm']] = (ds,a,b,start)
        lines.append(f"| {d['seed']}/{d['arm']}/{d['case']['tag']} | {a['reward']:.5f}/{a['e2e']:.5f}/{a['p99']:.5f} | "
                     f"{ds[0]:+.5f}/{ds[1]:+.5f} | {ds0[0]:+.5f}/{ds0[1]:+.5f} | {a['severe']}/{a['censored']}/{a['terminal']} | {a['cpu']:.2f}/{b['cpu']:.2f} | {a['wire']:.2f} |")
    lines += ['', '## 正式研究判断','']
    if done_eval:
        match_training_flows(plan)
        dset = [(s,tag,*effect) for (s,tag,a),effect in effects.items() if a=='D']
        by_seed = {s:(mean(ds[0] for z,_,ds,*_ in dset if z==s),mean(ds[1] for z,_,ds,*_ in dset if z==s))
                   for s in plan['seeds']}
        avg = lambda k: mean(ds[k] for _,_,ds,*_ in dset)
        risk = [(s,{k:a[k]-b[k] for k in ('severe','censored','terminal','p99','cpu','wire')})
                for s,tag,_,a,b,_ in dset if tag=='pressure']
        control = lambda other,k: mean(a[k]-effects[s,tag,other][1][k] for s,tag,_,a,_,_ in dset)
        initial = lambda k: mean(a[k]-start[k] for _,_,_,a,_,start in dset)
        late = {}
        for s in plan['seeds']:
            f = training[s,'D']['fixed']
            late[s] = all(f[g]['reward']<f[3024]['reward']-.10 and f[g]['e2e']>f[3024]['e2e']+.01
                          for g in (4032,5040))
        criteria = {'S 总体 Reward/E2E':avg(0)>1e-8 and avg(1)<=1e-8,
                    '至少两种子 S Reward 正增益':sum(x[0]>1e-8 for x in by_seed.values())>=2,
                    'Seed14 S 非负':by_seed[20261114][0]>=-1e-8,
                    '逐压力场景可靠性':all(all(x[k]<=0 for k in ('severe','censored','terminal')) for _,x in risk),
                    '联合较双初始的 Reward/E2E':initial('reward')>0 and initial('e2e')<0,
                    '后期固定验证':not any(late.values())}
        lines += [f"D 组 S 相对同布局先验平均 ΔReward {avg(0):+.6f}，平均 ΔE2E {avg(1):+.6f} s。分种子 {by_seed}。",
                  f"压力场景同布局差 {risk}。联合相对 GS0 的 ΔReward/E2E {initial('reward'):+.6f}/{initial('e2e'):+.6f} s。",
                  *[f"D 整体相对 {a} 的 ΔReward/E2E/P99 {control(a,'reward'):+.6f}/{control(a,'e2e'):+.6f}/{control(a,'p99'):+.6f}。" for a in ('B','C')],
                  '',*[(f"- {k} {'通过' if v else '未通过'}。") for k,v in criteria.items()],
                  '', ('通过预登记门槛。可进入独立复验。' if all(criteria.values()) else '未通过预登记门槛。保留负面证据，维持 EXP079 对照。')]
    else:
        lines.append('5040 GS 正式端点和同布局独立结果尚未齐备。S 是否独立超越资源先验、D/S 是否具备正式联合可靠性均待验证。')
    paths = figures(training,held)
    if paths:
        lines += ['', '## 原始图表', '', *[f'![{Path(p).stem}]({p})' for p in paths]]
    migration = ROOT/'smoke'/'legacy-migration'/'verification.json'
    if migration.exists():
        lines += ['', 'Seed14 C 的 GS504 迁移逐项核对了 D/S、优化器、Replay、RNG 和账本位置；只更新实验配置及环境指纹元数据。']
    lines += ['', '## 原始记录与恢复', '',
              '原始物理分叉、案件注册、训练轨迹、固定验证、逐槽独立评价、best/last/恢复检查点和图表在 `runs/results/EXP080/`。',
              ' `python -m scripts.exp080_run all --workers 6` 可从最近保存的回合继续。 `python -m scripts.exp080_run report` 依据已完成账本重建报告。','']
    atomic(REVIEW,lambda f:f.write('\n'.join(lines).encode('utf8')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=('audit','smoke','train','evaluate','all','report'))
    p.add_argument('--workers',type=int,default=6)
    args = p.parse_args()
    if args.workers<1:
        p.error('--workers must be positive')
    torch.set_num_threads(1)
    tiny = args.task=='smoke'
    root = ROOT/'smoke' if tiny else ROOT
    with run_lock(RunFiles(root/'workflow')):
        if args.task=='report':
            render()
            return
        base,_ = inputs()
        plan = register(root,base,tiny)
        if tiny:
            smoke(root,base,plan)
            return
        if args.task in ('audit','all'):
            audit(root,base,plan)
            render()
        if args.task in ('train','all'):
            if not phys.read(root/'diagnosis.json').get('stage_vs_system_conflicts'):
                raise ValueError('Run audit and verify a physical stage/system ranking conflict before formal training')
            parallel(training,[(root,base,plan,s,a) for s,a in product(plan['seeds'],ARMS)],args.workers)
        if args.task in ('evaluate','all'):
            parallel(evaluation,[(root,base,plan,c) for c in plan['held']],args.workers)
        if args.task!='audit':
            render()


if __name__=='__main__':
    main()
