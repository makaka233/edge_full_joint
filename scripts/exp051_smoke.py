"""Exercise real paired forking, learning and atomic resume with a minimal horizon."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from edge_joint.adaptive_runner import source
from edge_joint.adaptive_storage import append, random_state
from scripts.exp027_run import fingerprint
from scripts.exp044_run import frozen, same_flow, write
from scripts.exp047_run import experiences, read_model, rows_at, run_files
from scripts.exp047_smoke import equivalent, rejects
from scripts.exp049_report import coverage
from scripts.exp049_run import SEEDS
from scripts.exp050_train import ThreeStepGate, replay_key, replay_links
from scripts.exp051_report import extra_figures
from scripts.exp051_run import ROOT, collect_block, registration, train


def without_timing(row):
    return {k:v for k,v in row.items() if k not in ('wall_seconds','episode_wall_seconds','profile',
                                                  'gs','block','arm','seed','gate_epsilon','gate_updates_at_decision')}


def smoke():
    start = perf_counter()
    torch.set_num_threads(1)
    plan, base = registration(1)
    seed = SEEDS[0]
    mini = deepcopy(plan)
    mini.update(experiment='EXP051 smoke',slots=5,blocks=2,fixed_at=[0])
    key = str(seed)
    for arm in ('A','B'):
        mini['training_by_arm'][arm][key][0]['loads'] = [.32]*5
        mini['training_by_arm'][arm][key][1]['loads'] = [.32,.55,.55,.55,.32]
    mini['validation'][key]['loads'] = [.32]*5
    stamp = fingerprint(mini)
    a,b,foreign = (ThreeStepGate(arm,s,stamp,feature_arm='A',rng_arm='A')
                   for arm,s in (('A',seed),('B',seed),('B',SEEDS[1])))
    assert equivalent(a.online.state_dict(),b.online.state_dict())
    assert a.policy_rng is not b.policy_rng and a.replay_rng is not b.replay_rng
    assert a.optimizer.param_groups[0]['params'][0] is not b.optimizer.param_groups[0]['params'][0]
    assert a.online.center.data_ptr()!=b.online.center.data_ptr()
    assert not equivalent(a.online.state_dict(),foreign.online.state_dict())
    ds = frozen(base,plan,seed,5)
    initial_ds = [{k:v.detach().clone() for k,v in net.state_dict().items()} for net in (ds[1],ds[2].online)]
    before_rng = random_state()
    left,weight,_,_,cost_a = collect_block(base,mini['training_by_arm']['A'][key][0],ds,a,1,mini['epsilon'][0],limit=0)
    assert equivalent(before_rng, random_state())
    assert all(equivalent(stored,net.state_dict())
               for stored,net in zip(initial_ds,(ds[1],ds[2].online),strict=True))
    for r in left:
        r.update(gs=r['period'],block=1)
    a.fit_scaler(left)
    a.collect(experiences(left,weight,a.identity))
    a_updates = a.improve()
    ROOT.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=ROOT,prefix='smoke-') as folder:
        root = Path(folder)
        train(root,mini,base,'B',seed,5)
        paths = run_files(root,'B',seed)
        right = rows_at(paths/'train.jsonl')
        edges = rows_at(paths/'aux.jsonl')
        saved = read_model(paths/'resume_latest.pt')['gate']
        assert equivalent(a.policy_rng.bit_generator.state,saved['policy_rng'])
        same_flow(left,right)
        assert [without_timing(r) for r in left] == [without_timing(r) for r in right]
        assert len(edges)==1 and saved['updates']==48 and len(saved['replay'])==4
        pair = edges[0]
        assert pair['opening']==fingerprint(pair['snapshot'])
        factual,auxiliary = (pair[k] for k in ('factual','auxiliary'))
        f,g = (v['rows'][0] for v in (factual,auxiliary))
        assert {f['action'],g['action']}=={'KEEP','REDEPLOY'}
        kept,switched = (f,g) if f['action']=='KEEP' else (g,f)
        assert kept['old_matrix']==kept['active_matrix'] and kept['switch_cost']==0.
        assert switched['active_matrix']==pair['proposal']
        assert factual['transition']['x']==auxiliary['transition']['x']
        assert auxiliary['transition']['next']==auxiliary['rows'][1]['timing']['state']
        for row in (*auxiliary['rows'],*factual['rows']):
            expected = -(10.*(row['e2e_penalized'] or 0.)+row['holding_cost']+row['switch_cost'])
            assert row['reward_raw']==expected
        links = replay_links(saved['replay'])
        added = saved['replay'][-1]
        assert all('branch' not in links[k] for k in links if isinstance(k,int))
        assert links[replay_key(added)]==added['continuation'][0]
        assert links[replay_key(added['continuation'][0])]==added['continuation'][1]
        assert replay_key(added['continuation'][1]) not in links
        assert not equivalent(b.online.state_dict(),saved['online'])
        assert not equivalent(b.target.state_dict(),saved['target'])
        assert saved['optimizer']['state']
        rejects(lambda: b.restore(a.snapshot()))
        rejects(lambda: foreign.restore(saved))
        rejects(lambda: a.collect([added]))
        b.restore(saved)
        assert equivalent(b.snapshot(),saved)
        append(paths/'aux.jsonl',[{'uncommitted':True}])
        train(root,mini,base,'B',seed,10)
        res = read_model(paths/'resume_latest.pt')
        assert res['last']['gs']==10 and res['gate']['updates']==96 and res['resumes']==1
        pairs = rows_at(paths/'aux.jsonl')
        assert len(pairs)==2 and pairs[0]==pair and pairs[0]['branch']!=pairs[1]['branch']
        assert sum('branch' in r for r in res['gate']['replay'])==2
        assert 'already committed GS10' in train(root,mini,base,'B',seed,10)
        b_updates = rows_at(paths/'updates.jsonl')
        assert all(sum(u['target_horizons'])==16 for u in b_updates)
        assert all(torch.isfinite(torch.tensor([u['gradient'],u['td_mae']])).all() for u in b_updates)
        a_cov = coverage(left,a.replay,a.replay_rng.bit_generator.state,seed,periods=5,by_source=True)
        b_cov = coverage(rows_at(paths/'train.jsonl'),res['gate']['replay'],res['gate']['replay_rng'],seed,
                         periods=5,by_source=True)
        reports = {('A',seed):dict(coverage=a_cov,updates=[dict(u,update=i) for i,u in enumerate(a_updates,1)]),
                   ('B',seed):dict(coverage=b_cov,updates=b_updates)}
        blocks = rows_at(paths/'blocks.jsonl')
        measured = {k:sum(r[k] for r in blocks) for k in cost_a}
        extra_figures(root,reports,{'A':dict(cost_a,wall_seconds=cost_a['main_wall_seconds']),
                                    'B':dict(measured,wall_seconds=res['wall_seconds'])})
        assert len(list(root.glob('fig*.png')))==3
    write(ROOT/'smoke.json',dict(source=source(),wall_seconds=perf_counter()-start,seed=seed,
                                main_and_fixed_slots=20,auxiliary_slots=8,figures=3,recovery=True,
                                measured=measured,updates=96,matched_policy_rng=True,exact_factual_ledger=True,
                                sampled_auxiliary=sum(r['n'] for r in b_cov if r['source']=='auxiliary' and r['kind']=='draws'),
                                target_horizons=[sum(r['target_horizons'][i] for r in b_updates) for i in range(3)]))
    print(ROOT/'smoke.json')
