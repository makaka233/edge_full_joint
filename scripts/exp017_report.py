"""Validate EXP017 paired evidence and apply the preregistered pilot gate."""
import argparse
import base64
import json
from pathlib import Path
from shutil import copy2

import numpy as np

from edge_joint.exploration import ARMS,normalized_entropy,probabilities,settings
from edge_joint.inventory_joint import FIXED,SEEDS
from scripts.exp009 import _e2e,rows,write_json
from scripts.exp010_report import archive
from scripts.exp016_report import paired,require,summarize,verify_short
from scripts.exp017_forensics import CASES,base_rows


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def conditional_states(path,epsilon,gs):
    node_uniform,greedy=np.zeros(32),np.zeros(32)
    expected={a:np.zeros(32) for a in ARMS}
    entropy={a:0. for a in ARMS}
    multi=0
    for row in rows(path):
        if row['phase']!='official' or len(row['legal'])<2:
            continue
        nodes=np.array(row['legal'])
        q=np.frombuffer(base64.b64decode(row['q_f32le']),dtype='<f4').astype(float)
        node_uniform[nodes]+=1./len(nodes)
        greedy[row['greedy']]+=1.
        multi+=1
        for arm in ARMS:
            p=probabilities(q[nodes],*settings(arm,gs,epsilon))
            expected[arm][nodes]+=p
            entropy[arm]+=normalized_entropy(p)
    leader=int(np.argmax(greedy))
    return dict(multi_actions=multi, q_top_node=leader,greedy_fraction=float(greedy[leader]/multi),
                uniform_opportunity=float(node_uniform[leader]/multi),
                expected_choice_at_q_top={a:float(expected[a][leader]/multi) for a in ARMS},
                per_arm_normalized_entropy={a:entropy[a]/multi for a in ARMS},
                uniform_node_distribution=(node_uniform/multi).tolist(),greedy_node_distribution=(greedy/multi).tolist())


def spent_time(requests):
    residence={k:[] for k in ('wireless','wired','cpu')}
    for request in requests:
        if request['final_finish'] is None:
            continue
        stages={}
        for index,event,time,_ in request['events']:
            stages.setdefault(index,{})[event]=time
        upload=stages[0]['ready']-request['arrival']
        transport=sum(s['compute']-s['ready'] for s in stages.values())
        compute=sum(s['finish']-s['compute'] for s in stages.values())
        require(abs(upload+transport+compute-(request['final_finish']-request['arrival']))<1e-8,
                'Resource residence does not add up to actual request delay')
        for key,value in zip(residence,(upload,transport,compute),strict=True):
            residence[key].append(value)
    return {name:_e2e(values) for name,values in residence.items()}


def forensic(work):
    results={}
    for seed, targets in CASES.items():
        for gs in targets:
            name=f'{seed}_{gs}'
            a_dir=work/f'case_{name}_A'
            a=read_json(a_dir/'result.json')
            groups={}
            for arm in ARMS:
                path=work/f'case_{name}_{arm}'
                result=read_json(path/'result.json')
                for field in ('matrix_packed_hex','s_policy_fingerprint','trace_sha256','nested_trace_hash'):
                    require(a[field]==result[field],f'Physical counterfactual changed {field}')
                requests=[r for r in rows(path/'requests.jsonl.gz') if r['cohort']]
                recovered=[r for r in requests if r['official_censored'] and not r['final_censored']]
                unresolved=sum(r['final_censored'] for r in requests)
                finishes=[r['final_finish']-r['arrival'] for r in requests if r['final_finish'] is not None]
                deadline={w['window']:w['official_stop'] for w in result['shadow']}
                added=[r['final_finish']-deadline[r['window']] for r in recovered]
                require(len(requests)==result['requests'] and
                        len(added)+unresolved==result['censored'],'Extended drain ledger differs')
                groups[arm]=dict(reward=result['reward'],completed=result['completed'],censored=result['censored'],
                                 penalized_e2e=_e2e(result['penalty_adjusted_delays_seconds']),
                                 completed_e2e=_e2e(result['completed_delays_seconds']),
                                 peak_resources=result['peak_resources'],visited_nodes=sum(x>0 for x in result['chosen_nodes']),
                                 top_node_share=result['largest_node_share'],shadow_recovered=len(added),unresolved=unresolved,
                                 actual_after_shadow=_e2e(finishes),extra_delay_after_cutoff=_e2e(added),
                                 actual_residence=spent_time(requests),
                                 recovered_beyond_penalty=sum(d>10. for d in added))
            results[name]=dict(role=a['registered_role'],original_exact=a['exact_original'],arms=groups,
                               same_states=conditional_states(a_dir/'decisions.jsonl.gz',a['epsilon'],gs))
    return results


def pilot(work,out=None):
    result={}
    for seed in SEEDS:
        original={k:base_rows(seed,k)[:64] for k in ('train','antithetic')}
        original_fixed=[x for x in base_rows(seed,'fixed') if x['gs'] in (0,64)]
        scores={}
        traces={}
        fixed_traces={}
        initial_rows={}
        for arm in ARMS:
            root=work/f'seed{seed}_{arm}'
            fit=read_json(root/'fit.json')
            require((fit['gs'],fit['d_updates'],fit['s_updates'])==(64,64,512),'Pilot optimizer budget differs')
            data={k:rows(root/f'{k}.jsonl') for k in ('train','antithetic','updates','fixed')}
            groups=(data['train'],data['antithetic'],data['updates'])
            paired(*groups,64)
            for file in ('train','antithetic','fixed'):
                for row in data[file]:
                    verify_short(row)
            changed={agent:sum(u[f'{agent}_parameter_delta']>0. for u in data['updates']) for agent in ('d','s')}
            require(all(changed.values()),'Expected both agents to update')
            require(all(sum(r['chosen_nodes'])==r['s_decisions'] for r in data['train']+data['antithetic']),
                    'Exploration decision count differs')
            for row in data['train']+data['antithetic']:
                registered=settings(arm,row['gs'],row['epsilon'])
                observed=tuple(row[k] for k in ('effective_epsilon','softmax_weight','temperature'))
                require(registered==observed,'Actual exploratory distribution differs from protocol')
            traces[arm]=[r['trace_sha256'] for r in data['train']]
            if arm=='A':
                for file in ('train','antithetic'):
                    require(all(all(a[k]==b[k] for k in ('reward','matrix_packed_hex','s_policy_fingerprint','s_version',
                                                        's_decisions','penalized_e2e','censored'))
                                for a,b in zip(data[file],original[file],strict=True)),'A changed EXP016 physiology')
                require(all(a['reward']==b['reward'] and a['matrix_packed_hex']==b['matrix_packed_hex']
                            for a,b in zip(data['fixed'],original_fixed,strict=True)),'A did not reproduce fixed outcomes')
            init=[r for r in data['fixed'] if r['gs']==0]
            fixed=[r for r in data['fixed'] if r['gs']==64]
            require([r['scene'] for r in init]==[r['scene'] for r in fixed]==list(FIXED),'Fixed cohort differs')
            fixed_traces[arm]=[r['trace_sha256'] for r in data['fixed']]
            initial_rows[arm]=[(r['reward'],r['matrix_packed_hex']) for r in init]
            beginning=data['train'][:32]
            n=sum(r['multi_action_decisions'] for r in beginning)
            mean=float(sum(r['conditional_entropy']*r['multi_action_decisions']
                           for r in beginning if r['multi_action_decisions'])/n)
            scores[arm]=dict(train=summarize(data['train']),fixed0=summarize(init),fixed64=summarize(fixed),
                             parameter_change_steps=changed,
                             visits=float(np.mean([r['visited_node_fraction'] for r in beginning])),
                             entropy=mean,largest_node_share=float(np.mean([r['largest_node_share'] for r in beginning])),
                             online_seconds=fit['online_seconds'],wall_seconds=fit['wall_seconds'])
            if out is not None:
                for sheet,r in data.items():
                    archive(out/'raw'/f'seed{seed}_{arm}'/f'{sheet}.jsonl.gz',r)
        require(all(p==traces['A'] for p in traces.values()),'Pilot demand streams differ')
        require(all(p==fixed_traces['A'] for p in fixed_traces.values()),'Fixed validation request streams differ')
        require(all(p==initial_rows['A'] for p in initial_rows.values()),'Initial physical policies differ')
        result[str(seed)]=scores
    return result


def gate(pilot_results):
    verdict={}
    eligible=[]
    for arm in ARMS[1:]:
        verdict[arm]={}
        gains=[]
        for seed,group in pilot_results.items():
            a,b=group['A'],group[arm]
            left,right=a['train'],b['train']
            initial,follow=a['fixed64'],b['fixed64']
            gain=right['worst5']-left['worst5']
            gains.append(gain)
            verdict[arm][seed]=dict(fewer_spikes=right['spikes']<left['spikes'],worst5_gain=gain>=.5,
                                    worst_reward=right['worst_reward']>=left['worst_reward'],
                                    censor=right['censored']<=left['censored'],
                                    entropy=b['entropy']>=a['entropy'],visits=b['visits']>=a['visits']-.02,
                                    fixed_reward=follow['reward']>=initial['reward']-.05,
                                    fixed_mean=follow['penalized_e2e']['mean']<=initial['penalized_e2e']['mean']+.005,
                                    fixed_p99=follow['penalized_e2e']['p99']<=initial['penalized_e2e']['p99']+.040)
        if all(all(x.values()) for x in verdict[arm].values()):
            eligible.append((min(gains),-ARMS.index(arm),arm))
    return dict(selected=max(eligible)[2] if eligible else None,criteria=verdict)


def publish_evidence(work,out):
    for seed,targets in CASES.items():
        prefix=out/'raw'/f'prefix_{seed}.json'
        prefix.parent.mkdir(parents=True,exist_ok=True)
        copy2(work/f'weights_{seed}'/'prefix.json',prefix)
        for gs in targets:
            for arm in ARMS:
                case=f'case_{seed}_{gs}_{arm}'
                folder=out/'raw'/'forensics'/case
                folder.mkdir(parents=True,exist_ok=True)
                for name in ('result.json','decisions.jsonl.gz','requests.jsonl.gz'):
                    copy2(work/case/name,folder/name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gate-only',action='store_true')
    parser.add_argument('--redraw',action='store_true')
    a=parser.parse_args()
    if a.redraw:
        from scripts.exp017_figures import figures
        metrics=read_json(a.out/'metrics.json')
        for seed in SEEDS:
            figures(a.out,seed,metrics)
        raise SystemExit(0)
    if a.work is None:
        parser.error('--work must locate the original ledgers')
    a.out.mkdir(parents=True,exist_ok=True)
    case_results=forensic(a.work)
    pilot_results=pilot(a.work,None if a.gate_only else a.out)
    metrics=dict(forensics=case_results,pilot=pilot_results,gate=gate(pilot_results))
    write_json(a.out/'metrics.json',metrics)
    if not a.gate_only:
        from scripts.exp017_figures import figures
        publish_evidence(a.work,a.out)
        for seed in SEEDS:
            figures(a.out,seed,metrics)
