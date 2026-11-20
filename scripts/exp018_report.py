"""Verify EXP018 paired physical evidence and preregistered progression gates."""
import argparse
import json
from pathlib import Path
from shutil import copy2

import numpy as np
import torch

from edge_joint.inventory_joint import FIXED,SEEDS
from edge_s.agent import SchedulingAgent
from scripts.exp009 import _e2e,rows,write_json
from scripts.exp010_report import archive
from scripts.exp016_report import paired,require,summarize,verify_short
from scripts.exp017_forensics import CASES
from scripts.exp017_report import spent_time


REFERENCE=Path('research/EXP017')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def cases(work,out,weights):
    old=read(REFERENCE/'metrics.json')['forensics']
    starts={}
    audit={}
    for seed,gs_list in CASES.items():
        w=SchedulingAgent(seed).online.encoder[0].weight.detach()
        starts[str(seed)]=dict(inputs=int(w.shape[1]),base_norm=float(w[:,:16].norm()),
                               competition_norm=float(w[:,16:24].norm()))
        for gs in gs_list:
            key=f'{seed}_{gs}'
            source=work/f'case_{key}_R'
            raw=read(source/'result.json')
            reference=read(REFERENCE/'raw'/'forensics'/f'case_{key}_A'/'result.json')
            for field in ('matrix_packed_hex','s_policy_fingerprint','trace_sha256','nested_trace_hash','s_version'):
                require(raw[field]==reference[field],f'{key}: unequal causal state: {field}')
            require(raw['reward']==raw['reward'] and sum(raw['chosen_nodes'])==raw['s_decisions'],
                    f'{key}: invalid resource action ledger')
            verify_short(raw)
            decisions=rows(source/'decisions.jsonl.gz')
            reserve=(1.-raw['resource_weight'])*raw['epsilon']
            require(all(d['chosen'] in d['legal'] and d['chosen_probability']>=reserve/len(d['legal'])-1e-12
                        for d in decisions),'Illegal action or invalid full-support mixture')
            count=sum(x['phase']=='official' for x in decisions)
            require(count==raw['s_decisions'],'Physical action trace count differs')
            cohort=[q for q in rows(source/'requests.jsonl.gz') if q['cohort']]
            require(len(cohort)==raw['requests'] and sum(q['official_censored'] for q in cohort)==raw['censored'],
                    'Cohort count not reconciled')
            completed=[q['final_finish']-q['arrival'] for q in cohort if q['final_finish'] is not None]
            b=dict(reward=raw['reward'],completed=raw['completed'],censored=raw['censored'],
                   penalized_e2e=_e2e(raw['penalty_adjusted_delays_seconds']),
                   completed_e2e=_e2e(raw['completed_delays_seconds']),
                   actual_after_shadow=_e2e(completed),actual_residence=spent_time(cohort),
                   unresolved=sum(q['final_censored'] for q in cohort),peak_resources=raw['peak_resources'],
                   visits=raw['visited_node_fraction'],soft_draws=raw['resource_draws'],
                   decisions=raw['s_decisions'],resource_weight=raw['resource_weight'])
            a=old[key]['arms']['A'].copy()
            a['visits']=reference['visited_node_fraction']
            a['decisions']=reference['s_decisions']
            actual_a,actual_b=(r['actual_after_shadow'] for r in (a,b))
            rule=b['reward']-a['reward']
            weight=torch.load(weights/f'weights_{seed}'/f's{gs}.pt',weights_only=True)['s']['encoder.0.weight']
            audit[key]=dict(role=raw['registered_role'],A=a,R=b,rule_reward_delta=rule,
                            s_competition_norm=float(weight[:,16:24].norm()),
                            official_reward_gain_and_real_e2e=rule>=.5 and actual_b['mean']<actual_a['mean'],
                            nondeterioration=dict(reward=rule>=-2.,censored=b['censored']<=a['censored'],
                                                  shadow_p99=actual_b['p99']<=1.10*actual_a['p99'],
                                                  visits=b['visits']>=.9*a['visits'],unresolved=b['unresolved']<=a['unresolved']))
            target=out/'raw'/'forensics'/source.name
            target.mkdir(parents=True,exist_ok=True)
            for name in ('result.json','decisions.jsonl.gz','requests.jsonl.gz'):
                copy2(source/name,target/name)
    checks=dict(three_useful=sum(t['official_reward_gain_and_real_e2e'] for t in audit.values())>=3,
                every_case_nondeteriorates=all(all(t['nondeterioration'].values()) for t in audit.values()),
                lower_censor=sum(x['R']['censored']-x['A']['censored'] for x in audit.values())<0)
    for kind in ('cpu','wired'):
        checks[f'one_{kind}_improves']=any(t['R']['peak_resources'][kind]['max_jobs']<
                                          t['A']['peak_resources'][kind]['max_jobs'] for t in audit.values())
    summary=dict(cases=audit,initial_network=starts,checks=checks,pass_causal=all(checks.values()))
    write_json(out/'causal.json',summary)


def pilot(work,out):
    reference=read(REFERENCE/'metrics.json')['pilot']
    summary={}
    for seed in SEEDS:
        base=reference[str(seed)]['A']
        root=work/f'seed{seed}'
        result=read(root/'fit.json')
        require((result['gs'],result['d_updates'],result['s_updates'])==(64,64,512),'Incorrect pilot budgets')
        logs={name:rows(root/f'{name}.jsonl') for name in ('train','antithetic','updates','fixed')}
        paired(logs['train'],logs['antithetic'],logs['updates'],64)
        for name in ('train','antithetic','fixed'):
            for r in logs[name]:
                verify_short(r)
        ref=REFERENCE/'raw'/f'seed{seed}_A'
        original=rows(ref/'train.jsonl.gz')
        require([r['trace_sha256'] for r in original]==[r['trace_sha256'] for r in logs['train']],
                'Pilot request flow differs')
        for key in ('reward','trace_sha256','matrix_packed_hex'):
            old=[r[key] for r in rows(ref/'fixed.jsonl.gz') if r['gs']==0]
            new=[r[key] for r in logs['fixed'] if r['gs']==0]
            require(old==new,'Pilot greedy initialization differs from the baseline')
        tail=[r for r in logs['fixed'] if r['gs']==64]
        require([r['scene'] for r in tail]==list(FIXED) and not any(r['resource_draws'] for r in logs['fixed']),
                'Fixed evaluation did not use the original greedy rule')
        t,f=summarize(logs['train']),summarize(tail)
        a,g=base['train'],base['fixed64']
        visits=float(np.mean([r['visited_node_fraction'] for r in logs['train'][:32]]))
        flags=dict(fewer_spikes=t['spikes']<a['spikes'],worst5=t['worst5']>=a['worst5']+.5,
                   worst_reward=t['worst_reward']>=a['worst_reward'],censor=t['censored']<=a['censored'],
                   visits=visits>=base['visits']-.02,reward=f['reward']>=g['reward']-.05,
                   e2e=f['penalized_e2e']['mean']<=g['penalized_e2e']['mean']+.005,
                   p99=f['penalized_e2e']['p99']<=g['penalized_e2e']['p99']+.040)
        summary[str(seed)]=dict(A=base,R=dict(train=t,fixed64=f,visits=visits,
                                             online_seconds=result['online_seconds'],
                                             wall_seconds=result['wall_seconds'],
                                             direct_rule_decisions=sum(r['resource_draws'] for r in logs['train'])),
                                criteria=flags,passed=all(flags.values()))
        for name,data in logs.items():
            archive(out/'raw'/f'seed{seed}_R64'/f'{name}.jsonl.gz',data)
    write_json(out/'pilot.json',dict(seeds=summary,pass_pilot=all(v['passed'] for v in summary.values())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('cases','pilot'))
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--weights',type=Path)
    a=parser.parse_args()
    if a.action=='cases':
        cases(a.work,a.out,a.weights or a.work)
    else:
        pilot(a.work,a.out)
