"""One fixed policy-generated matrix, one fixed request trace, two ORIGINAL CPU disciplines.

This compares CPU scheduling only; wired links ALWAYS remain sqrt-KKT.
It is a physical compatibility probe, not a training or convergence experiment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from research.history.legacy_edit.training import new_actor
from research.history.legacy_edit.rollout import sample_layout
from edge_d.world import make_world
from edge_d.physical import evaluate_layout

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='Write JSON to an explicit path; otherwise print only')
    args=parser.parse_args()
    seed=20261113;ps=1234;ph=2345
    record,_=sample_layout(seed=seed,scene=0,split='train',actor=new_actor(seed),
          policy_seed=ps,physical_seed=ph,max_actions=700)
    packed=bytes.fromhex(record['matrix_packed_hex'])
    X=np.unpackbits(np.frombuffer(packed,dtype=np.uint8))[:28*32].astype(bool).reshape(28,32)
    metrics={}
    for mode in ('finish_pf','kkt'):
        e=make_world(seed,0,'train')
        e.c.cpu_allocator=mode
        info=evaluate_layout(e,X,ph,20.)
        metrics[mode]={'reward':float(info['reward']),'penalized_e2e':float(info['penalized_delay']),
                       'censored':int(info['failed']),'trace':info['nested_trace_hash'],
                       'max_conservation_error':max(w['conservation'] for w in info['windows'])}
    assert metrics['kkt']['trace']==metrics['finish_pf']['trace'],'Request trace differed'
    assert all(np.isfinite(v['reward']) and v['max_conservation_error']<1e-5 for v in metrics.values())
    assert abs(metrics['finish_pf']['reward']-record['reward'])<1e-8
    output={'source':'original physics.py and cpu_allocation.py','matrix':record['matrix_sha256'],
            'wireless':'equal share original','wired':'sqrt KKT original (both runs)',
            'cpu':'finish_pf versus original sqrt KKT','results':metrics}
    if args.out:
        args.out.write_text(json.dumps(output,indent=2),encoding='utf8')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
