"""Verify request identities, actual routes and injected work from all long runs."""
import argparse
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from struct import Struct

import numpy as np

from edge_d.world import make_world
from edge_joint.adaptive_storage import atomic, line
from scripts.exp024_report import read
from scripts.exp027_run import ARMS, PERIODS, RESULTS, registry


def audit_case(case):
    name = case['id']
    output = RESULTS/f'long.{name}.paths.json'
    if output.exists():
        return name
    clocks = list(range(24,24*(PERIODS+1),24))
    results = {}
    world = make_world(case['environment'],case['scene'],'holdout')
    pack = Struct('>dqqq').pack
    for arm in ARMS:
        prefix = f'long.{name}.{arm}'
        rows = read(RESULTS/f'{prefix}.periods.jsonl')
        work = read(RESULTS/f'{prefix}.work.jsonl')
        arrivals = [hashlib.sha256() for _ in clocks]
        counts = np.zeros(PERIODS,dtype=int)
        cpu = np.zeros((PERIODS,world.N,world.K),dtype=int)
        wired = np.zeros((PERIODS,len(world.w.directed),world.K),dtype=int)
        placements = Counter()
        routes = Counter()
        n = 0
        with gzip.open(RESULTS/f'{prefix}.requests.jsonl.gz','rb') as traces:
            for record in traces:
                request,t,user,access,service,finish,spans = json.loads(record)
                if request!=n:
                    raise ValueError('Missing or duplicate request number')
                n += 1
                i = int(t//24)
                arrivals[i].update(pack(t,user,access,service))
                counts[i] += 1
                chain = world.c.services[service]
                starts,legal,links = -1,(),()
                hop_index = 0
                chosen = -1
                for idx,kind,when,node in spans:
                    if kind=='origin':
                        starts = node
                    elif kind=='legal':
                        legal = node
                    elif kind=='ready':
                        chosen = node
                        if chosen not in legal:
                            raise ValueError('Illegal selected node in request trace')
                        stage = chain[idx]
                        placements[stage,chosen] += 1
                        routes[stage,starts,chosen] += 1
                        links = world.w.routes[starts][chosen]
                        hop_index = 0
                    elif kind=='hop':
                        if node!=links[hop_index]:
                            raise ValueError('Event diverged from the recorded fixed route')
                        hop_index += 1
                        j = bisect_left(clocks,when)
                        if j<PERIODS:
                            wired[j,node,chain[idx]] += 1
                    elif kind=='compute':
                        if chosen!=node or hop_index!=len(links):
                            raise ValueError('A computation started at the wrong node or with missing hops')
                        j = bisect_left(clocks,when)
                        if j<PERIODS:
                            cpu[j,node,chain[idx]] += 1
        assert [a.hexdigest() for a in arrivals]==[r['trace_sha256'] for r in rows]
        assert counts.tolist()==[r['requests'] for r in rows]
        injected_cpu = cpu @ world.w.stages[:,0]
        injected_wired = wired @ world.w.stages[:,1]
        cpu_error = np.max(np.abs(injected_cpu-np.asarray([r['cpu'] for r in work])[:,:,1]))
        wire_error = np.max(np.abs(injected_wired-np.asarray([r['wired'] for r in work])[:,:,1]))
        if max(cpu_error,wire_error)>1e-6:
            raise AssertionError('Request events do not reconcile with admitted CPU or link work')
        stages_by_edge = wired.sum(0)
        results[arm] = dict(requests=n,cpu_injection_error=float(cpu_error),wired_injection_error=float(wire_error),
                            stage_to_node=[[*k,v] for k,v in sorted(placements.items())],
                            stage_routes=[[*k,v] for k,v in sorted(routes.items())],
                            stages_by_edge=[[*map(int,k),int(stages_by_edge[tuple(k)])]
                                            for k in np.argwhere(stages_by_edge>0)])
    atomic(output,lambda f:f.write(line(dict(case=name,arms=results))))
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--available',action='store_true',help='audit completed pairs only')
    args = parser.parse_args()
    cases = [c for c in registry() if (RESULTS/f'long.{c["id"]}.paired.json').exists()]
    if len(cases)!=32 and not args.available:
        parser.error('All pairs must finish before the final request path audit')
    planned = [c for c in cases if not (RESULTS/f'long.{c["id"]}.paths.json').exists()]
    if args.workers==1:
        for case in planned:
            print(audit_case(case),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for job in as_completed([pool.submit(audit_case,c) for c in planned]):
                print(job.result(),flush=True)


if __name__ == '__main__':
    main()
