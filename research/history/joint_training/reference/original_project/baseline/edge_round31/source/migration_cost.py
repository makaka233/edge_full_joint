"""Optional migration-aware reward: route work in physical units, not a transition simulator.

For each addition, choose nearest existing image source (old matrix), otherwise
registry node 0. A stage's assumed transferable image is a fraction of its SSD
assets. Sum Mbit on directed links; bottleneck seconds is a lower bound on
completion for these fixed routes with no competing business traffic. This is
only a cost proxy: no simultaneous old/new service or actual downtime modeled.
"""
import numpy as np

def account(w,old,new,p):
    fraction=float(p['image_fraction']);weight=float(p['weight']);registry=int(p['registry_node'])
    if not 0<fraction<=1 or weight<0 or not 0<=registry<w.c.n:raise ValueError('invalid migration scenario')
    work=np.zeros(len(w.directed));gb=0.
    for k,n in np.argwhere(new&~old):
        sources=np.flatnonzero(old[k]);src=int(sources[np.argmin(w.path_seconds_per_mbit[sources,n])]) if len(sources) else registry
        size=float(w.stages[k,4]*fraction);gb+=size
        work[list(w.routes[src][n])]+=size*8000.
    cap=np.array([e[2] for e in w.directed]);busy=work/cap;bound=float(busy.max(initial=0));return dict(migration_cost=weight*bound/w.c.period_seconds,migration_image_gb=gb,migration_link_mbit=float(work.sum()),migration_bottleneck_seconds=bound)
