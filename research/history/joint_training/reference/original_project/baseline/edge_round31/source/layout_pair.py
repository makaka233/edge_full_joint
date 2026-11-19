"""One actual physical alternative per eligible macro state, training only.
The main action is already committed. Never select between the two layouts.
"""
import copy, numpy as np

def collect_layout_pair(pre,post,d,s,info,seed,gs,epsilon,scale,rng_for):
    if not pre.old.any(1).all():return None,None
    z=copy.deepcopy(pre);actual=post.old.copy();st=pre.state();st.phase=2;st.draft=actual
    if info['decision']=='KEEP':
        z.apply(z.BUILD);rng=rng_for(seed,28271,gs)
        while True:
            a,_,_=d.act(z.state(),z.legal(),rng);alt=z.apply(a)
            if alt is not None:break
        label='BUILD'
    else:
        alt=z.apply(z.KEEP);label='KEEP'
    ast=pre.state();ast.phase=2;ast.draft=alt.copy()
    ns,ai,_,_,_=z.commit(alt,s.scheduler(rng_for(seed,30,gs),epsilon),int(rng_for(seed,90,gs).integers(0,2**31-1)),gs,False)
    assert ai['trace_sha256']==info['trace_sha256'] and np.array_equal(ns.demand,post.state().demand)
    pair=([st,ast],[info['reward']*scale,ai['reward']*scale])
    log=dict(gs=gs,actual=info['decision'],alternative=label,actual_reward=info['reward'],alternative_reward=ai['reward'],alternative_n=ai['n'],alternative_failed=ai['failed'],alternative_replicas=ai['replicas'],alternative_delay=ai['penalized_delay'],trace_sha256=ai['trace_sha256'],alternative_matrix_hex=np.packbits(alt.ravel()).tobytes().hex())
    return pair,log
