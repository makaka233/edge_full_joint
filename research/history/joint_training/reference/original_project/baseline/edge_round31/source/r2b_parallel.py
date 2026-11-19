"""Exact block-parallel D120 continuation: S,D fixed within each four-GS block.
The short main commits remain sequential; long evaluations are independent read-only
copies of their pre-commit state. Results are joined in GS order before any update.
No reward, action, observation, random stream, or optimizer rule is changed.
"""
from r2b_joint import *
from r2a_sampling import nested,hasharr,Simulator,summarize,agg
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def long_job(payload):
    e,X,ss,seed,eps=pickle.loads(payload);torch.set_num_threads(1)
    s=SchedulingAgent(e.c);s.load_state_dict(ss)
    tick=time.monotonic();b,chosen,arr,tr=nested(e,seed)
    sim=Simulator(e.w,X,s.scheduler(np.random.default_rng(seed+418001),eps),b-30,collect=False,cohort=(b,b+120,b+128));sim.add_arrivals(arr);sim.run_until(b+128);bins={}
    for j in range(0,120,6):
        qs=[q for q in sim.requests if b+j<=q.arrival.time<b+j+6];v=summarize(qs,b+j+14,e.failure_penalty);v['hash']=hasharr([a for a in arr if b+j<=a.time<b+j+6]);bins[str(j)]=v
    z=agg(list(bins.values()))
    for j,(_,items) in zip(chosen,tr):assert bins[str(j)]['hash']==hasharr([a for a in items if b+j<=a.time<b+j+6])
    return dict(d_penalized_delay=z['delay'],d_failed=z['failed'],d_n=z['n'],d_protocol='D120',nested_trace_hash=hasharr(arr),A_cohorts=list(chosen),long_wall_seconds=time.monotonic()-tick)

def run_parallel(seed,steps,workers=2):
    c,x,e,s,p=setup(seed);torch.manual_seed(seed+32);v=Critic(len(e.w.context()));ao=torch.optim.Adam(p.parameters(),lr=x['r0_d_lr']);vo=torch.optim.Adam(v.parameters(),lr=.003);rng=np.random.default_rng(seed+771);out=OUT/f'D120_{seed}';ck=torch.load(out/'latest.pt',weights_only=False,map_location='cpu');p.load_state_dict(ck['d']);v.load_state_dict(ck['critic']);s.load_state_dict(ck['s']);ao.load_state_dict(ck['actor_opt']);vo.load_state_dict(ck['critic_opt']);e=ck['env'];dec=Decode(e);rng.bit_generator.state=ck['rng'];start=ck['gs'];best=ck['best'];assert start%BLOCK==0 and digest(p)==ck['d_hash'] and digest(v)==ck['v_hash'] and digest(s.online)==ck['s_hash']
    with gzip.open(out/'latest_replay.pkl.gz','rb') as f:replay=pickle.load(f)
    print('PARALLEL EXACT',seed,start,workers,flush=True)
    # On this bounded CPU run, allow the short control arms to release their replay memory first.
    waiting=[OUT/f'{a}_{z}_extend.exit' for a in ('A','s_only') for z in (20261114,20261113) if (OUT/f'{a}_{z}_extend.log').exists()]
    while any(not f.exists() for f in waiting):time.sleep(2)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for startblock in range(start+1,steps+1,BLOCK):
            tickblock=time.monotonic();roll=[];ss=s.state_dict()
            for gs in range(startblock,startblock+BLOCK):
                tick=time.monotonic();o,a,X,di=execute(e,p,dec,rng);before=statehash(e);old=hexmat(e.old);eps=c.epsilon_end+(c.epsilon_start-c.epsilon_end)*max(0,1-((gs-1)//BLOCK*BLOCK)/c.epsilon_decay_steps);ps=int(rng_for(seed,900,gs).integers(0,2**31-1));e.current_s_version=s.version
                future=pool.submit(long_job,pickle.dumps((e,X,ss,ps,eps),protocol=5))
                _,r,ts,terminated,truncated=commit_nested(e,X,s,ps,gs,eps,'A',True);assert not terminated and not truncated and ts and all(t.mask[t.action] for t in ts);replay.add(ts);replay.prune(gs);nxt=observe(e)
                with torch.no_grad():vv=float(v(o));lp=float(logprob(p,o,a)[0])
                row=dict(gs=gs,seed=seed,arm='D120',action=a,mode=a['mode'],old_hex=old,matrix_hex=hexmat(X),state_hash=before,next_state_hash=statehash(e),d_hash=digest(p),s_hash=digest(s.online),s_version=s.version,epsilon=eps,collected=len(ts),d_logp=lp,value_norm=vv,**di,**r)
                roll.append(dict(o=o,a=a,nxt=nxt,gs=gs,row=row,future=future,short_wall=time.monotonic()-tick))
            for b in roll:
                z=b['future'].result();r=b['row'];assert r['nested_trace_hash']==z['nested_trace_hash'] and r['A_cohorts']==z['A_cohorts'];r.update(z);r['d_reward']=-(e.cost['delay_weight']*r['d_penalized_delay']+r['holding_cost']+r['switch_cost']);r['wall_seconds']=b['short_wall']+z['long_wall_seconds'];b['r']=.1*r['d_reward'];write(out/'train.jsonl',r);print(seed,'D120',b['gs'],round(r['reward'],3),round(r['d_reward'],3),r['failed'],flush=True)
            gs=roll[-1]['gs']
            with torch.no_grad():
                vals=[float(v(b['o'])) for b in roll];boot=float(v(roll[-1]['nxt']));adv,tgt=value_targets([SCALE*b['r'] for b in roll],vals,boot,GAMMA,x['r0_lambda']);targets=[(b['o'],float(t)) for b,t in zip(roll,tgt)]
            losses=[-logprob(p,b['o'],b['a'])[0]*float(ad)/SCALE for b,ad in zip(roll,adv)];al=torch.stack(losses).mean();ao.zero_grad();al.backward();gn=float(nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True));ao.step();vt=torch.tensor([t for _,t in targets]);vl=[]
            for _ in range(CRITIC_STEPS):
                pred=torch.stack([v(o) for o,_ in targets]);loss=F.smooth_l1_loss(pred,vt);vo.zero_grad();loss.backward();nn.utils.clip_grad_norm_(v.parameters(),5,error_if_nonfinite=True);vo.step();vl.append(float(loss.detach()))
            sl=[]
            for ui in range(x['s_updates_per_period']*BLOCK):sl.append(s.update(replay.sample(x['s_batch'],gs,rng_for(seed,902,gs,ui))))
            s.version+=1;write(out/'updates.jsonl',dict(gs=gs,actor_loss=float(al.detach()),actor_grad=gn,v_loss_first=vl[0],v_loss_last=vl[-1],values=vals,targets=vt.tolist(),adv=adv.tolist(),bootstrap=boot,qdiag=[],d_hash=digest(p),v_hash=digest(v),s_hash=digest(s.online),s_updates=s.updates,s_loss=float(np.mean([a['s_loss'] for a in sl])),block_wall_seconds=time.monotonic()-tickblock,parallel_workers=workers))
            if gs in (8,16,32,64):
                score=r0.evaluate(p,s,c,x,seed,gs,out/'eval.jsonl','D120')
                if score>best:best=score;save1(out,'best.pt',p,v,s,ao,vo,e,replay,rng,gs,seed,'D120',best)
            save1(out,'latest.pt',p,v,s,ao,vo,e,replay,rng,gs,seed,'D120',best)
            if gs==32:save1(out,'gs32.pt',p,v,s,ao,vo,e,replay,rng,gs,seed,'D120',best)
    (out/'completed.json').write_text(json.dumps(dict(gs=steps,actor_updates=steps//BLOCK,critic_updates=steps//BLOCK*CRITIC_STEPS,s_updates=s.updates),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--steps',type=int,default=64);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();run_parallel(a.seed,a.steps,a.workers)
