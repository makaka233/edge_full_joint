"""R12 S-only matched random D/S initialization, frozen D and continuing S replay.
Controls have same Markov main commitments as joint and do not use D auxiliary samples.
"""
import argparse,copy,gzip,pickle,time
from pathlib import Path
import numpy as np,torch
import r2b_joint as b
from r12 import joint as q
ROOT=Path(__file__).resolve().parent

def save(out,seed,gs,e,p,s,replay,rng):
 ck=dict(gs=gs,d=p.state_dict(),s=s.state_dict(),seed=seed,arm='S',
         d_hash=b.digest(p),s_hash=b.digest(s.online),env=e,rng=rng.bit_generator.state,
         replay_gzip=gzip.compress(pickle.dumps(replay,protocol=5),compresslevel=1))
 tmp=out/'latest.tmp';torch.save(ck,tmp);tmp.replace(out/'latest.pt')

def run(seed,steps,resume):
 assert steps>=0 and steps%2==0
 c,x,e,s,p=q.setup(seed,'D');initial=b.digest(p)
 out=ROOT/'experiments'/'S'/'r51/results'/f'{seed}_D';out.mkdir(parents=True,exist_ok=resume)
 rng=np.random.default_rng(seed+771);replay=b.SReplay(c.s_replay_transitions,c.replay_max_age);start=0
 if resume:
  ck=torch.load(out/'latest.pt',weights_only=False,map_location='cpu')
  p.load_state_dict(ck['d']);s.load_state_dict(ck['s']);e=ck['env'];rng.bit_generator.state=ck['rng'];start=ck['gs']
  assert start%2==0 and b.digest(p)==ck['d_hash'] and b.digest(s.online)==ck['s_hash']
  replay=pickle.loads(gzip.decompress(ck['replay_gzip']));q.repair_logs(out,start)
  evfile=out/'eval.jsonl'
  scored={__import__('json').loads(line)['checkpoint'] for line in evfile.read_text().splitlines()} if evfile.exists() else set()
  if start not in scored and (start==0 or start%8==0):
   score=q.evaluate(c,x,p,s,seed,'S',start,evfile);q.maybe_best(out,score,start,seed,'S')
 else:
  save(out,seed,0,e,p,s,replay,rng);score=q.evaluate(c,x,p,s,seed,'S',0,out/'eval.jsonl');q.maybe_best(out,score,0,seed,'S')
 for gs in range(start+1,steps+1):
  tick=time.monotonic();o,a,X,di=b.execute(e,p,b.Decode(e),rng)
  ph=q.physical(e,o,a,X)
  eps=c.epsilon_end+(c.epsilon_start-c.epsilon_end)*max(0,1-((gs-1)//2*2)/c.epsilon_decay_steps)
  ps=int(b.rng_for(seed,900,gs).integers(0,2**31-1))
  r,ts=q.rec(e,s,p,X,a,ps,gs,eps,collect=True)
  assert ts and all(z.mask[z.action] and z.collection_step==gs for z in ts)
  replay.add(ts);replay.prune(gs)
  r.update(ph,**di,arm='S',collected=len(ts),replay_size=len(replay),total_wall=time.monotonic()-tick)
  b.write(out/'train.jsonl',r)
  if gs%2==0:
   oldsh=b.digest(s.online)
   vals=[s.update(replay.sample(x['s_batch'],gs,b.rng_for(seed,902,gs,j))) for j in range(2*x['s_updates_per_period'])]
   s.version+=1
   b.write(out/'updates.jsonl',dict(gs=gs,d_before=initial,d_after=b.digest(p),s_before=oldsh,s_after=b.digest(s.online),s_updates=s.updates,s_loss=float(np.mean([v['s_loss'] for v in vals]))))
   save(out,seed,gs,e,p,s,replay,rng)
  b.write(out/'wall.jsonl',dict(gs=gs,seed=seed,arm='S',wall_complete_s=time.monotonic()-tick,physical_commit=1,aux_d80_calls=0,D_updates=0,S_updates=s.updates,latest_d_hash=b.digest(p),latest_s_hash=b.digest(s.online)))
  print(seed,'S',gs,r['reward'],r['penalized_delay'],r['failed'],flush=True)
  if gs % 8 == 0 or gs==steps:
   score=q.evaluate(c,x,p,s,seed,'S',gs,out/'eval.jsonl');q.maybe_best(out,score,gs,seed,'S')
 assert b.digest(p)==initial,'D modified in S-only control'
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--steps',type=int,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();run(a.seed,a.steps,a.resume)
