"""R5 random joint: main one commit, auxiliary H3 frozen-version, two-GS blocks."""
import argparse,copy,json,time,gzip,pickle
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
import r2b_joint as b
from interaction import Interaction
from r12.credit import make_pair,loss_for,gradient_contract
from r12.stability import ALGORITHM_VERSION,entropy_bonus,bounded_adam_step
from r9.continuous_credit import measure
ROOT=Path(__file__).resolve().parent
VALIDATION_SEEDS=(975101,975103)
class Policy(Interaction):
 def __init__(self,base,arm):
  super().__init__(base)
  # Separate macro trainable head; random encoder stays frozen as R41.
  for v in self.base.macro.parameters():v.requires_grad_(True)
  self.arm=arm
  if arm in ("C","D","F"):
   self.macro_base=copy.deepcopy(base)
   for n,v in self.macro_base.named_parameters():v.requires_grad_(arm!="F" and n.startswith("macro."))
   for n,v in self.base.named_parameters():v.requires_grad_(not n.startswith(("macro.","value.")))
  self.macro_res=nn.Linear(4,1);nn.init.zeros_(self.macro_res.weight);nn.init.zeros_(self.macro_res.bias)
 def forward(self,o):
  mo,rf,ct,nl,v=super().forward(o);mo=self.macro_base(o)[0] if self.arm in ("C","D","F") else mo;x,_,old,_=o;cpu=x[0,:,5]*20;need=x[:,0,0]*2*x[:,0,9]*8;cap=old.float()@cpu
  rho=need/cap.clamp_min(1e-8);features=torch.stack((torch.log1p(rho.clamp_max(100)).max(),torch.log1p(rho.clamp_max(100)).mean(),need.sum()/cpu.sum(),old.float().mean()))
  if self.arm=="F":features=(features-self.macro_mu)/self.macro_sd
  return mo+torch.cat((mo.new_zeros(1),self.macro_res(features))),rf,ct,nl,v

def setup(seed,arm="A"):
 c,x,e,s,base=b.setup(seed);p=Policy(base,arm)
 return c,x,e,s,p

def rec(e,s,p,X,a,ps,gs,eps,protocol='A',collect=False):
 old=e.old.copy();h=b.statehash(e);t=time.monotonic();e.current_s_version=s.version
 _,r,ts,term,trunc=b.commit_nested(e,X,s,ps,gs,eps,protocol,collect)
 assert not term and not trunc and np.array_equal(e.old,X)
 need=e.w.node_demand()[:,e.w.stage_service].sum(0)*e.w.stages[:,0] # next demand, not used for pre-state pressure
 return dict(gs=gs,action=a,old_hex=b.hexmat(old),matrix_hex=b.hexmat(X),state_hash=h,next_state_hash=b.statehash(e),request_seed=ps,d_hash=b.digest(p),s_hash=b.digest(s.online),s_version=s.version,epsilon=eps,wall=time.monotonic()-t,**r),ts

def proposal(o,a):
 y=o[2].numpy().copy()
 if a['mode']==1:
  for k in np.flatnonzero(a['refresh']):y[k]=False;y[k,a['order'][k]]=True
 return y

def physical(e,o,a,X):
 y=proposal(o,a);need=e.w.node_demand()[:,e.w.stage_service].sum(0)*e.w.stages[:,0];cap=X@e.w.nodes[:,0]
 return dict(proposal_hex=b.hexmat(y),proposal_counts=y.sum(1).tolist(),actual_counts=X.sum(1).tolist(),actual_cpu=cap.tolist(),pressure=(need/cap).tolist(),projection_hamming=int(np.count_nonzero(X!=y)))

def branch(base,s,p,a,seed,gs,eps):
 e=copy.deepcopy(base);o=b.observe(e);X=e.old.copy() if a['mode']==0 else b.project(e,b.Decode(e),o,a);b.valid(e,X);rows=[]
 for h in range(3):
  if h:o,a,X,_=b.execute(e,p,b.Decode(e),b.rng_for(seed,1902,gs,h))
  ph=physical(e,o,a,X);ps=int(b.rng_for(seed,900,gs+h).integers(0,2**31-1));r,ts=rec(e,s,p,X,a,ps,gs+h,eps);assert not ts;r.update(ph);rows.append(r)
 return dict(G=sum(.95**h*.1*r['reward'] for h,r in enumerate(rows)),rows=rows)

def long_branch(base,s,p,a,seed,gs,eps):
 # Exact R9-B corrected D80 physical credit. Main/S Replay remains short window.
 e=copy.deepcopy(base);o=b.observe(e)
 X=e.old.copy() if a['mode']==0 else b.project(e,b.Decode(e),o,a)
 b.valid(e,X);rows=[]
 for h in range(3):
  if h:o,a,X,_=b.execute(e,p,b.Decode(e),b.rng_for(seed,1902,gs,h))
  ph=physical(e,o,a,X);ps=int(b.rng_for(seed,900,gs+h).integers(0,2**31-1))
  z=measure(e,s,X,ps,eps)
  r,ts=rec(e,s,p,X,a,ps,gs+h,eps,'A');assert not ts
  r.update(ph,d80_e2e=z['e2e'],d80_failed=z['failed'],d80_n=z['n'],d80_reward=-(e.cost['delay_weight']*z['e2e']+r['holding_cost']+r['switch_cost']))
  rows.append(r)
 return dict(G=sum(.95**h*.1*r['d80_reward'] for h,r in enumerate(rows)),rows=rows)
branch=long_branch

def matched(*branches):
 for i in range(3):
  a=branches[0]['rows'][i]
  assert all((z['rows'][i]['nested_trace_hash'],z['rows'][i]['trace_sha256'],z['rows'][i]['n'])==(a['nested_trace_hash'],a['trace_sha256'],a['n']) for z in branches)

def aux(base,s,p,o,a,seed,gs,eps):
 covered=bool(o[2].any(1).all());ar=b.rng_for(seed,1901,gs);ba=a if a['mode']==1 else b.build(p,o,ar);need=base.w.node_demand()[:,base.w.stage_service].sum(0)*base.w.stages[:,0];cap=base.old@base.w.nodes[:,0];rr=need/np.maximum(cap,1e-8);q=.5*np.ones(b.K)/b.K+.5*rr**2/(rr**2).sum() if covered else np.ones(b.K)/b.K;k=int(ar.choice(b.K,p=q));weight=1/(b.K*q[k]);mo=p(o)[0];pb=mo.softmax(0)[1] if covered else mo.sum()*0+1
 if covered:
  keep,ref,pr=make_pair(p,o,ba,k,ar);z0=branch(base,s,p,keep,seed,gs,eps);z1=branch(base,s,p,ref,seed,gs,eps);zk=branch(base,s,p,{'mode':0,'refresh':[],'counts':[],'order':[],'jitter_seed':None},seed,gs,eps);matched(z0,z1,zk);delta=z1['G']-z0['G'];gb=(1-pr)*z0['G']+pr*z1['G'];loss=-pb*(gb-zk['G'])+pb.detach()*loss_for(p,o,k,ref,delta,weight);branches={'retain':z0,'refresh':z1,'macro_keep':zk};md=gb-zk['G']
 else:
  # All rows forced refresh, no retain/KEEP. Independent row replacement, symmetric score.
  ref=copy.deepcopy(ba);fresh=b.build(p,o,ar)
  for key in ('refresh','counts','order'):ref[key][k]=fresh[key][k]
  z0=branch(base,s,p,ba,seed,gs,eps);z1=branch(base,s,p,ref,seed,gs,eps);matched(z0,z1);delta=z1['G']-z0['G'];loss=-.5*weight*delta*(b.row_lp(p,o,ref,k)-b.row_lp(p,o,ba,k));branches={'refresh0':z0,'refresh1':z1};md=None;pr=1.
 return loss,dict(gs=gs,covered=covered,stage=k,q=q.tolist(),weight=weight,macro_p=float(pb.detach()),refresh_p=pr,delta=delta,macro_delta=md,branches=branches,d_hash=b.digest(p),s_hash=b.digest(s.online),main_mode=a['mode'])

def evaluate(c,x,p,s,seed,arm,gs,out,day=False,fixed=False):
 rows=[]
 for j,ds in enumerate(VALIDATION_SEEDS):
  e=b.environment(c,x);e.reset(ds,0);rng=np.random.default_rng(seed+987101+j*9991)
  n=12 if day else 4
  for t in range(1,n+1):
   if fixed:raise ValueError('Minimal runner only validates learned policies')
   o,a,X,_=b.execute(e,p,b.Decode(e),rng)
   ph=physical(e,o,a,X);ps=int(b.rng_for(ds,808,t).integers(0,2**31-1))
   r,_=rec(e,s,p,X,a,ps,t,0)
   r.update(ph,checkpoint=gs,arm=arm,eval_step=t,validation_seed=ds,
            validation_run=j,day=day,fixed=fixed)
   b.write(out,r);rows.append(r)
 return float(np.mean([z['reward'] for z in rows]))

def maybe_best(out,score,gs,seed,arm):
 # Two frozen 4GS validation tracks ONLY: never select from independent demands or D120.
 meta_file=out/'best_metadata.json'
 prev=json.loads(meta_file.read_text()) if meta_file.exists() else None
 if prev is not None and float(score)<=float(prev['best_validation_reward']):return False
 ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
 best={k:ck[k] for k in ('d','s','gs','seed','arm','d_hash','s_hash')}
 best['algorithm_version']=ck.get('algorithm_version','S-only')
 tmp=out/'best.tmp';torch.save(best,tmp);tmp.replace(out/'best.pt')
 meta=dict(seed=seed,arm=arm,best_gs=gs,best_validation_reward=float(score),
   selection='strictly higher mean actual Reward on two frozen Markov 4GS tracks',
   validation_seeds=list(VALIDATION_SEEDS),validation_gs=8,algorithm_version=ALGORITHM_VERSION)
 meta_file.write_text(json.dumps(meta,indent=2,ensure_ascii=False))
 return True

def save(out,p,s,opt,e,replay,rng,gs,seed,arm,anchors):
 # One atomic latest.pt contains compressed Replay, optimizer, env, RNG.
 # best.pt is weights-only and is selected by fixed validation Reward.
 replay_blob=gzip.compress(pickle.dumps(replay,protocol=5),compresslevel=1)
 ck=dict(d=p.state_dict(),s=s.state_dict(),opt=opt.state_dict(),env=e,rng=rng.bit_generator.state,
         gs=gs,seed=seed,arm=arm,d_hash=b.digest(p),s_hash=b.digest(s.online),replay_gzip=replay_blob,algorithm_version=ALGORITHM_VERSION,trust_anchors=anchors)
 tmp=out/'latest.tmp';torch.save(ck,tmp);tmp.replace(out/'latest.pt')

def repair_logs(out,start):
 # If a previous run was interrupted at an odd GS, preserve but exclude
 # incomplete trailing records before replaying from the last atomic GS.
 for name in ('train.jsonl','aux.jsonl','wall.jsonl','updates.jsonl','adam_audit.jsonl','eval.jsonl'):
  f=out/name
  if not f.exists():continue
  keep=[];extra=[]
  for line in f.read_text().splitlines(keepends=True):
   r=json.loads(line)
   g=int(r.get('checkpoint',0) if name=='eval.jsonl' else r.get('gs',0))
   (keep if g<=start else extra).append(line)
  if extra:
   with (out/(f'aborted_{name}')).open('a',encoding='utf8') as dest:dest.writelines(extra)
   tmp=out/(name+'.tmp');tmp.write_text(''.join(keep));tmp.replace(f)

def run(seed,arm,steps,resume,coverage=False):
 assert arm == "A"  # minimal R12-A direct joint learner
 # Identical randomly initialized D/S, same natural main Markov process and S DDQN.
 # Arm B only: no D auxiliary gradients/optimizer steps during first eight GS.
 c,x,e,s,p=setup(seed,"D");out=ROOT/"experiments"/arm/"r51/results"/f"{seed}_D";out.mkdir(parents=True,exist_ok=resume);rng=np.random.default_rng(seed+771);replay=b.SReplay(c.s_replay_transitions,c.replay_max_age);opt=torch.optim.Adam([v for v in p.parameters() if v.requires_grad],lr=float(x["r0_d_lr"]));anchors=[];start=0;initialD=b.digest(p);initialS=b.digest(s.online)
 if resume:
  ck=torch.load(out/'latest.pt',weights_only=False,map_location='cpu');assert ck.get('algorithm_version')==ALGORITHM_VERSION, 'Old optimizer/checkpoint requires a fresh run or an explicitly audited migration';p.load_state_dict(ck['d']);s.load_state_dict(ck['s']);opt.load_state_dict(ck['opt']);e=ck['env'];rng.bit_generator.state=ck['rng'];start=ck['gs'];assert start%2==0 and b.digest(p)==ck['d_hash'] and b.digest(s.online)==ck['s_hash'];replay=pickle.loads(gzip.decompress(ck['replay_gzip']));anchors=ck['trust_anchors'];repair_logs(out,start)
  evfile=out/'eval.jsonl'
  scored={json.loads(line)['checkpoint'] for line in evfile.read_text().splitlines()} if evfile.exists() else set()
  if start not in scored and (start==0 or start%8==0):
   score=evaluate(c,x,p,s,seed,arm,start,evfile);maybe_best(out,score,start,seed,arm)
 else:
  assert gradient_contract()<1e-12 and len(replay)==0 and s.updates==0 and not opt.state;save(out,p,s,opt,e,replay,rng,0,seed,arm,anchors);score=evaluate(c,x,p,s,seed,arm,0,out/'eval.jsonl');maybe_best(out,score,0,seed,arm)
 losses=[];versions=[];obsbuf=[]
 for gs in range(start+1,steps+1):
  tick=time.monotonic();o,a,X,di=b.execute(e,p,b.Decode(e),rng);
  if not anchors: anchors.append(o)  # preserve the first natural observed deployment state
  elif gs%2==1:
   anchors.append(o)
   if len(anchors)>9:anchors.pop(1)
  base=copy.deepcopy(e);ph=physical(e,o,a,X);eps=c.epsilon_end+(c.epsilon_start-c.epsilon_end)*max(0,1-((gs-1)//2*2)/c.epsilon_decay_steps);ps=int(b.rng_for(seed,900,gs).integers(0,2**31-1));r,ts=rec(e,s,p,X,a,ps,gs,eps,collect=True);assert e.period==base.period+1
  if True:assert ts and all(t.mask[t.action] and t.collection_step==gs for t in ts);replay.add(ts);replay.prune(gs)
  if arm=="A" or gs>8:
   before=b.statehash(e);size=len(replay);loss,ar=aux(base,s,p,o,a,seed,gs,eps);assert b.statehash(e)==before and len(replay)==size;assert ar.get('state_source','natural')!='natural' or ar['branches'][next(iter(ar['branches']))]['rows'][0]['nested_trace_hash']==r['nested_trace_hash'];losses.append(loss);obsbuf.append(ar.pop("_obs",o));versions.append((ar['d_hash'],ar['s_hash']));b.write(out/'aux.jsonl',ar)
  r.update(ph,**di,proposal_density=float(np.mean(np.array(ph['proposal_counts'])/b.N)),projected_fraction=ph['projection_hamming']/(b.K*b.N),arm=arm,collected=len(ts),replay_size=len(replay),total_wall=time.monotonic()-tick,d_update_eligible=(arm=='A' or gs>8),d_warmup_frozen=(arm=='B' and gs<=8));b.write(out/'train.jsonl',r)
  if gs%2==0:
   du=None;gn=None;sh=b.digest(s.online);dh=b.digest(p)
   if arm=="A" or gs>8:
    assert versions==[(dh,sh)]*2;loss=torch.stack(losses).mean()-entropy_bonus(p,obsbuf);opt.zero_grad();loss.backward();gn=float(nn.utils.clip_grad_norm_([v for v in p.parameters() if v.requires_grad],5,error_if_nonfinite=True));before_params={n:v.detach().clone() for n,v in p.named_parameters() if v.requires_grad};before_out=[tuple(t.detach().clone() for t in p(oo)[:4]) for oo in obsbuf];step_audit=bounded_adam_step(p,opt,obsbuf+anchors);du=float(loss.detach());after_out=[tuple(t.detach() for t in p(oo)[:4]) for oo in obsbuf];kl=[]
    for prev,after in zip(before_out,after_out):
     def bern(a,b):return (a.sigmoid()*(F.logsigmoid(a)-F.logsigmoid(b))+(1-a.sigmoid())*(F.logsigmoid(-a)-F.logsigmoid(-b))).mean().item()
     kl.append(dict(macro=float((prev[0].softmax(0)*(prev[0].log_softmax(0)-after[0].log_softmax(0))).sum()),refresh=bern(prev[1],after[1]),count=31*bern(prev[2],after[2]),node_first=float((prev[3].softmax(-1)*(prev[3].log_softmax(-1)-after[3].log_softmax(-1))).sum(-1).mean())))
    b.write(out/'adam_audit.jsonl',dict(gs=gs,delta={n:float((v.detach()-before_params[n]).norm()) for n,v in p.named_parameters() if n in before_params},kl=kl,trust_step=step_audit));losses=[];versions=[];obsbuf=[]
   if True:
    sl=[s.update(replay.sample(x['s_batch'],gs,b.rng_for(seed,902,gs,j))) for j in range(2*x['s_updates_per_period'])];s.version+=1
   else:sl=[]
   b.write(out/'updates.jsonl',dict(gs=gs,d_before=dh,d_after=b.digest(p),s_before=sh,s_after=b.digest(s.online),d_loss=du,d_grad=gn,d_trust=step_audit if du is not None else None,s_updates=s.updates,s_loss=float(np.mean([v['s_loss'] for v in sl])) if sl else None));save(out,p,s,opt,e,replay,rng,gs,seed,arm,anchors)
  b.write(out/"wall.jsonl",dict(gs=gs,seed=seed,arm=arm,wall_complete_s=time.monotonic()-tick,physical_commit=1,aux_d80_calls=(0 if arm=="B" and gs<=8 else len(ar["branches"])*3),D_updates=int(gs//2 if arm=="A" else max(gs-8,0)//2),S_updates=s.updates,latest_d_hash=b.digest(p),latest_s_hash=b.digest(s.online)))
  print(seed,arm,gs,r['reward'],r['penalized_delay'],r['failed'],flush=True)
  if gs % 8 == 0 or gs==steps:
   score=evaluate(c,x,p,s,seed,arm,gs,out/'eval.jsonl');maybe_best(out,score,gs,seed,arm)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--arm',choices=['A'],default='A');ap.add_argument('--steps',type=int,default=16);ap.add_argument('--resume',action='store_true');a=ap.parse_args();assert a.steps%2==0;run(a.seed,a.arm,a.steps,a.resume)
