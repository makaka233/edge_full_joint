"""R1: separate normalized critic, matched macro/row credit. No auxiliary selection."""
from r0_joint import *
import r0_joint as r0
from r2a_sampling import commit_nested
OUT=ROOT/'r2b_results'; SCALE=.05; GAMMA=.95; BLOCK=4; CRITIC_STEPS=32
class Critic(nn.Module):
 def __init__(self,d):
  super().__init__();self.enc=nn.Sequential(nn.Linear(20,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU());self.net=nn.Sequential(nn.Linear(96+d,64),nn.SiLU(),nn.Linear(64,1))
 def forward(self,o):
  h=self.enc(o[0]);return self.net(torch.cat((h.mean((0,1)),h.mean(1).amax(0),h.mean(0).amax(0),o[1]))).squeeze()
def build(p,o,rng):
 with torch.no_grad():m,rf,c,loc,_=p(o)
 ref=rng.random(K)<torch.sigmoid(rf).numpy() if bool(o[2].any(1).all()) else np.ones(K,bool);a={'mode':1,'refresh':ref.tolist(),'counts':[0]*K,'order':[[] for _ in range(K)],'jitter_seed':int(rng.integers(0,2**31-1))}
 for k in np.flatnonzero(ref):
  n=1+int(rng.binomial(N-1,float(torch.sigmoid(c[k]))));a['counts'][k]=n;a['order'][k]=np.argsort(-(loc[k].numpy()+rng.gumbel(size=N)))[:n].tolist()
 return a
def project(e,d,o,a):
 y=o[2].numpy().copy()
 for k in np.flatnonzero(a['refresh']):y[k]=False;y[k,a['order'][k]]=True
 return d(e,np.where(y,1.,-1.)+np.random.default_rng(a['jitter_seed']).uniform(-1,1,(K,N))*.0001)[0]
def row_lp(p,o,a,k):
 m,ref,c,loc,v=p(o);rf=a['refresh'][k];lp=-F.binary_cross_entropy_with_logits(ref[k],torch.tensor(float(rf))) if bool(o[2].any(1).all()) else ref[k]*0
 if rf:
  lp=lp+torch.distributions.Binomial(total_count=N-1,logits=c[k]).log_prob(torch.tensor(float(a['counts'][k]-1)));z=loc[k];mask=torch.zeros(N,dtype=torch.bool)
  for j in a['order'][k]:lp=lp+F.log_softmax(z.masked_fill(mask,-1e9),0)[j];mask=mask.clone();mask[j]=True
 return lp
def branch(base,s,X,ps,seed,gs,eps):
 e=copy.deepcopy(base);_,r,_,_,_=e.commit(X,s.scheduler(rng_for(seed,901,gs),eps),ps,gs,False);return r,observe(e),statehash(e)
def save1(out,name,p,v,s,ao,vo,e,replay,rng,gs,seed,arm,best):
 torch.save(dict(d=p.state_dict(),critic=v.state_dict(),s=s.state_dict(),actor_opt=ao.state_dict(),critic_opt=vo.state_dict(),env=e,rng=rng.bit_generator.state,gs=gs,seed=seed,arm=arm,best=best,d_hash=digest(p),v_hash=digest(v),s_hash=digest(s.online)),out/name)
 if name=='latest.pt':
  with gzip.open(out/'latest_replay.pkl.gz','wb',compresslevel=1) as f:pickle.dump(replay,f,protocol=5)
def run(seed,arm,steps,resume=False):
 c,x,e,s,p=setup(seed);learnD=arm!='s_only';protocol='D120' if arm=='D120' else 'A';torch.manual_seed(seed+32);v=Critic(len(e.w.context()));ao=torch.optim.Adam(p.parameters(),lr=x['r0_d_lr']);vo=torch.optim.Adam(v.parameters(),lr=.003);dec=Decode(e);rng=np.random.default_rng(seed+771);replay=SReplay(c.s_replay_transitions,c.replay_max_age);out=OUT/f'{arm}_{seed}';out.mkdir(parents=True,exist_ok=resume);roll=[];start=0
 if resume:
  ck=torch.load(out/'latest.pt',weights_only=False,map_location='cpu');p.load_state_dict(ck['d']);v.load_state_dict(ck['critic']);s.load_state_dict(ck['s']);ao.load_state_dict(ck['actor_opt']);vo.load_state_dict(ck['critic_opt']);e=ck['env'];dec=Decode(e);rng.bit_generator.state=ck['rng'];start=ck['gs'];best=ck['best'];assert digest(p)==ck['d_hash'] and digest(v)==ck['v_hash'] and digest(s.online)==ck['s_hash']
  with gzip.open(out/'latest_replay.pkl.gz','rb') as f:replay=pickle.load(f)
 else:
  best=r0.evaluate(p,s,c,x,seed,0,out/'eval.jsonl',arm);save1(out,'best.pt',p,v,s,ao,vo,e,replay,rng,0,seed,arm,best)
 for gs in range(start+1,steps+1):
  tick=time.monotonic();o,a,X,di=execute(e,p,dec,rng);base=copy.deepcopy(e);before=statehash(e);old=hexmat(e.old);covered=bool(e.old.any(1).all());eps=c.epsilon_end+(c.epsilon_start-c.epsilon_end)*max(0,1-((gs-1)//BLOCK*BLOCK)/c.epsilon_decay_steps);ps=int(rng_for(seed,900,gs).integers(0,2**31-1));e.current_s_version=s.version
  _,r,ts,terminated,truncated=commit_nested(e,X,s,ps,gs,eps,protocol,True);assert not terminated and not truncated and ts and all(t.mask[t.action] for t in ts);replay.add(ts);replay.prune(gs);nxt=observe(e)
  with torch.no_grad():vv=float(v(o));lp=float(logprob(p,o,a)[0])
  b=dict(o=o,a=a,r=.1*r['d_reward'],nxt=nxt,value=vv,gs=gs,covered=covered)
  if arm=='paired':
   arng=np.random.default_rng(seed+901000+gs);ba=a if a['mode']==1 else build(p,o,arng);bx=X if a['mode']==1 else project(base,dec,o,ba);kr,kn,kh=(r,nxt,statehash(e)) if a['mode']==0 else branch(base,s,base.old.copy(),ps,seed,gs,eps) if covered else (None,None,None);br,bn,bh=(r,nxt,statehash(e)) if a['mode']==1 else branch(base,s,bx,ps,seed,gs,eps)
   k=int(arng.integers(K));fresh=build(p,o,arng);alt=copy.deepcopy(ba)
   for key in ('refresh','counts','order'):alt[key][k]=fresh[key][k]
   ax=project(base,dec,o,alt);ar,an,ah=branch(base,s,ax,ps,seed,gs,eps)
   assert br['trace_sha256']==ar['trace_sha256']==r['trace_sha256'] and (kr is None or kr['trace_sha256']==r['trace_sha256']);assert br['n']==ar['n']==r['n'] and (kr is None or kr['n']==r['n']);b.update(ba=ba,k=k,br=.1*br['reward'],bn=bn,kr=None if kr is None else .1*kr['reward'],kn=kn,ar=.1*ar['reward'],an=an)
   for tag,rr,xx,hh in [('build',br,bx,bh),('row',ar,ax,ah),('keep',kr,base.old,kh)]:
    if rr is not None:write(out/'pairs.jsonl',dict(gs=gs,tag=tag,row=k,main_mode=a['mode'],state_hash=before,next_hash=hh,matrix_hex=hexmat(xx),old_hex=old,proposal_action=ba if tag=='build' else alt if tag=='row' else {'mode':0},**rr))
  roll.append(b);write(out/'train.jsonl',dict(gs=gs,seed=seed,arm=arm,action=a,mode=a['mode'],old_hex=old,matrix_hex=hexmat(X),state_hash=before,next_state_hash=statehash(e),d_hash=digest(p),s_hash=digest(s.online),s_version=s.version,epsilon=eps,collected=len(ts),d_logp=lp,value_norm=vv,wall_seconds=time.monotonic()-tick,**di,**r))
  if gs%BLOCK==0:
   if gs==BLOCK:
    with torch.no_grad():v.net[-1].bias.add_(float(np.mean([b['r'] for b in roll]))-float(torch.stack([v(b['o']) for b in roll]).mean()))
   with torch.no_grad():
    vals=[float(v(b['o'])) for b in roll];boot=float(v(roll[-1]['nxt']));adv,tgt=value_targets([SCALE*b['r'] for b in roll],vals,boot,GAMMA,x['r0_lambda']);targets=[(b['o'],float(t)) for b,t in zip(roll,tgt)];losses=[];qdiag=[]
    for b in roll:
     if arm=='paired':
      qb=b['br']+GAMMA*float(v(b['bn']))/SCALE;qa=b['ar']+GAMMA*float(v(b['an']))/SCALE;qk=b['kr']+GAMMA*float(v(b['kn']))/SCALE if b['covered'] else float(v(b['o']))/SCALE;b.update(qb=qb,qa=qa,qk=qk);qdiag.append([qb,qk,qa])
   for b,ad in zip(roll,adv):
    o=b['o'];m=p(o)[0];l,_=logprob(p,o,b['a'])
    if arm!='paired':loss=-l*float(ad)/SCALE
    else:
     ba=b['ba'];bl,_=logprob(p,o,ba);pb=torch.softmax(m,0)[1] if b['covered'] else m.sum()*0+1;cond=bl-F.log_softmax(m,0)[1] if b['covered'] else bl
     loss=-pb*(b['qb']-b['qk'])-pb.detach()*(cond*(b['qb']-b['qk'])+row_lp(p,o,ba,b['k'])*(b['qk']-b['qa']))
    losses.append(loss)
   al=torch.stack(losses).mean();ao.zero_grad();al.backward();gn=float(nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True));ao.step() if learnD else None;vt=torch.tensor([t for _,t in targets]);vl=[]
   for _ in range(CRITIC_STEPS if learnD else 0):
    pred=torch.stack([v(o) for o,_ in targets]);loss=F.smooth_l1_loss(pred,vt);vo.zero_grad();loss.backward();nn.utils.clip_grad_norm_(v.parameters(),5,error_if_nonfinite=True);vo.step();vl.append(float(loss.detach()))
   sl=[]
   for ui in range(x['s_updates_per_period']*BLOCK):sl.append(s.update(replay.sample(x['s_batch'],gs,rng_for(seed,902,gs,ui))))
   s.version+=1;write(out/'updates.jsonl',dict(gs=gs,actor_loss=float(al.detach()),actor_grad=gn,v_loss_first=vl[0] if vl else None,v_loss_last=vl[-1] if vl else None,values=vals,targets=vt.tolist(),adv=adv.tolist(),bootstrap=boot,qdiag=qdiag,d_hash=digest(p),v_hash=digest(v),s_hash=digest(s.online),s_updates=s.updates,s_loss=float(np.mean([a['s_loss'] for a in sl]))));roll=[]
  print(seed,arm,gs,round(r['reward'],3),round(r['d_reward'],3),r['failed'],flush=True)
  if gs in (8,16,32,64):
   score=r0.evaluate(p,s,c,x,seed,gs,out/'eval.jsonl',arm)
   if score>best:best=score;save1(out,'best.pt',p,v,s,ao,vo,e,replay,rng,gs,seed,arm,best)
  if gs%BLOCK==0:save1(out,'latest.pt',p,v,s,ao,vo,e,replay,rng,gs,seed,arm,best)
  if gs==32:save1(out,'gs32.pt',p,v,s,ao,vo,e,replay,rng,gs,seed,arm,best)
 (out/'completed.json').write_text(json.dumps(dict(gs=steps,actor_updates=steps//BLOCK if learnD else 0,critic_updates=steps//BLOCK*CRITIC_STEPS if learnD else 0,s_updates=s.updates),indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--arm',choices=['A','D120','s_only'],required=True);ap.add_argument('--steps',type=int,default=32);ap.add_argument('--resume',action='store_true');a=ap.parse_args();assert a.steps%BLOCK==0
 if a.arm=='D120' and a.resume and os.environ.get('R2B_SERIAL','0')!='1':
  from r2b_parallel import run_parallel
  run_parallel(a.seed,a.steps,int(os.environ.get('R2B_WORKERS','2')))
 else:run(a.seed,a.arm,a.steps,a.resume)
