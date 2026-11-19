"""Fixed-trace D/S checkpoint swaps. Read-only evidence on user's two checkpoints."""
import json,sys,time
from pathlib import Path
import torch,numpy as np
from r12 import joint as q
out=Path(sys.argv[3] if len(sys.argv)>3 else '/mnt/data/R12A_factorial_audit')
out.mkdir(parents=True,exist_ok=True)
seed=20261114
c16=torch.load(sys.argv[1],weights_only=True,map_location='cpu')
c100=torch.load(sys.argv[2],weights_only=False,map_location='cpu')
for d,s in [('16','16'),('16','100'),('100','16'),('100','100')]:
  c,x,e,agent,policy=q.setup(seed,'D')
  ck_d=c16 if d=='16' else c100
  ck_s=c16 if s=='16' else c100
  policy.load_state_dict(ck_d['d']);agent.load_state_dict(ck_s['s'])
  tic=time.monotonic();f=out/f'D{d}_S{s}.jsonl'
  reward=q.evaluate(c,x,policy,agent,seed,'A',int(d)*1000+int(s),f)
  rows=[json.loads(l) for l in f.open()]
  builds=[r for r in rows if r['action']['mode']==1]
  summary=dict(d=d,s=s,reward=float(reward),e2e=float(sum(r['penalized_delay']*r['n'] for r in rows)/sum(r['n'] for r in rows)),failed=int(sum(r['failed'] for r in rows)),mean_projected_fraction=float(np.mean([r['projection_hamming']/(q.b.K*q.b.N) for r in rows])),mean_proposed_count=float(np.mean([sum(r['proposal_counts']) for r in builds])) if builds else 0,wall_s=time.monotonic()-tic,valid_seed=[int(r['validation_seed']) for r in rows])
  print('FACTORIAL',json.dumps(summary),flush=True)
  (out/f'D{d}_S{s}.summary.json').write_text(json.dumps(summary,indent=2))
