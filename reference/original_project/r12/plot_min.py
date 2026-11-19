"""Draw raw online and held-out validation Reward/E2E; no smoothing."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent

def run(seed,arm):
 d=R/'experiments'/arm/'r51/results'/f'{seed}_D'
 if not (d/'train.jsonl').exists():raise FileNotFoundError(d/'train.jsonl')
 rows=[json.loads(x) for x in (d/'train.jsonl').read_text().splitlines() if x.strip()]
 ev=[json.loads(x) for x in (d/'eval.jsonl').read_text().splitlines() if x.strip()]
 grouped={}
 for r in ev:grouped.setdefault(r['checkpoint'],[]).append(r)
 X=[x['gs'] for x in rows]
 d.mkdir(parents=True,exist_ok=True)
 for k,key,label in [('reward','reward','Reward (higher better)'),('e2e','penalized_delay','Request-weighted E2E (s)')]:
  fig,ax=plt.subplots(figsize=(9,4.6))
  # Each main GS is one actual measurement. No drops, filtering or smoothing.
  ax.plot(X,[float(t['reward'] if k=='reward' else t['penalized_delay']) for t in rows],'-o',ms=2.6,label='online raw')
  kk=sorted(grouped)
  if k=='reward':v=[sum(z['reward'] for z in grouped[i])/len(grouped[i]) for i in kk]
  else:v=[sum(z['penalized_delay']*z['n'] for z in grouped[i])/sum(z['n'] for z in grouped[i]) for i in kk]
  ax.plot(kk,v,'--s',ms=4,label='frozen validation')
  ax.set(xlabel='GS (2 logical hours / GS)',ylabel=label,title=f'R12-A {arm}, seed {seed}: {k} raw')
  ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(d/f'{k}_raw.png',dpi=160);plt.close(fig)
  print(d/f'{k}_raw.png')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--arm',choices=['A','S'],default='A');v=a.parse_args();run(v.seed,v.arm)
