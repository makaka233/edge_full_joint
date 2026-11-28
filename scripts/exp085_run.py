"""EXP085 bounded learnability experiment, preceding conditional physical iteration."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts import exp084_run as parent
from scripts.exp082_run import table

ROOT=Path('runs/results/EXP085')
REVIEW=Path('research/EXP085/AI_RESEARCH_REVIEW.md')

def compact(x):
    x=np.asarray(x);raw=np.sign(x)*np.expm1(abs(x))
    stages=raw[92:372].reshape(28,10);nodes=raw[372:660].reshape(32,9);links=raw[660:]
    q4=lambda a:np.quantile(a,[.5,.9,.99,1])
    q3=lambda a:np.array([np.mean(a),np.quantile(a,.9),np.max(a)])
    z=np.r_[raw[:12],q4(stages[:,4]),q4(stages[:,5]),q4(stages[:,5]-stages[:,4]),
            q3(nodes[:,1]),q3(nodes[:,2]),q3(nodes[:,2]-nodes[:,1]),q3(nodes[:,3]),q3(links),
            q3(stages[:,7]-stages[:,6]),q3(stages[:,8]),stages[:,:2].sum(0),stages[:,9].max()]
    assert len(z)==48 and np.isfinite(z).all()
    return parent.slog(z).astype(np.float32)

def data(part):
    samples=[];manifest={}
    for path in sorted((parent.ROOT/'data').glob(f'{part}.*.json')):
        d=parent.old.read(path);manifest[path.as_posix()]=parent.old.digest(path)
        for p in d['points']:
            risk=[int(any(p['costs'][a]['8']['metrics'][k]>0 for k in ('severe','censored')) or p['costs'][a]['8']['admitted_unfinished_at_deadline']>0) for a in ('0','1')]
            samples.append(dict(id=f"{d['case']['id']}.{p['period']}",seed=d['case']['seed'],x=p['x'],compact=compact(p['x']).tolist(),y=p['labels'],risk=risk,
                backlog=[np.log1p(p['costs'][a]['8']['metrics']['cpu']) for a in ('0','1')]))
    return samples,manifest

class Net(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.Sequential(nn.Linear(48,32),nn.Tanh(),nn.Linear(32,16),nn.Tanh(),nn.Linear(16,6))
    def forward(self,x):return self.layers(x)

def assess(pred,samples):
    y=np.array([p['y'] for p in samples]);risk=np.array([p['risk'] for p in samples]);act=np.array(pred)[:,1]>0
    dangerous=risk[np.arange(len(risk)),act.astype(int)]>risk[np.arange(len(risk)),1-act.astype(int)]
    return dict(value=parent.score(pred,y),dangerous_choices=int(dangerous.sum()),
                severe_states=int(risk.any(1).sum()),
                severe_value=parent.score(np.array(pred)[risk.any(1)],y[risk.any(1)]) if risk.any(1).any() else None,
                horizon_reversals=int((y[:,0]*y[:,1]<0).sum()),
                by_seed={str(s):parent.score(np.array(pred)[[p['seed']==s for p in samples]],y[[p['seed']==s for p in samples]]) for s in sorted({p['seed'] for p in samples})})

def fit(kind,seed,tr,va):
    torch.manual_seed(seed);net=Net();initial=parent.old.weights_digest(net.state_dict())
    x=torch.tensor([p['compact'] for p in tr]);y=torch.tensor([p['y'] for p in tr]);vx=torch.tensor([p['compact'] for p in va]);vy=torch.tensor([p['y'] for p in va])
    mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(.05);x=(x-mean)/scale;vx=(vx-mean)/scale
    ys=(y.abs().median(0).values if kind.startswith('robust') else y.square().mean(0).sqrt()).clamp_min(.1)
    risks=torch.tensor([p['risk'] for p in tr],dtype=torch.float32);backlog=torch.tensor([p['backlog'] for p in tr],dtype=torch.float32)
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed)
    best=float('inf');curves=[];folder=ROOT/'models'/f'{kind}.{seed}';folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    for step in range(1,2001):
        ix=rng.integers(len(x),size=32);out=net(x[ix]);target=y[ix]/ys
        if kind=='mse':loss=(out[:,:2]-target).square().mean()
        else:
            value=F.smooth_l1_loss(out[:,:2],target)
            valid=y[ix].abs()>.03
            rank=F.binary_cross_entropy_with_logits(out[:,:2][valid],(y[ix][valid]>0).float()) if valid.any() else value*0
            risk=F.binary_cross_entropy_with_logits(out[:,2:4],risks[ix])
            load=F.smooth_l1_loss(out[:,4:6],backlog[ix])
            loss=value+.1*(rank+risk+load)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        optimizer.zero_grad();loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True));optimizer.step()
        if step%50==0:
            with torch.no_grad():v=net(vx)[:,:2]*ys;mae=float((v-vy).abs().mean())
            curves.append(dict(step=step,loss=float(loss.detach()),gradient=gradient,validation_mae=mae))
            if mae<best:best=mae;best_state=deepcopy(net.state_dict());best_step=step
    common=dict(mean=mean,scale=scale,y_scale=ys,seed=seed,kind=kind,initial=initial)
    for name,st in (('best',best_state),('last',net.state_dict())):
        parent.old.atomic(folder/f'{name}.pt',lambda f,st=st:torch.save(dict(**common,network=st),f))
    parent.old.atomic(folder/'resume_latest.pt',lambda f:torch.save(dict(**common,network=net.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),step=2000,best_network=best_state,best_step=best_step),f))
    net.load_state_dict(best_state)
    with torch.no_grad():prediction=(net(vx)[:,:2]*ys).numpy();tp=(net(x)[:,:2]*ys).numpy();risk_prob=net(vx)[:,2:4].sigmoid().numpy()
    result=dict(kind=kind,seed=seed,best_step=best_step,updates=2000,parameter_count=sum(p.numel() for p in net.parameters()),
        initial=initial,final=parent.old.weights_digest(best_state),validation=assess(prediction,va),training=assess(tp,tr),
        prediction=prediction.tolist(),risk_probability=risk_prob.tolist(),curves=curves,seconds=time.perf_counter()-started)
    parent.save(folder/'result.json',result);print(kind,seed,'MAE8',result['validation']['value']['8']['mae'],'regret',result['validation']['value']['8']['physical_regret'],flush=True)
    return result

def run():
    torch.set_num_threads(1);started=time.perf_counter();ROOT.mkdir(parents=True,exist_ok=True)
    planpath=ROOT/'protocol.json'
    if not planpath.exists():
        parent.save(planpath,dict(protocol=parent.old.digest(REVIEW.with_name('PROTOCOL.md')),source=parent.old.source(),
            final_keys=[dict(seed=s,family=f,population=parent.old.stable_seed('EXP085-final-v1',s,f,'population'),scene=parent.old.stable_seed('EXP085-final-v1',s,f,'scene'),field=parent.old.stable_seed('EXP085-final-v1',s,f,'field')) for s in parent.SEEDS for f in parent.FAMILIES]))
    elif parent.old.read(planpath)['protocol']!=parent.old.digest(REVIEW.with_name('PROTOCOL.md')):raise ValueError('Protocol changed')
    tr,tm=data('train');va,vm=data('validation');assert len(tr)==216 and len(va)==72
    parent.save(ROOT/'dataset.json',dict(train=tr,validation=va,source_sha256={**tm,**vm}))
    baseline=[]
    for seed in parent.NETS:
        p=parent.predictor(parent.ROOT,seed)
        pred=[p(s['x'],0)[1] for s in va]
        baseline.append(dict(seed=seed,validation=assess(pred,va)))
    constant=assess(np.tile(np.mean([p['y'] for p in tr],axis=0),(len(va),1)),va)
    results=[fit(kind,seed,tr,va) for kind in ('mse','robust') for seed in (8501,8502,8503)]
    passed=[]
    for r,old in zip([r for r in results if r['kind']=='robust'],baseline):
        v=r['validation'];a=v['value']['8'];b=old['validation']['value']['8'];c=constant['value']['8']
        passed.append(a['mae']<min(b['mae'],c['mae']) and a['physical_regret']<min(b['physical_regret'],c['physical_regret'])
            and min(a['keep_accuracy'],a['redeploy_accuracy'])>=.55 and v['dangerous_choices']<=old['validation']['dangerous_choices'])
    # Compare opposite-sign near neighbours without leaking validation labels into fitting.
    tx=np.array([p['compact'] for p in tr]);vx=np.array([p['compact'] for p in va]);scale=np.maximum(tx.std(0),.05)
    distance=np.sqrt(np.mean(((vx[:,None,:]-tx[None,:,:])/scale)**2,axis=2));near=distance.argmin(1)
    collisions=[dict(validation=va[i]['id'],train=tr[j]['id'],distance=float(distance[i,j]),train_delta=tr[j]['y'][1],validation_delta=va[i]['y'][1]) for i,j in enumerate(near) if tr[j]['y'][1]*va[i]['y'][1]<0 and min(abs(tr[j]['y'][1]),abs(va[i]['y'][1]))>.03]
    result=dict(baseline=baseline,constant=constant,results=results,passed=passed,proceed=sum(passed)>=2,nearest_opposite=collisions,seconds=time.perf_counter()-started,source=parent.old.source())
    parent.save(ROOT/'offline.json',result)
    report(result)
    print('P1 complete; proceed to physical iteration:',result['proceed'],flush=True)

def report(r):
    lines=['# EXP085 稳健相对收益学习', '',f'代码 {r["source"]["commit"]}；EXP084 complete归档34114c8。',
        '本轮已实际完成原216训练/72验证样本、三初始化、紧凑MSE及唯一稳健候选的2000更新对照。没有改D/S、Reward或物理环境。',
        'EXP084 764维/51106参数、验证MAE不及常数、独立方向近随机、高风险REDEPLOY漏判、固定KEEP续策不一致均保留为失败证据。',
        '48维紧凑观测；稳健幅度回归+0.1方向辅助+真实风险/积压辅助。最终动作按原单位预测Δ8>0，不按分类概率/风险阈值。风险头不加人工Reward惩罚。',
        'Huber回归不保证估计条件均值，仍须真实物理验证，不能仅凭方向命中宣布期望Reward最优。原4/8GS标签仍为固定KEEP续策，尚不代表自适应策略价值。']
    rows=[]
    for name,v in [('constant',r['constant'])]+[(f'EXP084-{b["seed"]}',b['validation']) for b in r['baseline']]+[(f'{a["kind"]}-{a["seed"]}',a['validation']) for a in r['results']]:
        m=v['value']['8'];rows.append([name,f"{m['mae']:.6f}",m['keep_accuracy'],m['redeploy_accuracy'],f"{m['physical_regret']:.6f}",v['dangerous_choices'],v['severe_states'],v['horizon_reversals']])
    table(lines,['模型','8GS MAE','KEEP识别','REDEPLOY识别','实际收益regret','危险动作漏判','严重起点','4/8反转'],rows)
    for a in r['results']:
        lines += ['',f"{a['kind']}/{a['seed']}：参数{a['parameter_count']}，best更新{a['best_step']}；训练MAE8={a['training']['value']['8']['mae']:.5f}；逐模型验证={a['validation']['by_seed']}；严重状态={a['validation']['severe_value']}。"]
    lines += ['',f'三初始化预登记通过={r["passed"]}。',
        'P1达到物理策略迭代启动条件，尚未进行P2/P4；不能宣布最终通过。' if r['proceed'] else 'P1未达到物理策略迭代启动条件。按预登记停止昂贵P2/P4，不扩大物理测试、不修改阈值或追加候选。未执行的在线实验不是已验证失败。',
        f'紧凑标准化距离最近的异号训练/验证邻居：{len(r["nearest_opposite"])}个。仅说明泛化困难线索，不证明原物理状态不可辨识；详见offline.json。',
        '结论：目前不能将机制加入论文主方案；是否解决高风险漏判须以表中危险选择计数及后续独立物理收益判定，不以loss下降代替。',
        f'本轮耗时{r["seconds"]:.2f}s。原EXP084的408开口保持只读；新数据/模型/逐更新曲线位于runs/results/EXP085。新最终随机流已预登记但未执行。',
        '复现：`python -m scripts.exp085_run offline`。没有启动后台长训或恢复旧SAC。']
    fig,axes=parent.old.plt.subplots(2,1,figsize=(11,8),sharex=True)
    for a in r['results']:
        for ax,k in zip(axes,('loss','validation_mae')):
            ax.plot([v['step'] for v in a['curves']],[v[k] for v in a['curves']],label=f"{a['kind']}-{a['seed']}");ax.set_ylabel(k)
    axes[0].legend(ncol=3);axes[-1].set_xlabel('Optimizer updates; offline physical value regression')
    fig.savefig(ROOT/'fig.learning.png',dpi=130,bbox_inches='tight');parent.old.plt.close(fig)
    lines+=['','![原始学习记录](../../../runs/results/EXP085/fig.learning.png)','本轮没有新在线Reward/E2E/P99轨迹，不虚构曲线；原失败证据见EXP084完整报告。','']
    REVIEW.parent.mkdir(parents=True,exist_ok=True);parent.old.atomic(REVIEW,lambda f:f.write('\n'.join(lines).encode('utf8')))

def policy(kind,seed=8501):
    st=torch.load(ROOT/'models'/f'{kind}.{seed}'/'best.pt',weights_only=True,map_location='cpu')
    net=Net();net.load_state_dict(st['network']);net.eval()
    @torch.inference_mode()
    def choose(x,p):
        v=net((torch.tensor(compact(x))-st['mean'])/st['scale'])[:2]*st['y_scale']
        return bool(v[1]>0),v.tolist()
    return choose

def iteration_case(base,history,c,kind,round_):
    torch.set_num_threads(1);path=ROOT/f'iteration{round_}'/f"{c['id']}.json"
    if path.exists():
        cached=parent.old.read(path)
        if cached['continuation_sha']!=parent.old.digest(ROOT/'models'/f'{kind}.8501'/'best.pt'):raise ValueError('Cached continuation policy changed')
        return str(path)
    model=parent.weights(base,history,c['seed']);choose=policy(kind);points=[];started=time.perf_counter()
    def collect(opening,x,audit,candidate):
        p,point,ctx=opening;arms={}
        for a in (0,1):
            def continuation(z,j):return (bool(a),None) if j==p else choose(z,j)
            branch,_=parent.rollout(base,c,model,continuation,opening=opening,end=p+8)
            assert branch['rows'][0]['timing']['candidate']==candidate and branch['rows'][0]['timing']['x']==x
            arms[str(a)]=branch
        parent.old.same_flow(arms['0']['rows'],arms['1']['rows'])
        costs={a:{str(h):parent.branch_cost(v,h) for h in (4,8)} for a,v in arms.items()}
        risk=[int(any(costs[a]['8']['metrics'][k]>0 for k in ('severe','censored')) or costs[a]['8']['admitted_unfinished_at_deadline']>0) for a in ('0','1')]
        points.append(dict(id=f"{c['id']}.{p}",seed=c['seed'],x=x,compact=compact(x).tolist(),
            y=[costs['1'][str(h)]['aligned']-costs['0'][str(h)]['aligned'] for h in (4,8)],risk=risk,
            backlog=[np.log1p(costs[a]['8']['metrics']['cpu']) for a in ('0','1')],
            opening=parent.boundary_record(*point,c['population'],'holdout',c['scene'],p),candidate=candidate,arms=arms,costs=costs))
    visit,_=parent.rollout(base,c,model,choose,collector=collect)
    parent.save(path,dict(case=c,points=points,visit=visit,seconds=time.perf_counter()-started,
        continuation_kind=kind,continuation_sha=parent.old.digest(ROOT/'models'/f'{kind}.8501'/'best.pt')))
    return f'{c["id"]} complete'

def final_case(base,history,c,kind):
    torch.set_num_threads(1);path=ROOT/'evaluation'/f"{c['id']}.json"
    if path.exists():return str(path)
    model=parent.weights(base,history,c['seed']);arms={}
    for name in (*parent.FIXED,'EXP084',*(f'EXP085-{s}' for s in (8501,8502,8503))):
        choose=parent.predictor(parent.ROOT,8401) if name=='EXP084' else policy(kind,int(name.split('-')[1])) if name.startswith('EXP085') else name
        result,_=parent.rollout(base,c,model,choose)
        if arms:parent.old.same_flow(next(iter(arms.values()))['rows'],result['rows'])
        arms[name]=result
    parent.save(path,dict(case=c,arms=arms,kind=kind,source=parent.old.source()))
    return c['id']+' final complete'

def physical_report(state):
    report(parent.old.read(ROOT/'offline.json'))
    lines=['','## 策略一致物理验证状态',str(state),'']
    documents=[]
    for p in sorted((ROOT/'evaluation').glob('*.json')):
        d=parent.old.read(p)
        documents.append(d)
        table(lines,[d['case']['id'],'R','E2E','P99','严重/截尾/未完成','CPU/链路','REDEPLOY','保持间隔'],[
            [a,m['reward'],m['e2e'],m['p99'],f"{m['severe']}/{m['censored']}/{m['terminal']}",f"{m['cpu']}/{m['wire']}",m['redeploy'],m['intervals']]
            for a,v in d['arms'].items() for m in [v['metrics']]])
        fig,axes=parent.old.plt.subplots(3,1,figsize=(11,8),sharex=True)
        for arm,v in d['arms'].items():
            for ax,key in zip(axes,('reward_raw','e2e_completed','p99')):
                ax.plot([r['period'] for r in v['rows']],[r[key] for r in v['rows']],label=arm,lw=.7);ax.set_ylabel(key)
        axes[0].legend(ncol=4,fontsize=7);axes[-1].set_xlabel('Physical GS')
        name=f"fig.{d['case']['id']}.png";fig.savefig(ROOT/name,dpi=120,bbox_inches='tight');parent.old.plt.close(fig)
        lines+=['',f'![原始未平滑轨迹](../../../runs/results/EXP085/{name})']
    if len(documents)==20:
        accepted=[]
        for netseed in (8501,8502,8503):
            name=f'EXP085-{netseed}';gaps={};risk=[]
            for seed in parent.SEEDS:
                cases=[d for d in documents if d['case']['seed']==seed]
                gaps[seed]=[float(np.mean([d['arms'][name]['metrics'][k]-d['arms']['2h']['metrics'][k] for d in cases])) for k in ('reward','e2e','p99')]
            for d in documents:
                a=d['arms'][name]['metrics'];b=d['arms']['2h']['metrics']
                if (d['case']['family']=='pressure' and any(a[k]>b[k] for k in ('severe','censored','terminal'))) or a['p99']>b['p99']*1.05:risk.append(d['case']['id'])
            avg=np.mean(list(gaps.values()),axis=0);ok=avg[0]>0 and avg[1]<=0 and sum(v[0]>0 and v[1]<=0 for v in gaps.values())>=3 and not risk
            accepted.append(ok);lines += [f'网络{netseed} 相对2h逐模型[ΔR,ΔE2E,ΔP99]={gaps}；越界={risk}；通过={ok}。']
            for fixed in ('4h','8h','EXP084'):
                lines += [f'网络{netseed} 相对{fixed}总体配对均值：'+str({k:float(np.mean([d['arms'][name]['metrics'][k]-d['arms'][fixed]['metrics'][k] for d in documents])) for k in ('reward','e2e','p99')})]
        lines += ['三初始化均支持晋级，仍需后续复现，不能声称无限期收敛。' if all(accepted) else '未通过正式可靠性与收益验收，不建议加入论文主方案，不自动继续迭代。']
    elif state.get('selected') is None and state['status'].startswith('stopped'):
        lines += ['策略一致目标下验证未通过，停止P4；固定KEEP标签上的改进尚不能转化为可靠自适应价值，不建议加入论文主方案。']
    with REVIEW.open('a',encoding='utf8') as f:f.write('\n'.join(lines))

def physical(workers):
    torch.set_num_threads(1)
    if not parent.old.read(ROOT/'offline.json')['proceed']:raise ValueError('P1 gate failed')
    base=parent.old.read(parent.old.EXP043/'protocol.json')['plan'];history=parent.old.read(parent.old.ROOT/'protocol.json')['plan']
    for seed,sha in parent.old.read(parent.ROOT/'protocol.json')['plan']['models'].items():
        if parent.old.digest(parent.old.files(parent.old.ROOT,int(seed))/'last.pt')!=sha:raise ValueError('Frozen D/S model changed')
    registry=ROOT/'physical_protocol.json'
    if registry.exists():plan=parent.old.read(registry)
    else:
        cases=[]
        for round_ in (1,2):
            for part in ('train','validation'):
                for seed in parent.SEEDS[:3]:
                    for fam in parent.FAMILIES:
                        c=parent.demand_selection(base,f'EXP085-iteration{round_}-{part}',seed,fam,0)
                        c.update(id=f'i{round_}.{part}.{seed}.{fam}',part=part,round=round_,seed=seed,family=fam,periods=32,openings=[4,12,20,28],decoder_key=parent.old.stable_seed('EXP085-iteration-field',round_,part,seed,fam))
                        cases.append(c)
        final=[]
        for k in parent.old.read(ROOT/'protocol.json')['final_keys']:
            final.append(dict(id=f"held.{k['seed']}.{k['family']}",seed=k['seed'],family=k['family'],population=k['population'],scene=k['scene'],decoder_key=k['field'],periods=72,openings=[]))
        plan=dict(cases=cases,final=final,primary_policy_seed=8501,source=parent.old.source(),protocol=parent.old.digest(REVIEW.with_name('PROTOCOL.md')),physical_protocol=parent.old.digest(REVIEW.with_name('PHYSICAL_PROTOCOL.md')))
        parent.save(registry,plan)
    started=time.perf_counter();state=dict(status='running',rounds=[],source=parent.old.source());kind='robust';selected=None
    try:
        for round_ in (1,2):
            tasks=[(base,history,c,kind,round_) for c in plan['cases'] if c['round']==round_]
            parent.jobs(iteration_case,tasks,workers)
            groups={'train':[],'validation':[]}
            for p in sorted((ROOT/f'iteration{round_}').glob('*.json')):
                d=parent.old.read(p)
                groups[d['case']['part']].extend([{k:v for k,v in point.items() if k not in ('opening','arms','costs')} for point in d['points']])
            tr,va=groups['train'],groups['validation'];assert len(tr)==len(va)==48
            previous=policy(kind);prior=assess([previous(p['x'],0)[1] for p in va],va)
            constant=assess(np.tile(np.mean([p['y'] for p in tr],axis=0),(len(va),1)),va)
            candidate=f'robust_i{round_}';fits=[fit(candidate,s,tr,va) for s in (8501,8502,8503)]
            passed=[]
            for fit_ in fits:
                a=fit_['validation'];v=a['value']['8'];b=prior['value']['8'];c=constant['value']['8']
                passed.append(v['mae']<min(b['mae'],c['mae']) and v['physical_regret']<min(b['physical_regret'],c['physical_regret']) and min(v['keep_accuracy'],v['redeploy_accuracy'])>=.55 and a['dangerous_choices']<=prior['dangerous_choices'])
            state['rounds'].append(dict(round=round_,passed=passed,previous=prior,constant=constant,results=[f['validation'] for f in fits]))
            parent.save(ROOT/'physical_status.json',state);physical_report(state)
            if sum(passed)<2:break
            selected=kind=candidate
        if selected:
            parent.jobs(final_case,[(base,history,c,selected) for c in plan['final']],workers)
            state.update(status='evaluation complete',selected=selected)
        else:state.update(status='stopped at policy-consistent validation',selected=None)
        state['seconds']=time.perf_counter()-started;parent.save(ROOT/'physical_status.json',state);physical_report(state)
        print(state['status'],flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));parent.save(ROOT/'physical_status.json',state);physical_report(state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['offline','physical','report']);p.add_argument('--workers',type=int,default=5);args=p.parse_args()
    if args.mode=='offline':run()
    elif args.mode=='physical':physical(args.workers)
    else:report(parent.old.read(ROOT/'offline.json'))
