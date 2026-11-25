"""Publish the completed five-seed joint run into the EXP055 handoff."""

from pathlib import Path
from statistics import mean

from comparison.report import records
from edge_joint.adaptive_runner import FIXED, read_checkpoint
from edge_joint.adaptive_storage import RunFiles, atomic, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp043_run import verify_work
from scripts.exp044_run import same_flow
from scripts.exp047_run import read
from scripts.exp055_run import ROOT


REVIEW = Path('research/EXP055/AI_RESEARCH_REVIEW.md')
START = '<!-- BEGIN EXP055 JOINT RESULTS -->'
END = '<!-- END EXP055 JOINT RESULTS -->'
ARMS = ('gs0','last','best')
LIMIT = ('spikes','censored','terminal','dropped')
ENVIRONMENTS = {'original':'原环境', 'persistent':'持久热点'}
SCENARIOS = {'natural':'自然', 'pulse':'脉冲'}


def read_seed(plan,seed,cases,out,identity):
    with run_lock(out):
        required = ('last.pt','best.pt','resume_latest.pt','train.jsonl','updates.jsonl','fixed.jsonl')
        missing = next((out/f for f in required if not (out/f).is_file()),None)
        if missing is not None:
            return None, str(missing)
        state = read_checkpoint(out/'resume_latest.pt')
        p = state['episode']
        model_stamps = {k:digest(out/f'{k}.pt') for k in ARMS[1:]}
        projections = {k:read_checkpoint(out/f'{k}.pt') for k in ARMS[1:]}
        for model in (state,*projections.values()):
            if (model['origin']!={'kind':'random'} or model['environment_id']!=fingerprint(plan) or
                    model['config']!=plan['config'][str(seed)]):
                raise ValueError(f'Unexpected D/S checkpoint for seed {seed}')
        if p<1 or projections['last']['episode']!=p or projections['best']['episode']!=state['best']['episode']:
            return None, f'seed {seed} has no committed checkpoint projections at its current target'
        evaluations = {}
        for v,c in cases:
            path = out/f'eval.{v}.{c}.episode{p}.json'
            if not path.is_file():
                return None, str(path)
            data = read(path)
            key,pop,stamp = identity(plan,seed,v,c,model_stamps)
            if (data['stamp']!=stamp or data['model_sha']!=model_stamps or set(data['arms'])!=set(ARMS) or
                    (data['seed'],data['variant'],data['profile'],data['scene'],data['population'])!=(seed,v,c,key,pop)):
                raise ValueError(f'Independent evaluation does not match the current checkpoint: {path}')
            evaluations[v,c] = data
        if any((out/f).stat().st_size!=size for f,size in state['offsets'].items()):
            raise ValueError(f'Uncommitted training log records for seed {seed}')
        train,fixed,updates = (list(records(out/f'{k}.jsonl')) for k in ('train','fixed','updates'))

    periods = state['config']['periods']
    gs = p*periods
    if ([r['gs'] for r in train]!=list(range(1,gs+1)) or
            [u['gs'] for u in updates]!=list(range(periods,gs+1,periods)) or
            (updates[-1]['d_updates'],updates[-1]['s_updates'])!=(state['d_updates'],state['s']['updates'])):
        raise ValueError(f'Training log and saved update counts differ for seed {seed}')
    stages = sorted({r['episode'] for r in fixed})
    if not {0,p,projections['best']['episode']}.issubset(stages):
        raise ValueError(f'Missing fixed validation for seed {seed}')
    fixed_scores = {}
    for step in stages:
        rows = [r for r in fixed if r['episode']==step]
        if [(r['scene'],r['period']) for r in rows]!=[(s,i+1) for s in FIXED for i in range(periods)]:
            raise ValueError(f'Incomplete fixed validation at seed {seed} episode {step}')
        if step==0:
            baseline = rows
        else:
            same_flow(baseline,rows)
        fixed_scores[step] = summarize(rows)
    scores = {}
    error = verify_work(train)
    for k,d in evaluations.items():
        scores[k] = {}
        for arm in ARMS:
            raw = d['arms'][arm]['rows']
            if len(raw)!=periods:
                raise ValueError(f'Truncated evaluation for seed {seed} {k} {arm}')
            same_flow(d['arms']['gs0']['rows'],raw)
            error = max(error,verify_work(raw))
            scores[k][arm] = summarize(raw)
    width = min(64, gs//2)
    worst_gs = min(train,key=lambda r:r['reward_raw'])['gs']
    return dict(seed=seed,episodes=p,gs=gs,d=state['d_updates'],s=state['s']['updates'],
                best=projections['best']['episode']*periods,wall=state['wall_seconds'],
                d_changed=sum(u['d_change']>0. for u in updates),s_changed=sum(u['s_change']>0. for u in updates),
                max_error=error,train=summarize(train),early=summarize(train[:width]),late=summarize(train[-width:]),
                width=width,worst_gs=worst_gs,fixed=fixed_scores,scores=scores,
                train_source=state['source_history'],eval_source=[d['source'] for d in evaluations.values()]),None


def metric(value,digits):
    return '—' if value is None else f'{value:.{digits}f}'


def triplet(score):
    return ' / '.join(metric(score[k],n) for k,n in (('reward',3),('e2e',4),('p99',3)))


def averages(values):
    means = {}
    for k in ('reward','e2e','p99'):
        samples = [v[k] for v in values]
        means[k] = None if None in samples else mean(samples)
    return means


def both_better(old,new):
    return (old['e2e'] is not None and new['e2e'] is not None and
            new['reward']>old['reward'] and new['e2e']<old['e2e'])


def source_counts(entries):
    counts = {}
    for s in entries:
        suffix = '（有未提交跟踪改动）' if s['tracked_dirty'] else ''
        key = f"`{s['commit']}`{suffix}"
        counts[key] = counts.get(key,0)+1
    return '；'.join(f'{key} × {count}' for key,count in sorted(counts.items()))


def markdown(runs,cases):
    base = runs[0]
    paired = [(run['seed'],v,c,run['scores'][v,c]) for run in runs for v,c in cases]
    all_scores = {a:[d[a] for _,_,_,d in paired] for a in ARMS}
    avg = {a:averages(values) for a,values in all_scores.items()}
    wins = [(seed,v,c) for seed,v,c,d in paired if both_better(d['gs0'],d['last'])]
    comparable_p99 = [(seed,v,c,d) for seed,v,c,d in paired
                      if d['last']['p99'] is not None and d['gs0']['p99'] is not None]
    exceptions = [(seed,v,c,d['last']['p99']-d['gs0']['p99']) for seed,v,c,d in comparable_p99
                  if d['last']['p99']>d['gs0']['p99']]
    seed_wins = sum(both_better(averages([d['gs0'] for d in r['scores'].values()]),
                                averages([d['last'] for d in r['scores'].values()])) for r in runs)
    old,new = avg['gs0'],avg['last']
    worst = min(runs,key=lambda r:r['train']['worst'])
    severe = sum(r['train']['spikes'] for r in runs)
    lines = [
        '## 五种子正式联合训练与独立评价','',
        f"固定两小时部署从随机初始化完成 {len(runs)} 个种子，每种子 {base['episodes']} 个 episode、"
        f"{base['gs']} GS。新训练的 last 相对同种子 GS0 的独立平均 Reward 为 {old['reward']:.3f} → "
        f"{new['reward']:.3f}，平均完成 E2E 为 {metric(old['e2e'],4)} → {metric(new['e2e'],4)} s。"
        f"Reward 与 E2E 同时改善的有 {seed_wins}/{len(runs)} 个种子（各自四场景均值）和 "
        f"{len(wins)}/{len(paired)} 个独立配对场景。"
        f"独立 P99 为 {metric(old['p99'],3)} → {metric(new['p99'],3)} s，"
        f"可比较的 {len(comparable_p99)} 个场景中有 {len(exceptions)} 个升高。"
        f"在线训练仍有 {severe} 个严重期，最差发生在 {worst['seed']} 的 GS{worst['worst_gs']}"
        f"（Reward {worst['train']['worst']:.3f}）。D、S 各自的净贡献需要单独消融，且本试验的部署时机固定。",'',
        '### 训练记录与更新核验','',
        'Reward 按 GS 平均。E2E 按完成请求平均，P99 对期内记账的惩罚后请求样本计算。'
        '严重期为 Reward ≤ −20；截尾为截止时仍未完成的请求，终局未完成为 episode 尾部尚存请求。'
        'D/S 参数变化列给出分别观测到非零参数变化的 episode 数。','',
        '| 种子 | D / S 更新 | 参数变化 D / S | 全程 Reward / E2E s / P99 s | 最差 Reward 所在 GS | 严重期 / 截尾 / 终局未完成 | 训练计时 s |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in runs:
        t = r['train']
        lines.append(f"| {r['seed']} | {r['d']} / {r['s']} | {r['d_changed']} / {r['s_changed']} | {triplet(t)} | "
                     f"{t['worst']:.3f}（GS{r['worst_gs']}） | {t['spikes']} / {t['censored']} / {t['terminal']} | {r['wall']:.1f} |")
    lines.extend([
        '', f"原始训练前后各 {base['width']} GS 的五种子平均及固定验证使用各自账本计算。"
        '固定验证流用于选 best。','',
        '| 种子 | 在线最初窗口 | 在线最后窗口 | 固定验证 GS0 | 固定验证末期 | best GS |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for r in runs:
        lines.append(f"| {r['seed']} | {triplet(r['early'])} | {triplet(r['late'])} | "
                     f"{triplet(r['fixed'][0])} | {triplet(r['fixed'][r['episodes']])} | {r['best']} |")
    training_summaries = (
        averages([r['early'] for r in runs]), averages([r['late'] for r in runs]),
        averages([r['fixed'][0] for r in runs]), averages([r['fixed'][r['episodes']] for r in runs]),
    )
    lines.append(f"| 五种子均值 | {' | '.join(map(triplet,training_summaries))} | — |")
    lines.extend([
        '', '各三项均依次为原始 Reward、完成 E2E（s）、惩罚后 P99（s）。','',
        '### 独立评价','',
        '训练使用原环境。独立评价冻结权重、关闭探索，覆盖原环境和持久热点的自然与单槽 0.65 CPU 提供负载脉冲。'
        '每个场景持续 8 个 24 秒物理槽，GS0、last 和 best 在场景内使用同一请求流。'
        '每种子的四个环境×负载组合，是成组配对观察。总体按场景等权平均，场景内 E2E/P99 来自请求样本。'
        '表内三项依次为原始 Reward、完成 E2E（s）、惩罚后 P99（s）。缺少可定义的时延时标为 —。','',
        '| 环境与负载 | 场景数 | GS0 | last | best |','|---|---:|---:|---:|---:|',
    ])
    for v,c in cases:
        subset = [d for _,variant,profile,d in paired if (v,c)==(variant,profile)]
        values = [triplet(averages([d[a] for d in subset])) for a in ARMS]
        lines.append(f"| {ENVIRONMENTS[v]} / {SCENARIOS[c]} | {len(subset)} | {' | '.join(values)} |")
    lines.append(f"| 合计 | {len(paired)} | {' | '.join(triplet(avg[a]) for a in ARMS)} |")
    lines.extend(['','| 种子 | 独立环境 / 负载 | GS0 | last | best |','|---|---|---:|---:|---:|'])
    for seed,v,c,scores in paired:
        values = ' | '.join(triplet(scores[a]) for a in ARMS)
        lines.append(f'| {seed} | {ENVIRONMENTS[v]} / {SCENARIOS[c]} | {values} |')
    lines.extend([
        '', '| 策略 | 请求数 | 严重期 | 截尾 | 终局未完成 | 丢弃 | 最大 CPU 积压 CU | 最大有线积压 Mbit |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for a,values in all_scores.items():
        counts = ' | '.join(str(sum(s[k] for s in values)) for k in ('requests',*LIMIT))
        lines.append(f"| {a} | {counts} | {max(s['cpu'] for s in values):.3f} | {max(s['wire'] for s in values):.3f} |")
    if exceptions:
        cases_text = '、'.join(f'{s} {ENVIRONMENTS[v]}/{SCENARIOS[c]} +{d:.3f} s' for s,v,c,d in exceptions)
        lines.extend(['',f'last 的 P99 高于 GS0 的场景包括 {cases_text}。'])
    lines.extend([
        '', '### 证据范围与版本','',
        f"- 训练与评价原账本的最大资源守恒误差为 `{max(r['max_error'] for r in runs):.2e}`。"
        '汇总核验了训练步、检查点更新计数、独立评价模型 SHA256 与配对请求指纹。',
        f"- 训练运行段记录 {source_counts(s for r in runs for s in r['train_source'])}。",
        f"- 独立评价场景记录 {source_counts(s for r in runs for s in r['eval_source'])}。"
        '有未提交改动的运行保留当时的标志。相关并行评价修复之后提交为 `4a788e7`。',
        '- 独立评价为每种子四条短时场景。总体均值只覆盖登记流、这些随机种子和运行长度。'
        '继续训练的稳定性、未覆盖的高压模式与学习式部署时机仍需要额外证据。'
        '原账本可供复算。[本地训练与独立评价账本](../../runs/results/EXP055/)',
    ])
    return '\n'.join(lines)


def merge(text,body):
    block = f'{START}\n{body}\n{END}'
    if START in text or END in text:
        left,sep,rest = text.partition(START)
        _,end,right = rest.partition(END)
        if not sep or not end or START in rest or END in left or END in right:
            raise ValueError('Invalid generated-result markers in the EXP055 handoff')
        return left+block+right
    heading = '## 1. 已执行实验与条件价值范围'
    index = text.index(heading)
    return text[:index]+block+'\n\n'+text[index:]


def write_review(plan,seeds,cases,output,identity):
    with run_lock(RunFiles(ROOT/'joint-review')):
        runs = []
        for seed in seeds:
            evidence,pending = read_seed(plan,seed,cases,output(seed,False),identity)
            if evidence is None:
                print(f'Joint report waiting for {pending}; existing report preserved',flush=True)
                return False
            runs.append(evidence)
        if len({r['episodes'] for r in runs})!=1:
            print('Joint report requires one training target across all seeds; existing report preserved',flush=True)
            return False
        text = REVIEW.read_text(encoding='utf-8')
        updated = merge(text,markdown(runs,cases))
        if updated!=text:
            atomic(REVIEW,lambda f:f.write(updated.encode('utf-8')))
        print(REVIEW,flush=True)
    return True
