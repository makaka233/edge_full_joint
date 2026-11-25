"""Verify the five resumed runs and write their paired results into the handoff."""

from itertools import chain
from pathlib import Path
from statistics import mean

from comparison.report import records
from edge_joint.adaptive_runner import FIXED, read_checkpoint
from edge_joint.adaptive_storage import atomic, run_lock
from scripts.exp027_run import fingerprint
from scripts.exp029_report import summarize
from scripts.exp029_run import digest
from scripts.exp043_run import verify_work
from scripts.exp044_run import same_flow
from scripts.exp047_run import read
from scripts.exp055_joint_report import ENVIRONMENTS, SCENARIOS, averages, both_better, source_counts, triplet
from scripts.exp056_long import START, destination, verify_prefix
from scripts.exp056_run import CASES, ROOT, SEEDS


REVIEW = Path('research/EXP056/AI_RESEARCH_REVIEW.md')
BEGIN = '<!-- BEGIN EXP056 GS1008 -->'
END = '<!-- END EXP056 GS1008 -->'
TARGET = 126
FAILURES = ('spikes','censored','terminal')


def seed_evidence(registered,seed):
    out = destination(seed)
    old = registered['files'][str(seed)]
    files = {key:ROOT/f'long-eval.seed{seed}.{key[0]}.{key[1]}.episode{TARGET}.json' for key in CASES}
    required = (out/'resume_latest.pt',out/'last.pt',out/'continuation.json',*files.values())
    if not all(f.is_file() for f in required):
        return None
    with run_lock(out):
        state,last = (read_checkpoint(out/f'{k}.pt') for k in ('resume_latest','last'))
        if (state['episode'],last['episode'])!=(TARGET,TARGET):
            return None
        imported = read(out/'continuation.json')['import']
        if (imported['episode']!=START or imported['seed']!=seed or
                imported['files']['resume_latest.pt']['sha256']!=old['resume'] or
                any(x['origin']!={'kind':'random'} or x['config']['seed']!=seed or
                    x['environment_id']!=registered['exp055'] for x in (state,last)) or
                state['config']!=last['config']):
            raise ValueError(f'Unexpected continuation checkpoint for seed {seed}')
        for f,n in state['offsets'].items():
            archived = imported['files'][f]
            if (out/f).stat().st_size!=n:
                raise ValueError(f'Uncommitted training ledger for seed {seed}: {f}')
            verify_prefix(out/f,archived['size'],archived['sha256'])
        train,updates,fixed = (list(records(out/f'{k}.jsonl')) for k in ('train','updates','fixed'))
        model_sha = digest(out/'last.pt')
    periods = state['config']['periods']
    split,final = START*periods,TARGET*periods
    if ([r['gs'] for r in train]!=list(range(1,final+1)) or
            [r['gs'] for r in updates]!=list(range(periods,final+1,periods)) or
            (updates[-1]['d_updates'],updates[-1]['s_updates'])!=(state['d_updates'],state['s']['updates'])):
        raise ValueError(f'Incomplete continuation ledger for seed {seed}')
    fixed_endpoints = [[r for r in fixed if r['episode']==p] for p in (START,TARGET)]
    if any(len(r)!=periods*len(FIXED) for r in fixed_endpoints):
        raise ValueError(f'Missing fixed validation endpoints for seed {seed}')
    same_flow(*fixed_endpoints)
    scenarios = {}
    for (v,c),path in files.items():
        data = read(path)
        meta = old['case'][f'{v}.{c}']
        expected = fingerprint(dict(model=model_sha,meta=meta,episode=TARGET,registered=fingerprint(registered)))
        if (data['stamp']!=expected or (data['seed'],data['variant'],data['profile'],data['episode'])!=(seed,v,c,TARGET)
                or digest(Path(meta['path']))!=meta['sha256']):
            raise ValueError(f'Independent evaluation cannot be paired: {path}')
        earlier = read(Path(meta['path']))['arms']['last']['rows']
        newer = data['last']['rows']
        if len(earlier)!=periods or len(newer)!=periods:
            raise ValueError(f'Truncated independent evaluation: {path}')
        same_flow(earlier,newer)
        scenarios[v,c] = dict(before=summarize(earlier),after=summarize(newer),source=data['source'])
    continuation = updates[START:]
    initial_train,continued_train = train[:split],train[split:]
    return dict(seed=seed,scenarios=scenarios,online=(summarize(initial_train),summarize(continued_train)),
                fixed=tuple(summarize(r) for r in fixed_endpoints),
                d=state['d_updates'],s=state['s']['updates'],
                d_changes=sum(r['d_change']>0. for r in continuation),
                s_changes=sum(r['s_change']>0. for r in continuation),
                target_changes=sum(r['s_target_change']>0. for r in continuation),
                error=verify_work(continued_train),wall=state['wall_seconds'],
                source=[r for r in state['source_history'] if r['target_episode']>START])


def fail_counts(s):
    return ' / '.join(str(s[k]) for k in FAILURES)


def paired_means(pairs):
    return [averages([p[k] for p in pairs]) for k in ('before','after')]


def markdown(runs):
    pairs = [p for r in runs for p in r['scenarios'].values()]
    before,after = paired_means(pairs)
    wins = sum(both_better(p['before'],p['after']) for p in pairs)
    seed_wins = sum(both_better(*paired_means(list(r['scenarios'].values()))) for r in runs)
    p99_worse = sum(p['after']['p99']>p['before']['p99'] for p in pairs)
    losses = sorted((p['after']['reward']-p['before']['reward'],r['seed'],v,c)
                    for r in runs for (v,c),p in r['scenarios'].items())
    delta,s,v,c = losses[0]
    lines = [
        '### 正式结果（GS504 → GS1008）','',
        f'五种子都完成了累计 {TARGET} 个 episode 和 1008 GS。20 个配对独立场景的总体指标为 '
        f'{triplet(before)} → {triplet(after)}（Reward / 完成 E2E s / 惩罚后 P99 s）。'
        f'Reward 与 E2E 同时改善的有 {seed_wins}/5 个种子（各自四场景均值）和 {wins}/20 个场景，'
        f'{p99_worse}/20 个场景的 P99 上升。',
        f'最差配对 Reward 变化为 {s} {ENVIRONMENTS[v]} / {SCENARIOS[c]} 的 {delta:+.3f}。'
        '总体均值和单条轨迹的变化均需保留，两个端点不足以证明长期收敛。','',
        '#### 配对独立评价','',
        '两端都用冻结 last，同一场景逐槽复核请求指纹。先按原始请求样本计算场景 E2E/P99，再跨场景等权。'
        '各格指标依次为 Reward / 完成 E2E（s）/ 惩罚后 P99（s）。','',
        '| 环境 / 负载 | GS504 | GS1008 | ΔReward / ΔE2E s / ΔP99 s |',
        '|---|---:|---:|---:|',
    ]
    for v,c in CASES:
        a,b = paired_means([r['scenarios'][v,c] for r in runs])
        d = ' / '.join(f'{b[k]-a[k]:+.4f}' for k in ('reward','e2e','p99'))
        lines.append(f'| {ENVIRONMENTS[v]} / {SCENARIOS[c]} | {triplet(a)} | {triplet(b)} | {d} |')
    lines.extend(['','| 种子 | GS504 四场景均值 | GS1008 四场景均值 | ΔReward / ΔE2E s / ΔP99 s |',
                  '|---|---:|---:|---:|'])
    for r in runs:
        a,b = paired_means(list(r['scenarios'].values()))
        d = ' / '.join(f'{b[k]-a[k]:+.4f}' for k in ('reward','e2e','p99'))
        lines.append(f"| {r['seed']} | {triplet(a)} | {triplet(b)} | {d} |")
    lines.extend(['','| 种子 | 环境 / 负载 | GS504 | GS1008 |',
                  '|---|---|---:|---:|'])
    for r in runs:
        for (v,c),p in r['scenarios'].items():
            lines.append(f"| {r['seed']} | {ENVIRONMENTS[v]} / {SCENARIOS[c]} | "
                         f"{triplet(p['before'])} | {triplet(p['after'])} |")
    lines.extend(['','| 端点 | 请求 | 严重期 / 截尾 / 终局未完成 | 最大 CPU CU / 有线积压 Mbit |',
                  '|---|---:|---:|---:|'])
    for gs,k in ((504,'before'),(1008,'after')):
        metrics = [p[k] for p in pairs]
        counts = fail_counts({f:sum(p[f] for p in metrics) for f in FAILURES})
        peak = ' / '.join(f'{max(d[k] for d in metrics):.3f}' for k in ('cpu','wire'))
        lines.append(f"| GS{gs} | {sum(d['requests'] for d in metrics)} | {counts} | {peak} |")
    lines.extend(['','#### 训练、固定验证与风险','',
                  '在线前后两段各 504 GS，保留所有严重期。严重期定义为 Reward ≤ −20。'
                  '固定验证只用于观察期内趋势及选模，与上方独立评价分开。','',
                  '| 种子 | 在线前段 / 续训段 Reward | 固定验证 GS504 | 固定验证 GS1008 | '
                  '前段严重期 / 截尾 / 终局 | 续训段严重期 / 截尾 / 终局 | 续训最差 Reward |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for r in runs:
        a,b = r['online']
        x,y = r['fixed']
        lines.append(f"| {r['seed']} | {a['reward']:.3f} / {b['reward']:.3f} | {triplet(x)} | {triplet(y)} | "
                     f"{fail_counts(a)} | {fail_counts(b)} | {b['worst']:.3f} |")
    fx,fy = (averages([r['fixed'][i] for r in runs]) for i in range(2))
    before_risk,after_risk = (fail_counts({k:sum(r['online'][i][k] for r in runs) for k in FAILURES}) for i in range(2))
    lines.extend([
        '',
        f'固定验证五种子均值为 {triplet(fx)} → {triplet(fy)}。'
        f'在线五种子的严重期 / 截尾 / 终局计数为 {before_risk} → {after_risk}。','',
        '| 种子 | 累计 D / S 更新 | 续训非零变化的 episode 数 D / S / S 目标网 | 累计训练计时 s |',
        '|---|---:|---:|---:|',
    ])
    for r in runs:
        lines.append(f"| {r['seed']} | {r['d']} / {r['s']} | "
                     f"{r['d_changes']} / {r['s_changes']} / {r['target_changes']} | {r['wall']:.1f} |")
    lines.extend([
        '', f"验证了继承的日志字节前缀和已提交偏移。续训最大资源对账误差为 {max(r['error'] for r in runs):.2e}。",
        f"续训来源为 {source_counts(chain.from_iterable(r['source'] for r in runs))}。"
        f"独立评价来源为 {source_counts(p['source'] for p in pairs)}。",
        '独立结果覆盖原四类短场景。该预算无法推断持续高负载下的失效率，也不能用来重新估计 GS1008 的组件单独贡献。','',
    ])
    return '\n'.join(lines)


def write_long_review():
    registered = read(ROOT/'protocol.json')['plan']
    runs = [seed_evidence(registered,s) for s in SEEDS]
    if any(r is None for r in runs):
        return
    text = REVIEW.read_text(encoding='utf-8')
    body = f'{BEGIN}\n{markdown(runs)}{END}'
    if BEGIN in text or END in text:
        head,start,rest = text.partition(BEGIN)
        _,stop,tail = rest.partition(END)
        if not start or not stop or BEGIN in rest or END in head or END in tail:
            raise ValueError('Invalid EXP056 report markers')
        updated = head+body+tail
    else:
        marker = '\n## 5. 计算预算与复现'
        index = text.index(marker)
        updated = text[:index]+body+'\n'+text[index:]
    if updated!=text:
        atomic(REVIEW,lambda f:f.write(updated.encode('utf-8')))
    print(REVIEW,flush=True)
