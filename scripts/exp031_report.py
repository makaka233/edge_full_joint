"""Analyze EXP031 frozen pairs and produce raw curves and incident attribution."""

import csv
import io
import json

import numpy as np

from edge_d.world import make_world
from edge_joint.adaptive_storage import atomic, line
from scripts.exp019_report import unpack
from scripts.exp024_evaluate import assert_flow
from scripts.exp027_run import PROFILES
from scripts.exp029_report import COLORS, intervals, layout_difference, summarize, summarize_quartet
from scripts.exp028_report import iter_json
from scripts.exp030_continue import SEEDS, prefix
from scripts.exp030_frozen import RESULTS as EXP030
from scripts.exp030_report import archive_flows, plot_raw
from scripts.exp031_frozen import FOCAL_PERIOD, PAIRS, RESULTS, SEED, archived_path, destination, registry


METRICS = ('reward_raw', 'e2e_completed', 'p95', 'p99')
TITLES = ('原始 Reward', '真实完成 E2E（秒）', '惩罚后 P95（秒）', '惩罚后 P99（秒）')


def relative_arms(rows):
    a = rows['A']
    return {p:layout_difference(a, v) for p, v in rows.items() if p != 'A'}


def incident(case, cells):
    env = make_world(case['environment'], case['scene'], 'holdout')
    selected = {p:next(r for r in rows if r['period'] == FOCAL_PERIOD) for p, rows in cells.items()}
    matrices = {p:unpack(env, r['active_matrix']) for p, r in selected.items()}
    current = {p:dict(metrics={k:r[k] for k in METRICS}, queue_in=r['backlog_in'],
                      queue_out=r['backlog_out'], cpu=r['cpu'], wired=r['wired'],
                      holding=r['holding_cost'], switch=r['switch_cost'],
                      supports=[np.flatnonzero(row).tolist() for row in matrices[p]],
                      added=np.argwhere(matrices[p] & ~matrices['A']).tolist(),
                      removed=np.argwhere(matrices['A'] & ~matrices[p]).tolist())
               for p, r in selected.items()}
    return dict(period=FOCAL_PERIOD, cells=current, link_index=env.w.directed)


def plot_archive(rows, path):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained', sharex=True)
    for axis, key, title in zip(axes.flat, METRICS, TITLES, strict=True):
        for (label, data), color in zip(rows.items(), (COLORS[0], COLORS[2]), strict=True):
            axis.plot([r['period'] for r in data], [r[key] for r in data], color=color,
                      label={'A': 'EXP029 校准续训', 'D': 'EXP030 负载条件校准续训'}[label])
        axis.axvline(FOCAL_PERIOD, color='grey', linestyle=':', linewidth=1.)
        axis.set_title(title)
        axis.grid(alpha=.2)
    axes[0, 0].legend()
    for axis in axes[1]:
        axis.set_xlabel('业务期（24 秒）')
    fig.suptitle('Seed16 持续 0.70 第 2 条历史轨迹 · GS5512 原始归档')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def set_fonts():
    from matplotlib import pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def archive_preview():
    set_fonts()
    case = registry()[0]
    archive = json.loads(archived_path(case).read_bytes())
    rows = {p:archive['rows'][PAIRS[p][0]+'.last'] for p in ('A', 'D')}
    assert_flow(*rows.values())
    stats = {p:summarize(periods) for p, periods in rows.items()}
    delta = {k:stats['D'][k]-v for k, v in stats['A'].items()}
    result = dict(case=case, stats=stats, delta=delta, layout=relative_arms(rows),
                  incident=incident(case, rows))
    atomic(RESULTS/'archive.preview.json', lambda f:f.write(line(result)))
    plot_archive(rows, RESULTS/'archive.sustained070.2.raw.png')
    print('Archived A/D reanalysis only. Delta Reward %.5f, E2E %.2f ms, P99 %.2f ms'
          % (delta['reward'], 1000*delta['e2e'], 1000*delta['p99']), flush=True)
    return result


def effects(cases):
    metrics = cases[0]['contrasts']
    out = {}
    for key in metrics:
        labels = (*metrics[key], 'D_given_S1', 'S_given_D1')
        pairs = []
        for case in cases:
            c = case['contrasts'][key]
            pairs.append(dict(c, D_given_S1=c['D']+c['interaction'],
                              S_given_D1=c['S']+c['interaction']))
        out[key] = {factor:intervals({p:[v[factor] for case, v in zip(cases, pairs, strict=True)
                                         if case['case']['profile'] == p] for p in PROFILES},
                                      ('EXP031', SEED, key, factor)) for factor in labels}
    return out


def regression_gate(result):
    """Apply the registered screen without treating it as a learning-mechanism finding."""
    classifications = {}
    for key, sign in (('reward', -1), ('e2e', 1), ('p99', 1)):
        values = result[key]
        adverse = lambda label: (values[label]['lower'] > 0 if sign == 1
                                  else values[label]['upper'] < 0)
        dominant = [label for label in ('D', 'S', 'interaction')
                    if adverse(label) and sign*values[label]['estimate'] >
                    max(abs(values[x]['estimate']) for x in ('D', 'S', 'interaction') if x != label)]
        attribution = dominant[0] if dominant and adverse('joint') else None
        if attribution in ('D', 'S') and not adverse('D_given_S1' if attribution == 'D' else 'S_given_D1'):
            attribution = None
        classifications[key] = dict(adverse_joint=adverse('joint'), dominant=attribution)
    main = {classifications[key]['dominant'] for key in ('reward', 'e2e')
            if classifications[key]['adverse_joint']}
    direction = next(iter(main)) if len(main) == 1 else None
    return dict(metrics=classifications, attribution=direction,
                status='mechanism audit required' if direction else 'insufficient for targeted change')


def observation_alias(record):
    """Find same-input states with different already outstanding work."""
    candidates = []
    for left, right in (('A', 'C'), ('B', 'D')):
        a, b = (record['cells'][p]['observations'] for p in (left, right))
        for x, y in zip(a, b, strict=True):
            if x['period'] != y['period']:
                raise ValueError('Observation periods differ')
            if x['input'] == y['input']:
                cpu = np.abs(np.subtract(x['cpu'], y['cpu']))
                wire = np.abs(np.subtract(x['wire'], y['wire']))
                candidates.append(dict(pair=[left, right], period=x['period'],
                                       max_node_gap_cu=float(cpu.max()), node=int(cpu.argmax()),
                                       max_link_gap_mbit=float(wire.max()), link=int(wire.argmax()),
                                       states=[x, y]))
    return dict(exact_input_pairs=len(candidates),
                max_cpu_example=max(candidates, key=lambda x:x['max_node_gap_cu'], default=None),
                max_link_example=max(candidates, key=lambda x:x['max_link_gap_mbit'], default=None))


def export_cases(selected):
    fields = ('reward', 'e2e', 'p95', 'p99', 'worst', 'worst1', 'worst5', 'cpu', 'wire',
              'censored', 'terminal', 'dropped', 'switch', 'holding')
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(('case', 'cell', 'opening_jobs', *fields))
    for case in selected:
        for label, cell in case['arms'].items():
            writer.writerow((case['case']['id'], label, case['opening_jobs'], *(cell[k] for k in fields)))
    atomic(RESULTS/'frozen.paired.csv', lambda f:f.write(buffer.getvalue().encode('utf8')))


def report(protocol):
    set_fonts()
    registered_source = json.loads((RESULTS/'frozen.protocol.json').read_bytes())['source']
    prior = archive_flows()
    exp030 = json.loads((EXP030/'frozen.review.json').read_bytes())
    prior.update(h for group in exp030['groups'].values() for case in group['cases'] for h in case['flow'])
    for path in EXP030.glob(f'bounded.holdout.seed{SEED}.*.json'):
        saved = json.loads(path.read_bytes())
        prior.update(r['trace_sha256'] for r in saved['rows']['exp029.last'])
    for seed in SEEDS:
        base = prefix(seed, 'exp029', 'bounded')
        for suffix in ('train.jsonl', 'fixed.jsonl'):
            prior.update(r['trace_sha256'] for r in iter_json(base/suffix))
    historical, all_new, plots, runtime = None, [], [], 0.
    for case in registry():
        path = destination(case)
        record = json.loads(path.read_bytes())
        if record['case'] != json.loads(line(case)) or record['protocol_sha256'] != protocol:
            raise ValueError(f'Unexpected case source in {path}')
        if (record['source']['commit'] != registered_source['commit']
                or record['source']['tracked_dirty'] != registered_source['tracked_dirty']):
            raise ValueError(f'Execution source differs from registration for {path}')
        for cell in record['cells'].values():
            if not (RESULTS/cell['requests']).is_file():
                raise FileNotFoundError(cell['requests'])
        summary = summarize_quartet(record, prior)
        summary.update(path=path.name, source=record['source'], observation_alias=observation_alias(record))
        runtime += record['wall_seconds']
        trace = dict(case=case, cells={p:dict(rows=[{k:r[k] for k in METRICS} for r in cell['rows']])
                                      for p, cell in record['cells'].items()})
        if case['historical']:
            rows = {p:v['rows'] for p, v in record['cells'].items()}
            detail = incident(case, rows)
            for p, cell in record['cells'].items():
                detail['cells'][p].update(
                    work=next(r for r in cell['work'] if r['period'] == FOCAL_PERIOD),
                    boundary=cell['incident_state'],
                    stages=[r for r in cell['stages'] if r[0] == FOCAL_PERIOD],
                    hops=[r for r in cell['hops'] if r[0] == FOCAL_PERIOD])
            historical = dict(**summary, focal=detail)
            plot_raw([trace], RESULTS/'frozen.history.raw.png', 'EXP031 Seed16 · 历史事故四格复现')
        else:
            if len(summary['flow']) != 24 or len(set(summary['flow'])) != 24:
                raise ValueError(f'Expected 24 distinct exogenous periods: {path}')
            prior.update(summary['flow'])
            all_new.append(summary)
            plots.append(trace)
    if historical is None or len(all_new) != 16:
        raise ValueError('The registered historical and 16 new quartets are required')
    result = effects(all_new)
    gate = regression_gate(result)
    atomic(RESULTS/'frozen.review.json', lambda f:f.write(line(dict(
        independent=all_new, historical=historical, effects=result, screen=gate,
        total_quartet_seconds=runtime))))
    export_cases([historical, *all_new])
    plot_raw(plots, RESULTS/'frozen.independent.raw.png', 'EXP031 Seed16 · 16 条新轨迹冻结四格')
    print(json.dumps(gate, ensure_ascii=False), flush=True)
