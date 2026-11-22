"""Causal CPU/link load bounds for fixed service chains and fixed network paths."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class TransportRisk:
    # All ratios are relative to capacity, with queued work amortized over the horizon.
    joint_ratio: float
    unavoidable_link_ratio: np.ndarray
    cpu_ratio: np.ndarray
    link_ratio: np.ndarray


def transport_risk(env, matrix, demand, *, cpu_work=None, link_work=None, seconds=24.):
    """Minimize a common utilization ratio over fractional service-chain routes.

    Each chain is expanded into access and execution layers. Layer flow is in
    requests/s. Arc costs are CU/request at the destination and Mbit/request
    on each of the physical route's directed links. The relaxation omits
    latency, indivisibility, future traffic, and downstream work of old jobs.
    """
    world = env.w
    if (matrix.shape != (env.K, env.N) or not matrix.any(1).all()
            or demand.shape != (env.N, len(env.c.services)) or np.any(demand < 0) or seconds <= 0):
        raise ValueError('Invalid deployment, demand or horizon for transport diagnostics')
    caps = np.concatenate((world.nodes[:,0], [r[2] for r in world.directed]))
    committed = np.concatenate((np.zeros(env.N) if cpu_work is None else cpu_work,
                                np.zeros(len(caps)-env.N) if link_work is None else link_work))/seconds
    eq_rows, eq_cols, eq_values, balance = [], [], [], []
    cap_rows, cap_cols, cap_values = [], [], []
    unavoidable = committed[env.N:].copy()
    columns = 0
    for service, chain in enumerate(env.c.services):
        sources = np.flatnonzero(demand[:,service] > 0)
        if not len(sources):
            continue
        balance.extend(demand[sources,service])
        start = len(balance)-len(sources)
        for i, stage in enumerate(chain):
            destinations = np.flatnonzero(matrix[stage])
            next_start = len(balance)
            if i+1 < len(chain):
                balance.extend(np.zeros(len(destinations)))
            cpu, mbit = world.stages[stage,:2]
            paths = world.path_incidence[sources[:,None], destinations]
            if i == 0:
                unavoidable += (demand[sources,service] @ paths.all(axis=1)) * mbit
            else:
                unavoidable += demand[:,service].sum() * paths.all(axis=(0,1)) * mbit
            for a, source in enumerate(sources):
                for b, dest in enumerate(destinations):
                    eq_rows.append(start + a)
                    eq_cols.append(columns)
                    eq_values.append(1.)
                    if i+1 < len(chain):
                        eq_rows.append(next_start + b)
                        eq_cols.append(columns)
                        eq_values.append(-1.)
                    route = world.routes[source][dest]
                    cap_rows.extend((dest, *(env.N+link for link in route)))
                    cap_cols.extend([columns]*(1+len(route)))
                    cap_values.extend((cpu, *([mbit]*len(route))))
                    columns += 1
            sources, start = destinations, next_start
    resource = coo_matrix((cap_values,(cap_rows,cap_cols)), shape=(len(caps),columns)).tocsr()
    cap_rows.extend(range(len(caps)))
    cap_cols.extend([columns]*len(caps))
    cap_values.extend(-caps)
    ub = coo_matrix((cap_values,(cap_rows,cap_cols)), shape=(len(caps),columns+1)).tocsr()
    eq = coo_matrix((eq_values,(eq_rows,eq_cols)), shape=(len(balance),columns+1)).tocsr()
    objective = np.zeros(columns+1)
    objective[-1] = 1.
    solution = linprog(objective, A_eq=eq, b_eq=balance, A_ub=ub, b_ub=-committed, method='highs')
    if not solution.success:
        raise RuntimeError(f'Service-chain transport solver: {solution.message}')
    ratios = (resource @ solution.x[:-1]+committed)/caps
    return TransportRisk(float(solution.fun), unavoidable/caps[env.N:], ratios[:env.N], ratios[env.N:])
