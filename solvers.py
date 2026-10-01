"""Minimum-beacon solvers — robot beacon localization project.

Reusable module extracted from optimization.ipynb: greedy + ILP (CP-SAT) + LP-relaxation lower bound.

Model: min sum_j x_j  s.t.  sum_j V_ij x_j >= r_i  for every sample point i, x_j in {0,1}
where r_i = min(k, total number of candidate beacons visible from that point), so the problem is always feasible.
"""

import time

import numpy as np

__all__ = ["K_DEFAULT", "requirements", "greedy_cover", "ilp_cover",
           "lp_bound", "verify", "constraint_generation"]

K_DEFAULT = 3   # minimum number of visible beacons needed to localize


def requirements(V, k=K_DEFAULT):
    """Coverage requirement of each point: r_i = min(k, total visible candidates)."""
    return np.minimum(k, V.sum(axis=1)).astype(int)


def greedy_cover(V, k=K_DEFAULT):
    """Greedy k-fold cover; returns the list of selected beacon indices."""
    N, M = V.shape
    deficit = requirements(V, k).copy()
    Vb = V.astype(bool)
    used = np.zeros(M, dtype=bool)
    chosen = []
    while deficit.max() > 0:
        need = deficit > 0
        gain = (Vb & need[:, None]).sum(axis=0)
        gain[used] = -1
        j = int(gain.argmax())
        if gain[j] <= 0:
            break
        used[j] = True
        chosen.append(j)
        deficit[Vb[:, j]] = np.maximum(0, deficit[Vb[:, j]] - 1)
    return chosen


def ilp_cover(V, k=K_DEFAULT, hint=None, time_limit=60.0):
    """Minimum k-fold cover with CP-SAT. Returns (selected indices, status string)."""
    from ortools.sat.python import cp_model

    N, M = V.shape
    req = requirements(V, k)
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f'x{j}') for j in range(M)]
    for i in range(N):
        model.Add(sum(x[j] for j in range(M) if V[i, j]) >= int(req[i]))
    model.Minimize(sum(x))
    if hint:
        hs = set(hint)
        for j in range(M):
            model.AddHint(x[j], 1 if j in hs else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    sel = [j for j in range(M) if solver.Value(x[j])]
    name = {cp_model.OPTIMAL: 'OPTIMAL', cp_model.FEASIBLE: 'FEASIBLE',
            cp_model.INFEASIBLE: 'INFEASIBLE'}.get(status, str(status))
    return sel, name


def lp_bound(V, k=K_DEFAULT):
    """Optimal value of the LP relaxation (a lower bound for the integer problem). Requires pulp."""
    import pulp

    N, M = V.shape
    req = requirements(V, k)
    p = pulp.LpProblem('lp', pulp.LpMinimize)
    x = [pulp.LpVariable(f'x{j}', lowBound=0, upBound=1) for j in range(M)]
    p += pulp.lpSum(x)
    for i in range(N):
        p += pulp.lpSum(x[j] for j in range(M) if V[i, j]) >= int(req[i])
    p.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.value(p.objective)


def verify(V, chosen, k=K_DEFAULT):
    """Check a solution; returns (whether every point meets its requirement, actual coverage count per point)."""
    counts = (V[:, chosen].sum(axis=1) if len(chosen)
              else np.zeros(V.shape[0], dtype=int))
    return bool((counts >= requirements(V, k)).all()), counts


def constraint_generation(beacons, cells, W1, W2, P_init, P_check,
                          k=K_DEFAULT, max_iter=25, verbose=False, reuse=True, keep=None, scores=None):
    """Constraint generation (see refinement.ipynb): solve on the coarse samples, verify on P_check,
    add the violating points back into the model, and iterate until there are zero violations.

    Returns (selected beacon indices, list of iteration records). The solution is verified point by point on P_check.
    It is optimal for P_init and P_check together: the active set is a subset of P_init ∪ P_check (the two grids
    need not overlap), so its optimum is a lower bound that the final, violation-free solution attains.
    The records also hold the seconds spent on setup, ILP solves, verification and model updates
    (t_init, t_ilp, t_verify, t_update).

    reuse:  cache each beacon's visibility on P_check across iterations and warm-start every re-solve from the
            previous solution — same guarantee, much faster on large maps. False runs the plain algorithm.
    keep:   optional candidate indices to optimize over instead of all of them, e.g. the network's top-K
            (see scaling.ipynb). Requirements still count ALL candidates: whenever a point cannot reach
            min(k, #visible candidates) with the kept ones, the missing candidates it sees are added back,
            best `scores` first. The result is feasible for the full problem and optimal over the final kept set.
    """
    from visibility import visibility_matrix

    M = len(beacons)
    order = np.argsort(-np.asarray(scores, dtype=float), kind='stable') if scores is not None else np.arange(M)
    cols = list(range(M)) if keep is None else sorted({int(j) for j in keep})   # candidates in the model

    def vis(idx, P):
        return visibility_matrix([beacons[j] for j in idx], P, cells, W1, W2)

    def missing(R):
        """Pruned candidates to add back so that every row of R (full-candidate visibility) is servable"""
        kept = np.zeros(M, dtype=bool)
        kept[cols] = True
        need = np.minimum(k, R.sum(axis=1)) - R[:, kept].sum(axis=1)
        add = []
        for i in np.flatnonzero(need > 0):
            for j in order:
                if need[i] <= 0:
                    break
                if R[i, j] and not kept[j]:
                    kept[j] = True
                    add.append(int(j))
                    need -= R[:, j]
        return add

    t0 = time.perf_counter()
    A = P_init.copy()
    V_A = vis(cols, A)
    if keep is not None:
        low = np.flatnonzero(V_A.sum(axis=1) < k)
        add = missing(vis(range(M), A[low])) if len(low) else []
        if add:
            V_A = np.hstack([V_A, vis(add, A)])
            cols += add
    log, cache, prev = [], {}, None
    t_init = time.perf_counter() - t0
    for it in range(1, max_iter + 1):
        t0 = time.perf_counter()
        if reuse and prev:
            pos = {j: c for c, j in enumerate(cols)}
            hint = [pos[j] for j in prev]
        else:
            hint = greedy_cover(V_A, k)
        sel, status = ilp_cover(V_A, k=k, hint=hint)
        sol = [cols[c] for c in sel]
        t1 = time.perf_counter()
        if reuse:
            new = [j for j in sol if j not in cache]
            if new:
                cache.update(zip(new, vis(new, P_check).T))
            V_sel = np.column_stack([cache[j] for j in sol])
        else:
            V_sel = vis(sol, P_check)
        bad = np.where(V_sel.sum(axis=1) < k)[0]
        t2 = time.perf_counter()
        log.append({'iter': it, 'active': len(A), 'n_beacons': len(sol),
                    'status': status, 'violations': len(bad), 'kept': len(cols),
                    't_init': t_init if it == 1 else 0.0, 't_ilp': t1 - t0, 't_verify': t2 - t1, 't_update': 0.0})
        if verbose:
            print(f'  iter {it}: |A|={len(A):6d}  n={len(sol):3d} ({status})  '
                  f'violations={len(bad)}')
        if len(bad) == 0:
            return sol, log
        # full-candidate visibility of the violating points; if a point cannot see k beacons even
        # with all candidates, its requirement is min(k, total visible), so re-check whether it truly violates
        P_bad = P_check[bad]
        V_bad = vis(range(M), P_bad)
        truly = V_sel[bad].sum(axis=1) < np.minimum(k, V_bad.sum(axis=1))
        if not truly.any():
            log[-1]['violations'] = 0
            log[-1]['unsatisfiable'] = int(len(bad))
            return sol, log
        add = missing(V_bad[truly]) if keep is not None else []
        if add:                                   # pruned candidates the new points need
            V_A = np.hstack([V_A, vis(add, A)])
            cols += add
        V_A = np.vstack([V_A, V_bad[truly][:, cols]])
        A = np.vstack([A, P_bad[truly]])
        prev = sol
        log[-1]['t_update'] = time.perf_counter() - t2
    raise RuntimeError('did not converge within max_iter iterations')
