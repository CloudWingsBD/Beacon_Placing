"""Minimum-beacon solvers — robot beacon localization project.

Reusable module extracted from optimization.ipynb: greedy + ILP (CP-SAT) + LP-relaxation lower bound.

Model: min sum_j x_j  s.t.  sum_j V_ij x_j >= r_i  for every sample point i, x_j in {0,1}
where r_i = min(k, total number of candidate beacons visible from that point), so the problem is always feasible.
"""

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
                          k=K_DEFAULT, max_iter=25, verbose=False):
    """Constraint generation (see refinement.ipynb): solve on the coarse samples, verify on P_check,
    add the violating points back into the model, and iterate until there are zero violations.

    Returns (selected beacon indices, list of iteration records). The solution is verified point by point on P_check.
    It is optimal for P_init and P_check together: the active set is a subset of P_init ∪ P_check (the two grids
    need not overlap), so its optimum is a lower bound that the final, violation-free solution attains.
    """
    from visibility import visibility_matrix

    A = P_init.copy()
    V_A = visibility_matrix(beacons, A, cells, W1, W2)
    log = []
    for it in range(1, max_iter + 1):
        sol, status = ilp_cover(V_A, k=k, hint=greedy_cover(V_A, k))
        V_sel = visibility_matrix([beacons[j] for j in sol], P_check, cells, W1, W2)
        bad = np.where(V_sel.sum(axis=1) < k)[0]
        log.append({'iter': it, 'active': len(A), 'n_beacons': len(sol),
                    'status': status, 'violations': len(bad)})
        if verbose:
            print(f'  iter {it}: |A|={len(A):6d}  n={len(sol):3d} ({status})  '
                  f'violations={len(bad)}')
        if len(bad) == 0:
            return sol, log
        # full-candidate visibility of the violating points; if a point cannot see k beacons even
        # with all candidates, its requirement is min(k, total visible), so re-check whether it truly violates
        P_bad = P_check[bad]
        V_bad = visibility_matrix(beacons, P_bad, cells, W1, W2)
        truly = V_sel[bad].sum(axis=1) < np.minimum(k, V_bad.sum(axis=1))
        if not truly.any():
            log[-1]['violations'] = 0
            log[-1]['unsatisfiable'] = int(len(bad))
            return sol, log
        V_A = np.vstack([V_A, V_bad[truly]])
        A = np.vstack([A, P_bad[truly]])
    raise RuntimeError('did not converge within max_iter iterations')
