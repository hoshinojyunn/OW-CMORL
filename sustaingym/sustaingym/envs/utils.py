from __future__ import annotations

import cvxpy as cp


def _preferred_solver() -> str | None:
    installed = set(cp.installed_solvers())
    for solver in ("MOSEK", "CLARABEL", "ECOS", "OSQP", "SCS"):
        if solver in installed:
            return solver
    return None


def solve_mosek(prob: cp.Problem, verbose: int = 0) -> None:
    """Uses cvxpy solvers to solve optimization problem.

    Args:
        prob: optimization problem
        verbose: if >= 2, prints if MOSEK solver failed
    """
    solver_name = _preferred_solver()
    if solver_name is None:
        prob.solve(warm_start=True)
        return
    try:
        prob.solve(warm_start=True, solver=solver_name)
    except cp.SolverError:
        fallback_name = "CLARABEL" if solver_name != "CLARABEL" else None
        if fallback_name is not None and fallback_name in set(cp.installed_solvers()):
            prob.solve(warm_start=True, solver=fallback_name)
        else:
            prob.solve(warm_start=True)
        if verbose >= 2:
            print(f'Default solver {solver_name} failed in action projection. Trying fallback.')
            if prob.status != 'optimal':
                print(f'prob.status = {prob.status}')
        if 'infeasible' in prob.status:
            # your problem should never be infeasible. So now go debug
            import pdb
            pdb.set_trace()  # :)
