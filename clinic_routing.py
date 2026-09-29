"""Small, explicit simulator and routing models used by the teaching notebook."""
from dataclasses import dataclass
import time

import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Settings:
    range_km: float = 10.0
    stops: int = 8
    hidden: tuple = (8,)
    capacity: float = 250.0
    cost_per_km: float = 1.0
    sigma: float = 0.15
    periods: int = 1500
    test_periods: int = 400
    seed: int = 42
    time_limit: float = 60.0
    threads: int = 2
    mip_gap: float = 1e-6
    benchmark_error: float = 0.5  # bound on total objective error from interpolation


def validate_inputs(distance, base, features, start, gamma, cfg):
    n = len(base)
    if n < 1 or np.shape(distance) != (n, n) or np.shape(features) != (n, 3):
        raise ValueError("Expected an n by n distance matrix and n by 3 feature matrix.")
    if not all(np.isfinite(a).all() for a in (distance, base, features)):
        raise ValueError("Distances, demand and features must be finite; check disconnected roads.")
    if np.any(distance < 0) or np.any(base <= 0):
        raise ValueError("Distances must be nonnegative and base demand strictly positive.")
    if not isinstance(start, (int, np.integer)) or not 0 <= start < n:
        raise ValueError("The depot index must identify one ward.")
    if not isinstance(cfg.stops, (int, np.integer)) or not 1 <= cfg.stops <= n:
        raise ValueError(f"stops must be an integer from 1 to {n}, including the depot.")
    values = [gamma, cfg.range_km, cfg.capacity, cfg.cost_per_km, cfg.sigma,
              cfg.time_limit, cfg.mip_gap, cfg.benchmark_error]
    if not np.isfinite(values).all() or gamma <= -1:
        raise ValueError("Parameters must be finite and gamma must be greater than -1.")
    if min(cfg.range_km, cfg.capacity, cfg.time_limit, cfg.benchmark_error) <= 0:
        raise ValueError("Range, capacity, time limit and benchmark error must be positive.")
    if min(cfg.cost_per_km, cfg.sigma, cfg.mip_gap) < 0:
        raise ValueError("Cost, noise and MIP gap must be nonnegative.")
    if cfg.periods < 2 or cfg.test_periods < 2 or cfg.threads < 1:
        raise ValueError("Use at least two training/test periods and one solver thread.")
    if not cfg.hidden or any(not isinstance(h, int) or h < 1 for h in cfg.hidden):
        raise ValueError("hidden must contain positive integer layer sizes.")


def weights(distance, range_km):
    """Symmetric geographic interaction; routing retains directed road distances."""
    W = np.exp(-(distance + distance.T) / (2 * range_km))
    np.fill_diagonal(W, 0)
    return W


def expected_service(mu, capacity=250.0, sigma=0.15):
    """E[min(capacity, mu*epsilon)], for mean-one lognormal epsilon."""
    mu = np.asarray(mu, dtype=float)
    if np.any(mu < 0) or not np.isfinite(mu).all():
        raise ValueError("Mean demand must be finite and nonnegative.")
    if sigma == 0:
        return np.minimum(capacity, mu)
    safe = np.maximum(mu, np.finfo(float).tiny)
    a = (np.log(capacity) - np.log(safe) + sigma**2 / 2) / sigma
    value = safe * ndtr(a - sigma) + capacity * ndtr(-a)
    return np.where(mu == 0, 0.0, value)


def history(base, features, W, gamma, cfg, *, test=False):
    """Synthetic full-information history, with independent held-out periods.

    A zero-visit period supplies the hypothetical isolated-ward outcome. These
    potential outcomes at every ward are a simulation assumption, not clinic logs.
    Architecture and solve order never change the generated data.
    """
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, int(test), 2718]))
    periods = cfg.test_periods if test else cfg.periods
    n = len(base)
    rows, targets, means = [], [], []
    for period in range(periods):
        y = np.zeros(n)
        k = 0 if period % 10 == 0 else rng.integers(1, min(10, n) + 1)
        y[rng.choice(n, k, replace=False)] = 1
        E = np.minimum(1, W @ y)
        mu = base * (1 + gamma * E)
        eps = rng.lognormal(-cfg.sigma**2 / 2, cfg.sigma, n)
        rows.append(np.column_stack([features, E]))
        targets.append(np.minimum(cfg.capacity, mu * eps))
        means.append(expected_service(mu, cfg.capacity, cfg.sigma))
    return np.vstack(rows), np.concatenate(targets), np.concatenate(means)


def fit_predictors(base, features, W, gamma, cfg):
    X, target, _ = history(base, features, W, gamma, cfg)
    Xt, yt, expected = history(base, features, W, gamma, cfg, test=True)
    models = {
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=cfg.hidden, solver="lbfgs", max_iter=3000,
            random_state=cfg.seed, tol=1e-5)),
    }
    diagnostics = []
    for name, model in models.items():
        t = time.perf_counter()
        with threadpool_limits(limits=1):
            model.fit(X, target)
        pred = np.clip(model.predict(Xt), 0, cfg.capacity)
        diagnostics.append(dict(predictor=name, training_rows=len(target), test_rows=len(yt),
                                heldout_r2=r2_score(yt, pred), heldout_mae=mean_absolute_error(yt, pred),
                                heldout_expected_mae=mean_absolute_error(expected, pred),
                                fit_seconds=time.perf_counter() - t))
    # A careful static baseline: per-ward historical mean of capped service.
    historical = target.reshape(cfg.periods, len(base)).mean(axis=0)
    return models, historical, pd.DataFrame(diagnostics), X[:3], target[:3]


def benchmark_segments(b, gamma, cfg):
    """Concave secant under-estimator and an analytic uniform error bound.

    For mean-one lognormal noise, |h''(E)| <= (b*gamma)^2 *
    exp(sigma^2)/(capacity*sigma*sqrt(2*pi)). Interpolation error on an
    interval of width delta is at most |h''|*delta^2/8.
    """
    curvature = (b * gamma)**2 * np.exp(cfg.sigma**2) / (
        cfg.capacity * cfg.sigma * np.sqrt(2 * np.pi))
    segments = max(1, int(np.ceil(np.sqrt(curvature * cfg.stops / (8 * cfg.benchmark_error)))))
    E = np.linspace(0, 1, segments + 1)
    h = expected_service(b * (1 + gamma * E), cfg.capacity, cfg.sigma)
    slope = np.diff(h) / np.diff(E)
    intercept = h[:-1] - slope * E[:-1]
    return slope, intercept, curvature / (8 * segments**2)


def subtour_callback(model, where):
    if where != GRB.Callback.MIPSOL:
        return
    try:
        xv = model.cbGetSolution(model._arcs)
        succ = {i: j for (i, j), value in xv.items() if value > 0.5}
        onpath, k = {model._start}, model._start
        while k in succ and succ[k] not in onpath:
            k = succ[k]
            onpath.add(k)
        seen = set()
        for i in succ:
            if i in onpath or i in seen:
                continue
            cycle, k = [], i
            while k in succ and k not in seen:
                seen.add(k)
                cycle.append(k)
                k = succ[k]
            inside = gp.quicksum(model._arcs[a, b] for a in cycle for b in cycle if a != b)
            visits = gp.quicksum(model._visits[a] for a in cycle)
            for k in cycle:
                model.cbLazy(inside <= visits - model._visits[k])
                model._cuts += 1
    except Exception as exc:
        model._callback_error = exc
        model.terminate()


def build_model(distance, base, features, start, gamma, cfg, variant, predictor=None, env=None):
    validate_inputs(distance, base, features, start, gamma, cfg)
    if variant not in {"static", "embedded", "historical", "benchmark"}:
        raise ValueError(f"Unknown variant: {variant}")
    n = len(base)
    W = weights(distance, cfg.range_km)
    m = gp.Model(env=env)
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = cfg.time_limit
    m.Params.Threads = cfg.threads
    m.Params.Seed = cfg.seed
    m.Params.MIPGap = cfg.mip_gap
    m.Params.LazyConstraints = 1
    x = m.addVars([(i, j) for i in range(n) for j in range(n) if i != j], vtype=GRB.BINARY, name="arc")
    y = m.addVars(n, vtype=GRB.BINARY, name="visit")
    z = m.addVars(n, lb=0, ub=cfg.capacity, name="service")
    m.addConstr(y[start] == 1)
    m.addConstr(gp.quicksum(x[i, start] for i in range(n) if i != start) == 0)
    for i in range(n):
        if i != start:
            m.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == y[i])
        m.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) <= y[i])
        m.addConstr(z[i] <= cfg.capacity * y[i])
    m.addConstr(x.sum() == y.sum() - 1)
    m.addConstr(y.sum() <= cfg.stops)
    interpolation_errors = np.zeros(n)
    if variant in {"embedded", "benchmark"}:
        raw = m.addVars(n, lb=0, ub=float(W.sum(axis=1).max()), name="raw_exposure")
        E = m.addVars(n, lb=0, ub=1, name="exposure")
        for i in range(n):
            m.addConstr(raw[i] == gp.quicksum(W[i, j] * y[j] for j in range(n)))
            m.addGenConstrMin(E[i], [raw[i]], constant=1)
    if variant == "benchmark":
        for i in range(n):
            if cfg.sigma == 0:
                m.addConstr(z[i] <= base[i] * (1 + gamma * E[i]))
            else:
                slopes, intercepts, interpolation_errors[i] = benchmark_segments(base[i], gamma, cfg)
                for slope, intercept in zip(slopes, intercepts):
                    m.addConstr(z[i] <= float(slope) * E[i] + float(intercept))
    elif variant == "embedded":
        inp = m.addMVar((n, 4), lb=-GRB.INFINITY, name="features")
        out = m.addMVar(n, lb=-GRB.INFINITY, name="prediction")
        nonnegative = m.addVars(n, lb=0, name="nonnegative_prediction")
        for i in range(n):
            for j in range(3):
                inp[i, j].LB = inp[i, j].UB = features[i, j]
            inp[i, 3].LB, inp[i, 3].UB = 0, 1
            m.addConstr(inp[i, 3] == E[i])
        add_predictor_constr(m, predictor, inp, out.reshape((n, 1)))
        for i in range(n):
            m.addGenConstrMax(nonnegative[i], [out[i].item()], constant=0)
            m.addConstr(z[i] <= nonnegative[i])
    else:
        pred = predictor if variant == "historical" else predictor.predict(np.column_stack([features, np.zeros(n)]))
        for i, value in enumerate(np.clip(pred, 0, cfg.capacity)):
            m.addConstr(z[i] <= float(value))
    m.setObjective(z.sum() - cfg.cost_per_km * gp.quicksum(distance[i, j] * x[i, j] for i, j in x), GRB.MAXIMIZE)
    m._arcs, m._visits, m._start, m._cuts, m._callback_error = x, y, start, 0, None
    m.update()
    if m.NumVars > 2000 or m.NumConstrs > 2000:
        nv, nc = m.NumVars, m.NumConstrs
        m.dispose()
        raise ValueError(f"Teaching licence budget exceeded: {nv} variables, {nc} linear constraints. Use fewer wards/units or a larger benchmark_error.")
    bound = float(np.sort(interpolation_errors)[-cfg.stops:].sum())
    return m, x, y, z, bound


def validate_route(x_values, y_values, start, stop_limit):
    selected = {i for i, value in enumerate(y_values) if value > 0.5}
    arcs = [(i, j) for (i, j), value in x_values.items() if value > 0.5]
    if start not in selected or len(selected) > stop_limit:
        raise RuntimeError("Invalid selected stops.")
    successor = dict(arcs)
    if len(successor) != len(arcs):
        raise RuntimeError("More than one outgoing arc.")
    order = [start]
    while order[-1] in successor:
        nxt = successor[order[-1]]
        if nxt in order:
            raise RuntimeError("Cycle in extracted route.")
        order.append(nxt)
    if set(order) != selected or len(arcs) != len(order) - 1:
        raise RuntimeError("Disconnected route or inconsistent visit indicators.")
    return order


def solve(distance, base, features, start, gamma, cfg, variant, predictor=None, label=None, env=None):
    t = time.perf_counter()
    m, x, y, z, error = build_model(distance, base, features, start, gamma, cfg, variant, predictor, env)
    build_seconds = time.perf_counter() - t
    try:
        m.optimize(subtour_callback)
        if m._callback_error is not None:
            raise RuntimeError("Subtour callback failed") from m._callback_error
        if m.SolCount == 0:
            raise RuntimeError(f"No feasible route: Gurobi status {m.Status}. Check inputs or increase time_limit.")
        yy = np.array([y[i].X for i in range(len(base))])
        order = validate_route(m.getAttr("X", x), yy, start, cfg.stops)
        km = sum(distance[a, b] for a, b in zip(order[:-1], order[1:]))
        service = expected_service(base * (1 + gamma * np.minimum(1, weights(distance, cfg.range_km) @ yy)), cfg.capacity, cfg.sigma)
        expected = float(service[yy > 0.5].sum())
        predicted = sum(z[i].X for i in range(len(base)))
        if any(z[i].X > cfg.capacity * yy[i] + 1e-5 for i in range(len(base))):
            raise RuntimeError("Service exceeds visit capacity.")
        if not np.isclose(predicted - cfg.cost_per_km * km, m.ObjVal, atol=1e-5):
            raise RuntimeError("Extracted route objective differs from solver objective.")
        return dict(policy=label or variant, gamma=gamma, stops=len(order), km=float(km),
                    predicted_service=float(predicted), expected_service=expected,
                    expected_objective=expected - cfg.cost_per_km * km,
                    doses_per_km=expected / km if km > 0 else None,
                    solver_objective=m.ObjVal, solver_bound=m.ObjBound, status=m.Status, mip_gap=m.MIPGap,
                    benchmark_interpolation_error=error,
                    expected_objective_upper_bound=m.ObjBound + error if variant == "benchmark" else None,
                    variables=m.NumVars, linear_constraints=m.NumConstrs, general_constraints=m.NumGenConstrs,
                    lazy_cuts=m._cuts, nodes=m.NodeCount, work=m.Work, build_seconds=build_seconds,
                    solve_seconds=m.Runtime, order=order)
    finally:
        m.dispose()


def run_scenario(distance, base, features, start, gamma, cfg):
    validate_inputs(distance, base, features, start, gamma, cfg)
    models, historical, diagnostics, sample_x, sample_y = fit_predictors(
        base, features, weights(distance, cfg.range_km), gamma, cfg)
    # Quiet environment: do not publish personal licence metadata in notebook output.
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        rows = []
        for name, predictor in models.items():
            for variant in ["static", "embedded"]:
                rows.append(solve(distance, base, features, start, gamma, cfg, variant,
                                  predictor, f"{variant} {name}", env))
        rows.append(solve(distance, base, features, start, gamma, cfg, "historical", historical, "historical mean", env))
        rows.append(solve(distance, base, features, start, gamma, cfg, "benchmark", env=env))
    return pd.DataFrame(rows), diagnostics, pd.DataFrame(
        np.column_stack([sample_x, sample_y]), columns=["log population", "unvaccinated", "inaccessibility", "exposure", "served"])
