"""Mean-variance optimizer utilities using cvxpy."""
import numpy as np
import cvxpy as cp


def _ensure_psd(Sigma, eps=1e-6):
    """Regularize covariance matrix to be PSD by adding eps to diagonal."""
    Sigma = np.array(Sigma, dtype=float)
    # Symmetrize
    Sigma = 0.5 * (Sigma + Sigma.T)
    # add small value to diagonal
    Sigma += np.eye(Sigma.shape[0]) * eps
    return Sigma


def mean_variance_opt(mu, Sigma, risk_aversion=1.0, long_only=True, max_weight=0.4, solver=cp.ECOS):
    """Solve: max mu^T w - risk_aversion * w^T Sigma w

    Args:
        mu (array-like): expected returns vector (n,)
        Sigma (array-like): covariance matrix (n,n)
        risk_aversion (float): higher => more risk averse
        long_only (bool): whether to constrain w >= 0
        max_weight (float or None): upper bound on individual weights
        solver: cvxpy solver

    Returns:
        np.array weights (n,)
    """
    mu = np.array(mu).flatten()
    Sigma = _ensure_psd(Sigma)
    n = len(mu)
    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, Sigma)
    objective = cp.Maximize(ret - risk_aversion * risk)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight is not None:
        constraints.append(w <= max_weight)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=solver, verbose=False)
    except Exception:
        # fallback to SCS
        prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        # fallback: equal weights
        return np.ones(n) / n
    # numerical cleanup
    w_opt = np.array(w.value).flatten()
    w_opt[w_opt < 1e-8] = 0.0
    w_opt = w_opt / w_opt.sum()
    return w_opt


def efficient_frontier(mu, Sigma, points=20, long_only=True, solver=cp.ECOS):
    """Compute efficient frontier (returns, risks, weight matrix).

    Solves min variance given a target return sweep.
    """
    mu = np.array(mu).flatten()
    Sigma = _ensure_psd(Sigma)
    n = len(mu)
    returns = []
    risks = []
    weights = []
    mu_min = float(mu.min())
    mu_max = float(mu.max())
    for tgt in np.linspace(mu_min, mu_max, points):
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, mu @ w >= tgt]
        if long_only:
            constraints.append(w >= 0)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        try:
            prob.solve(solver=solver, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)
        if w.value is not None:
            weights.append(np.array(w.value).flatten())
            risks.append(float(np.sqrt(np.dot(weights[-1], Sigma.dot(weights[-1])))))
            returns.append(float(mu @ weights[-1]))
    return np.array(returns), np.array(risks), np.array(weights)
