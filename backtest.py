"""Backtesting helpers and metrics."""
import numpy as np
import pandas as pd


def compute_portfolio_return(weights, returns_df):
    """Daily portfolio returns series given weights and returns DataFrame."""
    # align columns
    cols = returns_df.columns
    w = np.array(weights).flatten()
    if len(w) != len(cols):
        raise ValueError("weights length must match number of assets")
    return (returns_df.values * w).sum(axis=1)


def annualized_sharpe(returns, freq=252, rf=0.0):
    mean = np.nanmean(returns) * freq
    std = np.nanstd(returns) * np.sqrt(freq)
    if std == 0:
        return 0.0
    return (mean - rf) / std


def max_drawdown(cum_returns):
    """cum_returns: pandas Series of cumulative returns (index dates)"""
    peak = cum_returns.cummax()
    dd = (cum_returns - peak) / peak
    return float(dd.min())


def var_cvar(returns, alpha=0.05):
    """Return parametric VaR (normal approx) and empirical CVaR by simulation percentile."""
    # empirical VaR
    var = np.percentile(returns, 100 * alpha)
    cvar = returns[returns <= var].mean() if (returns <= var).any() else var
    return float(var), float(cvar)
