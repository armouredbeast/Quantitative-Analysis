# main.py
"""End-to-end pipeline that runs data download, optimization, ML signal, and backtest."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import download_prices, compute_returns, train_test_split_by_date
from optimizer import mean_variance_opt, efficient_frontier
from ml_signal import train_linear_signal, signal_to_tilt
from backtest import compute_portfolio_return, annualized_sharpe, max_drawdown, var_cvar


TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
START = "2016-01-01"
SPLIT_DATE = "2021-01-01"


def run_pipeline(tickers=TICKERS, start=START, split_date=SPLIT_DATE):
    prices = download_prices(tickers, start=start)
    returns = compute_returns(prices)
    train, test = train_test_split_by_date(returns, split_date)

    mu = train.mean().values
    Sigma = np.cov(train.values.T)

    # Base optimization
    w_base = mean_variance_opt(mu, Sigma, risk_aversion=0.5, long_only=True, max_weight=0.4)
    print("Base optimized weights:")
    for t, wt in zip(tickers, np.round(w_base, 4)):
        print(f"  {t}: {wt}")

    # Efficient frontier (optional plot)
    ret_grid, risk_grid, weights_grid = efficient_frontier(mu, Sigma, points=12)
    plt.figure(figsize=(6,4))
    plt.plot(risk_grid, ret_grid, '-o')
    plt.xlabel('Risk (Std Dev)')
    plt.ylabel('Return')
    plt.title('Efficient Frontier (Train)')
    plt.grid(True)
    plt.savefig('efficient_frontier.png', bbox_inches='tight')
    plt.close()

    # ML signal on SPY
    pred, model = train_linear_signal(train['SPY'], lags=5)
    tilt = signal_to_tilt(pred, scale=0.2)
    print(f"ML signal (pred): {pred:.6f}, tilt applied to SPY: {tilt:.4f}")

    # Apply tilt to first asset (SPY) and renormalize
    w_tilt = w_base.copy()
    w_tilt[0] = max(0.0, w_tilt[0] + tilt)
    w_tilt = w_tilt / w_tilt.sum()
    print("Tilted weights:")
    for t, wt in zip(tickers, np.round(w_tilt,4)):
        print(f"  {t}: {wt}")

    # Backtest on test set
    port_rets = pd.Series(compute_portfolio_return(w_tilt, test), index=test.index)
    cum = (1 + port_rets).cumprod()
    plt.figure(figsize=(8,4))
    plt.plot(cum.index, cum.values, label='Opt+Tilt')
    plt.title('Equity Curve (Test)')
    plt.legend()
    plt.grid(True)
    plt.savefig('equity_curve.png', bbox_inches='tight')
    plt.close()

    sharpe = annualized_sharpe(port_rets.values)
    mdd = max_drawdown(cum)
    var95, cvar95 = var_cvar(port_rets.values, alpha=0.05)

    print('\nPerformance (Test period):')
    print(f'  Annualized Sharpe: {sharpe:.3f}')
    print(f'  Max Drawdown: {mdd:.3f}')
    print(f'  VaR(5%): {var95:.4f}, CVaR(5%): {cvar95:.4f}')

    # Save results to CSV for quick inspection
    res_df = pd.DataFrame({
        'ticker': tickers,
        'weight_base': w_base,
        'weight_tilt': w_tilt
    })
    res_df.to_csv('weights.csv', index=False)
    print('\nSaved: efficient_frontier.png, equity_curve.png, weights.csv')


if __name__ == '__main__':
    run_pipeline()