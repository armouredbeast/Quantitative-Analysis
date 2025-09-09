# data_loader.py
"""Data download and preprocessing helpers (robust to yfinance output formats)."""
import yfinance as yf
import pandas as pd
import numpy as np

def _extract_adj_close(df):
    """
    Given the raw DataFrame returned by yf.download, extract a clean DataFrame of adjusted close prices.
    Handles:
      - single ticker (Series or DataFrame)
      - multiple tickers with columns like ('Adj Close', TICKER)
      - when 'Adj Close' is not present, fall back to 'Close'
      - if columns are single-level with tickers, assume they are adjusted already
    """
    # If it's a Series, convert to DataFrame
    if isinstance(df, pd.Series):
        return df.to_frame()

    # If DataFrame has a MultiIndex columns (e.g., ('Adj Close', 'SPY'))
    if isinstance(df.columns, pd.MultiIndex):
        # Try to pick 'Adj Close' level
        if 'Adj Close' in df.columns.levels[0]:
            out = df.xs('Adj Close', axis=1, level=0)
            return out
        # fallback to 'Close' if Adj Close missing
        if 'Close' in df.columns.levels[0]:
            out = df.xs('Close', axis=1, level=0)
            return out

    # If columns are single-level
    if 'Adj Close' in df.columns:
        return df['Adj Close'].to_frame() if isinstance(df['Adj Close'], pd.Series) else df['Adj Close']
    if 'Close' in df.columns:
        # Sometimes yf returns 'Close' only (possibly already adjusted)
        return df['Close'].to_frame() if isinstance(df['Close'], pd.Series) else df['Close']

    # If columns look like tickers already (single-level and not 'Adj Close'/'Close'),
    # assume these are already the price series per ticker
    # e.g., when user passed group of tickers and auto_adjust=True, output may be single-level tickers
    # Verify values are numeric
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if len(numeric_cols) > 0:
        return df[numeric_cols]

    # As a last resort, try to flatten columns and look for any column containing 'Adj' or 'Close'
    flat_cols = [str(c) for c in df.columns]
    for col in flat_cols:
        if 'Adj' in col or 'Close' in col:
            return df[col]

    # If nothing works, raise informative error
    raise KeyError("Could not find price column in yfinance data. Columns: {}".format(list(df.columns)))

def download_prices(tickers, start="2016-01-01", end=None, auto_adjust=True):
    """
    Download adjusted close (or close) prices for given tickers robustly.
    Returns a DataFrame with tickers as columns and Date index.
    """
    # yfinance changelog: auto_adjust behavior has changed; we default to auto_adjust=True to simplify.
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=auto_adjust)
    prices = _extract_adj_close(raw)

    # If single column/series, ensure DataFrame with one column named by ticker
    if isinstance(prices, pd.Series):
        # if user passed a single ticker string, rename to that ticker
        if isinstance(tickers, str):
            prices = prices.to_frame(name=tickers)
        else:
            prices = prices.to_frame()

    # Ensure we have columns named by tickers when possible:
    # If yfinance returned e.g., columns ['SPY'] already, fine.
    # Otherwise, try to rename numeric columns with provided tickers if counts match.
    if list(prices.columns) != list(np.atleast_1d(tickers)):
        # if counts match, assign tickers
        if len(prices.columns) == len(np.atleast_1d(tickers)):
            prices.columns = list(np.atleast_1d(tickers))

    # Drop rows with all NaNs
    prices = prices.dropna(how='all')
    return prices

def compute_returns(prices):
    """Compute simple daily returns from price series."""
    returns = prices.pct_change().dropna()
    return returns

def train_test_split_by_date(returns, split_date):
    """Split returns DataFrame into train/test by date string (inclusive for train)."""
    train = returns.loc[:split_date].copy()
    test = returns.loc[split_date:].copy()
    return train, test