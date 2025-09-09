"""Simple ML signal builders for financial time series."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def build_lag_features(series, lags=5):
    """Build lagged features DataFrame for a pandas Series."""
    s = series.dropna()
    df = pd.DataFrame({series.name: s})
    for i in range(1, lags + 1):
        df[f"lag{i}"] = df[series.name].shift(i)
    df = df.dropna()
    return df


def train_linear_signal(returns_series, lags=5):
    """Train a linear model to predict next-day return and return last-day prediction.

    Returns: (prediction_scalar, trained_model)
    """
    df = build_lag_features(returns_series, lags=lags)
    X = df[[f"lag{i}" for i in range(1, lags + 1)]].values
    y = df[returns_series.name].values
    model = LinearRegression()
    model.fit(X, y)
    # last available feature row for prediction
    last_feat = X[-1].reshape(1, -1)
    pred = model.predict(last_feat)[0]
    return float(pred), model


def signal_to_tilt(pred_signal, scale=0.2):
    """Map prediction to a small tilt in weights using tanh squashing.

    scale: max absolute tilt fraction to apply to a single asset.
    """
    return float(np.tanh(pred_signal) * scale)
