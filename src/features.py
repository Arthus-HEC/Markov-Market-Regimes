"""
features.py

Feature engineering utilities for the Markov market regimes project.

This module transforms clean price data into financial features used to define
observable market regimes:

1. daily log-returns;
2. rolling realized volatility;
3. volatility threshold;
4. clean feature DataFrame ready for regime classification.
"""

from typing import Optional

import numpy as np
import pandas as pd


def compute_log_returns(
    prices: pd.DataFrame,
    price_column: str = "Close",
    return_column: str = "log_return",
) -> pd.DataFrame:
    """
    Compute daily log-returns from a price DataFrame.

    Parameters
    ----------
    prices:
        DataFrame indexed by date, containing a price column.
    price_column:
        Name of the price column.
    return_column:
        Name of the output log-return column.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with an additional log-return column.
    """
    if price_column not in prices.columns:
        raise ValueError(f"Column '{price_column}' not found in prices DataFrame.")

    data = prices.copy()

    if (data[price_column] <= 0).any():
        raise ValueError("Prices must be strictly positive to compute log-returns.")

    data[return_column] = np.log(data[price_column] / data[price_column].shift(1))

    return data


def compute_realized_volatility(
    data: pd.DataFrame,
    return_column: str = "log_return",
    volatility_column: str = "realized_volatility",
    window: int = 20,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    Compute rolling annualized realized volatility.

    Realized volatility at date t is computed only from returns available up to
    date t. This avoids look-ahead bias.

    Parameters
    ----------
    data:
        DataFrame containing a log-return column.
    return_column:
        Name of the log-return column.
    volatility_column:
        Name of the output volatility column.
    window:
        Rolling window length.
    annualization_factor:
        Number of trading days used for annualization.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with an additional realized volatility column.
    """
    if return_column not in data.columns:
        raise ValueError(f"Column '{return_column}' not found in DataFrame.")

    if window <= 1:
        raise ValueError("Window length must be greater than 1.")

    output = data.copy()

    output[volatility_column] = (
        output[return_column]
        .rolling(window=window)
        .std()
        * np.sqrt(annualization_factor)
    )

    return output


def compute_volatility_threshold(
    data: pd.DataFrame,
    volatility_column: str = "realized_volatility",
    method: str = "median",
    quantile: Optional[float] = None,
) -> float:
    """
    Compute a volatility threshold used to classify low-volatility and
    high-volatility regimes.

    Parameters
    ----------
    data:
        DataFrame containing a realized volatility column.
    volatility_column:
        Name of the volatility column.
    method:
        Threshold method. Supported values: "median" or "quantile".
    quantile:
        Quantile level if method="quantile". Example: 0.75.

    Returns
    -------
    float
        Volatility threshold.
    """
    if volatility_column not in data.columns:
        raise ValueError(f"Column '{volatility_column}' not found in DataFrame.")

    vol = data[volatility_column].dropna()

    if vol.empty:
        raise ValueError("Volatility series is empty after dropping missing values.")

    if method == "median":
        return float(vol.median())

    if method == "quantile":
        if quantile is None:
            raise ValueError("You must provide a quantile when method='quantile'.")
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be strictly between 0 and 1.")
        return float(vol.quantile(quantile))

    raise ValueError("Unsupported method. Use 'median' or 'quantile'.")


def build_feature_dataframe(
    prices: pd.DataFrame,
    price_column: str = "Close",
    return_column: str = "log_return",
    volatility_column: str = "realized_volatility",
    window: int = 20,
    annualization_factor: int = 252,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Build the full feature DataFrame used for regime classification.

    Parameters
    ----------
    prices:
        Clean price DataFrame indexed by date.
    price_column:
        Name of the price column.
    return_column:
        Name of the log-return column.
    volatility_column:
        Name of the realized volatility column.
    window:
        Rolling window length for volatility.
    annualization_factor:
        Number of trading days used for annualization.
    dropna:
        Whether to remove rows with missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame containing prices, log-returns, and realized volatility.
    """
    data = compute_log_returns(
        prices=prices,
        price_column=price_column,
        return_column=return_column,
    )

    data = compute_realized_volatility(
        data=data,
        return_column=return_column,
        volatility_column=volatility_column,
        window=window,
        annualization_factor=annualization_factor,
    )

    if dropna:
        data = data.dropna(subset=[return_column, volatility_column])

    return data


def validate_feature_dataframe(
    data: pd.DataFrame,
    price_column: str = "Close",
    return_column: str = "log_return",
    volatility_column: str = "realized_volatility",
) -> None:
    """
    Validate that the feature DataFrame is usable by the regime classification step.

    Parameters
    ----------
    data:
        Feature DataFrame to validate.
    price_column:
        Name of the price column.
    return_column:
        Name of the log-return column.
    volatility_column:
        Name of the realized volatility column.

    Raises
    ------
    ValueError
        If required columns are missing or contain invalid values.
    """
    required_columns = {price_column, return_column, volatility_column}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("Feature DataFrame is empty.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Feature DataFrame index must be a pandas DatetimeIndex.")

    if data[[return_column, volatility_column]].isna().any().any():
        raise ValueError("Feature DataFrame contains missing values.")

    if (data[volatility_column] < 0).any():
        raise ValueError("Realized volatility must be non-negative.")


if __name__ == "__main__":
    from data import load_price_data_from_csv, validate_price_data

    prices = load_price_data_from_csv("data/btc_usd_prices.csv")
    validate_price_data(prices)

    features = build_feature_dataframe(
        prices=prices,
        price_column="Close",
        window=20,
    )

    validate_feature_dataframe(features)

    threshold = compute_volatility_threshold(features, method="median")

    print(features.head())
    print()
    print(f"Number of observations: {len(features)}")
    print(f"Volatility threshold: {threshold:.4f}")

