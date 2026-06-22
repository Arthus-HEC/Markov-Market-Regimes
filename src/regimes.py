"""
regimes.py

Regime classification utilities for the Markov market regimes project.

This module classifies each trading day into one of four observable regimes
based on:

1. the sign of the daily log-return;
2. the level of realized volatility relative to a volatility threshold.

Regimes:
1 = Bull Low Volatility
2 = Bull High Volatility
3 = Bear Low Volatility
4 = Bear High Volatility
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from features import (
    build_feature_dataframe,
    compute_volatility_threshold,
    validate_feature_dataframe,
)


REGIME_LABELS: Dict[int, str] = {
    1: "Bull Low Volatility",
    2: "Bull High Volatility",
    3: "Bear Low Volatility",
    4: "Bear High Volatility",
}


def classify_market_regimes(
    data: pd.DataFrame,
    return_column: str = "log_return",
    volatility_column: str = "realized_volatility",
    regime_column: str = "regime",
    label_column: str = "regime_label",
    volatility_threshold: Optional[float] = None,
    threshold_method: str = "median",
    quantile: Optional[float] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Classify each observation into one of four observable market regimes.

    The classification rule is:

    1 = Bull Low Volatility:
        log_return >= 0 and realized_volatility < threshold

    2 = Bull High Volatility:
        log_return >= 0 and realized_volatility >= threshold

    3 = Bear Low Volatility:
        log_return < 0 and realized_volatility < threshold

    4 = Bear High Volatility:
        log_return < 0 and realized_volatility >= threshold

    Parameters
    ----------
    data:
        Feature DataFrame containing log-returns and realized volatility.
    return_column:
        Name of the log-return column.
    volatility_column:
        Name of the realized volatility column.
    regime_column:
        Name of the output regime column.
    label_column:
        Name of the output regime label column.
    volatility_threshold:
        Optional fixed volatility threshold. If None, the threshold is computed
        from the data using threshold_method.
    threshold_method:
        Method used to compute the threshold if volatility_threshold is None.
        Supported values: "median" or "quantile".
    quantile:
        Quantile level if threshold_method="quantile".

    Returns
    -------
    tuple[pd.DataFrame, float]
        DataFrame with regime columns added, and the volatility threshold used.
    """
    validate_feature_dataframe(
        data,
        return_column=return_column,
        volatility_column=volatility_column,
    )

    output = data.copy()

    if volatility_threshold is None:
        volatility_threshold = compute_volatility_threshold(
            output,
            volatility_column=volatility_column,
            method=threshold_method,
            quantile=quantile,
        )

    bull = output[return_column] >= 0
    high_vol = output[volatility_column] >= volatility_threshold

    conditions = [
        bull & ~high_vol,
        bull & high_vol,
        ~bull & ~high_vol,
        ~bull & high_vol,
    ]

    choices = [1, 2, 3, 4]

    output[regime_column] = np.select(
        conditions,
        choices,
        default=np.nan,
    )

    output[regime_column] = output[regime_column].astype(int)
    output[label_column] = output[regime_column].map(REGIME_LABELS)

    return output, float(volatility_threshold)


def compute_regime_counts(
    data: pd.DataFrame,
    regime_column: str = "regime",
    normalize: bool = False,
) -> pd.Series:
    """
    Count the number of observations in each regime.

    Parameters
    ----------
    data:
        DataFrame containing a regime column.
    regime_column:
        Name of the regime column.
    normalize:
        If True, return frequencies instead of raw counts.

    Returns
    -------
    pd.Series
        Counts or frequencies indexed by regime labels.
    """
    if regime_column not in data.columns:
        raise ValueError(f"Column '{regime_column}' not found in DataFrame.")

    counts = data[regime_column].value_counts(normalize=normalize).sort_index()
    counts.index = counts.index.map(REGIME_LABELS)

    return counts


def add_regime_indicators(
    data: pd.DataFrame,
    regime_column: str = "regime",
) -> pd.DataFrame:
    """
    Add one-hot encoded regime indicator columns.

    Parameters
    ----------
    data:
        DataFrame containing a regime column.
    regime_column:
        Name of the regime column.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with one-hot regime indicator columns.
    """
    if regime_column not in data.columns:
        raise ValueError(f"Column '{regime_column}' not found in DataFrame.")

    output = data.copy()

    for regime_id, regime_label in REGIME_LABELS.items():
        column_name = (
            "is_"
            + regime_label.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        output[column_name] = (output[regime_column] == regime_id).astype(int)

    return output


def validate_regime_dataframe(
    data: pd.DataFrame,
    regime_column: str = "regime",
    label_column: str = "regime_label",
) -> None:
    """
    Validate that the DataFrame contains a usable regime classification.

    Parameters
    ----------
    data:
        Regime DataFrame to validate.
    regime_column:
        Name of the regime column.
    label_column:
        Name of the regime label column.

    Raises
    ------
    ValueError
        If required regime columns are missing or contain invalid values.
    """
    required_columns = {regime_column, label_column}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("Regime DataFrame is empty.")

    valid_regimes = set(REGIME_LABELS.keys())
    observed_regimes = set(data[regime_column].unique())

    invalid_regimes = observed_regimes.difference(valid_regimes)
    if invalid_regimes:
        raise ValueError(f"Invalid regime values found: {invalid_regimes}")

    if data[label_column].isna().any():
        raise ValueError("Some regime labels are missing.")


if __name__ == "__main__":
    from data import load_price_data_from_csv, validate_price_data

    prices = load_price_data_from_csv("data/spy_prices.csv")
    validate_price_data(prices)

    features = build_feature_dataframe(
        prices=prices,
        price_column="Close",
        window=20,
    )

    validate_feature_dataframe(features)

    regimes, threshold = classify_market_regimes(features)
    validate_regime_dataframe(regimes)

    counts = compute_regime_counts(regimes, normalize=False)
    frequencies = compute_regime_counts(regimes, normalize=True)

    print(regimes[["Close", "log_return", "realized_volatility", "regime", "regime_label"]].head())
    print()
    print(f"Volatility threshold: {threshold:.4f}")
    print()
    print("Regime counts:")
    print(counts)
    print()
    print("Regime frequencies:")
    print((frequencies * 100).round(2).astype(str) + "%")

