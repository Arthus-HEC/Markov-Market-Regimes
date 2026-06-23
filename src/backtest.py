"""
backtest.py

Backtesting utilities for the Markov market regimes project.

This module tests a simple regime-based allocation strategy.

The key discipline is to avoid look-ahead bias:
the allocation for date t is based on the regime observed at date t-1.

Baseline allocation rule:
1 = Bull Low Volatility  -> 100% invested
2 = Bull High Volatility -> 75% invested
3 = Bear Low Volatility  -> 50% invested
4 = Bear High Volatility -> 0% invested
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from features import build_feature_dataframe, validate_feature_dataframe
from regimes import classify_market_regimes, validate_regime_dataframe


DEFAULT_ALLOCATION_RULE = {
    1: 1.00,  # Bull Low Volatility
    2: 0.75,  # Bull High Volatility
    3: 0.75,  # Bear Low Volatility
    4: 0.50,  # Bear High Volatility
}

def validate_allocation_rule(
    allocation_rule: Dict[int, float],
    minimum_allocation: float = 0.0,
    maximum_allocation: float = 1.0,
) -> None:
    """
    Validate that the allocation rule is well-defined.

    Parameters
    ----------
    allocation_rule:
        Dictionary mapping regime IDs to portfolio allocations.
    minimum_allocation:
        Minimum allowed allocation.
    maximum_allocation:
        Maximum allowed allocation.

    Raises
    ------
    ValueError
        If the allocation rule is invalid.
    """
    required_regimes = {1, 2, 3, 4}
    provided_regimes = set(allocation_rule.keys())

    missing_regimes = required_regimes.difference(provided_regimes)
    if missing_regimes:
        raise ValueError(f"Missing allocation for regimes: {missing_regimes}")

    for regime, allocation in allocation_rule.items():
        if regime not in required_regimes:
            raise ValueError(f"Invalid regime in allocation rule: {regime}")

        if not minimum_allocation <= allocation <= maximum_allocation:
            raise ValueError(
                f"Allocation for regime {regime} must be between "
                f"{minimum_allocation} and {maximum_allocation}."
            )


def compute_lagged_allocations(
    data: pd.DataFrame,
    regime_column: str = "regime",
    allocation_rule: Optional[Dict[int, float]] = None,
    allocation_column: str = "allocation",
    lag: int = 1,
    initial_allocation: float = 0.0,
) -> pd.DataFrame:
    """
    Compute lagged portfolio allocations from observed regimes.

    The allocation at date t is based on the regime observed at date t-lag.

    Parameters
    ----------
    data:
        DataFrame containing a regime column.
    regime_column:
        Name of the regime column.
    allocation_rule:
        Dictionary mapping regime IDs to allocations.
    allocation_column:
        Name of the output allocation column.
    lag:
        Number of periods by which the allocation is lagged.
    initial_allocation:
        Allocation used when lagged information is not available.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with an allocation column.
    """
    if allocation_rule is None:
        allocation_rule = DEFAULT_ALLOCATION_RULE

    validate_allocation_rule(allocation_rule)

    if regime_column not in data.columns:
        raise ValueError(f"Column '{regime_column}' not found in DataFrame.")

    if lag < 1:
        raise ValueError("Lag must be at least 1 to avoid look-ahead bias.")

    output = data.copy()

    raw_allocation = output[regime_column].map(allocation_rule)

    if raw_allocation.isna().any():
        raise ValueError("Some regimes could not be mapped to allocations.")

    output[allocation_column] = (
        raw_allocation
        .shift(lag)
        .fillna(initial_allocation)
        .astype(float)
    )

    return output


def compute_turnover(
    allocations: pd.Series,
    initial_allocation: float = 0.0,
) -> pd.Series:
    """
    Compute daily portfolio turnover.

    Turnover is defined as the absolute change in allocation.

    Parameters
    ----------
    allocations:
        Series of portfolio allocations.
    initial_allocation:
        Allocation before the first observation.

    Returns
    -------
    pd.Series
        Daily turnover series.
    """
    previous_allocations = allocations.shift(1).fillna(initial_allocation)
    turnover = (allocations - previous_allocations).abs()

    return turnover


def run_regime_backtest(
    data: pd.DataFrame,
    return_column: str = "log_return",
    regime_column: str = "regime",
    allocation_rule: Optional[Dict[int, float]] = None,
    transaction_cost_bps: float = 0.0,
    risk_free_rate_daily: float = 0.0,
    allocation_column: str = "allocation",
    turnover_column: str = "turnover",
    cost_column: str = "transaction_cost",
    strategy_return_column: str = "strategy_log_return",
    strategy_net_return_column: str = "strategy_net_log_return",
    benchmark_return_column: str = "benchmark_log_return",
) -> pd.DataFrame:
    """
    Run a regime-based allocation backtest.

    The strategy uses the previous day's regime to determine today's exposure.

    Parameters
    ----------
    data:
        DataFrame containing log-returns and regimes.
    return_column:
        Name of the log-return column.
    regime_column:
        Name of the regime column.
    allocation_rule:
        Dictionary mapping regime IDs to allocations.
    transaction_cost_bps:
        Transaction cost in basis points paid on allocation turnover.
        Example: 5 means 5 basis points.
    risk_free_rate_daily:
        Daily log-return earned on the cash allocation.
        Default is 0.
    allocation_column:
        Name of the allocation column.
    turnover_column:
        Name of the turnover column.
    cost_column:
        Name of the transaction cost column.
    strategy_return_column:
        Name of the gross strategy return column.
    strategy_net_return_column:
        Name of the net strategy return column.
    benchmark_return_column:
        Name of the benchmark return column.

    Returns
    -------
    pd.DataFrame
        Backtest DataFrame with allocations, returns, costs, and cumulative values.
    """
    if return_column not in data.columns:
        raise ValueError(f"Column '{return_column}' not found in DataFrame.")

    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative.")

    output = compute_lagged_allocations(
        data=data,
        regime_column=regime_column,
        allocation_rule=allocation_rule,
        allocation_column=allocation_column,
        lag=1,
        initial_allocation=0.0,
    )

    output[turnover_column] = compute_turnover(output[allocation_column])

    cost_rate = transaction_cost_bps / 10_000.0
    output[cost_column] = cost_rate * output[turnover_column]

    output[benchmark_return_column] = output[return_column]

    output[strategy_return_column] = (
        output[allocation_column] * output[return_column]
        + (1.0 - output[allocation_column]) * risk_free_rate_daily
    )

    # Transaction costs are subtracted as a log-return approximation.
    output[strategy_net_return_column] = (
        output[strategy_return_column] - output[cost_column]
    )

    output["benchmark_value"] = compute_cumulative_value(
        output[benchmark_return_column]
    )

    output["strategy_value"] = compute_cumulative_value(
        output[strategy_return_column]
    )

    output["strategy_net_value"] = compute_cumulative_value(
        output[strategy_net_return_column]
    )

    return output


def compute_cumulative_value(
    log_returns: pd.Series,
    initial_value: float = 1.0,
) -> pd.Series:
    """
    Compute cumulative portfolio value from log-returns.

    Parameters
    ----------
    log_returns:
        Series of log-returns.
    initial_value:
        Initial portfolio value.

    Returns
    -------
    pd.Series
        Cumulative value series.
    """
    if initial_value <= 0:
        raise ValueError("initial_value must be strictly positive.")

    return initial_value * np.exp(log_returns.fillna(0.0).cumsum())


def compute_max_drawdown(
    cumulative_values: pd.Series,
) -> float:
    """
    Compute the maximum drawdown of a cumulative value series.

    Parameters
    ----------
    cumulative_values:
        Portfolio cumulative value.

    Returns
    -------
    float
        Maximum drawdown, expressed as a negative number.
    """
    if cumulative_values.empty:
        raise ValueError("Cumulative value series is empty.")

    running_max = cumulative_values.cummax()
    drawdowns = cumulative_values / running_max - 1.0

    return float(drawdowns.min())


def compute_cagr(
    cumulative_values: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the compound annual growth rate.

    Parameters
    ----------
    cumulative_values:
        Portfolio cumulative value.
    periods_per_year:
        Number of periods per year.

    Returns
    -------
    float
        CAGR.
    """
    if cumulative_values.empty:
        raise ValueError("Cumulative value series is empty.")

    if len(cumulative_values) < 2:
        return np.nan

    start_value = cumulative_values.iloc[0]
    end_value = cumulative_values.iloc[-1]

    if start_value <= 0 or end_value <= 0:
        raise ValueError("Cumulative values must be strictly positive.")

    number_of_years = len(cumulative_values) / periods_per_year

    return float((end_value / start_value) ** (1.0 / number_of_years) - 1.0)


def compute_performance_metrics(
    log_returns: pd.Series,
    cumulative_values: pd.Series,
    turnover: Optional[pd.Series] = None,
    periods_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Compute standard performance metrics.

    Parameters
    ----------
    log_returns:
        Strategy or benchmark log-returns.
    cumulative_values:
        Cumulative value series.
    turnover:
        Optional turnover series.
    periods_per_year:
        Number of periods per year.
    annual_risk_free_rate:
        Annual risk-free rate used in Sharpe ratio.

    Returns
    -------
    pd.Series
        Performance metrics.
    """
    clean_returns = log_returns.dropna()

    if clean_returns.empty:
        raise ValueError("Return series is empty.")

    total_return = cumulative_values.iloc[-1] / cumulative_values.iloc[0] - 1.0
    cagr = compute_cagr(cumulative_values, periods_per_year=periods_per_year)

    annualized_log_return = clean_returns.mean() * periods_per_year
    annualized_volatility = clean_returns.std(ddof=1) * np.sqrt(periods_per_year)

    if annualized_volatility == 0 or np.isnan(annualized_volatility):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (
            annualized_log_return - annual_risk_free_rate
        ) / annualized_volatility

    max_drawdown = compute_max_drawdown(cumulative_values)

    if turnover is None:
        total_turnover = np.nan
        average_turnover = np.nan
    else:
        total_turnover = float(turnover.sum())
        average_turnover = float(turnover.mean())

    metrics = pd.Series(
        {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "annualized_log_return": float(annualized_log_return),
            "annualized_volatility": float(annualized_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "total_turnover": total_turnover,
            "average_turnover": average_turnover,
        }
    )

    return metrics


def compare_backtest_performance(
    backtest: pd.DataFrame,
    periods_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Compare buy-and-hold, gross regime strategy, and net regime strategy.

    Parameters
    ----------
    backtest:
        Output of run_regime_backtest.
    periods_per_year:
        Number of periods per year.
    annual_risk_free_rate:
        Annual risk-free rate used in Sharpe ratio.

    Returns
    -------
    pd.DataFrame
        Performance comparison table.
    """
    benchmark_metrics = compute_performance_metrics(
        log_returns=backtest["benchmark_log_return"],
        cumulative_values=backtest["benchmark_value"],
        turnover=None,
        periods_per_year=periods_per_year,
        annual_risk_free_rate=annual_risk_free_rate,
    )

    strategy_metrics = compute_performance_metrics(
        log_returns=backtest["strategy_log_return"],
        cumulative_values=backtest["strategy_value"],
        turnover=backtest["turnover"],
        periods_per_year=periods_per_year,
        annual_risk_free_rate=annual_risk_free_rate,
    )

    strategy_net_metrics = compute_performance_metrics(
        log_returns=backtest["strategy_net_log_return"],
        cumulative_values=backtest["strategy_net_value"],
        turnover=backtest["turnover"],
        periods_per_year=periods_per_year,
        annual_risk_free_rate=annual_risk_free_rate,
    )

    comparison = pd.DataFrame(
        {
            "Buy and Hold": benchmark_metrics,
            "Regime Strategy": strategy_metrics,
            "Regime Strategy Net": strategy_net_metrics,
        }
    ).T

    return comparison


def format_performance_table(
    performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Format a performance table for terminal display.

    Parameters
    ----------
    performance:
        Raw performance comparison table.

    Returns
    -------
    pd.DataFrame
        Formatted performance table.
    """
    formatted = performance.copy()

    percentage_columns = [
        "total_return",
        "cagr",
        "annualized_log_return",
        "annualized_volatility",
        "max_drawdown",
        "average_turnover",
    ]

    for column in percentage_columns:
        formatted[column] = (formatted[column] * 100).map(lambda x: f"{x:.2f}%")

    formatted["sharpe_ratio"] = formatted["sharpe_ratio"].map(lambda x: f"{x:.2f}")
    formatted["total_turnover"] = formatted["total_turnover"].map(lambda x: f"{x:.2f}")

    return formatted


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

    regimes, threshold = classify_market_regimes(features)
    validate_regime_dataframe(regimes)

    backtest = run_regime_backtest(
        data=regimes,
        transaction_cost_bps=5.0,
        risk_free_rate_daily=0.0,
    )

    performance = compare_backtest_performance(backtest)
    formatted_performance = format_performance_table(performance)

    print("Volatility threshold:")
    print(f"{threshold:.4f}")
    print()

    print("Backtest preview:")
    print(
        backtest[
            [
                "Close",
                "log_return",
                "regime",
                "regime_label",
                "allocation",
                "turnover",
                "benchmark_value",
                "strategy_value",
                "strategy_net_value",
            ]
        ].head()
    )
    print()

    print("Performance comparison:")
    print(formatted_performance)
