"""
plots.py

Plotting utilities for the Markov market regimes project.

This module generates the main figures used in the report, README, and website:

1. asset price with market regimes;
2. realized volatility and volatility threshold;
3. transition matrix heatmap;
4. cumulative performance comparison;
5. drawdown comparison.

The module uses matplotlib only, to keep the project lightweight and standard.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import (
    compare_backtest_performance,
    run_regime_backtest,
)
from features import build_feature_dataframe, validate_feature_dataframe
from markov import (
    compute_transition_counts,
    estimate_transition_matrix,
    summarize_markov_chain,
    validate_transition_matrix,
)
from regimes import (
    REGIME_LABELS,
    classify_market_regimes,
    validate_regime_dataframe,
)


FIGURES_DIR = Path("figures")


def ensure_output_directory(output_dir: Path = FIGURES_DIR) -> None:
    """
    Create the output directory if it does not exist.

    Parameters
    ----------
    output_dir:
        Directory where figures will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Path = FIGURES_DIR,
    dpi: int = 300,
) -> Path:
    """
    Save a matplotlib figure.

    Parameters
    ----------
    fig:
        Figure to save.
    filename:
        Output filename.
    output_dir:
        Directory where the figure will be saved.
    dpi:
        Resolution of the saved figure.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    ensure_output_directory(output_dir)

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_price_with_regimes(
    data: pd.DataFrame,
    price_column: str = "Close",
    regime_column: str = "regime",
    title: str = "Asset price with market regimes",
    filename: Optional[str] = "regimes_over_time.png",
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """
    Plot asset price and highlight the market regime through colored points.

    Parameters
    ----------
    data:
        DataFrame containing prices and regimes.
    price_column:
        Name of the price column.
    regime_column:
        Name of the regime column.
    title:
        Plot title.
    filename:
        If provided, save the figure under this filename.
    output_dir:
        Directory where the figure will be saved.

    Returns
    -------
    Optional[Path]
        Saved figure path if filename is provided, otherwise None.
    """
    if price_column not in data.columns:
        raise ValueError(f"Column '{price_column}' not found in DataFrame.")

    if regime_column not in data.columns:
        raise ValueError(f"Column '{regime_column}' not found in DataFrame.")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        data.index,
        data[price_column],
        linewidth=1.2,
        label="Close price",
    )

    for regime_id, regime_label in REGIME_LABELS.items():
        subset = data[data[regime_column] == regime_id]

        ax.scatter(
            subset.index,
            subset[price_column],
            s=8,
            label=regime_label,
            alpha=0.75,
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if filename is None:
        return None

    return save_figure(fig, filename, output_dir=output_dir)


def plot_realized_volatility(
    data: pd.DataFrame,
    volatility_threshold: float,
    volatility_column: str = "realized_volatility",
    title: str = "Realized volatility and threshold",
    filename: Optional[str] = "volatility_threshold.png",
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """
    Plot rolling realized volatility and the volatility threshold.

    Parameters
    ----------
    data:
        DataFrame containing realized volatility.
    volatility_threshold:
        Threshold separating low-volatility and high-volatility regimes.
    volatility_column:
        Name of the volatility column.
    title:
        Plot title.
    filename:
        If provided, save the figure under this filename.
    output_dir:
        Directory where the figure will be saved.

    Returns
    -------
    Optional[Path]
        Saved figure path if filename is provided, otherwise None.
    """
    if volatility_column not in data.columns:
        raise ValueError(f"Column '{volatility_column}' not found in DataFrame.")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        data.index,
        data[volatility_column],
        linewidth=1.2,
        label="Realized volatility",
    )

    ax.axhline(
        volatility_threshold,
        linestyle="--",
        linewidth=1.2,
        label="Volatility threshold",
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if filename is None:
        return None

    return save_figure(fig, filename, output_dir=output_dir)


def plot_transition_matrix_heatmap(
    transition_matrix: pd.DataFrame,
    title: str = "Estimated transition matrix",
    filename: Optional[str] = "transition_matrix_heatmap.png",
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """
    Plot a heatmap of the estimated transition matrix.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic Markov transition matrix.
    title:
        Plot title.
    filename:
        If provided, save the figure under this filename.
    output_dir:
        Directory where the figure will be saved.

    Returns
    -------
    Optional[Path]
        Saved figure path if filename is provided, otherwise None.
    """
    validate_transition_matrix(transition_matrix)

    matrix = transition_matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Next regime")
    ax.set_ylabel("Current regime")

    ax.set_xticks(np.arange(len(transition_matrix.columns)))
    ax.set_yticks(np.arange(len(transition_matrix.index)))

    ax.set_xticklabels(
        transition_matrix.columns,
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_yticklabels(
        transition_matrix.index,
        fontsize=8,
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.tight_layout()

    if filename is None:
        return None

    return save_figure(fig, filename, output_dir=output_dir)


def plot_cumulative_performance(
    backtest: pd.DataFrame,
    title: str = "Cumulative performance",
    filename: Optional[str] = "strategy_vs_benchmark.png",
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """
    Plot cumulative performance of buy-and-hold and regime strategies.

    Parameters
    ----------
    backtest:
        Output of run_regime_backtest.
    title:
        Plot title.
    filename:
        If provided, save the figure under this filename.
    output_dir:
        Directory where the figure will be saved.

    Returns
    -------
    Optional[Path]
        Saved figure path if filename is provided, otherwise None.
    """
    required_columns = {
        "benchmark_value",
        "strategy_value",
        "strategy_net_value",
    }

    missing_columns = required_columns.difference(backtest.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in backtest DataFrame: {missing_columns}")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        backtest.index,
        backtest["benchmark_value"],
        linewidth=1.4,
        label="Buy and hold",
    )
    ax.plot(
        backtest.index,
        backtest["strategy_value"],
        linewidth=1.4,
        label="Regime strategy",
    )
    ax.plot(
        backtest.index,
        backtest["strategy_net_value"],
        linewidth=1.4,
        label="Regime strategy net",
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if filename is None:
        return None

    return save_figure(fig, filename, output_dir=output_dir)


def compute_drawdown_series(
    cumulative_values: pd.Series,
) -> pd.Series:
    """
    Compute drawdown series from cumulative values.

    Parameters
    ----------
    cumulative_values:
        Portfolio cumulative value.

    Returns
    -------
    pd.Series
        Drawdown series.
    """
    if cumulative_values.empty:
        raise ValueError("Cumulative value series is empty.")

    running_max = cumulative_values.cummax()
    drawdown = cumulative_values / running_max - 1.0

    return drawdown


def plot_drawdown_comparison(
    backtest: pd.DataFrame,
    title: str = "Drawdown comparison",
    filename: Optional[str] = "drawdown_comparison.png",
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """
    Plot drawdowns of buy-and-hold and regime strategies.

    Parameters
    ----------
    backtest:
        Output of run_regime_backtest.
    title:
        Plot title.
    filename:
        If provided, save the figure under this filename.
    output_dir:
        Directory where the figure will be saved.

    Returns
    -------
    Optional[Path]
        Saved figure path if filename is provided, otherwise None.
    """
    required_columns = {
        "benchmark_value",
        "strategy_value",
        "strategy_net_value",
    }

    missing_columns = required_columns.difference(backtest.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in backtest DataFrame: {missing_columns}")

    benchmark_drawdown = compute_drawdown_series(backtest["benchmark_value"])
    strategy_drawdown = compute_drawdown_series(backtest["strategy_value"])
    strategy_net_drawdown = compute_drawdown_series(backtest["strategy_net_value"])

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        backtest.index,
        benchmark_drawdown,
        linewidth=1.2,
        label="Buy and hold",
    )
    ax.plot(
        backtest.index,
        strategy_drawdown,
        linewidth=1.2,
        label="Regime strategy",
    )
    ax.plot(
        backtest.index,
        strategy_net_drawdown,
        linewidth=1.2,
        label="Regime strategy net",
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if filename is None:
        return None

    return save_figure(fig, filename, output_dir=output_dir)


def generate_all_figures(
    regimes: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    backtest: pd.DataFrame,
    volatility_threshold: float,
    output_dir: Path = FIGURES_DIR,
) -> None:
    """
    Generate all project figures.

    Parameters
    ----------
    regimes:
        DataFrame containing prices, features, and regimes.
    transition_matrix:
        Estimated transition matrix.
    backtest:
        Backtest output.
    volatility_threshold:
        Volatility threshold used for regime classification.
    output_dir:
        Directory where figures will be saved.
    """
    ensure_output_directory(output_dir)

    saved_paths = []

    saved_paths.append(
        plot_price_with_regimes(
            regimes,
            filename="regimes_over_time.png",
            output_dir=output_dir,
        )
    )

    saved_paths.append(
        plot_realized_volatility(
            regimes,
            volatility_threshold=volatility_threshold,
            filename="volatility_threshold.png",
            output_dir=output_dir,
        )
    )

    saved_paths.append(
        plot_transition_matrix_heatmap(
            transition_matrix,
            filename="transition_matrix_heatmap.png",
            output_dir=output_dir,
        )
    )

    saved_paths.append(
        plot_cumulative_performance(
            backtest,
            filename="strategy_vs_benchmark.png",
            output_dir=output_dir,
        )
    )

    saved_paths.append(
        plot_drawdown_comparison(
            backtest,
            filename="drawdown_comparison.png",
            output_dir=output_dir,
        )
    )

    print("Saved figures:")
    for path in saved_paths:
        print(f"- {path}")


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

    transition_counts = compute_transition_counts(regimes)
    transition_matrix = estimate_transition_matrix(transition_counts)
    validate_transition_matrix(transition_matrix)

    backtest = run_regime_backtest(
        data=regimes,
        transaction_cost_bps=5.0,
        risk_free_rate_daily=0.0,
    )

    performance = compare_backtest_performance(backtest)
    markov_summary = summarize_markov_chain(
        transition_counts=transition_counts,
        transition_matrix=transition_matrix,
    )

    print("Markov chain summary:")
    print(markov_summary.round(4))
    print()

    print("Performance comparison:")
    print(performance.round(4))
    print()

    generate_all_figures(
        regimes=regimes,
        transition_matrix=transition_matrix,
        backtest=backtest,
        volatility_threshold=threshold,
        output_dir=FIGURES_DIR,
    )

