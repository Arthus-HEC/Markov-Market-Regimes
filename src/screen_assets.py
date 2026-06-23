"""
screen_assets.py

Multi-asset screening script for the Markov market regimes project.

The goal is to test several regime-based allocation rules across multiple assets
and identify where the Markov regime framework is most useful.

The script compares:
1. Buy-and-hold
2. Regime Strategy
3. Regime Strategy Net of transaction costs

It saves the full results to:

    data/screening_results.csv

Run from the project root with:

    python src/screen_assets.py
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from data import download_price_data, validate_price_data
from features import build_feature_dataframe, validate_feature_dataframe
from regimes import classify_market_regimes, validate_regime_dataframe
from markov import compute_transition_counts, estimate_transition_matrix
from backtest import run_regime_backtest, compare_backtest_performance


# ============================================================
# Configuration
# ============================================================

TICKERS: List[str] = [
    "SPY",       # S&P 500 ETF
    "QQQ",       # Nasdaq 100 ETF
    "IWM",       # Russell 2000 ETF
    "EEM",       # Emerging markets ETF
    "TLT",       # Long-term US Treasuries ETF
    "GLD",       # Gold ETF
    "BTC-USD",   # Bitcoin
    "ARKK",      # High-growth innovation ETF
    "XLE",       # Energy sector ETF
    "XLF",       # Financial sector ETF
]

START_DATE = "2015-01-01"
END_DATE = None

VOLATILITY_WINDOW = 20
TRANSACTION_COST_BPS = 5.0

OUTPUT_PATH = Path("data/screening_results.csv")


# ============================================================
# Allocation rules
# ============================================================

BASELINE_RULE: Dict[int, float] = {
    1: 1.00,  # Bull Low Volatility
    2: 0.75,  # Bull High Volatility
    3: 0.50,  # Bear Low Volatility
    4: 0.00,  # Bear High Volatility
}

CRASH_FILTER_RULE: Dict[int, float] = {
    1: 1.00,  # Bull Low Volatility
    2: 1.00,  # Bull High Volatility
    3: 1.00,  # Bear Low Volatility
    4: 0.00,  # Bear High Volatility
}

DEFENSIVE_OVERLAY_RULE: Dict[int, float] = {
    1: 1.00,  # Bull Low Volatility
    2: 1.00,  # Bull High Volatility
    3: 0.75,  # Bear Low Volatility
    4: 0.25,  # Bear High Volatility
}

VOLATILITY_TARGET_RULE: Dict[int, float] = {
    1: 1.00,  # Bull Low Volatility
    2: 0.75,  # Bull High Volatility
    3: 0.75,  # Bear Low Volatility
    4: 0.50,  # Bear High Volatility
}

ALLOCATION_RULES: Dict[str, Dict[int, float]] = {
    "baseline": BASELINE_RULE,
    "crash_filter": CRASH_FILTER_RULE,
    "defensive_overlay": DEFENSIVE_OVERLAY_RULE,
    "volatility_target": VOLATILITY_TARGET_RULE,
}


# ============================================================
# Metric extraction
# ============================================================

def extract_metrics(
    performance: pd.DataFrame,
    ticker: str,
    rule_name: str,
) -> Dict[str, float]:
    """
    Extract useful metrics from the backtest performance table.

    Parameters
    ----------
    performance:
        Output of compare_backtest_performance.
    ticker:
        Asset ticker.
    rule_name:
        Name of the allocation rule.

    Returns
    -------
    dict
        Flattened metrics for one ticker and one allocation rule.
    """
    buy_hold = performance.loc["Buy and Hold"]
    strategy = performance.loc["Regime Strategy"]
    strategy_net = performance.loc["Regime Strategy Net"]

    row = {
        "ticker": ticker,
        "rule": rule_name,

        "buy_hold_total_return": buy_hold["total_return"],
        "strategy_total_return": strategy["total_return"],
        "strategy_net_total_return": strategy_net["total_return"],

        "buy_hold_cagr": buy_hold["cagr"],
        "strategy_cagr": strategy["cagr"],
        "strategy_net_cagr": strategy_net["cagr"],

        "buy_hold_volatility": buy_hold["annualized_volatility"],
        "strategy_volatility": strategy["annualized_volatility"],
        "strategy_net_volatility": strategy_net["annualized_volatility"],

        "buy_hold_sharpe": buy_hold["sharpe_ratio"],
        "strategy_sharpe": strategy["sharpe_ratio"],
        "strategy_net_sharpe": strategy_net["sharpe_ratio"],

        "buy_hold_max_drawdown": buy_hold["max_drawdown"],
        "strategy_max_drawdown": strategy["max_drawdown"],
        "strategy_net_max_drawdown": strategy_net["max_drawdown"],

        "strategy_total_turnover": strategy["total_turnover"],
        "strategy_net_total_turnover": strategy_net["total_turnover"],
    }

    row["net_cagr_minus_buy_hold"] = (
        row["strategy_net_cagr"] - row["buy_hold_cagr"]
    )

    row["net_sharpe_minus_buy_hold"] = (
        row["strategy_net_sharpe"] - row["buy_hold_sharpe"]
    )

    # Drawdowns are negative.
    # A positive value means the net strategy has a smaller drawdown.
    row["net_drawdown_improvement"] = (
        row["strategy_net_max_drawdown"] - row["buy_hold_max_drawdown"]
    )

    row["net_volatility_reduction"] = (
        row["buy_hold_volatility"] - row["strategy_net_volatility"]
    )

    return row


# ============================================================
# Screening logic
# ============================================================

def run_single_asset_screen(
    ticker: str,
    allocation_rule: Dict[int, float],
    rule_name: str,
) -> Dict[str, float]:
    """
    Run the full pipeline for one asset and one allocation rule.

    Parameters
    ----------
    ticker:
        Asset ticker.
    allocation_rule:
        Mapping from regime IDs to allocations.
    rule_name:
        Name of the allocation rule.

    Returns
    -------
    dict
        Screening metrics for this asset and this rule.
    """
    prices = download_price_data(
        ticker=ticker,
        start=START_DATE,
        end=END_DATE,
    )
    validate_price_data(prices)

    features = build_feature_dataframe(
        prices=prices,
        price_column="Close",
        window=VOLATILITY_WINDOW,
    )
    validate_feature_dataframe(features)

    regimes, volatility_threshold = classify_market_regimes(features)
    validate_regime_dataframe(regimes)

    transition_counts = compute_transition_counts(regimes)
    transition_matrix = estimate_transition_matrix(transition_counts)

    backtest = run_regime_backtest(
        data=regimes,
        allocation_rule=allocation_rule,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        risk_free_rate_daily=0.0,
    )

    performance = compare_backtest_performance(backtest)

    row = extract_metrics(
        performance=performance,
        ticker=ticker,
        rule_name=rule_name,
    )

    row["volatility_threshold"] = volatility_threshold
    row["n_observations"] = len(regimes)

    diagonal = transition_matrix.to_numpy().diagonal()
    row["average_regime_persistence"] = float(diagonal.mean())

    return row


def format_screening_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Format screening results for terminal display.

    Parameters
    ----------
    results:
        Raw screening DataFrame.

    Returns
    -------
    pd.DataFrame
        Formatted table.
    """
    columns_to_display = [
        "ticker",
        "rule",
        "buy_hold_cagr",
        "strategy_net_cagr",
        "net_cagr_minus_buy_hold",
        "buy_hold_sharpe",
        "strategy_net_sharpe",
        "net_sharpe_minus_buy_hold",
        "buy_hold_max_drawdown",
        "strategy_net_max_drawdown",
        "net_drawdown_improvement",
        "buy_hold_volatility",
        "strategy_net_volatility",
        "net_volatility_reduction",
        "strategy_net_total_turnover",
    ]

    formatted = results[columns_to_display].copy()

    percentage_columns = [
        "buy_hold_cagr",
        "strategy_net_cagr",
        "net_cagr_minus_buy_hold",
        "buy_hold_max_drawdown",
        "strategy_net_max_drawdown",
        "net_drawdown_improvement",
        "buy_hold_volatility",
        "strategy_net_volatility",
        "net_volatility_reduction",
    ]

    for column in percentage_columns:
        formatted[column] = (formatted[column] * 100).map(lambda x: f"{x:.2f}%")

    numeric_columns = [
        "buy_hold_sharpe",
        "strategy_net_sharpe",
        "net_sharpe_minus_buy_hold",
        "strategy_net_total_turnover",
    ]

    for column in numeric_columns:
        formatted[column] = formatted[column].map(lambda x: f"{x:.2f}")

    return formatted


def add_ranking_scores(results: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple ranking scores to compare asset-rule combinations.

    The ranking is not meant to be a formal optimization criterion. It is a
    practical way to identify interesting candidates for the report.

    Higher is better:
    - net Sharpe improvement;
    - drawdown improvement;
    - volatility reduction;
    - net CAGR improvement.

    Lower is better:
    - turnover.
    """
    output = results.copy()

    output["rank_sharpe"] = output["net_sharpe_minus_buy_hold"].rank(
        ascending=False
    )
    output["rank_drawdown"] = output["net_drawdown_improvement"].rank(
        ascending=False
    )
    output["rank_volatility"] = output["net_volatility_reduction"].rank(
        ascending=False
    )
    output["rank_cagr"] = output["net_cagr_minus_buy_hold"].rank(
        ascending=False
    )
    output["rank_turnover"] = output["strategy_net_total_turnover"].rank(
        ascending=True
    )

    output["combined_rank"] = (
        output["rank_sharpe"]
        + output["rank_drawdown"]
        + output["rank_volatility"]
        + output["rank_cagr"]
        + 0.5 * output["rank_turnover"]
    )

    output = output.sort_values(
        by=["combined_rank", "net_sharpe_minus_buy_hold"],
        ascending=[True, False],
    )

    return output


def main() -> None:
    """
    Run the multi-asset screening.
    """
    rows = []

    for ticker in TICKERS:
        for rule_name, allocation_rule in ALLOCATION_RULES.items():
            print(f"Running {ticker} with rule '{rule_name}'...")

            try:
                row = run_single_asset_screen(
                    ticker=ticker,
                    allocation_rule=allocation_rule,
                    rule_name=rule_name,
                )
                rows.append(row)

            except Exception as error:
                print(f"  Skipped {ticker} / {rule_name} because of error: {error}")

    if not rows:
        raise RuntimeError("No asset could be processed successfully.")

    results = pd.DataFrame(rows)
    results = add_ranking_scores(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Screening results saved to:")
    print(OUTPUT_PATH)
    print()

    print("Top results by combined rank:")
    formatted = format_screening_results(results.head(20))
    print(formatted.to_string(index=False))


if __name__ == "__main__":
    main()