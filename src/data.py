"""
data.py

Data loading utilities for the Markov market regimes project.

This module provides two ways to obtain price data:

1. Download daily market data from Yahoo Finance using yfinance.
2. Load previously saved price data from a local CSV file.

The rest of the project should receive a clean pandas DataFrame indexed by date,
with at least one column named "Close".
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = {"Close"}


def _clean_yfinance_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean the output returned by yfinance.

    yfinance may return either:
    - a simple DataFrame with columns such as Open, High, Low, Close, Volume;
    - a MultiIndex DataFrame, especially when several tickers are requested.

    This helper makes the output compatible with the rest of the project.
    """
    if data.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # If yfinance returns MultiIndex columns, select the requested ticker.
    if isinstance(data.columns, pd.MultiIndex):
        level_0 = data.columns.get_level_values(0)
        level_1 = data.columns.get_level_values(1)

        # Case 1: columns look like ("Close", "SPY")
        if ticker in level_1:
            data = data.xs(ticker, axis=1, level=1)

        # Case 2: columns look like ("SPY", "Close")
        elif ticker in level_0:
            data = data.xs(ticker, axis=1, level=0)

        else:
            raise ValueError(
                f"Could not find ticker {ticker} in yfinance MultiIndex columns."
            )

    # Standardize column names as strings.
    data.columns = [str(col).strip() for col in data.columns]

    if "Close" not in data.columns:
        raise ValueError(
            "The downloaded data does not contain a 'Close' column. "
            "Check the ticker, date range, or yfinance output format."
        )

    # Keep only useful columns if available.
    useful_columns = [
        col for col in ["Open", "High", "Low", "Close", "Volume"]
        if col in data.columns
    ]

    data = data[useful_columns].copy()

    # Ensure the index is a proper datetime index.
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    # Remove rows with missing close prices.
    data = data.dropna(subset=["Close"])

    return data


def download_price_data(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download historical price data from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Asset ticker. Example: "SPY", "^GSPC", "AAPL", "BTC-USD".
    start:
        Start date in YYYY-MM-DD format.
    end:
        Optional end date in YYYY-MM-DD format. If None, yfinance downloads
        data up to the most recent available date.
    interval:
        Data frequency. For this project, the default is daily data: "1d".
    auto_adjust:
        If True, OHLC prices are adjusted for dividends and splits.

    Returns
    -------
    pd.DataFrame
        Clean price DataFrame indexed by date, with at least a "Close" column.
    """
    data = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )

    clean_data = _clean_yfinance_columns(data, ticker=ticker)

    return clean_data


def load_price_data_from_csv(
    path: str,
    date_column: str = "Date",
    price_column: str = "Close",
) -> pd.DataFrame:
    """
    Load price data from a local CSV file.

    The CSV file must contain:
    - one date column;
    - one price column.

    Parameters
    ----------
    path:
        Path to the CSV file.
    date_column:
        Name of the date column in the CSV file.
    price_column:
        Name of the price column in the CSV file.

    Returns
    -------
    pd.DataFrame
        Clean price DataFrame indexed by date, with a "Close" column.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    data = pd.read_csv(file_path)

    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' not found in CSV file.")

    if price_column not in data.columns:
        raise ValueError(f"Price column '{price_column}' not found in CSV file.")

    data[date_column] = pd.to_datetime(data[date_column])

    data = data[[date_column, price_column]].copy()
    data = data.rename(columns={date_column: "Date", price_column: "Close"})
    data = data.set_index("Date")
    data = data.sort_index()
    data = data.dropna(subset=["Close"])

    return data


def save_price_data(data: pd.DataFrame, path: str) -> None:
    """
    Save a price DataFrame to CSV.

    Parameters
    ----------
    data:
        Price DataFrame indexed by date.
    path:
        Destination path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(output_path)


def validate_price_data(data: pd.DataFrame) -> None:
    """
    Validate that the price DataFrame is usable by the rest of the project.

    Parameters
    ----------
    data:
        Price DataFrame to validate.

    Raises
    ------
    ValueError
        If the DataFrame is empty, does not contain "Close", has duplicated
        dates, or contains non-positive close prices.
    """
    if data.empty:
        raise ValueError("Price data is empty.")

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Price data index must be a pandas DatetimeIndex.")

    if data.index.has_duplicates:
        raise ValueError("Price data contains duplicated dates.")

    if (data["Close"] <= 0).any():
        raise ValueError("Close prices must be strictly positive.")


if __name__ == "__main__":
    prices = download_price_data(ticker="SPY", start="2010-01-01")
    validate_price_data(prices)

    save_price_data(prices, "data/spy_prices.csv")

    print(prices.head())
    print()
    print(f"Downloaded {len(prices)} rows.")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
