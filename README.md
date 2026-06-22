# Market Regime Detection with Markov Chains

A transparent quantitative finance project using discrete-time Markov chains to classify market regimes, estimate regime transition probabilities, and test whether regime persistence can support a simple dynamic allocation strategy.

## Overview

Financial markets rarely behave homogeneously over time. Periods of calm upward trends, high volatility, drawdowns, and stress episodes tend to alternate. This project builds a simple and interpretable framework to study such market regimes using Markov chains.

The objective is not to predict asset prices directly. Instead, the project investigates whether observable market conditions exhibit enough persistence to support risk-aware portfolio allocation.

Starting from historical asset prices, the project:

1. computes daily log-returns;
2. estimates rolling realized volatility;
3. classifies each trading day into an observable market regime;
4. estimates the transition matrix of the resulting Markov chain;
5. studies regime persistence, stationary probabilities, and expected regime durations;
6. backtests a simple regime-based allocation strategy against buy-and-hold.

This project is designed as a clean baseline before moving to more advanced models such as rolling transition matrices, clustering-based regimes, Markov regime-switching models, or Hidden Markov Models.

## Why This Project?

This project connects several core skills used in quantitative finance:

* probability and Markov chains;
* financial time series analysis;
* volatility estimation;
* maximum likelihood estimation;
* backtesting discipline;
* portfolio risk management;
* Python implementation and reproducible research.

The methodology is deliberately simple and interpretable. The goal is to build a robust baseline rather than a black-box trading model.

## Regime Definition

Each trading day is classified using two observable quantities:

* the sign of the daily log-return;
* the level of realized volatility relative to a chosen threshold.

The four regimes are:

| State | Regime               | Interpretation                            |
| ----: | -------------------- | ----------------------------------------- |
|     1 | Bull Low Volatility  | Positive return, low realized volatility  |
|     2 | Bull High Volatility | Positive return, high realized volatility |
|     3 | Bear Low Volatility  | Negative return, low realized volatility  |
|     4 | Bear High Volatility | Negative return, high realized volatility |

The classification is intentionally simple. It provides an interpretable baseline rather than a fully data-driven regime discovery model.

## Methodology

Let ( P_t ) denote the price of an asset at date ( t ). The daily log-return is defined as:

```math
r_t = \log\left(\frac{P_t}{P_{t-1}}\right).
```

Realized volatility is estimated over a rolling window of length ( w ):

```math
\sigma_t = \sqrt{252} \, \widehat{\mathrm{Std}}(r_{t-w+1}, \ldots, r_t).
```

The regime process ( (X_t) ) is then modeled as a finite-state Markov chain:

```math
\mathbb{P}(X_{t+1}=j \mid X_t=i, X_{t-1}, \ldots)
=
\mathbb{P}(X_{t+1}=j \mid X_t=i).
```

The transition matrix is:

```math
P = (p_{ij})_{1 \leq i,j \leq 4},
```

where:

```math
p_{ij} = \mathbb{P}(X_{t+1}=j \mid X_t=i).
```

Given an observed path of regimes, the transition probabilities are estimated by maximum likelihood:

```math
\widehat{p}_{ij}
=
\frac{N_{ij}}{\sum_{k=1}^{4} N_{ik}},
```

where ( N_{ij} ) is the number of observed transitions from regime ( i ) to regime ( j ).

## Empirical Pipeline

The empirical workflow is:

1. Download daily adjusted close prices.
2. Compute daily log-returns.
3. Estimate rolling realized volatility.
4. Define high-volatility and low-volatility regimes.
5. Classify each trading day into one of four regimes.
6. Estimate transition counts and transition probabilities.
7. Compute persistence and expected regime durations.
8. Compute the stationary distribution implied by the transition matrix.
9. Define a lagged allocation rule based on the previous day’s regime.
10. Backtest the strategy against a buy-and-hold benchmark.
11. Evaluate the results using standard performance metrics.

## Figures

### Asset Price with Regime Classification

This figure displays the historical asset price and highlights each observation according to its estimated market regime.

![Asset price with regime classification](figures/regimes_over_time.png)

### Realized Volatility and Threshold

This figure shows the rolling annualized realized volatility and the threshold used to separate low-volatility and high-volatility regimes.

![Realized volatility and threshold](figures/volatility_threshold.png)

### Estimated Transition Matrix

The transition matrix summarizes the conditional dynamics of market regimes. Diagonal coefficients measure regime persistence.

![Transition matrix heatmap](figures/transition_matrix_heatmap.png)

### Cumulative Performance

The regime-based allocation strategy is compared to a buy-and-hold benchmark.

![Cumulative performance](figures/strategy_vs_benchmark.png)

### Drawdown Comparison

This figure compares the drawdowns of the benchmark and the regime-based strategy.

![Drawdown comparison](figures/drawdown_comparison.png)

## Backtesting Discipline

To avoid look-ahead bias, the allocation for day ( t ) is based only on the regime observed at day ( t-1 ):

```math
a_t = g(X_{t-1}),
```

where ( a_t \in [0,1] ) is the portfolio exposure to the risky asset.

The baseline allocation rule is:

| Previous regime      | Allocation |
| -------------------- | ---------: |
| Bull Low Volatility  |       100% |
| Bull High Volatility |        75% |
| Bear Low Volatility  |        50% |
| Bear High Volatility |         0% |

This rule is heuristic. Its purpose is to test whether reducing exposure during unfavorable regimes can improve risk-adjusted performance or reduce drawdowns.

## Performance Metrics

The strategy is evaluated using:

* total return;
* compound annual growth rate;
* annualized log-return;
* annualized volatility;
* Sharpe ratio;
* maximum drawdown;
* total turnover;
* average turnover.

The objective is not only to maximize raw return. A regime-based strategy may still be useful if it reduces drawdowns or improves risk-adjusted performance.

## Latest Backtest Results

The following table should be updated after running the notebook.

| Strategy            | Total Return | CAGR | Annualized Volatility | Sharpe Ratio | Max Drawdown | Total Turnover |
| ------------------- | -----------: | ---: | --------------------: | -----------: | -----------: | -------------: |
| Buy and Hold        |           -- |   -- |                    -- |           -- |           -- |             -- |
| Regime Strategy     |           -- |   -- |                    -- |           -- |           -- |             -- |
| Regime Strategy Net |           -- |   -- |                    -- |           -- |           -- |             -- |

## Markov Chain Summary

The fitted Markov chain can be summarized using persistence, expected duration, stationary probability, and the number of outgoing transitions observed in the sample.

| Regime               | Persistence | Expected Duration | Stationary Probability | Outgoing Transitions |
| -------------------- | ----------: | ----------------: | ---------------------: | -------------------: |
| Bull Low Volatility  |          -- |                -- |                     -- |                   -- |
| Bull High Volatility |          -- |                -- |                     -- |                   -- |
| Bear Low Volatility  |          -- |                -- |                     -- |                   -- |
| Bear High Volatility |          -- |                -- |                     -- |                   -- |

## Project Structure

```text
markov-market-regimes/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- report/
|   |-- markov_market_regimes.pdf
|   |-- markov_market_regimes.tex
|
|-- notebooks/
|   |-- 01_markov_market_regimes.ipynb
|
|-- src/
|   |-- data.py
|   |-- features.py
|   |-- regimes.py
|   |-- markov.py
|   |-- backtest.py
|   |-- plots.py
|
|-- figures/
|   |-- regimes_over_time.png
|   |-- volatility_threshold.png
|   |-- transition_matrix_heatmap.png
|   |-- strategy_vs_benchmark.png
|   |-- drawdown_comparison.png
|
|-- data/
|   |-- .gitkeep
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/markov-market-regimes.git
cd markov-market-regimes
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

First, download and save the price data:

```bash
python src/data.py
```

Then run the individual modules:

```bash
python src/features.py
python src/regimes.py
python src/markov.py
python src/backtest.py
python src/plots.py
```

Alternatively, run the full notebook:

```bash
jupyter notebook notebooks/01_markov_market_regimes.ipynb
```

The generated figures are saved in the `figures/` folder.

## Report

A short mathematical report is available in the `report/` folder. It presents the probabilistic framework, the Markov chain model, the maximum likelihood estimator of the transition matrix, and the interpretation of regime persistence.

## Possible Extensions

This project is intentionally simple and interpretable. Natural extensions include:

* estimating rolling transition matrices;
* using quantile-based or adaptive volatility thresholds;
* comparing the Markov model to an independent regime model;
* using clustering to define regimes;
* implementing Hidden Markov Models;
* estimating regime-conditional expected returns and variances;
* deriving allocation rules from a mean-variance or utility maximization problem;
* adding more realistic transaction costs, slippage, and cash returns.

## Disclaimer

This project is for educational and research purposes only. It does not constitute investment advice. The strategy implemented here is a simplified illustration of regime-based allocation and should not be used for live trading without further validation.

## Author

**Arthus Goujon**
Mathematics & Economics student interested in quantitative finance, probability, financial markets, and applied modeling.

Website: [arthusgoujon.xyz](https://arthusgoujon.xyz)
