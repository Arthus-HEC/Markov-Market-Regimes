"""
markov.py

Markov chain utilities for the Markov market regimes project.

This module estimates and analyzes a finite-state Markov chain from an observed
sequence of market regimes.

Main outputs:
1. transition count matrix;
2. transition probability matrix;
3. stationary distribution;
4. expected regime durations;
5. log-likelihood comparison with an independent-regime model.
"""

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from features import build_feature_dataframe, validate_feature_dataframe
from regimes import (
    REGIME_LABELS,
    classify_market_regimes,
    validate_regime_dataframe,
)


DEFAULT_STATES = [1, 2, 3, 4]


def compute_transition_counts(
    data: pd.DataFrame,
    regime_column: str = "regime",
    states: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Compute the transition count matrix.

    The entry (i, j) is the number of observed transitions from regime i to
    regime j.

    Parameters
    ----------
    data:
        DataFrame containing a regime column.
    regime_column:
        Name of the regime column.
    states:
        Ordered list of states. If None, defaults to [1, 2, 3, 4].

    Returns
    -------
    pd.DataFrame
        Transition count matrix indexed and columned by regime labels.
    """
    if states is None:
        states = DEFAULT_STATES

    states = list(states)

    if regime_column not in data.columns:
        raise ValueError(f"Column '{regime_column}' not found in DataFrame.")

    regimes = data[regime_column].dropna().astype(int).to_numpy()

    if len(regimes) < 2:
        raise ValueError("At least two regime observations are required.")

    invalid_states = set(regimes).difference(states)
    if invalid_states:
        raise ValueError(f"Invalid states found in regime sequence: {invalid_states}")

    counts = pd.DataFrame(
        data=0,
        index=states,
        columns=states,
        dtype=int,
    )

    for current_state, next_state in zip(regimes[:-1], regimes[1:]):
        counts.loc[current_state, next_state] += 1

    counts.index = [REGIME_LABELS[state] for state in states]
    counts.columns = [REGIME_LABELS[state] for state in states]

    return counts


def estimate_transition_matrix(
    transition_counts: pd.DataFrame,
    smoothing_alpha: float = 0.0,
) -> pd.DataFrame:
    """
    Estimate the Markov transition probability matrix.

    The maximum likelihood estimator is:

        p_hat_ij = N_ij / sum_j N_ij

    If smoothing_alpha > 0, Laplace smoothing is used:

        p_hat_ij = (N_ij + alpha) / (N_i + K * alpha)

    where K is the number of states.

    Parameters
    ----------
    transition_counts:
        Transition count matrix.
    smoothing_alpha:
        Optional Laplace smoothing parameter. Default is 0.0.

    Returns
    -------
    pd.DataFrame
        Row-stochastic transition probability matrix.
    """
    if smoothing_alpha < 0:
        raise ValueError("smoothing_alpha must be non-negative.")

    counts = transition_counts.astype(float).copy()
    number_of_states = counts.shape[1]

    if smoothing_alpha > 0:
        counts = counts + smoothing_alpha

    row_sums = counts.sum(axis=1)

    if (row_sums == 0).any():
        empty_rows = row_sums[row_sums == 0].index.tolist()
        raise ValueError(
            "Some states have no outgoing transitions and cannot be estimated "
            f"without smoothing: {empty_rows}. "
            "Use smoothing_alpha > 0 to handle this case."
        )

    transition_matrix = counts.div(row_sums, axis=0)

    if smoothing_alpha > 0:
        expected_row_sums = transition_matrix.sum(axis=1)
        if not np.allclose(expected_row_sums, 1.0):
            raise ValueError("Transition matrix rows do not sum to one.")

    return transition_matrix


def validate_transition_matrix(
    transition_matrix: pd.DataFrame,
    tolerance: float = 1e-10,
) -> None:
    """
    Validate that a matrix is a proper transition matrix.

    Parameters
    ----------
    transition_matrix:
        Candidate transition matrix.
    tolerance:
        Numerical tolerance for row sums.

    Raises
    ------
    ValueError
        If probabilities are invalid or rows do not sum to one.
    """
    if transition_matrix.empty:
        raise ValueError("Transition matrix is empty.")

    values = transition_matrix.to_numpy(dtype=float)

    if np.isnan(values).any():
        raise ValueError("Transition matrix contains NaN values.")

    if (values < -tolerance).any():
        raise ValueError("Transition matrix contains negative probabilities.")

    row_sums = values.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=tolerance):
        raise ValueError("Rows of the transition matrix must sum to one.")


def compute_stationary_distribution(
    transition_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Compute the stationary distribution of a finite Markov chain.

    The stationary distribution pi satisfies:

        pi P = pi

    Parameters
    ----------
    transition_matrix:
        Row-stochastic transition matrix.

    Returns
    -------
    pd.Series
        Stationary distribution indexed by regime labels.
    """
    validate_transition_matrix(transition_matrix)

    matrix = transition_matrix.to_numpy(dtype=float)

    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)

    index = np.argmin(np.abs(eigenvalues - 1.0))
    stationary_vector = np.real(eigenvectors[:, index])

    # Eigenvectors are defined up to a multiplicative constant.
    # We force a non-negative normalization.
    if stationary_vector.sum() < 0:
        stationary_vector = -stationary_vector

    stationary_vector = np.maximum(stationary_vector, 0)

    total = stationary_vector.sum()
    if total <= 0:
        raise ValueError("Could not compute a valid stationary distribution.")

    stationary_vector = stationary_vector / total

    return pd.Series(
        stationary_vector,
        index=transition_matrix.index,
        name="stationary_probability",
    )


def compute_expected_durations(
    transition_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Compute expected regime durations.

    If p_ii is the probability of remaining in regime i, then the expected
    duration of regime i is:

        E[D_i] = 1 / (1 - p_ii)

    The duration counts the initial day spent in the regime.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic transition matrix.

    Returns
    -------
    pd.Series
        Expected duration of each regime.
    """
    validate_transition_matrix(transition_matrix)

    diagonal = np.diag(transition_matrix.to_numpy(dtype=float))

    durations = []
    for persistence in diagonal:
        if np.isclose(persistence, 1.0):
            durations.append(np.inf)
        else:
            durations.append(1.0 / (1.0 - persistence))

    return pd.Series(
        durations,
        index=transition_matrix.index,
        name="expected_duration",
    )


def compute_markov_log_likelihood(
    transition_counts: pd.DataFrame,
    transition_matrix: pd.DataFrame,
) -> float:
    """
    Compute the log-likelihood of the Markov model.

    Parameters
    ----------
    transition_counts:
        Transition count matrix.
    transition_matrix:
        Transition probability matrix.

    Returns
    -------
    float
        Markov model log-likelihood.
    """
    counts = transition_counts.to_numpy(dtype=float)
    probabilities = transition_matrix.to_numpy(dtype=float)

    if counts.shape != probabilities.shape:
        raise ValueError("Counts and transition matrix must have the same shape.")

    mask = counts > 0

    if (probabilities[mask] <= 0).any():
        raise ValueError(
            "Observed transitions have zero probability under the transition matrix."
        )

    log_likelihood = np.sum(counts[mask] * np.log(probabilities[mask]))

    return float(log_likelihood)


def compute_independent_log_likelihood(
    transition_counts: pd.DataFrame,
) -> Tuple[float, pd.Series]:
    """
    Compute the log-likelihood of an independent-regime model.

    In this benchmark model, the next regime does not depend on the current
    regime:

        P(X_{t+1} = j | X_t = i) = q_j

    Parameters
    ----------
    transition_counts:
        Transition count matrix.

    Returns
    -------
    tuple[float, pd.Series]
        Independent model log-likelihood and estimated unconditional
        probabilities q_j.
    """
    counts = transition_counts.to_numpy(dtype=float)
    column_counts = counts.sum(axis=0)
    total_transitions = column_counts.sum()

    if total_transitions <= 0:
        raise ValueError("No transitions available.")

    q = column_counts / total_transitions

    mask = counts > 0

    if (q == 0).any():
        zero_states = transition_counts.columns[q == 0].tolist()
        raise ValueError(
            "Some regimes are never observed as next states under the "
            f"independent model: {zero_states}."
        )

    log_q = np.log(q)
    log_likelihood = np.sum(counts * log_q.reshape(1, -1))

    q_series = pd.Series(
        q,
        index=transition_counts.columns,
        name="independent_probability",
    )

    return float(log_likelihood), q_series


def compute_likelihood_ratio_statistic(
    transition_counts: pd.DataFrame,
    transition_matrix: pd.DataFrame,
) -> float:
    """
    Compute the likelihood ratio statistic comparing:

    1. the first-order Markov model;
    2. the independent-regime model.

    A larger value indicates that conditioning on the current regime improves
    the fit.

    Parameters
    ----------
    transition_counts:
        Transition count matrix.
    transition_matrix:
        Transition probability matrix.

    Returns
    -------
    float
        Likelihood ratio statistic.
    """
    markov_ll = compute_markov_log_likelihood(
        transition_counts=transition_counts,
        transition_matrix=transition_matrix,
    )

    independent_ll, _ = compute_independent_log_likelihood(
        transition_counts=transition_counts,
    )

    likelihood_ratio = 2.0 * (markov_ll - independent_ll)

    return float(likelihood_ratio)


def summarize_markov_chain(
    transition_counts: pd.DataFrame,
    transition_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact summary of the fitted Markov chain.

    The summary contains:
    - regime persistence p_ii;
    - expected duration;
    - stationary probability.

    Parameters
    ----------
    transition_counts:
        Transition count matrix.
    transition_matrix:
        Transition probability matrix.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by regime label.
    """
    validate_transition_matrix(transition_matrix)

    persistence = pd.Series(
        np.diag(transition_matrix.to_numpy(dtype=float)),
        index=transition_matrix.index,
        name="persistence",
    )

    expected_durations = compute_expected_durations(transition_matrix)
    stationary_distribution = compute_stationary_distribution(transition_matrix)

    summary = pd.concat(
        [persistence, expected_durations, stationary_distribution],
        axis=1,
    )

    summary["outgoing_transitions"] = transition_counts.sum(axis=1).astype(int)

    return summary


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

    transition_counts = compute_transition_counts(regimes)
    transition_matrix = estimate_transition_matrix(transition_counts)
    validate_transition_matrix(transition_matrix)

    stationary_distribution = compute_stationary_distribution(transition_matrix)
    expected_durations = compute_expected_durations(transition_matrix)
    summary = summarize_markov_chain(transition_counts, transition_matrix)

    markov_ll = compute_markov_log_likelihood(
        transition_counts=transition_counts,
        transition_matrix=transition_matrix,
    )
    independent_ll, independent_probabilities = compute_independent_log_likelihood(
        transition_counts=transition_counts,
    )
    likelihood_ratio = compute_likelihood_ratio_statistic(
        transition_counts=transition_counts,
        transition_matrix=transition_matrix,
    )

    print("Volatility threshold:")
    print(f"{threshold:.4f}")
    print()

    print("Transition counts:")
    print(transition_counts)
    print()

    print("Transition matrix:")
    print(transition_matrix.round(4))
    print()

    print("Stationary distribution:")
    print(stationary_distribution.round(4))
    print()

    print("Expected durations:")
    print(expected_durations.round(2))
    print()

    print("Markov chain summary:")
    print(summary.round(4))
    print()

    print("Log-likelihood comparison:")
    print(f"Markov log-likelihood:      {markov_ll:.2f}")
    print(f"Independent log-likelihood: {independent_ll:.2f}")
    print(f"Likelihood ratio statistic: {likelihood_ratio:.2f}")

