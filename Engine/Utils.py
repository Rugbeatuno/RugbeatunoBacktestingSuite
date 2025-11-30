import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, norm, t
from math import sqrt
import numpy as np
import math

from typing import Tuple


def dollar_format(value: float) -> str:
    """
    Formats a float value as a dollar price string.

    Args:
        value (float): The value to format.

    Returns:
        str: The formatted dollar price string.
    """
    if value is None:
        return 'None'
    return f"${value:,.2f}"


def percentage_format(value: float) -> str:
    """
    Formats a float value as a percentage string.

    Args:
        value (float): The value to format.
    """
    if value is None:
        return 'None'

    return f'{(100 * value):.2f}%'


def number_format(value: float) -> str:
    if value is None:
        return 'None'

    return f'{value:.2f}'


def confidence_interval(k: int, n: int, alpha=0.05):
    """
    Wilson score interval (recommended for proportions).
    Returns p_hat, lower, upper.
    """
    p_hat = k / n
    z = norm.ppf(1 - alpha/2)
    z2 = z*z
    denom = 1 + z2 / n
    center = (p_hat + (z2 / (2*n))) / denom
    half = (z * sqrt((p_hat*(1-p_hat)/n) + (z2/(4*n*n)))) / denom
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    return p_hat, lower, upper


def confidence_interval_mean(
    sample_mean: float,
    sample_std: float,
    n: int,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Calculates the Student's t-confidence interval for a population mean.

    This interval is recommended when the population standard deviation is
    unknown and the sample size (n) is relatively small, or when the
    population is normally distributed.

    Args:
        sample_mean (float): The mean of the sample (x_bar).
        sample_std (float): The standard deviation of the sample (s).
        n (int): The size of the sample. Must be > 1.
        alpha (float): The significance level (e.g., 0.05 for 95% confidence).

    Returns:
        Tuple[float, float, float]: A tuple containing (sample_mean, lower_bound, upper_bound).
    """
    if n <= 1:
        raise ValueError(
            "Sample size 'n' must be greater than 1 to calculate degrees of freedom.")

    # Degrees of Freedom (df) is n - 1
    df = n - 1

    # 1. Find the Critical T-Value
    # We use (1 - alpha/2) because it's a two-tailed test
    t_score = t.ppf(1 - alpha / 2, df)

    # 2. Calculate the Standard Error of the Mean (SE)
    standard_error = sample_std / math.sqrt(n)

    # 3. Calculate the Margin of Error (ME)
    margin_of_error = t_score * standard_error

    # 4. Calculate the bounds
    lower = sample_mean - margin_of_error
    upper = sample_mean + margin_of_error

    # The confidence level is 1 - alpha
    # confidence_level = (1 - alpha) * 100

    return sample_mean, lower, upper


def plot_price_series(price_series: list, labels: list, pct_of_bars: float = 1, figsize=(14, 6)):
    """
    Efficiently plot a large price series with optional skipping.

    Parameters
    ----------
    price_series : pd.Series
        Time-indexed price data.
    skip_bar : int
        Plot every Nth bar (1 = no skipping).
    figsize : tuple
        Size of the plot.
    """
    if isinstance(price_series, list):
        for i in price_series:
            if not isinstance(i, pd.Series):
                raise TypeError("price_series must be a list of pandas Series")

    pct_of_bars = int(1 / pct_of_bars)

    if len(labels) != len(price_series):
        raise IndexError("Each price series must have a label")

    plt.figure(figsize=figsize)

    # Downsample by slicing
    for i in range(len(price_series)):
        series_to_plot = price_series[i]
        if pct_of_bars > 1:
            series_to_plot = price_series[i].iloc[::pct_of_bars]

        plt.plot(series_to_plot.index, series_to_plot.values,
                 linewidth=0.8, label=labels[i])

    plt.title(f"Price Series (showing every {pct_of_bars} bars)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.legend()
    plt.show()


def bars_per_year(timeframe: str) -> int:
    """Return the approximate number of bars per year for a timeframe string."""
    if timeframe.endswith("min"):
        minutes = int(timeframe.replace("min", ""))
        return (365.25 * 24 * 60) / minutes

    if timeframe.endswith("H"):
        hours = int(timeframe.replace("H", ""))
        return (365.25 * 24) / hours

    if timeframe.endswith("D"):
        days = int(timeframe.replace("D", ""))
        return 365.25 / days

    raise ValueError(f"Unsupported timeframe: {timeframe}")


def p_value_one_tailed(mean: float, std: float, observed: float, tail="right") -> float:
    """
    Returns the 1-tailed p-value assuming normal distribution.

    tail = "right" → P(X >= observed)
    tail = "left"  → P(X <= observed)
    """
    if std <= 0:
        raise ValueError("std deviation must be > 0")

    z = (observed - mean) / std

    if tail == "right":
        # probability of being >= observed
        return 1 - norm.cdf(z)

    elif tail == "left":
        # probability of being <= observed
        return norm.cdf(z)

    else:
        raise ValueError("tail must be 'right' or 'left'")


def get_max_consecutive_streak(arr: np.ndarray) -> int:
    """
    Finds the maximum number of consecutive identical values in a 1D NumPy array.

    Example: [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4, 2, 4, 4] -> returns 5 (for the 4s)
    """
    if arr.size == 0:
        return 0
    if arr.size == 1:
        return 1

    # 1. Identify where the array values change
    # np.diff(arr) != 0 returns a boolean array (True at change points).
    # np.nonzero gets the indices of those True values.
    change_indices = np.nonzero(np.diff(arr))[0]
    # For arr, this is: array([ 2, 4, 5, 10, 11])

    # 2. Define the boundaries of all consecutive blocks
    # np.r_ combines the start index (0), the indices *after* the change points (+1),
    # and the final index (arr.shape[0], which is 14 for the example).
    boundaries = np.r_[0, change_indices + 1, arr.shape[0]]
    # For arr, this is: array([ 0, 3, 5, 6, 11, 12, 14])

    # 3. Calculate the length (duration) of each block
    # np.diff calculates the difference between adjacent boundaries.
    run_lengths = np.diff(boundaries)
    # For arr, this is: array([3, 2, 1, 5, 1, 2])

    # 4. Find the maximum length
    max_streak = np.max(run_lengths)

    return max_streak
