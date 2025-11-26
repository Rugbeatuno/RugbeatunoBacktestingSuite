import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, norm
from math import sqrt


def dollar_format(value: float) -> str:
    """
    Formats a float value as a dollar price string.

    Args:
        value (float): The value to format.

    Returns:
        str: The formatted dollar price string.
    """
    return f"${value:,.2f}"


def percentage_format(value: float) -> str:
    """
    Formats a float value as a percentage string.

    Args:
        value (float): The value to format.
    """
    return f'{(100 * value):.2f}%'


def number_format(value: float) -> str:
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
