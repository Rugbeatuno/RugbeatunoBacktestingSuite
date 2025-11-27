import pandas_ta as ta
import pandas as pd
import numpy as np

from Engine.Backtester import Backtester


def Parameter(name: str, min_value: float, max_value: float, step: float):
    """
    Defines a parameter for optimization.

    Parameters
    ----------
    name : str
        Name of the parameter.
    min_value : float
        Minimum value of the parameter.
    max_value : float
        Maximum value of the parameter.
    step : float
        Step size for the parameter.
    """

    def __init__(self, name: str, min_value: float, max_value: float, step: float):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.step = step


def optimize(price_timeseries: pd.DataFrame,
             timeframe: str,
             signal_generator: function,
             weights: list[Parameter],
             tick_value: float,
             contract_multiplier: float,
             slippage_ticks: float
             ) -> dict:
    """
    Performs a Monte Carlo optimization by permuting the price series
    and running backtests to evaluate strategy robustness.

    Parameters
    ----------
    price_timeseries : pd.DataFrame
        DataFrame containing the price data.
    timeframe : str
        Timeframe of the data (e.g., '1min', '5min', '1D').
    signal : pd.Series
        Trading signal series.
    tick_value : float
        Value of a single tick.
    contract_multiplier : float
        Contract multiplier.
    slippage_ticks : float
        Slippage in ticks.
    iterations : int
        Number of Monte Carlo iterations.

    Returns
    -------
    dict
        Dictionary containing statistics from the optimization.
    """
    backtester = Backtester(price_timeseries, timeframe, signal,
                            tick_value=tick_value,
                            contract_multiplier=contract_multiplier,
                            slippage_ticks=slippage_ticks)

    results = backtester.monte_carlo_optimization(iterations=iterations)
    return results
