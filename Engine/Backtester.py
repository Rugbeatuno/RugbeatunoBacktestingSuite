import pandas as pd
import numpy as np

from Engine.BacktestResults import BacktestResults


class Backtester:

    def __init__(self, data_timeseries: pd.DataFrame, initial_equity: float = 100_000):
        self.data_timeseries = data_timeseries
        self.initial_equity = initial_equity

        self.buy_and_hold_return: float = None

    def full_backtest(self, signal: pd.Series) -> BacktestResults:
        pass

    def quick_backtest(self, signal: pd.Series) -> float:
        pass

    def calculate_buy_and_hold_return(self) -> float:
        pass
