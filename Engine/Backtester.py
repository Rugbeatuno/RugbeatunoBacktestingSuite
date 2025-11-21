import pandas as pd
import numpy as np

from Engine.BacktestResults import BacktestResults
from Engine.DataTimeseries import DataTimeseries


class Backtester:

    def __init__(self, data_timeseries: DataTimeseries, initial_equity: float = 100_000):
        self.data_timeseries: DataTimeseries = data_timeseries
        self.start_date = self.data_timeseries.get_start_date()
        self.end_date = self.data_timeseries.get_end_date()

        self.initial_equity = initial_equity

        self.close_prices: pd.Series = self.data_timeseries.get_close_prices()

        self.buy_and_hold_return: float = self.calculate_buy_and_hold_return()

    def full_backtest(self, signal: pd.Series) -> BacktestResults:
        pass

    def quick_backtest(self, signal: pd.Series) -> BacktestResults:
        if len(signal) != len(self.close_prices):
            raise ValueError(
                f"Signal length must match the length of close prices. {len(signal)} != {len(self.close_prices)}")

        # Calculate log returns
        log_returns = np.log(self.close_prices / self.close_prices.shift(1))

        # Handle NaN values introduced by shift
        signal = signal.shift(1).fillna(0)
        log_returns = log_returns.fillna(0)

        # Calculate strategy returns
        strategy_returns = signal * log_returns

        # Calculate cumulative log return
        cumulative_log_return = strategy_returns.sum()

        # Convert to percentage return
        percentage_return = np.exp(cumulative_log_return) - 1

        # Create BacktestResults object
        end_equity = self.initial_equity * (1 + percentage_return)

        results = BacktestResults(
            start_date=self.start_date,
            end_date=self.end_date,
            start_equity=self.initial_equity,
            end_equity=end_equity,
            percent_return=percentage_return,
            log_return=cumulative_log_return
        )

        return results

    def calculate_buy_and_hold_return(self) -> float:
        return self.close_prices.iloc[-1] / self.close_prices.iloc[0] - 1
