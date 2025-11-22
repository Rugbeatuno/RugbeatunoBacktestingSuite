import pandas as pd
import numpy as np

from Engine.DataTimeseries import DataTimeseries
from Engine.EquityCurve import EquityCurve
from Engine.Utils import dollar_format, percentage_format

# equity curves
# trade win distribution, skew analysis
# optimal trade times etc
# monte carlo simulations
# monte carlo permulation test
# hyperparameter stationary testing


class Backtester:

    def __init__(self, data_timeseries: DataTimeseries, signal: pd.Series, initial_equity: float = 100_000):
        self.data_timeseries: DataTimeseries = data_timeseries
        self.close_prices: pd.Series = self.data_timeseries.get_close_prices()

        print((self.close_prices == 0).value_counts())

        self.signal: pd.Series = signal

        self.start_date = self.data_timeseries.get_start_date()
        self.end_date = self.data_timeseries.get_end_date()
        self.duration: pd.Timedelta = self.end_date - self.start_date

        self.initial_equity = initial_equity
        self.end_equity: float = None

        self.buy_hold_return: float = self.calculate_buy_and_hold_return()
        self.percent_return: float = None
        self.log_returns: pd.Series = None
        self.cum_log_return: float = None

        self.quick_backtest()

        self.equity_curve: EquityCurve = self.generate_equity_curve()
        self.bh_equity_curve: EquityCurve = self.generate_buy_hold_equity_curve()

        self.sharpe_ratio: float = None
        self.sortino_ratio: float = None
        self.profit_factor: float = None

        self.alpha: float = None
        self.beta: float = None

        self.trades: list = []
        self.total_trades: int = len(self.trades)
        self.win_rate: float = None

        self.max_drawdown: float = None
        self.max_drawdown_duration: pd.Timedelta = None
        self.consecutive_losses: int = None

    def __str__(self):
        return f'''
        Backtest Results from {self.start_date} to {self.end_date} ({self.duration})
        Equity: {dollar_format(self.initial_equity)} -> {dollar_format(self.end_equity)}
        Return (%): {percentage_format(self.percent_return)}
        Log Return: {self.cum_log_return:.4f}
        Buy & Hold Return (%): {percentage_format(self.buy_hold_return)}
        '''

    def full_backtest(self, signal: pd.Series) -> None:
        pass

    def quick_backtest(self) -> None:
        if len(self.signal) != len(self.close_prices):
            raise ValueError(
                f"Signal length must match the length of close prices. {len(self.signal)} != {len(self.close_prices)}")

        # Handle NaN values introduced by shift
        shifted_signal = self.signal.shift(1).fillna(0)
        self.log_returns = shifted_signal * \
            np.log(self.close_prices / self.close_prices.shift(1)).fillna(0)
        self.cum_log_return = self.log_returns.sum()
        self.percent_return = np.exp(self.cum_log_return) - 1

        self.end_equity = self.initial_equity * np.exp(self.cum_log_return)

    def calculate_buy_and_hold_return(self) -> float:
        return self.close_prices.iloc[-1] / self.close_prices.iloc[0] - 1

    def generate_equity_curve(self) -> EquityCurve:
        return EquityCurve(self.initial_equity, self.signal, self.log_returns)

    def generate_buy_hold_equity_curve(self) -> EquityCurve:
        bh_log_returns = np.log(
            self.close_prices / self.close_prices.shift(1)).fillna(0)
        return EquityCurve(self.initial_equity, self.signal, bh_log_returns)
