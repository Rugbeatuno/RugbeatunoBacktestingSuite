import pandas as pd

from Engine.EquityCurve import EquityCurve
from Engine.Utils import dollar_format


class BacktestResults:
    # equity curves
    # trade win distribution, skew analysis
    # optimal trade times etc
    # monte carlo simulations
    # monte carlo permulation test
    # hyperparameter stationary testing

    def __init__(self,
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp,
                 percent_return: float,
                 log_return: float,
                 buy_hold_return: float,
                 start_equity: float = 100_000,
                 end_equity: float = None
                 ):

        self.start_date: pd.Timestamp = start_date
        self.end_date: pd.Timestamp = end_date
        self.duration: pd.Timedelta = pd.Timedelta(
            self.start_date, self.end_date)

        self.buy_hold_return: float = buy_hold_return
        self.start_equity: float = start_equity
        self.end_equity: float = end_equity
        self.equity_curve: EquityCurve = None
        self.percent_return: float = percent_return
        self.log_return: float = log_return

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

    def __repr__(self):
        return f'''
            Backtest Results from {self.start_date} to {self.end_date} ({self.duration})
            Equity: {dollar_format(self.start_equity)} -> {dollar_format(self.end_equity)}
            Return (%): {self.percent_return:.2f}%
            Log Return: {self.log_return:.4f}
            Buy & Hold Return (%): {self.buy_hold_return:.2f}%'''

    def generate_equity_curve(self):
        pass
